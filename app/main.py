import asyncio
import json
import logging
import time
import unicodedata
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

import redis
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import Depends, FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.agents.chat_graph import build_default_chat_graph, thread_id_for
from app.agents.llm import LLMConfigError, ensure_llm_configured
from app.brief.generate import (
    BriefDeps,
    BriefInputError,
    ClientNotFoundError,
    build_default_deps,
    default_fecha,
    run_brief,
)
from app.brief.scheduler import build_brief_job, lock_key, register_brief_job, release_lock
from app.brief.store import BriefStore, validate_client_id, validate_fecha
from app.core.audit import (
    log_chat_request,
    log_event,
    request_context,
    request_id_var,
    safe_url,
)
from app.core.auth import (
    CurrentPrincipal,
    admin_startup_warning,
    require_admin,
    resolve_cliente_id,
    startup_warning,
)
from app.core.bedrock_errors import llm_exception_to_http
from app.core.config import Settings, get_settings
from app.core.errors import api_error, install_error_handling
from app.core.redis_client import get_redis_client
from app.core.traceability import audit_traceability
from app.prompts.registry import PromptError, brief_prompt_source, chat_prompt_source
from app.rag.ingest_news import (
    INGEST_LAST_SUCCESS_KEY,
    IngestReport,
    build_ingest_job,
    sanitize_error,
)
from app.rag.scheduler import maybe_start_scheduler
from app.rag.stats import DEFAULT_TOP, MAX_TOP, build_summary_job
from app.tools.search_news import warm_news_store

logger = logging.getLogger(__name__)


async def _warm_prompts(settings: Settings) -> None:
    """Fetch both system prompts so the first request hits the prompt cache."""
    await chat_prompt_source(settings).aget()
    await brief_prompt_source(settings).aget()


async def warm_up(settings: Settings) -> None:
    """Build the lazy singletons (chat graph, brief deps, news store, prompts) before traffic.

    Steps run concurrently, each failure is logged and swallowed (the lazy path on the first
    request stays as fallback) and the whole warm-up is bounded by WARMUP_TIMEOUT_SECONDS.
    """

    async def step(name: str, call) -> None:
        started = time.perf_counter()
        try:
            await call()
        except Exception as exc:
            logger.warning("Warm-up step %s failed: %s: %s", name, type(exc).__name__, exc)
            return
        logger.info("Warm-up step %s done in %.0f ms", name, (time.perf_counter() - started) * 1000)

    steps = [
        step("graph", lambda: get_chat_graph()),
        step("brief", lambda: run_in_threadpool(get_brief_deps)),
        step("news", lambda: run_in_threadpool(warm_news_store, settings)),
        step("prompts", lambda: _warm_prompts(settings)),
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*steps), timeout=settings.warmup_timeout_seconds)
    except TimeoutError:
        logger.warning(
            "Warm-up timed out after %gs; remaining parts build lazily.",
            settings.warmup_timeout_seconds,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the singletons; start the news-ingestion scheduler only when ENABLE_SCHEDULER=true."""
    settings = get_settings()
    warning = startup_warning(settings)
    if warning:
        logger.warning(warning)
    admin_warning = admin_startup_warning(settings)
    if admin_warning:
        logger.warning(admin_warning)
    scheduler = maybe_start_scheduler(settings, job=build_ingest_job(settings))
    if scheduler is not None:
        register_brief_job(scheduler, build_brief_job(settings))
    app.state.scheduler = scheduler
    if settings.warmup_on_startup:
        await warm_up(settings)
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)


app = FastAPI(title="financial-ai", lifespan=lifespan)
app.state.scheduler = None
install_error_handling(app)


def get_redis() -> redis.Redis:
    return get_redis_client()


@app.get("/health")
def health(client: Annotated[redis.Redis, Depends(get_redis)]) -> dict:
    try:
        client.ping()
    except Exception:
        return {"status": "degraded", "redis": "down"}
    health_body: dict = {"status": "ok", "redis": "up"}
    news = _news_health(client)
    if news:
        health_body["news"] = news
    return health_body


def _news_health(client) -> dict:
    """Age of the last successful ingestion (absent when never recorded; never raises)."""
    try:
        raw = client.get(INGEST_LAST_SUCCESS_KEY)
        if isinstance(raw, bytes):
            raw = raw.decode()
        if not raw:
            return {}
        age = (datetime.now(UTC) - datetime.fromisoformat(raw)).total_seconds()
        return {"last_success": raw, "last_success_age_seconds": max(0, int(age))}
    except Exception:
        return {}


# ------------------------------------------------------------------ /chat

_graph = None
MAX_MESSAGE_CHARS = 4000
_KEEP_CONTROL = {"\n", "\t"}


def clean_text(value: str) -> str:
    """Drop control and invisible format characters (keeping newline and tab), then trim."""
    kept = (
        ch for ch in value if ch in _KEEP_CONTROL or unicodedata.category(ch) not in ("Cc", "Cf")
    )
    return "".join(kept).strip()


async def get_chat_graph():
    """Lazily build the production graph once; tests override this dependency."""
    global _graph
    if _graph is None:
        settings = get_settings()
        try:
            ensure_llm_configured(settings)
        except LLMConfigError as exc:
            raise api_error(503, exc.code, exc.message) from exc
        try:
            _graph = await build_default_chat_graph(settings)
        except PromptError as exc:
            raise llm_exception_to_http(exc) from exc
    return _graph


class ChatRequest(BaseModel):
    # Optional when an API key identifies the client; must match it if both are given.
    cliente_id: str | None = Field(default=None, min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("message", mode="before")
    @classmethod
    def _clean_message(cls, value):
        return clean_text(value) if isinstance(value, str) else value


class ToolCallInfo(BaseModel):
    name: str
    args: dict
    status: str  # "ok" | "error"


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_calls: list[ToolCallInfo]
    # Converse stopReason of the final model message; `truncated` = hit BEDROCK_MAX_TOKENS.
    # `tool_loop_limit` = the turn hit CHAT_MAX_TOOL_ITERATIONS.
    stop_reason: str | None = None
    truncated: bool = False
    # True when the model's reply held an explicit buy/sell recommendation and was replaced.
    recommendation_blocked: bool = False


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def _result_status(message: ToolMessage | None) -> str:
    if message is None or message.status == "error":
        return "error"
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return "ok"
    return "error" if isinstance(payload, dict) and "error" in payload else "ok"


def _extract_turn(messages: list) -> tuple[str, list[ToolCallInfo], str | None, bool]:
    """Reply, tool calls, stopReason and guardrail flag of the latest turn."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    turn = messages[start + 1 :]
    results = {m.tool_call_id: m for m in turn if isinstance(m, ToolMessage)}
    calls = [
        ToolCallInfo(name=tc["name"], args=tc["args"], status=_result_status(results.get(tc["id"])))
        for m in turn
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]
    final = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
    stop_reason = final.response_metadata.get("stopReason") if final else None
    blocked = bool(final) and final.response_metadata.get("guardrail") == "recommendation_blocked"
    return (_text(final.content) if final else ""), calls, stop_reason, blocked


async def _invoke_chat(graph, message: str, config: dict, timeout: float) -> dict:
    try:
        async with asyncio.timeout(timeout) as deadline:
            return await graph.ainvoke({"messages": [("user", message)]}, config)
    except TimeoutError as exc:
        if not deadline.expired():  # a timeout raised inside the graph, not our deadline
            raise
        raise api_error(
            504, "CHAT_TIMEOUT", f"The assistant did not answer within {timeout:g} seconds."
        ) from exc


def _chat_config(cliente_id: str, session_id: str, settings: Settings) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id_for(cliente_id, session_id),
            "cliente_id": cliente_id,
            "session_id": session_id,
        },
        # agent + tools per allowed round, plus the final answer and the guardrail step.
        "recursion_limit": 2 * settings.chat_max_tool_iterations + 8,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    principal: CurrentPrincipal,
    body: ChatRequest,
    graph: Annotated[Any, Depends(get_chat_graph)],
) -> ChatResponse:
    cliente_id = resolve_cliente_id(principal, body.cliente_id)
    settings = get_settings()
    session_id = body.session_id or uuid.uuid4().hex
    config = _chat_config(cliente_id, session_id, settings)
    started = time.perf_counter()
    audit: dict[str, Any] = {
        "cliente_id": cliente_id,
        "session_id": session_id,
        "message_chars": len(body.message),
        "status": "ok",
        "tool_calls": [],
    }
    # Correlation id + client/session visible to every tool and LLM audit line of this turn;
    # tools read the client from here, never from model-supplied arguments.
    try:
        with request_context(cliente_id=cliente_id, session_id=session_id):
            try:
                result = await _invoke_chat(
                    graph, body.message, config, settings.chat_timeout_seconds
                )
            except Exception as exc:
                http_exc = exc if hasattr(exc, "status_code") else llm_exception_to_http(exc)
                detail = http_exc.detail if isinstance(http_exc.detail, dict) else {}
                audit["error_code"] = detail.get("error", "ERROR")
                audit["status"] = "timeout" if audit["error_code"] == "CHAT_TIMEOUT" else "error"
                raise http_exc from exc
            reply, tool_calls, stop_reason, blocked = _extract_turn(result["messages"])
            untraced = [] if blocked else audit_traceability(result["messages"])
            audit.update(
                tool_calls=[call.name for call in tool_calls],
                tool_errors=sum(call.status == "error" for call in tool_calls),
                stop_reason=stop_reason,
                guardrail_blocked=blocked,
                untraced_numbers=len(untraced),
            )
    finally:
        log_chat_request(duration_ms=round((time.perf_counter() - started) * 1000, 1), **audit)
    return ChatResponse(
        reply=reply,
        session_id=session_id,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        truncated=stop_reason == "max_tokens",
        recommendation_blocked=blocked,
    )


# -------------------------------------------------------------- /chat/stream


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_turn(graph, message: str, config: dict, timeout: float, stream_tokens: bool):
    """Yield `(event, data)` pairs for one turn; the last one is `("state", values)`.

    Events: `tool_start`/`tool_end` {name} and `token` {text} (only with CHAT_STREAM_TOKENS).
    The final graph state is read back from the checkpointer once the run ends.
    """
    try:
        async with asyncio.timeout(timeout) as deadline:
            async for event in graph.astream_events(
                {"messages": [("user", message)]}, config, version="v2"
            ):
                kind = event["event"]
                if kind in ("on_tool_start", "on_tool_end"):
                    name = "tool_start" if kind == "on_tool_start" else "tool_end"
                    yield name, {"name": event["name"]}
                elif (
                    stream_tokens
                    and kind == "on_chat_model_stream"
                    and event.get("metadata", {}).get("langgraph_node") == "agent"
                ):
                    text = _text(event["data"]["chunk"].content)
                    if text:
                        yield "token", {"text": text}
            state = await graph.aget_state(config)
    except TimeoutError as exc:
        if not deadline.expired():  # a timeout raised inside the graph, not our deadline
            raise
        raise api_error(
            504, "CHAT_TIMEOUT", f"The assistant did not answer within {timeout:g} seconds."
        ) from exc
    yield "state", state.values


@app.post("/chat/stream")
async def chat_stream(
    principal: CurrentPrincipal,
    body: ChatRequest,
    graph: Annotated[Any, Depends(get_chat_graph)],
) -> StreamingResponse:
    """Server-Sent Events variant of POST /chat (same request, same final payload in `done`).

    Events: `tool_start`/`tool_end` {name}, `token` {text} (only with CHAT_STREAM_TOKENS) and a
    final `done` with the POST /chat body, which is authoritative: the guardrail runs after the
    model and may replace what the tokens showed. A failure after the stream opened cannot change
    the HTTP status, so it arrives as an `error` event {code, status, message, request_id}.
    """
    cliente_id = resolve_cliente_id(principal, body.cliente_id)
    settings = get_settings()
    session_id = body.session_id or uuid.uuid4().hex
    config = _chat_config(cliente_id, session_id, settings)

    async def events():
        started = time.perf_counter()
        audit: dict[str, Any] = {
            "cliente_id": cliente_id,
            "session_id": session_id,
            "message_chars": len(body.message),
            "status": "ok",
            "tool_calls": [],
        }
        try:
            with request_context(cliente_id=cliente_id, session_id=session_id) as request_id:
                try:
                    values: dict = {}
                    async for name, data in _stream_turn(
                        graph,
                        body.message,
                        config,
                        settings.chat_timeout_seconds,
                        settings.chat_stream_tokens,
                    ):
                        if name == "state":
                            values = data
                        else:
                            yield _sse(name, data)
                    reply, tool_calls, stop_reason, blocked = _extract_turn(values["messages"])
                    untraced = [] if blocked else audit_traceability(values["messages"])
                    audit.update(
                        tool_calls=[call.name for call in tool_calls],
                        tool_errors=sum(call.status == "error" for call in tool_calls),
                        stop_reason=stop_reason,
                        guardrail_blocked=blocked,
                        untraced_numbers=len(untraced),
                    )
                    done = ChatResponse(
                        reply=reply,
                        session_id=session_id,
                        tool_calls=tool_calls,
                        stop_reason=stop_reason,
                        truncated=stop_reason == "max_tokens",
                        recommendation_blocked=blocked,
                    )
                    yield _sse("done", done.model_dump())
                except Exception as exc:
                    http_exc = exc if hasattr(exc, "status_code") else llm_exception_to_http(exc)
                    detail = http_exc.detail if isinstance(http_exc.detail, dict) else {}
                    audit["error_code"] = detail.get("error", "ERROR")
                    audit["status"] = (
                        "timeout" if audit["error_code"] == "CHAT_TIMEOUT" else "error"
                    )
                    yield _sse(
                        "error",
                        {
                            "code": audit["error_code"],
                            "status": http_exc.status_code,
                            "message": detail.get("message", ""),
                            "request_id": request_id,
                        },
                    )
        finally:
            log_chat_request(duration_ms=round((time.perf_counter() - started) * 1000, 1), **audit)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----------------------------------------------------------- /brief/preview

_brief_deps: BriefDeps | None = None


def get_brief_deps() -> BriefDeps:
    """Lazily build the production brief dependencies once; tests override this dependency."""
    global _brief_deps
    if _brief_deps is None:
        try:
            _brief_deps = build_default_deps(get_settings())
        except LLMConfigError as exc:
            raise api_error(503, exc.code, exc.message) from exc
        except PromptError as exc:
            raise llm_exception_to_http(exc) from exc
    return _brief_deps


_CLIENT_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class BriefPreviewRequest(BaseModel):
    # Optional when an API key identifies the client; must match it if both are given.
    cliente_id: str | None = Field(default=None, pattern=_CLIENT_PATTERN)
    fecha: str | None = Field(default=None, description="YYYY-MM-DD; default today (Buenos Aires)")

    @field_validator("fecha")
    @classmethod
    def _valid_fecha(cls, value):
        if value is not None:
            validate_fecha(value)
        return value


class BriefPreviewResponse(BaseModel):
    fecha: str
    texto_brief: str
    html: str
    activos_mencionados: list[str]
    degraded: bool
    missing_inputs: list[str]


def _run_brief_or_raise(deps: BriefDeps, cliente_id: str, fecha: str | None, **options):
    try:
        return run_brief(deps, cliente_id, fecha, **options)
    except ClientNotFoundError as exc:
        raise api_error(404, "CLIENT_NOT_FOUND", str(exc)) from exc
    except BriefInputError as exc:
        raise api_error(502, "PORTFOLIO_UNAVAILABLE", str(exc)) from exc
    except Exception as exc:
        raise llm_exception_to_http(exc) from exc


@app.post("/brief/preview", response_model=BriefPreviewResponse)
def brief_preview(
    principal: CurrentPrincipal,
    body: BriefPreviewRequest,
    deps: Annotated[BriefDeps, Depends(get_brief_deps)],
) -> dict:
    """Generate a brief with live data. It is neither stored nor sent."""
    cliente_id = resolve_cliente_id(principal, body.cliente_id)
    return _run_brief_or_raise(deps, cliente_id, body.fecha).preview()


# -------------------------------------------------------------------- /brief


class BriefRunRequest(BriefPreviewRequest):
    store: bool = Field(default=True, strict=True)
    send: bool = Field(default=False, strict=True)


class BriefRunResponse(BaseModel):
    cliente_id: str
    fecha: str
    texto_brief: str
    html: str
    activos_mencionados: list[str]
    degraded: bool
    missing_inputs: list[str]
    stored: bool
    sent: bool
    delivery_status: str
    generated_at: str
    prompt: dict[str, str] | None = None
    request_id: str | None = None


class StoredBriefResponse(BaseModel):
    cliente_id: str
    fecha: str
    texto_brief: str
    html: str | None = None
    activos_mencionados: list[str]
    degraded: bool | None = None
    missing_inputs: list[str] | None = None
    generated_at: str | None = None
    prompt: dict[str, str] | None = None
    request_id: str | None = None


@app.post("/brief", response_model=BriefRunResponse)
def brief_run(
    principal: CurrentPrincipal,
    body: BriefRunRequest,
    deps: Annotated[BriefDeps, Depends(get_brief_deps)],
    redis_client: Annotated[redis.Redis, Depends(get_redis)],
) -> dict:
    """Generate the brief with the same pipeline as the CLI and the daily job.

    Stores it (default) and optionally sends the simulated email. The per-client/date Redis lock
    of the scheduler guards the run: a concurrent run answers 409 BRIEF_IN_PROGRESS.
    """
    cliente_id = resolve_cliente_id(principal, body.cliente_id)
    fecha = body.fecha or default_fecha(deps.now())
    key, token = lock_key(cliente_id, fecha), uuid.uuid4().hex
    try:
        acquired = redis_client.set(key, token, nx=True, ex=deps.settings.brief_lock_ttl_seconds)
    except redis.RedisError as exc:
        raise api_error(503, "STORAGE_UNAVAILABLE", "Brief lock storage unavailable.") from exc
    if not acquired:
        raise api_error(
            409,
            "BRIEF_IN_PROGRESS",
            f"A brief for {cliente_id} on {fecha} is already being generated.",
        )
    try:
        result = _run_brief_or_raise(deps, cliente_id, fecha, store=body.store, send=body.send)
    finally:
        release_lock(redis_client, key, token)
    delivery_status = str(result.delivery.get("status", "not_requested"))
    return {
        "cliente_id": result.cliente_id,
        "fecha": result.fecha,
        "texto_brief": result.texto_brief,
        "html": result.html,
        "activos_mencionados": result.activos_mencionados,
        "degraded": result.degraded,
        "missing_inputs": result.missing_inputs,
        "stored": result.stored,
        "sent": delivery_status == "sent",
        "delivery_status": delivery_status,
        "generated_at": result.generated_at,
        "prompt": result.prompt,
        "request_id": request_id_var.get(),
    }


@app.get("/brief/latest", response_model=StoredBriefResponse)
def brief_latest(
    principal: CurrentPrincipal,
    redis_client: Annotated[redis.Redis, Depends(get_redis)],
    cliente_id: Annotated[str | None, Query(pattern=_CLIENT_PATTERN)] = None,
) -> dict:
    """The latest stored brief of the authenticated client."""
    client = resolve_cliente_id(principal, cliente_id)
    validate_client_id(client)
    try:
        brief = BriefStore(redis_client).latest(client)
    except redis.RedisError as exc:
        raise api_error(503, "STORAGE_UNAVAILABLE", "Brief storage unavailable.") from exc
    if brief is None:
        raise api_error(404, "BRIEF_NOT_FOUND", f"No brief stored yet for client {client}.")
    return {**brief, "cliente_id": client, "request_id": request_id_var.get()}


# --------------------------------------------------------------- /news/ingest


class FailedFeed(BaseModel):
    url: str
    error: str


class FeedStatResponse(BaseModel):
    url: str
    status: str
    fetched: int
    new: int
    unchanged: int
    old: int
    invalid: int
    indexed: int
    latency_ms: float
    error: str | None = None


class IngestResponse(BaseModel):
    indexed: int
    skipped_old: int
    skipped_unchanged: int = 0
    skipped_invalid: int = 0
    purged: int
    skipped_locked: bool
    failed_feeds: list[FailedFeed]
    feeds: list[FeedStatResponse] = []
    purge_error: str | None = None
    duration_ms: float
    request_id: str | None = None


def get_ingest_job() -> Callable[[], IngestReport]:
    """The ingestion cycle (same job as the scheduler and the CLI); tests override this."""
    return build_ingest_job(get_settings())


def _clean_error(text: str) -> str:
    """URLs reduced to scheme://host/path, secrets redacted, truncated."""
    return sanitize_error(text)


def _sanitize_opt(text: str | None) -> str | None:
    return None if text is None else _clean_error(text)


def _ingest_exception_to_http(exc: Exception):
    if hasattr(exc, "status_code"):
        return exc
    if isinstance(exc, redis.RedisError):
        return api_error(503, "STORAGE_UNAVAILABLE", "News storage (Redis) unavailable.")
    if isinstance(exc, ValueError):
        return api_error(
            503,
            "NEWS_CONFIG_INVALID",
            f"Invalid news/embedding configuration: {_clean_error(str(exc))}",
        )
    if isinstance(exc, (ClientError, NoCredentialsError, PromptError)):
        return llm_exception_to_http(exc)
    logger.error("News ingestion failed", exc_info=exc)
    return api_error(502, "INGEST_FAILED", f"News ingestion failed: {type(exc).__name__}.")


@app.post("/news/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
async def news_ingest(job: Annotated[Callable[[], IngestReport], Depends(get_ingest_job)]) -> dict:
    """Run one news-ingestion cycle now (admin only; global, not per client).

    Uses the scheduler's job and Redis lock: a cycle already running answers 409.
    """
    if not get_settings().news_rss_urls:
        raise api_error(422, "NEWS_RSS_URLS_EMPTY", "NEWS_RSS_URLS is empty: nothing to ingest.")
    started = time.perf_counter()
    with request_context() as request_id:
        try:
            report = await run_in_threadpool(job)
        except Exception as exc:
            http_exc = _ingest_exception_to_http(exc)
            detail = http_exc.detail if isinstance(http_exc.detail, dict) else {}
            log_event(
                "news",
                "news_ingest",
                level=logging.WARNING,
                status="error",
                error_code=detail.get("error"),
            )
            raise http_exc from exc
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        log_event(
            "news",
            "news_ingest",
            status="locked" if report.skipped_locked else "ok",
            indexed=report.indexed,
            skipped_old=report.skipped_old,
            skipped_unchanged=report.skipped_unchanged,
            skipped_invalid=report.skipped_invalid,
            purged=report.purged,
            purge_failed=report.purge_error is not None,
            failed_feeds=len(report.failed_feeds),
            duration_ms=duration_ms,
        )
        if report.skipped_locked:
            raise api_error(
                409, "INGEST_IN_PROGRESS", "Another news ingestion cycle is already running."
            )
        return {
            "indexed": report.indexed,
            "skipped_old": report.skipped_old,
            "skipped_unchanged": report.skipped_unchanged,
            "skipped_invalid": report.skipped_invalid,
            "purged": report.purged,
            "skipped_locked": report.skipped_locked,
            "failed_feeds": [
                {"url": safe_url(url), "error": _clean_error(error)}
                for url, error in report.failed_feeds
            ],
            "feeds": [
                {**feed, "url": safe_url(feed["url"]), "error": _sanitize_opt(feed["error"])}
                for feed in report.feeds_as_dicts()
            ],
            "purge_error": _sanitize_opt(report.purge_error),
            "duration_ms": duration_ms,
            "request_id": request_id,
        }


# ---------------------------------------------------------------- /news/stats


def get_news_summary_job() -> Callable[[int], dict]:
    """`top -> index summary` (same data as `python -m app.rag.stats`); tests override this."""
    return build_summary_job(get_settings())


@app.get("/news/stats", dependencies=[Depends(require_admin)])
async def news_stats(
    job: Annotated[Callable[[int], dict], Depends(get_news_summary_job)],
    top: Annotated[int, Query(ge=1, le=MAX_TOP)] = DEFAULT_TOP,
) -> dict:
    """Aggregate summary of everything indexed (admin only): counts, never item content."""
    with request_context() as request_id:
        try:
            summary = await run_in_threadpool(job, top)
        except Exception as exc:
            if isinstance(exc, (redis.RedisError, ValueError)) or hasattr(exc, "status_code"):
                http_exc = _ingest_exception_to_http(exc)
            else:
                logger.error("News stats failed", exc_info=exc)
                http_exc = api_error(502, "NEWS_STATS_FAILED", "News index summary failed.")
            detail = http_exc.detail if isinstance(http_exc.detail, dict) else {}
            log_event(
                "news",
                "news_stats",
                level=logging.WARNING,
                status="error",
                error_code=detail.get("error"),
            )
            raise http_exc from exc
        log_event(
            "news",
            "news_stats",
            status="ok",
            total_documents=summary["total_documents"],
            index_exists=summary["index"]["exists"],
            expiring_24h=summary["expiring_24h"],
            sources=len(summary["by_source"]),
            tickers=len(summary["by_ticker"]),
            untagged=summary["untagged"],
            top=top,
        )
        return {**summary, "request_id": request_id}
