"""Daily brief generation: a LangGraph pipeline run by the job (never by the LLM).

    load_portfolio -> [prices | macro | news] (parallel) -> generate -> validate
        -> (generate once more | fallback) -> assemble -> store -> send

Tools are called here, by the job. The generation prompt is non-conversational: the model gets
the already-resolved data (rendered into the system prompt's `{{variables}}`; the user turn is a
short trigger), has no tools bound, and writes ONLY the narrative. The price table
and the disclaimer are rendered deterministically from the data (`app.brief.render`).

CLI: `uv run python -m app.brief.generate --cliente client_001 [--fecha YYYY-MM-DD]
    [--store] [--send]`
"""

import argparse
import json
import logging
import operator
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict
from zoneinfo import ZoneInfo

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents.llm import build_chat_model, build_request_metadata, ensure_llm_configured
from app.brief.delivery import send_brief_email
from app.brief.render import AssetRow, compose_text, render_html
from app.brief.store import BriefStore, validate_client_id, validate_fecha
from app.brief.validate import validate_final_text, validate_narrative
from app.core.audit import log_brief_run, log_llm_call, request_context
from app.core.concurrency import map_in_threads
from app.core.config import Settings, get_settings
from app.core.redis_client import get_redis_client
from app.prompts.registry import PromptError, PromptSource, brief_prompt_source
from app.tools.errors import ToolError
from app.tools.get_macro import default_macro_providers, make_get_macro
from app.tools.get_portfolio import CURRENCY, make_get_portfolio
from app.tools.get_price import make_get_price
from app.tools.search_news import make_search_news

logger = logging.getLogger(__name__)

BRIEF_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
MACRO_VARIABLES = ("dolar_ccl", "dolar_mep", "riesgo_pais")
NEWS_WINDOW_HOURS = 24
MAX_ATTEMPTS = 2  # first generation + one retry with the violation list
UNAVAILABLE = "no disponible"
NO_NEWS = "sin noticias en las últimas 24 horas"
_MARKET_BY_CURRENCY = {currency: market for market, currency in CURRENCY.items()}


class ClientNotFoundError(LookupError):
    """The client has no portfolio."""


class BriefInputError(RuntimeError):
    """The portfolio could not be loaded, so no brief can be built."""


@dataclass
class BriefDeps:
    """Everything the pipeline needs; every dependency is injectable for tests."""

    get_portfolio: Callable[..., list]
    get_price: Callable[..., dict]
    get_macro: Callable[..., dict]
    search_news: Callable[..., list]
    model: Any  # chat model WITHOUT tools bound
    settings: Settings
    prompt_source: PromptSource  # system prompt, read on every generation (cached)
    store: BriefStore | None = None
    send_email: Callable[..., dict] = send_brief_email
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    # Scheduled jobs prefer a data-only brief over no brief when the LLM is unreachable.
    degrade_on_llm_error: bool = False


@dataclass
class BriefResult:
    cliente_id: str
    fecha: str
    texto_brief: str
    html: str
    activos_mencionados: list[str]
    degraded: bool
    missing_inputs: list[str]
    generated_at: str
    attempts: int = 0
    violations: list[str] = field(default_factory=list)
    stored: bool = False
    delivery: dict = field(default_factory=lambda: {"status": "not_requested"})
    duration_ms: float = 0.0
    prompt: dict[str, str] | None = None  # {name, version, hash} of the system prompt used

    def preview(self) -> dict:
        """The /brief/preview response body."""
        return {
            "fecha": self.fecha,
            "texto_brief": self.texto_brief,
            "html": self.html,
            "activos_mencionados": self.activos_mencionados,
            "degraded": self.degraded,
            "missing_inputs": self.missing_inputs,
        }


class BriefState(TypedDict, total=False):
    cliente_id: str
    fecha: str
    do_store: bool
    do_send: bool
    holdings: list[dict]
    prices: dict[str, dict]
    macro: dict[str, dict]
    news: dict[str, list[dict] | None]
    missing_inputs: Annotated[list[str], operator.add]
    llm_usage: Annotated[list[dict], operator.add]
    prompt_data: dict
    prompt: dict[str, str]
    narrative: str | None
    attempts: int
    violations: list[str]
    llm_error: str | None
    degraded: bool
    texto_brief: str
    html: str
    activos_mencionados: list[str]
    generated_at: str
    stored: bool
    delivery: dict


# ------------------------------------------------------------------ helpers


def default_fecha(now: datetime) -> str:
    return now.astimezone(BRIEF_TZ).date().isoformat()


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def _market_of_row(row: dict) -> str:
    if row.get("moneda") in _MARKET_BY_CURRENCY:
        return _MARKET_BY_CURRENCY[row["moneda"]]
    return "byma" if row["ticker"].upper().endswith(".BA") else "nyse"


def _asset_rows(state: BriefState) -> list[AssetRow]:
    rows = []
    for h in state["holdings"]:
        p = state["prices"].get(h["ticker"]) or {}
        rows.append(
            AssetRow(
                ticker=h["ticker"],
                moneda=h["moneda"],
                mercado=h["mercado"],
                peso_pct=h.get("peso_pct"),
                precio=p.get("precio_cierre"),
                fecha_precio=p.get("fecha"),
                fuente=p.get("fuente"),
                var_1d=p.get("variacion_dia"),
                var_5d=p.get("variacion_semana"),
                var_mes=p.get("variacion_mes"),
            )
        )
    return rows


def build_prompt_data(fecha: str, rows: list[AssetRow], state: BriefState) -> dict:
    """Resolved data for the model; anything unavailable is the explicit string 'no disponible'."""

    def v(value):
        return UNAVAILABLE if value is None else value

    cartera = [
        {
            "ticker": r.ticker,
            "moneda": r.moneda,
            "peso_pct": v(r.peso_pct),
            "precio": v(r.precio),
            "fecha_precio": v(r.fecha_precio),
            "fuente": v(r.fuente),
            "var_1d_pct": v(r.var_1d),
            "var_5d_pct": v(r.var_5d),
            "var_mes_pct": v(r.var_mes),
        }
        for r in rows
    ]
    macro = {}
    for name in MACRO_VARIABLES:
        item = state["macro"].get(name)
        macro[name] = (
            UNAVAILABLE
            if item is None
            else {
                "valor": item.get("valor"),
                "fecha": item.get("fecha"),
                "fuente": item.get("fuente"),
            }
        )
    noticias = {}
    for r in rows:
        items = state["news"].get(r.ticker)
        if items is None:
            noticias[r.ticker] = UNAVAILABLE
        elif not items:
            noticias[r.ticker] = NO_NEWS
        else:
            noticias[r.ticker] = [
                {k: n.get(k, "") for k in ("titulo", "resumen", "fuente", "fecha")} for n in items
            ]
    return {
        "fecha": fecha,
        "ventana_noticias_horas": NEWS_WINDOW_HOURS,
        "cartera": cartera,
        "macro": macro,
        "noticias": noticias,
    }


BRIEF_TRIGGER = "Generá el brief."


def _json_block(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def brief_prompt_variables(data: dict, bank_name: str) -> dict[str, str]:
    """Values of the brief prompt's `{{variables}}`: the resolved data as compact JSON blocks.

    Derived from the same `data` the validator checks the narrative against, so every number
    the model sees is traceable.
    """
    cartera = [{k: row[k] for k in ("ticker", "moneda", "peso_pct")} for row in data["cartera"]]
    precios = [
        {
            k: row[k]
            for k in (
                "ticker",
                "precio",
                "fecha_precio",
                "fuente",
                "var_1d_pct",
                "var_5d_pct",
                "var_mes_pct",
            )
        }
        for row in data["cartera"]
    ]
    return {
        "banco": bank_name,
        "fecha": data["fecha"],
        "cartera": _json_block(cartera),
        "precios": _json_block(precios),
        "macro": _json_block(data["macro"]),
        "noticias": _json_block(data["noticias"]),
    }


def _user_message(violations: list[str]) -> str:
    """Short fixed trigger; the data already travels in the system prompt."""
    if not violations:
        return BRIEF_TRIGGER
    listing = "\n".join(f"- {v}" for v in violations)
    return (
        f"{BRIEF_TRIGGER}\n\nTu texto anterior fue rechazado por estas violaciones:\n"
        f"{listing}\nEscribí de nuevo los ítems narrativos corrigiéndolas."
    )


# ----------------------------------------------------------------- pipeline


def build_brief_graph(deps: BriefDeps):
    settings = deps.settings

    def load_portfolio(state: BriefState) -> dict:
        cliente_id = state["cliente_id"]
        try:
            rows = deps.get_portfolio(cliente_id=cliente_id)
        except ToolError as exc:
            if exc.code == "CLIENT_NOT_FOUND":
                raise ClientNotFoundError(f"No portfolio found for client {cliente_id}.") from exc
            raise BriefInputError(f"{exc.code}: {exc.message}") from exc
        holdings, seen = [], set()
        for row in rows:
            if row["ticker"] in seen:
                continue
            seen.add(row["ticker"])
            market = _market_of_row(row)
            holdings.append(
                {
                    "ticker": row["ticker"],
                    "mercado": market,
                    "moneda": CURRENCY[market],
                    "peso_pct": row.get("peso_porcentual"),
                }
            )
        if not holdings:
            raise BriefInputError(f"Client {cliente_id} has an empty portfolio.")
        return {"holdings": holdings}

    def prices(state: BriefState) -> dict:
        # Not reused from get_portfolio (it keeps only the close, the brief needs the
        # variations and source too); get_price is cached, so this is cheap after the portfolio.
        def fetch(h: dict) -> tuple[str, dict | None]:
            try:
                return h["ticker"], deps.get_price(ticker=h["ticker"], mercado=h["mercado"])
            except Exception as exc:
                logger.warning("price unavailable for %s: %s", h["ticker"], exc)
                return h["ticker"], None

        found, missing = {}, []
        for ticker, price in map_in_threads(fetch, state["holdings"]):
            if price is None:
                missing.append(f"price:{ticker}")
            else:
                found[ticker] = price
        return {"prices": found, "missing_inputs": missing}

    def macro(state: BriefState) -> dict:
        found, missing = {}, []
        for name in MACRO_VARIABLES:
            try:
                found[name] = deps.get_macro(variable=name)
            except Exception as exc:
                logger.warning("macro unavailable for %s: %s", name, exc)
                missing.append(f"macro:{name}")
        return {"macro": found, "missing_inputs": missing}

    def news(state: BriefState) -> dict:
        # Embedding + vector search per ticker are independent: run them concurrently.
        def search(h: dict) -> tuple[str, list[dict] | None]:
            ticker = h["ticker"]
            try:
                return ticker, deps.search_news(
                    query=ticker.removesuffix(".BA"), ticker=ticker, horas=NEWS_WINDOW_HOURS
                )
            except ToolError as exc:
                # NO_NEWS_FOUND is a valid empty result, not a failure
                return ticker, [] if exc.code == "NO_NEWS_FOUND" else None
            except Exception as exc:
                logger.warning("news unavailable for %s: %s", ticker, exc)
                return ticker, None

        found: dict[str, list[dict] | None] = {}
        missing = []
        for ticker, items in map_in_threads(search, state["holdings"]):
            found[ticker] = items
            if items is None:
                missing.append(f"news:{ticker}")
        return {"news": found, "missing_inputs": missing}

    def generate(state: BriefState) -> dict:
        data = build_prompt_data(state["fecha"], _asset_rows(state), state)
        attempts = state.get("attempts", 0) + 1
        # Outside the LLM try/except below: a prompt failure must surface, not degrade silently.
        prompt_version = deps.prompt_source.get()
        prompt_meta = prompt_version.metadata()  # identifies the raw template, not the render
        messages = [
            SystemMessage(
                content=prompt_version.render(**brief_prompt_variables(data, settings.bank_name))
            ),
            HumanMessage(content=_user_message(state.get("violations", []))),
        ]
        metadata = build_request_metadata(
            {"app": settings.app_name, "env": settings.app_env},
            state["cliente_id"],
            f"brief-{state['fecha']}",
        )
        try:
            # Blocking call: /brief and /brief/preview are sync endpoints, so it occupies one
            # threadpool worker (not the event loop); the CLI and the job share this path.
            response = deps.model.invoke(messages, request_metadata=metadata)
        except Exception as exc:
            if not deps.degrade_on_llm_error:
                raise
            logger.error("LLM unavailable, degrading brief: %s", exc)
            return {
                "prompt_data": data,
                "prompt": prompt_meta,
                "attempts": attempts,
                "narrative": None,
                "llm_error": f"{type(exc).__name__}: {exc}",
            }
        log_llm_call(
            response,
            cliente_id=state["cliente_id"],
            session_id=f"brief-{state['fecha']}",
            prompt=prompt_version,
        )
        usage = getattr(response, "usage_metadata", None) or {}
        return {
            "prompt_data": data,
            "prompt": prompt_meta,
            "attempts": attempts,
            "narrative": _content_text(response.content).strip(),
            "llm_usage": [
                {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }
            ],
        }

    def validate(state: BriefState) -> dict:
        if state.get("narrative") is None:
            return {"violations": ["LLM unavailable"]}
        narrative = state["narrative"]
        violations = validate_narrative(narrative, state["prompt_data"])
        violations += validate_final_text(compose_text(narrative, _asset_rows(state)))
        return {"violations": violations}

    def route_after_validate(state: BriefState) -> str:
        if not state["violations"]:
            return "assemble"
        if state.get("narrative") is not None and state["attempts"] < MAX_ATTEMPTS:
            return "generate"
        return "fallback"

    def fallback(state: BriefState) -> dict:
        logger.warning(
            "brief for %s degraded to data-only fallback: %s",
            state["cliente_id"],
            state["violations"],
        )
        return {"narrative": "", "degraded": True}

    def assemble(state: BriefState) -> dict:
        rows = _asset_rows(state)
        narrative = state.get("narrative") or ""
        degraded = state.get("degraded", False)
        return {
            "texto_brief": compose_text(narrative, rows),
            "html": render_html(state["fecha"], narrative, rows, degraded),
            "activos_mencionados": [r.ticker for r in rows],
            "degraded": degraded,
            "generated_at": deps.now().isoformat(),
            "missing_inputs": [],  # reducer: nothing to add
        }

    def store(state: BriefState) -> dict:
        if not (state.get("do_store") and deps.store):
            return {"stored": False}
        deps.store.save(
            state["cliente_id"],
            state["fecha"],
            state["texto_brief"],
            state["activos_mencionados"],
            html=state["html"],
            degraded=state["degraded"],
            missing_inputs=sorted(state.get("missing_inputs", [])),
            generated_at=state["generated_at"],
            prompt=state.get("prompt"),
        )
        return {"stored": True}

    def send(state: BriefState) -> dict:
        if not state.get("do_send"):
            return {"delivery": {"status": "not_requested"}}
        try:
            delivery = deps.send_email(
                settings,
                fecha=state["fecha"],
                texto=state["texto_brief"],
                html=state["html"],
            )
        except Exception as exc:  # delivery must never fail a stored brief
            delivery = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            logger.warning("brief delivery failed: %s", delivery["error"])
        return {"delivery": delivery}

    builder = StateGraph(BriefState)
    for name, node in [
        ("load_portfolio", load_portfolio),
        ("prices", prices),
        ("macro", macro),
        ("news", news),
        ("generate", generate),
        ("validate", validate),
        ("fallback", fallback),
        ("assemble", assemble),
        ("store", store),
        ("send", send),
    ]:
        builder.add_node(name, node)
    builder.add_edge(START, "load_portfolio")
    for branch in ("prices", "macro", "news"):
        builder.add_edge("load_portfolio", branch)
    builder.add_edge(["prices", "macro", "news"], "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"assemble": "assemble", "generate": "generate", "fallback": "fallback"},
    )
    builder.add_edge("fallback", "assemble")
    builder.add_edge("assemble", "store")
    builder.add_edge("store", "send")
    builder.add_edge("send", END)
    return builder.compile()


def run_brief(
    deps: BriefDeps,
    cliente_id: str,
    fecha: str | None = None,
    *,
    store: bool = False,
    send: bool = False,
) -> BriefResult:
    """Run the pipeline for one client. `fecha` only labels the brief (data is always latest)."""
    validate_client_id(cliente_id)
    fecha = fecha or default_fecha(deps.now())
    validate_fecha(fecha)
    started = time.perf_counter()
    try:
        # Reuses the id bound by /brief/preview; scheduler and CLI runs get a fresh one.
        with request_context(cliente_id=cliente_id):
            state = build_brief_graph(deps).invoke(
                {
                    "cliente_id": cliente_id,
                    "fecha": fecha,
                    "do_store": store,
                    "do_send": send,
                    "missing_inputs": [],
                    "llm_usage": [],
                }
            )
    except Exception as exc:
        log_brief_run(
            status="error",
            cliente_id=cliente_id,
            fecha=fecha,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    usage = state.get("llm_usage", [])
    result = BriefResult(
        cliente_id=cliente_id,
        fecha=fecha,
        texto_brief=state["texto_brief"],
        html=state["html"],
        activos_mencionados=state["activos_mencionados"],
        degraded=state["degraded"],
        missing_inputs=sorted(state.get("missing_inputs", [])),
        generated_at=state["generated_at"],
        attempts=state.get("attempts", 0),
        violations=state.get("violations", []),
        stored=state.get("stored", False),
        delivery=state.get("delivery", {"status": "not_requested"}),
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
        prompt=state.get("prompt"),
    )
    log_brief_run(
        status="ok",
        cliente_id=cliente_id,
        fecha=fecha,
        duration_ms=result.duration_ms,
        missing_inputs=result.missing_inputs,
        degraded=result.degraded,
        attempts=result.attempts,
        stored=result.stored,
        delivery=result.delivery.get("status"),
        llm={
            "calls": len(usage),
            "input_tokens": sum(u["input_tokens"] for u in usage),
            "output_tokens": sum(u["output_tokens"] for u in usage),
        },
        llm_error=state.get("llm_error"),
        **({f"prompt_{k}": v for k, v in result.prompt.items()} if result.prompt else {}),
    )
    return result


# --------------------------------------------------------- production wiring


def build_default_deps(settings: Settings | None = None, **overrides) -> BriefDeps:
    """Production wiring: real tools + Bedrock model (no tools bound) + Redis brief store.

    Raises LLMConfigError if the Bedrock configuration is incomplete and PromptError
    (PROMPT_ID_MISSING) if BEDROCK_PROMPT_BRIEF_ID is unset. Makes no network call.
    """
    settings = settings or get_settings()
    ensure_llm_configured(settings)
    prompt_source = overrides.pop("prompt_source", None) or brief_prompt_source(settings)
    redis_client = get_redis_client(settings)
    http = httpx.Client(timeout=8, follow_redirects=True)
    get_price = make_get_price(redis_client=redis_client, settings=settings)
    get_macro = make_get_macro(
        redis_client=redis_client, settings=settings, providers=default_macro_providers(http)
    )
    deps = BriefDeps(
        get_portfolio=make_get_portfolio(get_price=get_price, get_macro=get_macro),
        get_price=get_price,
        get_macro=get_macro,
        search_news=make_search_news(settings=settings, redis_client=redis_client),
        model=build_chat_model(settings),
        settings=settings,
        prompt_source=prompt_source,
        store=BriefStore(redis_client),
    )
    for key, value in overrides.items():
        setattr(deps, key, value)
    return deps


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None, *, deps_factory=build_default_deps) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.brief.generate")
    parser.add_argument("--cliente", required=True, help="client id, e.g. client_001")
    parser.add_argument("--fecha", help="brief date YYYY-MM-DD (default: today in Buenos Aires)")
    parser.add_argument("--store", action="store_true", help="save the brief in Redis")
    parser.add_argument("--send", action="store_true", help="send the (simulated) email")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    try:
        validate_client_id(args.cliente)
        if args.fecha:
            validate_fecha(args.fecha)
        deps = deps_factory(get_settings(), degrade_on_llm_error=args.store)
        result = run_brief(deps, args.cliente, args.fecha, store=args.store, send=args.send)
    except ClientNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    except PromptError as exc:
        print(f"ERROR: {exc.code}: {exc.message}")
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2

    print(result.texto_brief)
    print()
    print(
        json.dumps(
            {
                **{
                    k: v
                    for k, v in asdict(result).items()
                    if k
                    in ("cliente_id", "fecha", "degraded", "missing_inputs", "stored", "delivery")
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
