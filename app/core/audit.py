import contextlib
import contextvars
import functools
import inspect
import json
import logging
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from urllib.parse import urlsplit

from app.tools.errors import ToolError

AUDIT_LOGGER_NAME = "audit.tools"
logger = logging.getLogger(AUDIT_LOGGER_NAME)

REDACTED = "[REDACTED]"
MAX_TEXT = 200

# ------------------------------------------------------------ request context

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
cliente_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "audit_cliente_id", default=None
)
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "audit_session_id", default=None
)
# Per-tool-call scratchpad: tools/caches/sources add facts (cache hit, source, ...) here.
_call_detail: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "audit_call_detail", default=None
)


@contextlib.contextmanager
def request_context(
    request_id: str | None = None,
    cliente_id: str | None = None,
    session_id: str | None = None,
):
    """Bind a correlation id (and optional client/session) for everything run inside.

    An id already bound by an outer entry point is reused, so nested entry points (e.g. a brief
    run started from an endpoint) stay on the same correlation id.
    """
    rid = request_id or request_id_var.get() or uuid.uuid4().hex[:16]
    tokens = [(request_id_var, request_id_var.set(rid))]
    if cliente_id is not None:
        tokens.append((cliente_id_var, cliente_id_var.set(cliente_id)))
    if session_id is not None:
        tokens.append((session_id_var, session_id_var.set(session_id)))
    try:
        yield rid
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def annotate(**fields) -> None:
    """Attach facts to the audit record of the tool call in progress (no-op outside a call)."""
    detail = _call_detail.get()
    if detail is not None:
        detail.update(fields)


# ------------------------------------------------------------------ sanitizer

_SECRET_KEY = re.compile(
    r"pass(word|wd)?|secret|token|api[_-]?key|authorization|credential|access[_-]?key|smtp",
    re.IGNORECASE,
)
# Portfolio contents and free text that must never reach the logs (only sizes are kept).
_HOLDING_KEYS = {
    "cantidad",
    "valorizacion",
    "precio_actual",
    "peso_porcentual",
    "holdings",
    "tenencias",
    "monto",
    "amount",
    "rows",
}
_TEXT_KEYS = {"message", "messages", "texto_brief", "html", "content", "body"}
_SECRET_VALUE = [
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), REDACTED),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer " + REDACTED),
    (
        re.compile(
            r"(?i)\b(pass(?:word|wd)?|pwd|secret|token|api[_-]?key)(\s*[=:]\s*)[^\s,;&\"']+"
        ),
        r"\1\2" + REDACTED,
    ),
]


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...[truncated {len(text) - max_len} chars]"


def _clean_text(text: str, max_len: int) -> str:
    for pattern, repl in _SECRET_VALUE:
        text = pattern.sub(repl, text)
    return _truncate(text, max_len)


def sanitize(value, *, max_len: int = MAX_TEXT, _key: str | None = None):
    """JSON-safe copy of `value` that is safe to log.

    Redacts secret-looking keys and values (AWS key ids, bearer tokens, `password=...`),
    portfolio amounts/holdings and free-text bodies, and truncates long strings.
    """
    if _key is not None:
        lowered = _key.lower()
        if _SECRET_KEY.search(lowered) or lowered in _HOLDING_KEYS:
            return REDACTED
        if lowered in _TEXT_KEYS:
            size = len(value) if hasattr(value, "__len__") else 0
            return f"{REDACTED} ({size} chars)"
    if isinstance(value, dict):
        return {str(k): sanitize(v, max_len=max_len, _key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize(v, max_len=max_len) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clean_text(value, max_len)
    return _clean_text(repr(value), max_len)


# ------------------------------------------------------------- structured events


def redacted_exc_info(exc: BaseException | None = None):
    """`exc_info` triple whose message is scrubbed, so tracebacks cannot leak secrets.

    Keeps the exception class name, module and traceback frames; the message is redacted and
    truncated, and the cause chain is dropped.
    """
    if exc is None:
        exc = sys.exc_info()[1]
    if exc is None:
        return None
    cls = exc.__class__
    clean = type(cls.__name__, (Exception,), {"__module__": cls.__module__})(
        _clean_text(str(exc), 500)
    )
    return (type(clean), clean, exc.__traceback__)


def log_event(
    component: str,
    event: str,
    *,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields,
) -> None:
    """Emit one JSON line on `audit.tools.<component>` (fields are sanitized, id attached)."""
    target = logging.getLogger(f"{AUDIT_LOGGER_NAME}.{component}")
    if not target.isEnabledFor(level):
        return
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "request_id": request_id_var.get(),
        **{k: sanitize(v, _key=k) for k, v in fields.items()},
    }
    target.log(
        level,
        json.dumps(record, default=repr, ensure_ascii=False),
        exc_info=redacted_exc_info() if exc_info else False,
    )


def safe_url(url) -> str:
    """scheme://host/path only: never the query string or credentials."""
    parts = urlsplit(str(url))
    return f"{parts.scheme}://{parts.hostname or ''}{parts.path}"


def log_upstream(
    source: str, operation: str, url, started: float, status: int | None = None, **extra
) -> None:
    """One line per upstream HTTP call: status and latency, never headers or bodies."""
    ok = status < 400 if status is not None else "error_type" not in extra
    log_event(
        "http",
        "upstream",
        level=logging.INFO if ok else logging.WARNING,
        source=source,
        operation=operation,
        url=safe_url(url),
        status=status,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        **extra,
    )


# ------------------------------------------------------------------ tool audit


def _bound_params(func, args, kwargs) -> dict:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        return dict(bound.arguments)
    except TypeError:
        return {"args": list(args), **kwargs}


def _emit(name, params, status, started, detail, result=None, error=None) -> None:
    cliente_id = cliente_id_var.get()
    if cliente_id is None and isinstance(params.get("cliente_id"), str):
        cliente_id = params["cliente_id"]
    if "result_size" not in detail and isinstance(result, (list, dict)):
        detail = {**detail, "result_size": len(result)}
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": "tool_call",
        "tool": name,
        "request_id": request_id_var.get(),
        "cliente_id": cliente_id,
        "session_id": session_id_var.get(),
        "params": sanitize(params),
        "status": status,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "cache": None,
        "source": None,
        "result_size": None,
        **{k: sanitize(v, _key=k) for k, v in detail.items()},
    }
    level, exc_info = logging.INFO, False
    if error is not None:
        record["error"] = _clean_text(f"{type(error).__name__}: {error}", 300)
        record["error_type"] = type(error).__name__
        record["error_code"] = error.code if isinstance(error, ToolError) else None
        if isinstance(error, ToolError):  # expected domain error: no traceback
            level = logging.WARNING
        else:
            level, exc_info = logging.ERROR, redacted_exc_info(error)
    logger.log(level, json.dumps(record, default=repr, ensure_ascii=False), exc_info=exc_info)


def audited_tool(func=None, *, name: str | None = None):
    """Log every tool call as one structured JSON line (sync and async).

    The line carries tool, request/client/session ids, sanitized params, status, error
    code/type, duration, and whatever the tool, cache and sources `annotate()` during the call
    (cache hit/miss, source, result size, ...). Idempotent: an already audited function is
    returned unchanged.
    """
    if func is None:
        return lambda f: audited_tool(f, name=name)
    if getattr(func, "__audited__", False):
        return func
    tool_name = name or func.__name__

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            params = _bound_params(func, args, kwargs)
            detail: dict = {}
            token = _call_detail.set(detail)
            started = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                _emit(tool_name, params, "error", started, detail, error=exc)
                raise
            finally:
                _call_detail.reset(token)
            _emit(tool_name, params, "ok", started, detail, result=result)
            return result

        async_wrapper.__audited__ = True
        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        params = _bound_params(func, args, kwargs)
        detail: dict = {}
        token = _call_detail.set(detail)
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            _emit(tool_name, params, "error", started, detail, error=exc)
            raise
        finally:
            _call_detail.reset(token)
        _emit(tool_name, params, "ok", started, detail, result=result)
        return result

    wrapper.__audited__ = True
    return wrapper


BRIEF_AUDIT_LOGGER_NAME = "audit.brief"
brief_logger = logging.getLogger(BRIEF_AUDIT_LOGGER_NAME)


def prompt_fields(prompt) -> dict:
    """Audit fields for the prompt that produced an output (`PromptVersion` or None)."""
    if prompt is None:
        return {}
    return {
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_hash": prompt.hash,
    }


def log_brief_run(**fields) -> None:
    """Log one JSON line per brief run (missing inputs, degraded, delivery, LLM usage)."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": "brief_run",
        "request_id": request_id_var.get(),
        **fields,
    }
    brief_logger.info(json.dumps(record, default=repr, ensure_ascii=False))


CHAT_AUDIT_LOGGER_NAME = "audit.chat"
chat_logger = logging.getLogger(CHAT_AUDIT_LOGGER_NAME)


def log_chat_request(*, duration_ms: float, **fields) -> None:
    """One JSON line per /chat turn: ids, status, duration, tool names; never the message text."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": "chat_request",
        "request_id": request_id_var.get(),
        "duration_ms": duration_ms,
        **{k: sanitize(v, _key=k) for k, v in fields.items()},
    }
    chat_logger.info(json.dumps(record, default=repr, ensure_ascii=False))


LLM_AUDIT_LOGGER_NAME = "audit.llm"
llm_logger = logging.getLogger(LLM_AUDIT_LOGGER_NAME)


def log_llm_call(message, *, cliente_id: str | None, session_id: str | None, prompt=None) -> None:
    """Log one structured JSON line per Converse response (stop reason, tokens, latency).

    `message` is the AIMessage returned by ChatBedrockConverse. LangChain folds cache tokens
    into `input_tokens`, so the uncached Converse `inputTokens` is recovered here. `prompt` is the
    `PromptVersion` used, logged as prompt_name / prompt_version / prompt_hash.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    cache_read = details.get("cache_read", 0)
    cache_write = details.get("cache_creation", 0)
    meta = getattr(message, "response_metadata", None) or {}
    latency = (meta.get("metrics") or {}).get("latencyMs")
    if isinstance(latency, list):
        latency = latency[0] if latency else None
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": "llm_call",
        "request_id": request_id_var.get(),
        "cliente_id": cliente_id,
        "session_id": session_id,
        "stop_reason": meta.get("stopReason"),
        "input_tokens": max(usage.get("input_tokens", 0) - cache_read - cache_write, 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cache_read_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
        "latency_ms": latency,
        **prompt_fields(prompt),
    }
    llm_logger.info(json.dumps(record, ensure_ascii=False))
