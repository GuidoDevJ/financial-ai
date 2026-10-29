import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.core.audit import AUDIT_LOGGER_NAME, annotate, log_event


def _ttl_remaining(redis_client, key: str) -> int | None:
    try:
        return int(redis_client.ttl(key))
    except Exception:
        return None


STALE_SUFFIX = ":stale"


def _stale_answer(redis_client, key: str, exc: Exception) -> dict | None:
    """Last-known value of `key` flagged stale, or None when there is none (or Redis is down)."""
    try:
        raw = redis_client.get(key + STALE_SUFFIX)
        entry = json.loads(raw) if raw is not None else None
    except Exception:
        return None
    value = entry.get("value") if isinstance(entry, dict) else None
    if not isinstance(value, dict):
        return None
    cached_at = str(entry.get("cached_at", ""))
    reason = str(getattr(exc, "code", type(exc).__name__))
    annotate(stale=True)
    log_event(
        "cache",
        "stale_served",
        level=logging.WARNING,
        key=key,
        reason=reason,
        cached_at=cached_at,
    )
    return {
        **value,
        "stale": True,
        "as_of": value.get("fecha") or cached_at[:10],
        "cached_at": cached_at,
        "stale_reason": reason,
    }


def cached_call(
    redis_client,
    key: str,
    ttl: int,
    compute: Callable[[], Any],
    *,
    stale_ttl: int | None = None,
) -> Any:
    """Return the cached JSON value for `key`, else compute and cache it.

    Only successful results are cached (exceptions propagate before the write). Redis failures
    degrade to a direct computation instead of failing the tool. Hit/miss/failure is logged on
    `audit.tools.cache` and recorded on the surrounding tool-call audit line.

    With `stale_ttl`, every success also leaves a longer-lived copy under `<key>:stale`. If
    `compute` later fails and that copy exists, it is returned flagged `stale: true` with its
    as-of date instead of the error (never promoted back to the fresh cache).
    """
    cache_state = "miss"
    try:
        raw = redis_client.get(key)
        if raw is not None:
            value = json.loads(raw)
            annotate(cache="hit", cache_key=key, cache_ttl=ttl)
            # The extra TTL round trip only happens when the DEBUG line will actually be logged.
            if logging.getLogger(f"{AUDIT_LOGGER_NAME}.cache").isEnabledFor(logging.DEBUG):
                log_event(
                    "cache",
                    "cache",
                    level=logging.DEBUG,
                    result="hit",
                    key=key,
                    ttl=ttl,
                    ttl_remaining=_ttl_remaining(redis_client, key),
                )
            return value
    except Exception as exc:
        cache_state = "error"
        log_event(
            "cache",
            "cache",
            level=logging.WARNING,
            exc_info=True,
            result="read_error",
            key=key,
            error_type=type(exc).__name__,
        )

    annotate(cache=cache_state, cache_key=key, cache_ttl=ttl)
    if cache_state == "miss":
        log_event("cache", "cache", level=logging.DEBUG, result="miss", key=key, ttl=ttl)

    try:
        value = compute()
    except Exception as exc:
        stale = _stale_answer(redis_client, key, exc) if stale_ttl else None
        if stale is None:
            raise
        return stale

    try:
        redis_client.setex(key, ttl, json.dumps(value))
        if stale_ttl:
            entry = {"cached_at": datetime.now(UTC).isoformat(), "value": value}
            redis_client.setex(key + STALE_SUFFIX, stale_ttl, json.dumps(entry))
    except Exception as exc:
        log_event(
            "cache",
            "cache",
            level=logging.WARNING,
            exc_info=True,
            result="write_error",
            key=key,
            error_type=type(exc).__name__,
        )
    return value
