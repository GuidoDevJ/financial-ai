"""Daily brief job: weekdays 07:30 America/Argentina/Buenos_Aires, one brief per mock client.

Registered on the same APScheduler instance as the news ingestion, behind ENABLE_SCHEDULER.
Each (client, date) is guarded by a Redis lock (SET NX EX) and skipped when already stored, so
several instances generate a brief only once. Each client is isolated from the others.

TODO: no Argentine holiday calendar; the job also runs on national holidays (weekdays only).
"""

import logging
import uuid
from collections.abc import Callable

import redis

from app.brief.generate import BriefDeps, build_default_deps, default_fecha, run_brief
from app.core.config import Settings
from app.core.redis_client import get_redis_client
from app.tools.get_portfolio import load_mock_portfolios

logger = logging.getLogger(__name__)

BRIEF_JOB_ID = "daily_brief"
BRIEF_TIMEZONE = "America/Argentina/Buenos_Aires"
BRIEF_HOUR, BRIEF_MINUTE = 7, 30
MISFIRE_GRACE_SECONDS = 3600  # still run if the process was briefly down at 07:30


def mock_client_ids() -> list[str]:
    return sorted(load_mock_portfolios())


def lock_key(cliente_id: str, fecha: str) -> str:
    return f"lock:brief:{cliente_id}:{fecha}"


def release_lock(redis_client, key: str, token: str) -> None:
    # Compare-then-delete so we never release a lock that expired and was re-taken.
    try:
        if redis_client.get(key) == token:
            redis_client.delete(key)
    except redis.RedisError:
        logger.warning("could not release brief lock %s", key, exc_info=True)


def _run_one(
    deps: BriefDeps, redis_client, settings: Settings, cliente_id: str, fecha: str, force: bool
) -> dict:
    store = deps.store
    if not force and store is not None and store.exists(cliente_id, fecha):
        return {"cliente_id": cliente_id, "status": "skipped_exists"}
    key, token = lock_key(cliente_id, fecha), uuid.uuid4().hex
    if not redis_client.set(key, token, nx=True, ex=settings.brief_lock_ttl_seconds):
        return {"cliente_id": cliente_id, "status": "skipped_locked"}
    try:
        # Re-check under the lock: another instance may have just finished this brief.
        if not force and store is not None and store.exists(cliente_id, fecha):
            return {"cliente_id": cliente_id, "status": "skipped_exists"}
        result = run_brief(deps, cliente_id, fecha, store=True, send=True)
        return {
            "cliente_id": cliente_id,
            "status": "generated",
            "degraded": result.degraded,
            "missing_inputs": result.missing_inputs,
            "delivery": result.delivery.get("status"),
        }
    finally:
        release_lock(redis_client, key, token)


def run_daily_briefs(
    deps: BriefDeps,
    redis_client,
    settings: Settings,
    *,
    fecha: str | None = None,
    client_ids: list[str] | None = None,
    force: bool = False,
) -> list[dict]:
    """Generate, store and send the brief of every client. Never raises for one client's failure."""
    fecha = fecha or default_fecha(deps.now())
    results = []
    for cliente_id in client_ids if client_ids is not None else mock_client_ids():
        try:
            results.append(_run_one(deps, redis_client, settings, cliente_id, fecha, force))
        except Exception as exc:
            logger.exception("brief failed for %s", cliente_id)
            results.append(
                {
                    "cliente_id": cliente_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def build_brief_job(settings: Settings) -> Callable[[], list[dict]]:
    """Zero-argument job; builds its dependencies lazily on every run and never raises."""

    def job() -> list[dict]:
        try:
            deps = build_default_deps(settings, degrade_on_llm_error=True)
            results = run_daily_briefs(deps, get_redis_client(settings), settings)
        except Exception:
            logger.exception("daily brief job could not start")
            return []
        logger.info("daily brief job finished: %s", results)
        return results

    return job


def register_brief_job(scheduler, job: Callable[[], object]) -> None:
    scheduler.add_job(
        job,
        "cron",
        day_of_week="mon-fri",
        hour=BRIEF_HOUR,
        minute=BRIEF_MINUTE,
        timezone=BRIEF_TIMEZONE,
        id=BRIEF_JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_SECONDS,
    )
