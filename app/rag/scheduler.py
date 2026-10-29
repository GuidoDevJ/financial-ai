"""APScheduler registration for the periodic news ingestion (disabled by default)."""

import logging
from collections.abc import Callable

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import Settings

logger = logging.getLogger(__name__)

JOB_ID = "ingest_news"


def register_ingest_job(scheduler, job: Callable[[], object], interval_minutes: int) -> None:
    """Add (or replace) the interval job. One instance at a time, overlapping runs coalesced."""
    scheduler.add_job(
        job,
        "interval",
        minutes=interval_minutes,
        id=JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )


def maybe_start_scheduler(
    settings: Settings,
    *,
    job: Callable[[], object],
    scheduler_factory: Callable[[], BackgroundScheduler] = BackgroundScheduler,
):
    """Start the scheduler only when ENABLE_SCHEDULER=true; otherwise return None."""
    if not settings.enable_scheduler:
        return None
    scheduler = scheduler_factory()
    register_ingest_job(scheduler, job, settings.ingest_interval_minutes)
    scheduler.start()
    logger.info("ingestion scheduler started (every %s min)", settings.ingest_interval_minutes)
    return scheduler
