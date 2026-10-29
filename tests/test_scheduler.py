from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.rag.scheduler import JOB_ID, maybe_start_scheduler, register_ingest_job


def test_scheduler_disabled_by_default():
    assert Settings(_env_file=None).enable_scheduler is False
    assert Settings(_env_file=None).ingest_interval_minutes == 90


def test_maybe_start_returns_none_when_disabled():
    started = []
    result = maybe_start_scheduler(
        Settings(_env_file=None), job=lambda: started.append(1), scheduler_factory=lambda: 1 / 0
    )
    assert result is None and started == []


def test_enabled_registers_interval_job_and_starts():
    s = Settings(_env_file=None, enable_scheduler=True, ingest_interval_minutes=60)
    sched = maybe_start_scheduler(s, job=lambda: None, scheduler_factory=BackgroundScheduler)
    try:
        assert sched.running
        job = sched.get_job(JOB_ID)
        assert job.trigger.interval.total_seconds() == 3600
    finally:
        sched.shutdown(wait=False)


def test_register_ingest_job_is_idempotent_and_single_instance():
    sched = BackgroundScheduler()
    sched.start(paused=True)  # replace_existing only applies once the scheduler is running
    try:
        register_ingest_job(sched, lambda: None, interval_minutes=30)
        register_ingest_job(sched, lambda: None, interval_minutes=45)
        jobs = sched.get_jobs()
        assert len(jobs) == 1 and jobs[0].max_instances == 1
        assert jobs[0].trigger.interval.total_seconds() == 45 * 60
    finally:
        sched.shutdown(wait=False)


def test_app_lifespan_does_not_start_scheduler_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_SCHEDULER", raising=False)
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        with TestClient(app):
            assert getattr(app.state, "scheduler", None) is None
    finally:
        get_settings.cache_clear()


def test_app_lifespan_starts_and_stops_scheduler_when_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_SCHEDULER", "true")
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr("app.main.build_ingest_job", lambda settings: lambda: None)
    try:
        with TestClient(app):
            sched = app.state.scheduler
            assert sched is not None and sched.running
        assert not sched.running
    finally:
        get_settings.cache_clear()
        app.state.scheduler = None
