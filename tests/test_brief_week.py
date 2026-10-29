"""Acceptance (offline simulation): briefs for all mock clients on every weekday of a week.

The real criterion needs the scheduler running for a week against live sources (pending manual
step in docs/pilot-checklist.md); this proves the job logic holds across consecutive weekdays.
"""

from datetime import date, timedelta

from apscheduler.triggers.cron import CronTrigger

from app.brief.scheduler import (
    BRIEF_HOUR,
    BRIEF_MINUTE,
    BRIEF_TIMEZONE,
    mock_client_ids,
    run_daily_briefs,
)
from app.brief.store import BriefStore
from app.core.config import Settings
from tests.test_brief_scheduler import make_deps, redis_

MONDAY = date(2026, 10, 12)


def test_a_full_week_of_weekday_briefs_generates_for_every_client():
    r = redis_()
    deps = make_deps(r)
    store = BriefStore(r)
    weekdays = [(MONDAY + timedelta(days=i)).isoformat() for i in range(5)]
    for fecha in weekdays:
        results = run_daily_briefs(deps, r, Settings(_env_file=None), fecha=fecha)
        assert {x["status"] for x in results} == {"generated"}, fecha
    for cid in mock_client_ids():
        for fecha in weekdays:
            brief = store.get(cid, fecha)
            assert brief and brief["texto_brief"].strip(), (cid, fecha)
        assert store.latest(cid)["fecha"] == weekdays[-1]


def test_the_cron_fires_on_each_weekday_and_skips_weekends():
    trigger = CronTrigger(
        day_of_week="mon-fri", hour=BRIEF_HOUR, minute=BRIEF_MINUTE, timezone=BRIEF_TIMEZONE
    )
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(BRIEF_TIMEZONE)
    now = datetime(2026, 10, 12, 0, 0, tzinfo=tz)  # Monday
    fired = []
    for _ in range(7):
        nxt = trigger.get_next_fire_time(None, now)
        fired.append(nxt.date().isoformat())
        now = nxt + timedelta(minutes=1)
    assert fired[:5] == [(MONDAY + timedelta(days=i)).isoformat() for i in range(5)]
    assert fired[5] == "2026-10-19"  # Saturday and Sunday skipped
