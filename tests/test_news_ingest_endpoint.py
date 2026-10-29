import hmac
import json
import logging
import threading

import fakeredis
import pytest
import redis
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from app.core import auth
from app.core.config import Settings
from app.main import app, get_ingest_job
from app.rag.ingest_news import INGEST_LOCK_KEY, FeedStat, IngestReport, run_ingestion

ADMIN = "admin-secret-key"
CLIENT_KEYS = '{"key-one": "c1"}'
URLS = "https://feed.example/rss"


def teardown_function():
    app.dependency_overrides.clear()


def client_with(job) -> TestClient:
    app.dependency_overrides[get_ingest_job] = lambda: job
    return TestClient(app)


def ok_report(**kw) -> IngestReport:
    return IngestReport(indexed=3, skipped_old=1, purged=2, **kw)


def ingest(client, key=None):
    return client.post("/news/ingest", headers={"X-API-Key": key} if key else {})


@pytest.fixture(autouse=True)
def _urls(set_env):
    set_env(news_rss_urls=URLS)


# ------------------------------------------------------------------ auth matrix


def test_dev_without_any_keys_is_allowed():
    assert ingest(client_with(ok_report)).status_code == 200


@pytest.mark.parametrize("env", ["prod", "staging"])
def test_outside_dev_without_admin_keys_is_503(set_env, env):
    set_env(app_env=env, auth_api_keys=CLIENT_KEYS)
    r = ingest(client_with(ok_report), "key-one")
    assert r.status_code == 503 and r.json()["error"]["code"] == "ADMIN_NOT_CONFIGURED"


def test_missing_key_is_401_when_admin_keys_are_set(set_env):
    set_env(admin_api_keys=ADMIN)
    r = ingest(client_with(ok_report))
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHENTICATED"


def test_unknown_key_is_401(set_env):
    set_env(admin_api_keys=ADMIN, auth_api_keys=CLIENT_KEYS)
    r = ingest(client_with(ok_report), "nope")
    assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_API_KEY"


def test_client_key_is_403_not_admin(set_env):
    set_env(admin_api_keys=ADMIN, auth_api_keys=CLIENT_KEYS)
    r = ingest(client_with(ok_report), "key-one")
    assert r.status_code == 403 and r.json()["error"]["code"] == "ADMIN_REQUIRED"


@pytest.mark.parametrize("raw", [ADMIN, f"other, {ADMIN}", json.dumps(["other", ADMIN])])
def test_admin_key_is_accepted_in_csv_or_json(set_env, raw):
    set_env(admin_api_keys=raw, auth_api_keys=CLIENT_KEYS)
    assert ingest(client_with(ok_report), ADMIN).status_code == 200


def test_dev_without_admin_keys_still_requires_a_client_key(set_env):
    set_env(auth_api_keys=CLIENT_KEYS)
    c = client_with(ok_report)
    assert ingest(c).status_code == 401
    assert ingest(c, "key-one").status_code == 200


def test_admin_keys_are_compared_in_constant_time(set_env, monkeypatch):
    set_env(admin_api_keys="a1,a2,a3")
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    ingest(client_with(ok_report), "a1")
    assert len(calls) == 3


def test_invalid_admin_keys_setting_fails_fast_without_echoing_it():
    with pytest.raises(ValueError) as err:
        Settings(_env_file=None, admin_api_keys='["ok", 5]')
    assert "ok" not in str(err.value).replace("ADMIN_API_KEYS", "")


def test_startup_warns_in_dev_without_admin_keys(caplog):
    caplog.set_level(logging.WARNING)
    with TestClient(app):
        pass
    assert any("ADMIN_API_KEYS" in r.getMessage() for r in caplog.records)


def test_startup_is_quiet_with_admin_keys(set_env, caplog):
    set_env(admin_api_keys=ADMIN)
    caplog.set_level(logging.WARNING)
    with TestClient(app):
        pass
    assert not any("ADMIN_API_KEYS" in r.getMessage() for r in caplog.records)


# -------------------------------------------------------------------- behaviour


def test_success_shape_and_request_id():
    report = ok_report(failed_feeds=[("https://f.example/rss?token=abc#x", "HTTPStatusError: 500")])
    r = ingest(client_with(lambda: report))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "indexed",
        "skipped_old",
        "skipped_unchanged",
        "skipped_invalid",
        "purged",
        "purge_error",
        "feeds",
        "skipped_locked",
        "failed_feeds",
        "duration_ms",
        "request_id",
    }
    assert (body["indexed"], body["skipped_old"], body["purged"]) == (3, 1, 2)
    assert body["skipped_locked"] is False and body["duration_ms"] >= 0
    assert body["request_id"] == r.headers["x-request-id"]
    assert body["failed_feeds"] == [
        {"url": "https://f.example/rss", "error": "HTTPStatusError: 500"}
    ]


def test_new_counters_feed_stats_and_purge_error_in_response():
    report = ok_report(
        skipped_unchanged=7,
        skipped_invalid=2,
        purge_error="ConnectionError: redis://:pw@h down",
        feeds=[
            FeedStat(
                url="https://f.example/rss?token=abc",
                status="failed",
                fetched=5,
                invalid=1,
                latency_ms=12.5,
                error="InvalidFeed: see https://f.example/x?apikey=SECRET1",
            )
        ],
    )
    body = ingest(client_with(lambda: report)).json()
    assert (body["skipped_unchanged"], body["skipped_invalid"]) == (7, 2)
    [feed] = body["feeds"]
    assert feed["url"] == "https://f.example/rss" and feed["status"] == "failed"
    assert (feed["fetched"], feed["invalid"], feed["latency_ms"]) == (5, 1, 12.5)
    assert set(feed) == {
        "url", "status", "fetched", "new", "unchanged", "old", "invalid", "indexed",
        "latency_ms", "error",
    }  # fmt: skip
    assert "SECRET1" not in feed["error"] and "pw@h" not in body["purge_error"]


def test_audit_event_includes_new_counters(set_env, caplog):
    caplog.set_level(logging.INFO)
    report = ok_report(skipped_unchanged=7, skipped_invalid=2)
    ingest(client_with(lambda: report))
    lines = [r.getMessage() for r in caplog.records if r.name.startswith("audit.tools.news")]
    event = json.loads(next(line for line in lines if "news_ingest" in line))
    assert (event["skipped_unchanged"], event["skipped_invalid"]) == (7, 2)


def test_failed_feed_errors_are_sanitized_and_truncated():
    nasty = "ConnectError: https://u:p@f.example/x?apikey=SECRET1 token=SECRET2 " + "z" * 1000
    r = ingest(client_with(lambda: IngestReport(failed_feeds=[("https://f.example/x", nasty)])))
    error = r.json()["failed_feeds"][0]["error"]
    assert "SECRET1" not in error and "SECRET2" not in error and "u:p" not in error
    assert len(error) < 400


def test_job_runs_off_the_event_loop_thread():
    seen = {}

    def job():
        seen["thread"] = threading.current_thread()
        return ok_report()

    ingest(client_with(job))
    assert seen["thread"] is not threading.main_thread()


def test_lock_held_is_409_with_standard_body():
    fake = fakeredis.FakeRedis(decode_responses=True)
    fake.set(INGEST_LOCK_KEY, "someone-else")

    def job():
        return run_ingestion(Settings(_env_file=None), redis_client=fake, store=None)

    r = ingest(client_with(job))
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "INGEST_IN_PROGRESS" and err["request_id"] and err["message"]
    assert fake.get(INGEST_LOCK_KEY) == "someone-else"  # not released by the loser


def test_empty_rss_urls_is_422_and_does_not_run(set_env):
    set_env(news_rss_urls="")
    ran = []
    r = ingest(client_with(lambda: ran.append(1)))
    assert r.status_code == 422 and r.json()["error"]["code"] == "NEWS_RSS_URLS_EMPTY"
    assert not ran


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (redis.ConnectionError("boom redis://:pw@h"), 503, "STORAGE_UNAVAILABLE"),
        (ValueError("Unknown vector size"), 503, "NEWS_CONFIG_INVALID"),
        (
            ClientError({"Error": {"Code": "ExpiredTokenException"}}, "InvokeModel"),
            503,
            "AWS_CREDENTIALS_EXPIRED",
        ),
        (
            ClientError({"Error": {"Code": "ValidationException"}}, "InvokeModel"),
            502,
            "BEDROCK_VALIDATION",
        ),
    ],
)
def test_infrastructure_errors_use_the_error_mappers(exc, status, code):
    def job():
        raise exc

    r = ingest(client_with(job))
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert "pw@h" not in r.text


# ------------------------------------------------------------------------ audit


def test_audit_event_has_counts_only_and_no_secrets(set_env, caplog):
    set_env(admin_api_keys=ADMIN)
    caplog.set_level(logging.INFO)
    report = ok_report(failed_feeds=[("https://f.example/rss", "ConnectError: LEAK-TEXT")])
    r = ingest(client_with(lambda: report), ADMIN)
    assert r.status_code == 200
    lines = [rec.getMessage() for rec in caplog.records if rec.name.startswith("audit.tools.news")]
    event = json.loads(next(line for line in lines if "news_ingest" in line))
    assert event["request_id"] == r.json()["request_id"]
    assert (event["indexed"], event["purged"], event["failed_feeds"]) == (3, 2, 1)
    everything = "\n".join(rec.getMessage() for rec in caplog.records)
    assert ADMIN not in everything and "LEAK-TEXT" not in everything
