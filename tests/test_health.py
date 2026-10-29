import fakeredis
from fastapi.testclient import TestClient

from app.main import app, get_redis


def client_with(redis_obj):
    app.dependency_overrides[get_redis] = lambda: redis_obj
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health_ok():
    r = client_with(fakeredis.FakeRedis()).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "redis": "up"}


def test_health_degraded_when_redis_down():
    class Down:
        def ping(self):
            raise ConnectionError("down")

    r = client_with(Down()).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "degraded", "redis": "down"}


def test_health_reports_age_of_last_successful_ingestion():
    from datetime import UTC, datetime, timedelta

    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("ingest:last_success", (datetime.now(UTC) - timedelta(minutes=5)).isoformat())
    body = client_with(r).get("/health").json()
    assert body["status"] == "ok" and body["redis"] == "up"
    assert 295 <= body["news"]["last_success_age_seconds"] <= 330


def test_health_ignores_garbage_last_success():
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("ingest:last_success", "not-a-date")
    assert client_with(r).get("/health").json() == {"status": "ok", "redis": "up"}
