import fakeredis
import pytest
import redis
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.brief.render import DISCLAIMER
from app.brief.scheduler import lock_key
from app.brief.store import BriefStore
from app.main import app, get_brief_deps, get_redis
from app.prompts.registry import PromptError
from tests.fakes import ScriptedChatModel, scripted
from tests.test_brief_generate import GOOD, Tools, make_deps

FECHA = "2026-10-15"
RUN_KEYS = {
    "cliente_id",
    "fecha",
    "texto_brief",
    "html",
    "activos_mencionados",
    "degraded",
    "missing_inputs",
    "stored",
    "sent",
    "delivery_status",
    "generated_at",
    "prompt",
    "request_id",
}


def teardown_function():
    app.dependency_overrides.clear()


def setup(deps_model=None, *, tools=None, send_email=None, redis_client=None):
    r = redis_client or fakeredis.FakeRedis(decode_responses=True)
    store = BriefStore(r)
    deps = make_deps(
        deps_model or scripted(AIMessage(content=GOOD)),
        tools,
        store=store,
        send_email=send_email,
    )
    app.dependency_overrides[get_brief_deps] = lambda: deps
    app.dependency_overrides[get_redis] = lambda: r
    return TestClient(app), store, r


def authed(set_env):
    set_env(auth_api_keys='{"key-1": "client_001", "key-2": "client_002"}')
    return {"X-API-Key": "key-1"}


# ---------------------------------------------------------------- POST /brief


def test_generates_stores_and_returns_the_full_payload(set_env):
    headers = authed(set_env)
    client, store, _ = setup()
    r = client.post("/brief", json={}, headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == RUN_KEYS
    assert body["cliente_id"] == "client_001" and body["fecha"] == FECHA
    assert body["texto_brief"].startswith(GOOD) and body["texto_brief"].endswith(DISCLAIMER)
    assert "<table>" in body["html"]
    assert body["degraded"] is False and body["missing_inputs"] == []
    assert body["stored"] is True and body["sent"] is False
    assert body["delivery_status"] == "not_requested"
    assert body["prompt"] == {"name": "brief", "version": "2", "hash": body["prompt"]["hash"]}
    assert body["request_id"] == r.headers["x-request-id"]
    assert store.get("client_001", FECHA)["texto_brief"] == body["texto_brief"]


def test_store_false_does_not_persist_and_send_true_delivers(set_env):
    headers = authed(set_env)
    sent = []

    def send(settings, **kw):
        sent.append(kw["fecha"])
        return {"status": "sent"}

    client, store, _ = setup(send_email=send)
    body = client.post(
        "/brief", json={"store": False, "send": True, "fecha": "2026-10-09"}, headers=headers
    ).json()
    assert body["stored"] is False and body["sent"] is True and body["delivery_status"] == "sent"
    assert body["fecha"] == "2026-10-09" and sent == ["2026-10-09"]
    assert store.get("client_001", "2026-10-09") is None


def test_failed_delivery_is_reported_not_raised(set_env):
    headers = authed(set_env)

    def send(settings, **kw):
        raise RuntimeError("smtp down")

    client, store, _ = setup(send_email=send)
    body = client.post("/brief", json={"send": True}, headers=headers).json()
    assert body["sent"] is False and body["delivery_status"] == "failed"
    assert body["stored"] is True


def test_degraded_and_missing_inputs_are_returned(set_env):
    headers = authed(set_env)
    tools = Tools(macro={})
    client, _, _ = setup(tools=tools)
    body = client.post("/brief", json={}, headers=headers).json()
    assert "macro:riesgo_pais" in body["missing_inputs"]


def test_same_pipeline_as_preview(set_env):
    headers = authed(set_env)
    client, _, _ = setup(scripted(AIMessage(content=GOOD), AIMessage(content=GOOD)))
    run = client.post("/brief", json={"store": False}, headers=headers).json()
    preview = client.post("/brief/preview", json={}, headers=headers).json()
    for key in ("fecha", "texto_brief", "html", "activos_mencionados", "degraded"):
        assert run[key] == preview[key]


def test_requires_a_key_and_binds_the_client(set_env):
    headers = authed(set_env)
    client, store, _ = setup()
    assert client.post("/brief", json={}).status_code == 401
    mismatch = client.post("/brief", json={"cliente_id": "client_002"}, headers=headers)
    assert mismatch.status_code == 403
    assert mismatch.json()["error"]["code"] == "CLIENT_MISMATCH"
    assert store.get("client_002", FECHA) is None
    other_key = client.post("/brief", json={}, headers={"X-API-Key": "key-2"})
    assert other_key.json()["cliente_id"] == "client_002"


def test_dev_mode_without_keys_needs_the_body_client():
    client, store, _ = setup()
    assert client.post("/brief", json={}).json()["error"]["code"] == "CLIENT_ID_REQUIRED"
    ok = client.post("/brief", json={"cliente_id": "client_001"})
    assert ok.status_code == 200 and store.get("client_001", FECHA)


def test_outside_dev_without_keys_is_503(set_env):
    set_env(app_env="prod")
    client, _, _ = setup()
    r = client.post("/brief", json={"cliente_id": "client_001"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "AUTH_NOT_CONFIGURED"


@pytest.mark.parametrize(
    "body",
    [
        {"fecha": "15/10/2026"},
        {"fecha": "2026-13-45"},
        {"store": "maybe"},
        {"cliente_id": "../x"},
        {"send": 5},
    ],
)
def test_invalid_body_is_422(set_env, body):
    headers = authed(set_env)
    client, _, _ = setup()
    r = client.post("/brief", json=body, headers=headers)
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_client_is_404(set_env):
    headers = authed(set_env)
    client, _, _ = setup(tools=Tools(rows=None))
    r = client.post("/brief", json={}, headers=headers)
    assert r.status_code == 404 and r.json()["error"]["code"] == "CLIENT_NOT_FOUND"


# --------------------------------------------------------------- lock (409)


def test_concurrent_run_is_409_and_does_not_touch_the_model(set_env):
    headers = authed(set_env)
    model = scripted(AIMessage(content=GOOD))
    client, store, r = setup(model)
    r.set(lock_key("client_001", FECHA), "other-instance", nx=True, ex=60)
    resp = client.post("/brief", json={}, headers=headers)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "BRIEF_IN_PROGRESS"
    assert model.seen == [] and store.get("client_001", FECHA) is None
    assert r.get(lock_key("client_001", FECHA)) == "other-instance"  # not stolen


def test_lock_is_released_after_success_and_after_failure(set_env):
    headers = authed(set_env)
    client, _, r = setup()
    client.post("/brief", json={}, headers=headers)
    assert r.get(lock_key("client_001", FECHA)) is None

    class Boom(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("llm down")

    client, _, r = setup(Boom(responses=[AIMessage(content="x")], seen=[]))
    assert client.post("/brief", json={}, headers=headers).status_code == 502
    assert r.get(lock_key("client_001", FECHA)) is None
    assert client.post("/brief", json={}, headers=headers).status_code == 502  # retryable


def test_redis_down_at_lock_time_is_503(set_env):
    headers = authed(set_env)

    class Down:
        def set(self, *a, **k):
            raise redis.ConnectionError("down")

    client, _, _ = setup()
    app.dependency_overrides[get_redis] = lambda: Down()
    r = client.post("/brief", json={}, headers=headers)
    assert r.status_code == 503 and r.json()["error"]["code"] == "STORAGE_UNAVAILABLE"


# ------------------------------------------------------------- error mapping


@pytest.mark.parametrize(
    ("code", "status", "error"),
    [
        ("ThrottlingException", 429, "BEDROCK_THROTTLED"),
        ("AccessDeniedException", 403, "BEDROCK_ACCESS_DENIED"),
        ("ExpiredTokenException", 503, "AWS_CREDENTIALS_EXPIRED"),
        ("ModelTimeoutException", 504, "BEDROCK_TIMEOUT"),
    ],
)
def test_bedrock_errors_use_the_shared_mapping(set_env, code, status, error):
    headers = authed(set_env)

    class Boom(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise ClientError({"Error": {"Code": code, "Message": "m"}}, "Converse")

    client, _, _ = setup(Boom(responses=[AIMessage(content="x")], seen=[]))
    r = client.post("/brief", json={}, headers=headers)
    assert r.status_code == status and r.json()["error"]["code"] == error
    assert r.json()["error"]["request_id"] == r.headers["x-request-id"]
    if status == 429:
        assert r.headers["retry-after"] == "5"


def test_prompt_errors_are_mapped(set_env, monkeypatch):
    headers = authed(set_env)
    client, _, _ = setup()
    app.dependency_overrides.pop(get_brief_deps)

    def not_configured(settings, **kw):
        raise PromptError("PROMPT_ID_MISSING", "BEDROCK_PROMPT_BRIEF_ID is not set.", status=503)

    monkeypatch.setattr("app.main.build_default_deps", not_configured)
    monkeypatch.setattr("app.main._brief_deps", None)
    r = client.post("/brief", json={}, headers=headers)
    assert r.status_code == 503 and r.json()["error"]["code"] == "PROMPT_ID_MISSING"


# ---------------------------------------------------------- GET /brief/latest


def test_latest_returns_the_stored_brief_of_the_authenticated_client(set_env):
    headers = authed(set_env)
    client, store, _ = setup()
    client.post("/brief", json={"fecha": "2026-10-14"}, headers=headers)
    client.post("/brief", json={"fecha": "2026-10-15"}, headers=headers)
    r = client.get("/brief/latest", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["fecha"] == "2026-10-15" and body["cliente_id"] == "client_001"
    assert body["texto_brief"].endswith(DISCLAIMER) and "<table>" in body["html"]
    assert body["activos_mencionados"] == ["GGAL", "YPFD.BA"]
    assert body["degraded"] is False and body["request_id"] == r.headers["x-request-id"]


def test_latest_never_returns_another_clients_brief(set_env):
    headers = authed(set_env)
    client, store, _ = setup()
    store.save("client_002", "2026-10-15", "BRIEF PRIVADO", ["X"])
    r = client.get("/brief/latest", headers=headers)
    assert r.status_code == 404 and r.json()["error"]["code"] == "BRIEF_NOT_FOUND"
    mismatch = client.get("/brief/latest", params={"cliente_id": "client_002"}, headers=headers)
    assert mismatch.status_code == 403 and "PRIVADO" not in mismatch.text
    other = client.get("/brief/latest", headers={"X-API-Key": "key-2"})
    assert other.status_code == 200 and other.json()["texto_brief"] == "BRIEF PRIVADO"


def test_latest_requires_auth_and_dev_needs_a_query_client(set_env):
    client, store, _ = setup()
    store.save("client_001", "2026-10-15", "texto", ["GGAL"])
    assert client.get("/brief/latest").json()["error"]["code"] == "CLIENT_ID_REQUIRED"
    assert client.get("/brief/latest", params={"cliente_id": "client_001"}).status_code == 200
    set_env(auth_api_keys='{"key-1": "client_001"}')
    assert client.get("/brief/latest").status_code == 401


def test_latest_storage_failure_is_503(set_env):
    headers = authed(set_env)

    class Down:
        def zrevrange(self, *a, **k):
            raise redis.ConnectionError("down")

    client, _, _ = setup()
    app.dependency_overrides[get_redis] = lambda: Down()
    r = client.get("/brief/latest", headers=headers)
    assert r.status_code == 503 and r.json()["error"]["code"] == "STORAGE_UNAVAILABLE"


def test_latest_for_a_malformed_query_client_is_422(set_env):
    client, _, _ = setup()
    r = client.get("/brief/latest", params={"cliente_id": "../x"})
    assert r.status_code == 422
