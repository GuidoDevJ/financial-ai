import hmac
import logging

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.brief.render import DISCLAIMER
from app.core import auth
from app.core.audit import cliente_id_var
from app.core.config import Settings
from app.main import app, get_brief_deps
from tests.fakes import scripted
from tests.test_brief_endpoint import GOOD, make_deps
from tests.test_chat_endpoint import client_for, graph_with

KEYS = '{"key-one": "c1", "key-two": "c2"}'


def teardown_function():
    app.dependency_overrides.clear()


def chat(client, headers=None, **body):
    payload = {"message": "hola", **body}
    return client.post("/chat", json=payload, headers=headers or {})


def ok_graph():
    return graph_with(AIMessage(content="Hola."))


def test_missing_key_is_401_with_consistent_body(set_env):
    set_env(auth_api_keys=KEYS)
    r = chat(client_for(ok_graph()), cliente_id="c1")
    assert r.status_code == 401
    err = r.json()["error"]
    assert err["code"] == "UNAUTHENTICATED" and err["request_id"]
    assert r.headers["x-request-id"] == err["request_id"]


@pytest.mark.parametrize("key", ["wrong", "key-on", "KEY-ONE", "key-one-extra"])
def test_invalid_key_is_401(set_env, key):
    set_env(auth_api_keys=KEYS)
    r = chat(client_for(ok_graph()), {"X-API-Key": key}, cliente_id="c1")
    assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_API_KEY"


def test_valid_key_and_matching_body_is_200(set_env):
    set_env(auth_api_keys=KEYS)
    r = chat(client_for(ok_graph()), {"X-API-Key": "key-one"}, cliente_id="c1")
    assert r.status_code == 200 and r.json()["reply"] == "Hola."


def test_body_client_different_from_authenticated_is_403(set_env):
    set_env(auth_api_keys=KEYS)
    r = chat(client_for(ok_graph()), {"X-API-Key": "key-one"}, cliente_id="c2")
    assert r.status_code == 403 and r.json()["error"]["code"] == "CLIENT_MISMATCH"


def test_body_without_cliente_id_uses_the_authenticated_one(set_env):
    set_env(auth_api_keys=KEYS)
    seen = {}

    class Spy:
        async def ainvoke(self, payload, config):
            seen["cliente_id"] = cliente_id_var.get()
            seen["configurable"] = config["configurable"]["cliente_id"]
            return {"messages": [AIMessage(content="ok")]}

    r = chat(client_for(Spy()), {"X-API-Key": "key-two"})
    assert r.status_code == 200
    assert seen == {"cliente_id": "c2", "configurable": "c2"}


def test_without_keys_in_dev_the_body_client_is_trusted_and_required():
    c = client_for(ok_graph())
    assert chat(c, cliente_id="anyone").status_code == 200
    r = chat(c)
    assert r.status_code == 422 and r.json()["error"]["code"] == "CLIENT_ID_REQUIRED"


@pytest.mark.parametrize("env", ["prod", "staging"])
def test_without_keys_outside_dev_is_503(set_env, env):
    set_env(app_env=env)
    r = chat(client_for(ok_graph()), cliente_id="c1")
    assert r.status_code == 503 and r.json()["error"]["code"] == "AUTH_NOT_CONFIGURED"


def test_keys_are_compared_in_constant_time(set_env, monkeypatch):
    set_env(auth_api_keys=KEYS)
    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    chat(client_for(ok_graph()), {"X-API-Key": "key-two"}, cliente_id="c2")
    assert len(calls) == 2  # every configured key is compared, no early exit


def test_health_stays_public(set_env):
    set_env(auth_api_keys=KEYS)
    app.dependency_overrides.clear()
    from app.main import get_redis

    app.dependency_overrides[get_redis] = lambda: fakeredis.FakeRedis()
    assert TestClient(app).get("/health").status_code == 200


def brief_client(set_env_fn=None):
    app.dependency_overrides[get_brief_deps] = lambda: make_deps(scripted(AIMessage(content=GOOD)))
    return TestClient(app)


def test_brief_preview_requires_a_key_and_binds_the_client(set_env):
    set_env(auth_api_keys='{"key-one": "client_001"}')
    c = brief_client()
    assert c.post("/brief/preview", json={"cliente_id": "client_001"}).status_code == 401
    other = c.post(
        "/brief/preview", json={"cliente_id": "client_002"}, headers={"X-API-Key": "key-one"}
    )
    assert other.status_code == 403 and other.json()["error"]["code"] == "CLIENT_MISMATCH"
    ok = c.post("/brief/preview", json={}, headers={"X-API-Key": "key-one"})
    assert ok.status_code == 200 and ok.json()["texto_brief"].endswith(DISCLAIMER)


def test_brief_preview_outside_dev_without_keys_is_503(set_env):
    set_env(app_env="prod")
    r = brief_client().post("/brief/preview", json={"cliente_id": "client_001"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "AUTH_NOT_CONFIGURED"


def test_startup_warns_when_auth_is_disabled_in_dev(caplog):
    caplog.set_level(logging.WARNING)
    with TestClient(app):
        pass
    assert any("AUTH_API_KEYS" in r.getMessage() for r in caplog.records)


def test_startup_is_quiet_when_keys_are_configured(set_env, caplog):
    set_env(auth_api_keys=KEYS)
    caplog.set_level(logging.WARNING)
    with TestClient(app):
        pass
    assert not any("AUTH_API_KEYS" in r.getMessage() for r in caplog.records)


def test_startup_check_messages():
    dev = Settings(_env_file=None, app_env="dev")
    prod = Settings(_env_file=None, app_env="prod")
    keyed = Settings(_env_file=None, auth_api_keys='{"k": "c1"}')
    assert auth.startup_warning(dev) and "dev" in auth.startup_warning(dev)
    assert auth.startup_warning(prod) and "503" in auth.startup_warning(prod)
    assert auth.startup_warning(keyed) is None
