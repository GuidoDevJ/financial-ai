import fakeredis
import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agents.llm import LLMConfigError
from app.brief.render import DISCLAIMER
from app.brief.store import BriefStore
from app.main import app, get_brief_deps
from tests.fakes import ScriptedChatModel, scripted
from tests.test_brief_generate import GOOD, Tools, make_deps


def client_with(deps):
    app.dependency_overrides[get_brief_deps] = lambda: deps
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_preview_returns_brief_without_storing_or_sending():
    store = BriefStore(fakeredis.FakeRedis(decode_responses=True))
    sent = []
    deps = make_deps(
        scripted(AIMessage(content=GOOD)),
        store=store,
        send_email=lambda *a, **k: sent.append(1) or {},
    )
    r = client_with(deps).post("/brief/preview", json={"cliente_id": "client_001"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "fecha",
        "texto_brief",
        "html",
        "activos_mencionados",
        "degraded",
        "missing_inputs",
    }
    assert body["fecha"] == "2026-10-15" and body["degraded"] is False
    assert body["texto_brief"].startswith(GOOD) and body["texto_brief"].endswith(DISCLAIMER)
    assert "<table>" in body["html"]
    assert store.get("client_001", "2026-10-15") is None and sent == []


def test_preview_accepts_explicit_fecha():
    deps = make_deps(scripted(AIMessage(content=GOOD)))
    r = client_with(deps).post(
        "/brief/preview", json={"cliente_id": "client_001", "fecha": "2026-10-09"}
    )
    assert r.json()["fecha"] == "2026-10-09"


def test_unknown_client_is_404():
    deps = make_deps(scripted(AIMessage(content=GOOD)), Tools(rows=None))
    r = client_with(deps).post("/brief/preview", json={"cliente_id": "nope"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "CLIENT_NOT_FOUND"


@pytest.mark.parametrize(
    "body",
    [
        {"cliente_id": ""},
        {"cliente_id": "../x"},
        {"cliente_id": "client_001", "fecha": "15/10/2026"},
        {"cliente_id": "client_001", "fecha": "2026-13-45"},
        {},
    ],
)
def test_invalid_body_is_422(body):
    deps = make_deps(scripted(AIMessage(content=GOOD)))
    assert client_with(deps).post("/brief/preview", json=body).status_code == 422


def _boom(exc):
    class Boom(ScriptedChatModel):
        def _generate(self, *a, **k):
            raise exc

    return Boom(responses=[AIMessage(content="x")], seen=[])


@pytest.mark.parametrize(
    "code,status,error",
    [
        ("ThrottlingException", 429, "BEDROCK_THROTTLED"),
        ("AccessDeniedException", 403, "BEDROCK_ACCESS_DENIED"),
        ("ExpiredTokenException", 503, "AWS_CREDENTIALS_EXPIRED"),
        ("ModelTimeoutException", 504, "BEDROCK_TIMEOUT"),
    ],
)
def test_bedrock_errors_share_the_chat_mapping(code, status, error):
    exc = ClientError({"Error": {"Code": code, "Message": "m"}}, "Converse")
    r = client_with(make_deps(_boom(exc))).post("/brief/preview", json={"cliente_id": "client_001"})
    assert r.status_code == status and r.json()["error"]["code"] == error
    if status == 429:
        assert r.headers["retry-after"] == "5"


def test_missing_credentials_and_generic_failures():
    r = client_with(make_deps(_boom(NoCredentialsError()))).post(
        "/brief/preview", json={"cliente_id": "client_001"}
    )
    assert r.status_code == 503 and r.json()["error"]["code"] == "AWS_CREDENTIALS_MISSING"
    r = client_with(make_deps(_boom(RuntimeError("x")))).post(
        "/brief/preview", json={"cliente_id": "client_001"}
    )
    assert r.status_code == 502 and r.json()["error"]["code"] == "LLM_ERROR"


def test_default_dependency_returns_503_when_llm_is_not_configured(monkeypatch):
    def not_configured(settings, **kw):
        raise LLMConfigError("BEDROCK_MODEL_ID_MISSING", "BEDROCK_MODEL_ID is not set.")

    monkeypatch.setattr("app.main.build_default_deps", not_configured)
    monkeypatch.setattr("app.main._brief_deps", None)
    r = TestClient(app).post("/brief/preview", json={"cliente_id": "client_001"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "BEDROCK_MODEL_ID_MISSING"
