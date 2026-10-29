import fakeredis
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agents.chat_graph import build_chat_graph
from app.main import app, get_chat_graph
from app.tools.errors import ToolError
from app.tools.get_macro import make_get_macro
from app.tools.registry import build_tools
from tests.fakes import CHAT_PROMPT, scripted, tool_call_msg


def _price_stub(ticker: str, mercado: str) -> dict:
    raise ToolError("NO_DATA", "n/a")


def _macro_ok(variable):
    return {"valor": 611.0, "fecha": "2026-10-14", "fuente": "x"}


def graph_with(*msgs):
    r = fakeredis.FakeRedis(decode_responses=True)
    tools = build_tools(
        get_price=_price_stub,
        get_macro=make_get_macro(redis_client=r, providers={"riesgo_pais": [_macro_ok]}),
    )
    return build_chat_graph(scripted(*msgs), tools, prompt_source=CHAT_PROMPT)


def client_for(graph):
    async def dep():
        return graph

    app.dependency_overrides[get_chat_graph] = dep
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_chat_returns_reply_and_tool_calls():
    g = graph_with(
        tool_call_msg("get_macro", {"variable": "riesgo_pais"}),
        AIMessage(content="El riesgo pais es 611."),
    )
    r = client_for(g).post("/chat", json={"cliente_id": "c1", "message": "riesgo pais?"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "El riesgo pais es 611."
    assert body["session_id"]
    assert body["tool_calls"] == [
        {"name": "get_macro", "args": {"variable": "riesgo_pais"}, "status": "ok"}
    ]


def test_tool_error_status_reported_and_session_reused():
    g = graph_with(
        tool_call_msg("get_price", {"ticker": "ZZZ", "mercado": "nyse"}),
        AIMessage(content="Sin dato."),
    )
    r = client_for(g).post(
        "/chat", json={"cliente_id": "c1", "session_id": "s1", "message": "precio ZZZ"}
    )
    body = r.json()
    assert body["session_id"] == "s1"
    assert body["tool_calls"][0]["status"] == "error"


def test_validation_422():
    c = client_for(graph_with(AIMessage(content="x")))
    assert c.post("/chat", json={"cliente_id": "c1"}).status_code == 422
    assert c.post("/chat", json={"cliente_id": "", "message": "hi"}).status_code == 422


def test_llm_failure_is_502_structured():
    class Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("bedrock down")

    r = client_for(Boom()).post("/chat", json={"cliente_id": "c1", "message": "hola"})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "LLM_ERROR"


@pytest.mark.parametrize(
    ("code", "status", "error"),
    [
        ("AccessDeniedException", 403, "BEDROCK_ACCESS_DENIED"),
        ("ThrottlingException", 429, "BEDROCK_THROTTLED"),
        ("ValidationException", 502, "BEDROCK_VALIDATION"),
        ("ModelNotReadyException", 503, "BEDROCK_UNAVAILABLE"),
        ("ServiceUnavailableException", 503, "BEDROCK_UNAVAILABLE"),
        ("ModelTimeoutException", 504, "BEDROCK_TIMEOUT"),
        ("InternalServerException", 502, "LLM_ERROR"),
    ],
)
def test_bedrock_client_errors_are_mapped(code, status, error):
    class Failing:
        async def ainvoke(self, *a, **k):
            raise ClientError({"Error": {"Code": code, "Message": "m"}}, "Converse")

    r = client_for(Failing()).post("/chat", json={"cliente_id": "c1", "message": "hola"})
    assert r.status_code == status
    assert r.json()["error"]["code"] == error
    if status == 429:
        assert int(r.headers["retry-after"]) > 0
    if error == "BEDROCK_ACCESS_DENIED":
        assert "model access" in r.json()["error"]["message"]


@pytest.mark.parametrize("code", ["ExpiredTokenException", "ExpiredToken"])
def test_expired_credentials_are_503(code):
    class Expired:
        async def ainvoke(self, *a, **k):
            raise ClientError({"Error": {"Code": code, "Message": "token expired"}}, "Converse")

    r = client_for(Expired()).post("/chat", json={"cliente_id": "c1", "message": "hola"})
    assert r.status_code == 503
    detail = r.json()["error"]
    assert detail["code"] == "AWS_CREDENTIALS_EXPIRED"
    assert "temporary credentials" in detail["message"]


def test_max_tokens_sets_truncated_flag():
    g = graph_with(AIMessage(content="cortado...", response_metadata={"stopReason": "max_tokens"}))
    body = client_for(g).post("/chat", json={"cliente_id": "c1", "message": "hola"}).json()
    assert body["truncated"] is True
    assert body["stop_reason"] == "max_tokens"


def test_normal_reply_is_not_truncated():
    g = graph_with(AIMessage(content="ok", response_metadata={"stopReason": "end_turn"}))
    body = client_for(g).post("/chat", json={"cliente_id": "c1", "message": "hola"}).json()
    assert body["truncated"] is False
    assert body["stop_reason"] == "end_turn"


def test_no_credentials_at_runtime_is_503():
    from botocore.exceptions import NoCredentialsError

    class NoCreds:
        async def ainvoke(self, *a, **k):
            raise NoCredentialsError()

    r = client_for(NoCreds()).post("/chat", json={"cliente_id": "c1", "message": "hola"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "AWS_CREDENTIALS_MISSING"


def test_unconfigured_llm_is_503(monkeypatch):
    import app.main as main
    from app.agents.llm import LLMConfigError

    def boom(settings):
        raise LLMConfigError("BEDROCK_MODEL_ID_MISSING", "BEDROCK_MODEL_ID is not set.")

    monkeypatch.setattr(main, "_graph", None)
    monkeypatch.setattr(main, "ensure_llm_configured", boom)
    r = TestClient(app).post("/chat", json={"cliente_id": "c1", "message": "hola"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "BEDROCK_MODEL_ID_MISSING"
