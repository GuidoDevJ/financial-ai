"""POST /chat/stream: SSE variant of /chat (same request, same final payload in `done`)."""

import json

import fakeredis
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.agents.chat_graph import build_chat_graph
from app.main import app, get_chat_graph
from app.tools.get_macro import make_get_macro
from app.tools.registry import build_tools
from tests.fakes import CHAT_PROMPT, ScriptedChatModel, scripted, tool_call_msg


def _macro_ok(variable):
    return {"valor": 611.0, "fecha": "2026-10-14", "fuente": "x"}


def _price_stub(ticker, mercado):
    return {"ticker": ticker, "fecha": "2026-10-14", "precio_cierre": 1.0, "fuente": "x"}


def graph_with(model):
    r = fakeredis.FakeRedis(decode_responses=True)
    tools = build_tools(
        get_price=_price_stub,
        get_macro=make_get_macro(redis_client=r, providers={"riesgo_pais": [_macro_ok]}),
    )
    return build_chat_graph(model, tools, prompt_source=CHAT_PROMPT)


def client_for(graph):
    async def dep():
        return graph

    app.dependency_overrides[get_chat_graph] = dep
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_emits_tool_progress_and_a_done_event_equal_to_the_chat_response():
    msgs = [
        tool_call_msg("get_macro", {"variable": "riesgo_pais"}),
        AIMessage(content="El riesgo pais es 611."),
    ]
    payload = {"cliente_id": "c1", "session_id": "s1", "message": "riesgo pais?"}
    plain = client_for(graph_with(scripted(*msgs))).post("/chat", json=payload).json()

    r = client_for(graph_with(scripted(*msgs))).post("/chat/stream", json=payload)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [name for name, _ in events]
    assert names[0] == "tool_start" and "tool_end" in names and names[-1] == "done"
    assert events[0][1] == {"name": "get_macro"}
    assert events[-1][1] == plain
    assert "token" not in names  # off by default


def test_stream_is_contract_compatible_with_chat_validation_and_auth():
    c = client_for(graph_with(scripted(AIMessage(content="x"))))
    assert c.post("/chat/stream", json={"cliente_id": "c1"}).status_code == 422
    assert c.post("/chat/stream", json={"cliente_id": "", "message": "hi"}).status_code == 422


class StreamingModel(ScriptedChatModel):
    """Fake model that streams its reply word by word."""

    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGenerationChunk

        for word in ("El ", "riesgo ", "pais."):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))


def test_tokens_are_streamed_only_when_enabled(set_env):
    set_env(CHAT_STREAM_TOKENS="true")
    model = StreamingModel(responses=[AIMessage(content="unused")], seen=[])
    r = client_for(graph_with(model)).post(
        "/chat/stream", json={"cliente_id": "c1", "message": "riesgo?"}
    )
    events = parse_sse(r.text)
    tokens = [data["text"] for name, data in events if name == "token"]
    assert "".join(tokens) == "El riesgo pais."
    assert events[-1][0] == "done" and events[-1][1]["reply"] == "El riesgo pais."


def test_guardrail_replacement_is_authoritative_in_done(set_env):
    set_env(CHAT_STREAM_TOKENS="true")
    model = StreamingModel(responses=[AIMessage(content="unused")], seen=[])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGenerationChunk

        for word in ("Te recomiendo ", "comprar GGAL ", "ahora."):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))

    StreamingModel._astream = _astream
    try:
        r = client_for(graph_with(model)).post(
            "/chat/stream", json={"cliente_id": "c1", "message": "que hago?"}
        )
    finally:
        del StreamingModel._astream
    done = parse_sse(r.text)[-1]
    assert done[0] == "done" and done[1]["recommendation_blocked"] is True
    assert "comprar GGAL" not in done[1]["reply"]


def test_llm_failure_becomes_an_error_event_with_the_same_code():
    class Failing:
        def astream_events(self, *a, **k):
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "m"}}, "Converse"
            )

    r = client_for(Failing()).post("/chat/stream", json={"cliente_id": "c1", "message": "hola"})
    assert r.status_code == 200  # the stream was already open
    name, data = parse_sse(r.text)[-1]
    assert name == "error"
    assert data["code"] == "BEDROCK_THROTTLED" and data["status"] == 429
    assert data["request_id"]


def test_timeout_becomes_a_chat_timeout_error_event(set_env):
    import asyncio

    set_env(CHAT_TIMEOUT_SECONDS="0.05")

    class Slow:
        async def astream_events(self, *a, **k):
            await asyncio.sleep(5)
            yield {}

    r = client_for(Slow()).post("/chat/stream", json={"cliente_id": "c1", "message": "hola"})
    name, data = parse_sse(r.text)[-1]
    assert name == "error" and data["code"] == "CHAT_TIMEOUT" and data["status"] == 504
