import asyncio
import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.chat_graph import LOOP_LIMIT_MESSAGE, build_chat_graph, make_checkpointer
from app.core.config import Settings
from app.main import app, get_chat_graph
from app.tools.registry import build_tools
from tests.fakes import CHAT_PROMPT, ScriptedChatModel, scripted
from tests.test_chat_endpoint import client_for, graph_with


def teardown_function():
    app.dependency_overrides.clear()


def post(client, message="hola", **extra):
    return client.post("/chat", json={"cliente_id": "c1", "message": message, **extra})


def assert_error_body(response, status, code):
    assert response.status_code == status
    err = response.json()["error"]
    assert set(err) >= {"code", "message", "request_id"}
    assert err["code"] == code and err["message"] and err["request_id"]
    assert response.headers["x-request-id"] == err["request_id"]
    return err


# ------------------------------------------------------------- input validation


@pytest.mark.parametrize(
    "body",
    [
        {"cliente_id": "c1"},
        {"cliente_id": "c1", "message": ""},
        {"cliente_id": "c1", "message": "   \n\t "},
        {"cliente_id": "c1", "message": "\x00\x01\x02"},
        {"cliente_id": "c1", "message": "x" * 4001},
        {"cliente_id": "c1", "message": 5},
    ],
)
def test_invalid_message_is_422_with_the_error_contract(body):
    r = client_for(graph_with(AIMessage(content="x"))).post("/chat", json=body)
    err = assert_error_body(r, 422, "VALIDATION_ERROR")
    assert "message" in err["message"]


def test_message_of_exactly_4000_chars_is_accepted():
    assert post(client_for(graph_with(AIMessage(content="x"))), "y" * 4000).status_code == 200


def test_422_never_echoes_the_submitted_text():
    secret = "super-secret-" + "x" * 4100
    r = client_for(graph_with(AIMessage(content="x"))).post(
        "/chat", json={"cliente_id": "c1", "message": secret}
    )
    assert "super-secret" not in r.text


def test_control_characters_are_stripped_but_newlines_and_tabs_kept():
    model = scripted(AIMessage(content="ok"))
    tools = build_tools(get_price=lambda **k: {}, get_macro=lambda **k: {})
    g = build_chat_graph(model, tools, prompt_source=CHAT_PROMPT)
    # NUL, BEL, DEL, CR and the bidi override (U+202E) are removed; newline and tab stay.
    r = post(client_for(g), "  ho\x00la\x07‮ mundo\r\nlinea\t2\x7f  ")
    assert r.status_code == 200
    assert model.seen[0][-1].content == "hola mundo\nlinea\t2"


# ------------------------------------------------------------ error body / ids


def test_unknown_route_uses_the_error_contract():
    assert_error_body(TestClient(app).get("/nope"), 404, "NOT_FOUND")


def test_unhandled_exception_is_a_500_with_the_contract_and_no_leak():
    async def broken():
        raise RuntimeError("db password=hunter2")

    app.dependency_overrides[get_chat_graph] = broken
    r = post(TestClient(app, raise_server_exceptions=False))
    err = assert_error_body(r, 500, "INTERNAL_ERROR")
    assert "hunter2" not in r.text and "RuntimeError" not in err["message"]


def test_bedrock_errors_use_the_contract_and_keep_retry_after():
    from botocore.exceptions import ClientError

    class Throttled:
        async def ainvoke(self, *a, **k):
            raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "m"}}, "x")

    r = post(client_for(Throttled()))
    assert_error_body(r, 429, "BEDROCK_THROTTLED")
    assert r.headers["retry-after"] == "5"


def test_request_id_header_on_success_and_incoming_id_is_honoured():
    c = client_for(graph_with(AIMessage(content="ok"), AIMessage(content="ok")))
    first = post(c)
    assert first.status_code == 200 and len(first.headers["x-request-id"]) >= 8
    mine = uuid.uuid4().hex
    again = c.post(
        "/chat", json={"cliente_id": "c1", "message": "hola"}, headers={"X-Request-ID": mine}
    )
    assert again.headers["x-request-id"] == mine


def test_hostile_incoming_request_id_is_replaced():
    c = client_for(graph_with(AIMessage(content="ok")))
    r = c.post(
        "/chat",
        json={"cliente_id": "c1", "message": "hola"},
        headers={"X-Request-ID": "bad id with spaces;<script>"},
    )
    assert r.headers["x-request-id"] != "bad id with spaces;<script>"
    assert r.headers["x-request-id"].isalnum()


def test_chat_audit_record_has_duration_ids_and_no_message_text(caplog):
    caplog.set_level(logging.INFO, logger="audit.chat")
    c = client_for(graph_with(AIMessage(content="respuesta")))
    r = post(c, "texto privado del cliente", session_id="s1")
    [rec] = [json.loads(x.getMessage()) for x in caplog.records if x.name == "audit.chat"]
    assert rec["event"] == "chat_request" and rec["status"] == "ok"
    assert rec["request_id"] == r.headers["x-request-id"]
    assert rec["cliente_id"] == "c1" and rec["session_id"] == "s1"
    assert isinstance(rec["duration_ms"], float) and rec["duration_ms"] >= 0
    assert rec["tool_calls"] == [] and rec["message_chars"] == len("texto privado del cliente")
    assert "texto privado" not in caplog.text


def test_chat_audit_record_on_error(caplog):
    caplog.set_level(logging.INFO, logger="audit.chat")

    class Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("x")

    post(client_for(Boom()))
    [rec] = [json.loads(x.getMessage()) for x in caplog.records if x.name == "audit.chat"]
    assert rec["status"] == "error" and rec["error_code"] == "LLM_ERROR"
    assert rec["duration_ms"] >= 0


# --------------------------------------------------------------------- timeout


class Slow:
    async def ainvoke(self, *a, **k):
        await asyncio.sleep(2)
        return {"messages": [AIMessage(content="late")]}


def test_slow_turn_is_a_504_chat_timeout(set_env, caplog):
    set_env(chat_timeout_seconds="0.05")
    caplog.set_level(logging.INFO, logger="audit.chat")
    r = post(client_for(Slow()))
    err = assert_error_body(r, 504, "CHAT_TIMEOUT")
    assert "0.05" in err["message"]
    [rec] = [json.loads(x.getMessage()) for x in caplog.records if x.name == "audit.chat"]
    assert rec["status"] == "timeout" and rec["error_code"] == "CHAT_TIMEOUT"


def test_an_inner_timeout_error_is_not_mistaken_for_our_deadline():
    class InnerTimeout:
        async def ainvoke(self, *a, **k):
            raise TimeoutError("socket timed out")

    r = post(client_for(InnerTimeout()))
    assert_error_body(r, 502, "LLM_ERROR")


# --------------------------------------------------------------- tool loop cap


class Looping(ScriptedChatModel):
    """Always asks for another tool call (fresh message ids every time)."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        call = {"name": "get_macro", "args": {"variable": "riesgo_pais"}, "id": uuid.uuid4().hex}
        msg = AIMessage(content="", tool_calls=[call], id=uuid.uuid4().hex)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def looping_graph(cap):
    tools = build_tools(
        get_price=lambda **k: {},
        get_macro=lambda variable: {"valor": 611.0, "fecha": "2026-10-14", "fuente": "x"},
    )
    return build_chat_graph(
        Looping(responses=[AIMessage(content="x")], seen=[]),
        tools,
        prompt_source=CHAT_PROMPT,
        max_tool_iterations=cap,
    )


async def test_tool_loop_stops_at_the_cap_with_a_graceful_message():
    result = await looping_graph(3).ainvoke(
        {"messages": [("user", "riesgo pais")]}, {"configurable": {"thread_id": "t"}}
    )
    tool_rounds = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_rounds) == 3
    last = result["messages"][-1]
    assert last.content == LOOP_LIMIT_MESSAGE
    assert last.response_metadata["stopReason"] == "tool_loop_limit"


def test_tool_loop_cap_through_the_endpoint():
    r = post(client_for(looping_graph(2)))
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == LOOP_LIMIT_MESSAGE and body["stop_reason"] == "tool_loop_limit"
    assert len(body["tool_calls"]) == 2


async def test_cap_counts_only_the_current_turn():
    graph = looping_graph(2)
    cfg = {"configurable": {"thread_id": "t2"}}
    for _ in range(2):  # each turn may use its own full budget
        result = await graph.ainvoke({"messages": [("user", "otra vez")]}, cfg)
        assert result["messages"][-1].content == LOOP_LIMIT_MESSAGE
    assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 4


# ---------------------------------------------------------------- Redis down


async def test_checkpointer_falls_back_to_memory_and_logs(monkeypatch, caplog):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.redis import aio

    class Down:
        def __init__(self, *a, **k):
            pass

        async def asetup(self):
            raise ConnectionError("redis down")

    monkeypatch.setattr(aio, "AsyncRedisSaver", Down)
    caplog.set_level(logging.WARNING)
    saver = await make_checkpointer(Settings(_env_file=None))
    assert isinstance(saver, MemorySaver)
    assert any("checkpointer_fallback" in r.getMessage() for r in caplog.records)


def test_chat_works_end_to_end_on_a_memory_checkpointer():
    g = graph_with(AIMessage(content="uno"), AIMessage(content="dos"))
    c = client_for(g)
    assert post(c, session_id="s").json()["reply"] == "uno"
    assert post(c, session_id="s").json()["reply"] == "dos"
