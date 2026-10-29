import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.traceability import CONTEXT_RADIUS, number_contexts, untraced_numbers
from app.main import app
from tests.fakes import tool_call_msg
from tests.test_chat_endpoint import client_for, graph_with


def teardown_function():
    app.dependency_overrides.clear()


def turn(reply, tool_payloads, user="consulta"):
    messages = [HumanMessage(content=user)]
    for i, payload in enumerate(tool_payloads):
        messages.append(AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": str(i)}]))
        messages.append(ToolMessage(content=json.dumps(payload), tool_call_id=str(i)))
    messages.append(AIMessage(content=reply))
    return messages


MACRO = {"variable": "riesgo_pais", "valor": 611.0, "fecha": "2026-10-14", "fuente": "x"}
PRICE = {"ticker": "GGAL", "fecha": "2026-10-14", "precio_cierre": 12.3456, "variacion_dia": 3.1}


def test_numbers_that_come_from_tool_outputs_are_traced():
    reply = "El riesgo país es 611 (2026-10-14). GGAL cerró en 12,35 con +3,1% en el día."
    assert untraced_numbers(turn(reply, [MACRO, PRICE])) == []


def test_invented_numbers_are_reported():
    reply = "El riesgo país es 611 y el dólar CCL 1.612,50."
    assert untraced_numbers(turn(reply, [MACRO])) == ["1.612,50"]


def test_numbers_typed_by_the_user_and_single_digits_are_not_data():
    reply = "Con 100 acciones, tenés 2 posiciones: 1) GGAL y 2) YPF."
    assert untraced_numbers(turn(reply, [], user="Tengo 100 acciones de GGAL")) == []


def test_error_payloads_do_not_justify_numbers():
    err = {"error": "NO_DATA", "message": "no data"}
    assert untraced_numbers(turn("GGAL vale 12,5.", [err])) == ["12,5"]


def test_no_final_reply_means_nothing_to_check():
    assert untraced_numbers([HumanMessage(content="x")]) == []


def test_chat_logs_untraced_numbers_as_an_audit_warning(caplog):
    caplog.set_level(logging.WARNING)
    g = graph_with(AIMessage(content="El dólar CCL está en 1.999,99."))
    client_for(g).post("/chat", json={"cliente_id": "c1", "message": "dólar?"})
    hits = [r for r in caplog.records if "untraced_numbers" in r.getMessage()]
    assert len(hits) == 1 and hits[0].levelno == logging.WARNING
    record = json.loads(hits[0].getMessage())
    assert record["numbers"] == ["1.999,99"] and record["request_id"]
    assert "dólar CCL está en 1.999,99" in record["contexts"]["1.999,99"]


def test_context_snippets_are_bounded_around_each_number():
    reply = "a" * 200 + " 20 " + "b" * 200
    contexts = number_contexts(reply, ["20"])
    assert "20" in contexts["20"] and len(contexts["20"]) <= 2 * CONTEXT_RADIUS + len("20")


def test_chat_is_silent_when_every_number_comes_from_a_tool(caplog):
    caplog.set_level(logging.WARNING)
    g = graph_with(
        tool_call_msg("get_macro", {"variable": "riesgo_pais"}),
        AIMessage(content="El riesgo país es 611 según x (2026-10-14)."),
    )
    r = client_for(g).post("/chat", json={"cliente_id": "c1", "message": "riesgo país?"})
    assert r.status_code == 200
    assert not [rec for rec in caplog.records if "untraced_numbers" in rec.getMessage()]
