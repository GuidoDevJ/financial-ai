"""Offline adversarial evaluation of the no-recommendation rule.

Every question of tests/data/adversarial_recommendations.json goes through the real /chat
endpoint and the real output guardrail with a scripted fake LLM that answers either with a
compliant reply (refusal or neutral comparison) or with a violating one (explicit buy/sell
advice). Compliant replies must reach the client untouched; violating ones must be replaced.
"""

import json
import logging
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.agents.chat_graph import build_chat_graph
from app.brief.validate import find_recommendations_outside_refusals
from app.core.guardrail import SAFE_REFUSAL, check_reply
from app.main import app
from app.tools.registry import build_tools
from tests.fakes import CHAT_PROMPT, scripted
from tests.test_chat_endpoint import client_for

DATA = Path(__file__).parent / "data" / "adversarial_recommendations.json"
ITEMS = json.loads(DATA.read_text(encoding="utf-8"))["items"]
IDS = [item["id"] for item in ITEMS]


def teardown_function():
    app.dependency_overrides.clear()


def chat_graph(*replies, model=None):
    tools = build_tools(get_price=lambda **k: {}, get_macro=lambda **k: {})
    return build_chat_graph(
        model or scripted(*[AIMessage(content=r) for r in replies]),
        tools,
        prompt_source=CHAT_PROMPT,
    )


def ask(item, reply, caplog=None):
    client = client_for(chat_graph(reply))
    return client.post("/chat", json={"cliente_id": "c1", "message": item["question"]})


# ------------------------------------------------------------------- the data set


def test_data_set_is_large_and_varied():
    spanish = [i for i in ITEMS if i["category"] != "english"]
    assert len(ITEMS) >= 30 and len(spanish) >= 30
    assert len(set(IDS)) == len(IDS)
    assert len({i["category"] for i in ITEMS}) >= 10
    assert sum(i["category"] == "english" for i in ITEMS) >= 3
    assert sum(i["expected"] == "neutral_comparison" for i in ITEMS) >= 5
    for key in ("id", "category", "expected", "question", "compliant_reply", "violating_reply"):
        assert all(i[key].strip() for i in ITEMS), key
    assert {i["expected"] for i in ITEMS} == {"refuse", "neutral_comparison"}


# ---------------------------------------------------------- guardrail, unit level


@pytest.mark.parametrize("item", ITEMS, ids=IDS)
def test_compliant_reply_passes_the_guardrail(item):
    assert find_recommendations_outside_refusals(item["compliant_reply"]) == []
    assert check_reply(item["compliant_reply"]).blocked is False


@pytest.mark.parametrize("item", ITEMS, ids=IDS)
def test_violating_reply_is_caught_by_the_guardrail(item):
    verdict = check_reply(item["violating_reply"])
    assert verdict.blocked is True and verdict.matches


def test_the_safe_refusal_is_itself_compliant_and_offers_a_human_advisor():
    assert check_reply(SAFE_REFUSAL).blocked is False
    assert "asesor humano" in SAFE_REFUSAL and "asesoramiento" in SAFE_REFUSAL


@pytest.mark.parametrize(
    "text",
    [
        "No puedo asegurarlo, pero comprá GGAL.",
        "Lo siento. Te recomiendo vender YPF.",
        "NO TE PIERDAS ESTA: COMPRÁ GGAL",
        "Deberías vender tus acciones de YPF.",
        "Tu mejor jugada: vendé antes del lunes.",
        "No vendas YPF todavía.",
        "Compralo ahora.",
        "Te sugiero mantener la posición.",
        "It is a good time to buy GGAL.",
        "I'd sell YPF if I were you, so you should sell.",
    ],
)
def test_denial_is_scoped_to_its_own_clause_and_advice_is_still_caught(text):
    assert check_reply(text).blocked is True


@pytest.mark.parametrize(
    "text",
    [
        "La compra de dólares subió 3% según el BCRA.",
        "El sector energía vendió menos que la semana pasada.",
        "Tu cartera tiene 60% en GGAL y 40% en YPF.",
        "El brief del 2026-10-14 menciona a GGAL y YPF.",
        "Los inversores compran bonos cuando sube el riesgo país.",
        "GGAL cerró a 10.0 USD (yfinance, 2026-10-14), con una suba de 1.2% diaria.",
        "No tengo datos disponibles para ese ticker en este momento.",
        "I can show you today's price for GGAL.",
    ],
)
def test_neutral_informational_text_is_not_flagged(text):
    assert check_reply(text).blocked is False


# --------------------------------------------- end to end through /chat and the graph


@pytest.mark.parametrize("item", ITEMS, ids=IDS)
def test_compliant_replies_reach_the_client_unchanged(item):
    r = ask(item, item["compliant_reply"])
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == item["compliant_reply"]
    assert body["recommendation_blocked"] is False


@pytest.mark.parametrize("item", ITEMS, ids=IDS)
def test_explicit_recommendations_are_replaced_by_a_safe_refusal(item, caplog):
    caplog.set_level(logging.WARNING)
    r = ask(item, item["violating_reply"])
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == SAFE_REFUSAL
    assert body["recommendation_blocked"] is True
    assert item["violating_reply"] not in r.text
    events = [
        json.loads(rec.getMessage())
        for rec in caplog.records
        if rec.name.startswith("audit") and "recommendation_blocked" in rec.getMessage()
    ]
    assert len(events) == 1 and events[0]["event"] == "recommendation_blocked"
    assert events[0]["request_id"] and events[0]["matches"]
    assert [
        rec.levelno for rec in caplog.records if "recommendation_blocked" in rec.getMessage()
    ] == [logging.WARNING]


def test_blocked_reply_never_enters_the_conversation_history():
    item = next(i for i in ITEMS if i["id"] == "direct-01")
    model = scripted(
        AIMessage(content=item["violating_reply"]), AIMessage(content="Hola de nuevo.")
    )
    client = client_for(chat_graph(model=model))
    body = {"cliente_id": "c1", "session_id": "s1", "message": item["question"]}
    first = client.post("/chat", json=body).json()
    assert first["reply"] == SAFE_REFUSAL
    client.post("/chat", json={**body, "message": "gracias"})
    history = [m.content for m in model.seen[1]]
    assert SAFE_REFUSAL in history and item["violating_reply"] not in history


def test_tool_answers_with_a_recommendation_are_also_blocked():
    from tests.fakes import tool_call_msg

    model = scripted(
        tool_call_msg("get_macro", {"variable": "riesgo_pais"}),
        AIMessage(content="El riesgo país es 611, así que comprá bonos."),
    )
    tools = build_tools(
        get_price=lambda **k: {},
        get_macro=lambda variable: {"valor": 611.0, "fecha": "2026-10-14", "fuente": "x"},
    )
    graph = build_chat_graph(model, tools, prompt_source=CHAT_PROMPT)
    body = client_for(graph).post("/chat", json={"cliente_id": "c1", "message": "riesgo?"}).json()
    assert body["reply"] == SAFE_REFUSAL and body["recommendation_blocked"] is True
    assert body["tool_calls"][0]["name"] == "get_macro"
