"""Acceptance criterion: chat response time below 5 seconds in 95% of cases.

Deterministic offline check of the whole /chat path (auth, graph, tools, guardrail, audit) with a
scripted LLM and mocked data sources that sleep a few milliseconds like real ones. It guards the
application overhead; `scripts/bench_chat.py` measures a live deployment.
"""

import json
import logging
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agents.chat_graph import build_chat_graph
from app.core.stats import percentile
from app.main import app, get_chat_graph
from app.tools.registry import build_tools
from tests.fakes import CHAT_PROMPT, scripted, tool_call_msg

CALLS = 100
P95_BUDGET_SECONDS = 5.0
SOURCE_DELAY = 0.005  # per mocked market-data call


def slow_price(ticker, mercado):
    time.sleep(SOURCE_DELAY)
    return {"ticker": ticker, "fecha": "2026-10-14", "precio_cierre": 12.35, "fuente": "mock"}


def slow_macro(variable):
    time.sleep(SOURCE_DELAY)
    return {"variable": variable, "valor": 611.0, "fecha": "2026-10-14", "fuente": "mock"}


def make_client():
    tools = build_tools(get_price=slow_price, get_macro=slow_macro)
    model = scripted(
        tool_call_msg("get_price", {"ticker": "GGAL", "mercado": "nyse"}, "p"),
        tool_call_msg("get_macro", {"variable": "riesgo_pais"}, "m"),
        AIMessage(content="GGAL cerró en 12,35 (mock, 2026-10-14) y el riesgo país en 611."),
    )
    graph = build_chat_graph(model, tools, prompt_source=CHAT_PROMPT)

    async def dep():
        return graph

    app.dependency_overrides[get_chat_graph] = dep
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_p95_chat_latency_is_under_five_seconds(caplog):
    caplog.set_level(logging.INFO, logger="audit.chat")
    client = make_client()
    durations = []
    for i in range(CALLS):
        started = time.perf_counter()
        r = client.post(
            "/chat",
            json={
                "cliente_id": "c1",
                "session_id": f"s{i}",
                "message": "precio GGAL y riesgo país",
            },
        )
        durations.append(time.perf_counter() - started)
        assert r.status_code == 200 and len(r.json()["tool_calls"]) == 2
    assert len(durations) >= 100
    assert percentile(durations, 95) < P95_BUDGET_SECONDS

    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "audit.chat"]
    assert len(records) == CALLS and all(rec["status"] == "ok" for rec in records)
    assert all(isinstance(rec["duration_ms"], float) for rec in records)
    assert percentile([rec["duration_ms"] for rec in records], 95) < P95_BUDGET_SECONDS * 1000


@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [
        ([1, 2, 3, 4, 5], 50, 3),
        ([1, 2, 3, 4, 5], 95, 5),
        ([5], 99, 5),
        (list(range(1, 101)), 95, 95),
    ],
)
def test_percentile_nearest_rank(values, pct, expected):
    assert percentile(values, pct) == expected


def test_percentile_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        percentile([], 95)
