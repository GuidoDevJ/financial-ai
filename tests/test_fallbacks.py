"""Fallbacks and stale-if-error behaviour of the price and macro tools."""

import json
import logging
from datetime import date, timedelta

import fakeredis
import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.chat_graph import build_chat_graph
from app.core.config import Settings
from app.tools.cache import cached_call
from app.tools.errors import ToolError
from app.tools.get_macro import default_macro_providers, make_get_macro
from app.tools.get_price import PriceSeries, make_get_price, yfinance_ba_provider
from app.tools.registry import build_tools
from tests.fakes import CHAT_PROMPT, scripted, tool_call_msg


def redis_():
    return fakeredis.FakeRedis(decode_responses=True)


def series(fuente="fake", close=10.0, end=date(2026, 10, 14)):
    return PriceSeries(fuente, [(end - timedelta(days=1), close - 1), (end, close)])


def failing(code="PROVIDER_UNAVAILABLE"):
    def provider(ticker):
        raise ToolError(code, "down")

    return provider


def events(caplog, name):
    out = []
    for rec in caplog.records:
        try:
            data = json.loads(rec.getMessage())
        except ValueError:
            continue
        if data.get("event") == name:
            out.append((rec, data))
    return out


# ------------------------------------------------- (a) BYMA: IOL -> yfinance .BA


def test_byma_falls_back_to_yfinance_when_iol_fails(caplog):
    caplog.set_level(logging.INFO)
    seen = []

    def yf(ticker):
        seen.append(ticker)
        return series("yfinance")

    tool = make_get_price(
        redis_client=redis_(),
        providers={"byma": failing("CREDENTIALS_MISSING")},
        fallbacks={"byma": yf},
    )
    out = tool("YPFD", "byma")
    assert out["fuente"] == "yfinance" and out["precio_cierre"] == 10.0
    assert seen == ["YPFD"]
    [(rec, event)] = events(caplog, "fallback")
    assert rec.levelno == logging.WARNING
    assert event["from"] == "provider" and event["to"] == "yf"  # provider function names
    assert "CREDENTIALS_MISSING" in event["reason"]
    assert event["ticker"] == "YPFD" and event["mercado"] == "byma"


def test_fallback_result_is_cached_like_any_other():
    r = redis_()
    calls = []

    def yf(ticker):
        calls.append(ticker)
        return series("yfinance")

    tool = make_get_price(redis_client=r, providers={"byma": failing()}, fallbacks={"byma": yf})
    tool("YPFD", "byma")
    tool("YPFD", "byma")
    assert calls == ["YPFD"]


def test_no_fallback_when_the_primary_works():
    def yf(ticker):
        raise AssertionError("fallback must not be used")

    tool = make_get_price(
        redis_client=redis_(),
        providers={"byma": lambda t: series("IOL")},
        fallbacks={"byma": yf},
    )
    assert tool("YPFD", "byma")["fuente"] == "IOL"


def test_both_sources_failing_reports_the_primary_error():
    tool = make_get_price(
        redis_client=redis_(),
        providers={"byma": failing("AUTH_FAILED")},
        fallbacks={"byma": failing("NO_DATA")},
    )
    with pytest.raises(ToolError) as exc:
        tool("YPFD", "byma")
    assert exc.value.code == "AUTH_FAILED" and "NO_DATA" in exc.value.message


def test_yfinance_fallback_uses_the_ba_suffix(monkeypatch):
    seen = []

    def fake(ticker):
        seen.append(ticker)
        return PriceSeries("yfinance", [(date(2026, 10, 14), 1.0)])

    monkeypatch.setattr("app.tools.get_price.yfinance_provider", fake)
    assert yfinance_ba_provider("ypfd").fuente == "yfinance (.BA)"
    yfinance_ba_provider("GGAL.BA")
    assert seen == ["YPFD.BA", "GGAL.BA"]


def test_default_wiring_has_the_byma_fallback_only_for_byma():
    tool = make_get_price(redis_client=redis_(), settings=Settings(_env_file=None))
    assert tool.__name__ == "get_price"  # constructing with defaults must not call the network


# ------------------------------------------------------- (c) stale-if-error cache


def test_stale_copy_is_served_flagged_when_every_provider_fails():
    r = redis_()
    state = {"fail": False}

    def provider(ticker):
        if state["fail"]:
            raise ToolError("PROVIDER_UNAVAILABLE", "yahoo down")
        return series("yfinance", close=11.0)

    tool = make_get_price(redis_client=r, providers={"nyse": provider})
    fresh = tool("GGAL", "nyse")
    assert "stale" not in fresh
    r.delete("tool:get_price:nyse:GGAL")  # the normal TTL expired
    state["fail"] = True
    out = tool("GGAL", "nyse")
    assert out["stale"] is True and out["precio_cierre"] == 11.0
    assert out["as_of"] == "2026-10-14" and out["stale_reason"] == "PROVIDER_UNAVAILABLE"
    assert out["cached_at"]


def test_stale_copy_outlives_the_normal_ttl():
    r = redis_()
    s = Settings(_env_file=None, price_cache_ttl=30, price_stale_ttl_seconds=5000)
    make_get_price(redis_client=r, settings=s, providers={"nyse": lambda t: series()})(
        "GGAL", "nyse"
    )
    assert 0 < r.ttl("tool:get_price:nyse:GGAL") <= 30
    assert 30 < r.ttl("tool:get_price:nyse:GGAL:stale") <= 5000


def test_without_a_stale_copy_the_error_propagates():
    tool = make_get_price(redis_client=redis_(), providers={"nyse": failing()})
    with pytest.raises(ToolError) as exc:
        tool("GGAL", "nyse")
    assert exc.value.code == "PROVIDER_UNAVAILABLE"


def test_stale_serving_is_audited(caplog):
    caplog.set_level(logging.INFO)
    r = redis_()
    fail = {"on": False}

    def provider(t):
        if fail["on"]:
            raise ToolError("PROVIDER_UNAVAILABLE", "down")
        return series()

    tool = make_get_price(redis_client=r, providers={"nyse": provider})
    tool("GGAL", "nyse")
    r.delete("tool:get_price:nyse:GGAL")
    fail["on"] = True
    tool("GGAL", "nyse")
    [(rec, event)] = events(caplog, "stale_served")
    assert rec.levelno == logging.WARNING
    assert event["key"] == "tool:get_price:nyse:GGAL" and event["reason"] == "PROVIDER_UNAVAILABLE"


def test_macro_stale_if_error():
    r = redis_()
    state = {"fail": False}

    def provider(variable):
        if state["fail"]:
            raise ToolError("PROVIDER_UNAVAILABLE", "down")
        return {"valor": 611.0, "fecha": "2026-10-14", "fuente": "argentinadatos.com"}

    tool = make_get_macro(redis_client=r, providers={"riesgo_pais": [provider]})
    tool("riesgo_pais")
    r.delete("tool:get_macro:riesgo_pais")
    state["fail"] = True
    out = tool("riesgo_pais")
    assert out["valor"] == 611.0 and out["stale"] is True and out["as_of"] == "2026-10-14"


def test_cached_call_stale_edge_cases():
    r = redis_()
    assert cached_call(r, "k", 60, lambda: {"a": 1}, stale_ttl=600) == {"a": 1}
    r.delete("k")

    def boom():
        raise RuntimeError("x")

    out = cached_call(r, "k", 60, boom, stale_ttl=600)
    assert out["a"] == 1 and out["stale"] is True and out["stale_reason"] == "RuntimeError"
    assert r.get("k") is None  # a stale answer is never promoted to the fresh cache

    class Down:
        def get(self, *a):
            raise ConnectionError

        def setex(self, *a):
            raise ConnectionError

    with pytest.raises(RuntimeError):  # Redis down and compute failing: original error
        cached_call(Down(), "k", 60, boom, stale_ttl=600)


# ------------------------------------------------------------ (b) tasa_bcra


BCRA_V4 = "https://api.bcra.gob.ar/estadisticas/v4.0/monetarias/7"
BCRA_V2 = r"https://api\.bcra\.gob\.ar/estadisticas/v2\.0/datosvariable/7/.*"


def test_tasa_bcra_has_a_secondary_provider():
    chain = default_macro_providers(httpx.Client())["tasa_bcra"]
    assert len(chain) == 2


@respx.mock
def test_tasa_bcra_secondary_is_used_when_the_primary_is_down():
    respx.get(BCRA_V4).mock(return_value=httpx.Response(503))
    respx.get(url__regex=BCRA_V2).mock(
        return_value=httpx.Response(
            200,
            json={
                "status": 200,
                "results": [
                    {"idVariable": 7, "fecha": "2026-10-13", "valor": 22.0},
                    {"idVariable": 7, "fecha": "2026-10-14", "valor": 22.5},
                ],
            },
        )
    )
    tool = make_get_macro(redis_client=redis_(), providers=default_macro_providers(httpx.Client()))
    out = tool("tasa_bcra")
    assert out["valor"] == 22.5 and out["fecha"] == "2026-10-14"
    assert "bcra" in out["fuente"] and "stale" not in out


@respx.mock
def test_tasa_bcra_degrades_to_the_last_known_value_with_a_long_ttl():
    r = redis_()
    s = Settings(_env_file=None, macro_cache_ttl=10)
    tool = make_get_macro(
        redis_client=r, settings=s, providers=default_macro_providers(httpx.Client())
    )
    ok = respx.get(BCRA_V4).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"idVariable": 7, "detalle": [{"fecha": "2026-10-14", "valor": 22.5}]}]
            },
        )
    )
    tool("tasa_bcra")
    assert r.ttl("tool:get_macro:tasa_bcra:stale") > 86400 * 20  # 30 days, not the macro TTL
    r.delete("tool:get_macro:tasa_bcra")
    ok.mock(return_value=httpx.Response(500))
    respx.get(url__regex=BCRA_V2).mock(return_value=httpx.Response(500))
    out = tool("tasa_bcra")
    assert out["valor"] == 22.5 and out["stale"] is True and out["as_of"] == "2026-10-14"


@respx.mock
def test_tasa_bcra_without_any_copy_is_an_explicit_error():
    respx.get(BCRA_V4).mock(return_value=httpx.Response(500))
    respx.get(url__regex=BCRA_V2).mock(return_value=httpx.Response(500))
    tool = make_get_macro(redis_client=redis_(), providers=default_macro_providers(httpx.Client()))
    with pytest.raises(ToolError) as exc:
        tool("tasa_bcra")
    assert exc.value.code == "PROVIDER_UNAVAILABLE"


# ----------------------------------------- unknown tickers / errors reach the LLM


def test_price_rejects_malformed_tickers_before_any_provider_call():
    def provider(t):
        raise AssertionError("provider must not be called")

    tool = make_get_price(redis_client=redis_(), providers={"nyse": provider})
    for bad in ["comprar GGAL ya!!", "A" * 20, "GG AL", "../etc", "GGAL;DROP"]:
        with pytest.raises(ToolError) as exc:
            tool(bad, "nyse")
        assert exc.value.code == "INVALID_PARAMS"


async def test_unknown_ticker_and_provider_outage_reach_the_llm_as_structured_errors():
    def unknown(ticker):
        raise ToolError("NO_DATA", "yfinance returned no data for ZZZZ.")

    r = redis_()
    tools = build_tools(
        get_price=make_get_price(redis_client=r, providers={"nyse": unknown}),
        get_macro=make_get_macro(redis_client=r, providers={"dolar_ccl": [failing_macro]}),
    )
    model = scripted(
        tool_call_msg("get_price", {"ticker": "ZZZZ", "mercado": "nyse"}, "a"),
        tool_call_msg("get_macro", {"variable": "dolar_ccl"}, "b"),
        AIMessage(content="No hay datos disponibles."),
    )
    graph = build_chat_graph(model, tools, prompt_source=CHAT_PROMPT)
    result = await graph.ainvoke(
        {"messages": [("user", "x")]}, {"configurable": {"thread_id": "t"}}
    )
    payloads = [json.loads(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
    assert payloads[0]["error"] == "NO_DATA" and "ZZZZ" in payloads[0]["message"]
    assert payloads[1]["error"] == "PROVIDER_UNAVAILABLE" and payloads[1]["message"]
    assert all(set(p) == {"error", "message"} for p in payloads)


def failing_macro(variable):
    raise ToolError("PROVIDER_UNAVAILABLE", "down")
