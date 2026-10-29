"""Audit/logging coverage for every tool: success, cache hit, error and fallback paths."""

import json
import logging
from datetime import UTC, date, datetime, timedelta

import fakeredis
import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from llama_index.core.vector_stores import SimpleVectorStore

from app.agents.chat_graph import build_chat_graph
from app.brief.store import BriefStore
from app.core.audit import (
    AUDIT_LOGGER_NAME,
    annotate,
    audited_tool,
    request_context,
    sanitize,
)
from app.core.config import Settings
from app.main import app, get_chat_graph
from app.rag.ingest_news import NewsItem, item_to_document
from app.rag.store import NewsStore
from app.tools.cache import cached_call
from app.tools.errors import ToolError
from app.tools.get_last_brief import make_get_last_brief
from app.tools.get_macro import default_macro_providers, make_get_macro
from app.tools.get_portfolio import make_get_portfolio
from app.tools.get_price import IolPriceProvider, PriceSeries, make_get_price
from app.tools.registry import build_tools
from app.tools.search_news import make_search_news
from tests.fakes import CHAT_PROMPT, BagOfWordsEmbedding, scripted, tool_call_msg

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
SECRET = "S3cr3t-Pa55w0rd-do-not-log"
TOKEN = "tok-abcdef-1234567890-do-not-log"


@pytest.fixture(autouse=True)
def _debug_level(caplog):
    caplog.set_level(logging.DEBUG, logger="audit")
    caplog.set_level(logging.DEBUG, logger="app")


def events(caplog, name=None):
    """Parsed JSON records emitted under the `audit.*` loggers."""
    out = []
    for r in caplog.records:
        if not r.name.startswith("audit."):
            continue
        try:
            data = json.loads(r.getMessage())
        except ValueError:
            continue
        data["_level"], data["_logger"], data["_exc"] = r.levelname, r.name, r.exc_info
        out.append(data)
    if name:
        out = [e for e in out if e.get("event") == name]
    return out


def tool_records(caplog, tool=None):
    recs = [r for r in events(caplog) if r["_logger"] == AUDIT_LOGGER_NAME]
    return [r for r in recs if tool is None or r["tool"] == tool]


def all_text(caplog):
    return "\n".join(r.getMessage() for r in caplog.records)


def redis_():
    return fakeredis.FakeRedis(decode_responses=True)


def series(closes, end=date(2026, 10, 14), fuente="fake"):
    n = len(closes)
    return PriceSeries(
        fuente=fuente, points=[(end - timedelta(days=n - 1 - i), c) for i, c in enumerate(closes)]
    )


# ------------------------------------------------------------------ sanitizer


def test_sanitize_redacts_secrets_and_truncates_text():
    out = sanitize(
        {
            "password": SECRET,
            "iol_password": SECRET,
            "Authorization": f"Bearer {TOKEN}",
            "aws_secret_access_key": "x" * 40,
            "note": f"call with password={SECRET} and key AKIAABCDEFGHIJKLMNOP",
            "long": "a" * 5000,
            "cantidad": 10,
            "valorizacion": 12345.67,
            "ticker": "GGAL",
            "n": 3,
        },
        max_len=100,
    )
    dumped = json.dumps(out)
    assert SECRET not in dumped and TOKEN not in dumped
    assert "AKIAABCDEFGHIJKLMNOP" not in dumped
    assert out["password"] == "[REDACTED]" and out["Authorization"] == "[REDACTED]"
    assert out["cantidad"] == "[REDACTED]" and out["valorizacion"] == "[REDACTED]"
    assert len(out["long"]) < 200 and "truncated" in out["long"]
    assert out["ticker"] == "GGAL" and out["n"] == 3


def test_sanitize_handles_nested_and_odd_types():
    out = sanitize({"a": [{"token": "t"}, ("x", object())]})
    assert out["a"][0] == {"token": "[REDACTED]"}
    assert isinstance(out["a"][1][1], str)


# ------------------------------------------------------- uniform audit record


def test_record_has_uniform_fields_and_context(caplog):
    @audited_tool
    def t(cliente_id, query):
        annotate(source="redis", cache="hit")
        return [1, 2, 3]

    with request_context(request_id="req-1", cliente_id="c1", session_id="s1"):
        t("c1", "x" * 1000)
    [rec] = tool_records(caplog)
    assert rec["request_id"] == "req-1"
    assert rec["cliente_id"] == "c1" and rec["session_id"] == "s1"
    assert rec["status"] == "ok" and rec["_level"] == "INFO"
    assert rec["source"] == "redis" and rec["cache"] == "hit"
    assert rec["result_size"] == 3
    assert len(rec["params"]["query"]) < 400
    assert rec["duration_ms"] >= 0


def test_request_context_generates_and_nests(caplog):
    with request_context() as outer:
        assert outer
        with request_context(cliente_id="c9") as inner:
            assert inner == outer


def test_expected_error_is_warning_without_traceback(caplog):
    @audited_tool
    def t():
        raise ToolError("NO_DATA", "nothing")

    with pytest.raises(ToolError):
        t()
    [rec] = tool_records(caplog)
    assert rec["status"] == "error" and rec["error_code"] == "NO_DATA"
    assert rec["error_type"] == "ToolError"
    assert rec["_level"] == "WARNING" and not rec["_exc"]


def test_unexpected_error_is_error_with_traceback_and_redacted(caplog):
    @audited_tool
    def t():
        raise RuntimeError(f"boom password={SECRET}")

    with pytest.raises(RuntimeError):
        t()
    [rec] = tool_records(caplog)
    assert rec["_level"] == "ERROR" and rec["_exc"]
    assert rec["error_type"] == "RuntimeError" and rec["error_code"] is None
    assert SECRET not in json.dumps({k: v for k, v in rec.items() if k != "_exc"})
    assert SECRET not in caplog.text  # includes the formatted traceback
    assert "RuntimeError" in caplog.text


def test_registry_audits_plain_functions_once(caplog):
    @audited_tool
    def get_macro(variable):
        return {"valor": 1}

    tools = {
        t.name: t
        for t in build_tools(
            get_price=lambda ticker, mercado: {"precio_cierre": 1}, get_macro=get_macro
        )
    }
    with request_context(request_id="r2", cliente_id="c1"):
        tools["get_price"].invoke({"ticker": "GGAL", "mercado": "nyse"})
        tools["get_macro"].invoke({"variable": "riesgo_pais"})
    assert [r["tool"] for r in tool_records(caplog)] == ["get_price", "get_macro"]
    assert all(r["request_id"] == "r2" and r["cliente_id"] == "c1" for r in tool_records(caplog))


def test_registry_logs_unexpected_exception_once_with_traceback(caplog):
    def get_price(ticker, mercado):
        raise RuntimeError("kaput")

    tools = {t.name: t for t in build_tools(get_price=get_price, get_macro=lambda variable: {})}
    out = tools["get_price"].invoke({"ticker": "GGAL", "mercado": "nyse"})
    assert json.loads(out)["error"] == "INTERNAL_ERROR"
    [rec] = tool_records(caplog, "get_price")
    assert rec["_level"] == "ERROR" and rec["_exc"]


def test_chat_endpoint_propagates_request_and_client_ids(caplog):
    r = redis_()
    tools = build_tools(
        get_price=lambda ticker, mercado: {"precio_cierre": 1},
        get_macro=make_get_macro(
            redis_client=r,
            providers={
                "riesgo_pais": [lambda v: {"valor": 6.0, "fecha": "2026-10-14", "fuente": "x"}]
            },
        ),
    )
    graph = build_chat_graph(
        scripted(tool_call_msg("get_macro", {"variable": "riesgo_pais"}), AIMessage(content="ok")),
        tools,
        prompt_source=CHAT_PROMPT,
    )

    async def dep():
        return graph

    app.dependency_overrides[get_chat_graph] = dep
    try:
        resp = TestClient(app).post(
            "/chat", json={"cliente_id": "c1", "session_id": "s1", "message": "hola " * 500}
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    [rec] = tool_records(caplog, "get_macro")
    assert rec["request_id"] and rec["cliente_id"] == "c1" and rec["session_id"] == "s1"
    assert "hola hola hola hola hola hola hola hola hola hola" not in all_text(caplog)


# ---------------------------------------------------------------------- cache


def test_cache_miss_then_hit_logged_with_key_and_ttl(caplog):
    r = redis_()
    cached_call(r, "k1", 60, lambda: {"a": 1})
    cached_call(r, "k1", 60, lambda: {"a": 1})
    evs = events(caplog, "cache")
    assert [e["result"] for e in evs] == ["miss", "hit"]
    assert all(e["key"] == "k1" for e in evs)
    assert evs[0]["ttl"] == 60
    assert 0 < evs[1]["ttl_remaining"] <= 60


def test_cache_hit_skips_ttl_lookup_unless_debug(caplog):
    class CountingRedis(fakeredis.FakeRedis):
        ttl_calls = 0

        def ttl(self, name):
            type(self).ttl_calls += 1
            return super().ttl(name)

    r = CountingRedis(decode_responses=True)
    caplog.set_level(logging.INFO, logger="audit.tools")
    cached_call(r, "k1", 60, lambda: {"a": 1})
    cached_call(r, "k1", 60, lambda: {"a": 1})
    assert CountingRedis.ttl_calls == 0


def test_cache_failure_warns_with_traceback(caplog):
    class Down:
        def get(self, *a):
            raise ConnectionError("down")

        def setex(self, *a):
            raise ConnectionError("down")

    assert cached_call(Down(), "k", 60, lambda: 1) == 1
    warns = [e for e in events(caplog, "cache") if e["_level"] == "WARNING"]
    assert {w["result"] for w in warns} == {"read_error", "write_error"}
    assert all(w["_exc"] for w in warns)


# ------------------------------------------------------------------- get_price


def test_price_success_and_cache_hit_audit(caplog):
    tool = make_get_price(
        redis_client=redis_(),
        providers={"nyse": lambda t: series([10, 11, 12])},
        today=lambda: date(2026, 10, 15),
    )
    with request_context(cliente_id="c1"):
        tool("ggal", "nyse")
        tool("GGAL", "nyse")
    first, second = tool_records(caplog, "get_price")
    assert first["cache"] == "miss" and second["cache"] == "hit"
    assert first["source"] == "fake" and first["as_of"] == "2026-10-14"
    assert first["stale"] is False
    assert first["params"] == {"ticker": "ggal", "mercado": "nyse"}
    price_evs = events(caplog, "price")
    assert price_evs[0]["ticker"] == "GGAL" and price_evs[0]["source"] == "fake"
    assert price_evs[0]["as_of"] == "2026-10-14" and price_evs[0]["stale"] is False


def test_price_stale_flag_warns(caplog):
    tool = make_get_price(
        redis_client=redis_(),
        providers={"nyse": lambda t: series([10, 11], end=date(2026, 10, 1))},
        today=lambda: date(2026, 10, 15),
    )
    tool("GGAL", "nyse")
    [rec] = tool_records(caplog, "get_price")
    assert rec["stale"] is True
    [ev] = events(caplog, "price")
    assert ev["stale"] is True and ev["_level"] == "WARNING"


def test_price_expected_provider_error_warns_without_traceback(caplog):
    def provider(t):
        raise ToolError("NO_DATA", "empty")

    tool = make_get_price(redis_client=redis_(), providers={"nyse": provider})
    with pytest.raises(ToolError):
        tool("GGAL", "nyse")
    [rec] = tool_records(caplog, "get_price")
    assert rec["status"] == "error" and rec["error_code"] == "NO_DATA" and rec["cache"] == "miss"
    assert rec["_level"] == "WARNING" and not rec["_exc"]


def test_price_unexpected_provider_error_has_traceback(caplog):
    def provider(t):
        raise RuntimeError("socket exploded")

    tool = make_get_price(redis_client=redis_(), providers={"nyse": provider})
    with pytest.raises(ToolError):
        tool("GGAL", "nyse")
    errs = [e for e in events(caplog, "provider_error") if e["_exc"]]
    assert errs and errs[0]["source"] == "nyse" and errs[0]["_level"] == "ERROR"


def test_price_invalid_params_logged(caplog):
    tool = make_get_price(redis_client=redis_(), providers={"nyse": lambda t: series([1])})
    with pytest.raises(ToolError):
        tool("GGAL", "mars")
    [rec] = tool_records(caplog, "get_price")
    assert rec["error_code"] == "INVALID_PARAMS" and rec["cache"] is None


def iol(settings=None):
    s = settings or Settings(_env_file=None, iol_user="u", iol_password=SECRET)
    return IolPriceProvider(s, client=httpx.Client())


@respx.mock
def test_iol_logs_status_and_latency_without_secrets(caplog):
    respx.post("https://api.invertironline.com/token").mock(
        return_value=httpx.Response(200, json={"access_token": TOKEN, "expires_in": 900})
    )
    respx.get(url__regex=r".*seriehistorica.*").mock(
        return_value=httpx.Response(
            200, json=[{"ultimoPrecio": 10.0, "fechaHora": "2026-10-14T17:00:00"}]
        )
    )
    iol()("YPFD.BA")
    ups = events(caplog, "upstream")
    assert [(u["source"], u["status"]) for u in ups] == [("iol", 200), ("iol", 200)]
    assert all(u["latency_ms"] >= 0 for u in ups)
    assert ups[0]["operation"] == "token"
    assert SECRET not in all_text(caplog) and TOKEN not in all_text(caplog)


@respx.mock
def test_iol_auth_failure_logged_without_secrets(caplog):
    respx.post("https://api.invertironline.com/token").mock(return_value=httpx.Response(400))
    with pytest.raises(ToolError):
        iol()("YPFD.BA")
    [up] = events(caplog, "upstream")
    assert up["status"] == 400 and up["_level"] == "WARNING"
    assert SECRET not in all_text(caplog)


def test_iol_missing_credentials_warns(caplog):
    with pytest.raises(ToolError) as e:
        IolPriceProvider(Settings(_env_file=None), client=httpx.Client())("YPFD.BA")
    assert e.value.code == "CREDENTIALS_MISSING"
    assert any(x["_level"] == "WARNING" for x in events(caplog, "iol_credentials"))


# ------------------------------------------------------------------- get_macro


def test_macro_fallback_logged_with_reason(caplog):
    def primary(variable):
        raise ToolError("PROVIDER_UNAVAILABLE", f"down password={SECRET}")

    def backup(variable):
        return {"valor": 1500.0, "fecha": "2026-10-14", "fuente": "backup"}

    tool = make_get_macro(redis_client=redis_(), providers={"dolar_ccl": [primary, backup]})
    tool("dolar_ccl")
    [fb] = events(caplog, "fallback")
    assert fb["from"] == "primary" and fb["to"] == "backup" and fb["_level"] == "WARNING"
    assert "PROVIDER_UNAVAILABLE" in fb["reason"]
    [rec] = tool_records(caplog, "get_macro")
    assert rec["source"] == "backup" and rec["fallback"] is True and rec["cache"] == "miss"
    assert SECRET not in all_text(caplog)


def test_macro_cache_hit_audit_and_all_failed(caplog):
    r = redis_()
    ok = lambda v: {"valor": 6.0, "fecha": "2026-10-14", "fuente": "x"}  # noqa: E731
    tool = make_get_macro(redis_client=r, providers={"riesgo_pais": [ok]})
    tool("riesgo_pais")
    tool("riesgo_pais")
    first, second = tool_records(caplog, "get_macro")
    assert (first["cache"], second["cache"]) == ("miss", "hit")
    assert second["result_size"] == 4

    def bad(v):
        raise RuntimeError("x")

    tool2 = make_get_macro(redis_client=redis_(), providers={"dolar_ccl": [bad, bad]})
    with pytest.raises(ToolError):
        tool2("dolar_ccl")
    [ev] = events(caplog, "providers_exhausted")
    assert ev["_level"] == "WARNING" and ev["attempts"] == 2
    assert tool_records(caplog, "get_macro")[-1]["error_code"] == "PROVIDER_UNAVAILABLE"


@respx.mock
def test_macro_http_status_and_latency_logged(caplog):
    respx.get("https://dolarapi.com/v1/dolares/contadoconliqui").mock(
        return_value=httpx.Response(
            200, json={"venta": 1500, "compra": 1490, "fechaActualizacion": "2026-10-14T10:00:00Z"}
        )
    )
    tool = make_get_macro(redis_client=redis_(), providers=default_macro_providers(httpx.Client()))
    tool("dolar_ccl")
    [up] = events(caplog, "upstream")
    assert up["source"] == "dolarapi" and up["status"] == 200 and up["latency_ms"] >= 0
    assert up["url"] == "https://dolarapi.com/v1/dolares/contadoconliqui"


@respx.mock
def test_macro_http_error_status_warns_and_strips_query(caplog):
    respx.get(url__regex=r"https://api.bcra.gob.ar/.*").mock(return_value=httpx.Response(503))
    tool = make_get_macro(redis_client=redis_(), providers=default_macro_providers(httpx.Client()))
    with pytest.raises(ToolError):
        tool("tasa_bcra")
    ups = events(caplog, "upstream")  # primary and secondary BCRA routes
    assert len(ups) == 2
    for up in ups:
        assert up["source"] == "bcra" and up["status"] == 503 and up["_level"] == "WARNING"
        assert "?" not in up["url"]


# ----------------------------------------------------------------- search_news


def news_tool(caplog_items=True):
    store = NewsStore(SimpleVectorStore(), BagOfWordsEmbedding())
    if caplog_items:
        store.upsert(
            [
                item_to_document(
                    NewsItem(
                        "Galicia sube tras balance",
                        "GGAL lidera el panel",
                        "https://x.com/1",
                        "Ambito",
                        NOW - timedelta(hours=3),
                    )
                )
            ]
        )
    return make_search_news(store=store, settings=Settings(_env_file=None), now=lambda: NOW)


def test_news_search_logs_query_topk_hits_scores_and_ticker(caplog):
    tool = news_tool()
    out = tool(query="Galicia balance", ticker="GGAL", horas=48)
    assert len(out) == 1
    [ev] = events(caplog, "news_search")
    assert ev["query"] == "Galicia balance" and ev["ticker"] == "GGAL"
    assert ev["top_k"] == 4 and ev["hours"] == 48 and ev["hits"] == 1
    assert ev["score_min"] is not None and ev["score_max"] >= ev["score_min"]
    [rec] = tool_records(caplog, "search_news")
    assert rec["result_size"] == 1 and rec["source"] == "news_store"
    assert "Galicia sube tras balance" not in all_text(caplog)
    assert "GGAL lidera el panel" not in all_text(caplog)


def test_news_long_query_truncated_and_no_results_warns(caplog):
    tool = news_tool(caplog_items=False)
    with pytest.raises(ToolError):
        tool(query="zzz " * 300)
    [ev] = events(caplog, "news_search")
    assert len(ev["query"]) < 400 and ev["hits"] == 0
    [rec] = tool_records(caplog, "search_news")
    assert rec["error_code"] == "NO_NEWS_FOUND" and rec["_level"] == "WARNING"


def test_news_store_failure_logged_with_traceback(caplog):
    def factory():
        raise RuntimeError("redis stack down")

    tool = make_search_news(store_factory=factory, settings=Settings(_env_file=None))
    with pytest.raises(ToolError):
        tool(query="x")
    errs = [e for e in events(caplog, "news_store_error") if e["_exc"]]
    assert errs and errs[0]["_level"] == "ERROR"
    assert tool_records(caplog, "search_news")[0]["error_code"] == "NEWS_STORE_UNAVAILABLE"


# ---------------------------------------------------------------- get_portfolio


def portfolio_tool(price=None, macro=None):
    def get_price(ticker, mercado):
        if price:
            return price(ticker, mercado)
        return {"precio_cierre": 123456.0}

    return make_get_portfolio(
        get_price=get_price,
        get_macro=macro or (lambda variable: {"valor": 1000.0}),
        portfolios={
            "c1": [
                {"ticker": "GGAL", "cantidad": 7777, "mercado": "nyse"},
                {"ticker": "YPFD.BA", "cantidad": 8888, "mercado": "byma"},
            ]
        },
    )


def test_portfolio_found_logs_counts_only(caplog):
    with request_context(cliente_id="c1"):
        portfolio_tool()(cliente_id="c1")
    [ev] = events(caplog, "portfolio")
    assert ev["cliente_id"] == "c1" and ev["found"] is True
    assert ev["n_holdings"] == 2 and ev["priced"] == 2 and ev["failed"] == 0
    [rec] = tool_records(caplog, "get_portfolio")
    assert rec["result_size"] == 2 and rec["source"] == "mock_portfolios"
    text = all_text(caplog)
    for leaked in ("7777", "8888", "123456", "YPFD"):
        assert leaked not in text


def test_portfolio_not_found_warns(caplog):
    with pytest.raises(ToolError):
        portfolio_tool()(cliente_id="nope")
    [ev] = events(caplog, "portfolio")
    assert ev["found"] is False and ev["_level"] == "WARNING"
    assert tool_records(caplog, "get_portfolio")[0]["error_code"] == "CLIENT_NOT_FOUND"


def test_portfolio_price_failures_counted_by_code(caplog):
    def price(ticker, mercado):
        if ticker == "GGAL":
            raise ToolError("NO_DATA", "n/a")
        raise RuntimeError("boom")

    portfolio_tool(price=price)(cliente_id="c1")
    [ev] = events(caplog, "portfolio")
    assert ev["failed"] == 2 and ev["priced"] == 0
    assert ev["failures"] == {"NO_DATA": 1, "PRICE_ERROR": 1}
    assert ev["_level"] == "WARNING"


def test_portfolio_weight_fallback_when_ccl_missing_warns(caplog):
    def macro(variable):
        raise ToolError("PROVIDER_UNAVAILABLE", "down")

    portfolio_tool(macro=macro)(cliente_id="c1")
    [ev] = events(caplog, "portfolio_weights")
    assert ev["_level"] == "WARNING" and ev["reason"] == "ccl_unavailable"


# --------------------------------------------------------------- get_last_brief


def brief_tool():
    store = BriefStore(redis_())
    store.save("c1", "2026-10-14", "SECRET BRIEF TEXT body", ["GGAL", "YPFD"])
    return make_get_last_brief(store=store)


def test_last_brief_found_logs_ids_and_counts_not_text(caplog):
    with request_context(cliente_id="c1"):
        brief_tool()(cliente_id="c1")
    [ev] = events(caplog, "last_brief")
    assert ev["cliente_id"] == "c1" and ev["found"] is True
    assert ev["fecha"] == "2026-10-14" and ev["assets"] == 2 and ev["text_chars"] > 0
    [rec] = tool_records(caplog, "get_last_brief")
    assert rec["source"] == "redis" and rec["result_size"] == 2
    assert "SECRET BRIEF TEXT" not in all_text(caplog)


def test_last_brief_not_found_and_invalid_and_storage_down(caplog):
    tool = brief_tool()
    with pytest.raises(ToolError):
        tool(cliente_id="c2")
    with pytest.raises(ToolError):
        tool(cliente_id="bad id!")
    ev = events(caplog, "last_brief")[0]
    assert ev["found"] is False and ev["_level"] == "WARNING"
    codes = [r["error_code"] for r in tool_records(caplog, "get_last_brief")]
    assert codes == ["NO_BRIEF_FOUND", "INVALID_PARAMS"]

    import redis as redis_lib

    class Down:
        def zrevrange(self, *a):
            raise redis_lib.RedisError("down")

    with pytest.raises(ToolError):
        make_get_last_brief(store=BriefStore(Down()))(cliente_id="c1")
    assert tool_records(caplog, "get_last_brief")[-1]["error_code"] == "STORAGE_UNAVAILABLE"
    assert any(e["_exc"] for e in events(caplog, "storage_error"))
