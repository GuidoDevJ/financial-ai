import fakeredis
import pytest

from app.brief.store import BriefStore
from app.tools.errors import ToolError
from app.tools.get_last_brief import make_get_last_brief
from app.tools.get_portfolio import make_get_portfolio

PORTFOLIOS = {
    "c1": [
        {"ticker": "GGAL", "cantidad": 10, "mercado": "nyse"},
        {"ticker": "YPFD.BA", "cantidad": 100, "mercado": "byma"},
    ],
    "c2": [{"ticker": "MELI", "cantidad": 2, "mercado": "nyse"}],
    "c3": [{"ticker": "BMA.BA", "cantidad": 5}, {"ticker": "KO", "cantidad": 1}],
}


def prices(table):
    def get_price(ticker, mercado):
        value = table[(ticker, mercado)]
        if isinstance(value, Exception):
            raise value
        return {"ticker": ticker, "precio_cierre": value, "fecha": "2026-10-14", "fuente": "t"}

    return get_price


def macro(valor=1000.0):
    def get_macro(variable):
        assert variable == "dolar_ccl"
        if isinstance(valor, Exception):
            raise valor
        return {"variable": variable, "valor": valor, "fecha": "2026-10-14", "fuente": "t"}

    return get_macro


def build(table, valor=1000.0, portfolios=PORTFOLIOS):
    return make_get_portfolio(
        get_price=prices(table), get_macro=macro(valor), portfolios=portfolios
    )


# ---------------------------------------------------------------- portfolio


def test_valuation_and_weights_in_single_currency():
    fn = build({("MELI", "nyse"): 100.0})
    [row] = fn(cliente_id="c2")
    assert row == {
        "ticker": "MELI",
        "cantidad": 2,
        "precio_actual": 100.0,
        "moneda": "USD",
        "valorizacion": 200.0,
        "peso_porcentual": 100.0,
    }


def test_mixed_currencies_weights_use_ccl_conversion():
    # GGAL: 10 * 10 USD = 100 USD = 100_000 ARS ; YPFD.BA: 100 * 1000 ARS = 100_000 ARS
    fn = build({("GGAL", "nyse"): 10.0, ("YPFD.BA", "byma"): 1000.0}, valor=1000.0)
    rows = {r["ticker"]: r for r in fn(cliente_id="c1")}
    assert rows["GGAL"]["valorizacion"] == 100.0 and rows["GGAL"]["moneda"] == "USD"
    assert rows["YPFD.BA"]["valorizacion"] == 100000.0 and rows["YPFD.BA"]["moneda"] == "ARS"
    assert rows["GGAL"]["peso_porcentual"] == 50.0
    assert rows["YPFD.BA"]["peso_porcentual"] == 50.0


def test_mixed_currencies_without_ccl_leaves_weights_unknown_explicitly():
    fn = build(
        {("GGAL", "nyse"): 10.0, ("YPFD.BA", "byma"): 1000.0}, valor=RuntimeError("ccl down")
    )
    rows = fn(cliente_id="c1")
    assert all(r["peso_porcentual"] is None for r in rows)
    assert all("CCL" in r["nota_peso"] for r in rows)


def test_price_failure_reported_per_asset_not_zeroed():
    fn = build({("GGAL", "nyse"): ToolError("NO_DATA", "no price"), ("YPFD.BA", "byma"): 1000.0})
    rows = {r["ticker"]: r for r in fn(cliente_id="c1")}
    assert rows["GGAL"]["error"] == "NO_DATA"
    assert rows["GGAL"]["cantidad"] == 10
    assert "precio_actual" not in rows["GGAL"] and "valorizacion" not in rows["GGAL"]
    assert rows["YPFD.BA"]["peso_porcentual"] == 100.0  # only priced asset(s) share the total
    assert "nota_peso" in rows["YPFD.BA"]


def test_unexpected_price_exception_also_reported_per_asset():
    fn = build({("MELI", "nyse"): ValueError("bad")})
    [row] = fn(cliente_id="c2")
    assert row["error"] == "PRICE_ERROR" and "bad" in row["message"]


def test_unknown_client_is_explicit_error():
    with pytest.raises(ToolError) as exc:
        build({})(cliente_id="nope")
    assert exc.value.code == "CLIENT_NOT_FOUND"


def test_empty_client_id_invalid():
    with pytest.raises(ToolError) as exc:
        build({})(cliente_id=" ")
    assert exc.value.code == "INVALID_PARAMS"


def test_market_inferred_from_suffix_when_missing():
    seen = []

    def get_price(ticker, mercado):
        seen.append((ticker, mercado))
        return {"precio_cierre": 1.0}

    fn = make_get_portfolio(get_price=get_price, get_macro=macro(), portfolios=PORTFOLIOS)
    fn(cliente_id="c3")
    assert seen == [("BMA.BA", "byma"), ("KO", "nyse")]


def test_default_mock_file_loads_and_covers_three_clients():
    fn = make_get_portfolio(
        get_price=lambda ticker, mercado: {"precio_cierre": 1.0}, get_macro=macro(1.0)
    )
    for cid in ("client_001", "client_002", "client_003"):
        assert len(fn(cliente_id=cid)) >= 3


# -------------------------------------------------------------------- briefs


@pytest.fixture
def briefs():
    return BriefStore(fakeredis.FakeRedis(decode_responses=True))


def test_save_get_roundtrip_uses_spec_key(briefs):
    briefs.save("c1", "2026-10-14", "texto", ["GGAL", "YPF"])
    assert briefs.redis.exists("brief:c1:2026-10-14")
    assert briefs.get("c1", "2026-10-14") == {
        "fecha": "2026-10-14",
        "texto_brief": "texto",
        "activos_mencionados": ["GGAL", "YPF"],
    }
    assert briefs.get("c1", "2026-10-13") is None
    assert briefs.get("c2", "2026-10-14") is None


def test_latest_returns_most_recent_date_not_last_written(briefs):
    briefs.save("c1", "2026-10-14", "nuevo", [])
    briefs.save("c1", "2026-10-09", "viejo", [])
    assert briefs.latest("c1")["fecha"] == "2026-10-14"
    assert briefs.latest("c2") is None


def test_latest_skips_expired_entries(briefs):
    briefs.save("c1", "2026-10-14", "nuevo", [])
    briefs.save("c1", "2026-10-09", "viejo", [])
    briefs.redis.delete("brief:c1:2026-10-14")  # e.g. evicted
    assert briefs.latest("c1")["fecha"] == "2026-10-09"


def test_save_overwrites_same_day(briefs):
    briefs.save("c1", "2026-10-14", "a", [])
    briefs.save("c1", "2026-10-14", "b", [])
    assert briefs.get("c1", "2026-10-14")["texto_brief"] == "b"
    assert briefs.redis.zcard("briefidx:c1") == 1


@pytest.mark.parametrize("bad", ["", "a:b", "x y", "../x"])
def test_store_rejects_unsafe_client_ids(briefs, bad):
    with pytest.raises(ValueError):
        briefs.save(bad, "2026-10-14", "t", [])


def test_store_rejects_bad_date(briefs):
    with pytest.raises(ValueError):
        briefs.save("c1", "14/10/2026", "t", [])


# --------------------------------------------------------------- get_last_brief


def test_tool_latest_and_by_date(briefs):
    briefs.save("c1", "2026-10-14", "hoy", ["GGAL"])
    briefs.save("c1", "2026-10-13", "ayer", [])
    fn = make_get_last_brief(store=briefs)
    assert fn(cliente_id="c1")["texto_brief"] == "hoy"
    assert fn(cliente_id="c1", fecha="2026-10-13")["texto_brief"] == "ayer"


def test_tool_none_found_is_explicit_error(briefs):
    fn = make_get_last_brief(store=briefs)
    with pytest.raises(ToolError) as exc:
        fn(cliente_id="c1")
    assert exc.value.code == "NO_BRIEF_FOUND"
    with pytest.raises(ToolError) as exc:
        fn(cliente_id="c1", fecha="2026-01-01")
    assert exc.value.code == "NO_BRIEF_FOUND"


def test_tool_validates_params(briefs):
    fn = make_get_last_brief(store=briefs)
    for kwargs in (
        {"cliente_id": ""},
        {"cliente_id": "c1", "fecha": "ayer"},
        {"cliente_id": "a:b"},
    ):
        with pytest.raises(ToolError) as exc:
            fn(**kwargs)
        assert exc.value.code == "INVALID_PARAMS"


def test_tool_redis_down_is_explicit_error():
    import redis

    class BadRedis:
        def get(self, *a):
            raise redis.ConnectionError("down")

        zrevrange = get

    fn = make_get_last_brief(store=BriefStore(BadRedis()))
    with pytest.raises(ToolError) as exc:
        fn(cliente_id="c1")
    assert exc.value.code == "STORAGE_UNAVAILABLE"


def test_prices_are_fetched_concurrently_and_rows_keep_holding_order():
    import time

    holdings = [{"ticker": f"T{i}", "cantidad": 1, "mercado": "nyse"} for i in range(6)]

    def slow_price(ticker, mercado):
        time.sleep(0.2)
        if ticker == "T2":
            raise ToolError("NO_DATA", "n/a")
        return {"ticker": ticker, "precio_cierre": 10.0, "fecha": "2026-10-14", "fuente": "t"}

    tool = make_get_portfolio(get_price=slow_price, get_macro=macro(), portfolios={"c9": holdings})
    started = time.perf_counter()
    rows = tool(cliente_id="c9")
    assert time.perf_counter() - started < 0.7  # sequential would take 1.2s
    assert [r["ticker"] for r in rows] == [f"T{i}" for i in range(6)]
    assert rows[2]["error"] == "NO_DATA" and "valorizacion" in rows[0]
