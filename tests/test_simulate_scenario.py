import pytest

from app.tools.errors import ToolError
from app.tools.get_portfolio_performance import compute_performance
from app.tools.simulate_scenario import make_simulate_scenario

PORTFOLIOS = {
    "c1": [
        {"ticker": "GGAL", "cantidad": 10, "mercado": "nyse"},
        {"ticker": "PAMP.BA", "cantidad": 100, "mercado": "byma"},
    ],
    "ars": [{"ticker": "PAMP.BA", "cantidad": 100, "mercado": "byma"}],
}
PRICES = {
    "GGAL": {"fecha": "2026-10-21", "precio_cierre": 100.0},
    "PAMP.BA": {"fecha": "2026-10-21", "precio_cierre": 200.0},
}


class Fakes:
    def __init__(self, prices=None, ccl=1000.0):
        self.prices = PRICES if prices is None else prices
        self.ccl = ccl
        self.macro_calls = 0

    def get_price(self, ticker, mercado):
        value = self.prices[ticker]
        if isinstance(value, Exception):
            raise value
        return {"ticker": ticker, **value}

    def get_macro(self, variable):
        self.macro_calls += 1
        if self.ccl is None:
            raise ToolError("PROVIDER_UNAVAILABLE", "down")
        return {"valor": self.ccl, "fecha": "2026-10-21", "fuente": "dolarapi.com"}


def build(fakes=None):
    fakes = fakes or Fakes()
    return make_simulate_scenario(
        get_price=fakes.get_price, get_macro=fakes.get_macro, portfolios=PORTFOLIOS
    )


def test_compute_performance_reports_absolute_values_in_ars():
    holding = {
        "ticker": "GGAL",
        "cantidad": 10,
        "moneda": "USD",
        "precio_actual": 110.0,
        "variacion_pct": 10.0,
    }
    out = compute_performance([holding], fx0=1000, fx1=1100)
    assert out["valor_inicial_ars"] == 1_000_000.0 and out["valor_final_ars"] == 1_210_000.0


def test_ccl_shock_only_moves_usd_assets():
    # Start: GGAL 10 * 100 * 1000 = 1,000,000 + PAMP 20,000. CCL +10% adds 100,000.
    out = build()(cliente_id="c1", ccl_pct=10, precio_pct=0)
    assert out["tipo_cambio_actual"] == 1000.0 and out["tipo_cambio_escenario"] == 1100.0
    assert out["valor_inicial_ars"] == 1_020_000.0 and out["valor_final_ars"] == 1_120_000.0
    assert out["retorno_total_pct"] == pytest.approx(9.8, abs=0.01)
    assert out["efecto_cambiario_pp"] == pytest.approx(9.8, abs=0.01)
    assert out["efecto_precio_pp"] == 0


def test_price_shock_moves_every_asset_in_its_own_currency():
    out = build()(cliente_id="c1", ccl_pct=0, precio_pct=-10)
    assert out["retorno_total_pct"] == pytest.approx(-10.0, abs=0.01)
    assert out["efecto_cambiario_pp"] == 0


def test_both_shocks_combine():
    out = build()(cliente_id="c1", ccl_pct=10, precio_pct=-10)
    assert out["retorno_total_pct"] == pytest.approx(-1.18, abs=0.01)


def test_zero_shocks_change_nothing():
    out = build()(cliente_id="c1", ccl_pct=0, precio_pct=0)
    assert out["retorno_total_pct"] == 0 and out["valor_inicial_ars"] == out["valor_final_ars"]


def test_ars_only_portfolio_never_asks_for_the_rate_and_notes_the_ignored_shock():
    fakes = Fakes()
    out = build(fakes)(cliente_id="ars", ccl_pct=10, precio_pct=0)
    assert fakes.macro_calls == 0 and out["tipo_cambio_actual"] is None
    assert out["retorno_total_pct"] == 0
    assert any("CCL" in n for n in out["notas"])


def test_the_result_is_labelled_hypothetical_and_echoes_the_shocks():
    out = build()(cliente_id="c1", ccl_pct=5, precio_pct=-3)
    assert (out["ccl_pct"], out["precio_pct"]) == (5, -3)
    assert "hipótesis" in out["aviso"] and "no es una predicción" in out["aviso"]
    assert out["fecha_precios"] == "2026-10-21"


def test_holdings_without_price_are_excluded_and_reported():
    prices = {**PRICES, "GGAL": ToolError("NO_DATA", "no price")}
    out = build(Fakes(prices=prices))(cliente_id="c1", ccl_pct=0, precio_pct=-10)
    assert out["excluidos"] == [{"ticker": "GGAL", "error": "NO_DATA"}]
    assert [c["ticker"] for c in out["contribuyentes"]] == ["PAMP.BA"]


def test_missing_rate_with_usd_holdings_is_an_error():
    with pytest.raises(ToolError) as exc:
        build(Fakes(ccl=None))(cliente_id="c1", ccl_pct=10, precio_pct=0)
    assert exc.value.code == "PROVIDER_UNAVAILABLE"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ccl_pct": 150, "precio_pct": 0},
        {"ccl_pct": -51, "precio_pct": 0},
        {"ccl_pct": 0, "precio_pct": -100},
        {"ccl_pct": 0, "precio_pct": float("nan")},
    ],
)
def test_out_of_range_shocks_are_rejected(kwargs):
    with pytest.raises(ToolError) as exc:
        build()(cliente_id="c1", **kwargs)
    assert exc.value.code == "INVALID_PARAMS"


@pytest.mark.parametrize("cid,code", [("  ", "INVALID_PARAMS"), ("nadie", "CLIENT_NOT_FOUND")])
def test_invalid_client_is_rejected(cid, code):
    with pytest.raises(ToolError) as exc:
        build()(cliente_id=cid, ccl_pct=0, precio_pct=0)
    assert exc.value.code == code


def test_the_tool_is_registered_session_bound_with_numeric_shocks():
    from app.tools.registry import build_tools

    fakes = Fakes()
    tools = build_tools(
        get_price=fakes.get_price, get_macro=fakes.get_macro, simulate_scenario=build(fakes)
    )
    tool = next(t for t in tools if t.name == "simulate_scenario")
    props = tool.args_schema.model_json_schema()["properties"]
    assert set(props) == {"ccl_pct", "precio_pct"}
