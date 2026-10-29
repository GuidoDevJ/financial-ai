from datetime import date

import pytest

from app.tools.errors import ToolError
from app.tools.get_portfolio_performance import make_get_portfolio_performance

PORTFOLIOS = {
    "c1": [
        {"ticker": "GGAL", "cantidad": 10, "mercado": "nyse"},
        {"ticker": "PAMP.BA", "cantidad": 100, "mercado": "byma"},
    ],
    "ars": [{"ticker": "PAMP.BA", "cantidad": 100, "mercado": "byma"}],
}
PRICES = {
    "GGAL": {"fecha": "2026-10-21", "precio_cierre": 110.0, "variacion_semana": 10.0},
    "PAMP.BA": {"fecha": "2026-10-21", "precio_cierre": 220.0, "variacion_semana": 10.0},
}
CCL = {"2026-10-14": 1000.0, "2026-10-21": 1100.0}


class Recorder:
    def __init__(self, prices=None, ccl=None, macro=1200.0):
        self.prices = PRICES if prices is None else prices
        self.ccl = CCL if ccl is None else ccl
        self.macro = macro
        self.ccl_dates = []
        self.macro_calls = 0

    def get_price(self, ticker, mercado):
        value = self.prices[ticker]
        if isinstance(value, Exception):
            raise value
        return {"ticker": ticker, **value}

    def get_ccl_at(self, fecha):
        self.ccl_dates.append(fecha)
        if fecha not in self.ccl:
            raise ToolError("PROVIDER_UNAVAILABLE", "down")
        return {"valor": self.ccl[fecha], "fecha": fecha, "fuente": "argentinadatos.com"}

    def get_macro(self, variable):
        self.macro_calls += 1
        return {"valor": self.macro, "fecha": "2026-10-21", "fuente": "dolarapi.com"}


def build(rec, today=date(2026, 10, 22)):
    return make_get_portfolio_performance(
        get_price=rec.get_price,
        get_macro=rec.get_macro,
        get_ccl_at=rec.get_ccl_at,
        portfolios=PORTFOLIOS,
        today=lambda: today,
    )


def test_mixed_portfolio_splits_price_and_exchange_effects():
    rec = Recorder()
    out = build(rec)(cliente_id="c1", periodo="semana")

    # GGAL: start 10*100*1000 = 1,000,000 ARS; PAMP: start 100*200 = 20,000 ARS.
    assert rec.ccl_dates == ["2026-10-14", "2026-10-21"]
    assert (out["tipo_cambio_inicial"], out["tipo_cambio_final"]) == (1000.0, 1100.0)
    assert out["retorno_total_pct"] == pytest.approx(20.78, abs=0.01)
    assert out["efecto_cambiario_pp"] == pytest.approx(10.78, abs=0.01)
    assert out["efecto_precio_pp"] == pytest.approx(10.0, abs=0.01)
    assert [c["ticker"] for c in out["contribuyentes"]] == ["GGAL", "PAMP.BA"]
    assert out["periodo"] == "semana" and out["fecha_fin"] == "2026-10-21"
    assert "tenencia actual" in out["supuesto"]


def test_ars_only_portfolio_never_asks_for_exchange_rates():
    rec = Recorder()
    out = build(rec)(cliente_id="ars", periodo="semana")
    assert rec.ccl_dates == [] and rec.macro_calls == 0
    assert out["efecto_cambiario_pp"] == 0 and out["tipo_cambio_inicial"] is None


@pytest.mark.parametrize(
    "periodo,base_date", [("dia", "2026-10-20"), ("semana", "2026-10-14"), ("mes", "2026-09-21")]
)
def test_the_starting_rate_uses_the_same_lookback_as_get_price(periodo, base_date):
    prices = {t: {**v, f"variacion_{periodo}": 5.0} for t, v in PRICES.items()}
    rec = Recorder(prices=prices, ccl={base_date: 1000.0, "2026-10-21": 1100.0})
    build(rec)(cliente_id="c1", periodo=periodo)
    assert rec.ccl_dates[0] == base_date


def test_a_closing_date_of_today_reads_the_live_rate_from_get_macro():
    rec = Recorder(ccl={"2026-10-14": 1000.0})
    out = build(rec, today=date(2026, 10, 21))(cliente_id="c1", periodo="semana")
    assert rec.macro_calls == 1 and out["tipo_cambio_final"] == 1200.0


def test_holdings_without_price_or_variation_are_excluded_with_a_reason():
    prices = {
        "GGAL": ToolError("NO_DATA", "no price"),
        "PAMP.BA": {"fecha": "2026-10-21", "precio_cierre": 220.0, "variacion_semana": None},
    }
    with pytest.raises(ToolError) as exc:
        build(Recorder(prices=prices))(cliente_id="c1", periodo="semana")
    assert exc.value.code == "NO_DATA"

    prices["PAMP.BA"]["variacion_semana"] = 10.0
    out = build(Recorder(prices=prices))(cliente_id="c1", periodo="semana")
    assert [e["ticker"] for e in out["excluidos"]] == ["GGAL"]
    assert out["excluidos"][0]["error"] == "NO_DATA"
    assert [c["ticker"] for c in out["contribuyentes"]] == ["PAMP.BA"]


def test_missing_starting_rate_degrades_to_price_effect_with_a_note():
    out = build(Recorder(ccl={"2026-10-21": 1100.0}))(cliente_id="c1", periodo="semana")
    assert out["efecto_cambiario_pp"] == 0
    assert out["efecto_precio_pp"] == pytest.approx(10.0, abs=0.01)
    assert any("Tipo de cambio inicial" in n for n in out["notas"])


def test_no_exchange_rate_at_all_is_an_error_when_there_are_usd_holdings():
    rec = Recorder(ccl={})
    rec.get_macro = lambda variable: (_ for _ in ()).throw(ToolError("PROVIDER_UNAVAILABLE", "x"))
    with pytest.raises(ToolError) as exc:
        build(rec, today=date(2026, 10, 21))(cliente_id="c1", periodo="semana")
    assert exc.value.code == "PROVIDER_UNAVAILABLE"


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"cliente_id": "c1", "periodo": "año"}, "INVALID_PARAMS"),
        ({"cliente_id": "  ", "periodo": "dia"}, "INVALID_PARAMS"),
        ({"cliente_id": "nadie", "periodo": "dia"}, "CLIENT_NOT_FOUND"),
    ],
)
def test_invalid_input_is_rejected(kwargs, code):
    with pytest.raises(ToolError) as exc:
        build(Recorder())(**kwargs)
    assert exc.value.code == code


def test_the_tool_is_registered_session_bound_with_a_period_enum():
    from app.tools.registry import build_tools

    rec = Recorder()
    tools = build_tools(
        get_price=rec.get_price,
        get_macro=rec.get_macro,
        get_portfolio_performance=build(rec),
    )
    tool = next(t for t in tools if t.name == "get_portfolio_performance")
    schema = tool.args_schema.model_json_schema()
    assert "cliente_id" not in schema["properties"]
    assert schema["properties"]["periodo"]["enum"] == ["dia", "semana", "mes"]
