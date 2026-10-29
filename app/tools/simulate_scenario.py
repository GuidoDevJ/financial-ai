"""simulate_scenario tool: "what if the CCL rises 10%?" applied to the client's current portfolio.

Reuses the attribution math of `get_portfolio_performance`: the current prices are the starting
point and the scenario prices are `price * (1 + shock)`. USD assets are converted to ARS at the
current CCL and at the shocked CCL. It is a hypothetical calculation, never a forecast.
"""

from collections.abc import Callable

from app.core.audit import annotate, audited_tool
from app.tools.errors import ToolError
from app.tools.get_portfolio import _market_of, load_mock_portfolios
from app.tools.get_portfolio_performance import compute_performance

SHOCK_MIN, SHOCK_MAX = -50.0, 100.0
NOTICE = "Es una hipótesis de cálculo, no es una predicción ni una recomendación."
ASSUMPTION = (
    "El shock de precio se aplica igual a todos los activos, en su propia moneda; los precios en "
    "pesos no reaccionan al CCL. Se asume la tenencia actual."
)


def _check_shock(name: str, value: float) -> float:
    shock = float(value)
    if not SHOCK_MIN <= shock <= SHOCK_MAX:
        raise ToolError("INVALID_PARAMS", f"{name} must be between {SHOCK_MIN} and {SHOCK_MAX}.")
    return shock


def make_simulate_scenario(
    *,
    get_price: Callable[..., dict],
    get_macro: Callable[..., dict],
    portfolios: dict[str, list[dict]] | None = None,
):
    """Build the audited `simulate_scenario(cliente_id, ccl_pct, precio_pct)`."""
    data = portfolios if portfolios is not None else load_mock_portfolios()

    @audited_tool
    def simulate_scenario(cliente_id: str, ccl_pct: float = 0, precio_pct: float = 0) -> dict:
        cid = (cliente_id or "").strip()
        if not cid:
            raise ToolError("INVALID_PARAMS", "cliente_id must not be empty.")
        ccl_shock = _check_shock("ccl_pct", ccl_pct)
        price_shock = _check_shock("precio_pct", precio_pct)
        holdings = data.get(cid)
        annotate(source="mock_portfolios")
        if holdings is None:
            raise ToolError("CLIENT_NOT_FOUND", f"No portfolio found for client {cid}.")

        scenario, excluded, dates = [], [], []
        for holding in holdings:
            market = _market_of(holding)
            try:
                price = get_price(ticker=holding["ticker"], mercado=market)
                base = float(price["precio_cierre"])
                dates.append(price["fecha"])
            except ToolError as exc:
                excluded.append({"ticker": holding["ticker"], "error": exc.code})
                continue
            except Exception as exc:
                excluded.append({"ticker": holding["ticker"], "error": type(exc).__name__})
                continue
            scenario.append(
                {
                    "ticker": holding["ticker"],
                    "cantidad": holding["cantidad"],
                    "moneda": "USD" if market == "nyse" else "ARS",
                    "precio_actual": base * (1 + price_shock / 100),
                    "variacion_pct": price_shock,
                }
            )
        if not scenario:
            raise ToolError("NO_DATA", "No holding has price data to simulate.")

        notes = []
        fx0 = fx1 = None
        if any(h["moneda"] == "USD" for h in scenario):
            try:
                fx0 = float(get_macro(variable="dolar_ccl")["valor"])
            except Exception as exc:
                raise ToolError(
                    "PROVIDER_UNAVAILABLE", f"The current CCL rate is unavailable: {exc}"
                ) from exc
            fx1 = round(fx0 * (1 + ccl_shock / 100), 2)
        elif ccl_shock:
            notes.append("El shock del CCL no tiene efecto: la cartera no tiene activos en USD.")
        result = compute_performance(scenario, fx0=fx0 or 1.0, fx1=fx1 or 1.0)
        return {
            "ccl_pct": ccl_pct,
            "precio_pct": precio_pct,
            "fecha_precios": max(dates),
            "tipo_cambio_actual": fx0,
            "tipo_cambio_escenario": fx1,
            **result,
            "excluidos": excluded,
            "notas": notes,
            "supuesto": ASSUMPTION,
            "aviso": NOTICE,
        }

    return simulate_scenario
