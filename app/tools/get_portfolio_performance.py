"""Portfolio performance attribution: how much of a period's move came from price vs. exchange rate.

Pure and deterministic: the LLM only narrates what this module computes. Values are expressed in
ARS. ARS assets only have a price effect; USD assets split into a price effect at the starting
rate (`q * (p1 - p0) * fx0`) and an exchange effect on the closing price (`q * p1 * (fx1 - fx0)`),
which add up exactly to `q * p1 * fx1 - q * p0 * fx0`.
"""

import logging
from collections.abc import Callable
from datetime import date, timedelta

from app.core.audit import annotate, audited_tool, log_event
from app.tools.errors import ToolError
from app.tools.get_portfolio import _market_of, load_mock_portfolios

DECIMALS = 2
# Same lookbacks as `get_price` (`variacion_dia` is the previous close, about one day back).
LOOKBACK_DAYS = {"dia": 1, "semana": 7, "mes": 30}
ASSUMPTION = "Se asume la tenencia actual durante todo el período."


def _start_price(holding: dict) -> float:
    variation = holding["variacion_pct"]
    if variation <= -100:
        raise ValueError(f"{holding['ticker']}: variation of {variation}% has no starting price.")
    return holding["precio_actual"] / (1 + variation / 100)


def _effects_ars(holding: dict, fx0: float, fx1: float) -> tuple[float, float, float]:
    """Return (starting value, price effect, exchange effect) of one holding, in ARS."""
    qty, p1, p0 = holding["cantidad"], holding["precio_actual"], _start_price(holding)
    if holding["moneda"] == "USD":
        return qty * p0 * fx0, qty * (p1 - p0) * fx0, qty * p1 * (fx1 - fx0)
    return qty * p0, qty * (p1 - p0), 0.0


def _pp(value: float, start_total: float) -> float:
    return round(value / start_total * 100, DECIMALS) if start_total else 0.0


def compute_performance(holdings: list[dict], *, fx0: float, fx1: float) -> dict:
    """Attribute the portfolio's move over a period to each holding and to price vs. exchange rate.

    `holdings` carry `ticker`, `cantidad`, `moneda`, `precio_actual` and `variacion_pct` (the
    price change over the period, in percent). `fx0`/`fx1` are the CCL rate at the start and end
    of the period. The current quantities are assumed to be held during the whole period.
    """
    rows = [(h["ticker"], *_effects_ars(h, fx0, fx1)) for h in holdings]
    start_total = sum(start for _, start, _, _ in rows)
    price_total = sum(price for _, _, price, _ in rows)
    exchange_total = sum(exchange for _, _, _, exchange in rows)
    contributors = [
        {
            "ticker": ticker,
            "contribucion_pp": _pp(price + exchange, start_total),
            "efecto_precio_pp": _pp(price, start_total),
            "efecto_cambiario_pp": _pp(exchange, start_total),
        }
        for ticker, _, price, exchange in rows
    ]
    contributors.sort(key=lambda c: abs(c["contribucion_pp"]), reverse=True)
    return {
        "valor_inicial_ars": round(start_total, DECIMALS),
        "valor_final_ars": round(start_total + price_total + exchange_total, DECIMALS),
        "retorno_total_pct": _pp(price_total + exchange_total, start_total),
        "efecto_precio_pp": _pp(price_total, start_total),
        "efecto_cambiario_pp": _pp(exchange_total, start_total),
        "contribuyentes": contributors,
    }


def _rate(fetch: Callable[[], dict]) -> float | None:
    try:
        return float(fetch()["valor"])
    except Exception as exc:
        log_event(
            "portfolio",
            "performance_rate_unavailable",
            level=logging.WARNING,
            error_type=type(exc).__name__,
        )
        return None


def _priced(get_price: Callable[..., dict], holding: dict, field: str) -> dict:
    market = _market_of(holding)
    price = get_price(ticker=holding["ticker"], mercado=market)
    if price.get(field) is None:
        raise ToolError("NO_VARIATION", f"No {field} available for {holding['ticker']}.")
    return {
        "ticker": holding["ticker"],
        "cantidad": holding["cantidad"],
        "moneda": "USD" if market == "nyse" else "ARS",
        "precio_actual": float(price["precio_cierre"]),
        "variacion_pct": float(price[field]),
        "fecha": price["fecha"],
    }


def make_get_portfolio_performance(
    *,
    get_price: Callable[..., dict],
    get_macro: Callable[..., dict],
    get_ccl_at: Callable[..., dict],
    portfolios: dict[str, list[dict]] | None = None,
    today: Callable[[], date] = date.today,
):
    """Build the audited `get_portfolio_performance(cliente_id, periodo)`.

    Prices and variations come from the injected `get_price`; the CCL at the start and end of the
    window from `get_ccl_at` (the live rate from `get_macro` when the closing date is today).
    """
    data = portfolios if portfolios is not None else load_mock_portfolios()

    @audited_tool
    def get_portfolio_performance(cliente_id: str, periodo: str) -> dict:
        cid = (cliente_id or "").strip()
        period = (periodo or "").strip().lower()
        if not cid:
            raise ToolError("INVALID_PARAMS", "cliente_id must not be empty.")
        if period not in LOOKBACK_DAYS:
            raise ToolError("INVALID_PARAMS", f"periodo must be one of {list(LOOKBACK_DAYS)}.")
        holdings = data.get(cid)
        annotate(source="mock_portfolios")
        if holdings is None:
            raise ToolError("CLIENT_NOT_FOUND", f"No portfolio found for client {cid}.")

        priced, excluded = [], []
        for holding in holdings:
            try:
                priced.append(_priced(get_price, holding, f"variacion_{period}"))
            except ToolError as exc:
                excluded.append({"ticker": holding["ticker"], "error": exc.code})
            except Exception as exc:
                excluded.append({"ticker": holding["ticker"], "error": type(exc).__name__})
        if not priced:
            raise ToolError("NO_DATA", "No holding has price data for the requested period.")

        end = max(date.fromisoformat(h["fecha"]) for h in priced)
        notes = []
        fx0 = fx1 = None
        if any(h["moneda"] == "USD" for h in priced):
            start = end - timedelta(days=LOOKBACK_DAYS[period])
            fx0 = _rate(lambda: get_ccl_at(fecha=start.isoformat()))
            fx1 = _rate(
                (lambda: get_ccl_at(fecha=end.isoformat()))
                if end < today()
                else (lambda: get_macro(variable="dolar_ccl"))
            )
            if fx0 is None and fx1 is None:
                raise ToolError("PROVIDER_UNAVAILABLE", "No CCL rate could be obtained.")
            if fx0 is None or fx1 is None:
                missing = "inicial" if fx0 is None else "final"
                fx0 = fx1 = fx0 if fx1 is None else fx1
                notes.append(
                    f"Tipo de cambio {missing} no disponible: el efecto cambiario no se calculó."
                )
        result = compute_performance(priced, fx0=fx0 or 1.0, fx1=fx1 or 1.0)
        return {
            "periodo": period,
            "fecha_fin": end.isoformat(),
            "tipo_cambio_inicial": fx0,
            "tipo_cambio_final": fx1,
            **result,
            "excluidos": excluded,
            "notas": notes,
            "supuesto": ASSUMPTION,
        }

    return get_portfolio_performance
