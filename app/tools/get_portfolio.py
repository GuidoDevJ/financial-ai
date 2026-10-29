"""get_portfolio tool: mock client holdings valued with live prices (no core-banking yet)."""

import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from app.core.audit import annotate, audited_tool, log_event
from app.core.concurrency import map_in_threads
from app.tools.errors import ToolError

MOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolios_mock.json"
CURRENCY = {"nyse": "USD", "byma": "ARS"}


def load_mock_portfolios(path: Path = MOCK_PATH) -> dict[str, list[dict]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _market_of(holding: dict) -> str:
    return holding.get("mercado") or (
        "byma" if holding["ticker"].upper().endswith(".BA") else "nyse"
    )


def _add_weights(rows: list[dict], get_macro: Callable[..., dict], total_assets: int) -> None:
    """Fill `peso_porcentual` over the priced rows; mixed currencies are converted via CCL."""
    priced = [r for r in rows if "valorizacion" in r]
    if not priced:
        return
    notes = []
    if len(priced) < total_assets:
        notes.append("Weight computed only over assets with a price.")
    ccl = None
    if len({r["moneda"] for r in priced}) > 1:
        try:
            ccl = float(get_macro(variable="dolar_ccl")["valor"])
        except Exception as exc:
            log_event(
                "portfolio",
                "portfolio_weights",
                level=logging.WARNING,
                reason="ccl_unavailable",
                error_type=type(exc).__name__,
                priced=len(priced),
            )
            for r in priced:
                r["peso_porcentual"] = None
                r["nota_peso"] = (
                    f"Weight unavailable: portfolio mixes USD and ARS and the CCL rate could "
                    f"not be obtained ({type(exc).__name__})."
                )
            return
        notes.append(f"USD values converted to ARS at CCL {ccl} for the weight.")

    def in_common_currency(row: dict) -> float:
        return row["valorizacion"] * ccl if ccl and row["moneda"] == "USD" else row["valorizacion"]

    total = sum(in_common_currency(r) for r in priced)
    for r in priced:
        r["peso_porcentual"] = round(in_common_currency(r) / total * 100, 2) if total else None
        if notes:
            r["nota_peso"] = " ".join(notes)


def make_get_portfolio(
    *,
    get_price: Callable[..., dict],
    get_macro: Callable[..., dict],
    portfolios: dict[str, list[dict]] | None = None,
):
    """Build the audited `get_portfolio(cliente_id)`; prices come from the injected get_price."""
    data = portfolios if portfolios is not None else load_mock_portfolios()

    @audited_tool
    def get_portfolio(cliente_id: str) -> list[dict]:
        cid = (cliente_id or "").strip()
        if not cid:
            raise ToolError("INVALID_PARAMS", "cliente_id must not be empty.")
        holdings = data.get(cid)
        annotate(source="mock_portfolios")
        if holdings is None:
            log_event("portfolio", "portfolio", level=logging.WARNING, cliente_id=cid, found=False)
            raise ToolError("CLIENT_NOT_FOUND", f"No portfolio found for client {cid}.")

        def value(holding: dict) -> dict:
            ticker, qty, market = holding["ticker"], holding["cantidad"], _market_of(holding)
            row = {"ticker": ticker, "cantidad": qty}
            try:
                price = float(get_price(ticker=ticker, mercado=market)["precio_cierre"])
            except ToolError as exc:
                return {**row, "error": exc.code, "message": exc.message}
            except Exception as exc:
                return {**row, "error": "PRICE_ERROR", "message": f"{type(exc).__name__}: {exc}"}
            return {
                **row,
                "precio_actual": price,
                "moneda": CURRENCY[market],
                "valorizacion": round(qty * price, 2),
            }

        # Prices are independent network calls: fetch them concurrently, rows keep holding order.
        rows = map_in_threads(value, holdings)
        _add_weights(rows, get_macro, len(rows))
        failures = Counter(r["error"] for r in rows if "error" in r)
        # Counts and error codes only: never tickers, quantities or valuations.
        log_event(
            "portfolio",
            "portfolio",
            level=logging.WARNING if failures else logging.INFO,
            cliente_id=cid,
            found=True,
            n_holdings=len(rows),
            priced=len(rows) - sum(failures.values()),
            failed=sum(failures.values()),
            failures=dict(failures),
        )
        return rows

    return get_portfolio
