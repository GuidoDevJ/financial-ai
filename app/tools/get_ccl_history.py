"""get_ccl_at tool: the CCL (contado con liquidación) sell rate on a past date.

Needed to split a portfolio's move into price and exchange effects. The only source is the
argentinadatos history list (ascending `{fecha, venta, ...}` rows, one per calendar day). Past
rates never change, so they are cached for a day per date; the compact row is cached, not the
5,000-row payload.
"""

from bisect import bisect_right
from collections.abc import Callable
from datetime import date

import httpx

from app.core.audit import annotate, audited_tool
from app.tools.cache import cached_call
from app.tools.errors import ToolError
from app.tools.get_macro import _DOLAR_CASA, ARGENTINADATOS, _get_json

SOURCE = "argentinadatos.com"
CACHE_TTL_SECONDS = 24 * 60 * 60


def make_get_ccl_at(
    *,
    redis_client,
    client: httpx.Client | None = None,
    today: Callable[[], date] = date.today,
):
    """Build the audited, cached `get_ccl_at(fecha)` function (fecha = ISO date, strictly past)."""
    client = client or httpx.Client(timeout=8, follow_redirects=True)
    url = f"{ARGENTINADATOS}/v1/cotizaciones/dolares/{_DOLAR_CASA['dolar_ccl']}"

    @audited_tool
    def get_ccl_at(fecha: str) -> dict:
        try:
            day = date.fromisoformat((fecha or "").strip())
        except ValueError:
            raise ToolError("INVALID_PARAMS", "fecha must be an ISO date (YYYY-MM-DD).") from None
        if day >= today():
            raise ToolError("INVALID_PARAMS", "fecha must be before today; use get_macro instead.")

        def compute() -> dict:
            rows = _get_json(client, url, "argentinadatos")
            dates = [str(row["fecha"])[:10] for row in rows]
            index = bisect_right(dates, day.isoformat()) - 1
            if index < 0:
                raise ToolError("NO_DATA", f"No CCL rate on or before {day.isoformat()}.")
            annotate(source=SOURCE)
            return {
                "valor": float(rows[index]["venta"]),
                "fecha": dates[index],
                "fuente": SOURCE,
            }

        return cached_call(
            redis_client, f"tool:get_ccl_at:{day.isoformat()}", CACHE_TTL_SECONDS, compute
        )

    return get_ccl_at
