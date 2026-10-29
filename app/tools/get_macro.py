"""get_macro tool: Argentine macro variables from community/public APIs with a fallback chain."""

import logging
import time
from collections.abc import Callable
from datetime import date, timedelta

import httpx

from app.core.audit import annotate, audited_tool, log_event, log_upstream
from app.core.config import Settings, get_settings
from app.tools.cache import cached_call
from app.tools.errors import ToolError

VARIABLES = ("dolar_ccl", "dolar_mep", "riesgo_pais", "tasa_bcra")

DOLARAPI = "https://dolarapi.com"
ARGENTINADATOS = "https://api.argentinadatos.com"
BCRA = "https://api.bcra.gob.ar"
# The BCRA policy-rate series (ids 160/161) stopped updating in 2025-07; BADLAR (id 7) is the
# current reference rate exposed by the official API, so `tasa_bcra` maps to it.
BCRA_BADLAR_ID = 7

# provider(variable) -> {"valor", "fecha", "fuente"}
MacroProvider = Callable[[str], dict]

_DOLAR_CASA = {"dolar_ccl": "contadoconliqui", "dolar_mep": "bolsa"}


def _get_json(client: httpx.Client, url: str, source: str = "http"):
    started = time.perf_counter()
    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        log_upstream(source, "get", url, started, error_type=type(exc).__name__)
        raise ToolError("PROVIDER_UNAVAILABLE", f"{url}: {exc}") from exc
    log_upstream(source, "get", url, started, resp.status_code)
    if resp.status_code != 200:
        raise ToolError("PROVIDER_UNAVAILABLE", f"{url} returned HTTP {resp.status_code}.")
    return resp.json()


def default_macro_providers(client: httpx.Client) -> dict[str, list[MacroProvider]]:
    """Primary + fallback provider chain per variable."""

    def dolarapi(variable: str) -> dict:
        # Verified: GET /v1/dolares/{contadoconliqui|bolsa} -> {venta, compra, fechaActualizacion}
        data = _get_json(client, f"{DOLARAPI}/v1/dolares/{_DOLAR_CASA[variable]}", "dolarapi")
        return {
            "valor": float(data["venta"]),
            "fecha": str(data["fechaActualizacion"])[:10],
            "fuente": "dolarapi.com",
        }

    def argentinadatos_dolar(variable: str) -> dict:
        # TODO(unverified): shape is a full history list [{casa, compra, venta, fecha}] (large
        # payload); only the last entry is used. Fallback only.
        data = _get_json(
            client,
            f"{ARGENTINADATOS}/v1/cotizaciones/dolares/{_DOLAR_CASA[variable]}",
            "argentinadatos",
        )
        last = data[-1]
        return {
            "valor": float(last["venta"]),
            "fecha": str(last["fecha"])[:10],
            "fuente": "argentinadatos.com",
        }

    def riesgo_ultimo(variable: str) -> dict:
        # Verified: GET /v1/finanzas/indices/riesgo-pais/ultimo -> {valor, fecha}
        data = _get_json(
            client, f"{ARGENTINADATOS}/v1/finanzas/indices/riesgo-pais/ultimo", "argentinadatos"
        )
        return {
            "valor": float(data["valor"]),
            "fecha": str(data["fecha"])[:10],
            "fuente": "argentinadatos.com",
        }

    def riesgo_historial(variable: str) -> dict:
        # TODO(unverified): list endpoint assumed to return [{fecha, valor}] ascending.
        data = _get_json(
            client, f"{ARGENTINADATOS}/v1/finanzas/indices/riesgo-pais", "argentinadatos"
        )
        last = data[-1]
        return {
            "valor": float(last["valor"]),
            "fecha": str(last["fecha"])[:10],
            "fuente": "argentinadatos.com",
        }

    def bcra_badlar(variable: str) -> dict:
        # Verified: GET /estadisticas/v4.0/monetarias/{id}?limit=1
        #   -> {results: [{idVariable, detalle: [{fecha, valor}]}]}
        url = f"{BCRA}/estadisticas/v4.0/monetarias/{BCRA_BADLAR_ID}?limit=1"
        data = _get_json(client, url, "bcra")
        row = data["results"][0]["detalle"][0]
        return {
            "valor": float(row["valor"]),
            "fecha": str(row["fecha"])[:10],
            "fuente": "api.bcra.gob.ar (BADLAR bancos privados, % n.a.)",
        }

    def bcra_badlar_v2(variable: str) -> dict:
        # TODO(unverified): legacy v2.0 endpoint of the same official API, assumed to return
        #   {results: [{idVariable, fecha, valor}, ...]} for a date range. Same series (BADLAR),
        #   so it is a second route to the same number; validate against the real API.
        today = date.today()
        url = (
            f"{BCRA}/estadisticas/v2.0/datosvariable/{BCRA_BADLAR_ID}/"
            f"{(today - timedelta(days=10)).isoformat()}/{today.isoformat()}"
        )
        rows = _get_json(client, url, "bcra")["results"]
        row = max(rows, key=lambda r: str(r["fecha"]))
        return {
            "valor": float(row["valor"]),
            "fecha": str(row["fecha"])[:10],
            "fuente": "api.bcra.gob.ar (BADLAR bancos privados, % n.a.)",
        }

    return {
        "dolar_ccl": [dolarapi, argentinadatos_dolar],
        "dolar_mep": [dolarapi, argentinadatos_dolar],
        "riesgo_pais": [riesgo_ultimo, riesgo_historial],
        # Last resort for every variable is the stale-if-error copy (30 days for tasa_bcra).
        "tasa_bcra": [bcra_badlar, bcra_badlar_v2],
    }


def make_get_macro(
    *,
    redis_client,
    settings: Settings | None = None,
    providers: dict[str, list[MacroProvider]] | None = None,
):
    """Build the audited, cached `get_macro(variable)` function."""
    settings = settings or get_settings()
    if providers is None:
        providers = default_macro_providers(httpx.Client(timeout=8, follow_redirects=True))

    @audited_tool
    def get_macro(variable: str) -> dict:
        variable_n = (variable or "").strip().lower()
        if variable_n not in VARIABLES:
            raise ToolError("INVALID_PARAMS", f"variable must be one of {list(VARIABLES)}.")

        def compute() -> dict:
            chain = providers.get(variable_n, [])
            names = [getattr(p, "__name__", type(p).__name__) for p in chain]
            errors = []
            for i, provider in enumerate(chain):
                try:
                    data = provider(variable_n)
                except Exception as exc:  # try the next provider in the chain
                    reason = f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"
                    errors.append(f"{type(exc).__name__}: {exc}")
                    has_next = i + 1 < len(chain)
                    log_event(
                        "macro",
                        "fallback" if has_next else "provider_failed",
                        level=logging.WARNING,
                        exc_info=not isinstance(exc, ToolError),
                        variable=variable_n,
                        **{"from": names[i], "to": names[i + 1] if has_next else None},
                        reason=reason,
                    )
                    continue
                annotate(source=data.get("fuente"), fallback=i > 0, attempts=i + 1)
                return {"variable": variable_n, **data}
            detail = "; ".join(errors) or "no provider configured"
            log_event(
                "macro",
                "providers_exhausted",
                level=logging.WARNING,
                variable=variable_n,
                attempts=len(chain),
            )
            raise ToolError(
                "PROVIDER_UNAVAILABLE", f"No source could provide {variable_n} ({detail})."
            )

        stale_ttl = (
            settings.tasa_bcra_stale_ttl_seconds
            if variable_n == "tasa_bcra"
            else settings.macro_stale_ttl_seconds
        )
        return cached_call(
            redis_client,
            f"tool:get_macro:{variable_n}",
            settings.macro_cache_ttl,
            compute,
            stale_ttl=stale_ttl or None,
        )

    return get_macro
