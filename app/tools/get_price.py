"""get_price tool: closing price and daily/weekly variation of an Argentine equity or ADR."""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from app.core.audit import annotate, audited_tool, log_event, log_upstream
from app.core.config import Settings, get_settings
from app.tools.cache import cached_call
from app.tools.errors import ToolError

MERCADOS = ("nyse", "byma")
IOL_MARKET = {"byma": "bCBA"}  # IOL market code for BYMA (Buenos Aires)
# A last close older than this many calendar days is flagged stale (covers weekends/holidays).
STALE_AFTER_DAYS = 4
# Plain symbols (GGAL, YPFD.BA, BRK-B, ^MERV): rejects sentences and injection attempts early.
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")


@dataclass(frozen=True)
class PriceSeries:
    """Daily closing prices (ascending by date) plus the name of the source."""

    fuente: str
    points: list[tuple[date, float]]


PriceProvider = Callable[[str], PriceSeries]


def _pct(new: float, old: float) -> float | None:
    if not old:
        return None
    return round((new / old - 1) * 100, 2)


def build_output(ticker: str, series: PriceSeries) -> dict:
    points = sorted(series.points)
    if not points:
        raise ToolError("NO_DATA", f"No price data available for {ticker}.")
    last_date, last_close = points[-1]
    variacion_dia = _pct(last_close, points[-2][1]) if len(points) > 1 else None

    # Weekly variation: latest close on or before (last_date - 7 days); None if history too short.
    target = last_date - timedelta(days=7)
    base = [p for p in points if p[0] <= target]
    variacion_semana = _pct(last_close, base[-1][1]) if base else None

    # Monthly variation: same rule with a 30-day lookback (needs ~6 weeks of history).
    month_base = [p for p in points if p[0] <= last_date - timedelta(days=30)]
    variacion_mes = _pct(last_close, month_base[-1][1]) if month_base else None

    return {
        "ticker": ticker,
        "fecha": last_date.isoformat(),
        "precio_cierre": round(last_close, 4),
        "variacion_dia": variacion_dia,
        "variacion_semana": variacion_semana,
        "variacion_mes": variacion_mes,
        "fuente": series.fuente,
    }


# ---------------------------------------------------------------- providers


def yfinance_provider(ticker: str) -> PriceSeries:
    """NYSE/ADR prices via yfinance (unofficial but free). Injectable for tests."""
    import yfinance as yf

    started = time.perf_counter()
    frame = yf.Ticker(ticker).history(period="2mo")
    rows = 0 if frame is None else len(frame)
    # yfinance hides the HTTP exchange, so only latency and the row count are known.
    log_upstream("yfinance", "history", "https://query1.finance.yahoo.com", started, row_count=rows)
    if frame is None or frame.empty:
        raise ToolError("NO_DATA", f"yfinance returned no data for {ticker}.")
    points = [(idx.date(), float(close)) for idx, close in frame["Close"].dropna().items()]
    return PriceSeries(fuente="yfinance", points=points)


def yfinance_ba_provider(ticker: str) -> PriceSeries:
    """BYMA prices via yfinance using the Buenos Aires `.BA` suffix (fallback for IOL)."""
    symbol = ticker.upper()
    if not symbol.endswith(".BA"):
        symbol += ".BA"
    series = yfinance_provider(symbol)
    return PriceSeries(fuente="yfinance (.BA)", points=series.points)


def _provider_name(provider) -> str:
    return getattr(provider, "__name__", type(provider).__name__)


class IolPriceProvider:
    """BYMA prices via the IOL (InvertirOnline) REST API with OAuth password flow.

    Verified from IOL docs: bearer token valid 15 minutes, sent on every request.
    TODO(unverified): exact token request body (form: username, password, grant_type=password),
    the historical series path and its response fields (ultimoPrecio, fechaHora). Validate
    against the IOL sandbox/production with real credentials before relying on it.
    """

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self._settings = settings or get_settings()
        # Short timeout: a slow IOL must fail fast so the yfinance fallback can answer in time.
        self._client = client or httpx.Client(timeout=self._settings.iol_timeout_seconds)
        self._token: str | None = None
        self._token_expires_at = 0.0

    def _get_token(self) -> str:
        s = self._settings
        if not (s.iol_user and s.iol_password):
            log_event(
                "iol",
                "iol_credentials",
                level=logging.WARNING,
                source="iol",
                reason="IOL_USER / IOL_PASSWORD not configured",
            )
            raise ToolError(
                "CREDENTIALS_MISSING",
                "IOL credentials (IOL_USER / IOL_PASSWORD) are not configured; "
                "BYMA prices are unavailable.",
            )
        if self._token and time.monotonic() < self._token_expires_at:
            log_event("iol", "iol_token", level=logging.DEBUG, source="iol", reused=True)
            return self._token
        token_url = f"{s.iol_base_url}/token"
        started = time.perf_counter()
        try:
            resp = self._client.post(
                token_url,
                data={"username": s.iol_user, "password": s.iol_password, "grant_type": "password"},
            )
        except httpx.HTTPError as exc:
            log_upstream("iol", "token", token_url, started, error_type=type(exc).__name__)
            raise ToolError(
                "PROVIDER_UNAVAILABLE", f"IOL token request failed: {type(exc).__name__}"
            ) from exc
        log_upstream("iol", "token", token_url, started, resp.status_code)
        if resp.status_code != 200:
            raise ToolError("AUTH_FAILED", f"IOL authentication failed ({resp.status_code}).")
        body = resp.json()
        self._token = body["access_token"]
        # Refresh a minute early.
        self._token_expires_at = time.monotonic() + max(int(body.get("expires_in", 900)) - 60, 0)
        return self._token

    def __call__(self, ticker: str) -> PriceSeries:
        token = self._get_token()
        symbol = ticker.upper().removesuffix(".BA")
        today = date.today()
        start = today - timedelta(days=45)
        # TODO(unverified): historical series path.
        url = (
            f"{self._settings.iol_base_url}/api/v2/{IOL_MARKET['byma']}/Titulos/{symbol}"
            f"/cotizacion/seriehistorica/{start.isoformat()}/{today.isoformat()}/sinAjustar"
        )
        started = time.perf_counter()
        try:
            resp = self._client.get(url, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            log_upstream("iol", "history", url, started, error_type=type(exc).__name__)
            raise ToolError(
                "PROVIDER_UNAVAILABLE", f"IOL request failed: {type(exc).__name__}"
            ) from exc
        log_upstream("iol", "history", url, started, resp.status_code, symbol=symbol)
        if resp.status_code == 401:
            self._token = None
            raise ToolError("AUTH_FAILED", "IOL rejected the access token.")
        if resp.status_code != 200:
            raise ToolError("PROVIDER_UNAVAILABLE", f"IOL returned HTTP {resp.status_code}.")
        by_day: dict[date, float] = {}
        for row in resp.json():
            price = row.get("ultimoPrecio")
            if price:
                day = datetime.fromisoformat(row["fechaHora"]).date()
                by_day[day] = float(price)
        if not by_day:
            raise ToolError("NO_DATA", f"IOL returned no price history for {symbol}.")
        return PriceSeries(fuente="IOL", points=sorted(by_day.items()))


# --------------------------------------------------------------------- tool


def make_get_price(
    *,
    redis_client,
    settings: Settings | None = None,
    providers: dict[str, PriceProvider] | None = None,
    fallbacks: dict[str, PriceProvider] | None = None,
    today: Callable[[], date] = date.today,
):
    """Build the audited, cached `get_price(ticker, mercado)` function.

    `fallbacks[mercado]` is tried when the primary provider fails (by default BYMA falls back
    from IOL to yfinance with the `.BA` suffix). When every source fails, a stale-if-error copy
    of the last good answer is returned flagged `stale: true`.
    """
    settings = settings or get_settings()
    if providers is None:
        providers = {"nyse": yfinance_provider, "byma": IolPriceProvider(settings)}
        if fallbacks is None:
            fallbacks = {"byma": yfinance_ba_provider}
    fallbacks = fallbacks or {}

    @audited_tool
    def get_price(ticker: str, mercado: str) -> dict:
        ticker_n = (ticker or "").strip().upper()
        mercado_n = (mercado or "").strip().lower()
        if not ticker_n:
            raise ToolError("INVALID_PARAMS", "ticker must not be empty.")
        if not TICKER_RE.fullmatch(ticker_n):
            raise ToolError(
                "INVALID_PARAMS", "ticker must be a plain symbol such as GGAL or YPFD.BA."
            )
        if mercado_n not in MERCADOS:
            raise ToolError("INVALID_PARAMS", f"mercado must be one of {list(MERCADOS)}.")
        provider = providers.get(mercado_n)
        if provider is None:
            raise ToolError("PROVIDER_UNAVAILABLE", f"No provider configured for {mercado_n}.")

        def fetch(source: PriceProvider) -> PriceSeries:
            try:
                return source(ticker_n)
            except ToolError as exc:
                log_event(
                    "price",
                    "provider_error",
                    level=logging.WARNING,
                    source=mercado_n,
                    ticker=ticker_n,
                    error_code=exc.code,
                )
                raise
            except Exception as exc:
                log_event(
                    "price",
                    "provider_error",
                    level=logging.ERROR,
                    exc_info=True,
                    source=mercado_n,
                    ticker=ticker_n,
                    error_type=type(exc).__name__,
                )
                raise ToolError("PROVIDER_ERROR", f"{type(exc).__name__}: {exc}") from exc

        def compute() -> dict:
            try:
                series = fetch(provider)
            except ToolError as primary:
                backup = fallbacks.get(mercado_n)
                if backup is None:
                    raise
                log_event(
                    "price",
                    "fallback",
                    level=logging.WARNING,
                    mercado=mercado_n,
                    ticker=ticker_n,
                    **{"from": _provider_name(provider), "to": _provider_name(backup)},
                    reason=f"{primary.code}: {primary.message}",
                )
                try:
                    series = fetch(backup)
                except ToolError as second:
                    raise ToolError(
                        primary.code,
                        f"{primary.message} (fallback {_provider_name(backup)} also failed: "
                        f"{second.code})",
                    ) from second
                annotate(fallback=True)
            return build_output(ticker_n, series)

        key = f"tool:get_price:{mercado_n}:{ticker_n}"
        out = cached_call(
            redis_client,
            key,
            settings.price_cache_ttl,
            compute,
            stale_ttl=settings.price_stale_ttl_seconds or None,
        )
        as_of = out["fecha"]
        stale = bool(out.get("stale")) or (today() - date.fromisoformat(as_of)).days > (
            STALE_AFTER_DAYS
        )
        annotate(source=out["fuente"], as_of=as_of, stale=stale)
        log_event(
            "price",
            "price",
            level=logging.WARNING if stale else logging.INFO,
            ticker=ticker_n,
            mercado=mercado_n,
            source=out["fuente"],
            as_of=as_of,
            stale=stale,
        )
        return out

    return get_price
