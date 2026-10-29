from datetime import date, timedelta

import fakeredis
import httpx
import pytest
import respx

from app.core.config import Settings
from app.tools.errors import ToolError
from app.tools.get_price import (
    IolPriceProvider,
    PriceSeries,
    make_get_price,
)

KEYS = {"ticker", "fecha", "precio_cierre", "variacion_dia", "variacion_semana", "fuente"}
OPTIONAL_KEYS = {"variacion_mes"}  # additive field (Week 3); the original keys are unchanged


def series(closes, end=date(2026, 10, 14), fuente="fake"):
    n = len(closes)
    pts = [(end - timedelta(days=n - 1 - i), c) for i, c in enumerate(closes)]
    return PriceSeries(fuente=fuente, points=pts)


def redis_():
    return fakeredis.FakeRedis(decode_responses=True)


def test_output_contract_and_variations():
    closes = [100.0] * 7 + [110.0, 121.0]  # 9 consecutive days
    tool = make_get_price(redis_client=redis_(), providers={"nyse": lambda t: series(closes)})
    out = tool("ggal", "NYSE")
    assert set(out) == KEYS | OPTIONAL_KEYS
    assert out["ticker"] == "GGAL"
    assert out["fecha"] == "2026-10-14"
    assert out["precio_cierre"] == 121.0
    assert out["variacion_dia"] == 10.0
    assert out["variacion_semana"] == 21.0  # vs 2026-10-07 (close 100)
    assert out["fuente"] == "fake"


def test_week_variation_none_when_history_too_short():
    tool = make_get_price(redis_client=redis_(), providers={"nyse": lambda t: series([10, 11])})
    out = tool("GGAL", "nyse")
    assert out["variacion_dia"] == 10.0
    assert out["variacion_semana"] is None


def test_cache_hit_avoids_provider_and_keys_are_normalized():
    calls = []

    def provider(t):
        calls.append(t)
        return series([1, 2, 3])

    r = redis_()
    tool = make_get_price(redis_client=r, providers={"nyse": provider})
    first = tool("ggal", "nyse")
    assert tool(" GGAL ", "NYSE") == first
    assert calls == ["GGAL"]
    assert any("GGAL" in k for k in r.keys("*"))


def test_errors_not_cached_and_typed():
    def provider(t):
        raise ToolError("NO_DATA", "no data")

    r = redis_()
    tool = make_get_price(redis_client=r, providers={"nyse": provider})
    with pytest.raises(ToolError) as exc:
        tool("GGAL", "nyse")
    assert exc.value.code == "NO_DATA"
    assert r.keys("*") == []


def test_unexpected_provider_exception_wrapped():
    def provider(t):
        raise RuntimeError("kaboom")

    tool = make_get_price(redis_client=redis_(), providers={"nyse": provider})
    with pytest.raises(ToolError) as exc:
        tool("GGAL", "nyse")
    assert exc.value.code == "PROVIDER_ERROR"


@pytest.mark.parametrize("ticker,mercado", [("", "nyse"), ("GGAL", "mervalx")])
def test_invalid_params(ticker, mercado):
    tool = make_get_price(redis_client=redis_(), providers={"nyse": lambda t: series([1, 2])})
    with pytest.raises(ToolError) as exc:
        tool(ticker, mercado)
    assert exc.value.code == "INVALID_PARAMS"


def test_ttl_from_settings():
    r = redis_()
    s = Settings(_env_file=None, price_cache_ttl=33)
    tool = make_get_price(
        redis_client=r, settings=s, providers={"nyse": lambda t: series([1, 2, 3])}
    )
    tool("GGAL", "nyse")
    (key,) = [k for k in r.keys("*") if not k.endswith(":stale")]
    assert 0 < r.ttl(key) <= 33


def test_default_settings_ttls():
    assert Settings(_env_file=None).price_cache_ttl == 900
    assert Settings(_env_file=None).macro_cache_ttl == 300


def test_iol_timeout_is_short_and_configurable():
    assert Settings(_env_file=None).iol_timeout_seconds == 3.5
    provider = IolPriceProvider(Settings(_env_file=None, iol_timeout_seconds=2.0))
    assert provider._client.timeout.read == 2.0


# ---- IOL provider ----


def iol(user="u", password="p"):
    s = Settings(_env_file=None, iol_user=user, iol_password=password)
    return IolPriceProvider(s, client=httpx.Client())


def test_iol_missing_credentials_is_explicit_error():
    with pytest.raises(ToolError) as exc:
        iol(user="", password="")("GGAL")
    assert exc.value.code == "CREDENTIALS_MISSING"


@respx.mock
def test_iol_token_flow_and_history():
    token = respx.post("https://api.invertironline.com/token").mock(
        return_value=httpx.Response(200, json={"access_token": "tok", "expires_in": 900})
    )
    hist = respx.get(url__regex=r".*/api/v2/bCBA/Titulos/YPFD/cotizacion/seriehistorica/.*").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"ultimoPrecio": 100.0, "fechaHora": "2026-10-13T17:00:00"},
                {"ultimoPrecio": 105.0, "fechaHora": "2026-10-14T17:00:00"},
            ],
        )
    )
    provider = iol()
    result = provider("YPFD.BA")
    assert result.fuente == "IOL"
    assert result.points[-1] == (date(2026, 10, 14), 105.0)
    assert hist.calls.last.request.headers["authorization"] == "Bearer tok"
    provider("YPFD")
    assert token.call_count == 1  # token reused


@respx.mock
def test_iol_auth_failure():
    respx.post("https://api.invertironline.com/token").mock(return_value=httpx.Response(400))
    with pytest.raises(ToolError) as exc:
        iol()("GGAL")
    assert exc.value.code == "AUTH_FAILED"


@respx.mock
def test_iol_empty_history_is_no_data():
    respx.post("https://api.invertironline.com/token").mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 900})
    )
    respx.get(url__regex=r".*seriehistorica.*").mock(return_value=httpx.Response(200, json=[]))
    with pytest.raises(ToolError) as exc:
        iol()("GGAL")
    assert exc.value.code == "NO_DATA"


def test_month_variation_uses_close_on_or_before_30_days_back():
    closes = [100.0] * 40 + [150.0]  # 41 consecutive days ending 2026-10-14
    tool = make_get_price(redis_client=redis_(), providers={"nyse": lambda t: series(closes)})
    out = tool("GGAL", "nyse")
    assert out["variacion_mes"] == 50.0  # vs 2026-09-14 (close 100)


def test_month_variation_none_when_history_too_short():
    tool = make_get_price(
        redis_client=redis_(), providers={"nyse": lambda t: series([100.0] * 10 + [110.0])}
    )
    out = tool("GGAL", "nyse")
    assert out["variacion_semana"] == 10.0
    assert out["variacion_mes"] is None


@respx.mock
def test_iol_history_window_covers_a_month_back():
    respx.post("https://api.invertironline.com/token").mock(
        return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 900})
    )
    hist = respx.get(url__regex=r".*seriehistorica.*").mock(
        return_value=httpx.Response(
            200, json=[{"ultimoPrecio": 1.0, "fechaHora": "2026-10-14T17:00:00"}]
        )
    )
    iol()("GGAL")
    start = str(hist.calls.last.request.url).split("/seriehistorica/")[1].split("/")[0]
    assert (date.today() - date.fromisoformat(start)).days >= 35
