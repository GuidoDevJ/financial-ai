from datetime import date

import fakeredis
import httpx
import pytest
import respx

from app.tools.errors import ToolError
from app.tools.get_ccl_history import make_get_ccl_at

URL = "https://api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui"
HISTORY = [
    {"casa": "contadoconliqui", "compra": 1600.0, "venta": 1610.0, "fecha": "2026-10-14"},
    {"casa": "contadoconliqui", "compra": 1615.0, "venta": 1620.0, "fecha": "2026-10-16"},
    {"casa": "contadoconliqui", "compra": 1630.0, "venta": 1640.0, "fecha": "2026-10-20"},
]
TODAY = date(2026, 10, 22)


def tool(redis_client=None):
    return make_get_ccl_at(
        redis_client=redis_client or fakeredis.FakeRedis(decode_responses=True),
        client=httpx.Client(),
        today=lambda: TODAY,
    )


@respx.mock
def test_returns_the_sell_rate_of_the_requested_date():
    respx.get(URL).mock(return_value=httpx.Response(200, json=HISTORY))
    assert tool()("2026-10-16") == {
        "valor": 1620.0,
        "fecha": "2026-10-16",
        "fuente": "argentinadatos.com",
    }


@respx.mock
def test_a_date_without_a_row_carries_the_last_known_rate_forward():
    respx.get(URL).mock(return_value=httpx.Response(200, json=HISTORY))
    out = tool()("2026-10-18")
    assert out["valor"] == 1620.0 and out["fecha"] == "2026-10-16"


@respx.mock
def test_a_date_before_the_history_starts_is_an_error():
    respx.get(URL).mock(return_value=httpx.Response(200, json=HISTORY))
    with pytest.raises(ToolError) as exc:
        tool()("2026-01-01")
    assert exc.value.code == "NO_DATA"


@pytest.mark.parametrize("bad", ["", "ayer", "2026-13-40", "2026-10-22", "2026-10-23"])
def test_invalid_or_non_past_dates_are_rejected_without_calling_the_provider(bad):
    with respx.mock:
        route = respx.get(URL).mock(return_value=httpx.Response(200, json=HISTORY))
        with pytest.raises(ToolError) as exc:
            tool()(bad)
    assert exc.value.code == "INVALID_PARAMS" and not route.called


@respx.mock
def test_past_rates_are_cached_per_date():
    route = respx.get(URL).mock(return_value=httpx.Response(200, json=HISTORY))
    get_ccl_at = tool()
    get_ccl_at("2026-10-20")
    get_ccl_at("2026-10-20")
    assert route.call_count == 1


@respx.mock
def test_provider_failure_surfaces_as_provider_unavailable_and_is_not_cached():
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    respx.get(URL).mock(return_value=httpx.Response(503))
    with pytest.raises(ToolError) as exc:
        tool(redis_client)("2026-10-20")
    assert exc.value.code == "PROVIDER_UNAVAILABLE"
    assert redis_client.keys("*") == []
