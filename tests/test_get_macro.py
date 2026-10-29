import fakeredis
import httpx
import pytest
import respx

from app.core.config import Settings
from app.tools.errors import ToolError
from app.tools.get_macro import default_macro_providers, make_get_macro

KEYS = {"variable", "valor", "fecha", "fuente"}


def redis_():
    return fakeredis.FakeRedis(decode_responses=True)


def ok(valor=1.0, fuente="p"):
    return lambda variable: {"valor": valor, "fecha": "2026-10-14", "fuente": fuente}


def fail(variable):
    raise ToolError("PROVIDER_UNAVAILABLE", "down")


def test_contract_and_cache():
    calls = []

    def p(variable):
        calls.append(variable)
        return {"valor": 600.0, "fecha": "2026-10-14", "fuente": "x"}

    tool = make_get_macro(redis_client=redis_(), providers={"riesgo_pais": [p]})
    out = tool("riesgo_pais")
    assert out == {"variable": "riesgo_pais", "valor": 600.0, "fecha": "2026-10-14", "fuente": "x"}
    assert set(out) == KEYS
    tool(" RIESGO_PAIS")
    assert calls == ["riesgo_pais"]


def test_fallback_chain():
    tool = make_get_macro(
        redis_client=redis_(), providers={"dolar_ccl": [fail, ok(1500.0, "backup")]}
    )
    out = tool("dolar_ccl")
    assert out["valor"] == 1500.0 and out["fuente"] == "backup"


def test_all_providers_fail_not_cached():
    r = redis_()
    tool = make_get_macro(redis_client=r, providers={"dolar_ccl": [fail, fail]})
    with pytest.raises(ToolError) as exc:
        tool("dolar_ccl")
    assert exc.value.code == "PROVIDER_UNAVAILABLE"
    assert r.keys("*") == []


def test_unexpected_exception_falls_through():
    def weird(variable):
        raise KeyError("x")

    tool = make_get_macro(redis_client=redis_(), providers={"dolar_mep": [weird, ok()]})
    assert tool("dolar_mep")["valor"] == 1.0


def test_invalid_variable():
    tool = make_get_macro(redis_client=redis_(), providers={})
    with pytest.raises(ToolError) as exc:
        tool("bitcoin")
    assert exc.value.code == "INVALID_PARAMS"


def test_ttl_from_settings():
    r = redis_()
    s = Settings(_env_file=None, macro_cache_ttl=44)
    make_get_macro(redis_client=r, settings=s, providers={"dolar_mep": [ok()]})("dolar_mep")
    (key,) = [k for k in r.keys("*") if not k.endswith(":stale")]
    assert 0 < r.ttl(key) <= 44


# ---- default HTTP providers ----


@respx.mock
def test_default_ccl_primary_dolarapi():
    respx.get("https://dolarapi.com/v1/dolares/contadoconliqui").mock(
        return_value=httpx.Response(
            200,
            json={
                "venta": 1616,
                "compra": 1615.5,
                "fechaActualizacion": "2026-10-15T19:57:00.000Z",
            },
        )
    )
    primary = default_macro_providers(httpx.Client())["dolar_ccl"][0]
    assert primary("dolar_ccl") == {
        "valor": 1616.0,
        "fecha": "2026-10-15",
        "fuente": "dolarapi.com",
    }


@respx.mock
def test_default_mep_uses_bolsa():
    route = respx.get("https://dolarapi.com/v1/dolares/bolsa").mock(
        return_value=httpx.Response(
            200, json={"venta": 1500, "fechaActualizacion": "2026-10-15T10:00:00Z"}
        )
    )
    default_macro_providers(httpx.Client())["dolar_mep"][0]("dolar_mep")
    assert route.called


@respx.mock
def test_default_riesgo_pais_and_fallback():
    respx.get("https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo").mock(
        return_value=httpx.Response(200, json={"valor": 611, "fecha": "2026-10-14"})
    )
    respx.get("https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais").mock(
        return_value=httpx.Response(
            200, json=[{"fecha": "2026-10-13", "valor": 600}, {"fecha": "2026-10-14", "valor": 611}]
        )
    )
    chain = default_macro_providers(httpx.Client())["riesgo_pais"]
    assert len(chain) == 2
    assert chain[0]("riesgo_pais")["valor"] == 611.0
    assert chain[1]("riesgo_pais") == {
        "valor": 611.0,
        "fecha": "2026-10-14",
        "fuente": "argentinadatos.com",
    }


@respx.mock
def test_default_tasa_bcra():
    respx.get("https://api.bcra.gob.ar/estadisticas/v4.0/monetarias/7").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"idVariable": 7, "detalle": [{"fecha": "2026-10-14", "valor": 22.5}]}]
            },
        )
    )
    out = default_macro_providers(httpx.Client())["tasa_bcra"][0]("tasa_bcra")
    assert out["valor"] == 22.5 and out["fecha"] == "2026-10-14"
    assert "bcra" in out["fuente"]


@respx.mock
def test_http_error_becomes_tool_error():
    respx.get("https://dolarapi.com/v1/dolares/bolsa").mock(return_value=httpx.Response(503))
    with pytest.raises(ToolError) as exc:
        default_macro_providers(httpx.Client())["dolar_mep"][0]("dolar_mep")
    assert exc.value.code == "PROVIDER_UNAVAILABLE"
