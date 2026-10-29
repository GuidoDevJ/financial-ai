import random

import pytest

from app.tools.get_portfolio_performance import compute_performance

# Two rounded parts can differ from the rounded whole by up to 0.01 (plus float noise).
FLOAT_EPSILON = 1e-9


def holding(ticker, qty, currency, price, variation_pct):
    return {
        "ticker": ticker,
        "cantidad": qty,
        "moneda": currency,
        "precio_actual": price,
        "variacion_pct": variation_pct,
    }


def random_portfolio(rng):
    return [
        holding(
            f"T{i}",
            rng.randint(1, 500),
            rng.choice(["ARS", "USD"]),
            round(rng.uniform(1, 5000), 2),
            round(rng.uniform(-15, 15), 2),
        )
        for i in range(rng.randint(1, 8))
    ]


@pytest.mark.parametrize("seed", range(25))
def test_contributions_add_up_to_the_total_return(seed):
    rng = random.Random(seed)
    portfolio = random_portfolio(rng)
    fx0 = round(rng.uniform(900, 1700), 2)
    fx1 = round(fx0 * rng.uniform(0.9, 1.1), 2)

    out = compute_performance(portfolio, fx0=fx0, fx1=fx1)

    # Each contribution is rounded to 2 decimals, so the sum may drift half a cent per row.
    tolerance = 0.005 * (len(portfolio) + 1)
    contributions = sum(c["contribucion_pp"] for c in out["contribuyentes"])
    assert contributions == pytest.approx(out["retorno_total_pct"], abs=tolerance)
    assert out["efecto_precio_pp"] + out["efecto_cambiario_pp"] == pytest.approx(
        out["retorno_total_pct"], abs=tolerance
    )
    for c in out["contribuyentes"]:
        assert c["efecto_precio_pp"] + c["efecto_cambiario_pp"] == pytest.approx(
            c["contribucion_pp"], abs=0.01 + FLOAT_EPSILON
        )


def test_ars_assets_have_no_exchange_effect():
    out = compute_performance([holding("PAMP.BA", 100, "ARS", 110.0, 10.0)], fx0=1000, fx1=1200)
    assert out["efecto_cambiario_pp"] == 0
    assert out["retorno_total_pct"] == pytest.approx(10.0, abs=0.01)


def test_usd_asset_splits_price_and_exchange_effects():
    # p0 = 100, p1 = 110 (+10%); fx 1000 -> 1100 (+10%). Total = 1.1 * 1.1 - 1 = 21%.
    out = compute_performance([holding("GGAL", 10, "USD", 110.0, 10.0)], fx0=1000, fx1=1100)
    assert out["efecto_precio_pp"] == pytest.approx(10.0, abs=0.01)
    assert out["efecto_cambiario_pp"] == pytest.approx(11.0, abs=0.01)
    assert out["retorno_total_pct"] == pytest.approx(21.0, abs=0.01)


def test_contributors_are_sorted_by_absolute_impact():
    out = compute_performance(
        [
            holding("SMALL", 1, "ARS", 100.0, 1.0),
            holding("BIG", 100, "ARS", 100.0, -5.0),
            holding("MID", 10, "ARS", 100.0, 3.0),
        ],
        fx0=1000,
        fx1=1000,
    )
    assert [c["ticker"] for c in out["contribuyentes"]] == ["BIG", "MID", "SMALL"]
