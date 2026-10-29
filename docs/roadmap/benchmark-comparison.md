# Idea: "Did I beat the dollar, inflation and a fixed-term deposit?"

Status: **documented, not scheduled.** Builds on `get_portfolio_performance` and `get_ccl_at`.

## Problem

The question an Argentine investor really asks is not "did my portfolio go up?" but "did I do
better than doing nothing?". Today the assistant explains *why* the portfolio moved (price vs.
exchange effect) but not whether the move was good relative to the usual alternatives.

## What the client sees

A short comparison for a closed period, in percentage points, with source and date per line:

| Alternative | Return | Source |
| --- | --- | --- |
| Your portfolio (ARS) | x,x% | `get_portfolio_performance` |
| Dollar (CCL) | x,x% | `get_ccl_at` |
| Inflation (INDEC) | x,x% | argentinadatos |
| Fixed-term deposit (BADLAR) | x,x% | BCRA |

The assistant narrates the table and never ranks the options as advice.

## Data sources

| Series | Endpoint | Status |
| --- | --- | --- |
| CCL history | `argentinadatos` `/v1/cotizaciones/dolares/contadoconliqui` | Verified (daily rows since 2013, ascending) |
| Fixed-term rate | BCRA `/estadisticas/v4.0/monetarias/7` (BADLAR, nominal annual %) | Already used by `get_macro("tasa_bcra")`; only the latest value, no history helper yet |
| Inflation | `argentinadatos` `/v1/finanzas/indices/inflacion` | Verified 2026-10-22: monthly `{fecha, valor}` in %, last row 2026-09-30 (1.9) |

## Calculation (deterministic, in code)

- **Dollar:** `fx1 / fx0 - 1`.
- **Inflation:** compound the monthly rates of the months fully inside the window.
- **Fixed-term deposit:** `(1 + BADLAR / 365 * days) - 1` (simple accrual on the nominal rate).
  Needs the rate at the start of the window, so a BADLAR history helper is required.
- **Portfolio:** `retorno_total_pct` from `compute_performance`.

## Constraints

- Inflation is published with a lag (the latest month is the previous one). It is only valid for
  closed calendar months. For `dia` and `semana` the row must say "not available", never an
  estimate.
- The portfolio return assumes the current holdings for the whole period (same caveat as
  `get_portfolio_performance`).
- BADLAR is a nominal rate of private banks, an approximation of a deposit, not an offer.

## Guardrails

- Comparing is not recommending. The prompt must keep forbidding "switch to a deposit" or
  "buy dollars"; add an adversarial test next to `tests/test_adversarial_recommendations.py`.
- Every number must come from the tool output so `untraced_numbers` stays empty.

## Suggested slices

1. `get_benchmarks(periodo)` pure function and tests (property: each benchmark equals its
   closed-form value for fixed inputs).
2. BADLAR history provider and inflation provider with caching, like `get_ccl_at`.
3. Session-bound tool, registry wiring, prompt rule and example.
4. Optional: a section in the daily brief.

## Open questions

- Should the comparison be offered for `mes` only, or also `semana` with inflation omitted?
- Is BADLAR the right deposit proxy for the bank's own customers, or should the bank's rate be
  injected?
