# ADR 0020: Explicit four-baseline classical reconstruction

- Status: accepted for the project-study reconstruction
- Date: 2026-07-14

## Context

The project results contain four classical LSM comparisons with shared market inputs and estimator
settings but materially different dividend models. Recovering them through notebooks or mutable
configuration files would make model identity, volatility semantics, and random populations
ambiguous.

## Decision

Freeze four typed variants and retain their submitted result aliases:

| Variant | Submitted alias | Dividend model |
|---|---|---|
| `exact_cash` | `IT_GBM_SPOT` | exact event-aligned cash jumps under `liquidator` limited liability |
| `escrowed_cash_spot_matched` | `IT_GBMT_FD_AH` | risky prepaid forward plus deterministic cash, with origin dollar-volatility matching |
| `forecast_yield` | `IT_GBMT_FY` | constant continuous forecast-dividend yield |
| `historical_yield` | `IT_GBMT_HY` | constant continuous historical-dividend yield |

All use `underlying_price`, `strike`, `yte`, `risk_free_rate`, and
`historical_volatility_90`; the cash variants share the exact forecast event lists. The immutable
reconstruction estimator uses 150 base intervals, origin exercise excluded, degree-four ITM LSM,
500 training paths, and 500 independent evaluation paths. Regression tolerance, condition limit,
tie rule, random seeds, domain roots, and worker count are all explicit. Financial/estimator
identity is separate from execution identity.

One materialized Polars boundary converts the shared scalar columns and target to a single
Fortran-ordered Float64 allocation whose columns are directly borrowable by the native engine.
Event child values stay zero-copy unless the caller explicitly permits a rechunk. Stateful pricing
is a native batch call outside Polars expressions.

## Consequences

- No variant may infer a dividend representation or volatility convention.
- The four variants share row order and immutable input bytes, making comparisons paired by
  construction.
- Historical submitted metrics are qualitative reconstruction references; immutable new runs name
  their engine, random, and independent-evaluation identity rather than claiming exact legacy RNG
  equivalence.
- Float64 classical predictions and metrics are not silently downcast to the neural Float32
  boundary.
- A complete four-way result is persisted atomically as one content-addressed safe-tensors file.
  It stores the target once, every native result/diagnostic column, exact Float64 metrics, and a
  canonical manifest binding source hashes, full code revisions, dependency lock, engine/compiler,
  prepared input bytes, and execution identity. Local filesystem paths are excluded.
