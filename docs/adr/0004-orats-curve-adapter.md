# ADR 0004: Explicit ORATS continuous-rate curve adapter

Status: accepted

The study adapter consumes one ORATS quote date at a time from a Polars `LazyFrame`. Raw
`trade_date` and `expirDate` values are parsed strictly from the historical delivery format; `yte`
and `iRate` must already be `Float64`. The adapter never infers units, casts financial numbers,
selects among conflicting rows, or drops invalid values.

ORATS defines `iRate` as a continuous risk-free rate and `yte` as years to expiration. Each unique
maturity is converted to a log discount factor using

```text
log P(0, t) = -iRate(t) * t.
```

The exact origin is inserted when absent. Model event times are interpolated linearly in log
discount space, which makes the instantaneous forward rate piecewise constant between source
nodes. Extrapolation is rejected. Negative rates remain valid; finite, representable discount
factors and an exact unit discount at the origin are mandatory in the engine.

Every maturity must map to exactly one rate and expiration, and expirations must increase with
maturity. The returned object retains the quote date, source maturities, rates, expirations, and
origin-insertion decision as provenance. Dataset, adapter, and engine revisions belong in the
immutable run manifest; licensed rows do not.
