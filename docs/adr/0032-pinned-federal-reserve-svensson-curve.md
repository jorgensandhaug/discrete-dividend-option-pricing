# ADR 0032: Pin the published Federal Reserve Svensson curve

## Status

Accepted.

## Context

The project requires a daily continuously compounded risk-free zero yield at each option maturity.
Refitting Treasury constant-maturity quotes would introduce a second estimation problem and mix par
yield observations with the zero-yield quantity used by the pricing models. The Federal Reserve
publishes daily nominal Svensson parameters together with continuously compounded fitted zero
yields, but the downloadable file is a mutable current vintage and tabulates zero yields only at
whole-year maturities.

## Decision

One command accepts an explicitly hashed and sized vintage of the Federal Reserve nominal yield
curve CSV. The half-open date horizon, percentage-to-decimal scale, continuously compounded quote
convention, Polars runtime, and durable streaming Parquet settings are part of a content-addressed
request. `BETA0` through `BETA3` are converted from percentage points to decimal rates; `TAU1` and
`TAU2` remain in years.

The projection evaluates the published Svensson formula at the option's positive year fraction. It
records the stable numerical floor `1e-8` years and the exact zero-maturity limit
`(BETA0 + BETA1) / 100`, so the existing short-end branch is algebraically continuous with the same
curve rather than flattening sub-one-year options to the published one-year yield. As an independent
source check, the one-year yield reconstructed from every complete parameter row must agree with
published continuously compounded `SVENY01` within its four-decimal percentage-point rounding
tolerance.

Rows inside the requested calendar horizon are either complete or fully unavailable. Fully
unavailable rows are excluded and counted because they represent dates for which no curve was
published; partial rows, duplicate dates, nonfinite parameters, nonpositive decay scales, and a
parameter/published-yield inconsistency fail before output. The resulting Parquet and receipt bind
the complete source and output identities and can be verified without retaining the source CSV.

## Consequences

The paper uses one coherent published zero-curve model and quote convention at the exact option
maturity. There is no par-yield refit, one-year short-end flattening, carry-forward, interpolation,
repair, or silent fallback. A future Federal Reserve revision or Polars upgrade produces a distinct
artifact instead of silently changing an existing study input.
