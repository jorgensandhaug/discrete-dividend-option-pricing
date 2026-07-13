# ADR 0010: Trailing observed-cash dividend features

Status: accepted for the project as-submitted reconstruction

Canonical history contains ticker, ex-date, and finite nonnegative `Float64` cash amount. Exact
duplicate rows are rejected; distinct components on one ex-date are additive. The project identity
requires history coverage for every ticker and uses the inclusive interval from quote date minus 52
weeks through quote date. A missing ticker history is not silently interpreted as zero.

The lazy implementation aggregates same-date cash once, forms a per-ticker cumulative series, and
uses two validated backward as-of lookups per unique ticker/quote date. It therefore avoids
expanding every option row against every historical payment. The recovered scalar feature is
`log(1 + trailing cash / split-adjusted close) / (lookback weeks / 52)`; it remains separate from
the forward discrete-dividend schedule.
