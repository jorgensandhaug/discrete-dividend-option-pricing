# ADR 0033: Verified natural strike groups for classical pricing

- Status: accepted for project classical reconstruction
- Date: 2026-07-14

## Context

The project feature artifact contains several strikes for one underlying quote and expiry. Those
contracts share a simulated state only when every path input is identical. Repeating native path
generation for each strike wastes the dominant state-process work, but grouping on an incomplete
key or selecting an arbitrary representative could silently change a financial model.

Stateful pricing must also remain outside Polars expressions. A lazy query is appropriate for
projection and ordering; simulation, policy fitting, and evaluation are a separate native terminal
operation.

## Decision

Define the natural group key as `(ticker, quote_date, expiry_date)`. The lazy adapter selects only
the public key and classical input columns, then sorts by the natural key and strike. It does not
collect or price.

At the explicit materialized boundary, reject the request unless:

- every natural key is present and groups are contiguous;
- strike is unique inside each natural group; and
- spot, maturity, rate, 90-day volatility, forecast yield, historical yield, exact forecast cash
  times, and exact forecast cash amounts each have exactly one value inside the group.

Only after those checks does the adapter retain one path state and cash schedule per group. Strikes
and targets remain row-aligned and in exact materialized order. Contiguous uint32 group offsets,
group states, row values, and ragged event buffers are immutable and content-hashed.

The grouped native call uses one training/evaluation path pair per natural state and common random
numbers across its strikes. Each strike and each dividend variant still fits and evaluates its own
stopping policy. A failure stops the complete four-variant run and returns no partial result. No
float tolerance, representative selection, sorting inside the native engine, cast, event shift,
regression change, or fallback is permitted.

## Evidence

- Synthetic tests reject non-contiguous natural keys, a changed group volatility, and duplicate
  strikes, then execute all four dividend variants through the grouped native boundary.
- The receipted available strict-open-horizon cohort contains 512,567 contracts and compacts to
  74,215 exact natural states, with a mean 6.91 and maximum 38 strikes per state. Every constancy and
  uniqueness check passes.
- Full lazy projection and native-input preparation took 0.152 seconds on the Apple M4 Pro
  development machine.
- A licensed-data same-request benchmark is recorded in
  [`../performance/2026-07-14-grouped-classical-prefix.md`](../performance/2026-07-14-grouped-classical-prefix.md).

## Consequences

- Group-addressed and legacy row-addressed runs have different declared random identities and need
  not match digits. Their financial specifications and per-strike estimators remain the same.
- Common random numbers make within-state strike comparisons paired but induce Monte Carlo
  dependence; paper-level inference must respect the quote/state clustering already present in the
  data.
- Polars remains the lazy preparation engine, while Rust owns the stateful numerical kernel. There
  is no opaque row callback or Polars plugin dependency in the financial engine.
- The row-addressed path remains available as an audit comparator. The command requires an explicit
  `--population-addressing` value and dispatches to the corresponding separately typed runner; it
  never changes topology automatically.
- Classical artifact format version three records `natural_strike_groups` population addressing
  and the simulation-unit count separately from output observations. The loader remains compatible
  with version-two row-addressed artifacts.
