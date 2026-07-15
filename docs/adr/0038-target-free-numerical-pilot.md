# ADR 0038: Target-free deterministic and LSM numerical pilot

- Status: accepted for the first predeclared pilot
- Date: 2026-07-15

## Context

The publication estimator cannot inherit the submitted 500-path, 150-date configuration merely
because it exists, and it cannot choose new budgets by consulting market-price errors or the prior
model ranking. Path population, finite exercise mesh, direct-integration grid, upper boundary, and
quadrature order are separate approximation identities. Performance matters because every path
budget is multiplied by many natural stock-state groups and complete randomized replications, but
speed cannot authorize a different financial contract or a hidden numerical fallback.

The engine now provides two materially independent routes on the same explicit finite call
schedule. The production direct-quadrature method performs no simulation, regression, stopping-rule
fit, or random sampling. Independent nonproduction references include segmented adaptive
integration for the published one-dividend case and a separately implemented event-aligned
interpolated CRR tree. Cross-method agreement is evidence within a declared error envelope; it does
not assert that different finite numerical methods should be digit-identical.

## Decision

Add a versioned target-free pilot with eight synthetic cases. The panel covers one and multiple
strictly pre-expiry cash events in the nonnegative-rate theorem domain, short maturity, continuous
yield, negative rates, terminal cash, simultaneous cash and continuous yield, longer maturity, high
volatility, and in/at-the-money calls. It contains no market option price, fitted empirical target,
legacy model output, or prior ranking.

Every case is priced on the complete joint deterministic sequence:

- `(upper=400, points=10,001, normal order=512)`;
- `(upper=600, points=20,001, normal order=1,024)`; and
- `(upper=800, points=40,001, normal order=2,048)`.

General-domain cases retain uniform Bermudan meshes with 16, 32, 64, and 128 base intervals,
augmented by exact cash-event times. The valuation origin is an exercise opportunity. Cases inside
the nonnegative-rate, zero-continuous-yield, strictly pre-expiry-cash theorem domain instead exercise
only at cash-event left limits; they are never moved there from the general domain at runtime.

LSM retains nested 2,048, 8,192, and 32,768 fitting and independent evaluation paths. Each level
uses the same 12 predeclared replication seeds and case-specific Philox domains, while fitting and
evaluation occupy different domains. Degree four, strict-greater exercise, rank tolerance `1e-12`,
condition limit `1e8`, and fail-on-insufficient-support are invariant. The execution identity is
separate from the scientific identity; the first receipt predeclares four workers and portable Rust
`f64` math.

The following gates are fixed before the first complete run:

- the last joint deterministic change is at most `0.003 + 0.0003 |V|`;
- the finest upper-boundary gap is at most `0.002`;
- the last general-mesh change is at most `0.02 + 0.001 |V|`;
- immediate-exercise and stock-dominance bounds hold within the deterministic envelope;
- the finest LSM replication mean differs from the deterministic value by at most
  `0.03 + 0.003 |V|`, plus the deterministic envelope and a two-sided 99% Student-t margin for 11
  degrees of freedom (`3.1058065155392804` times the replication standard error);
- the paired mean change from 8,192 to 32,768 paths is at most `0.02 + 0.002 |V|`, plus the same
  critical value times the paired replication standard error; and
- the finest mean native evaluation standard error is at most `0.08 + 0.005 |V|`.

All levels, replications, seeds, domains, diagnostics, wall times, gates, source revisions, runtime
versions, specification hash, and execution hash are written to one non-overwriting canonical JSON
receipt. A completed receipt returns a failing process status if any gate misses. A pricing,
regression, financial-domain, arithmetic, or artifact failure aborts without a partial success
receipt. No gate triggers refinement, retry, degree reduction, regularization, changed mesh,
alternate backend, or method substitution.

## Consequences

- A passing pilot supports a numerical configuration decision; it does not by itself select the
  final full-cohort budget or establish an empirical model ranking.
- A failing v1 receipt remains evidence. Any new sequence or tolerance requires a new named pilot
  identity rather than overwriting v1.
- Timings are recorded but are not correctness gates. Performance selection occurs only among
  financially admitted configurations after the numerical evidence is complete.
- The later cohort-scale pilot may use recovered financial inputs without market-price targets to
  assess group-level throughput and paired price stability. Market errors remain excluded until the
  estimator identity is frozen.

## Engine evidence

- Engine revision: `4fe64a4823e2d5ceeb826279a58101ecda3a0455`
- [Engine direct-quadrature ADR](https://github.com/jorgensandhaug/finoptions-engine/blob/4fe64a4823e2d5ceeb826279a58101ecda3a0455/docs/adr/0017-event-aligned-direct-quadrature.md)
- [Engine performance checkpoint](https://github.com/jorgensandhaug/finoptions-engine/blob/4fe64a4823e2d5ceeb826279a58101ecda3a0455/docs/performance/2026-07-15-event-aligned-quadrature.md)
