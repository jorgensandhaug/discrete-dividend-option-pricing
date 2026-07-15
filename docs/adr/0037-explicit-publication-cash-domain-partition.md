# ADR 0037: Explicit publication cash-domain partition

- Status: superseded for new publication runs by ADR 0046; retained as historical evidence
- Date: 2026-07-15

## Context

The engine's cash-dividend-date exercise mesh removes intermediate exercise work only for
fixed-strike calls with deterministic cash events, nonnegative rates, and no cash event at expiry.
A publication cohort may contain both eligible and ineligible market states. Passing the complete
cohort to the specialization would be financially invalid; catching an engine error and retrying
selected rows on another mesh would be a hidden fallback.

Natural strike groups are the correct partition unit. Spot, maturity, rate, volatility, yields, and
the complete cash schedule are constant inside each verified group, while strike-specific policies
remain separate. The partition must not inspect market option prices, model predictions, losses, or
eventual rankings.

## Decision

Create a separately hashed publication study with no production default. Its estimator requires
strict regression support, an origin-inclusive general mesh, and the closed partition rule
`nonnegative_rate_strictly_pre_expiry_v1`. Path counts, general-mesh intervals, basis degree, and
replication count are supplied only by a later accepted numerical-pilot receipt.

At a materialized boundary, classify every verified natural group exactly once:

- `specialized`: constant continuously compounded rate is nonnegative and no cash-event time is
  bit-identical to maturity;
- `general`: the exact complement—negative rate or terminal cash.

The materializer creates compact immutable arrays for each domain, records original group and row
indices, proves that both maps are complete disjoint permutations, and hashes the source prepared
identity, publication study, domain order, and reconstruction maps. It never reads targets while
classifying groups.

Every publication replication uses the fixed reordered unit order `specialized, general`. The
second partition's random-domain root is the first root plus the specialized group count. This
preserves identical namespace assignment across the four variants and every rerun. Exact-cash and
escrowed-cash variants use `CashDividendExerciseMesh` only on the specialized partition and the
declared uniform mesh on the general partition. Forecast- and historical-yield variants use the
uniform mesh on both partitions. Empty partitions are skipped rather than sent as invalid native
requests.

All twelve native result/diagnostic columns are scattered through the immutable original-row maps.
Overlap, omission, changed partition length, or invalid stopping counts fails the complete variant;
no partial result escapes. The merged source-order output is the only input to aggregate metrics.

Paired seeds across different mesh identities are not described as pathwise-identical Brownian
increments. Exact/escrowed cash share an identical mesh within each domain, as do the two yield
variants. A stronger cross-mesh coupling would require a separately named random construction.

## Consequences

- The optimized cash path can be used on a heterogeneous cohort without weakening its theorem
  checks or creating a runtime fallback.
- General-domain rows remain part of the same declared empirical cohort and use the named finite
  general mesh whose convergence is evaluated; they are not discarded, imputed, shifted, or
  automatically retried after a specialization failure.
- The partition identity becomes an immutable input to replication and artifact identities.
- Publication execution remains unavailable until the target-free numerical pilot supplies an
  accepted estimator budget; fixture budgets exist only in tests.
