# ADR 0048: Frozen cash-LSM estimator and replication plan

- Status: accepted; complete production artifact pending
- Date: 2026-07-15
- Engine revision: `cd14031a8b1cd03b8258c73c739155a4161f88bc`

## Context

ADR 0047 removes simulation from the two continuous-yield variants. Only the direct liquidator
cash-jump and spot-matched escrowed-cash variants retain fitted stopping policies and Monte Carlo
valuation. Their publication identity must now fix the path budget, general exercise mesh,
regression method, random replications, and random namespaces before a complete result is run.

The recovered submitted-window union contains 96,666 natural states and 678,122 contracts. The
nonnegative-rate theorem domain contains 94,906 states and uses only its calendar-owned
pre-dividend exercise opportunities. The remaining 1,760 negative-rate states require the general
origin-inclusive American mesh. Those states have 31--365 calendar-day maturities, a 219-day
median, and at most four forecast cash events. Although they contain only 2% of states, their dense
mesh dominates LSM runtime.

Configuration selection must remain independent of observed option prices and historical model
rankings. A failed regression, missing support, invalid event, arithmetic error, or incomplete
replication must stop rather than trigger a smaller basis, another seed, a coarser mesh, or a
different method.

## Decision

Freeze `PROJECT_PUBLICATION_CASH_LSM_SPEC` with:

- 128 origin-inclusive uniform intervals in the negative-rate general domain;
- the exact calendar-owned ragged schedule and origin excluded in the nonnegative-rate theorem
  domain;
- 32,768 fitting paths and a disjoint 32,768-path evaluation population;
- degree-four standardized polynomial continuation regression;
- rank tolerance `1e-12` and condition limit `1e8`;
- strict-greater exercise ties; and
- `InsufficientExerciseSupportPolicy.FAIL`.

Freeze `PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC` with 12 replications. Replication `j` uses
`(0x243F6A8885A308D3 + j * 0x9E3779B97F4A7C15) mod 2^64` as the common seed for its disjoint fitting
and evaluation roles. The roles occupy group-domain blocks rooted at 10,000 and 1,000,000. The
complete group count is validated against both half-open blocks before execution. The modular seed
period is checked mathematically, and every declared replication has a distinct seed.

The two cash variants share a seed and group-domain addressing within each replication, producing
the intended paired comparison. Fitting and evaluation never share a random population. The
Student-t critical value for 11 degrees of freedom is fixed at `3.1058065155392804`. Runtime thread
count and CPU math backend remain explicit per-replication execution fields and hashes; neither is
selected automatically by the study or replication plan.

`PROJECT_PUBLICATION_STUDY_SPEC` combines that cash estimator with the deterministic yield profile
from ADR 0047. A complete publication result requires all 12 cash replications and the two
deterministic yield results. Cash publication predictions are arithmetic means of the complete
replication vectors; between-replication variability and native evaluation uncertainty remain
separate diagnostics. A successful subset is not a complete result.

## Target-free evidence

The retained v5 synthetic pilot evaluated 12 replications at 2,048, 8,192, 32,768, and 131,072
fitting/evaluation paths. Across its pure cash cases, the 32,768-to-131,072 replication-mean change
was at most `$0.0213`. The negative-rate cash reference changed by `$0.0004807` between 128 and 256
general intervals. The earlier reason for extending the mesh to 256 was a terminal-cash case;
ADR 0046 now places real nonnegative-rate terminal-cash states in the exact calendar-owned domain,
so that case no longer determines the general-domain production mesh.

Full-cohort throughput probes did not evaluate targets, errors, or rankings. At 2,048 paths and 256
general intervals, both exact-domain cash models took 12.8 seconds combined while both general-
domain models took 33.9 seconds combined. At 8,192 paths and the selected 128 intervals, the exact
pair took 41.4 seconds and the general pair 69.2 seconds on 14 Apple M4 Pro threads. Every request
completed with zero missing- or insufficient-support nodes. Minimum fitted support was at least 671
paths exact and 70 general in the first probe, then 2,755 exact and 711 general in the second; all
observed condition numbers were far below the fixed `1e8` limit. These timings size the production
run but do not select a model or result.

## Consequences

- The paper now has one complete immutable study identity rather than an unresolved numerical
  budget.
- The expected expensive work is bounded to two cash variants times 12 replications; deterministic
  yield prices are not recomputed as stochastic estimates.
- Reproduction on a non-Apple host uses an explicitly different execution identity while retaining
  the same financial study, random plan, paths, and mesh.
- Any new path count, mesh, basis, random construction, replication count, or critical value is a
  new named specification and requires retained target-free evidence.
- Production execution may resume only from verified immutable replication artifacts; it may not
  infer completion from a mutable filename or silently skip a failed seed.
