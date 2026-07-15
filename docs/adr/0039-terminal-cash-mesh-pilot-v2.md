# ADR 0039: Append one general mesh level after the terminal-cash v1 miss

- Status: accepted for the second predeclared pilot
- Date: 2026-07-15

## Context

The complete v1 target-free pilot returned failure status after 181.556 seconds. Sixty of 61 gates
passed. The sole miss was the terminal-cash case's uniform-mesh change from 64 to 128 intervals:
`0.0462706999` against the predeclared `0.0312997752` limit. The complete sequence was
`11.0178222073`, `11.1684970812`, `11.2535044519`, and `11.2997751518` at 16, 32, 64, and 128
intervals. The last two adjacent changes contracted by factors near 0.56 and 0.54, but that pattern
is evidence for another declared level, not permission to extrapolate a passing value.

All deterministic integration, upper-boundary, financial-bound, LSM reference, paired path-drift,
and evaluation-standard-error gates passed for all cases. V1 therefore isolated finite general
exercise timing as the unresolved approximation dimension. The receipt is retained under SHA-256
`4c097e93f0832d2449e6d63a3251df5596ec1c255a29d64a26a8b45ccfecd3a3` and summarized in the
[v1 checkpoint](../performance/2026-07-15-target-free-pilot-v1.md).

## Decision

Define `project-publication-target-free-v2` by appending 256 to the general uniform Bermudan mesh
sequence. Apply that sequence to every general-domain case because the publication estimator owns
one general mesh identity; do not special-case terminal cash after observing its result.
The resulting frozen specification SHA-256 is
`4f04f0530dcb1df484492ed9ae56ea658dc18811c8d50d06513ad1e1be71fc73`.

Retain v1's eight cases, three joint direct-quadrature levels, 2,048/8,192/32,768 path levels, 12
replications, seeds, case and fitting/evaluation domains, degree-four regression, rank and condition
limits, strict exercise tie, fail-on-insufficient-support policy, four-thread portable execution,
and every numerical tolerance. The v2 acceptance gate is the newly observed 128-to-256 change under
the unchanged formula `0.02 + 0.001 |V|`.

V2 does not infer a 256-interval production default. A pass admits the level to subsequent
cohort-scale target-free performance and price-stability work; a miss is retained and requires a
new named identity. No result changes the mesh or tolerance at runtime.
