# ADR 0040: Append one LSM path level after the high-volatility v2 miss

- Status: accepted for the third predeclared pilot
- Date: 2026-07-15

## Context

V2 resolved v1's terminal-cash mesh failure by appending 256 general intervals. The complete run
again returned failure status because one different gate missed: in the long-maturity,
high-volatility in-the-money case, the paired 8,192-to-32,768 path mean change was `0.2951096237`
against the frozen uncertainty-adjusted `0.2882414507` limit. Sixty of 61 gates passed. The miss was
small, but accepting it by widening the tolerance after observation would defeat the pilot.

The v2 receipt is retained under SHA-256
`3f3d85f5436602f5726ef1a569701b56bf353185851df6cc07ae38f9ef4ed8a7` and summarized in the
[v2 checkpoint](../performance/2026-07-15-target-free-pilot-v2.md). Deterministic, mesh, financial,
reference-envelope, and native standard-error evidence isolates path refinement as the unresolved
dimension.

## Decision

Define `project-publication-target-free-v3` by appending `p4`, with 131,072 policy-fitting paths and
131,072 independent policy-evaluation paths. This preserves the v1 sequence's factor-four refinement
ratio and uses the same 12 replication seeds, case domains, and disjoint fitting/evaluation domains,
so every p3 path population is a prefix of p4. The resulting frozen specification SHA-256 is
`6f7843bde65ec6629027b3f98b5f6f113f717be92697828467f9ea49c0d4cf27`.

Retain v2's cases, 16/32/64/128/256 general meshes, all three deterministic levels, regression
degree, rank and condition limits, exercise rule, strict support policy, numerical tolerances, and
four-thread portable execution. V3 evaluates the unchanged paired-drift formula on p3-to-p4
per-seed differences. It does not automatically select 131,072 paths for the full cohort.

Any v3 miss remains a failed receipt and requires a new named identity. Runtime cannot reduce the
path count, reuse fitting paths for valuation, change regression, or substitute a method.
