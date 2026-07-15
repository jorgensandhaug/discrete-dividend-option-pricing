# ADR 0041: Diagnose terminal-cash LSM policy bias before further refinement

- Status: accepted for diagnostic execution
- Date: 2026-07-15

## Context

V3 resolved v2's path-drift miss but failed the terminal-cash reference envelope. At 256 general
exercise intervals, the deterministic value is `11.3240479650`; 32,768-path and 131,072-path LSM
replication means are `11.1932375450` and `11.1932011155`. The unchanged mean despite a factor-four
population increase, small p4 replication standard error, full regression support, and moderate
condition diagnostics make another path increase scientifically unmotivated.

Engine `7fcbf1ac2163daf59118c3a6e0e7b6b20b160c3a` independently cross-checks the same
terminal-cash schedule against a 16,384-step interpolated CRR tree within `0.002`. The production
quadrature method and independent tree share neither interpolation geometry nor backward transition
implementation. The remaining candidate is the global polynomial continuation-policy
approximation across 255 exercise nodes.

## Decision

Run one target-free diagnostic for each currently supported polynomial degree zero through four.
Every degree keeps the v3 terminal-cash case, 256-interval reference schedule, three deterministic
levels, complete 16/32/64/128/256 mesh sequence, p3 and p4 path levels, 12 seeds, exact v3 terminal
case domains, disjoint fitting/evaluation populations, rank and condition limits, strict exercise
tie, fail-on-insufficient-support policy, gates, and four-thread portable execution.

The diagnostic does not select the degree whose mean happens to lie closest to quadrature. It asks
whether population-stable bias changes systematically with the admitted polynomial space. Degree
four's v3 receipt is already evidence; rerunning its case-only identity verifies that the diagnostic
uses the same paths. All five specification hashes are frozen in tests before execution and all
receipts are retained whether their inherited gates pass or fail.

The degree-zero-through-four specification SHA-256 values are, in order,
`77bffc33ae67049cf764429e9cb1a1ae07c0ad7781334cd8b00fc6df2b9f7ad2`,
`b0e643d387d98bf07c149de6090567ead88eee4cea7fc4b3d76612539d5313f4`,
`17408240538036511a07a1e867f392fb961639e7d881abac799072175a4dc746`,
`33515f847f52dd19c3dbb7f1f9650b3f5fba6024bd035a16a044434ce696d2e6`, and
`beba98e74c2392ab0775b05842c1aa2b818701da956a01abf46f570d5d29fdfa`.

If no supported degree removes the bias stably, the next action is to design and independently test
a different continuation estimator rather than widening a gate or adding paths. Candidate methods
must retain out-of-sample policy evaluation, explicit numerical diagnostics, bounded memory, and
fatal unsupported-domain behavior. No diagnostic result changes the publication estimator at
runtime.
