# Deterministic publication-yield integration

- Date: 2026-07-15
- Benchmark engine revision: `77d34e358dad7ba075216629189f5a4bd9bd1f74`
- Pinned paper engine revision: `cd14031a8b1cd03b8258c73c739155a4161f88bc`
- Machine: Apple M4 Pro, macOS arm64, eight explicit CPU threads
- Source: receipt-verified private feature artifact; aggregate evidence only
- Artifact SHA-256: `aa778e3e1b362ffc32ac6ff273ca7d751438990eb0e935045340809ff6068808`
- Cohort: exact recoverable union of the submitted 117 test windows
- Shape: 96,666 natural states and 678,122 call contracts
- Rate partition: 94,906 nonnegative-rate states and 1,760 negative-rate states
- Selection: target-free; observed option prices were not read

The pinned descendant changes only Python-side typing of invalid scalar method counts; the Rust
solver, selected profile, and numerical output path are unchanged from the benchmark revision.

## Fixed profile

`(boundary quadrature=13, ordinary iterations=16, collocation nodes=6, price quadrature=21,
maximum relative boundary residual=1e-4, maximum relative obstacle projection=1e-6)`.

## Final paper-route validation

| Yield input | Direct full-group | Partitioned paper route | Max residual | Max projection |
|---|---:|---:|---:|---:|
| Forecast | 0.549 s | 0.549 s | 1.298e-5 | 2.622e-7 |
| Historical | 0.880 s | 0.876 s | 1.504e-6 | 3.016e-7 |

The paper route restores both row-aligned outputs and group-aligned diagnostics through the exact
cash/general partition maps. For each yield representation, its prices, origin-exercise decisions,
boundary residuals, obstacle projections, and European-theorem flags were bitwise identical to one
direct source-ordered request. The forecast route used the exact zero-yield/nonnegative-rate theorem
for 38,340 groups; the historical route required boundary solves for every group.

Preparation from the verified feature artifact through submitted-window selection, natural-group
materialization, XNYS exercise schedules, and publication partitions took 0.283 seconds in this
warm run. That preparation timing is reported for integration context and is not part of the
financial-method comparison.

## Nested numerical evidence

The separately declared over-resolved profile was
`(81, 48, 32, 131, 1e-8, 1e-8)`.

| Comparison with selected profile | Forecast | Historical |
|---|---:|---:|
| Maximum absolute price difference | $0.0004012 | $0.0003998 |
| Mean absolute price difference | $0.00001385 | $0.00001618 |
| RMSE | $0.00003459 | $0.00003493 |
| 99.9th percentile absolute difference | $0.0003135 | $0.0003155 |
| Differences above $0.001 | 0 | 0 |
| Origin-decision disagreements | 188 | 430 |

Decision disagreements occurred at the discretized free boundary. Both profiles passed the same
financial bounds and their explicit convergence gates. A faster four-node candidate was not
selected because its subsecond whole-cohort saving was immaterial relative to its larger numerical
margin. No market target or historical ranking entered that decision.
