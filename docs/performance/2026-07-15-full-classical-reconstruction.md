# Full recoverable classical reconstruction - 2026-07-15

## Contract

- host: Apple M4 Pro, 14 logical CPU cores, 48 GB unified memory;
- runtime: locked CPython 3.14, Polars 1.42.1, and the explicit Apple Accelerate math backend;
- paper revision: `30cde75b4e881552c7ac96559189e11695dea47f`;
- engine revision: `3a650d95b42826019e1604ec94393f37f6c38e4b`;
- source feature SHA-256:
  `aa778e3e1b362ffc32ac6ff273ca7d751438990eb0e935045340809ff6068808`;
- study SHA-256: `13d2afe8cf36e8c275b3329ca894d5e724cb4ab9eb551d200f22c7cce4dfc937`;
- prepared grouped-input SHA-256:
  `4ea3fa536935a98685357a35c015a098763847a3ce0e3fa590fb2f5207b3dc05`;
- result artifact SHA-256:
  `ac58771adabc600fd183140d04e805457c4fa96e1ed0f9cfd03e048acaa0cae5`;
- request SHA-256: `36659ec9f92569e84fc575922e14a1a47346cb1f1648174c8590e228b43476f3`;
- fixed submitted-compatible test union: 117 windows over `[2015-01-01, 2024-08-11)`;
- 678,122 contracts and 96,666 proven natural strike groups;
- 500 training and 500 independent evaluation paths per simulation group;
- degree-four ITM polynomial OLS with explicit insufficient-support continuation;
- distinct training/evaluation seeds and nonoverlapping group-addressed Philox domains.

The recovered cohort is 96.911% of the 699,737 observations in the submitted result table. All rows
for DUK, EPD, MO, SO, and VZ are recovered; 21,615 later ETR rows inside the fixed test union remain
unavailable. No missing row, prediction, event, or model state was imputed.

## Aggregate result

| Variant | N | Current MAPE (%) | Submitted MAPE (%) | Current RMSE | Submitted RMSE |
|---|---:|---:|---:|---:|---:|
| Exact cash jump | 678,122 | 5.359951 | 4.695443 | 0.532752 | 0.651005 |
| Escrowed cash, spot matched | 678,122 | 5.335945 | 4.750231 | 0.534680 | 0.666347 |
| Forecast continuous yield | 678,122 | 6.058589 | 4.924086 | 0.568239 | 0.659504 |
| Historical continuous yield | 678,122 | 6.522954 | 5.253302 | 0.577759 | 0.665882 |

The high-level comparison survives: both discrete-cash representations outperform both
continuous-yield representations on MAPE and RMSE, and historical yield is weakest. Exact and
escrowed cash remain close. Their MAPE ordering reverses in this reconstruction, while exact cash
has the lowest RMSE. This is qualitative reconstruction evidence, not exact numerical replication
or a statistical significance claim.

The declared sparse-support rule was used for only three contracts and 19 exercise nodes in each
cash variant. It was never used by either continuous-yield variant. No ordinary unsupported node
was returned, every accepted regression had at least five ITM rows, and every rank, condition,
finite-arithmetic, stopping-count, event, price-bound, manifest, and artifact-content gate passed.

## Performance

The clean one-shot process completed in 1,040.58 seconds (17 minutes 20.58 seconds), with 7,969.83
user CPU seconds, 123.97 system seconds, and 1,363,427,328 bytes maximum RSS. The 96,666 natural
states reduce the path-simulation population by 85.7% relative to row addressing. The verified
safe-tensors artifact contains 162,757,920 bytes and was written only after all four variants and
metrics completed.

These are local end-to-end engineering measurements, not portable performance thresholds. The run
uses one paired training/evaluation seed pair and 500 paths per population. Publication inference
still requires predeclared whole-study random replications and date/ticker-aware uncertainty;
option rows are dependent, MAPE emphasizes cheap contracts, the ETR tail is incomplete, and the
new engine intentionally differs from the submitted solver, random generator, and in-sample
valuation implementation.
