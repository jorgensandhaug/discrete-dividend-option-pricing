# Real project classical preparation - 2026-07-15

## Contract

- host: Apple M4 Pro, 14 logical CPU cores, 48 GB unified memory;
- runtime: locked CPython 3.14 and Polars 1.42.1;
- source: receipt-verified 19-column mixed-source project artifact;
- source artifact SHA-256:
  `aa778e3e1b362ffc32ac6ff273ca7d751438990eb0e935045340809ff6068808`;
- schedule: the 117 fixed submitted-study test windows, with exact union
  `[2015-01-01, 2024-08-11)`;
- selected contracts: 678,122;
- classical study SHA-256:
  `c92a16625ddc8e4feded1c0563fde417d4fc2c89253811d0f8a22afb193094e7`;
- cash boundary: `at_or_before`, with a terminal cash event applied before expiry payoff;
- benchmark excludes simulation, LSM fitting/evaluation, result persistence, and interpreter startup.

Both runs verified the detached producer receipt before scanning, rehashed the Parquet and receipt
after collection, and proved that every one of the 117 test partitions is nonempty. Event buffers
were allowed one explicit rechunk. No event was clipped, shifted, dropped, or synthesized.

## Result

| Measurement | Row-addressed audit | Natural strike groups |
|---|---:|---:|
| Materialized contracts | 678,122 | 678,122 |
| Simulation units | 678,122 | 96,666 |
| Cash-event values | 774,159 | 98,860 |
| Receipt verification | 0.0538 s | 0.0543 s |
| Projection, sort where required, and collection | 0.0071 s | 0.1406 s |
| Reverification and 117-window validation | 0.0068 s | 0.0099 s |
| Exact validation and native preparation | 0.0269 s | 0.1101 s |
| Instrumented total | 0.0945 s | 0.3150 s |
| Materialized frame bytes | 61,211,328 | 65,437,665 |
| Prepared native bytes | 61,211,336 | 18,231,684 |
| Whole-process maximum RSS | 734,560,256 | 800,833,536 |

The grouped path spends additional time and transient memory on its deterministic natural-key and
strike sort, but reduces persistent native input storage by 70.2% and the dominant path-simulation
population by 85.7%. It verified one constant state and unique strikes inside every
`(ticker, quote_date, expiry_date)` group. The row path remains a separately addressed audit
comparator and is never selected as an automatic fallback.

The exact prepared identities are
`55b85e625b7b67fb1f5d02844cb2f8665b12c0755dd7ab5a5183d228bfb4ac59` for row addressing and
`ca274e8e0737abd8803f9a13b0866ad9a55c5239fa1abbd04fede45a0737b521` for grouped addressing.
These are local engineering measurements under warm filesystem state, not pricing-throughput or
empirical-result claims.

## Supersession note

This preparation checkpoint predates the recovered submitted estimator's explicit sparse-support
policy. Its strict-policy study and prepared-input hashes remain valid historical identities, but
they are not the reconstruction identity used for pricing. The scalar/event bytes and cohort counts
did not change. The complete explicitly identified run is recorded in
[the full classical reconstruction checkpoint](2026-07-15-full-classical-reconstruction.md).
