# Calendar-owned exercise materialization — 2026-07-15

## Contract

- host: Apple M4 Pro, 14 logical CPU cores, 48 GB unified memory;
- runtime: locked CPython 3.14, NumPy 2.5.1, and Polars 1.42.1;
- source: receipt-verified 19-column recovered project feature artifact, SHA-256
  `aa778e3e1b362ffc32ac6ff273ca7d751438990eb0e935045340809ff6068808`;
- calendar: XNYS 2009--2025 from `exchange_calendars` 4.13.2, cross-checked against XNAS;
- calendar asset SHA-256:
  `19683a870cf0eb1983b1727c0c62fded512c1b7440c0812e2984ea666311cd3b`;
- day count: ACT/365F civil dates, preserving source `Float64` values;
- partition: nonnegative-rate explicit cash schedules versus negative-rate uniform-mesh domain;
- no simulation, regression, result writing, network access, calendar-date repair, or fallback.

Both measurements loaded and verified the detached producer receipt, lazily selected the requested
cohort, created sorted natural strike groups, validated every date/time correspondence, generated
the exact ragged exercise schedules, built complete disjoint source-order maps, and reverified the
source artifact after consumption.

## Result

| Measurement | Full recovered artifact | Submitted 117-window test union |
|---|---:|---:|
| Contracts | 894,509 | 678,122 |
| Natural strike groups | 127,185 | 96,666 |
| Cash/exercise events | 136,128 | 98,860 |
| Terminal-cash groups | 2,393 | 2,325 |
| Exact-domain groups | 117,535 | 94,906 |
| Exact-domain rows | 816,421 | 664,185 |
| Negative-rate general groups | 9,650 | 1,760 |
| Negative-rate general rows | 78,088 | 13,937 |
| Frame collection | 0.092 s | 0.066 s |
| Native grouping | 0.077 s | 0.050 s |
| Schedule validation/materialization | 0.165 s | 0.123 s |
| Partition materialization | 0.031 s | 0.022 s |
| Instrumented total | 0.366 s | 0.261 s |

The full schedule SHA-256 is
`26aa9e41e6d3bd1221813438deaad4e0534616f40c1a2893df646b97fd4c8638`. The submitted-union
schedule SHA-256 is
`3008e547d044d3985452b41e99a08c9a8df6131750daee8b7e3996bfb13ac5e6`. The prepared natural-group
identities remain `063eb42e...` and `4ea3fa53...`, respectively.

This is a warm local engineering measurement, not a portable performance threshold or a pricing
result. An audit-only two-path study identity exercised the complete partition constructor; its
partition hash is deliberately not a production replication identity. Production path counts,
whole-study replications, and empirical inference remain separately predeclaration-gated.
