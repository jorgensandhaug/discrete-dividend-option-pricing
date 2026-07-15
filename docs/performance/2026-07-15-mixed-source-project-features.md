# Mixed-source project feature materialization — 2026-07-15

## Contract

- host: Apple M4 Pro, 14 logical CPU cores, 48 GB unified memory;
- runtime: locked CPython 3.14 and Polars 1.42.1;
- long-form partition: five submitted tickers, `Float64` financial values, full project horizon;
- compacted partition: ETR, canonical raw-ORATS `Float32`, through 2019-11-07;
- support: verified six-ticker forecast dividends, historical dividends, underlying history, and
  submitted NSS parameters;
- output: ordered 19-column experiment schema, Zstandard level 3, 131,072-row groups;
- licensed source rows and local paths remain outside Git.

The option partitions are ticker-disjoint. Each is filtered in its source precision and narrowed
only after cohort selection. The benchmark includes source verification, long-form pairing and
validation, quote filtering, the temporary uncompressed quote stage, all feature joins, durable
output, source re-verification, hashing, receipt writing, and detached output validation.

## Result

| Measurement | Result |
|---|---:|
| Recoverable rows from 2011 onward | 894,509 |
| Recoverable descriptive rows from 2012 | 855,495 |
| Recoverable rows on or after 2015-01-01 | 689,887 |
| Output columns | 19 |
| Output bytes | 6,957,440 |
| Verified v2 materialization | 7.77 s |
| Independent v2 fresh-root materialization | 6.08 s |
| Nine-column quote-stage fresh-root materialization | 5.87 s |

The fresh-root run reproduced request identity
`6b5e8683f6d5530fc90e39e7e0c68cb848c48dd7ba4612f3a44ab3f105e1fa59`, pipeline identity
`c3b0adfa99f0bfd323e1c8d165a7a9a58b0cbd3592db8999012effd5c8960637`, output byte count,
per-ticker coverage, and SHA-256
`aa778e3e1b362ffc32ac6ff273ca7d751438990eb0e935045340809ff6068808`.
All 894,509 rows are complete and finite, contract keys are unique, and each support enrichment
retains the entire recoverable quote cohort. Pruning raw and diagnostic quote fields immediately
after filtering reduced the latest fresh-root wall time while leaving the request identity and
output bytes unchanged.

These are local engineering measurements, not publication-performance claims. The request receipt
binds the Polars version and thread-pool size because they are part of the byte-producing execution
identity.
