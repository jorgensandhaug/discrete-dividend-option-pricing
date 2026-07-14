# Project-feature prefix benchmark (2026-07-14)

## Scope

This benchmark composes all five verified real-data inputs over the available development prefix.
It reports only aggregate identities and measurements:

- option source: 1,193,653 normalized rows in six annual Parquets over the half-open interval
  `[2011-01-01, 2016-02-18)`;
- support sources: the exact six-ticker forecast and historical-dividend projections, normalized
  underlying history, and reconstructed normalized NSS parameters;
- output: 270,589 complete 42-column rows from 2011-01-03 through 2016-02-17, with no null rows;
- output bytes: 10,789,589; SHA-256
  `1cbcb137735abce8743c54827110339f87741d6a2c5ba47d828fee4c712b3aa4`.

The run also exercised two recovered submission rules on real data. Forecast events are selected
strictly after their snapshot date, while an absent ticker in an otherwise verified sparse snapshot
is an empty schedule. Quotes on dates absent from the reconstructed NSS table are removed by the
submitted inner date intersection. Missing forecast dates, malformed rows, duplicate consumed
events, and malformed NSS rows remain fatal.

## Baseline and staged result

The Apple M4 Pro run used Polars 1.42.1 with its 14-thread pool and warm filesystem cache. The
original one-plan execution verified sources in 0.1662 seconds, built and validated the pipeline in
0.2835 seconds, and wrote the final Parquet in 0.1871 seconds. Measured internal end-to-end time was
0.6430 seconds, process wall time was 0.98 seconds, and peak resident set size was 807,518,208 bytes.

The production materializer now streams the quote-only cohort to one temporary uncompressed
Parquet and enriches that smaller cohort. The exact final output bytes and SHA-256 were unchanged.
Scan nodes fell from 20 to 14, internal time fell to 0.5276 seconds, and measured process peak RSS
fell to 706,674,688 bytes. That is an 18.0% time reduction and a 12.5% peak-RSS reduction on the
same prefix. The temporary stage is removed before the artifact receipt is committed.

## Thread-width tradeoff

Four alternating runs per width were measured while the full ORATS acquisition continued in the
background, so medians are more meaningful than individual timings. The active thread pool is now
part of the artifact writer identity because one- and two-thread floating-point reductions produced
different valid output bytes.

| Polars threads | Median internal total | Observed peak RSS range |
|---:|---:|---:|
| 1 | 1.255 s | 341--362 MB |
| 2 | 1.298 s | 365--376 MB |
| 4 | 1.261 s | 435--450 MB |
| 8 | 1.031 s | 524--556 MB |
| 14 | 0.941 s | 642--677 MB |

The default automatic thread pool remains the speed-first setting. Limited-memory machines can set
`POLARS_MAX_THREADS` before Python starts; Polars documents that the pool cannot be resized after
initialization. The materializer records the resulting width and rejects a runtime mismatch. The
pipeline uses the streaming Parquet engine, whose documented batch execution reduces memory
pressure; an explicit chunk-size override did not improve this workload and is therefore not
frozen. See the official [thread-pool](https://docs.pola.rs/api/python/dev/reference/api/polars.thread_pool_size.html)
and [streaming sink](https://docs.pola.rs/api/python/dev/reference/api/polars.LazyFrame.sink_parquet.html)
documentation. This prefix remains performance evidence, not a final-paper runtime claim.

A subsequent scale-up gate used 2,538,710 compacted option rows through 2019-11-07, the pinned
Federal Reserve Svensson source, and the exact dividend and underlying projections. It produced
523,551 complete rows for all six tickers. Staged and original one-plan execution emitted the exact
same 21,135,022 bytes and SHA-256. The quote-stage write plus enrichment sink took 0.4823 seconds;
the direct sink took 0.7563 seconds in the same warm-cache process, a 36.2% reduction. The temporary
uncompressed cohort occupied 53,130,876 bytes and was deleted after use.
