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

## Baseline result

The Apple M4 Pro run used Polars 1.42.1 with warm filesystem cache. Verified source loading took
0.1662 seconds, pipeline validation/build took 0.2835 seconds, the streaming Parquet sink took
0.1871 seconds, and measured internal end-to-end time was 0.6430 seconds. Process wall time was
0.98 seconds. Peak resident set size was 807,518,208 bytes.

The optimized lazy plan contains 20 Parquet scan nodes. Throughput is already sufficient for rapid
prefix iteration, but peak memory is not yet the desired limited-hardware result. The next gate is
to reduce repeated quote-lineage scans and measure thread-count tradeoffs without weakening source,
coverage, or financial-domain validation. This prefix is performance evidence, not a final-paper
runtime claim.
