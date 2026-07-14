# ORATS annual-compaction benchmark (2026-07-14)

## Scope

This benchmark uses real licensed ORATS projections but reports only aggregate identities and
timings. It is a development prefix, not the final paper artifact:

- frozen source prefix: 2,297 trading days from 2007-01-03 through 2016-02-17;
- benchmark selection: half-open interval `[2011-01-01, 2016-02-18)`;
- selected input: 1,289 daily Parquets, 1,193,653 rows, and 36,911,516 bytes;
- output: 6 annual Parquets, the same 1,193,653 rows, and 20,137,728 bytes.

The annual artifact is 54.6% of the daily projected bytes. A full eager equality check confirmed
identical ordered schemas and exact row-by-row equality across all 17 columns. A floating-point
sum/mean aggregation was not last-bit identical because the annual row groups change parallel
reduction order; no stored row value changed.

## Results

Measurements ran on the Apple M4 Pro development machine with Polars 1.42.1. Files were warm in the
OS cache after compaction. Times below are internal `perf_counter` intervals unless explicitly
identified as process wall time.

| Operation | Daily files | Annual files | Speedup |
|---|---:|---:|---:|
| Cryptographic/content verification | 3.1570 s | 0.0370 s | 85.3x |
| Lazy metadata row count | 0.02660 s | 0.000791 s | 33.6x |
| Four-price filter plus three aggregates, median of 7 alternating runs | 0.04938 s | 0.005573 s | 8.86x |
| Full 17-column eager collection | 0.06354 s | 0.01260 s | 5.04x |

The one-time compaction command took 8.94 seconds wall time and 315,211,776 bytes maximum resident
set size. That measurement includes verification of all 2,297 files in the source prefix, annual
streaming writes, post-consumption source rehashes, output validation, manifest creation, and a
detached reload. It is deliberately more work than an unchecked concatenation.

## Decision

Keep the daily projection as the restart/recovery boundary and the annual projection as the repeated
study-scan boundary. The real prefix establishes a material improvement in verification, planning,
scan throughput, and storage while preserving exact ordered rows. Re-run the same benchmark on the
complete 2011--2024 paper horizon before locking final execution settings; do not extrapolate these
warm-cache prefix timings as a cluster claim.
