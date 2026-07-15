# Real project neural preparation — 2026-07-15

## Contract

- host: Apple M4 Pro, 14 logical CPU cores, 48 GB unified memory;
- runtime: locked CPython 3.14, Polars 1.42.1, and eight native preparation threads;
- source: receipt-verified 19-column mixed-source project artifact;
- source rows: 894,509 before the rolling-schedule horizon projection;
- schedule: 117 fixed project windows and all five reconstructed neural variants;
- event tensors: five chronological right-padded slots with masks and lengths, plus twelve bins;
- benchmark excludes neural runtime loading, training, inference, and artifact writing.

The preparation boundary scans the minimal union of required columns once, sorts the rolling
horizon once, validates and borrows the ragged list buffers once, constructs each distinct complete
representation once, and exposes phase ranges as read-only row views. The detached producer receipt
is fully verified before the scan; Parquet and receipt are rehashed after preparation.

## Result

| Measurement | Result |
|---|---:|
| Receipt and artifact load | 0.0686 s |
| One-pass five-model preparation | 0.3290 s |
| Post-preparation Parquet and receipt rehash | 0.0039 s |
| Rows in the union of rolling windows | 843,730 |
| Maximum resident set reported by `/usr/bin/time -l` | 888,455,168 bytes |

The real artifact's maximum forecast-event count is five. Exactly 89 rows reach that bound and no
row exceeds it, so capacity overflow remains a checked error rather than truncation. The measured
schema identity is `eae3db69d3ed5fe303cec4bc48cb6804dd04e0d5a56768ccbdd14017791610ba`.

This is a local engineering measurement, not a training-throughput or publication-performance
claim. Wall time includes warm local filesystem state and should be compared only under an explicit
matching environment.
