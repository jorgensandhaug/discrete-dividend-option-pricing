# TabPFN input-preparation checkpoint — 2026-07-14

This checkpoint measures only the accepted project cash-bin to flat-table allocation. It does not
install or execute TabPFN. The benchmark creates deterministic project-shaped source tensors before
timing, then copies six scalar columns and twelve chronological cash bins into the one row-major
float32 matrix a future local estimator requires. The target remains zero-copy.

Environment: Apple M4 Pro, 14 physical cores, 48 GiB RAM, macOS 26.2, CPython 3.14.6, uv 0.11.28.
Command, run in five fresh processes:

```shell
.venv/bin/python -m benchmarks.tabpfn_preparation --rows 1000000
```

Median allocation time was `0.054593 s`, or approximately `18.32 million rows/s`. The output table
is exactly 1,000,000 × 18 float32 values (`72,000,000` bytes). Median peak whole-process RSS was
`477,462,528` bytes; this includes interpreter/import state and the already-created source scalar,
bin, and target arrays, so it is not incremental allocation size. All five processes produced
prepared hash `6aa1e7268303b6210c0daceba19446cf702dddd6e9b0291135c9bd4f66e97b87`
and parent source hash `d9839d0d2c15fc135a434ea7d9bbf23be4a197bc3dbbd748b9ff2221b578d2b9`.

The copy is not a measured bottleneck relative to future foundation-model inference. Eliminating it
would require a different direct-to-final-buffer preparation API and would complicate the existing
shared neural representation. No such optimization is admitted without an end-to-end profile.

The benchmark support code normalizes `ru_maxrss` explicitly: Darwin reports bytes and Linux
reports kibibytes. Other platforms fail instead of emitting a mislabeled memory metric.
