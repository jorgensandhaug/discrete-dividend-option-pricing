# Target-free publication numerical pilot v2

- Date: 2026-07-15
- Paper revision: `dbe85fa2bb0a897b7c379553a8915a1b71d0ba38`
- Engine revision: `4fe64a4823e2d5ceeb826279a58101ecda3a0455`
- Specification SHA-256: `4f04f0530dcb1df484492ed9ae56ea658dc18811c8d50d06513ad1e1be71fc73`
- Execution SHA-256: `51d5e84e8cc099d403d15269883eadf504962c29e2ffc38ee3f1b99cd22828d2`
- Receipt SHA-256: `3f3d85f5436602f5726ef1a569701b56bf353185851df6cc07ae38f9ef4ed8a7`
- Runtime: CPython 3.14.6, Rust 1.97.0, four threads, portable Rust `f64`, Apple M4 Pro
- Complete wall time: 204.382 seconds
- Outcome: failed closed; 60 of 61 predeclared gates passed

V2 changed only the general mesh sequence by appending 256 intervals. The terminal-cash value
moved from `11.2997751518` at 128 intervals to `11.3240479650` at 256, a `0.0242728132` change
below the unchanged `0.0313240480` limit. Every general-mesh gate then passed.

The sole v2 miss was the long-maturity, high-volatility in-the-money case's paired LSM path drift.
The 8,192-path and 32,768-path replication means were `24.7590216733` and `25.0541312970`; the
paired mean shift was `0.2951096237` against the predeclared uncertainty-adjusted `0.2882414507`
limit. This was a gate on paired per-seed differences, not a comparison of two independent means.

Every deterministic integration, boundary, financial-bound, mesh, LSM reference-envelope, and
native evaluation-standard-error gate passed. The smallest accepted regression population was 709
in-the-money fitting paths and the largest accepted condition diagnostic was 2,280.39. No support
continuation, retry, fallback, or alternate method occurred.

V2 remains unchanged. The next pilot preserves the existing factor-four path refinement and appends
131,072 fitting plus 131,072 independent evaluation paths. No other scientific or execution value
changes.
