# Target-free publication numerical pilot v4

- Date: 2026-07-15
- Paper revision: `e4e81a9f2448f242883966fe38d266c4f297ccab`
- Engine revision: `e7dfbb0216456f51fe7b7857a4dc50b4a827ebce`
- Specification SHA-256: `344670190d05856d90e0ab3b2d9a084a4de32cb1c3e68bc224d943edc21e0c32`
- Execution SHA-256: `51d5e84e8cc099d403d15269883eadf504962c29e2ffc38ee3f1b99cd22828d2`
- Receipt SHA-256: `7a625cdb1fdc39f97db8162625ac17f8d1e09a02401e3f178c3ddec59937d6f2`
- Runtime: CPython 3.14.6, Rust 1.97.0, four threads, portable Rust `f64`, Apple M4 Pro
- Complete wall time: 139.202 seconds
- Outcome: passed all 61 predeclared gates across all 8 cases

V4 changed only the terminal-cash case's exercise domain from every pre-expiry mesh node to its
final declared predecessor. It preserved v3's other seven cases and every quadrature, mesh, path,
random-stream, regression, support, tie, tolerance, and gate identity.

| Case | Deterministic q3 | LSM p4 mean | Absolute gap | Gate limit |
|---|---:|---:|---:|---:|
| Published one-cash ATM | 11.6559872504 | 11.6772149008 | 0.0212276504 | 0.1042264193 |
| Three cash events ATM | 12.0444844571 | 12.0323241277 | 0.0121603294 | 0.1182389850 |
| Short cash call ITM | 9.0765420349 | 9.0966842850 | 0.0201422501 | 0.0980215347 |
| Continuous yield ATM | 8.9405417958 | 8.9397336380 | 0.0008081578 | 0.0951643781 |
| Negative-rate cash ATM | 8.2666972323 | 8.2535291467 | 0.0131680855 | 0.0896754814 |
| Terminal cash ATM | 11.3240479650 | 11.2936038368 | 0.0304441281 | 0.1144318415 |
| Mixed cash and yield ATM | 11.1333399149 | 11.1041164977 | 0.0292234172 | 0.1162436793 |
| Long high-volatility ITM | 25.2151790030 | 25.0976568867 | 0.1175221163 | 0.2197902107 |

The terminal-case deterministic value is bit-identical to v3. Its p4 mean/reference gap fell from
`0.1308468495` to `0.0304441281` on the same stochastic paths and polynomial degree. Every other
case reproduces its v3 observation because its specification is unchanged.

This passing synthetic pilot is the numerical baseline for subsequent estimator engineering. It is
not a market-data fit, a legacy-result target, a model comparison, or an uncertainty analysis for
the paper's empirical conclusions.
