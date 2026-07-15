# Explicit terminal-cash predecessor diagnostic

- Date: 2026-07-15
- Paper revision: `c3fc4e88c6f5f2c444e5839f37a45c74dbecbaa2`
- Engine revision: `e7dfbb0216456f51fe7b7857a4dc50b4a827ebce`
- Specification SHA-256: `f4700ffe31903ff810b2e773efe286b2f2850e6ec316ac469b471a57480e494d`
- Execution SHA-256: `51d5e84e8cc099d403d15269883eadf504962c29e2ffc38ee3f1b99cd22828d2`
- Receipt SHA-256: `2bf0d9fb5381cbf10810a70adf7f4d299bb19aa6dedf8fcff1e9a1b4557b4d77`
- Runtime: CPython 3.14.6, Rust 1.97.0, four threads, portable Rust `f64`, Apple M4 Pro
- Complete wall time: 22.366 seconds
- Outcome: passed all 8 predeclared gates

This diagnostic changed only the terminal-cash case's exercise domain from every pre-expiry node to
the final declared predecessor node. It retained the exact degree-four diagnostic's q1/q2/q3
sequence, 16/32/64/128/256 mesh sequence, p3/p4 nested fitting and independent evaluation paths,
12 seeds and random domains, regression controls, strict support and tie rules, tolerances, and
gates.

The deterministic q3 value was unchanged from the full 256-opportunity schedule at
`11.3240479650`; q2-to-q3 changed by `0.0001744130`. The 128-to-256 predecessor-mesh change was
`0.0242728132` against a `0.0313240480` limit.

| Level | Fitting paths | Evaluation paths | Replication mean | Replication SE | Mean native SE | Regressions per replication |
|---|---:|---:|---:|---:|---:|---:|
| p3 | 32,768 | 32,768 | 11.2868175411 | 0.0289062256 | 0.0985197521 | 1 |
| p4 | 131,072 | 131,072 | 11.2936038368 | 0.0141871308 | 0.0492083188 | 1 |

The p4/reference gap was `0.0304441281` against an uncertainty-adjusted `0.1144318415` limit. The
paired p3-to-p4 drift was `0.0067862958` against `0.1228588531`. Every replication fitted its single
decision with at least 64,944 in-the-money p4 training paths; the largest p4 regression condition
number was `169.613`.

Compared with the same degree-four paths under all 255 regression nodes, the corrected exercise
geometry both removed the stable bias and reduced complete diagnostic time from 56.340 to 22.366
seconds. This is evidence for an explicit sparse exercise-time API, not permission to infer calendar
predecessors inside the pricing engine.
