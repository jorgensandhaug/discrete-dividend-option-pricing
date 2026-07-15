# Target-free publication numerical pilot v1

- Date: 2026-07-15
- Paper revision: `3e9c1c34c463ddc5c4dc9cb85c9807fd56d60a5f`
- Engine revision: `4fe64a4823e2d5ceeb826279a58101ecda3a0455`
- Specification SHA-256: `e8680f30b61c68b170ec5080f8bbc98aac1a9571f17b612a187ecc075feff2fa`
- Execution SHA-256: `51d5e84e8cc099d403d15269883eadf504962c29e2ffc38ee3f1b99cd22828d2`
- Receipt SHA-256: `4c097e93f0832d2449e6d63a3251df5596ec1c255a29d64a26a8b45ccfecd3a3`
- Runtime: CPython 3.14.6, Rust 1.97.0, four threads, portable Rust `f64`, Apple M4 Pro
- Complete wall time: 181.556 seconds
- Outcome: failed closed; 60 of 61 predeclared gates passed

All eight cases completed at every deterministic, mesh, path, and replication level. Seven cases
passed every gate. The only miss was the last uniform-mesh change for terminal cash:

| Base intervals | Deterministic value | Change from prior |
| ---: | ---: | ---: |
| 16 | 11.0178222073 | — |
| 32 | 11.1684970812 | 0.1506748739 |
| 64 | 11.2535044519 | 0.0850073707 |
| 128 | 11.2997751518 | 0.0462706999 |

The frozen limit at the 128-interval value was `0.02 + 0.001 |V| = 0.0312997752`. The observed
change therefore failed rather than being accepted as close enough. This is consistent with a
terminal cash event whose same-node pre-cash exercise phase is not represented directly by the
finite schedule: a uniform pre-expiry mesh approaches that decision from earlier dates.

Every direct-quadrature level-change and boundary gate passed. Every LSM mean/reference envelope,
paired 8,192-to-32,768 path-drift gate, and native evaluation-standard-error gate also passed. The
smallest accepted regression population was 698 in-the-money fitting paths and the largest accepted
condition diagnostic was 3,183.48; no sparse-support continuation, fallback, retry, or alternate
method occurred.

V1 remains unchanged. The next pilot appends 256 general base intervals and changes no case,
quadrature level, path population, replication, seed, domain, regression rule, or tolerance.
