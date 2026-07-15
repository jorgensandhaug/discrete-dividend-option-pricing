# Target-free publication numerical pilot v3

- Date: 2026-07-15
- Paper revision: `6723bdaf8e36dab8c8b26075a26d70dcbd0faba7`
- Engine revision: `4fe64a4823e2d5ceeb826279a58101ecda3a0455`
- Specification SHA-256: `6f7843bde65ec6629027b3f98b5f6f113f717be92697828467f9ea49c0d4cf27`
- Execution SHA-256: `51d5e84e8cc099d403d15269883eadf504962c29e2ffc38ee3f1b99cd22828d2`
- Receipt SHA-256: `931b40fc1e86af3dfb9fd0115523fd2dde509097cf8edbca2a4880a094642323`
- Runtime: CPython 3.14.6, Rust 1.97.0, four threads, portable Rust `f64`, Apple M4 Pro
- Complete wall time: 318.206 seconds
- Outcome: failed closed; 60 of 61 predeclared gates passed

V3 appended 131,072 fitting and 131,072 independent evaluation paths while preserving every p3
path as a p4 prefix. The v2 long/high-volatility drift resolved: its mean moved from `25.0541312970`
at p3 to `25.0976568867` at p4 against the deterministic `25.2151790030`, and all of that case's
gates passed.

The sole v3 miss was the terminal-cash LSM reference envelope. Its 256-opportunity deterministic
value was `11.3240479650`; p3 and p4 replication means were `11.1932375450` and `11.1932011155`.
The p3-to-p4 change was only `0.0000364294`, while the p4/reference gap was `0.1308468495` against
the unchanged uncertainty-adjusted `0.1185747552` limit. Increasing paths by four therefore reduced
sampling error without reducing this gap.

All 255 continuation regressions were fitted in every p4 replication. Terminal-case regression
conditions remained between roughly 121 and 206 at their maxima, with at least 64,854 in-the-money
fitting paths; sparse support and numerical conditioning do not explain the gap. Each p4 policy sent
roughly half of 131,072 evaluation paths to early exercise and half to expiry.

Engine revision `7fcbf1ac2163daf59118c3a6e0e7b6b20b160c3a` subsequently added a permanent
independent check: direct quadrature on this exact 256-opportunity terminal-cash schedule agrees with
a separately implemented 16,384-step interpolated CRR tree within `0.002`. The remaining evidence
therefore points to LSM continuation-policy approximation rather than deterministic integration or
Monte Carlo population size.
