# Compact terminal-predecessor numerical diagnostic

Date: 2026-07-15

- Paper revision: `128134925b0f8b860d624ddcda185ae1899395ce`
- Engine revision: `818691e6f7638e6bd4fa781c52302792636717ae`
- Spec SHA-256: `e2aa1fd264a7521070fa7d693cacb668da4c2f4d486dae0ef146e6a4b3d9ae4f`
- Execution SHA-256: `51d5e84e8cc099d403d15269883eadf504962c29e2ffc38ee3f1b99cd22828d2`
- Receipt SHA-256: `7358175cd0b91002c659ba39bf2ae90c9d80211dcd3cd5968b0b99669f55d8de`
- Execution: four workers, portable Rust `f64`
- Result: 7/7 gates passed

## Retained sequence

| Level | Price | Change from prior | Wall time |
|---|---:|---:|---:|
| q1 | 11.3315269094 | — | 0.0029 s |
| q2 | 11.3276516812 | -0.0038752281 | 0.0086 s |
| q3 | 11.3257173321 | -0.0019343491 | 0.0263 s |
| p3 mean | 11.3373099123 | — | 0.0203 s total over 12 replications |
| p4 mean | 11.3274465278 | -0.0098633845 paired mean | 0.0870 s total over 12 replications |

The q3 upper-boundary gap was `3.41e-13`. P3 used 32,768 policy and 32,768 independent valuation
paths per replication; p4 used 131,072 of each. Every policy fitted one regression. The minimum p4
in-the-money fitting population was 65,127, the maximum condition number was 175.542, and the p4
mean native standard error was 0.0493553.

| Gate | Observed | Limit |
|---|---:|---:|
| Deterministic q2-to-q3 change | 0.0019343491 | 0.0063977152 |
| Deterministic upper-boundary gap | 3.41e-13 | 0.0020000000 |
| Immediate-exercise lower bound | 0 | 0.0063977152 |
| Stock dominance | 0 | 0.0063977152 |
| P4 mean to q3 reference | 0.0017291957 | 0.1166120170 |
| P3-to-p4 paired drift | 0.0098633845 | 0.1159309121 |
| P4 mean native standard error | 0.0493553254 | 0.1366285867 |

## Performance interpretation

The compact schedule completed in 0.157 seconds versus 22.366 seconds for the otherwise identical
redundant 256-node predecessor diagnostic: about 143x lower end-to-end wall time. Retained
quadrature work was about 237x faster and retained LSM replication work about 64x faster. This is
the expected benefit of removing transition-only nodes; it does not change the mathematical
contract.

The compact q3 reference differs from the redundant-grid q3 value by 0.00166937. Exact GBM has the
semigroup property, but the deterministic solver interpolates after each transition, so finite
quadrature/grid accumulation is not bitwise invariant to redundant subdivisions. The compact
sequence independently passed its frozen deterministic convergence and boundary gates. LSM random
addressing also changes with the event grid, which is why the complete stochastic envelope was
rerun rather than inferred from v4.

## Reproduction

```console
python3 tools/dev.py -- publication-numerical-pilot \
  --pilot terminal-compact-predecessor-v1 \
  --threads 4 \
  --math-backend rust_std_f64 \
  --output artifacts/numerical-pilot/terminal-compact-predecessor-v1-portable-4t.json
```

Receipt writes are atomic and non-overwriting. A missed gate returns a nonzero process status.
