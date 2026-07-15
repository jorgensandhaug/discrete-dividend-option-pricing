# Target-free publication numerical pilot v5

Date: 2026-07-15

- Paper revision: `b9d586f5ef5ddc5149a84e7097930adb02d17089`
- Engine revision: `818691e6f7638e6bd4fa781c52302792636717ae`
- Spec SHA-256: `8dff50a6cef9c93271c745f2f6cb784552516d739d3e109884b451378493ce64`
- Execution SHA-256: `51d5e84e8cc099d403d15269883eadf504962c29e2ffc38ee3f1b99cd22828d2`
- Receipt SHA-256: `a8e05bba21db9cdf24beac3bdf5e20dbbc5777b9799a858e81135257945285ea`
- Execution: four workers, portable Rust `f64`
- Result: 8/8 cases and 60/60 gates passed in 93.220 seconds

## Results

| Case | Exercise domain | q3 reference | p4 mean | Absolute gap | Limit |
|---|---|---:|---:|---:|---:|
| Published one cash ATM | cash-event theorem | 11.6559872504 | 11.6772149008 | 0.0212276504 | 0.1042264193 |
| Three cash events ATM | cash-event theorem | 12.0444844571 | 12.0323241277 | 0.0121603294 | 0.1182389850 |
| Short cash call ITM | cash-event theorem | 9.0765420349 | 9.0966842850 | 0.0201422501 | 0.0980215347 |
| Continuous yield ATM | uniform Bermudan | 8.9405417958 | 8.9397336380 | 0.0008081578 | 0.0951643781 |
| Negative-rate cash ATM | uniform Bermudan | 8.2666972323 | 8.2535291467 | 0.0131680855 | 0.0896754814 |
| Terminal cash ATM | compact predecessor | 11.3257173321 | 11.3274465278 | 0.0017291957 | 0.1166120170 |
| Mixed cash and yield ATM | uniform Bermudan | 11.1333399149 | 11.1041164977 | 0.0292234172 | 0.1162436793 |
| Long/high-volatility ITM | uniform Bermudan | 25.2151790030 | 25.0976568867 | 0.1175221163 | 0.2197902107 |

All nonterminal cases are numerically identical to v4 after wall-clock fields are removed. V5
changes only the terminal case from a redundant 256-transition predecessor schedule to the exact
three-node schedule `[0, 255/256, 1]`. That case independently passed its deterministic convergence,
boundary, financial-bound, LSM-reference, paired-path-drift and native-standard-error gates.

V5 completed about 1.49x faster than v4's 139.202 seconds. This is not a direct whole-estimator
benchmark because ordinary system timing varies and the engine revision changed; the isolated,
same-revision schedule comparison in ADR 0044 is the controlled performance evidence.

## Interpretation boundary

The pilot validates the numerical implementation against a target-free deterministic method over a
predeclared synthetic domain. It does not use recovered project prices, reproduce the submitted
ranking, establish empirical uncertainty, or select a method because it resembles a legacy result.
General continuous-yield, negative-rate, mixed-dividend and high-volatility cases retain their
declared 256-opportunity Bermudan approximation until a separate scientific decision changes them.

## Reproduction

```console
python3 tools/dev.py -- publication-numerical-pilot \
  --pilot v5 \
  --threads 4 \
  --math-backend rust_std_f64 \
  --output artifacts/numerical-pilot/v5-portable-4t.json
```

Receipt writes are atomic and non-overwriting. Any missed gate returns a nonzero process status.
