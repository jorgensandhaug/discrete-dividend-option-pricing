# Grouped classical prefix checkpoint: 2026-07-14

This checkpoint uses licensed local data and records aggregate counts/hashes only. It is a
performance and integration result, not a publishable empirical table.

## Preparation gate

The receipted strict-open-horizon feature cohort has SHA-256
`66cdb1a9599e342b31a79cdcc8601371c86f84ebb8da3a9a907f9560ad0fb920` and 512,567
contracts. The verified lazy/materialized adapter produced:

| Measure | Result |
|---|---:|
| Natural `(ticker, quote_date, expiry_date)` states | 74,215 |
| Mean strikes per state | 6.9065 |
| Minimum / maximum strikes | 1 / 38 |
| Lazy projection and collection | 0.0987 s |
| Exact group validation and native preparation | 0.0536 s |
| Total preparation | 0.1524 s |
| Prepared group identity | `f047db96a5e787d09428a77a242869434cc15e0f2f2876f984bc8d1d45178640` |

No duplicate strike or nonconstant simulated field was observed. The fresh preparation process
reported approximately 552 MB maximum resident set size while holding both the materialized Polars
frame and compact native buffers.

## Pricing A/B

A contiguous 512-state prefix retained 1,188 contracts (2.32 strikes per state) and priced all four
reconstructed dividend variants. Both paths used 150 intervals, 500 training paths, 500 disjoint
evaluation paths, degree-four ITM regression, the same seeds/domain roots, 14 threads, and the
explicit Apple vForce backend.

| Execution | Wall time | Contract-variants/s |
|---|---:|---:|
| Group-addressed common paths | 1.7095 s | 2,779.8 |
| Row-addressed independent paths | 2.4293 s | 1,956.1 |

The grouped path was 1.421 times as fast end to end. Its combined price SHA-256 was
`7ff743946d943513ab2be2924bcdd9a236ea18a21920753e24319b904758b7c6`; the row-addressed
price SHA-256 was `ef1a3c4be5e42be2bc288cd6f9715513958a5e40506ee3281392be083a5f1321`.
Different hashes are expected because random domains are assigned per state rather than per strike.

Both requests preserved the same qualitative ordering: the two cash representations outperformed
forecast yield, which outperformed historical yield on this small prefix. That observation is only
an integration check; finite-path differences and the nonrepresentative prefix preclude empirical
interpretation.

