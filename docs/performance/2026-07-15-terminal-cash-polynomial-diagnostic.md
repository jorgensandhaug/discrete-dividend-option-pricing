# Terminal-cash polynomial-policy diagnostic

- Date: 2026-07-15
- Paper revision: `01b1abf`
- Engine revision: `7fcbf1ac2163daf59118c3a6e0e7b6b20b160c3a`
- Runtime: CPython 3.14.6, Rust 1.97.0, four threads, portable Rust `f64`, Apple M4 Pro
- Outcome: all five diagnostics failed the inherited LSM/reference envelope

Every row below used the exact v3 terminal-cash case and 256-opportunity schedule, q1/q2/q3
deterministic sequence, p3/p4 nested paths, 12 randomized replications, random domains, strict
support and exercise rules, tolerances, and gates. Only the global polynomial degree changed. The
deterministic q3 reference was `11.3240479650` in every run.

| Degree | p3 mean | p4 mean | p4/reference gap | Gate limit | Wall seconds |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.3080811007 | 8.3127650354 | 3.0112829296 | 0.0928598098 | 60.454 |
| 1 | 11.1086013872 | 11.1233064017 | 0.2007415633 | 0.1157595829 | 60.998 |
| 2 | 11.1006249650 | 11.1038898790 | 0.2201580860 | 0.1101551199 | 63.143 |
| 3 | 11.1678886012 | 11.1309398879 | 0.1931080771 | 0.1206937552 | 59.176 |
| 4 | 11.1932375450 | 11.1932011155 | 0.1308468495 | 0.1185747552 | 56.340 |

The degree-four result is numerically closest, but no supported global polynomial removes the
population-stable downward policy bias. Selecting the closest degree would tune the method to this
case and is rejected. The evidence instead motivates changing the exercise-policy architecture.

The receipt SHA-256 values for degrees zero through four are, respectively:

- `aabfd65c031762e0dfe3e0a6a86d9fc0c6768913db61f1a6b037f63c24989de2`
- `193a8e5fd3ca9968f6b1d9745380bbea014f1c961361cf84547e19c974f52455`
- `a31d431cbb443146c63595c2682c0044cdb2a276d0d40579ebec4dfbba2888d0`
- `d75b6b0864a413cf41c9b92e1a4d288242c0d29da1d2b48119258feb9f78497a`
- `4c2c10da22dfbb2241d0a295874b253407d2275e7c3d51dedaf88286b63beb30`
