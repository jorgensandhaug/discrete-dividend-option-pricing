# ADR 0047: Deterministic publication pricing for continuous-yield variants

- Status: accepted
- Date: 2026-07-15
- Engine revision: `cd14031a8b1cd03b8258c73c739155a4161f88bc`
- Supersedes: the continuous-yield routing in ADR 0036 and ADR 0046

## Context

The forecast-yield and historical-yield variants assume one constant proportional dividend yield,
constant volatility, and a constant continuously compounded rate over each option horizon. Their
American calls therefore do not require simulated paths or a fitted stopping policy. American
call-put symmetry transforms each call into a put, and normalizing the transformed put strike makes
one exercise boundary reusable by every strike sharing the same natural market state.

The paper must retain American exercise, cover the recovered cohort's nonnegative yields and both
positive and negative rates, expose numerical diagnostics, and fail rather than substitute another
method. The numerical profile must also be selected without consulting observed option prices or
model rankings.

## Decision

Use the engine's grouped `AndersenLake` method for both continuous-yield variants. The paper owns
the immutable `PROJECT_PUBLICATION_YIELD_BOUNDARY_SPEC`:

- boundary Gauss--Legendre order 13;
- exactly 16 ordinary fixed-point-B iterations;
- 6 positive-time Chebyshev collocation nodes plus the exact right limit at zero;
- price Gauss--Legendre order 21;
- maximum relative boundary residual `1e-4`; and
- maximum relative Snell-obstacle projection `1e-6` of spot.

The controls are required parts of the separately hashed study identity. There is no engine or
host-selected default, adaptive refinement, retry, alternate accelerator, LSM fallback, boundary
clipping, or target-dependent selection. Unsupported yields, numerical-domain violations,
excessive residuals or projections, and financial-bound failures abort the named variant and make
the complete replication fail.

For zero-yield calls with nonnegative rates, the exact no-early-exercise theorem routes directly to
the European Black--Scholes formula. Every other admitted state uses call-put symmetry and the
single-boundary solver. The exact stopping payoff applies inside the solved exercise region. The
solver returns origin-exercise decisions per contract and boundary residual, obstacle projection,
and theorem-route diagnostics per natural state.

The existing exhaustive rate partition remains the cash-dividend method boundary. Yield pricing
may traverse the same two immutable partitions for source-order composition, but it does not
change method at that boundary. Both pieces receive the same deterministic method, and their row-
and group-aligned arrays are restored through the verified partition maps. Random streams and LSM
path budgets now belong only to the two cash-dividend variants.

## Target-free evidence

The selected profile was evaluated on all 96,666 natural states and 678,122 contracts in the exact
recoverable submitted-window union without reading option-price targets. Against the separately
declared over-resolved profile `(81, 48, 32, 131, 1e-8, 1e-8)`, maximum price differences were
`$0.0004012` for forecast yields and `$0.0003998` for historical yields. RMSE was approximately
`$0.000035` for both, and no contract differed by `$0.001`.

On eight Apple M4 Pro threads, direct pricing took 0.549 seconds for forecast yields and 0.880
seconds for historical yields in the final integration run. The paper's partitioned route took
0.549 and 0.876 seconds, respectively. Prices, origin decisions, residuals, obstacle projections,
and theorem-route flags were bitwise identical to the corresponding direct full-group requests.
The largest observed residuals were `1.298e-5` and `1.504e-6`; the largest obstacle projections
were `2.622e-7` and `3.016e-7`.

The aggregate receipt is recorded in
[`../performance/2026-07-15-deterministic-publication-yields.md`](../performance/2026-07-15-deterministic-publication-yields.md).
The engine's mathematical construction, independent test evidence, and interpolation choices are
recorded in [engine ADR 0019](https://github.com/jorgensandhaug/finoptions-engine/blob/cd14031a8b1cd03b8258c73c739155a4161f88bc/docs/adr/0019-grouped-continuous-yield-american-calls.md).

## Consequences

- Both yield variants are deterministic and have no Monte Carlo estimator uncertainty.
- One boundary solve serves all strikes in a natural state, preserving the intended grouped fast
  path and making full-cohort iteration subsecond per yield representation on the measured host.
- Cash-dividend variants retain their separately declared LSM methods and random identities.
- Publication inference must distinguish deterministic yield output from replication-averaged cash
  estimates while comparing all four variants on the same recovered rows.
- A future change to any boundary control is a new study identity and requires retained target-free
  evidence; runtime self-tuning is not permitted.

## Sources

- Andersen, Lake, and Offengenden (2016), <https://doi.org/10.21314/JCF.2016.312>
- Healy (2021), <https://arxiv.org/abs/2109.15157>
- Fritsch and Butland (1984), <https://doi.org/10.1137/0905021>
