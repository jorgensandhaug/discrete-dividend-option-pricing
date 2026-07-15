# ADR 0044: Compact explicit terminal-predecessor diagnostic

- Status: accepted before execution
- Date: 2026-07-15

## Context

Pilot v4 passed every predeclared deterministic and stochastic gate after limiting the synthetic
terminal-cash call to its prior eligible decision at `255/256` years. Its scalar diagnostic schedule
retained the other 254 uniform points as transition-only nodes. The production engine's exact
exercise-time batch API intentionally constructs the minimal union of declared exercise times, cash
events, origin and expiry. Exact GBM transitions make the redundant and compact grids equal in law,
but removing nodes changes finite-sample Philox normal addressing and therefore requires a frozen
numerical check.

The engine never derives a predecessor. For this synthetic diagnostic, `255/256` is the declared
prior eligible time. A future empirical adapter must provide each predecessor from its trading
calendar and contract timestamps.

## Decision

Add `compact_terminal_cash_predecessor` as a pilot-only schedule identity. It constructs exactly
`[0, 255/256, 1]` for the existing one-year terminal-cash case, exercises only at `255/256`, applies
the terminal cash before expiry payoff, and retains the predecessor diagnostic's cases, q1--q3
quadrature levels, p3--p4 nested path levels, 12 seeds, random domains, degree-four policy and all
acceptance tolerances. There is no mesh-refinement gate because this is one exact finite schedule,
not a Bermudan mesh approximation.

The pre-execution spec hash is
`e2aa1fd264a7521070fa7d693cacb668da4c2f4d486dae0ef146e6a4b3d9ae4f`. Engine revision
`818691e6f7638e6bd4fa781c52302792636717ae` separately verifies explicit schedule construction,
bitwise parity with identical uniform and cash-specialized batch schedules, and rejection of invalid
times before simulation. The scalar diagnostic and batch route use the same event-plan and LSM
kernels after schedule construction.

Run the diagnostic before changing the full eight-case pilot. A pass authorizes freezing a v5 that
changes only the terminal case from the redundant predecessor domain to the compact domain; a miss
is retained and investigated without tuning this diagnostic.
