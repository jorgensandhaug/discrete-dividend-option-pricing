# ADR 0043: Apply the passing terminal predecessor domain to the full pilot

- Status: accepted for pilot execution
- Date: 2026-07-15

## Context

V3 passed 60 of 61 frozen gates. Its sole miss was stable downward LSM policy bias for terminal
cash under 255 pre-expiry exercise regressions. Increasing paths and testing every supported global
polynomial degree did not remove the bias. The single-change predecessor diagnostic then retained
the exact deterministic value and passed all eight case gates with one economically relevant
regression. The complete causal sequence is retained in ADRs 0040 through 0042.

That case-only pass is not yet evidence that one complete publication panel and its shared method
identity pass together. A new full-panel version must be frozen before execution.

## Decision

Create target-free pilot v4 by changing only `terminal_cash_atm.exercise_domain` from
`uniform_bermudan` to `terminal_cash_predecessor`. Preserve v3's other seven cases, case order,
q1/q2/q3 quadrature sequence, 16/32/64/128/256 general mesh sequence, p1/p2/p3/p4 nested path
sequence, 12 seeds and case-addressed random domains, degree-four regression, rank and condition
limits, independent fitting/evaluation populations, strict support and exercise rules, all
tolerances, and all gates.

The terminal case still retains every mesh transition and its terminal cash time. Only its exercise
nodes change to strictly pre-expiry cash events plus the final declared predecessor. V4 therefore
does not move the cash event, alter terminal cash-before-payoff ordering, change a stochastic path,
or relax a numerical envelope.

The complete specification SHA-256 is
`344670190d05856d90e0ab3b2d9a084a4de32cb1c3e68bc224d943edc21e0c32`, frozen in tests before
execution. Run it with four threads and portable Rust `f64`; retain the non-overwriting receipt
whether it passes or fails. A pass validates the synthetic numerical panel, not an empirical model
ranking and not a real-data calendar mapping.
