# ADR 0042: Test an explicit terminal-cash predecessor exercise domain

- Status: accepted; diagnostic passed
- Date: 2026-07-15

## Context

The retained degree-zero-through-four diagnostic shows stable downward LSM policy bias for every
supported global polynomial basis. Degree four remains closest to direct quadrature, but selecting
it for that reason would tune the estimator to one synthetic case rather than explain the error.
The complete results are recorded in
[`../performance/2026-07-15-terminal-cash-polynomial-diagnostic.md`](../performance/2026-07-15-terminal-cash-polynomial-diagnostic.md).

For a fixed-strike call with zero proportional yield and a nonnegative rate, exercise between
deterministic cash events is dominated by continuation. At any time before the next cash-event
decision, holding at least a European call to that decision gives
`C >= S - K exp(-r delta) >= S - K`. This is the interval-wise no-early-exercise argument from
Merton's [theory of rational option pricing](https://doi.org/10.2307/3003143). The single-dividend
American-call literature likewise locates the decision just before the ex-dividend instant; see
Whaley's [corrected known-dividend valuation](https://doi.org/10.1016/0304-405X(81)90013-1).

The synthetic terminal event deliberately represents an ex-date and expiry collapsed to the same
date. That does not create a same-instant exercise phase: the stock becomes ex-dividend before the
expiry observation. The Options Industry Council states that dividend entitlement requires call
exercise before the ex-dividend date. A real-data implementation must therefore receive the prior
eligible trading decision from explicit contract and market-calendar data.

Engine `e7dfbb0216456f51fe7b7857a4dc50b4a827ebce` permanently checks the exact synthetic
geometry. Direct quadrature with all 256 pre-expiry exercise nodes agrees within `0.002` both with
the same grid restricted to its final predecessor node and with an independent 16,384-step
interpolated CRR tree.

## Decision

Add one pilot-only `terminal_cash_predecessor` exercise domain. It fails validation unless the case
has deterministic cash, zero proportional yield, a nonnegative rate, and a final cash event exactly
at maturity. For each declared base mesh, it retains every transition time but admits exercise only
at strictly pre-expiry cash-event nodes and at the final base node before the terminal event. It
does not shift the terminal cash event or create an exercise phase at expiry.

Run one diagnostic that changes only this domain from the degree-four case. It retains the exact v3
terminal-case p3/p4 fitting and independent evaluation path prefixes, 12 seeds, random domains,
quadrature sequence, 16/32/64/128/256 mesh sequence, regression controls, strict support and tie
rules, tolerances, gates, and four-thread portable execution. Its frozen specification SHA-256 is
`f4700ffe31903ff810b2e773efe286b2f2850e6ec316ac469b471a57480e494d`.

The diagnostic receipt is retained whether it passes or fails. A pass would support a production
API that requires an explicit row-level terminal predecessor time; it would not authorize the
engine to infer a business day, subtract an epsilon, or silently select this domain. A failure would
send the estimator search to a different continuation-policy family without widening any gate.

## Outcome

The diagnostic passed all eight predeclared gates. The p4 mean/reference gap fell from
`0.1308468495` on the 255-regression policy to `0.0304441281` on the one-regression predecessor
policy, while the deterministic value remained bit-identical at `11.3240479650`. The p3-to-p4
paired drift was `0.0067862958`, well inside its `0.1228588531` limit. Complete retained evidence is
in
[`../performance/2026-07-15-terminal-cash-predecessor-diagnostic.md`](../performance/2026-07-15-terminal-cash-predecessor-diagnostic.md).

This resolves the synthetic terminal-cash policy bias without changing a basis, path population,
gate, event time, or terminal event order. Production work may now add a generic exact exercise-time
input and let the study layer provide the predecessor from explicit calendar semantics. The engine
still must not infer that time.
