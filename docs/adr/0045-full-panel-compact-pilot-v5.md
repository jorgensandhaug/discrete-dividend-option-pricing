# ADR 0045: Full-panel compact exact-schedule pilot v5

- Status: accepted before execution
- Date: 2026-07-15

## Context

Full-panel v4 passed all 61 target-free gates, using one explicitly declared pre-terminal-cash
exercise opportunity for the terminal case while retaining a redundant 256-node transition grid.
The separately frozen compact diagnostic in ADR 0044 removed only those transition-only nodes. It
passed all seven inherited applicable gates and reduced isolated end-to-end wall time by about 143x.

That diagnostic is necessary but not sufficient to replace the first complete passing baseline.
The full panel must preserve the other cases and confirm their unchanged results under a new,
immutable study identity.

## Decision

Create `project-publication-target-free-v5` by changing only `terminal_cash_atm.exercise_domain`
from `terminal_cash_predecessor` to `compact_terminal_cash_predecessor`. Retain v4's other seven
cases and case order, q1--q3 quadrature sequence, general mesh sequence, p1--p4 nested path sequence,
12 replications, random identities, degree-four policy, diagnostics and numerical tolerances.

The terminal case has seven rather than eight gates because its compact schedule is an exact finite
set, not a mesh-refinement sequence. All general Bermudan cases retain their mesh convergence gate.
The pre-execution v5 specification SHA-256 is
`8dff50a6cef9c93271c745f2f6cb784552516d739d3e109884b451378493ce64`.

Execute the complete portable four-worker panel once and retain every level and replication in a
non-overwriting receipt. A pass makes v5 the numerical baseline for the exact-schedule publication
estimator. A miss remains evidence and does not authorize tuning this frozen specification.
