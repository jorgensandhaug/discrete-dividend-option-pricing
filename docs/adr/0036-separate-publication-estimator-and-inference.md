# ADR 0036: Separate publication estimator, convergence, and inference

- Status: accepted architecture; numerical budgets remain evidence-gated
- Date: 2026-07-15

## Context

The project reconstruction identifies the submitted comparison and its qualitative result. It is
not a publication estimator: it intentionally retains a finite 150-interval exercise mesh,
degree-four regression, 500 fitting paths, 500 evaluation paths, and the submitted sparse-support
continuation rule. Reusing that identity for a paper would confound historical reconstruction,
numerical approximation, randomized estimator uncertainty, and empirical uncertainty across
dependent option quotes.

The publication workflow also has to remain practical on limited hardware. Performance work may
specialize a financial method only when the specialization's theorem domain is represented by a
typed request and validated before execution. A faster kernel is not allowed to change exercise,
cash-event, discounting, volatility, or statistical semantics silently.

## Decision

Create a separately named and separately hashed publication study. It never changes
`PROJECT_CLASSICAL_RECONSTRUCTION_SPEC` or interprets a reconstruction artifact as publication
inference.

The publication workflow has four independently auditable layers.

### 1. Financial estimator

- Fit each LSM stopping policy and value it on disjoint random populations.
- Include the valuation-origin exercise decision whenever the declared opportunity set admits it.
- Select `InsufficientExerciseSupportPolicy.FAIL`. Insufficient rows, absent support entered by an
  evaluation path, rank deficiency, excessive condition number, non-finite arithmetic, invalid
  event order, or an unsupported financial domain fails the run. No degree reduction,
  regularization, support expansion, event movement, backend change, or result clipping is allowed.
- Preserve common random numbers across strikes and variants that share an identical time mesh,
  while retaining separately fitted strike policies. Replications remain seed-paired across
  different mesh identities, but that pairing is not described as pathwise-identical Brownian
  increments: consuming one counter stream on different time grids does not establish that claim.
  A stronger cross-mesh coupling requires its own named random construction and validation.
- Treat the independently evaluated policy value as a randomized lower-bound estimator for the
  declared finite opportunity set. It is not labeled an exact American-option price.

An optimized opportunity set may be added only as a new typed method. For example, a deterministic
cash-dividend call can use pre-dividend exercise opportunities only after validating every theorem
condition, including the rate domain and left-limit ordering at a terminal cash event. Rows outside
that domain use a separately declared general method; they are never sent through an implicit
fallback. Closed-form European reductions receive the same requirement.

### 2. Numerical design

Path counts, exercise meshes, polynomial basis, and randomized-replication count are not hard-coded
from the submitted configuration and are not chosen from empirical model rankings. A deterministic
stress panel is selected without reading market targets. It spans the admitted input domain,
includes published and analytic hand cases, and overweights boundaries such as short maturity,
near-the-money states, cash events near valuation or expiry, and the minimum admitted rate.

Candidate budgets form predeclared nested sequences. Common random numbers compare adjacent path
budgets; nested exercise sets compare adjacent meshes. A candidate becomes admissible only if every
declared absolute/relative price-stability limit, regression diagnostic, sampling-error limit, and
independent-reference check passes for every stress stratum. Failure stops selection. The accepted
budget and the complete pilot receipt become immutable inputs to the production request; no
production runner searches for a cheaper passing setting.

At least one independent deterministic implementation validates the one-dimensional cash-dividend
slice on the stress panel through analytic identities, financial bounds, hand cases, and
predeclared cross-method convergence. Agreement is an evidence envelope, not an assertion that one
implementation is an oracle.

### 3. Randomized-estimator uncertainty

Run a predeclared number of independent, role-separated fitting/evaluation replications. Each
replication is paired across the four dividend variants and uses nonoverlapping random namespaces.
The publication prediction is the arithmetic mean of the complete replication vector for each
contract. Between-replication variability, per-replication native standard errors, convergence
diagnostics, and seed identities are persisted separately. A missing or failed replication makes
the complete study incomplete; successful partial output is not a publication result.

Randomized quasi-Monte Carlo, Brownian-bridge construction, antithetic sampling, or a control
variate may replace the current Philox construction only under a separately named random-method
identity and a measured variance/time gate. No random method is selected automatically by device or
workload.

### 4. Empirical model comparison

Compute row-level paired loss differences from the completed replication-mean predictions. Retain
the submitted MAPE and RMSE for historical comparison, but report MAE, bias, relative bias, and
SMAPE as diagnostics so no conclusion rests on one low-price-sensitive metric.

Empirical uncertainty resamples complete quote-date panels with a stationary date-block bootstrap.
This preserves same-date cross-sectional dependence and serial dependence in adjacent trading
dates without pretending option rows are independent or relying on six ticker clusters for an
asymptotic approximation. Block-length selection, bootstrap seed, replication count, confidence
level, loss functions, pairwise contrasts, and multiplicity procedure are fixed in a hashed
inference specification before inspecting inferential results. The recovered cohort is analyzed as
declared, with no imputation of unavailable licensed rows; cohort sensitivities receive distinct
identities.

## Consequences

- Historical reconstruction, numerical validation, Monte Carlo uncertainty, and empirical
  inference cannot be mixed in one result identity.
- A performance optimization must either be exactly semantics-preserving or become a new typed
  financial/random method with its own evidence gate.
- Numerical pilot outcomes can determine compute budgets under frozen tolerances, but cannot select
  a dividend representation, metric, empirical cohort, or favorable conclusion.
- The public paper repository can document the complete method without containing licensed rows,
  machine-local paths, private audit findings, or unpublished result archaeology.
- Publication execution remains blocked until the strict estimator, convergence receipt, paired
  replication artifact, and date-block inference artifact are implemented and verified.

## Sources

- [Longstaff and Schwartz (2001), least-squares Monte Carlo](https://doi.org/10.1093/rfs/14.1.113)
- [Clément, Lamberton, and Protter (2002), LSM convergence](https://doi.org/10.1007/s007800200071)
- [Glasserman and Yu (2004), paths versus basis functions](https://doi.org/10.1214/105051604000000846)
- [Andersen and Broadie (2004), primal-dual bounds](https://doi.org/10.1287/mnsc.1040.0258)
- [Dion and L'Ecuyer (2010), randomized QMC for American options](https://www.informs-sim.org/wsc10papers/250.pdf)
- [Politis and Romano (1994), stationary bootstrap](https://doi.org/10.1080/01621459.1994.10476870)
- [Hansen, Lunde, and Nason (2011), model confidence sets](https://doi.org/10.3982/ECTA5771)
