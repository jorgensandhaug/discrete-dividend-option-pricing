# ADR 0003: Immutable study, execution, and artifact identity

Status: accepted

A versioned study specification defines data availability, cohort, filters, features, financial
model, estimator, comparisons, and metrics. A separate execution specification defines backend,
threads/device, memory, batching, and artifact location. Validation resolves both into one immutable
canonical manifest containing dataset/schema hashes, adapter/engine/paper revisions, dependency
locks, hardware identity, and every value capable of changing the result.

Artifact locations derive from relevant manifest hashes. Production code never selects a wildcard
or “latest” result. Any change to a financial event, path set, seed, basis, solver, precision, or
estimator configuration creates a different result identity rather than an execution override.

