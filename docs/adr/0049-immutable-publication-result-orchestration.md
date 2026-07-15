# ADR 0049: Immutable publication-result orchestration

- Status: accepted; first production artifact pending
- Date: 2026-07-15
- Engine revision: `cd14031a8b1cd03b8258c73c739155a4161f88bc`

## Context

The frozen publication study requires 12 expensive paired cash-LSM replications and two fast,
deterministic continuous-yield results. A local run may be interrupted, but the legacy repository's
mutable `raw.parquet` and `processed.parquet` caches are scientifically unsafe: their filenames do
not prove which source, code, estimator, random namespace, or execution configuration produced
their contents. Recomputing deterministic yield prices for every stochastic seed also wastes work
and incorrectly suggests that the yield methods have Monte Carlo uncertainty.

The final numerical intervals have a narrow interpretation. Repeating the independently seeded
cash estimator measures numerical Monte Carlo variability conditional on the fixed observed
cohort. It does not measure dependence among option rows, time-series sampling uncertainty, or
out-of-sample generalization uncertainty.

## Decision

Each seed produces one cash-only safe-tensors artifact containing the target once, both cash
variants' complete native LSM diagnostics, and their project metrics. Its canonical manifest binds:

- the verified feature artifact, feature schema, dependency lock, paper revision, and engine
  revision;
- the publication-study and replication-plan hashes;
- the exact seed, random-domain roots, thread count, and CPU math backend;
- the complete partition hash and exact/general group counts;
- paper, engine, and Rust compiler versions; and
- independent hashes of every result tensor and metric payload.

The file is named by the SHA-256 digest of its complete bytes. A request-addressed receipt is
written once through an atomic hard link and contains only the exact request hash and artifact
content hash. A missing receipt means the request is not complete. An existing malformed,
conflicting, renamed, or unverifiable receipt is fatal; it never causes a retry, alternate seed, or
cache substitution.

After all 12 canonical replication indices exist, finalization streams them in order and keeps only
two Float64 Welford accumulators per cash variant. It requires one shared execution profile and
verifies every source, partition, target, study, plan, replication index, request hash, and content
hash. It then prices each deterministic yield variant exactly once under a separate explicit
deterministic CPU execution identity.

The final content-addressed artifact contains:

- the target vector;
- cash replication-mean price vectors and per-row sample standard deviations;
- every individual cash replication metric, metrics evaluated on the mean prediction, and
  two-sided 99% Student-t intervals for each metric's Monte Carlo replication mean;
- both deterministic yield price vectors and complete native boundary diagnostics; and
- the ordered content, request, and execution hashes of all 12 prerequisite cash artifacts.

The loader independently recomputes metrics from stored price vectors and recomputes every Monte
Carlo metric summary from the stored per-replication metrics. A successful subset can exist as
verified cash artifacts, but only the complete final artifact is a publication-study result.

## Consequences

- An interrupted run resumes only exact verified requests and retains completed expensive seeds.
- No mutable `latest`, filename convention, timestamp, or notebook state can authorize reuse.
- Deterministic yield methods carry no fabricated replication uncertainty.
- The final artifact is compact: it references full per-seed diagnostics rather than duplicating
  them, while retaining the cash mean and dispersion needed for analysis.
- The reported Student-t intervals must always be labelled Monte Carlo numerical uncertainty. Date-
  blocked or cluster-aware empirical inference remains a separate analysis layer under ADR 0036.
