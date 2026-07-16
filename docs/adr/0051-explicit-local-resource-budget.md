# ADR 0051: Explicit local resource budget and phase isolation

- Status: accepted
- Date: 2026-07-16

## Context

The complete project reconstruction contains hundreds of independent LSM calls and 585 rolling
neural fits. Content addressing makes these units safely resumable, but it does not make arbitrary
local parallelism desirable. Launching many LSM processes, CPU trainers and MLX contexts together
duplicates the recovered feature population, framework runtimes, simulation workspaces and unified
GPU allocations. Aggregate throughput can improve while the interactive workstation becomes
unusable and starts compressing or swapping memory.

Financial execution specifications answer a different question. Path counts, random streams,
backend, precision and estimator thread count can be part of a result identity; a workstation
budget determines when and alongside what that request may execute. The latter must not silently
rewrite the estimator to fit the machine.

## Decision

The interactive Apple-Silicon workflow admits exactly one heavy process at a time. Its declared
budget is `project_interactive_one_phase_v1`:

- one heavy model or simulation process;
- at most eight explicitly requested estimator threads;
- four Polars/data-preparation threads;
- two hidden BLAS, OpenMP, NumExpr and Accelerate threads;
- OS nice increment 10; and
- no automatic restart.

LSM completion, LSTM training and each MLX model family are separate phases. A neural phase may not
start until the faithful-LSM completion manifest exists. Completed content-addressed units remain
resumable, but a failed phase stops the queue; it is never replaced by a different backend, worker
count, batch size or model.

`LocalResourceBudget` owns a path-free manifest and the exact process thread environment. The
installed console entry resolves its resource request without importing the study package, acquires
`acquire_local_heavy_process_lease`, installs all thread-pool variables, applies the nice increment,
and only then imports the numerical CLI. Requested `--threads` and `--data-threads` values above the
budget fail before that import; the launcher never rewrites them. Lease acquisition is nonblocking:
overlap fails immediately instead of waiting, retrying, or choosing another execution profile. The
lease requires an absolute regular non-symlink file on an existing local directory. Cluster and
dedicated-machine schedules require distinct named budgets and admission controls; they must not
reuse the interactive identifier with higher concurrency.

The four heavy study CLI commands acquire this lease before importing numerical dependencies,
reading the feature artifact, or initializing a model. A tiny separately packaged bootstrap exists
because importing the research package intentionally exposes a broad typed API and therefore cannot
serve as a pre-import boundary. By default commands share a user-specific file in the operating
system's temporary directory, so separate checkouts still contend on one local admission point.
Advanced orchestrators may pass one absolute `--local-resource-lock`; changing that path does not
change financial or artifact identity. Direct Python callers must install the same environment and
hold the same context manager before importing numerical modules.

## Consequences

- The workstation remains usable while long reproduction phases progress.
- Only one copy of the prepared population and one framework/device context is live per phase.
- Resource policy is inspectable and testable instead of being encoded in tmux sessions or agent
  behavior.
- A second local heavy phase is rejected before framework or data initialization.
- Excess estimator or data concurrency is rejected rather than reduced behind the caller's back.
- The local schedule may take longer than an aggressively sharded run; cluster throughput remains a
  separate, explicitly declared optimization.
- The budget does not claim to enforce a portable RSS percentage. Memory is measured per workload,
  and a materially larger live footprint requires a new benchmark and budget decision.
