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
- two Polars/data-preparation threads;
- one hidden BLAS, OpenMP, NumExpr and Accelerate thread;
- OS nice increment 10; and
- no automatic restart.

LSM completion, LSTM training and each MLX model family are separate phases. A neural phase may not
start until the faithful-LSM completion manifest exists. Completed content-addressed units remain
resumable, but a failed phase stops the queue; it is never replaced by a different backend, worker
count, batch size or model.

`LocalResourceBudget` owns a path-free manifest and the exact child thread environment. Launchers
must hold `acquire_local_heavy_process_lease` across the complete heavy child lifetime and record
the budget manifest in their operational receipt. Lease acquisition is nonblocking: overlap fails
immediately instead of waiting, retrying, or choosing another execution profile. The lease requires
an absolute regular non-symlink file on an existing local directory. Cluster and dedicated-machine
schedules require distinct named budgets and admission controls; they must not reuse the interactive
identifier with higher concurrency.

The four heavy study CLI commands acquire this lease before reading the feature artifact or
initializing a model. By default they share a user-specific file in the operating system's temporary
directory, so separate checkouts still contend on one local admission point. Advanced orchestrators
may pass one absolute `--local-resource-lock`; changing that path does not change financial or
artifact identity. Direct Python callers must hold the same context manager across their complete
heavy phase.

## Consequences

- The workstation remains usable while long reproduction phases progress.
- Only one copy of the prepared population and one framework/device context is live per phase.
- Resource policy is inspectable and testable instead of being encoded in tmux sessions or agent
  behavior.
- A second local heavy phase is rejected before framework or data initialization.
- The local schedule may take longer than an aggressively sharded run; cluster throughput remains a
  separate, explicitly declared optimization.
- The budget does not claim to enforce a portable RSS percentage. Memory is measured per workload,
  and a materially larger live footprint requires a new benchmark and budget decision.
