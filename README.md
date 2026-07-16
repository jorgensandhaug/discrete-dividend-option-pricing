# Discrete-dividend option pricing

A reproducible implementation of the project-thesis study on learning American call-option prices
from forecast discrete cash dividends. The central experiment asks whether preserving the timing and
size of individual dividend events improves neural pricing relative to replacing the schedule with a
single continuous-yield feature.

The repository contains the licensed-data adapters, point-in-time feature pipeline, five frozen
neural models, rolling evaluation, immutable result artifacts, and aggregate-only paper tables.
Generic pricing kernels live in the sibling `finoptions-engine` repository at an exact pinned
revision.

This repository is private and pre-release. Raw or derived licensed rows, targets, predictions,
credentials, and machine-local paths are never committed.

## Study

All models use the same market inputs, target, rolling windows, optimizer, and scalar network. They
differ only in how dividends enter the model:

| Model | Dividend representation |
|---|---|
| Direct cash dividends (LSTM) | Ordered event sequence with exact logical length |
| Direct cash dividends (attention) | Fixed-capacity learned event attention |
| Direct cash dividends (12 bins) | Fixed maturity-bucket vector |
| Forecast continuous yield | One scalar derived from forecast dividends |
| Historical continuous yield | One trailing realized-dividend scalar |

The submitted schedule contains 117 rolling experiments. Each window uses a three-year lookback,
reserves the final 30 days for validation, evaluates on the following 30 days, and advances by 30
days. Metrics are pooled over observations after every window completes; window RMSE values are
never averaged.

The authoritative submitted framework-neutral identity—including event order, slot policy,
features, architecture, initialization, loss, optimizer, scheduler, batching, and early stopping—is
defined as `PROJECT_AS_SUBMITTED_NEURAL_STUDY_SPEC` in
[`neural_spec.py`](src/discrete_dividend_study/neural_spec.py). PyTorch and MLX implement that one
identity and are tested against the same canonical parameters, forward outputs, and gradients.

## Quick start

Check out `finoptions-engine` beside this repository. Prerequisites are `uv`, Rust 1.97.0, and
`just`.

```shell
just test
just check
just full
```

The development launcher synchronizes the locked environment and rebuilds the native sibling only
when its inputs change. Ruff and ty define the Python quality boundary; Clippy and Rust tests are
owned by the engine repository.

## Reproduce the neural study

First materialize one verified project feature artifact from licensed inputs. The provider adapters,
cleaning stages, support-data contracts, and exact commands are documented in the
[workflow reference](docs/workflows.md). No notebook defines a production transformation.

Run or exactly resume all five models and all 117 rolling windows:

```shell
uv run --locked discrete-dividend-study run-neural-study \
  --features /private/features/project_experiment_features.parquet \
  --feature-artifact-kind project-experiment \
  --artifacts /private/artifacts/neural \
  --profile apple-silicon \
  --data-threads 4 \
  --parameter-root-seed 2026071401 \
  --shuffle-root-seed 2026071402
```

The command validates the complete schedule and every runtime before training. Each model/window
result is immutable and content-addressed, so an interrupted run resumes only when its exact request
receipt matches. A changed source, code revision, dependency lock, model specification, seed,
backend, or rolling phase is a different request.

Render a paper table only from the verified complete 585-run study manifest:

```shell
uv run --locked discrete-dividend-study render-neural-table \
  --artifact /private/artifacts/neural/studies/<sha256>.study.json \
  --format markdown
```

The renderer rejects partial schedules and artifacts produced by another model specification. Its
output contains aggregate metrics and the source artifact hash, never licensed observations or
machine-local paths.

## Performance and local resources

The Apple-Silicon profile uses the explicitly selected native backend for each model. The portable
CPU profile is separately named; a missing backend is an error and never triggers substitution.
Project-shaped synthetic benchmarks measure preparation, training, inference, throughput, and peak
memory without provider data.

Interactive heavy work uses one process at a time, at most eight estimator threads, four data
threads, and two hidden threads per numerical pool. The installed launcher obtains the shared lease
and applies the budget before importing Polars, NumPy, the engine, PyTorch, or MLX. Overlapping or
over-budget work fails immediately. A cluster or dedicated host must declare a different execution
profile rather than changing the scientific model.

## Classical benchmarks

The repository also reconstructs the project’s four classical pricing variants and contains a
separate, stricter cash-dividend LSM extension with independent fitting and valuation paths,
deterministic continuous-yield comparators, and date-panel inference. These are benchmark and
follow-on analyses; they are not the project paper’s primary contribution.

Current aggregate results for that extension are retained in
[`results`](results/README.md). They do not replace the five-model neural result table.

## Repository map

- `src/discrete_dividend_study`: typed data adapters, feature construction, neural models,
  orchestration, artifacts, metrics, and table renderers.
- `config`: the exact sibling-engine revision.
- `docs/workflows.md`: complete licensed-data and experiment commands.
- `docs/adr`: financial, numerical, data, and execution contracts.
- `benchmarks`: provider-free performance harnesses.
- `results`: aggregate-only follow-on classical benchmark output.
- `tools/dev.py`: locked, state-aware local development entry point.

## Scientific contract

- Exercise, dividend, maturity, curve, day-count, source-precision, event-order, and padding
  semantics are explicit.
- Unsupported financial, numerical, schema, coverage, or runtime domains fail; there are no silent
  fallbacks, imputations, clipping rules, or backend substitutions.
- Training, validation, and test phases are disjoint and content-hashed.
- Model identity is framework-neutral; execution and hardware identity are recorded separately.
- Licensed data remain local while aggregate tables retain cryptographic provenance.
- Legacy numbers are qualitative reconstruction targets, not an exact-digit acceptance test.

Later TabPFN experiments remain an isolated experiment family. They do not alter the submitted
five-model comparison or appear in its result artifact.
