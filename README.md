# Discrete-dividend option pricing

A reproducible reconstruction and publication extension of the project-thesis comparison between
explicit cash-dividend and continuous-yield option-pricing models. The repository owns licensed
data adapters, point-in-time feature construction, frozen study specifications, immutable result
artifacts, empirical inference, and publication tables. Generic pricing kernels live in the
private sibling `finoptions-engine` at an exact pinned revision.

This repository is private and pre-release. Raw or derived licensed rows, targets, predictions,
credentials, and machine-local paths are never committed.

## Result

On 678,122 submitted-test observations across 2,400 quote-date panels, both explicit cash models
have lower MAPE than both continuous-yield approximations. The cash-jump model also has lower RMSE
than both yield models after the predeclared Holm correction. The liquidator and spot-matched
escrowed-cash conventions are not statistically distinguishable from each other.

The exact aggregate estimates, confidence intervals, paired tests, and LaTeX fragments are in
[results](results/README.md). All values are bound to verified inference artifact
`0fe2e85baefd1befc22476c92b7595659aeac4528e9899a191b0abb09c57b2f3`.

## Quick start

Check out `finoptions-engine` beside this repository. Prerequisites are `uv`, Rust 1.97.0, and
`just`.

```shell
just test
just check
just full
```

The state-aware launcher synchronizes the locked environment and rebuilds the native sibling only
when its inputs change. Ruff and ty define the Python quality boundary; Clippy and the Rust test
suite are owned by the engine repository.

Heavy local phases use the typed `project_interactive_one_phase_v1` resource budget: one process at
a time, bounded data and hidden CPU thread pools, low OS priority, and no automatic restart. A
cluster or dedicated host must declare a different budget rather than silently increasing local
parallelism. Heavy CLI commands acquire the shared local lease automatically and fail immediately
if another study phase owns it. See [ADR 0051](docs/adr/0051-explicit-local-resource-budget.md).

## Reproduce the publication study

First materialize a verified `project-experiment` feature artifact from licensed inputs. The
manifest-first acquisition, cleaning, support-data, and feature commands are documented in the
[workflow reference](docs/workflows.md). No notebook defines a production transformation.

Run or exactly resume the 12 paired cash-LSM replications and the deterministic yield methods:

```shell
uv run --locked discrete-dividend-study run-publication-cash-replications \
  --features /private/features/project_experiment_features.parquet \
  --feature-artifact-kind project-experiment \
  --artifacts /private/artifacts/publication \
  --threads 14 \
  --math-backend apple_accelerate \
  --event-buffer-policy require_zero_copy
```

Then run the frozen date-panel bootstrap against the verified complete pricing result:

```shell
uv run --locked discrete-dividend-study run-publication-empirical-inference \
  --features /private/features/project_experiment_features.parquet \
  --feature-artifact-kind project-experiment \
  --artifacts /private/artifacts/publication \
  --publication-result /private/artifacts/publication/result.safetensors \
  --threads 14 \
  --event-buffer-policy require_zero_copy
```

Render aggregate-only output without exposing licensed rows:

```shell
uv run --locked discrete-dividend-study render-publication-inference-table \
  --artifact /private/artifacts/publication/inference.safetensors \
  --table primary-models \
  --format markdown
```

Every expensive stage is content-addressed and resumes only through a verified immutable request
receipt. A dirty checkout, changed dependency lock, altered source, missing cohort, invalid event,
or mismatched engine revision fails before execution.

## Repository map

- `src/discrete_dividend_study`: typed adapters, transformations, specifications, orchestration,
  artifacts, inference, and table renderers.
- `config`: the exact sibling-engine revision.
- `results`: verified aggregate-only paper outputs.
- `docs/adr`: financial, numerical, data, and inference decisions.
- `docs/workflows.md`: detailed licensed-data and experiment command reference.
- `benchmarks`: synthetic performance harnesses that require no provider data.
- `tools/dev.py`: locked, state-aware local development entry point.

## Scientific contract

- Exercise, dividend, maturity, curve, day-count, and source-precision semantics are explicit.
- Unsupported financial or numerical domains fail; there are no silent fallbacks or imputations.
- Cash-LSM policies use separate fitting and valuation populations.
- Pricing Monte Carlo uncertainty and empirical sampling uncertainty are reported separately.
- The primary population, metrics, block bootstrap, random mapping, sensitivities, and Holm family
  were frozen before inspecting the selected-identity result.
- The known late ETR coverage gap is handled by mandatory named cohorts, never by invented rows.

The publication estimator is specified by [ADR 0048](docs/adr/0048-frozen-cash-lsm-replication-plan.md),
immutable finalization by [ADR 0049](docs/adr/0049-immutable-publication-result-orchestration.md),
and empirical inference by [ADR 0050](docs/adr/0050-predeclared-empirical-inference.md).

## Future experiments

The neural and TabPFN paths remain separate, bounded experiment families. Their typed preparation,
training, checkpoint, and benchmark infrastructure is retained, but they are not part of the
classical publication result and require a new declared compute phase before any claim is made.
