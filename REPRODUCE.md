# Reproduction

## 1. Create the locked environment

```shell
uv sync --all-groups --extra torch --extra mlx --locked
just check
just test
```

The lock resolves `finoptions-engine` from its public `v0.1.0` release. The Apple profile uses a
measured combination of PyTorch CPU and MLX GPU. The portable profile uses one-thread PyTorch CPU.

## 2. Prepare licensed inputs

Transform provider data into the canonical table in [`DATA.md`](DATA.md). The built-in acquisition,
projection, compaction, and feature commands are discoverable from:

```shell
uv run --locked discrete-dividend-study --help
```

Those stages are strict and content-addressed. If an organization writes its own provider adapter,
it must preserve the final ordered schema and produce the verified feature artifact and receipt
expected by the study loader.

## 3. Freeze the neural cohort

```shell
uv run --locked discrete-dividend-study freeze-neural-cohort \
  --features /private/features/project_experiment_features.parquet \
  --feature-artifact-kind project-experiment \
  --artifacts /private/artifacts/neural \
  --data-threads 4
```

This command verifies the feature bytes and receipt, source-code and dependency identities, the
complete rolling schedule, every nonempty phase, submitted ticker coverage, and the exact prepared
tensor bytes for all five models. It writes only an aggregate content-addressed manifest. Any
invalid input fails before training.

## 4. Run the rolling neural experiment

```shell
uv run --locked discrete-dividend-study run-neural-study \
  --features /private/features/project_experiment_features.parquet \
  --feature-artifact-kind project-experiment \
  --artifacts /private/artifacts/neural \
  --profile portable-cpu \
  --data-threads 4 \
  --parameter-root-seed 2026071401 \
  --shuffle-root-seed 2026071402
```

The schedule contains 117 rolling windows. Every window uses a three-year lookback, the final 30
days of that lookback for validation, the following 30 days for testing, and a 30-day step. Five
models produce 585 immutable window runs.

The command validates the entire schedule and all requested runtimes before training. Re-running
the same request resumes completed windows; changing data, code, dependencies, configuration,
seeds, backend, or rolling phase creates a different request identity.

## 5. Render aggregate results

```shell
uv run --locked discrete-dividend-study render-neural-table \
  --artifact /private/artifacts/neural/studies/<sha256>.study.json \
  --format markdown
```

The renderer requires every expected model and window and rejects mixed cohorts or another study
specification. Output contains only aggregate metrics and source hashes, not licensed rows or
machine-local paths.

## 6. Archive a run

Retain together:

- the exact Git revision and `uv.lock`;
- the feature Parquet and receipt;
- the cohort manifest;
- the complete study manifest and referenced window artifacts;
- the selected execution profile and seeds; and
- the aggregate rendered tables.

Exact market-data rows cannot be redistributed under this repository's license. Researchers must
obtain the required data rights independently.
