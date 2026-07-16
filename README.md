# Discrete-dividend option pricing

Reference implementation for *American Stock Option Pricing Using Discrete Dividend Forecasts: A
Neural Network and Least Squares Monte Carlo Approach*.

The study tests whether retaining the timing and size of forecast cash dividends improves American
call-option pricing relative to replacing the schedule with a continuous-yield feature. It compares
five neural representations under the same rolling windows, market features, target, optimizer,
and evaluation procedure:

| Model | Dividend input |
|---|---|
| Direct LSTM | Ordered future cash-dividend events |
| Event attention | Learned representation of future cash-dividend events |
| Twelve bins | Cash amounts grouped by time to payment |
| Forecast yield | One scalar derived from forecast dividends |
| Historical yield | One trailing realized-dividend scalar |

The typed paper configuration is exposed as `PROJECT_STUDY` in
[`paper.py`](src/discrete_dividend_study/paper.py). PyTorch and MLX execute the same
framework-neutral model parameters and are tested for matching forward and gradient behavior.

## Install

Prerequisites are [uv](https://docs.astral.sh/uv/), Rust 1.97.0, and `just`.

```shell
git clone https://github.com/jorgensandhaug/discrete-dividend-option-pricing.git
cd discrete-dividend-option-pricing
uv sync --all-groups --extra torch --extra mlx --locked
just test
```

`just check` runs formatting, Ruff, and strict `ty` checks. The exact engine release is resolved
from the lock file and does not require another repository beside this one.

## Data

Market and dividend-forecast data used by the study are licensed and cannot be redistributed.
[`DATA.md`](DATA.md) defines the complete provider-neutral feature table, dtypes, units, and row
semantics required by the experiment. The repository contains no observations, targets,
predictions, credentials, or trained weights.

Researchers with equivalent data can adapt it to that contract. Existing ORATS-oriented adapters
are included for users licensed to access the corresponding source fields.

## Reproduce the study

Given a verified feature artifact and its receipt, freeze the exact five-model input cohort before
training:

```shell
uv run --locked discrete-dividend-study freeze-neural-cohort \
  --features /private/features/project_experiment_features.parquet \
  --feature-artifact-kind project-experiment \
  --artifacts /private/artifacts/neural \
  --data-threads 4
```

The aggregate-only cohort artifact hashes every prepared model representation and all 117 phase
row counts without copying licensed observations into the repository. Then run the complete study:

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

Use `--profile portable-cpu` on non-Apple hardware. Runtime substitution is never silent: an
unavailable requested backend is an error.

After all 585 model-window runs complete, render the aggregate table:

```shell
uv run --locked discrete-dividend-study render-neural-table \
  --artifact /private/artifacts/neural/studies/<sha256>.study.json \
  --format markdown
```

Every window result is immutable and content-addressed. An interrupted run resumes only when the
data, configuration, seeds, software revisions, and runtime identity match. See
[`REPRODUCE.md`](REPRODUCE.md) for the complete workflow.

## Code map

- `paper.py`: the small public surface for the frozen study and five model identities.
- `neural_spec.py`: typed architecture and training configuration.
- `torch_neural.py` and `mlx_neural.py`: backend implementations of that configuration.
- `project_pipeline.py`: point-in-time feature construction and filtering.
- `neural_cohort.py`: fail-fast schedule, coverage, and prepared-tensor admission.
- `neural_study.py`: rolling execution and exact resume.
- `finoptions-engine`: the separately versioned Rust pricing and simulation dependency.

## License and citation

Code is licensed under Apache-2.0. Citation metadata for the project and its three authors are in
[`CITATION.cff`](CITATION.cff).
