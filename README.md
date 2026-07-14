# Discrete-dividend option pricing

Private pre-release repository for the reproducible reconstruction and extension of the
project-thesis option-pricing study.

The repository owns licensed-data adapters, point-in-time selection and cleaning, study
specifications, neural models, rolling experiments, statistical analysis, and publication outputs.
Generic financial types and pricing kernels come from the private sibling `finoptions-engine`
repository at a pinned revision.

Raw and derived licensed row-level data are never committed. Production workflows will build
content-addressed Parquet artifacts locally from explicit provider adapters; notebooks consume
those artifacts and do not define the pipeline.

## Development

Check out `finoptions-engine` beside this repository, then run:

```shell
just test
just check
just bench-neural nn_bin 100000 10000 10
just bench-inference nn_bin 100000 8192
just bench-rolling-preparation full_horizon 500000 117 8
```

The state-aware launcher synchronizes the locked environment and rebuilds a changed Rust extension
without publishing a package. PyTorch and MLX are separate optional extras; the development lock
installs both, while the Linux PyTorch source is explicitly CPU-only. The repository remains private
and no publication or public release is authorized.

The neural benchmark uses deterministic project-shaped synthetic rows, emits canonical JSON, and
records metric-history and trained-weight hashes beside throughput and peak process memory. It never
requires or fabricates provider data.

The first production data boundary is intentionally small:

```python
import polars as pl
from finoptions import EventSchedule

from discrete_dividend_study import adapt_raw_orats_curve

schedule = EventSchedule(
    times=(0.0, 0.25, 0.5, 1.0),
    dividends=(),
    exercise_nodes=(),
    expiry_node=3,
)
adapted = adapt_raw_orats_curve(pl.scan_parquet("local-orats-slice.parquet"), schedule)
curve = adapted.curve
```

The adapter remains lazy until it reduces the selected rows to maturity nodes, validates source
agreement and calendar ordering, and rejects extrapolation. Its financial contract is recorded in
[ADR 0004](docs/adr/0004-orats-curve-adapter.md).

Daily option deliveries use the same strict boundary: `scan_raw_orats_csv` owns the CSV parse types,
and `canonicalize_raw_orats_options` returns the exact shared 17-column thesis representation
without filtering, dropping, or imputing rows. Licensed inputs and generated artifacts stay local.

`add_quote_features` and `apply_quote_filter` keep feature definitions and cohort selection separate.
The immutable `PROJECT_AS_SUBMITTED_FILTER` records every project-study bound and comparison rather
than recovering hidden defaults at runtime.

`prepare_underlying_history` validates the daily adjusted-price contract, canonicalizes observation
order, and computes the immutable project rolling-volatility specification with complete windows and
explicit sample degrees of freedom and annualization.
`join_underlying_history` then validates grouped as-of ordering and support coverage while retaining
the exact matched observation date and staleness beside every quote.

`scan_raw_orats_dividend_forecast`, `canonicalize_raw_orats_dividend_forecast`, and
`select_future_orats_dividend_forecasts` preserve the provider snapshot before applying the strict
submitted future-event boundary. Cash amounts remain `Float64`; missing daily snapshots are errors,
not implicit carry-forwards.

`add_forecast_dividend_features` builds event schedules once per unique option horizon, verifies
point-in-time snapshot coverage, and fans them back to quotes without reordering. The submitted
zero-event, same-date-component, expiry-boundary, day-count, and scalar-feature policies are carried
by one immutable specification.

`add_historical_dividend_features` uses a cumulative two-as-of algorithm on unique ticker/date keys
for the explicit 52-week observed-cash window. It validates history coverage and exact endpoint
semantics before joining totals back to quote rows.

`add_nss_rate_feature` reproduces the exact-date submitted short-end and NSS scalar-rate rule with
strict `Float64` parameters and a cancellation-safe evaluation. Pricing-engine discount curves keep
their own explicit provider and quote-convention adapter.

`build_project_as_submitted_features` is the single complete lazy composition. It first enforces the
frozen six-ticker study universe, pushes quote-only predicates before expensive joins, applies the
complete-row policy after all support data, and emits one fixed ordered 42-column model/research
contract.

`PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC` resolves the calendar-day train, validation, and test
identity into a typed 117-window manifest. Phase bounds are half-open, and a one-scan coverage
preflight fails before training if any required partition is empty.

`PROJECT_CLASSICAL_RECONSTRUCTION_SPEC` freezes the four submitted classical comparison names and
their distinct exact-cash, escrowed-cash, forecast-yield, and historical-yield semantics. One
materialized boundary prepares shared immutable Float64 columns and zero-copy ragged cash children;
stateful contract-parallel pricing stays outside Polars and receives a separate explicit CPU/random
execution specification. A completed paired run is one content-addressed safe-tensors artifact
containing the target once, every variant's native diagnostics, Float64 metrics, and a canonical
source/code/engine/execution manifest without local paths. The contract is recorded in
[ADR 0020](docs/adr/0020-explicit-classical-reconstruction.md).

`PROJECT_NEURAL_STUDY_SPEC` is the framework-neutral identity of the five project neural variants.
`prepare_project_neural_batch` validates a materialized phase and creates only the model's declared
float32 scalar, target, and dividend representation. Cash-event children stay zero-copy when Polars
storage permits it; any rechunk is an explicit caller policy, and every invalid layout fails before
training.

Neural epochs use deterministic fixed-shape batches while preserving every logical row. PyTorch and
MLX share explicit optimizer, scheduling, early-stopping, and best-weight semantics. Reusable model
artifacts are atomic content-addressed safe-tensors files bound to study, phase hashes, seeds,
runtime, parameters, and complete metric history; pickle and mutable “latest” aliases are excluded.

`prepare_project_neural_window` collects all three projected phases together and hashes the exact
read-only model tensors. `run_neural_window` then trains the explicitly profiled backend, performs
bounded fixed-batch inference, computes the submitted pooled MAPE/RMSE and diagnostic metrics, and
emits cross-referenced checkpoint, prediction, and canonical JSON run artifacts. Source identity is
resolved only from caller-supplied feature and lock files plus full Git commits; no production path
searches for “latest”, clips an invalid prediction, changes backend, or skips a failed window.

`prepare_project_neural_rolling_source` is the production whole-study boundary. It collects the
minimal full horizon once, prepares each distinct model representation once, and exposes every
overlapping phase as an immutable zero-copy row view. `run_project_neural_rolling_study` preflights
the complete schedule, derives role/model/window-separated seeds, and resumes only through an exact
pre-training request receipt. Its final content-addressed manifest contains every model/window
artifact reference and observation-pooled metrics for each model separately; it never averages
window RMSE or combines metrics across different models. The full contract is recorded in
[ADR 0019](docs/adr/0019-one-pass-neural-study-and-exact-resume.md).

From clean paper and engine checkouts at the pinned engine revision, the complete private-data
classical comparison has one explicit entry point:

```shell
uv run --locked discrete-dividend-study run-classical-study \
  --features /absolute/path/project-features.parquet \
  --artifacts /absolute/path/immutable-run-artifacts \
  --threads 8 \
  --math-backend apple_accelerate_vforce \
  --training-seed 2026071401 \
  --training-domain-start 0 \
  --evaluation-seed 2026071402 \
  --evaluation-domain-start 1000000000 \
  --event-buffer-policy allow_rechunk
```

The command projects and materializes only the ten classical input columns, verifies that the
feature artifact and dependency lock did not change during preparation, and prints only aggregate
metrics plus the verified content-addressed artifact identity. `require_zero_copy` is available
when the Parquet list buffers are already contiguous; rechunking is never implicit.
Use `rust_std_f64` instead on non-macOS systems; backend identity is mandatory and embedded in the
artifact rather than selected from the host automatically.

The complete private-data neural run remains a separate explicit command:

```shell
uv run --locked discrete-dividend-study run-neural-study \
  --features /absolute/path/project-features.parquet \
  --artifacts /absolute/path/immutable-run-artifacts \
  --profile apple-silicon \
  --data-threads 8 \
  --parameter-root-seed 2026071401 \
  --shuffle-root-seed 2026071402
```

There is no automatic profile, seed, thread count, source discovery, dirty-checkout allowance, or
resume wildcard. The portable profile must be requested explicitly as `portable-cpu`.
