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
just bench-tabpfn-preparation 1000000
```

The state-aware launcher synchronizes the locked environment and rebuilds a changed Rust extension
without publishing a package. PyTorch and MLX are separate optional extras; the development lock
installs both, while the Linux PyTorch source is explicitly CPU-only. The repository remains private
and no publication or public release is authorized.

The neural benchmark uses deterministic project-shaped synthetic rows, emits canonical JSON, and
records metric-history and trained-weight hashes beside throughput and peak process memory. It never
requires or fabricates provider data.

The publication estimator has a separate target-free numerical pilot. It retains every declared
direct-quadrature level, general exercise mesh, LSM path level, and randomized replication in one
non-overwriting JSON receipt; any missed gate returns a nonzero status and changes nothing
automatically:

```shell
python3 tools/dev.py -- publication-numerical-pilot \
  --pilot v5 \
  --threads 4 \
  --math-backend rust_std_f64 \
  --output artifacts/numerical-pilot/v5-portable.json
```

The frozen synthetic panel, tolerances, and limits are recorded in
[ADR 0038](docs/adr/0038-target-free-numerical-pilot.md). V1's retained terminal-cash mesh miss and
the single-dimension v2 extension are recorded in
[ADR 0039](docs/adr/0039-terminal-cash-mesh-pilot-v2.md). Neither contains a market price target or
legacy ranking. V2's retained high-volatility path-drift miss and the v3 path extension are in
[ADR 0040](docs/adr/0040-high-volatility-path-pilot-v3.md). A passing pilot is not an empirical
conclusion. The stable terminal-cash policy bias across all supported global polynomial degrees and
the frozen explicit-predecessor diagnostic are recorded in
[ADR 0041](docs/adr/0041-terminal-cash-policy-diagnostic.md) and
[ADR 0042](docs/adr/0042-explicit-terminal-cash-predecessor-diagnostic.md). The frozen predecessor
diagnostic passed without changing its basis, paths, tolerances, cash time, or terminal event order;
production code must still receive the real predecessor explicitly from calendar semantics. The
single-change full-panel v4 specification is frozen in
[ADR 0043](docs/adr/0043-full-panel-predecessor-pilot-v4.md); it passed all 61 frozen gates. This is
a retained synthetic numerical baseline, not an empirical conclusion. The compact-grid diagnostic
and single-change full-panel v5 are recorded in
[ADR 0044](docs/adr/0044-compact-explicit-terminal-predecessor-diagnostic.md) and
[ADR 0045](docs/adr/0045-full-panel-compact-pilot-v5.md). V5 passed all 60 applicable gates and is
the current target-free numerical baseline; it does not establish an empirical ranking. The real
XNYS schedule materialization and its two exhaustive rate domains are specified in
[ADR 0046](docs/adr/0046-calendar-owned-publication-exercise-schedules.md).

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

Raw archive acquisition is manifest-first and encrypted. Credentials are accepted only through
`ORATS_FTP_USER` and `ORATS_FTP_PASSWORD`; they are never command arguments or artifact fields.
Inventory performs no download:

```shell
export ORATS_FTP_USER='...'
export ORATS_FTP_PASSWORD='...'

uv run --locked discrete-dividend-study inventory-orats-archives \
  --start-date 2007-01-03 \
  --end-date 2026-07-10 \
  --manifest /absolute/private/path/orats-options-inventory.json
```

The inventory summary prints the exact entry count and compressed bytes. Transfer requires the
caller to repeat that count, then resumes partial ZIPs and cryptographically revalidates completed
ones:

```shell
uv run --locked discrete-dividend-study download-orats-archives \
  --manifest /absolute/private/path/orats-options-inventory.json \
  --destination /absolute/private/path/orats-options-raw \
  --confirm-entry-count "$SOURCE_ENTRY_COUNT"
```

The example count is illustrative of one observed inventory, not a hard-coded expectation. A fresh
manifest is authoritative. Explicit FTPS, MLSD/MLST identity, pre/post-transfer verification,
byte-exact resume, SHA-256 receipts, and refusal to overwrite are recorded in
[ADR 0022](docs/adr/0022-manifest-first-orats-acquisition.md).

Verified raw ZIPs are projected into small immutable daily Parquet files before any study cleaning.
The ticker order and exact manifest count are deliberate confirmation inputs:

```shell
uv run --locked discrete-dividend-study materialize-orats-options \
  --manifest /absolute/private/path/orats-options-inventory.json \
  --raw-directory /absolute/private/path/orats-options-raw \
  --output-root /absolute/private/path/orats-options-projected \
  --projection-manifest /absolute/private/path/orats-options-projection-set.json \
  --scratch /absolute/private/path/temporary-extraction \
  --ticker DUK --ticker EPD --ticker ETR --ticker MO --ticker SO --ticker VZ \
  --confirm-entry-count 4913
```

The command verifies every raw receipt, requires one exact daily CSV member, pushes the declared
ticker filter into a strict Polars scan, validates normalized schema and quote-date identity, and
writes content-addressed Zstandard Parquet plus complete cryptographic receipts. The count remains
illustrative; it must equal the supplied manifest. Completion also writes one compact aggregate
manifest that binds the source inventory bytes and identity to every ordered daily Parquet and
sidecar SHA-256. Loading that set rehashes every referenced file before exposing one lazy Polars
scan. The bounded, restart-safe contracts are recorded in
[ADR 0023](docs/adr/0023-verified-orats-option-projection.md) and
[ADR 0027](docs/adr/0027-aggregate-option-projection-identity.md).

Compact the exact project horizon once before repeated feature scans. The upper date bound is
exclusive, and the source count must match the verified daily projection set:

```shell
uv run --locked discrete-dividend-study compact-orats-options \
  --projection-manifest /absolute/private/path/orats-options-projection-set.json \
  --inventory /absolute/private/path/orats-options-inventory.json \
  --projection-root /absolute/private/path/orats-options-projected \
  --output-root /absolute/private/path/orats-options-annual \
  --compaction-manifest /absolute/private/path/orats-options-annual.json \
  --start-date 2011-01-01 \
  --end-date-exclusive 2024-09-28 \
  --confirm-source-entry-count "$SOURCE_ENTRY_COUNT"
```

The streaming operation performs no filtering or financial transformation: it preserves ordered
rows while replacing thousands of daily files with roughly fourteen annual files. Annual receipts
and one aggregate manifest bind the date selection, source-entry hashes, schema, physical writer,
row totals, and exact output bytes. The detached verified loader therefore keeps downstream scans
fast without weakening provenance. See
[ADR 0030](docs/adr/0030-verified-annual-option-compaction.md).

The separate annual dividend-forecast archive uses its exact production HTTPS origin and needs no
credential. Freeze the complete year range before transfer:

```shell
uv run --locked discrete-dividend-study inventory-orats-dividend-archives \
  --start-year 2007 \
  --end-year 2024 \
  --manifest /absolute/private/path/orats-dividend-inventory.json

uv run --locked discrete-dividend-study download-orats-dividend-archives \
  --manifest /absolute/private/path/orats-dividend-inventory.json \
  --destination /absolute/private/path/orats-dividend-raw \
  --confirm-entry-count 18
```

The manifest freezes every annual Content-Length, Last-Modified value, and strong S3 ETag. Exact
range resume uses `If-Match`; every completed ZIP receives a local SHA-256 receipt. The count must
equal the supplied manifest, and the transport never follows redirects or substitutes a full GET
for a rejected range. [ADR 0024](docs/adr/0024-manifest-first-orats-dividend-archives.md) records the
contract.

Project the verified annual ZIPs into the same bounded study universe while retaining complete
snapshot coverage:

```shell
uv run --locked discrete-dividend-study materialize-orats-dividends \
  --manifest /absolute/private/path/orats-dividend-inventory.json \
  --raw-directory /absolute/private/path/orats-dividend-raw \
  --output-root /absolute/private/path/orats-dividend-projected \
  --projection-manifest /absolute/private/path/orats-dividend-projection-set.json \
  --scratch /absolute/private/path/temporary-dividend-extraction \
  --ticker DUK --ticker EPD --ticker ETR --ticker MO --ticker SO --ticker VZ \
  --confirm-entry-count 18
```

Every regular snapshot is CRC-checked and hashed, all rows receive a strict structural preflight,
and each annual receipt records raw, selected, and exact canonical duplicate row counts for every
forecast date—including dates with zero selected rows. The provenance output preserves duplicates;
they remain invalid study inputs until an explicit downstream policy resolves them. The output also
deliberately precedes the separate future-event rule. [ADR 0025](docs/adr/0025-verified-orats-dividend-projection.md)
records the accepted historical ZIP layouts and artifact contract.

Missing ticker keys fail by default. A fully receipted bounded projection may explicitly add
`--unidentified-row-policy record-and-exclude`; the command still rejects missing financial fields
or extra content, and the policy becomes part of the projection hash.

Completion writes one compact annual projection-set manifest. It binds the exact HTTPS inventory,
every full annual sidecar, every Parquet SHA-256, aggregate anomaly counts, and the complete snapshot
count before exposing one ordered lazy scan. [ADR 0028](docs/adr/0028-aggregate-dividend-projection-identity.md)
records this boundary.

Historical realized cash dividends enter through a separate licensed Parquet boundary. The source
hash and row count come from a private acquisition receipt; neither is inferred from a filename:

```shell
uv run --locked discrete-dividend-study materialize-historical-dividends \
  --source /absolute/private/path/all_hist_div.parquet \
  --expected-source-sha256 "$SOURCE_SHA256" \
  --expected-source-rows "$SOURCE_ROWS" \
  --maximum-ex-date 2024-09-27 \
  --output-root /absolute/private/path/historical-dividend-projected \
  --ticker DUK --ticker EPD --ticker ETR --ticker MO --ticker SO --ticker VZ
```

The command requires the exact five-column source schema, rejects null keys, nonfinite cash, source
duplicates, missing declared tickers, negative selected cash, and duplicates created by projection.
It preserves provider rows byte-for-byte upstream, records aggregate unused-field diagnostics, and
writes only the bounded three-column study history under source and specification hashes. No
provider-wide anomaly is silently repaired or reinterpreted. [ADR 0026](docs/adr/0026-verified-historical-dividend-projection.md)
records the boundary.

The risk-free curve is projected directly from one pinned current-vintage Federal Reserve nominal
Svensson CSV:

```shell
uv run --locked discrete-dividend-study materialize-fed-nominal-curve \
  --source /private/curve/feds200628.csv \
  --expected-source-sha256 "$FED_CURVE_SHA256" \
  --expected-source-size-bytes "$FED_CURVE_BYTES" \
  --start-date 2011-01-01 --end-date-exclusive 2024-09-28 \
  --output-root /private/curve/projected
```

The source's Svensson betas are converted from percentage points to decimal continuously
compounded rates while its tau parameters remain in years. Every complete parameter row must
reproduce the published one-year zero yield within the source's stated rounding precision. Positive
option maturities use the same Svensson curve at their exact year fraction; the short-end metadata
stores its stable numerical floor and exact zero-maturity limit, not a one-year flat-rate
substitution. Partial or inconsistent rows fail, while fully unavailable calendar rows are explicitly
counted and excluded. [ADR 0032](docs/adr/0032-pinned-federal-reserve-svensson-curve.md) records the
financial and provenance contract.

The externally prepared underlying history enters through `verify_pinned_parquet`. The caller must
provide its exact SHA-256 and row count; the ordered schema must match without casts or extra
columns, and long-running consumers rehash the source after use. The Federal Reserve projection is
verified by the same pinned-Parquet boundary before feature construction. Paths are deliberately
excluded from relocatable identities. The complete feature schema and every frozen stage
specification also contribute to one `project_feature_pipeline_sha256` identity.
[ADR 0029](docs/adr/0029-pinned-support-parquet.md) records the small generic boundary.

Materialize the single closed feature artifact only after all five identities are known:

```shell
uv run --locked discrete-dividend-study materialize-project-features \
  --option-compaction-manifest /private/options/annual-set.json \
  --option-compaction-root /private/options/annual \
  --forecast-manifest /private/dividends/projection-set.json \
  --forecast-inventory /private/dividends/inventory.json \
  --forecast-root /private/dividends/projected \
  --historical-dividends /private/history/historical_dividends.parquet \
  --underlying-history /private/support/underlying.parquet \
  --underlying-sha256 "$UNDERLYING_SHA256" --underlying-rows "$UNDERLYING_ROWS" \
  --nss-parameters /private/support/nss.parquet \
  --nss-sha256 "$NSS_SHA256" --nss-rows "$NSS_ROWS" \
  --output-root /private/features
```

The request hash binds every source, every transformation choice, the exact 42-column schema, and
the byte-producing Polars runtime before a streaming sink runs. Only forecast snapshots matching
consumed project quote dates are relevant; each consumed snapshot date remains strict. Sources are
reverified after the sink, and the result can then be loaded from its Parquet and receipt alone. A
benchmark subset cannot be passed accidentally: the option source must declare the exact six
tickers and half-open `[2011-01-01, 2024-09-28)` project horizon.
[ADR 0031](docs/adr/0031-content-addressed-project-features.md) records this boundary.

A provider-neutral long-form quote source can enter the same submitted experiment pipeline without
fabricating unused ORATS diagnostics. The long-form boundary is exactly
`ticker, quote_date, expiry_date, strike, underlying_price, bid, ask, volume, option_type`, with
dates and strings plus `Float64` financial values. When source coverage is split by ticker, use the
closed mixed-source materializer:

```shell
uv run --locked discrete-dividend-study materialize-project-experiment-features \
  --long-form-options /private/options/long_form.parquet \
  --long-form-sha256 "$LONG_FORM_SHA256" --long-form-rows "$LONG_FORM_ROWS" \
  --option-compaction-manifest /private/options/annual-set.json \
  --option-compaction-root /private/options/annual \
  --forecast-manifest /private/dividends/projection-set.json \
  --forecast-inventory /private/dividends/inventory.json \
  --forecast-root /private/dividends/projected \
  --historical-dividends /private/history/historical_dividends.parquet \
  --underlying-history /private/support/underlying.parquet \
  --underlying-sha256 "$UNDERLYING_SHA256" --underlying-rows "$UNDERLYING_ROWS" \
  --nss-parameters /private/support/nss.parquet \
  --nss-sha256 "$NSS_SHA256" --nss-rows "$NSS_ROWS" \
  --output-root /private/experiment-features
```

Every strict quote predicate is evaluated in its source representation: long-form `Float64` values
remain `Float64`, while the canonical raw-ORATS projection remains `Float32`. Only surviving rows
are narrowed to one common `Float32` quote-cohort schema. The two ticker domains must be disjoint
and must exactly partition the submitted six-name universe; there is no precedence, date fallback,
row repair, or deduplication. The result is a detached content-addressed 19-column feature artifact
containing only fields consumed by the project experiments and their aggregate diagnostics.
[ADR 0034](docs/adr/0034-source-precision-mixed-option-artifact.md) records this boundary.

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

`build_project_experiment_features` is the provider-minimal equivalent. It applies the same
financial transformations and complete-row policy but emits only the ordered 19-column contract
consumed by the submitted classical and neural experiment families.

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

The publication estimator is deliberately separate from that historical identity. Its strict
support behavior, convergence receipt, paired randomized replications, and date-block empirical
inference are specified in
[ADR 0036](docs/adr/0036-separate-publication-estimator-and-inference.md); no reconstruction
artifact is promoted to a publication result implicitly.

The publication input boundary binds exact civil dates to a checked-in, versioned XNYS session
asset. It preserves pre-expiry cash-event left limits and maps terminal cash only to the verified
prior session. Verified natural states then form two exhaustive domains: nonnegative-rate cash
groups use their explicit ragged exercise schedules, while negative-rate cash groups retain the
declared origin-inclusive uniform American mesh. Both yield variants use that uniform mesh in both
domains. Immutable group/row maps prove complete coverage and restore every native diagnostic to
source order without a retry or fallback. ADR 0037 retains the historical first partition; new
runs use [ADR 0046](docs/adr/0046-calendar-owned-publication-exercise-schedules.md).

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
resolved only from an explicitly typed, receipt-verified feature artifact, the caller-supplied lock
file, and full Git commits; no production path searches for “latest”, clips an invalid prediction,
changes backend, or skips a failed window.

`prepare_project_neural_rolling_source` is the production whole-study boundary. It collects the
minimal full horizon once, prepares each distinct model representation once, and exposes every
overlapping phase as an immutable zero-copy row view. `run_project_neural_rolling_study` preflights
the complete schedule, derives role/model/window-separated seeds, and resumes only through an exact
pre-training request receipt. Its final content-addressed manifest contains every model/window
artifact reference and observation-pooled metrics for each model separately; it never averages
window RMSE or combines metrics across different models. The full contract is recorded in
[ADR 0019](docs/adr/0019-one-pass-neural-study-and-exact-resume.md).

`prepare_project_tabpfn_binned_batch` is a deliberately narrower future-benchmark boundary. It
copies the existing six scalar features and twelve project cash bins once into one immutable
row-major float32 table, retains the positive target without transformation, and exposes zero-copy
rolling slices. It does not install or execute TabPFN, acquire weights, or reinterpret the result as
a financial pricer. [ADR 0021](docs/adr/0021-bounded-tabpfn-input.md) records that limit.

From clean paper and engine checkouts at the pinned engine revision, the complete private-data
classical comparison has one explicit entry point:

```shell
uv run --locked discrete-dividend-study run-classical-study \
  --features /absolute/path/project_experiment_features.parquet \
  --feature-artifact-kind project-experiment \
  --artifacts /absolute/path/immutable-run-artifacts \
  --threads 8 \
  --math-backend apple_accelerate \
  --training-seed 2026071401 \
  --training-domain-start 0 \
  --evaluation-seed 2026071402 \
  --evaluation-domain-start 1000000000 \
  --population-addressing natural_strike_groups \
  --event-buffer-policy allow_rechunk
```

The command first verifies the detached producer receipt, true schema identity, bytes, and aggregate
coverage, derives the frozen 117-window submitted schedule, and lazily selects its exact adjacent
test union `[2015-01-01, 2024-08-11)`. Every declared test partition must remain nonempty. It then
projects and orders only the required classical columns. The artifact kind is mandatory:
`project-experiment` selects the 19-column mixed-source artifact and `project-full` selects the
42-column canonical artifact. Kinds are never inferred and a failed loader never falls through to
the other contract. The command then proves that every `(ticker, quote_date, expiry_date)` group has
one exact simulated state and unique strikes before
entering the native engine. It verifies that the feature artifact and dependency lock did not
change during preparation, and prints only aggregate metrics plus the verified content-addressed
artifact identity. Use the explicit `contract_rows` addressing only for the independent-path audit
comparator; execution topology is never inferred or silently changed. `require_zero_copy` is
available when the Parquet list buffers are already contiguous; rechunking is never implicit.
Use `rust_std_f64` instead on non-macOS systems; backend identity is mandatory and embedded in the
artifact rather than selected from the host automatically.

Both cash-dividend variants declare `at_or_before` maturity semantics: when a source ex-date equals
expiration, the engine applies the cash event before terminal payoff. Events after maturity fail;
no default, clipping, time shift, event deletion, or numerical epsilon is available.

Render a verified aggregate-only table directly from an immutable artifact; the output includes the
source artifact hash and no licensed row-level values or machine-local paths:

```shell
uv run --locked discrete-dividend-study render-classical-table \
  --artifact /absolute/path/result.safetensors \
  --format latex
```

`markdown` is the other supported explicit format.

The complete private-data neural run remains a separate explicit command:

```shell
uv run --locked discrete-dividend-study run-neural-study \
  --features /absolute/path/project_experiment_features.parquet \
  --feature-artifact-kind project-experiment \
  --artifacts /absolute/path/immutable-run-artifacts \
  --profile apple-silicon \
  --data-threads 8 \
  --parameter-root-seed 2026071401 \
  --shuffle-root-seed 2026071402
```

There is no automatic profile, seed, thread count, artifact-kind inference, source discovery,
dirty-checkout allowance, or resume wildcard. Parquet and receipt are rehashed after preparation;
either changing before execution is fatal. The portable profile must be requested explicitly as
`portable-cpu`. [ADR 0035](docs/adr/0035-verified-study-feature-boundary.md) records this boundary.
