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
```

The state-aware launcher synchronizes the locked environment and rebuilds a changed Rust extension
without publishing a package. The repository remains private and no publication or public release
is authorized.

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
