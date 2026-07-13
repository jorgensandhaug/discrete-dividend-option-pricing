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
