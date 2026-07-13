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

