# ADR 0002: Lazy data preparation and explicit pricing artifacts

Status: accepted

Stateless data preparation uses Polars lazy scans and built-in expressions. Validated pricing inputs
are materialized through native Parquet sinks into content-addressed immutable artifacts. Stateful
simulation, regression, pricing, and model training consume bounded row groups/batches and write
new immutable artifacts; they are not represented as dataframe expressions.

This boundary preserves predicate/projection pushdown, bounds memory, makes recovery explicit, and
keeps the engine independent of Polars' native plugin release graph. Notebooks may scan results but
do not define production transformations.

