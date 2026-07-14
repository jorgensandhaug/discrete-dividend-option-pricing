# ADR 0027: Bind daily option projections into one verified set

## Status

Accepted.

## Context

The raw ORATS delivery and each projected trading day already have strict cryptographic receipts,
but a complete study run spans thousands of independent Parquet files. Inferring a dataset from a
directory glob would leave the selected inventory, ordering, completeness, and exact daily bytes
implicit. Embedding all licensed rows or all verbose daily receipts in a second artifact would add
unnecessary storage and parsing work.

## Decision

Every complete projection command must write one canonical aggregate manifest. It pins the exact
source-inventory file and ordered-entry identity, projection specification and normalized schema,
actual first/last trading dates, row and byte totals, and a compact entry for every day. The source
inventory hash separately pins the requested calendar bounds, including non-trading endpoints. Each entry records canonical
relative paths plus the SHA-256 and aggregate identity of both the daily Parquet and its full
sidecar receipt. Dates must be strictly increasing and exactly cover the source inventory.

The verified loader rejects missing files, symlinks in the referenced path, unsafe relative paths,
source-inventory drift, receipt drift, output drift, count drift, and specification drift. Only
after every file is rehashed does it expose the ordered paths as one Polars lazy scan. Existing
manifests are reusable only when they verify to the exact same requested set; there is no directory
discovery, partial-set mode, repair, or silent fallback.

## Consequences

The project feature build can consume one small identity artifact while retaining the independent,
restart-safe daily projections. Verification performs one sequential read of every small projected
file and receipt, which is intentional and much cheaper than rereading the raw compressed feed.
Licensed data remain outside Git and are neither copied nor embedded in the manifest.
