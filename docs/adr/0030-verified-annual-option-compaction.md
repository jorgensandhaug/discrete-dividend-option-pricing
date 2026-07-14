# ADR 0030: Detach annual option compaction from daily projection layout

## Status

Accepted.

## Context

The verified ORATS option projection is intentionally one Parquet and sidecar per trading day. That
shape is ideal for restartable acquisition but expensive for repeated research scans: the project
period spans thousands of files, repeated open and metadata work dominates small-machine iteration,
and a downstream artifact should not need the raw daily layout merely to verify a derived input.

## Decision

An explicit half-open date selection streams the verified daily projections into one immutable
Parquet per calendar year. The physical writer contract pins the Polars version, streaming engine,
ordering, compression, statistics, row-group size, and durability policy. Each annual receipt binds
the exact ordered daily-entry slice, source rows and bytes, output rows and bytes, schema, date
bounds, ticker set, and file hash. One aggregate manifest binds every annual receipt and output plus
the exact upstream projection-manifest identity.

Initial compaction verifies the complete daily source and rechecks selected source bytes after each
sink. The completed annual artifact is then self-contained: its loader requires only the aggregate
manifest and annual output root, rehashes every receipt and output, and independently checks ordered
schema, rows, dates, and ticker sets. Historical years may contain different subsets of the declared
ticker universe. There is no glob, inferred date bound, runtime substitution, partial-set mode,
repair, or silent fallback.

## Consequences

Feature construction opens one file per selected year instead of one per trading day and can move
or archive the daily projection layout without weakening the compacted artifact's identity. The
extra one-time sequential read and write materially reduces every later iteration. A Polars upgrade
changes the physical writer identity and produces a new content-addressed compaction instead of
pretending byte-level equivalence.
