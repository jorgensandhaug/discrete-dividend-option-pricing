# ADR 0029: Verify external support Parquet by bytes, rows, and ordered schema

## Status

Accepted.

## Context

The project feature pipeline consumes small support tables whose acquisition or fitting process may
remain provider-specific: underlying price history and daily fitted curve parameters are the first
two examples. Treating a plausible filename or compatible subset of columns as the input identity
would make replacement, schema coercion, and provider revision invisible. Duplicating a separate
adapter framework for every small Parquet would add more policy than the study needs.

## Decision

One generic verifier accepts a regular non-symlink Parquet path plus an explicit expected SHA-256,
positive row count, and exact ordered Polars schema. It hashes the bytes before parsing, rejects
missing, extra, reordered, or differently typed columns, verifies the row count, and detects a file
identity change during verification. The resulting relocatable identity contains only filename,
size, byte hash, rows, and ordered-schema hash. A separate post-consumption check rehashes the file
so a long-running materialization cannot accept a source that changed after preflight.

The verifier performs no casts, date filtering, null repair, path discovery, provider inference, or
fallback. Domain-specific content validation remains in the feature stage that owns those financial
semantics. The complete project pipeline separately hashes every frozen transformation
specification and its exact output schema.

## Consequences

Private data can move between machines without changing its scientific identity, while a revised
provider file cannot masquerade as the earlier source. The verifier adds two sequential reads when
a consuming operation uses the post-check; for support files this is intentionally cheap compared
with recomputation or scanning the licensed option feed.
