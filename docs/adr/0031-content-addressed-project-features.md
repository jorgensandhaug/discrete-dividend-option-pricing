# ADR 0031: Close project features over five verified sources

## Status

Accepted.

## Context

The submitted 42-column feature table depends on licensed option quotes, point-in-time dividend
forecasts, realized dividends, underlying history, and fitted NSS parameters. A notebook that opens
plausible filenames cannot prove which bytes, cleaning decisions, or writer runtime produced a
model input. Repeatedly scanning thousands of daily option files also prevents practical iteration
on limited hardware.

## Decision

Feature materialization accepts only five loader-produced verified handles: detached annual option
compaction, the complete dividend-forecast projection set, the bounded historical-dividend
projection, and exact pinned underlying and NSS Parquets. One pre-run request hash covers the
relocatable identities of every source file, the complete feature-pipeline and ordered-schema
hashes, and the exact durable streaming Parquet writer. The request hash names the output directory.

The pipeline selects forecast snapshots by the exact ticker and quote-date keys that survive the
quote-only predicates. Therefore malformed or duplicated provider history that cannot enter the
study neither changes nor rejects the cohort, while every consumed snapshot is still validated
strictly. The output is written once through a streaming sink, sources are reverified after use,
and the final receipt records exact bytes, rows, date bounds, schema, and the submitted six-ticker
universe. The detached loader needs only the feature Parquet and its receipt.

## Consequences

Model and notebook entry points consume one immutable, content-addressed Parquet rather than
reimplementing data preparation. Licensed rows and machine-local paths never enter Git or the
receipt. Source, policy, schema, runtime, or output drift produces a different identity or an early
error; there is no discovery, coercion, repair, or fallback.
