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
The option compaction must declare exactly the submitted six tickers and half-open quote horizon
`[2011-01-01, 2024-09-28)`, and historical dividends use the matching inclusive 2024-09-27 cap; a
benchmark subset or broader projection fails before any sink runs.

The pipeline restricts forecast data to the exact quote dates that survive quote-only predicates,
validates those snapshots, and applies the submitted strict `ex_date > forecast_date` event
selection. ORATS is treated as a sparse event feed: the forecast date itself must exist, while an
absent ticker inside that verified snapshot is an explicit empty event schedule. Thus malformed or
duplicated provider history outside consumed dates cannot reject the cohort, a missing consumed
snapshot date remains fatal, and sparse ticker absence is never an implicit fallback.

The Federal Reserve NSS source has no parameters on some equity-trading dates. The submitted inner
join is reproduced as an explicit quote-date intersection before expensive support joins; malformed
or duplicated NSS rows are still rejected. The output is written once through a streaming sink,
sources are reverified after use, and the final receipt records exact bytes, rows, date bounds,
schema, and the submitted six-ticker universe. The detached loader needs only the feature Parquet
and its receipt.

## Consequences

Model and notebook entry points consume one immutable, content-addressed Parquet rather than
reimplementing data preparation. Licensed rows and machine-local paths never enter Git or the
receipt. Source, policy, schema, runtime, or output drift produces a different identity or an early
error; there is no discovery, coercion, repair, or fallback.
