# ADR 0034: Preserve source precision across disjoint option partitions

## Status

Accepted.

## Context

The submitted project cohort can be reconstructed from two licensed option boundaries with
different physical representations. One is a provider-neutral long-form table whose financial
values are `Float64`; the other is the canonical raw-ORATS projection whose selected quote values
are `Float32`. Strict price, moneyness, intrinsic-value, and spread comparisons can move rows across
their boundaries if values are narrowed before filtering. A union that silently prefers one source,
fills missing dates, or deduplicates overlapping contracts would create a new undocumented dataset.

The full 42-column ORATS artifact also contains vendor diagnostics that neither submitted experiment
family consumes. Requiring a compatible source to invent those columns would weaken provenance and
make the reusable research boundary needlessly provider-specific.

## Decision

Long-form input has one exact nine-column schema and must contain one call and one put for every
selected `(ticker, quote_date, expiry_date, strike)` key. Every selected side is consumed exactly
once; nulls, nonfinite values, unsupported option types, duplicate sides, missing sides, missing
declared tickers, or unequal call/put spot values fail before output. Pairing preserves every
financial value as `Float64`.

Quote features and the frozen submitted predicates support two explicit numerical identities.
Long-form rows are derived and filtered in `Float64`; raw-ORATS rows remain `Float32`. Each surviving
partition is then narrowed to the same ordered canonical quote-cohort schema. The partition union
requires pairwise-disjoint, exact ticker declarations and the complete submitted ticker universe.
It has no source precedence, temporal fallback, repair, or deduplication semantics.

Immediately after the predicates, raw bid, ask, volume, put-side, intrinsic-value, and spread
diagnostic columns are pruned from the temporary stage. Only the nine quote fields consumed by
support enrichment and the final experiment schema cross the materialization boundary.

The minimal enrichment path applies the same exact-date NSS intersection, point-in-time forecast
selection, realized-dividend window, underlying as-of join, volatility definition, rate rule, and
complete-row policy as the full project pipeline. It emits one ordered 19-column experiment schema.
A combined pipeline hash binds both precision-specific pipeline hashes and the exact source/ticker
allocation.

Materialization accepts only loader-produced verified handles for the long-form Parquet, annual
ORATS compaction, forecast-dividend set, historical-dividend projection, underlying history, and NSS
parameters. A request hash binds every relocatable source identity, the combined pipeline, schema,
allocation, and durable Polars writer. Sources are reverified after the streaming sink. The detached
loader needs only the output Parquet and receipt and rechecks bytes, schema, nulls, finiteness,
contract-key uniqueness, per-ticker coverage, and all aggregate identities.

Coverage reports the full horizon, the 2012 descriptive cohort, and the factual post-2015 cohort.
It does not infer that a date-threshold cohort is identical to missing legacy rolling-output rows.
The artifact version is part of the request identity, so a receipt-schema change cannot reuse an
older request directory.

## Consequences

The cohort cannot change merely because an adapter chose a smaller floating-point dtype. Partial
coverage remains visible per ticker rather than being hidden behind a same-ticker fallback. The
artifact is substantially smaller than the full vendor schema and is directly consumable by every
submitted experiment family. Licensed rows and machine-local paths remain outside Git and outside
the relocatable receipt. Supporting an overlapping or temporal source splice would require a new
explicit data contract and pipeline identity.
