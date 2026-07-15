# ADR 0006: Shared quote features and explicit project filter

Status: accepted for the project as-submitted reconstruction

The paper pipeline derives mid prices, civil-day time to expiration divided by 365, call/put
moneyness, signed intrinsic values, and two separately named spread conventions through lazy Polars
expressions. `project_call_spread` is `(call ask - call mid) / call mid`; the ordinary diagnostic
`call_spread` is `(call ask - call bid) / call mid`. These definitions and their `Float32`/`Float64`
output types are validated before filtering.

Every filtering value is carried by an immutable `QuoteFilter`. The frozen
`PROJECT_AS_SUBMITTED_FILTER` requires 3--365 calendar days, call bid and ask strictly above 0.10,
put bid and ask strictly above zero, call moneyness in the closed interval [1, 2], the submitted
intrinsic/ask tolerance, the submitted call-spread limit, quote dates from 2011-01-01, and complete
joined rows. Strict versus inclusive comparisons are part of the identity.

Quote-domain predicates execute before expensive support-data joins. Their arithmetic and
comparisons retain the input source's declared `Float32` or `Float64` representation; narrowing a
surviving row is a separate later boundary. The explicit complete-row null policy executes after
support enrichment so it covers the complete featured row. Input normalization, quote predicates,
support-data joins, complete-row selection, and later corrected or robustness cohorts remain
distinct immutable stages; no convenience default selects among them.
