# ADR 0026: Verified historical-dividend projection

Status: accepted

Historical realized cash dividends arrive as a licensed Parquet outside Git. Admission requires an
explicit expected SHA-256 and row count; a plausible filename is never evidence. The source must be
a regular non-symlink file with the exact ordered schema `ticker: String`, `ex_date: Date`, `amount:
Float64`, `freq: Int64`, and `pay_date: Date`. The complete source is structurally preflighted before
selection. Null rows, blank ticker keys, nonfinite cash, and exact five-column duplicates fail.

Provider-wide zero cash, frequency codes, and payment-date ordering are not silently assigned new
economic meanings. They are aggregate receipt diagnostics because the project consumer uses only
ticker, ex-date, and cash amount. The declared ticker universe and maximum ex-date are then pushed
into the lazy scan. Every declared ticker must be observed. The bounded three-column result rejects
negative cash and exact duplicates introduced by dropping unused source fields, while retaining
distinct same-date cash components for the downstream additive rule.

The projection specification freezes sorted unique uppercase tickers, the inclusive maximum
ex-date, Zstandard level 3, a 65,536-row group, and schema version. Output lives at
`<root>/<source-sha256>/<projection-spec-sha256>/historical_dividends.parquet`. Its canonical JSON
receipt binds source name, bytes, hash, rows and schema hash; provider-wide aggregate diagnostics;
excluded-row accounting; complete specification and schema hashes; selected ticker and date
coverage; and output rows, bytes and SHA-256. The source is stat- and hash-revalidated after
collection. Existing output is reusable only after receipt parsing, fresh byte hashing, schema,
row-count, ticker, and date-summary validation. Partial, tampered, or mismatched artifacts fail and
are never overwritten or repaired.
