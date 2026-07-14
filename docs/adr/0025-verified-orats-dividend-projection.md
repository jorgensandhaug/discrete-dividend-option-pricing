# ADR 0025: Verified ORATS dividend projection

Status: accepted

Each verified annual dividend ZIP is admitted through its S3/SHA receipt and a bounded member
inventory. Every member must have a relative safe path, be non-encrypted and non-symlink, and fit
explicit per-member, total-byte, and member-count limits. Non-metadata provider files are accepted
only under the archive root or its exact `{year}ORATSStockDivForecasts` directory and must be one of
the observed named feeds: regular stock forecasts, dividends, special dividends, historical growth,
or the limited historical stock-dividend-trend feed. AppleDouble files beneath `__MACOSX` are
recognized as packaging metadata and never extracted.

The regular forecast identity accepts the observed singular and plural names with either an
eight-digit forecast date or date plus four-digit feed time. Exactly one regular member per forecast
date is required and every date must belong to the archive year. Regular members are streamed to a
unique scratch directory, read completely for ZIP CRC validation, and SHA-256 hashed. The source ZIP
is size- and hash-revalidated after extraction.

Polars scans every extracted snapshot together under a five-column sentinel schema. A first
streaming aggregate proves that every snapshot is nonempty and every row has exactly the four
provider fields, including the historically optional terminal delimiter. A second streaming scan
pushes down the declared ticker set. Only that bounded materialization is canonicalized into exact
date, Float64 cash amount, and UInt8 frequency types, financially validated, and deterministically
sorted. This provenance artifact preserves past, same-day, future, zero-amount, distinct same-date,
and exact duplicate provider rows. Duplicate counts are explicit receipt data and remain invalid
study inputs until a separate declared policy resolves them; the submitted `ex_date > forecast_date`
study rule is likewise a separate explicit transformation.

Missing or blank ticker keys fail by default. The projection identity may instead declare
`record-and-exclude`: only rows with a missing ticker and otherwise complete typed financial fields
are excluded from a ticker projection, and their count is bound to the affected snapshot and annual
receipt. Missing financial fields and extra fields always fail. This is an explicit key-domain
decision, not a parser fallback.

Annual Zstandard Parquet lives at `<root>/<projection-spec-sha256>/<year>.parquet`. Its receipt binds
the annual remote object and raw ZIP SHA-256, complete ticker/resource/physical specification,
canonical schema hash, every snapshot's member path, date, CRC32, byte counts, SHA-256, full raw row
count, selected row count, and selected canonical duplicate count, plus output rows, duplicate rows,
unidentified source rows, bytes, observed tickers, and SHA-256. Existing output is reusable only
after exact receipt, size, and fresh hash validation. Scratch is removed after success or failure;
no malformed or partial artifact is repaired, overwritten, or silently skipped.
