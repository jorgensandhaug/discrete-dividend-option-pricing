# ADR 0005: Strict lazy ORATS option normalization

Status: accepted

One uncompressed ORATS daily CSV is scanned lazily under an explicit parse contract. The 17 fields
used by both submitted studies are parsed directly into their historical `Float32` representation;
`yte` and `iRate` remain `Float64` for the separately defined curve adapter. Unowned vendor columns
remain strings and are projected away unless another adapter explicitly defines their semantics.
Schema inference, lossy text decoding, parse-error skipping, glob expansion, and missing-column
insertion are disabled.

Normalization strictly parses the observed `M/D/YYYY` quote and expiration dates, selects and
renames the shared 17 columns, and preserves input order, duplicate rows, and nulls. It performs no
quote filtering, imputation, sorting, or unit inference. Those are separately versioned stages, so
the as-submitted project filter and future corrected filters cannot be blended into ingestion.

Raw ZIP acquisition, byte hashes, extraction, source licensing, and artifact manifests belong to
the provider/storage layer. Licensed raw or normalized rows are never committed to this repository.
