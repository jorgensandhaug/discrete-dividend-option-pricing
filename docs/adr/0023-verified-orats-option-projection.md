# ADR 0023: Verified bounded ORATS option projection

Status: accepted

Each acquired daily option ZIP is admitted only through its immutable archive receipt. Projection
accepts exactly one non-directory, non-encrypted, non-symlink member named
`ORATS_SMV_Strikes_YYYYMMDD.csv` at the ZIP root. The member has an explicit two-GiB upper bound,
is read completely so ZIP CRC validation cannot be skipped, and is hashed while streaming to a
temporary CSV. The source ZIP is size- and SHA-256-revalidated after extraction. Wrong paths,
additional files, malformed CSV fields, archive changes, and stale scratch files are errors.

The projection specification freezes a sorted unique uppercase ticker set and the physical Parquet
settings. Polars pushes the ticker predicate into the strict lazy CSV scan, canonicalizes only the
owned 17 option columns, and collects with the streaming engine. The materialized frame must have
the exact normalized schema, contain one quote date equal to the archive date, and contain only
declared tickers. A day may contain only a subset of the declared universe because securities need
not exist or have listed options throughout the full study horizon; a completely empty day is an
error.

Daily outputs use Zstandard level 3, 65,536-row groups, statistics, and a deterministic location:
`<root>/<projection-spec-sha256>/<year>/<YYYY-MM-DD>.parquet`. The sidecar receipt binds the remote
inventory entry, raw ZIP SHA-256, member name, CRC32, compressed and uncompressed sizes,
uncompressed SHA-256, complete projection specification and hash, normalized schema hash, observed
tickers, row and byte counts, and output SHA-256. Parquet and receipt are published atomically. An
existing pair is reusable only after exact receipt, size, and fresh output-hash validation; missing,
partial, mismatched, or same-size-corrupted artifacts never trigger overwrite or fallback.

The temporary full-universe CSV is deleted after every successful or failed attempt. Only the
small declared ticker projection persists, keeping downstream scans fast and storage bounded while
retaining the raw cryptographically receipted ZIP as the licensed source of truth. Neither artifact
belongs in Git.
