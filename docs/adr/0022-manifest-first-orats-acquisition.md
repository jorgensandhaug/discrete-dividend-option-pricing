# ADR 0022: Manifest-first encrypted ORATS acquisition

Status: accepted

Licensed ORATS daily option archives are acquired only through explicit FTPS with certificate
validation, TLS 1.2 or newer, passive transfer, and protected control and data channels. Plain FTP,
implicit TLS, SFTP substitution, web scraping, and protocol fallback are not part of the adapter.
Credentials exist only in the process environment and are excluded from object representations,
command arguments, manifests, receipts, and output.

Acquisition is split into two explicit operations. First, MLSD inventories a closed civil-date
interval and records each canonical remote path, byte size, UTC modification timestamp, and a hash
over the ordered entry identities. The immutable manifest is written before transfer. Second, the
caller supplies that manifest and confirms its exact entry count. Every new or partial transfer
verifies its MLST identity before and after RETR, resumes only from the exact local partial byte
count, checks the completed size, computes SHA-256, and atomically publishes the raw ZIP with a
sidecar receipt.

A completed local archive is reusable only when the requested remote identity equals the receipt,
the local byte count still equals the remote inventory, and a fresh SHA-256 equals the receipt.
Reuse is deliberately detached from current server availability: once the immutable inventory,
archive, and receipt agree, a later remote deletion or permission change cannot invalidate or block
the acquired historical bytes. Remote identity checks remain mandatory before and after every
unfinished transfer.
Missing counterparts, changed server facts, malformed machine listings, oversized partial files,
REST rejection, interrupted transfers, and same-size local corruption are errors. No condition
causes an overwrite, restart-from-zero, alternate listing method, or unencrypted retry.

Raw ZIP retention and normalized Parquet production are separate stages. The receipt identifies the
licensed delivery; later adapters identify the exact ZIP member, uncompressed bytes, projection,
study universe, schema, row count, and produced artifact. Neither raw nor derived licensed data is
committed to Git.
