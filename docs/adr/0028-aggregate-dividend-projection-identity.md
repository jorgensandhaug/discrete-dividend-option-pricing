# ADR 0028: Bind annual dividend projections into one verified set

## Status

Accepted.

## Context

Annual ORATS dividend archives and their projected Parquet outputs already have strict receipts,
but project feature construction needs one closed forecast dataset. Directory discovery would leave
year completeness, projection-policy identity, receipt selection, and exact bytes implicit. The
annual sidecars are intentionally detailed because they identify every daily snapshot and anomaly;
copying those full receipts into another manifest would only duplicate them.

## Decision

Every complete annual projection command must write one canonical aggregate manifest. It pins the
source inventory file and ordered-entry identity, exact ticker/anomaly projection policy, canonical
schema, inclusive year range, aggregate row/duplicate/unidentified/snapshot/byte totals, and one
compact entry for every year. Each annual entry records canonical relative paths, the sidecar hash,
the output hash, and the aggregate fields needed to cross-check the full sidecar.

The verified loader rejects missing years, source-inventory drift, sidecar drift, output drift,
symlinks, unsafe relative paths, count drift, and specification drift. Only after every referenced
file is rehashed does it expose the ordered annual Parquets as one Polars lazy scan. Existing
manifests are reusable only when they verify to exactly the requested complete set. There is no glob,
partial-set mode, automatic repair, anomaly suppression, or silent fallback.

## Consequences

The feature materializer can consume one small identity artifact while retaining rich daily
snapshot provenance in the annual sidecars. Verification reads each bounded annual Parquet and JSON
sidecar once; it does not reread the much larger raw archives. Licensed rows remain outside Git and
are neither copied nor embedded in the manifest.
