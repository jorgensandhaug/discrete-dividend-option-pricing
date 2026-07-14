# ADR 0024: Manifest-first ORATS dividend archives

Status: accepted

Annual ORATS dividend-forecast archives are acquired only from the exact production S3 HTTPS
origin and canonical `{year}ORATSStockDivForecasts.zip` object names. Certificate validation and
TLS 1.2 or newer are mandatory. HTTP, alternate origins, redirects, content-encoding substitution,
URL discovery, and protocol fallback are excluded. This archive endpoint does not require or admit
credentials.

Acquisition begins with one HEAD request for every year in a closed interval. Each response must
have the exact final URL, status 200, identity encoding, a positive Content-Length, a UTC
Last-Modified value, one strong S3 ETag, and explicit byte-range support. The immutable inventory
records those identities in exact year order and hashes them independently of observation time. A
missing year or duplicate/ambiguous header fails the complete inventory.

Download verifies the HEAD identity before and after GET. A new transfer requires status 200 and
the exact full Content-Length. A partial transfer resumes only with `If-Match` for the inventoried
ETag and an exact byte range; status 206, Content-Range, response length, ETag, modification time,
encoding, and final URL must all agree. Range rejection or a full-response substitution preserves
the partial file and fails. The S3 ETag remains a remote version fact rather than being treated as a
portable content hash.

Completion requires the inventoried byte count and a locally computed SHA-256. The ZIP and its
complete source/ETag/SHA sidecar receipt are published atomically. Existing files are reusable only
after exact receipt, size, and fresh SHA-256 validation; missing counterparts, stale identities,
same-size corruption, symlinks, and partials larger than the remote object are errors. Raw annual
archives remain licensed local inputs and are never committed to Git.
