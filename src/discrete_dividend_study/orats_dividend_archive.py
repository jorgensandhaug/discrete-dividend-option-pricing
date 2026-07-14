"""Manifest-first HTTPS acquisition of annual ORATS dividend-forecast archives."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable
    from email.message import Message
    from http.client import HTTPMessage, HTTPResponse
    from typing import IO

ORATS_DIVIDEND_ARCHIVE_INVENTORY_VERSION = 1
ORATS_DIVIDEND_ARCHIVE_RECEIPT_VERSION = 1
ORATS_DIVIDEND_ARCHIVE_BASE_URL = "https://s3.us-east-1.amazonaws.com/assets.orats.com"
ORATS_DIVIDEND_ARCHIVE_TIMEOUT_SECONDS = 60.0
ORATS_DIVIDEND_ARCHIVE_TRANSFER_BLOCK_BYTES = 8 * 1024 * 1024

_FIRST_ARCHIVE_YEAR = 2007
_MAXIMUM_ARCHIVE_YEAR = 9999
_MINIMUM_TRANSFER_BLOCK_BYTES = 64 * 1024
_SHA256_HEX_LENGTH = 64
_ETAG_PATTERN = re.compile(r'"[0-9a-f]{32}(?:-[1-9][0-9]*)?"')


class OratsDividendArchiveError(RuntimeError):
    """Base class for invalid annual dividend archive operations."""


@dataclass(frozen=True, slots=True)
class OratsDividendArchiveConfigurationError(OratsDividendArchiveError):
    """Static HTTPS archive settings violate the explicit contract."""

    reason: str

    def __str__(self) -> str:
        """Describe the invalid setting."""
        return self.reason


@dataclass(frozen=True, slots=True)
class OratsDividendArchiveTransportError(OratsDividendArchiveError):
    """An authenticated HTTPS request failed."""

    method: str
    url: str
    reason: str

    def __str__(self) -> str:
        """Describe the failed remote operation."""
        return f"ORATS dividend HTTPS {self.method} failed for {self.url}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OratsDividendArchiveIntegrityError(OratsDividendArchiveError):
    """A remote identity, manifest, transfer, or receipt is invalid."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"ORATS dividend archive integrity failure{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OratsDividendHttpsSpec:
    """Fixed provider endpoint with bounded request and transfer settings."""

    base_url: str = ORATS_DIVIDEND_ARCHIVE_BASE_URL
    timeout_seconds: float = ORATS_DIVIDEND_ARCHIVE_TIMEOUT_SECONDS
    transfer_block_bytes: int = ORATS_DIVIDEND_ARCHIVE_TRANSFER_BLOCK_BYTES

    def __post_init__(self) -> None:
        """Reject endpoint substitutions and unbounded transport settings."""
        if self.base_url != ORATS_DIVIDEND_ARCHIVE_BASE_URL:
            msg = "ORATS dividend archive base URL must be the explicit production HTTPS endpoint"
            raise OratsDividendArchiveConfigurationError(msg)
        parsed = urlsplit(self.base_url)
        if parsed.scheme != "https" or parsed.netloc != "s3.us-east-1.amazonaws.com":
            msg = "ORATS dividend archive endpoint must use the expected HTTPS origin"
            raise OratsDividendArchiveConfigurationError(msg)
        if not self.timeout_seconds > 0.0:
            msg = "ORATS dividend HTTPS timeout must be positive"
            raise OratsDividendArchiveConfigurationError(msg)
        if self.transfer_block_bytes < _MINIMUM_TRANSFER_BLOCK_BYTES:
            msg = "ORATS dividend transfer block must be at least 65536 bytes"
            raise OratsDividendArchiveConfigurationError(msg)


def orats_dividend_archive_name(year: int) -> str:
    """Return the one canonical annual provider filename."""
    if not _FIRST_ARCHIVE_YEAR <= year <= _MAXIMUM_ARCHIVE_YEAR:
        msg = "ORATS dividend archive year must be in [2007, 9999]"
        raise OratsDividendArchiveConfigurationError(msg)
    return f"{year}ORATSStockDivForecasts.zip"


def orats_dividend_archive_url(year: int) -> str:
    """Return the exact HTTPS identity for one annual archive."""
    return f"{ORATS_DIVIDEND_ARCHIVE_BASE_URL}/{orats_dividend_archive_name(year)}"


@dataclass(frozen=True, slots=True, order=True)
class OratsDividendArchiveEntry:
    """One annual S3 object identity observed with HEAD."""

    year: int
    url: str
    size_bytes: int
    modified_at_utc: datetime
    etag: str

    def __post_init__(self) -> None:
        """Reject an incomplete or internally inconsistent HTTP identity."""
        if self.url != orats_dividend_archive_url(self.year):
            msg = "annual archive URL disagrees with its year"
            raise OratsDividendArchiveIntegrityError(msg, self.url)
        if self.size_bytes <= 0:
            msg = "annual archive size must be positive"
            raise OratsDividendArchiveIntegrityError(msg, self.url)
        if self.modified_at_utc.tzinfo != UTC:
            msg = "annual archive modification time must be timezone-aware UTC"
            raise OratsDividendArchiveIntegrityError(msg, self.url)
        if _ETAG_PATTERN.fullmatch(self.etag) is None:
            msg = "annual archive ETag is malformed or weak"
            raise OratsDividendArchiveIntegrityError(msg, self.url)

    @property
    def local_name(self) -> str:
        """Return the canonical local filename."""
        return orats_dividend_archive_name(self.year)

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical remote identity mapping."""
        return {
            "etag": self.etag,
            "modified_at_utc": self.modified_at_utc.isoformat().replace("+00:00", "Z"),
            "size_bytes": self.size_bytes,
            "url": self.url,
            "year": self.year,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and validate one exact annual entry."""
        expected = {"etag", "modified_at_utc", "size_bytes", "url", "year"}
        mapping = _exact_mapping(raw, expected, "dividend archive entry")
        etag = mapping["etag"]
        modified_raw = mapping["modified_at_utc"]
        size_bytes = mapping["size_bytes"]
        url = mapping["url"]
        year = mapping["year"]
        if type(etag) is not str or type(modified_raw) is not str or type(url) is not str:
            msg = "dividend archive entry fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg)
        if type(size_bytes) is not int or type(year) is not int or not modified_raw.endswith("Z"):
            msg = "dividend archive entry fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg)
        try:
            modified = datetime.fromisoformat(f"{modified_raw[:-1]}+00:00")
        except ValueError as error:
            msg = "dividend archive entry fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg) from error
        return cls(
            year=year,
            url=url,
            size_bytes=size_bytes,
            modified_at_utc=modified,
            etag=etag,
        )


def orats_dividend_archive_entries_sha256(
    entries: Iterable[OratsDividendArchiveEntry],
) -> str:
    """Hash the ordered annual object identities."""
    return hashlib.sha256(_canonical_json_bytes([entry.as_dict() for entry in entries])).hexdigest()


@dataclass(frozen=True, slots=True)
class OratsDividendArchiveInventory:
    """A closed year interval with exact remote object identities."""

    base_url: str
    start_year: int
    end_year: int
    observed_at_utc: datetime
    entries: tuple[OratsDividendArchiveEntry, ...]
    entries_sha256: str
    schema_version: int = ORATS_DIVIDEND_ARCHIVE_INVENTORY_VERSION

    def __post_init__(self) -> None:
        """Validate range completeness, ordering, and content hash."""
        if self.schema_version != ORATS_DIVIDEND_ARCHIVE_INVENTORY_VERSION:
            msg = "unsupported ORATS dividend archive inventory version"
            raise OratsDividendArchiveIntegrityError(msg)
        if self.base_url != ORATS_DIVIDEND_ARCHIVE_BASE_URL:
            msg = "dividend archive inventory base URL is unsupported"
            raise OratsDividendArchiveIntegrityError(msg)
        if not _FIRST_ARCHIVE_YEAR <= self.start_year <= self.end_year <= _MAXIMUM_ARCHIVE_YEAR:
            msg = "dividend archive inventory year range is invalid"
            raise OratsDividendArchiveIntegrityError(msg)
        if self.observed_at_utc.tzinfo != UTC:
            msg = "dividend archive inventory observation time must be UTC"
            raise OratsDividendArchiveIntegrityError(msg)
        expected_years = tuple(range(self.start_year, self.end_year + 1))
        if tuple(entry.year for entry in self.entries) != expected_years:
            msg = "dividend archive inventory must contain exactly one ordered entry per year"
            raise OratsDividendArchiveIntegrityError(msg)
        if self.entries_sha256 != orats_dividend_archive_entries_sha256(self.entries):
            msg = "dividend archive inventory entry hash mismatch"
            raise OratsDividendArchiveIntegrityError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical inventory mapping."""
        return {
            "base_url": self.base_url,
            "end_year": self.end_year,
            "entries": [entry.as_dict() for entry in self.entries],
            "entries_sha256": self.entries_sha256,
            "observed_at_utc": self.observed_at_utc.isoformat().replace("+00:00", "Z"),
            "schema_version": self.schema_version,
            "start_year": self.start_year,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and rehash one exact inventory."""
        expected = {
            "base_url",
            "end_year",
            "entries",
            "entries_sha256",
            "observed_at_utc",
            "schema_version",
            "start_year",
        }
        mapping = _exact_mapping(raw, expected, "dividend archive inventory")
        base_url = mapping["base_url"]
        end_year = mapping["end_year"]
        entries_hash = mapping["entries_sha256"]
        observed_raw = mapping["observed_at_utc"]
        schema_version = mapping["schema_version"]
        start_year = mapping["start_year"]
        entries_raw = mapping["entries"]
        if type(base_url) is not str or type(entries_hash) is not str:
            msg = "dividend archive inventory fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg)
        if type(observed_raw) is not str or type(entries_raw) is not list:
            msg = "dividend archive inventory fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg)
        if any(type(value) is not int for value in (start_year, end_year, schema_version)):
            msg = "dividend archive inventory fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg)
        if not observed_raw.endswith("Z"):
            msg = "dividend archive inventory fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg)
        try:
            observed = datetime.fromisoformat(f"{observed_raw[:-1]}+00:00")
        except ValueError as error:
            msg = "dividend archive inventory fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg) from error
        return cls(
            base_url=base_url,
            start_year=cast("int", start_year),
            end_year=cast("int", end_year),
            observed_at_utc=observed,
            entries=tuple(
                OratsDividendArchiveEntry.from_dict(value)
                for value in cast("list[object]", entries_raw)
            ),
            entries_sha256=entries_hash,
            schema_version=cast("int", schema_version),
        )


@dataclass(frozen=True, slots=True)
class OratsDividendArchiveReceipt:
    """Cryptographic receipt for one completed annual archive."""

    entry: OratsDividendArchiveEntry
    local_name: str
    sha256: str
    schema_version: int = ORATS_DIVIDEND_ARCHIVE_RECEIPT_VERSION

    def __post_init__(self) -> None:
        """Validate local name and digest identity."""
        if self.schema_version != ORATS_DIVIDEND_ARCHIVE_RECEIPT_VERSION:
            msg = "unsupported ORATS dividend archive receipt version"
            raise OratsDividendArchiveIntegrityError(msg)
        if self.local_name != self.entry.local_name:
            msg = "dividend archive receipt local name is not canonical"
            raise OratsDividendArchiveIntegrityError(msg)
        if not _is_sha256(self.sha256):
            msg = "dividend archive receipt SHA-256 is malformed"
            raise OratsDividendArchiveIntegrityError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical receipt mapping."""
        return {
            "entry": self.entry.as_dict(),
            "local_name": self.local_name,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and validate one exact receipt."""
        expected = {"entry", "local_name", "schema_version", "sha256"}
        mapping = _exact_mapping(raw, expected, "dividend archive receipt")
        local_name = mapping["local_name"]
        schema_version = mapping["schema_version"]
        sha256 = mapping["sha256"]
        if type(local_name) is not str or type(schema_version) is not int:
            msg = "dividend archive receipt fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg)
        if type(sha256) is not str:
            msg = "dividend archive receipt fields are malformed"
            raise OratsDividendArchiveIntegrityError(msg)
        return cls(
            entry=OratsDividendArchiveEntry.from_dict(mapping["entry"]),
            local_name=local_name,
            sha256=sha256,
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class VerifiedOratsDividendArchive:
    """An annual archive whose bytes match its immutable receipt."""

    path: Path
    receipt: OratsDividendArchiveReceipt


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        del req, fp, code, msg, headers, newurl
        return None


class OratsDividendHttpsClient:
    """Strict HTTPS HEAD/GET client with byte-exact range resume."""

    def __init__(self, spec: OratsDividendHttpsSpec) -> None:
        """Build one certificate-validating TLS opener without making a request."""
        self._spec = spec
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        self._opener = urllib.request.build_opener(
            _RejectRedirects(),
            urllib.request.HTTPSHandler(context=context),
        )

    def _open(self, request: urllib.request.Request) -> HTTPResponse:
        try:
            return cast(
                "HTTPResponse",
                self._opener.open(
                    request,
                    timeout=self._spec.timeout_seconds,
                ),
            )
        except (OSError, urllib.error.HTTPError, urllib.error.URLError, ssl.SSLError) as error:
            raise OratsDividendArchiveTransportError(
                request.get_method(),
                request.full_url,
                str(error),
            ) from error

    def _head(self, year: int) -> OratsDividendArchiveEntry:
        url = orats_dividend_archive_url(year)
        request = urllib.request.Request(  # noqa: S310 - exact validated provider HTTPS URL
            url,
            headers={"Accept-Encoding": "identity"},
            method="HEAD",
        )
        with self._open(request) as response:
            return _entry_from_response(year, response, expected_status=200)

    def inventory(
        self,
        *,
        start_year: int,
        end_year: int,
        observed_at_utc: datetime | None = None,
    ) -> OratsDividendArchiveInventory:
        """HEAD and hash exactly one annual archive for every requested year."""
        if not _FIRST_ARCHIVE_YEAR <= start_year <= end_year <= _MAXIMUM_ARCHIVE_YEAR:
            msg = "ORATS dividend archive inventory year range is invalid"
            raise OratsDividendArchiveConfigurationError(msg)
        observed = observed_at_utc or datetime.now(UTC)
        if observed.tzinfo != UTC:
            msg = "ORATS dividend archive observation time must be UTC"
            raise OratsDividendArchiveConfigurationError(msg)
        entries = tuple(self._head(year) for year in range(start_year, end_year + 1))
        return OratsDividendArchiveInventory(
            base_url=self._spec.base_url,
            start_year=start_year,
            end_year=end_year,
            observed_at_utc=observed,
            entries=entries,
            entries_sha256=orats_dividend_archive_entries_sha256(entries),
        )

    def _verify_remote_entry(self, expected: OratsDividendArchiveEntry) -> None:
        if self._head(expected.year) != expected:
            msg = "remote annual archive identity changed after inventory"
            raise OratsDividendArchiveIntegrityError(msg, expected.url)

    def _existing_receipt(
        self,
        entry: OratsDividendArchiveEntry,
        final: Path,
        receipt_path: Path,
    ) -> OratsDividendArchiveReceipt | None:
        if not final.exists() and not receipt_path.exists():
            return None
        if not final.is_file() or final.is_symlink():
            msg = "completed annual archive must be a non-symlink file"
            raise OratsDividendArchiveIntegrityError(msg, str(final))
        if not receipt_path.is_file() or receipt_path.is_symlink():
            msg = "completed annual archive receipt must be a non-symlink file"
            raise OratsDividendArchiveIntegrityError(msg, str(receipt_path))
        receipt = _read_receipt(receipt_path)
        if receipt.entry != entry or receipt.local_name != final.name:
            msg = "existing annual archive receipt disagrees with the requested remote identity"
            raise OratsDividendArchiveIntegrityError(msg, str(receipt_path))
        if final.stat().st_size != entry.size_bytes:
            msg = "existing annual archive size disagrees with its receipt"
            raise OratsDividendArchiveIntegrityError(msg, str(final))
        if _file_sha256(final, self._spec.transfer_block_bytes) != receipt.sha256:
            msg = "existing annual archive bytes disagree with its receipt SHA-256"
            raise OratsDividendArchiveIntegrityError(msg, str(final))
        return receipt

    @staticmethod
    def _partial_offset(partial: Path, entry: OratsDividendArchiveEntry) -> int:
        if not partial.exists():
            return 0
        if not partial.is_file() or partial.is_symlink():
            msg = "partial annual archive must be a non-symlink file"
            raise OratsDividendArchiveIntegrityError(msg, str(partial))
        offset = partial.stat().st_size
        if offset > entry.size_bytes:
            msg = "partial annual archive is larger than the remote object"
            raise OratsDividendArchiveIntegrityError(msg, str(partial))
        return offset

    def _transfer(
        self,
        entry: OratsDividendArchiveEntry,
        partial: Path,
        offset: int,
    ) -> None:
        if offset == entry.size_bytes:
            return
        headers = {"Accept-Encoding": "identity", "If-Match": entry.etag}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(  # noqa: S310 - exact validated provider HTTPS URL
            entry.url,
            headers=headers,
            method="GET",
        )
        expected_status = 206 if offset else 200
        mode = "ab" if partial.exists() else "xb"
        with self._open(request) as response:
            _validate_get_response(response, entry, offset, expected_status)
            written = offset
            try:
                with partial.open(mode) as destination:
                    while block := response.read(self._spec.transfer_block_bytes):
                        if written + len(block) > entry.size_bytes:
                            msg = "HTTPS response exceeds the inventoried byte count"
                            raise OratsDividendArchiveIntegrityError(msg, entry.url)
                        destination.write(block)
                        written += len(block)
                    destination.flush()
                    os.fsync(destination.fileno())
            except (OSError, http.client.HTTPException) as error:
                method = "GET"
                raise OratsDividendArchiveTransportError(method, entry.url, str(error)) from error

    def download(
        self,
        entry: OratsDividendArchiveEntry,
        destination_directory: str | Path,
    ) -> OratsDividendArchiveReceipt:
        """Resume and cryptographically receipt one unchanged annual object."""
        self._verify_remote_entry(entry)
        directory = Path(destination_directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir() or directory.is_symlink():
            msg = "annual archive destination must be a non-symlink directory"
            raise OratsDividendArchiveIntegrityError(msg, str(directory))
        final = directory / entry.local_name
        partial = directory / f".{entry.local_name}.part"
        receipt_path = directory / f"{entry.local_name}.receipt.json"
        existing = self._existing_receipt(entry, final, receipt_path)
        if existing is not None:
            return existing
        offset = self._partial_offset(partial, entry)
        self._transfer(entry, partial, offset)
        self._verify_remote_entry(entry)
        return self._finalize(entry, partial, final, receipt_path)

    def _finalize(
        self,
        entry: OratsDividendArchiveEntry,
        partial: Path,
        final: Path,
        receipt_path: Path,
    ) -> OratsDividendArchiveReceipt:
        if partial.stat().st_size != entry.size_bytes:
            msg = "downloaded annual archive size disagrees with its inventory"
            raise OratsDividendArchiveIntegrityError(msg, str(partial))
        receipt = OratsDividendArchiveReceipt(
            entry=entry,
            local_name=final.name,
            sha256=_file_sha256(partial, self._spec.transfer_block_bytes),
        )
        partial.replace(final)
        try:
            _write_json_atomic(receipt_path, receipt.as_dict())
        except BaseException:
            final.replace(partial)
            raise
        return receipt


def _entry_from_response(
    year: int,
    response: HTTPResponse,
    *,
    expected_status: int,
) -> OratsDividendArchiveEntry:
    url = orats_dividend_archive_url(year)
    if response.status != expected_status or response.url != url:
        msg = "HTTPS response status or final URL disagrees with the exact request"
        raise OratsDividendArchiveIntegrityError(msg, url)
    _validate_identity_encoding(response.headers, url)
    if _single_header(response.headers, "Accept-Ranges", url) != "bytes":
        msg = "annual archive endpoint does not explicitly support byte ranges"
        raise OratsDividendArchiveIntegrityError(msg, url)
    size_raw = _single_header(response.headers, "Content-Length", url)
    if not size_raw.isascii() or not size_raw.isdecimal() or int(size_raw) <= 0:
        msg = "annual archive Content-Length is malformed"
        raise OratsDividendArchiveIntegrityError(msg, url)
    modified = _parse_http_datetime(_single_header(response.headers, "Last-Modified", url), url)
    etag = _single_header(response.headers, "ETag", url)
    return OratsDividendArchiveEntry(
        year=year,
        url=url,
        size_bytes=int(size_raw),
        modified_at_utc=modified,
        etag=etag,
    )


def _validate_get_response(
    response: HTTPResponse,
    entry: OratsDividendArchiveEntry,
    offset: int,
    expected_status: int,
) -> None:
    if response.status != expected_status or response.url != entry.url:
        msg = "GET response status or final URL disagrees with the exact request"
        raise OratsDividendArchiveIntegrityError(msg, entry.url)
    _validate_identity_encoding(response.headers, entry.url)
    if _single_header(response.headers, "ETag", entry.url) != entry.etag:
        msg = "GET response ETag disagrees with the inventory"
        raise OratsDividendArchiveIntegrityError(msg, entry.url)
    modified = _parse_http_datetime(
        _single_header(response.headers, "Last-Modified", entry.url),
        entry.url,
    )
    if modified != entry.modified_at_utc:
        msg = "GET response modification time disagrees with the inventory"
        raise OratsDividendArchiveIntegrityError(msg, entry.url)
    expected_length = entry.size_bytes - offset
    length_raw = _single_header(response.headers, "Content-Length", entry.url)
    if not length_raw.isascii() or not length_raw.isdecimal() or int(length_raw) != expected_length:
        msg = "GET response Content-Length disagrees with the requested range"
        raise OratsDividendArchiveIntegrityError(msg, entry.url)
    ranges = response.headers.get_all("Content-Range", failobj=[])
    if offset:
        expected_range = f"bytes {offset}-{entry.size_bytes - 1}/{entry.size_bytes}"
        if ranges != [expected_range]:
            msg = "GET response Content-Range disagrees with the exact resume offset"
            raise OratsDividendArchiveIntegrityError(msg, entry.url)
    elif ranges:
        msg = "full GET response unexpectedly includes Content-Range"
        raise OratsDividendArchiveIntegrityError(msg, entry.url)


def _validate_identity_encoding(headers: Message, location: str) -> None:
    encodings = headers.get_all("Content-Encoding", failobj=[])
    if encodings not in ([], ["identity"]):
        msg = "HTTPS response uses an unsupported content encoding"
        raise OratsDividendArchiveIntegrityError(msg, location)


def _single_header(headers: Message, name: str, location: str) -> str:
    values = headers.get_all(name, failobj=[])
    if len(values) != 1 or not values[0] or values[0].strip() != values[0]:
        msg = f"HTTPS response must contain one canonical {name} header"
        raise OratsDividendArchiveIntegrityError(msg, location)
    return values[0]


def _parse_http_datetime(value: str, location: str) -> datetime:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError) as error:
        msg = "HTTPS Last-Modified is malformed"
        raise OratsDividendArchiveIntegrityError(msg, location) from error
    if parsed.tzinfo is None:
        msg = "HTTPS Last-Modified must be timezone-aware"
        raise OratsDividendArchiveIntegrityError(msg, location)
    return parsed.astimezone(UTC)


def _exact_mapping(raw: object, expected: set[str], label: str) -> dict[str, object]:
    if type(raw) is not dict:
        msg = f"{label} must be a JSON object"
        raise OratsDividendArchiveIntegrityError(msg)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected or any(type(key) is not str for key in mapping):
        msg = f"{label} keys must be exactly {sorted(expected)!r}"
        raise OratsDividendArchiveIntegrityError(msg)
    return cast("dict[str, object]", mapping)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path, block_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    if path.exists():
        msg = "dividend archive JSON artifact already exists"
        raise OratsDividendArchiveIntegrityError(msg, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        msg = "dividend archive JSON partial already exists"
        raise OratsDividendArchiveIntegrityError(msg, str(temporary))
    try:
        with temporary.open("xb") as destination:
            destination.write(_canonical_json_bytes(value) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_orats_dividend_archive_inventory(
    path: str | Path,
    inventory: OratsDividendArchiveInventory,
) -> None:
    """Atomically publish a new annual inventory without overwrite."""
    _write_json_atomic(Path(path).expanduser(), inventory.as_dict())


def read_orats_dividend_archive_inventory(path: str | Path) -> OratsDividendArchiveInventory:
    """Read and fully revalidate one annual inventory."""
    source = Path(path).expanduser()
    if not source.is_file() or source.is_symlink():
        msg = "dividend archive inventory must be an existing non-symlink file"
        raise OratsDividendArchiveIntegrityError(msg, str(source))
    try:
        raw = json.loads(source.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "dividend archive inventory is unreadable"
        raise OratsDividendArchiveIntegrityError(msg, str(source)) from error
    return OratsDividendArchiveInventory.from_dict(raw)


def _read_receipt(path: Path) -> OratsDividendArchiveReceipt:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "dividend archive receipt is unreadable"
        raise OratsDividendArchiveIntegrityError(msg, str(path)) from error
    return OratsDividendArchiveReceipt.from_dict(raw)


def load_verified_orats_dividend_archive(path: str | Path) -> VerifiedOratsDividendArchive:
    """Load and cryptographically verify one annual archive and sidecar receipt."""
    archive = Path(path).expanduser()
    if not archive.is_file() or archive.is_symlink():
        msg = "annual dividend archive must be an existing non-symlink file"
        raise OratsDividendArchiveIntegrityError(msg, str(archive))
    receipt_path = archive.with_name(f"{archive.name}.receipt.json")
    if not receipt_path.is_file() or receipt_path.is_symlink():
        msg = "annual dividend archive receipt must be an existing non-symlink file"
        raise OratsDividendArchiveIntegrityError(msg, str(receipt_path))
    receipt = _read_receipt(receipt_path)
    if receipt.local_name != archive.name or receipt.entry.size_bytes != archive.stat().st_size:
        msg = "annual dividend archive filename or size disagrees with its receipt"
        raise OratsDividendArchiveIntegrityError(msg, str(archive))
    actual_hash = _file_sha256(archive, ORATS_DIVIDEND_ARCHIVE_TRANSFER_BLOCK_BYTES)
    if actual_hash != receipt.sha256:
        msg = "annual dividend archive bytes disagree with its receipt SHA-256"
        raise OratsDividendArchiveIntegrityError(msg, str(archive))
    return VerifiedOratsDividendArchive(path=archive, receipt=receipt)


__all__ = [
    "ORATS_DIVIDEND_ARCHIVE_BASE_URL",
    "ORATS_DIVIDEND_ARCHIVE_INVENTORY_VERSION",
    "ORATS_DIVIDEND_ARCHIVE_RECEIPT_VERSION",
    "ORATS_DIVIDEND_ARCHIVE_TIMEOUT_SECONDS",
    "ORATS_DIVIDEND_ARCHIVE_TRANSFER_BLOCK_BYTES",
    "OratsDividendArchiveConfigurationError",
    "OratsDividendArchiveEntry",
    "OratsDividendArchiveError",
    "OratsDividendArchiveIntegrityError",
    "OratsDividendArchiveInventory",
    "OratsDividendArchiveReceipt",
    "OratsDividendArchiveTransportError",
    "OratsDividendHttpsClient",
    "OratsDividendHttpsSpec",
    "VerifiedOratsDividendArchive",
    "load_verified_orats_dividend_archive",
    "orats_dividend_archive_entries_sha256",
    "orats_dividend_archive_name",
    "orats_dividend_archive_url",
    "read_orats_dividend_archive_inventory",
    "write_orats_dividend_archive_inventory",
]
