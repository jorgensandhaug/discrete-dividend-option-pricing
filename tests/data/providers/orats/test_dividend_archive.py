"""Tests for manifest-first HTTPS ORATS dividend archive acquisition."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from email.message import Message
from typing import TYPE_CHECKING, Self, cast
from urllib.parse import urlsplit

import pytest

import discrete_dividend_study.data.providers.orats.dividend_archive as archive_module
from discrete_dividend_study.data.providers.orats.dividend_archive import (
    ORATS_DIVIDEND_ARCHIVE_BASE_URL,
    OratsDividendArchiveConfigurationError,
    OratsDividendArchiveEntry,
    OratsDividendArchiveIntegrityError,
    OratsDividendArchiveInventory,
    OratsDividendHttpsClient,
    OratsDividendHttpsSpec,
    load_verified_orats_dividend_archive,
    orats_dividend_archive_entries_sha256,
    orats_dividend_archive_name,
    orats_dividend_archive_url,
    read_orats_dividend_archive_inventory,
    write_orats_dividend_archive_inventory,
)

if TYPE_CHECKING:
    import urllib.request
    from pathlib import Path
    from types import TracebackType


YEAR = 2024
PAYLOAD = b"synthetic-annual-dividend-archive"
MODIFIED = datetime(2024, 12, 31, 3, 21, tzinfo=UTC)
OBSERVED = datetime(2026, 7, 14, 12, 30, tzinfo=UTC)
ETAG = '"0123456789abcdef0123456789abcdef"'


def _headers(*, offset: int | None = None, accept_ranges: bool = True) -> Message:
    headers = Message()
    headers["Content-Length"] = str(len(PAYLOAD) if offset is None else len(PAYLOAD) - offset)
    headers["Last-Modified"] = "Tue, 31 Dec 2024 03:21:00 GMT"
    headers["ETag"] = ETAG
    if accept_ranges:
        headers["Accept-Ranges"] = "bytes"
    if offset is not None:
        headers["Content-Range"] = f"bytes {offset}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
    return headers


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        url: str,
        headers: Message,
        payload: bytes = b"",
    ) -> None:
        self.status = status
        self.url = url
        self.headers = headers
        self._payload = payload
        self._offset = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._payload) - self._offset
        end = min(self._offset + amount, len(self._payload))
        block = self._payload[self._offset : end]
        self._offset = end
        return block


class _FakeOpener:
    def __init__(self, *, accept_ranges: bool = True, wrong_range_status: bool = False) -> None:
        self.accept_ranges = accept_ranges
        self.wrong_range_status = wrong_range_status
        self.requests: list[tuple[str, dict[str, str]]] = []

    def open(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _FakeResponse:
        assert timeout > 0.0
        method = request.get_method()
        headers = {name.lower(): value for name, value in request.header_items()}
        self.requests.append((method, headers))
        url = request.full_url
        assert urlsplit(url).scheme == "https"
        if method == "HEAD":
            return _FakeResponse(
                status=200,
                url=url,
                headers=_headers(accept_ranges=self.accept_ranges),
            )
        range_header = headers.get("range")
        offset = (
            0
            if range_header is None
            else int(range_header.removeprefix("bytes=").removesuffix("-"))
        )
        status = 200 if offset == 0 or self.wrong_range_status else 206
        return _FakeResponse(
            status=status,
            url=url,
            headers=_headers(offset=offset or None),
            payload=PAYLOAD[offset:],
        )


def _client(fake: _FakeOpener) -> OratsDividendHttpsClient:
    client = OratsDividendHttpsClient(OratsDividendHttpsSpec(transfer_block_bytes=64 * 1024))
    client._opener = cast("urllib.request.OpenerDirector", fake)  # noqa: SLF001 - transport fake
    return client


def _entry() -> OratsDividendArchiveEntry:
    return OratsDividendArchiveEntry(
        year=YEAR,
        url=orats_dividend_archive_url(YEAR),
        size_bytes=len(PAYLOAD),
        modified_at_utc=MODIFIED,
        etag=ETAG,
    )


def test_exact_https_endpoint_and_entry_identity_are_not_configurable() -> None:
    """Protocol, origin, filename, year, timezone, and ETag are explicit invariants."""
    assert orats_dividend_archive_name(YEAR) == "2024ORATSStockDivForecasts.zip"
    assert orats_dividend_archive_url(YEAR).startswith(f"{ORATS_DIVIDEND_ARCHIVE_BASE_URL}/")
    with pytest.raises(OratsDividendArchiveConfigurationError, match="production HTTPS"):
        OratsDividendHttpsSpec(base_url="http://assets.orats.com")
    with pytest.raises(OratsDividendArchiveIntegrityError, match="URL disagrees"):
        OratsDividendArchiveEntry(
            year=YEAR,
            url=f"{ORATS_DIVIDEND_ARCHIVE_BASE_URL}/wrong.zip",
            size_bytes=1,
            modified_at_utc=MODIFIED,
            etag=ETAG,
        )
    with pytest.raises(OratsDividendArchiveIntegrityError, match="ETag"):
        OratsDividendArchiveEntry(
            year=YEAR,
            url=orats_dividend_archive_url(YEAR),
            size_bytes=1,
            modified_at_utc=MODIFIED,
            etag='W/"weak"',
        )


def test_head_inventory_is_complete_hashed_and_immutable(tmp_path: Path) -> None:
    """Exactly one canonical HEAD identity per year is frozen before any GET."""
    fake = _FakeOpener()
    inventory = _client(fake).inventory(
        start_year=YEAR,
        end_year=YEAR,
        observed_at_utc=OBSERVED,
    )
    assert inventory.entries == (_entry(),)
    assert inventory.entries_sha256 == orats_dividend_archive_entries_sha256(inventory.entries)
    assert fake.requests == [("HEAD", {"Accept-encoding".lower(): "identity"})]

    path = tmp_path / "dividend-inventory.json"
    write_orats_dividend_archive_inventory(path, inventory)
    assert read_orats_dividend_archive_inventory(path) == inventory
    with pytest.raises(OratsDividendArchiveIntegrityError, match="already exists"):
        write_orats_dividend_archive_inventory(path, inventory)

    with pytest.raises(OratsDividendArchiveIntegrityError, match="exactly one ordered"):
        OratsDividendArchiveInventory(
            base_url=ORATS_DIVIDEND_ARCHIVE_BASE_URL,
            start_year=YEAR,
            end_year=YEAR + 1,
            observed_at_utc=OBSERVED,
            entries=(_entry(),),
            entries_sha256=orats_dividend_archive_entries_sha256((_entry(),)),
        )


def test_range_resume_hash_receipt_and_exact_reuse(tmp_path: Path) -> None:
    """An interrupted object resumes at its byte count and completed bytes are rehashed."""
    fake = _FakeOpener()
    client = _client(fake)
    partial = tmp_path / f".{orats_dividend_archive_name(YEAR)}.part"
    offset = 9
    partial.write_bytes(PAYLOAD[:offset])

    first = client.download(_entry(), tmp_path)
    final = tmp_path / orats_dividend_archive_name(YEAR)
    assert final.read_bytes() == PAYLOAD
    assert first.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    get_headers = next(headers for method, headers in fake.requests if method == "GET")
    assert get_headers["range"] == f"bytes={offset}-"
    assert get_headers["if-match"] == ETAG

    before = len(fake.requests)
    second = client.download(_entry(), tmp_path)
    assert second == first
    assert all(method == "HEAD" for method, _headers_value in fake.requests[before:])
    assert load_verified_orats_dividend_archive(final).receipt == first

    final.write_bytes(b"x" * len(PAYLOAD))
    with pytest.raises(OratsDividendArchiveIntegrityError, match="receipt SHA-256"):
        client.download(_entry(), tmp_path)


def test_existing_zero_byte_partial_is_a_valid_full_get_target(tmp_path: Path) -> None:
    """A pre-payload interruption resumes without requiring deletion or overwrite."""
    fake = _FakeOpener()
    partial = tmp_path / f".{orats_dividend_archive_name(YEAR)}.part"
    partial.touch()
    receipt = _client(fake).download(_entry(), tmp_path)
    assert receipt.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    get_headers = next(headers for method, headers in fake.requests if method == "GET")
    assert "range" not in get_headers


def test_missing_range_support_and_range_fallback_are_errors(tmp_path: Path) -> None:
    """The client never substitutes a full response when exact resume is unavailable."""
    with pytest.raises(OratsDividendArchiveIntegrityError, match="Accept-Ranges"):
        _client(_FakeOpener(accept_ranges=False)).inventory(
            start_year=YEAR,
            end_year=YEAR,
        )

    partial = tmp_path / f".{orats_dividend_archive_name(YEAR)}.part"
    partial.write_bytes(PAYLOAD[:5])
    with pytest.raises(OratsDividendArchiveIntegrityError, match="status or final URL"):
        _client(_FakeOpener(wrong_range_status=True)).download(_entry(), tmp_path)
    assert partial.read_bytes() == PAYLOAD[:5]


def test_remote_identity_change_stops_before_get(tmp_path: Path) -> None:
    """A stale manifest cannot authorize a changed annual S3 object."""
    changed = OratsDividendArchiveEntry(
        year=YEAR,
        url=orats_dividend_archive_url(YEAR),
        size_bytes=len(PAYLOAD),
        modified_at_utc=MODIFIED,
        etag='"ffffffffffffffffffffffffffffffff"',
    )
    fake = _FakeOpener()
    with pytest.raises(OratsDividendArchiveIntegrityError, match="changed after inventory"):
        _client(fake).download(changed, tmp_path)
    assert all(method == "HEAD" for method, _headers_value in fake.requests)


def test_manifest_parser_rejects_same_shape_hash_tampering(tmp_path: Path) -> None:
    """Changing one remote fact without recomputing identity invalidates the manifest."""
    inventory = _client(_FakeOpener()).inventory(
        start_year=YEAR,
        end_year=YEAR,
        observed_at_utc=OBSERVED,
    )
    raw = inventory.as_dict()
    entries = cast("list[dict[str, object]]", raw["entries"])
    entries[0]["size_bytes"] = len(PAYLOAD) + 1
    with pytest.raises(OratsDividendArchiveIntegrityError, match="entry hash mismatch"):
        OratsDividendArchiveInventory.from_dict(raw)

    path = tmp_path / "not-json.json"
    path.write_text("not JSON")
    with pytest.raises(OratsDividendArchiveIntegrityError, match="unreadable"):
        read_orats_dividend_archive_inventory(path)


def test_response_header_parser_rejects_duplicates() -> None:
    """Ambiguous S3 identity headers never collapse to the first plausible value."""
    headers = _headers()
    headers["ETag"] = ETAG
    response = _FakeResponse(
        status=200,
        url=orats_dividend_archive_url(YEAR),
        headers=headers,
    )
    with pytest.raises(OratsDividendArchiveIntegrityError, match="one canonical ETag"):
        archive_module._entry_from_response(  # noqa: SLF001 - pure header boundary
            YEAR,
            cast("archive_module.HTTPResponse", response),
            expected_status=200,
        )
