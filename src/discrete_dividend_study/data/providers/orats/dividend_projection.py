"""Immutable ticker projection of verified annual ORATS dividend archives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Self, cast

import polars as pl

from discrete_dividend_study.data.providers.orats.dividend_archive import (
    OratsDividendArchiveEntry,
    VerifiedOratsDividendArchive,
)
from discrete_dividend_study.data.providers.orats.dividends import (
    ORATS_DIVIDEND_DTYPES,
    canonicalize_raw_orats_dividend_forecast,
)

ORATS_DIVIDEND_PROJECTION_VERSION: Final = 3
ORATS_DIVIDEND_PROJECTION_RECEIPT_VERSION: Final = 3
ORATS_DIVIDEND_PROJECTION_COMPRESSION: Final = "zstd"
ORATS_DIVIDEND_PROJECTION_COMPRESSION_LEVEL: Final = 3
ORATS_DIVIDEND_PROJECTION_ROW_GROUP_SIZE: Final = 65_536
ORATS_DIVIDEND_PROJECTION_MAX_MEMBERS: Final = 10_000
ORATS_DIVIDEND_PROJECTION_MAX_MEMBER_BYTES: Final = 2 * 1024 * 1024 * 1024
ORATS_DIVIDEND_PROJECTION_MAX_TOTAL_BYTES: Final = 16 * 1024 * 1024 * 1024
ORATS_DIVIDEND_PROJECTION_BLOCK_BYTES: Final = 8 * 1024 * 1024

_TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]*")
_FEED_PATTERN = re.compile(
    r"(?P<date>[0-9]{8})(?:[0-9]{4})?ORATS"
    r"(?P<kind>StockDivForecasts?|Dividends|SpecialDividends|StockDivHistGrowth|StockDivsTrend)"
    r"[.]txt",
)
_REGULAR_KINDS = frozenset({"StockDivForecast", "StockDivForecasts"})
_SHA256_LENGTH = 64
_CRC32_LENGTH = 8
_SCAN_SOURCE_COLUMN = "_source_path"
_SCAN_TERMINAL_COLUMN = "_terminal_field"
_SCAN_DTYPES = {
    "ticker": pl.String,
    "ex_date_raw": pl.String,
    "amount": pl.Float64,
    "frequency_per_year": pl.UInt8,
    _SCAN_TERMINAL_COLUMN: pl.String,
}
OratsDividendUnidentifiedRowPolicy = Literal["reject", "record-and-exclude"]


class OratsDividendProjectionError(RuntimeError):
    """Base class for an invalid annual dividend projection."""


@dataclass(slots=True)
class OratsDividendProjectionSpecError(OratsDividendProjectionError):
    """Projection settings are incomplete, ambiguous, or unsupported."""

    reason: str

    def __str__(self) -> str:
        """Describe the invalid static setting."""
        return self.reason


@dataclass(slots=True)
class OratsDividendProjectionContentError(OratsDividendProjectionError):
    """An annual ZIP, feed member, or output violates its exact identity."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"ORATS dividend projection failed{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OratsDividendProjectionSpec:
    """Exact ticker, resource, and physical Parquet identity."""

    tickers: tuple[str, ...]
    unidentified_row_policy: OratsDividendUnidentifiedRowPolicy = "reject"
    compression: Literal["zstd"] = ORATS_DIVIDEND_PROJECTION_COMPRESSION
    compression_level: int = ORATS_DIVIDEND_PROJECTION_COMPRESSION_LEVEL
    row_group_size: int = ORATS_DIVIDEND_PROJECTION_ROW_GROUP_SIZE
    max_members: int = ORATS_DIVIDEND_PROJECTION_MAX_MEMBERS
    max_member_bytes: int = ORATS_DIVIDEND_PROJECTION_MAX_MEMBER_BYTES
    max_total_uncompressed_bytes: int = ORATS_DIVIDEND_PROJECTION_MAX_TOTAL_BYTES
    schema_version: int = ORATS_DIVIDEND_PROJECTION_VERSION

    def __post_init__(self) -> None:
        """Reject implicit ticker normalization and unsupported physical settings."""
        if self.schema_version != ORATS_DIVIDEND_PROJECTION_VERSION:
            msg = "unsupported ORATS dividend projection version"
            raise OratsDividendProjectionSpecError(msg)
        if not self.tickers:
            msg = "ORATS dividend projection requires at least one ticker"
            raise OratsDividendProjectionSpecError(msg)
        if self.tickers != tuple(sorted(set(self.tickers))):
            msg = "ORATS dividend projection tickers must be sorted and unique"
            raise OratsDividendProjectionSpecError(msg)
        if any(_TICKER_PATTERN.fullmatch(ticker) is None for ticker in self.tickers):
            msg = "ORATS dividend projection tickers must use canonical uppercase symbols"
            raise OratsDividendProjectionSpecError(msg)
        if self.unidentified_row_policy not in ("reject", "record-and-exclude"):
            msg = "unsupported unidentified-row policy"
            raise OratsDividendProjectionSpecError(msg)
        if self.compression != ORATS_DIVIDEND_PROJECTION_COMPRESSION:
            msg = "ORATS dividend projection compression must be zstd"
            raise OratsDividendProjectionSpecError(msg)
        if self.compression_level != ORATS_DIVIDEND_PROJECTION_COMPRESSION_LEVEL:
            msg = "ORATS dividend projection compression level must be 3"
            raise OratsDividendProjectionSpecError(msg)
        if self.row_group_size != ORATS_DIVIDEND_PROJECTION_ROW_GROUP_SIZE:
            msg = "ORATS dividend projection row group size must be 65536"
            raise OratsDividendProjectionSpecError(msg)
        self._validate_bounds()

    def _validate_bounds(self) -> None:
        if not 1 <= self.max_members <= ORATS_DIVIDEND_PROJECTION_MAX_MEMBERS:
            msg = "ORATS dividend projection member bound is invalid"
            raise OratsDividendProjectionSpecError(msg)
        if not 1 <= self.max_member_bytes <= ORATS_DIVIDEND_PROJECTION_MAX_MEMBER_BYTES:
            msg = "ORATS dividend projection per-member byte bound is invalid"
            raise OratsDividendProjectionSpecError(msg)
        if not 1 <= self.max_total_uncompressed_bytes <= ORATS_DIVIDEND_PROJECTION_MAX_TOTAL_BYTES:
            msg = "ORATS dividend projection total byte bound is invalid"
            raise OratsDividendProjectionSpecError(msg)
        if self.max_member_bytes > self.max_total_uncompressed_bytes:
            msg = "ORATS dividend per-member bound must not exceed its total bound"
            raise OratsDividendProjectionSpecError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical projection identity mapping."""
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "max_member_bytes": self.max_member_bytes,
            "max_members": self.max_members,
            "max_total_uncompressed_bytes": self.max_total_uncompressed_bytes,
            "row_group_size": self.row_group_size,
            "schema_version": self.schema_version,
            "tickers": list(self.tickers),
            "unidentified_row_policy": self.unidentified_row_policy,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully validate one exact projection identity."""
        expected = {
            "compression",
            "compression_level",
            "max_member_bytes",
            "max_members",
            "max_total_uncompressed_bytes",
            "row_group_size",
            "schema_version",
            "tickers",
            "unidentified_row_policy",
        }
        mapping = _exact_mapping(raw, expected, "dividend projection spec")
        compression = mapping["compression"]
        tickers_raw = mapping["tickers"]
        unidentified_row_policy = mapping["unidentified_row_policy"]
        integer_names = (
            "compression_level",
            "max_member_bytes",
            "max_members",
            "max_total_uncompressed_bytes",
            "row_group_size",
            "schema_version",
        )
        if (
            type(compression) is not str
            or type(tickers_raw) is not list
            or type(unidentified_row_policy) is not str
        ):
            msg = "dividend projection spec fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_names):
            msg = "dividend projection spec fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        if any(type(ticker) is not str for ticker in tickers_raw):
            msg = "dividend projection spec ticker fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        return cls(
            tickers=tuple(cast("list[str]", tickers_raw)),
            unidentified_row_policy=cast(
                "OratsDividendUnidentifiedRowPolicy",
                unidentified_row_policy,
            ),
            compression=cast("Literal['zstd']", compression),
            compression_level=cast("int", mapping["compression_level"]),
            row_group_size=cast("int", mapping["row_group_size"]),
            max_members=cast("int", mapping["max_members"]),
            max_member_bytes=cast("int", mapping["max_member_bytes"]),
            max_total_uncompressed_bytes=cast("int", mapping["max_total_uncompressed_bytes"]),
            schema_version=cast("int", mapping["schema_version"]),
        )


def orats_dividend_projection_spec_sha256(spec: OratsDividendProjectionSpec) -> str:
    """Return the exact ticker/resource/physical projection identity."""
    return hashlib.sha256(_canonical_json_bytes(spec.as_dict())).hexdigest()


def orats_dividend_schema_sha256() -> str:
    """Return the ordered canonical dividend schema identity."""
    payload = [(name, str(dtype)) for name, dtype in ORATS_DIVIDEND_DTYPES.items()]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class OratsDividendSnapshotReceipt:
    """Exact identity and selected row count for one regular daily snapshot member."""

    forecast_date: date
    member_name: str
    member_crc32: str
    member_compressed_bytes: int
    member_uncompressed_bytes: int
    member_uncompressed_sha256: str
    raw_rows: int
    unidentified_rows: int
    selected_rows: int
    selected_duplicate_rows: int

    def __post_init__(self) -> None:
        """Validate member/date/hash/count identities."""
        match = _FEED_PATTERN.fullmatch(PurePosixPath(self.member_name).name)
        if match is None or match.group("kind") not in _REGULAR_KINDS:
            msg = "snapshot member is not a regular ORATS forecast file"
            raise OratsDividendProjectionContentError(msg, self.member_name)
        if _date_from_digits(match.group("date"), self.member_name) != self.forecast_date:
            msg = "snapshot member date disagrees with its forecast date"
            raise OratsDividendProjectionContentError(msg, self.member_name)
        if len(self.member_crc32) != _CRC32_LENGTH or not _is_lower_hex(self.member_crc32):
            msg = "snapshot member CRC32 is malformed"
            raise OratsDividendProjectionContentError(msg, self.member_name)
        if self.member_compressed_bytes <= 0 or self.member_uncompressed_bytes <= 0:
            msg = "snapshot member byte counts must be positive"
            raise OratsDividendProjectionContentError(msg, self.member_name)
        if not _is_sha256(self.member_uncompressed_sha256):
            msg = "snapshot member SHA-256 is malformed"
            raise OratsDividendProjectionContentError(msg, self.member_name)
        if (
            self.raw_rows <= 0
            or not 0 <= self.unidentified_rows <= self.raw_rows
            or not 0 <= self.selected_rows <= self.raw_rows - self.unidentified_rows
        ):
            msg = "snapshot raw/selected row counts are invalid"
            raise OratsDividendProjectionContentError(msg, self.member_name)
        if (
            not 0 <= self.selected_duplicate_rows <= self.selected_rows
            or self.selected_duplicate_rows == 1
        ):
            msg = "snapshot selected duplicate-row count is invalid"
            raise OratsDividendProjectionContentError(msg, self.member_name)

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical snapshot mapping."""
        return {
            "forecast_date": self.forecast_date.isoformat(),
            "member_compressed_bytes": self.member_compressed_bytes,
            "member_crc32": self.member_crc32,
            "member_name": self.member_name,
            "member_uncompressed_bytes": self.member_uncompressed_bytes,
            "member_uncompressed_sha256": self.member_uncompressed_sha256,
            "raw_rows": self.raw_rows,
            "selected_duplicate_rows": self.selected_duplicate_rows,
            "selected_rows": self.selected_rows,
            "unidentified_rows": self.unidentified_rows,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and validate one exact snapshot receipt."""
        expected = {
            "forecast_date",
            "member_compressed_bytes",
            "member_crc32",
            "member_name",
            "member_uncompressed_bytes",
            "member_uncompressed_sha256",
            "raw_rows",
            "selected_duplicate_rows",
            "selected_rows",
            "unidentified_rows",
        }
        mapping = _exact_mapping(raw, expected, "dividend snapshot receipt")
        forecast_raw = mapping["forecast_date"]
        string_names = ("member_crc32", "member_name", "member_uncompressed_sha256")
        integer_names = (
            "member_compressed_bytes",
            "member_uncompressed_bytes",
            "raw_rows",
            "selected_duplicate_rows",
            "selected_rows",
            "unidentified_rows",
        )
        if type(forecast_raw) is not str or any(
            type(mapping[name]) is not str for name in string_names
        ):
            msg = "dividend snapshot receipt fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_names):
            msg = "dividend snapshot receipt fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        try:
            forecast_date = date.fromisoformat(forecast_raw)
        except ValueError as error:
            msg = "dividend snapshot receipt fields are malformed"
            raise OratsDividendProjectionContentError(msg) from error
        return cls(
            forecast_date=forecast_date,
            member_name=cast("str", mapping["member_name"]),
            member_crc32=cast("str", mapping["member_crc32"]),
            member_compressed_bytes=cast("int", mapping["member_compressed_bytes"]),
            member_uncompressed_bytes=cast("int", mapping["member_uncompressed_bytes"]),
            member_uncompressed_sha256=cast("str", mapping["member_uncompressed_sha256"]),
            raw_rows=cast("int", mapping["raw_rows"]),
            selected_duplicate_rows=cast("int", mapping["selected_duplicate_rows"]),
            selected_rows=cast("int", mapping["selected_rows"]),
            unidentified_rows=cast("int", mapping["unidentified_rows"]),
        )


@dataclass(frozen=True, slots=True)
class OratsDividendProjectionReceipt:
    """Complete annual source/member/spec/output identity."""

    source_entry: OratsDividendArchiveEntry
    source_archive_sha256: str
    projection_spec: OratsDividendProjectionSpec
    projection_spec_sha256: str
    canonical_schema_sha256: str
    snapshots: tuple[OratsDividendSnapshotReceipt, ...]
    output_name: str
    output_rows: int
    output_duplicate_rows: int
    output_bytes: int
    output_sha256: str
    observed_tickers: tuple[str, ...]
    source_unidentified_rows: int
    schema_version: int = ORATS_DIVIDEND_PROJECTION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        """Validate all receipt cross-identities without reading local files."""
        self._validate_source_and_spec()
        self._validate_snapshots()
        self._validate_output()

    def _validate_source_and_spec(self) -> None:
        if self.schema_version != ORATS_DIVIDEND_PROJECTION_RECEIPT_VERSION:
            msg = "unsupported ORATS dividend projection receipt version"
            raise OratsDividendProjectionContentError(msg)
        if not _is_sha256(self.source_archive_sha256):
            msg = "dividend projection source archive SHA-256 is malformed"
            raise OratsDividendProjectionContentError(msg)
        if self.projection_spec_sha256 != orats_dividend_projection_spec_sha256(
            self.projection_spec,
        ):
            msg = "dividend projection spec hash disagrees with the embedded specification"
            raise OratsDividendProjectionContentError(msg)
        if self.canonical_schema_sha256 != orats_dividend_schema_sha256():
            msg = "dividend projection canonical schema hash is unsupported"
            raise OratsDividendProjectionContentError(msg)

    def _validate_snapshots(self) -> None:
        dates = tuple(snapshot.forecast_date for snapshot in self.snapshots)
        if not dates or dates != tuple(sorted(set(dates))):
            msg = "dividend projection snapshots must have unique ordered dates"
            raise OratsDividendProjectionContentError(msg)
        if len(self.snapshots) > self.projection_spec.max_members:
            msg = "dividend projection snapshots exceed the declared member bound"
            raise OratsDividendProjectionContentError(msg)
        if any(
            snapshot.member_uncompressed_bytes > self.projection_spec.max_member_bytes
            or snapshot.member_compressed_bytes > self.source_entry.size_bytes
            for snapshot in self.snapshots
        ):
            msg = "dividend projection snapshot bytes violate source or resource bounds"
            raise OratsDividendProjectionContentError(msg)
        if (
            sum(snapshot.member_uncompressed_bytes for snapshot in self.snapshots)
            > self.projection_spec.max_total_uncompressed_bytes
        ):
            msg = "dividend projection snapshot bytes exceed the declared total bound"
            raise OratsDividendProjectionContentError(msg)
        if any(value.year != self.source_entry.year for value in dates):
            msg = "dividend projection snapshot lies outside its archive year"
            raise OratsDividendProjectionContentError(msg)

    def _validate_output(self) -> None:
        if sum(snapshot.selected_rows for snapshot in self.snapshots) != self.output_rows:
            msg = "dividend projection snapshot rows disagree with output rows"
            raise OratsDividendProjectionContentError(msg)
        if (
            sum(snapshot.selected_duplicate_rows for snapshot in self.snapshots)
            != self.output_duplicate_rows
        ):
            msg = "dividend projection snapshot duplicates disagree with output duplicates"
            raise OratsDividendProjectionContentError(msg)
        if sum(snapshot.unidentified_rows for snapshot in self.snapshots) != (
            self.source_unidentified_rows
        ):
            msg = "dividend projection snapshot unidentified rows disagree with their total"
            raise OratsDividendProjectionContentError(msg)
        if (
            self.projection_spec.unidentified_row_policy == "reject"
            and self.source_unidentified_rows != 0
        ):
            msg = "reject policy cannot receipt unidentified source rows"
            raise OratsDividendProjectionContentError(msg)
        if self.output_name != f"{self.source_entry.year}.parquet":
            msg = "dividend projection output name disagrees with its archive year"
            raise OratsDividendProjectionContentError(msg)
        if (
            self.output_rows <= 0
            or not 0 <= self.output_duplicate_rows <= self.output_rows
            or self.output_duplicate_rows == 1
            or self.source_unidentified_rows < 0
            or self.output_bytes <= 0
            or not _is_sha256(self.output_sha256)
        ):
            msg = "dividend projection output identity is malformed"
            raise OratsDividendProjectionContentError(msg)
        if self.observed_tickers != tuple(sorted(set(self.observed_tickers))):
            msg = "dividend projection observed tickers must be sorted and unique"
            raise OratsDividendProjectionContentError(msg)
        if not self.observed_tickers or not set(self.observed_tickers).issubset(
            self.projection_spec.tickers,
        ):
            msg = "dividend projection observed tickers must be a nonempty declared subset"
            raise OratsDividendProjectionContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical annual receipt mapping."""
        return {
            "canonical_schema_sha256": self.canonical_schema_sha256,
            "observed_tickers": list(self.observed_tickers),
            "output_bytes": self.output_bytes,
            "output_duplicate_rows": self.output_duplicate_rows,
            "output_name": self.output_name,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "projection_spec": self.projection_spec.as_dict(),
            "projection_spec_sha256": self.projection_spec_sha256,
            "schema_version": self.schema_version,
            "snapshots": [snapshot.as_dict() for snapshot in self.snapshots],
            "source_archive_sha256": self.source_archive_sha256,
            "source_entry": self.source_entry.as_dict(),
            "source_unidentified_rows": self.source_unidentified_rows,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully validate one exact annual receipt."""
        expected = {
            "canonical_schema_sha256",
            "observed_tickers",
            "output_bytes",
            "output_duplicate_rows",
            "output_name",
            "output_rows",
            "output_sha256",
            "projection_spec",
            "projection_spec_sha256",
            "schema_version",
            "snapshots",
            "source_archive_sha256",
            "source_entry",
            "source_unidentified_rows",
        }
        mapping = _exact_mapping(raw, expected, "dividend projection receipt")
        string_names = (
            "canonical_schema_sha256",
            "output_name",
            "output_sha256",
            "projection_spec_sha256",
            "source_archive_sha256",
        )
        integer_names = (
            "output_bytes",
            "output_duplicate_rows",
            "output_rows",
            "schema_version",
            "source_unidentified_rows",
        )
        if any(type(mapping[name]) is not str for name in string_names):
            msg = "dividend projection receipt fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_names):
            msg = "dividend projection receipt fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        snapshots_raw = mapping["snapshots"]
        tickers_raw = mapping["observed_tickers"]
        if type(snapshots_raw) is not list or type(tickers_raw) is not list:
            msg = "dividend projection receipt fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        if any(type(ticker) is not str for ticker in tickers_raw):
            msg = "dividend projection receipt ticker fields are malformed"
            raise OratsDividendProjectionContentError(msg)
        return cls(
            source_entry=OratsDividendArchiveEntry.from_dict(mapping["source_entry"]),
            source_archive_sha256=cast("str", mapping["source_archive_sha256"]),
            projection_spec=OratsDividendProjectionSpec.from_dict(mapping["projection_spec"]),
            projection_spec_sha256=cast("str", mapping["projection_spec_sha256"]),
            canonical_schema_sha256=cast("str", mapping["canonical_schema_sha256"]),
            snapshots=tuple(
                OratsDividendSnapshotReceipt.from_dict(value)
                for value in cast("list[object]", snapshots_raw)
            ),
            output_name=cast("str", mapping["output_name"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_duplicate_rows=cast("int", mapping["output_duplicate_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            observed_tickers=tuple(cast("list[str]", tickers_raw)),
            source_unidentified_rows=cast("int", mapping["source_unidentified_rows"]),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class MaterializedOratsDividendProjection:
    """One verified annual Parquet output and complete receipt."""

    path: Path
    receipt: OratsDividendProjectionReceipt


@dataclass(frozen=True, slots=True)
class _ExtractedSnapshot:
    forecast_date: date
    member_name: str
    path: Path
    crc32: str
    compressed_bytes: int
    uncompressed_bytes: int
    uncompressed_sha256: str


@dataclass(frozen=True, slots=True)
class _SourceRowCounts:
    raw_rows: int
    unidentified_rows: int


@dataclass(frozen=True, slots=True)
class _ProjectionDestination:
    output: Path
    receipt: Path


def materialize_orats_dividend_projection(
    source: VerifiedOratsDividendArchive,
    spec: OratsDividendProjectionSpec,
    output_root: str | Path,
    scratch_root: str | Path,
) -> MaterializedOratsDividendProjection:
    """Project one verified annual ZIP into an immutable canonical ticker Parquet."""
    output_directory = _output_directory(output_root, spec)
    output_path = output_directory / f"{source.receipt.entry.year}.parquet"
    receipt_path = output_path.with_name(f"{output_path.name}.receipt.json")
    existing = _load_existing(source, spec, output_path, receipt_path)
    if existing is not None:
        return existing

    scratch_parent = _scratch_parent(scratch_root)
    scratch = scratch_parent / (
        f".orats-dividend-{source.receipt.entry.year}-{source.receipt.sha256[:16]}.scratch"
    )
    if scratch.exists() or scratch.is_symlink():
        msg = "dividend projection scratch directory already exists"
        raise OratsDividendProjectionContentError(msg, str(scratch))
    scratch.mkdir()
    try:
        extracted = _extract_regular_snapshots(source, spec, scratch)
        _verify_source_after_extraction(source)
        frame, raw_rows = _collect_projection(extracted, spec)
        snapshots = _snapshot_receipts(extracted, frame, raw_rows)
        return _write_projection(
            source,
            spec,
            frame,
            snapshots,
            _ProjectionDestination(output=output_path, receipt=receipt_path),
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=False)


def load_verified_orats_dividend_projection(
    path: str | Path,
) -> MaterializedOratsDividendProjection:
    """Load and verify one annual projected output without requiring the raw ZIP."""
    output_path = Path(path).expanduser()
    receipt_path = output_path.with_name(f"{output_path.name}.receipt.json")
    if not receipt_path.is_file() or receipt_path.is_symlink():
        msg = "annual dividend projection receipt must be a non-symlink file"
        raise OratsDividendProjectionContentError(msg, str(receipt_path))
    receipt = _read_receipt(receipt_path)
    return _verify_materialized_output(output_path, receipt)


def _extract_regular_snapshots(
    source: VerifiedOratsDividendArchive,
    spec: OratsDividendProjectionSpec,
    scratch: Path,
) -> tuple[_ExtractedSnapshot, ...]:
    try:
        with zipfile.ZipFile(source.path, mode="r") as archive:
            regular = _validate_zip_members(archive.infolist(), source, spec)
            extracted = tuple(
                _extract_snapshot(archive, member, source, scratch) for member in regular
            )
    except OratsDividendProjectionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        msg = "annual dividend ZIP could not be read and CRC-verified"
        raise OratsDividendProjectionContentError(msg, str(source.path)) from error
    return tuple(sorted(extracted, key=lambda item: item.forecast_date))


def _validate_zip_members(
    members: list[zipfile.ZipInfo],
    source: VerifiedOratsDividendArchive,
    spec: OratsDividendProjectionSpec,
) -> tuple[zipfile.ZipInfo, ...]:
    if not 1 <= len(members) <= spec.max_members:
        msg = "annual dividend ZIP violates the explicit member-count bound"
        raise OratsDividendProjectionContentError(msg, str(source.path))
    total_bytes = sum(member.file_size for member in members)
    if not 1 <= total_bytes <= spec.max_total_uncompressed_bytes:
        msg = "annual dividend ZIP violates the explicit total uncompressed byte bound"
        raise OratsDividendProjectionContentError(msg, str(source.path))
    regular: list[tuple[date, zipfile.ZipInfo]] = []
    for member in members:
        _validate_safe_member(member, source, spec)
        if member.is_dir() or _is_macos_metadata(member.filename):
            continue
        forecast_date, kind = _validate_feed_member(member, source)
        if kind in _REGULAR_KINDS:
            if member.file_size <= 0 or member.compress_size <= 0:
                msg = "regular dividend snapshot member must have positive byte counts"
                raise OratsDividendProjectionContentError(msg, member.filename)
            regular.append((forecast_date, member))
    dates = tuple(item[0] for item in regular)
    if not dates or len(dates) != len(set(dates)):
        msg = "annual dividend ZIP must contain one unique regular snapshot per forecast date"
        raise OratsDividendProjectionContentError(msg, str(source.path))
    return tuple(member for _forecast_date, member in sorted(regular))


def _validate_safe_member(
    member: zipfile.ZipInfo,
    source: VerifiedOratsDividendArchive,
    spec: OratsDividendProjectionSpec,
) -> None:
    path = PurePosixPath(member.filename)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        msg = "annual dividend ZIP contains an unsafe member path"
        raise OratsDividendProjectionContentError(msg, member.filename)
    if member.flag_bits & 0x1:
        msg = "encrypted annual dividend ZIP members are unsupported"
        raise OratsDividendProjectionContentError(msg, member.filename)
    if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK:
        msg = "symbolic-link annual dividend ZIP members are forbidden"
        raise OratsDividendProjectionContentError(msg, member.filename)
    if not 0 <= member.file_size <= spec.max_member_bytes:
        msg = "annual dividend ZIP member violates the per-member byte bound"
        raise OratsDividendProjectionContentError(msg, member.filename)
    if member.compress_size < 0 or member.compress_size > source.receipt.entry.size_bytes:
        msg = "annual dividend ZIP member compressed size is inconsistent"
        raise OratsDividendProjectionContentError(msg, member.filename)


def _validate_feed_member(
    member: zipfile.ZipInfo,
    source: VerifiedOratsDividendArchive,
) -> tuple[date, str]:
    year = source.receipt.entry.year
    path = PurePosixPath(member.filename)
    expected_directory = f"{year}ORATSStockDivForecasts"
    if path.parts not in ((path.name,), (expected_directory, path.name)):
        msg = "annual dividend feed member has an unsupported directory layout"
        raise OratsDividendProjectionContentError(msg, member.filename)
    match = _FEED_PATTERN.fullmatch(path.name)
    if match is None:
        msg = "annual dividend ZIP contains an unsupported provider feed member"
        raise OratsDividendProjectionContentError(msg, member.filename)
    forecast_date = _date_from_digits(match.group("date"), member.filename)
    if forecast_date.year != year:
        msg = "annual dividend feed member date lies outside its archive year"
        raise OratsDividendProjectionContentError(msg, member.filename)
    return forecast_date, match.group("kind")


def _extract_snapshot(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    source: VerifiedOratsDividendArchive,
    scratch: Path,
) -> _ExtractedSnapshot:
    forecast_date, _kind = _validate_feed_member(member, source)
    destination = scratch / PurePosixPath(member.filename).name
    digest = hashlib.sha256()
    written = 0
    with archive.open(member, mode="r") as member_source, destination.open("xb") as member_output:
        while block := member_source.read(ORATS_DIVIDEND_PROJECTION_BLOCK_BYTES):
            member_output.write(block)
            digest.update(block)
            written += len(block)
        member_output.flush()
        os.fsync(member_output.fileno())
    if written != member.file_size:
        msg = "extracted dividend snapshot bytes disagree with ZIP metadata"
        raise OratsDividendProjectionContentError(msg, member.filename)
    return _ExtractedSnapshot(
        forecast_date=forecast_date,
        member_name=member.filename,
        path=destination,
        crc32=f"{member.CRC:08x}",
        compressed_bytes=member.compress_size,
        uncompressed_bytes=member.file_size,
        uncompressed_sha256=digest.hexdigest(),
    )


def _scan_extracted_snapshots(extracted: tuple[_ExtractedSnapshot, ...]) -> pl.LazyFrame:
    return pl.scan_csv(
        [item.path for item in extracted],
        has_header=False,
        separator=" ",
        quote_char=None,
        schema=_SCAN_DTYPES,
        ignore_errors=False,
        infer_schema=False,
        encoding="utf8",
        low_memory=False,
        rechunk=False,
        try_parse_dates=False,
        glob=False,
        raise_if_empty=True,
        truncate_ragged_lines=True,
        missing_columns="insert",
        include_file_paths=_SCAN_SOURCE_COLUMN,
    )


def _collect_projection(
    extracted: tuple[_ExtractedSnapshot, ...],
    spec: OratsDividendProjectionSpec,
) -> tuple[pl.DataFrame, dict[Path, _SourceRowCounts]]:
    scan = _scan_extracted_snapshots(extracted)
    try:
        structure = (
            scan.group_by(_SCAN_SOURCE_COLUMN)
            .agg(
                pl.len().alias("raw_rows"),
                (
                    pl.col("ticker").is_null()
                    | pl.col("ticker").str.strip_chars().eq("").fill_null(value=False)
                )
                .sum()
                .alias("unidentified_rows"),
                pl.any_horizontal(
                    pl.col("ex_date_raw").is_null(),
                    pl.col("amount").is_null(),
                    pl.col("frequency_per_year").is_null(),
                )
                .sum()
                .alias("rows_with_missing_financial_fields"),
                pl.col(_SCAN_TERMINAL_COLUMN).is_not_null().sum().alias("rows_with_extra_fields"),
            )
            .collect(engine="streaming")
        )
        raw_rows = _validate_structure(structure, extracted, spec)
        selected = (
            scan.filter(pl.col("ticker").is_in(spec.tickers))
            .select(
                "ticker",
                "ex_date_raw",
                "amount",
                "frequency_per_year",
                pl.col(_SCAN_SOURCE_COLUMN)
                .str.extract(r"([0-9]{8})(?:[0-9]{4})?ORATSStockDivForecasts?[.]txt$", 1)
                .str.strptime(pl.Date, format="%Y%m%d", strict=True, exact=True)
                .alias("forecast_date"),
            )
            .collect(engine="streaming")
        )
    except pl.exceptions.PolarsError as error:
        msg = "annual dividend snapshot parse failed under the strict schema"
        raise OratsDividendProjectionContentError(msg) from error
    if selected.is_empty():
        msg = "annual dividend archive contains no rows for the declared ticker projection"
        raise OratsDividendProjectionContentError(msg)
    canonical = canonicalize_raw_orats_dividend_forecast(
        selected.lazy(),
        duplicate_policy="preserve",
    ).collect()
    if canonical.schema != pl.Schema(ORATS_DIVIDEND_DTYPES):
        msg = "annual dividend projection materialized an unsupported schema"
        raise OratsDividendProjectionContentError(msg)
    return (
        canonical.sort(
            "forecast_date",
            "ticker",
            "ex_date",
            "frequency_per_year",
            "amount",
        ),
        raw_rows,
    )


def _validate_structure(
    structure: pl.DataFrame,
    extracted: tuple[_ExtractedSnapshot, ...],
    spec: OratsDividendProjectionSpec,
) -> dict[Path, _SourceRowCounts]:
    expected = {item.path for item in extracted}
    observed = {Path(value) for value in structure.get_column(_SCAN_SOURCE_COLUMN).to_list()}
    if observed != expected:
        msg = "one or more regular dividend snapshots contain no parseable rows"
        raise OratsDividendProjectionContentError(msg)
    missing = structure.get_column("rows_with_missing_financial_fields").sum()
    extra = structure.get_column("rows_with_extra_fields").sum()
    if missing != 0 or extra != 0:
        msg = "regular dividend snapshots are not exactly four space-delimited fields"
        raise OratsDividendProjectionContentError(msg)
    unidentified = structure.get_column("unidentified_rows").sum()
    if unidentified != 0 and spec.unidentified_row_policy == "reject":
        msg = "regular dividend snapshots contain unidentified rows under the reject policy"
        raise OratsDividendProjectionContentError(msg)
    return {
        Path(path): _SourceRowCounts(raw_rows=rows, unidentified_rows=unidentified_rows)
        for path, rows, unidentified_rows in structure.select(
            _SCAN_SOURCE_COLUMN,
            "raw_rows",
            "unidentified_rows",
        ).iter_rows()
    }


def _snapshot_receipts(
    extracted: tuple[_ExtractedSnapshot, ...],
    frame: pl.DataFrame,
    raw_rows: dict[Path, _SourceRowCounts],
) -> tuple[OratsDividendSnapshotReceipt, ...]:
    selected_diagnostics = {
        forecast_date: (selected_rows, selected_duplicate_rows)
        for forecast_date, selected_rows, selected_duplicate_rows in (
            frame.group_by("forecast_date")
            .agg(
                pl.len().alias("selected_rows"),
                pl.struct(tuple(ORATS_DIVIDEND_DTYPES))
                .is_duplicated()
                .sum()
                .alias("selected_duplicate_rows"),
            )
            .iter_rows()
        )
    }
    return tuple(
        OratsDividendSnapshotReceipt(
            forecast_date=item.forecast_date,
            member_name=item.member_name,
            member_crc32=item.crc32,
            member_compressed_bytes=item.compressed_bytes,
            member_uncompressed_bytes=item.uncompressed_bytes,
            member_uncompressed_sha256=item.uncompressed_sha256,
            raw_rows=raw_rows[item.path].raw_rows,
            unidentified_rows=raw_rows[item.path].unidentified_rows,
            selected_rows=selected_diagnostics.get(item.forecast_date, (0, 0))[0],
            selected_duplicate_rows=selected_diagnostics.get(item.forecast_date, (0, 0))[1],
        )
        for item in extracted
    )


def _write_projection(
    source: VerifiedOratsDividendArchive,
    spec: OratsDividendProjectionSpec,
    frame: pl.DataFrame,
    snapshots: tuple[OratsDividendSnapshotReceipt, ...],
    destination: _ProjectionDestination,
) -> MaterializedOratsDividendProjection:
    output_path = destination.output
    receipt_path = destination.receipt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.part")
    if partial.exists():
        msg = "annual dividend projection output partial already exists"
        raise OratsDividendProjectionContentError(msg, str(partial))
    try:
        frame.write_parquet(
            partial,
            compression=spec.compression,
            compression_level=spec.compression_level,
            statistics=True,
            row_group_size=spec.row_group_size,
        )
        _fsync_file(partial)
        receipt = OratsDividendProjectionReceipt(
            source_entry=source.receipt.entry,
            source_archive_sha256=source.receipt.sha256,
            projection_spec=spec,
            projection_spec_sha256=orats_dividend_projection_spec_sha256(spec),
            canonical_schema_sha256=orats_dividend_schema_sha256(),
            snapshots=snapshots,
            output_name=output_path.name,
            output_rows=frame.height,
            output_duplicate_rows=sum(snapshot.selected_duplicate_rows for snapshot in snapshots),
            output_bytes=partial.stat().st_size,
            output_sha256=_file_sha256(partial),
            observed_tickers=tuple(sorted(frame.get_column("ticker").unique().to_list())),
            source_unidentified_rows=sum(snapshot.unidentified_rows for snapshot in snapshots),
        )
        partial.replace(output_path)
        try:
            _write_json_atomic(receipt_path, receipt.as_dict())
        except BaseException:
            output_path.replace(partial)
            raise
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return MaterializedOratsDividendProjection(path=output_path, receipt=receipt)


def _load_existing(
    source: VerifiedOratsDividendArchive,
    spec: OratsDividendProjectionSpec,
    output_path: Path,
    receipt_path: Path,
) -> MaterializedOratsDividendProjection | None:
    if not output_path.exists() and not receipt_path.exists():
        return None
    if not output_path.is_file() or output_path.is_symlink():
        msg = "existing annual dividend projection must be a non-symlink file"
        raise OratsDividendProjectionContentError(msg, str(output_path))
    if not receipt_path.is_file() or receipt_path.is_symlink():
        msg = "existing annual dividend projection receipt must be a non-symlink file"
        raise OratsDividendProjectionContentError(msg, str(receipt_path))
    receipt = _read_receipt(receipt_path)
    if receipt.source_entry != source.receipt.entry:
        msg = "existing dividend projection source entry disagrees with the requested archive"
        raise OratsDividendProjectionContentError(msg, str(receipt_path))
    if receipt.source_archive_sha256 != source.receipt.sha256 or receipt.projection_spec != spec:
        msg = "existing dividend projection receipt disagrees with its source or specification"
        raise OratsDividendProjectionContentError(msg, str(receipt_path))
    return _verify_materialized_output(output_path, receipt)


def _verify_materialized_output(
    output_path: Path,
    receipt: OratsDividendProjectionReceipt,
) -> MaterializedOratsDividendProjection:
    if not output_path.is_file() or output_path.is_symlink():
        msg = "annual dividend projection must be a non-symlink file"
        raise OratsDividendProjectionContentError(msg, str(output_path))
    if (
        receipt.output_name != output_path.name
        or receipt.output_bytes != output_path.stat().st_size
    ):
        msg = "annual dividend projection output name or size disagrees with its receipt"
        raise OratsDividendProjectionContentError(msg, str(output_path))
    if receipt.output_sha256 != _file_sha256(output_path):
        msg = "annual dividend projection output bytes disagree with its receipt SHA-256"
        raise OratsDividendProjectionContentError(msg, str(output_path))
    try:
        lazy = pl.scan_parquet(output_path, rechunk=False)
        schema = lazy.collect_schema()
        summary = lazy.select(
            pl.len().alias("rows"),
            pl.col("forecast_date").min().alias("forecast_date_min"),
            pl.col("forecast_date").max().alias("forecast_date_max"),
        ).collect()
        observed_tickers = tuple(
            lazy.select("ticker").unique().sort("ticker").collect().get_column("ticker").to_list(),
        )
    except pl.exceptions.PolarsError as error:
        msg = "annual dividend projection is not readable Parquet"
        raise OratsDividendProjectionContentError(msg, str(output_path)) from error
    if schema != pl.Schema(ORATS_DIVIDEND_DTYPES):
        msg = "annual dividend projection schema disagrees with its receipt contract"
        raise OratsDividendProjectionContentError(msg, str(output_path))
    row = summary.row(0, named=True)
    selected_snapshot_dates = tuple(
        snapshot.forecast_date for snapshot in receipt.snapshots if snapshot.selected_rows > 0
    )
    if not selected_snapshot_dates:
        msg = "annual dividend projection receipt has no selected snapshot dates"
        raise OratsDividendProjectionContentError(msg, str(output_path))
    if (
        row["rows"] != receipt.output_rows
        or row["forecast_date_min"] != selected_snapshot_dates[0]
        or row["forecast_date_max"] != selected_snapshot_dates[-1]
        or observed_tickers != receipt.observed_tickers
    ):
        msg = "annual dividend projection content summary disagrees with its receipt"
        raise OratsDividendProjectionContentError(msg, str(output_path))
    return MaterializedOratsDividendProjection(path=output_path, receipt=receipt)


def _output_directory(root: str | Path, spec: OratsDividendProjectionSpec) -> Path:
    output_root = Path(root).expanduser()
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        msg = "dividend projection output root must be absent or a non-symlink directory"
        raise OratsDividendProjectionContentError(msg, str(output_root))
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        msg = "dividend projection output parent must be an existing non-symlink directory"
        raise OratsDividendProjectionContentError(msg, str(output_root.parent))
    return output_root / orats_dividend_projection_spec_sha256(spec)


def _scratch_parent(root: str | Path) -> Path:
    scratch = Path(root).expanduser()
    if not scratch.is_dir() or scratch.is_symlink():
        msg = "dividend projection scratch root must be an existing non-symlink directory"
        raise OratsDividendProjectionContentError(msg, str(scratch))
    return scratch


def _verify_source_after_extraction(source: VerifiedOratsDividendArchive) -> None:
    if not source.path.is_file() or source.path.is_symlink():
        msg = "annual dividend source archive changed during projection"
        raise OratsDividendProjectionContentError(msg, str(source.path))
    if source.path.stat().st_size != source.receipt.entry.size_bytes:
        msg = "annual dividend source byte count changed during projection"
        raise OratsDividendProjectionContentError(msg, str(source.path))
    if _file_sha256(source.path) != source.receipt.sha256:
        msg = "annual dividend source SHA-256 changed during projection"
        raise OratsDividendProjectionContentError(msg, str(source.path))


def _is_macos_metadata(filename: str) -> bool:
    return PurePosixPath(filename).parts[0] == "__MACOSX"


def _date_from_digits(value: str, location: str) -> date:
    try:
        return date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    except ValueError as error:
        msg = "annual dividend member contains an invalid date"
        raise OratsDividendProjectionContentError(msg, location) from error


def _exact_mapping(raw: object, expected: set[str], label: str) -> dict[str, object]:
    if type(raw) is not dict:
        msg = f"{label} must be a JSON object"
        raise OratsDividendProjectionContentError(msg)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected or any(type(key) is not str for key in mapping):
        msg = f"{label} keys must be exactly {sorted(expected)!r}"
        raise OratsDividendProjectionContentError(msg)
    return cast("dict[str, object]", mapping)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _write_json_atomic(path: Path, value: object) -> None:
    if path.exists():
        msg = "annual dividend projection receipt already exists"
        raise OratsDividendProjectionContentError(msg, str(path))
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        msg = "annual dividend projection receipt partial already exists"
        raise OratsDividendProjectionContentError(msg, str(temporary))
    try:
        with temporary.open("xb") as destination:
            destination.write(_canonical_json_bytes(value) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_receipt(path: Path) -> OratsDividendProjectionReceipt:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "annual dividend projection receipt is unreadable"
        raise OratsDividendProjectionContentError(msg, str(path)) from error
    return OratsDividendProjectionReceipt.from_dict(raw)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(ORATS_DIVIDEND_PROJECTION_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and _is_lower_hex(value)


def _is_lower_hex(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)


__all__ = [
    "ORATS_DIVIDEND_PROJECTION_BLOCK_BYTES",
    "ORATS_DIVIDEND_PROJECTION_COMPRESSION",
    "ORATS_DIVIDEND_PROJECTION_COMPRESSION_LEVEL",
    "ORATS_DIVIDEND_PROJECTION_MAX_MEMBERS",
    "ORATS_DIVIDEND_PROJECTION_MAX_MEMBER_BYTES",
    "ORATS_DIVIDEND_PROJECTION_MAX_TOTAL_BYTES",
    "ORATS_DIVIDEND_PROJECTION_RECEIPT_VERSION",
    "ORATS_DIVIDEND_PROJECTION_ROW_GROUP_SIZE",
    "ORATS_DIVIDEND_PROJECTION_VERSION",
    "MaterializedOratsDividendProjection",
    "OratsDividendProjectionContentError",
    "OratsDividendProjectionError",
    "OratsDividendProjectionReceipt",
    "OratsDividendProjectionSpec",
    "OratsDividendProjectionSpecError",
    "OratsDividendSnapshotReceipt",
    "OratsDividendUnidentifiedRowPolicy",
    "load_verified_orats_dividend_projection",
    "materialize_orats_dividend_projection",
    "orats_dividend_projection_spec_sha256",
    "orats_dividend_schema_sha256",
]
