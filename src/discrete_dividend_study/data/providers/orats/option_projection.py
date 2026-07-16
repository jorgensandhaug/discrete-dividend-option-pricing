"""Immutable projection of verified ORATS option ZIPs into bounded thesis Parquet files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, Self, cast

import polars as pl

if TYPE_CHECKING:
    from datetime import date

from discrete_dividend_study.data.providers.orats.archive import (
    OratsArchiveEntry,
    VerifiedOratsArchive,
)
from discrete_dividend_study.data.providers.orats.options import (
    ORATS_NORMALIZED_OPTION_DTYPES,
    canonicalize_raw_orats_options,
    scan_raw_orats_csv,
)

ORATS_OPTION_PROJECTION_VERSION: Final = 1
ORATS_OPTION_PROJECTION_RECEIPT_VERSION: Final = 1
ORATS_OPTION_PROJECTION_COMPRESSION: Final = "zstd"
ORATS_OPTION_PROJECTION_COMPRESSION_LEVEL: Final = 3
ORATS_OPTION_PROJECTION_ROW_GROUP_SIZE: Final = 65_536
ORATS_OPTION_PROJECTION_MAX_MEMBER_BYTES: Final = 2 * 1024 * 1024 * 1024
ORATS_OPTION_PROJECTION_BLOCK_BYTES: Final = 8 * 1024 * 1024

_TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]*")
_SHA256_LENGTH = 64
_CRC32_LENGTH = 8


class OratsOptionProjectionError(RuntimeError):
    """Base class for an invalid ZIP projection request or artifact."""


@dataclass(slots=True)
class OratsOptionProjectionSpecError(OratsOptionProjectionError):
    """Projection settings are incomplete, ambiguous, or unsupported."""

    reason: str

    def __str__(self) -> str:
        """Describe the invalid static setting."""
        return self.reason


@dataclass(slots=True)
class OratsOptionProjectionContentError(OratsOptionProjectionError):
    """A verified ZIP, CSV member, or projected frame violates its identity."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"ORATS option projection failed{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OratsOptionProjectionSpec:
    """Exact ticker projection and physical Parquet identity."""

    tickers: tuple[str, ...]
    compression: Literal["zstd"] = ORATS_OPTION_PROJECTION_COMPRESSION
    compression_level: int = ORATS_OPTION_PROJECTION_COMPRESSION_LEVEL
    row_group_size: int = ORATS_OPTION_PROJECTION_ROW_GROUP_SIZE
    max_member_bytes: int = ORATS_OPTION_PROJECTION_MAX_MEMBER_BYTES
    schema_version: int = ORATS_OPTION_PROJECTION_VERSION

    def __post_init__(self) -> None:
        """Reject noncanonical ticker or physical settings."""
        if self.schema_version != ORATS_OPTION_PROJECTION_VERSION:
            msg = "unsupported ORATS option projection schema version"
            raise OratsOptionProjectionSpecError(msg)
        if not self.tickers:
            msg = "ORATS option projection requires at least one ticker"
            raise OratsOptionProjectionSpecError(msg)
        if self.tickers != tuple(sorted(self.tickers)) or len(self.tickers) != len(
            set(self.tickers),
        ):
            msg = "ORATS option projection tickers must be sorted and unique"
            raise OratsOptionProjectionSpecError(msg)
        if any(_TICKER_PATTERN.fullmatch(ticker) is None for ticker in self.tickers):
            msg = "ORATS option projection tickers must use canonical uppercase symbols"
            raise OratsOptionProjectionSpecError(msg)
        if self.compression != ORATS_OPTION_PROJECTION_COMPRESSION:
            msg = "ORATS option projection compression must be zstd"
            raise OratsOptionProjectionSpecError(msg)
        if self.compression_level != ORATS_OPTION_PROJECTION_COMPRESSION_LEVEL:
            msg = "ORATS option projection compression level must be 3"
            raise OratsOptionProjectionSpecError(msg)
        if self.row_group_size != ORATS_OPTION_PROJECTION_ROW_GROUP_SIZE:
            msg = "ORATS option projection row group size must be 65536"
            raise OratsOptionProjectionSpecError(msg)
        if not 1 <= self.max_member_bytes <= ORATS_OPTION_PROJECTION_MAX_MEMBER_BYTES:
            msg = "ORATS option projection member bound must be in [1, 2147483648]"
            raise OratsOptionProjectionSpecError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical projection identity mapping."""
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "max_member_bytes": self.max_member_bytes,
            "row_group_size": self.row_group_size,
            "schema_version": self.schema_version,
            "tickers": list(self.tickers),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and validate an exact projection identity."""
        expected = {
            "compression",
            "compression_level",
            "max_member_bytes",
            "row_group_size",
            "schema_version",
            "tickers",
        }
        mapping = _exact_mapping(raw, expected, "projection spec")
        compression = mapping["compression"]
        compression_level = mapping["compression_level"]
        max_member_bytes = mapping["max_member_bytes"]
        row_group_size = mapping["row_group_size"]
        schema_version = mapping["schema_version"]
        tickers_raw = mapping["tickers"]
        if type(compression) is not str or type(compression_level) is not int:
            msg = "projection spec fields are malformed"
            raise OratsOptionProjectionContentError(msg)
        if type(max_member_bytes) is not int or type(row_group_size) is not int:
            msg = "projection spec fields are malformed"
            raise OratsOptionProjectionContentError(msg)
        if type(schema_version) is not int or type(tickers_raw) is not list:
            msg = "projection spec fields are malformed"
            raise OratsOptionProjectionContentError(msg)
        if any(type(ticker) is not str for ticker in tickers_raw):
            msg = "projection spec ticker fields are malformed"
            raise OratsOptionProjectionContentError(msg)
        return cls(
            tickers=tuple(cast("list[str]", tickers_raw)),
            compression=cast("Literal['zstd']", compression),
            compression_level=compression_level,
            row_group_size=row_group_size,
            max_member_bytes=max_member_bytes,
            schema_version=schema_version,
        )


def orats_option_projection_spec_sha256(spec: OratsOptionProjectionSpec) -> str:
    """Return the exact ticker/physical projection identity."""
    return hashlib.sha256(_canonical_json_bytes(spec.as_dict())).hexdigest()


def orats_normalized_option_schema_sha256() -> str:
    """Return the ordered normalized schema identity."""
    payload = [(name, str(dtype)) for name, dtype in ORATS_NORMALIZED_OPTION_DTYPES.items()]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class OratsOptionProjectionReceipt:
    """Complete source/member/spec/output identity for one daily Parquet projection."""

    source_entry: OratsArchiveEntry
    source_archive_sha256: str
    member_name: str
    member_crc32: str
    member_compressed_bytes: int
    member_uncompressed_bytes: int
    member_uncompressed_sha256: str
    projection_spec: OratsOptionProjectionSpec
    projection_spec_sha256: str
    normalized_schema_sha256: str
    output_name: str
    output_rows: int
    output_bytes: int
    output_sha256: str
    observed_tickers: tuple[str, ...]
    schema_version: int = ORATS_OPTION_PROJECTION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        """Validate all cross-field identities without reading local files."""
        self._validate_source_and_member()
        self._validate_projection_and_output()

    def _validate_source_and_member(self) -> None:
        if self.schema_version != ORATS_OPTION_PROJECTION_RECEIPT_VERSION:
            msg = "unsupported ORATS option projection receipt version"
            raise OratsOptionProjectionContentError(msg)
        expected_member = f"ORATS_SMV_Strikes_{self.source_entry.trade_date:%Y%m%d}.csv"
        if self.member_name != expected_member:
            msg = "projection member name disagrees with the source trade date"
            raise OratsOptionProjectionContentError(msg)
        if len(self.member_crc32) != _CRC32_LENGTH or not _is_lower_hex(self.member_crc32):
            msg = "projection member CRC32 is malformed"
            raise OratsOptionProjectionContentError(msg)
        if self.member_compressed_bytes <= 0 or self.member_uncompressed_bytes <= 0:
            msg = "projection member byte counts must be positive"
            raise OratsOptionProjectionContentError(msg)
        if self.member_compressed_bytes > self.source_entry.size_bytes:
            msg = "projection member compressed bytes exceed the source archive"
            raise OratsOptionProjectionContentError(msg)
        if self.member_uncompressed_bytes > self.projection_spec.max_member_bytes:
            msg = "projection member exceeds the declared uncompressed byte bound"
            raise OratsOptionProjectionContentError(msg)
        for field_name, value in (
            ("source archive", self.source_archive_sha256),
            ("member", self.member_uncompressed_sha256),
        ):
            if len(value) != _SHA256_LENGTH or not _is_lower_hex(value):
                msg = f"projection {field_name} SHA-256 is malformed"
                raise OratsOptionProjectionContentError(msg)

    def _validate_projection_and_output(self) -> None:
        if self.projection_spec_sha256 != orats_option_projection_spec_sha256(
            self.projection_spec,
        ):
            msg = "projection spec hash disagrees with the embedded specification"
            raise OratsOptionProjectionContentError(msg)
        if self.normalized_schema_sha256 != orats_normalized_option_schema_sha256():
            msg = "projection normalized schema hash is unsupported"
            raise OratsOptionProjectionContentError(msg)
        expected_output = f"{self.source_entry.trade_date.isoformat()}.parquet"
        if self.output_name != expected_output:
            msg = "projection output name disagrees with the source trade date"
            raise OratsOptionProjectionContentError(msg)
        if self.output_rows <= 0 or self.output_bytes <= 0:
            msg = "projection output row and byte counts must be positive"
            raise OratsOptionProjectionContentError(msg)
        if self.observed_tickers != tuple(sorted(set(self.observed_tickers))):
            msg = "projection observed tickers must be sorted and unique"
            raise OratsOptionProjectionContentError(msg)
        if not self.observed_tickers or not set(self.observed_tickers).issubset(
            self.projection_spec.tickers,
        ):
            msg = "projection observed tickers must be a nonempty subset of the specification"
            raise OratsOptionProjectionContentError(msg)
        if len(self.output_sha256) != _SHA256_LENGTH or not _is_lower_hex(
            self.output_sha256,
        ):
            msg = "projection output SHA-256 is malformed"
            raise OratsOptionProjectionContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical receipt mapping."""
        return {
            "member_compressed_bytes": self.member_compressed_bytes,
            "member_crc32": self.member_crc32,
            "member_name": self.member_name,
            "member_uncompressed_bytes": self.member_uncompressed_bytes,
            "member_uncompressed_sha256": self.member_uncompressed_sha256,
            "normalized_schema_sha256": self.normalized_schema_sha256,
            "observed_tickers": list(self.observed_tickers),
            "output_bytes": self.output_bytes,
            "output_name": self.output_name,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "projection_spec": self.projection_spec.as_dict(),
            "projection_spec_sha256": self.projection_spec_sha256,
            "schema_version": self.schema_version,
            "source_archive_sha256": self.source_archive_sha256,
            "source_entry": self.source_entry.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully revalidate one projection receipt."""
        expected = {
            "member_compressed_bytes",
            "member_crc32",
            "member_name",
            "member_uncompressed_bytes",
            "member_uncompressed_sha256",
            "normalized_schema_sha256",
            "observed_tickers",
            "output_bytes",
            "output_name",
            "output_rows",
            "output_sha256",
            "projection_spec",
            "projection_spec_sha256",
            "schema_version",
            "source_archive_sha256",
            "source_entry",
        }
        mapping = _exact_mapping(raw, expected, "projection receipt")
        string_fields = (
            "member_crc32",
            "member_name",
            "member_uncompressed_sha256",
            "normalized_schema_sha256",
            "output_name",
            "output_sha256",
            "projection_spec_sha256",
            "source_archive_sha256",
        )
        integer_fields = (
            "member_compressed_bytes",
            "member_uncompressed_bytes",
            "output_bytes",
            "output_rows",
            "schema_version",
        )
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "projection receipt string fields are malformed"
            raise OratsOptionProjectionContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_fields):
            msg = "projection receipt integer fields are malformed"
            raise OratsOptionProjectionContentError(msg)
        observed_raw = mapping["observed_tickers"]
        if type(observed_raw) is not list or any(type(value) is not str for value in observed_raw):
            msg = "projection receipt ticker fields are malformed"
            raise OratsOptionProjectionContentError(msg)
        return cls(
            source_entry=OratsArchiveEntry.from_dict(mapping["source_entry"]),
            source_archive_sha256=cast("str", mapping["source_archive_sha256"]),
            member_name=cast("str", mapping["member_name"]),
            member_crc32=cast("str", mapping["member_crc32"]),
            member_compressed_bytes=cast("int", mapping["member_compressed_bytes"]),
            member_uncompressed_bytes=cast("int", mapping["member_uncompressed_bytes"]),
            member_uncompressed_sha256=cast("str", mapping["member_uncompressed_sha256"]),
            projection_spec=OratsOptionProjectionSpec.from_dict(mapping["projection_spec"]),
            projection_spec_sha256=cast("str", mapping["projection_spec_sha256"]),
            normalized_schema_sha256=cast("str", mapping["normalized_schema_sha256"]),
            output_name=cast("str", mapping["output_name"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            observed_tickers=tuple(cast("list[str]", observed_raw)),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class MaterializedOratsOptionProjection:
    """A verified daily Parquet output and its complete receipt."""

    path: Path
    receipt: OratsOptionProjectionReceipt


def materialize_orats_option_projection(
    source: VerifiedOratsArchive,
    spec: OratsOptionProjectionSpec,
    output_root: str | Path,
    scratch_root: str | Path,
) -> MaterializedOratsOptionProjection:
    """Project one verified raw ZIP to an immutable daily normalized Parquet artifact."""
    output_directory = _output_directory(output_root, spec, source.receipt.entry.trade_date)
    output_path = output_directory / f"{source.receipt.entry.trade_date.isoformat()}.parquet"
    receipt_path = output_path.with_name(f"{output_path.name}.receipt.json")
    existing = _load_existing_projection(source, spec, output_path, receipt_path)
    if existing is not None:
        return existing

    scratch_directory = _scratch_directory(scratch_root)
    csv_path = scratch_directory / f".{source.path.stem}.{source.receipt.sha256[:16]}.scratch.csv"
    if csv_path.exists():
        msg = "projection scratch CSV already exists"
        raise OratsOptionProjectionContentError(msg, str(csv_path))
    try:
        member = _extract_option_member(source, spec, csv_path)
        _verify_source_after_extraction(source)
        frame = _collect_projection(csv_path, source.receipt.entry.trade_date, spec)
        return _write_projection(
            source=source,
            spec=spec,
            member=member,
            frame=frame,
            destination=_ProjectionDestination(output=output_path, receipt=receipt_path),
        )
    finally:
        csv_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _ExtractedMember:
    name: str
    crc32: str
    compressed_bytes: int
    uncompressed_bytes: int
    uncompressed_sha256: str


@dataclass(frozen=True, slots=True)
class _ProjectionDestination:
    output: Path
    receipt: Path


def _extract_option_member(
    source: VerifiedOratsArchive,
    spec: OratsOptionProjectionSpec,
    destination: Path,
) -> _ExtractedMember:
    try:
        with zipfile.ZipFile(source.path, mode="r") as archive:
            members = tuple(member for member in archive.infolist() if not member.is_dir())
            if len(members) != 1:
                msg = "option ZIP must contain exactly one non-directory member"
                raise OratsOptionProjectionContentError(msg, str(source.path))
            member = members[0]
            _validate_member(member, source, spec)
            digest = hashlib.sha256()
            written = 0
            with (
                archive.open(member, mode="r") as member_source,
                destination.open(
                    "xb",
                ) as member_destination,
            ):
                while block := member_source.read(ORATS_OPTION_PROJECTION_BLOCK_BYTES):
                    member_destination.write(block)
                    digest.update(block)
                    written += len(block)
                member_destination.flush()
                os.fsync(member_destination.fileno())
    except OratsOptionProjectionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        msg = "option ZIP could not be read and CRC-verified"
        raise OratsOptionProjectionContentError(msg, str(source.path)) from error
    if written != member.file_size:
        msg = "extracted member byte count disagrees with ZIP metadata"
        raise OratsOptionProjectionContentError(msg, str(destination))
    return _ExtractedMember(
        name=member.filename,
        crc32=f"{member.CRC:08x}",
        compressed_bytes=member.compress_size,
        uncompressed_bytes=member.file_size,
        uncompressed_sha256=digest.hexdigest(),
    )


def _validate_member(
    member: zipfile.ZipInfo,
    source: VerifiedOratsArchive,
    spec: OratsOptionProjectionSpec,
) -> None:
    expected = f"ORATS_SMV_Strikes_{source.receipt.entry.trade_date:%Y%m%d}.csv"
    path = PurePosixPath(member.filename)
    if member.filename != expected or path.name != expected or len(path.parts) != 1:
        msg = "option ZIP member path is not the exact expected daily CSV name"
        raise OratsOptionProjectionContentError(msg, str(source.path))
    if member.flag_bits & 0x1:
        msg = "encrypted ZIP members are unsupported"
        raise OratsOptionProjectionContentError(msg, str(source.path))
    unix_mode = member.external_attr >> 16
    if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
        msg = "symbolic-link ZIP members are forbidden"
        raise OratsOptionProjectionContentError(msg, str(source.path))
    if not 1 <= member.file_size <= spec.max_member_bytes:
        msg = "option ZIP member violates the explicit uncompressed byte bound"
        raise OratsOptionProjectionContentError(msg, str(source.path))
    if not 1 <= member.compress_size <= source.receipt.entry.size_bytes:
        msg = "option ZIP member compressed size is inconsistent with its archive"
        raise OratsOptionProjectionContentError(msg, str(source.path))


def _verify_source_after_extraction(source: VerifiedOratsArchive) -> None:
    if not source.path.is_file() or source.path.is_symlink():
        msg = "source archive changed during projection"
        raise OratsOptionProjectionContentError(msg, str(source.path))
    if source.path.stat().st_size != source.receipt.entry.size_bytes:
        msg = "source archive byte count changed during projection"
        raise OratsOptionProjectionContentError(msg, str(source.path))
    if _file_sha256(source.path) != source.receipt.sha256:
        msg = "source archive SHA-256 changed during projection"
        raise OratsOptionProjectionContentError(msg, str(source.path))


def _collect_projection(
    csv_path: Path,
    trade_date: date,
    spec: OratsOptionProjectionSpec,
) -> pl.DataFrame:
    try:
        frame = canonicalize_raw_orats_options(
            scan_raw_orats_csv(csv_path).filter(pl.col("ticker").is_in(spec.tickers)),
        ).collect(engine="streaming")
    except pl.exceptions.PolarsError as error:
        msg = "option CSV projection failed under the strict parse schema"
        raise OratsOptionProjectionContentError(msg, str(csv_path)) from error
    if frame.is_empty():
        msg = "option CSV contains no rows for the declared ticker projection"
        raise OratsOptionProjectionContentError(msg, str(csv_path))
    if frame.schema != pl.Schema(ORATS_NORMALIZED_OPTION_DTYPES):
        msg = "option CSV projection materialized an unsupported schema"
        raise OratsOptionProjectionContentError(msg, str(csv_path))
    quote_dates = frame.get_column("quote_date").unique().to_list()
    if quote_dates != [trade_date]:
        msg = "option CSV quote dates disagree with the archive trade date"
        raise OratsOptionProjectionContentError(msg, str(csv_path))
    observed = set(frame.get_column("ticker").unique().to_list())
    if not observed or not observed.issubset(spec.tickers):
        msg = "option CSV materialized tickers outside the declared projection"
        raise OratsOptionProjectionContentError(msg, str(csv_path))
    return frame


def _write_projection(
    *,
    source: VerifiedOratsArchive,
    spec: OratsOptionProjectionSpec,
    member: _ExtractedMember,
    frame: pl.DataFrame,
    destination: _ProjectionDestination,
) -> MaterializedOratsOptionProjection:
    output_path = destination.output
    receipt_path = destination.receipt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.part")
    if partial.exists():
        msg = "projection output partial already exists"
        raise OratsOptionProjectionContentError(msg, str(partial))
    try:
        frame.write_parquet(
            partial,
            compression=spec.compression,
            compression_level=spec.compression_level,
            statistics=True,
            row_group_size=spec.row_group_size,
        )
        _fsync_file(partial)
        receipt = OratsOptionProjectionReceipt(
            source_entry=source.receipt.entry,
            source_archive_sha256=source.receipt.sha256,
            member_name=member.name,
            member_crc32=member.crc32,
            member_compressed_bytes=member.compressed_bytes,
            member_uncompressed_bytes=member.uncompressed_bytes,
            member_uncompressed_sha256=member.uncompressed_sha256,
            projection_spec=spec,
            projection_spec_sha256=orats_option_projection_spec_sha256(spec),
            normalized_schema_sha256=orats_normalized_option_schema_sha256(),
            output_name=output_path.name,
            output_rows=frame.height,
            output_bytes=partial.stat().st_size,
            output_sha256=_file_sha256(partial),
            observed_tickers=tuple(sorted(frame.get_column("ticker").unique().to_list())),
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
    return MaterializedOratsOptionProjection(path=output_path, receipt=receipt)


def _load_existing_projection(
    source: VerifiedOratsArchive,
    spec: OratsOptionProjectionSpec,
    output_path: Path,
    receipt_path: Path,
) -> MaterializedOratsOptionProjection | None:
    if not output_path.exists() and not receipt_path.exists():
        return None
    if not output_path.is_file() or output_path.is_symlink():
        msg = "existing projection output must be a non-symlink file"
        raise OratsOptionProjectionContentError(msg, str(output_path))
    if not receipt_path.is_file() or receipt_path.is_symlink():
        msg = "existing projection receipt must be a non-symlink file"
        raise OratsOptionProjectionContentError(msg, str(receipt_path))
    receipt = _read_receipt(receipt_path)
    if receipt.source_entry != source.receipt.entry:
        msg = "existing projection source entry disagrees with the requested archive"
        raise OratsOptionProjectionContentError(msg, str(receipt_path))
    if receipt.source_archive_sha256 != source.receipt.sha256:
        msg = "existing projection source hash disagrees with the requested archive"
        raise OratsOptionProjectionContentError(msg, str(receipt_path))
    if receipt.projection_spec != spec or receipt.output_name != output_path.name:
        msg = "existing projection receipt disagrees with the requested specification"
        raise OratsOptionProjectionContentError(msg, str(receipt_path))
    return _verify_materialized_output(output_path, receipt)


def load_verified_orats_option_projection(
    path: str | Path,
) -> MaterializedOratsOptionProjection:
    """Load and fully verify one daily projection without requiring the raw ZIP."""
    output_path = Path(path).expanduser()
    receipt_path = output_path.with_name(f"{output_path.name}.receipt.json")
    if not receipt_path.is_file() or receipt_path.is_symlink():
        msg = "daily option projection receipt must be a non-symlink file"
        raise OratsOptionProjectionContentError(msg, str(receipt_path))
    receipt = _read_receipt(receipt_path)
    return _verify_materialized_output(output_path, receipt)


def _verify_materialized_output(
    output_path: Path,
    receipt: OratsOptionProjectionReceipt,
) -> MaterializedOratsOptionProjection:
    if not output_path.is_file() or output_path.is_symlink():
        msg = "daily option projection must be a non-symlink file"
        raise OratsOptionProjectionContentError(msg, str(output_path))
    if receipt.output_name != output_path.name:
        msg = "daily option projection name disagrees with its receipt"
        raise OratsOptionProjectionContentError(msg, str(output_path))
    if receipt.output_bytes != output_path.stat().st_size:
        msg = "daily option projection byte count disagrees with its receipt"
        raise OratsOptionProjectionContentError(msg, str(output_path))
    if receipt.output_sha256 != _file_sha256(output_path):
        msg = "daily option projection bytes disagree with its receipt SHA-256"
        raise OratsOptionProjectionContentError(msg, str(output_path))
    try:
        lazy = pl.scan_parquet(output_path, rechunk=False)
        schema = lazy.collect_schema()
        summary = lazy.select(
            pl.len().alias("rows"),
            pl.col("quote_date").n_unique().alias("quote_dates"),
            pl.col("quote_date").min().alias("quote_date_min"),
            pl.col("quote_date").max().alias("quote_date_max"),
        ).collect()
        observed_tickers = tuple(
            lazy.select("ticker").unique().sort("ticker").collect().get_column("ticker").to_list(),
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        msg = "daily option projection is not readable Parquet"
        raise OratsOptionProjectionContentError(msg, str(output_path)) from error
    if schema != pl.Schema(ORATS_NORMALIZED_OPTION_DTYPES):
        msg = "daily option projection schema disagrees with its receipt contract"
        raise OratsOptionProjectionContentError(msg, str(output_path))
    row = summary.row(0, named=True)
    expected_date = receipt.source_entry.trade_date
    if (
        row["rows"] != receipt.output_rows
        or row["quote_dates"] != 1
        or row["quote_date_min"] != expected_date
        or row["quote_date_max"] != expected_date
        or observed_tickers != receipt.observed_tickers
    ):
        msg = "daily option projection content summary disagrees with its receipt"
        raise OratsOptionProjectionContentError(msg, str(output_path))
    return MaterializedOratsOptionProjection(path=output_path, receipt=receipt)


def _output_directory(
    root: str | Path,
    spec: OratsOptionProjectionSpec,
    trade_date: date,
) -> Path:
    output_root = Path(root).expanduser()
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        msg = "projection output root must be absent or a non-symlink directory"
        raise OratsOptionProjectionContentError(msg, str(output_root))
    parent = output_root.parent
    if not parent.is_dir() or parent.is_symlink():
        msg = "projection output parent must be an existing non-symlink directory"
        raise OratsOptionProjectionContentError(msg, str(parent))
    return output_root / orats_option_projection_spec_sha256(spec) / str(trade_date.year)


def _scratch_directory(root: str | Path) -> Path:
    scratch = Path(root).expanduser()
    if not scratch.is_dir() or scratch.is_symlink():
        msg = "projection scratch root must be an existing non-symlink directory"
        raise OratsOptionProjectionContentError(msg, str(scratch))
    return scratch


def _exact_mapping(raw: object, expected: set[str], label: str) -> dict[str, object]:
    if type(raw) is not dict:
        msg = f"{label} must be a JSON object"
        raise OratsOptionProjectionContentError(msg)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected or any(type(key) is not str for key in mapping):
        msg = f"{label} keys must be exactly {sorted(expected)!r}"
        raise OratsOptionProjectionContentError(msg)
    return cast("dict[str, object]", mapping)


def _read_receipt(path: Path) -> OratsOptionProjectionReceipt:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "projection receipt is unreadable"
        raise OratsOptionProjectionContentError(msg, str(path)) from error
    return OratsOptionProjectionReceipt.from_dict(raw)


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
        msg = "projection receipt already exists"
        raise OratsOptionProjectionContentError(msg, str(path))
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        msg = "projection receipt partial already exists"
        raise OratsOptionProjectionContentError(msg, str(temporary))
    payload = _canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(ORATS_OPTION_PROJECTION_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _is_lower_hex(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)


__all__ = [
    "ORATS_OPTION_PROJECTION_BLOCK_BYTES",
    "ORATS_OPTION_PROJECTION_COMPRESSION",
    "ORATS_OPTION_PROJECTION_COMPRESSION_LEVEL",
    "ORATS_OPTION_PROJECTION_MAX_MEMBER_BYTES",
    "ORATS_OPTION_PROJECTION_RECEIPT_VERSION",
    "ORATS_OPTION_PROJECTION_ROW_GROUP_SIZE",
    "ORATS_OPTION_PROJECTION_VERSION",
    "MaterializedOratsOptionProjection",
    "OratsOptionProjectionContentError",
    "OratsOptionProjectionError",
    "OratsOptionProjectionReceipt",
    "OratsOptionProjectionSpec",
    "OratsOptionProjectionSpecError",
    "load_verified_orats_option_projection",
    "materialize_orats_option_projection",
    "orats_normalized_option_schema_sha256",
    "orats_option_projection_spec_sha256",
]
