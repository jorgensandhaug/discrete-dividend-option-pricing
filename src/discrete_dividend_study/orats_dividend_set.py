"""One verified aggregate identity for a complete ORATS dividend projection set."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Self, cast

import polars as pl

from .orats_dividend_archive import (
    OratsDividendArchiveEntry,
    OratsDividendArchiveInventory,
    read_orats_dividend_archive_inventory,
)
from .orats_dividend_projection import (
    MaterializedOratsDividendProjection,
    OratsDividendProjectionReceipt,
    OratsDividendProjectionSpec,
    orats_dividend_projection_spec_sha256,
    orats_dividend_schema_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ORATS_DIVIDEND_PROJECTION_SET_VERSION: Final = 1
ORATS_DIVIDEND_PROJECTION_SET_BLOCK_BYTES: Final = 8 * 1024 * 1024
_SHA256_LENGTH: Final = 64
_FIRST_ARCHIVE_YEAR: Final = 2007
_MAXIMUM_ARCHIVE_YEAR: Final = 9999


class OratsDividendProjectionSetError(RuntimeError):
    """Base class for an invalid aggregate ORATS dividend projection identity."""


@dataclass(slots=True)
class OratsDividendProjectionSetContentError(OratsDividendProjectionSetError):
    """An aggregate manifest or one of its referenced files is invalid."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"ORATS dividend projection set failed{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OratsDividendProjectionSetEntry:
    """Compact identity of one annual output and its complete sidecar receipt."""

    year: int
    output_relative_path: str
    receipt_relative_path: str
    receipt_sha256: str
    output_rows: int
    output_duplicate_rows: int
    output_bytes: int
    output_sha256: str
    observed_tickers: tuple[str, ...]
    source_unidentified_rows: int
    snapshots: int

    def __post_init__(self) -> None:
        """Reject unsafe paths and malformed annual identities."""
        if (
            type(self.year) is not int
            or not _FIRST_ARCHIVE_YEAR <= self.year <= _MAXIMUM_ARCHIVE_YEAR
        ):
            msg = "projection-set year must be a plain int in [2007, 9999]"
            raise OratsDividendProjectionSetContentError(msg)
        _validate_relative_path(self.output_relative_path, "output")
        _validate_relative_path(self.receipt_relative_path, "receipt")
        if self.output_rows <= 0 or self.output_bytes <= 0 or self.snapshots <= 0:
            msg = "projection-set output rows, bytes, and snapshots must be positive"
            raise OratsDividendProjectionSetContentError(msg)
        if not 0 <= self.output_duplicate_rows <= self.output_rows:
            msg = "projection-set duplicate row count is invalid"
            raise OratsDividendProjectionSetContentError(msg)
        if self.source_unidentified_rows < 0:
            msg = "projection-set unidentified row count must be nonnegative"
            raise OratsDividendProjectionSetContentError(msg)
        if not _is_sha256(self.output_sha256) or not _is_sha256(self.receipt_sha256):
            msg = "projection-set file SHA-256 is malformed"
            raise OratsDividendProjectionSetContentError(msg)
        if self.observed_tickers != tuple(sorted(set(self.observed_tickers))):
            msg = "projection-set observed tickers must be sorted and unique"
            raise OratsDividendProjectionSetContentError(msg)
        if not self.observed_tickers:
            msg = "projection-set observed tickers must be nonempty"
            raise OratsDividendProjectionSetContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the exact canonical annual mapping."""
        return {
            "observed_tickers": list(self.observed_tickers),
            "output_bytes": self.output_bytes,
            "output_duplicate_rows": self.output_duplicate_rows,
            "output_relative_path": self.output_relative_path,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "receipt_relative_path": self.receipt_relative_path,
            "receipt_sha256": self.receipt_sha256,
            "snapshots": self.snapshots,
            "source_unidentified_rows": self.source_unidentified_rows,
            "year": self.year,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse one strict canonical annual mapping."""
        mapping = _exact_mapping(
            raw,
            {
                "observed_tickers",
                "output_bytes",
                "output_duplicate_rows",
                "output_relative_path",
                "output_rows",
                "output_sha256",
                "receipt_relative_path",
                "receipt_sha256",
                "snapshots",
                "source_unidentified_rows",
                "year",
            },
            "dividend projection-set entry",
        )
        string_fields = (
            "output_relative_path",
            "output_sha256",
            "receipt_relative_path",
            "receipt_sha256",
        )
        integer_fields = (
            "output_bytes",
            "output_duplicate_rows",
            "output_rows",
            "snapshots",
            "source_unidentified_rows",
            "year",
        )
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "dividend projection-set entry string fields are malformed"
            raise OratsDividendProjectionSetContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_fields):
            msg = "dividend projection-set entry integer fields are malformed"
            raise OratsDividendProjectionSetContentError(msg)
        observed_raw = mapping["observed_tickers"]
        if type(observed_raw) is not list or any(type(value) is not str for value in observed_raw):
            msg = "dividend projection-set entry ticker fields are malformed"
            raise OratsDividendProjectionSetContentError(msg)
        return cls(
            year=cast("int", mapping["year"]),
            output_relative_path=cast("str", mapping["output_relative_path"]),
            receipt_relative_path=cast("str", mapping["receipt_relative_path"]),
            receipt_sha256=cast("str", mapping["receipt_sha256"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_duplicate_rows=cast("int", mapping["output_duplicate_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            observed_tickers=tuple(cast("list[str]", observed_raw)),
            source_unidentified_rows=cast("int", mapping["source_unidentified_rows"]),
            snapshots=cast("int", mapping["snapshots"]),
        )


def orats_dividend_projection_set_entries_sha256(
    entries: Sequence[OratsDividendProjectionSetEntry],
) -> str:
    """Hash the exact ordered annual projection identities."""
    return hashlib.sha256(_canonical_json_bytes([entry.as_dict() for entry in entries])).hexdigest()


@dataclass(frozen=True, slots=True)
class OratsDividendProjectionSetManifest:
    """Complete inventory/spec/output identity for one annual projection set."""

    source_inventory_name: str
    source_inventory_sha256: str
    source_entries_sha256: str
    start_year: int
    end_year: int
    projection_spec: OratsDividendProjectionSpec
    projection_spec_sha256: str
    canonical_schema_sha256: str
    entries: tuple[OratsDividendProjectionSetEntry, ...]
    entries_sha256: str
    total_rows: int
    total_duplicate_rows: int
    total_bytes: int
    total_source_unidentified_rows: int
    total_snapshots: int
    schema_version: int = ORATS_DIVIDEND_PROJECTION_SET_VERSION

    def __post_init__(self) -> None:
        """Validate all aggregate cross-identities without reading referenced files."""
        self._validate_source_and_spec()
        self._validate_entries_and_totals()
        self._validate_entry_paths_and_tickers()

    def _validate_source_and_spec(self) -> None:
        if self.schema_version != ORATS_DIVIDEND_PROJECTION_SET_VERSION:
            msg = "unsupported ORATS dividend projection-set schema version"
            raise OratsDividendProjectionSetContentError(msg)
        if not self.source_inventory_name or Path(self.source_inventory_name).name != (
            self.source_inventory_name
        ):
            msg = "dividend projection-set source inventory name is malformed"
            raise OratsDividendProjectionSetContentError(msg)
        if not _is_sha256(self.source_inventory_sha256) or not _is_sha256(
            self.source_entries_sha256,
        ):
            msg = "dividend projection-set source inventory identity is malformed"
            raise OratsDividendProjectionSetContentError(msg)
        if self.projection_spec_sha256 != orats_dividend_projection_spec_sha256(
            self.projection_spec,
        ):
            msg = "dividend projection-set spec hash disagrees with its specification"
            raise OratsDividendProjectionSetContentError(msg)
        if self.canonical_schema_sha256 != orats_dividend_schema_sha256():
            msg = "dividend projection-set canonical schema hash is unsupported"
            raise OratsDividendProjectionSetContentError(msg)

    def _validate_entries_and_totals(self) -> None:
        if not self.entries:
            msg = "dividend projection-set entries must be nonempty"
            raise OratsDividendProjectionSetContentError(msg)
        years = tuple(entry.year for entry in self.entries)
        expected_years = tuple(range(self.start_year, self.end_year + 1))
        if years != expected_years:
            msg = "dividend projection-set must contain exactly one ordered entry per year"
            raise OratsDividendProjectionSetContentError(msg)
        if self.entries_sha256 != orats_dividend_projection_set_entries_sha256(self.entries):
            msg = "dividend projection-set entries hash disagrees with its entries"
            raise OratsDividendProjectionSetContentError(msg)
        totals = (
            ("rows", self.total_rows, sum(entry.output_rows for entry in self.entries)),
            (
                "duplicate rows",
                self.total_duplicate_rows,
                sum(entry.output_duplicate_rows for entry in self.entries),
            ),
            ("bytes", self.total_bytes, sum(entry.output_bytes for entry in self.entries)),
            (
                "unidentified rows",
                self.total_source_unidentified_rows,
                sum(entry.source_unidentified_rows for entry in self.entries),
            ),
            ("snapshots", self.total_snapshots, sum(entry.snapshots for entry in self.entries)),
        )
        for label, declared, observed in totals:
            if declared != observed:
                msg = f"dividend projection-set total {label} disagrees with its entries"
                raise OratsDividendProjectionSetContentError(msg)
        if self.total_rows <= 0 or self.total_bytes <= 0 or self.total_snapshots <= 0:
            msg = "dividend projection-set row, byte, and snapshot totals must be positive"
            raise OratsDividendProjectionSetContentError(msg)

    def _validate_entry_paths_and_tickers(self) -> None:
        spec_tickers = set(self.projection_spec.tickers)
        for entry in self.entries:
            expected_output = f"{self.projection_spec_sha256}/{entry.year}.parquet"
            expected_receipt = f"{expected_output}.receipt.json"
            if entry.output_relative_path != expected_output:
                msg = "dividend projection-set output path disagrees with its year and spec"
                raise OratsDividendProjectionSetContentError(msg, entry.output_relative_path)
            if entry.receipt_relative_path != expected_receipt:
                msg = "dividend projection-set receipt path disagrees with its output"
                raise OratsDividendProjectionSetContentError(msg, entry.receipt_relative_path)
            if not set(entry.observed_tickers).issubset(spec_tickers):
                msg = "dividend projection-set entry has tickers outside its specification"
                raise OratsDividendProjectionSetContentError(msg, entry.output_relative_path)

    def as_dict(self) -> dict[str, object]:
        """Return the exact canonical aggregate mapping."""
        return {
            "canonical_schema_sha256": self.canonical_schema_sha256,
            "end_year": self.end_year,
            "entries": [entry.as_dict() for entry in self.entries],
            "entries_sha256": self.entries_sha256,
            "projection_spec": self.projection_spec.as_dict(),
            "projection_spec_sha256": self.projection_spec_sha256,
            "schema_version": self.schema_version,
            "source_entries_sha256": self.source_entries_sha256,
            "source_inventory_name": self.source_inventory_name,
            "source_inventory_sha256": self.source_inventory_sha256,
            "start_year": self.start_year,
            "total_bytes": self.total_bytes,
            "total_duplicate_rows": self.total_duplicate_rows,
            "total_rows": self.total_rows,
            "total_snapshots": self.total_snapshots,
            "total_source_unidentified_rows": self.total_source_unidentified_rows,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully revalidate one aggregate dividend manifest."""
        mapping = _exact_mapping(
            raw,
            {
                "canonical_schema_sha256",
                "end_year",
                "entries",
                "entries_sha256",
                "projection_spec",
                "projection_spec_sha256",
                "schema_version",
                "source_entries_sha256",
                "source_inventory_name",
                "source_inventory_sha256",
                "start_year",
                "total_bytes",
                "total_duplicate_rows",
                "total_rows",
                "total_snapshots",
                "total_source_unidentified_rows",
            },
            "dividend projection-set manifest",
        )
        string_fields = (
            "canonical_schema_sha256",
            "entries_sha256",
            "projection_spec_sha256",
            "source_entries_sha256",
            "source_inventory_name",
            "source_inventory_sha256",
        )
        integer_fields = (
            "end_year",
            "schema_version",
            "start_year",
            "total_bytes",
            "total_duplicate_rows",
            "total_rows",
            "total_snapshots",
            "total_source_unidentified_rows",
        )
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "dividend projection-set manifest string fields are malformed"
            raise OratsDividendProjectionSetContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_fields):
            msg = "dividend projection-set manifest integer fields are malformed"
            raise OratsDividendProjectionSetContentError(msg)
        entries_raw = mapping["entries"]
        if type(entries_raw) is not list:
            msg = "dividend projection-set manifest entries are malformed"
            raise OratsDividendProjectionSetContentError(msg)
        return cls(
            source_inventory_name=cast("str", mapping["source_inventory_name"]),
            source_inventory_sha256=cast("str", mapping["source_inventory_sha256"]),
            source_entries_sha256=cast("str", mapping["source_entries_sha256"]),
            start_year=cast("int", mapping["start_year"]),
            end_year=cast("int", mapping["end_year"]),
            projection_spec=OratsDividendProjectionSpec.from_dict(mapping["projection_spec"]),
            projection_spec_sha256=cast("str", mapping["projection_spec_sha256"]),
            canonical_schema_sha256=cast("str", mapping["canonical_schema_sha256"]),
            entries=tuple(
                OratsDividendProjectionSetEntry.from_dict(entry)
                for entry in cast("list[object]", entries_raw)
            ),
            entries_sha256=cast("str", mapping["entries_sha256"]),
            total_rows=cast("int", mapping["total_rows"]),
            total_duplicate_rows=cast("int", mapping["total_duplicate_rows"]),
            total_bytes=cast("int", mapping["total_bytes"]),
            total_source_unidentified_rows=cast(
                "int",
                mapping["total_source_unidentified_rows"],
            ),
            total_snapshots=cast("int", mapping["total_snapshots"]),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class VerifiedOratsDividendProjectionSet:
    """A verified aggregate manifest and its ordered annual Parquet paths."""

    manifest_path: Path
    inventory_path: Path
    output_root: Path
    paths: tuple[Path, ...]
    manifest: OratsDividendProjectionSetManifest

    def scan(self) -> pl.LazyFrame:
        """Scan the verified ordered annual outputs without materializing licensed rows."""
        return pl.scan_parquet(list(self.paths))


def write_orats_dividend_projection_set_manifest(
    path: str | Path,
    inventory_path: str | Path,
    inventory: OratsDividendArchiveInventory,
    output_root: str | Path,
    projections: Sequence[MaterializedOratsDividendProjection],
) -> VerifiedOratsDividendProjectionSet:
    """Write or exactly reuse the aggregate identity for a complete projection run."""
    manifest_path = Path(path).expanduser()
    source_path = _regular_file(inventory_path, "source inventory")
    root = _regular_directory(output_root, "projection output root")
    parsed_inventory = read_orats_dividend_archive_inventory(source_path)
    if parsed_inventory != inventory:
        msg = "source inventory bytes disagree with the supplied dividend inventory"
        raise OratsDividendProjectionSetContentError(msg, str(source_path))
    manifest, output_paths = _build_manifest(source_path, inventory, root, projections)
    if manifest_path.exists():
        verified = load_verified_orats_dividend_projection_set(
            manifest_path,
            source_path,
            root,
        )
        if verified.manifest != manifest or verified.paths != output_paths:
            msg = "existing dividend projection-set manifest disagrees with the requested set"
            raise OratsDividendProjectionSetContentError(msg, str(manifest_path))
        return verified
    _write_new_json(manifest_path, manifest.as_dict())
    return VerifiedOratsDividendProjectionSet(
        manifest_path=manifest_path,
        inventory_path=source_path,
        output_root=root,
        paths=output_paths,
        manifest=manifest,
    )


def load_verified_orats_dividend_projection_set(
    path: str | Path,
    inventory_path: str | Path,
    output_root: str | Path,
) -> VerifiedOratsDividendProjectionSet:
    """Load a manifest and rehash every referenced receipt and annual Parquet."""
    manifest_path = _regular_file(path, "dividend projection-set manifest")
    source_path = _regular_file(inventory_path, "source dividend inventory")
    root = _regular_directory(output_root, "dividend projection output root")
    manifest = _read_manifest(manifest_path)
    if manifest.source_inventory_name != source_path.name:
        msg = "source inventory name disagrees with the dividend projection-set manifest"
        raise OratsDividendProjectionSetContentError(msg, str(source_path))
    if manifest.source_inventory_sha256 != _file_sha256(source_path):
        msg = "source inventory bytes disagree with the dividend projection-set manifest"
        raise OratsDividendProjectionSetContentError(msg, str(source_path))
    inventory = read_orats_dividend_archive_inventory(source_path)
    _validate_inventory(manifest, inventory)
    output_paths = tuple(
        _verify_entry(root, source_entry, manifest, entry)
        for source_entry, entry in zip(inventory.entries, manifest.entries, strict=True)
    )
    return VerifiedOratsDividendProjectionSet(
        manifest_path=manifest_path,
        inventory_path=source_path,
        output_root=root,
        paths=output_paths,
        manifest=manifest,
    )


def _build_manifest(
    inventory_path: Path,
    inventory: OratsDividendArchiveInventory,
    output_root: Path,
    projections: Sequence[MaterializedOratsDividendProjection],
) -> tuple[OratsDividendProjectionSetManifest, tuple[Path, ...]]:
    if len(projections) != len(inventory.entries):
        msg = "projection count must exactly equal the dividend inventory entry count"
        raise OratsDividendProjectionSetContentError(msg)
    if not projections:
        msg = "dividend projection set must be nonempty"
        raise OratsDividendProjectionSetContentError(msg)
    projection_spec = projections[0].receipt.projection_spec
    projection_spec_hash = orats_dividend_projection_spec_sha256(projection_spec)
    entries: list[OratsDividendProjectionSetEntry] = []
    output_paths: list[Path] = []
    for source_entry, projection in zip(inventory.entries, projections, strict=True):
        receipt = projection.receipt
        if receipt.source_entry != source_entry:
            msg = "annual projection source entry disagrees with the dividend inventory"
            raise OratsDividendProjectionSetContentError(msg, str(projection.path))
        if receipt.projection_spec != projection_spec:
            msg = "annual projections do not share one exact projection specification"
            raise OratsDividendProjectionSetContentError(msg, str(projection.path))
        expected_output = output_root / projection_spec_hash / f"{source_entry.year}.parquet"
        if projection.path != expected_output:
            msg = "annual projection path disagrees with the canonical output root"
            raise OratsDividendProjectionSetContentError(msg, str(projection.path))
        output = _regular_child_file(output_root, projection.path, "annual projection output")
        receipt_path = output.with_name(f"{output.name}.receipt.json")
        sidecar = _read_projection_receipt(receipt_path)
        if sidecar != receipt:
            msg = "annual projection sidecar disagrees with the supplied receipt"
            raise OratsDividendProjectionSetContentError(msg, str(receipt_path))
        _verify_output_identity(output, receipt)
        entries.append(
            OratsDividendProjectionSetEntry(
                year=source_entry.year,
                output_relative_path=output.relative_to(output_root).as_posix(),
                receipt_relative_path=receipt_path.relative_to(output_root).as_posix(),
                receipt_sha256=_file_sha256(receipt_path),
                output_rows=receipt.output_rows,
                output_duplicate_rows=receipt.output_duplicate_rows,
                output_bytes=receipt.output_bytes,
                output_sha256=receipt.output_sha256,
                observed_tickers=receipt.observed_tickers,
                source_unidentified_rows=receipt.source_unidentified_rows,
                snapshots=len(receipt.snapshots),
            ),
        )
        output_paths.append(output)
    frozen_entries = tuple(entries)
    manifest = OratsDividendProjectionSetManifest(
        source_inventory_name=inventory_path.name,
        source_inventory_sha256=_file_sha256(inventory_path),
        source_entries_sha256=inventory.entries_sha256,
        start_year=inventory.start_year,
        end_year=inventory.end_year,
        projection_spec=projection_spec,
        projection_spec_sha256=projection_spec_hash,
        canonical_schema_sha256=orats_dividend_schema_sha256(),
        entries=frozen_entries,
        entries_sha256=orats_dividend_projection_set_entries_sha256(frozen_entries),
        total_rows=sum(entry.output_rows for entry in frozen_entries),
        total_duplicate_rows=sum(entry.output_duplicate_rows for entry in frozen_entries),
        total_bytes=sum(entry.output_bytes for entry in frozen_entries),
        total_source_unidentified_rows=sum(
            entry.source_unidentified_rows for entry in frozen_entries
        ),
        total_snapshots=sum(entry.snapshots for entry in frozen_entries),
    )
    return manifest, tuple(output_paths)


def _validate_inventory(
    manifest: OratsDividendProjectionSetManifest,
    inventory: OratsDividendArchiveInventory,
) -> None:
    if manifest.source_entries_sha256 != inventory.entries_sha256:
        msg = "dividend inventory entries disagree with the projection-set manifest"
        raise OratsDividendProjectionSetContentError(msg)
    if manifest.start_year != inventory.start_year or manifest.end_year != inventory.end_year:
        msg = "dividend inventory year bounds disagree with the projection-set manifest"
        raise OratsDividendProjectionSetContentError(msg)
    if len(manifest.entries) != len(inventory.entries):
        msg = "dividend inventory count disagrees with the projection-set manifest"
        raise OratsDividendProjectionSetContentError(msg)


def _verify_entry(
    root: Path,
    source_entry: OratsDividendArchiveEntry,
    manifest: OratsDividendProjectionSetManifest,
    entry: OratsDividendProjectionSetEntry,
) -> Path:
    output = _regular_child_file(
        root, root / entry.output_relative_path, "annual projection output"
    )
    receipt_path = _regular_child_file(
        root,
        root / entry.receipt_relative_path,
        "annual projection receipt",
    )
    if _file_sha256(receipt_path) != entry.receipt_sha256:
        msg = "annual projection receipt bytes disagree with the projection-set manifest"
        raise OratsDividendProjectionSetContentError(msg, str(receipt_path))
    receipt = _read_projection_receipt(receipt_path)
    if receipt.source_entry != source_entry:
        msg = "annual projection receipt source disagrees with the dividend inventory"
        raise OratsDividendProjectionSetContentError(msg, str(receipt_path))
    if receipt.projection_spec != manifest.projection_spec:
        msg = "annual projection receipt spec disagrees with the projection-set manifest"
        raise OratsDividendProjectionSetContentError(msg, str(receipt_path))
    if (
        receipt.output_rows != entry.output_rows
        or receipt.output_duplicate_rows != entry.output_duplicate_rows
        or receipt.output_bytes != entry.output_bytes
        or receipt.output_sha256 != entry.output_sha256
        or receipt.observed_tickers != entry.observed_tickers
        or receipt.source_unidentified_rows != entry.source_unidentified_rows
        or len(receipt.snapshots) != entry.snapshots
    ):
        msg = "annual projection receipt identity disagrees with the projection-set entry"
        raise OratsDividendProjectionSetContentError(msg, str(receipt_path))
    _verify_output_identity(output, receipt)
    return output


def _verify_output_identity(path: Path, receipt: OratsDividendProjectionReceipt) -> None:
    if path.name != receipt.output_name:
        msg = "annual projection filename disagrees with its receipt"
        raise OratsDividendProjectionSetContentError(msg, str(path))
    if path.stat().st_size != receipt.output_bytes:
        msg = "annual projection byte count disagrees with its receipt"
        raise OratsDividendProjectionSetContentError(msg, str(path))
    if _file_sha256(path) != receipt.output_sha256:
        msg = "annual projection bytes disagree with its receipt SHA-256"
        raise OratsDividendProjectionSetContentError(msg, str(path))


def _read_manifest(path: Path) -> OratsDividendProjectionSetManifest:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "dividend projection-set manifest is unreadable"
        raise OratsDividendProjectionSetContentError(msg, str(path)) from error
    return OratsDividendProjectionSetManifest.from_dict(raw)


def _read_projection_receipt(path: Path) -> OratsDividendProjectionReceipt:
    regular = _regular_file(path, "annual projection receipt")
    try:
        raw = json.loads(regular.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "annual projection receipt is unreadable"
        raise OratsDividendProjectionSetContentError(msg, str(path)) from error
    return OratsDividendProjectionReceipt.from_dict(raw)


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.is_symlink():
        msg = f"{label} must be an existing non-symlink file"
        raise OratsDividendProjectionSetContentError(msg, str(candidate))
    return candidate


def _regular_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_dir() or candidate.is_symlink():
        msg = f"{label} must be an existing non-symlink directory"
        raise OratsDividendProjectionSetContentError(msg, str(candidate))
    return candidate


def _regular_child_file(root: Path, path: Path, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        msg = f"{label} must remain below the declared output root"
        raise OratsDividendProjectionSetContentError(msg, str(path)) from error
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if not current.is_dir() or current.is_symlink():
            msg = f"{label} parent must be a non-symlink directory"
            raise OratsDividendProjectionSetContentError(msg, str(current))
    return _regular_file(path, label)


def _validate_relative_path(value: str, label: str) -> None:
    if type(value) is not str or not value:
        msg = f"dividend projection-set {label} path is malformed"
        raise OratsDividendProjectionSetContentError(msg)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        msg = f"dividend projection-set {label} path must be canonical and relative"
        raise OratsDividendProjectionSetContentError(msg, value)


def _write_new_json(path: Path, value: object) -> None:
    if path.exists():
        msg = "dividend projection-set manifest already exists"
        raise OratsDividendProjectionSetContentError(msg, str(path))
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        msg = "dividend projection-set manifest parent must be a non-symlink directory"
        raise OratsDividendProjectionSetContentError(msg, str(parent))
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        msg = "dividend projection-set manifest partial already exists"
        raise OratsDividendProjectionSetContentError(msg, str(temporary))
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
        while block := source.read(ORATS_DIVIDEND_PROJECTION_SET_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _exact_mapping(raw: object, expected: set[str], label: str) -> dict[str, object]:
    if type(raw) is not dict:
        msg = f"{label} must be a JSON object"
        raise OratsDividendProjectionSetContentError(msg)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected or any(type(key) is not str for key in mapping):
        msg = f"{label} keys must be exactly {sorted(expected)!r}"
        raise OratsDividendProjectionSetContentError(msg)
    return cast("dict[str, object]", mapping)


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "ORATS_DIVIDEND_PROJECTION_SET_BLOCK_BYTES",
    "ORATS_DIVIDEND_PROJECTION_SET_VERSION",
    "OratsDividendProjectionSetContentError",
    "OratsDividendProjectionSetEntry",
    "OratsDividendProjectionSetError",
    "OratsDividendProjectionSetManifest",
    "VerifiedOratsDividendProjectionSet",
    "load_verified_orats_dividend_projection_set",
    "orats_dividend_projection_set_entries_sha256",
    "write_orats_dividend_projection_set_manifest",
]
