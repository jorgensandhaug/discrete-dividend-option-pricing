"""One verified aggregate identity for a complete ORATS option projection set."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Self, cast

import polars as pl

from .orats_archive import OratsArchiveEntry, OratsArchiveInventory, read_orats_archive_inventory
from .orats_option_projection import (
    MaterializedOratsOptionProjection,
    OratsOptionProjectionReceipt,
    OratsOptionProjectionSpec,
    orats_normalized_option_schema_sha256,
    orats_option_projection_spec_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ORATS_OPTION_PROJECTION_SET_VERSION: Final = 1
ORATS_OPTION_PROJECTION_SET_BLOCK_BYTES: Final = 8 * 1024 * 1024
_SHA256_LENGTH: Final = 64


class OratsOptionProjectionSetError(RuntimeError):
    """Base class for an invalid aggregate ORATS option projection identity."""


@dataclass(slots=True)
class OratsOptionProjectionSetContentError(OratsOptionProjectionSetError):
    """An aggregate manifest or one of its referenced files is invalid."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"ORATS option projection set failed{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OratsOptionProjectionSetEntry:
    """Compact identity of one daily output and its full sidecar receipt."""

    trade_date: date
    output_relative_path: str
    receipt_relative_path: str
    receipt_sha256: str
    output_rows: int
    output_bytes: int
    output_sha256: str
    observed_tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject unsafe paths and malformed daily identities."""
        if type(self.trade_date) is not date:
            msg = "projection-set trade date must be a plain date"
            raise OratsOptionProjectionSetContentError(msg)
        _validate_relative_path(self.output_relative_path, "output")
        _validate_relative_path(self.receipt_relative_path, "receipt")
        if self.output_rows <= 0 or self.output_bytes <= 0:
            msg = "projection-set output row and byte counts must be positive"
            raise OratsOptionProjectionSetContentError(msg)
        if not _is_sha256(self.output_sha256) or not _is_sha256(self.receipt_sha256):
            msg = "projection-set file SHA-256 is malformed"
            raise OratsOptionProjectionSetContentError(msg)
        if self.observed_tickers != tuple(sorted(set(self.observed_tickers))):
            msg = "projection-set observed tickers must be sorted and unique"
            raise OratsOptionProjectionSetContentError(msg)
        if not self.observed_tickers:
            msg = "projection-set observed tickers must be nonempty"
            raise OratsOptionProjectionSetContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the exact canonical daily mapping."""
        return {
            "observed_tickers": list(self.observed_tickers),
            "output_bytes": self.output_bytes,
            "output_relative_path": self.output_relative_path,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "receipt_relative_path": self.receipt_relative_path,
            "receipt_sha256": self.receipt_sha256,
            "trade_date": self.trade_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse one strict canonical daily mapping."""
        mapping = _exact_mapping(
            raw,
            {
                "observed_tickers",
                "output_bytes",
                "output_relative_path",
                "output_rows",
                "output_sha256",
                "receipt_relative_path",
                "receipt_sha256",
                "trade_date",
            },
            "projection-set entry",
        )
        string_fields = (
            "output_relative_path",
            "output_sha256",
            "receipt_relative_path",
            "receipt_sha256",
            "trade_date",
        )
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "projection-set entry string fields are malformed"
            raise OratsOptionProjectionSetContentError(msg)
        if type(mapping["output_rows"]) is not int or type(mapping["output_bytes"]) is not int:
            msg = "projection-set entry integer fields are malformed"
            raise OratsOptionProjectionSetContentError(msg)
        observed_raw = mapping["observed_tickers"]
        if type(observed_raw) is not list or any(type(value) is not str for value in observed_raw):
            msg = "projection-set entry ticker fields are malformed"
            raise OratsOptionProjectionSetContentError(msg)
        try:
            trade_date = date.fromisoformat(cast("str", mapping["trade_date"]))
        except ValueError as error:
            msg = "projection-set trade date is malformed"
            raise OratsOptionProjectionSetContentError(msg) from error
        return cls(
            trade_date=trade_date,
            output_relative_path=cast("str", mapping["output_relative_path"]),
            receipt_relative_path=cast("str", mapping["receipt_relative_path"]),
            receipt_sha256=cast("str", mapping["receipt_sha256"]),
            output_rows=mapping["output_rows"],
            output_bytes=mapping["output_bytes"],
            output_sha256=cast("str", mapping["output_sha256"]),
            observed_tickers=tuple(cast("list[str]", observed_raw)),
        )


def orats_option_projection_set_entries_sha256(
    entries: Sequence[OratsOptionProjectionSetEntry],
) -> str:
    """Hash the exact ordered daily projection identities."""
    return hashlib.sha256(_canonical_json_bytes([entry.as_dict() for entry in entries])).hexdigest()


@dataclass(frozen=True, slots=True)
class OratsOptionProjectionSetManifest:
    """Complete inventory/spec/output identity for one daily projection set."""

    source_inventory_name: str
    source_inventory_sha256: str
    source_entries_sha256: str
    first_trade_date: date
    last_trade_date: date
    projection_spec: OratsOptionProjectionSpec
    projection_spec_sha256: str
    normalized_schema_sha256: str
    entries: tuple[OratsOptionProjectionSetEntry, ...]
    entries_sha256: str
    total_rows: int
    total_bytes: int
    schema_version: int = ORATS_OPTION_PROJECTION_SET_VERSION

    def __post_init__(self) -> None:
        """Validate all aggregate cross-identities without reading referenced files."""
        self._validate_source_and_spec()
        self._validate_entries_and_totals()
        self._validate_entry_paths_and_tickers()

    def _validate_source_and_spec(self) -> None:
        if self.schema_version != ORATS_OPTION_PROJECTION_SET_VERSION:
            msg = "unsupported ORATS option projection-set schema version"
            raise OratsOptionProjectionSetContentError(msg)
        if not self.source_inventory_name or Path(self.source_inventory_name).name != (
            self.source_inventory_name
        ):
            msg = "projection-set source inventory name is malformed"
            raise OratsOptionProjectionSetContentError(msg)
        if not _is_sha256(self.source_inventory_sha256) or not _is_sha256(
            self.source_entries_sha256,
        ):
            msg = "projection-set source inventory identity is malformed"
            raise OratsOptionProjectionSetContentError(msg)
        if self.projection_spec_sha256 != orats_option_projection_spec_sha256(
            self.projection_spec,
        ):
            msg = "projection-set spec hash disagrees with its embedded specification"
            raise OratsOptionProjectionSetContentError(msg)
        if self.normalized_schema_sha256 != orats_normalized_option_schema_sha256():
            msg = "projection-set normalized schema hash is unsupported"
            raise OratsOptionProjectionSetContentError(msg)

    def _validate_entries_and_totals(self) -> None:
        if not self.entries:
            msg = "projection-set entries must be nonempty"
            raise OratsOptionProjectionSetContentError(msg)
        dates = tuple(entry.trade_date for entry in self.entries)
        if dates != tuple(sorted(set(dates))):
            msg = "projection-set entries must have strictly increasing unique dates"
            raise OratsOptionProjectionSetContentError(msg)
        if self.first_trade_date != dates[0] or self.last_trade_date != dates[-1]:
            msg = "projection-set trading-date bounds disagree with its entries"
            raise OratsOptionProjectionSetContentError(msg)
        if self.entries_sha256 != orats_option_projection_set_entries_sha256(self.entries):
            msg = "projection-set entries hash disagrees with its entries"
            raise OratsOptionProjectionSetContentError(msg)
        if self.total_rows != sum(entry.output_rows for entry in self.entries):
            msg = "projection-set total rows disagree with its entries"
            raise OratsOptionProjectionSetContentError(msg)
        if self.total_bytes != sum(entry.output_bytes for entry in self.entries):
            msg = "projection-set total bytes disagree with its entries"
            raise OratsOptionProjectionSetContentError(msg)
        if self.total_rows <= 0 or self.total_bytes <= 0:
            msg = "projection-set totals must be positive"
            raise OratsOptionProjectionSetContentError(msg)

    def _validate_entry_paths_and_tickers(self) -> None:
        spec_tickers = set(self.projection_spec.tickers)
        for entry in self.entries:
            expected_output = (
                f"{self.projection_spec_sha256}/{entry.trade_date.year}/"
                f"{entry.trade_date.isoformat()}.parquet"
            )
            expected_receipt = f"{expected_output}.receipt.json"
            if entry.output_relative_path != expected_output:
                msg = "projection-set output path disagrees with its date and specification"
                raise OratsOptionProjectionSetContentError(msg, entry.output_relative_path)
            if entry.receipt_relative_path != expected_receipt:
                msg = "projection-set receipt path disagrees with its output"
                raise OratsOptionProjectionSetContentError(msg, entry.receipt_relative_path)
            if not set(entry.observed_tickers).issubset(spec_tickers):
                msg = "projection-set entry contains tickers outside its specification"
                raise OratsOptionProjectionSetContentError(msg, entry.output_relative_path)

    def as_dict(self) -> dict[str, object]:
        """Return the exact canonical aggregate mapping."""
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "entries_sha256": self.entries_sha256,
            "first_trade_date": self.first_trade_date.isoformat(),
            "last_trade_date": self.last_trade_date.isoformat(),
            "normalized_schema_sha256": self.normalized_schema_sha256,
            "projection_spec": self.projection_spec.as_dict(),
            "projection_spec_sha256": self.projection_spec_sha256,
            "schema_version": self.schema_version,
            "source_entries_sha256": self.source_entries_sha256,
            "source_inventory_name": self.source_inventory_name,
            "source_inventory_sha256": self.source_inventory_sha256,
            "start_date": self.start_date.isoformat(),
            "total_bytes": self.total_bytes,
            "total_rows": self.total_rows,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully revalidate one aggregate manifest."""
        mapping = _exact_mapping(
            raw,
            {
                "entries",
                "entries_sha256",
                "first_trade_date",
                "last_trade_date",
                "normalized_schema_sha256",
                "projection_spec",
                "projection_spec_sha256",
                "schema_version",
                "source_entries_sha256",
                "source_inventory_name",
                "source_inventory_sha256",
                "total_bytes",
                "total_rows",
            },
            "projection-set manifest",
        )
        string_fields = (
            "entries_sha256",
            "first_trade_date",
            "last_trade_date",
            "normalized_schema_sha256",
            "projection_spec_sha256",
            "source_entries_sha256",
            "source_inventory_name",
            "source_inventory_sha256",
        )
        integer_fields = ("schema_version", "total_bytes", "total_rows")
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "projection-set manifest string fields are malformed"
            raise OratsOptionProjectionSetContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_fields):
            msg = "projection-set manifest integer fields are malformed"
            raise OratsOptionProjectionSetContentError(msg)
        entries_raw = mapping["entries"]
        if type(entries_raw) is not list:
            msg = "projection-set manifest entries are malformed"
            raise OratsOptionProjectionSetContentError(msg)
        try:
            first_trade_date = date.fromisoformat(cast("str", mapping["first_trade_date"]))
            last_trade_date = date.fromisoformat(cast("str", mapping["last_trade_date"]))
        except ValueError as error:
            msg = "projection-set manifest date bounds are malformed"
            raise OratsOptionProjectionSetContentError(msg) from error
        return cls(
            source_inventory_name=cast("str", mapping["source_inventory_name"]),
            source_inventory_sha256=cast("str", mapping["source_inventory_sha256"]),
            source_entries_sha256=cast("str", mapping["source_entries_sha256"]),
            first_trade_date=first_trade_date,
            last_trade_date=last_trade_date,
            projection_spec=OratsOptionProjectionSpec.from_dict(mapping["projection_spec"]),
            projection_spec_sha256=cast("str", mapping["projection_spec_sha256"]),
            normalized_schema_sha256=cast("str", mapping["normalized_schema_sha256"]),
            entries=tuple(OratsOptionProjectionSetEntry.from_dict(entry) for entry in entries_raw),
            entries_sha256=cast("str", mapping["entries_sha256"]),
            total_rows=cast("int", mapping["total_rows"]),
            total_bytes=cast("int", mapping["total_bytes"]),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class VerifiedOratsOptionProjectionSet:
    """A verified aggregate manifest and its ordered daily Parquet paths."""

    manifest_path: Path
    inventory_path: Path
    output_root: Path
    paths: tuple[Path, ...]
    manifest: OratsOptionProjectionSetManifest

    def scan(self) -> pl.LazyFrame:
        """Scan the verified ordered daily outputs without materializing licensed rows."""
        return pl.scan_parquet(list(self.paths))


def write_orats_option_projection_set_manifest(
    path: str | Path,
    inventory_path: str | Path,
    inventory: OratsArchiveInventory,
    output_root: str | Path,
    projections: Sequence[MaterializedOratsOptionProjection],
) -> VerifiedOratsOptionProjectionSet:
    """Write or exactly reuse the aggregate identity for a complete projection run."""
    manifest_path = Path(path).expanduser()
    source_path = _regular_file(inventory_path, "source inventory")
    root = _regular_directory(output_root, "projection output root")
    parsed_inventory = read_orats_archive_inventory(source_path)
    if parsed_inventory != inventory:
        msg = "source inventory bytes disagree with the supplied inventory"
        raise OratsOptionProjectionSetContentError(msg, str(source_path))
    manifest, output_paths = _build_manifest(source_path, inventory, root, projections)
    if manifest_path.exists():
        verified = load_verified_orats_option_projection_set(
            manifest_path,
            source_path,
            root,
        )
        if verified.manifest != manifest or verified.paths != output_paths:
            msg = "existing projection-set manifest disagrees with the requested projection set"
            raise OratsOptionProjectionSetContentError(msg, str(manifest_path))
        return verified
    _write_new_json(manifest_path, manifest.as_dict())
    return VerifiedOratsOptionProjectionSet(
        manifest_path=manifest_path,
        inventory_path=source_path,
        output_root=root,
        paths=output_paths,
        manifest=manifest,
    )


def load_verified_orats_option_projection_set(
    path: str | Path,
    inventory_path: str | Path,
    output_root: str | Path,
) -> VerifiedOratsOptionProjectionSet:
    """Load a manifest and rehash every referenced receipt and projected Parquet."""
    manifest_path = _regular_file(path, "projection-set manifest")
    source_path = _regular_file(inventory_path, "source inventory")
    root = _regular_directory(output_root, "projection output root")
    manifest = _read_manifest(manifest_path)
    if manifest.source_inventory_name != source_path.name:
        msg = "source inventory name disagrees with the projection-set manifest"
        raise OratsOptionProjectionSetContentError(msg, str(source_path))
    if manifest.source_inventory_sha256 != _file_sha256(source_path):
        msg = "source inventory bytes disagree with the projection-set manifest"
        raise OratsOptionProjectionSetContentError(msg, str(source_path))
    inventory = read_orats_archive_inventory(source_path)
    _validate_inventory(manifest, inventory)
    output_paths = tuple(
        _verify_entry(root, source_entry, manifest, entry)
        for source_entry, entry in zip(inventory.entries, manifest.entries, strict=True)
    )
    return VerifiedOratsOptionProjectionSet(
        manifest_path=manifest_path,
        inventory_path=source_path,
        output_root=root,
        paths=output_paths,
        manifest=manifest,
    )


def _build_manifest(
    inventory_path: Path,
    inventory: OratsArchiveInventory,
    output_root: Path,
    projections: Sequence[MaterializedOratsOptionProjection],
) -> tuple[OratsOptionProjectionSetManifest, tuple[Path, ...]]:
    if len(projections) != len(inventory.entries):
        msg = "projection count must exactly equal the source inventory entry count"
        raise OratsOptionProjectionSetContentError(msg)
    if not projections:
        msg = "projection set must be nonempty"
        raise OratsOptionProjectionSetContentError(msg)
    projection_spec = projections[0].receipt.projection_spec
    projection_spec_hash = orats_option_projection_spec_sha256(projection_spec)
    entries: list[OratsOptionProjectionSetEntry] = []
    output_paths: list[Path] = []
    for source_entry, projection in zip(inventory.entries, projections, strict=True):
        receipt = projection.receipt
        if receipt.source_entry != source_entry:
            msg = "daily projection source entry disagrees with the source inventory"
            raise OratsOptionProjectionSetContentError(msg, str(projection.path))
        if receipt.projection_spec != projection_spec:
            msg = "daily projections do not share one exact projection specification"
            raise OratsOptionProjectionSetContentError(msg, str(projection.path))
        expected_output = (
            output_root
            / projection_spec_hash
            / str(source_entry.trade_date.year)
            / f"{source_entry.trade_date.isoformat()}.parquet"
        )
        if projection.path != expected_output:
            msg = "daily projection path disagrees with the canonical output root"
            raise OratsOptionProjectionSetContentError(msg, str(projection.path))
        output = _regular_child_file(output_root, projection.path, "daily projection output")
        receipt_path = output.with_name(f"{output.name}.receipt.json")
        sidecar = _read_projection_receipt(receipt_path)
        if sidecar != receipt:
            msg = "daily projection sidecar disagrees with the supplied receipt"
            raise OratsOptionProjectionSetContentError(msg, str(receipt_path))
        _verify_output_identity(output, receipt)
        entries.append(
            OratsOptionProjectionSetEntry(
                trade_date=source_entry.trade_date,
                output_relative_path=output.relative_to(output_root).as_posix(),
                receipt_relative_path=receipt_path.relative_to(output_root).as_posix(),
                receipt_sha256=_file_sha256(receipt_path),
                output_rows=receipt.output_rows,
                output_bytes=receipt.output_bytes,
                output_sha256=receipt.output_sha256,
                observed_tickers=receipt.observed_tickers,
            ),
        )
        output_paths.append(output)
    frozen_entries = tuple(entries)
    manifest = OratsOptionProjectionSetManifest(
        source_inventory_name=inventory_path.name,
        source_inventory_sha256=_file_sha256(inventory_path),
        source_entries_sha256=inventory.entries_sha256,
        first_trade_date=inventory.entries[0].trade_date,
        last_trade_date=inventory.entries[-1].trade_date,
        projection_spec=projection_spec,
        projection_spec_sha256=projection_spec_hash,
        normalized_schema_sha256=orats_normalized_option_schema_sha256(),
        entries=frozen_entries,
        entries_sha256=orats_option_projection_set_entries_sha256(frozen_entries),
        total_rows=sum(entry.output_rows for entry in frozen_entries),
        total_bytes=sum(entry.output_bytes for entry in frozen_entries),
    )
    return manifest, tuple(output_paths)


def _validate_inventory(
    manifest: OratsOptionProjectionSetManifest,
    inventory: OratsArchiveInventory,
) -> None:
    if manifest.source_entries_sha256 != inventory.entries_sha256:
        msg = "source inventory entries disagree with the projection-set manifest"
        raise OratsOptionProjectionSetContentError(msg)
    if (
        manifest.first_trade_date != inventory.entries[0].trade_date
        or manifest.last_trade_date != inventory.entries[-1].trade_date
    ):
        msg = "source inventory trading-date bounds disagree with the projection-set manifest"
        raise OratsOptionProjectionSetContentError(msg)
    if len(manifest.entries) != len(inventory.entries):
        msg = "source inventory count disagrees with the projection-set manifest"
        raise OratsOptionProjectionSetContentError(msg)


def _verify_entry(
    root: Path,
    source_entry: OratsArchiveEntry,
    manifest: OratsOptionProjectionSetManifest,
    entry: OratsOptionProjectionSetEntry,
) -> Path:
    output = _regular_child_file(root, root / entry.output_relative_path, "daily projection output")
    receipt_path = _regular_child_file(
        root,
        root / entry.receipt_relative_path,
        "daily projection receipt",
    )
    if _file_sha256(receipt_path) != entry.receipt_sha256:
        msg = "daily projection receipt bytes disagree with the projection-set manifest"
        raise OratsOptionProjectionSetContentError(msg, str(receipt_path))
    receipt = _read_projection_receipt(receipt_path)
    if receipt.source_entry != source_entry:
        msg = "daily projection receipt source disagrees with the source inventory"
        raise OratsOptionProjectionSetContentError(msg, str(receipt_path))
    if receipt.projection_spec != manifest.projection_spec:
        msg = "daily projection receipt spec disagrees with the projection-set manifest"
        raise OratsOptionProjectionSetContentError(msg, str(receipt_path))
    if (
        receipt.output_rows != entry.output_rows
        or receipt.output_bytes != entry.output_bytes
        or receipt.output_sha256 != entry.output_sha256
        or receipt.observed_tickers != entry.observed_tickers
    ):
        msg = "daily projection receipt identity disagrees with the projection-set entry"
        raise OratsOptionProjectionSetContentError(msg, str(receipt_path))
    _verify_output_identity(output, receipt)
    return output


def _verify_output_identity(path: Path, receipt: OratsOptionProjectionReceipt) -> None:
    if path.name != receipt.output_name:
        msg = "daily projection filename disagrees with its receipt"
        raise OratsOptionProjectionSetContentError(msg, str(path))
    if path.stat().st_size != receipt.output_bytes:
        msg = "daily projection byte count disagrees with its receipt"
        raise OratsOptionProjectionSetContentError(msg, str(path))
    if _file_sha256(path) != receipt.output_sha256:
        msg = "daily projection bytes disagree with its receipt SHA-256"
        raise OratsOptionProjectionSetContentError(msg, str(path))


def _read_manifest(path: Path) -> OratsOptionProjectionSetManifest:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "projection-set manifest is unreadable"
        raise OratsOptionProjectionSetContentError(msg, str(path)) from error
    return OratsOptionProjectionSetManifest.from_dict(raw)


def _read_projection_receipt(path: Path) -> OratsOptionProjectionReceipt:
    regular = _regular_file(path, "daily projection receipt")
    try:
        raw = json.loads(regular.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "daily projection receipt is unreadable"
        raise OratsOptionProjectionSetContentError(msg, str(path)) from error
    return OratsOptionProjectionReceipt.from_dict(raw)


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.is_symlink():
        msg = f"{label} must be an existing non-symlink file"
        raise OratsOptionProjectionSetContentError(msg, str(candidate))
    return candidate


def _regular_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_dir() or candidate.is_symlink():
        msg = f"{label} must be an existing non-symlink directory"
        raise OratsOptionProjectionSetContentError(msg, str(candidate))
    return candidate


def _regular_child_file(root: Path, path: Path, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        msg = f"{label} must remain below the declared output root"
        raise OratsOptionProjectionSetContentError(msg, str(path)) from error
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if not current.is_dir() or current.is_symlink():
            msg = f"{label} parent must be a non-symlink directory"
            raise OratsOptionProjectionSetContentError(msg, str(current))
    return _regular_file(path, label)


def _validate_relative_path(value: str, label: str) -> None:
    if type(value) is not str or not value:
        msg = f"projection-set {label} path is malformed"
        raise OratsOptionProjectionSetContentError(msg)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        msg = f"projection-set {label} path must be canonical and relative"
        raise OratsOptionProjectionSetContentError(msg, value)


def _write_new_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        msg = "projection-set manifest already exists"
        raise OratsOptionProjectionSetContentError(msg, str(path))
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        msg = "projection-set manifest parent must be an existing non-symlink directory"
        raise OratsOptionProjectionSetContentError(msg, str(parent))
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        msg = "projection-set manifest partial already exists"
        raise OratsOptionProjectionSetContentError(msg, str(temporary))
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
        while block := source.read(ORATS_OPTION_PROJECTION_SET_BLOCK_BYTES):
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
        raise OratsOptionProjectionSetContentError(msg)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected or any(type(key) is not str for key in mapping):
        msg = f"{label} keys must be exactly {sorted(expected)!r}"
        raise OratsOptionProjectionSetContentError(msg)
    return cast("dict[str, object]", mapping)


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "ORATS_OPTION_PROJECTION_SET_BLOCK_BYTES",
    "ORATS_OPTION_PROJECTION_SET_VERSION",
    "OratsOptionProjectionSetContentError",
    "OratsOptionProjectionSetEntry",
    "OratsOptionProjectionSetError",
    "OratsOptionProjectionSetManifest",
    "VerifiedOratsOptionProjectionSet",
    "load_verified_orats_option_projection_set",
    "orats_option_projection_set_entries_sha256",
    "write_orats_option_projection_set_manifest",
]
