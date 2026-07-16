"""Verified annual compaction of immutable daily ORATS option projections."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, Self, cast

import polars as pl

from .orats_option_projection import orats_normalized_option_schema_sha256
from .orats_option_set import (
    OratsOptionProjectionSetEntry,
    VerifiedOratsOptionProjectionSet,
    load_verified_orats_option_projection_set,
    orats_option_projection_set_entries_sha256,
)
from .orats_options import ORATS_NORMALIZED_OPTION_DTYPES

if TYPE_CHECKING:
    from collections.abc import Sequence

ORATS_OPTION_COMPACTION_VERSION: Final = 1
ORATS_OPTION_COMPACTION_RECEIPT_VERSION: Final = 1
ORATS_OPTION_COMPACTION_MANIFEST_VERSION: Final = 1
ORATS_OPTION_COMPACTION_COMPRESSION: Final = "zstd"
ORATS_OPTION_COMPACTION_COMPRESSION_LEVEL: Final = 3
ORATS_OPTION_COMPACTION_ROW_GROUP_SIZE: Final = 131_072
ORATS_OPTION_COMPACTION_BLOCK_BYTES: Final = 8 * 1024 * 1024
_SHA256_LENGTH: Final = 64


class OratsOptionCompactionError(RuntimeError):
    """Base class for an invalid ORATS option compaction request or artifact."""


@dataclass(slots=True)
class OratsOptionCompactionSpecError(OratsOptionCompactionError):
    """A compaction selection or physical write setting is unsupported."""

    reason: str

    def __str__(self) -> str:
        """Describe the invalid static setting."""
        return self.reason


@dataclass(slots=True)
class OratsOptionCompactionContentError(OratsOptionCompactionError):
    """A source, receipt, output, or aggregate identity is invalid."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"ORATS option compaction failed{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ParquetCompactionWriteSpec:
    """Exact physical identity of the streaming annual Parquet sink."""

    polars_version: str = pl.__version__
    compression: Literal["zstd"] = ORATS_OPTION_COMPACTION_COMPRESSION
    compression_level: int = ORATS_OPTION_COMPACTION_COMPRESSION_LEVEL
    row_group_size: int = ORATS_OPTION_COMPACTION_ROW_GROUP_SIZE
    statistics: bool = True
    maintain_order: bool = True
    sync_on_close: Literal["all"] = "all"
    engine: Literal["streaming"] = "streaming"

    def __post_init__(self) -> None:
        """Reject implicit or unbounded physical write policies."""
        if type(self.polars_version) is not str or not self.polars_version:
            msg = "compaction Polars version must be a nonempty string"
            raise OratsOptionCompactionSpecError(msg)
        if self.compression != ORATS_OPTION_COMPACTION_COMPRESSION:
            msg = "compaction compression must be zstd"
            raise OratsOptionCompactionSpecError(msg)
        if self.compression_level != ORATS_OPTION_COMPACTION_COMPRESSION_LEVEL:
            msg = "compaction compression level must be 3"
            raise OratsOptionCompactionSpecError(msg)
        if self.row_group_size != ORATS_OPTION_COMPACTION_ROW_GROUP_SIZE:
            msg = "compaction row group size must be 131072"
            raise OratsOptionCompactionSpecError(msg)
        if type(self.statistics) is not bool or not self.statistics:
            msg = "compaction Parquet statistics must be enabled"
            raise OratsOptionCompactionSpecError(msg)
        if type(self.maintain_order) is not bool or not self.maintain_order:
            msg = "compaction must preserve source row order"
            raise OratsOptionCompactionSpecError(msg)
        if self.sync_on_close != "all" or self.engine != "streaming":
            msg = "compaction requires durable streaming Parquet sinks"
            raise OratsOptionCompactionSpecError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical physical writer identity."""
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "engine": self.engine,
            "maintain_order": self.maintain_order,
            "polars_version": self.polars_version,
            "row_group_size": self.row_group_size,
            "statistics": self.statistics,
            "sync_on_close": self.sync_on_close,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse an exact physical writer mapping without coercion."""
        mapping = _exact_mapping(
            raw,
            {
                "compression",
                "compression_level",
                "engine",
                "maintain_order",
                "polars_version",
                "row_group_size",
                "statistics",
                "sync_on_close",
            },
            "compaction writer spec",
        )
        string_fields = ("compression", "engine", "polars_version", "sync_on_close")
        integer_fields = ("compression_level", "row_group_size")
        boolean_fields = ("maintain_order", "statistics")
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "compaction writer string fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_fields):
            msg = "compaction writer integer fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        if any(type(mapping[name]) is not bool for name in boolean_fields):
            msg = "compaction writer boolean fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        return cls(
            polars_version=cast("str", mapping["polars_version"]),
            compression=cast("Literal['zstd']", mapping["compression"]),
            compression_level=cast("int", mapping["compression_level"]),
            row_group_size=cast("int", mapping["row_group_size"]),
            statistics=cast("bool", mapping["statistics"]),
            maintain_order=cast("bool", mapping["maintain_order"]),
            sync_on_close=cast("Literal['all']", mapping["sync_on_close"]),
            engine=cast("Literal['streaming']", mapping["engine"]),
        )


@dataclass(frozen=True, slots=True)
class OratsOptionCompactionSpec:
    """Explicit half-open date selection plus separate physical write identity."""

    start_date: date
    end_date_exclusive: date
    writer: ParquetCompactionWriteSpec = ParquetCompactionWriteSpec()
    schema_version: int = ORATS_OPTION_COMPACTION_VERSION

    def __post_init__(self) -> None:
        """Reject ambiguous date scopes and unsupported schema versions."""
        if self.schema_version != ORATS_OPTION_COMPACTION_VERSION:
            msg = "unsupported ORATS option compaction schema version"
            raise OratsOptionCompactionSpecError(msg)
        if type(self.start_date) is not date or type(self.end_date_exclusive) is not date:
            msg = "compaction bounds must be plain dates"
            raise OratsOptionCompactionSpecError(msg)
        if self.start_date >= self.end_date_exclusive:
            msg = "compaction requires a nonempty half-open date interval"
            raise OratsOptionCompactionSpecError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical selection and writer identity."""
        return {
            "end_date_exclusive": self.end_date_exclusive.isoformat(),
            "schema_version": self.schema_version,
            "start_date": self.start_date.isoformat(),
            "writer": self.writer.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse an exact compaction specification without inferred bounds."""
        mapping = _exact_mapping(
            raw,
            {"end_date_exclusive", "schema_version", "start_date", "writer"},
            "compaction spec",
        )
        if type(mapping["start_date"]) is not str or type(mapping["end_date_exclusive"]) is not str:
            msg = "compaction date bounds are malformed"
            raise OratsOptionCompactionContentError(msg)
        if type(mapping["schema_version"]) is not int:
            msg = "compaction schema version is malformed"
            raise OratsOptionCompactionContentError(msg)
        try:
            start_date = date.fromisoformat(mapping["start_date"])
            end_date_exclusive = date.fromisoformat(mapping["end_date_exclusive"])
        except ValueError as error:
            msg = "compaction date bounds are malformed"
            raise OratsOptionCompactionContentError(msg) from error
        return cls(
            start_date=start_date,
            end_date_exclusive=end_date_exclusive,
            writer=ParquetCompactionWriteSpec.from_dict(mapping["writer"]),
            schema_version=mapping["schema_version"],
        )


def orats_option_compaction_spec_sha256(spec: OratsOptionCompactionSpec) -> str:
    """Hash the exact date-selection and physical-writer specification."""
    return hashlib.sha256(_canonical_json_bytes(spec.as_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class OratsOptionCompactionReceipt:
    """Complete source slice and output identity for one annual Parquet."""

    year: int
    first_trade_date: date
    last_trade_date: date
    source_manifest_sha256: str
    source_projection_entries_sha256: str
    source_year_entries_sha256: str
    source_entry_count: int
    source_rows: int
    source_bytes: int
    compaction_spec: OratsOptionCompactionSpec
    compaction_spec_sha256: str
    normalized_schema_sha256: str
    output_name: str
    output_rows: int
    output_bytes: int
    output_sha256: str
    observed_tickers: tuple[str, ...]
    schema_version: int = ORATS_OPTION_COMPACTION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        """Validate cross-field annual identities without reading local files."""
        self._validate_version_dates_and_hashes()
        self._validate_spec_and_output()

    def _validate_version_dates_and_hashes(self) -> None:
        if self.schema_version != ORATS_OPTION_COMPACTION_RECEIPT_VERSION:
            msg = "unsupported ORATS option compaction receipt version"
            raise OratsOptionCompactionContentError(msg)
        if (
            type(self.year) is not int
            or type(self.first_trade_date) is not date
            or type(self.last_trade_date) is not date
            or self.year != self.first_trade_date.year
        ):
            msg = "compaction receipt year disagrees with its first trade date"
            raise OratsOptionCompactionContentError(msg)
        if self.last_trade_date.year != self.year or self.first_trade_date > self.last_trade_date:
            msg = "compaction receipt trade-date bounds are malformed"
            raise OratsOptionCompactionContentError(msg)
        if not (
            self.compaction_spec.start_date
            <= self.first_trade_date
            <= self.last_trade_date
            < self.compaction_spec.end_date_exclusive
        ):
            msg = "compaction receipt dates fall outside its declared selection"
            raise OratsOptionCompactionContentError(msg)
        for value in (
            self.source_manifest_sha256,
            self.source_projection_entries_sha256,
            self.source_year_entries_sha256,
            self.normalized_schema_sha256,
            self.output_sha256,
        ):
            if not _is_sha256(value):
                msg = "compaction receipt SHA-256 is malformed"
                raise OratsOptionCompactionContentError(msg)

    def _validate_spec_and_output(self) -> None:
        if self.compaction_spec_sha256 != orats_option_compaction_spec_sha256(
            self.compaction_spec,
        ):
            msg = "compaction receipt spec hash disagrees with its embedded specification"
            raise OratsOptionCompactionContentError(msg)
        if self.normalized_schema_sha256 != orats_normalized_option_schema_sha256():
            msg = "compaction receipt normalized schema hash is unsupported"
            raise OratsOptionCompactionContentError(msg)
        counts = (
            self.source_entry_count,
            self.source_rows,
            self.source_bytes,
            self.output_rows,
            self.output_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in counts):
            msg = "compaction receipt counts must be positive plain integers"
            raise OratsOptionCompactionContentError(msg)
        if self.output_rows != self.source_rows:
            msg = "compaction output rows must exactly equal selected source rows"
            raise OratsOptionCompactionContentError(msg)
        if self.output_name != f"{self.year}.parquet":
            msg = "compaction output name disagrees with its year"
            raise OratsOptionCompactionContentError(msg)
        if self.observed_tickers != tuple(sorted(set(self.observed_tickers))):
            msg = "compaction observed tickers must be sorted and unique"
            raise OratsOptionCompactionContentError(msg)
        if not self.observed_tickers:
            msg = "compaction observed tickers must be nonempty"
            raise OratsOptionCompactionContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical annual receipt mapping."""
        return {
            "compaction_spec": self.compaction_spec.as_dict(),
            "compaction_spec_sha256": self.compaction_spec_sha256,
            "first_trade_date": self.first_trade_date.isoformat(),
            "last_trade_date": self.last_trade_date.isoformat(),
            "normalized_schema_sha256": self.normalized_schema_sha256,
            "observed_tickers": list(self.observed_tickers),
            "output_bytes": self.output_bytes,
            "output_name": self.output_name,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "schema_version": self.schema_version,
            "source_bytes": self.source_bytes,
            "source_entry_count": self.source_entry_count,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_projection_entries_sha256": self.source_projection_entries_sha256,
            "source_rows": self.source_rows,
            "source_year_entries_sha256": self.source_year_entries_sha256,
            "year": self.year,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully validate one exact annual receipt."""
        mapping = _exact_mapping(
            raw,
            {
                "compaction_spec",
                "compaction_spec_sha256",
                "first_trade_date",
                "last_trade_date",
                "normalized_schema_sha256",
                "observed_tickers",
                "output_bytes",
                "output_name",
                "output_rows",
                "output_sha256",
                "schema_version",
                "source_bytes",
                "source_entry_count",
                "source_manifest_sha256",
                "source_projection_entries_sha256",
                "source_rows",
                "source_year_entries_sha256",
                "year",
            },
            "compaction receipt",
        )
        string_fields = (
            "compaction_spec_sha256",
            "first_trade_date",
            "last_trade_date",
            "normalized_schema_sha256",
            "output_name",
            "output_sha256",
            "source_manifest_sha256",
            "source_projection_entries_sha256",
            "source_year_entries_sha256",
        )
        integer_fields = (
            "output_bytes",
            "output_rows",
            "schema_version",
            "source_bytes",
            "source_entry_count",
            "source_rows",
            "year",
        )
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "compaction receipt string fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_fields):
            msg = "compaction receipt integer fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        observed_raw = mapping["observed_tickers"]
        if type(observed_raw) is not list or any(type(value) is not str for value in observed_raw):
            msg = "compaction receipt ticker fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        try:
            first_trade_date = date.fromisoformat(cast("str", mapping["first_trade_date"]))
            last_trade_date = date.fromisoformat(cast("str", mapping["last_trade_date"]))
        except ValueError as error:
            msg = "compaction receipt date fields are malformed"
            raise OratsOptionCompactionContentError(msg) from error
        return cls(
            year=cast("int", mapping["year"]),
            first_trade_date=first_trade_date,
            last_trade_date=last_trade_date,
            source_manifest_sha256=cast("str", mapping["source_manifest_sha256"]),
            source_projection_entries_sha256=cast(
                "str",
                mapping["source_projection_entries_sha256"],
            ),
            source_year_entries_sha256=cast("str", mapping["source_year_entries_sha256"]),
            source_entry_count=cast("int", mapping["source_entry_count"]),
            source_rows=cast("int", mapping["source_rows"]),
            source_bytes=cast("int", mapping["source_bytes"]),
            compaction_spec=OratsOptionCompactionSpec.from_dict(mapping["compaction_spec"]),
            compaction_spec_sha256=cast("str", mapping["compaction_spec_sha256"]),
            normalized_schema_sha256=cast("str", mapping["normalized_schema_sha256"]),
            output_name=cast("str", mapping["output_name"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            observed_tickers=tuple(cast("list[str]", observed_raw)),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class MaterializedOratsOptionCompactionYear:
    """One verified annual Parquet and its complete source/output receipt."""

    path: Path
    receipt_path: Path
    receipt: OratsOptionCompactionReceipt


@dataclass(frozen=True, slots=True)
class OratsOptionCompactionManifestEntry:
    """Compact identity of one annual output and its sidecar receipt."""

    year: int
    output_relative_path: str
    receipt_relative_path: str
    receipt_sha256: str
    source_year_entries_sha256: str
    first_trade_date: date
    last_trade_date: date
    source_entry_count: int
    source_bytes: int
    output_rows: int
    output_bytes: int
    output_sha256: str
    observed_tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject malformed paths, dates, hashes, and aggregates."""
        if (
            type(self.year) is not int
            or type(self.first_trade_date) is not date
            or type(self.last_trade_date) is not date
            or self.first_trade_date.year != self.year
        ):
            msg = "compaction manifest entry year is malformed"
            raise OratsOptionCompactionContentError(msg)
        if self.last_trade_date.year != self.year or self.first_trade_date > self.last_trade_date:
            msg = "compaction manifest entry dates are malformed"
            raise OratsOptionCompactionContentError(msg)
        _validate_relative_path(self.output_relative_path, "output")
        _validate_relative_path(self.receipt_relative_path, "receipt")
        for value in (self.receipt_sha256, self.source_year_entries_sha256, self.output_sha256):
            if not _is_sha256(value):
                msg = "compaction manifest entry SHA-256 is malformed"
                raise OratsOptionCompactionContentError(msg)
        for value in (
            self.source_entry_count,
            self.source_bytes,
            self.output_rows,
            self.output_bytes,
        ):
            if type(value) is not int or value <= 0:
                msg = "compaction manifest entry counts must be positive plain integers"
                raise OratsOptionCompactionContentError(msg)
        if self.observed_tickers != tuple(sorted(set(self.observed_tickers))):
            msg = "compaction manifest entry tickers must be sorted and unique"
            raise OratsOptionCompactionContentError(msg)
        if not self.observed_tickers:
            msg = "compaction manifest entry tickers must be nonempty"
            raise OratsOptionCompactionContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical annual manifest-entry mapping."""
        return {
            "first_trade_date": self.first_trade_date.isoformat(),
            "last_trade_date": self.last_trade_date.isoformat(),
            "observed_tickers": list(self.observed_tickers),
            "output_bytes": self.output_bytes,
            "output_relative_path": self.output_relative_path,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "receipt_relative_path": self.receipt_relative_path,
            "receipt_sha256": self.receipt_sha256,
            "source_entry_count": self.source_entry_count,
            "source_bytes": self.source_bytes,
            "source_year_entries_sha256": self.source_year_entries_sha256,
            "year": self.year,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse one strict canonical annual manifest entry."""
        mapping = _exact_mapping(
            raw,
            {
                "first_trade_date",
                "last_trade_date",
                "observed_tickers",
                "output_bytes",
                "output_relative_path",
                "output_rows",
                "output_sha256",
                "receipt_relative_path",
                "receipt_sha256",
                "source_entry_count",
                "source_bytes",
                "source_year_entries_sha256",
                "year",
            },
            "compaction manifest entry",
        )
        string_fields = (
            "first_trade_date",
            "last_trade_date",
            "output_relative_path",
            "output_sha256",
            "receipt_relative_path",
            "receipt_sha256",
            "source_year_entries_sha256",
        )
        integer_fields = (
            "output_bytes",
            "output_rows",
            "source_bytes",
            "source_entry_count",
            "year",
        )
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "compaction manifest entry string fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_fields):
            msg = "compaction manifest entry integer fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        observed_raw = mapping["observed_tickers"]
        if type(observed_raw) is not list or any(type(value) is not str for value in observed_raw):
            msg = "compaction manifest entry ticker fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        try:
            first_trade_date = date.fromisoformat(cast("str", mapping["first_trade_date"]))
            last_trade_date = date.fromisoformat(cast("str", mapping["last_trade_date"]))
        except ValueError as error:
            msg = "compaction manifest entry dates are malformed"
            raise OratsOptionCompactionContentError(msg) from error
        return cls(
            year=cast("int", mapping["year"]),
            output_relative_path=cast("str", mapping["output_relative_path"]),
            receipt_relative_path=cast("str", mapping["receipt_relative_path"]),
            receipt_sha256=cast("str", mapping["receipt_sha256"]),
            source_year_entries_sha256=cast("str", mapping["source_year_entries_sha256"]),
            first_trade_date=first_trade_date,
            last_trade_date=last_trade_date,
            source_entry_count=cast("int", mapping["source_entry_count"]),
            source_bytes=cast("int", mapping["source_bytes"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            observed_tickers=tuple(cast("list[str]", observed_raw)),
        )


def orats_option_compaction_manifest_entries_sha256(
    entries: Sequence[OratsOptionCompactionManifestEntry],
) -> str:
    """Hash the exact ordered annual artifact identities."""
    return hashlib.sha256(_canonical_json_bytes([entry.as_dict() for entry in entries])).hexdigest()


@dataclass(frozen=True, slots=True)
class OratsOptionCompactionManifest:
    """Complete source selection and annual output identity."""

    source_manifest_name: str
    source_manifest_sha256: str
    source_projection_entries_sha256: str
    source_projection_spec_sha256: str
    projected_tickers: tuple[str, ...]
    normalized_schema_sha256: str
    compaction_spec: OratsOptionCompactionSpec
    compaction_spec_sha256: str
    selected_source_entries_sha256: str
    selected_first_trade_date: date
    selected_last_trade_date: date
    selected_source_entry_count: int
    selected_source_rows: int
    selected_source_bytes: int
    entries: tuple[OratsOptionCompactionManifestEntry, ...]
    entries_sha256: str
    total_output_rows: int
    total_output_bytes: int
    schema_version: int = ORATS_OPTION_COMPACTION_MANIFEST_VERSION

    def __post_init__(self) -> None:
        """Validate complete aggregate cross-identities without reading files."""
        self._validate_source_and_spec()
        self._validate_entries()
        self._validate_totals()
        self._validate_paths_and_tickers()

    def _validate_source_and_spec(self) -> None:
        if self.schema_version != ORATS_OPTION_COMPACTION_MANIFEST_VERSION:
            msg = "unsupported ORATS option compaction manifest version"
            raise OratsOptionCompactionContentError(msg)
        if not self.source_manifest_name or Path(self.source_manifest_name).name != (
            self.source_manifest_name
        ):
            msg = "compaction source manifest name is malformed"
            raise OratsOptionCompactionContentError(msg)
        for value in (
            self.source_manifest_sha256,
            self.source_projection_entries_sha256,
            self.source_projection_spec_sha256,
            self.normalized_schema_sha256,
            self.selected_source_entries_sha256,
            self.entries_sha256,
        ):
            if not _is_sha256(value):
                msg = "compaction manifest SHA-256 is malformed"
                raise OratsOptionCompactionContentError(msg)
        if self.normalized_schema_sha256 != orats_normalized_option_schema_sha256():
            msg = "compaction manifest normalized schema hash is unsupported"
            raise OratsOptionCompactionContentError(msg)
        if (
            type(self.projected_tickers) is not tuple
            or any(type(ticker) is not str for ticker in self.projected_tickers)
            or self.projected_tickers != tuple(sorted(set(self.projected_tickers)))
        ):
            msg = "compaction manifest projected tickers must be sorted and unique"
            raise OratsOptionCompactionContentError(msg)
        if not self.projected_tickers:
            msg = "compaction manifest projected tickers must be nonempty"
            raise OratsOptionCompactionContentError(msg)
        if self.compaction_spec_sha256 != orats_option_compaction_spec_sha256(
            self.compaction_spec,
        ):
            msg = "compaction manifest spec hash disagrees with its embedded specification"
            raise OratsOptionCompactionContentError(msg)

    def _validate_entries(self) -> None:
        if not self.entries:
            msg = "compaction manifest annual entries must be nonempty"
            raise OratsOptionCompactionContentError(msg)
        years = tuple(entry.year for entry in self.entries)
        if years != tuple(sorted(set(years))):
            msg = "compaction manifest years must be strictly increasing and unique"
            raise OratsOptionCompactionContentError(msg)
        if self.entries_sha256 != orats_option_compaction_manifest_entries_sha256(self.entries):
            msg = "compaction manifest entries hash disagrees with its entries"
            raise OratsOptionCompactionContentError(msg)

    def _validate_totals(self) -> None:
        if (
            self.selected_first_trade_date != self.entries[0].first_trade_date
            or self.selected_last_trade_date != self.entries[-1].last_trade_date
        ):
            msg = "compaction selected date bounds disagree with annual entries"
            raise OratsOptionCompactionContentError(msg)
        if self.selected_source_entry_count != sum(
            entry.source_entry_count for entry in self.entries
        ):
            msg = "compaction selected source count disagrees with annual entries"
            raise OratsOptionCompactionContentError(msg)
        if self.selected_source_rows != sum(entry.output_rows for entry in self.entries):
            msg = "compaction selected source rows disagree with annual entries"
            raise OratsOptionCompactionContentError(msg)
        if self.total_output_rows != self.selected_source_rows:
            msg = "compaction output rows disagree with selected source rows"
            raise OratsOptionCompactionContentError(msg)
        if self.selected_source_bytes != sum(entry.source_bytes for entry in self.entries):
            msg = "compaction selected source bytes disagree with annual entries"
            raise OratsOptionCompactionContentError(msg)
        if self.total_output_bytes != sum(entry.output_bytes for entry in self.entries):
            msg = "compaction output bytes disagree with annual entries"
            raise OratsOptionCompactionContentError(msg)
        for value in (
            self.selected_source_entry_count,
            self.selected_source_rows,
            self.selected_source_bytes,
            self.total_output_rows,
            self.total_output_bytes,
        ):
            if type(value) is not int or value <= 0:
                msg = "compaction manifest totals must be positive plain integers"
                raise OratsOptionCompactionContentError(msg)

    def _validate_paths_and_tickers(self) -> None:
        projected_tickers = set(self.projected_tickers)
        for entry in self.entries:
            expected_output = f"{self.compaction_spec_sha256}/{entry.year}.parquet"
            if entry.output_relative_path != expected_output:
                msg = "compaction output path disagrees with its specification and year"
                raise OratsOptionCompactionContentError(msg, entry.output_relative_path)
            if entry.receipt_relative_path != f"{expected_output}.receipt.json":
                msg = "compaction receipt path disagrees with its output"
                raise OratsOptionCompactionContentError(msg, entry.receipt_relative_path)
            if not set(entry.observed_tickers).issubset(projected_tickers):
                msg = "annual compaction output has tickers outside the source projection"
                raise OratsOptionCompactionContentError(msg, entry.output_relative_path)

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical aggregate mapping."""
        return {
            "compaction_spec": self.compaction_spec.as_dict(),
            "compaction_spec_sha256": self.compaction_spec_sha256,
            "entries": [entry.as_dict() for entry in self.entries],
            "entries_sha256": self.entries_sha256,
            "normalized_schema_sha256": self.normalized_schema_sha256,
            "projected_tickers": list(self.projected_tickers),
            "schema_version": self.schema_version,
            "selected_first_trade_date": self.selected_first_trade_date.isoformat(),
            "selected_last_trade_date": self.selected_last_trade_date.isoformat(),
            "selected_source_bytes": self.selected_source_bytes,
            "selected_source_entries_sha256": self.selected_source_entries_sha256,
            "selected_source_entry_count": self.selected_source_entry_count,
            "selected_source_rows": self.selected_source_rows,
            "source_manifest_name": self.source_manifest_name,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_projection_entries_sha256": self.source_projection_entries_sha256,
            "source_projection_spec_sha256": self.source_projection_spec_sha256,
            "total_output_bytes": self.total_output_bytes,
            "total_output_rows": self.total_output_rows,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully validate one canonical aggregate manifest."""
        mapping = _exact_mapping(
            raw,
            {
                "compaction_spec",
                "compaction_spec_sha256",
                "entries",
                "entries_sha256",
                "normalized_schema_sha256",
                "projected_tickers",
                "schema_version",
                "selected_first_trade_date",
                "selected_last_trade_date",
                "selected_source_bytes",
                "selected_source_entries_sha256",
                "selected_source_entry_count",
                "selected_source_rows",
                "source_manifest_name",
                "source_manifest_sha256",
                "source_projection_entries_sha256",
                "source_projection_spec_sha256",
                "total_output_bytes",
                "total_output_rows",
            },
            "compaction manifest",
        )
        string_fields = (
            "compaction_spec_sha256",
            "entries_sha256",
            "normalized_schema_sha256",
            "selected_first_trade_date",
            "selected_last_trade_date",
            "selected_source_entries_sha256",
            "source_manifest_name",
            "source_manifest_sha256",
            "source_projection_entries_sha256",
            "source_projection_spec_sha256",
        )
        integer_fields = (
            "schema_version",
            "selected_source_bytes",
            "selected_source_entry_count",
            "selected_source_rows",
            "total_output_bytes",
            "total_output_rows",
        )
        if any(type(mapping[name]) is not str for name in string_fields):
            msg = "compaction manifest string fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        if any(type(mapping[name]) is not int for name in integer_fields):
            msg = "compaction manifest integer fields are malformed"
            raise OratsOptionCompactionContentError(msg)
        entries_raw = mapping["entries"]
        if type(entries_raw) is not list:
            msg = "compaction manifest annual entries are malformed"
            raise OratsOptionCompactionContentError(msg)
        projected_tickers_raw = mapping["projected_tickers"]
        if type(projected_tickers_raw) is not list or any(
            type(value) is not str for value in projected_tickers_raw
        ):
            msg = "compaction manifest projected tickers are malformed"
            raise OratsOptionCompactionContentError(msg)
        try:
            selected_first_trade_date = date.fromisoformat(
                cast("str", mapping["selected_first_trade_date"]),
            )
            selected_last_trade_date = date.fromisoformat(
                cast("str", mapping["selected_last_trade_date"]),
            )
        except ValueError as error:
            msg = "compaction manifest selected date bounds are malformed"
            raise OratsOptionCompactionContentError(msg) from error
        return cls(
            source_manifest_name=cast("str", mapping["source_manifest_name"]),
            source_manifest_sha256=cast("str", mapping["source_manifest_sha256"]),
            source_projection_entries_sha256=cast(
                "str",
                mapping["source_projection_entries_sha256"],
            ),
            source_projection_spec_sha256=cast(
                "str",
                mapping["source_projection_spec_sha256"],
            ),
            projected_tickers=tuple(cast("list[str]", projected_tickers_raw)),
            normalized_schema_sha256=cast("str", mapping["normalized_schema_sha256"]),
            compaction_spec=OratsOptionCompactionSpec.from_dict(mapping["compaction_spec"]),
            compaction_spec_sha256=cast("str", mapping["compaction_spec_sha256"]),
            selected_source_entries_sha256=cast(
                "str",
                mapping["selected_source_entries_sha256"],
            ),
            selected_first_trade_date=selected_first_trade_date,
            selected_last_trade_date=selected_last_trade_date,
            selected_source_entry_count=cast("int", mapping["selected_source_entry_count"]),
            selected_source_rows=cast("int", mapping["selected_source_rows"]),
            selected_source_bytes=cast("int", mapping["selected_source_bytes"]),
            entries=tuple(
                OratsOptionCompactionManifestEntry.from_dict(entry) for entry in entries_raw
            ),
            entries_sha256=cast("str", mapping["entries_sha256"]),
            total_output_rows=cast("int", mapping["total_output_rows"]),
            total_output_bytes=cast("int", mapping["total_output_bytes"]),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class VerifiedOratsOptionCompaction:
    """One verified aggregate manifest and its ordered annual Parquet files."""

    manifest_path: Path
    output_root: Path
    paths: tuple[Path, ...]
    manifest: OratsOptionCompactionManifest

    def scan(self) -> pl.LazyFrame:
        """Lazily scan the verified annual outputs in chronological order."""
        return pl.scan_parquet(list(self.paths), rechunk=False)


@dataclass(frozen=True, slots=True)
class _SelectedYear:
    """Ordered source entries and paths selected for one calendar year."""

    year: int
    entries: tuple[OratsOptionProjectionSetEntry, ...]
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _AnnualMaterializationContext:
    """Shared immutable identities for every annual sink in one run."""

    source: VerifiedOratsOptionProjectionSet
    spec: OratsOptionCompactionSpec
    source_manifest_sha256: str
    output_directory: Path


@dataclass(frozen=True, slots=True)
class _ManifestBuildContext:
    """Shared identities needed to construct the aggregate manifest."""

    source: VerifiedOratsOptionProjectionSet
    spec: OratsOptionCompactionSpec
    source_manifest_sha256: str
    output_root: Path


@dataclass(frozen=True, slots=True)
class _AnnualOutputSummary:
    """Typed aggregate inspection of one newly written annual output."""

    rows: int
    tickers: tuple[str, ...]
    first_date: date
    last_date: date


def materialize_orats_option_compaction(
    source: VerifiedOratsOptionProjectionSet,
    spec: OratsOptionCompactionSpec,
    output_root: str | Path,
    manifest_path: str | Path,
) -> VerifiedOratsOptionCompaction:
    """Stream daily projections into restart-safe annual files and bind one manifest."""
    verified_source = _require_verified_source(source)
    _require_writer_runtime(spec.writer)
    root = _output_root(output_root)
    destination_manifest = Path(manifest_path).expanduser()
    if destination_manifest.exists() or destination_manifest.is_symlink():
        existing = load_verified_orats_option_compaction(
            destination_manifest,
            root,
        )
        if existing.manifest.compaction_spec != spec or not _manifest_matches_source(
            existing.manifest,
            verified_source,
        ):
            msg = (
                "existing compaction manifest disagrees with the requested source or specification"
            )
            raise OratsOptionCompactionContentError(msg, str(destination_manifest))
        return existing
    _validate_new_manifest_path(destination_manifest)
    selected = _select_source_years(verified_source, spec)
    source_manifest_sha256 = _file_sha256(verified_source.manifest_path)
    output_directory = root / orats_option_compaction_spec_sha256(spec)
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink():
        msg = "compaction specification directory must not be a symlink"
        raise OratsOptionCompactionContentError(msg, str(output_directory))
    annual_context = _AnnualMaterializationContext(
        source=verified_source,
        spec=spec,
        source_manifest_sha256=source_manifest_sha256,
        output_directory=output_directory,
    )
    annual = tuple(_materialize_year(selected_year, annual_context) for selected_year in selected)
    if _file_sha256(verified_source.manifest_path) != source_manifest_sha256:
        msg = "source projection manifest changed during annual compaction"
        raise OratsOptionCompactionContentError(msg, str(source.manifest_path))
    manifest = _build_manifest(
        selected,
        annual,
        _ManifestBuildContext(
            source=verified_source,
            spec=spec,
            source_manifest_sha256=source_manifest_sha256,
            output_root=root,
        ),
    )
    _write_json_atomic(destination_manifest, manifest.as_dict(), "compaction manifest")
    return load_verified_orats_option_compaction(destination_manifest, root)


def load_verified_orats_option_compaction(
    manifest_path: str | Path,
    output_root: str | Path,
) -> VerifiedOratsOptionCompaction:
    """Verify a detached compaction manifest, annual receipts, and Parquet outputs."""
    path = _regular_file(manifest_path, "compaction manifest")
    root = _regular_directory(output_root, "compaction output root")
    manifest = _read_manifest(path)
    paths = tuple(_verify_manifest_entry(root, manifest, entry) for entry in manifest.entries)
    return VerifiedOratsOptionCompaction(
        manifest_path=path,
        output_root=root,
        paths=paths,
        manifest=manifest,
    )


def verify_orats_option_compaction_unchanged(source: VerifiedOratsOptionCompaction) -> None:
    """Rehash annual outputs and receipts after a consuming operation."""
    if (
        _read_manifest(_regular_file(source.manifest_path, "compaction manifest"))
        != source.manifest
    ):
        msg = "compaction manifest changed after verification"
        raise OratsOptionCompactionContentError(msg, str(source.manifest_path))
    for entry, path in zip(source.manifest.entries, source.paths, strict=True):
        regular = _regular_file(path, "annual output")
        if (
            regular.stat().st_size != entry.output_bytes
            or _file_sha256(regular) != entry.output_sha256
        ):
            msg = "annual output changed after compaction verification"
            raise OratsOptionCompactionContentError(msg, str(path))
        receipt_path = source.output_root / entry.receipt_relative_path
        if _file_sha256(_regular_file(receipt_path, "annual receipt")) != entry.receipt_sha256:
            msg = "annual receipt changed after compaction verification"
            raise OratsOptionCompactionContentError(msg, str(receipt_path))


def _require_verified_source(
    source: VerifiedOratsOptionProjectionSet,
) -> VerifiedOratsOptionProjectionSet:
    verified = load_verified_orats_option_projection_set(
        source.manifest_path,
        source.inventory_path,
        source.output_root,
    )
    if verified != source:
        msg = "supplied source object disagrees with its verified projection set"
        raise OratsOptionCompactionContentError(msg, str(source.manifest_path))
    return verified


def _require_writer_runtime(spec: ParquetCompactionWriteSpec) -> None:
    if spec.polars_version != pl.__version__:
        msg = (
            "compaction writer Polars version disagrees with the active runtime: "
            f"requested {spec.polars_version}, active {pl.__version__}"
        )
        raise OratsOptionCompactionSpecError(msg)


def _select_source_years(
    source: VerifiedOratsOptionProjectionSet,
    spec: OratsOptionCompactionSpec,
) -> tuple[_SelectedYear, ...]:
    source_first = source.manifest.first_trade_date
    source_end_exclusive = date.fromordinal(source.manifest.last_trade_date.toordinal() + 1)
    if spec.start_date < source_first or spec.end_date_exclusive > source_end_exclusive:
        msg = "compaction interval must be fully covered by the source projection set"
        raise OratsOptionCompactionContentError(msg, str(source.manifest_path))
    pairs = tuple(
        (entry, path)
        for entry, path in zip(source.manifest.entries, source.paths, strict=True)
        if spec.start_date <= entry.trade_date < spec.end_date_exclusive
    )
    if not pairs:
        msg = "compaction interval selects no source trading days"
        raise OratsOptionCompactionContentError(msg, str(source.manifest_path))
    years: list[_SelectedYear] = []
    for year in sorted({entry.trade_date.year for entry, _ in pairs}):
        annual = tuple((entry, path) for entry, path in pairs if entry.trade_date.year == year)
        years.append(
            _SelectedYear(
                year=year,
                entries=tuple(entry for entry, _ in annual),
                paths=tuple(path for _, path in annual),
            ),
        )
    return tuple(years)


def _materialize_year(
    selected: _SelectedYear,
    context: _AnnualMaterializationContext,
) -> MaterializedOratsOptionCompactionYear:
    output = context.output_directory / f"{selected.year}.parquet"
    receipt_path = output.with_name(f"{output.name}.receipt.json")
    existing = _load_existing_year(
        selected,
        context,
        output,
        receipt_path,
    )
    if existing is not None:
        return existing
    partial = output.with_name(f".{output.name}.part")
    if partial.exists() or partial.is_symlink():
        msg = "compaction output partial already exists"
        raise OratsOptionCompactionContentError(msg, str(partial))
    writer = context.spec.writer
    try:
        pl.scan_parquet(list(selected.paths), rechunk=False).sink_parquet(
            partial,
            compression=writer.compression,
            compression_level=writer.compression_level,
            statistics=writer.statistics,
            row_group_size=writer.row_group_size,
            maintain_order=writer.maintain_order,
            sync_on_close=writer.sync_on_close,
            mkdir=False,
            engine=writer.engine,
        )
        _fsync_file(partial)
        _verify_selected_source_year(selected)
        summary = _inspect_annual_output(partial)
        expected_rows = sum(entry.output_rows for entry in selected.entries)
        _validate_annual_summary(partial, selected, summary)
        receipt = OratsOptionCompactionReceipt(
            year=selected.year,
            first_trade_date=summary.first_date,
            last_trade_date=summary.last_date,
            source_manifest_sha256=context.source_manifest_sha256,
            source_projection_entries_sha256=context.source.manifest.entries_sha256,
            source_year_entries_sha256=orats_option_projection_set_entries_sha256(
                selected.entries,
            ),
            source_entry_count=len(selected.entries),
            source_rows=expected_rows,
            source_bytes=sum(entry.output_bytes for entry in selected.entries),
            compaction_spec=context.spec,
            compaction_spec_sha256=orats_option_compaction_spec_sha256(context.spec),
            normalized_schema_sha256=context.source.manifest.normalized_schema_sha256,
            output_name=output.name,
            output_rows=summary.rows,
            output_bytes=partial.stat().st_size,
            output_sha256=_file_sha256(partial),
            observed_tickers=summary.tickers,
        )
        partial.replace(output)
        try:
            _write_json_atomic(receipt_path, receipt.as_dict(), "annual receipt")
        except BaseException:
            output.replace(partial)
            raise
    except pl.exceptions.PolarsError as error:
        partial.unlink(missing_ok=True)
        msg = "streaming annual Parquet compaction failed"
        raise OratsOptionCompactionContentError(msg, str(output)) from error
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return MaterializedOratsOptionCompactionYear(
        path=output,
        receipt_path=receipt_path,
        receipt=receipt,
    )


def _inspect_annual_output(path: Path) -> _AnnualOutputSummary:
    scan = pl.scan_parquet(path, rechunk=False)
    if scan.collect_schema() != pl.Schema(ORATS_NORMALIZED_OPTION_DTYPES):
        msg = "annual output materialized an unsupported ordered schema"
        raise OratsOptionCompactionContentError(msg, str(path))
    summary = scan.select(
        pl.len().alias("rows"),
        pl.col("ticker").unique().sort().implode().alias("tickers"),
        pl.col("quote_date").min().alias("first_date"),
        pl.col("quote_date").max().alias("last_date"),
    ).collect(engine="streaming")
    rows = summary.get_column("rows").item()
    tickers = summary.get_column("tickers").item()
    first_date = summary.get_column("first_date").item()
    last_date = summary.get_column("last_date").item()
    if type(rows) is not int or not isinstance(tickers, pl.Series):
        msg = "annual output aggregate types are unsupported"
        raise OratsOptionCompactionContentError(msg, str(path))
    if type(first_date) is not date or type(last_date) is not date:
        msg = "annual output date aggregate types are unsupported"
        raise OratsOptionCompactionContentError(msg, str(path))
    ticker_values = tickers.to_list()
    if any(type(value) is not str for value in ticker_values):
        msg = "annual output ticker aggregate types are unsupported"
        raise OratsOptionCompactionContentError(msg, str(path))
    return _AnnualOutputSummary(
        rows=rows,
        tickers=tuple(cast("list[str]", ticker_values)),
        first_date=first_date,
        last_date=last_date,
    )


def _validate_annual_summary(
    path: Path,
    selected: _SelectedYear,
    summary: _AnnualOutputSummary,
) -> None:
    expected_rows = sum(entry.output_rows for entry in selected.entries)
    expected_tickers = tuple(
        sorted({ticker for entry in selected.entries for ticker in entry.observed_tickers}),
    )
    if summary.rows != expected_rows:
        msg = "annual output rows disagree with selected daily receipts"
        raise OratsOptionCompactionContentError(msg, str(path))
    if summary.tickers != expected_tickers:
        msg = "annual output tickers disagree with selected daily receipts"
        raise OratsOptionCompactionContentError(msg, str(path))
    if (
        summary.first_date != selected.entries[0].trade_date
        or summary.last_date != selected.entries[-1].trade_date
    ):
        msg = "annual output date bounds disagree with selected daily receipts"
        raise OratsOptionCompactionContentError(msg, str(path))


def _verify_selected_source_year(selected: _SelectedYear) -> None:
    for entry, path in zip(selected.entries, selected.paths, strict=True):
        regular = _regular_file(path, "selected daily source")
        if (
            regular.stat().st_size != entry.output_bytes
            or _file_sha256(regular) != entry.output_sha256
        ):
            msg = "selected daily source changed during compaction"
            raise OratsOptionCompactionContentError(msg, str(path))


def _load_existing_year(
    selected: _SelectedYear,
    context: _AnnualMaterializationContext,
    output: Path,
    receipt_path: Path,
) -> MaterializedOratsOptionCompactionYear | None:
    if not output.exists() and not receipt_path.exists():
        return None
    regular_output = _regular_file(output, "existing annual output")
    regular_receipt = _regular_file(receipt_path, "existing annual receipt")
    receipt = _read_receipt(regular_receipt)
    expected_year_hash = orats_option_projection_set_entries_sha256(selected.entries)
    expected_rows = sum(entry.output_rows for entry in selected.entries)
    if (
        receipt.year != selected.year
        or receipt.first_trade_date != selected.entries[0].trade_date
        or receipt.last_trade_date != selected.entries[-1].trade_date
        or receipt.source_manifest_sha256 != context.source_manifest_sha256
        or receipt.source_projection_entries_sha256 != context.source.manifest.entries_sha256
        or receipt.source_year_entries_sha256 != expected_year_hash
        or receipt.source_entry_count != len(selected.entries)
        or receipt.source_rows != expected_rows
        or receipt.source_bytes != sum(entry.output_bytes for entry in selected.entries)
        or receipt.compaction_spec != context.spec
        or receipt.output_name != output.name
    ):
        msg = "existing annual receipt disagrees with the requested source selection"
        raise OratsOptionCompactionContentError(msg, str(receipt_path))
    _verify_output_identity(regular_output, receipt)
    summary = _inspect_annual_output(regular_output)
    _validate_annual_summary(regular_output, selected, summary)
    return MaterializedOratsOptionCompactionYear(
        path=regular_output,
        receipt_path=regular_receipt,
        receipt=receipt,
    )


def _build_manifest(
    selected: tuple[_SelectedYear, ...],
    annual: tuple[MaterializedOratsOptionCompactionYear, ...],
    context: _ManifestBuildContext,
) -> OratsOptionCompactionManifest:
    entries = tuple(
        OratsOptionCompactionManifestEntry(
            year=artifact.receipt.year,
            output_relative_path=artifact.path.relative_to(context.output_root).as_posix(),
            receipt_relative_path=artifact.receipt_path.relative_to(
                context.output_root,
            ).as_posix(),
            receipt_sha256=_file_sha256(artifact.receipt_path),
            source_year_entries_sha256=artifact.receipt.source_year_entries_sha256,
            first_trade_date=artifact.receipt.first_trade_date,
            last_trade_date=artifact.receipt.last_trade_date,
            source_entry_count=artifact.receipt.source_entry_count,
            source_bytes=artifact.receipt.source_bytes,
            output_rows=artifact.receipt.output_rows,
            output_bytes=artifact.receipt.output_bytes,
            output_sha256=artifact.receipt.output_sha256,
            observed_tickers=artifact.receipt.observed_tickers,
        )
        for artifact in annual
    )
    selected_entries = tuple(entry for year in selected for entry in year.entries)
    return OratsOptionCompactionManifest(
        source_manifest_name=context.source.manifest_path.name,
        source_manifest_sha256=context.source_manifest_sha256,
        source_projection_entries_sha256=context.source.manifest.entries_sha256,
        source_projection_spec_sha256=context.source.manifest.projection_spec_sha256,
        projected_tickers=context.source.manifest.projection_spec.tickers,
        normalized_schema_sha256=context.source.manifest.normalized_schema_sha256,
        compaction_spec=context.spec,
        compaction_spec_sha256=orats_option_compaction_spec_sha256(context.spec),
        selected_source_entries_sha256=orats_option_projection_set_entries_sha256(
            selected_entries,
        ),
        selected_first_trade_date=selected_entries[0].trade_date,
        selected_last_trade_date=selected_entries[-1].trade_date,
        selected_source_entry_count=len(selected_entries),
        selected_source_rows=sum(entry.output_rows for entry in selected_entries),
        selected_source_bytes=sum(entry.output_bytes for entry in selected_entries),
        entries=entries,
        entries_sha256=orats_option_compaction_manifest_entries_sha256(entries),
        total_output_rows=sum(entry.output_rows for entry in entries),
        total_output_bytes=sum(entry.output_bytes for entry in entries),
    )


def _manifest_matches_source(
    manifest: OratsOptionCompactionManifest,
    source: VerifiedOratsOptionProjectionSet,
) -> bool:
    selected = _select_source_years(source, manifest.compaction_spec)
    try:
        _validate_manifest_selection(manifest, selected)
    except OratsOptionCompactionContentError:
        return False
    return (
        manifest.source_manifest_name == source.manifest_path.name
        and manifest.source_manifest_sha256 == _file_sha256(source.manifest_path)
        and manifest.source_projection_entries_sha256 == source.manifest.entries_sha256
        and manifest.source_projection_spec_sha256 == source.manifest.projection_spec_sha256
        and manifest.projected_tickers == source.manifest.projection_spec.tickers
        and manifest.normalized_schema_sha256 == source.manifest.normalized_schema_sha256
        and len(manifest.entries) == len(selected)
    )


def _validate_manifest_selection(
    manifest: OratsOptionCompactionManifest,
    selected: tuple[_SelectedYear, ...],
) -> None:
    entries = tuple(entry for year in selected for entry in year.entries)
    if manifest.selected_source_entries_sha256 != orats_option_projection_set_entries_sha256(
        entries,
    ):
        msg = "compaction selected source entries disagree with its manifest"
        raise OratsOptionCompactionContentError(msg)
    if (
        manifest.selected_first_trade_date != entries[0].trade_date
        or manifest.selected_last_trade_date != entries[-1].trade_date
        or manifest.selected_source_entry_count != len(entries)
        or manifest.selected_source_rows != sum(entry.output_rows for entry in entries)
        or manifest.selected_source_bytes != sum(entry.output_bytes for entry in entries)
    ):
        msg = "compaction selected source aggregates disagree with its manifest"
        raise OratsOptionCompactionContentError(msg)


def _verify_manifest_entry(
    root: Path,
    manifest: OratsOptionCompactionManifest,
    entry: OratsOptionCompactionManifestEntry,
) -> Path:
    output = _regular_child_file(root, root / entry.output_relative_path, "annual output")
    receipt_path = _regular_child_file(
        root,
        root / entry.receipt_relative_path,
        "annual receipt",
    )
    if _file_sha256(receipt_path) != entry.receipt_sha256:
        msg = "annual receipt bytes disagree with the compaction manifest"
        raise OratsOptionCompactionContentError(msg, str(receipt_path))
    receipt = _read_receipt(receipt_path)
    if (
        receipt.year != entry.year
        or receipt.source_manifest_sha256 != manifest.source_manifest_sha256
        or receipt.source_projection_entries_sha256 != manifest.source_projection_entries_sha256
        or receipt.source_year_entries_sha256 != entry.source_year_entries_sha256
        or receipt.compaction_spec != manifest.compaction_spec
        or receipt.first_trade_date != entry.first_trade_date
        or receipt.last_trade_date != entry.last_trade_date
        or receipt.source_entry_count != entry.source_entry_count
        or receipt.source_rows != entry.output_rows
        or receipt.source_bytes != entry.source_bytes
        or receipt.compaction_spec_sha256 != manifest.compaction_spec_sha256
        or receipt.normalized_schema_sha256 != manifest.normalized_schema_sha256
        or receipt.output_name != Path(entry.output_relative_path).name
        or receipt.output_rows != entry.output_rows
        or receipt.output_bytes != entry.output_bytes
        or receipt.output_sha256 != entry.output_sha256
        or receipt.observed_tickers != entry.observed_tickers
    ):
        msg = "annual receipt identity disagrees with the compaction manifest"
        raise OratsOptionCompactionContentError(msg, str(receipt_path))
    _verify_output_identity(output, receipt)
    summary = _inspect_annual_output(output)
    if (
        summary.rows != entry.output_rows
        or summary.tickers != entry.observed_tickers
        or summary.first_date != entry.first_trade_date
        or summary.last_date != entry.last_trade_date
    ):
        msg = "annual output content summary disagrees with the compaction manifest"
        raise OratsOptionCompactionContentError(msg, str(output))
    return output


def _verify_output_identity(path: Path, receipt: OratsOptionCompactionReceipt) -> None:
    if path.name != receipt.output_name:
        msg = "annual output filename disagrees with its receipt"
        raise OratsOptionCompactionContentError(msg, str(path))
    if path.stat().st_size != receipt.output_bytes:
        msg = "annual output byte count disagrees with its receipt"
        raise OratsOptionCompactionContentError(msg, str(path))
    if _file_sha256(path) != receipt.output_sha256:
        msg = "annual output bytes disagree with its receipt SHA-256"
        raise OratsOptionCompactionContentError(msg, str(path))


def _output_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        msg = "compaction output root must be absent or a non-symlink directory"
        raise OratsOptionCompactionContentError(msg, str(root))
    parent = root.parent
    if not parent.is_dir() or parent.is_symlink():
        msg = "compaction output parent must be an existing non-symlink directory"
        raise OratsOptionCompactionContentError(msg, str(parent))
    root.mkdir(exist_ok=True)
    return root


def _validate_new_manifest_path(path: Path) -> None:
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        msg = "compaction manifest parent must be an existing non-symlink directory"
        raise OratsOptionCompactionContentError(msg, str(parent))


def _read_manifest(path: Path) -> OratsOptionCompactionManifest:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "compaction manifest is unreadable"
        raise OratsOptionCompactionContentError(msg, str(path)) from error
    return OratsOptionCompactionManifest.from_dict(raw)


def _read_receipt(path: Path) -> OratsOptionCompactionReceipt:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "annual compaction receipt is unreadable"
        raise OratsOptionCompactionContentError(msg, str(path)) from error
    return OratsOptionCompactionReceipt.from_dict(raw)


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.is_symlink():
        msg = f"{label} must be an existing non-symlink file"
        raise OratsOptionCompactionContentError(msg, str(candidate))
    return candidate


def _regular_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_dir() or candidate.is_symlink():
        msg = f"{label} must be an existing non-symlink directory"
        raise OratsOptionCompactionContentError(msg, str(candidate))
    return candidate


def _regular_child_file(root: Path, path: Path, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        msg = f"{label} must remain below the declared output root"
        raise OratsOptionCompactionContentError(msg, str(path)) from error
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if not current.is_dir() or current.is_symlink():
            msg = f"{label} parent must be a non-symlink directory"
            raise OratsOptionCompactionContentError(msg, str(current))
    return _regular_file(path, label)


def _validate_relative_path(value: str, label: str) -> None:
    if type(value) is not str or not value:
        msg = f"compaction {label} path is malformed"
        raise OratsOptionCompactionContentError(msg)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        msg = f"compaction {label} path must be canonical and relative"
        raise OratsOptionCompactionContentError(msg, value)


def _write_json_atomic(path: Path, value: object, label: str) -> None:
    if path.exists() or path.is_symlink():
        msg = f"{label} already exists"
        raise OratsOptionCompactionContentError(msg, str(path))
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists() or temporary.is_symlink():
        msg = f"{label} partial already exists"
        raise OratsOptionCompactionContentError(msg, str(temporary))
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


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(ORATS_OPTION_COMPACTION_BLOCK_BYTES):
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
        raise OratsOptionCompactionContentError(msg)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected or any(type(key) is not str for key in mapping):
        msg = f"{label} keys must be exactly {sorted(expected)!r}"
        raise OratsOptionCompactionContentError(msg)
    return cast("dict[str, object]", mapping)


def _is_sha256(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ORATS_OPTION_COMPACTION_BLOCK_BYTES",
    "ORATS_OPTION_COMPACTION_COMPRESSION",
    "ORATS_OPTION_COMPACTION_COMPRESSION_LEVEL",
    "ORATS_OPTION_COMPACTION_MANIFEST_VERSION",
    "ORATS_OPTION_COMPACTION_RECEIPT_VERSION",
    "ORATS_OPTION_COMPACTION_ROW_GROUP_SIZE",
    "ORATS_OPTION_COMPACTION_VERSION",
    "MaterializedOratsOptionCompactionYear",
    "OratsOptionCompactionContentError",
    "OratsOptionCompactionError",
    "OratsOptionCompactionManifest",
    "OratsOptionCompactionManifestEntry",
    "OratsOptionCompactionReceipt",
    "OratsOptionCompactionSpec",
    "OratsOptionCompactionSpecError",
    "ParquetCompactionWriteSpec",
    "VerifiedOratsOptionCompaction",
    "load_verified_orats_option_compaction",
    "materialize_orats_option_compaction",
    "orats_option_compaction_manifest_entries_sha256",
    "orats_option_compaction_spec_sha256",
    "verify_orats_option_compaction_unchanged",
]
