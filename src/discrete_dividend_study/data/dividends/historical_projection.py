"""Immutable projection of licensed historical cash-dividend Parquet inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Self, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

HISTORICAL_DIVIDEND_SOURCE_DTYPES: Final = {
    "ticker": pl.String,
    "ex_date": pl.Date,
    "amount": pl.Float64,
    "freq": pl.Int64,
    "pay_date": pl.Date,
}
HISTORICAL_DIVIDEND_PROJECTION_DTYPES: Final = {
    "ticker": pl.String,
    "ex_date": pl.Date,
    "amount": pl.Float64,
}
HISTORICAL_DIVIDEND_PROJECTION_VERSION: Final = 1
HISTORICAL_DIVIDEND_PROJECTION_RECEIPT_VERSION: Final = 1
HISTORICAL_DIVIDEND_PROJECTION_COMPRESSION: Final = "zstd"
HISTORICAL_DIVIDEND_PROJECTION_COMPRESSION_LEVEL: Final = 3
HISTORICAL_DIVIDEND_PROJECTION_ROW_GROUP_SIZE: Final = 65_536

_OUTPUT_NAME: Final = "historical_dividends.parquet"
_SHA256_LENGTH: Final = 64
_HASH_BLOCK_BYTES: Final = 8 * 1024 * 1024
_TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9.-]*")


class HistoricalDividendProjectionError(RuntimeError):
    """Base class for rejected licensed source data or projected output."""


@dataclass(slots=True)
class HistoricalDividendProjectionSpecError(HistoricalDividendProjectionError):
    """Projection settings are incomplete, ambiguous, or unsupported."""

    reason: str

    def __str__(self) -> str:
        """Describe the invalid static setting."""
        return self.reason


@dataclass(slots=True)
class HistoricalDividendProjectionContentError(HistoricalDividendProjectionError):
    """A source or projected artifact violates its declared identity."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"historical-dividend projection failed{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class HistoricalDividendProjectionSpec:
    """Exact ticker/date projection and physical Parquet identity."""

    tickers: tuple[str, ...]
    maximum_ex_date: date
    compression: Literal["zstd"] = HISTORICAL_DIVIDEND_PROJECTION_COMPRESSION
    compression_level: int = HISTORICAL_DIVIDEND_PROJECTION_COMPRESSION_LEVEL
    row_group_size: int = HISTORICAL_DIVIDEND_PROJECTION_ROW_GROUP_SIZE
    schema_version: int = HISTORICAL_DIVIDEND_PROJECTION_VERSION

    def __post_init__(self) -> None:
        """Reject inferred ticker canonicalization and unsupported physical settings."""
        if self.schema_version != HISTORICAL_DIVIDEND_PROJECTION_VERSION:
            msg = "unsupported historical-dividend projection schema version"
            raise HistoricalDividendProjectionSpecError(
                msg,
            )
        if not self.tickers:
            msg = "historical-dividend projection requires at least one ticker"
            raise HistoricalDividendProjectionSpecError(
                msg,
            )
        if self.tickers != tuple(sorted(self.tickers)) or len(self.tickers) != len(
            set(self.tickers),
        ):
            msg = "historical-dividend projection tickers must be sorted and unique"
            raise HistoricalDividendProjectionSpecError(
                msg,
            )
        if any(_TICKER_PATTERN.fullmatch(ticker) is None for ticker in self.tickers):
            msg = "historical-dividend projection tickers must use canonical uppercase symbols"
            raise HistoricalDividendProjectionSpecError(
                msg,
            )
        if type(self.maximum_ex_date) is not date:
            msg = "historical-dividend maximum ex-date must be a plain date"
            raise HistoricalDividendProjectionSpecError(
                msg,
            )
        if self.compression != HISTORICAL_DIVIDEND_PROJECTION_COMPRESSION:
            msg = "historical-dividend projection compression must be zstd"
            raise HistoricalDividendProjectionSpecError(
                msg,
            )
        if self.compression_level != HISTORICAL_DIVIDEND_PROJECTION_COMPRESSION_LEVEL:
            msg = "historical-dividend projection compression level must be 3"
            raise HistoricalDividendProjectionSpecError(
                msg,
            )
        if self.row_group_size != HISTORICAL_DIVIDEND_PROJECTION_ROW_GROUP_SIZE:
            msg = "historical-dividend projection row group size must be 65536"
            raise HistoricalDividendProjectionSpecError(
                msg,
            )

    def as_dict(self) -> dict[str, object]:
        """Return the canonical projection identity mapping."""
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "maximum_ex_date": self.maximum_ex_date.isoformat(),
            "row_group_size": self.row_group_size,
            "schema_version": self.schema_version,
            "tickers": list(self.tickers),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and validate an exact projection identity."""
        mapping = _exact_mapping(
            raw,
            {
                "compression",
                "compression_level",
                "maximum_ex_date",
                "row_group_size",
                "schema_version",
                "tickers",
            },
            "projection spec",
        )
        compression = mapping["compression"]
        compression_level = mapping["compression_level"]
        maximum_ex_date = mapping["maximum_ex_date"]
        row_group_size = mapping["row_group_size"]
        schema_version = mapping["schema_version"]
        tickers_raw = mapping["tickers"]
        if (
            type(compression) is not str
            or type(compression_level) is not int
            or type(maximum_ex_date) is not str
            or type(row_group_size) is not int
            or type(schema_version) is not int
            or type(tickers_raw) is not list
            or any(type(ticker) is not str for ticker in tickers_raw)
        ):
            msg = "projection spec fields are malformed"
            raise HistoricalDividendProjectionContentError(msg)
        try:
            parsed_maximum = date.fromisoformat(maximum_ex_date)
        except ValueError as error:
            msg = "projection maximum ex-date is malformed"
            raise HistoricalDividendProjectionContentError(
                msg,
            ) from error
        return cls(
            tickers=tuple(cast("list[str]", tickers_raw)),
            maximum_ex_date=parsed_maximum,
            compression=cast("Literal['zstd']", compression),
            compression_level=compression_level,
            row_group_size=row_group_size,
            schema_version=schema_version,
        )


def historical_dividend_projection_spec_sha256(
    spec: HistoricalDividendProjectionSpec,
) -> str:
    """Return the exact ticker/date/physical projection identity."""
    return hashlib.sha256(_canonical_json_bytes(spec.as_dict())).hexdigest()


def historical_dividend_source_schema_sha256() -> str:
    """Return the ordered licensed source schema identity."""
    return _schema_sha256(HISTORICAL_DIVIDEND_SOURCE_DTYPES)


def historical_dividend_projection_schema_sha256() -> str:
    """Return the ordered study projection schema identity."""
    return _schema_sha256(HISTORICAL_DIVIDEND_PROJECTION_DTYPES)


@dataclass(frozen=True, slots=True)
class HistoricalDividendSourceIdentity:
    """Pinned byte, row, and schema identity of one licensed source Parquet."""

    name: str
    size_bytes: int
    sha256: str
    rows: int
    schema_sha256: str

    def __post_init__(self) -> None:
        """Reject malformed or unsupported source identities."""
        if not self.name or Path(self.name).name != self.name:
            msg = "source name is malformed"
            raise HistoricalDividendProjectionContentError(msg)
        if self.size_bytes <= 0 or self.rows <= 0 or not _is_sha256(self.sha256):
            msg = "source byte identity is malformed"
            raise HistoricalDividendProjectionContentError(msg)
        if self.schema_sha256 != historical_dividend_source_schema_sha256():
            msg = "source schema hash is unsupported"
            raise HistoricalDividendProjectionContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the pinned source identity mapping."""
        return {
            "name": self.name,
            "rows": self.rows,
            "schema_sha256": self.schema_sha256,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and validate one pinned source identity."""
        mapping = _exact_mapping(
            raw,
            {"name", "rows", "schema_sha256", "sha256", "size_bytes"},
            "source identity",
        )
        if (
            type(mapping["name"]) is not str
            or type(mapping["rows"]) is not int
            or type(mapping["schema_sha256"]) is not str
            or type(mapping["sha256"]) is not str
            or type(mapping["size_bytes"]) is not int
        ):
            msg = "source identity fields are malformed"
            raise HistoricalDividendProjectionContentError(msg)
        return cls(
            name=mapping["name"],
            rows=mapping["rows"],
            schema_sha256=mapping["schema_sha256"],
            sha256=mapping["sha256"],
            size_bytes=mapping["size_bytes"],
        )


@dataclass(frozen=True, slots=True)
class HistoricalDividendProjectionReceipt:
    """Complete source/spec/quality/output identity for one projected artifact."""

    source: HistoricalDividendSourceIdentity
    projection_spec: HistoricalDividendProjectionSpec
    projection_spec_sha256: str
    projection_schema_sha256: str
    source_tickers: int
    source_ex_date_min: date
    source_ex_date_max: date
    source_nonpositive_amount_rows: int
    source_negative_amount_rows: int
    source_pay_before_ex_rows: int
    excluded_ticker_rows: int
    excluded_after_maximum_ex_date_rows: int
    output_name: str
    output_rows: int
    output_bytes: int
    output_sha256: str
    output_ex_date_min: date
    output_ex_date_max: date
    observed_tickers: tuple[str, ...]
    schema_version: int = HISTORICAL_DIVIDEND_PROJECTION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        """Validate cross-field source, selection, and output identities."""
        self._validate_identities()
        self._validate_counts()
        self._validate_dates_and_universe()

    def _validate_identities(self) -> None:
        if self.schema_version != HISTORICAL_DIVIDEND_PROJECTION_RECEIPT_VERSION:
            msg = "unsupported historical-dividend projection receipt version"
            raise HistoricalDividendProjectionContentError(
                msg,
            )
        if self.projection_spec_sha256 != historical_dividend_projection_spec_sha256(
            self.projection_spec,
        ):
            msg = "projection spec hash disagrees with its embedded specification"
            raise HistoricalDividendProjectionContentError(
                msg,
            )
        if self.projection_schema_sha256 != historical_dividend_projection_schema_sha256():
            msg = "projection schema hash is unsupported"
            raise HistoricalDividendProjectionContentError(msg)

    def _validate_counts(self) -> None:
        counts = (
            self.source_tickers,
            self.source_nonpositive_amount_rows,
            self.source_negative_amount_rows,
            self.source_pay_before_ex_rows,
            self.excluded_ticker_rows,
            self.excluded_after_maximum_ex_date_rows,
            self.output_rows,
            self.output_bytes,
        )
        if self.source_tickers <= 0 or any(value < 0 for value in counts[1:]):
            msg = "receipt counts are malformed"
            raise HistoricalDividendProjectionContentError(msg)
        if self.output_rows <= 0 or self.output_bytes <= 0:
            msg = "output must be nonempty"
            raise HistoricalDividendProjectionContentError(msg)
        if self.source_negative_amount_rows > self.source_nonpositive_amount_rows:
            msg = "negative source count exceeds nonpositive source count"
            raise HistoricalDividendProjectionContentError(
                msg,
            )
        if (
            self.output_rows + self.excluded_ticker_rows + self.excluded_after_maximum_ex_date_rows
            != self.source.rows
        ):
            msg = "receipt row accounting disagrees with the source"
            raise HistoricalDividendProjectionContentError(
                msg,
            )

    def _validate_dates_and_universe(self) -> None:
        if self.source_ex_date_min > self.source_ex_date_max:
            msg = "source date bounds are inverted"
            raise HistoricalDividendProjectionContentError(msg)
        if self.output_ex_date_min > self.output_ex_date_max:
            msg = "output date bounds are inverted"
            raise HistoricalDividendProjectionContentError(msg)
        if self.output_ex_date_max > self.projection_spec.maximum_ex_date:
            msg = "output exceeds the declared maximum ex-date"
            raise HistoricalDividendProjectionContentError(
                msg,
            )
        if self.output_name != _OUTPUT_NAME or not _is_sha256(self.output_sha256):
            msg = "output identity is malformed"
            raise HistoricalDividendProjectionContentError(msg)
        if self.observed_tickers != self.projection_spec.tickers:
            msg = "output must contain every declared ticker exactly once in its universe identity"
            raise HistoricalDividendProjectionContentError(
                msg,
            )

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical receipt mapping."""
        return {
            "excluded_after_maximum_ex_date_rows": self.excluded_after_maximum_ex_date_rows,
            "excluded_ticker_rows": self.excluded_ticker_rows,
            "observed_tickers": list(self.observed_tickers),
            "output_bytes": self.output_bytes,
            "output_ex_date_max": self.output_ex_date_max.isoformat(),
            "output_ex_date_min": self.output_ex_date_min.isoformat(),
            "output_name": self.output_name,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "projection_schema_sha256": self.projection_schema_sha256,
            "projection_spec": self.projection_spec.as_dict(),
            "projection_spec_sha256": self.projection_spec_sha256,
            "schema_version": self.schema_version,
            "source": self.source.as_dict(),
            "source_ex_date_max": self.source_ex_date_max.isoformat(),
            "source_ex_date_min": self.source_ex_date_min.isoformat(),
            "source_negative_amount_rows": self.source_negative_amount_rows,
            "source_nonpositive_amount_rows": self.source_nonpositive_amount_rows,
            "source_pay_before_ex_rows": self.source_pay_before_ex_rows,
            "source_tickers": self.source_tickers,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully revalidate one projection receipt."""
        expected = {
            "excluded_after_maximum_ex_date_rows",
            "excluded_ticker_rows",
            "observed_tickers",
            "output_bytes",
            "output_ex_date_max",
            "output_ex_date_min",
            "output_name",
            "output_rows",
            "output_sha256",
            "projection_schema_sha256",
            "projection_spec",
            "projection_spec_sha256",
            "schema_version",
            "source",
            "source_ex_date_max",
            "source_ex_date_min",
            "source_negative_amount_rows",
            "source_nonpositive_amount_rows",
            "source_pay_before_ex_rows",
            "source_tickers",
        }
        mapping = _exact_mapping(raw, expected, "projection receipt")
        string_fields = (
            "output_ex_date_max",
            "output_ex_date_min",
            "output_name",
            "output_sha256",
            "projection_schema_sha256",
            "projection_spec_sha256",
            "source_ex_date_max",
            "source_ex_date_min",
        )
        integer_fields = (
            "excluded_after_maximum_ex_date_rows",
            "excluded_ticker_rows",
            "output_bytes",
            "output_rows",
            "schema_version",
            "source_negative_amount_rows",
            "source_nonpositive_amount_rows",
            "source_pay_before_ex_rows",
            "source_tickers",
        )
        observed_raw = mapping["observed_tickers"]
        if (
            any(type(mapping[name]) is not str for name in string_fields)
            or any(type(mapping[name]) is not int for name in integer_fields)
            or type(observed_raw) is not list
            or any(type(value) is not str for value in observed_raw)
        ):
            msg = "projection receipt fields are malformed"
            raise HistoricalDividendProjectionContentError(msg)
        try:
            parsed_dates = {
                name: date.fromisoformat(cast("str", mapping[name])) for name in string_fields[:2]
            }
            parsed_source_min = date.fromisoformat(cast("str", mapping["source_ex_date_min"]))
            parsed_source_max = date.fromisoformat(cast("str", mapping["source_ex_date_max"]))
        except ValueError as error:
            msg = "projection receipt date fields are malformed"
            raise HistoricalDividendProjectionContentError(
                msg,
            ) from error
        return cls(
            source=HistoricalDividendSourceIdentity.from_dict(mapping["source"]),
            projection_spec=HistoricalDividendProjectionSpec.from_dict(
                mapping["projection_spec"],
            ),
            projection_spec_sha256=cast("str", mapping["projection_spec_sha256"]),
            projection_schema_sha256=cast("str", mapping["projection_schema_sha256"]),
            source_tickers=cast("int", mapping["source_tickers"]),
            source_ex_date_min=parsed_source_min,
            source_ex_date_max=parsed_source_max,
            source_nonpositive_amount_rows=cast(
                "int",
                mapping["source_nonpositive_amount_rows"],
            ),
            source_negative_amount_rows=cast("int", mapping["source_negative_amount_rows"]),
            source_pay_before_ex_rows=cast("int", mapping["source_pay_before_ex_rows"]),
            excluded_ticker_rows=cast("int", mapping["excluded_ticker_rows"]),
            excluded_after_maximum_ex_date_rows=cast(
                "int",
                mapping["excluded_after_maximum_ex_date_rows"],
            ),
            output_name=cast("str", mapping["output_name"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            output_ex_date_min=parsed_dates["output_ex_date_min"],
            output_ex_date_max=parsed_dates["output_ex_date_max"],
            observed_tickers=tuple(cast("list[str]", observed_raw)),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class MaterializedHistoricalDividendProjection:
    """Verified study projection and its complete receipt."""

    path: Path
    receipt: HistoricalDividendProjectionReceipt


@dataclass(frozen=True, slots=True)
class _SourceDiagnostics:
    """Aggregate-only licensed source diagnostics used in the receipt."""

    rows: int
    tickers: int
    ex_date_min: date
    ex_date_max: date
    nonpositive_amount_rows: int
    negative_amount_rows: int
    pay_before_ex_rows: int
    excluded_ticker_rows: int
    excluded_after_maximum_ex_date_rows: int


def materialize_historical_dividend_projection(
    source_path: str | Path,
    *,
    expected_source_sha256: str,
    expected_source_rows: int,
    spec: HistoricalDividendProjectionSpec,
    output_root: str | Path,
) -> MaterializedHistoricalDividendProjection:
    """Validate one licensed source and write an immutable bounded study projection."""
    source = _verify_source_path(
        source_path,
        expected_sha256=expected_source_sha256,
        expected_rows=expected_source_rows,
    )
    output_directory = (
        _output_root(output_root) / source.sha256 / historical_dividend_projection_spec_sha256(spec)
    )
    output_path = output_directory / _OUTPUT_NAME
    receipt_path = output_path.with_name(f"{output_path.name}.receipt.json")
    existing = _load_existing(source, spec, output_path, receipt_path)
    if existing is not None:
        return existing

    source_stat = _regular_file_stat(Path(source_path), "licensed source")
    frame, diagnostics = _collect_projection(Path(source_path), source, spec)
    _verify_unchanged_source(Path(source_path), source, source_stat)
    output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_private_directory(output_directory)
    temporary_output = output_directory / f".{_OUTPUT_NAME}.{os.getpid()}.tmp"
    temporary_receipt = output_directory / f".{_OUTPUT_NAME}.{os.getpid()}.receipt.tmp"
    if temporary_output.exists() or temporary_receipt.exists():
        msg = "temporary projection path already exists"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(output_directory),
        )
    try:
        frame.write_parquet(
            temporary_output,
            compression=spec.compression,
            compression_level=spec.compression_level,
            row_group_size=spec.row_group_size,
            statistics="full",
        )
        temporary_output.chmod(0o600)
        output_stat = _regular_file_stat(temporary_output, "temporary projection")
        output_sha256 = _sha256_file(temporary_output)
        output_ex_date_min = frame.get_column("ex_date").min()
        output_ex_date_max = frame.get_column("ex_date").max()
        if type(output_ex_date_min) is not date or type(output_ex_date_max) is not date:
            msg = "projected date bounds are malformed"
            raise HistoricalDividendProjectionContentError(msg)
        receipt = HistoricalDividendProjectionReceipt(
            source=source,
            projection_spec=spec,
            projection_spec_sha256=historical_dividend_projection_spec_sha256(spec),
            projection_schema_sha256=historical_dividend_projection_schema_sha256(),
            source_tickers=diagnostics.tickers,
            source_ex_date_min=diagnostics.ex_date_min,
            source_ex_date_max=diagnostics.ex_date_max,
            source_nonpositive_amount_rows=diagnostics.nonpositive_amount_rows,
            source_negative_amount_rows=diagnostics.negative_amount_rows,
            source_pay_before_ex_rows=diagnostics.pay_before_ex_rows,
            excluded_ticker_rows=diagnostics.excluded_ticker_rows,
            excluded_after_maximum_ex_date_rows=(diagnostics.excluded_after_maximum_ex_date_rows),
            output_name=_OUTPUT_NAME,
            output_rows=frame.height,
            output_bytes=output_stat.st_size,
            output_sha256=output_sha256,
            output_ex_date_min=output_ex_date_min,
            output_ex_date_max=output_ex_date_max,
            observed_tickers=tuple(frame.get_column("ticker").unique().sort().to_list()),
        )
        _write_new_private_json(temporary_receipt, receipt.as_dict())
        if output_path.exists() or receipt_path.exists():
            msg = "projection destination appeared during materialization"
            raise HistoricalDividendProjectionContentError(
                msg,
                str(output_directory),
            )
        temporary_output.replace(output_path)
        temporary_receipt.replace(receipt_path)
        return _verify_materialized(output_path, receipt_path, receipt)
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_receipt.unlink(missing_ok=True)


def load_verified_historical_dividend_projection(
    path: str | Path,
) -> MaterializedHistoricalDividendProjection:
    """Load and fully verify one projected output without requiring the licensed source."""
    output_path = Path(path).expanduser()
    receipt_path = output_path.with_name(f"{output_path.name}.receipt.json")
    _regular_file_stat(output_path, "projected output")
    _regular_file_stat(receipt_path, "projection receipt")
    try:
        raw = json.loads(receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "projection receipt is unreadable"
        raise HistoricalDividendProjectionContentError(msg, str(receipt_path)) from error
    receipt = HistoricalDividendProjectionReceipt.from_dict(raw)
    if output_path.name != receipt.output_name:
        msg = "projected output name disagrees with its receipt"
        raise HistoricalDividendProjectionContentError(msg, str(output_path))
    return _verify_materialized(output_path, receipt_path, receipt)


def _verify_source_path(
    source_path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> HistoricalDividendSourceIdentity:
    path = Path(source_path)
    if not _is_sha256(expected_sha256):
        msg = "expected source SHA-256 must be 64 lowercase hexadecimal characters"
        raise HistoricalDividendProjectionSpecError(
            msg,
        )
    if type(expected_rows) is not int or expected_rows <= 0:
        msg = "expected source rows must be a plain positive int"
        raise HistoricalDividendProjectionSpecError(
            msg,
        )
    source_stat = _regular_file_stat(path, "licensed source")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        msg = "licensed source SHA-256 disagrees with the required identity"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    lazy = _scan_exact_source(path)
    observed_rows = lazy.select(pl.len()).collect().item()
    if type(observed_rows) is not int or observed_rows != expected_rows:
        msg = "licensed source row count disagrees with the required identity"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    return HistoricalDividendSourceIdentity(
        name=path.name,
        size_bytes=source_stat.st_size,
        sha256=observed_sha256,
        rows=observed_rows,
        schema_sha256=historical_dividend_source_schema_sha256(),
    )


def _collect_projection(
    path: Path,
    source: HistoricalDividendSourceIdentity,
    spec: HistoricalDividendProjectionSpec,
) -> tuple[pl.DataFrame, _SourceDiagnostics]:
    lazy = _scan_exact_source(path)
    diagnostic_row = (
        lazy.select(
            pl.len().alias("rows"),
            pl.col("ticker").n_unique().alias("tickers"),
            pl.col("ex_date").min().alias("ex_date_min"),
            pl.col("ex_date").max().alias("ex_date_max"),
            pl.any_horizontal(pl.all().is_null()).sum().alias("rows_with_nulls"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_tickers"),
            (~pl.col("amount").is_finite()).fill_null(value=False).sum().alias("nonfinite_amounts"),
            (pl.col("amount") <= 0.0).fill_null(value=False).sum().alias("nonpositive_amounts"),
            (pl.col("amount") < 0.0).fill_null(value=False).sum().alias("negative_amounts"),
            (pl.col("pay_date") < pl.col("ex_date"))
            .fill_null(value=False)
            .sum()
            .alias("pay_before_ex"),
            pl.struct(tuple(HISTORICAL_DIVIDEND_SOURCE_DTYPES))
            .is_duplicated()
            .sum()
            .alias("duplicate_source_rows"),
            (~pl.col("ticker").is_in(spec.tickers))
            .fill_null(value=False)
            .sum()
            .alias("excluded_ticker_rows"),
            (
                pl.col("ticker").is_in(spec.tickers)
                & (pl.col("ex_date") > pl.lit(spec.maximum_ex_date))
            )
            .fill_null(value=False)
            .sum()
            .alias("excluded_after_maximum_ex_date_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    violations = tuple(
        (name, _required_int(diagnostic_row, name))
        for name in (
            "rows_with_nulls",
            "blank_tickers",
            "nonfinite_amounts",
            "duplicate_source_rows",
        )
        if _required_int(diagnostic_row, name) != 0
    )
    if violations:
        msg = f"licensed source aggregate violations: {violations!r}"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    projected = (
        lazy.filter(
            pl.col("ticker").is_in(spec.tickers),
            pl.col("ex_date") <= pl.lit(spec.maximum_ex_date),
        )
        .select(tuple(HISTORICAL_DIVIDEND_PROJECTION_DTYPES))
        .sort("ticker", "ex_date", "amount")
        .collect()
    )
    if projected.is_empty():
        msg = "projection selected no rows"
        raise HistoricalDividendProjectionContentError(msg, str(path))
    projected_tickers = tuple(projected.get_column("ticker").unique().sort().to_list())
    if projected_tickers != spec.tickers:
        missing = len(set(spec.tickers) - set(projected_tickers))
        msg = f"projection is missing {missing} declared ticker histories"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    projected_violations = projected.select(
        (pl.col("amount") < 0.0).sum().alias("negative_amounts"),
        pl.struct(tuple(HISTORICAL_DIVIDEND_PROJECTION_DTYPES))
        .is_duplicated()
        .sum()
        .alias("duplicate_rows"),
    ).row(0, named=True)
    invalid_projected = tuple(
        (name, _required_int(projected_violations, name))
        for name in ("negative_amounts", "duplicate_rows")
        if _required_int(projected_violations, name) != 0
    )
    if invalid_projected:
        msg = f"selected history aggregate violations: {invalid_projected!r}"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    rows = _required_int(diagnostic_row, "rows")
    if rows != source.rows:
        msg = "licensed source changed while collecting diagnostics"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    ex_date_min = diagnostic_row["ex_date_min"]
    ex_date_max = diagnostic_row["ex_date_max"]
    if type(ex_date_min) is not date or type(ex_date_max) is not date:
        msg = "source date bounds are malformed"
        raise HistoricalDividendProjectionContentError(msg, str(path))
    diagnostics = _SourceDiagnostics(
        rows=rows,
        tickers=_required_int(diagnostic_row, "tickers"),
        ex_date_min=ex_date_min,
        ex_date_max=ex_date_max,
        nonpositive_amount_rows=_required_int(diagnostic_row, "nonpositive_amounts"),
        negative_amount_rows=_required_int(diagnostic_row, "negative_amounts"),
        pay_before_ex_rows=_required_int(diagnostic_row, "pay_before_ex"),
        excluded_ticker_rows=_required_int(diagnostic_row, "excluded_ticker_rows"),
        excluded_after_maximum_ex_date_rows=_required_int(
            diagnostic_row,
            "excluded_after_maximum_ex_date_rows",
        ),
    )
    if diagnostics.rows != (
        projected.height
        + diagnostics.excluded_ticker_rows
        + diagnostics.excluded_after_maximum_ex_date_rows
    ):
        msg = "projection row accounting is incomplete"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    return projected, diagnostics


def _load_existing(
    source: HistoricalDividendSourceIdentity,
    spec: HistoricalDividendProjectionSpec,
    output_path: Path,
    receipt_path: Path,
) -> MaterializedHistoricalDividendProjection | None:
    if not output_path.exists() and not receipt_path.exists():
        return None
    if not output_path.exists() or not receipt_path.exists():
        msg = "existing projection is incomplete"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(output_path.parent),
        )
    try:
        raw = json.loads(receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "existing projection receipt is unreadable"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(receipt_path),
        ) from error
    receipt = HistoricalDividendProjectionReceipt.from_dict(raw)
    if receipt.source != source or receipt.projection_spec != spec:
        msg = "existing projection receipt disagrees with the requested source or specification"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(receipt_path),
        )
    return _verify_materialized(output_path, receipt_path, receipt)


def _verify_materialized(
    output_path: Path,
    receipt_path: Path,
    receipt: HistoricalDividendProjectionReceipt,
) -> MaterializedHistoricalDividendProjection:
    output_stat = _regular_file_stat(output_path, "projected output")
    _regular_file_stat(receipt_path, "projection receipt")
    if output_stat.st_size != receipt.output_bytes:
        msg = "projected output byte count disagrees with its receipt"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(output_path),
        )
    if _sha256_file(output_path) != receipt.output_sha256:
        msg = "projected output receipt SHA-256 disagrees with local bytes"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(output_path),
        )
    lazy = pl.scan_parquet(output_path, rechunk=False)
    schema = lazy.collect_schema()
    if schema != pl.Schema(HISTORICAL_DIVIDEND_PROJECTION_DTYPES):
        msg = "projected output schema disagrees with its receipt contract"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(output_path),
        )
    observed = lazy.select(
        pl.len().alias("rows"),
        pl.col("ex_date").min().alias("ex_date_min"),
        pl.col("ex_date").max().alias("ex_date_max"),
    ).collect()
    row = observed.row(0, named=True)
    if _required_int(row, "rows") != receipt.output_rows:
        msg = "projected output row count disagrees with its receipt"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(output_path),
        )
    observed_tickers = tuple(
        lazy.select("ticker").unique().sort("ticker").collect().get_column("ticker").to_list(),
    )
    if (
        row["ex_date_min"] != receipt.output_ex_date_min
        or row["ex_date_max"] != receipt.output_ex_date_max
        or observed_tickers != receipt.observed_tickers
    ):
        msg = "projected output content summary disagrees with its receipt"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(output_path),
        )
    return MaterializedHistoricalDividendProjection(path=output_path, receipt=receipt)


def _scan_exact_source(path: Path) -> pl.LazyFrame:
    try:
        lazy = pl.scan_parquet(path, rechunk=False)
        schema = lazy.collect_schema()
    except (OSError, pl.exceptions.PolarsError) as error:
        msg = "licensed source is not a readable Parquet file"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        ) from error
    if schema != pl.Schema(HISTORICAL_DIVIDEND_SOURCE_DTYPES):
        observed = tuple((name, str(dtype)) for name, dtype in schema.items())
        expected = tuple(
            (name, str(dtype)) for name, dtype in HISTORICAL_DIVIDEND_SOURCE_DTYPES.items()
        )
        msg = f"licensed source schema mismatch: expected {expected!r}; observed {observed!r}"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    return lazy


def _verify_unchanged_source(
    path: Path,
    source: HistoricalDividendSourceIdentity,
    before: os.stat_result,
) -> None:
    after = _regular_file_stat(path, "licensed source")
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_after != identity_before or _sha256_file(path) != source.sha256:
        msg = "licensed source changed during projection"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )


def _output_root(value: str | Path) -> Path:
    root = Path(value)
    if root.exists():
        _require_private_directory(root)
    else:
        root.mkdir(parents=True, mode=0o700)
        _require_private_directory(root)
    return root


def _require_private_directory(path: Path) -> None:
    file_stat = path.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISDIR(file_stat.st_mode):
        msg = "projection directory must be a real directory"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )


def _regular_file_stat(path: Path, label: str) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except OSError as error:
        msg = f"{label} is unavailable"
        raise HistoricalDividendProjectionContentError(msg, str(path)) from error
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        msg = f"{label} must be a regular non-symlink file"
        raise HistoricalDividendProjectionContentError(
            msg,
            str(path),
        )
    if file_stat.st_size <= 0:
        msg = f"{label} must be nonempty"
        raise HistoricalDividendProjectionContentError(msg, str(path))
    return file_stat


def _write_new_private_json(path: Path, value: object) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, mode="wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: Mapping[str, type[pl.DataType]]) -> str:
    payload = [(name, str(dtype)) for name, dtype in schema.items()]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _is_sha256(value: str) -> bool:
    return (
        len(value) == _SHA256_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_int(row: dict[str, object], name: str) -> int:
    value = row[name]
    if type(value) is not int:
        msg = f"aggregate diagnostic {name!r} is malformed"
        raise HistoricalDividendProjectionContentError(
            msg,
        )
    return value


def _exact_mapping(raw: object, expected: set[str], label: str) -> dict[str, object]:
    if type(raw) is not dict or any(type(key) is not str for key in raw):
        msg = f"{label} must be an object"
        raise HistoricalDividendProjectionContentError(msg)
    mapping = cast("dict[str, object]", raw)
    if set(mapping) != expected:
        msg = f"{label} fields are not exact"
        raise HistoricalDividendProjectionContentError(msg)
    return mapping


__all__ = [
    "HISTORICAL_DIVIDEND_PROJECTION_DTYPES",
    "HISTORICAL_DIVIDEND_SOURCE_DTYPES",
    "HistoricalDividendProjectionContentError",
    "HistoricalDividendProjectionError",
    "HistoricalDividendProjectionReceipt",
    "HistoricalDividendProjectionSpec",
    "HistoricalDividendProjectionSpecError",
    "HistoricalDividendSourceIdentity",
    "MaterializedHistoricalDividendProjection",
    "historical_dividend_projection_schema_sha256",
    "historical_dividend_projection_spec_sha256",
    "historical_dividend_source_schema_sha256",
    "load_verified_historical_dividend_projection",
    "materialize_historical_dividend_projection",
]
