"""Verified projection of Federal Reserve nominal Svensson curve data."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal, Self, cast

import polars as pl

from .nss_rates import NSS_PARAMETER_DTYPES
from .pinned_parquet import VerifiedPinnedParquet, verify_pinned_parquet

FED_NOMINAL_CURVE_URL: Final = (
    "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
)
FED_NOMINAL_CURVE_SOURCE_SKIP_LINES: Final = 9
FED_NOMINAL_CURVE_VERSION: Final = 1
FED_NOMINAL_CURVE_RECEIPT_VERSION: Final = 1
FED_NOMINAL_CURVE_OUTPUT_NAME: Final = "nss_parameters.parquet"
FED_NOMINAL_CURVE_COMPRESSION: Final = "zstd"
FED_NOMINAL_CURVE_COMPRESSION_LEVEL: Final = 3
FED_NOMINAL_CURVE_ROW_GROUP_SIZE: Final = 65_536
FED_NOMINAL_CURVE_HASH_BLOCK_BYTES: Final = 8 * 1024 * 1024
FED_NOMINAL_CURVE_RATE_SCALE: Final = 0.01
FED_NOMINAL_CURVE_SHORTEST_MATURITY_YEARS: Final = 1.0e-8
FED_NOMINAL_CURVE_PUBLISHED_YIELD_TOLERANCE_PERCENTAGE_POINTS: Final = 5.01e-5
FED_NOMINAL_CURVE_SHORT_END_RATE_DEFINITION: Final = "nss-zero-maturity-limit"
FED_NOMINAL_CURVE_PROJECT_START_DATE: Final = date(2011, 1, 1)
FED_NOMINAL_CURVE_PROJECT_END_DATE_EXCLUSIVE: Final = date(2024, 9, 28)
_SOURCE_COLUMNS: Final = (
    "Date",
    "BETA0",
    "BETA1",
    "BETA2",
    "BETA3",
    "SVENY01",
    "TAU1",
    "TAU2",
)
_SOURCE_VALUE_COLUMNS: Final = _SOURCE_COLUMNS[1:]
_SHA256_LENGTH: Final = 64
_SMALL_LOADING_ARGUMENT: Final = 1.0e-4


class FederalReserveCurveError(RuntimeError):
    """Base class for an invalid source, request, or projected curve artifact."""


@dataclass(slots=True)
class FederalReserveCurveContentError(FederalReserveCurveError):
    """Source or output content violates the declared curve contract."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without emitting market rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"Federal Reserve curve projection failed{location}: {self.reason}"


@dataclass(slots=True)
class FederalReserveCurveSpecError(FederalReserveCurveError):
    """A requested date, unit, or physical writer policy is unsupported."""

    reason: str

    def __str__(self) -> str:
        """Return the exact rejected contract reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class FederalReserveCurveProjectionSpec:
    """Exact source columns, units, horizon, and Svensson short-end policy."""

    start_date: date
    end_date_exclusive: date
    shortest_maturity_years: float = FED_NOMINAL_CURVE_SHORTEST_MATURITY_YEARS
    rate_scale: float = FED_NOMINAL_CURVE_RATE_SCALE
    shortest_rate_definition: Literal["nss-zero-maturity-limit"] = "nss-zero-maturity-limit"
    source_zero_yield_column: Literal["SVENY01"] = "SVENY01"
    source_zero_yield_compounding: Literal["continuous"] = "continuous"
    source_zero_yield_tolerance_percentage_points: float = (
        FED_NOMINAL_CURVE_PUBLISHED_YIELD_TOLERANCE_PERCENTAGE_POINTS
    )
    schema_version: int = FED_NOMINAL_CURVE_VERSION

    def __post_init__(self) -> None:
        """Reject ambiguous bounds, unit conversions, and short-end substitutions."""
        if self.schema_version != FED_NOMINAL_CURVE_VERSION:
            reason = "unsupported curve projection version"
            raise FederalReserveCurveSpecError(reason)
        if (
            type(self.start_date) is not date
            or type(self.end_date_exclusive) is not date
            or self.start_date >= self.end_date_exclusive
        ):
            reason = "curve projection requires plain increasing dates"
            raise FederalReserveCurveSpecError(reason)
        if (
            type(self.shortest_maturity_years) is not float
            or self.shortest_maturity_years != FED_NOMINAL_CURVE_SHORTEST_MATURITY_YEARS
        ):
            reason = "short-end maturity must equal the stable NSS numerical floor"
            raise FederalReserveCurveSpecError(reason)
        if type(self.rate_scale) is not float or self.rate_scale != FED_NOMINAL_CURVE_RATE_SCALE:
            reason = "Federal Reserve percentage rates must scale by 0.01"
            raise FederalReserveCurveSpecError(reason)
        if self.source_zero_yield_column != "SVENY01":
            reason = "published zero-yield validation source must be SVENY01"
            raise FederalReserveCurveSpecError(reason)
        if self.source_zero_yield_compounding != "continuous":
            reason = "SVENY01 must retain continuous compounding"
            raise FederalReserveCurveSpecError(reason)
        if self.shortest_rate_definition != FED_NOMINAL_CURVE_SHORT_END_RATE_DEFINITION:
            reason = "shortest rate must be the exact NSS zero-maturity limit"
            raise FederalReserveCurveSpecError(reason)
        if (
            type(self.source_zero_yield_tolerance_percentage_points) is not float
            or self.source_zero_yield_tolerance_percentage_points
            != FED_NOMINAL_CURVE_PUBLISHED_YIELD_TOLERANCE_PERCENTAGE_POINTS
        ):
            reason = "published-yield validation tolerance must match source rounding"
            raise FederalReserveCurveSpecError(reason)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical financial projection mapping."""
        return {
            "end_date_exclusive": self.end_date_exclusive.isoformat(),
            "rate_scale": self.rate_scale,
            "schema_version": self.schema_version,
            "shortest_maturity_years": self.shortest_maturity_years,
            "shortest_rate_definition": self.shortest_rate_definition,
            "source_zero_yield_column": self.source_zero_yield_column,
            "source_zero_yield_compounding": self.source_zero_yield_compounding,
            "source_zero_yield_tolerance_percentage_points": (
                self.source_zero_yield_tolerance_percentage_points
            ),
            "start_date": self.start_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse an exact projection mapping without coercion."""
        mapping = _exact_mapping(
            raw,
            {
                "end_date_exclusive",
                "rate_scale",
                "schema_version",
                "shortest_maturity_years",
                "shortest_rate_definition",
                "source_zero_yield_column",
                "source_zero_yield_compounding",
                "source_zero_yield_tolerance_percentage_points",
                "start_date",
            },
            "curve projection spec",
        )
        if type(mapping["start_date"]) is not str or type(mapping["end_date_exclusive"]) is not str:
            reason = "curve projection dates are malformed"
            raise FederalReserveCurveContentError(reason)
        try:
            start_date = date.fromisoformat(mapping["start_date"])
            end_date = date.fromisoformat(mapping["end_date_exclusive"])
        except ValueError as error:
            reason = "curve projection dates are malformed"
            raise FederalReserveCurveContentError(reason) from error
        return cls(
            start_date=start_date,
            end_date_exclusive=end_date,
            shortest_maturity_years=cast("float", mapping["shortest_maturity_years"]),
            rate_scale=cast("float", mapping["rate_scale"]),
            shortest_rate_definition=cast(
                "Literal['nss-zero-maturity-limit']",
                mapping["shortest_rate_definition"],
            ),
            source_zero_yield_column=cast(
                "Literal['SVENY01']", mapping["source_zero_yield_column"]
            ),
            source_zero_yield_compounding=cast(
                "Literal['continuous']",
                mapping["source_zero_yield_compounding"],
            ),
            source_zero_yield_tolerance_percentage_points=cast(
                "float",
                mapping["source_zero_yield_tolerance_percentage_points"],
            ),
            schema_version=cast("int", mapping["schema_version"]),
        )


PROJECT_FED_NOMINAL_CURVE_SPEC: Final = FederalReserveCurveProjectionSpec(
    start_date=FED_NOMINAL_CURVE_PROJECT_START_DATE,
    end_date_exclusive=FED_NOMINAL_CURVE_PROJECT_END_DATE_EXCLUSIVE,
)


@dataclass(frozen=True, slots=True)
class FederalReserveCurveWriteSpec:
    """Physical identity of the projected Parquet sink."""

    polars_version: str = pl.__version__
    compression: Literal["zstd"] = FED_NOMINAL_CURVE_COMPRESSION
    compression_level: int = FED_NOMINAL_CURVE_COMPRESSION_LEVEL
    row_group_size: int = FED_NOMINAL_CURVE_ROW_GROUP_SIZE
    statistics: bool = True
    maintain_order: bool = True
    sync_on_close: Literal["all"] = "all"
    engine: Literal["streaming"] = "streaming"

    def __post_init__(self) -> None:
        """Reject a runtime-independent or nondurable byte identity."""
        if type(self.polars_version) is not str or not self.polars_version:
            reason = "writer Polars version must be nonempty"
            raise FederalReserveCurveSpecError(reason)
        expected = (
            self.compression == FED_NOMINAL_CURVE_COMPRESSION,
            type(self.compression_level) is int
            and self.compression_level == FED_NOMINAL_CURVE_COMPRESSION_LEVEL,
            type(self.row_group_size) is int
            and self.row_group_size == FED_NOMINAL_CURVE_ROW_GROUP_SIZE,
            type(self.statistics) is bool and self.statistics,
            type(self.maintain_order) is bool and self.maintain_order,
            self.sync_on_close == "all",
            self.engine == "streaming",
        )
        if not all(expected):
            reason = "curve writer must use the frozen durable streaming Parquet contract"
            raise FederalReserveCurveSpecError(reason)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical physical writer mapping."""
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
            "curve writer spec",
        )
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


FED_NOMINAL_CURVE_WRITE_SPEC: Final = FederalReserveCurveWriteSpec()


@dataclass(frozen=True, slots=True)
class FederalReserveCurveReceipt:
    """Complete source, policy, diagnostic, and output identity."""

    source_name: str
    source_size_bytes: int
    source_sha256: str
    source_url: str
    projection_spec: FederalReserveCurveProjectionSpec
    writer: FederalReserveCurveWriteSpec
    request_sha256: str
    selected_source_rows: int
    excluded_unavailable_rows: int
    output_name: str
    output_rows: int
    output_bytes: int
    output_sha256: str
    output_date_min: date
    output_date_max: date
    schema_version: int = FED_NOMINAL_CURVE_RECEIPT_VERSION

    def __post_init__(self) -> None:
        """Validate all serialized identities before any file is trusted."""
        if self.schema_version != FED_NOMINAL_CURVE_RECEIPT_VERSION:
            reason = "unsupported curve receipt version"
            raise FederalReserveCurveContentError(reason)
        if (
            type(self.source_name) is not str
            or Path(self.source_name).name != self.source_name
            or not self.source_name
            or type(self.source_url) is not str
            or self.source_url != FED_NOMINAL_CURVE_URL
        ):
            reason = "curve source identity is malformed"
            raise FederalReserveCurveContentError(reason)
        if (
            type(self.source_size_bytes) is not int
            or self.source_size_bytes <= 0
            or not _is_sha256(self.source_sha256)
            or type(self.projection_spec) is not FederalReserveCurveProjectionSpec
            or type(self.writer) is not FederalReserveCurveWriteSpec
        ):
            reason = "curve source or request identity is malformed"
            raise FederalReserveCurveContentError(reason)
        if self.request_sha256 != federal_reserve_curve_request_sha256(
            self.source_name,
            self.source_size_bytes,
            self.source_sha256,
            self.projection_spec,
            self.writer,
        ):
            reason = "curve request hash is inconsistent"
            raise FederalReserveCurveContentError(reason)
        counts = (
            self.selected_source_rows,
            self.excluded_unavailable_rows,
            self.output_rows,
            self.output_bytes,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            reason = "curve receipt counts are malformed"
            raise FederalReserveCurveContentError(reason)
        if (
            self.output_rows <= 0
            or self.output_bytes <= 0
            or self.selected_source_rows != self.output_rows + self.excluded_unavailable_rows
            or self.output_name != FED_NOMINAL_CURVE_OUTPUT_NAME
            or not _is_sha256(self.output_sha256)
        ):
            reason = "curve output identity is malformed"
            raise FederalReserveCurveContentError(reason)
        if (
            type(self.output_date_min) is not date
            or type(self.output_date_max) is not date
            or self.output_date_min > self.output_date_max
            or self.output_date_min < self.projection_spec.start_date
            or self.output_date_max >= self.projection_spec.end_date_exclusive
        ):
            reason = "curve output date summary is malformed"
            raise FederalReserveCurveContentError(reason)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical receipt mapping."""
        return {
            "excluded_unavailable_rows": self.excluded_unavailable_rows,
            "output_bytes": self.output_bytes,
            "output_date_max": self.output_date_max.isoformat(),
            "output_date_min": self.output_date_min.isoformat(),
            "output_name": self.output_name,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "projection_spec": self.projection_spec.as_dict(),
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "selected_source_rows": self.selected_source_rows,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "source_url": self.source_url,
            "writer": self.writer.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and revalidate one exact curve receipt."""
        expected = {
            "excluded_unavailable_rows",
            "output_bytes",
            "output_date_max",
            "output_date_min",
            "output_name",
            "output_rows",
            "output_sha256",
            "projection_spec",
            "request_sha256",
            "schema_version",
            "selected_source_rows",
            "source_name",
            "source_sha256",
            "source_size_bytes",
            "source_url",
            "writer",
        }
        mapping = _exact_mapping(raw, expected, "curve receipt")
        string_fields = (
            "output_date_max",
            "output_date_min",
            "output_name",
            "output_sha256",
            "request_sha256",
            "source_name",
            "source_sha256",
            "source_url",
        )
        integer_fields = (
            "excluded_unavailable_rows",
            "output_bytes",
            "output_rows",
            "schema_version",
            "selected_source_rows",
            "source_size_bytes",
        )
        if any(type(mapping[name]) is not str for name in string_fields) or any(
            type(mapping[name]) is not int for name in integer_fields
        ):
            reason = "curve receipt fields are malformed"
            raise FederalReserveCurveContentError(reason)
        try:
            date_min = date.fromisoformat(cast("str", mapping["output_date_min"]))
            date_max = date.fromisoformat(cast("str", mapping["output_date_max"]))
        except ValueError as error:
            reason = "curve receipt dates are malformed"
            raise FederalReserveCurveContentError(reason) from error
        return cls(
            source_name=cast("str", mapping["source_name"]),
            source_size_bytes=cast("int", mapping["source_size_bytes"]),
            source_sha256=cast("str", mapping["source_sha256"]),
            source_url=cast("str", mapping["source_url"]),
            projection_spec=FederalReserveCurveProjectionSpec.from_dict(
                mapping["projection_spec"],
            ),
            writer=FederalReserveCurveWriteSpec.from_dict(mapping["writer"]),
            request_sha256=cast("str", mapping["request_sha256"]),
            selected_source_rows=cast("int", mapping["selected_source_rows"]),
            excluded_unavailable_rows=cast("int", mapping["excluded_unavailable_rows"]),
            output_name=cast("str", mapping["output_name"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            output_date_min=date_min,
            output_date_max=date_max,
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class MaterializedFederalReserveCurve:
    """One verified projected parameter Parquet and its provenance receipt."""

    path: Path
    receipt_path: Path
    receipt: FederalReserveCurveReceipt
    pinned: VerifiedPinnedParquet

    def scan(self) -> pl.LazyFrame:
        """Lazily scan the exact verified projection."""
        return self.pinned.scan()


@dataclass(frozen=True, slots=True)
class _ProjectionDiagnostics:
    selected_rows: int
    unavailable_rows: int
    partial_rows: int
    duplicate_dates: int
    invalid_complete_rows: int
    inconsistent_published_yields: int


def federal_reserve_curve_request_sha256(
    source_name: str,
    source_size_bytes: int,
    source_sha256: str,
    spec: FederalReserveCurveProjectionSpec,
    writer: FederalReserveCurveWriteSpec,
) -> str:
    """Hash the source bytes, financial policy, and byte-producing writer."""
    payload = {
        "source": {
            "name": source_name,
            "sha256": source_sha256,
            "size_bytes": source_size_bytes,
            "url": FED_NOMINAL_CURVE_URL,
        },
        "spec": spec.as_dict(),
        "writer": writer.as_dict(),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def materialize_federal_reserve_curve(
    source_path: str | Path,
    *,
    expected_source_sha256: str,
    expected_source_size_bytes: int,
    spec: FederalReserveCurveProjectionSpec,
    output_root: str | Path,
) -> MaterializedFederalReserveCurve:
    """Verify one official CSV vintage and project a closed decimal-rate Parquet."""
    writer = FED_NOMINAL_CURVE_WRITE_SPEC
    _require_writer_runtime(writer)
    source, initial, source_sha256 = _verified_source_identity(
        source_path,
        expected_source_sha256,
        expected_source_size_bytes,
    )
    request_sha256 = federal_reserve_curve_request_sha256(
        source.name,
        initial.st_size,
        source_sha256,
        spec,
        writer,
    )
    root = _output_root(output_root)
    directory = root / request_sha256
    output = directory / FED_NOMINAL_CURVE_OUTPUT_NAME
    receipt_path = output.with_name(f"{output.name}.receipt.json")
    if output.exists() or receipt_path.exists():
        existing = load_verified_federal_reserve_curve(output)
        if existing.receipt.request_sha256 != request_sha256:
            reason = "existing curve projection disagrees with the request"
            raise FederalReserveCurveContentError(reason, str(output))
        return existing
    directory.mkdir(parents=True, exist_ok=False)
    partial = output.with_name(f".{output.name}.part")
    partial_receipt = receipt_path.with_name(f".{receipt_path.name}.part")
    try:
        selected = _selected_source_frame(source, spec)
        diagnostics = _source_diagnostics(selected, spec)
        _validate_diagnostics(diagnostics, source)
        _project_complete_rows(selected, spec).sink_parquet(
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
        _require_source_unchanged(source, initial, expected_source_sha256)
        pinned = _verify_projected_output(partial)
        date_min, date_max = _output_date_bounds(pinned)
        receipt = FederalReserveCurveReceipt(
            source_name=source.name,
            source_size_bytes=initial.st_size,
            source_sha256=source_sha256,
            source_url=FED_NOMINAL_CURVE_URL,
            projection_spec=spec,
            writer=writer,
            request_sha256=request_sha256,
            selected_source_rows=diagnostics.selected_rows,
            excluded_unavailable_rows=diagnostics.unavailable_rows,
            output_name=FED_NOMINAL_CURVE_OUTPUT_NAME,
            output_rows=pinned.identity.rows,
            output_bytes=pinned.identity.size_bytes,
            output_sha256=pinned.identity.sha256,
            output_date_min=date_min,
            output_date_max=date_max,
        )
        _write_new_json(partial_receipt, receipt.as_dict())
        partial.replace(output)
        try:
            partial_receipt.replace(receipt_path)
        except BaseException:
            output.replace(partial)
            raise
    except pl.exceptions.PolarsError as error:
        reason = "curve CSV parsing or streaming projection failed"
        raise FederalReserveCurveContentError(reason, str(source)) from error
    finally:
        partial.unlink(missing_ok=True)
        partial_receipt.unlink(missing_ok=True)
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return load_verified_federal_reserve_curve(output)


def load_verified_federal_reserve_curve(
    path: str | Path,
) -> MaterializedFederalReserveCurve:
    """Load a detached projection and verify receipt, bytes, schema, rows, and dates."""
    output = _regular_file(path, "curve projection")
    receipt_path = _regular_file(
        output.with_name(f"{output.name}.receipt.json"),
        "curve projection receipt",
    )
    try:
        receipt = FederalReserveCurveReceipt.from_dict(json.loads(receipt_path.read_bytes()))
    except (OSError, json.JSONDecodeError) as error:
        reason = "curve projection receipt is unreadable"
        raise FederalReserveCurveContentError(reason, str(receipt_path)) from error
    if output.parent.name != receipt.request_sha256:
        reason = "curve output directory disagrees with its request identity"
        raise FederalReserveCurveContentError(reason, str(output.parent))
    if output.name != receipt.output_name or output.stat().st_size != receipt.output_bytes:
        reason = "curve output name or byte count disagrees with its receipt"
        raise FederalReserveCurveContentError(reason, str(output))
    pinned = verify_pinned_parquet(
        output,
        expected_sha256=receipt.output_sha256,
        expected_rows=receipt.output_rows,
        expected_schema=NSS_PARAMETER_DTYPES,
    )
    date_min, date_max = _output_date_bounds(pinned)
    if date_min != receipt.output_date_min or date_max != receipt.output_date_max:
        reason = "curve output date summary disagrees with its receipt"
        raise FederalReserveCurveContentError(reason, str(output))
    return MaterializedFederalReserveCurve(output, receipt_path, receipt, pinned)


def _selected_source_frame(
    source: Path,
    spec: FederalReserveCurveProjectionSpec,
) -> pl.LazyFrame:
    overrides = {"Date": pl.Date, **dict.fromkeys(_SOURCE_VALUE_COLUMNS, pl.Float64)}
    scan = pl.scan_csv(
        source,
        skip_lines=FED_NOMINAL_CURVE_SOURCE_SKIP_LINES,
        null_values="NA",
        schema_overrides=overrides,
        infer_schema_length=1_000,
        rechunk=False,
    )
    schema = scan.collect_schema()
    mismatches = tuple(
        (name, str(dtype), str(schema.get(name)))
        for name, dtype in overrides.items()
        if schema.get(name) != dtype
    )
    if mismatches:
        reason = f"Federal Reserve source columns have incompatible dtypes: {mismatches!r}"
        raise FederalReserveCurveContentError(reason, str(source))
    return scan.select(*_SOURCE_COLUMNS).filter(
        pl.col("Date").is_between(
            spec.start_date,
            spec.end_date_exclusive,
            closed="left",
        ),
    )


def _stable_source_nss_loadings(argument: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    square = argument * argument
    cube = square * argument
    fourth = square * square
    level_series = 1.0 - argument / 2.0 + square / 6.0 - cube / 24.0 + fourth / 120.0
    curvature_series = argument / 2.0 - square / 3.0 + cube / 8.0 - fourth / 30.0
    exponential = (-argument).exp()
    level_direct = (1.0 - exponential) / argument
    level = (
        pl.when(argument.abs() < _SMALL_LOADING_ARGUMENT).then(level_series).otherwise(level_direct)
    )
    curvature = (
        pl.when(argument.abs() < _SMALL_LOADING_ARGUMENT)
        .then(curvature_series)
        .otherwise(level_direct - exponential)
    )
    return level, curvature


def _source_nss_zero_yield_percentage_points(maturity_years: float) -> pl.Expr:
    maturity = pl.lit(maturity_years, dtype=pl.Float64)
    level1, curvature1 = _stable_source_nss_loadings(maturity / pl.col("TAU1"))
    _, curvature2 = _stable_source_nss_loadings(maturity / pl.col("TAU2"))
    return (
        pl.col("BETA0")
        + pl.col("BETA1") * level1
        + pl.col("BETA2") * curvature1
        + pl.col("BETA3") * curvature2
    )


def _source_diagnostics(
    selected: pl.LazyFrame,
    spec: FederalReserveCurveProjectionSpec,
) -> _ProjectionDiagnostics:
    any_null = pl.any_horizontal(*(pl.col(name).is_null() for name in _SOURCE_VALUE_COLUMNS))
    all_null = pl.all_horizontal(*(pl.col(name).is_null() for name in _SOURCE_VALUE_COLUMNS))
    invalid_complete = (~any_null) & pl.any_horizontal(
        *(~pl.col(name).is_finite() for name in _SOURCE_VALUE_COLUMNS),
        pl.col("TAU1") <= 0.0,
        pl.col("TAU2") <= 0.0,
    )
    published_yield_difference = (
        _source_nss_zero_yield_percentage_points(1.0) - pl.col(spec.source_zero_yield_column)
    ).abs()
    inconsistent_published_yield = (
        (~any_null)
        & ~invalid_complete
        & (published_yield_difference > spec.source_zero_yield_tolerance_percentage_points)
    )
    row = (
        selected.select(
            pl.len().alias("selected_rows"),
            all_null.sum().alias("unavailable_rows"),
            (any_null & ~all_null).sum().alias("partial_rows"),
            pl.col("Date").is_duplicated().sum().alias("duplicate_dates"),
            invalid_complete.sum().alias("invalid_complete_rows"),
            inconsistent_published_yield.sum().alias("inconsistent_published_yields"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    values = (
        row["selected_rows"],
        row["unavailable_rows"],
        row["partial_rows"],
        row["duplicate_dates"],
        row["invalid_complete_rows"],
        row["inconsistent_published_yields"],
    )
    if any(type(value) is not int for value in values):
        reason = "curve source diagnostics are malformed"
        raise FederalReserveCurveContentError(reason)
    return _ProjectionDiagnostics(*cast("tuple[int, int, int, int, int, int]", values))


def _validate_diagnostics(diagnostics: _ProjectionDiagnostics, source: Path) -> None:
    if diagnostics.selected_rows <= 0:
        reason = "curve source date selection is empty"
        raise FederalReserveCurveContentError(reason, str(source))
    if diagnostics.selected_rows == diagnostics.unavailable_rows:
        reason = "curve source has no available selected rows"
        raise FederalReserveCurveContentError(reason, str(source))
    violations = {
        "partial_rows": diagnostics.partial_rows,
        "duplicate_dates": diagnostics.duplicate_dates,
        "invalid_complete_rows": diagnostics.invalid_complete_rows,
        "inconsistent_published_yields": diagnostics.inconsistent_published_yields,
    }
    observed = {name: value for name, value in violations.items() if value != 0}
    if observed:
        reason = f"curve source diagnostics contain violations: {observed!r}"
        raise FederalReserveCurveContentError(reason, str(source))


def _project_complete_rows(
    selected: pl.LazyFrame,
    spec: FederalReserveCurveProjectionSpec,
) -> pl.LazyFrame:
    available = selected.filter(
        pl.all_horizontal(*(pl.col(name).is_not_null() for name in _SOURCE_VALUE_COLUMNS)),
    )
    scale = spec.rate_scale
    return available.select(
        pl.col("Date").alias("date"),
        (pl.col("BETA0") * scale).alias("beta0"),
        (pl.col("BETA1") * scale).alias("beta1"),
        (pl.col("BETA2") * scale).alias("beta2"),
        (pl.col("BETA3") * scale).alias("beta3"),
        pl.col("TAU1").alias("tau1"),
        pl.col("TAU2").alias("tau2"),
        pl.lit(spec.shortest_maturity_years, dtype=pl.Float64).alias("shortest_maturity"),
        ((pl.col("BETA0") + pl.col("BETA1")) * scale).alias("shortest_rate"),
    ).sort("date")


def _verify_projected_output(path: Path) -> VerifiedPinnedParquet:
    rows = pl.scan_parquet(path, rechunk=False).select(pl.len()).collect().item()
    if type(rows) is not int or rows <= 0:
        reason = "curve projection output is empty"
        raise FederalReserveCurveContentError(reason, str(path))
    return verify_pinned_parquet(
        path,
        expected_sha256=_file_sha256(path),
        expected_rows=rows,
        expected_schema=NSS_PARAMETER_DTYPES,
    )


def _output_date_bounds(source: VerifiedPinnedParquet) -> tuple[date, date]:
    row = (
        source.scan()
        .select(
            pl.col("date").min().alias("date_min"),
            pl.col("date").max().alias("date_max"),
            pl.col("date").is_duplicated().sum().alias("duplicate_dates"),
            pl.any_horizontal(pl.all().is_null()).sum().alias("rows_with_nulls"),
            pl.any_horizontal(
                *(~pl.col(name).is_finite() for name in NSS_PARAMETER_DTYPES if name != "date"),
            )
            .sum()
            .alias("rows_with_nonfinite_values"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if (
        type(row["date_min"]) is not date
        or type(row["date_max"]) is not date
        or row["duplicate_dates"] != 0
        or row["rows_with_nulls"] != 0
        or row["rows_with_nonfinite_values"] != 0
    ):
        reason = "curve projection content summary is invalid"
        raise FederalReserveCurveContentError(reason, str(source.path))
    return row["date_min"], row["date_max"]


def _validate_expected_source(expected_sha256: str, expected_size_bytes: int) -> None:
    if type(expected_sha256) is not str or not _is_sha256(expected_sha256):
        reason = "expected curve source SHA-256 is malformed"
        raise FederalReserveCurveSpecError(reason)
    if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
        reason = "expected curve source bytes must be positive"
        raise FederalReserveCurveSpecError(reason)


def _verified_source_identity(
    source_path: str | Path,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[Path, os.stat_result, str]:
    _validate_expected_source(expected_sha256, expected_size_bytes)
    source = _regular_file(source_path, "curve source")
    initial = source.stat()
    if initial.st_size != expected_size_bytes:
        reason = "source byte count disagrees with the expected identity"
        raise FederalReserveCurveContentError(reason, str(source))
    source_sha256 = _file_sha256(source)
    if source_sha256 != expected_sha256:
        reason = "source bytes disagree with the expected SHA-256"
        raise FederalReserveCurveContentError(reason, str(source))
    return source, initial, source_sha256


def _require_writer_runtime(writer: FederalReserveCurveWriteSpec) -> None:
    if writer.polars_version != pl.__version__:
        reason = "curve writer Polars version disagrees with the active runtime"
        raise FederalReserveCurveSpecError(reason)


def _require_source_unchanged(source: Path, initial: os.stat_result, sha256: str) -> None:
    final = source.stat()
    if (
        final.st_dev != initial.st_dev
        or final.st_ino != initial.st_ino
        or final.st_size != initial.st_size
        or final.st_mtime_ns != initial.st_mtime_ns
        or _file_sha256(source) != sha256
    ):
        reason = "curve source changed during projection"
        raise FederalReserveCurveContentError(reason, str(source))


def _regular_file(path: str | Path, label: str) -> Path:
    source = Path(path).expanduser()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        reason = f"{label} must be an existing nonempty non-symlink file"
        raise FederalReserveCurveContentError(reason, str(source))
    return source


def _output_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        reason = "curve output root must be absent or a non-symlink directory"
        raise FederalReserveCurveContentError(reason, str(root))
    if not root.parent.is_dir() or root.parent.is_symlink():
        reason = "curve output parent must be an existing non-symlink directory"
        raise FederalReserveCurveContentError(reason, str(root.parent))
    root.mkdir(exist_ok=True)
    return root


def _write_new_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        reason = "curve receipt already exists"
        raise FederalReserveCurveContentError(reason, str(path))
    with path.open("xb") as destination:
        destination.write(_canonical_json_bytes(value) + b"\n")
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(FED_NOMINAL_CURVE_HASH_BLOCK_BYTES):
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
        reason = f"{label} must be a JSON object"
        raise FederalReserveCurveContentError(reason)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected or any(type(key) is not str for key in mapping):
        reason = f"{label} keys must be exactly {sorted(expected)!r}"
        raise FederalReserveCurveContentError(reason)
    return cast("dict[str, object]", mapping)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "FED_NOMINAL_CURVE_COMPRESSION",
    "FED_NOMINAL_CURVE_COMPRESSION_LEVEL",
    "FED_NOMINAL_CURVE_HASH_BLOCK_BYTES",
    "FED_NOMINAL_CURVE_OUTPUT_NAME",
    "FED_NOMINAL_CURVE_PROJECT_END_DATE_EXCLUSIVE",
    "FED_NOMINAL_CURVE_PROJECT_START_DATE",
    "FED_NOMINAL_CURVE_PUBLISHED_YIELD_TOLERANCE_PERCENTAGE_POINTS",
    "FED_NOMINAL_CURVE_RATE_SCALE",
    "FED_NOMINAL_CURVE_RECEIPT_VERSION",
    "FED_NOMINAL_CURVE_ROW_GROUP_SIZE",
    "FED_NOMINAL_CURVE_SHORTEST_MATURITY_YEARS",
    "FED_NOMINAL_CURVE_SHORT_END_RATE_DEFINITION",
    "FED_NOMINAL_CURVE_SOURCE_SKIP_LINES",
    "FED_NOMINAL_CURVE_URL",
    "FED_NOMINAL_CURVE_VERSION",
    "FED_NOMINAL_CURVE_WRITE_SPEC",
    "PROJECT_FED_NOMINAL_CURVE_SPEC",
    "FederalReserveCurveContentError",
    "FederalReserveCurveError",
    "FederalReserveCurveProjectionSpec",
    "FederalReserveCurveReceipt",
    "FederalReserveCurveSpecError",
    "FederalReserveCurveWriteSpec",
    "MaterializedFederalReserveCurve",
    "federal_reserve_curve_request_sha256",
    "load_verified_federal_reserve_curve",
    "materialize_federal_reserve_curve",
]
