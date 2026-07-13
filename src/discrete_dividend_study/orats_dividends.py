"""Strict daily ORATS cash-dividend forecast ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl

from .schema import require_columns

if TYPE_CHECKING:
    from collections.abc import Mapping

ORATS_DIVIDEND_FREQUENCIES: Final = frozenset({0, 1, 2, 3, 4, 12})
ORATS_RAW_DIVIDEND_DTYPES = {
    "ticker": pl.String,
    "ex_date_raw": pl.String,
    "amount": pl.Float64,
    "frequency_per_year": pl.UInt8,
    "forecast_date": pl.Date,
}
ORATS_DIVIDEND_DTYPES = {
    "ticker": pl.String,
    "forecast_date": pl.Date,
    "ex_date": pl.Date,
    "amount": pl.Float64,
    "frequency_per_year": pl.UInt8,
}
_ORATS_DIVIDEND_FILENAME = re.compile(
    r"^(?P<forecast_date>\d{8})(?:\d{4})?ORATSStockDivForecasts?\.txt$",
)
_RAW_FIELD_COUNT = 4


class OratsDividendError(ValueError):
    """Base class for invalid ORATS dividend-forecast inputs."""


@dataclass(frozen=True, slots=True)
class OratsDividendSourceError(OratsDividendError):
    """A local dividend delivery is absent or has an unsupported identity."""

    source: Path
    reason: str

    def __str__(self) -> str:
        """Describe the source contract failure without inspecting rows."""
        return f"invalid ORATS dividend source {self.source}: {self.reason}"


@dataclass(frozen=True, slots=True)
class OratsDividendSchemaError(OratsDividendError):
    """Required raw or canonical dividend columns have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed dtypes without exposing forecast rows."""
        return f"ORATS dividend column dtype mismatches: {self.mismatches!r}"


@dataclass(frozen=True, slots=True)
class OratsDividendContentError(OratsDividendError):
    """Dividend rows violate the strict regular-forecast contract."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing licensed values."""
        return f"invalid ORATS dividend content: {self.violations!r}"


def _forecast_date_from_source(source: Path) -> date:
    match = _ORATS_DIVIDEND_FILENAME.fullmatch(source.name)
    if match is None:
        raise OratsDividendSourceError(
            source=source,
            reason="filename does not identify one regular ORATS forecast snapshot",
        )
    value = match.group("forecast_date")
    try:
        return date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    except ValueError as error:
        raise OratsDividendSourceError(
            source=source,
            reason="filename begins with an invalid forecast date",
        ) from error


def scan_raw_orats_dividend_forecast(source: str | Path) -> pl.LazyFrame:
    """Lazily scan one uncompressed regular ORATS forecast snapshot.

    Historical files use either an eight-digit date or a date plus four-digit intraday time before
    `ORATSStockDivForecasts.txt`. The first eight digits own `forecast_date`. The feed is exactly
    four space-delimited financial fields and may contain one terminal delimiter. A byte-level
    structural preflight proves that every row has exactly those fields and at most one terminal
    space before Polars' legacy-compatible terminal-delimiter handling is enabled. Missing columns,
    ragged nonempty fields, malformed UTF-8, parse errors, and empty files are never ignored.

    Raises:
        OratsDividendSourceError: If the source path or filename is unsupported.

    """
    source = Path(source).expanduser()
    if not source.is_file():
        raise OratsDividendSourceError(source=source, reason="file does not exist")
    forecast_date = _forecast_date_from_source(source)
    _validate_space_delimited_rows(source)
    raw_file_dtypes = {
        name: dtype for name, dtype in ORATS_RAW_DIVIDEND_DTYPES.items() if name != "forecast_date"
    }
    return pl.scan_csv(
        source,
        has_header=False,
        separator=" ",
        quote_char=None,
        schema=raw_file_dtypes,
        ignore_errors=False,
        infer_schema=False,
        encoding="utf8",
        low_memory=False,
        rechunk=False,
        try_parse_dates=False,
        glob=False,
        raise_if_empty=True,
        truncate_ragged_lines=True,
        missing_columns="raise",
    ).with_columns(pl.lit(forecast_date, dtype=pl.Date).alias("forecast_date"))


def _validate_space_delimited_rows(source: Path) -> None:
    try:
        with source.open("rb") as handle:
            row_count = 0
            for line_number, raw_line in enumerate(handle, start=1):
                row_count = line_number
                line = raw_line.removesuffix(b"\n").removesuffix(b"\r")
                if line.endswith(b" "):
                    line = line[:-1]
                fields = line.split(b" ")
                if len(fields) != _RAW_FIELD_COUNT or any(not field for field in fields):
                    raise OratsDividendSourceError(
                        source=source,
                        reason=f"line {line_number} is not exactly four space-delimited fields",
                    )
            if row_count == 0:
                raise OratsDividendSourceError(source=source, reason="file contains no rows")
    except OSError as error:
        raise OratsDividendSourceError(source=source, reason="file could not be read") from error


def _validate_dtypes(
    frame: pl.LazyFrame,
    expected: Mapping[str, type[pl.DataType]],
) -> pl.LazyFrame:
    validated = require_columns(frame, tuple(expected))
    schema = validated.collect_schema()
    mismatches = tuple(
        (name, str(dtype), str(schema[name]))
        for name, dtype in expected.items()
        if schema[name] != dtype
    )
    if mismatches:
        raise OratsDividendSchemaError(mismatches=mismatches)
    return validated


def _raw_content_violations(frame: pl.LazyFrame) -> tuple[tuple[str, int], ...]:
    diagnostics = (
        frame.select(
            pl.len().alias("rows"),
            pl.col("ticker").is_null().sum().alias("null_tickers"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_tickers"),
            pl.col("ex_date_raw").is_null().sum().alias("null_ex_dates"),
            pl.col("ex_date_raw")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_ex_dates"),
            pl.col("amount").is_null().sum().alias("null_amounts"),
            (~pl.col("amount").is_finite()).fill_null(value=False).sum().alias("nonfinite_amounts"),
            (pl.col("amount") < 0.0).fill_null(value=False).sum().alias("negative_amounts"),
            pl.col("frequency_per_year").is_null().sum().alias("null_frequencies"),
            (~pl.col("frequency_per_year").is_in(ORATS_DIVIDEND_FREQUENCIES))
            .fill_null(value=False)
            .sum()
            .alias("unsupported_frequencies"),
            pl.struct(
                "ticker",
                "forecast_date",
                "ex_date_raw",
                "amount",
                "frequency_per_year",
            )
            .is_duplicated()
            .sum()
            .alias("duplicate_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    if diagnostics["rows"] == 0:
        return (("empty_input", 1),)
    return tuple(
        (name, value)
        for name, value in diagnostics.items()
        if name != "rows" and type(value) is int and value != 0
    )


def canonicalize_raw_orats_dividend_forecast(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Project raw fields into typed forecast-date, ex-date, amount, and frequency rows.

    This stage preserves vendor row order, zero-amount placeholders, past/same-day records, and
    distinct same-date cash components. Amounts are parsed directly as `Float64`, matching the
    submitted loader and the engine's durable cash-event precision. Ex-dates use strict ISO
    `YYYY-MM-DD` parsing. The future-event selection is intentionally a separate operation.

    Raises:
        MissingColumnsError: If a required raw field is absent.
        OratsDividendSchemaError: If a required raw field has the wrong dtype.
        OratsDividendContentError: If a row is empty, malformed, duplicated, or outside the feed
            frequency/amount domain.

    """
    validated = _validate_dtypes(frame, ORATS_RAW_DIVIDEND_DTYPES)
    violations = _raw_content_violations(validated)
    if violations:
        raise OratsDividendContentError(violations=violations)
    canonical = validated.select(
        "ticker",
        "forecast_date",
        pl.col("ex_date_raw")
        .str.strptime(pl.Date, format="%Y-%m-%d", strict=True, exact=True, cache=True)
        .alias("ex_date"),
        "amount",
        "frequency_per_year",
    )
    try:
        canonical.select(pl.col("ex_date").null_count()).collect()
    except pl.exceptions.PolarsError as error:
        raise OratsDividendContentError(violations=(("invalid_ex_dates", 1),)) from error
    return canonical


def _validate_canonical_content(frame: pl.LazyFrame) -> None:
    diagnostics = (
        frame.select(
            pl.len().alias("rows"),
            pl.any_horizontal(pl.all().is_null()).sum().alias("rows_with_nulls"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_tickers"),
            (~pl.col("amount").is_finite()).fill_null(value=False).sum().alias("nonfinite_amounts"),
            (pl.col("amount") < 0.0).fill_null(value=False).sum().alias("negative_amounts"),
            (~pl.col("frequency_per_year").is_in(ORATS_DIVIDEND_FREQUENCIES))
            .fill_null(value=False)
            .sum()
            .alias("unsupported_frequencies"),
            pl.struct(tuple(ORATS_DIVIDEND_DTYPES)).is_duplicated().sum().alias("duplicate_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    violations = (
        (("empty_input", 1),)
        if diagnostics["rows"] == 0
        else tuple(
            (name, value)
            for name, value in diagnostics.items()
            if name != "rows" and type(value) is int and value != 0
        )
    )
    if violations:
        raise OratsDividendContentError(violations=violations)


def select_future_orats_dividend_forecasts(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Select and deterministically order records strictly after their forecast date.

    The strict `ex_date > forecast_date` boundary reproduces the submitted forecast-artifact
    producer and prevents a same-day or historical cash row from becoming a future model event.
    Zero-amount provider placeholders and distinct components on the same ex-date remain
    represented; their downstream cash-event treatment requires an explicit separate policy.

    Raises:
        MissingColumnsError: If a required canonical field is absent.
        OratsDividendSchemaError: If a canonical field has the wrong dtype.
        OratsDividendContentError: If canonical rows are empty, malformed, or duplicated.

    """
    validated = _validate_dtypes(frame, ORATS_DIVIDEND_DTYPES).select(
        tuple(ORATS_DIVIDEND_DTYPES),
    )
    _validate_canonical_content(validated)
    return validated.filter(pl.col("ex_date") > pl.col("forecast_date")).sort(
        "forecast_date",
        "ticker",
        "ex_date",
        "frequency_per_year",
        "amount",
    )


__all__ = [
    "ORATS_DIVIDEND_DTYPES",
    "ORATS_DIVIDEND_FREQUENCIES",
    "ORATS_RAW_DIVIDEND_DTYPES",
    "OratsDividendContentError",
    "OratsDividendError",
    "OratsDividendSchemaError",
    "OratsDividendSourceError",
    "canonicalize_raw_orats_dividend_forecast",
    "scan_raw_orats_dividend_forecast",
    "select_future_orats_dividend_forecasts",
]
