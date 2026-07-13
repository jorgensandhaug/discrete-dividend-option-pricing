"""Strict ORATS cash-dividend forecast ingestion tests."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from discrete_dividend_study import (
    ORATS_DIVIDEND_DTYPES,
    OratsDividendContentError,
    OratsDividendSchemaError,
    OratsDividendSourceError,
    canonicalize_raw_orats_dividend_forecast,
    scan_raw_orats_dividend_forecast,
    select_future_orats_dividend_forecasts,
)

RAW_ROW_COUNT = 5


def write_forecast(path: Path, rows: str) -> Path:
    """Write a synthetic provider-format snapshot."""
    path.write_text(rows, encoding="utf-8")
    return path


def test_scan_and_canonicalize_both_historical_filename_and_terminator_forms(
    tmp_path: Path,
) -> None:
    """Date-plus-time filenames and an optional terminal delimiter have one strict schema."""
    source = write_forecast(
        tmp_path / "202401020610ORATSStockDivForecasts.txt",
        "AAA 2024-02-01 0.25 4\nAAA 2024-05-01 0 4 \n",
    )
    observed = canonicalize_raw_orats_dividend_forecast(
        scan_raw_orats_dividend_forecast(source),
    ).collect()
    assert observed.schema == pl.Schema(ORATS_DIVIDEND_DTYPES)
    assert observed.get_column("forecast_date").to_list() == [date(2024, 1, 2)] * 2
    assert observed.get_column("amount").to_list() == [0.25, 0.0]
    assert observed.get_column("frequency_per_year").to_list() == [4, 4]


def test_future_selection_has_strict_boundary_and_deterministic_event_order(
    tmp_path: Path,
) -> None:
    """Past and same-day rows are explicit inputs but never future model events."""
    source = write_forecast(
        tmp_path / "20240102ORATSStockDivForecast.txt",
        "AAA 2024-01-01 9 1\nAAA 2024-01-02 8 1\nAAA 2024-03-01 0.4 4\n"
        "AAA 2024-02-01 0.1 1\nAAA 2024-02-01 0.2 4\n",
    )
    canonical = canonicalize_raw_orats_dividend_forecast(
        scan_raw_orats_dividend_forecast(source),
    )
    assert canonical.collect().height == RAW_ROW_COUNT
    future = select_future_orats_dividend_forecasts(canonical).collect()
    assert future.get_column("ex_date").to_list() == [
        date(2024, 2, 1),
        date(2024, 2, 1),
        date(2024, 3, 1),
    ]
    assert future.get_column("frequency_per_year").to_list() == [1, 4, 4]


def test_source_identity_and_nonempty_extra_fields_fail_explicitly(tmp_path: Path) -> None:
    """The snapshot date cannot be guessed and terminal content cannot be truncated."""
    with pytest.raises(OratsDividendSourceError, match="does not identify"):
        scan_raw_orats_dividend_forecast(
            write_forecast(tmp_path / "dividends.txt", "AAA 2024-02-01 0.25 4\n"),
        )
    with pytest.raises(OratsDividendSourceError, match="file does not exist"):
        scan_raw_orats_dividend_forecast(
            tmp_path / "20240102ORATSStockDivForecasts.txt",
        )
    source = write_forecast(
        tmp_path / "20240102ORATSStockDivForecasts.txt",
        "AAA 2024-02-01 0.25 4 unexpected\n",
    )
    with pytest.raises(OratsDividendSourceError, match="exactly four"):
        scan_raw_orats_dividend_forecast(source)


@pytest.mark.parametrize(
    ("rows", "violation"),
    [
        ("AAA 2024-02-01 -0.25 4\n", "negative_amounts"),
        ("AAA 2024-02-01 0.25 5\n", "unsupported_frequencies"),
        (
            "AAA 2024-02-01 0.25 4\nAAA 2024-02-01 0.25 4\n",
            "duplicate_rows",
        ),
        ("AAA not-a-date 0.25 4\n", "invalid_ex_dates"),
    ],
)
def test_invalid_financial_rows_fail_before_future_selection(
    tmp_path: Path,
    rows: str,
    violation: str,
) -> None:
    """Amounts, frequency codes, duplicates, and dates are never repaired or inferred."""
    source = write_forecast(
        tmp_path / "20240102ORATSStockDivForecasts.txt",
        rows,
    )
    with pytest.raises(OratsDividendContentError, match=violation):
        canonicalize_raw_orats_dividend_forecast(scan_raw_orats_dividend_forecast(source))


def test_canonical_precision_is_not_coerced() -> None:
    """Durable cash amounts remain Float64 through future selection."""
    frame = pl.LazyFrame(
        {
            "ticker": ["AAA"],
            "forecast_date": [date(2024, 1, 2)],
            "ex_date": [date(2024, 2, 1)],
            "amount": [0.25],
            "frequency_per_year": [4],
        },
        schema={**ORATS_DIVIDEND_DTYPES, "amount": pl.Float32},
    )
    with pytest.raises(OratsDividendSchemaError, match="dtype mismatches"):
        select_future_orats_dividend_forecasts(frame)
