"""Tests for immutable licensed historical-dividend projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from discrete_dividend_study.cli import main
from discrete_dividend_study.data.dividends.historical_projection import (
    HISTORICAL_DIVIDEND_PROJECTION_DTYPES,
    HistoricalDividendProjectionContentError,
    HistoricalDividendProjectionReceipt,
    HistoricalDividendProjectionSpec,
    HistoricalDividendProjectionSpecError,
    historical_dividend_projection_spec_sha256,
    load_verified_historical_dividend_projection,
    materialize_historical_dividend_projection,
)

if TYPE_CHECKING:
    from pathlib import Path


MAXIMUM_EX_DATE = date(2024, 9, 27)
EXCLUDED_TICKER_ROWS = 2
NONPOSITIVE_SOURCE_ROWS = 2


def _spec() -> HistoricalDividendProjectionSpec:
    return HistoricalDividendProjectionSpec(
        tickers=("DUK", "VZ"),
        maximum_ex_date=MAXIMUM_EX_DATE,
    )


def _source_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["DUK", "DUK", "VZ", "AAPL", "IBM"],
            "ex_date": [
                date(2023, 1, 2),
                date(2025, 1, 2),
                date(2023, 2, 2),
                date(2023, 3, 2),
                date(2023, 4, 2),
            ],
            "amount": [0.9, 1.0, 0.65, 0.0, -0.5],
            "freq": [4, 4, 4, 0, 1],
            "pay_date": [
                date(2023, 1, 5),
                date(2025, 1, 5),
                date(1970, 1, 1),
                date(2023, 3, 3),
                date(2023, 4, 3),
            ],
        },
        schema={
            "ticker": pl.String,
            "ex_date": pl.Date,
            "amount": pl.Float64,
            "freq": pl.Int64,
            "pay_date": pl.Date,
        },
    )


def _write_source(path: Path, frame: pl.DataFrame | None = None) -> str:
    selected = _source_frame() if frame is None else frame
    selected.write_parquet(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_projection_is_pinned_filtered_receipted_and_exactly_reusable(tmp_path: Path) -> None:
    """A licensed input becomes one bounded content-addressed study artifact."""
    source = tmp_path / "all_hist_div.parquet"
    source_sha256 = _write_source(source)
    first = materialize_historical_dividend_projection(
        source,
        expected_source_sha256=source_sha256,
        expected_source_rows=_source_frame().height,
        spec=_spec(),
        output_root=tmp_path / "projected",
    )
    expected = (
        tmp_path
        / "projected"
        / source_sha256
        / historical_dividend_projection_spec_sha256(_spec())
        / "historical_dividends.parquet"
    )
    assert first.path == expected
    assert first.receipt.output_rows == len(_spec().tickers)
    assert first.receipt.excluded_ticker_rows == EXCLUDED_TICKER_ROWS
    assert first.receipt.excluded_after_maximum_ex_date_rows == 1
    assert first.receipt.source_nonpositive_amount_rows == NONPOSITIVE_SOURCE_ROWS
    assert first.receipt.source_negative_amount_rows == 1
    assert first.receipt.source_pay_before_ex_rows == 1
    assert first.receipt.observed_tickers == _spec().tickers

    frame = pl.read_parquet(expected)
    assert frame.schema == pl.Schema(HISTORICAL_DIVIDEND_PROJECTION_DTYPES)
    assert frame.get_column("ticker").to_list() == ["DUK", "VZ"]
    observed_maximum = cast("date", frame.get_column("ex_date").max())
    assert observed_maximum <= MAXIMUM_EX_DATE

    receipt_path = expected.with_name(f"{expected.name}.receipt.json")
    raw_receipt = json.loads(receipt_path.read_bytes())
    assert raw_receipt == first.receipt.as_dict()
    assert HistoricalDividendProjectionReceipt.from_dict(raw_receipt) == first.receipt
    second = materialize_historical_dividend_projection(
        source,
        expected_source_sha256=source_sha256,
        expected_source_rows=_source_frame().height,
        spec=_spec(),
        output_root=tmp_path / "projected",
    )
    assert second == first
    assert load_verified_historical_dividend_projection(expected) == first


def test_source_hash_schema_rows_and_selected_universe_are_never_inferred(tmp_path: Path) -> None:
    """Every source identity and requested ticker history is caller-owned and exact."""
    source = tmp_path / "source.parquet"
    source_sha256 = _write_source(source)
    with pytest.raises(HistoricalDividendProjectionContentError, match="SHA-256"):
        materialize_historical_dividend_projection(
            source,
            expected_source_sha256="0" * 64,
            expected_source_rows=_source_frame().height,
            spec=_spec(),
            output_root=tmp_path / "wrong-hash",
        )
    with pytest.raises(HistoricalDividendProjectionContentError, match="row count"):
        materialize_historical_dividend_projection(
            source,
            expected_source_sha256=source_sha256,
            expected_source_rows=99,
            spec=_spec(),
            output_root=tmp_path / "wrong-rows",
        )

    malformed = tmp_path / "malformed.parquet"
    malformed_frame = _source_frame().with_columns(pl.col("amount").cast(pl.Float32))
    malformed_sha256 = _write_source(malformed, malformed_frame)
    with pytest.raises(HistoricalDividendProjectionContentError, match="schema mismatch"):
        materialize_historical_dividend_projection(
            malformed,
            expected_source_sha256=malformed_sha256,
            expected_source_rows=malformed_frame.height,
            spec=_spec(),
            output_root=tmp_path / "wrong-schema",
        )

    with pytest.raises(HistoricalDividendProjectionContentError, match="missing 1"):
        materialize_historical_dividend_projection(
            source,
            expected_source_sha256=source_sha256,
            expected_source_rows=_source_frame().height,
            spec=HistoricalDividendProjectionSpec(
                tickers=("DUK", "MISSING"),
                maximum_ex_date=MAXIMUM_EX_DATE,
            ),
            output_root=tmp_path / "missing-ticker",
        )


@pytest.mark.parametrize("violation", ["negative", "duplicate"])
def test_selected_history_violations_fail_instead_of_being_repaired(
    tmp_path: Path,
    violation: str,
) -> None:
    """Negative selected cash and duplicates introduced by projection remain fatal."""
    source = tmp_path / f"{violation}.parquet"
    frame = _source_frame()
    if violation == "negative":
        frame = frame.with_columns(
            pl.when(pl.col("ticker") == "DUK")
            .then(pl.lit(-0.1))
            .otherwise(pl.col("amount"))
            .alias("amount"),
        )
    else:
        duplicate_after_projection = pl.DataFrame(
            {
                "ticker": ["DUK"],
                "ex_date": [date(2023, 1, 2)],
                "amount": [0.9],
                "freq": [12],
                "pay_date": [date(2023, 1, 6)],
            },
            schema=frame.schema,
        )
        frame = pl.concat((frame, duplicate_after_projection))
    source_sha256 = _write_source(source, frame)
    with pytest.raises(HistoricalDividendProjectionContentError, match=violation):
        materialize_historical_dividend_projection(
            source,
            expected_source_sha256=source_sha256,
            expected_source_rows=frame.height,
            spec=_spec(),
            output_root=tmp_path / "projected",
        )


def test_existing_output_is_never_reused_after_same_size_tampering(tmp_path: Path) -> None:
    """A matching path and byte count cannot substitute for receipt-bound bytes."""
    source = tmp_path / "source.parquet"
    source_sha256 = _write_source(source)
    materialized = materialize_historical_dividend_projection(
        source,
        expected_source_sha256=source_sha256,
        expected_source_rows=_source_frame().height,
        spec=_spec(),
        output_root=tmp_path / "projected",
    )
    size = materialized.path.stat().st_size
    materialized.path.write_bytes(b"x" * size)
    with pytest.raises(HistoricalDividendProjectionContentError, match="receipt SHA-256"):
        materialize_historical_dividend_projection(
            source,
            expected_source_sha256=source_sha256,
            expected_source_rows=_source_frame().height,
            spec=_spec(),
            output_root=tmp_path / "projected",
        )


def test_projection_spec_refuses_implicit_canonicalization_and_coercion() -> None:
    """Ticker order, duplication, case, dates, and physical settings are explicit."""
    for tickers in ((), ("VZ", "DUK"), ("DUK", "DUK"), ("duk",)):
        with pytest.raises(HistoricalDividendProjectionSpecError):
            HistoricalDividendProjectionSpec(
                tickers=tickers,
                maximum_ex_date=MAXIMUM_EX_DATE,
            )
    invalid_date: object = "2024-09-27"
    with pytest.raises(HistoricalDividendProjectionSpecError, match="plain date"):
        HistoricalDividendProjectionSpec(
            tickers=("DUK",),
            maximum_ex_date=cast("date", invalid_date),
        )
    with pytest.raises(HistoricalDividendProjectionSpecError, match="compression level"):
        replace(_spec(), compression_level=9)


def test_cli_binds_source_identity_date_cap_and_ticker_universe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public command requires every source and selection identity explicitly."""
    source = tmp_path / "source.parquet"
    source_sha256 = _write_source(source)
    arguments = (
        "materialize-historical-dividends",
        "--source",
        str(source),
        "--expected-source-sha256",
        source_sha256,
        "--expected-source-rows",
        str(_source_frame().height),
        "--maximum-ex-date",
        MAXIMUM_EX_DATE.isoformat(),
        "--output-root",
        str(tmp_path / "projected"),
        "--ticker",
        "DUK",
        "--ticker",
        "VZ",
    )
    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == len(_spec().tickers)
    assert summary["source_sha256"] == source_sha256
    assert summary["projection_spec_sha256"] == historical_dividend_projection_spec_sha256(
        _spec(),
    )
    assert main(arguments) == 0
