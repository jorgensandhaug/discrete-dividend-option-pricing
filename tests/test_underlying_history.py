"""Underlying-history and historical-volatility contract tests."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta
from typing import cast

import polars as pl
import pytest

from discrete_dividend_study import (
    HistoricalVolatilitySpec,
    HistoricalVolatilitySpecError,
    UnderlyingHistoryContentError,
    UnderlyingHistorySchemaError,
    prepare_underlying_history,
)

TEST_SPEC = HistoricalVolatilitySpec(
    windows=(2, 3),
    annualization_periods_per_year=4.0,
    degrees_of_freedom=1,
)
SHORT_WINDOW = 2
LONG_WINDOW = 3


def history_rows() -> pl.LazyFrame:
    """Return two deliberately interleaved ticker histories with positive daily closes."""
    origin = date(2024, 1, 2)
    return pl.LazyFrame(
        {
            "ticker": ["B", "A", "B", "A", "B", "A", "B", "A"],
            "date": [origin + timedelta(days=offset) for offset in (0, 0, 1, 1, 2, 2, 3, 3)],
            "split_adjusted_close": [50.0, 100.0, 51.0, 110.0, 52.0, 99.0, 53.0, 108.0],
            "dividend_and_split_adjusted_close": [
                50.0,
                100.0,
                51.0,
                110.0,
                52.0,
                99.0,
                53.0,
                108.0,
            ],
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "split_adjusted_close": pl.Float64,
            "dividend_and_split_adjusted_close": pl.Float64,
        },
    )


def test_preparation_is_canonical_and_uses_complete_sample_windows() -> None:
    """Canonical sorting, adjusted-return formula, ddof, annualizer, and null lead-in are exact."""
    observed = prepare_underlying_history(history_rows(), TEST_SPEC).collect()
    assert observed.select("ticker", "date").rows() == sorted(
        observed.select("ticker", "date").rows(),
    )
    assert observed.schema["historical_volatility_2"] == pl.Float64

    ticker_a = observed.filter(pl.col("ticker") == "A")
    assert ticker_a.get_column("historical_volatility_2").null_count() == SHORT_WINDOW
    returns = [math.log(110.0 / 100.0), math.log(99.0 / 110.0)]
    mean = sum(returns) / len(returns)
    sample_variance = sum((value - mean) ** 2 for value in returns)
    expected = math.sqrt(sample_variance) * math.sqrt(4.0)
    assert ticker_a.get_column("historical_volatility_2")[2] == pytest.approx(expected)
    assert ticker_a.get_column("historical_volatility_3").null_count() == LONG_WINDOW


@pytest.mark.parametrize(
    ("column", "replacement", "violation"),
    [
        ("ticker", pl.lit(" "), "blank_tickers"),
        ("split_adjusted_close", pl.lit(0.0), "nonpositive_split_close"),
        (
            "dividend_and_split_adjusted_close",
            pl.lit(float("inf")),
            "nonfinite_total_return_close",
        ),
    ],
)
def test_invalid_history_content_fails_before_feature_materialization(
    column: str,
    replacement: pl.Expr,
    violation: str,
) -> None:
    """Blank identities and invalid prices cannot reach rolling arithmetic."""
    with pytest.raises(UnderlyingHistoryContentError, match=violation):
        prepare_underlying_history(
            history_rows().with_columns(replacement.alias(column)), TEST_SPEC
        )


def test_duplicate_daily_keys_and_empty_inputs_are_rejected() -> None:
    """A daily series never resolves duplicate observations or an absent source silently."""
    duplicate = pl.concat([history_rows(), history_rows().slice(0, 1)])
    with pytest.raises(UnderlyingHistoryContentError, match="duplicate_keys"):
        prepare_underlying_history(duplicate, TEST_SPEC)
    with pytest.raises(UnderlyingHistoryContentError, match="empty_input"):
        prepare_underlying_history(history_rows().limit(0), TEST_SPEC)


def test_schema_and_estimator_identity_reject_coercions() -> None:
    """Runtime precision and estimator-defining values must be explicit and valid."""
    with pytest.raises(UnderlyingHistorySchemaError, match="dtype mismatches"):
        prepare_underlying_history(
            history_rows().with_columns(pl.col("split_adjusted_close").cast(pl.Float32)),
            TEST_SPEC,
        )
    invalid_window: object = True
    with pytest.raises(HistoricalVolatilitySpecError, match="strictly increasing"):
        replace(TEST_SPEC, windows=(cast("int", invalid_window), 3))
    with pytest.raises(HistoricalVolatilitySpecError, match="strictly increasing"):
        replace(TEST_SPEC, windows=(3, 2))
    with pytest.raises(HistoricalVolatilitySpecError, match="below every window"):
        replace(TEST_SPEC, degrees_of_freedom=2)
    with pytest.raises(HistoricalVolatilitySpecError, match="finite positive"):
        replace(TEST_SPEC, annualization_periods_per_year=float("nan"))
