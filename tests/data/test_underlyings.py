"""Underlying-history and historical-volatility contract tests."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta
from typing import cast

import polars as pl
import pytest

from discrete_dividend_study.core.schema import ConflictingColumnsError
from discrete_dividend_study.data.underlyings import (
    HistoricalVolatilitySpec,
    HistoricalVolatilitySpecError,
    UnderlyingHistoryContentError,
    UnderlyingHistoryJoinError,
    UnderlyingHistoryJoinSchemaError,
    UnderlyingHistoryJoinSpec,
    UnderlyingHistoryJoinSpecError,
    UnderlyingHistorySchemaError,
    join_underlying_history,
    prepare_underlying_history,
)

TEST_SPEC = HistoricalVolatilitySpec(
    windows=(2, 3),
    annualization_periods_per_year=4.0,
    degrees_of_freedom=1,
)
SHORT_WINDOW = 2
LONG_WINDOW = 3
UNLIMITED_EXACT_JOIN = UnderlyingHistoryJoinSpec(
    allow_exact_matches=True,
    maximum_calendar_lag_days=None,
)


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


def quote_rows() -> pl.LazyFrame:
    """Return sorted exact-date and weekend-like quote keys for both synthetic tickers."""
    return pl.LazyFrame(
        {
            "ticker": ["A", "A", "B"],
            "quote_date": [date(2024, 1, 3), date(2024, 1, 6), date(2024, 1, 5)],
            "quote_id": [1, 2, 3],
        },
        schema={"ticker": pl.String, "quote_date": pl.Date, "quote_id": pl.Int64},
    )


def test_backward_join_records_exact_support_and_staleness_without_reordering() -> None:
    """The join preserves quote order and exposes both matched date and civil-day lag."""
    history = prepare_underlying_history(history_rows(), TEST_SPEC)
    observed = join_underlying_history(
        quote_rows(), history, TEST_SPEC, UNLIMITED_EXACT_JOIN
    ).collect()
    assert observed.get_column("quote_id").to_list() == [1, 2, 3]
    assert observed.get_column("underlying_observation_date").to_list() == [
        date(2024, 1, 3),
        date(2024, 1, 5),
        date(2024, 1, 5),
    ]
    assert observed.get_column("underlying_observation_lag_days").to_list() == [0, 1, 0]


def test_join_rejects_unordered_unsupported_and_overstale_quotes() -> None:
    """Grouped as-of preconditions and coverage are fail-fast rather than Polars assumptions."""
    history = prepare_underlying_history(history_rows(), TEST_SPEC)
    with pytest.raises(UnderlyingHistoryJoinError, match="out_of_order_quote_rows"):
        join_underlying_history(quote_rows().reverse(), history, TEST_SPEC, UNLIMITED_EXACT_JOIN)
    with pytest.raises(UnderlyingHistoryJoinError, match="unsupported_quote_dates"):
        join_underlying_history(
            quote_rows().with_columns(pl.lit("MISSING").alias("ticker")).sort("quote_date"),
            history,
            TEST_SPEC,
            UNLIMITED_EXACT_JOIN,
        )
    with pytest.raises(UnderlyingHistoryJoinError, match="maximum_observed_lag"):
        join_underlying_history(
            quote_rows(),
            history,
            TEST_SPEC,
            replace(UNLIMITED_EXACT_JOIN, maximum_calendar_lag_days=0),
        )


def test_strict_previous_observation_and_join_spec_are_explicit() -> None:
    """Disabling exact matches selects a prior date and cannot pair with a zero-day tolerance."""
    history = prepare_underlying_history(history_rows(), TEST_SPEC)
    strict_previous = UnderlyingHistoryJoinSpec(
        allow_exact_matches=False,
        maximum_calendar_lag_days=3,
    )
    observed = join_underlying_history(quote_rows(), history, TEST_SPEC, strict_previous).collect()
    assert observed.get_column("underlying_observation_date")[0] == date(2024, 1, 2)
    invalid_bool: object = 1
    with pytest.raises(UnderlyingHistoryJoinSpecError, match="plain bool"):
        replace(
            strict_previous,
            allow_exact_matches=cast("bool", invalid_bool),
        )
    with pytest.raises(UnderlyingHistoryJoinSpecError, match="positive or None"):
        replace(strict_previous, maximum_calendar_lag_days=0)


def test_join_support_schema_is_not_coerced() -> None:
    """Prepared volatility precision is part of the join contract."""
    history = prepare_underlying_history(history_rows(), TEST_SPEC).with_columns(
        pl.col("historical_volatility_2").cast(pl.Float32),
    )
    with pytest.raises(UnderlyingHistoryJoinSchemaError, match="dtype mismatches"):
        join_underlying_history(quote_rows(), history, TEST_SPEC, UNLIMITED_EXACT_JOIN)

    with pytest.raises(ConflictingColumnsError, match="already exist"):
        join_underlying_history(
            quote_rows().with_columns(pl.lit(0.2).alias("historical_volatility_2")),
            prepare_underlying_history(history_rows(), TEST_SPEC),
            TEST_SPEC,
            UNLIMITED_EXACT_JOIN,
        )
