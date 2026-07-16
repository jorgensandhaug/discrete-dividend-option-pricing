"""Trailing observed-cash dividend feature contract tests."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from typing import cast

import polars as pl
import pytest

from discrete_dividend_study.core.schema import ConflictingColumnsError
from discrete_dividend_study.data.dividends.historical_features import (
    HISTORICAL_DIVIDEND_FEATURE_DTYPES,
    HistoricalDividendContentError,
    HistoricalDividendFeatureSpec,
    HistoricalDividendSchemaError,
    HistoricalDividendSpecError,
    add_historical_dividend_features,
)

INCLUSIVE_SPEC = HistoricalDividendFeatureSpec(
    lookback_weeks=4,
    weeks_per_year=52.0,
    include_quote_date=True,
    include_lookback_boundary=True,
    require_ticker_history=True,
)


def quote_rows() -> pl.LazyFrame:
    """Return duplicate contracts on one quote date in deliberate input order."""
    return pl.LazyFrame(
        {
            "ticker": ["AAA", "AAA"],
            "quote_date": [date(2024, 1, 29)] * 2,
            "split_adjusted_close": [100.0, 100.0],
            "contract_id": [2, 1],
        },
        schema={
            "ticker": pl.String,
            "quote_date": pl.Date,
            "split_adjusted_close": pl.Float64,
            "contract_id": pl.Int64,
        },
    )


def history_rows() -> pl.LazyFrame:
    """Return lower/upper boundary, same-date, prior, and future payments."""
    return pl.LazyFrame(
        {
            "ticker": ["AAA"] * 6,
            "ex_date": [
                date(2023, 12, 31),
                date(2024, 1, 1),
                date(2024, 1, 15),
                date(2024, 1, 15),
                date(2024, 1, 29),
                date(2024, 2, 1),
            ],
            "amount": [9.0, 0.1, 0.2, 0.3, 0.4, 8.0],
        },
        schema={"ticker": pl.String, "ex_date": pl.Date, "amount": pl.Float64},
    )


def test_inclusive_window_sums_components_once_and_preserves_quote_order() -> None:
    """Both endpoints and two same-date components produce one unique-date total."""
    observed = add_historical_dividend_features(
        quote_rows(), history_rows(), INCLUSIVE_SPEC
    ).collect()
    assert observed.get_column("contract_id").to_list() == [2, 1]
    for name, dtype in HISTORICAL_DIVIDEND_FEATURE_DTYPES.items():
        assert observed.schema[name] == dtype
    row = observed.row(0, named=True)
    assert row["historical_total_dividend_in_lookback_window"] == pytest.approx(1.0)
    expected = math.log(1.01) / (4 / 52)
    assert row["historical_dividend_yield"] == pytest.approx(expected)


def test_exclusive_boundaries_and_missing_ticker_policy_are_explicit() -> None:
    """Boundary and coverage changes require a different immutable specification."""
    exclusive = replace(
        INCLUSIVE_SPEC,
        include_quote_date=False,
        include_lookback_boundary=False,
    )
    observed = add_historical_dividend_features(quote_rows(), history_rows(), exclusive).collect()
    assert observed["historical_total_dividend_in_lookback_window"][0] == pytest.approx(0.5)

    missing_quotes = quote_rows().with_columns(pl.lit("MISSING").alias("ticker"))
    with pytest.raises(HistoricalDividendContentError, match="missing_ticker_histories"):
        add_historical_dividend_features(missing_quotes, history_rows(), INCLUSIVE_SPEC)
    allowed = replace(INCLUSIVE_SPEC, require_ticker_history=False)
    missing = add_historical_dividend_features(missing_quotes, history_rows(), allowed).collect()
    assert missing["historical_total_dividend_in_lookback_window"][0] == 0.0
    assert missing["historical_dividend_yield"][0] == 0.0


def test_invalid_history_quote_domains_and_schema_fail_before_asof() -> None:
    """Amounts, adjusted closes, exact duplicates, and precision are never repaired."""
    with pytest.raises(HistoricalDividendContentError, match="negative_history_amounts"):
        add_historical_dividend_features(
            quote_rows(),
            history_rows().with_columns(pl.lit(-1.0).alias("amount")),
            INCLUSIVE_SPEC,
        )
    duplicate = pl.concat([history_rows(), history_rows().slice(0, 1)])
    with pytest.raises(HistoricalDividendContentError, match="duplicate_history_rows"):
        add_historical_dividend_features(quote_rows(), duplicate, INCLUSIVE_SPEC)
    with pytest.raises(HistoricalDividendContentError, match="nonpositive_split_adjusted_closes"):
        add_historical_dividend_features(
            quote_rows().with_columns(pl.lit(0.0).alias("split_adjusted_close")),
            history_rows(),
            INCLUSIVE_SPEC,
        )
    with pytest.raises(HistoricalDividendSchemaError, match="dtype mismatches"):
        add_historical_dividend_features(
            quote_rows(),
            history_rows().with_columns(pl.col("amount").cast(pl.Float32)),
            INCLUSIVE_SPEC,
        )
    with pytest.raises(ConflictingColumnsError, match="already exist"):
        add_historical_dividend_features(
            quote_rows().with_columns(pl.lit(0.0).alias("historical_dividend_yield")),
            history_rows(),
            INCLUSIVE_SPEC,
        )


def test_spec_rejects_coerced_boundaries_and_invalid_annualizers() -> None:
    """Week window, annualizer, and endpoint policies are fully typed."""
    invalid_flag: object = 1
    with pytest.raises(HistoricalDividendSpecError, match="plain bool"):
        replace(INCLUSIVE_SPEC, include_quote_date=cast("bool", invalid_flag))
    with pytest.raises(HistoricalDividendSpecError, match="plain positive int"):
        replace(INCLUSIVE_SPEC, lookback_weeks=0)
    with pytest.raises(HistoricalDividendSpecError, match="finite positive"):
        replace(INCLUSIVE_SPEC, weeks_per_year=float("inf"))
