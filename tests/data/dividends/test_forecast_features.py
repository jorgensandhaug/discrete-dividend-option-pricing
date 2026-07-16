"""Point-in-time forecast-dividend feature contract tests."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from typing import cast

import polars as pl
import pytest

from discrete_dividend_study.core.schema import ConflictingColumnsError
from discrete_dividend_study.data.dividends.forecast_features import (
    DIVIDEND_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
    DividendFeatureContentError,
    DividendFeatureSchemaError,
    DividendFeatureSpecError,
    ForecastDividendFeatureSpec,
    add_forecast_dividend_features,
)

EXPECTED_COMPONENT_COUNT = 4


def quote_rows() -> pl.LazyFrame:
    """Return duplicate contracts on one horizon plus a shorter expiration."""
    return pl.LazyFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "quote_date": [date(2024, 1, 2)] * 3,
            "expiry_date": [date(2024, 4, 1), date(2024, 4, 1), date(2024, 2, 1)],
            "underlying_price": [100.0, 100.0, 100.0],
            "contract_id": [1, 2, 3],
        },
        schema={
            "ticker": pl.String,
            "quote_date": pl.Date,
            "expiry_date": pl.Date,
            "underlying_price": pl.Float32,
            "contract_id": pl.Int64,
        },
    )


def forecast_rows() -> pl.LazyFrame:
    """Return zero, same-date components, an expiry-date event, and a later event."""
    return pl.LazyFrame(
        {
            "ticker": ["AAA"] * 5,
            "forecast_date": [date(2024, 1, 2)] * 5,
            "ex_date": [
                date(2024, 1, 15),
                date(2024, 2, 1),
                date(2024, 2, 1),
                date(2024, 4, 1),
                date(2024, 5, 1),
            ],
            "amount": [0.0, 0.1, 0.2, 0.4, 0.5],
            "frequency_per_year": [4, 1, 4, 4, 4],
        },
        schema={
            "ticker": pl.String,
            "forecast_date": pl.Date,
            "ex_date": pl.Date,
            "amount": pl.Float64,
            "frequency_per_year": pl.UInt8,
        },
    )


def test_submitted_schedule_preserves_components_zero_events_and_quote_order() -> None:
    """One unique horizon schedule fans out exactly to duplicate contracts."""
    observed = add_forecast_dividend_features(
        quote_rows(),
        forecast_rows(),
        PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
    ).collect()
    assert observed.get_column("contract_id").to_list() == [1, 2, 3]
    for name, dtype in DIVIDEND_FEATURE_DTYPES.items():
        assert observed.schema[name] == dtype
    first = observed.row(0, named=True)
    assert first["forecasted_discrete_ex_dates"] == [
        date(2024, 1, 15),
        date(2024, 2, 1),
        date(2024, 2, 1),
        date(2024, 4, 1),
    ]
    assert first["forecasted_discrete_amounts"] == [0.0, 0.1, 0.2, 0.4]
    assert first["forecasted_dividend_count"] == EXPECTED_COMPONENT_COUNT
    assert first["forecasted_total_dividend_in_period"] == pytest.approx(0.7)
    horizon = (date(2024, 4, 1) - date(2024, 1, 2)).days / 365
    assert first["forecasted_dividend_yield"] == pytest.approx(math.log(1.007) / horizon)
    assert observed.row(2, named=True)["forecasted_discrete_amounts"] == [0.0, 0.1, 0.2]


def test_alternative_event_policies_have_distinct_results() -> None:
    """Zero removal, same-date aggregation, and exclusive expiry are explicit identities."""
    alternative = ForecastDividendFeatureSpec(
        snapshot_coverage="ticker-date",
        include_expiry_date=False,
        retain_zero_amount_events=False,
        aggregate_same_date_components=True,
        year_basis_days=365.0,
    )
    observed = add_forecast_dividend_features(
        quote_rows(),
        forecast_rows(),
        alternative,
    ).collect()
    first = observed.row(0, named=True)
    assert first["forecasted_discrete_ex_dates"] == [date(2024, 2, 1)]
    assert first["forecasted_discrete_amounts"] == pytest.approx([0.3])


def test_valid_snapshot_with_no_in_horizon_events_has_typed_empty_schedule() -> None:
    """A covered ticker/date can validly have zero events before a short expiration."""
    short_quote = (
        quote_rows()
        .slice(0, 1)
        .with_columns(
            pl.lit(date(2024, 1, 10)).alias("expiry_date"),
        )
    )
    observed = add_forecast_dividend_features(
        short_quote,
        forecast_rows(),
        PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
    ).collect()
    row = observed.row(0, named=True)
    assert row["forecasted_discrete_ex_dates"] == []
    assert row["forecasted_discrete_amounts"] == []
    assert row["forecasted_total_dividend_in_period"] == 0.0
    assert row["forecasted_dividend_count"] == 0
    assert row["forecasted_dividend_yield"] == 0.0


def test_missing_snapshot_and_nonfuture_rows_are_not_zero_filled() -> None:
    """Missing availability differs from a valid empty in-horizon schedule."""
    with pytest.raises(DividendFeatureContentError, match="missing_ticker_date_forecasts"):
        add_forecast_dividend_features(
            quote_rows().with_columns(pl.lit("MISSING").alias("ticker")),
            forecast_rows(),
            replace(
                PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
                snapshot_coverage="ticker-date",
            ),
        )
    with pytest.raises(DividendFeatureContentError, match="nonfuture_forecast_rows"):
        add_forecast_dividend_features(
            quote_rows(),
            forecast_rows().with_columns(pl.col("forecast_date").alias("ex_date")),
            PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
        )


def test_sparse_snapshot_policy_distinguishes_empty_ticker_from_missing_date() -> None:
    """A verified sparse snapshot may omit a ticker, but the snapshot date itself is mandatory."""
    absent_ticker = quote_rows().with_columns(pl.lit("MISSING").alias("ticker"))
    observed = add_forecast_dividend_features(
        absent_ticker,
        forecast_rows(),
        PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
    ).collect()
    assert observed["forecasted_dividend_count"].to_list() == [0, 0, 0]
    assert observed["forecasted_discrete_amounts"].to_list() == [[], [], []]
    with pytest.raises(DividendFeatureContentError, match="missing_forecast_dates"):
        add_forecast_dividend_features(
            quote_rows().with_columns(pl.lit(date(2024, 1, 3)).alias("quote_date")),
            forecast_rows(),
            PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
        )


def test_quote_domains_dtypes_and_output_ownership_are_strict() -> None:
    """Invalid spot/horizon, precision changes, and stage-name collisions fail before joining."""
    with pytest.raises(DividendFeatureContentError, match="nonpositive_underlying_prices"):
        add_forecast_dividend_features(
            quote_rows().with_columns(pl.lit(0.0, dtype=pl.Float32).alias("underlying_price")),
            forecast_rows(),
            PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
        )
    with pytest.raises(DividendFeatureSchemaError, match="dtype mismatches"):
        add_forecast_dividend_features(
            quote_rows().with_columns(pl.col("underlying_price").cast(pl.Float64)),
            forecast_rows(),
            PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
        )
    with pytest.raises(ConflictingColumnsError, match="already exist"):
        add_forecast_dividend_features(
            quote_rows().with_columns(pl.lit(0.0).alias("forecasted_dividend_yield")),
            forecast_rows(),
            PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
        )


def test_feature_spec_rejects_coercions_and_invalid_year_basis() -> None:
    """Event and annualization policies cannot be inferred from truthy values."""
    invalid_flag: object = 1
    with pytest.raises(DividendFeatureSpecError, match="plain bool"):
        replace(
            PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
            include_expiry_date=cast("bool", invalid_flag),
        )
    with pytest.raises(DividendFeatureSpecError, match="finite positive"):
        replace(PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC, year_basis_days=float("nan"))
