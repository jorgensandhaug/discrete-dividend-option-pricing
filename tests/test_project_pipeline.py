"""End-to-end project as-submitted feature pipeline test."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from discrete_dividend_study import (
    HISTORICAL_DIVIDEND_DTYPES,
    NSS_PARAMETER_DTYPES,
    ORATS_DIVIDEND_DTYPES,
    ORATS_NORMALIZED_OPTION_DTYPES,
    PROJECT_AS_SUBMITTED_FEATURE_COLUMNS,
    PROJECT_AS_SUBMITTED_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_TICKERS,
    ProjectFeatureUniverseError,
    build_project_as_submitted_features,
    project_feature_schema_sha256,
)

HISTORY_ROWS = 430
SHORTEST_RATE = 0.019
SHA256_HEX_LENGTH = 64


def test_project_feature_schema_has_one_stable_ordered_identity() -> None:
    """The manifest schema hash is lowercase SHA-256 over the ordered feature contract."""
    first = project_feature_schema_sha256()
    assert first == project_feature_schema_sha256()
    assert len(first) == SHA256_HEX_LENGTH
    assert first == first.lower()


def normalized_options() -> pl.LazyFrame:
    """Return one passing quote and one strict-call-price rejection per study ticker."""
    tickers = [ticker for ticker in PROJECT_AS_SUBMITTED_TICKERS for _ in range(2)]
    return pl.LazyFrame(
        {
            "ticker": tickers,
            "quote_date": [date(2024, 10, 1)] * len(tickers),
            "expiry_date": [date(2024, 11, 1)] * len(tickers),
            "strike": [80.0, 80.0] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "call_ask": [21.0, 0.1] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "call_bid": [20.5, 0.1] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "put_ask": [1.0, 1.0] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "put_bid": [0.9, 0.9] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "underlying_price": [100.0, 100.0] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "call_volume": [10.0, 10.0] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "put_volume": [5.0, 5.0] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "call_open_interest": [100.0, 100.0] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "put_open_interest": [50.0, 50.0] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "call_theoretical_value": [20.8, 0.1] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "put_theoretical_value": [0.95, 0.95] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "call_iv": [0.2, 0.2] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "put_iv": [0.21, 0.21] * len(PROJECT_AS_SUBMITTED_TICKERS),
        },
        schema=ORATS_NORMALIZED_OPTION_DTYPES,
    )


def underlying_history() -> pl.LazyFrame:
    """Return enough adjusted observations for every submitted volatility window."""
    origin = date(2023, 8, 1)
    dates = [origin + timedelta(days=index) for index in range(HISTORY_ROWS)]
    values = [90.0 + index * 0.01 for index in range(HISTORY_ROWS)]
    return pl.LazyFrame(
        {
            "ticker": [
                ticker for ticker in PROJECT_AS_SUBMITTED_TICKERS for _ in range(HISTORY_ROWS)
            ],
            "date": dates * len(PROJECT_AS_SUBMITTED_TICKERS),
            "split_adjusted_close": values * len(PROJECT_AS_SUBMITTED_TICKERS),
            "dividend_and_split_adjusted_close": values * len(PROJECT_AS_SUBMITTED_TICKERS),
        },
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "split_adjusted_close": pl.Float64,
            "dividend_and_split_adjusted_close": pl.Float64,
        },
    )


def future_forecasts() -> pl.LazyFrame:
    """Return one exact-date future cash event inside the option horizon."""
    return pl.LazyFrame(
        {
            "ticker": list(PROJECT_AS_SUBMITTED_TICKERS),
            "forecast_date": [date(2024, 10, 1)] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "ex_date": [date(2024, 10, 15)] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "amount": [0.5] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "frequency_per_year": [4] * len(PROJECT_AS_SUBMITTED_TICKERS),
        },
        schema=ORATS_DIVIDEND_DTYPES,
    )


def historical_dividends() -> pl.LazyFrame:
    """Return one observed cash payment inside the trailing year."""
    return pl.LazyFrame(
        {
            "ticker": list(PROJECT_AS_SUBMITTED_TICKERS),
            "ex_date": [date(2024, 9, 1)] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "amount": [0.4] * len(PROJECT_AS_SUBMITTED_TICKERS),
        },
        schema=HISTORICAL_DIVIDEND_DTYPES,
    )


def nss_parameters() -> pl.LazyFrame:
    """Return one finite exact-date submitted parameter row."""
    return pl.LazyFrame(
        {
            "date": [date(2024, 10, 1)],
            "beta0": [0.03],
            "beta1": [-0.01],
            "beta2": [0.02],
            "beta3": [0.005],
            "tau1": [1.5],
            "tau2": [3.0],
            "shortest_maturity": [0.25],
            "shortest_rate": [SHORTEST_RATE],
        },
        schema=NSS_PARAMETER_DTYPES,
    )


def test_project_pipeline_has_one_ordered_typed_complete_feature_contract() -> None:
    """All stages compose lazily; strict rejected/null rows cannot enter the artifact."""
    observed = build_project_as_submitted_features(
        normalized_options(),
        underlying_history(),
        future_forecasts(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    assert observed.shape == (
        len(PROJECT_AS_SUBMITTED_TICKERS),
        len(PROJECT_AS_SUBMITTED_FEATURE_COLUMNS),
    )
    assert observed.columns == list(PROJECT_AS_SUBMITTED_FEATURE_COLUMNS)
    assert observed.schema == pl.Schema(PROJECT_AS_SUBMITTED_FEATURE_DTYPES)
    assert observed.null_count().row(0) == (0,) * len(PROJECT_AS_SUBMITTED_FEATURE_COLUMNS)
    assert observed["forecasted_discrete_ex_dates"].to_list() == [[date(2024, 10, 15)]] * len(
        PROJECT_AS_SUBMITTED_TICKERS
    )
    assert observed["risk_free_rate"].to_list() == [SHORTEST_RATE] * len(
        PROJECT_AS_SUBMITTED_TICKERS
    )


def test_project_pipeline_rejects_an_incomplete_frozen_universe() -> None:
    """A missing submitted ticker cannot silently shrink the study cohort."""
    incomplete = normalized_options().filter(pl.col("ticker") != PROJECT_AS_SUBMITTED_TICKERS[0])
    with pytest.raises(ProjectFeatureUniverseError, match="DUK"):
        build_project_as_submitted_features(
            incomplete,
            underlying_history(),
            future_forecasts(),
            historical_dividends(),
            nss_parameters(),
        )
