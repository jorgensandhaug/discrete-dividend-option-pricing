"""End-to-end project as-submitted feature pipeline test."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from discrete_dividend_study.data.dividends.historical_features import HISTORICAL_DIVIDEND_DTYPES
from discrete_dividend_study.data.options.filters import QuoteNumericPrecision
from discrete_dividend_study.data.options.quotes import (
    FLOAT32_PAIRED_OPTION_QUOTE_DTYPES,
    FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
)
from discrete_dividend_study.data.pipeline import (
    PROJECT_AS_SUBMITTED_FEATURE_COLUMNS,
    PROJECT_AS_SUBMITTED_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_TICKERS,
    PROJECT_EXPERIMENT_FEATURE_COLUMNS,
    PROJECT_EXPERIMENT_FEATURE_DTYPES,
    ProjectFeatureUniverseError,
    build_project_as_submitted_features,
    build_project_experiment_features,
    build_project_experiment_quote_partition,
    project_experiment_feature_pipeline_sha256,
    project_experiment_feature_schema_sha256,
    project_feature_pipeline_sha256,
    project_feature_schema_sha256,
)
from discrete_dividend_study.data.providers.orats.dividends import ORATS_DIVIDEND_DTYPES
from discrete_dividend_study.data.providers.orats.options import ORATS_NORMALIZED_OPTION_DTYPES
from discrete_dividend_study.data.rates import NSS_PARAMETER_DTYPES

HISTORY_ROWS = 430
SHORTEST_RATE = 0.019
SHA256_HEX_LENGTH = 64


def test_project_feature_schema_has_one_stable_ordered_identity() -> None:
    """The manifest schema hash is lowercase SHA-256 over the ordered feature contract."""
    first = project_feature_schema_sha256()
    assert first == project_feature_schema_sha256()
    assert len(first) == SHA256_HEX_LENGTH
    assert first == first.lower()


def test_project_feature_pipeline_has_one_complete_stable_identity() -> None:
    """All frozen stage choices contribute to one lowercase SHA-256 identity."""
    first = project_feature_pipeline_sha256()
    assert first == project_feature_pipeline_sha256()
    assert len(first) == SHA256_HEX_LENGTH
    assert first == first.lower()


def test_project_experiment_contract_has_distinct_stable_identities() -> None:
    """The provider-minimal experiment boundary is ordered, typed, and independently hashed."""
    schema_hash = project_experiment_feature_schema_sha256()
    pipeline_hash = project_experiment_feature_pipeline_sha256()
    assert schema_hash == project_experiment_feature_schema_sha256()
    assert pipeline_hash == project_experiment_feature_pipeline_sha256()
    assert len(schema_hash) == SHA256_HEX_LENGTH
    assert len(pipeline_hash) == SHA256_HEX_LENGTH
    assert schema_hash != project_feature_schema_sha256()
    assert pipeline_hash != project_feature_pipeline_sha256()
    assert (
        project_experiment_feature_pipeline_sha256(QuoteNumericPrecision.FLOAT64) != pipeline_hash
    )


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


def normalized_options_with_unavailable_nss_date() -> pl.LazyFrame:
    """Append otherwise valid quotes on a date absent from the submitted NSS artifact."""
    unavailable = normalized_options().with_columns(
        pl.lit(date(2024, 10, 2)).alias("quote_date"),
        pl.lit(date(2024, 11, 2)).alias("expiry_date"),
    )
    return pl.concat((normalized_options(), unavailable))


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


def future_forecasts_with_irrelevant_duplicate_snapshot() -> pl.LazyFrame:
    """Append a duplicated old snapshot that cannot affect any selected quote."""
    old = pl.LazyFrame(
        {
            "ticker": [PROJECT_AS_SUBMITTED_TICKERS[0]] * 2,
            "forecast_date": [date(2009, 1, 2)] * 2,
            "ex_date": [date(2009, 2, 1)] * 2,
            "amount": [0.5] * 2,
            "frequency_per_year": [4] * 2,
        },
        schema=ORATS_DIVIDEND_DTYPES,
    )
    return pl.concat((future_forecasts(), old))


def future_forecasts_with_consumed_nonfuture_rows() -> pl.LazyFrame:
    """Append same-day rows that the submitted source producer explicitly removed."""
    same_day = pl.LazyFrame(
        {
            "ticker": list(PROJECT_AS_SUBMITTED_TICKERS),
            "forecast_date": [date(2024, 10, 1)] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "ex_date": [date(2024, 10, 1)] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "amount": [999.0] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "frequency_per_year": [4] * len(PROJECT_AS_SUBMITTED_TICKERS),
        },
        schema=ORATS_DIVIDEND_DTYPES,
    )
    return pl.concat((future_forecasts(), same_day))


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


def test_minimal_paired_pipeline_is_value_identical_on_every_experiment_field() -> None:
    """Unused ORATS diagnostics are unnecessary for either submitted experiment family."""
    complete = build_project_as_submitted_features(
        normalized_options(),
        underlying_history(),
        future_forecasts(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    minimal = build_project_experiment_features(
        normalized_options().select(tuple(FLOAT32_PAIRED_OPTION_QUOTE_DTYPES)),
        underlying_history(),
        future_forecasts(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    assert minimal.schema == pl.Schema(PROJECT_EXPERIMENT_FEATURE_DTYPES)
    assert minimal.columns == list(PROJECT_EXPERIMENT_FEATURE_COLUMNS)
    assert minimal.equals(complete.select(PROJECT_EXPERIMENT_FEATURE_COLUMNS))


def test_source_precision_is_preserved_until_after_strict_quote_predicates() -> None:
    """A Float64 boundary quote cannot enter the cohort through premature narrowing."""
    source = pl.LazyFrame(
        {
            "ticker": ["DUK"],
            "quote_date": [date(2024, 10, 1)],
            "expiry_date": [date(2024, 11, 1)],
            "strike": [1.0],
            "call_ask": [1.11],
            "call_bid": [1.10],
            "put_ask": [0.02],
            "put_bid": [0.01],
            "underlying_price": [2.00000001],
            "call_volume": [1.0],
            "put_volume": [1.0],
        },
        schema=FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
    )
    preserved = build_project_experiment_quote_partition(
        source,
        nss_parameters(),
        tickers=("DUK",),
        source_precision=QuoteNumericPrecision.FLOAT64,
    ).collect()
    narrowed_first = build_project_experiment_quote_partition(
        source.select(
            *(
                pl.col(name).cast(dtype)
                for name, dtype in FLOAT32_PAIRED_OPTION_QUOTE_DTYPES.items()
            )
        ),
        nss_parameters(),
        tickers=("DUK",),
        source_precision=QuoteNumericPrecision.FLOAT32,
    ).collect()
    assert preserved.height == 0
    assert narrowed_first.height == 1


def test_project_pipeline_validates_only_forecast_snapshots_consumed_by_quotes() -> None:
    """Provider history outside exact quote dates cannot reject or alter the project cohort."""
    baseline = build_project_as_submitted_features(
        normalized_options(),
        underlying_history(),
        future_forecasts(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    with_irrelevant_history = build_project_as_submitted_features(
        normalized_options(),
        underlying_history(),
        future_forecasts_with_irrelevant_duplicate_snapshot(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    assert with_irrelevant_history.equals(baseline)


def test_project_pipeline_reproduces_submitted_strict_future_source_selection() -> None:
    """Consumed past or same-day vendor rows are validated, then explicitly excluded."""
    baseline = build_project_as_submitted_features(
        normalized_options(),
        underlying_history(),
        future_forecasts(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    with_nonfuture_rows = build_project_as_submitted_features(
        normalized_options(),
        underlying_history(),
        future_forecasts_with_consumed_nonfuture_rows(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    assert with_nonfuture_rows.equals(baseline)


def test_project_pipeline_reproduces_submitted_nss_inner_date_intersection() -> None:
    """Quotes without a fitted NSS date are explicitly removed before support feature joins."""
    baseline = build_project_as_submitted_features(
        normalized_options(),
        underlying_history(),
        future_forecasts(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    with_unavailable_date = build_project_as_submitted_features(
        normalized_options_with_unavailable_nss_date(),
        underlying_history(),
        future_forecasts(),
        historical_dividends(),
        nss_parameters(),
    ).collect()
    assert with_unavailable_date.equals(baseline)


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
