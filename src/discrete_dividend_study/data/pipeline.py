"""One obvious composition for the project as-submitted feature artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

from discrete_dividend_study.core.schema import forbid_columns, require_columns
from discrete_dividend_study.data.dividends.forecast_features import (
    DIVIDEND_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
    add_forecast_dividend_features,
)
from discrete_dividend_study.data.dividends.historical_features import (
    HISTORICAL_DIVIDEND_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_HISTORICAL_DIVIDEND_SPEC,
    add_historical_dividend_features,
)
from discrete_dividend_study.data.options.filters import (
    PROJECT_AS_SUBMITTED_FILTER,
    QUOTE_FEATURE_DTYPES,
    QuoteNumericPrecision,
    add_quote_features,
    apply_complete_row_policy,
    apply_quote_predicates,
)
from discrete_dividend_study.data.providers.orats.dividends import (
    select_future_orats_dividend_forecasts,
)
from discrete_dividend_study.data.providers.orats.options import ORATS_NORMALIZED_OPTION_DTYPES
from discrete_dividend_study.data.rates import (
    PROJECT_AS_SUBMITTED_NSS_RATE_SPEC,
    add_nss_rate_feature,
)
from discrete_dividend_study.data.underlyings import (
    PROJECT_AS_SUBMITTED_UNDERLYING_JOIN_SPEC,
    PROJECT_HISTORICAL_VOLATILITY_SPEC,
    join_underlying_history,
    prepare_underlying_history,
)

PROJECT_AS_SUBMITTED_FEATURE_DTYPES = {
    **ORATS_NORMALIZED_OPTION_DTYPES,
    **QUOTE_FEATURE_DTYPES,
    **DIVIDEND_FEATURE_DTYPES,
    "split_adjusted_underlying_price": pl.Float64,
    **{
        f"historical_volatility_{window}": pl.Float64
        for window in PROJECT_HISTORICAL_VOLATILITY_SPEC.windows
    },
    "risk_free_rate": pl.Float64,
    **HISTORICAL_DIVIDEND_FEATURE_DTYPES,
}
PROJECT_AS_SUBMITTED_FEATURE_COLUMNS = tuple(PROJECT_AS_SUBMITTED_FEATURE_DTYPES)
PROJECT_EXPERIMENT_FEATURE_DTYPES: Final = {
    "ticker": pl.String,
    "quote_date": pl.Date,
    "expiry_date": pl.Date,
    "strike": pl.Float32,
    "underlying_price": pl.Float32,
    "call_mid_price": pl.Float32,
    "days_to_expiry": pl.Int64,
    "yte": pl.Float64,
    "call_moneyness": pl.Float32,
    "forecasted_total_dividend_in_period": pl.Float64,
    "forecasted_discrete_ex_dates": pl.List(pl.Date),
    "forecasted_discrete_amounts": pl.List(pl.Float64),
    "forecasted_years_to_discrete_ex_dates": pl.List(pl.Float64),
    "forecasted_dividend_count": pl.UInt32,
    "forecasted_dividend_yield": pl.Float64,
    "historical_volatility_90": pl.Float64,
    "risk_free_rate": pl.Float64,
    "historical_total_dividend_in_lookback_window": pl.Float64,
    "historical_dividend_yield": pl.Float64,
}
PROJECT_EXPERIMENT_FEATURE_COLUMNS: Final = tuple(PROJECT_EXPERIMENT_FEATURE_DTYPES)
PROJECT_AS_SUBMITTED_TICKERS: Final = ("DUK", "EPD", "ETR", "MO", "SO", "VZ")
_PROJECT_QUOTE_COHORT_DTYPES: Final = {
    **ORATS_NORMALIZED_OPTION_DTYPES,
    **QUOTE_FEATURE_DTYPES,
}
PROJECT_EXPERIMENT_QUOTE_COHORT_DTYPES: Final = {
    name: PROJECT_EXPERIMENT_FEATURE_DTYPES[name]
    for name in (
        "ticker",
        "quote_date",
        "expiry_date",
        "strike",
        "underlying_price",
        "call_mid_price",
        "days_to_expiry",
        "yte",
        "call_moneyness",
    )
}
_PROJECT_FEATURE_SCHEMA_HASH_FORMAT: Final = "project-as-submitted-feature-schema-v1"
_PROJECT_FEATURE_PIPELINE_HASH_FORMAT: Final = "project-as-submitted-feature-pipeline-v1"
_PROJECT_EXPERIMENT_SCHEMA_HASH_FORMAT: Final = "project-experiment-feature-schema-v1"
_PROJECT_EXPERIMENT_PIPELINE_HASH_FORMAT: Final = "project-experiment-feature-pipeline-v1"


class ProjectFeatureError(ValueError):
    """Base class for an invalid resolved project feature artifact."""


@dataclass(slots=True)
class ProjectFeatureSchemaError(ProjectFeatureError):
    """A project input or the composed artifact has incorrect dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe output schema differences without exposing study rows."""
        return f"project feature dtype mismatches: {self.mismatches!r}"


@dataclass(slots=True)
class ProjectFeatureSpecError(ProjectFeatureError):
    """A project pipeline declaration is malformed or outside the frozen study domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid declaration without inspecting licensed rows."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(slots=True)
class ProjectFeatureUniverseError(ProjectFeatureError):
    """The normalized source lacks one or more frozen study tickers."""

    missing_tickers: tuple[str, ...]

    def __str__(self) -> str:
        """Identify absent public ticker symbols without exposing licensed rows."""
        return f"project feature source is missing study tickers: {self.missing_tickers!r}"


def project_feature_schema_sha256() -> str:
    """Hash the exact ordered project feature names and Polars dtypes."""
    return _feature_schema_sha256(
        _PROJECT_FEATURE_SCHEMA_HASH_FORMAT,
        PROJECT_AS_SUBMITTED_FEATURE_DTYPES,
    )


def project_experiment_feature_schema_sha256() -> str:
    """Hash the minimal ordered feature contract consumed by every project experiment."""
    return _feature_schema_sha256(
        _PROJECT_EXPERIMENT_SCHEMA_HASH_FORMAT,
        PROJECT_EXPERIMENT_FEATURE_DTYPES,
    )


def _feature_schema_sha256(
    format_identity: str,
    dtypes: Mapping[str, type[pl.DataType] | pl.DataType],
) -> str:
    hasher = hashlib.sha256()
    for value in (
        format_identity,
        *(item for pair in dtypes.items() for item in map(str, pair)),
    ):
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        hasher.update(encoded)
    return hasher.hexdigest()


def project_feature_pipeline_sha256() -> str:
    """Hash every immutable transformation choice in the complete project pipeline."""
    return _feature_pipeline_sha256(
        format_identity=_PROJECT_FEATURE_PIPELINE_HASH_FORMAT,
        feature_schema_sha256=project_feature_schema_sha256(),
    )


def project_experiment_feature_pipeline_sha256(
    source_precision: QuoteNumericPrecision = QuoteNumericPrecision.FLOAT32,
) -> str:
    """Hash the same financial pipeline under its minimal experiment output schema."""
    if type(source_precision) is not QuoteNumericPrecision:
        raise ProjectFeatureSchemaError(
            mismatches=(("source_precision", "QuoteNumericPrecision", repr(source_precision)),),
        )
    return _feature_pipeline_sha256(
        format_identity=_PROJECT_EXPERIMENT_PIPELINE_HASH_FORMAT,
        feature_schema_sha256=project_experiment_feature_schema_sha256(),
        quote_source_precision=source_precision.value,
    )


def _feature_pipeline_sha256(
    *,
    format_identity: str,
    feature_schema_sha256: str,
    quote_source_precision: str | None = None,
) -> str:
    quote_filter = PROJECT_AS_SUBMITTED_FILTER
    volatility = PROJECT_HISTORICAL_VOLATILITY_SPEC
    underlying_join = PROJECT_AS_SUBMITTED_UNDERLYING_JOIN_SPEC
    forecast = PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC
    historical = PROJECT_AS_SUBMITTED_HISTORICAL_DIVIDEND_SPEC
    nss = PROJECT_AS_SUBMITTED_NSS_RATE_SPEC
    payload = {
        "feature_schema_sha256": feature_schema_sha256,
        "forecast_dividends": {
            "aggregate_same_date_components": forecast.aggregate_same_date_components,
            "include_expiry_date": forecast.include_expiry_date,
            "retain_zero_amount_events": forecast.retain_zero_amount_events,
            "snapshot_coverage": forecast.snapshot_coverage,
            "year_basis_days": forecast.year_basis_days,
        },
        "format": format_identity,
        "forecast_source_selection": "exact-project-quote-dates-then-strict-future-v1",
        "historical_dividends": {
            "include_lookback_boundary": historical.include_lookback_boundary,
            "include_quote_date": historical.include_quote_date,
            "lookback_weeks": historical.lookback_weeks,
            "require_ticker_history": historical.require_ticker_history,
            "weeks_per_year": historical.weeks_per_year,
        },
        "historical_volatility": {
            "annualization_periods_per_year": volatility.annualization_periods_per_year,
            "degrees_of_freedom": volatility.degrees_of_freedom,
            "windows": list(volatility.windows),
        },
        "nss_rate": {
            "minimum_nss_maturity": nss.minimum_nss_maturity,
            "use_observed_shortest_rate": nss.use_observed_shortest_rate,
        },
        "nss_source_selection": "submitted-inner-quote-date-intersection-v1",
        "quote_filter": {
            "drop_nulls": quote_filter.drop_nulls,
            "intrinsic_ask_multiplier": quote_filter.intrinsic_ask_multiplier,
            "max_call_moneyness": quote_filter.max_call_moneyness,
            "max_days_to_expiry": quote_filter.max_days_to_expiry,
            "max_project_call_spread": quote_filter.max_project_call_spread,
            "max_put_moneyness": quote_filter.max_put_moneyness,
            "max_quote_date": (
                None
                if quote_filter.max_quote_date is None
                else quote_filter.max_quote_date.isoformat()
            ),
            "min_call_moneyness": quote_filter.min_call_moneyness,
            "min_call_price": quote_filter.min_call_price,
            "min_days_to_expiry": quote_filter.min_days_to_expiry,
            "min_put_moneyness": quote_filter.min_put_moneyness,
            "min_put_price": quote_filter.min_put_price,
            "min_quote_date": (
                None
                if quote_filter.min_quote_date is None
                else quote_filter.min_quote_date.isoformat()
            ),
        },
        "tickers": list(PROJECT_AS_SUBMITTED_TICKERS),
        "underlying_join": {
            "allow_exact_matches": underlying_join.allow_exact_matches,
            "maximum_calendar_lag_days": underlying_join.maximum_calendar_lag_days,
        },
    }
    if quote_source_precision is not None:
        payload["quote_source_precision"] = quote_source_precision
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _select_project_as_submitted_universe(frame: pl.LazyFrame) -> pl.LazyFrame:
    return _select_project_tickers(frame, PROJECT_AS_SUBMITTED_TICKERS)


def _select_project_tickers(
    frame: pl.LazyFrame,
    tickers: tuple[str, ...],
) -> pl.LazyFrame:
    if (
        type(tickers) is not tuple
        or not tickers
        or tickers != tuple(sorted(set(tickers)))
        or any(ticker not in PROJECT_AS_SUBMITTED_TICKERS for ticker in tickers)
    ):
        raise ProjectFeatureSpecError(
            field="tickers",
            observed=tickers,
            requirement=(
                f"a sorted unique nonempty tuple drawn from {PROJECT_AS_SUBMITTED_TICKERS!r}"
            ),
        )
    validated = require_columns(frame, ("ticker",))
    ticker_dtype = validated.collect_schema()["ticker"]
    if ticker_dtype != pl.String:
        raise ProjectFeatureSchemaError(
            mismatches=(("ticker", str(pl.String), str(ticker_dtype)),),
        )
    available = frozenset(
        validated.filter(pl.col("ticker").is_in(tickers))
        .select("ticker")
        .unique()
        .collect()["ticker"]
        .to_list(),
    )
    missing = tuple(ticker for ticker in tickers if ticker not in available)
    if missing:
        raise ProjectFeatureUniverseError(missing_tickers=missing)
    return validated.filter(pl.col("ticker").is_in(tickers))


def select_project_as_submitted_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Project enriched rows into the exact ordered project feature contract."""
    required = (
        *(
            name
            for name in PROJECT_AS_SUBMITTED_FEATURE_COLUMNS
            if name != "split_adjusted_underlying_price"
        ),
        "split_adjusted_close",
    )
    validated = require_columns(
        forbid_columns(frame, ("split_adjusted_underlying_price",)),
        required,
    )
    projected = validated.select(
        *(
            pl.col("split_adjusted_close").alias(name)
            if name == "split_adjusted_underlying_price"
            else pl.col(name)
            for name in PROJECT_AS_SUBMITTED_FEATURE_COLUMNS
        ),
    )
    schema = projected.collect_schema()
    mismatches = tuple(
        (name, str(dtype), str(schema[name]))
        for name, dtype in PROJECT_AS_SUBMITTED_FEATURE_DTYPES.items()
        if schema[name] != dtype
    )
    if mismatches:
        raise ProjectFeatureSchemaError(mismatches=mismatches)
    return projected


def select_project_experiment_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Project enriched rows into the minimal contract needed by every submitted experiment."""
    validated = require_columns(frame, PROJECT_EXPERIMENT_FEATURE_COLUMNS)
    projected = validated.select(PROJECT_EXPERIMENT_FEATURE_COLUMNS)
    schema = projected.collect_schema()
    mismatches = tuple(
        (name, str(dtype), str(schema[name]))
        for name, dtype in PROJECT_EXPERIMENT_FEATURE_DTYPES.items()
        if schema[name] != dtype
    )
    if mismatches:
        raise ProjectFeatureSchemaError(mismatches=mismatches)
    return projected


def build_project_quote_cohort(
    normalized_options: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build the exact quote-only cohort before all expensive support joins."""
    return _build_project_quote_cohort(
        normalized_options,
        nss_parameters,
        output_dtypes=_PROJECT_QUOTE_COHORT_DTYPES,
        required_tickers=PROJECT_AS_SUBMITTED_TICKERS,
        source_precision=QuoteNumericPrecision.FLOAT32,
    )


def build_project_experiment_quote_cohort(
    paired_options: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
    *,
    source_precision: QuoteNumericPrecision = QuoteNumericPrecision.FLOAT32,
) -> pl.LazyFrame:
    """Build the submitted cohort from only the provider fields used by experiments."""
    return _build_project_quote_cohort(
        paired_options,
        nss_parameters,
        output_dtypes=PROJECT_EXPERIMENT_QUOTE_COHORT_DTYPES,
        required_tickers=PROJECT_AS_SUBMITTED_TICKERS,
        source_precision=source_precision,
    )


def build_project_experiment_quote_partition(
    paired_options: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
    *,
    tickers: tuple[str, ...],
    source_precision: QuoteNumericPrecision,
) -> pl.LazyFrame:
    """Filter one explicit submitted-ticker partition before cross-source union."""
    return _build_project_quote_cohort(
        paired_options,
        nss_parameters,
        output_dtypes=PROJECT_EXPERIMENT_QUOTE_COHORT_DTYPES,
        required_tickers=tickers,
        source_precision=source_precision,
    )


def _build_project_quote_cohort(
    options: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
    *,
    output_dtypes: Mapping[str, type[pl.DataType] | pl.DataType],
    required_tickers: tuple[str, ...],
    source_precision: QuoteNumericPrecision,
) -> pl.LazyFrame:
    selected_options = _select_project_tickers(options, required_tickers)
    quotes = apply_quote_predicates(
        add_quote_features(selected_options, precision=source_precision),
        PROJECT_AS_SUBMITTED_FILTER,
        precision=source_precision,
    )
    quotes = quotes.join(
        nss_parameters.select(pl.col("date").alias("quote_date")).unique(),
        on="quote_date",
        how="semi",
    )
    source_schema = quotes.collect_schema()
    projected = quotes.select(
        *(
            pl.col(name) if source_schema[name] == dtype else pl.col(name).cast(dtype)
            for name, dtype in output_dtypes.items()
        )
    )
    schema = projected.collect_schema()
    mismatches = tuple(
        (name, str(dtype), str(schema[name]))
        for name, dtype in output_dtypes.items()
        if schema[name] != dtype
    )
    if mismatches:
        raise ProjectFeatureSchemaError(mismatches=mismatches)
    return projected


def enrich_project_quote_cohort(
    quotes: pl.LazyFrame,
    underlying_history: pl.LazyFrame,
    future_dividend_forecasts: pl.LazyFrame,
    historical_dividends: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
) -> pl.LazyFrame:
    """Validate support sources and enrich one already-filtered quote cohort."""
    complete = _enrich_project_quote_cohort(
        quotes,
        underlying_history,
        future_dividend_forecasts,
        historical_dividends,
        nss_parameters,
    )
    return select_project_as_submitted_features(complete)


def enrich_project_experiment_quote_cohort(
    quotes: pl.LazyFrame,
    underlying_history: pl.LazyFrame,
    future_dividend_forecasts: pl.LazyFrame,
    historical_dividends: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
) -> pl.LazyFrame:
    """Enrich a minimal paired cohort without fabricating unused vendor diagnostics."""
    complete = _enrich_project_quote_cohort(
        quotes,
        underlying_history,
        future_dividend_forecasts,
        historical_dividends,
        nss_parameters,
    )
    return select_project_experiment_features(complete)


def _enrich_project_quote_cohort(
    quotes: pl.LazyFrame,
    underlying_history: pl.LazyFrame,
    future_dividend_forecasts: pl.LazyFrame,
    historical_dividends: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
) -> pl.LazyFrame:
    relevant_forecast_dates = quotes.select(
        pl.col("quote_date").alias("forecast_date"),
    ).unique()
    relevant_forecasts = select_future_orats_dividend_forecasts(
        future_dividend_forecasts.join(
            relevant_forecast_dates,
            on="forecast_date",
            how="semi",
        )
    )
    prepared_underlying = prepare_underlying_history(
        underlying_history,
        PROJECT_HISTORICAL_VOLATILITY_SPEC,
    )
    enriched = join_underlying_history(
        quotes,
        prepared_underlying,
        PROJECT_HISTORICAL_VOLATILITY_SPEC,
        PROJECT_AS_SUBMITTED_UNDERLYING_JOIN_SPEC,
    )
    enriched = add_forecast_dividend_features(
        enriched,
        relevant_forecasts,
        PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
    )
    enriched = add_historical_dividend_features(
        enriched,
        historical_dividends,
        PROJECT_AS_SUBMITTED_HISTORICAL_DIVIDEND_SPEC,
    )
    enriched = add_nss_rate_feature(
        enriched,
        nss_parameters,
        PROJECT_AS_SUBMITTED_NSS_RATE_SPEC,
    )
    return apply_complete_row_policy(enriched, PROJECT_AS_SUBMITTED_FILTER)


def build_project_as_submitted_features(
    normalized_options: pl.LazyFrame,
    underlying_history: pl.LazyFrame,
    future_dividend_forecasts: pl.LazyFrame,
    historical_dividends: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compose the immutable project as-submitted data transformation.

    Quote-only predicates and the submitted NSS date intersection execute before support joins.
    The declared complete-row null policy executes only after every support feature is present.
    Because every join independently validates coverage and preserves quote identity, the result is
    cohort-equivalent to the submitted final cleaner while avoiding expensive work on rejected
    quotes. The returned lazy frame has one exact 42-column ordered schema and no provider-specific
    helper columns.

    All inputs are canonical typed frames; acquisition and licensed storage remain explicit
    upstream stages.

    """
    quotes = build_project_quote_cohort(normalized_options, nss_parameters)
    return enrich_project_quote_cohort(
        quotes,
        underlying_history,
        future_dividend_forecasts,
        historical_dividends,
        nss_parameters,
    )


def build_project_experiment_features(  # noqa: PLR0913 - each independent source is explicit
    paired_options: pl.LazyFrame,
    underlying_history: pl.LazyFrame,
    future_dividend_forecasts: pl.LazyFrame,
    historical_dividends: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
    *,
    source_precision: QuoteNumericPrecision = QuoteNumericPrecision.FLOAT32,
) -> pl.LazyFrame:
    """Compose the submitted study from the minimal provider-neutral paired quote contract.

    This path applies the same cohort, point-in-time support, dividend, rate, and complete-row
    policies as ``build_project_as_submitted_features``. It deliberately emits only fields consumed
    by the classical/neural experiments or their aggregate diagnostics, so a compatible long-form
    source does not need to invent open interest, vendor theoretical values, or implied volatility.
    """
    quotes = build_project_experiment_quote_cohort(
        paired_options,
        nss_parameters,
        source_precision=source_precision,
    )
    return enrich_project_experiment_quote_cohort(
        quotes,
        underlying_history,
        future_dividend_forecasts,
        historical_dividends,
        nss_parameters,
    )


__all__ = [
    "PROJECT_AS_SUBMITTED_FEATURE_COLUMNS",
    "PROJECT_AS_SUBMITTED_FEATURE_DTYPES",
    "PROJECT_AS_SUBMITTED_TICKERS",
    "PROJECT_EXPERIMENT_FEATURE_COLUMNS",
    "PROJECT_EXPERIMENT_FEATURE_DTYPES",
    "PROJECT_EXPERIMENT_QUOTE_COHORT_DTYPES",
    "ProjectFeatureError",
    "ProjectFeatureSchemaError",
    "ProjectFeatureSpecError",
    "ProjectFeatureUniverseError",
    "build_project_as_submitted_features",
    "build_project_experiment_features",
    "build_project_experiment_quote_cohort",
    "build_project_experiment_quote_partition",
    "build_project_quote_cohort",
    "enrich_project_experiment_quote_cohort",
    "enrich_project_quote_cohort",
    "project_experiment_feature_pipeline_sha256",
    "project_experiment_feature_schema_sha256",
    "project_feature_pipeline_sha256",
    "project_feature_schema_sha256",
    "select_project_as_submitted_features",
    "select_project_experiment_features",
]
