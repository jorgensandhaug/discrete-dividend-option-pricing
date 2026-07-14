"""One obvious composition for the project as-submitted feature artifact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import polars as pl

from .dividend_features import (
    DIVIDEND_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC,
    add_forecast_dividend_features,
)
from .historical_dividends import (
    HISTORICAL_DIVIDEND_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_HISTORICAL_DIVIDEND_SPEC,
    add_historical_dividend_features,
)
from .nss_rates import PROJECT_AS_SUBMITTED_NSS_RATE_SPEC, add_nss_rate_feature
from .orats_options import ORATS_NORMALIZED_OPTION_DTYPES
from .quote_filters import (
    PROJECT_AS_SUBMITTED_FILTER,
    QUOTE_FEATURE_DTYPES,
    add_quote_features,
    apply_complete_row_policy,
    apply_quote_predicates,
)
from .schema import forbid_columns, require_columns
from .underlying_history import (
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
PROJECT_AS_SUBMITTED_TICKERS: Final = ("DUK", "EPD", "ETR", "MO", "SO", "VZ")
_PROJECT_FEATURE_SCHEMA_HASH_FORMAT: Final = "project-as-submitted-feature-schema-v1"


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
class ProjectFeatureUniverseError(ProjectFeatureError):
    """The normalized source lacks one or more frozen study tickers."""

    missing_tickers: tuple[str, ...]

    def __str__(self) -> str:
        """Identify absent public ticker symbols without exposing licensed rows."""
        return f"project feature source is missing study tickers: {self.missing_tickers!r}"


def project_feature_schema_sha256() -> str:
    """Hash the exact ordered project feature names and Polars dtypes."""
    hasher = hashlib.sha256()
    for value in (
        _PROJECT_FEATURE_SCHEMA_HASH_FORMAT,
        *(item for pair in PROJECT_AS_SUBMITTED_FEATURE_DTYPES.items() for item in map(str, pair)),
    ):
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        hasher.update(encoded)
    return hasher.hexdigest()


def _select_project_as_submitted_universe(frame: pl.LazyFrame) -> pl.LazyFrame:
    validated = require_columns(frame, ("ticker",))
    ticker_dtype = validated.collect_schema()["ticker"]
    if ticker_dtype != pl.String:
        raise ProjectFeatureSchemaError(
            mismatches=(("ticker", str(pl.String), str(ticker_dtype)),),
        )
    available = frozenset(
        validated.filter(pl.col("ticker").is_in(PROJECT_AS_SUBMITTED_TICKERS))
        .select("ticker")
        .unique()
        .collect()["ticker"]
        .to_list(),
    )
    missing = tuple(ticker for ticker in PROJECT_AS_SUBMITTED_TICKERS if ticker not in available)
    if missing:
        raise ProjectFeatureUniverseError(missing_tickers=missing)
    return validated.filter(pl.col("ticker").is_in(PROJECT_AS_SUBMITTED_TICKERS))


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


def build_project_as_submitted_features(
    normalized_options: pl.LazyFrame,
    underlying_history: pl.LazyFrame,
    future_dividend_forecasts: pl.LazyFrame,
    historical_dividends: pl.LazyFrame,
    nss_parameters: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compose the immutable project as-submitted data transformation.

    Quote-only predicates execute before support joins for predicate pushdown and reduced work. The
    declared complete-row null policy executes only after every support feature is present. Because
    every join independently validates coverage and preserves quote identity, this staging is
    cohort-equivalent to the submitted final cleaner while avoiding expensive work on rejected
    quotes. The returned lazy frame has one exact 42-column ordered schema and no provider-specific
    helper columns.

    All inputs are canonical typed frames; acquisition, licensed storage, and future-event
    selection remain explicit upstream stages.

    """
    selected_options = _select_project_as_submitted_universe(normalized_options)
    quotes = apply_quote_predicates(
        add_quote_features(selected_options),
        PROJECT_AS_SUBMITTED_FILTER,
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
        future_dividend_forecasts,
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
    complete = apply_complete_row_policy(enriched, PROJECT_AS_SUBMITTED_FILTER)
    return select_project_as_submitted_features(complete)


__all__ = [
    "PROJECT_AS_SUBMITTED_FEATURE_COLUMNS",
    "PROJECT_AS_SUBMITTED_FEATURE_DTYPES",
    "PROJECT_AS_SUBMITTED_TICKERS",
    "ProjectFeatureError",
    "ProjectFeatureSchemaError",
    "ProjectFeatureUniverseError",
    "build_project_as_submitted_features",
    "project_feature_schema_sha256",
    "select_project_as_submitted_features",
]
