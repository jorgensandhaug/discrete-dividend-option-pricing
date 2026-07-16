"""Explicit per-option features from point-in-time cash-dividend forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import polars as pl

from discrete_dividend_study.core.schema import forbid_columns, require_columns
from discrete_dividend_study.data.providers.orats.dividends import (
    ORATS_DIVIDEND_DTYPES,
    ORATS_DIVIDEND_FREQUENCIES,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

DIVIDEND_FEATURE_DTYPES = {
    "forecasted_total_dividend_in_period": pl.Float64,
    "forecasted_discrete_ex_dates": pl.List(pl.Date),
    "forecasted_discrete_amounts": pl.List(pl.Float64),
    "forecasted_years_to_discrete_ex_dates": pl.List(pl.Float64),
    "forecasted_dividend_count": pl.UInt32,
    "forecasted_dividend_yield": pl.Float64,
}
_QUOTE_DIVIDEND_INPUT_DTYPES = {
    "ticker": pl.String,
    "quote_date": pl.Date,
    "expiry_date": pl.Date,
    "underlying_price": pl.Float32,
}


class DividendFeatureError(ValueError):
    """Base class for invalid dividend-feature specifications or inputs."""


@dataclass(slots=True)
class DividendFeatureSchemaError(DividendFeatureError):
    """Quote or forecast feature inputs have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed dtypes without exposing market rows."""
        return f"dividend-feature input dtype mismatches: {self.mismatches!r}"


@dataclass(slots=True)
class DividendFeatureContentError(DividendFeatureError):
    """Quote or forecast rows violate the dividend-feature contract."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing licensed rows."""
        return f"invalid dividend-feature content: {self.violations!r}"


@dataclass(slots=True)
class DividendFeatureSpecError(DividendFeatureError):
    """A dividend-feature specification is outside its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid specification value."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(frozen=True, slots=True)
class ForecastDividendFeatureSpec:
    """Identity of forecast-event selection and the submitted scalar-yield proxy."""

    snapshot_coverage: Literal["forecast-date", "ticker-date"]
    include_expiry_date: bool
    retain_zero_amount_events: bool
    aggregate_same_date_components: bool
    year_basis_days: float

    def __post_init__(self) -> None:
        """Reject coerced policy flags and undefined year bases."""
        if self.snapshot_coverage not in ("forecast-date", "ticker-date"):
            raise DividendFeatureSpecError(
                field="snapshot_coverage",
                observed=self.snapshot_coverage,
                requirement="'forecast-date' or 'ticker-date'",
            )
        for field in (
            "include_expiry_date",
            "retain_zero_amount_events",
            "aggregate_same_date_components",
        ):
            value = getattr(self, field)
            if type(value) is not bool:
                raise DividendFeatureSpecError(
                    field=field,
                    observed=value,
                    requirement="a plain bool",
                )
        if (
            type(self.year_basis_days) is not float
            or not math.isfinite(self.year_basis_days)
            or self.year_basis_days <= 0.0
        ):
            raise DividendFeatureSpecError(
                field="year_basis_days",
                observed=self.year_basis_days,
                requirement="a finite positive plain float",
            )


PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC: Final = ForecastDividendFeatureSpec(
    snapshot_coverage="forecast-date",
    include_expiry_date=True,
    retain_zero_amount_events=True,
    aggregate_same_date_components=False,
    year_basis_days=365.0,
)


def _validate_dtypes(
    frame: pl.LazyFrame,
    expected: Mapping[str, type[pl.DataType] | pl.DataType],
    *,
    prefix: str,
) -> pl.LazyFrame:
    validated = require_columns(frame, tuple(expected))
    schema = validated.collect_schema()
    mismatches = tuple(
        (f"{prefix}.{name}", str(dtype), str(schema[name]))
        for name, dtype in expected.items()
        if schema[name] != dtype
    )
    if mismatches:
        raise DividendFeatureSchemaError(mismatches=mismatches)
    return validated


def _input_violations(
    quotes: pl.LazyFrame,
    forecasts: pl.LazyFrame,
) -> tuple[tuple[str, int], ...]:
    quote_diagnostics = (
        quotes.select(
            pl.len().alias("quote_rows"),
            pl.any_horizontal(
                pl.col("ticker").is_null(),
                pl.col("quote_date").is_null(),
                pl.col("expiry_date").is_null(),
                pl.col("underlying_price").is_null(),
            )
            .sum()
            .alias("quote_rows_with_nulls"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_quote_tickers"),
            (~pl.col("underlying_price").is_finite())
            .fill_null(value=False)
            .sum()
            .alias("nonfinite_underlying_prices"),
            (pl.col("underlying_price") <= 0.0)
            .fill_null(value=False)
            .sum()
            .alias("nonpositive_underlying_prices"),
            (pl.col("expiry_date") <= pl.col("quote_date"))
            .fill_null(value=False)
            .sum()
            .alias("nonpositive_option_horizons"),
        )
        .collect()
        .row(0, named=True)
    )
    forecast_diagnostics = (
        forecasts.select(
            pl.len().alias("forecast_rows"),
            pl.any_horizontal(pl.all().is_null()).sum().alias("forecast_rows_with_nulls"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_forecast_tickers"),
            (~pl.col("amount").is_finite())
            .fill_null(value=False)
            .sum()
            .alias("nonfinite_forecast_amounts"),
            (pl.col("amount") < 0.0)
            .fill_null(value=False)
            .sum()
            .alias("negative_forecast_amounts"),
            (~pl.col("frequency_per_year").is_in(ORATS_DIVIDEND_FREQUENCIES))
            .fill_null(value=False)
            .sum()
            .alias("unsupported_forecast_frequencies"),
            (pl.col("ex_date") <= pl.col("forecast_date"))
            .fill_null(value=False)
            .sum()
            .alias("nonfuture_forecast_rows"),
            pl.struct(tuple(ORATS_DIVIDEND_DTYPES))
            .is_duplicated()
            .sum()
            .alias("duplicate_forecast_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    combined = {**quote_diagnostics, **forecast_diagnostics}
    violations: list[tuple[str, int]] = []
    if combined["quote_rows"] == 0:
        violations.append(("empty_quotes", 1))
    if combined["forecast_rows"] == 0:
        violations.append(("empty_forecasts", 1))
    violations.extend(
        (name, value)
        for name, value in combined.items()
        if name not in {"quote_rows", "forecast_rows"} and type(value) is int and value != 0
    )
    return tuple(violations)


def _validate_snapshot_coverage(
    quotes: pl.LazyFrame,
    forecasts: pl.LazyFrame,
    spec: ForecastDividendFeatureSpec,
) -> None:
    if spec.snapshot_coverage == "ticker-date":
        keys = ("ticker", "quote_date")
        required = quotes.select(*keys).unique()
        available = forecasts.select(
            "ticker",
            pl.col("forecast_date").alias("quote_date"),
        ).unique()
        violation = "missing_ticker_date_forecasts"
    else:
        keys = ("quote_date",)
        required = quotes.select(*keys).unique()
        available = forecasts.select(
            pl.col("forecast_date").alias("quote_date"),
        ).unique()
        violation = "missing_forecast_dates"
    missing = required.join(available, on=keys, how="anti").select(pl.len()).collect().item()
    if type(missing) is int and missing != 0:
        raise DividendFeatureContentError(violations=((violation, missing),))


def _cash_components(
    forecasts: pl.LazyFrame,
    spec: ForecastDividendFeatureSpec,
) -> pl.LazyFrame:
    selected = forecasts
    if not spec.retain_zero_amount_events:
        selected = selected.filter(pl.col("amount") > 0.0)
    if spec.aggregate_same_date_components:
        return selected.group_by("ticker", "forecast_date", "ex_date").agg(
            pl.col("amount").sum().alias("amount"),
        )
    return selected.select("ticker", "forecast_date", "ex_date", "amount")


def _event_schedules(
    quotes: pl.LazyFrame,
    forecasts: pl.LazyFrame,
    spec: ForecastDividendFeatureSpec,
) -> pl.LazyFrame:
    horizons = quotes.select("ticker", "quote_date", "expiry_date").unique()
    joined = horizons.join(
        _cash_components(forecasts, spec),
        left_on=("ticker", "quote_date"),
        right_on=("ticker", "forecast_date"),
        how="inner",
        validate="m:m",
        coalesce=True,
    )
    expiry_predicate = (
        pl.col("ex_date") <= pl.col("expiry_date")
        if spec.include_expiry_date
        else pl.col("ex_date") < pl.col("expiry_date")
    )
    return (
        joined.filter(expiry_predicate)
        .sort("ticker", "quote_date", "expiry_date", "ex_date", "amount")
        .group_by("ticker", "quote_date", "expiry_date", maintain_order=True)
        .agg(
            pl.col("amount").sum().alias("forecasted_total_dividend_in_period"),
            pl.col("ex_date").alias("forecasted_discrete_ex_dates"),
            pl.col("amount").alias("forecasted_discrete_amounts"),
            (
                (pl.col("ex_date") - pl.col("quote_date")).dt.total_days()
                / pl.lit(spec.year_basis_days, dtype=pl.Float64)
            ).alias("forecasted_years_to_discrete_ex_dates"),
            pl.len().alias("forecasted_dividend_count"),
        )
    )


def add_forecast_dividend_features(
    quotes: pl.LazyFrame,
    forecasts: pl.LazyFrame,
    spec: ForecastDividendFeatureSpec,
) -> pl.LazyFrame:
    """Attach point-in-time forecast schedules and the submitted scalar-yield proxy.

    Quotes must already have positive horizons and positive finite ORATS spot prices. Forecast rows
    must already be the strict future set. Snapshot coverage follows the explicit specification:
    either every quote ticker/date is required, or every quote date is required while ticker
    absence inside that verified sparse event snapshot denotes an empty schedule. Schedules are
    computed once per unique ticker/quote/expiry horizon and joined many-to-one back to quotes.
    Output order follows quote order.

    The scalar `forecasted_dividend_yield` is named as the submitted feature and is exactly
    `log(1 + total undiscounted forecast cash / spot) / (calendar DTE / year_basis_days)`. It is not
    used to alter the discrete cash-event schedule produced beside it.

    Raises:
        MissingColumnsError: If a required quote or forecast column is absent.
        ConflictingColumnsError: If a quote already contains a stage-owned output name.
        DividendFeatureSchemaError: If a required input column has the wrong dtype.
        DividendFeatureContentError: If rows are empty, invalid, nonfuture, duplicated, or lack a
            matching ticker/date snapshot.

    """
    validated_quotes = _validate_dtypes(
        require_columns(
            forbid_columns(quotes, tuple(DIVIDEND_FEATURE_DTYPES)),
            tuple(_QUOTE_DIVIDEND_INPUT_DTYPES),
        ),
        _QUOTE_DIVIDEND_INPUT_DTYPES,
        prefix="quotes",
    )
    validated_forecasts = _validate_dtypes(
        forecasts,
        ORATS_DIVIDEND_DTYPES,
        prefix="forecasts",
    ).select(tuple(ORATS_DIVIDEND_DTYPES))
    violations = _input_violations(validated_quotes, validated_forecasts)
    if violations:
        raise DividendFeatureContentError(violations=violations)
    _validate_snapshot_coverage(validated_quotes, validated_forecasts, spec)

    schedules = _event_schedules(validated_quotes, validated_forecasts, spec)
    joined = validated_quotes.join(
        schedules,
        on=("ticker", "quote_date", "expiry_date"),
        how="left",
        validate="m:1",
        coalesce=True,
        maintain_order="left",
    ).with_columns(
        pl.col("forecasted_total_dividend_in_period").fill_null(value=0.0),
        pl.col("forecasted_discrete_ex_dates").fill_null(
            value=pl.lit([], dtype=pl.List(pl.Date)),
        ),
        pl.col("forecasted_discrete_amounts").fill_null(
            value=pl.lit([], dtype=pl.List(pl.Float64)),
        ),
        pl.col("forecasted_years_to_discrete_ex_dates").fill_null(
            value=pl.lit([], dtype=pl.List(pl.Float64)),
        ),
        pl.col("forecasted_dividend_count").fill_null(value=0),
    )
    horizon = (pl.col("expiry_date") - pl.col("quote_date")).dt.total_days() / pl.lit(
        spec.year_basis_days, dtype=pl.Float64
    )
    cash_to_spot = pl.col("forecasted_total_dividend_in_period") / pl.col(
        "underlying_price",
    ).cast(pl.Float64)
    return joined.with_columns(
        ((1.0 + cash_to_spot).log() / horizon).alias("forecasted_dividend_yield"),
    )


__all__ = [
    "DIVIDEND_FEATURE_DTYPES",
    "PROJECT_AS_SUBMITTED_FORECAST_DIVIDEND_SPEC",
    "DividendFeatureContentError",
    "DividendFeatureError",
    "DividendFeatureSchemaError",
    "DividendFeatureSpecError",
    "ForecastDividendFeatureSpec",
    "add_forecast_dividend_features",
]
