"""Strict preparation of daily underlying-price histories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import polars as pl

from .schema import require_columns

if TYPE_CHECKING:
    from collections.abc import Mapping

UNDERLYING_HISTORY_DTYPES = {
    "ticker": pl.String,
    "date": pl.Date,
    "split_adjusted_close": pl.Float64,
    "dividend_and_split_adjusted_close": pl.Float64,
}


class UnderlyingHistoryError(ValueError):
    """Base class for invalid underlying history or estimator specifications."""


@dataclass(frozen=True, slots=True)
class UnderlyingHistorySchemaError(UnderlyingHistoryError):
    """Required underlying-history columns have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed dtypes without exposing market rows."""
        return f"underlying-history column dtype mismatches: {self.mismatches!r}"


@dataclass(frozen=True, slots=True)
class UnderlyingHistoryContentError(UnderlyingHistoryError):
    """Underlying-history rows violate the canonical daily-series contract."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing provider values."""
        return f"invalid underlying-history content: {self.violations!r}"


@dataclass(frozen=True, slots=True)
class HistoricalVolatilitySpecError(UnderlyingHistoryError):
    """A historical-volatility specification is outside its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid estimator value."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(frozen=True, slots=True)
class HistoricalVolatilitySpec:
    """Identity of complete-window sample volatility on daily adjusted log returns."""

    windows: tuple[int, ...]
    annualization_periods_per_year: float
    degrees_of_freedom: int

    def __post_init__(self) -> None:
        """Reject coercions, duplicate windows, and undefined sample estimators."""
        if type(self.windows) is not tuple or not self.windows:
            raise HistoricalVolatilitySpecError(
                field="windows",
                observed=self.windows,
                requirement="a nonempty tuple of strictly increasing plain positive ints",
            )
        if any(type(window) is not int or window <= 0 for window in self.windows):
            raise HistoricalVolatilitySpecError(
                field="windows",
                observed=self.windows,
                requirement="a nonempty tuple of strictly increasing plain positive ints",
            )
        if any(right <= left for left, right in pairwise(self.windows)):
            raise HistoricalVolatilitySpecError(
                field="windows",
                observed=self.windows,
                requirement="a nonempty tuple of strictly increasing plain positive ints",
            )
        if (
            type(self.annualization_periods_per_year) is not float
            or not math.isfinite(self.annualization_periods_per_year)
            or self.annualization_periods_per_year <= 0.0
        ):
            raise HistoricalVolatilitySpecError(
                field="annualization_periods_per_year",
                observed=self.annualization_periods_per_year,
                requirement="a finite positive plain float",
            )
        if (
            type(self.degrees_of_freedom) is not int
            or self.degrees_of_freedom < 0
            or self.degrees_of_freedom >= self.windows[0]
        ):
            raise HistoricalVolatilitySpecError(
                field="degrees_of_freedom",
                observed=self.degrees_of_freedom,
                requirement="a plain nonnegative int below every window",
            )


PROJECT_HISTORICAL_VOLATILITY_SPEC: Final = HistoricalVolatilitySpec(
    windows=(30, 90, 126, 252),
    annualization_periods_per_year=252.0,
    degrees_of_freedom=1,
)


def _validate_dtypes(
    frame: pl.LazyFrame,
    expected: Mapping[str, type[pl.DataType]],
) -> pl.LazyFrame:
    validated = require_columns(frame, tuple(expected))
    schema = validated.collect_schema()
    mismatches = tuple(
        (name, str(dtype), str(schema[name]))
        for name, dtype in expected.items()
        if schema[name] != dtype
    )
    if mismatches:
        raise UnderlyingHistorySchemaError(mismatches=mismatches)
    return validated


def _content_violations(frame: pl.LazyFrame) -> tuple[tuple[str, int], ...]:
    diagnostics = (
        frame.select(
            pl.len().alias("rows"),
            pl.col("ticker").is_null().sum().alias("null_tickers"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_tickers"),
            pl.col("date").is_null().sum().alias("null_dates"),
            pl.struct("ticker", "date").is_duplicated().sum().alias("duplicate_keys"),
            pl.col("split_adjusted_close").is_null().sum().alias("null_split_close"),
            (~pl.col("split_adjusted_close").is_finite())
            .fill_null(value=False)
            .sum()
            .alias("nonfinite_split_close"),
            (pl.col("split_adjusted_close") <= 0.0)
            .fill_null(value=False)
            .sum()
            .alias("nonpositive_split_close"),
            pl.col("dividend_and_split_adjusted_close")
            .is_null()
            .sum()
            .alias("null_total_return_close"),
            (~pl.col("dividend_and_split_adjusted_close").is_finite())
            .fill_null(value=False)
            .sum()
            .alias("nonfinite_total_return_close"),
            (pl.col("dividend_and_split_adjusted_close") <= 0.0)
            .fill_null(value=False)
            .sum()
            .alias("nonpositive_total_return_close"),
        )
        .collect()
        .row(0, named=True)
    )
    if diagnostics["rows"] == 0:
        return (("empty_input", 1),)
    return tuple(
        (name, value)
        for name, value in diagnostics.items()
        if name != "rows" and type(value) is int and value != 0
    )


def prepare_underlying_history(
    frame: pl.LazyFrame,
    spec: HistoricalVolatilitySpec,
) -> pl.LazyFrame:
    """Validate, canonicalize, and add complete-window annualized volatility columns.

    Returns exactly the four canonical input columns followed by one `Float64`
    `historical_volatility_{window}` column per requested window. Rows are ordered by ticker and
    observation date because provider delivery order has no financial meaning. Volatility is the
    sample standard deviation of `log(P_t / P_{t-1})` from dividend-and-split-adjusted closes,
    multiplied by the square root of the declared annual observation count. A value is emitted only
    after a complete window of non-null returns exists.

    Raises:
        MissingColumnsError: If a required source column is absent.
        UnderlyingHistorySchemaError: If a required source column has the wrong dtype.
        UnderlyingHistoryContentError: If keys or prices are empty, duplicated, or invalid.

    """
    validated = _validate_dtypes(frame, UNDERLYING_HISTORY_DTYPES).select(
        tuple(UNDERLYING_HISTORY_DTYPES),
    )
    violations = _content_violations(validated)
    if violations:
        raise UnderlyingHistoryContentError(violations=violations)

    canonical = validated.sort("ticker", "date")
    total_return_close = pl.col("dividend_and_split_adjusted_close")
    log_return = (total_return_close / total_return_close.shift(1)).log()
    annualizer = pl.lit(spec.annualization_periods_per_year, dtype=pl.Float64).sqrt()
    return canonical.with_columns(
        (
            log_return.rolling_std(
                window_size=window,
                min_samples=window,
                ddof=spec.degrees_of_freedom,
            )
            * annualizer
        )
        .over("ticker")
        .alias(f"historical_volatility_{window}")
        for window in spec.windows
    )


__all__ = [
    "PROJECT_HISTORICAL_VOLATILITY_SPEC",
    "UNDERLYING_HISTORY_DTYPES",
    "HistoricalVolatilitySpec",
    "HistoricalVolatilitySpecError",
    "UnderlyingHistoryContentError",
    "UnderlyingHistoryError",
    "UnderlyingHistorySchemaError",
    "prepare_underlying_history",
]
