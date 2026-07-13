"""Strict preparation of daily underlying-price histories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import polars as pl

from .schema import forbid_columns, require_columns

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
class UnderlyingHistoryJoinSchemaError(UnderlyingHistoryError):
    """Quote or prepared-history join columns have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed join dtypes without exposing market rows."""
        return f"underlying-history join dtype mismatches: {self.mismatches!r}"


@dataclass(frozen=True, slots=True)
class UnderlyingHistoryJoinError(UnderlyingHistoryError):
    """A backward underlying-history join violates its declared content contract."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate join violations without exposing provider values."""
        return f"invalid underlying-history join: {self.violations!r}"


@dataclass(frozen=True, slots=True)
class UnderlyingHistoryJoinSpecError(UnderlyingHistoryError):
    """An underlying-history as-of join value is outside its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid join specification value."""
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


@dataclass(frozen=True, slots=True)
class UnderlyingHistoryJoinSpec:
    """Identity of a backward point-in-time underlying-history join."""

    allow_exact_matches: bool
    maximum_calendar_lag_days: int | None

    def __post_init__(self) -> None:
        """Reject coerced booleans, negative lags, and an impossible strict-zero lag."""
        if type(self.allow_exact_matches) is not bool:
            raise UnderlyingHistoryJoinSpecError(
                field="allow_exact_matches",
                observed=self.allow_exact_matches,
                requirement="a plain bool",
            )
        maximum_lag = self.maximum_calendar_lag_days
        if maximum_lag is not None and (type(maximum_lag) is not int or maximum_lag < 0):
            raise UnderlyingHistoryJoinSpecError(
                field="maximum_calendar_lag_days",
                observed=maximum_lag,
                requirement="None or a plain nonnegative int",
            )
        if not self.allow_exact_matches and maximum_lag == 0:
            raise UnderlyingHistoryJoinSpecError(
                field="maximum_calendar_lag_days",
                observed=maximum_lag,
                requirement="positive or None when exact matches are disabled",
            )


PROJECT_HISTORICAL_VOLATILITY_SPEC: Final = HistoricalVolatilitySpec(
    windows=(30, 90, 126, 252),
    annualization_periods_per_year=252.0,
    degrees_of_freedom=1,
)
PROJECT_AS_SUBMITTED_UNDERLYING_JOIN_SPEC: Final = UnderlyingHistoryJoinSpec(
    allow_exact_matches=True,
    maximum_calendar_lag_days=None,
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


def _validate_join_dtypes(
    quotes: pl.LazyFrame,
    history: pl.LazyFrame,
    volatility_spec: HistoricalVolatilitySpec,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    support_columns = (
        "underlying_observation_date",
        "underlying_observation_lag_days",
        "split_adjusted_close",
        "dividend_and_split_adjusted_close",
        *(f"historical_volatility_{window}" for window in volatility_spec.windows),
    )
    validated_quotes = require_columns(
        forbid_columns(quotes, support_columns),
        ("ticker", "quote_date"),
    )
    quote_schema = validated_quotes.collect_schema()
    expected_history = {
        **UNDERLYING_HISTORY_DTYPES,
        **{f"historical_volatility_{window}": pl.Float64 for window in volatility_spec.windows},
    }
    validated_history = require_columns(history, tuple(expected_history))
    history_schema = validated_history.collect_schema()
    mismatches = tuple(
        (
            f"quotes.{name}",
            str(expected),
            str(quote_schema[name]),
        )
        for name, expected in {"ticker": pl.String, "quote_date": pl.Date}.items()
        if quote_schema[name] != expected
    ) + tuple(
        (
            f"history.{name}",
            str(expected),
            str(history_schema[name]),
        )
        for name, expected in expected_history.items()
        if history_schema[name] != expected
    )
    if mismatches:
        raise UnderlyingHistoryJoinSchemaError(mismatches=mismatches)
    return validated_quotes, validated_history.select(tuple(expected_history))


def _join_input_violations(
    quotes: pl.LazyFrame,
    history: pl.LazyFrame,
) -> tuple[tuple[str, int], ...]:
    quote_diagnostics = (
        quotes.select(
            pl.len().alias("quote_rows"),
            pl.col("ticker").is_null().sum().alias("null_quote_tickers"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_quote_tickers"),
            pl.col("quote_date").is_null().sum().alias("null_quote_dates"),
            (pl.col("quote_date").diff().over("ticker").dt.total_days() < 0)
            .fill_null(value=False)
            .sum()
            .alias("out_of_order_quote_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    history_diagnostics = (
        history.select(
            pl.len().alias("history_rows"),
            pl.col("ticker").is_null().sum().alias("null_history_tickers"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_history_tickers"),
            pl.col("date").is_null().sum().alias("null_history_dates"),
            pl.struct("ticker", "date").is_duplicated().sum().alias("duplicate_history_keys"),
            (pl.col("date").diff().over("ticker").dt.total_days() < 0)
            .fill_null(value=False)
            .sum()
            .alias("out_of_order_history_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    combined = {**quote_diagnostics, **history_diagnostics}
    violations: list[tuple[str, int]] = []
    if combined["quote_rows"] == 0:
        violations.append(("empty_quotes", 1))
    if combined["history_rows"] == 0:
        violations.append(("empty_history", 1))
    violations.extend(
        (name, value)
        for name, value in combined.items()
        if name not in {"quote_rows", "history_rows"} and type(value) is int and value != 0
    )
    return tuple(violations)


def _support_date_frame(history: pl.LazyFrame) -> pl.LazyFrame:
    return history.select(
        "ticker",
        pl.col("date").alias("underlying_observation_date"),
    )


def _validate_join_coverage(
    quotes: pl.LazyFrame,
    history: pl.LazyFrame,
    join_spec: UnderlyingHistoryJoinSpec,
) -> None:
    coverage = (
        quotes.select("ticker", "quote_date")
        .unique()
        .sort("ticker", "quote_date")
        .join_asof(
            _support_date_frame(history),
            left_on="quote_date",
            right_on="underlying_observation_date",
            by="ticker",
            strategy="backward",
            allow_exact_matches=join_spec.allow_exact_matches,
            check_sortedness=False,
        )
        .with_columns(
            (pl.col("quote_date") - pl.col("underlying_observation_date"))
            .dt.total_days()
            .alias("underlying_observation_lag_days"),
        )
        .select(
            pl.col("underlying_observation_date").is_null().sum().alias("unsupported_quote_dates"),
            (pl.col("underlying_observation_lag_days") < 0)
            .fill_null(value=False)
            .sum()
            .alias("forward_looking_matches"),
            pl.col("underlying_observation_lag_days").max().alias("maximum_observed_lag"),
        )
        .collect()
        .row(0, named=True)
    )
    violations = [
        (name, value)
        for name, value in coverage.items()
        if name != "maximum_observed_lag" and type(value) is int and value != 0
    ]
    maximum_lag = coverage["maximum_observed_lag"]
    allowed_lag = join_spec.maximum_calendar_lag_days
    if allowed_lag is not None and type(maximum_lag) is int and maximum_lag > allowed_lag:
        violations.append(("maximum_observed_lag", maximum_lag))
    if violations:
        raise UnderlyingHistoryJoinError(violations=tuple(violations))


def join_underlying_history(
    quotes: pl.LazyFrame,
    history: pl.LazyFrame,
    volatility_spec: HistoricalVolatilitySpec,
    join_spec: UnderlyingHistoryJoinSpec,
) -> pl.LazyFrame:
    """Join prepared history backward without changing quote order or hiding staleness.

    The quote and history inputs must already be nondecreasing by date within each ticker. The
    function validates that invariant because Polars cannot verify grouped as-of ordering itself.
    It preflights every unique ticker/date for backward support and any declared maximum civil-day
    lag, then records the matched observation date and lag beside the requested price/volatility
    columns. `None` is an explicit unlimited-lag identity, not an inferred default.

    Raises:
        MissingColumnsError: If a required quote or prepared-history column is absent.
        UnderlyingHistoryJoinSchemaError: If a required join column has the wrong dtype.
        UnderlyingHistoryJoinError: If inputs are empty, invalid, unordered, or unsupported.

    """
    validated_quotes, validated_history = _validate_join_dtypes(
        quotes,
        history,
        volatility_spec,
    )
    violations = _join_input_violations(validated_quotes, validated_history)
    if violations:
        raise UnderlyingHistoryJoinError(violations=violations)
    _validate_join_coverage(validated_quotes, validated_history, join_spec)

    support = validated_history.rename({"date": "underlying_observation_date"})
    return validated_quotes.join_asof(
        support,
        left_on="quote_date",
        right_on="underlying_observation_date",
        by="ticker",
        strategy="backward",
        allow_exact_matches=join_spec.allow_exact_matches,
        check_sortedness=False,
    ).with_columns(
        (pl.col("quote_date") - pl.col("underlying_observation_date"))
        .dt.total_days()
        .alias("underlying_observation_lag_days"),
    )


__all__ = [
    "PROJECT_AS_SUBMITTED_UNDERLYING_JOIN_SPEC",
    "PROJECT_HISTORICAL_VOLATILITY_SPEC",
    "UNDERLYING_HISTORY_DTYPES",
    "HistoricalVolatilitySpec",
    "HistoricalVolatilitySpecError",
    "UnderlyingHistoryContentError",
    "UnderlyingHistoryError",
    "UnderlyingHistoryJoinError",
    "UnderlyingHistoryJoinSchemaError",
    "UnderlyingHistoryJoinSpec",
    "UnderlyingHistoryJoinSpecError",
    "UnderlyingHistorySchemaError",
    "join_underlying_history",
    "prepare_underlying_history",
]
