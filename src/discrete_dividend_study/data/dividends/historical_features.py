"""Efficient trailing cash-dividend features from canonical history."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from discrete_dividend_study.core.schema import forbid_columns, require_columns

if TYPE_CHECKING:
    from collections.abc import Mapping

HISTORICAL_DIVIDEND_DTYPES = {
    "ticker": pl.String,
    "ex_date": pl.Date,
    "amount": pl.Float64,
}
HISTORICAL_DIVIDEND_FEATURE_DTYPES = {
    "historical_total_dividend_in_lookback_window": pl.Float64,
    "historical_dividend_yield": pl.Float64,
}
_HISTORICAL_QUOTE_DTYPES = {
    "ticker": pl.String,
    "quote_date": pl.Date,
    "split_adjusted_close": pl.Float64,
}


class HistoricalDividendError(ValueError):
    """Base class for invalid historical-dividend feature inputs."""


@dataclass(slots=True)
class HistoricalDividendSchemaError(HistoricalDividendError):
    """Quote or dividend-history inputs have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed dtypes without exposing rows."""
        return f"historical-dividend input dtype mismatches: {self.mismatches!r}"


@dataclass(slots=True)
class HistoricalDividendContentError(HistoricalDividendError):
    """Quote or dividend-history rows violate their declared contract."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing market rows."""
        return f"invalid historical-dividend content: {self.violations!r}"


@dataclass(slots=True)
class HistoricalDividendSpecError(HistoricalDividendError):
    """A historical-dividend specification is outside its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid specification value."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(frozen=True, slots=True)
class HistoricalDividendFeatureSpec:
    """Identity of a trailing observed-cash window and submitted annualized feature."""

    lookback_weeks: int
    weeks_per_year: float
    include_quote_date: bool
    include_lookback_boundary: bool
    require_ticker_history: bool

    def __post_init__(self) -> None:
        """Reject coerced bounds, invalid annualizers, and inferred coverage policies."""
        if type(self.lookback_weeks) is not int or self.lookback_weeks <= 0:
            raise HistoricalDividendSpecError(
                field="lookback_weeks",
                observed=self.lookback_weeks,
                requirement="a plain positive int",
            )
        if (
            type(self.weeks_per_year) is not float
            or not math.isfinite(self.weeks_per_year)
            or self.weeks_per_year <= 0.0
        ):
            raise HistoricalDividendSpecError(
                field="weeks_per_year",
                observed=self.weeks_per_year,
                requirement="a finite positive plain float",
            )
        for field in (
            "include_quote_date",
            "include_lookback_boundary",
            "require_ticker_history",
        ):
            value = getattr(self, field)
            if type(value) is not bool:
                raise HistoricalDividendSpecError(
                    field=field,
                    observed=value,
                    requirement="a plain bool",
                )


PROJECT_AS_SUBMITTED_HISTORICAL_DIVIDEND_SPEC: Final = HistoricalDividendFeatureSpec(
    lookback_weeks=52,
    weeks_per_year=52.0,
    include_quote_date=True,
    include_lookback_boundary=True,
    require_ticker_history=True,
)


def _validate_dtypes(
    frame: pl.LazyFrame,
    expected: Mapping[str, type[pl.DataType]],
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
        raise HistoricalDividendSchemaError(mismatches=mismatches)
    return validated


def _input_violations(
    quotes: pl.LazyFrame,
    history: pl.LazyFrame,
) -> tuple[tuple[str, int], ...]:
    quote_diagnostics = (
        quotes.select(
            pl.len().alias("quote_rows"),
            pl.any_horizontal(
                pl.col("ticker").is_null(),
                pl.col("quote_date").is_null(),
                pl.col("split_adjusted_close").is_null(),
            )
            .sum()
            .alias("quote_rows_with_nulls"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_quote_tickers"),
            (~pl.col("split_adjusted_close").is_finite())
            .fill_null(value=False)
            .sum()
            .alias("nonfinite_split_adjusted_closes"),
            (pl.col("split_adjusted_close") <= 0.0)
            .fill_null(value=False)
            .sum()
            .alias("nonpositive_split_adjusted_closes"),
        )
        .collect()
        .row(0, named=True)
    )
    history_diagnostics = (
        history.select(
            pl.len().alias("history_rows"),
            pl.any_horizontal(pl.all().is_null()).sum().alias("history_rows_with_nulls"),
            pl.col("ticker")
            .str.strip_chars()
            .eq("")
            .fill_null(value=False)
            .sum()
            .alias("blank_history_tickers"),
            (~pl.col("amount").is_finite())
            .fill_null(value=False)
            .sum()
            .alias("nonfinite_history_amounts"),
            (pl.col("amount") < 0.0).fill_null(value=False).sum().alias("negative_history_amounts"),
            pl.struct(tuple(HISTORICAL_DIVIDEND_DTYPES))
            .is_duplicated()
            .sum()
            .alias("duplicate_history_rows"),
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


def _validate_ticker_coverage(
    quotes: pl.LazyFrame,
    history: pl.LazyFrame,
    spec: HistoricalDividendFeatureSpec,
) -> None:
    if not spec.require_ticker_history:
        return
    missing = (
        quotes.select("ticker")
        .unique()
        .join(history.select("ticker").unique(), on="ticker", how="anti")
        .select(pl.len())
        .collect()
        .item()
    )
    if type(missing) is int and missing != 0:
        raise HistoricalDividendContentError(violations=(("missing_ticker_histories", missing),))


def _daily_cumulative_history(history: pl.LazyFrame) -> pl.LazyFrame:
    return (
        history.group_by("ticker", "ex_date")
        .agg(pl.col("amount").sum().alias("amount"))
        .sort("ticker", "ex_date")
        .with_columns(
            pl.col("amount").cum_sum().over("ticker").alias("_cumulative_dividend"),
        )
    )


def _trailing_totals(
    quotes: pl.LazyFrame,
    history: pl.LazyFrame,
    spec: HistoricalDividendFeatureSpec,
) -> pl.LazyFrame:
    keys = quotes.select("ticker", "quote_date").unique().sort("ticker", "quote_date")
    cumulative = _daily_cumulative_history(history)
    with_upper = keys.join_asof(
        cumulative.select(
            "ticker",
            pl.col("ex_date").alias("_upper_ex_date"),
            pl.col("_cumulative_dividend").alias("_upper_cumulative"),
        ),
        left_on="quote_date",
        right_on="_upper_ex_date",
        by="ticker",
        strategy="backward",
        allow_exact_matches=spec.include_quote_date,
        check_sortedness=False,
    ).with_columns(
        (pl.col("quote_date") - pl.duration(weeks=spec.lookback_weeks)).alias(
            "_lookback_start",
        ),
    )
    with_lower = with_upper.join_asof(
        cumulative.select(
            "ticker",
            pl.col("ex_date").alias("_lower_ex_date"),
            pl.col("_cumulative_dividend").alias("_lower_cumulative"),
        ),
        left_on="_lookback_start",
        right_on="_lower_ex_date",
        by="ticker",
        strategy="backward",
        allow_exact_matches=not spec.include_lookback_boundary,
        check_sortedness=False,
    )
    return with_lower.select(
        "ticker",
        "quote_date",
        (
            pl.col("_upper_cumulative").fill_null(value=0.0)
            - pl.col("_lower_cumulative").fill_null(value=0.0)
        ).alias("historical_total_dividend_in_lookback_window"),
    )


def add_historical_dividend_features(
    quotes: pl.LazyFrame,
    history: pl.LazyFrame,
    spec: HistoricalDividendFeatureSpec,
) -> pl.LazyFrame:
    """Attach trailing observed-cash totals and the submitted annualized scalar feature.

    Same-date cash components are additive. A grouped cumulative-sum/two-as-of plan computes each
    unique ticker/date window without expanding every option row against every historical payment,
    then joins the result many-to-one in original quote order. The immutable specification controls
    both endpoints, the exact week-based window, annualization, and whether absent ticker history is
    admissible.

    `historical_dividend_yield` is exactly
    `log(1 + trailing cash / split-adjusted close) / (lookback weeks / weeks per year)` and does not
    modify discrete forecast schedules.

    Raises:
        MissingColumnsError: If a required quote or history column is absent.
        ConflictingColumnsError: If a quote already contains a stage-owned output name.
        HistoricalDividendSchemaError: If a required input column has the wrong dtype.
        HistoricalDividendContentError: If inputs are empty, invalid, duplicated, or violate the
            declared ticker-coverage policy.

    """
    validated_quotes = _validate_dtypes(
        forbid_columns(quotes, tuple(HISTORICAL_DIVIDEND_FEATURE_DTYPES)),
        _HISTORICAL_QUOTE_DTYPES,
        prefix="quotes",
    )
    validated_history = _validate_dtypes(
        history,
        HISTORICAL_DIVIDEND_DTYPES,
        prefix="history",
    ).select(tuple(HISTORICAL_DIVIDEND_DTYPES))
    violations = _input_violations(validated_quotes, validated_history)
    if violations:
        raise HistoricalDividendContentError(violations=violations)
    _validate_ticker_coverage(validated_quotes, validated_history, spec)

    totals = _trailing_totals(validated_quotes, validated_history, spec)
    joined = validated_quotes.join(
        totals,
        on=("ticker", "quote_date"),
        how="left",
        validate="m:1",
        coalesce=True,
        maintain_order="left",
    ).with_columns(
        pl.col("historical_total_dividend_in_lookback_window").fill_null(value=0.0),
    )
    annualized_window = spec.lookback_weeks / spec.weeks_per_year
    cash_to_spot = pl.col("historical_total_dividend_in_lookback_window") / pl.col(
        "split_adjusted_close",
    )
    return joined.with_columns(
        ((1.0 + cash_to_spot).log() / annualized_window).alias("historical_dividend_yield"),
    )


__all__ = [
    "HISTORICAL_DIVIDEND_DTYPES",
    "HISTORICAL_DIVIDEND_FEATURE_DTYPES",
    "PROJECT_AS_SUBMITTED_HISTORICAL_DIVIDEND_SPEC",
    "HistoricalDividendContentError",
    "HistoricalDividendError",
    "HistoricalDividendFeatureSpec",
    "HistoricalDividendSchemaError",
    "HistoricalDividendSpecError",
    "add_historical_dividend_features",
]
