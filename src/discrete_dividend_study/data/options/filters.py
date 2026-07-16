"""Shared option-quote features and explicit immutable study filters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import polars as pl

from discrete_dividend_study.core.schema import require_columns
from discrete_dividend_study.data.options.quotes import (
    FLOAT32_PAIRED_OPTION_QUOTE_DTYPES,
    FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

QUOTE_FEATURE_DTYPES = {
    "call_mid_price": pl.Float32,
    "put_mid_price": pl.Float32,
    "days_to_expiry": pl.Int64,
    "yte": pl.Float64,
    "call_moneyness": pl.Float32,
    "put_moneyness": pl.Float32,
    "call_intrinsic": pl.Float32,
    "put_intrinsic": pl.Float32,
    "project_call_spread": pl.Float32,
    "call_spread": pl.Float32,
    "put_spread": pl.Float32,
}
FLOAT64_QUOTE_FEATURE_DTYPES: Final = {
    name: dtype if dtype == pl.Int64 else pl.Float64 for name, dtype in QUOTE_FEATURE_DTYPES.items()
}
_FLOAT32_QUOTE_FEATURE_INPUT_DTYPES: Final = dict(FLOAT32_PAIRED_OPTION_QUOTE_DTYPES)
_FLOAT64_QUOTE_FEATURE_INPUT_DTYPES: Final = dict(FLOAT64_PAIRED_OPTION_QUOTE_DTYPES)


class QuoteNumericPrecision(StrEnum):
    """Exact floating-point representation used while evaluating quote predicates."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class QuoteFilterError(ValueError):
    """Base class for invalid feature or filter specifications."""


@dataclass(slots=True)
class QuoteSchemaError(QuoteFilterError):
    """Normalized or derived quote columns have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed dtypes without exposing licensed rows."""
        return f"option quote column dtype mismatches: {self.mismatches!r}"


@dataclass(slots=True)
class QuoteFilterSpecError(QuoteFilterError):
    """An immutable quote-filter value is outside its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid specification value."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(frozen=True, slots=True)
class QuoteFilter:
    """Complete identity of the currently supported call-quote filtering operation."""

    min_quote_date: date | None
    max_quote_date: date | None
    min_days_to_expiry: int
    max_days_to_expiry: int
    min_call_price: float
    min_put_price: float
    min_call_moneyness: float
    max_call_moneyness: float
    min_put_moneyness: float
    max_put_moneyness: float
    intrinsic_ask_multiplier: float
    max_project_call_spread: float
    drop_nulls: bool

    def __post_init__(self) -> None:
        """Reject inferred types, reversed bounds, and non-finite filter values."""
        _validate_optional_date("min_quote_date", self.min_quote_date)
        _validate_optional_date("max_quote_date", self.max_quote_date)
        if (
            self.min_quote_date is not None
            and self.max_quote_date is not None
            and self.max_quote_date < self.min_quote_date
        ):
            raise QuoteFilterSpecError(
                field="max_quote_date",
                observed=self.max_quote_date,
                requirement="on or after min_quote_date",
            )
        _validate_plain_int("min_days_to_expiry", self.min_days_to_expiry, minimum=0)
        _validate_plain_int("max_days_to_expiry", self.max_days_to_expiry, minimum=0)
        if self.max_days_to_expiry < self.min_days_to_expiry:
            raise QuoteFilterSpecError(
                field="max_days_to_expiry",
                observed=self.max_days_to_expiry,
                requirement="at least min_days_to_expiry",
            )
        for field, value in (
            ("min_call_price", self.min_call_price),
            ("min_put_price", self.min_put_price),
            ("min_call_moneyness", self.min_call_moneyness),
            ("max_call_moneyness", self.max_call_moneyness),
            ("min_put_moneyness", self.min_put_moneyness),
            ("max_put_moneyness", self.max_put_moneyness),
            ("intrinsic_ask_multiplier", self.intrinsic_ask_multiplier),
            ("max_project_call_spread", self.max_project_call_spread),
        ):
            _validate_nonnegative_float(field, value)
        if self.max_call_moneyness < self.min_call_moneyness:
            raise QuoteFilterSpecError(
                field="max_call_moneyness",
                observed=self.max_call_moneyness,
                requirement="at least min_call_moneyness",
            )
        if self.max_put_moneyness < self.min_put_moneyness:
            raise QuoteFilterSpecError(
                field="max_put_moneyness",
                observed=self.max_put_moneyness,
                requirement="at least min_put_moneyness",
            )
        if self.intrinsic_ask_multiplier < 1.0:
            raise QuoteFilterSpecError(
                field="intrinsic_ask_multiplier",
                observed=self.intrinsic_ask_multiplier,
                requirement="a plain float at least 1",
            )
        if type(self.drop_nulls) is not bool:
            raise QuoteFilterSpecError(
                field="drop_nulls",
                observed=self.drop_nulls,
                requirement="a plain bool",
            )


def _validate_optional_date(field: str, value: date | None) -> None:
    if value is not None and type(value) is not date:
        raise QuoteFilterSpecError(
            field=field,
            observed=value,
            requirement="None or a plain datetime.date",
        )


def _validate_plain_int(field: str, value: int, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise QuoteFilterSpecError(
            field=field,
            observed=value,
            requirement=f"a plain int at least {minimum}",
        )


def _validate_nonnegative_float(field: str, value: float) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise QuoteFilterSpecError(
            field=field,
            observed=value,
            requirement="a finite nonnegative plain float",
        )


PROJECT_AS_SUBMITTED_FILTER: Final = QuoteFilter(
    min_quote_date=date(2011, 1, 1),
    max_quote_date=None,
    min_days_to_expiry=3,
    max_days_to_expiry=365,
    min_call_price=0.1,
    min_put_price=0.0,
    min_call_moneyness=1.0,
    max_call_moneyness=2.0,
    min_put_moneyness=0.0,
    max_put_moneyness=9_999_999.0,
    intrinsic_ask_multiplier=1.0001,
    max_project_call_spread=0.05,
    drop_nulls=True,
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
        raise QuoteSchemaError(mismatches=mismatches)
    return validated


def add_quote_features(
    frame: pl.LazyFrame,
    *,
    precision: QuoteNumericPrecision = QuoteNumericPrecision.FLOAT32,
) -> pl.LazyFrame:
    """Add shared deterministic quote features without filtering or collecting rows.

    `project_call_spread` is the exact project-study convention `(ask - mid) / mid`.
    `call_spread` and `put_spread` retain the full `(ask - bid) / mid` diagnostics. Intrinsic
    columns retain signed `spot - strike` and `strike - spot` values.

    Raises:
        MissingColumnsError: If a normalized option column is absent.
        QuoteSchemaError: If a normalized option column has the wrong dtype.

    """
    input_dtypes, _ = _precision_dtypes(precision)
    validated = _validate_dtypes(frame, input_dtypes)
    with_mid_and_days = validated.with_columns(
        ((pl.col("call_bid") + pl.col("call_ask")) / 2).alias("call_mid_price"),
        ((pl.col("put_bid") + pl.col("put_ask")) / 2).alias("put_mid_price"),
        (pl.col("expiry_date") - pl.col("quote_date")).dt.total_days().alias("days_to_expiry"),
    )
    with_state = with_mid_and_days.with_columns(
        (pl.col("days_to_expiry") / 365).alias("yte"),
        (pl.col("underlying_price") / pl.col("strike")).alias("call_moneyness"),
        (pl.col("strike") / pl.col("underlying_price")).alias("put_moneyness"),
        (pl.col("underlying_price") - pl.col("strike")).alias("call_intrinsic"),
        (pl.col("strike") - pl.col("underlying_price")).alias("put_intrinsic"),
    )
    return with_state.with_columns(
        ((pl.col("call_ask") - pl.col("call_mid_price")) / pl.col("call_mid_price")).alias(
            "project_call_spread",
        ),
        ((pl.col("call_ask") - pl.col("call_bid")) / pl.col("call_mid_price")).alias(
            "call_spread",
        ),
        ((pl.col("put_ask") - pl.col("put_bid")) / pl.col("put_mid_price")).alias(
            "put_spread",
        ),
    )


def apply_quote_predicates(
    frame: pl.LazyFrame,
    spec: QuoteFilter,
    *,
    precision: QuoteNumericPrecision = QuoteNumericPrecision.FLOAT32,
) -> pl.LazyFrame:
    """Apply only the explicit quote-domain predicates and preserve extra columns.

    This operation can run before expensive support joins because every predicate depends only on
    normalized quote features. The separately named complete-row policy remains a post-join stage.

    Raises:
        MissingColumnsError: If a required derived quote feature is absent.
        QuoteSchemaError: If a required derived quote feature has the wrong dtype.

    """
    _, feature_dtypes = _precision_dtypes(precision)
    validated = _validate_dtypes(frame, feature_dtypes)
    predicate = (
        (pl.col("days_to_expiry") >= spec.min_days_to_expiry)
        & (pl.col("days_to_expiry") <= spec.max_days_to_expiry)
        & (pl.col("call_bid") > spec.min_call_price)
        & (pl.col("call_ask") > spec.min_call_price)
        & (pl.col("put_bid") > spec.min_put_price)
        & (pl.col("put_ask") > spec.min_put_price)
        & (pl.col("call_moneyness") >= spec.min_call_moneyness)
        & (pl.col("call_moneyness") <= spec.max_call_moneyness)
        & (pl.col("put_moneyness") >= spec.min_put_moneyness)
        & (pl.col("put_moneyness") <= spec.max_put_moneyness)
    )
    if spec.min_quote_date is not None:
        predicate &= pl.col("quote_date") >= spec.min_quote_date
    if spec.max_quote_date is not None:
        predicate &= pl.col("quote_date") <= spec.max_quote_date

    return (
        validated.filter(predicate)
        .filter(pl.col("call_intrinsic") < pl.col("call_ask") * spec.intrinsic_ask_multiplier)
        .filter(pl.col("project_call_spread") < spec.max_project_call_spread)
    )


def apply_complete_row_policy(frame: pl.LazyFrame, spec: QuoteFilter) -> pl.LazyFrame:
    """Apply the declared post-join null policy without changing quote predicates."""
    return frame.drop_nulls() if spec.drop_nulls else frame


def apply_quote_filter(
    frame: pl.LazyFrame,
    spec: QuoteFilter,
    *,
    precision: QuoteNumericPrecision = QuoteNumericPrecision.FLOAT32,
) -> pl.LazyFrame:
    """Apply one explicit quote identity in a single convenient operation.

    Extra columns are preserved. When `drop_nulls` is true, null removal covers the complete row at
    the point where this function is called. Performance-sensitive pipelines may call
    `apply_quote_predicates` before joins and `apply_complete_row_policy` after them; the result is
    equivalent when joins preserve quote identity.

    Raises:
        MissingColumnsError: If a required derived quote feature is absent.
        QuoteSchemaError: If a required derived quote feature has the wrong dtype.

    """
    filtered = apply_quote_predicates(frame, spec, precision=precision)
    return apply_complete_row_policy(filtered, spec)


def _precision_dtypes(
    precision: QuoteNumericPrecision,
) -> tuple[
    Mapping[str, type[pl.DataType]],
    Mapping[str, type[pl.DataType]],
]:
    if precision is QuoteNumericPrecision.FLOAT32:
        return _FLOAT32_QUOTE_FEATURE_INPUT_DTYPES, QUOTE_FEATURE_DTYPES
    if precision is QuoteNumericPrecision.FLOAT64:
        return _FLOAT64_QUOTE_FEATURE_INPUT_DTYPES, FLOAT64_QUOTE_FEATURE_DTYPES
    raise QuoteFilterSpecError(
        field="precision",
        observed=precision,
        requirement="QuoteNumericPrecision.FLOAT32 or QuoteNumericPrecision.FLOAT64",
    )


__all__ = [
    "FLOAT64_QUOTE_FEATURE_DTYPES",
    "PROJECT_AS_SUBMITTED_FILTER",
    "QUOTE_FEATURE_DTYPES",
    "QuoteFilter",
    "QuoteFilterError",
    "QuoteFilterSpecError",
    "QuoteNumericPrecision",
    "QuoteSchemaError",
    "add_quote_features",
    "apply_complete_row_policy",
    "apply_quote_filter",
    "apply_quote_predicates",
]
