"""Strict evaluation of the project as-submitted NSS rate feature."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from .schema import forbid_columns, require_columns

if TYPE_CHECKING:
    from collections.abc import Mapping

NSS_PARAMETER_DTYPES = {
    "date": pl.Date,
    "beta0": pl.Float64,
    "beta1": pl.Float64,
    "beta2": pl.Float64,
    "beta3": pl.Float64,
    "tau1": pl.Float64,
    "tau2": pl.Float64,
    "shortest_maturity": pl.Float64,
    "shortest_rate": pl.Float64,
}
_NSS_QUOTE_DTYPES = {
    "quote_date": pl.Date,
    "yte": pl.Float64,
}
_SMALL_LOADING_ARGUMENT = 1.0e-4


class NssRateError(ValueError):
    """Base class for invalid NSS rate inputs or specifications."""


@dataclass(slots=True)
class NssRateSchemaError(NssRateError):
    """Quote or parameter inputs have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed dtypes without exposing rows."""
        return f"NSS rate input dtype mismatches: {self.mismatches!r}"


@dataclass(slots=True)
class NssRateContentError(NssRateError):
    """Quote or NSS parameter rows violate their declared contract."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing market values."""
        return f"invalid NSS rate content: {self.violations!r}"


@dataclass(slots=True)
class NssRateSpecError(NssRateError):
    """An NSS evaluation specification is outside its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid specification value."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(frozen=True, slots=True)
class NssRateFeatureSpec:
    """Identity of the submitted NSS maturity and short-end evaluation rule."""

    minimum_nss_maturity: float
    use_observed_shortest_rate: bool

    def __post_init__(self) -> None:
        """Reject a nonpositive numerical domain or inferred short-end policy."""
        if (
            type(self.minimum_nss_maturity) is not float
            or not math.isfinite(self.minimum_nss_maturity)
            or self.minimum_nss_maturity <= 0.0
        ):
            raise NssRateSpecError(
                field="minimum_nss_maturity",
                observed=self.minimum_nss_maturity,
                requirement="a finite positive plain float",
            )
        if type(self.use_observed_shortest_rate) is not bool:
            raise NssRateSpecError(
                field="use_observed_shortest_rate",
                observed=self.use_observed_shortest_rate,
                requirement="a plain bool",
            )


PROJECT_AS_SUBMITTED_NSS_RATE_SPEC: Final = NssRateFeatureSpec(
    minimum_nss_maturity=1.0e-8,
    use_observed_shortest_rate=True,
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
        raise NssRateSchemaError(mismatches=mismatches)
    return validated


def _input_violations(
    quotes: pl.LazyFrame,
    parameters: pl.LazyFrame,
) -> tuple[tuple[str, int], ...]:
    quote_diagnostics = (
        quotes.select(
            pl.len().alias("quote_rows"),
            pl.col("quote_date").is_null().sum().alias("null_quote_dates"),
            pl.col("yte").is_null().sum().alias("null_maturities"),
            (~pl.col("yte").is_finite()).fill_null(value=False).sum().alias("nonfinite_maturities"),
            (pl.col("yte") < 0.0).fill_null(value=False).sum().alias("negative_maturities"),
        )
        .collect()
        .row(0, named=True)
    )
    parameter_diagnostics = (
        parameters.select(
            pl.len().alias("parameter_rows"),
            pl.any_horizontal(pl.all().is_null()).sum().alias("parameter_rows_with_nulls"),
            pl.any_horizontal(
                *(~pl.col(name).is_finite() for name in NSS_PARAMETER_DTYPES if name != "date"),
            )
            .fill_null(value=False)
            .sum()
            .alias("parameter_rows_with_nonfinite_values"),
            pl.col("date").is_duplicated().sum().alias("duplicate_parameter_dates"),
            pl.any_horizontal(
                pl.col("tau1") <= 0.0,
                pl.col("tau2") <= 0.0,
                pl.col("shortest_maturity") <= 0.0,
            )
            .fill_null(value=False)
            .sum()
            .alias("nonpositive_parameter_scales"),
        )
        .collect()
        .row(0, named=True)
    )
    combined = {**quote_diagnostics, **parameter_diagnostics}
    violations: list[tuple[str, int]] = []
    if combined["quote_rows"] == 0:
        violations.append(("empty_quotes", 1))
    if combined["parameter_rows"] == 0:
        violations.append(("empty_parameters", 1))
    violations.extend(
        (name, value)
        for name, value in combined.items()
        if name not in {"quote_rows", "parameter_rows"} and type(value) is int and value != 0
    )
    return tuple(violations)


def _stable_nss_loadings(argument: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    square = argument * argument
    cube = square * argument
    fourth = square * square
    level_series = 1.0 - argument / 2.0 + square / 6.0 - cube / 24.0 + fourth / 120.0
    curvature_series = argument / 2.0 - square / 3.0 + cube / 8.0 - fourth / 30.0
    exponential = (-argument).exp()
    level_direct = (1.0 - exponential) / argument
    level = (
        pl.when(argument.abs() < _SMALL_LOADING_ARGUMENT).then(level_series).otherwise(level_direct)
    )
    curvature = (
        pl.when(argument.abs() < _SMALL_LOADING_ARGUMENT)
        .then(curvature_series)
        .otherwise(level_direct - exponential)
    )
    return level, curvature


def nss_rate_expression(
    maturity: pl.Expr,
    spec: NssRateFeatureSpec,
) -> pl.Expr:
    """Return a numerically stable NSS rate expression for joined parameter columns.

    The small-argument series is algebraically the same Nelson-Siegel loading and changes only its
    floating-point evaluation near zero. It is not a different curve model.

    """
    evaluated_maturity = pl.max_horizontal(
        maturity,
        pl.lit(spec.minimum_nss_maturity, dtype=pl.Float64),
    )
    level1, curvature1 = _stable_nss_loadings(evaluated_maturity / pl.col("tau1"))
    _, curvature2 = _stable_nss_loadings(evaluated_maturity / pl.col("tau2"))
    nss_rate = (
        pl.col("beta0")
        + pl.col("beta1") * level1
        + pl.col("beta2") * curvature1
        + pl.col("beta3") * curvature2
    )
    if spec.use_observed_shortest_rate:
        return (
            pl.when(maturity <= pl.col("shortest_maturity"))
            .then(pl.col("shortest_rate"))
            .otherwise(nss_rate)
        )
    return nss_rate


def add_nss_rate_feature(
    quotes: pl.LazyFrame,
    parameters: pl.LazyFrame,
    spec: NssRateFeatureSpec,
) -> pl.LazyFrame:
    """Join exact-date parameters and add the project as-submitted scalar rate.

    Every quote date must have exactly one finite parameter row. The submitted short-end policy uses
    the observed shortest-maturity rate at and below that maturity; longer maturities use the NSS
    expression. The resulting `risk_free_rate` is a scalar study feature. Conversion into an engine
    discount curve remains a separate adapter with its own quote-convention identity.

    Raises:
        MissingColumnsError: If a required quote or parameter column is absent.
        ConflictingColumnsError: If quotes already contain `risk_free_rate`.
        NssRateSchemaError: If an input column has the wrong dtype.
        NssRateContentError: If rows are empty, invalid, duplicated, or lack exact-date parameters.

    """
    validated_quotes = _validate_dtypes(
        forbid_columns(quotes, ("risk_free_rate",)),
        _NSS_QUOTE_DTYPES,
        prefix="quotes",
    )
    validated_parameters = _validate_dtypes(
        parameters,
        NSS_PARAMETER_DTYPES,
        prefix="parameters",
    ).select(tuple(NSS_PARAMETER_DTYPES))
    violations = _input_violations(validated_quotes, validated_parameters)
    if violations:
        raise NssRateContentError(violations=violations)

    joined = validated_quotes.join(
        validated_parameters,
        left_on="quote_date",
        right_on="date",
        how="left",
        validate="m:1",
        coalesce=True,
        maintain_order="left",
    )
    missing = joined.select(pl.col("beta0").is_null().sum()).collect().item()
    if type(missing) is int and missing != 0:
        raise NssRateContentError(violations=(("missing_parameter_dates", missing),))
    return joined.with_columns(
        nss_rate_expression(pl.col("yte"), spec).alias("risk_free_rate"),
    ).drop(
        "beta0",
        "beta1",
        "beta2",
        "beta3",
        "tau1",
        "tau2",
        "shortest_maturity",
        "shortest_rate",
    )


__all__ = [
    "NSS_PARAMETER_DTYPES",
    "PROJECT_AS_SUBMITTED_NSS_RATE_SPEC",
    "NssRateContentError",
    "NssRateError",
    "NssRateFeatureSpec",
    "NssRateSchemaError",
    "NssRateSpecError",
    "add_nss_rate_feature",
    "nss_rate_expression",
]
