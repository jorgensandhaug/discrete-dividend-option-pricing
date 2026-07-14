"""Immutable identities for the four reconstructed project classical baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from finoptions import (
    CpuExecutor,
    CpuMathBackend,
    DividendPolicy,
    ExerciseTieRule,
    Lsm,
    OriginExercise,
    RandomStream,
    UniformExerciseMesh,
)

_STUDY_HASH_FORMAT: Final = "project-classical-reconstruction-study-v1"
_EXECUTION_HASH_FORMAT: Final = "project-classical-execution-v2"
_MAX_U32: Final = (2**32) - 1
_MAX_U64: Final = (2**64) - 1
_MIN_EVALUATION_PATHS: Final = 2
_MAX_POLYNOMIAL_DEGREE: Final = 4


class ClassicalSpecError(ValueError):
    """A classical study or execution identity is incomplete or unsupported."""


@dataclass(slots=True)
class ClassicalSpecValueError(ClassicalSpecError):
    """One named specification value violates its closed domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid declaration without interpreting it."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


class ClassicalVariant(StrEnum):
    """The four classical option-pricing comparisons recovered from the project thesis."""

    EXACT_CASH = "exact_cash"
    ESCROWED_CASH_SPOT_MATCHED = "escrowed_cash_spot_matched"
    FORECAST_YIELD = "forecast_yield"
    HISTORICAL_YIELD = "historical_yield"


class ClassicalDividendRepresentation(StrEnum):
    """Closed dividend-model identities admitted by the reconstructed study."""

    EXACT_CASH_JUMP = "exact_cash_jump"
    ESCROWED_PREPAID_FORWARD = "escrowed_prepaid_forward"
    CONTINUOUS_FORECAST_YIELD = "continuous_forecast_yield"
    CONTINUOUS_HISTORICAL_YIELD = "continuous_historical_yield"


@dataclass(frozen=True, slots=True)
class ClassicalVariantSpec:
    """One named result and its exact dividend representation."""

    variant: ClassicalVariant
    submitted_result_name: str
    dividend_representation: ClassicalDividendRepresentation

    def __post_init__(self) -> None:
        """Reject aliases or representation pairings outside the recovered comparison."""
        if type(self.variant) is not ClassicalVariant:
            raise ClassicalSpecValueError(
                field="variant",
                observed=self.variant,
                requirement="ClassicalVariant",
            )
        if type(self.submitted_result_name) is not str or not self.submitted_result_name:
            raise ClassicalSpecValueError(
                field="submitted_result_name",
                observed=self.submitted_result_name,
                requirement="a nonempty plain string",
            )
        expected = _REPRESENTATION_BY_VARIANT[self.variant]
        if self.dividend_representation is not expected:
            raise ClassicalSpecValueError(
                field=f"{self.variant.value}.dividend_representation",
                observed=self.dividend_representation,
                requirement=repr(expected),
            )


@dataclass(frozen=True, slots=True)
class ClassicalEstimatorSpec:
    """Financial and statistical identity of the reconstructed Bermudan LSM estimator."""

    base_intervals: int
    origin_exercise: OriginExercise
    training_paths: int
    evaluation_paths: int
    polynomial_degree: int
    rank_tolerance: float
    condition_limit: float
    tie_rule: ExerciseTieRule

    def __post_init__(self) -> None:
        """Reject incomplete meshes, path populations, or numerical regression contracts."""
        for field, value in (
            ("base_intervals", self.base_intervals),
            ("training_paths", self.training_paths),
            ("evaluation_paths", self.evaluation_paths),
        ):
            if type(value) is not int or not 1 <= value <= _MAX_U32:
                raise ClassicalSpecValueError(
                    field=field,
                    observed=value,
                    requirement=f"a plain int in [1, {_MAX_U32}]",
                )
        if self.evaluation_paths < _MIN_EVALUATION_PATHS:
            raise ClassicalSpecValueError(
                field="evaluation_paths",
                observed=self.evaluation_paths,
                requirement="at least two paths for a sample standard error",
            )
        if type(self.origin_exercise) is not OriginExercise:
            raise ClassicalSpecValueError(
                field="origin_exercise",
                observed=self.origin_exercise,
                requirement="OriginExercise",
            )
        if (
            type(self.polynomial_degree) is not int
            or not 0 <= self.polynomial_degree <= _MAX_POLYNOMIAL_DEGREE
        ):
            raise ClassicalSpecValueError(
                field="polynomial_degree",
                observed=self.polynomial_degree,
                requirement="a plain int in [0, 4]",
            )
        for field, value in (
            ("rank_tolerance", self.rank_tolerance),
            ("condition_limit", self.condition_limit),
        ):
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ClassicalSpecValueError(
                    field=field,
                    observed=value,
                    requirement="a finite positive plain float",
                )
        if self.condition_limit <= 1.0:
            raise ClassicalSpecValueError(
                field="condition_limit",
                observed=self.condition_limit,
                requirement="a finite plain float greater than one",
            )
        if type(self.tie_rule) is not ExerciseTieRule:
            raise ClassicalSpecValueError(
                field="tie_rule",
                observed=self.tie_rule,
                requirement="ExerciseTieRule",
            )

    def mesh(self) -> UniformExerciseMesh:
        """Build the exact engine mesh without adding an execution choice."""
        return UniformExerciseMesh(
            base_intervals=self.base_intervals,
            origin_exercise=self.origin_exercise,
        )

    def method(self, execution: ClassicalExecutionSpec) -> Lsm:
        """Resolve random namespaces from a separate explicit execution specification."""
        if type(execution) is not ClassicalExecutionSpec:
            raise ClassicalSpecValueError(
                field="execution",
                observed=execution,
                requirement="ClassicalExecutionSpec",
            )
        return Lsm(
            training_paths=self.training_paths,
            evaluation_paths=self.evaluation_paths,
            training_stream=RandomStream(
                seed=execution.training_seed,
                domain=execution.training_domain_start,
            ),
            evaluation_stream=RandomStream(
                seed=execution.evaluation_seed,
                domain=execution.evaluation_domain_start,
            ),
            polynomial_degree=self.polynomial_degree,
            rank_tolerance=self.rank_tolerance,
            condition_limit=self.condition_limit,
            tie_rule=self.tie_rule,
        )


@dataclass(frozen=True, slots=True)
class ClassicalStudySpec:
    """Framework-free financial identity shared by every project classical run."""

    variants: tuple[ClassicalVariantSpec, ...]
    estimator: ClassicalEstimatorSpec
    spot_column: str
    strike_column: str
    maturity_column: str
    rate_column: str
    volatility_column: str
    target_column: str
    forecast_yield_column: str
    historical_yield_column: str
    cash_times_column: str
    cash_amounts_column: str
    exact_cash_policy: DividendPolicy

    def __post_init__(self) -> None:
        """Require every baseline once and every source semantic explicitly."""
        if (
            type(self.variants) is not tuple
            or len(self.variants) != len(ClassicalVariant)
            or any(type(variant) is not ClassicalVariantSpec for variant in self.variants)
        ):
            raise ClassicalSpecValueError(
                field="variants",
                observed=self.variants,
                requirement="one ClassicalVariantSpec for every ClassicalVariant",
            )
        identities = tuple(variant.variant for variant in self.variants)
        if len(set(identities)) != len(identities) or set(identities) != set(ClassicalVariant):
            raise ClassicalSpecValueError(
                field="variants",
                observed=identities,
                requirement="each ClassicalVariant exactly once",
            )
        aliases = tuple(variant.submitted_result_name for variant in self.variants)
        if len(set(aliases)) != len(aliases):
            raise ClassicalSpecValueError(
                field="submitted_result_name",
                observed=aliases,
                requirement="unique aliases",
            )
        if type(self.estimator) is not ClassicalEstimatorSpec:
            raise ClassicalSpecValueError(
                field="estimator",
                observed=self.estimator,
                requirement="ClassicalEstimatorSpec",
            )
        columns = self.input_columns()
        if any(type(column) is not str or not column for column in columns) or len(
            set(columns)
        ) != len(columns):
            raise ClassicalSpecValueError(
                field="input columns",
                observed=columns,
                requirement="distinct nonempty plain strings",
            )
        if type(self.exact_cash_policy) is not DividendPolicy:
            raise ClassicalSpecValueError(
                field="exact_cash_policy",
                observed=self.exact_cash_policy,
                requirement="DividendPolicy",
            )

    def variant(self, identity: ClassicalVariant) -> ClassicalVariantSpec:
        """Return one exact variant without accepting string aliases."""
        if type(identity) is not ClassicalVariant:
            raise ClassicalSpecValueError(
                field="identity",
                observed=identity,
                requirement="ClassicalVariant",
            )
        return next(variant for variant in self.variants if variant.variant is identity)

    def input_columns(self) -> tuple[str, ...]:
        """Return the minimal ordered materialized feature columns for all four variants."""
        return (
            self.spot_column,
            self.strike_column,
            self.maturity_column,
            self.rate_column,
            self.volatility_column,
            self.forecast_yield_column,
            self.historical_yield_column,
            self.target_column,
            self.cash_times_column,
            self.cash_amounts_column,
        )


@dataclass(frozen=True, slots=True)
class ClassicalExecutionSpec:
    """CPU resources and row-addressed random namespaces, separate from financial identity."""

    threads: int
    math_backend: CpuMathBackend
    training_seed: int
    training_domain_start: int
    evaluation_seed: int
    evaluation_domain_start: int

    def __post_init__(self) -> None:
        """Reject automatic resources, truncated seeds, or invalid random-domain roots."""
        if type(self.threads) is not int or self.threads <= 0:
            raise ClassicalSpecValueError(
                field="threads",
                observed=self.threads,
                requirement="a positive plain int",
            )
        if type(self.math_backend) is not CpuMathBackend:
            raise ClassicalSpecValueError(
                field="math_backend",
                observed=self.math_backend,
                requirement="CpuMathBackend",
            )
        for field, value in (
            ("training_seed", self.training_seed),
            ("evaluation_seed", self.evaluation_seed),
        ):
            if type(value) is not int or not 0 <= value <= _MAX_U64:
                raise ClassicalSpecValueError(
                    field=field,
                    observed=value,
                    requirement=f"a plain int in [0, {_MAX_U64}]",
                )
        for field, value in (
            ("training_domain_start", self.training_domain_start),
            ("evaluation_domain_start", self.evaluation_domain_start),
        ):
            if type(value) is not int or not 0 <= value <= _MAX_U32:
                raise ClassicalSpecValueError(
                    field=field,
                    observed=value,
                    requirement=f"a plain int in [0, {_MAX_U32}]",
                )

    def executor(self) -> CpuExecutor:
        """Construct the exact requested dedicated CPU executor."""
        return CpuExecutor(threads=self.threads, math_backend=self.math_backend)

    def validate_rows(self, rows: int) -> None:
        """Reject overflowing or colliding row-addressed random-domain intervals."""
        if type(rows) is not int or rows <= 0:
            raise ClassicalSpecValueError(
                field="rows",
                observed=rows,
                requirement="a positive plain int",
            )
        training_end = self.training_domain_start + rows - 1
        evaluation_end = self.evaluation_domain_start + rows - 1
        if training_end > _MAX_U32:
            raise ClassicalSpecValueError(
                field="training random-domain interval",
                observed=(self.training_domain_start, training_end),
                requirement=f"an interval within [0, {_MAX_U32}]",
            )
        if evaluation_end > _MAX_U32:
            raise ClassicalSpecValueError(
                field="evaluation random-domain interval",
                observed=(self.evaluation_domain_start, evaluation_end),
                requirement=f"an interval within [0, {_MAX_U32}]",
            )
        if max(self.training_domain_start, self.evaluation_domain_start) <= min(
            training_end,
            evaluation_end,
        ):
            raise ClassicalSpecValueError(
                field="training/evaluation random-domain intervals",
                observed=(
                    (self.training_domain_start, training_end),
                    (self.evaluation_domain_start, evaluation_end),
                ),
                requirement="disjoint intervals",
            )


_REPRESENTATION_BY_VARIANT: Final = {
    ClassicalVariant.EXACT_CASH: ClassicalDividendRepresentation.EXACT_CASH_JUMP,
    ClassicalVariant.ESCROWED_CASH_SPOT_MATCHED: (
        ClassicalDividendRepresentation.ESCROWED_PREPAID_FORWARD
    ),
    ClassicalVariant.FORECAST_YIELD: ClassicalDividendRepresentation.CONTINUOUS_FORECAST_YIELD,
    ClassicalVariant.HISTORICAL_YIELD: (
        ClassicalDividendRepresentation.CONTINUOUS_HISTORICAL_YIELD
    ),
}

PROJECT_CLASSICAL_RECONSTRUCTION_SPEC: Final = ClassicalStudySpec(
    variants=(
        ClassicalVariantSpec(
            variant=ClassicalVariant.EXACT_CASH,
            submitted_result_name="IT_GBM_SPOT",
            dividend_representation=ClassicalDividendRepresentation.EXACT_CASH_JUMP,
        ),
        ClassicalVariantSpec(
            variant=ClassicalVariant.ESCROWED_CASH_SPOT_MATCHED,
            submitted_result_name="IT_GBMT_FD_AH",
            dividend_representation=ClassicalDividendRepresentation.ESCROWED_PREPAID_FORWARD,
        ),
        ClassicalVariantSpec(
            variant=ClassicalVariant.FORECAST_YIELD,
            submitted_result_name="IT_GBMT_FY",
            dividend_representation=ClassicalDividendRepresentation.CONTINUOUS_FORECAST_YIELD,
        ),
        ClassicalVariantSpec(
            variant=ClassicalVariant.HISTORICAL_YIELD,
            submitted_result_name="IT_GBMT_HY",
            dividend_representation=ClassicalDividendRepresentation.CONTINUOUS_HISTORICAL_YIELD,
        ),
    ),
    estimator=ClassicalEstimatorSpec(
        base_intervals=150,
        origin_exercise=OriginExercise.EXCLUDED,
        training_paths=500,
        evaluation_paths=500,
        polynomial_degree=4,
        rank_tolerance=1.0e-12,
        condition_limit=1.0e8,
        tie_rule=ExerciseTieRule.STRICTLY_GREATER,
    ),
    spot_column="underlying_price",
    strike_column="strike",
    maturity_column="yte",
    rate_column="risk_free_rate",
    volatility_column="historical_volatility_90",
    target_column="call_mid_price",
    forecast_yield_column="forecasted_dividend_yield",
    historical_yield_column="historical_dividend_yield",
    cash_times_column="forecasted_years_to_discrete_ex_dates",
    cash_amounts_column="forecasted_discrete_amounts",
    exact_cash_policy=DividendPolicy.LIQUIDATOR,
)


def classical_study_sha256(study: ClassicalStudySpec) -> str:
    """Hash the complete financial/estimator identity without execution resources."""
    if type(study) is not ClassicalStudySpec:
        raise ClassicalSpecValueError(
            field="study",
            observed=study,
            requirement="ClassicalStudySpec",
        )
    payload = {
        "format": _STUDY_HASH_FORMAT,
        "variants": [
            {
                "variant": variant.variant.value,
                "submitted_result_name": variant.submitted_result_name,
                "dividend_representation": variant.dividend_representation.value,
            }
            for variant in study.variants
        ],
        "estimator": {
            "base_intervals": study.estimator.base_intervals,
            "origin_exercise": study.estimator.origin_exercise.value,
            "training_paths": study.estimator.training_paths,
            "evaluation_paths": study.estimator.evaluation_paths,
            "polynomial_degree": study.estimator.polynomial_degree,
            "rank_tolerance": study.estimator.rank_tolerance,
            "condition_limit": study.estimator.condition_limit,
            "tie_rule": study.estimator.tie_rule.value,
        },
        "input_columns": list(study.input_columns()),
        "exact_cash_policy": study.exact_cash_policy.value,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def classical_execution_sha256(execution: ClassicalExecutionSpec) -> str:
    """Hash exact CPU and random namespace choices separately from financial identity."""
    if type(execution) is not ClassicalExecutionSpec:
        raise ClassicalSpecValueError(
            field="execution",
            observed=execution,
            requirement="ClassicalExecutionSpec",
        )
    payload = {
        "format": _EXECUTION_HASH_FORMAT,
        "threads": execution.threads,
        "math_backend": execution.math_backend.value,
        "training_seed": execution.training_seed,
        "training_domain_start": execution.training_domain_start,
        "evaluation_seed": execution.evaluation_seed,
        "evaluation_domain_start": execution.evaluation_domain_start,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PROJECT_CLASSICAL_RECONSTRUCTION_SPEC",
    "ClassicalDividendRepresentation",
    "ClassicalEstimatorSpec",
    "ClassicalExecutionSpec",
    "ClassicalSpecError",
    "ClassicalSpecValueError",
    "ClassicalStudySpec",
    "ClassicalVariant",
    "ClassicalVariantSpec",
    "classical_execution_sha256",
    "classical_study_sha256",
]
