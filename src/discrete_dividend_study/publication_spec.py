"""Explicit identities for one publication-estimator replication."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from finoptions import (
    AmericanOptionError,
    AndersenLake,
    CpuExecutor,
    CpuMathBackend,
    DividendPolicy,
    ExerciseTieRule,
    ExplicitExerciseMesh,
    InputError,
    InsufficientExerciseSupportPolicy,
    Lsm,
    MaturityBoundary,
    OriginExercise,
    RaggedExerciseTimes,
    RandomStream,
    UniformExerciseMesh,
)

_STUDY_HASH_FORMAT: Final = "project-publication-study-v3"
_EXECUTION_HASH_FORMAT: Final = "project-publication-replication-execution-v1"
_MAX_U32: Final = (2**32) - 1
_MAX_U64: Final = (2**64) - 1
_MAX_POLYNOMIAL_DEGREE: Final = 4
_MIN_EVALUATION_PATHS: Final = 2


class PublicationSpecError(ValueError):
    """A publication study or replication identity is incomplete."""


@dataclass(slots=True)
class PublicationSpecValueError(PublicationSpecError):
    """One named publication specification value violates its domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid declaration without repairing it."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


class PublicationVariant(StrEnum):
    """The four paired dividend representations in the publication comparison."""

    CASH_JUMP_LIQUIDATOR = "cash_jump_liquidator"
    ESCROWED_CASH_SPOT_MATCHED = "escrowed_cash_spot_matched"
    FORECAST_YIELD = "forecast_yield"
    HISTORICAL_YIELD = "historical_yield"

    @property
    def uses_cash_schedule(self) -> bool:
        """Return whether this variant consumes the deterministic cash-event schedule."""
        return self in (
            PublicationVariant.CASH_JUMP_LIQUIDATOR,
            PublicationVariant.ESCROWED_CASH_SPOT_MATCHED,
        )


class PublicationCashPartitionRule(StrEnum):
    """Closed eligibility rule for exact cash-call exercise schedules."""

    NONNEGATIVE_RATE_EXACT_CALENDAR = "nonnegative_rate_exact_calendar_v2"


@dataclass(frozen=True, slots=True)
class PublicationCashLsmSpec:
    """One pilot-selected cash-dividend LSM identity with no production default."""

    general_base_intervals: int
    general_origin_exercise: OriginExercise
    training_paths: int
    evaluation_paths: int
    polynomial_degree: int
    rank_tolerance: float
    condition_limit: float
    tie_rule: ExerciseTieRule
    insufficient_exercise_support: InsufficientExerciseSupportPolicy
    cash_partition_rule: PublicationCashPartitionRule

    def __post_init__(self) -> None:
        """Require strict support and a complete general American exercise identity."""
        _validate_estimator_counts(self)
        _validate_estimator_regression(self)
        if self.general_origin_exercise is not OriginExercise.INCLUDED:
            raise PublicationSpecValueError(
                field="general_origin_exercise",
                observed=self.general_origin_exercise,
                requirement="OriginExercise.INCLUDED",
            )
        if type(self.tie_rule) is not ExerciseTieRule:
            raise PublicationSpecValueError(
                field="tie_rule",
                observed=self.tie_rule,
                requirement="ExerciseTieRule",
            )
        if self.insufficient_exercise_support is not InsufficientExerciseSupportPolicy.FAIL:
            raise PublicationSpecValueError(
                field="insufficient_exercise_support",
                observed=self.insufficient_exercise_support,
                requirement="InsufficientExerciseSupportPolicy.FAIL",
            )
        if type(self.cash_partition_rule) is not PublicationCashPartitionRule:
            raise PublicationSpecValueError(
                field="cash_partition_rule",
                observed=self.cash_partition_rule,
                requirement="PublicationCashPartitionRule",
            )

    def general_mesh(self) -> UniformExerciseMesh:
        """Build the separately declared mesh for general-domain cash rows."""
        return UniformExerciseMesh(
            base_intervals=self.general_base_intervals,
            origin_exercise=self.general_origin_exercise,
        )

    @staticmethod
    def exact_cash_mesh(schedules: RaggedExerciseTimes) -> ExplicitExerciseMesh:
        """Build the typed calendar-owned nonnegative-rate cash schedule."""
        if type(schedules) is not RaggedExerciseTimes:
            raise PublicationSpecValueError(
                field="schedules",
                observed=schedules,
                requirement="RaggedExerciseTimes",
            )
        return ExplicitExerciseMesh(
            schedules=schedules,
            origin_exercise=OriginExercise.EXCLUDED,
        )

    def method(
        self,
        execution: PublicationReplicationExecutionSpec,
        *,
        unit_offset: int,
    ) -> Lsm:
        """Build one partition method with original reordered-unit domain addressing."""
        if type(execution) is not PublicationReplicationExecutionSpec:
            raise PublicationSpecValueError(
                field="execution",
                observed=execution,
                requirement="PublicationReplicationExecutionSpec",
            )
        if type(unit_offset) is not int or not 0 <= unit_offset <= _MAX_U32:
            raise PublicationSpecValueError(
                field="unit_offset",
                observed=unit_offset,
                requirement=f"a plain int in [0, {_MAX_U32}]",
            )
        training_domain = execution.training_domain_start + unit_offset
        evaluation_domain = execution.evaluation_domain_start + unit_offset
        if training_domain > _MAX_U32 or evaluation_domain > _MAX_U32:
            raise PublicationSpecValueError(
                field="partition random-domain roots",
                observed=(training_domain, evaluation_domain),
                requirement=f"values within [0, {_MAX_U32}]",
            )
        return Lsm(
            training_paths=self.training_paths,
            evaluation_paths=self.evaluation_paths,
            training_stream=RandomStream(seed=execution.training_seed, domain=training_domain),
            evaluation_stream=RandomStream(
                seed=execution.evaluation_seed,
                domain=evaluation_domain,
            ),
            polynomial_degree=self.polynomial_degree,
            rank_tolerance=self.rank_tolerance,
            condition_limit=self.condition_limit,
            tie_rule=self.tie_rule,
            insufficient_exercise_support=self.insufficient_exercise_support,
        )


def _validate_estimator_counts(spec: PublicationCashLsmSpec) -> None:
    for field, value in (
        ("general_base_intervals", spec.general_base_intervals),
        ("training_paths", spec.training_paths),
        ("evaluation_paths", spec.evaluation_paths),
    ):
        if type(value) is not int or not 1 <= value <= _MAX_U32:
            raise PublicationSpecValueError(
                field=field,
                observed=value,
                requirement=f"a plain int in [1, {_MAX_U32}]",
            )
    if spec.evaluation_paths < _MIN_EVALUATION_PATHS:
        raise PublicationSpecValueError(
            field="evaluation_paths",
            observed=spec.evaluation_paths,
            requirement=f"at least {_MIN_EVALUATION_PATHS} paths",
        )


def _validate_estimator_regression(spec: PublicationCashLsmSpec) -> None:
    if (
        type(spec.polynomial_degree) is not int
        or not 0 <= spec.polynomial_degree <= _MAX_POLYNOMIAL_DEGREE
    ):
        raise PublicationSpecValueError(
            field="polynomial_degree",
            observed=spec.polynomial_degree,
            requirement=f"a plain int in [0, {_MAX_POLYNOMIAL_DEGREE}]",
        )
    for field, value in (
        ("rank_tolerance", spec.rank_tolerance),
        ("condition_limit", spec.condition_limit),
    ):
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise PublicationSpecValueError(
                field=field,
                observed=value,
                requirement="a finite positive plain float",
            )
    if spec.condition_limit <= 1.0:
        raise PublicationSpecValueError(
            field="condition_limit",
            observed=spec.condition_limit,
            requirement="a finite plain float greater than one",
        )


@dataclass(frozen=True, slots=True)
class PublicationYieldBoundarySpec:
    """One fixed deterministic continuous-yield boundary-method identity."""

    boundary_quadrature_order: int
    iterations: int
    collocation_nodes: int
    price_quadrature_order: int
    maximum_relative_boundary_residual: float
    maximum_relative_obstacle_projection: float

    def __post_init__(self) -> None:
        """Reject coerced controls and unsupported native configurations immediately."""
        for field in (
            "boundary_quadrature_order",
            "iterations",
            "collocation_nodes",
            "price_quadrature_order",
        ):
            value = getattr(self, field)
            if type(value) is not int:
                raise PublicationSpecValueError(
                    field=field,
                    observed=value,
                    requirement="a plain int admitted by AndersenLake",
                )
        for field in (
            "maximum_relative_boundary_residual",
            "maximum_relative_obstacle_projection",
        ):
            value = getattr(self, field)
            if type(value) is not float:
                raise PublicationSpecValueError(
                    field=field,
                    observed=value,
                    requirement="a plain float admitted by AndersenLake",
                )
        try:
            self.method()
        except (AmericanOptionError, InputError) as error:
            raise PublicationSpecValueError(
                field="yield_boundary_method",
                observed=str(error),
                requirement="a financially and numerically admitted AndersenLake configuration",
            ) from error

    def method(self) -> AndersenLake:
        """Build the reusable native boundary method with exactly these controls."""
        return AndersenLake(
            boundary_quadrature_order=self.boundary_quadrature_order,
            iterations=self.iterations,
            collocation_nodes=self.collocation_nodes,
            price_quadrature_order=self.price_quadrature_order,
            maximum_relative_boundary_residual=self.maximum_relative_boundary_residual,
            maximum_relative_obstacle_projection=self.maximum_relative_obstacle_projection,
        )


PROJECT_PUBLICATION_YIELD_BOUNDARY_SPEC: Final = PublicationYieldBoundarySpec(
    boundary_quadrature_order=13,
    iterations=16,
    collocation_nodes=6,
    price_quadrature_order=21,
    maximum_relative_boundary_residual=1.0e-4,
    maximum_relative_obstacle_projection=1.0e-6,
)
"""Target-free selected deterministic profile for both publication yield variants."""


@dataclass(frozen=True, slots=True)
class PublicationStudySpec:
    """Financial and estimator identity for one four-way publication comparison."""

    name: str
    variants: tuple[PublicationVariant, ...]
    cash_lsm: PublicationCashLsmSpec
    yield_boundary: PublicationYieldBoundarySpec
    exact_cash_policy: DividendPolicy
    cash_maturity_boundary: MaturityBoundary

    def __post_init__(self) -> None:
        """Reject aliases, reordered variants, or implicit cash-event semantics."""
        if type(self.name) is not str or not self.name:
            raise PublicationSpecValueError(
                field="name",
                observed=self.name,
                requirement="a nonempty plain string",
            )
        if self.variants != tuple(PublicationVariant):
            raise PublicationSpecValueError(
                field="variants",
                observed=self.variants,
                requirement="tuple(PublicationVariant) in canonical order",
            )
        if type(self.cash_lsm) is not PublicationCashLsmSpec:
            raise PublicationSpecValueError(
                field="cash_lsm",
                observed=self.cash_lsm,
                requirement="PublicationCashLsmSpec",
            )
        if type(self.yield_boundary) is not PublicationYieldBoundarySpec:
            raise PublicationSpecValueError(
                field="yield_boundary",
                observed=self.yield_boundary,
                requirement="PublicationYieldBoundarySpec",
            )
        if self.exact_cash_policy is not DividendPolicy.LIQUIDATOR:
            raise PublicationSpecValueError(
                field="exact_cash_policy",
                observed=self.exact_cash_policy,
                requirement="DividendPolicy.LIQUIDATOR",
            )
        if self.cash_maturity_boundary is not MaturityBoundary.AT_OR_BEFORE:
            raise PublicationSpecValueError(
                field="cash_maturity_boundary",
                observed=self.cash_maturity_boundary,
                requirement="MaturityBoundary.AT_OR_BEFORE",
            )


@dataclass(frozen=True, slots=True)
class PublicationReplicationExecutionSpec:
    """Device and random identity of one complete paired estimator replication."""

    replication: int
    threads: int
    math_backend: CpuMathBackend
    training_seed: int
    training_domain_start: int
    evaluation_seed: int
    evaluation_domain_start: int

    def __post_init__(self) -> None:
        """Reject automatic resources, invalid identities, or incomplete namespaces."""
        if type(self.replication) is not int or self.replication < 0:
            raise PublicationSpecValueError(
                field="replication",
                observed=self.replication,
                requirement="a nonnegative plain int",
            )
        if type(self.threads) is not int or self.threads <= 0:
            raise PublicationSpecValueError(
                field="threads",
                observed=self.threads,
                requirement="a positive plain int",
            )
        if type(self.math_backend) is not CpuMathBackend:
            raise PublicationSpecValueError(
                field="math_backend",
                observed=self.math_backend,
                requirement="CpuMathBackend",
            )
        for field, value, maximum in (
            ("training_seed", self.training_seed, _MAX_U64),
            ("evaluation_seed", self.evaluation_seed, _MAX_U64),
            ("training_domain_start", self.training_domain_start, _MAX_U32),
            ("evaluation_domain_start", self.evaluation_domain_start, _MAX_U32),
        ):
            if type(value) is not int or not 0 <= value <= maximum:
                raise PublicationSpecValueError(
                    field=field,
                    observed=value,
                    requirement=f"a plain int in [0, {maximum}]",
                )

    def executor(self) -> CpuExecutor:
        """Build the exact requested native CPU runtime."""
        return CpuExecutor(threads=self.threads, math_backend=self.math_backend)

    def validate_units(self, units: int) -> None:
        """Reject overflowing or overlapping full-replication domain intervals."""
        if type(units) is not int or units <= 0:
            raise PublicationSpecValueError(
                field="simulation_units",
                observed=units,
                requirement="a positive plain int",
            )
        training_end = self.training_domain_start + units
        evaluation_end = self.evaluation_domain_start + units
        if training_end > _MAX_U32 + 1 or evaluation_end > _MAX_U32 + 1:
            raise PublicationSpecValueError(
                field="random-domain intervals",
                observed=(training_end, evaluation_end),
                requirement=f"half-open intervals within [0, {_MAX_U32 + 1}]",
            )
        if max(self.training_domain_start, self.evaluation_domain_start) < min(
            training_end,
            evaluation_end,
        ):
            raise PublicationSpecValueError(
                field="training/evaluation random-domain intervals",
                observed=(
                    (self.training_domain_start, training_end),
                    (self.evaluation_domain_start, evaluation_end),
                ),
                requirement="disjoint half-open intervals",
            )


def publication_study_sha256(study: PublicationStudySpec) -> str:
    """Hash the complete financial and estimator identity."""
    if type(study) is not PublicationStudySpec:
        raise PublicationSpecValueError(
            field="study",
            observed=study,
            requirement="PublicationStudySpec",
        )
    cash_lsm = study.cash_lsm
    yield_boundary = study.yield_boundary
    payload = {
        "format": _STUDY_HASH_FORMAT,
        "name": study.name,
        "variants": [variant.value for variant in study.variants],
        "exact_cash_policy": study.exact_cash_policy.value,
        "cash_maturity_boundary": study.cash_maturity_boundary.value,
        "cash_lsm": {
            "general_base_intervals": cash_lsm.general_base_intervals,
            "general_origin_exercise": cash_lsm.general_origin_exercise.value,
            "training_paths": cash_lsm.training_paths,
            "evaluation_paths": cash_lsm.evaluation_paths,
            "polynomial_degree": cash_lsm.polynomial_degree,
            "rank_tolerance": float(cash_lsm.rank_tolerance).hex(),
            "condition_limit": float(cash_lsm.condition_limit).hex(),
            "tie_rule": cash_lsm.tie_rule.value,
            "insufficient_exercise_support": cash_lsm.insufficient_exercise_support.value,
            "cash_partition_rule": cash_lsm.cash_partition_rule.value,
            "exact_cash_mesh": "calendar_owned_explicit_times_origin_excluded",
        },
        "yield_boundary": {
            "boundary_quadrature_order": yield_boundary.boundary_quadrature_order,
            "iterations": yield_boundary.iterations,
            "collocation_nodes": yield_boundary.collocation_nodes,
            "price_quadrature_order": yield_boundary.price_quadrature_order,
            "maximum_relative_boundary_residual": float(
                yield_boundary.maximum_relative_boundary_residual
            ).hex(),
            "maximum_relative_obstacle_projection": float(
                yield_boundary.maximum_relative_obstacle_projection
            ).hex(),
        },
    }
    return _sha256_json(payload)


def publication_execution_sha256(execution: PublicationReplicationExecutionSpec) -> str:
    """Hash one complete replication's device and random identity."""
    if type(execution) is not PublicationReplicationExecutionSpec:
        raise PublicationSpecValueError(
            field="execution",
            observed=execution,
            requirement="PublicationReplicationExecutionSpec",
        )
    return _sha256_json(
        {
            "format": _EXECUTION_HASH_FORMAT,
            "replication": execution.replication,
            "threads": execution.threads,
            "math_backend": execution.math_backend.value,
            "training_seed": execution.training_seed,
            "training_domain_start": execution.training_domain_start,
            "evaluation_seed": execution.evaluation_seed,
            "evaluation_domain_start": execution.evaluation_domain_start,
        }
    )


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PROJECT_PUBLICATION_YIELD_BOUNDARY_SPEC",
    "PublicationCashLsmSpec",
    "PublicationCashPartitionRule",
    "PublicationReplicationExecutionSpec",
    "PublicationSpecError",
    "PublicationSpecValueError",
    "PublicationStudySpec",
    "PublicationVariant",
    "PublicationYieldBoundarySpec",
    "publication_execution_sha256",
    "publication_study_sha256",
]
