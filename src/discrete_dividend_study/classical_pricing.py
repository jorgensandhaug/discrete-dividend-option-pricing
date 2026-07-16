"""Stateful native pricing for the immutable four-variant classical study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from finoptions import (
    CpuExecutor,
    InputError,
    Lsm,
    LsmBatchResult,
    LsmError,
    price_lsm_call_batch,
    price_lsm_call_grouped_batch,
)

from .classical_groups import PreparedClassicalStrikeGroups
from .classical_inputs import PreparedClassicalInputs
from .classical_spec import (
    PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    ClassicalExecutionSpec,
    ClassicalPopulationAddressing,
    ClassicalStudySpec,
    ClassicalVariant,
    ClassicalVariantSpec,
)
from .project_metrics import (
    ProjectMetricError,
    ProjectPricingMetrics,
    evaluate_project_pricing_f64,
)


@dataclass(slots=True)
class ClassicalPricingContractError(RuntimeError):
    """A pricing request or result violates the complete-study contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected contract."""
        return f"invalid classical pricing contract: {self.reason}"


@dataclass(slots=True)
class ClassicalPricingError(RuntimeError):
    """One named variant failed without selecting another method or returning partial output."""

    variant: ClassicalVariant
    source: Exception

    def __str__(self) -> str:
        """Attach model identity to the original typed engine or metric failure."""
        return f"classical variant {self.variant.value!r} failed: {self.source}"


@dataclass(frozen=True, slots=True)
class ClassicalVariantPricingResult:
    """One variant's immutable native diagnostics and Float64 pricing metrics."""

    variant: ClassicalVariantSpec
    native: LsmBatchResult
    metrics: ProjectPricingMetrics

    def __post_init__(self) -> None:
        """Require complete row agreement and immutable native output arrays."""
        if type(self.variant) is not ClassicalVariantSpec:
            raise ClassicalPricingContractError(reason="variant must be ClassicalVariantSpec")
        if type(self.native) is not LsmBatchResult:
            raise ClassicalPricingContractError(reason="native must be LsmBatchResult")
        if type(self.metrics) is not ProjectPricingMetrics:
            raise ClassicalPricingContractError(reason="metrics must be ProjectPricingMetrics")
        if self.native.rows != self.metrics.observations:
            raise ClassicalPricingContractError(
                reason="native rows and metric observations must agree"
            )
        arrays = (
            self.native.prices,
            self.native.sample_standard_deviations,
            self.native.standard_errors,
            self.native.origin_decisions_fitted,
            self.native.origin_exercised,
            self.native.regression_nodes,
            self.native.no_training_support_nodes,
            self.native.insufficient_exercise_support_nodes,
            self.native.minimum_in_the_money_training_paths,
            self.native.maximum_regression_condition_numbers,
            self.native.early_exercise_paths,
            self.native.expiry_paths,
        )
        if any(array.flags.writeable for array in arrays):
            raise ClassicalPricingContractError(
                reason="native classical result arrays must be immutable"
            )


@dataclass(frozen=True, slots=True)
class ClassicalStudyPricingResult:
    """The complete ordered four-way paired comparison over one immutable input population."""

    study: ClassicalStudySpec
    rows: int
    variants: tuple[ClassicalVariantPricingResult, ...]

    def __post_init__(self) -> None:
        """Reject missing, reordered, duplicated, or row-misaligned model results."""
        if type(self.study) is not ClassicalStudySpec:
            raise ClassicalPricingContractError(reason="study must be ClassicalStudySpec")
        if type(self.rows) is not int or self.rows <= 0:
            raise ClassicalPricingContractError(reason="rows must be a positive plain int")
        if type(self.variants) is not tuple or any(
            type(result) is not ClassicalVariantPricingResult for result in self.variants
        ):
            raise ClassicalPricingContractError(
                reason="variants must be tuple[ClassicalVariantPricingResult, ...]"
            )
        observed = tuple(result.variant for result in self.variants)
        if observed != self.study.variants:
            raise ClassicalPricingContractError(
                reason="variant results must exactly follow the study order"
            )
        if any(result.native.rows != self.rows for result in self.variants):
            raise ClassicalPricingContractError(
                reason="every variant result must contain the shared row count"
            )

    def variant(self, identity: ClassicalVariant) -> ClassicalVariantPricingResult:
        """Return one result by typed identity without accepting legacy aliases."""
        expected = self.study.variant(identity)
        return next(result for result in self.variants if result.variant is expected)


def run_project_classical_study(
    inputs: PreparedClassicalInputs,
    execution: ClassicalExecutionSpec,
    study: ClassicalStudySpec = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
) -> ClassicalStudyPricingResult:
    """Price all four variants sequentially through one reusable native CPU executor.

    Variants intentionally use common row-addressed random populations, enabling paired numerical
    comparisons while preserving independent training and evaluation populations within each
    variant. A failure stops the study and returns no partial result.
    """
    if type(inputs) is not PreparedClassicalInputs:
        raise ClassicalPricingContractError(reason="inputs must be PreparedClassicalInputs")
    if type(execution) is not ClassicalExecutionSpec:
        raise ClassicalPricingContractError(reason="execution must be ClassicalExecutionSpec")
    if type(study) is not ClassicalStudySpec:
        raise ClassicalPricingContractError(reason="study must be ClassicalStudySpec")
    if inputs.study != study:
        raise ClassicalPricingContractError(
            reason="prepared inputs and requested classical study must be identical"
        )
    if execution.population_addressing is not ClassicalPopulationAddressing.CONTRACT_ROWS:
        raise ClassicalPricingContractError(
            reason="row inputs require contract_rows population addressing"
        )
    execution.validate_units(inputs.rows)
    executor = execution.executor()
    estimator = study.estimator
    method = estimator.method(execution)
    results = tuple(
        _price_variant(
            inputs,
            variant,
            execution_paths=estimator.evaluation_paths,
            executor=executor,
            method=method,
        )
        for variant in study.variants
    )
    return ClassicalStudyPricingResult(study=study, rows=inputs.rows, variants=results)


def run_project_classical_grouped_study(
    inputs: PreparedClassicalStrikeGroups,
    execution: ClassicalExecutionSpec,
    study: ClassicalStudySpec = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
) -> ClassicalStudyPricingResult:
    """Price all four variants with one exact path pair per natural strike group.

    Every variant uses the same group-addressed random populations. Every strike and variant still
    fits its own stopping policy and uses a disjoint out-of-sample evaluation population. A failure
    stops the study and returns no partial result.
    """
    if type(inputs) is not PreparedClassicalStrikeGroups:
        raise ClassicalPricingContractError(reason="inputs must be PreparedClassicalStrikeGroups")
    if type(execution) is not ClassicalExecutionSpec:
        raise ClassicalPricingContractError(reason="execution must be ClassicalExecutionSpec")
    if type(study) is not ClassicalStudySpec:
        raise ClassicalPricingContractError(reason="study must be ClassicalStudySpec")
    if inputs.study != study:
        raise ClassicalPricingContractError(
            reason="prepared strike groups and requested classical study must be identical"
        )
    if execution.population_addressing is not ClassicalPopulationAddressing.NATURAL_STRIKE_GROUPS:
        raise ClassicalPricingContractError(
            reason="grouped inputs require natural_strike_groups population addressing"
        )
    execution.validate_units(inputs.groups)
    executor = execution.executor()
    estimator = study.estimator
    method = estimator.method(execution)
    results = tuple(
        _price_grouped_variant(
            inputs,
            variant,
            execution_paths=estimator.evaluation_paths,
            executor=executor,
            method=method,
        )
        for variant in study.variants
    )
    return ClassicalStudyPricingResult(study=study, rows=inputs.rows, variants=results)


def _price_variant(
    inputs: PreparedClassicalInputs,
    variant: ClassicalVariantSpec,
    *,
    execution_paths: int,
    executor: CpuExecutor,
    method: Lsm,
) -> ClassicalVariantPricingResult:
    try:
        native = price_lsm_call_batch(
            initial_spots=inputs.initial_spots,
            strikes=inputs.strikes,
            maturities=inputs.maturities,
            rates=inputs.rates,
            volatilities=inputs.volatilities,
            dividends=inputs.dividends_for(variant.variant),
            mesh=inputs.study.estimator.mesh(),
            executor=executor,
            method=method,
        )
        _validate_stopping_counts(native, inputs.rows, execution_paths)
        metrics = evaluate_project_pricing_f64(inputs.targets, native.prices)
    except (InputError, LsmError, ProjectMetricError, ClassicalPricingContractError) as error:
        raise ClassicalPricingError(variant=variant.variant, source=error) from error
    return ClassicalVariantPricingResult(variant=variant, native=native, metrics=metrics)


def _price_grouped_variant(
    inputs: PreparedClassicalStrikeGroups,
    variant: ClassicalVariantSpec,
    *,
    execution_paths: int,
    executor: CpuExecutor,
    method: Lsm,
) -> ClassicalVariantPricingResult:
    try:
        native = price_lsm_call_grouped_batch(
            groups=inputs.engine_groups_for(variant.variant),
            mesh=inputs.study.estimator.mesh(),
            executor=executor,
            method=method,
        )
        _validate_stopping_counts(native, inputs.rows, execution_paths)
        metrics = evaluate_project_pricing_f64(inputs.targets, native.prices)
    except (InputError, LsmError, ProjectMetricError, ClassicalPricingContractError) as error:
        raise ClassicalPricingError(variant=variant.variant, source=error) from error
    return ClassicalVariantPricingResult(variant=variant, native=native, metrics=metrics)


def _validate_stopping_counts(native: LsmBatchResult, rows: int, evaluation_paths: int) -> None:
    stopping_paths = native.early_exercise_paths.astype(np.uint64) + native.expiry_paths.astype(
        np.uint64
    )
    if not np.array_equal(
        stopping_paths,
        np.full(rows, evaluation_paths, dtype=np.uint64),
    ):
        raise ClassicalPricingContractError(
            reason="native stopping counts do not partition evaluation paths"
        )


__all__ = [
    "ClassicalPricingContractError",
    "ClassicalPricingError",
    "ClassicalStudyPricingResult",
    "ClassicalVariantPricingResult",
    "run_project_classical_grouped_study",
    "run_project_classical_study",
]
