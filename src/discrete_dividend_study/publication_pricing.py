"""Fail-fast pricing of one paired publication-estimator replication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from finoptions import (
    CpuExecutor,
    InputError,
    Lsm,
    LsmBatchResult,
    LsmError,
    price_lsm_call_grouped_batch,
)

from .project_metrics import (
    ProjectMetricError,
    ProjectPricingMetrics,
    evaluate_project_pricing_f64,
)
from .publication_groups import (
    PreparedPublicationPartitions,
    PublicationCashDomain,
    PublicationGroupPartition,
    PublicationPartitionError,
    publication_partitions_sha256,
)
from .publication_spec import (
    PublicationReplicationExecutionSpec,
    PublicationStudySpec,
    PublicationVariant,
)

_SHA256_HEX_LENGTH = 64


@dataclass(slots=True)
class PublicationPricingContractError(RuntimeError):
    """A publication replication request or merged result is incomplete."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected contract."""
        return f"invalid publication pricing contract: {self.reason}"


@dataclass(slots=True)
class PublicationPricingError(RuntimeError):
    """One named variant/domain failed without a retry or partial result."""

    variant: PublicationVariant
    domain: PublicationCashDomain | None
    source: Exception

    def __str__(self) -> str:
        """Attach immutable method identities to the original typed failure."""
        domain = self.domain.value if self.domain is not None else "merged_source_order"
        return (
            f"publication variant {self.variant.value!r}, domain {domain!r} failed: {self.source}"
        )


@dataclass(frozen=True, slots=True)
class PublicationVariantReplicationResult:
    """One variant's source-ordered native diagnostics and metrics."""

    variant: PublicationVariant
    native: LsmBatchResult
    metrics: ProjectPricingMetrics

    def __post_init__(self) -> None:
        """Require complete row agreement and immutable engine outputs."""
        if type(self.variant) is not PublicationVariant:
            raise PublicationPricingContractError(reason="variant must be PublicationVariant")
        if type(self.native) is not LsmBatchResult:
            raise PublicationPricingContractError(reason="native must be LsmBatchResult")
        if type(self.metrics) is not ProjectPricingMetrics:
            raise PublicationPricingContractError(reason="metrics must be ProjectPricingMetrics")
        if self.native.rows != self.metrics.observations:
            raise PublicationPricingContractError(reason="native and metric row counts differ")


@dataclass(frozen=True, slots=True)
class PublicationStudyReplicationResult:
    """One complete seed-paired four-variant publication replication."""

    study: PublicationStudySpec
    execution: PublicationReplicationExecutionSpec
    partitions_sha256: str
    rows: int
    specialized_groups: int
    general_groups: int
    variants: tuple[PublicationVariantReplicationResult, ...]

    def __post_init__(self) -> None:
        """Reject missing, reordered, or population-misaligned variant output."""
        if type(self.study) is not PublicationStudySpec:
            raise PublicationPricingContractError(reason="study must be PublicationStudySpec")
        if type(self.execution) is not PublicationReplicationExecutionSpec:
            raise PublicationPricingContractError(
                reason="execution must be PublicationReplicationExecutionSpec"
            )
        if (
            type(self.partitions_sha256) is not str
            or len(self.partitions_sha256) != _SHA256_HEX_LENGTH
        ):
            raise PublicationPricingContractError(reason="partition identity must be SHA-256")
        if type(self.rows) is not int or self.rows <= 0:
            raise PublicationPricingContractError(reason="rows must be positive")
        if type(self.specialized_groups) is not int or self.specialized_groups < 0:
            raise PublicationPricingContractError(reason="specialized_groups must be nonnegative")
        if type(self.general_groups) is not int or self.general_groups < 0:
            raise PublicationPricingContractError(reason="general_groups must be nonnegative")
        if tuple(result.variant for result in self.variants) != self.study.variants:
            raise PublicationPricingContractError(reason="variant output order differs from study")
        if any(result.native.rows != self.rows for result in self.variants):
            raise PublicationPricingContractError(reason="variant output row counts differ")

    def variant(self, identity: PublicationVariant) -> PublicationVariantReplicationResult:
        """Return one typed result without accepting a string alias."""
        if type(identity) is not PublicationVariant:
            raise PublicationPricingContractError(reason="identity must be PublicationVariant")
        return next(result for result in self.variants if result.variant is identity)


def run_publication_replication(
    inputs: PreparedPublicationPartitions,
    execution: PublicationReplicationExecutionSpec,
    study: PublicationStudySpec,
) -> PublicationStudyReplicationResult:
    """Run one complete paired replication over the immutable two-domain partition."""
    if type(inputs) is not PreparedPublicationPartitions:
        raise PublicationPricingContractError(reason="inputs must be PreparedPublicationPartitions")
    if type(execution) is not PublicationReplicationExecutionSpec:
        raise PublicationPricingContractError(
            reason="execution must be PublicationReplicationExecutionSpec"
        )
    if type(study) is not PublicationStudySpec:
        raise PublicationPricingContractError(reason="study must be PublicationStudySpec")
    if inputs.study != study:
        raise PublicationPricingContractError(reason="partition and requested studies differ")
    execution.validate_units(inputs.simulation_units)
    executor = execution.executor()
    variants = tuple(
        _price_variant(inputs, execution, study, variant, executor) for variant in study.variants
    )
    return PublicationStudyReplicationResult(
        study=study,
        execution=execution,
        partitions_sha256=publication_partitions_sha256(inputs),
        rows=inputs.original_rows,
        specialized_groups=inputs.specialized.groups,
        general_groups=inputs.general.groups,
        variants=variants,
    )


def _price_variant(
    inputs: PreparedPublicationPartitions,
    execution: PublicationReplicationExecutionSpec,
    study: PublicationStudySpec,
    variant: PublicationVariant,
    executor: CpuExecutor,
) -> PublicationVariantReplicationResult:
    pieces: list[tuple[PublicationGroupPartition, LsmBatchResult]] = []
    unit_offset = 0
    for partition in inputs.ordered():
        if partition.groups:
            method = study.estimator.method(execution, unit_offset=unit_offset)
            pieces.append(
                (
                    partition,
                    _price_partition(
                        partition,
                        variant,
                        study,
                        executor=executor,
                        method=method,
                    ),
                )
            )
        unit_offset += partition.groups
    try:
        native = _merge_partition_results(pieces, inputs.original_rows)
        metrics = evaluate_project_pricing_f64(inputs.targets, native.prices)
    except (ProjectMetricError, PublicationPricingContractError) as error:
        raise PublicationPricingError(
            variant=variant,
            domain=None,
            source=error,
        ) from error
    return PublicationVariantReplicationResult(variant=variant, native=native, metrics=metrics)


def _price_partition(
    partition: PublicationGroupPartition,
    variant: PublicationVariant,
    study: PublicationStudySpec,
    *,
    executor: CpuExecutor,
    method: Lsm,
) -> LsmBatchResult:
    mesh = (
        study.estimator.specialized_cash_mesh()
        if variant.uses_cash_schedule and partition.domain is PublicationCashDomain.SPECIALIZED
        else study.estimator.general_mesh()
    )
    try:
        native = price_lsm_call_grouped_batch(
            groups=partition.engine_groups_for(variant),
            mesh=mesh,
            executor=executor,
            method=method,
        )
        _validate_stopping_counts(native, partition.rows, study.estimator.evaluation_paths)
    except (
        InputError,
        LsmError,
        PublicationPartitionError,
        PublicationPricingContractError,
    ) as error:
        raise PublicationPricingError(
            variant=variant,
            domain=partition.domain,
            source=error,
        ) from error
    return native


def _merge_partition_results(
    pieces: list[tuple[PublicationGroupPartition, LsmBatchResult]],
    rows: int,
) -> LsmBatchResult:
    prices = np.empty(rows, dtype=np.float64)
    sample_standard_deviations = np.empty(rows, dtype=np.float64)
    standard_errors = np.empty(rows, dtype=np.float64)
    origin_decisions_fitted = np.empty(rows, dtype=np.bool_)
    origin_exercised = np.empty(rows, dtype=np.bool_)
    regression_nodes = np.empty(rows, dtype=np.uint32)
    no_training_support_nodes = np.empty(rows, dtype=np.uint32)
    insufficient_exercise_support_nodes = np.empty(rows, dtype=np.uint32)
    minimum_in_the_money_training_paths = np.empty(rows, dtype=np.uint32)
    maximum_regression_condition_numbers = np.empty(rows, dtype=np.float64)
    early_exercise_paths = np.empty(rows, dtype=np.uint32)
    expiry_paths = np.empty(rows, dtype=np.uint32)
    assigned = np.zeros(rows, dtype=np.bool_)
    for partition, native in pieces:
        indices = partition.original_row_indices
        if np.any(assigned[indices]):
            raise PublicationPricingContractError(reason="partition output rows overlap")
        if native.rows != indices.size:
            raise PublicationPricingContractError(reason="partition output length changed")
        prices[indices] = native.prices
        sample_standard_deviations[indices] = native.sample_standard_deviations
        standard_errors[indices] = native.standard_errors
        origin_decisions_fitted[indices] = native.origin_decisions_fitted
        origin_exercised[indices] = native.origin_exercised
        regression_nodes[indices] = native.regression_nodes
        no_training_support_nodes[indices] = native.no_training_support_nodes
        insufficient_exercise_support_nodes[indices] = native.insufficient_exercise_support_nodes
        minimum_in_the_money_training_paths[indices] = native.minimum_in_the_money_training_paths
        maximum_regression_condition_numbers[indices] = native.maximum_regression_condition_numbers
        early_exercise_paths[indices] = native.early_exercise_paths
        expiry_paths[indices] = native.expiry_paths
        assigned[indices] = True
    if not bool(np.all(assigned)):
        raise PublicationPricingContractError(reason="partition output rows are incomplete")
    arrays = (
        prices,
        sample_standard_deviations,
        standard_errors,
        origin_decisions_fitted,
        origin_exercised,
        regression_nodes,
        no_training_support_nodes,
        insufficient_exercise_support_nodes,
        minimum_in_the_money_training_paths,
        maximum_regression_condition_numbers,
        early_exercise_paths,
        expiry_paths,
    )
    for array in arrays:
        array.flags.writeable = False
    return LsmBatchResult(
        prices=prices,
        sample_standard_deviations=sample_standard_deviations,
        standard_errors=standard_errors,
        origin_decisions_fitted=origin_decisions_fitted,
        origin_exercised=origin_exercised,
        regression_nodes=regression_nodes,
        no_training_support_nodes=no_training_support_nodes,
        insufficient_exercise_support_nodes=insufficient_exercise_support_nodes,
        minimum_in_the_money_training_paths=minimum_in_the_money_training_paths,
        maximum_regression_condition_numbers=maximum_regression_condition_numbers,
        early_exercise_paths=early_exercise_paths,
        expiry_paths=expiry_paths,
    )


def _validate_stopping_counts(native: LsmBatchResult, rows: int, evaluation_paths: int) -> None:
    stopping = native.early_exercise_paths.astype(np.uint64) + native.expiry_paths.astype(np.uint64)
    if not np.array_equal(stopping, np.full(rows, evaluation_paths, dtype=np.uint64)):
        raise PublicationPricingContractError(reason="stopping counts do not partition paths")


__all__ = [
    "PublicationPricingContractError",
    "PublicationPricingError",
    "PublicationStudyReplicationResult",
    "PublicationVariantReplicationResult",
    "run_publication_replication",
]
