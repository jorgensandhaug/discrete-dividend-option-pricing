"""Fail-fast pricing of one paired publication-estimator replication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from finoptions import (
    AmericanOptionError,
    AndersenLake,
    AndersenLakeBatchResult,
    CpuExecutor,
    InputError,
    Lsm,
    LsmBatchResult,
    LsmError,
    price_andersen_lake_call_grouped,
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

type PublicationNativeResult = LsmBatchResult | AndersenLakeBatchResult


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
    native: PublicationNativeResult
    metrics: ProjectPricingMetrics

    def __post_init__(self) -> None:
        """Require complete row agreement and immutable engine outputs."""
        if type(self.variant) is not PublicationVariant:
            raise PublicationPricingContractError(reason="variant must be PublicationVariant")
        expected = LsmBatchResult if self.variant.uses_cash_schedule else AndersenLakeBatchResult
        if type(self.native) is not expected:
            raise PublicationPricingContractError(
                reason=f"native must be {expected.__name__} for {self.variant.value}"
            )
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
    exact_groups: int
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
        if type(self.exact_groups) is not int or self.exact_groups < 0:
            raise PublicationPricingContractError(reason="exact_groups must be nonnegative")
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
    yield_method = study.yield_boundary.method()
    variants = tuple(
        _price_variant(inputs, execution, study, variant, executor, yield_method)
        for variant in study.variants
    )
    return PublicationStudyReplicationResult(
        study=study,
        execution=execution,
        partitions_sha256=publication_partitions_sha256(inputs),
        rows=inputs.original_rows,
        exact_groups=inputs.exact.groups,
        general_groups=inputs.general.groups,
        variants=variants,
    )


def _price_variant(  # noqa: PLR0913 - method identities stay explicit at dispatch
    inputs: PreparedPublicationPartitions,
    execution: PublicationReplicationExecutionSpec,
    study: PublicationStudySpec,
    variant: PublicationVariant,
    executor: CpuExecutor,
    yield_method: AndersenLake,
) -> PublicationVariantReplicationResult:
    native = (
        _price_cash_variant(inputs, execution, study, variant, executor)
        if variant.uses_cash_schedule
        else _price_yield_variant(inputs, variant, executor, yield_method)
    )
    try:
        metrics = evaluate_project_pricing_f64(inputs.targets, native.prices)
    except ProjectMetricError as error:
        raise PublicationPricingError(
            variant=variant,
            domain=None,
            source=error,
        ) from error
    return PublicationVariantReplicationResult(variant=variant, native=native, metrics=metrics)


def _price_cash_variant(
    inputs: PreparedPublicationPartitions,
    execution: PublicationReplicationExecutionSpec,
    study: PublicationStudySpec,
    variant: PublicationVariant,
    executor: CpuExecutor,
) -> LsmBatchResult:
    pieces: list[tuple[PublicationGroupPartition, LsmBatchResult]] = []
    unit_offset = 0
    for partition in inputs.ordered():
        if partition.groups:
            method = study.cash_lsm.method(execution, unit_offset=unit_offset)
            pieces.append(
                (
                    partition,
                    _price_cash_partition(
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
        return _merge_lsm_partition_results(pieces, inputs.original_rows)
    except PublicationPricingContractError as error:
        raise PublicationPricingError(
            variant=variant,
            domain=None,
            source=error,
        ) from error


def _price_cash_partition(
    partition: PublicationGroupPartition,
    variant: PublicationVariant,
    study: PublicationStudySpec,
    *,
    executor: CpuExecutor,
    method: Lsm,
) -> LsmBatchResult:
    if variant.uses_cash_schedule and partition.domain is PublicationCashDomain.EXACT:
        if partition.exercise_times is None:
            raise PublicationPricingContractError(
                reason="a nonempty exact cash partition has no exercise schedule"
            )
        mesh = study.cash_lsm.exact_cash_mesh(partition.exercise_times)
    else:
        mesh = study.cash_lsm.general_mesh()
    try:
        native = price_lsm_call_grouped_batch(
            groups=partition.engine_groups_for(variant),
            mesh=mesh,
            executor=executor,
            method=method,
        )
        _validate_stopping_counts(native, partition.rows, study.cash_lsm.evaluation_paths)
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


def _price_yield_variant(
    inputs: PreparedPublicationPartitions,
    variant: PublicationVariant,
    executor: CpuExecutor,
    method: AndersenLake,
) -> AndersenLakeBatchResult:
    pieces = [
        (
            partition,
            _price_yield_partition(
                partition,
                variant,
                executor=executor,
                method=method,
            ),
        )
        for partition in inputs.ordered()
        if partition.groups
    ]
    try:
        return _merge_yield_partition_results(
            pieces,
            groups=inputs.original_groups,
            rows=inputs.original_rows,
        )
    except PublicationPricingContractError as error:
        raise PublicationPricingError(variant=variant, domain=None, source=error) from error


def _price_yield_partition(
    partition: PublicationGroupPartition,
    variant: PublicationVariant,
    *,
    executor: CpuExecutor,
    method: AndersenLake,
) -> AndersenLakeBatchResult:
    if variant is PublicationVariant.FORECAST_YIELD:
        yields = partition.forecast_yields
    elif variant is PublicationVariant.HISTORICAL_YIELD:
        yields = partition.historical_yields
    else:
        raise PublicationPricingContractError(
            reason=f"{variant.value} is not a continuous-yield variant"
        )
    try:
        return price_andersen_lake_call_grouped(
            group_offsets=np.ascontiguousarray(partition.group_offsets, dtype=np.int64),
            group_spots=partition.initial_spots,
            group_maturities=partition.maturities,
            group_constant_continuously_compounded_rates=partition.rates,
            group_annualized_dividend_yields=yields,
            group_constant_volatilities=partition.volatilities,
            strikes=partition.strikes,
            executor=executor,
            method=method,
        )
    except (AmericanOptionError, InputError) as error:
        raise PublicationPricingError(
            variant=variant,
            domain=partition.domain,
            source=error,
        ) from error


def _merge_lsm_partition_results(
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


def _merge_yield_partition_results(
    pieces: list[tuple[PublicationGroupPartition, AndersenLakeBatchResult]],
    *,
    groups: int,
    rows: int,
) -> AndersenLakeBatchResult:
    prices = np.empty(rows, dtype=np.float64)
    exercised_at_origin = np.empty(rows, dtype=np.bool_)
    maximum_relative_boundary_residuals = np.empty(groups, dtype=np.float64)
    maximum_relative_obstacle_projections = np.empty(groups, dtype=np.float64)
    used_european_identity = np.empty(groups, dtype=np.bool_)
    assigned_rows = np.zeros(rows, dtype=np.bool_)
    assigned_groups = np.zeros(groups, dtype=np.bool_)
    for partition, native in pieces:
        row_indices = partition.original_row_indices
        group_indices = partition.original_group_indices
        if np.any(assigned_rows[row_indices]) or np.any(assigned_groups[group_indices]):
            raise PublicationPricingContractError(reason="yield partition outputs overlap")
        if native.rows != row_indices.size or native.groups != group_indices.size:
            raise PublicationPricingContractError(reason="yield partition output shape changed")
        prices[row_indices] = native.prices
        exercised_at_origin[row_indices] = native.exercised_at_origin
        maximum_relative_boundary_residuals[group_indices] = (
            native.maximum_relative_boundary_residuals
        )
        maximum_relative_obstacle_projections[group_indices] = (
            native.maximum_relative_obstacle_projections
        )
        used_european_identity[group_indices] = native.used_european_identity
        assigned_rows[row_indices] = True
        assigned_groups[group_indices] = True
    if not bool(np.all(assigned_rows)) or not bool(np.all(assigned_groups)):
        raise PublicationPricingContractError(reason="yield partition outputs are incomplete")
    arrays = (
        prices,
        exercised_at_origin,
        maximum_relative_boundary_residuals,
        maximum_relative_obstacle_projections,
        used_european_identity,
    )
    for array in arrays:
        array.flags.writeable = False
    return AndersenLakeBatchResult(
        prices=prices,
        exercised_at_origin=exercised_at_origin,
        maximum_relative_boundary_residuals=maximum_relative_boundary_residuals,
        maximum_relative_obstacle_projections=maximum_relative_obstacle_projections,
        used_european_identity=used_european_identity,
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
