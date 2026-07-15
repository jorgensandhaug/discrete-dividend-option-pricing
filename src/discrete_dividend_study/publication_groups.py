"""Immutable cash-domain partitioning of verified natural strike groups."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt
from finoptions import (
    BatchDividends,
    CashDividendBatch,
    ContinuousDividendYieldBatch,
    EscrowedCashDividendBatch,
    EscrowedVolatilityConvention,
    LsmStrikeGroups,
    RaggedCashDividends,
    RaggedExerciseTimes,
)

from .classical_groups import (
    PreparedClassicalStrikeGroups,
    prepared_classical_strike_groups_sha256,
)
from .publication_exercise import (
    PreparedPublicationExerciseSchedules,
    publication_exercise_schedules_sha256,
)
from .publication_spec import (
    PublicationCashPartitionRule,
    PublicationStudySpec,
    PublicationVariant,
    publication_study_sha256,
)

Float64Vector = npt.NDArray[np.float64]
UInt32Vector = npt.NDArray[np.uint32]

_FORMAT: Final = "project-publication-cash-domain-partitions-v2"
_FLOAT64: Final = np.dtype(np.float64)
_UINT32: Final = np.dtype(np.uint32)
_INT64: Final = np.dtype(np.int64)
_SHA256_HEX_LENGTH: Final = 64


class PublicationCashDomain(StrEnum):
    """The two explicit cash-call theorem domains."""

    EXACT = "nonnegative_rate_calendar_owned_exact_cash"
    GENERAL = "negative_rate_general"


@dataclass(slots=True)
class PublicationPartitionError(ValueError):
    """A publication partition is incomplete, overlapping, or financially misclassified."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected partition contract."""
        return f"invalid publication partition: {self.reason}"


class _HashWriter(Protocol):
    """Minimal streaming hash interface."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(frozen=True, slots=True, eq=False)
class PublicationGroupPartition:
    """One compact domain with immutable maps back to the source group and row order."""

    domain: PublicationCashDomain
    study: PublicationStudySpec
    original_group_indices: UInt32Vector
    original_row_indices: UInt32Vector
    group_offsets: UInt32Vector
    initial_spots: Float64Vector
    strikes: Float64Vector
    maturities: Float64Vector
    rates: Float64Vector
    volatilities: Float64Vector
    forecast_yields: Float64Vector
    historical_yields: Float64Vector
    targets: Float64Vector
    cash_dividends: RaggedCashDividends | None
    exercise_times: RaggedExerciseTimes | None

    def __post_init__(self) -> None:
        """Reject malformed compact axes, mappings, or theorem-domain membership."""
        _validate_partition_identity(self)
        _validate_partition_vectors(self)
        _validate_partition_shapes(self)
        _validate_partition_domain(self)
        _freeze_partition(self)

    @property
    def groups(self) -> int:
        """Return this partition's independently simulated state count."""
        return int(self.initial_spots.size)

    @property
    def rows(self) -> int:
        """Return this partition's strike-specific output count."""
        return int(self.strikes.size)

    def dividends_for(self, variant: PublicationVariant) -> BatchDividends:
        """Build the declared group-aligned dividend representation."""
        if self.cash_dividends is None:
            raise PublicationPartitionError(reason="an empty partition has no dividend request")
        if variant is PublicationVariant.CASH_JUMP_LIQUIDATOR:
            return CashDividendBatch(
                schedules=self.cash_dividends,
                policy=self.study.exact_cash_policy,
                maturity_boundary=self.study.cash_maturity_boundary,
            )
        if variant is PublicationVariant.ESCROWED_CASH_SPOT_MATCHED:
            return EscrowedCashDividendBatch(
                schedules=self.cash_dividends,
                volatility_convention=EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN,
                maturity_boundary=self.study.cash_maturity_boundary,
            )
        if variant is PublicationVariant.FORECAST_YIELD:
            return ContinuousDividendYieldBatch(annualized_decimals=self.forecast_yields)
        if variant is PublicationVariant.HISTORICAL_YIELD:
            return ContinuousDividendYieldBatch(annualized_decimals=self.historical_yields)
        raise PublicationPartitionError(reason=f"unsupported publication variant {variant!r}")

    def engine_groups_for(self, variant: PublicationVariant) -> LsmStrikeGroups:
        """Build the zero-copy engine facade for a nonempty partition."""
        if self.groups == 0:
            raise PublicationPartitionError(reason="an empty partition has no engine request")
        return LsmStrikeGroups(
            group_offsets=self.group_offsets,
            initial_spots=self.initial_spots,
            strikes=self.strikes,
            maturities=self.maturities,
            rates=self.rates,
            volatilities=self.volatilities,
            dividends=self.dividends_for(variant),
        )


def _validate_partition_identity(partition: PublicationGroupPartition) -> None:
    if type(partition.domain) is not PublicationCashDomain:
        raise PublicationPartitionError(reason="domain must be PublicationCashDomain")
    if type(partition.study) is not PublicationStudySpec:
        raise PublicationPartitionError(reason="study must be PublicationStudySpec")


def _validate_partition_vectors(partition: PublicationGroupPartition) -> None:
    for field, array, dtype in (
        ("original_group_indices", partition.original_group_indices, _UINT32),
        ("original_row_indices", partition.original_row_indices, _UINT32),
        ("group_offsets", partition.group_offsets, _UINT32),
        ("initial_spots", partition.initial_spots, _FLOAT64),
        ("strikes", partition.strikes, _FLOAT64),
        ("maturities", partition.maturities, _FLOAT64),
        ("rates", partition.rates, _FLOAT64),
        ("volatilities", partition.volatilities, _FLOAT64),
        ("forecast_yields", partition.forecast_yields, _FLOAT64),
        ("historical_yields", partition.historical_yields, _FLOAT64),
        ("targets", partition.targets, _FLOAT64),
    ):
        _validate_vector(array, field=field, dtype=dtype)


def _validate_partition_shapes(partition: PublicationGroupPartition) -> None:
    groups = partition.groups
    rows = partition.rows
    if partition.group_offsets.shape != (groups + 1,):
        raise PublicationPartitionError(reason="group offset count does not match groups")
    if partition.group_offsets[0] != 0 or partition.group_offsets[-1] != rows:
        raise PublicationPartitionError(reason="group offsets do not span partition rows")
    if groups and np.any(partition.group_offsets[1:] <= partition.group_offsets[:-1]):
        raise PublicationPartitionError(reason="partition contains an empty group")
    group_arrays = (
        partition.original_group_indices,
        partition.initial_spots,
        partition.maturities,
        partition.rates,
        partition.volatilities,
        partition.forecast_yields,
        partition.historical_yields,
    )
    row_arrays = (partition.original_row_indices, partition.strikes, partition.targets)
    if any(array.size != groups for array in group_arrays):
        raise PublicationPartitionError(reason="one group-aligned vector has the wrong length")
    if any(array.size != rows for array in row_arrays):
        raise PublicationPartitionError(reason="one row-aligned vector has the wrong length")
    _validate_partition_schedule_shapes(partition)


def _validate_partition_schedule_shapes(partition: PublicationGroupPartition) -> None:
    """Require dividend and exercise buffers exactly for their declared domains."""
    groups = partition.groups
    if groups == 0:
        if partition.cash_dividends is not None or partition.exercise_times is not None:
            raise PublicationPartitionError(
                reason="an empty partition must not own cash or exercise buffers"
            )
        return
    if type(partition.cash_dividends) is not RaggedCashDividends:
        raise PublicationPartitionError(reason="a nonempty partition requires cash buffers")
    if partition.cash_dividends.rows != groups:
        raise PublicationPartitionError(reason="cash schedules do not match group count")
    if partition.domain is PublicationCashDomain.EXACT:
        if type(partition.exercise_times) is not RaggedExerciseTimes:
            raise PublicationPartitionError(reason="the exact domain requires exercise schedules")
        if partition.exercise_times.rows != groups:
            raise PublicationPartitionError(reason="exercise schedules do not match group count")
    elif partition.exercise_times is not None:
        raise PublicationPartitionError(
            reason="only the nonempty exact domain may own exercise schedules"
        )


def _validate_partition_domain(partition: PublicationGroupPartition) -> None:
    if partition.groups == 0:
        return
    eligibility = np.greater_equal(partition.rates, 0.0)
    expected = eligibility if partition.domain is PublicationCashDomain.EXACT else ~eligibility
    if not bool(np.all(expected)):
        raise PublicationPartitionError(reason=f"misclassified {partition.domain.value} group")


def _freeze_partition(partition: PublicationGroupPartition) -> None:
    arrays = (
        partition.original_group_indices,
        partition.original_row_indices,
        partition.group_offsets,
        partition.initial_spots,
        partition.strikes,
        partition.maturities,
        partition.rates,
        partition.volatilities,
        partition.forecast_yields,
        partition.historical_yields,
        partition.targets,
    )
    if partition.cash_dividends is not None:
        arrays = (
            *arrays,
            partition.cash_dividends.offsets,
            partition.cash_dividends.times,
            partition.cash_dividends.amounts,
        )
    if partition.exercise_times is not None:
        arrays = (
            *arrays,
            partition.exercise_times.offsets,
            partition.exercise_times.times,
        )
    for array in arrays:
        array.flags.writeable = False


@dataclass(frozen=True, slots=True, eq=False)
class PreparedPublicationPartitions:
    """Complete disjoint partition plus source-order reconstruction maps."""

    study: PublicationStudySpec
    source_prepared_sha256: str
    exercise_schedules_sha256: str
    original_groups: int
    original_rows: int
    targets: Float64Vector
    exact: PublicationGroupPartition
    general: PublicationGroupPartition

    def __post_init__(self) -> None:
        """Prove exact group/row coverage once before any pricing call."""
        _validate_prepared_partition_header(self)
        _validate_prepared_partition_members(self)
        _validate_complete_partition_maps(self)
        self.targets.flags.writeable = False

    @property
    def simulation_units(self) -> int:
        """Return the complete reordered group count."""
        return self.exact.groups + self.general.groups

    def ordered(self) -> tuple[PublicationGroupPartition, PublicationGroupPartition]:
        """Return the immutable domain order used by random-domain addressing."""
        return (self.exact, self.general)


def _validate_prepared_partition_header(inputs: PreparedPublicationPartitions) -> None:
    if type(inputs.study) is not PublicationStudySpec:
        raise PublicationPartitionError(reason="study must be PublicationStudySpec")
    if (
        type(inputs.source_prepared_sha256) is not str
        or len(inputs.source_prepared_sha256) != _SHA256_HEX_LENGTH
    ):
        raise PublicationPartitionError(reason="source prepared identity must be SHA-256")
    if (
        type(inputs.exercise_schedules_sha256) is not str
        or len(inputs.exercise_schedules_sha256) != _SHA256_HEX_LENGTH
    ):
        raise PublicationPartitionError(reason="exercise schedule identity must be SHA-256")
    if type(inputs.original_groups) is not int or inputs.original_groups <= 0:
        raise PublicationPartitionError(reason="original_groups must be positive")
    if type(inputs.original_rows) is not int or inputs.original_rows <= 0:
        raise PublicationPartitionError(reason="original_rows must be positive")
    _validate_vector(inputs.targets, field="targets", dtype=_FLOAT64)
    if inputs.targets.size != inputs.original_rows:
        raise PublicationPartitionError(reason="source-order targets have the wrong length")


def _validate_prepared_partition_members(inputs: PreparedPublicationPartitions) -> None:
    if type(inputs.exact) is not PublicationGroupPartition:
        raise PublicationPartitionError(reason="exact must be PublicationGroupPartition")
    if type(inputs.general) is not PublicationGroupPartition:
        raise PublicationPartitionError(reason="general must be PublicationGroupPartition")
    if inputs.exact.domain is not PublicationCashDomain.EXACT:
        raise PublicationPartitionError(reason="exact partition has the wrong identity")
    if inputs.general.domain is not PublicationCashDomain.GENERAL:
        raise PublicationPartitionError(reason="general partition has the wrong identity")
    if inputs.exact.study != inputs.study or inputs.general.study != inputs.study:
        raise PublicationPartitionError(reason="partition study identities differ")


def _validate_complete_partition_maps(inputs: PreparedPublicationPartitions) -> None:
    group_map = np.concatenate(
        (inputs.exact.original_group_indices, inputs.general.original_group_indices)
    )
    row_map = np.concatenate(
        (inputs.exact.original_row_indices, inputs.general.original_row_indices)
    )
    if not np.array_equal(
        np.sort(group_map),
        np.arange(inputs.original_groups, dtype=np.uint32),
    ):
        raise PublicationPartitionError(reason="group partitions are not a complete permutation")
    if not np.array_equal(
        np.sort(row_map),
        np.arange(inputs.original_rows, dtype=np.uint32),
    ):
        raise PublicationPartitionError(reason="row partitions are not a complete permutation")


def prepare_publication_partitions(
    inputs: PreparedClassicalStrikeGroups,
    exercise_schedules: PreparedPublicationExerciseSchedules,
    study: PublicationStudySpec,
) -> PreparedPublicationPartitions:
    """Partition verified source groups without inspecting option-price targets."""
    if type(inputs) is not PreparedClassicalStrikeGroups:
        raise PublicationPartitionError(reason="inputs must be PreparedClassicalStrikeGroups")
    if type(study) is not PublicationStudySpec:
        raise PublicationPartitionError(reason="study must be PublicationStudySpec")
    if type(exercise_schedules) is not PreparedPublicationExerciseSchedules:
        raise PublicationPartitionError(
            reason="exercise_schedules must be PreparedPublicationExerciseSchedules"
        )
    if (
        study.estimator.cash_partition_rule
        is not PublicationCashPartitionRule.NONNEGATIVE_RATE_EXACT_CALENDAR
    ):
        raise PublicationPartitionError(reason="unsupported cash partition rule")
    source_sha256 = prepared_classical_strike_groups_sha256(inputs)
    if exercise_schedules.source_prepared_sha256 != source_sha256:
        raise PublicationPartitionError(reason="exercise schedules belong to another source")
    if exercise_schedules.groups != inputs.groups:
        raise PublicationPartitionError(reason="exercise schedule group count differs")
    eligible = np.greater_equal(inputs.rates, 0.0)
    exact = _subset_partition(
        inputs,
        exercise_schedules,
        study,
        PublicationCashDomain.EXACT,
        eligible,
    )
    general = _subset_partition(
        inputs,
        exercise_schedules,
        study,
        PublicationCashDomain.GENERAL,
        ~eligible,
    )
    targets = inputs.targets.view()
    targets.flags.writeable = False
    return PreparedPublicationPartitions(
        study=study,
        source_prepared_sha256=source_sha256,
        exercise_schedules_sha256=publication_exercise_schedules_sha256(exercise_schedules),
        original_groups=inputs.groups,
        original_rows=inputs.rows,
        targets=targets,
        exact=exact,
        general=general,
    )


def publication_partitions_sha256(inputs: PreparedPublicationPartitions) -> str:
    """Hash source identity, study, domain order, and exact reconstruction maps."""
    if type(inputs) is not PreparedPublicationPartitions:
        raise PublicationPartitionError(reason="inputs must be PreparedPublicationPartitions")
    hasher = hashlib.sha256()
    _hash_text(hasher, _FORMAT)
    _hash_text(hasher, publication_study_sha256(inputs.study))
    _hash_text(hasher, inputs.source_prepared_sha256)
    _hash_text(hasher, inputs.exercise_schedules_sha256)
    for partition in inputs.ordered():
        _hash_text(hasher, partition.domain.value)
        _hash_array(hasher, "original_group_indices", partition.original_group_indices)
        _hash_array(hasher, "original_row_indices", partition.original_row_indices)
        if partition.exercise_times is not None:
            _hash_array(hasher, "exercise_offsets", partition.exercise_times.offsets)
            _hash_array(hasher, "exercise_times", partition.exercise_times.times)
    return hasher.hexdigest()


def _subset_partition(
    source: PreparedClassicalStrikeGroups,
    exercise_schedules: PreparedPublicationExerciseSchedules,
    study: PublicationStudySpec,
    domain: PublicationCashDomain,
    group_mask: npt.NDArray[np.bool_],
) -> PublicationGroupPartition:
    group_indices = np.flatnonzero(group_mask).astype(np.uint32, copy=False)
    group_sizes = np.diff(source.group_offsets).astype(np.uint32, copy=False)
    selected_group_sizes = group_sizes[group_indices]
    group_offsets = np.empty(group_indices.size + 1, dtype=np.uint32)
    group_offsets[0] = 0
    if group_indices.size:
        group_offsets[1:] = np.cumsum(selected_group_sizes, dtype=np.uint32)

    row_mask = np.repeat(group_mask, group_sizes.astype(np.int64, copy=False))
    row_indices = np.flatnonzero(row_mask).astype(np.uint32, copy=False)
    cash_sizes = np.diff(source.cash_dividends.offsets).astype(np.int64, copy=False)
    cash_offsets = np.empty(group_indices.size + 1, dtype=np.int64)
    cash_offsets[0] = 0
    if group_indices.size:
        cash_offsets[1:] = np.cumsum(cash_sizes[group_indices], dtype=np.int64)
    event_mask = np.repeat(group_mask, cash_sizes)
    cash_times = np.ascontiguousarray(source.cash_dividends.times[event_mask])
    cash_amounts = np.ascontiguousarray(source.cash_dividends.amounts[event_mask])
    cash = (
        RaggedCashDividends(
            offsets=cash_offsets,
            times=cash_times,
            amounts=cash_amounts,
        )
        if group_indices.size
        else None
    )
    exercise: RaggedExerciseTimes | None = None
    if group_indices.size and domain is PublicationCashDomain.EXACT:
        exercise_sizes = np.diff(exercise_schedules.schedules.offsets).astype(
            np.int64,
            copy=False,
        )
        exercise_offsets = np.empty(group_indices.size + 1, dtype=np.int64)
        exercise_offsets[0] = 0
        exercise_offsets[1:] = np.cumsum(exercise_sizes[group_indices], dtype=np.int64)
        exercise_event_mask = np.repeat(group_mask, exercise_sizes)
        exercise = RaggedExerciseTimes(
            offsets=exercise_offsets,
            times=np.ascontiguousarray(exercise_schedules.schedules.times[exercise_event_mask]),
        )
    group_arrays = tuple(
        np.ascontiguousarray(array[group_indices])
        for array in (
            source.initial_spots,
            source.maturities,
            source.rates,
            source.volatilities,
            source.forecast_yields,
            source.historical_yields,
        )
    )
    row_arrays = tuple(
        np.ascontiguousarray(array[row_indices]) for array in (source.strikes, source.targets)
    )
    return PublicationGroupPartition(
        domain=domain,
        study=study,
        original_group_indices=np.ascontiguousarray(group_indices),
        original_row_indices=np.ascontiguousarray(row_indices),
        group_offsets=group_offsets,
        initial_spots=group_arrays[0],
        strikes=row_arrays[0],
        maturities=group_arrays[1],
        rates=group_arrays[2],
        volatilities=group_arrays[3],
        forecast_yields=group_arrays[4],
        historical_yields=group_arrays[5],
        targets=row_arrays[1],
        cash_dividends=cash,
        exercise_times=exercise,
    )


def _validate_vector(array: object, *, field: str, dtype: np.dtype[np.generic]) -> None:
    if type(array) is not np.ndarray:
        raise PublicationPartitionError(reason=f"{field} must be a plain numpy.ndarray")
    typed = cast("npt.NDArray[np.generic]", array)
    if typed.dtype != dtype or typed.ndim != 1 or not typed.flags.c_contiguous:
        raise PublicationPartitionError(
            reason=f"{field} must be a C-contiguous one-dimensional {dtype.name} vector"
        )


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode()
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _hash_array(hasher: _HashWriter, field: str, array: npt.NDArray[np.generic]) -> None:
    _hash_text(hasher, field)
    _hash_text(hasher, array.dtype.str)
    hasher.update(array.size.to_bytes(8, byteorder="little", signed=False))
    hasher.update(memoryview(array).cast("B"))


__all__ = [
    "PreparedPublicationPartitions",
    "PublicationCashDomain",
    "PublicationGroupPartition",
    "PublicationPartitionError",
    "prepare_publication_partitions",
    "publication_partitions_sha256",
]
