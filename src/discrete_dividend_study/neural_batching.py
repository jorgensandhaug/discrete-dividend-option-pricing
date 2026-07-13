"""Deterministic fixed-shape batch indices with explicit final-row masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

Int64Vector = npt.NDArray[np.int64]
Int64Matrix = npt.NDArray[np.int64]
BoolVector = npt.NDArray[np.bool_]
BoolMatrix = npt.NDArray[np.bool_]

_MAX_SEED: Final = (2**64) - 1


@dataclass(frozen=True, slots=True)
class NeuralBatchPlanError(ValueError):
    """A fixed-batch identity or generated epoch layout is invalid."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected batching contract."""
        return f"invalid neural batch plan: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralBatchPlan:
    """Immutable row count, physical batch shape, and shuffle-stream identity."""

    rows: int
    batch_size: int
    shuffle: bool
    seed: int | None

    def __post_init__(self) -> None:
        """Require a nonempty dataset and an explicit seed only when shuffling."""
        if type(self.rows) is not int or self.rows <= 0:
            raise NeuralBatchPlanError(reason="rows must be a positive plain int")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise NeuralBatchPlanError(reason="batch_size must be a positive plain int")
        if type(self.shuffle) is not bool:
            raise NeuralBatchPlanError(reason="shuffle must be a plain bool")
        if self.shuffle:
            _validate_seed(self.seed)
        elif self.seed is not None:
            raise NeuralBatchPlanError(reason="an unshuffled plan must not carry a seed")

    @property
    def batches(self) -> int:
        """Return the number of fixed physical batches, including a partial final batch."""
        return (self.rows + self.batch_size - 1) // self.batch_size


@dataclass(frozen=True, slots=True)
class FixedBatchIndices:
    """One fixed-width index row and its exact valid-sample mask."""

    indices: Int64Vector
    valid: BoolVector

    def __post_init__(self) -> None:
        """Reject writable, noncontiguous, mismatched, or empty arrays."""
        _validate_array(self.indices, field="indices", dtype=np.dtype(np.int64), dimensions=1)
        _validate_array(self.valid, field="valid", dtype=np.dtype(np.bool_), dimensions=1)
        if self.indices.size == 0 or self.valid.shape != self.indices.shape:
            raise NeuralBatchPlanError(reason="indices and valid must share one nonempty shape")

    @property
    def valid_rows(self) -> int:
        """Return the number of real observations represented by this physical batch."""
        return int(np.count_nonzero(self.valid))


@dataclass(frozen=True, slots=True)
class NeuralEpochBatchIndices:
    """One complete deterministic epoch of fixed-shape batches."""

    plan: NeuralBatchPlan
    epoch: int
    indices: Int64Matrix
    valid: BoolMatrix

    def __post_init__(self) -> None:
        """Prove exact once-only row coverage and canonical padding."""
        if type(self.plan) is not NeuralBatchPlan:
            raise NeuralBatchPlanError(reason="plan must be NeuralBatchPlan")
        if type(self.epoch) is not int or self.epoch < 0:
            raise NeuralBatchPlanError(reason="epoch must be a nonnegative plain int")
        _validate_array(self.indices, field="indices", dtype=np.dtype(np.int64), dimensions=2)
        _validate_array(self.valid, field="valid", dtype=np.dtype(np.bool_), dimensions=2)
        expected_shape = (self.plan.batches, self.plan.batch_size)
        if self.indices.shape != expected_shape or self.valid.shape != expected_shape:
            raise NeuralBatchPlanError(reason=f"epoch arrays must have shape {expected_shape!r}")
        if np.any(self.indices < 0) or np.any(self.indices >= self.plan.rows):
            raise NeuralBatchPlanError(reason="every physical index must address a real source row")
        if np.any(self.indices[~self.valid] != 0):
            raise NeuralBatchPlanError(
                reason="invalid final slots must use canonical filler index zero"
            )
        expected_valid = (
            np.arange(self.plan.batch_size)[None, :] < _valid_counts(self.plan)[:, None]
        )
        if not np.array_equal(self.valid, expected_valid):
            raise NeuralBatchPlanError(
                reason="valid samples must be left-packed with one partial tail"
            )
        covered = self.indices[self.valid]
        coverage_counts = np.bincount(covered, minlength=self.plan.rows)
        if covered.shape != (self.plan.rows,) or not np.all(coverage_counts == 1):
            raise NeuralBatchPlanError(reason="an epoch must cover every source row exactly once")
        if not self.plan.shuffle and not np.array_equal(
            covered,
            np.arange(self.plan.rows, dtype=np.int64),
        ):
            raise NeuralBatchPlanError(reason="an unshuffled epoch must preserve source order")

    def batch(self, index: int) -> FixedBatchIndices:
        """Return one immutable fixed-width row by exact integer position."""
        if type(index) is not int or not 0 <= index < self.plan.batches:
            raise IndexError(index)
        return FixedBatchIndices(indices=self.indices[index], valid=self.valid[index])


def build_neural_epoch_batch_indices(
    plan: NeuralBatchPlan,
    epoch: int,
) -> NeuralEpochBatchIndices:
    """Build one exact epoch without dropping or duplicating a real observation."""
    if type(plan) is not NeuralBatchPlan:
        raise NeuralBatchPlanError(reason="plan must be NeuralBatchPlan")
    if type(epoch) is not int or epoch < 0:
        raise NeuralBatchPlanError(reason="epoch must be a nonnegative plain int")
    order = np.arange(plan.rows, dtype=np.int64)
    if plan.shuffle:
        seed = _validate_seed(plan.seed)
        generator = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([seed, epoch])))
        generator.shuffle(order)
    physical_rows = plan.batches * plan.batch_size
    flat_indices = np.zeros(physical_rows, dtype=np.int64)
    flat_valid = np.zeros(physical_rows, dtype=np.bool_)
    flat_indices[: plan.rows] = order
    flat_valid[: plan.rows] = True
    indices = flat_indices.reshape(plan.batches, plan.batch_size)
    valid = flat_valid.reshape(plan.batches, plan.batch_size)
    indices.flags.writeable = False
    valid.flags.writeable = False
    return NeuralEpochBatchIndices(plan=plan, epoch=epoch, indices=indices, valid=valid)


def _valid_counts(plan: NeuralBatchPlan) -> Int64Vector:
    counts = np.full(plan.batches, plan.batch_size, dtype=np.int64)
    counts[-1] = plan.rows - ((plan.batches - 1) * plan.batch_size)
    return counts


def _validate_seed(seed: object) -> int:
    if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
        raise NeuralBatchPlanError(reason=f"seed must be a plain int in [0, {_MAX_SEED}]")
    return seed


def _validate_array(
    value: object,
    *,
    field: str,
    dtype: np.dtype[np.generic],
    dimensions: int,
) -> None:
    if type(value) is not np.ndarray:
        raise NeuralBatchPlanError(reason=f"{field} must be a plain numpy.ndarray")
    if value.dtype != dtype or value.ndim != dimensions or not value.flags.c_contiguous:
        raise NeuralBatchPlanError(
            reason=f"{field} must be a C-contiguous {dimensions}-D {dtype} array",
        )
    if value.flags.writeable:
        raise NeuralBatchPlanError(reason=f"{field} storage must be read-only")


__all__ = [
    "FixedBatchIndices",
    "NeuralBatchPlan",
    "NeuralBatchPlanError",
    "NeuralEpochBatchIndices",
    "build_neural_epoch_batch_indices",
]
