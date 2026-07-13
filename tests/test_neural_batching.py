"""Tests for deterministic fixed-shape neural batching without row loss."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from discrete_dividend_study.neural_batching import (
    NeuralBatchPlan,
    NeuralBatchPlanError,
    build_neural_epoch_batch_indices,
)

BATCH_SIZE = 128
PARTIAL_ROWS = 259
EXPECTED_BATCHES = 3
FINAL_VALID_ROWS = 3
SMALL_ROWS = 3


def test_partial_final_batch_is_masked_without_dropping_or_duplicating_rows() -> None:
    """A physical batch remains fixed width while logical coverage stays exact."""
    plan = NeuralBatchPlan(rows=PARTIAL_ROWS, batch_size=BATCH_SIZE, shuffle=False, seed=None)
    epoch = build_neural_epoch_batch_indices(plan, 0)
    assert plan.batches == EXPECTED_BATCHES
    assert epoch.indices.shape == (EXPECTED_BATCHES, BATCH_SIZE)
    assert epoch.valid.sum() == PARTIAL_ROWS
    np.testing.assert_array_equal(
        epoch.indices[epoch.valid],
        np.arange(PARTIAL_ROWS, dtype=np.int64),
    )
    final = epoch.batch(2)
    assert final.valid_rows == FINAL_VALID_ROWS
    np.testing.assert_array_equal(final.indices[:FINAL_VALID_ROWS], np.array([256, 257, 258]))
    np.testing.assert_array_equal(
        final.indices[FINAL_VALID_ROWS:],
        np.zeros(BATCH_SIZE - FINAL_VALID_ROWS, dtype=np.int64),
    )
    assert not final.valid[FINAL_VALID_ROWS:].any()


def test_dataset_smaller_than_batch_still_produces_one_valid_masked_batch() -> None:
    """Small rolling phases never disappear under a drop-last policy."""
    plan = NeuralBatchPlan(rows=SMALL_ROWS, batch_size=BATCH_SIZE, shuffle=False, seed=None)
    epoch = build_neural_epoch_batch_indices(plan, 0)
    batch = epoch.batch(0)
    assert plan.batches == 1
    assert batch.valid_rows == SMALL_ROWS
    np.testing.assert_array_equal(batch.indices[:SMALL_ROWS], np.array([0, 1, 2]))


def test_shuffle_stream_is_repeatable_and_epoch_specific() -> None:
    """The same seed and epoch repeat exactly while the next epoch changes order."""
    plan = NeuralBatchPlan(rows=1_003, batch_size=128, shuffle=True, seed=20261208)
    first = build_neural_epoch_batch_indices(plan, 7)
    repeated = build_neural_epoch_batch_indices(plan, 7)
    next_epoch = build_neural_epoch_batch_indices(plan, 8)
    np.testing.assert_array_equal(first.indices, repeated.indices)
    np.testing.assert_array_equal(first.valid, repeated.valid)
    assert not np.array_equal(first.indices[first.valid], next_epoch.indices[next_epoch.valid])
    np.testing.assert_array_equal(
        np.sort(first.indices[first.valid]),
        np.arange(plan.rows, dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("rows", "batch_size", "shuffle", "seed"),
    [
        (0, 128, False, None),
        (10, 0, False, None),
        (10, 128, True, None),
        (10, 128, False, 1),
        (10, 128, True, -1),
    ],
)
def test_invalid_plan_identity_fails(
    *,
    rows: int,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
) -> None:
    """Rows, physical shape, shuffle policy, and seed cannot be inferred or repaired."""
    with pytest.raises(NeuralBatchPlanError):
        NeuralBatchPlan(rows=rows, batch_size=batch_size, shuffle=shuffle, seed=seed)


def test_boolean_integer_fields_are_rejected() -> None:
    """Python's bool-as-int relationship cannot alter row, batch, seed, or epoch identity."""
    boolean_value: object = True
    invalid_rows = cast("int", boolean_value)
    with pytest.raises(NeuralBatchPlanError, match="rows"):
        NeuralBatchPlan(rows=invalid_rows, batch_size=128, shuffle=False, seed=None)
    plan = NeuralBatchPlan(rows=10, batch_size=128, shuffle=True, seed=1)
    invalid_epoch = cast("int", boolean_value)
    with pytest.raises(NeuralBatchPlanError, match="epoch"):
        build_neural_epoch_batch_indices(plan, invalid_epoch)
