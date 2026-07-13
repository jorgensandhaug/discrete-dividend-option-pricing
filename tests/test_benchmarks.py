"""Tests for licensed-data-free benchmark inputs and execution identities."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.neural_training import synthetic_neural_batch
from discrete_dividend_study.neural_spec import NeuralVariant

SYNTHETIC_ROWS = 7


@pytest.mark.parametrize("variant", list(NeuralVariant))
def test_synthetic_benchmark_batches_cover_every_declared_model(variant: NeuralVariant) -> None:
    """Benchmark rows satisfy strict model contracts and distinguish data phases."""
    training = synthetic_neural_batch(variant, SYNTHETIC_ROWS, phase=0)
    validation = synthetic_neural_batch(variant, SYNTHETIC_ROWS, phase=1)
    assert training.rows == SYNTHETIC_ROWS
    assert validation.rows == SYNTHETIC_ROWS
    assert training.model.variant is variant
    assert np.isfinite(training.scalar_features).all()
    assert np.isfinite(training.target).all()
    assert not np.array_equal(training.scalar_features, validation.scalar_features)
