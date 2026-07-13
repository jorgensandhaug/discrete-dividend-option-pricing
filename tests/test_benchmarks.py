"""Tests for licensed-data-free benchmark inputs and execution identities."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from benchmarks.neural_rolling_preparation import main as rolling_preparation_main
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


def test_rolling_preparation_strategies_have_identical_phase_bytes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The optimized full-horizon path preserves every repeated-path phase hash."""
    observed: dict[str, dict[str, object]] = {}
    for strategy in ("full_horizon", "repeated_windows"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "neural-rolling-preparation",
                "--strategy",
                strategy,
                "--rows",
                "2000",
                "--windows",
                "2",
                "--threads",
                "2",
            ],
        )
        rolling_preparation_main()
        observed[strategy] = json.loads(capsys.readouterr().out)
    assert observed["full_horizon"]["checksum"] == observed["repeated_windows"]["checksum"]
    assert observed["full_horizon"]["phase_count"] == observed["repeated_windows"]["phase_count"]
