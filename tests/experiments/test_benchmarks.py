"""Tests for licensed-data-free benchmark inputs and execution identities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks._support import BenchmarkPlatformError, normalize_max_rss_bytes
from benchmarks.neural_rolling_preparation import main as rolling_preparation_main
from benchmarks.neural_training import (
    ExecutionChoice,
    resolve_benchmark_execution,
    synthetic_neural_batch,
)
from discrete_dividend_study.models.neural.execution import NeuralBackend, NeuralDevice
from discrete_dividend_study.models.neural.specification import NeuralVariant

SYNTHETIC_ROWS = 7
RSS_SAMPLE = 123
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_neural_benchmark_recipe_uses_importable_package_module() -> None:
    """The documented recipe must preserve the repository root on Python's import path."""
    justfile = (REPOSITORY_ROOT / "justfile").read_text()
    assert "python -m benchmarks.neural_training" in justfile
    assert "python benchmarks/neural_training.py" not in justfile


def test_benchmark_execution_choice_is_explicit_and_backend_stable() -> None:
    """Alternative measurements resolve the requested backend rather than the host default."""
    variant = NeuralVariant.ATTENTION
    cpu = resolve_benchmark_execution(ExecutionChoice.TORCH_CPU, variant, 3)
    gpu = resolve_benchmark_execution(ExecutionChoice.MLX_GPU, variant, 3)
    assert (cpu.backend, cpu.device, cpu.cpu_threads) == (
        NeuralBackend.PYTORCH,
        NeuralDevice.CPU,
        3,
    )
    assert (gpu.backend, gpu.device, gpu.cpu_threads) == (
        NeuralBackend.MLX,
        NeuralDevice.MLX_GPU,
        None,
    )


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


def test_peak_rss_units_are_explicit_on_supported_platforms() -> None:
    """Darwin bytes and Linux kibibytes cannot be mislabeled as the same raw unit."""
    assert normalize_max_rss_bytes(RSS_SAMPLE, "Darwin") == RSS_SAMPLE
    assert normalize_max_rss_bytes(RSS_SAMPLE, "Linux") == RSS_SAMPLE * 1024
    with pytest.raises(BenchmarkPlatformError, match="undeclared"):
        normalize_max_rss_bytes(RSS_SAMPLE, "Plan9")
