"""Tests for licensed-data-free benchmark inputs and execution identities."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from benchmarks._support import BenchmarkPlatformError, normalize_max_rss_bytes
from benchmarks.neural_rolling_preparation import main as rolling_preparation_main
from benchmarks.neural_training import synthetic_neural_batch
from benchmarks.tabpfn_preparation import main as tabpfn_preparation_main
from discrete_dividend_study.neural_spec import NeuralVariant

SYNTHETIC_ROWS = 7
RSS_SAMPLE = 123
TABPFN_FEATURE_COLUMNS = 18
TABPFN_FLOAT32_BYTES = 4
TABPFN_ROWS = 1000


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


def test_tabpfn_preparation_benchmark_emits_reproducible_content(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The benchmark times only the one required flat allocation and identifies its bytes."""
    outputs: list[dict[str, object]] = []
    for _ in range(2):
        monkeypatch.setattr(
            sys,
            "argv",
            ["tabpfn-preparation", "--rows", str(TABPFN_ROWS)],
        )
        tabpfn_preparation_main()
        outputs.append(json.loads(capsys.readouterr().out))
    assert outputs[0]["prepared_sha256"] == outputs[1]["prepared_sha256"]
    assert outputs[0]["source_prepared_batch_sha256"] == outputs[1]["source_prepared_batch_sha256"]
    assert outputs[0]["feature_columns"] == TABPFN_FEATURE_COLUMNS
    assert outputs[0]["feature_payload_bytes"] == (
        TABPFN_ROWS * TABPFN_FEATURE_COLUMNS * TABPFN_FLOAT32_BYTES
    )
    assert outputs[0]["target_is_zero_copy"] is True
