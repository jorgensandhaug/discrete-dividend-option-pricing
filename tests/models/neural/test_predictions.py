"""Tests for safe test-prediction artifacts and embedded project metrics."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pytest

from discrete_dividend_study.experiments.windows import RollingWindow
from discrete_dividend_study.models.neural.checkpoints import write_neural_checkpoint
from discrete_dividend_study.models.neural.predictions import (
    NeuralPredictionError,
    NeuralPredictionRequest,
    load_neural_prediction,
    write_neural_prediction,
)
from tests.models.neural.test_checkpoints import _checkpoint_identity

if TYPE_CHECKING:
    from pathlib import Path

TEST_DATA_SHA256 = "c" * 64
WINDOW = RollingWindow(
    window_id=9,
    train_start=date(2023, 1, 1),
    validation_start=date(2023, 11, 1),
    test_start=date(2023, 12, 1),
    test_end=date(2024, 1, 1),
)


def test_prediction_is_safe_complete_content_addressed_and_idempotent(tmp_path: Path) -> None:
    """Identical test output resolves to one verified file containing exact pooled metrics."""
    request, runtime, summary = _checkpoint_identity()
    checkpoint_directory = tmp_path / "checkpoints"
    prediction_directory = tmp_path / "predictions"
    checkpoint_directory.mkdir()
    prediction_directory.mkdir()
    checkpoint = write_neural_checkpoint(
        checkpoint_directory,
        request=request,
        runtime=runtime,
        summary=summary,
        parameters=request.parameters.parameters,
    )
    prediction = np.ascontiguousarray(request.validation_batch.target * np.float32(1.1))
    prediction.flags.writeable = False
    first = write_neural_prediction(
        prediction_directory,
        NeuralPredictionRequest(
            study=request.study,
            window=WINDOW,
            test_batch=request.validation_batch,
            test_data_sha256=TEST_DATA_SHA256,
            checkpoint=checkpoint,
            prediction=prediction,
        ),
    )
    second = write_neural_prediction(
        prediction_directory,
        NeuralPredictionRequest(
            study=request.study,
            window=WINDOW,
            test_batch=request.validation_batch,
            test_data_sha256=TEST_DATA_SHA256,
            checkpoint=checkpoint,
            prediction=prediction,
        ),
    )
    assert first == second
    assert tuple(prediction_directory.iterdir()) == (first.path,)
    assert first.path.name == f"{first.sha256}.safetensors"
    assert first.manifest.window_id == WINDOW.window_id
    assert first.manifest.test_data_sha256 == TEST_DATA_SHA256
    assert first.manifest.checkpoint_sha256 == checkpoint.sha256
    assert first.metrics.observations == request.validation_batch.rows

    loaded = load_neural_prediction(first.path)
    assert loaded.artifact == first
    np.testing.assert_array_equal(loaded.target, request.validation_batch.target)
    np.testing.assert_array_equal(loaded.prediction, prediction)
    assert not loaded.target.flags.writeable
    assert not loaded.prediction.flags.writeable


def test_negative_predictions_and_tampered_files_fail_without_repair(tmp_path: Path) -> None:
    """Financially invalid output is never clipped and byte corruption fails before decoding."""
    request, runtime, summary = _checkpoint_identity()
    checkpoint_directory = tmp_path / "checkpoints"
    prediction_directory = tmp_path / "predictions"
    checkpoint_directory.mkdir()
    prediction_directory.mkdir()
    checkpoint = write_neural_checkpoint(
        checkpoint_directory,
        request=request,
        runtime=runtime,
        summary=summary,
        parameters=request.parameters.parameters,
    )
    negative = np.full(request.validation_batch.rows, -1.0, dtype=np.float32)
    negative.flags.writeable = False
    with pytest.raises(NeuralPredictionError, match="nonnegative"):
        write_neural_prediction(
            prediction_directory,
            NeuralPredictionRequest(
                study=request.study,
                window=WINDOW,
                test_batch=request.validation_batch,
                test_data_sha256=TEST_DATA_SHA256,
                checkpoint=checkpoint,
                prediction=negative,
            ),
        )

    valid = request.validation_batch.target.copy()
    valid.flags.writeable = False
    artifact = write_neural_prediction(
        prediction_directory,
        NeuralPredictionRequest(
            study=request.study,
            window=WINDOW,
            test_batch=request.validation_batch,
            test_data_sha256=TEST_DATA_SHA256,
            checkpoint=checkpoint,
            prediction=valid,
        ),
    )
    artifact.path.write_bytes(artifact.path.read_bytes() + b"tampered")
    with pytest.raises(NeuralPredictionError, match="filename hash"):
        load_neural_prediction(artifact.path)
