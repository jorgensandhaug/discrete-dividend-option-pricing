"""End-to-end tests for portable fixed-batch PyTorch training."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from discrete_dividend_study.models.neural.backends.torch import (
    prepare_torch_neural_batch,
    torch_prediction_numpy,
)
from discrete_dividend_study.models.neural.backends.torch_training import (
    predict_torch_project_model,
    train_torch_project_model,
)
from discrete_dividend_study.models.neural.parameters import initialize_neural_parameters
from discrete_dividend_study.models.neural.specification import PROJECT_STUDY, NeuralVariant
from discrete_dividend_study.models.neural.training import (
    NeuralTrainingError,
    NeuralTrainingRequest,
)
from tests.models.neural.test_backends import _PYTORCH_CPU_1 as PYTORCH_CPU
from tests.models.neural.test_backends import _prepared_batch as prepared_batch

TRAINING_SEED = 20261209
SHUFFLE_SEED = 20261210
TEST_BATCH_SIZE = 5
TEST_EPOCHS = 3
INFERENCE_GEOMETRY_ATOL = 2e-5
TRAINING_DATA_SHA256 = "a" * 64
VALIDATION_DATA_SHA256 = "b" * 64


def test_training_is_deterministic_restores_best_and_keeps_partial_rows() -> None:
    """Two identical CPU requests produce identical history and best-restored predictions."""
    execution = PYTORCH_CPU
    training = replace(
        PROJECT_STUDY.training,
        batch_size=TEST_BATCH_SIZE,
        maximum_epochs=TEST_EPOCHS,
        early_stopping_patience_epochs=TEST_EPOCHS,
    )
    study = replace(PROJECT_STUDY, training=training)
    training_batch = prepared_batch(NeuralVariant.FORECAST_YIELD)
    validation_batch = prepared_batch(NeuralVariant.FORECAST_YIELD)
    parameters = initialize_neural_parameters(study, training_batch.model, TRAINING_SEED)
    request = NeuralTrainingRequest(
        study=study,
        parameters=parameters,
        training_batch=training_batch,
        validation_batch=validation_batch,
        training_data_sha256=TRAINING_DATA_SHA256,
        validation_data_sha256=VALIDATION_DATA_SHA256,
        execution=execution,
        shuffle_seed=SHUFFLE_SEED,
    )
    first = train_torch_project_model(request)
    second = train_torch_project_model(request)
    assert first.summary == second.summary
    assert first.summary.control.completed_epochs == TEST_EPOCHS
    assert first.summary.control.best_epoch is not None
    assert first.summary.control.should_stop
    first_values = prepare_torch_neural_batch(validation_batch, first.network)
    second_values = prepare_torch_neural_batch(validation_batch, second.network)
    first_prediction = torch_prediction_numpy(first.network, first_values)
    second_prediction = torch_prediction_numpy(second.network, second_values)
    np.testing.assert_array_equal(first_prediction, second_prediction)
    fixed_prediction = predict_torch_project_model(
        first.network,
        validation_batch,
        batch_size=TEST_BATCH_SIZE,
    )
    np.testing.assert_array_equal(
        fixed_prediction,
        predict_torch_project_model(
            second.network,
            validation_batch,
            batch_size=TEST_BATCH_SIZE,
        ),
    )
    np.testing.assert_allclose(
        fixed_prediction,
        first_prediction[:, 0],
        rtol=0.0,
        atol=INFERENCE_GEOMETRY_ATOL,
    )
    assert np.isfinite(first_prediction).all()
    assert first_prediction.shape == (validation_batch.rows, 1)
    assert tuple(parameter.name for parameter in first.parameters) == tuple(
        sorted(parameter.name for parameter in parameters.parameters)
    )
    assert all(
        np.isfinite(parameter.values).all() and not parameter.values.flags.writeable
        for parameter in first.parameters
    )


def test_training_request_rejects_phase_aliasing() -> None:
    """One object cannot silently serve as both the fit and validation phase."""
    execution = PYTORCH_CPU
    batch = prepared_batch(NeuralVariant.BINS)
    parameters = initialize_neural_parameters(
        PROJECT_STUDY,
        batch.model,
        TRAINING_SEED,
    )
    with pytest.raises(NeuralTrainingError, match="distinct"):
        NeuralTrainingRequest(
            study=PROJECT_STUDY,
            parameters=parameters,
            training_batch=batch,
            validation_batch=batch,
            training_data_sha256=TRAINING_DATA_SHA256,
            validation_data_sha256=VALIDATION_DATA_SHA256,
            execution=execution,
            shuffle_seed=SHUFFLE_SEED,
        )


def test_training_request_rejects_untyped_or_aliased_data_identity() -> None:
    """Licensed phase content must have distinct canonical SHA-256 identities."""
    execution = PYTORCH_CPU
    training_batch = prepared_batch(NeuralVariant.BINS)
    validation_batch = prepared_batch(NeuralVariant.BINS)
    parameters = initialize_neural_parameters(
        PROJECT_STUDY,
        training_batch.model,
        TRAINING_SEED,
    )
    for training_hash, validation_hash in (
        ("A" * 64, VALIDATION_DATA_SHA256),
        (TRAINING_DATA_SHA256, "0" * 63),
        (TRAINING_DATA_SHA256, TRAINING_DATA_SHA256),
    ):
        with pytest.raises(NeuralTrainingError, match=r"hash|SHA-256"):
            NeuralTrainingRequest(
                study=PROJECT_STUDY,
                parameters=parameters,
                training_batch=training_batch,
                validation_batch=validation_batch,
                training_data_sha256=training_hash,
                validation_data_sha256=validation_hash,
                execution=execution,
                shuffle_seed=SHUFFLE_SEED,
            )
