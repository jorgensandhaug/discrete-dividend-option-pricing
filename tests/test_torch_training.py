"""End-to-end tests for portable fixed-batch PyTorch training."""

from __future__ import annotations

import runpy
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from discrete_dividend_study.neural_parameters import initialize_neural_parameters
from discrete_dividend_study.neural_spec import PROJECT_NEURAL_STUDY_SPEC, NeuralVariant
from discrete_dividend_study.neural_training import (
    NeuralTrainingError,
    NeuralTrainingRequest,
)
from discrete_dividend_study.torch_neural import (
    prepare_torch_neural_batch,
    torch_prediction_numpy,
)
from discrete_dividend_study.torch_training import train_torch_project_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from discrete_dividend_study.model_inputs import PreparedNeuralBatch
    from discrete_dividend_study.neural_execution import NeuralExecutionSpec

TRAINING_SEED = 20261209
SHUFFLE_SEED = 20261210
TEST_BATCH_SIZE = 5
TEST_EPOCHS = 3
TRAINING_DATA_SHA256 = "a" * 64
VALIDATION_DATA_SHA256 = "b" * 64


def test_training_is_deterministic_restores_best_and_keeps_partial_rows() -> None:
    """Two identical CPU requests produce identical history and best-restored predictions."""
    helpers = runpy.run_path("tests/test_neural_backends.py")
    prepared_batch = cast(
        "Callable[[NeuralVariant], PreparedNeuralBatch]",
        helpers["_prepared_batch"],
    )
    execution = cast("NeuralExecutionSpec", helpers["_PYTORCH_CPU_1"])
    training = replace(
        PROJECT_NEURAL_STUDY_SPEC.training,
        batch_size=TEST_BATCH_SIZE,
        maximum_epochs=TEST_EPOCHS,
        early_stopping_patience_epochs=TEST_EPOCHS,
    )
    study = replace(PROJECT_NEURAL_STUDY_SPEC, training=training)
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
    helpers = runpy.run_path("tests/test_neural_backends.py")
    prepared_batch = cast(
        "Callable[[NeuralVariant], PreparedNeuralBatch]",
        helpers["_prepared_batch"],
    )
    execution = cast("NeuralExecutionSpec", helpers["_PYTORCH_CPU_1"])
    batch = prepared_batch(NeuralVariant.BINS)
    parameters = initialize_neural_parameters(
        PROJECT_NEURAL_STUDY_SPEC,
        batch.model,
        TRAINING_SEED,
    )
    with pytest.raises(NeuralTrainingError, match="distinct"):
        NeuralTrainingRequest(
            study=PROJECT_NEURAL_STUDY_SPEC,
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
    helpers = runpy.run_path("tests/test_neural_backends.py")
    prepared_batch = cast(
        "Callable[[NeuralVariant], PreparedNeuralBatch]",
        helpers["_prepared_batch"],
    )
    execution = cast("NeuralExecutionSpec", helpers["_PYTORCH_CPU_1"])
    training_batch = prepared_batch(NeuralVariant.BINS)
    validation_batch = prepared_batch(NeuralVariant.BINS)
    parameters = initialize_neural_parameters(
        PROJECT_NEURAL_STUDY_SPEC,
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
                study=PROJECT_NEURAL_STUDY_SPEC,
                parameters=parameters,
                training_batch=training_batch,
                validation_batch=validation_batch,
                training_data_sha256=training_hash,
                validation_data_sha256=validation_hash,
                execution=execution,
                shuffle_seed=SHUFFLE_SEED,
            )
