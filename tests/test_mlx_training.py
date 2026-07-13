"""End-to-end tests for fixed-batch MLX training."""

from __future__ import annotations

import runpy
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from discrete_dividend_study.mlx_neural import (
    mlx_prediction_numpy,
    prepare_mlx_neural_batch,
)
from discrete_dividend_study.mlx_training import (
    predict_mlx_project_model,
    train_mlx_project_model,
)
from discrete_dividend_study.neural_execution import CompilationMode
from discrete_dividend_study.neural_parameters import initialize_neural_parameters
from discrete_dividend_study.neural_spec import PROJECT_NEURAL_STUDY_SPEC, NeuralVariant
from discrete_dividend_study.neural_training import NeuralTrainingRequest

if TYPE_CHECKING:
    from collections.abc import Callable

    from discrete_dividend_study.model_inputs import PreparedNeuralBatch
    from discrete_dividend_study.neural_execution import NeuralExecutionSpec
    from discrete_dividend_study.neural_spec import NeuralStudySpec

TRAINING_SEED = 20261209
SHUFFLE_SEED = 20261210
TEST_BATCH_SIZE = 5
TEST_EPOCHS = 2
INFERENCE_GEOMETRY_ATOL = 2e-5
TRAINING_DATA_SHA256 = "a" * 64
VALIDATION_DATA_SHA256 = "b" * 64


def test_eager_training_is_deterministic_and_restores_best() -> None:
    """Two identical MLX CPU requests produce identical history and predictions."""
    helpers = runpy.run_path("tests/test_neural_backends.py")
    prepared_batch = cast(
        "Callable[[NeuralVariant], PreparedNeuralBatch]",
        helpers["_prepared_batch"],
    )
    execution = cast("NeuralExecutionSpec", helpers["_MLX_CPU"])
    study = _small_study()
    training_batch = prepared_batch(NeuralVariant.BINS)
    validation_batch = prepared_batch(NeuralVariant.BINS)
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
    first = train_mlx_project_model(request)
    second = train_mlx_project_model(request)
    assert first.summary == second.summary
    assert first.summary.control.completed_epochs == TEST_EPOCHS
    assert first.summary.control.best_epoch is not None
    first_values = prepare_mlx_neural_batch(validation_batch, first.network)
    second_values = prepare_mlx_neural_batch(validation_batch, second.network)
    first_prediction = mlx_prediction_numpy(first.network, first_values)
    second_prediction = mlx_prediction_numpy(second.network, second_values)
    np.testing.assert_array_equal(first_prediction, second_prediction)
    fixed_prediction = predict_mlx_project_model(
        first.network,
        validation_batch,
        batch_size=TEST_BATCH_SIZE,
    )
    np.testing.assert_array_equal(
        fixed_prediction,
        predict_mlx_project_model(
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
    assert tuple(parameter.name for parameter in first.parameters) == tuple(
        sorted(parameter.name for parameter in parameters.parameters)
    )
    assert all(
        np.isfinite(parameter.values).all() and not parameter.values.flags.writeable
        for parameter in first.parameters
    )


def test_compiled_training_step_accepts_fixed_partial_batch_tree() -> None:
    """The production compiled path captures model and optimizer state with dynamic masks."""
    helpers = runpy.run_path("tests/test_neural_backends.py")
    prepared_batch = cast(
        "Callable[[NeuralVariant], PreparedNeuralBatch]",
        helpers["_prepared_batch"],
    )
    eager = cast("NeuralExecutionSpec", helpers["_MLX_CPU"])
    compiled = replace(eager, compilation=CompilationMode.COMPILED)
    study = _small_study(maximum_epochs=1)
    training_batch = prepared_batch(NeuralVariant.FORECAST_YIELD)
    validation_batch = prepared_batch(NeuralVariant.FORECAST_YIELD)
    parameters = initialize_neural_parameters(study, training_batch.model, TRAINING_SEED)
    result = train_mlx_project_model(
        NeuralTrainingRequest(
            study=study,
            parameters=parameters,
            training_batch=training_batch,
            validation_batch=validation_batch,
            training_data_sha256=TRAINING_DATA_SHA256,
            validation_data_sha256=VALIDATION_DATA_SHA256,
            execution=compiled,
            shuffle_seed=SHUFFLE_SEED,
        ),
    )
    assert result.summary.control.completed_epochs == 1
    assert result.summary.control.best_epoch == 0


def _small_study(*, maximum_epochs: int = TEST_EPOCHS) -> NeuralStudySpec:
    training = replace(
        PROJECT_NEURAL_STUDY_SPEC.training,
        batch_size=TEST_BATCH_SIZE,
        maximum_epochs=maximum_epochs,
        early_stopping_patience_epochs=maximum_epochs,
    )
    return replace(PROJECT_NEURAL_STUDY_SPEC, training=training)
