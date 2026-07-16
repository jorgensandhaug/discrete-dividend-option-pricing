"""Tests for safe immutable neural checkpoint identity and storage."""

from __future__ import annotations

import platform
import runpy
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from discrete_dividend_study.neural_checkpoint import (
    NeuralCheckpointError,
    load_neural_checkpoint,
    write_neural_checkpoint,
)
from discrete_dividend_study.neural_control import (
    advance_neural_control,
    initial_neural_control_state,
)
from discrete_dividend_study.neural_execution import (
    PYTORCH_VERSION,
    NeuralRuntimeIdentity,
)
from discrete_dividend_study.neural_parameters import (
    NeuralParameterSet,
    initialize_neural_parameters,
)
from discrete_dividend_study.neural_spec import (
    PROJECT_STUDY,
    NeuralVariant,
)
from discrete_dividend_study.neural_training import (
    NeuralEpochMetrics,
    NeuralTrainingError,
    NeuralTrainingRequest,
    NeuralTrainingSummary,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from discrete_dividend_study.model_inputs import PreparedNeuralBatch
    from discrete_dividend_study.neural_execution import NeuralExecutionSpec

PARAMETER_SEED = 20261209
SHUFFLE_SEED = 20261210
TRAINING_DATA_SHA256 = "a" * 64
VALIDATION_DATA_SHA256 = "b" * 64
TRAINING_LOSS = 1.5
VALIDATION_LOSS = 2.0


def test_checkpoint_is_safe_content_addressed_complete_and_idempotent(tmp_path: Path) -> None:
    """Identical best weights and run identity resolve to one verified immutable artifact."""
    request, runtime, summary = _checkpoint_identity()
    first = write_neural_checkpoint(
        tmp_path,
        request=request,
        runtime=runtime,
        summary=summary,
        parameters=request.parameters.parameters,
    )
    second = write_neural_checkpoint(
        tmp_path,
        request=request,
        runtime=runtime,
        summary=summary,
        parameters=request.parameters.parameters,
    )
    assert first == second
    assert first.path.name == f"{first.sha256}.safetensors"
    assert tuple(tmp_path.iterdir()) == (first.path,)

    loaded = load_neural_checkpoint(first.path)
    assert loaded.artifact == first
    assert loaded.summary == summary
    assert loaded.artifact.manifest.training_data_sha256 == TRAINING_DATA_SHA256
    assert loaded.artifact.manifest.validation_data_sha256 == VALIDATION_DATA_SHA256
    assert loaded.artifact.manifest.initial_parameters_sha256 == request.parameters.sha256
    assert loaded.artifact.manifest.parameter_seed == PARAMETER_SEED
    assert loaded.artifact.manifest.shuffle_seed == SHUFFLE_SEED
    assert loaded.artifact.manifest.backend_version == PYTORCH_VERSION
    expected = {parameter.name: parameter.values for parameter in request.parameters.parameters}
    observed = {parameter.name: parameter.values for parameter in loaded.parameters}
    assert observed.keys() == expected.keys()
    for name, values in observed.items():
        np.testing.assert_array_equal(values, expected[name])
        assert not values.flags.writeable


def test_checkpoint_rejects_runtime_or_parameter_drift(tmp_path: Path) -> None:
    """A checkpoint cannot be written under a runtime alias or incomplete weight tree."""
    request, runtime, summary = _checkpoint_identity()
    with pytest.raises(NeuralCheckpointError, match="runtime"):
        write_neural_checkpoint(
            tmp_path,
            request=request,
            runtime=replace(runtime, cpu_threads=2),
            summary=summary,
            parameters=request.parameters.parameters,
        )
    with pytest.raises(NeuralCheckpointError, match="names and shapes"):
        write_neural_checkpoint(
            tmp_path,
            request=request,
            runtime=runtime,
            summary=summary,
            parameters=request.parameters.parameters[:-1],
        )


def test_checkpoint_replays_and_rejects_inconsistent_metric_decisions(tmp_path: Path) -> None:
    """Stored learning-rate and best-weight decisions must follow the declared state machine."""
    request, runtime, summary = _checkpoint_identity()
    inconsistent = NeuralTrainingSummary(
        control=summary.control,
        epochs=(replace(summary.epochs[0], improved=False),),
    )
    with pytest.raises(NeuralTrainingError, match="decisions"):
        write_neural_checkpoint(
            tmp_path,
            request=request,
            runtime=runtime,
            summary=inconsistent,
            parameters=request.parameters.parameters,
        )


def test_checkpoint_fails_before_decoding_tampered_bytes(tmp_path: Path) -> None:
    """Content corruption is detected from the filename identity before tensor use."""
    request, runtime, summary = _checkpoint_identity()
    artifact = write_neural_checkpoint(
        tmp_path,
        request=request,
        runtime=runtime,
        summary=summary,
        parameters=request.parameters.parameters,
    )
    artifact.path.write_bytes(artifact.path.read_bytes() + b"tampered")
    with pytest.raises(NeuralCheckpointError, match="filename hash"):
        load_neural_checkpoint(artifact.path)


def _checkpoint_identity() -> tuple[
    NeuralTrainingRequest,
    NeuralRuntimeIdentity,
    NeuralTrainingSummary,
]:
    helpers = runpy.run_path("tests/test_neural_backends.py")
    prepared_batch = cast(
        "Callable[[NeuralVariant], PreparedNeuralBatch]",
        helpers["_prepared_batch"],
    )
    execution = cast("NeuralExecutionSpec", helpers["_PYTORCH_CPU_1"])
    study = replace(
        PROJECT_STUDY,
        training=replace(
            PROJECT_STUDY.training,
            maximum_epochs=1,
            early_stopping_patience_epochs=1,
        ),
    )
    training_batch = prepared_batch(NeuralVariant.BINS)
    validation_batch = prepared_batch(NeuralVariant.BINS)
    parameters: NeuralParameterSet = initialize_neural_parameters(
        study,
        training_batch.model,
        PARAMETER_SEED,
    )
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
    control = advance_neural_control(
        initial_neural_control_state(study.training),
        VALIDATION_LOSS,
        study.training,
    )
    summary = NeuralTrainingSummary(
        control=control,
        epochs=(
            NeuralEpochMetrics(
                epoch=0,
                training_loss=TRAINING_LOSS,
                validation_loss=VALIDATION_LOSS,
                learning_rate=control.learning_rate,
                improved=control.improved,
                learning_rate_reduced=control.learning_rate_reduced,
            ),
        ),
    )
    runtime = NeuralRuntimeIdentity(
        backend=execution.backend,
        backend_version=PYTORCH_VERSION,
        device=execution.device,
        precision=execution.precision,
        compilation=execution.compilation,
        cpu_threads=execution.cpu_threads,
        operating_system=platform.system(),
        machine=platform.machine(),
    )
    return request, runtime, summary
