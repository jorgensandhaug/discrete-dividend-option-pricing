"""Shared fail-fast contracts for optional neural backend realizations."""

from __future__ import annotations

from dataclasses import dataclass

from discrete_dividend_study.models.neural.execution import (
    NeuralBackend,
    NeuralExecutionError,
    NeuralExecutionSpec,
    NeuralExecutionSpecError,
    NeuralRuntimeIdentity,
    require_neural_runtime,
)
from discrete_dividend_study.models.neural.inputs import PreparedNeuralBatch
from discrete_dividend_study.models.neural.parameters import NeuralParameterSet
from discrete_dividend_study.models.neural.specification import NeuralModelSpec, NeuralStudySpec


@dataclass(slots=True)
class NeuralBackendContractError(NeuralExecutionError):
    """A model, parameter set, batch, and backend request do not share one identity."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected backend contract."""
        return f"invalid neural backend contract: {self.reason}"


def validate_neural_build_request(
    *,
    study: NeuralStudySpec,
    model: NeuralModelSpec,
    parameters: NeuralParameterSet,
    execution: NeuralExecutionSpec,
    backend: NeuralBackend,
) -> NeuralRuntimeIdentity:
    """Validate one complete build identity and resolve exactly its requested runtime."""
    if type(study) is not NeuralStudySpec:
        raise NeuralBackendContractError(reason="study must be NeuralStudySpec")
    if type(model) is not NeuralModelSpec or model not in study.models:
        raise NeuralBackendContractError(reason="model must be owned by study")
    if type(parameters) is not NeuralParameterSet:
        raise NeuralBackendContractError(reason="parameters must be NeuralParameterSet")
    if parameters.study != study or parameters.model != model:
        raise NeuralBackendContractError(
            reason="parameter set must identify the requested study and model",
        )
    if type(execution) is not NeuralExecutionSpec:
        raise NeuralExecutionSpecError(reason="execution must be NeuralExecutionSpec")
    if execution.backend is not backend:
        raise NeuralBackendContractError(
            reason=f"{backend.value} model requires a {backend.value} execution request",
        )
    return require_neural_runtime(execution)


def validate_prepared_batch(batch: PreparedNeuralBatch, model: NeuralModelSpec) -> None:
    """Require a nonempty prepared batch with the model's exact identity."""
    if type(batch) is not PreparedNeuralBatch:
        raise NeuralBackendContractError(reason="batch must be PreparedNeuralBatch")
    if batch.model != model:
        raise NeuralBackendContractError(reason="batch model does not match backend model")
    if batch.rows <= 0:
        raise NeuralBackendContractError(reason="batch must contain at least one row")


__all__ = [
    "NeuralBackendContractError",
    "validate_neural_build_request",
    "validate_prepared_batch",
]
