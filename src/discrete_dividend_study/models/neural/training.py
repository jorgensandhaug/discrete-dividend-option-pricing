"""Framework-neutral identities emitted by neural training backends."""

from __future__ import annotations

import math
from dataclasses import dataclass

from discrete_dividend_study.models.neural.execution import NeuralExecutionSpec
from discrete_dividend_study.models.neural.inputs import PreparedNeuralBatch
from discrete_dividend_study.models.neural.parameters import NeuralParameterSet
from discrete_dividend_study.models.neural.specification import NeuralStudySpec, TrainingSpec
from discrete_dividend_study.models.neural.training_control import (
    NeuralControlState,
    advance_neural_control,
    initial_neural_control_state,
)

_SHA256_HEX_LENGTH = 64


@dataclass(slots=True)
class NeuralTrainingError(RuntimeError):
    """A training backend encountered invalid numerical or run state."""

    reason: str

    def __str__(self) -> str:
        """Return the failed training contract."""
        return f"neural training failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralTrainingRequest:
    """One complete model, data-phase, runtime, and randomness training identity."""

    study: NeuralStudySpec
    parameters: NeuralParameterSet
    training_batch: PreparedNeuralBatch
    validation_batch: PreparedNeuralBatch
    training_data_sha256: str
    validation_data_sha256: str
    execution: NeuralExecutionSpec
    shuffle_seed: int

    def __post_init__(self) -> None:
        """Reject cross-model phases, aliases, and untyped run identity."""
        if type(self.study) is not NeuralStudySpec:
            raise NeuralTrainingError(reason="study must be NeuralStudySpec")
        if type(self.parameters) is not NeuralParameterSet:
            raise NeuralTrainingError(reason="parameters must be NeuralParameterSet")
        if (
            type(self.training_batch) is not PreparedNeuralBatch
            or type(self.validation_batch) is not PreparedNeuralBatch
        ):
            raise NeuralTrainingError(reason="training phases must be PreparedNeuralBatch")
        if self.training_batch is self.validation_batch:
            raise NeuralTrainingError(reason="training and validation batches must be distinct")
        _require_sha256(self.training_data_sha256, field="training_data_sha256")
        _require_sha256(self.validation_data_sha256, field="validation_data_sha256")
        if self.training_data_sha256 == self.validation_data_sha256:
            raise NeuralTrainingError(reason="training and validation data hashes must be distinct")
        model = self.parameters.model
        if (
            self.parameters.study != self.study
            or self.training_batch.model != model
            or self.validation_batch.model != model
        ):
            raise NeuralTrainingError(
                reason="study, parameters, and phases must identify one model"
            )
        if type(self.execution) is not NeuralExecutionSpec:
            raise NeuralTrainingError(reason="execution must be NeuralExecutionSpec")
        maximum_seed = (2**64) - 1
        if type(self.shuffle_seed) is not int or not 0 <= self.shuffle_seed <= maximum_seed:
            raise NeuralTrainingError(reason="shuffle_seed must be an unsigned 64-bit plain int")


@dataclass(frozen=True, slots=True)
class NeuralEpochMetrics:
    """Finite row-weighted metrics and controller decisions for one completed epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    learning_rate: float
    improved: bool
    learning_rate_reduced: bool

    def __post_init__(self) -> None:
        """Reject invalid metrics before persistence or scheduling."""
        if type(self.epoch) is not int or self.epoch < 0:
            raise NeuralTrainingError(reason="epoch must be a nonnegative plain int")
        for field, value in (
            ("training_loss", self.training_loss),
            ("validation_loss", self.validation_loss),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise NeuralTrainingError(reason=f"{field} must be finite and nonnegative")
        if type(self.learning_rate) is not float or not math.isfinite(self.learning_rate):
            raise NeuralTrainingError(reason="learning_rate must be a finite plain float")
        if self.learning_rate <= 0.0:
            raise NeuralTrainingError(reason="learning_rate must be positive")
        if type(self.improved) is not bool or type(self.learning_rate_reduced) is not bool:
            raise NeuralTrainingError(reason="epoch decisions must be plain bool values")


@dataclass(frozen=True, slots=True)
class NeuralTrainingSummary:
    """Complete controller and metric history after best-weight restoration."""

    control: NeuralControlState
    epochs: tuple[NeuralEpochMetrics, ...]

    def __post_init__(self) -> None:
        """Require a complete contiguous history matching the controller."""
        if type(self.control) is not NeuralControlState:
            raise NeuralTrainingError(reason="control must be NeuralControlState")
        if type(self.epochs) is not tuple or any(
            type(epoch) is not NeuralEpochMetrics for epoch in self.epochs
        ):
            raise NeuralTrainingError(reason="epochs must be tuple[NeuralEpochMetrics, ...]")
        if len(self.epochs) != self.control.completed_epochs:
            raise NeuralTrainingError(reason="metric history must match completed_epochs")
        if tuple(epoch.epoch for epoch in self.epochs) != tuple(range(len(self.epochs))):
            raise NeuralTrainingError(reason="metric history must use contiguous zero-based epochs")
        if not self.epochs or self.control.best_epoch is None:
            raise NeuralTrainingError(reason="training summary must contain a best completed epoch")
        if not self.control.should_stop:
            raise NeuralTrainingError(reason="training summary control must be terminal")


def validate_neural_training_summary(
    summary: NeuralTrainingSummary,
    training: TrainingSpec,
) -> None:
    """Replay every controller decision and reject inconsistent metric history."""
    if type(summary) is not NeuralTrainingSummary:
        raise NeuralTrainingError(reason="summary must be NeuralTrainingSummary")
    if type(training) is not TrainingSpec:
        raise NeuralTrainingError(reason="training must be TrainingSpec")
    state = initial_neural_control_state(training)
    for metrics in summary.epochs:
        state = advance_neural_control(state, metrics.validation_loss, training)
        if (
            metrics.learning_rate != state.learning_rate
            or metrics.improved is not state.improved
            or metrics.learning_rate_reduced is not state.learning_rate_reduced
        ):
            raise NeuralTrainingError(
                reason=f"epoch {metrics.epoch} decisions disagree with the training contract"
            )
    if state != summary.control:
        raise NeuralTrainingError(reason="final control state disagrees with metric history")


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NeuralTrainingError(reason=f"{field} must be a lowercase SHA-256 hex digest")


__all__ = [
    "NeuralEpochMetrics",
    "NeuralTrainingError",
    "NeuralTrainingRequest",
    "NeuralTrainingSummary",
    "validate_neural_training_summary",
]
