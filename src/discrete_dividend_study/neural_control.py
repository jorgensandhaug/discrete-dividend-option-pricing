"""Backend-neutral validation, plateau scheduling, and early-stopping state."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .neural_spec import TrainingSpec


@dataclass(frozen=True, slots=True)
class NeuralControlError(ValueError):
    """A training-control transition has an invalid identity or metric."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected controller contract."""
        return f"invalid neural training control: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralControlState:
    """Complete immutable state after zero or more validated epochs."""

    completed_epochs: int
    learning_rate: float
    best_validation_loss: float | None
    best_epoch: int | None
    epochs_without_improvement: int
    scheduler_best: float | None
    scheduler_bad_epochs: int
    scheduler_cooldown_remaining: int
    improved: bool
    learning_rate_reduced: bool
    should_stop: bool

    def __post_init__(self) -> None:
        """Reject malformed state before it can control optimizer or checkpoints."""
        if type(self.completed_epochs) is not int or self.completed_epochs < 0:
            raise NeuralControlError(reason="completed_epochs must be a nonnegative plain int")
        if not _finite_positive(self.learning_rate):
            raise NeuralControlError(reason="learning_rate must be finite and positive")
        for field, value in (
            ("epochs_without_improvement", self.epochs_without_improvement),
            ("scheduler_bad_epochs", self.scheduler_bad_epochs),
            ("scheduler_cooldown_remaining", self.scheduler_cooldown_remaining),
        ):
            if type(value) is not int or value < 0:
                raise NeuralControlError(reason=f"{field} must be a nonnegative plain int")
        for field, value in (
            ("improved", self.improved),
            ("learning_rate_reduced", self.learning_rate_reduced),
            ("should_stop", self.should_stop),
        ):
            if type(value) is not bool:
                raise NeuralControlError(reason=f"{field} must be a plain bool")
        _validate_optional_loss(self.best_validation_loss, field="best_validation_loss")
        _validate_optional_loss(self.scheduler_best, field="scheduler_best")
        if (self.best_validation_loss is None) != (self.best_epoch is None):
            raise NeuralControlError(
                reason="best loss and best epoch must be absent or present together"
            )
        if self.best_epoch is not None and (
            type(self.best_epoch) is not int or not 0 <= self.best_epoch < self.completed_epochs
        ):
            raise NeuralControlError(reason="best_epoch must identify a completed zero-based epoch")


def initial_neural_control_state(training: TrainingSpec) -> NeuralControlState:
    """Return the exact pre-training controller state."""
    if type(training) is not TrainingSpec:
        raise NeuralControlError(reason="training must be TrainingSpec")
    return NeuralControlState(
        completed_epochs=0,
        learning_rate=training.optimizer.learning_rate,
        best_validation_loss=None,
        best_epoch=None,
        epochs_without_improvement=0,
        scheduler_best=None,
        scheduler_bad_epochs=0,
        scheduler_cooldown_remaining=0,
        improved=False,
        learning_rate_reduced=False,
        should_stop=False,
    )


def advance_neural_control(
    state: NeuralControlState,
    validation_loss: float,
    training: TrainingSpec,
) -> NeuralControlState:
    """Apply one finite validation metric to scheduler and early-stopping state."""
    if type(state) is not NeuralControlState:
        raise NeuralControlError(reason="state must be NeuralControlState")
    if type(training) is not TrainingSpec:
        raise NeuralControlError(reason="training must be TrainingSpec")
    if state.should_stop:
        raise NeuralControlError(reason="a stopped controller cannot advance")
    if not _finite_nonnegative(validation_loss):
        raise NeuralControlError(reason="validation_loss must be a finite nonnegative plain float")

    epoch = state.completed_epochs
    improved = state.best_validation_loss is None or validation_loss < state.best_validation_loss
    best_loss = validation_loss if improved else state.best_validation_loss
    best_epoch = epoch if improved else state.best_epoch
    early_bad_epochs = 0 if improved else state.epochs_without_improvement + 1

    scheduler_best = state.scheduler_best
    scheduler_bad_epochs = state.scheduler_bad_epochs
    if scheduler_best is None or validation_loss < scheduler_best * (
        1.0 - training.scheduler.relative_threshold
    ):
        scheduler_best = validation_loss
        scheduler_bad_epochs = 0
    else:
        scheduler_bad_epochs += 1

    cooldown_remaining = state.scheduler_cooldown_remaining
    if cooldown_remaining > 0:
        cooldown_remaining -= 1
        scheduler_bad_epochs = 0

    learning_rate = state.learning_rate
    learning_rate_reduced = False
    if scheduler_bad_epochs > training.scheduler.patience_epochs:
        candidate = max(
            learning_rate * training.scheduler.factor,
            training.scheduler.minimum_learning_rate,
        )
        if learning_rate - candidate > training.scheduler.update_epsilon:
            learning_rate = candidate
            learning_rate_reduced = True
        cooldown_remaining = training.scheduler.cooldown_epochs
        scheduler_bad_epochs = 0

    completed_epochs = epoch + 1
    should_stop = (
        early_bad_epochs >= training.early_stopping_patience_epochs
        or completed_epochs >= training.maximum_epochs
    )
    return NeuralControlState(
        completed_epochs=completed_epochs,
        learning_rate=learning_rate,
        best_validation_loss=best_loss,
        best_epoch=best_epoch,
        epochs_without_improvement=early_bad_epochs,
        scheduler_best=scheduler_best,
        scheduler_bad_epochs=scheduler_bad_epochs,
        scheduler_cooldown_remaining=cooldown_remaining,
        improved=improved,
        learning_rate_reduced=learning_rate_reduced,
        should_stop=should_stop,
    )


def _validate_optional_loss(value: float | None, *, field: str) -> None:
    if value is not None and not _finite_nonnegative(value):
        raise NeuralControlError(reason=f"{field} must be None or a finite nonnegative plain float")


def _finite_positive(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and value > 0.0


def _finite_nonnegative(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and value >= 0.0


__all__ = [
    "NeuralControlError",
    "NeuralControlState",
    "advance_neural_control",
    "initial_neural_control_state",
]
