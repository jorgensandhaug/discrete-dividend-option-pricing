"""Tests for backend-neutral scheduler and early-stopping transitions."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from discrete_dividend_study.models.neural.training_control import (
    NeuralControlError,
    advance_neural_control,
    initial_neural_control_state,
)
from discrete_dividend_study.paper import PROJECT_STUDY

BASELINE_LOSS = 10.0
SMALL_IMPROVEMENT_LOSS = 9.9995
EARLY_STOP_COMPLETED_EPOCHS = 3


def test_plateau_learning_rate_sequence_matches_declared_relative_policy() -> None:
    """The independent controller agrees with the paper policy's boundary semantics."""
    training = PROJECT_STUDY.training
    optimizer_parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [optimizer_parameter],
        lr=training.optimizer.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training.scheduler.factor,
        patience=training.scheduler.patience_epochs,
        threshold=training.scheduler.relative_threshold,
        threshold_mode="rel",
        cooldown=training.scheduler.cooldown_epochs,
        min_lr=training.scheduler.minimum_learning_rate,
        eps=training.scheduler.update_epsilon,
    )
    metrics = [
        BASELINE_LOSS,
        SMALL_IMPROVEMENT_LOSS,
        *([10.1] * 10),
        9.0,
        *([9.1] * 11),
    ]
    state = initial_neural_control_state(training)
    for metric in metrics:
        scheduler.step(metric)
        state = advance_neural_control(state, metric, training)
        observed = float(optimizer.param_groups[0]["lr"])
        assert state.learning_rate == observed


def test_strict_early_stopping_tracks_best_epoch_independently_of_scheduler_threshold() -> None:
    """Any strict loss decrease checkpoints even when too small to reset the plateau counter."""
    training = PROJECT_STUDY.training
    state = initial_neural_control_state(training)
    state = advance_neural_control(state, BASELINE_LOSS, training)
    state = advance_neural_control(state, SMALL_IMPROVEMENT_LOSS, training)
    assert state.improved
    assert state.best_validation_loss == SMALL_IMPROVEMENT_LOSS
    assert state.best_epoch == 1
    assert state.scheduler_best == BASELINE_LOSS
    assert state.scheduler_bad_epochs == 1


def test_early_stopping_and_maximum_epoch_are_exact_terminal_conditions() -> None:
    """A stopped state cannot run an extra epoch or mutate after termination."""
    original = PROJECT_STUDY.training
    training = replace(original, early_stopping_patience_epochs=2, maximum_epochs=5)
    state = initial_neural_control_state(training)
    state = advance_neural_control(state, 1.0, training)
    state = advance_neural_control(state, 1.1, training)
    assert not state.should_stop
    state = advance_neural_control(state, 1.2, training)
    assert state.should_stop
    assert state.completed_epochs == EARLY_STOP_COMPLETED_EPOCHS
    with pytest.raises(NeuralControlError, match="stopped"):
        advance_neural_control(state, 0.9, training)

    maximum_only = replace(original, early_stopping_patience_epochs=10, maximum_epochs=2)
    state = initial_neural_control_state(maximum_only)
    state = advance_neural_control(state, 1.0, maximum_only)
    state = advance_neural_control(state, 0.9, maximum_only)
    assert state.should_stop
    assert state.completed_epochs == maximum_only.maximum_epochs


@pytest.mark.parametrize("loss", [math.nan, math.inf, -1.0])
def test_invalid_validation_metrics_fail_before_control_state_changes(loss: float) -> None:
    """Nonfinite or negative metrics never reach scheduling or checkpoint selection."""
    training = PROJECT_STUDY.training
    state = initial_neural_control_state(training)
    with pytest.raises(NeuralControlError, match="validation_loss"):
        advance_neural_control(state, loss, training)
