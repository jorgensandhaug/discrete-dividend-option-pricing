"""Eager or compiled MLX training for the declared project neural study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import mlx.core as mx  # ty: ignore[unresolved-import] - MLX does not ship core stubs
import numpy as np
import numpy.typing as npt
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from discrete_dividend_study.models.neural.backends.mlx import (
    MlxFixedNeuralBatch,
    MlxNeuralBatch,
    MlxProjectModel,
    build_mlx_project_model,
    mlx_fixed_smape_objective,
    prepare_mlx_neural_batch,
    snapshot_mlx_parameters,
)
from discrete_dividend_study.models.neural.batching import (
    NeuralBatchPlan,
    build_neural_epoch_batch_indices,
)
from discrete_dividend_study.models.neural.execution import CompilationMode
from discrete_dividend_study.models.neural.inputs import PreparedNeuralBatch
from discrete_dividend_study.models.neural.parameters import NeuralParameter
from discrete_dividend_study.models.neural.training import (
    NeuralEpochMetrics,
    NeuralTrainingError,
    NeuralTrainingRequest,
    NeuralTrainingSummary,
)
from discrete_dividend_study.models.neural.training_control import (
    advance_neural_control,
    initial_neural_control_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from discrete_dividend_study.models.neural.batching import NeuralEpochBatchIndices
    from discrete_dividend_study.models.neural.specification import NeuralStudySpec

type MlxFixedBatchTree = tuple[
    mx.array,
    mx.array,
    mx.array | None,
    mx.array | None,
    mx.array | None,
    mx.array | None,
    mx.array | None,
    mx.array,
]
type MlxTrainingStep = Callable[[MlxFixedBatchTree], tuple[mx.array, mx.array]]
type MlxValidationStep = Callable[[MlxFixedBatchTree], mx.array]
type MlxPredictionStep = Callable[[MlxFixedBatchTree], mx.array]
Float32Vector = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _MlxEpochIndices:
    layout: NeuralEpochBatchIndices
    indices: mx.array
    valid: mx.array


@dataclass(frozen=True, slots=True)
class MlxTrainingResult:
    """Best-restored MLX network and its complete immutable training summary."""

    network: MlxProjectModel
    summary: NeuralTrainingSummary
    parameters: tuple[NeuralParameter, ...]

    def __post_init__(self) -> None:
        """Reject an untyped network or summary."""
        if type(self.network) is not MlxProjectModel:
            raise NeuralTrainingError(reason="network must be MlxProjectModel")
        if type(self.summary) is not NeuralTrainingSummary:
            raise NeuralTrainingError(reason="summary must be NeuralTrainingSummary")
        if (
            type(self.parameters) is not tuple
            or not self.parameters
            or any(type(parameter) is not NeuralParameter for parameter in self.parameters)
        ):
            raise NeuralTrainingError(
                reason="parameters must be a nonempty tuple[NeuralParameter, ...]"
            )


def train_mlx_project_model(request: NeuralTrainingRequest) -> MlxTrainingResult:
    """Train one project model and restore its strictly best validation weights."""
    if type(request) is not NeuralTrainingRequest:
        raise NeuralTrainingError(reason="request must be NeuralTrainingRequest")
    study = request.study
    network = build_mlx_project_model(
        study=study,
        model=request.parameters.model,
        parameters=request.parameters,
        execution=request.execution,
    )
    training_values = prepare_mlx_neural_batch(request.training_batch, network)
    validation_values = prepare_mlx_neural_batch(request.validation_batch, network)
    training_plan = NeuralBatchPlan(
        rows=request.training_batch.rows,
        batch_size=study.training.batch_size,
        shuffle=study.training.shuffle_training,
        seed=request.shuffle_seed,
    )
    validation_plan = NeuralBatchPlan(
        rows=request.validation_batch.rows,
        batch_size=study.training.batch_size,
        shuffle=study.training.shuffle_validation,
        seed=None,
    )
    validation_epoch = build_neural_epoch_batch_indices(validation_plan, 0)
    validation_indices = _transfer_epoch_indices(validation_epoch)
    optimizer_spec = study.training.optimizer
    optimizer = optimizers.AdamW(
        learning_rate=optimizer_spec.learning_rate,
        betas=[optimizer_spec.beta1, optimizer_spec.beta2],
        eps=optimizer_spec.epsilon,
        weight_decay=optimizer_spec.weight_decay,
        bias_correction=optimizer_spec.bias_correction,
    )
    optimizer.init(network.trainable_parameters())
    training_step = _training_step(network, optimizer, study, request.execution.compilation)
    validation_step = _validation_step(network, study, request.execution.compilation)
    control = initial_neural_control_state(study.training)
    history: list[NeuralEpochMetrics] = []
    best_state: object | None = None

    while not control.should_stop:
        epoch = control.completed_epochs
        training_epoch = build_neural_epoch_batch_indices(training_plan, epoch)
        training_indices = _transfer_epoch_indices(training_epoch)
        training_loss = _train_epoch(
            network,
            optimizer,
            training_values,
            training_indices,
            training_step,
        )
        validation_loss = _validate_epoch(
            validation_values,
            validation_indices,
            validation_step,
        )
        control = advance_neural_control(control, validation_loss, study.training)
        if control.improved:
            best_state = network.parameters()
            mx.eval(best_state)
        _set_learning_rate(optimizer, control.learning_rate)
        history.append(
            NeuralEpochMetrics(
                epoch=epoch,
                training_loss=training_loss,
                validation_loss=validation_loss,
                learning_rate=control.learning_rate,
                improved=control.improved,
                learning_rate_reduced=control.learning_rate_reduced,
            ),
        )

    if best_state is None:
        raise NeuralTrainingError(reason="no finite best validation checkpoint was produced")
    network.update(best_state)
    network.eval()
    mx.eval(network.parameters())
    _require_finite_parameters(network)
    summary = NeuralTrainingSummary(control=control, epochs=tuple(history))
    return MlxTrainingResult(
        network=network,
        summary=summary,
        parameters=snapshot_mlx_parameters(network),
    )


def predict_mlx_project_model(
    network: MlxProjectModel,
    batch: PreparedNeuralBatch,
    *,
    batch_size: int,
) -> Float32Vector:
    """Predict every source row once with bounded fixed-shape compiled device batches."""
    if type(network) is not MlxProjectModel:
        raise NeuralTrainingError(reason="network must be MlxProjectModel")
    if type(batch) is not PreparedNeuralBatch:
        raise NeuralTrainingError(reason="batch must be PreparedNeuralBatch")
    source = prepare_mlx_neural_batch(batch, network)
    plan = NeuralBatchPlan(rows=batch.rows, batch_size=batch_size, shuffle=False, seed=None)
    layout = build_neural_epoch_batch_indices(plan, 0)
    epoch = _transfer_epoch_indices(layout)
    step = _prediction_step(network, network.runtime.compilation)
    network.eval()
    prediction = np.empty(batch.rows, dtype=np.float32)
    for batch_index in range(plan.batches):
        tree = _gather_fixed_batch_tree(
            source,
            epoch.indices[batch_index],
            epoch.valid[batch_index],
        )
        device_prediction = step(tree)
        mx.eval(device_prediction)
        physical = np.array(device_prediction, dtype=np.float32, copy=True)
        if physical.shape != (plan.batch_size, 1) or not np.isfinite(physical).all():
            raise NeuralTrainingError(reason="MLX returned invalid predictions")
        valid_rows = layout.valid_rows_at(batch_index)
        start = batch_index * plan.batch_size
        prediction[start : start + valid_rows] = physical[:valid_rows, 0]
    prediction.flags.writeable = False
    return prediction


def _training_step(
    network: MlxProjectModel,
    optimizer: optimizers.AdamW,
    study: NeuralStudySpec,
    compilation: CompilationMode,
) -> MlxTrainingStep:
    def loss_function(candidate: MlxProjectModel, tree: MlxFixedBatchTree) -> mx.array:
        fixed = _fixed_batch_from_tree(candidate, tree)
        return mlx_fixed_smape_objective(candidate(fixed.values), fixed, study.training)

    loss_and_gradient = nn.value_and_grad(network, loss_function)
    state = [network.state, optimizer.state]

    def step(tree: MlxFixedBatchTree) -> tuple[mx.array, mx.array]:
        loss, gradients = loss_and_gradient(network, tree)
        gradients_finite = _tree_is_finite(gradients)
        optimizer.update(network, gradients)
        parameters_finite = _tree_is_finite(network.parameters())
        return loss, gradients_finite & parameters_finite

    if compilation is CompilationMode.EAGER:
        return step
    if compilation is CompilationMode.COMPILED:
        return cast("MlxTrainingStep", partial(mx.compile, inputs=state, outputs=state)(step))
    raise NeuralTrainingError(reason=f"unsupported MLX compilation mode {compilation!r}")


def _validation_step(
    network: MlxProjectModel,
    study: NeuralStudySpec,
    compilation: CompilationMode,
) -> MlxValidationStep:
    def step(tree: MlxFixedBatchTree) -> mx.array:
        fixed = _fixed_batch_from_tree(network, tree)
        return mlx_fixed_smape_objective(network(fixed.values), fixed, study.training)

    if compilation is CompilationMode.EAGER:
        return step
    if compilation is CompilationMode.COMPILED:
        return cast("MlxValidationStep", partial(mx.compile, inputs=network.state)(step))
    raise NeuralTrainingError(reason=f"unsupported MLX compilation mode {compilation!r}")


def _prediction_step(
    network: MlxProjectModel,
    compilation: CompilationMode,
) -> MlxPredictionStep:
    def step(tree: MlxFixedBatchTree) -> mx.array:
        return network(_fixed_batch_from_tree(network, tree).values)

    if compilation is CompilationMode.EAGER:
        return step
    if compilation is CompilationMode.COMPILED:
        return cast("MlxPredictionStep", partial(mx.compile, inputs=network.state)(step))
    raise NeuralTrainingError(reason=f"unsupported MLX compilation mode {compilation!r}")


def _train_epoch(
    network: MlxProjectModel,
    optimizer: optimizers.AdamW,
    source: MlxNeuralBatch,
    epoch: _MlxEpochIndices,
    step: MlxTrainingStep,
) -> float:
    network.train()
    weighted_loss = 0.0
    logical_rows = 0
    state = [network.state, optimizer.state]
    for batch_index in range(epoch.layout.plan.batches):
        tree = _gather_fixed_batch_tree(
            source,
            epoch.indices[batch_index],
            epoch.valid[batch_index],
        )
        loss, finite = step(tree)
        mx.eval(state, loss, finite)
        if not bool(finite.item()):
            raise NeuralTrainingError(
                reason=f"nonfinite MLX gradient or parameter in batch {batch_index}",
            )
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise NeuralTrainingError(reason=f"nonfinite training loss in batch {batch_index}")
        valid_rows = epoch.layout.valid_rows_at(batch_index)
        weighted_loss += loss_value * valid_rows
        logical_rows += valid_rows
    return weighted_loss / logical_rows


def _validate_epoch(
    source: MlxNeuralBatch,
    epoch: _MlxEpochIndices,
    step: MlxValidationStep,
) -> float:
    weighted_loss = 0.0
    logical_rows = 0
    for batch_index in range(epoch.layout.plan.batches):
        tree = _gather_fixed_batch_tree(
            source,
            epoch.indices[batch_index],
            epoch.valid[batch_index],
        )
        loss = step(tree)
        mx.eval(loss)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise NeuralTrainingError(reason=f"nonfinite validation loss in batch {batch_index}")
        valid_rows = epoch.layout.valid_rows_at(batch_index)
        weighted_loss += loss_value * valid_rows
        logical_rows += valid_rows
    return weighted_loss / logical_rows


def _transfer_epoch_indices(epoch: NeuralEpochBatchIndices) -> _MlxEpochIndices:
    indices = mx.array(epoch.indices, dtype=mx.int64)
    valid = mx.array(epoch.valid, dtype=mx.bool_)
    mx.eval(indices, valid)
    return _MlxEpochIndices(layout=epoch, indices=indices, valid=valid)


def _gather_fixed_batch_tree(
    source: MlxNeuralBatch,
    indices: mx.array,
    valid: mx.array,
) -> MlxFixedBatchTree:
    """Slice one batch after source and complete epoch validation have already succeeded."""
    return (
        source.scalar_features[indices],
        source.target[indices],
        _gather_optional(source.event_times, indices),
        _gather_optional(source.event_amounts, indices),
        _gather_optional(source.event_mask, indices),
        _gather_optional(source.event_lengths, indices),
        _gather_optional(source.binned_amounts, indices),
        valid[:, None],
    )


def _gather_optional(value: mx.array | None, indices: mx.array) -> mx.array | None:
    return None if value is None else value[indices]


def _fixed_batch_from_tree(
    network: MlxProjectModel,
    tree: MlxFixedBatchTree,
) -> MlxFixedNeuralBatch:
    scalar, target, times, amounts, mask, lengths, bins, valid = tree
    return MlxFixedNeuralBatch(
        values=MlxNeuralBatch(
            model=network.model_spec,
            scalar_features=scalar,
            target=target,
            event_times=times,
            event_amounts=amounts,
            event_mask=mask,
            event_lengths=lengths,
            binned_amounts=bins,
        ),
        valid=valid,
    )


def _tree_is_finite(tree: object) -> mx.array:
    result = mx.ones((), dtype=mx.bool_)
    for _, value in tree_flatten(tree):
        result = result & mx.all(mx.isfinite(value))
    return result


def _set_learning_rate(optimizer: optimizers.AdamW, learning_rate: float) -> None:
    optimizer.learning_rate = learning_rate
    mx.eval(optimizer.learning_rate)
    observed = float(optimizer.learning_rate.item())
    expected = float(np.float32(learning_rate))
    if observed != expected:
        raise NeuralTrainingError(reason="MLX did not retain the float32 scheduled learning rate")


def _require_finite_parameters(network: MlxProjectModel) -> None:
    finite = _tree_is_finite(network.parameters())
    mx.eval(finite)
    if not bool(finite.item()):
        raise NeuralTrainingError(reason="best-restored MLX parameters are nonfinite")


__all__ = [
    "MlxTrainingResult",
    "predict_mlx_project_model",
    "train_mlx_project_model",
]
