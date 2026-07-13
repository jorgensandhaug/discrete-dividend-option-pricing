"""Portable eager PyTorch training for the declared project neural study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from .neural_batching import NeuralBatchPlan, build_neural_epoch_batch_indices
from .neural_control import advance_neural_control, initial_neural_control_state
from .neural_parameters import NeuralParameter
from .neural_training import (
    NeuralEpochMetrics,
    NeuralTrainingError,
    NeuralTrainingRequest,
    NeuralTrainingSummary,
)
from .torch_neural import (
    TorchProjectModel,
    build_torch_project_model,
    gather_torch_fixed_batch,
    prepare_torch_neural_batch,
    snapshot_torch_parameters,
    torch_fixed_smape_objective,
)

if TYPE_CHECKING:
    from .neural_batching import NeuralEpochBatchIndices
    from .neural_spec import NeuralStudySpec
    from .torch_neural import TorchNeuralBatch


@dataclass(frozen=True, slots=True)
class TorchTrainingResult:
    """Best-restored PyTorch network and its complete immutable training summary."""

    network: TorchProjectModel
    summary: NeuralTrainingSummary
    parameters: tuple[NeuralParameter, ...]

    def __post_init__(self) -> None:
        """Reject an untyped network or summary."""
        if type(self.network) is not TorchProjectModel:
            raise NeuralTrainingError(reason="network must be TorchProjectModel")
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


def train_torch_project_model(request: NeuralTrainingRequest) -> TorchTrainingResult:
    """Train one project model and restore its strictly best validation weights."""
    if type(request) is not NeuralTrainingRequest:
        raise NeuralTrainingError(reason="request must be NeuralTrainingRequest")
    study = request.study
    model = request.parameters.model
    network = build_torch_project_model(
        study=study,
        model=model,
        parameters=request.parameters,
        execution=request.execution,
    )
    training_values = prepare_torch_neural_batch(request.training_batch, network)
    validation_values = prepare_torch_neural_batch(request.validation_batch, network)
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
    optimizer_spec = study.training.optimizer
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=optimizer_spec.learning_rate,
        betas=(optimizer_spec.beta1, optimizer_spec.beta2),
        eps=optimizer_spec.epsilon,
        weight_decay=optimizer_spec.weight_decay,
    )
    control = initial_neural_control_state(study.training)
    history: list[NeuralEpochMetrics] = []
    best_state: dict[str, torch.Tensor] | None = None

    while not control.should_stop:
        epoch = control.completed_epochs
        training_epoch = build_neural_epoch_batch_indices(training_plan, epoch)
        training_loss = _train_epoch(
            network,
            optimizer,
            training_values,
            training_epoch,
            study,
        )
        validation_loss = _validate_epoch(
            network,
            validation_values,
            validation_epoch,
            study,
        )
        control = advance_neural_control(control, validation_loss, study.training)
        if control.improved:
            best_state = {
                name: value.detach().clone() for name, value in network.state_dict().items()
            }
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
    network.load_state_dict(best_state, strict=True)
    network.eval()
    _require_finite_parameters(network)
    summary = NeuralTrainingSummary(control=control, epochs=tuple(history))
    return TorchTrainingResult(
        network=network,
        summary=summary,
        parameters=snapshot_torch_parameters(network),
    )


def _train_epoch(
    network: TorchProjectModel,
    optimizer: torch.optim.AdamW,
    source: TorchNeuralBatch,
    epoch: NeuralEpochBatchIndices,
    study: NeuralStudySpec,
) -> float:
    network.train()
    weighted_loss = 0.0
    logical_rows = 0
    for batch_index in range(epoch.plan.batches):
        fixed_indices = epoch.batch(batch_index)
        fixed = gather_torch_fixed_batch(source, fixed_indices)
        optimizer.zero_grad(set_to_none=True)
        prediction = network(fixed.values)
        loss = torch_fixed_smape_objective(prediction, fixed, study.training)
        if not bool(torch.isfinite(loss).item()):
            raise NeuralTrainingError(reason=f"nonfinite training loss in batch {batch_index}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            network.parameters(),
            max_norm=math.inf,
            error_if_nonfinite=True,
            foreach=True,
        )
        optimizer.step()
        loss_value = float(loss.detach().item())
        weighted_loss += loss_value * fixed_indices.valid_rows
        logical_rows += fixed_indices.valid_rows
    _require_finite_parameters(network)
    return weighted_loss / logical_rows


def _validate_epoch(
    network: TorchProjectModel,
    source: TorchNeuralBatch,
    epoch: NeuralEpochBatchIndices,
    study: NeuralStudySpec,
) -> float:
    network.eval()
    weighted_loss = 0.0
    logical_rows = 0
    with torch.no_grad():
        for batch_index in range(epoch.plan.batches):
            fixed_indices = epoch.batch(batch_index)
            fixed = gather_torch_fixed_batch(source, fixed_indices)
            prediction = network(fixed.values)
            loss = torch_fixed_smape_objective(prediction, fixed, study.training)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise NeuralTrainingError(
                    reason=f"nonfinite validation loss in batch {batch_index}",
                )
            weighted_loss += loss_value * fixed_indices.valid_rows
            logical_rows += fixed_indices.valid_rows
    return weighted_loss / logical_rows


def _set_learning_rate(optimizer: torch.optim.AdamW, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    observed = cast("float", optimizer.param_groups[0]["lr"])
    if observed != learning_rate:
        raise NeuralTrainingError(reason="PyTorch did not retain the scheduled learning rate")


def _require_finite_parameters(network: TorchProjectModel) -> None:
    for name, parameter in network.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise NeuralTrainingError(reason=f"nonfinite trained parameter {name!r}")


__all__ = [
    "TorchTrainingResult",
    "train_torch_project_model",
]
