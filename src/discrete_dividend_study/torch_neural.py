"""PyTorch realization of the framework-neutral project neural models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import torch
from finoptions import BinnedCashDividends, PaddedCashDividends
from torch import nn

from .neural_backend import (
    NeuralBackendContractError,
    validate_neural_build_request,
    validate_prepared_batch,
)
from .neural_execution import (
    NeuralBackend,
    NeuralDevice,
    NeuralExecutionSpec,
    NeuralRuntimeIdentity,
)
from .neural_parameters import NeuralParameter
from .neural_spec import (
    BinnedDividendInput,
    DirectDividendLstmInput,
    DividendYieldInput,
    EventAttentionInput,
    EventFeature,
    NeuralModelSpec,
    NeuralStudySpec,
    TrainingSpec,
)

if TYPE_CHECKING:
    from .model_inputs import PreparedNeuralBatch
    from .neural_batching import FixedBatchIndices
    from .neural_parameters import NeuralParameterSet

Float32Matrix = npt.NDArray[np.float32]
_LSTM_PARAMETER: Final = re.compile(
    r"^lstm\.(?P<kind>weight_ih|weight_hh|bias_ih|bias_hh)_l(?P<layer>[0-9]+)$",
)


@dataclass(frozen=True, slots=True)
class TorchNeuralBatch:
    """One fully transferred model batch on a single explicit PyTorch device."""

    model: NeuralModelSpec
    scalar_features: torch.Tensor
    target: torch.Tensor
    event_times: torch.Tensor | None
    event_amounts: torch.Tensor | None
    event_mask: torch.Tensor | None
    event_lengths: torch.Tensor | None
    binned_amounts: torch.Tensor | None

    def __post_init__(self) -> None:
        """Reject inconsistent shapes, dtypes, devices, and representation fields."""
        rows, device = _validate_batch_base(self)
        dividend_input = self.model.dividend_input
        if (
            type(dividend_input) is DirectDividendLstmInput
            or type(dividend_input) is EventAttentionInput
        ):
            _validate_event_batch(self, rows, dividend_input.layout.capacity, device)
        elif type(dividend_input) is BinnedDividendInput:
            _validate_binned_batch(self, rows, dividend_input.layout.bins, device)
        elif type(dividend_input) is DividendYieldInput:
            if self.binned_amounts is not None:
                raise NeuralBackendContractError(reason="yield batch cannot contain bins")
            _require_no_events(self)


@dataclass(frozen=True, slots=True)
class TorchFixedNeuralBatch:
    """One fixed-width gathered batch and its explicit logical-row mask."""

    values: TorchNeuralBatch
    valid: torch.Tensor

    def __post_init__(self) -> None:
        """Require a nonempty boolean column mask on the values' exact device."""
        if type(self.values) is not TorchNeuralBatch:
            raise NeuralBackendContractError(reason="values must be TorchNeuralBatch")
        valid = _validate_tensor(self.valid, field="valid", dtype=torch.bool, dimensions=2)
        if tuple(valid.shape) != (self.values.scalar_features.shape[0], 1):
            raise NeuralBackendContractError(reason="valid must have shape (batch_size, 1)")
        if valid.device != self.values.scalar_features.device:
            raise NeuralBackendContractError(reason="valid must use the model-batch device")


class TorchProjectModel(nn.Module):
    """One exact eager PyTorch realization loaded from canonical parameters."""

    def __init__(
        self,
        study: NeuralStudySpec,
        model: NeuralModelSpec,
        runtime: NeuralRuntimeIdentity,
    ) -> None:
        """Construct the declared architecture without accepting framework defaults."""
        super().__init__()
        self.study_spec = study
        self.model_spec = model
        self.runtime = runtime
        dividend_input = model.dividend_input
        if type(dividend_input) is DirectDividendLstmInput:
            self.lstm = nn.LSTM(
                input_size=len(dividend_input.event_features),
                hidden_size=dividend_input.hidden_size,
                num_layers=dividend_input.layers,
                bias=dividend_input.bias,
                batch_first=True,
            )
        else:
            self.lstm = None
        if type(dividend_input) is EventAttentionInput:
            self.embedding = nn.ModuleList(
                nn.Linear(
                    len(dividend_input.event_features)
                    if layer == 0
                    else dividend_input.embedding_width,
                    dividend_input.embedding_width,
                )
                for layer in range(dividend_input.embedding_layers)
            )
            self.attention = nn.Linear(dividend_input.embedding_width, 1)
        else:
            self.embedding = nn.ModuleList()
            self.attention = None
        self.mlp = nn.ModuleList(
            nn.Linear(
                model.trunk_input_width if layer == 0 else study.mlp.hidden_width,
                study.mlp.hidden_width,
            )
            for layer in range(study.mlp.hidden_layers)
        )
        self.output = nn.Linear(study.mlp.hidden_width, 1)

    def forward(self, batch: TorchNeuralBatch) -> torch.Tensor:
        """Price one already-validated batch with exact mask and length semantics."""
        if type(batch) is not TorchNeuralBatch or batch.model != self.model_spec:
            raise NeuralBackendContractError(reason="batch does not identify this model")
        trunk = batch.scalar_features
        dividend_input = self.model_spec.dividend_input
        if type(dividend_input) is DirectDividendLstmInput:
            events = _event_tensor(batch, dividend_input.event_features)
            if self.lstm is None or batch.event_lengths is None:
                raise NeuralBackendContractError(reason="LSTM inputs or module are absent")
            hidden, _ = self.lstm(events)
            indices = torch.clamp(batch.event_lengths - 1, min=0).view(-1, 1, 1)
            indices = indices.expand(-1, 1, dividend_input.hidden_size)
            encoded = hidden.gather(1, indices).squeeze(1)
            encoded = torch.where(
                (batch.event_lengths > 0).view(-1, 1),
                encoded,
                torch.zeros_like(encoded),
            )
            trunk = torch.cat((trunk, encoded), dim=1)
        elif type(dividend_input) is EventAttentionInput:
            events = _event_tensor(batch, dividend_input.event_features)
            if self.attention is None or batch.event_mask is None:
                raise NeuralBackendContractError(reason="attention inputs or module are absent")
            embedded = events
            for layer in self.embedding:
                embedded = torch.relu(layer(embedded))
            logits = self.attention(embedded).squeeze(-1)
            encoded = _masked_attention_sum(embedded, logits, batch.event_mask)
            trunk = torch.cat((trunk, encoded), dim=1)
        elif type(dividend_input) is BinnedDividendInput:
            if batch.binned_amounts is None:
                raise NeuralBackendContractError(reason="binned inputs are absent")
            trunk = torch.cat((trunk, batch.binned_amounts), dim=1)
        for layer in self.mlp:
            trunk = torch.nn.functional.leaky_relu(
                layer(trunk),
                negative_slope=self.study_spec.mlp.leaky_relu_negative_slope,
            )
        return torch.nn.functional.softplus(self.output(trunk))


def build_torch_project_model(
    *,
    study: NeuralStudySpec,
    model: NeuralModelSpec,
    parameters: NeuralParameterSet,
    execution: NeuralExecutionSpec,
) -> TorchProjectModel:
    """Build one exact PyTorch model or fail if its declared runtime is unavailable."""
    runtime = validate_neural_build_request(
        study=study,
        model=model,
        parameters=parameters,
        execution=execution,
        backend=NeuralBackend.PYTORCH,
    )
    network = TorchProjectModel(study, model, runtime)
    device = _torch_device(runtime.device)
    network.to(device=device, dtype=torch.float32)
    _load_parameters(network, parameters)
    return network


def prepare_torch_neural_batch(
    batch: PreparedNeuralBatch,
    network: TorchProjectModel,
) -> TorchNeuralBatch:
    """Transfer one validated NumPy/native batch exactly once to the model device."""
    if type(network) is not TorchProjectModel:
        raise NeuralBackendContractError(reason="network must be TorchProjectModel")
    validate_prepared_batch(batch, network.model_spec)
    device = _torch_device(network.runtime.device)
    scalar = _tensor(batch.scalar_features, device=device)
    target = _tensor(batch.target[:, None], device=device)
    event_times: torch.Tensor | None = None
    event_amounts: torch.Tensor | None = None
    event_mask: torch.Tensor | None = None
    event_lengths: torch.Tensor | None = None
    binned_amounts: torch.Tensor | None = None
    if type(batch.dividends) is PaddedCashDividends:
        event_times = _tensor(batch.dividends.times, device=device)
        event_amounts = _tensor(batch.dividends.amounts, device=device)
        event_mask = torch.tensor(batch.dividends.mask, dtype=torch.bool, device=device)
        event_lengths = torch.tensor(batch.dividends.lengths, dtype=torch.int64, device=device)
    elif type(batch.dividends) is BinnedCashDividends:
        binned_amounts = _tensor(batch.dividends.amounts, device=device)
    return TorchNeuralBatch(
        model=batch.model,
        scalar_features=scalar,
        target=target,
        event_times=event_times,
        event_amounts=event_amounts,
        event_mask=event_mask,
        event_lengths=event_lengths,
        binned_amounts=binned_amounts,
    )


def torch_prediction_numpy(
    network: TorchProjectModel,
    batch: TorchNeuralBatch,
) -> Float32Matrix:
    """Evaluate one batch and copy finite float32 predictions to host NumPy."""
    if type(network) is not TorchProjectModel:
        raise NeuralBackendContractError(reason="network must be TorchProjectModel")
    network.eval()
    with torch.no_grad():
        prediction = network(batch)
    result = prediction.detach().cpu().numpy().copy()
    if result.dtype != np.dtype(np.float32) or result.shape != (batch.target.shape[0], 1):
        raise NeuralBackendContractError(reason="PyTorch returned an invalid prediction layout")
    if not np.isfinite(result).all():
        raise NeuralBackendContractError(reason="PyTorch returned nonfinite predictions")
    return result


def torch_smape_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    training: TrainingSpec,
) -> torch.Tensor:
    """Return the declared mean symmetric absolute percentage error."""
    if prediction.shape != target.shape:
        raise NeuralBackendContractError(reason="prediction and target shapes must match")
    if prediction.dtype is not torch.float32 or target.dtype is not torch.float32:
        raise NeuralBackendContractError(reason="sMAPE inputs must be float32")
    denominator = (torch.abs(target) + torch.abs(prediction)) / 2.0
    return (
        torch.mean(torch.abs(target - prediction) / denominator)
        * training.objective_percentage_scale
    )


def torch_fixed_smape_objective(
    prediction: torch.Tensor,
    batch: TorchFixedNeuralBatch,
    training: TrainingSpec,
) -> torch.Tensor:
    """Return sMAPE over logical rows only, excluding canonical physical padding."""
    if prediction.shape != batch.values.target.shape:
        raise NeuralBackendContractError(reason="prediction and target shapes must match")
    denominator = (torch.abs(batch.values.target) + torch.abs(prediction)) / 2.0
    per_row = torch.abs(batch.values.target - prediction) / denominator
    masked_total = torch.where(batch.valid, per_row, torch.zeros_like(per_row)).sum()
    return masked_total / batch.valid.sum() * training.objective_percentage_scale


def gather_torch_fixed_batch(
    source: TorchNeuralBatch,
    batch: FixedBatchIndices,
) -> TorchFixedNeuralBatch:
    """Gather one fixed physical batch without dropping its partial logical tail."""
    if type(source) is not TorchNeuralBatch:
        raise NeuralBackendContractError(reason="source must be TorchNeuralBatch")
    if batch.valid_rows <= 0:
        raise NeuralBackendContractError(reason="a fixed batch must contain a logical row")
    device = source.scalar_features.device
    indices = torch.tensor(batch.indices, dtype=torch.int64, device=device)
    values = TorchNeuralBatch(
        model=source.model,
        scalar_features=torch.index_select(source.scalar_features, 0, indices),
        target=torch.index_select(source.target, 0, indices),
        event_times=_index_select(source.event_times, indices),
        event_amounts=_index_select(source.event_amounts, indices),
        event_mask=_index_select(source.event_mask, indices),
        event_lengths=_index_select(source.event_lengths, indices),
        binned_amounts=_index_select(source.binned_amounts, indices),
    )
    valid = torch.tensor(batch.valid[:, None], dtype=torch.bool, device=device)
    return TorchFixedNeuralBatch(values=values, valid=valid)


def _event_tensor(
    batch: TorchNeuralBatch,
    feature_order: tuple[EventFeature, ...],
) -> torch.Tensor:
    if batch.event_times is None or batch.event_amounts is None:
        raise NeuralBackendContractError(reason="event tensors are absent")
    columns: list[torch.Tensor] = []
    for feature in feature_order:
        if feature is EventFeature.OPTION_MATURITY_YEARS:
            maturity_index = batch.model.scalar_columns.index("yte")
            columns.append(
                batch.scalar_features[:, maturity_index, None].expand_as(batch.event_times)
            )
        elif feature is EventFeature.DIVIDEND_TIME_YEARS:
            columns.append(batch.event_times)
        elif feature in {
            EventFeature.DIVIDEND_CASH_AMOUNT,
            EventFeature.DIVIDEND_AMOUNT_OVER_SPOT,
        }:
            columns.append(batch.event_amounts)
        else:
            raise NeuralBackendContractError(reason=f"unsupported event feature {feature!r}")
    return torch.stack(columns, dim=2)


def _masked_attention_sum(
    embedded: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    has_events = mask.any(dim=1, keepdim=True)
    maximum = (
        torch.where(mask, logits, torch.finfo(logits.dtype).min)
        .max(
            dim=1,
            keepdim=True,
        )
        .values
    )
    maximum = torch.where(has_events, maximum, torch.zeros_like(maximum))
    unnormalized = torch.exp(logits - maximum) * mask
    denominator = unnormalized.sum(dim=1, keepdim=True)
    weights = unnormalized / torch.where(
        denominator > 0.0,
        denominator,
        torch.ones_like(denominator),
    )
    return (weights.unsqueeze(-1) * embedded).sum(dim=1)


def _load_parameters(network: TorchProjectModel, parameters: NeuralParameterSet) -> None:
    canonical_names = {parameter.name for parameter in parameters.parameters}
    consumed: set[str] = set()
    with torch.no_grad():
        for framework_name, destination in network.named_parameters():
            canonical_name = torch_parameter_name_to_canonical(framework_name)
            if canonical_name not in canonical_names:
                raise NeuralBackendContractError(
                    reason=f"missing canonical parameter {canonical_name!r}",
                )
            source = parameters.get(canonical_name)
            if tuple(destination.shape) != source.shape:
                raise NeuralBackendContractError(
                    reason=f"shape mismatch for canonical parameter {canonical_name!r}",
                )
            destination.copy_(torch.tensor(source, dtype=torch.float32, device=destination.device))
            consumed.add(canonical_name)
    if consumed != canonical_names:
        unconsumed = tuple(sorted(canonical_names - consumed))
        raise NeuralBackendContractError(
            reason=f"unconsumed canonical parameters: {unconsumed!r}",
        )


def torch_parameter_name_to_canonical(framework_name: str) -> str:
    """Map a PyTorch state-tree path to its canonical parameter identity."""
    match = _LSTM_PARAMETER.fullmatch(framework_name)
    if match is None:
        return framework_name
    kind = {
        "weight_ih": "weight_input",
        "weight_hh": "weight_hidden",
        "bias_ih": "bias_input",
        "bias_hh": "bias_hidden",
    }[match.group("kind")]
    return f"lstm.{match.group('layer')}.{kind}"


def snapshot_torch_parameters(network: TorchProjectModel) -> tuple[NeuralParameter, ...]:
    """Copy trained parameters to canonical, immutable, backend-neutral arrays."""
    if type(network) is not TorchProjectModel:
        raise NeuralBackendContractError(reason="network must be TorchProjectModel")
    snapshots: list[NeuralParameter] = []
    for framework_name, parameter in network.named_parameters():
        values = np.ascontiguousarray(parameter.detach().cpu().numpy(), dtype=np.float32).copy()
        values.flags.writeable = False
        snapshots.append(
            NeuralParameter(
                name=torch_parameter_name_to_canonical(framework_name),
                values=values,
            ),
        )
    return tuple(sorted(snapshots, key=lambda parameter: parameter.name))


def _tensor(values: Float32Matrix, *, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _torch_device(device: NeuralDevice) -> torch.device:
    name = {
        NeuralDevice.CPU: "cpu",
        NeuralDevice.MPS: "mps",
        NeuralDevice.CUDA: "cuda",
    }.get(device)
    if name is None:
        raise NeuralBackendContractError(reason=f"{device.value} is not a PyTorch device")
    return torch.device(name)


def _validate_tensor(
    value: object,
    *,
    field: str,
    dtype: torch.dtype,
    dimensions: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise NeuralBackendContractError(reason=f"{field} must be torch.Tensor")
    if value.dtype is not dtype or value.ndim != dimensions:
        raise NeuralBackendContractError(
            reason=f"{field} must be {dimensions}-D {dtype}; observed {value.dtype} {value.shape}",
        )
    return value


def _validate_batch_base(batch: TorchNeuralBatch) -> tuple[int, torch.device]:
    if type(batch.model) is not NeuralModelSpec:
        raise NeuralBackendContractError(reason="batch model must be NeuralModelSpec")
    scalar = _validate_tensor(
        batch.scalar_features,
        field="scalar_features",
        dtype=torch.float32,
        dimensions=2,
    )
    rows = scalar.shape[0]
    target = _validate_tensor(
        batch.target,
        field="target",
        dtype=torch.float32,
        dimensions=2,
    )
    if tuple(target.shape) != (rows, 1):
        raise NeuralBackendContractError(reason="target must have shape (rows, 1)")
    if batch.scalar_features.shape[1] != len(batch.model.scalar_columns):
        raise NeuralBackendContractError(reason="scalar feature width does not match model")
    device = batch.scalar_features.device
    if batch.target.device != device:
        raise NeuralBackendContractError(reason="all tensors must use one device")
    return rows, device


def _validate_event_batch(
    batch: TorchNeuralBatch,
    rows: int,
    capacity: int,
    device: torch.device,
) -> None:
    for value, field, dtype, dimensions in (
        (batch.event_times, "event_times", torch.float32, 2),
        (batch.event_amounts, "event_amounts", torch.float32, 2),
        (batch.event_mask, "event_mask", torch.bool, 2),
        (batch.event_lengths, "event_lengths", torch.int64, 1),
    ):
        tensor = _validate_tensor(value, field=field, dtype=dtype, dimensions=dimensions)
        expected = (rows,) if dimensions == 1 else (rows, capacity)
        if tuple(tensor.shape) != expected:
            raise NeuralBackendContractError(reason=f"{field} has incorrect shape")
        if tensor.device != device:
            raise NeuralBackendContractError(reason="all tensors must use one device")
    if batch.binned_amounts is not None:
        raise NeuralBackendContractError(reason="sequence batch cannot contain bins")


def _validate_binned_batch(
    batch: TorchNeuralBatch,
    rows: int,
    bins: int,
    device: torch.device,
) -> None:
    tensor = _validate_tensor(
        batch.binned_amounts,
        field="binned_amounts",
        dtype=torch.float32,
        dimensions=2,
    )
    if tuple(tensor.shape) != (rows, bins):
        raise NeuralBackendContractError(reason="binned_amounts has incorrect shape")
    if tensor.device != device:
        raise NeuralBackendContractError(reason="all tensors must use one device")
    _require_no_events(batch)


def _require_no_events(batch: TorchNeuralBatch) -> None:
    if any(
        value is not None
        for value in (
            batch.event_times,
            batch.event_amounts,
            batch.event_mask,
            batch.event_lengths,
        )
    ):
        raise NeuralBackendContractError(reason="non-event batch cannot contain event tensors")


def _index_select(value: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    return None if value is None else torch.index_select(value, 0, indices)


__all__ = [
    "TorchFixedNeuralBatch",
    "TorchNeuralBatch",
    "TorchProjectModel",
    "build_torch_project_model",
    "gather_torch_fixed_batch",
    "prepare_torch_neural_batch",
    "snapshot_torch_parameters",
    "torch_fixed_smape_objective",
    "torch_parameter_name_to_canonical",
    "torch_prediction_numpy",
    "torch_smape_objective",
]
