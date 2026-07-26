"""MLX realization of the framework-neutral project neural models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx  # ty: ignore[unresolved-import] - MLX does not ship core stubs
import numpy as np
import numpy.typing as npt
from finoptions import BinnedCashDividends, PaddedCashDividends
from mlx import nn
from mlx.utils import tree_flatten

from discrete_dividend_study.models.neural.backend import (
    NeuralBackendContractError,
    validate_neural_build_request,
    validate_prepared_batch,
)
from discrete_dividend_study.models.neural.execution import (
    NeuralBackend,
    NeuralExecutionSpec,
    NeuralRuntimeIdentity,
)
from discrete_dividend_study.models.neural.parameters import NeuralParameter
from discrete_dividend_study.models.neural.specification import (
    AttentionSlotPolicy,
    BinnedDividendInput,
    DirectDividendLstmInput,
    DividendBinOrder,
    DividendYieldInput,
    EventAttentionInput,
    EventFeature,
    EventSequenceOrder,
    NeuralModelSpec,
    NeuralStudySpec,
    TrainingSpec,
)

if TYPE_CHECKING:
    from discrete_dividend_study.models.neural.batching import FixedBatchIndices
    from discrete_dividend_study.models.neural.inputs import PreparedNeuralBatch
    from discrete_dividend_study.models.neural.parameters import NeuralParameterSet

Float32Matrix = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class MlxNeuralBatch:
    """One model batch materialized on the explicit MLX default device."""

    model: NeuralModelSpec
    scalar_features: mx.array
    target: mx.array
    event_times: mx.array | None
    event_amounts: mx.array | None
    event_mask: mx.array | None
    event_lengths: mx.array | None
    binned_amounts: mx.array | None

    def __post_init__(self) -> None:
        """Reject inconsistent shapes, dtypes, and representation fields."""
        rows = _validate_batch_base(self)
        dividend_input = self.model.dividend_input
        if (
            type(dividend_input) is DirectDividendLstmInput
            or type(dividend_input) is EventAttentionInput
        ):
            _validate_event_batch(self, rows, dividend_input.layout.capacity)
        elif type(dividend_input) is BinnedDividendInput:
            _validate_binned_batch(self, rows, dividend_input.layout.bins)
        elif type(dividend_input) is DividendYieldInput:
            if self.binned_amounts is not None:
                raise NeuralBackendContractError(reason="yield batch cannot contain bins")
            _require_no_events(self)
        else:
            raise NeuralBackendContractError(reason="unsupported dividend input")


@dataclass(frozen=True, slots=True)
class MlxFixedNeuralBatch:
    """One fixed-width gathered batch and its explicit logical-row mask."""

    values: MlxNeuralBatch
    valid: mx.array

    def __post_init__(self) -> None:
        """Require a boolean column mask matching the physical batch shape."""
        if type(self.values) is not MlxNeuralBatch:
            raise NeuralBackendContractError(reason="values must be MlxNeuralBatch")
        valid = _validate_array(self.valid, field="valid", dtype=mx.bool_, dimensions=2)
        if valid.shape != (self.values.scalar_features.shape[0], 1):
            raise NeuralBackendContractError(reason="valid must have shape (batch_size, 1)")


class MlxProjectModel(nn.Module):
    """One exact MLX realization loaded from canonical parameters."""

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
            if dividend_input.bias:
                raise NeuralBackendContractError(
                    reason="MLX project LSTM requires the declared bias-free recurrent contract",
                )
            self.lstm = [
                nn.LSTM(
                    len(dividend_input.event_features)
                    if layer == 0
                    else dividend_input.hidden_size,
                    dividend_input.hidden_size,
                    bias=False,
                )
                for layer in range(dividend_input.layers)
            ]
        else:
            self.lstm = []
        if type(dividend_input) is EventAttentionInput:
            self.embedding = [
                nn.Linear(
                    len(dividend_input.event_features)
                    if layer == 0
                    else dividend_input.embedding_width,
                    dividend_input.embedding_width,
                )
                for layer in range(dividend_input.embedding_layers)
            ]
            self.attention = nn.Linear(dividend_input.embedding_width, 1)
        else:
            self.embedding = []
            self.attention = None
        self.mlp = [
            nn.Linear(
                model.trunk_input_width if layer == 0 else study.mlp.hidden_width,
                study.mlp.hidden_width,
            )
            for layer in range(study.mlp.hidden_layers)
        ]
        self.output = nn.Linear(study.mlp.hidden_width, 1)

    def __call__(self, batch: MlxNeuralBatch) -> mx.array:
        """Price one already-validated batch with exact mask and length semantics."""
        if type(batch) is not MlxNeuralBatch or batch.model != self.model_spec:
            raise NeuralBackendContractError(reason="batch does not identify this model")
        trunk = self._trunk_input(batch)
        for layer in self.mlp:
            trunk = nn.leaky_relu(
                layer(trunk),
                negative_slope=self.study_spec.mlp.leaky_relu_negative_slope,
            )
        return nn.softplus(self.output(trunk))

    def _trunk_input(self, batch: MlxNeuralBatch) -> mx.array:
        """Append exactly the dividend representation declared by this model."""
        dividend_input = self.model_spec.dividend_input
        if type(dividend_input) is DirectDividendLstmInput:
            encoded = self._encode_lstm(batch, dividend_input)
            return mx.concatenate((batch.scalar_features, encoded), axis=1)
        if type(dividend_input) is EventAttentionInput:
            encoded = self._encode_attention(batch, dividend_input)
            return mx.concatenate((batch.scalar_features, encoded), axis=1)
        if type(dividend_input) is BinnedDividendInput:
            if batch.binned_amounts is None:
                raise NeuralBackendContractError(reason="binned inputs are absent")
            bins = _order_dividend_bins(batch.binned_amounts, dividend_input.order)
            return mx.concatenate((batch.scalar_features, bins), axis=1)
        if type(dividend_input) is DividendYieldInput:
            return batch.scalar_features
        raise NeuralBackendContractError(reason="unsupported dividend input")

    def _encode_lstm(
        self,
        batch: MlxNeuralBatch,
        dividend_input: DirectDividendLstmInput,
    ) -> mx.array:
        """Return the last real recurrent state, or zero for an empty schedule."""
        events = _event_tensor(batch, dividend_input.event_features)
        if batch.event_lengths is None:
            raise NeuralBackendContractError(reason="LSTM lengths are absent")
        hidden = _order_lstm_events(
            events,
            batch.event_lengths,
            dividend_input.sequence_order,
        )
        for layer in self.lstm:
            hidden, _ = layer(hidden)
        rows = mx.arange(batch.scalar_features.shape[0])
        indices = mx.maximum(batch.event_lengths - 1, 0)
        encoded = hidden[rows, indices]
        return mx.where(
            (batch.event_lengths > 0)[:, None],
            encoded,
            mx.zeros_like(encoded),
        )

    def _encode_attention(
        self,
        batch: MlxNeuralBatch,
        dividend_input: EventAttentionInput,
    ) -> mx.array:
        """Return the declared fixed-capacity event representation."""
        events = _event_tensor(batch, dividend_input.event_features)
        if self.attention is None or batch.event_mask is None:
            raise NeuralBackendContractError(reason="attention inputs or module are absent")
        embedded = events
        for layer in self.embedding:
            embedded = mx.maximum(layer(embedded), 0.0)
        logits = self.attention(embedded).squeeze(-1)
        return _attention_sum(
            embedded,
            logits,
            dividend_input.slot_policy,
        )


def build_mlx_project_model(
    *,
    study: NeuralStudySpec,
    model: NeuralModelSpec,
    parameters: NeuralParameterSet,
    execution: NeuralExecutionSpec,
) -> MlxProjectModel:
    """Build one exact MLX model or fail if its declared runtime is unavailable."""
    runtime = validate_neural_build_request(
        study=study,
        model=model,
        parameters=parameters,
        execution=execution,
        backend=NeuralBackend.MLX,
    )
    network = MlxProjectModel(study, model, runtime)
    _load_parameters(network, parameters)
    mx.eval(network.parameters())
    return network


def prepare_mlx_neural_batch(
    batch: PreparedNeuralBatch,
    network: MlxProjectModel,
) -> MlxNeuralBatch:
    """Materialize one validated NumPy/native batch on the requested MLX device."""
    if type(network) is not MlxProjectModel:
        raise NeuralBackendContractError(reason="network must be MlxProjectModel")
    validate_prepared_batch(batch, network.model_spec)
    scalar = mx.array(batch.scalar_features, dtype=mx.float32)
    target = mx.array(batch.target[:, None], dtype=mx.float32)
    event_times: mx.array | None = None
    event_amounts: mx.array | None = None
    event_mask: mx.array | None = None
    event_lengths: mx.array | None = None
    binned_amounts: mx.array | None = None
    if type(batch.dividends) is PaddedCashDividends:
        event_times = mx.array(batch.dividends.times, dtype=mx.float32)
        event_amounts = mx.array(batch.dividends.amounts, dtype=mx.float32)
        event_mask = mx.array(batch.dividends.mask, dtype=mx.bool_)
        event_lengths = mx.array(batch.dividends.lengths, dtype=mx.int64)
    elif type(batch.dividends) is BinnedCashDividends:
        binned_amounts = mx.array(batch.dividends.amounts, dtype=mx.float32)
    transferred = MlxNeuralBatch(
        model=batch.model,
        scalar_features=scalar,
        target=target,
        event_times=event_times,
        event_amounts=event_amounts,
        event_mask=event_mask,
        event_lengths=event_lengths,
        binned_amounts=binned_amounts,
    )
    mx.eval(
        transferred.scalar_features,
        transferred.target,
        transferred.event_times,
        transferred.event_amounts,
        transferred.event_mask,
        transferred.event_lengths,
        transferred.binned_amounts,
    )
    return transferred


def mlx_prediction_numpy(
    network: MlxProjectModel,
    batch: MlxNeuralBatch,
) -> Float32Matrix:
    """Evaluate one batch and copy finite float32 predictions to host NumPy."""
    if type(network) is not MlxProjectModel:
        raise NeuralBackendContractError(reason="network must be MlxProjectModel")
    prediction = network(batch)
    mx.eval(prediction)
    result = np.array(prediction, dtype=np.float32, copy=True)
    if result.dtype != np.dtype(np.float32) or result.shape != (batch.target.shape[0], 1):
        raise NeuralBackendContractError(reason="MLX returned an invalid prediction layout")
    if not np.isfinite(result).all():
        raise NeuralBackendContractError(reason="MLX returned nonfinite predictions")
    return result


def mlx_smape_objective(
    prediction: mx.array,
    target: mx.array,
    training: TrainingSpec,
) -> mx.array:
    """Return the declared mean symmetric absolute percentage error."""
    if prediction.shape != target.shape:
        raise NeuralBackendContractError(reason="prediction and target shapes must match")
    if prediction.dtype != mx.float32 or target.dtype != mx.float32:
        raise NeuralBackendContractError(reason="sMAPE inputs must be float32")
    denominator = (mx.abs(target) + mx.abs(prediction)) / 2.0
    return mx.mean(mx.abs(target - prediction) / denominator) * training.objective_percentage_scale


def mlx_fixed_smape_objective(
    prediction: mx.array,
    batch: MlxFixedNeuralBatch,
    training: TrainingSpec,
) -> mx.array:
    """Return sMAPE over logical rows only, excluding canonical physical padding."""
    if prediction.shape != batch.values.target.shape:
        raise NeuralBackendContractError(reason="prediction and target shapes must match")
    denominator = (mx.abs(batch.values.target) + mx.abs(prediction)) / 2.0
    per_row = mx.abs(batch.values.target - prediction) / denominator
    masked_total = mx.sum(mx.where(batch.valid, per_row, mx.zeros_like(per_row)))
    return masked_total / mx.sum(batch.valid) * training.objective_percentage_scale


def gather_mlx_fixed_batch(
    source: MlxNeuralBatch,
    batch: FixedBatchIndices,
) -> MlxFixedNeuralBatch:
    """Gather one lazy fixed physical batch without dropping its partial logical tail."""
    if type(source) is not MlxNeuralBatch:
        raise NeuralBackendContractError(reason="source must be MlxNeuralBatch")
    if batch.valid_rows <= 0:
        raise NeuralBackendContractError(reason="a fixed batch must contain a logical row")
    indices = mx.array(batch.indices, dtype=mx.int64)
    valid = mx.array(batch.valid, dtype=mx.bool_)
    return gather_mlx_device_batch(source, indices, valid)


def gather_mlx_device_batch(
    source: MlxNeuralBatch,
    indices: mx.array,
    valid: mx.array,
) -> MlxFixedNeuralBatch:
    """Gather one already-transferred fixed index row without per-batch host copies."""
    if type(source) is not MlxNeuralBatch:
        raise NeuralBackendContractError(reason="source must be MlxNeuralBatch")
    indices = _validate_array(indices, field="indices", dtype=mx.int64, dimensions=1)
    valid = _validate_array(valid, field="valid", dtype=mx.bool_, dimensions=1)
    if indices.shape != valid.shape or indices.size == 0:
        raise NeuralBackendContractError(reason="indices and valid must share a nonempty shape")
    values = MlxNeuralBatch(
        model=source.model,
        scalar_features=source.scalar_features[indices],
        target=source.target[indices],
        event_times=_index_select(source.event_times, indices),
        event_amounts=_index_select(source.event_amounts, indices),
        event_mask=_index_select(source.event_mask, indices),
        event_lengths=_index_select(source.event_lengths, indices),
        binned_amounts=_index_select(source.binned_amounts, indices),
    )
    return MlxFixedNeuralBatch(values=values, valid=valid[:, None])


def mlx_parameter_name_to_canonical(framework_name: str) -> str:
    """Map an MLX parameter-tree path to its canonical parameter identity."""
    parts = framework_name.split(".")
    match parts:
        case ["lstm", layer, gate]:
            kind = {"Wx": "weight_input", "Wh": "weight_hidden"}.get(gate)
            if kind is not None:
                return f"lstm.{layer}.{kind}"
    return framework_name


def snapshot_mlx_parameters(network: MlxProjectModel) -> tuple[NeuralParameter, ...]:
    """Copy trained parameters to canonical, immutable, backend-neutral arrays."""
    if type(network) is not MlxProjectModel:
        raise NeuralBackendContractError(reason="network must be MlxProjectModel")
    mx.eval(network.parameters())
    snapshots: list[NeuralParameter] = []
    for framework_name, parameter in tree_flatten(network.parameters()):
        values = np.ascontiguousarray(np.array(parameter, dtype=np.float32, copy=True))
        values.flags.writeable = False
        snapshots.append(
            NeuralParameter(
                name=mlx_parameter_name_to_canonical(framework_name),
                values=values,
            ),
        )
    return tuple(sorted(snapshots, key=lambda parameter: parameter.name))


def _event_tensor(
    batch: MlxNeuralBatch,
    feature_order: tuple[EventFeature, ...],
) -> mx.array:
    if batch.event_times is None or batch.event_amounts is None:
        raise NeuralBackendContractError(reason="event tensors are absent")
    columns: list[mx.array] = []
    for feature in feature_order:
        if feature is EventFeature.OPTION_MATURITY_YEARS:
            maturity_index = batch.model.scalar_columns.index("yte")
            maturity = mx.broadcast_to(
                batch.scalar_features[:, maturity_index, None],
                batch.event_times.shape,
            )
            columns.append(maturity)
        elif feature is EventFeature.DIVIDEND_TIME_YEARS:
            columns.append(batch.event_times)
        elif feature in {
            EventFeature.DIVIDEND_CASH_AMOUNT,
            EventFeature.DIVIDEND_AMOUNT_OVER_SPOT,
        }:
            columns.append(batch.event_amounts)
        else:
            raise NeuralBackendContractError(reason=f"unsupported event feature {feature!r}")
    return mx.stack(columns, axis=2)


def _order_lstm_events(
    events: mx.array,
    lengths: mx.array,
    order: EventSequenceOrder,
) -> mx.array:
    """Return right-padded events in the exact order consumed by the project LSTM."""
    if order is not EventSequenceOrder.FARTHEST_TO_NEAREST:
        raise NeuralBackendContractError(reason=f"unsupported LSTM event order {order!r}")
    positions = mx.broadcast_to(mx.arange(events.shape[1])[None, :], events.shape[:2])
    reverse = mx.where(
        positions < lengths[:, None],
        lengths[:, None] - 1 - positions,
        positions,
    )
    rows = mx.broadcast_to(mx.arange(events.shape[0])[:, None], events.shape[:2])
    return events[rows, reverse]


def _order_dividend_bins(
    bins: mx.array,
    order: DividendBinOrder,
) -> mx.array:
    """Return monthly cash bins in the exact order consumed by the project MLP."""
    if order is not DividendBinOrder.FARTHEST_TO_NEAREST:
        raise NeuralBackendContractError(reason=f"unsupported dividend-bin order {order!r}")
    return bins[:, ::-1]


def _attention_sum(
    embedded: mx.array,
    logits: mx.array,
    policy: AttentionSlotPolicy,
) -> mx.array:
    """Apply the paper model's fixed-capacity event softmax."""
    if policy is AttentionSlotPolicy.INCLUDE_ZERO_PADDING:
        weights = mx.softmax(logits, axis=1)
        return mx.sum(weights[..., None] * embedded, axis=1)
    raise NeuralBackendContractError(reason=f"unsupported attention slot policy {policy!r}")


def _load_parameters(network: MlxProjectModel, parameters: NeuralParameterSet) -> None:
    consumed: set[str] = set()
    for index, layer in enumerate(network.lstm):
        layer.Wx = _parameter(parameters, f"lstm.{index}.weight_input", consumed)
        layer.Wh = _parameter(parameters, f"lstm.{index}.weight_hidden", consumed)
    for index, layer in enumerate(network.embedding):
        layer.weight = _parameter(parameters, f"embedding.{index}.weight", consumed)
        layer.bias = _parameter(parameters, f"embedding.{index}.bias", consumed)
    if network.attention is not None:
        network.attention.weight = _parameter(parameters, "attention.weight", consumed)
        network.attention.bias = _parameter(parameters, "attention.bias", consumed)
    for index, layer in enumerate(network.mlp):
        layer.weight = _parameter(parameters, f"mlp.{index}.weight", consumed)
        layer.bias = _parameter(parameters, f"mlp.{index}.bias", consumed)
    network.output.weight = _parameter(parameters, "output.weight", consumed)
    network.output.bias = _parameter(parameters, "output.bias", consumed)
    canonical_names = {parameter.name for parameter in parameters.parameters}
    if consumed != canonical_names:
        unconsumed = tuple(sorted(canonical_names - consumed))
        raise NeuralBackendContractError(
            reason=f"unconsumed canonical parameters: {unconsumed!r}",
        )


def _parameter(
    parameters: NeuralParameterSet,
    name: str,
    consumed: set[str],
) -> mx.array:
    try:
        values = parameters.get(name)
    except KeyError as error:
        raise NeuralBackendContractError(reason=f"missing canonical parameter {name!r}") from error
    consumed.add(name)
    return mx.array(values, dtype=mx.float32)


def _validate_array(
    value: object,
    *,
    field: str,
    dtype: mx.Dtype,
    dimensions: int,
) -> mx.array:
    if not isinstance(value, mx.array):
        raise NeuralBackendContractError(reason=f"{field} must be mlx.core.array")
    if value.dtype != dtype or value.ndim != dimensions:
        raise NeuralBackendContractError(
            reason=f"{field} must be {dimensions}-D {dtype}; observed {value.dtype} {value.shape}",
        )
    return value


def _validate_batch_base(batch: MlxNeuralBatch) -> int:
    if type(batch.model) is not NeuralModelSpec:
        raise NeuralBackendContractError(reason="batch model must be NeuralModelSpec")
    scalar = _validate_array(
        batch.scalar_features,
        field="scalar_features",
        dtype=mx.float32,
        dimensions=2,
    )
    rows = scalar.shape[0]
    target = _validate_array(batch.target, field="target", dtype=mx.float32, dimensions=2)
    if target.shape != (rows, 1):
        raise NeuralBackendContractError(reason="target must have shape (rows, 1)")
    if scalar.shape[1] != len(batch.model.scalar_columns):
        raise NeuralBackendContractError(reason="scalar feature width does not match model")
    return rows


def _validate_event_batch(batch: MlxNeuralBatch, rows: int, capacity: int) -> None:
    for value, field, dtype, dimensions in (
        (batch.event_times, "event_times", mx.float32, 2),
        (batch.event_amounts, "event_amounts", mx.float32, 2),
        (batch.event_mask, "event_mask", mx.bool_, 2),
        (batch.event_lengths, "event_lengths", mx.int64, 1),
    ):
        array = _validate_array(value, field=field, dtype=dtype, dimensions=dimensions)
        expected = (rows,) if dimensions == 1 else (rows, capacity)
        if array.shape != expected:
            raise NeuralBackendContractError(reason=f"{field} has incorrect shape")
    if batch.binned_amounts is not None:
        raise NeuralBackendContractError(reason="sequence batch cannot contain bins")


def _validate_binned_batch(batch: MlxNeuralBatch, rows: int, bins: int) -> None:
    array = _validate_array(
        batch.binned_amounts,
        field="binned_amounts",
        dtype=mx.float32,
        dimensions=2,
    )
    if array.shape != (rows, bins):
        raise NeuralBackendContractError(reason="binned_amounts has incorrect shape")
    _require_no_events(batch)


def _require_no_events(batch: MlxNeuralBatch) -> None:
    if any(
        value is not None
        for value in (
            batch.event_times,
            batch.event_amounts,
            batch.event_mask,
            batch.event_lengths,
        )
    ):
        raise NeuralBackendContractError(reason="non-event batch cannot contain event arrays")


def _index_select(value: mx.array | None, indices: mx.array) -> mx.array | None:
    return None if value is None else value[indices]


__all__ = [
    "MlxFixedNeuralBatch",
    "MlxNeuralBatch",
    "MlxProjectModel",
    "build_mlx_project_model",
    "gather_mlx_device_batch",
    "gather_mlx_fixed_batch",
    "mlx_fixed_smape_objective",
    "mlx_parameter_name_to_canonical",
    "mlx_prediction_numpy",
    "mlx_smape_objective",
    "prepare_mlx_neural_batch",
    "snapshot_mlx_parameters",
]
