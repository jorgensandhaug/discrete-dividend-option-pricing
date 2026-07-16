"""Exact gates for the submitted fixed-capacity event-attention compatibility model."""

from __future__ import annotations

from typing import Final

import mlx.core as mx  # ty: ignore[unresolved-import] - MLX does not ship core stubs
import mlx.nn as mlx_nn
import numpy as np
import torch
from finoptions import PaddedCashDividends
from mlx.utils import tree_flatten

from discrete_dividend_study.mlx_neural import (
    MlxNeuralBatch,
    MlxProjectModel,
    build_mlx_project_model,
    mlx_parameter_name_to_canonical,
    mlx_prediction_numpy,
    mlx_smape_objective,
    prepare_mlx_neural_batch,
)
from discrete_dividend_study.mlx_neural import _attention_sum as mlx_attention_sum
from discrete_dividend_study.model_inputs import PreparedNeuralBatch
from discrete_dividend_study.neural_execution import (
    CompilationMode,
    NeuralBackend,
    NeuralDevice,
    NeuralExecutionSpec,
)
from discrete_dividend_study.neural_parameters import initialize_neural_parameters
from discrete_dividend_study.neural_spec import (
    PROJECT_AS_SUBMITTED_NEURAL_STUDY_SPEC,
    AttentionSlotPolicy,
    ModelPrecision,
    NeuralStudySpec,
    NeuralVariant,
)
from discrete_dividend_study.torch_neural import (
    TorchNeuralBatch,
    TorchProjectModel,
    build_torch_project_model,
    prepare_torch_neural_batch,
    torch_parameter_name_to_canonical,
    torch_prediction_numpy,
    torch_smape_objective,
)
from discrete_dividend_study.torch_neural import _attention_sum as torch_attention_sum

_ROWS: Final = 8
_CAPACITY: Final = 5
_SEED: Final = 20260716
_FORWARD_TOLERANCE: Final = 2e-6
_GRADIENT_TOLERANCE: Final = 2e-4
_MLX_CPU = NeuralExecutionSpec(
    backend=NeuralBackend.MLX,
    device=NeuralDevice.CPU,
    precision=ModelPrecision.FLOAT32,
    compilation=CompilationMode.EAGER,
    cpu_threads=None,
    inference_batch_size=_ROWS,
)
_PYTORCH_CPU = NeuralExecutionSpec(
    backend=NeuralBackend.PYTORCH,
    device=NeuralDevice.CPU,
    precision=ModelPrecision.FLOAT32,
    compilation=CompilationMode.EAGER,
    cpu_threads=1,
    inference_batch_size=_ROWS,
)


def test_submitted_attention_forward_and_gradients_agree_across_backends() -> None:
    """Independent graphs implement the recovered unmasked fixed-capacity softmax."""
    study = PROJECT_AS_SUBMITTED_NEURAL_STUDY_SPEC
    prepared = _prepared_embedding_batch(study)
    parameters = initialize_neural_parameters(study, prepared.model, _SEED)
    torch_model = build_torch_project_model(
        study=study,
        model=prepared.model,
        parameters=parameters,
        execution=_PYTORCH_CPU,
    )
    mlx_model = build_mlx_project_model(
        study=study,
        model=prepared.model,
        parameters=parameters,
        execution=_MLX_CPU,
    )
    torch_batch = prepare_torch_neural_batch(prepared, torch_model)
    mlx_batch = prepare_mlx_neural_batch(prepared, mlx_model)
    np.testing.assert_allclose(
        torch_prediction_numpy(torch_model, torch_batch),
        mlx_prediction_numpy(mlx_model, mlx_batch),
        rtol=0.0,
        atol=_FORWARD_TOLERANCE,
    )
    torch_gradients = _torch_gradients(torch_model, torch_batch, study)
    mlx_gradients = _mlx_gradients(mlx_model, mlx_batch, study)
    assert torch_gradients.keys() == mlx_gradients.keys()
    for name in torch_gradients:
        np.testing.assert_allclose(
            torch_gradients[name],
            mlx_gradients[name],
            rtol=0.0,
            atol=_GRADIENT_TOLERANCE,
            err_msg=name,
        )


def test_attention_slot_policies_equal_independent_closed_form_formulas() -> None:
    """Masking and submitted padding inclusion are explicit, finite, and materially distinct."""
    embedded_values = np.array(
        [
            [[1.0, 2.0], [4.0, 3.0], [2.0, 8.0]],
            [[3.0, 1.0], [5.0, 7.0], [9.0, 2.0]],
        ],
        dtype=np.float32,
    )
    logits_values = np.array([[0.2, -0.4, 0.7], [1.0, -1.0, 0.5]], dtype=np.float32)
    mask_values = np.array([[True, False, False], [False, False, False]])
    fixed_weights = _numpy_softmax(logits_values)
    expected_fixed = np.sum(fixed_weights[..., None] * embedded_values, axis=1)
    expected_masked = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)

    torch_embedded = torch.tensor(embedded_values)
    torch_logits = torch.tensor(logits_values)
    torch_mask = torch.tensor(mask_values)
    observed_torch_fixed = torch_attention_sum(
        torch_embedded,
        torch_logits,
        torch_mask,
        AttentionSlotPolicy.INCLUDE_ZERO_PADDING,
    ).numpy()
    observed_torch_masked = torch_attention_sum(
        torch_embedded,
        torch_logits,
        torch_mask,
        AttentionSlotPolicy.MASK_INVALID_EVENTS,
    ).numpy()

    mlx_embedded = mx.array(embedded_values)
    mlx_logits = mx.array(logits_values)
    mlx_mask = mx.array(mask_values)
    observed_mlx_fixed_array = mlx_attention_sum(
        mlx_embedded,
        mlx_logits,
        mlx_mask,
        AttentionSlotPolicy.INCLUDE_ZERO_PADDING,
    )
    observed_mlx_masked_array = mlx_attention_sum(
        mlx_embedded,
        mlx_logits,
        mlx_mask,
        AttentionSlotPolicy.MASK_INVALID_EVENTS,
    )
    mx.eval(observed_mlx_fixed_array, observed_mlx_masked_array)
    observed_mlx_fixed = np.array(observed_mlx_fixed_array, copy=True)
    observed_mlx_masked = np.array(observed_mlx_masked_array, copy=True)

    for observed in (observed_torch_fixed, observed_mlx_fixed):
        np.testing.assert_allclose(observed, expected_fixed, rtol=0.0, atol=3e-7)
    for observed in (observed_torch_masked, observed_mlx_masked):
        np.testing.assert_array_equal(observed, expected_masked)
    assert not np.array_equal(expected_fixed, expected_masked)


def _prepared_embedding_batch(study: NeuralStudySpec) -> PreparedNeuralBatch:
    model = study.model(NeuralVariant.EMBEDDING)
    scalar = np.empty((_ROWS, len(model.scalar_columns)), dtype=np.float32)
    base = {
        "strike": np.linspace(80.0, 120.0, _ROWS, dtype=np.float32),
        "underlying_price": np.linspace(95.0, 105.0, _ROWS, dtype=np.float32),
        "risk_free_rate": np.linspace(0.01, 0.04, _ROWS, dtype=np.float32),
        "call_moneyness": np.linspace(0.8, 1.2, _ROWS, dtype=np.float32),
        "historical_volatility_90": np.linspace(0.15, 0.35, _ROWS, dtype=np.float32),
        "yte": np.linspace(0.4, 1.0, _ROWS, dtype=np.float32),
    }
    for column, name in enumerate(model.scalar_columns):
        scalar[:, column] = base[name]
    lengths = np.array([0, 1, 2, 3, 4, 5, 1, 2], dtype=np.uint32)
    mask = np.arange(_CAPACITY, dtype=np.uint32)[None, :] < lengths[:, None]
    times = np.zeros((_ROWS, _CAPACITY), dtype=np.float32)
    amounts = np.zeros((_ROWS, _CAPACITY), dtype=np.float32)
    maturities = scalar[:, model.scalar_columns.index("yte")]
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length > 0:
            times[row, :length] = np.linspace(0.1, maturities[row], length, dtype=np.float32)
            amounts[row, :length] = np.linspace(0.005, 0.02, length, dtype=np.float32)
    return PreparedNeuralBatch(
        model=model,
        scalar_features=np.ascontiguousarray(scalar),
        target=np.linspace(0.5, 20.0, _ROWS, dtype=np.float32),
        dividends=PaddedCashDividends(
            times=times,
            amounts=amounts,
            mask=np.ascontiguousarray(mask),
            lengths=lengths,
        ),
    )


def _torch_gradients(
    network: TorchProjectModel,
    batch: TorchNeuralBatch,
    study: NeuralStudySpec,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    network.train()
    network.zero_grad(set_to_none=True)
    prediction = network(batch)
    loss = torch_smape_objective(prediction, batch.target, study.training)
    loss.backward()
    return {
        torch_parameter_name_to_canonical(name): parameter.grad.detach().cpu().numpy().copy()
        for name, parameter in network.named_parameters()
        if parameter.grad is not None
    }


def _mlx_gradients(
    network: MlxProjectModel,
    batch: MlxNeuralBatch,
    study: NeuralStudySpec,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    def loss_function(candidate: MlxProjectModel) -> mx.array:
        prediction = candidate(batch)
        return mlx_smape_objective(prediction, batch.target, study.training)

    _, gradients = mlx_nn.value_and_grad(network, loss_function)(network)
    mx.eval(gradients)
    return {
        mlx_parameter_name_to_canonical(name): np.array(value, dtype=np.float32, copy=True)
        for name, value in tree_flatten(gradients)
    }


def _numpy_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)
