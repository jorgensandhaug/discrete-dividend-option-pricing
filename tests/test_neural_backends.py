"""Cross-backend gates for the five canonical project neural models."""

from __future__ import annotations

from typing import Final

import mlx.core as mx  # ty: ignore[unresolved-import] - MLX does not ship core stubs
import mlx.nn as mlx_nn
import numpy as np
import pytest
from finoptions import BinnedCashDividends, PaddedCashDividends
from mlx.utils import tree_flatten

from discrete_dividend_study.mlx_neural import (
    MlxNeuralBatch,
    MlxProjectModel,
    build_mlx_project_model,
    gather_mlx_fixed_batch,
    mlx_fixed_smape_objective,
    mlx_parameter_name_to_canonical,
    mlx_prediction_numpy,
    mlx_smape_objective,
    prepare_mlx_neural_batch,
)
from discrete_dividend_study.mlx_neural import (
    _order_dividend_bins as order_mlx_dividend_bins,
)
from discrete_dividend_study.mlx_neural import (
    _order_lstm_events as order_mlx_lstm_events,
)
from discrete_dividend_study.model_inputs import PreparedNeuralBatch
from discrete_dividend_study.neural_backend import NeuralBackendContractError
from discrete_dividend_study.neural_batching import (
    NeuralBatchPlan,
    build_neural_epoch_batch_indices,
)
from discrete_dividend_study.neural_execution import (
    CompilationMode,
    NeuralBackend,
    NeuralDevice,
    NeuralExecutionSpec,
)
from discrete_dividend_study.neural_parameters import initialize_neural_parameters
from discrete_dividend_study.neural_spec import (
    PROJECT_NEURAL_STUDY_SPEC,
    BinnedDividendInput,
    DirectDividendLstmInput,
    ModelPrecision,
    NeuralVariant,
)
from discrete_dividend_study.torch_neural import (
    TorchNeuralBatch,
    TorchProjectModel,
    build_torch_project_model,
    gather_torch_fixed_batch,
    prepare_torch_neural_batch,
    torch_fixed_smape_objective,
    torch_parameter_name_to_canonical,
    torch_prediction_numpy,
    torch_smape_objective,
)
from discrete_dividend_study.torch_neural import (
    _order_dividend_bins as order_torch_dividend_bins,
)
from discrete_dividend_study.torch_neural import (
    _order_lstm_events as order_torch_lstm_events,
)

ROWS: Final = 8
CAPACITY: Final = 5
SEED: Final = 20261207
FORWARD_TOLERANCE: Final = 2e-6
GRADIENT_TOLERANCE: Final = 2e-4

_MLX_CPU = NeuralExecutionSpec(
    backend=NeuralBackend.MLX,
    device=NeuralDevice.CPU,
    precision=ModelPrecision.FLOAT32,
    compilation=CompilationMode.EAGER,
    cpu_threads=None,
    inference_batch_size=8,
)
_PYTORCH_CPU_1 = NeuralExecutionSpec(
    backend=NeuralBackend.PYTORCH,
    device=NeuralDevice.CPU,
    precision=ModelPrecision.FLOAT32,
    compilation=CompilationMode.EAGER,
    cpu_threads=1,
    inference_batch_size=8,
)


@pytest.mark.parametrize("variant", list(NeuralVariant))
def test_canonical_forward_and_parameter_gradients_agree_across_backends(
    variant: NeuralVariant,
) -> None:
    """Independent MLX and PyTorch graphs agree from one canonical parameter set."""
    prepared = _prepared_batch(variant)
    model = prepared.model
    parameters = initialize_neural_parameters(PROJECT_NEURAL_STUDY_SPEC, model, SEED)
    torch_model = build_torch_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=model,
        parameters=parameters,
        execution=_PYTORCH_CPU_1,
    )
    mlx_model = build_mlx_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=model,
        parameters=parameters,
        execution=_MLX_CPU,
    )
    torch_batch = prepare_torch_neural_batch(prepared, torch_model)
    mlx_batch = prepare_mlx_neural_batch(prepared, mlx_model)

    torch_prediction = torch_prediction_numpy(torch_model, torch_batch)
    mlx_prediction = mlx_prediction_numpy(mlx_model, mlx_batch)
    assert np.isfinite(torch_prediction).all()
    assert np.isfinite(mlx_prediction).all()
    np.testing.assert_allclose(
        torch_prediction,
        mlx_prediction,
        rtol=0.0,
        atol=FORWARD_TOLERANCE,
    )

    torch_gradients = _torch_gradients(torch_model, torch_batch)
    mlx_gradients = _mlx_gradients(mlx_model, mlx_batch)
    assert torch_gradients.keys() == mlx_gradients.keys()
    for name in torch_gradients:
        assert np.isfinite(torch_gradients[name]).all(), name
        assert np.isfinite(mlx_gradients[name]).all(), name
        np.testing.assert_allclose(
            torch_gradients[name],
            mlx_gradients[name],
            rtol=0.0,
            atol=GRADIENT_TOLERANCE,
            err_msg=name,
        )


def test_empty_event_schedules_have_finite_zero_weight_attention() -> None:
    """An all-masked first row remains finite in both graphs and their gradients."""
    prepared = _prepared_batch(NeuralVariant.EMBEDDING)
    assert isinstance(prepared.dividends, PaddedCashDividends)
    assert prepared.dividends.lengths[0] == 0
    model = prepared.model
    parameters = initialize_neural_parameters(PROJECT_NEURAL_STUDY_SPEC, model, SEED)
    torch_model = build_torch_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=model,
        parameters=parameters,
        execution=_PYTORCH_CPU_1,
    )
    mlx_model = build_mlx_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=model,
        parameters=parameters,
        execution=_MLX_CPU,
    )
    torch_batch = prepare_torch_neural_batch(prepared, torch_model)
    mlx_batch = prepare_mlx_neural_batch(prepared, mlx_model)
    assert np.isfinite(torch_prediction_numpy(torch_model, torch_batch)[0]).all()
    assert np.isfinite(mlx_prediction_numpy(mlx_model, mlx_batch)[0]).all()


def test_lstm_consumes_real_events_farthest_to_nearest() -> None:
    """Both backends expose the submitted reverse-chronological recurrent contract."""
    prepared = _prepared_batch(NeuralVariant.LSTM)
    model = prepared.model
    parameters = initialize_neural_parameters(PROJECT_NEURAL_STUDY_SPEC, model, SEED)
    torch_model = build_torch_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=model,
        parameters=parameters,
        execution=_PYTORCH_CPU_1,
    )
    mlx_model = build_mlx_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=model,
        parameters=parameters,
        execution=_MLX_CPU,
    )
    torch_batch = prepare_torch_neural_batch(prepared, torch_model)
    mlx_batch = prepare_mlx_neural_batch(prepared, mlx_model)
    assert torch_batch.event_times is not None
    assert torch_batch.event_lengths is not None
    assert mlx_batch.event_times is not None
    assert mlx_batch.event_lengths is not None

    assert isinstance(model.dividend_input, DirectDividendLstmInput)
    order = model.dividend_input.sequence_order
    torch_ordered = (
        order_torch_lstm_events(
            torch_batch.event_times[:, :, None],
            torch_batch.event_lengths,
            order,
        )[:, :, 0]
        .detach()
        .cpu()
        .numpy()
    )
    mlx_ordered_array = order_mlx_lstm_events(
        mlx_batch.event_times[:, :, None],
        mlx_batch.event_lengths,
        order,
    )[:, :, 0]
    mx.eval(mlx_ordered_array)
    mlx_ordered = np.array(mlx_ordered_array, copy=True)
    assert isinstance(prepared.dividends, PaddedCashDividends)
    expected = prepared.dividends.times.copy()
    for row, length_value in enumerate(prepared.dividends.lengths):
        length = int(length_value)
        expected[row, :length] = expected[row, :length][::-1]
    np.testing.assert_array_equal(torch_ordered, expected)
    np.testing.assert_array_equal(mlx_ordered, expected)


def test_binned_model_consumes_months_farthest_to_nearest() -> None:
    """Both backends expose the submitted reverse monthly-bin input contract."""
    prepared = _prepared_batch(NeuralVariant.BINS)
    assert isinstance(prepared.dividends, BinnedCashDividends)
    model = prepared.model
    parameters = initialize_neural_parameters(PROJECT_NEURAL_STUDY_SPEC, model, SEED)
    torch_model = build_torch_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=model,
        parameters=parameters,
        execution=_PYTORCH_CPU_1,
    )
    mlx_model = build_mlx_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=model,
        parameters=parameters,
        execution=_MLX_CPU,
    )
    torch_batch = prepare_torch_neural_batch(prepared, torch_model)
    mlx_batch = prepare_mlx_neural_batch(prepared, mlx_model)
    assert torch_batch.binned_amounts is not None
    assert mlx_batch.binned_amounts is not None
    bin_input = model.dividend_input
    assert isinstance(bin_input, BinnedDividendInput)
    torch_ordered = (
        order_torch_dividend_bins(
            torch_batch.binned_amounts,
            bin_input.order,
        )
        .detach()
        .cpu()
        .numpy()
    )
    mlx_ordered_array = order_mlx_dividend_bins(
        mlx_batch.binned_amounts,
        bin_input.order,
    )
    mx.eval(mlx_ordered_array)
    mlx_ordered = np.array(mlx_ordered_array, copy=True)
    expected = prepared.dividends.amounts[:, ::-1]
    np.testing.assert_array_equal(torch_ordered, expected)
    np.testing.assert_array_equal(mlx_ordered, expected)


@pytest.mark.parametrize("variant", list(NeuralVariant))
def test_fixed_partial_batch_mask_matches_exact_logical_row_objective(
    variant: NeuralVariant,
) -> None:
    """Padded physical slots contribute neither rows nor loss in either backend."""
    prepared = _prepared_batch(variant)
    parameters = initialize_neural_parameters(
        PROJECT_NEURAL_STUDY_SPEC,
        prepared.model,
        SEED,
    )
    torch_model = build_torch_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=prepared.model,
        parameters=parameters,
        execution=_PYTORCH_CPU_1,
    )
    mlx_model = build_mlx_project_model(
        study=PROJECT_NEURAL_STUDY_SPEC,
        model=prepared.model,
        parameters=parameters,
        execution=_MLX_CPU,
    )
    torch_source = prepare_torch_neural_batch(prepared, torch_model)
    mlx_source = prepare_mlx_neural_batch(prepared, mlx_model)
    epoch = build_neural_epoch_batch_indices(
        NeuralBatchPlan(rows=ROWS, batch_size=5, shuffle=False, seed=None),
        0,
    )
    final_indices = epoch.batch(1)
    torch_fixed = gather_torch_fixed_batch(torch_source, final_indices)
    mlx_fixed = gather_mlx_fixed_batch(mlx_source, final_indices)
    torch_prediction = torch_model(torch_fixed.values)
    mlx_prediction = mlx_model(mlx_fixed.values)
    torch_loss = float(
        torch_fixed_smape_objective(
            torch_prediction,
            torch_fixed,
            PROJECT_NEURAL_STUDY_SPEC.training,
        ).item(),
    )
    mlx_loss_array = mlx_fixed_smape_objective(
        mlx_prediction,
        mlx_fixed,
        PROJECT_NEURAL_STUDY_SPEC.training,
    )
    mx.eval(mlx_prediction, mlx_loss_array)
    mlx_loss = float(mlx_loss_array.item())

    valid_rows = final_indices.indices[final_indices.valid]
    valid_prediction = torch_prediction_numpy(torch_model, torch_fixed.values)[
        : final_indices.valid_rows
    ]
    valid_target = prepared.target[valid_rows, None]
    expected = float(
        np.mean(
            np.abs(valid_target - valid_prediction)
            / ((np.abs(valid_target) + np.abs(valid_prediction)) / 2.0),
        )
        * PROJECT_NEURAL_STUDY_SPEC.training.objective_percentage_scale,
    )
    assert torch_loss == pytest.approx(expected, abs=1e-5)
    assert mlx_loss == pytest.approx(expected, abs=1e-5)


def test_backend_build_request_never_substitutes_a_framework() -> None:
    """A mismatched backend request fails instead of constructing an available framework."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.BINS)
    parameters = initialize_neural_parameters(PROJECT_NEURAL_STUDY_SPEC, model, SEED)
    with pytest.raises(NeuralBackendContractError, match="requires a pytorch execution"):
        build_torch_project_model(
            study=PROJECT_NEURAL_STUDY_SPEC,
            model=model,
            parameters=parameters,
            execution=_MLX_CPU,
        )


def _prepared_batch(variant: NeuralVariant) -> PreparedNeuralBatch:
    model = PROJECT_NEURAL_STUDY_SPEC.model(variant)
    scalar = np.empty((ROWS, len(model.scalar_columns)), dtype=np.float32)
    base = {
        "strike": np.linspace(80.0, 120.0, ROWS, dtype=np.float32),
        "underlying_price": np.linspace(95.0, 105.0, ROWS, dtype=np.float32),
        "risk_free_rate": np.linspace(0.01, 0.04, ROWS, dtype=np.float32),
        "call_moneyness": np.linspace(0.8, 1.2, ROWS, dtype=np.float32),
        "historical_volatility_90": np.linspace(0.15, 0.35, ROWS, dtype=np.float32),
        "yte": np.linspace(0.4, 1.0, ROWS, dtype=np.float32),
        "forecasted_dividend_yield": np.linspace(0.005, 0.04, ROWS, dtype=np.float32),
        "historical_dividend_yield": np.linspace(0.01, 0.035, ROWS, dtype=np.float32),
    }
    for column, name in enumerate(model.scalar_columns):
        scalar[:, column] = base[name]
    target = np.linspace(0.5, 20.0, ROWS, dtype=np.float32)
    dividends: PaddedCashDividends | BinnedCashDividends | None
    if variant in {NeuralVariant.LSTM, NeuralVariant.EMBEDDING}:
        dividends = _padded_dividends(scalar[:, model.scalar_columns.index("yte")])
    elif variant is NeuralVariant.BINS:
        amounts = np.linspace(0.0, 0.03, ROWS * 12, dtype=np.float32).reshape(ROWS, 12)
        dividends = BinnedCashDividends(amounts=np.ascontiguousarray(amounts))
    else:
        dividends = None
    return PreparedNeuralBatch(
        model=model,
        scalar_features=np.ascontiguousarray(scalar),
        target=np.ascontiguousarray(target),
        dividends=dividends,
    )


def _padded_dividends(
    maturities: np.ndarray[tuple[int], np.dtype[np.float32]],
) -> PaddedCashDividends:
    lengths = np.array([0, 1, 2, 3, 4, 5, 1, 2], dtype=np.uint32)
    mask = np.arange(CAPACITY, dtype=np.uint32)[None, :] < lengths[:, None]
    times = np.zeros((ROWS, CAPACITY), dtype=np.float32)
    amounts = np.zeros((ROWS, CAPACITY), dtype=np.float32)
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length > 0:
            times[row, :length] = np.linspace(0.1, maturities[row], length, dtype=np.float32)
            amounts[row, :length] = np.linspace(0.005, 0.02, length, dtype=np.float32)
    return PaddedCashDividends(
        times=times,
        amounts=amounts,
        mask=np.ascontiguousarray(mask),
        lengths=lengths,
    )


def _torch_gradients(
    network: TorchProjectModel,
    batch: TorchNeuralBatch,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    network.train()
    network.zero_grad(set_to_none=True)
    prediction = network(batch)
    loss = torch_smape_objective(
        prediction,
        batch.target,
        PROJECT_NEURAL_STUDY_SPEC.training,
    )
    loss.backward()
    return {
        torch_parameter_name_to_canonical(name): parameter.grad.detach().cpu().numpy().copy()
        for name, parameter in network.named_parameters()
        if parameter.grad is not None
    }


def _mlx_gradients(
    network: MlxProjectModel,
    batch: MlxNeuralBatch,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    def loss_function(candidate: MlxProjectModel) -> mx.array:
        prediction = candidate(batch)
        return mlx_smape_objective(
            prediction,
            batch.target,
            PROJECT_NEURAL_STUDY_SPEC.training,
        )

    _, gradients = mlx_nn.value_and_grad(network, loss_function)(network)
    mx.eval(gradients)
    return {
        mlx_parameter_name_to_canonical(name): np.array(value, dtype=np.float32, copy=True)
        for name, value in tree_flatten(gradients)
    }
