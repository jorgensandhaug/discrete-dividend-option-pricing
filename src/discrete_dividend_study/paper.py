"""Complete neural experiment specified by the project paper."""

from __future__ import annotations

from typing import Final

from finoptions import (
    CashAmountScale,
    DividendBinSpec,
    MaturityBoundary,
    PaddedSequenceSpec,
    TimeBinClosure,
)

from discrete_dividend_study.models.neural.specification import (
    Activation,
    AttentionSlotPolicy,
    BinnedDividendInput,
    DirectDividendLstmInput,
    DividendBinOrder,
    DividendRepresentation,
    DividendYieldInput,
    EventAttentionInput,
    EventSequenceOrder,
    LstmGateOrder,
    MlpSpec,
    ModelPrecision,
    NeuralModelSpec,
    NeuralStudySpec,
    NeuralVariant,
    ObjectiveKind,
    OptimizerKind,
    OptimizerSpec,
    ParameterInitialization,
    PlateauSchedulerSpec,
    TrainingSpec,
)

_MARKET_FEATURES: Final = (
    "strike",
    "underlying_price",
    "risk_free_rate",
    "call_moneyness",
    "historical_volatility_90",
    "yte",
)
_TARGET: Final = "call_mid_price"
_EVENTS: Final = PaddedSequenceSpec(
    capacity=5,
    maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
    amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
)
_MONTHLY_BINS: Final = DividendBinSpec(
    bins=12,
    horizon=1.0,
    closure=TimeBinClosure.LEFT_OPEN_RIGHT_CLOSED,
    maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
    amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
)

DIRECT_LSTM: Final = NeuralModelSpec(
    variant=NeuralVariant.LSTM,
    submitted_result_name="R2_AT_NN_FD_lstm_direct_yte_1_8",
    scalar_columns=_MARKET_FEATURES,
    target_column=_TARGET,
    dividend_input=DirectDividendLstmInput(
        layout=_EVENTS,
        sequence_order=EventSequenceOrder.FARTHEST_TO_NEAREST,
        include_option_maturity=True,
        hidden_size=8,
        layers=1,
        bias=False,
        gate_order=LstmGateOrder.INPUT_FORGET_CELL_OUTPUT,
    ),
)

EVENT_ATTENTION: Final = NeuralModelSpec(
    variant=NeuralVariant.ATTENTION,
    submitted_result_name="R2_AT_NN_FD_emb8",
    scalar_columns=_MARKET_FEATURES,
    target_column=_TARGET,
    dividend_input=EventAttentionInput(
        layout=_EVENTS,
        include_option_maturity=True,
        embedding_width=8,
        embedding_layers=1,
        embedding_activation=Activation.RELU,
        slot_policy=AttentionSlotPolicy.INCLUDE_ZERO_PADDING,
    ),
)

TWELVE_BINS: Final = NeuralModelSpec(
    variant=NeuralVariant.BINS,
    submitted_result_name="R2_AT_NN_FD_12B",
    scalar_columns=_MARKET_FEATURES,
    target_column=_TARGET,
    dividend_input=BinnedDividendInput(
        layout=_MONTHLY_BINS,
        order=DividendBinOrder.FARTHEST_TO_NEAREST,
    ),
)

FORECAST_YIELD: Final = NeuralModelSpec(
    variant=NeuralVariant.FORECAST_YIELD,
    submitted_result_name="R2_AT_NN_FY",
    scalar_columns=(*_MARKET_FEATURES, "forecasted_dividend_yield"),
    target_column=_TARGET,
    dividend_input=DividendYieldInput(
        representation=DividendRepresentation.FORECAST_YIELD,
        column="forecasted_dividend_yield",
    ),
)

HISTORICAL_YIELD: Final = NeuralModelSpec(
    variant=NeuralVariant.HISTORICAL_YIELD,
    submitted_result_name="R2_AT_NN_HY",
    scalar_columns=(*_MARKET_FEATURES, "historical_dividend_yield"),
    target_column=_TARGET,
    dividend_input=DividendYieldInput(
        representation=DividendRepresentation.HISTORICAL_YIELD,
        column="historical_dividend_yield",
    ),
)

PROJECT_MODELS: Final[tuple[NeuralModelSpec, ...]] = (
    DIRECT_LSTM,
    EVENT_ATTENTION,
    TWELVE_BINS,
    FORECAST_YIELD,
    HISTORICAL_YIELD,
)
"""The five compared models in paper-table order."""

PROJECT_STUDY: Final = NeuralStudySpec(
    precision=ModelPrecision.FLOAT32,
    parameter_initialization=ParameterInitialization.UNIFORM_PCG64DXSM,
    mlp=MlpSpec(
        hidden_width=282,
        hidden_layers=4,
        hidden_activation=Activation.LEAKY_RELU,
        leaky_relu_negative_slope=3.158e-2,
        dropout_probability=0.0,
        layer_normalization=False,
        output_activation=Activation.SOFTPLUS,
    ),
    training=TrainingSpec(
        objective=ObjectiveKind.SMAPE,
        objective_percentage_scale=100.0,
        optimizer=OptimizerSpec(
            kind=OptimizerKind.ADAMW,
            learning_rate=8.6e-4,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
            bias_correction=True,
        ),
        scheduler=PlateauSchedulerSpec(
            factor=0.5,
            patience_epochs=10,
            minimum_learning_rate=5e-5,
            relative_threshold=1e-4,
            cooldown_epochs=0,
            update_epsilon=1e-8,
        ),
        batch_size=128,
        maximum_epochs=100,
        early_stopping_patience_epochs=50,
        shuffle_training=True,
        shuffle_validation=False,
        strict_validation_improvement=True,
        normalize_inputs=False,
        transform_target=False,
    ),
    models=PROJECT_MODELS,
)
"""Complete architecture, optimizer, loss, and five-model study identity."""

__all__ = [
    "DIRECT_LSTM",
    "EVENT_ATTENTION",
    "FORECAST_YIELD",
    "HISTORICAL_YIELD",
    "PROJECT_MODELS",
    "PROJECT_STUDY",
    "TWELVE_BINS",
]
