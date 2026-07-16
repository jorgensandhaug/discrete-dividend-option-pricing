"""Framework-neutral identities for the project paper's neural study."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Final

from finoptions import (
    CashAmountScale,
    DividendBinSpec,
    MaturityBoundary,
    PaddedSequenceSpec,
    TimeBinClosure,
)


class NeuralSpecError(ValueError):
    """Base class for an invalid neural-study identity."""


@dataclass(slots=True)
class NeuralSpecValueError(NeuralSpecError):
    """A neural-study value is outside its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid value and required domain."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


class NeuralVariant(StrEnum):
    """The five project-study neural model identities."""

    LSTM = "nn_lstm"
    EMBEDDING = "nn_emb"
    BINS = "nn_bin"
    FORECAST_YIELD = "nn_fy"
    HISTORICAL_YIELD = "nn_hy"


class DividendRepresentation(StrEnum):
    """Closed dividend representations admitted by the project neural study."""

    PADDED_LSTM = "padded_lstm"
    FIXED_CAPACITY_EVENT_ATTENTION = "fixed_capacity_event_attention"
    FIXED_TIME_BINS = "fixed_time_bins"
    FORECAST_YIELD = "forecast_yield"
    HISTORICAL_YIELD = "historical_yield"


class Activation(StrEnum):
    """Activation functions required by the paper architectures."""

    LEAKY_RELU = "leaky_relu"
    RELU = "relu"
    SOFTPLUS = "softplus"


class OptimizerKind(StrEnum):
    """Optimizer algorithms admitted by the current study."""

    ADAMW = "adamw"


class ObjectiveKind(StrEnum):
    """Training objectives admitted by the current study."""

    SMAPE = "smape"


class ModelPrecision(StrEnum):
    """Declared arithmetic precision at the neural model boundary."""

    FLOAT32 = "float32"


class ParameterInitialization(StrEnum):
    """Backend-independent parameter initialization algorithms."""

    UNIFORM_PCG64DXSM = "uniform_pcg64dxsm"


class LstmGateOrder(StrEnum):
    """Canonical row order for recurrent gate parameter matrices."""

    INPUT_FORGET_CELL_OUTPUT = "input_forget_cell_output"


class EventSequenceOrder(StrEnum):
    """Declared order in which a recurrent encoder consumes dividend events."""

    FARTHEST_TO_NEAREST = "farthest_to_nearest"


class DividendBinOrder(StrEnum):
    """Declared order in which fixed maturity bins enter a neural trunk."""

    FARTHEST_TO_NEAREST = "farthest_to_nearest"


class AttentionSlotPolicy(StrEnum):
    """Treatment of padded fixed-capacity event-attention slots."""

    INCLUDE_ZERO_PADDING = "include_zero_padding"


class EventFeature(StrEnum):
    """Canonical feature order inside one discrete-dividend event."""

    OPTION_MATURITY_YEARS = "option_maturity_years"
    DIVIDEND_TIME_YEARS = "dividend_time_years"
    DIVIDEND_CASH_AMOUNT = "dividend_cash_amount"
    DIVIDEND_AMOUNT_OVER_SPOT = "dividend_amount_over_spot"


@dataclass(frozen=True, slots=True)
class MlpSpec:
    """Shared scalar trunk and positive price head."""

    hidden_width: int
    hidden_layers: int
    hidden_activation: Activation
    leaky_relu_negative_slope: float
    dropout_probability: float
    layer_normalization: bool
    output_activation: Activation

    def __post_init__(self) -> None:
        """Reject inferred dimensions, probabilities, and activation mismatches."""
        _require_positive_int(self.hidden_width, field="mlp.hidden_width")
        _require_positive_int(self.hidden_layers, field="mlp.hidden_layers")
        if self.hidden_activation is not Activation.LEAKY_RELU:
            raise NeuralSpecValueError(
                field="mlp.hidden_activation",
                observed=self.hidden_activation,
                requirement="Activation.LEAKY_RELU",
            )
        _require_finite_positive_float(
            self.leaky_relu_negative_slope,
            field="mlp.leaky_relu_negative_slope",
        )
        _require_probability(self.dropout_probability, field="mlp.dropout_probability")
        _require_plain_bool(self.layer_normalization, field="mlp.layer_normalization")
        if self.output_activation is not Activation.SOFTPLUS:
            raise NeuralSpecValueError(
                field="mlp.output_activation",
                observed=self.output_activation,
                requirement="Activation.SOFTPLUS",
            )


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """Shared optimizer identity."""

    kind: OptimizerKind
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    bias_correction: bool

    def __post_init__(self) -> None:
        """Reject unsupported algorithms or invalid magnitudes."""
        if self.kind is not OptimizerKind.ADAMW:
            raise NeuralSpecValueError(
                field="optimizer.kind",
                observed=self.kind,
                requirement="OptimizerKind.ADAMW",
            )
        _require_finite_positive_float(self.learning_rate, field="optimizer.learning_rate")
        _require_probability(self.beta1, field="optimizer.beta1")
        _require_probability(self.beta2, field="optimizer.beta2")
        _require_finite_positive_float(self.epsilon, field="optimizer.epsilon")
        _require_finite_nonnegative_float(self.weight_decay, field="optimizer.weight_decay")
        _require_plain_bool(self.bias_correction, field="optimizer.bias_correction")
        if not self.bias_correction:
            raise NeuralSpecValueError(
                field="optimizer.bias_correction",
                observed=self.bias_correction,
                requirement="True for the paper's AdamW contract",
            )


@dataclass(frozen=True, slots=True)
class PlateauSchedulerSpec:
    """Explicit reduce-on-validation-plateau semantics."""

    factor: float
    patience_epochs: int
    minimum_learning_rate: float
    relative_threshold: float
    cooldown_epochs: int
    update_epsilon: float

    def __post_init__(self) -> None:
        """Reject an ineffective or underspecified schedule."""
        _require_finite_positive_float(self.factor, field="scheduler.factor")
        if self.factor >= 1.0:
            raise NeuralSpecValueError(
                field="scheduler.factor",
                observed=self.factor,
                requirement="a plain float in (0, 1)",
            )
        _require_nonnegative_int(self.patience_epochs, field="scheduler.patience_epochs")
        _require_finite_nonnegative_float(
            self.minimum_learning_rate,
            field="scheduler.minimum_learning_rate",
        )
        _require_finite_nonnegative_float(
            self.relative_threshold,
            field="scheduler.relative_threshold",
        )
        _require_nonnegative_int(self.cooldown_epochs, field="scheduler.cooldown_epochs")
        _require_finite_nonnegative_float(self.update_epsilon, field="scheduler.update_epsilon")


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """Framework-neutral training, validation, and checkpoint identity."""

    objective: ObjectiveKind
    objective_percentage_scale: float
    optimizer: OptimizerSpec
    scheduler: PlateauSchedulerSpec
    batch_size: int
    maximum_epochs: int
    early_stopping_patience_epochs: int
    shuffle_training: bool
    shuffle_validation: bool
    strict_validation_improvement: bool
    normalize_inputs: bool
    transform_target: bool

    def __post_init__(self) -> None:
        """Reject ambiguous optimization and checkpoint semantics."""
        if self.objective is not ObjectiveKind.SMAPE:
            raise NeuralSpecValueError(
                field="training.objective",
                observed=self.objective,
                requirement="ObjectiveKind.SMAPE",
            )
        _require_finite_positive_float(
            self.objective_percentage_scale,
            field="training.objective_percentage_scale",
        )
        if type(self.optimizer) is not OptimizerSpec:
            raise NeuralSpecValueError(
                field="training.optimizer",
                observed=self.optimizer,
                requirement="an OptimizerSpec",
            )
        if type(self.scheduler) is not PlateauSchedulerSpec:
            raise NeuralSpecValueError(
                field="training.scheduler",
                observed=self.scheduler,
                requirement="a PlateauSchedulerSpec",
            )
        _require_positive_int(self.batch_size, field="training.batch_size")
        _require_positive_int(self.maximum_epochs, field="training.maximum_epochs")
        _require_positive_int(
            self.early_stopping_patience_epochs,
            field="training.early_stopping_patience_epochs",
        )
        for field in (
            "shuffle_training",
            "shuffle_validation",
            "strict_validation_improvement",
            "normalize_inputs",
            "transform_target",
        ):
            _require_plain_bool(getattr(self, field), field=f"training.{field}")


@dataclass(frozen=True, slots=True)
class DirectDividendLstmInput:
    """Reverse-chronological sequence consumed with exact logical lengths."""

    layout: PaddedSequenceSpec
    sequence_order: EventSequenceOrder
    include_option_maturity: bool
    hidden_size: int
    layers: int
    bias: bool
    gate_order: LstmGateOrder

    def __post_init__(self) -> None:
        """Reject a malformed recurrent dividend encoder."""
        if type(self.layout) is not PaddedSequenceSpec:
            raise NeuralSpecValueError(
                field="lstm.layout",
                observed=self.layout,
                requirement="a PaddedSequenceSpec",
            )
        if self.sequence_order is not EventSequenceOrder.FARTHEST_TO_NEAREST:
            raise NeuralSpecValueError(
                field="lstm.sequence_order",
                observed=self.sequence_order,
                requirement="EventSequenceOrder.FARTHEST_TO_NEAREST",
            )
        _require_plain_bool(self.include_option_maturity, field="lstm.include_option_maturity")
        _require_positive_int(self.hidden_size, field="lstm.hidden_size")
        _require_positive_int(self.layers, field="lstm.layers")
        _require_plain_bool(self.bias, field="lstm.bias")
        if self.gate_order is not LstmGateOrder.INPUT_FORGET_CELL_OUTPUT:
            raise NeuralSpecValueError(
                field="lstm.gate_order",
                observed=self.gate_order,
                requirement="LstmGateOrder.INPUT_FORGET_CELL_OUTPUT",
            )

    @property
    def event_features(self) -> tuple[EventFeature, ...]:
        """Return the exact recurrent event-feature order."""
        return _event_features(
            include_option_maturity=self.include_option_maturity,
            amount_scale=self.layout.amount_scale,
        )


@dataclass(frozen=True, slots=True)
class EventAttentionInput:
    """Chronological event embeddings with an explicit fixed-capacity slot policy."""

    layout: PaddedSequenceSpec
    include_option_maturity: bool
    embedding_width: int
    embedding_layers: int
    embedding_activation: Activation
    slot_policy: AttentionSlotPolicy

    def __post_init__(self) -> None:
        """Reject a malformed masked event encoder."""
        if type(self.layout) is not PaddedSequenceSpec:
            raise NeuralSpecValueError(
                field="embedding.layout",
                observed=self.layout,
                requirement="a PaddedSequenceSpec",
            )
        _require_plain_bool(
            self.include_option_maturity,
            field="embedding.include_option_maturity",
        )
        _require_positive_int(self.embedding_width, field="embedding.embedding_width")
        _require_positive_int(self.embedding_layers, field="embedding.embedding_layers")
        if self.embedding_activation is not Activation.RELU:
            raise NeuralSpecValueError(
                field="embedding.embedding_activation",
                observed=self.embedding_activation,
                requirement="Activation.RELU",
            )
        if type(self.slot_policy) is not AttentionSlotPolicy:
            raise NeuralSpecValueError(
                field="embedding.slot_policy",
                observed=self.slot_policy,
                requirement="an AttentionSlotPolicy",
            )

    @property
    def event_features(self) -> tuple[EventFeature, ...]:
        """Return the exact embedded event-feature order."""
        return _event_features(
            include_option_maturity=self.include_option_maturity,
            amount_scale=self.layout.amount_scale,
        )


@dataclass(frozen=True, slots=True)
class BinnedDividendInput:
    """Fixed-width cash-dividend bins with an explicit neural-trunk order."""

    layout: DividendBinSpec
    order: DividendBinOrder

    def __post_init__(self) -> None:
        """Reject a non-native bin identity."""
        if type(self.layout) is not DividendBinSpec:
            raise NeuralSpecValueError(
                field="bins.layout",
                observed=self.layout,
                requirement="a DividendBinSpec",
            )
        if self.order is not DividendBinOrder.FARTHEST_TO_NEAREST:
            raise NeuralSpecValueError(
                field="bins.order",
                observed=self.order,
                requirement="DividendBinOrder.FARTHEST_TO_NEAREST",
            )


@dataclass(frozen=True, slots=True)
class DividendYieldInput:
    """One named scalar dividend-yield feature already present in the scalar matrix."""

    representation: DividendRepresentation
    column: str

    def __post_init__(self) -> None:
        """Reject event representations or blank feature identities."""
        if self.representation not in {
            DividendRepresentation.FORECAST_YIELD,
            DividendRepresentation.HISTORICAL_YIELD,
        }:
            raise NeuralSpecValueError(
                field="yield.representation",
                observed=self.representation,
                requirement="a scalar dividend-yield representation",
            )
        if type(self.column) is not str or not self.column:
            raise NeuralSpecValueError(
                field="yield.column",
                observed=self.column,
                requirement="a nonempty plain str",
            )


type DividendInput = (
    DirectDividendLstmInput | EventAttentionInput | BinnedDividendInput | DividendYieldInput
)


@dataclass(frozen=True, slots=True)
class NeuralModelSpec:
    """One complete project neural-model input and encoder identity."""

    variant: NeuralVariant
    submitted_result_name: str
    scalar_columns: tuple[str, ...]
    target_column: str
    dividend_input: DividendInput

    def __post_init__(self) -> None:
        """Reject duplicate features or inconsistent variant representations."""
        if type(self.variant) is not NeuralVariant:
            raise NeuralSpecValueError(
                field="model.variant",
                observed=self.variant,
                requirement="a NeuralVariant",
            )
        for field, value in (
            ("model.submitted_result_name", self.submitted_result_name),
            ("model.target_column", self.target_column),
        ):
            if type(value) is not str or not value:
                raise NeuralSpecValueError(
                    field=field,
                    observed=value,
                    requirement="a nonempty plain str",
                )
        if (
            type(self.scalar_columns) is not tuple
            or not self.scalar_columns
            or any(type(column) is not str or not column for column in self.scalar_columns)
            or len(set(self.scalar_columns)) != len(self.scalar_columns)
        ):
            raise NeuralSpecValueError(
                field="model.scalar_columns",
                observed=self.scalar_columns,
                requirement="a nonempty unique tuple[str, ...]",
            )
        expected = _representation_for_input(self.dividend_input)
        variant_representations = {
            NeuralVariant.LSTM: frozenset({DividendRepresentation.PADDED_LSTM}),
            NeuralVariant.EMBEDDING: frozenset(
                {DividendRepresentation.FIXED_CAPACITY_EVENT_ATTENTION},
            ),
            NeuralVariant.BINS: frozenset({DividendRepresentation.FIXED_TIME_BINS}),
            NeuralVariant.FORECAST_YIELD: frozenset({DividendRepresentation.FORECAST_YIELD}),
            NeuralVariant.HISTORICAL_YIELD: frozenset(
                {DividendRepresentation.HISTORICAL_YIELD},
            ),
        }[self.variant]
        if expected not in variant_representations:
            raise NeuralSpecValueError(
                field="model.dividend_input",
                observed=expected,
                requirement=f"one of {variant_representations!r} for {self.variant.value}",
            )
        if type(self.dividend_input) is DividendYieldInput and (
            self.scalar_columns[-1] != self.dividend_input.column
        ):
            raise NeuralSpecValueError(
                field="model.scalar_columns",
                observed=self.scalar_columns,
                requirement=f"to end with yield column {self.dividend_input.column!r}",
            )

    @property
    def dividend_representation(self) -> DividendRepresentation:
        """Return the closed representation implied by the typed input specification."""
        return _representation_for_input(self.dividend_input)

    @property
    def trunk_input_width(self) -> int:
        """Return the exact scalar-trunk width after dividend encoding."""
        dividend_input = self.dividend_input
        if type(dividend_input) is DirectDividendLstmInput:
            return len(self.scalar_columns) + dividend_input.hidden_size
        if type(dividend_input) is EventAttentionInput:
            return len(self.scalar_columns) + dividend_input.embedding_width
        if type(dividend_input) is BinnedDividendInput:
            return len(self.scalar_columns) + dividend_input.layout.bins
        return len(self.scalar_columns)


@dataclass(frozen=True, slots=True)
class NeuralStudySpec:
    """Shared architecture, training contract, precision, and five model identities."""

    precision: ModelPrecision
    parameter_initialization: ParameterInitialization
    mlp: MlpSpec
    training: TrainingSpec
    models: tuple[NeuralModelSpec, ...]

    def __post_init__(self) -> None:
        """Require one and only one model for every closed project variant."""
        if self.precision is not ModelPrecision.FLOAT32:
            raise NeuralSpecValueError(
                field="study.precision",
                observed=self.precision,
                requirement="ModelPrecision.FLOAT32",
            )
        if self.parameter_initialization is not ParameterInitialization.UNIFORM_PCG64DXSM:
            raise NeuralSpecValueError(
                field="study.parameter_initialization",
                observed=self.parameter_initialization,
                requirement="ParameterInitialization.UNIFORM_PCG64DXSM",
            )
        if type(self.mlp) is not MlpSpec:
            raise NeuralSpecValueError(
                field="study.mlp",
                observed=self.mlp,
                requirement="an MlpSpec",
            )
        if type(self.training) is not TrainingSpec:
            raise NeuralSpecValueError(
                field="study.training",
                observed=self.training,
                requirement="a TrainingSpec",
            )
        if type(self.models) is not tuple or any(
            type(model) is not NeuralModelSpec for model in self.models
        ):
            raise NeuralSpecValueError(
                field="study.models",
                observed=self.models,
                requirement="a tuple[NeuralModelSpec, ...]",
            )
        variants = tuple(model.variant for model in self.models)
        if len(variants) != len(NeuralVariant) or set(variants) != set(NeuralVariant):
            raise NeuralSpecValueError(
                field="study.models",
                observed=variants,
                requirement="exactly one model for every NeuralVariant",
            )
        result_names = tuple(model.submitted_result_name for model in self.models)
        if len(set(result_names)) != len(result_names):
            raise NeuralSpecValueError(
                field="study.models.submitted_result_name",
                observed=result_names,
                requirement="unique result names",
            )

    def model(self, variant: NeuralVariant) -> NeuralModelSpec:
        """Return one model by closed variant without string coercion."""
        if type(variant) is not NeuralVariant:
            raise NeuralSpecValueError(
                field="variant",
                observed=variant,
                requirement="a NeuralVariant",
            )
        return next(model for model in self.models if model.variant is variant)


def trainable_parameter_count(study: NeuralStudySpec, model: NeuralModelSpec) -> int:
    """Return the framework-independent trainable parameter count for one declared model."""
    if type(study) is not NeuralStudySpec:
        raise NeuralSpecValueError(
            field="study",
            observed=study,
            requirement="a NeuralStudySpec",
        )
    if type(model) is not NeuralModelSpec or model not in study.models:
        raise NeuralSpecValueError(
            field="model",
            observed=model,
            requirement="a model owned by study",
        )
    width = study.mlp.hidden_width
    mlp_parameters = (
        (model.trunk_input_width * width + width)
        + (study.mlp.hidden_layers - 1) * (width * width + width)
        + width
        + 1
    )
    dividend_input = model.dividend_input
    if type(dividend_input) is DirectDividendLstmInput:
        event_width = 3 if dividend_input.include_option_maturity else 2
        recurrent_parameters = 0
        for layer in range(dividend_input.layers):
            layer_input = event_width if layer == 0 else dividend_input.hidden_size
            recurrent_parameters += (
                4 * dividend_input.hidden_size * (layer_input + dividend_input.hidden_size)
            )
            if dividend_input.bias:
                recurrent_parameters += 8 * dividend_input.hidden_size
        return mlp_parameters + recurrent_parameters
    if type(dividend_input) is EventAttentionInput:
        event_width = 3 if dividend_input.include_option_maturity else 2
        embedding_parameters = event_width * dividend_input.embedding_width
        embedding_parameters += dividend_input.embedding_width
        embedding_parameters += (dividend_input.embedding_layers - 1) * (
            dividend_input.embedding_width * dividend_input.embedding_width
            + dividend_input.embedding_width
        )
        attention_parameters = dividend_input.embedding_width + 1
        return mlp_parameters + embedding_parameters + attention_parameters
    return mlp_parameters


def neural_study_sha256(study: NeuralStudySpec) -> str:
    """Return the canonical content identity of a complete framework-neutral study."""
    if type(study) is not NeuralStudySpec:
        raise NeuralSpecValueError(
            field="study",
            observed=study,
            requirement="a NeuralStudySpec",
        )
    payload = json.dumps(
        asdict(study),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _representation_for_input(dividend_input: DividendInput) -> DividendRepresentation:
    if type(dividend_input) is DirectDividendLstmInput:
        return DividendRepresentation.PADDED_LSTM
    if type(dividend_input) is EventAttentionInput:
        return DividendRepresentation.FIXED_CAPACITY_EVENT_ATTENTION
    if type(dividend_input) is BinnedDividendInput:
        return DividendRepresentation.FIXED_TIME_BINS
    if type(dividend_input) is DividendYieldInput:
        return dividend_input.representation
    raise NeuralSpecValueError(
        field="model.dividend_input",
        observed=dividend_input,
        requirement="a supported typed dividend input",
    )


def _event_features(
    *,
    include_option_maturity: bool,
    amount_scale: CashAmountScale,
) -> tuple[EventFeature, ...]:
    amount_feature = {
        CashAmountScale.ABSOLUTE: EventFeature.DIVIDEND_CASH_AMOUNT,
        CashAmountScale.DIVIDE_BY_SPOT: EventFeature.DIVIDEND_AMOUNT_OVER_SPOT,
    }[amount_scale]
    features = (EventFeature.DIVIDEND_TIME_YEARS, amount_feature)
    if include_option_maturity:
        return (EventFeature.OPTION_MATURITY_YEARS, *features)
    return features


def _require_plain_bool(value: object, *, field: str) -> None:
    if type(value) is not bool:
        raise NeuralSpecValueError(field=field, observed=value, requirement="a plain bool")


def _require_positive_int(value: object, *, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise NeuralSpecValueError(field=field, observed=value, requirement="a positive plain int")


def _require_nonnegative_int(value: object, *, field: str) -> None:
    if type(value) is not int or value < 0:
        raise NeuralSpecValueError(
            field=field,
            observed=value,
            requirement="a nonnegative plain int",
        )


def _require_finite_positive_float(value: object, *, field: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise NeuralSpecValueError(
            field=field,
            observed=value,
            requirement="a finite positive plain float",
        )


def _require_finite_nonnegative_float(value: object, *, field: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise NeuralSpecValueError(
            field=field,
            observed=value,
            requirement="a finite nonnegative plain float",
        )


def _require_probability(value: object, *, field: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise NeuralSpecValueError(
            field=field,
            observed=value,
            requirement="a finite plain float in [0, 1)",
        )


_BASE_SCALAR_COLUMNS: Final = (
    "strike",
    "underlying_price",
    "risk_free_rate",
    "call_moneyness",
    "historical_volatility_90",
    "yte",
)
_TARGET_COLUMN: Final = "call_mid_price"
_PROJECT_SEQUENCE_LAYOUT: Final = PaddedSequenceSpec(
    capacity=5,
    maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
    amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
)
_PROJECT_BIN_LAYOUT: Final = DividendBinSpec(
    bins=12,
    horizon=1.0,
    closure=TimeBinClosure.LEFT_OPEN_RIGHT_CLOSED,
    maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
    amount_scale=CashAmountScale.DIVIDE_BY_SPOT,
)

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
    models=(
        NeuralModelSpec(
            variant=NeuralVariant.LSTM,
            submitted_result_name="R2_AT_NN_FD_lstm_direct_yte_1_8",
            scalar_columns=_BASE_SCALAR_COLUMNS,
            target_column=_TARGET_COLUMN,
            dividend_input=DirectDividendLstmInput(
                layout=_PROJECT_SEQUENCE_LAYOUT,
                sequence_order=EventSequenceOrder.FARTHEST_TO_NEAREST,
                include_option_maturity=True,
                hidden_size=8,
                layers=1,
                bias=False,
                gate_order=LstmGateOrder.INPUT_FORGET_CELL_OUTPUT,
            ),
        ),
        NeuralModelSpec(
            variant=NeuralVariant.EMBEDDING,
            submitted_result_name="R2_AT_NN_FD_emb8",
            scalar_columns=_BASE_SCALAR_COLUMNS,
            target_column=_TARGET_COLUMN,
            dividend_input=EventAttentionInput(
                layout=_PROJECT_SEQUENCE_LAYOUT,
                include_option_maturity=True,
                embedding_width=8,
                embedding_layers=1,
                embedding_activation=Activation.RELU,
                slot_policy=AttentionSlotPolicy.INCLUDE_ZERO_PADDING,
            ),
        ),
        NeuralModelSpec(
            variant=NeuralVariant.BINS,
            submitted_result_name="R2_AT_NN_FD_12B",
            scalar_columns=_BASE_SCALAR_COLUMNS,
            target_column=_TARGET_COLUMN,
            dividend_input=BinnedDividendInput(
                layout=_PROJECT_BIN_LAYOUT,
                order=DividendBinOrder.FARTHEST_TO_NEAREST,
            ),
        ),
        NeuralModelSpec(
            variant=NeuralVariant.FORECAST_YIELD,
            submitted_result_name="R2_AT_NN_FY",
            scalar_columns=(*_BASE_SCALAR_COLUMNS, "forecasted_dividend_yield"),
            target_column=_TARGET_COLUMN,
            dividend_input=DividendYieldInput(
                representation=DividendRepresentation.FORECAST_YIELD,
                column="forecasted_dividend_yield",
            ),
        ),
        NeuralModelSpec(
            variant=NeuralVariant.HISTORICAL_YIELD,
            submitted_result_name="R2_AT_NN_HY",
            scalar_columns=(*_BASE_SCALAR_COLUMNS, "historical_dividend_yield"),
            target_column=_TARGET_COLUMN,
            dividend_input=DividendYieldInput(
                representation=DividendRepresentation.HISTORICAL_YIELD,
                column="historical_dividend_yield",
            ),
        ),
    ),
)

__all__ = [
    "PROJECT_STUDY",
    "Activation",
    "AttentionSlotPolicy",
    "BinnedDividendInput",
    "DirectDividendLstmInput",
    "DividendRepresentation",
    "DividendYieldInput",
    "EventAttentionInput",
    "EventFeature",
    "LstmGateOrder",
    "MlpSpec",
    "ModelPrecision",
    "NeuralModelSpec",
    "NeuralSpecError",
    "NeuralSpecValueError",
    "NeuralStudySpec",
    "NeuralVariant",
    "ObjectiveKind",
    "OptimizerKind",
    "OptimizerSpec",
    "ParameterInitialization",
    "PlateauSchedulerSpec",
    "TrainingSpec",
    "neural_study_sha256",
    "trainable_parameter_count",
]
