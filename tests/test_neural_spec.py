"""Tests for the immutable project neural-study identity."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from finoptions import CashAmountScale, MaturityBoundary, TimeBinClosure

from discrete_dividend_study.neural_spec import (
    PROJECT_AS_SUBMITTED_NEURAL_STUDY_SPEC,
    PROJECT_NEURAL_STUDY_SPEC,
    AttentionSlotPolicy,
    BinnedDividendInput,
    DirectDividendLstmInput,
    DividendBinOrder,
    DividendRepresentation,
    EventAttentionInput,
    EventFeature,
    EventSequenceOrder,
    LstmGateOrder,
    NeuralSpecValueError,
    NeuralVariant,
    neural_study_sha256,
    trainable_parameter_count,
)

BASE_COLUMNS = (
    "strike",
    "underlying_price",
    "risk_free_rate",
    "call_moneyness",
    "historical_volatility_90",
    "yte",
)
EXPECTED_SHARED_CONFIGURATION = (
    282,
    4,
    3.158e-2,
    0.0,
    8.6e-4,
    0.9,
    0.999,
    1e-8,
    0.01,
    0.5,
    10,
    5e-5,
    128,
    100,
    50,
)
EXPECTED_SEQUENCE_CAPACITY = 5
EXPECTED_BIN_COUNT = 12
SHA256_HEX_LENGTH = 64


def test_project_study_freezes_shared_architecture_and_training() -> None:
    """Recovered shared choices are values, not backend defaults."""
    study = PROJECT_NEURAL_STUDY_SPEC
    observed = (
        study.mlp.hidden_width,
        study.mlp.hidden_layers,
        study.mlp.leaky_relu_negative_slope,
        study.mlp.dropout_probability,
        study.training.optimizer.learning_rate,
        study.training.optimizer.beta1,
        study.training.optimizer.beta2,
        study.training.optimizer.epsilon,
        study.training.optimizer.weight_decay,
        study.training.scheduler.factor,
        study.training.scheduler.patience_epochs,
        study.training.scheduler.minimum_learning_rate,
        study.training.batch_size,
        study.training.maximum_epochs,
        study.training.early_stopping_patience_epochs,
    )
    assert observed == EXPECTED_SHARED_CONFIGURATION
    assert not study.mlp.layer_normalization

    training = study.training
    assert training.shuffle_training
    assert not training.shuffle_validation
    assert training.strict_validation_improvement
    assert not training.normalize_inputs
    assert not training.transform_target


def test_five_model_identities_have_exact_feature_order_and_representations() -> None:
    """Each result name maps to one and only one declared dividend input."""
    study = PROJECT_NEURAL_STUDY_SPEC
    expected = {
        NeuralVariant.LSTM: (
            "R2_AT_NN_FD_lstm_direct_yte_1_8",
            BASE_COLUMNS,
            DividendRepresentation.PADDED_LSTM,
        ),
        NeuralVariant.EMBEDDING: (
            "R2_AT_NN_FD_emb8",
            BASE_COLUMNS,
            DividendRepresentation.MASKED_EVENT_ATTENTION,
        ),
        NeuralVariant.BINS: (
            "R2_AT_NN_FD_12B",
            BASE_COLUMNS,
            DividendRepresentation.FIXED_TIME_BINS,
        ),
        NeuralVariant.FORECAST_YIELD: (
            "R2_AT_NN_FY",
            (*BASE_COLUMNS, "forecasted_dividend_yield"),
            DividendRepresentation.FORECAST_YIELD,
        ),
        NeuralVariant.HISTORICAL_YIELD: (
            "R2_AT_NN_HY",
            (*BASE_COLUMNS, "historical_dividend_yield"),
            DividendRepresentation.HISTORICAL_YIELD,
        ),
    }
    for variant, (result_name, columns, representation) in expected.items():
        model = study.model(variant)
        assert model.submitted_result_name == result_name
        assert model.scalar_columns == columns
        assert model.target_column == "call_mid_price"
        assert model.dividend_representation is representation


def test_event_layouts_are_canonical_explicit_and_length_or_mask_aware() -> None:
    """Event order, units, capacity, boundary, and bin closure are immutable."""
    lstm = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.LSTM).dividend_input
    embedding = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.EMBEDDING).dividend_input
    assert type(lstm) is DirectDividendLstmInput
    assert type(embedding) is EventAttentionInput
    for event_input in (lstm, embedding):
        assert event_input.layout.capacity == EXPECTED_SEQUENCE_CAPACITY
        assert event_input.layout.maturity_boundary is MaturityBoundary.AT_OR_BEFORE
        assert event_input.layout.amount_scale is CashAmountScale.DIVIDE_BY_SPOT
        assert event_input.include_option_maturity
        assert event_input.event_features == (
            EventFeature.OPTION_MATURITY_YEARS,
            EventFeature.DIVIDEND_TIME_YEARS,
            EventFeature.DIVIDEND_AMOUNT_OVER_SPOT,
        )
    assert lstm.gate_order is LstmGateOrder.INPUT_FORGET_CELL_OUTPUT
    assert lstm.sequence_order is EventSequenceOrder.FARTHEST_TO_NEAREST
    assert embedding.slot_policy is AttentionSlotPolicy.MASK_INVALID_EVENTS

    bin_input = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.BINS).dividend_input
    assert type(bin_input) is BinnedDividendInput
    bins = bin_input.layout
    assert bins.bins == EXPECTED_BIN_COUNT
    assert bins.horizon == 1.0
    assert bins.closure is TimeBinClosure.LEFT_OPEN_RIGHT_CLOSED
    assert bins.maturity_boundary is MaturityBoundary.AT_OR_BEFORE
    assert bins.amount_scale is CashAmountScale.DIVIDE_BY_SPOT
    assert bin_input.order is DividendBinOrder.FARTHEST_TO_NEAREST


def test_submitted_embedding_slot_policy_is_explicitly_distinct_from_corrected() -> None:
    """The recovered fixed-capacity softmax cannot be mistaken for masked attention."""
    corrected = PROJECT_NEURAL_STUDY_SPEC
    submitted = PROJECT_AS_SUBMITTED_NEURAL_STUDY_SPEC
    corrected_input = corrected.model(NeuralVariant.EMBEDDING).dividend_input
    submitted_input = submitted.model(NeuralVariant.EMBEDDING).dividend_input
    assert type(corrected_input) is EventAttentionInput
    assert type(submitted_input) is EventAttentionInput
    assert corrected_input.slot_policy is AttentionSlotPolicy.MASK_INVALID_EVENTS
    assert submitted_input.slot_policy is AttentionSlotPolicy.INCLUDE_ZERO_PADDING
    assert (
        submitted.model(NeuralVariant.EMBEDDING).dividend_representation
        is DividendRepresentation.FIXED_CAPACITY_EVENT_ATTENTION
    )
    assert neural_study_sha256(submitted) != neural_study_sha256(corrected)
    for variant in NeuralVariant:
        if variant is not NeuralVariant.EMBEDDING:
            assert submitted.model(variant) == corrected.model(variant)


def test_framework_independent_parameter_counts_are_frozen() -> None:
    """Backend implementations can be checked before any training run starts."""
    expected = {
        NeuralVariant.LSTM: 244_283,
        NeuralVariant.EMBEDDING: 243_972,
        NeuralVariant.BINS: 245_059,
        NeuralVariant.FORECAST_YIELD: 241_957,
        NeuralVariant.HISTORICAL_YIELD: 241_957,
    }
    for variant, count in expected.items():
        model = PROJECT_NEURAL_STUDY_SPEC.model(variant)
        assert trainable_parameter_count(PROJECT_NEURAL_STUDY_SPEC, model) == count


def test_study_content_identity_is_canonical_and_sensitive() -> None:
    """A study hash is stable yet changes with any result-capable declared value."""
    first = neural_study_sha256(PROJECT_NEURAL_STUDY_SPEC)
    second = neural_study_sha256(PROJECT_NEURAL_STUDY_SPEC)
    changed = replace(
        PROJECT_NEURAL_STUDY_SPEC,
        training=replace(PROJECT_NEURAL_STUDY_SPEC.training, maximum_epochs=101),
    )
    assert first == second
    assert len(first) == SHA256_HEX_LENGTH
    assert neural_study_sha256(changed) != first


def test_closed_variant_lookup_rejects_strings() -> None:
    """No string or boolean coercion can silently select a study model."""
    variant = cast("NeuralVariant", "nn_lstm")
    with pytest.raises(NeuralSpecValueError, match="NeuralVariant"):
        PROJECT_NEURAL_STUDY_SPEC.model(variant)
