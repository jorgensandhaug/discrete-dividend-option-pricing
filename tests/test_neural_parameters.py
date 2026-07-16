"""Tests for backend-independent canonical neural initialization."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from discrete_dividend_study.neural_parameters import initialize_neural_parameters
from discrete_dividend_study.neural_spec import (
    PROJECT_STUDY,
    NeuralSpecValueError,
    NeuralVariant,
    ParameterInitialization,
    trainable_parameter_count,
)

SEED = 20261207
SHA256_HEX_LENGTH = 64


def test_study_declares_backend_independent_initialization() -> None:
    """Framework defaults cannot change the initial optimization problem."""
    assert PROJECT_STUDY.parameter_initialization is ParameterInitialization.UNIFORM_PCG64DXSM


@pytest.mark.parametrize("variant", list(NeuralVariant))
def test_parameter_sets_are_complete_read_only_and_deterministic(variant: NeuralVariant) -> None:
    """Every variant resolves to an immutable parameter set with a stable content identity."""
    model = PROJECT_STUDY.model(variant)
    first = initialize_neural_parameters(PROJECT_STUDY, model, SEED)
    second = initialize_neural_parameters(PROJECT_STUDY, model, SEED)
    assert first.sha256 == second.sha256
    assert len(first.sha256) == SHA256_HEX_LENGTH
    assert sum(
        parameter.values.size for parameter in first.parameters
    ) == trainable_parameter_count(
        PROJECT_STUDY,
        model,
    )
    assert tuple(parameter.name for parameter in first.parameters) == tuple(
        parameter.name for parameter in second.parameters
    )
    for left, right in zip(first.parameters, second.parameters, strict=True):
        assert left.values.dtype == np.dtype(np.float32)
        assert left.values.flags.c_contiguous
        assert not left.values.flags.writeable
        np.testing.assert_array_equal(left.values, right.values)


def test_encoder_parameter_names_and_shapes_are_unambiguous() -> None:
    """LSTM gates and masked-attention layers have backend-neutral names and layouts."""
    lstm = initialize_neural_parameters(
        PROJECT_STUDY,
        PROJECT_STUDY.model(NeuralVariant.LSTM),
        SEED,
    )
    assert lstm.get("lstm.0.weight_input").shape == (32, 3)
    assert lstm.get("lstm.0.weight_hidden").shape == (32, 8)
    with pytest.raises(KeyError):
        lstm.get("lstm.0.bias_input")

    embedding = initialize_neural_parameters(
        PROJECT_STUDY,
        PROJECT_STUDY.model(NeuralVariant.EMBEDDING),
        SEED,
    )
    assert embedding.get("embedding.0.weight").shape == (8, 3)
    assert embedding.get("attention.weight").shape == (1, 8)


def test_seed_changes_content_and_invalid_seeds_fail() -> None:
    """The seed is exact unsigned run identity, never a boolean or wrapped integer."""
    model = PROJECT_STUDY.model(NeuralVariant.BINS)
    first = initialize_neural_parameters(PROJECT_STUDY, model, SEED)
    second = initialize_neural_parameters(PROJECT_STUDY, model, SEED + 1)
    assert first.sha256 != second.sha256
    invalid_seed_value: object = True
    invalid_seed = cast("int", invalid_seed_value)
    with pytest.raises(NeuralSpecValueError, match="plain int"):
        initialize_neural_parameters(PROJECT_STUDY, model, invalid_seed)
