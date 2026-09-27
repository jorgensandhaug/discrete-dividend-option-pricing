"""Tests for the pre-result five-model neural empirical-inference identity."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from discrete_dividend_study.neural_inference_spec import (
    PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC,
    NeuralInferenceSpecValueError,
    NeuralVariantContrast,
    neural_bootstrap_configurations,
    neural_empirical_inference_sha256,
)
from discrete_dividend_study.neural_spec import NeuralVariant
from discrete_dividend_study.publication_inference_spec import (
    PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
    PublicationBootstrapDesign,
)

_PAIRWISE_CONTRASTS = len(NeuralVariant) * (len(NeuralVariant) - 1) // 2
_PRIMARY_CONTRASTS = 3
_PRIMARY_METRICS = 2
_SHA256_HEX_LENGTH = 64


def test_primary_family_is_only_discrete_representations_against_forecast_yield() -> None:
    """The six Holm tests answer the paper question without every-pair multiplicity leakage."""
    spec = PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC
    assert spec.primary_contrasts == (
        NeuralVariantContrast(NeuralVariant.LSTM, NeuralVariant.FORECAST_YIELD),
        NeuralVariantContrast(NeuralVariant.EMBEDDING, NeuralVariant.FORECAST_YIELD),
        NeuralVariantContrast(NeuralVariant.BINS, NeuralVariant.FORECAST_YIELD),
    )
    assert len(spec.contrasts) == _PAIRWISE_CONTRASTS
    assert len(spec.primary_contrasts) == _PRIMARY_CONTRASTS
    assert len(spec.primary_metrics) == _PRIMARY_METRICS
    assert spec.bootstrap.domain_start != (
        PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC.bootstrap.domain_start
    )
    assert spec.bootstrap.seed != PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC.bootstrap.seed


def test_neural_bootstrap_configurations_are_complete_and_canonical() -> None:
    """Five ordinary block lengths precede one calendar-year composition diagnostic."""
    configurations = neural_bootstrap_configurations(PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC)
    assert configurations == (
        (PublicationBootstrapDesign.CIRCULAR_STATIONARY, 5.0),
        (PublicationBootstrapDesign.CIRCULAR_STATIONARY, 10.0),
        (PublicationBootstrapDesign.CIRCULAR_STATIONARY, 20.0),
        (PublicationBootstrapDesign.CIRCULAR_STATIONARY, 40.0),
        (PublicationBootstrapDesign.CIRCULAR_STATIONARY, 60.0),
        (
            PublicationBootstrapDesign.CALENDAR_YEAR_STRATIFIED_CIRCULAR_STATIONARY,
            20.0,
        ),
    )


def test_inference_hash_binds_every_contrast_and_resampling_choice() -> None:
    """An otherwise plausible seed or primary-family change has a different identity."""
    spec = PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC
    digest = neural_empirical_inference_sha256(spec)
    assert len(digest) == _SHA256_HEX_LENGTH
    assert digest == neural_empirical_inference_sha256(spec)
    changed_bootstrap = replace(spec.bootstrap, seed=spec.bootstrap.seed + 1)
    changed = replace(spec, bootstrap=changed_bootstrap)
    assert neural_empirical_inference_sha256(changed) != digest
    with pytest.raises(NeuralInferenceSpecValueError, match="primary_contrasts"):
        replace(spec, primary_contrasts=spec.primary_contrasts[:2])


def test_contrasts_reject_reverse_order_and_string_aliases() -> None:
    """Contrast signs cannot change through coercion or unordered construction."""
    with pytest.raises(NeuralInferenceSpecValueError, match="contrast order"):
        NeuralVariantContrast(NeuralVariant.FORECAST_YIELD, NeuralVariant.LSTM)
    with pytest.raises(NeuralInferenceSpecValueError, match="NeuralVariant values"):
        NeuralVariantContrast(cast("NeuralVariant", "nn_lstm"), NeuralVariant.FORECAST_YIELD)
