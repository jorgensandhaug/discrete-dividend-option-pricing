"""Tests for the pre-result empirical publication-inference identity."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from discrete_dividend_study.project_pipeline import PROJECT_AS_SUBMITTED_TICKERS
from discrete_dividend_study.publication_inference_spec import (
    PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
    PublicationBootstrapBlockLengthSampler,
    PublicationBootstrapInterval,
    PublicationBootstrapOriginSampler,
    PublicationBootstrapQuantile,
    PublicationBootstrapRandomGenerator,
    PublicationEmpiricalMetric,
    PublicationInferenceSpecValueError,
    PublicationMultiplicityProcedure,
    publication_empirical_inference_sha256,
)
from discrete_dividend_study.publication_spec import PublicationVariant

_SHA256_HEX_LENGTH = 64
_CONTRASTS = 6
_BOOTSTRAP_REPLICATIONS = 19_999
_BOOTSTRAP_DOMAIN_START = 2_000_000
_PRIMARY_BLOCK_LENGTH = 20.0
_CONFIDENCE_LEVEL = 0.95
_FAMILY_WISE_ERROR_RATE = 0.05


def test_empirical_inference_identity_is_complete_and_frozen_before_results() -> None:
    """Cohorts, metrics, contrasts, blocks, seed, interval, and FWER are explicit."""
    spec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC
    assert len(publication_empirical_inference_sha256(spec)) == _SHA256_HEX_LENGTH
    assert spec.cohorts[0].tickers == PROJECT_AS_SUBMITTED_TICKERS
    assert spec.cohorts[0].quote_date_end_inclusive is None
    assert spec.cohorts[1].tickers == ("DUK", "EPD", "MO", "SO", "VZ")
    assert spec.cohorts[2].quote_date_end_inclusive == date(2019, 11, 7)
    assert spec.primary_metrics == (
        PublicationEmpiricalMetric.MAPE_PERCENT,
        PublicationEmpiricalMetric.RMSE,
    )
    assert len(spec.contrasts) == _CONTRASTS
    assert spec.contrasts[0].left is PublicationVariant.CASH_JUMP_LIQUIDATOR
    assert spec.contrasts[-1].right is PublicationVariant.HISTORICAL_YIELD
    assert spec.bootstrap.replications == _BOOTSTRAP_REPLICATIONS
    assert spec.bootstrap.domain_start == _BOOTSTRAP_DOMAIN_START
    assert spec.bootstrap.replication_path_start == 0
    assert spec.bootstrap.random_generator is PublicationBootstrapRandomGenerator.PHILOX_4X32_10
    assert spec.bootstrap.origin_sampler is PublicationBootstrapOriginSampler.LEMIRE_U64_REJECTION
    assert (
        spec.bootstrap.block_length_sampler
        is PublicationBootstrapBlockLengthSampler.INVERSE_CDF_OPEN_F64_52_MIDPOINT_LIBM
    )
    assert spec.bootstrap.mean_block_length == _PRIMARY_BLOCK_LENGTH
    assert spec.bootstrap.sensitivity_mean_block_lengths == (5.0, 10.0, 20.0, 40.0, 60.0)
    assert spec.bootstrap.confidence_level == _CONFIDENCE_LEVEL
    assert spec.bootstrap.interval is PublicationBootstrapInterval.BASIC
    assert spec.bootstrap.quantile is PublicationBootstrapQuantile.LINEAR_TYPE_7
    assert spec.bootstrap.finite_sample_p_value_correction
    assert spec.multiplicity is PublicationMultiplicityProcedure.HOLM
    assert spec.family_wise_error_rate == _FAMILY_WISE_ERROR_RATE


def test_inference_hash_changes_and_invalid_shortcuts_fail() -> None:
    """No data-driven block replacement, omitted contrast, or incoherent alpha is admitted."""
    spec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC
    changed = replace(
        spec,
        bootstrap=replace(spec.bootstrap, seed=spec.bootstrap.seed + 1),
    )
    assert publication_empirical_inference_sha256(changed) != (
        publication_empirical_inference_sha256(spec)
    )
    with pytest.raises(PublicationInferenceSpecValueError, match="containing mean_block_length"):
        replace(spec.bootstrap, sensitivity_mean_block_lengths=(5.0, 10.0, 40.0))
    with pytest.raises(PublicationInferenceSpecValueError, match="all six canonical"):
        replace(spec, contrasts=spec.contrasts[:-1])
    with pytest.raises(PublicationInferenceSpecValueError, match="one minus"):
        replace(spec, family_wise_error_rate=0.01)
