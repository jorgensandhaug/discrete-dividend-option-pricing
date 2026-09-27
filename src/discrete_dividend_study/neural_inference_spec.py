"""Pre-result identity for paired date-panel inference over the five neural models."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Final

from .neural_spec import NeuralVariant
from .project_pipeline import PROJECT_AS_SUBMITTED_TICKERS
from .publication_inference_spec import (
    PublicationBootstrapBlockLengthSampler,
    PublicationBootstrapDesign,
    PublicationBootstrapInterval,
    PublicationBootstrapOriginSampler,
    PublicationBootstrapQuantile,
    PublicationBootstrapRandomGenerator,
    PublicationCohortSpec,
    PublicationEmpiricalMetric,
    PublicationMultiplicityProcedure,
    PublicationStationaryBootstrapSpec,
)

_HASH_FORMAT: Final = "project-neural-empirical-inference-v1"
_MAX_U32: Final = (2**32) - 1
_PRIMARY_BLOCK_LENGTH: Final = 20.0
_ETR_LAST_AVAILABLE_DATE: Final = date(2019, 11, 7)


class NeuralInferenceSpecError(ValueError):
    """The neural empirical-inference identity is incomplete or inconsistent."""


@dataclass(slots=True)
class NeuralInferenceSpecValueError(NeuralInferenceSpecError):
    """One named neural-inference declaration violates its closed domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the rejected value without repairing it."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(frozen=True, slots=True)
class NeuralVariantContrast:
    """One canonical left-minus-right paired neural-model contrast."""

    left: NeuralVariant
    right: NeuralVariant

    def __post_init__(self) -> None:
        """Require distinct variants in the closed enumeration order."""
        if type(self.left) is not NeuralVariant or type(self.right) is not NeuralVariant:
            raise NeuralInferenceSpecValueError(
                field="contrast variants",
                observed=(self.left, self.right),
                requirement="NeuralVariant values",
            )
        order = {variant: index for index, variant in enumerate(NeuralVariant)}
        if order[self.left] >= order[self.right]:
            raise NeuralInferenceSpecValueError(
                field="contrast order",
                observed=(self.left, self.right),
                requirement="distinct variants in canonical left-before-right order",
            )


@dataclass(frozen=True, slots=True)
class NeuralEmpiricalInferenceSpec:
    """Complete pre-result neural comparison, uncertainty, and multiplicity identity."""

    name: str
    cohorts: tuple[PublicationCohortSpec, ...]
    primary_metrics: tuple[PublicationEmpiricalMetric, ...]
    diagnostic_metrics: tuple[PublicationEmpiricalMetric, ...]
    contrasts: tuple[NeuralVariantContrast, ...]
    primary_contrasts: tuple[NeuralVariantContrast, ...]
    bootstrap: PublicationStationaryBootstrapSpec
    multiplicity: PublicationMultiplicityProcedure
    family_wise_error_rate: float

    def __post_init__(self) -> None:
        """Reject omitted populations, models, metrics, or error-rate control."""
        if type(self.name) is not str or not self.name:
            raise NeuralInferenceSpecValueError(
                field="name",
                observed=self.name,
                requirement="a nonempty plain string",
            )
        _validate_cohorts(self.cohorts)
        _validate_metrics(self.primary_metrics, self.diagnostic_metrics)
        if self.contrasts != _all_pairwise_contrasts():
            raise NeuralInferenceSpecValueError(
                field="contrasts",
                observed=self.contrasts,
                requirement="all ten canonical NeuralVariant pairs",
            )
        if self.primary_contrasts != _discrete_vs_forecast_yield_contrasts():
            raise NeuralInferenceSpecValueError(
                field="primary_contrasts",
                observed=self.primary_contrasts,
                requirement="the three discrete-representation versus forecast-yield contrasts",
            )
        if type(self.bootstrap) is not PublicationStationaryBootstrapSpec:
            raise NeuralInferenceSpecValueError(
                field="bootstrap",
                observed=self.bootstrap,
                requirement="PublicationStationaryBootstrapSpec",
            )
        _validate_domains(self)
        if self.multiplicity is not PublicationMultiplicityProcedure.HOLM:
            raise NeuralInferenceSpecValueError(
                field="multiplicity",
                observed=self.multiplicity,
                requirement="PublicationMultiplicityProcedure.HOLM",
            )
        if (
            type(self.family_wise_error_rate) is not float
            or not math.isfinite(self.family_wise_error_rate)
            or not 0.0 < self.family_wise_error_rate < 1.0
        ):
            raise NeuralInferenceSpecValueError(
                field="family_wise_error_rate",
                observed=self.family_wise_error_rate,
                requirement="a finite plain float strictly between zero and one",
            )
        if not math.isclose(
            self.family_wise_error_rate,
            1.0 - self.bootstrap.confidence_level,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise NeuralInferenceSpecValueError(
                field="family_wise_error_rate",
                observed=self.family_wise_error_rate,
                requirement="one minus the bootstrap confidence level",
            )


def _validate_cohorts(cohorts: tuple[PublicationCohortSpec, ...]) -> None:
    names = (
        "available_case_six_ticker_full_horizon",
        "five_ticker_full_horizon_without_etr",
        "six_ticker_common_horizon_through_etr",
    )
    if (
        type(cohorts) is not tuple
        or len(cohorts) != len(names)
        or tuple(cohort.name for cohort in cohorts) != names
    ):
        raise NeuralInferenceSpecValueError(
            field="cohorts",
            observed=cohorts,
            requirement="the three canonical coverage-cohort identities",
        )


def _validate_metrics(
    primary: tuple[PublicationEmpiricalMetric, ...],
    diagnostic: tuple[PublicationEmpiricalMetric, ...],
) -> None:
    if primary != (
        PublicationEmpiricalMetric.MAPE_PERCENT,
        PublicationEmpiricalMetric.RMSE,
    ):
        raise NeuralInferenceSpecValueError(
            field="primary_metrics",
            observed=primary,
            requirement="submitted MAPE followed by RMSE",
        )
    if diagnostic != (
        PublicationEmpiricalMetric.MAE,
        PublicationEmpiricalMetric.BIAS,
        PublicationEmpiricalMetric.RELATIVE_BIAS,
        PublicationEmpiricalMetric.SMAPE_PERCENT,
    ):
        raise NeuralInferenceSpecValueError(
            field="diagnostic_metrics",
            observed=diagnostic,
            requirement="MAE, bias, relative bias, and SMAPE in canonical order",
        )


def _validate_domains(spec: NeuralEmpiricalInferenceSpec) -> None:
    blocks = len(spec.bootstrap.sensitivity_mean_block_lengths)
    ordinary_count = len(spec.cohorts) * blocks
    ordinary_end = spec.bootstrap.domain_start + ordinary_count - 1
    stratified_end = spec.bootstrap.stratified_sensitivity_domain_start + len(spec.cohorts) - 1
    if ordinary_end > _MAX_U32 or stratified_end > _MAX_U32:
        raise NeuralInferenceSpecValueError(
            field="bootstrap domain ranges",
            observed=(ordinary_end, stratified_end),
            requirement="complete nonwrapping unsigned 32-bit ranges",
        )
    ordinary = range(spec.bootstrap.domain_start, ordinary_end + 1)
    stratified = range(spec.bootstrap.stratified_sensitivity_domain_start, stratified_end + 1)
    if any(domain in ordinary for domain in stratified):
        raise NeuralInferenceSpecValueError(
            field="bootstrap domain ranges",
            observed=(
                spec.bootstrap.domain_start,
                spec.bootstrap.stratified_sensitivity_domain_start,
            ),
            requirement="disjoint ordinary and stratified ranges",
        )


def _all_pairwise_contrasts() -> tuple[NeuralVariantContrast, ...]:
    return tuple(
        NeuralVariantContrast(left=left, right=right)
        for left, right in itertools.combinations(NeuralVariant, 2)
    )


def _discrete_vs_forecast_yield_contrasts() -> tuple[NeuralVariantContrast, ...]:
    return tuple(
        NeuralVariantContrast(left=variant, right=NeuralVariant.FORECAST_YIELD)
        for variant in (
            NeuralVariant.LSTM,
            NeuralVariant.EMBEDDING,
            NeuralVariant.BINS,
        )
    )


PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC: Final = NeuralEmpiricalInferenceSpec(
    name="project-neural-empirical-inference-v1",
    cohorts=(
        PublicationCohortSpec(
            name="available_case_six_ticker_full_horizon",
            tickers=PROJECT_AS_SUBMITTED_TICKERS,
            quote_date_end_inclusive=None,
        ),
        PublicationCohortSpec(
            name="five_ticker_full_horizon_without_etr",
            tickers=tuple(ticker for ticker in PROJECT_AS_SUBMITTED_TICKERS if ticker != "ETR"),
            quote_date_end_inclusive=None,
        ),
        PublicationCohortSpec(
            name="six_ticker_common_horizon_through_etr",
            tickers=PROJECT_AS_SUBMITTED_TICKERS,
            quote_date_end_inclusive=_ETR_LAST_AVAILABLE_DATE,
        ),
    ),
    primary_metrics=(
        PublicationEmpiricalMetric.MAPE_PERCENT,
        PublicationEmpiricalMetric.RMSE,
    ),
    diagnostic_metrics=(
        PublicationEmpiricalMetric.MAE,
        PublicationEmpiricalMetric.BIAS,
        PublicationEmpiricalMetric.RELATIVE_BIAS,
        PublicationEmpiricalMetric.SMAPE_PERCENT,
    ),
    contrasts=_all_pairwise_contrasts(),
    primary_contrasts=_discrete_vs_forecast_yield_contrasts(),
    bootstrap=PublicationStationaryBootstrapSpec(
        replications=19_999,
        seed=0xA4093822299F31D0,
        domain_start=3_000_000,
        replication_path_start=0,
        random_generator=PublicationBootstrapRandomGenerator.PHILOX_4X32_10,
        origin_sampler=PublicationBootstrapOriginSampler.LEMIRE_U64_REJECTION,
        block_length_sampler=(
            PublicationBootstrapBlockLengthSampler.INVERSE_CDF_OPEN_F64_52_MIDPOINT_LIBM
        ),
        mean_block_length=_PRIMARY_BLOCK_LENGTH,
        sensitivity_mean_block_lengths=(5.0, 10.0, _PRIMARY_BLOCK_LENGTH, 40.0, 60.0),
        stratified_sensitivity_design=(
            PublicationBootstrapDesign.CALENDAR_YEAR_STRATIFIED_CIRCULAR_STATIONARY
        ),
        stratified_sensitivity_domain_start=3_100_000,
        stratified_sensitivity_mean_block_length=_PRIMARY_BLOCK_LENGTH,
        confidence_level=0.95,
        interval=PublicationBootstrapInterval.BASIC,
        quantile=PublicationBootstrapQuantile.LINEAR_TYPE_7,
        finite_sample_p_value_correction=True,
    ),
    multiplicity=PublicationMultiplicityProcedure.HOLM,
    family_wise_error_rate=0.05,
)


def neural_bootstrap_configurations(
    spec: NeuralEmpiricalInferenceSpec,
) -> tuple[tuple[PublicationBootstrapDesign, float], ...]:
    """Return ordinary block sensitivities followed by the year-stratified diagnostic."""
    if type(spec) is not NeuralEmpiricalInferenceSpec:
        raise NeuralInferenceSpecValueError(
            field="spec",
            observed=spec,
            requirement="NeuralEmpiricalInferenceSpec",
        )
    return (
        *(
            (PublicationBootstrapDesign.CIRCULAR_STATIONARY, mean)
            for mean in spec.bootstrap.sensitivity_mean_block_lengths
        ),
        (
            spec.bootstrap.stratified_sensitivity_design,
            spec.bootstrap.stratified_sensitivity_mean_block_length,
        ),
    )


def neural_empirical_inference_sha256(spec: NeuralEmpiricalInferenceSpec) -> str:
    """Hash every population, estimand, resampling, and multiplicity choice."""
    if type(spec) is not NeuralEmpiricalInferenceSpec:
        raise NeuralInferenceSpecValueError(
            field="spec",
            observed=spec,
            requirement="NeuralEmpiricalInferenceSpec",
        )
    bootstrap = spec.bootstrap
    payload = {
        "format": _HASH_FORMAT,
        "name": spec.name,
        "cohorts": [
            {
                "name": cohort.name,
                "tickers": list(cohort.tickers),
                "quote_date_end_inclusive": (
                    cohort.quote_date_end_inclusive.isoformat()
                    if cohort.quote_date_end_inclusive is not None
                    else None
                ),
            }
            for cohort in spec.cohorts
        ],
        "primary_metrics": [metric.value for metric in spec.primary_metrics],
        "diagnostic_metrics": [metric.value for metric in spec.diagnostic_metrics],
        "contrasts": [
            {"left": contrast.left.value, "right": contrast.right.value}
            for contrast in spec.contrasts
        ],
        "primary_contrasts": [
            {"left": contrast.left.value, "right": contrast.right.value}
            for contrast in spec.primary_contrasts
        ],
        "bootstrap": {
            "resampling_unit": "complete_ordered_observed_quote_date_panel",
            "panel_order": "strictly_increasing_observed_quote_date",
            "within_panel_order": "stable_verified_neural_test_source_order",
            "replications": bootstrap.replications,
            "seed": bootstrap.seed,
            "domain_start": bootstrap.domain_start,
            "replication_path_start": bootstrap.replication_path_start,
            "random_generator": bootstrap.random_generator.value,
            "origin_sampler": bootstrap.origin_sampler.value,
            "block_length_sampler": bootstrap.block_length_sampler.value,
            "mean_block_length": float(bootstrap.mean_block_length).hex(),
            "sensitivity_mean_block_lengths": [
                float(value).hex() for value in bootstrap.sensitivity_mean_block_lengths
            ],
            "stratified_sensitivity_design": bootstrap.stratified_sensitivity_design.value,
            "stratified_sensitivity_domain_start": (bootstrap.stratified_sensitivity_domain_start),
            "stratified_sensitivity_mean_block_length": (
                float(bootstrap.stratified_sensitivity_mean_block_length).hex()
            ),
            "confidence_level": float(bootstrap.confidence_level).hex(),
            "interval": bootstrap.interval.value,
            "quantile": bootstrap.quantile.value,
            "finite_sample_p_value_correction": bootstrap.finite_sample_p_value_correction,
        },
        "multiplicity": spec.multiplicity.value,
        "family_wise_error_rate": float(spec.family_wise_error_rate).hex(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC",
    "NeuralEmpiricalInferenceSpec",
    "NeuralInferenceSpecError",
    "NeuralInferenceSpecValueError",
    "NeuralVariantContrast",
    "neural_bootstrap_configurations",
    "neural_empirical_inference_sha256",
]
