"""Pre-result identity for date-panel empirical publication inference."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

from .project_pipeline import PROJECT_AS_SUBMITTED_TICKERS
from .publication_spec import PublicationVariant

_HASH_FORMAT: Final = "project-publication-empirical-inference-v1"
_MAX_U32: Final = (2**32) - 1
_MAX_U64: Final = (2**64) - 1
_MIN_BOOTSTRAP_REPLICATIONS: Final = 999
_MIN_BLOCK_LENGTH: Final = 1.0
_PRIMARY_BLOCK_LENGTH: Final = 20.0
_COHORT_COUNT: Final = 3
_ETR_LAST_AVAILABLE_DATE: Final = date(2019, 11, 7)


class PublicationInferenceSpecError(ValueError):
    """The empirical-inference identity is incomplete or internally inconsistent."""


@dataclass(slots=True)
class PublicationInferenceSpecValueError(PublicationInferenceSpecError):
    """One named inference value violates its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid declaration without repairing it."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


class PublicationEmpiricalMetric(StrEnum):
    """Row-weighted project metrics retained for empirical comparison."""

    MAE = "mae"
    MAPE_PERCENT = "mape_percent"
    RMSE = "rmse"
    BIAS = "bias"
    RELATIVE_BIAS = "relative_bias"
    SMAPE_PERCENT = "smape_percent"


class PublicationBootstrapInterval(StrEnum):
    """Declared transformation of bootstrap draws into a two-sided interval."""

    BASIC = "basic_centered_two_sided"


class PublicationBootstrapRandomGenerator(StrEnum):
    """Counter-based generator used only for empirical resampling indices."""

    PHILOX_4X32_10 = "philox_4x32_10"


class PublicationBootstrapQuantile(StrEnum):
    """Exact sample-quantile interpolation convention for interval endpoints."""

    LINEAR_TYPE_7 = "linear_hyndman_fan_type_7"


class PublicationMultiplicityProcedure(StrEnum):
    """Family-wise error control for the primary pairwise tests."""

    HOLM = "holm_step_down"


@dataclass(frozen=True, slots=True)
class PublicationCohortSpec:
    """One explicit ticker and optional quote-date sensitivity population."""

    name: str
    tickers: tuple[str, ...]
    quote_date_end_inclusive: date | None

    def __post_init__(self) -> None:
        """Require a sorted nonempty ticker subset and an unambiguous optional cutoff."""
        if type(self.name) is not str or not self.name:
            raise PublicationInferenceSpecValueError(
                field="name",
                observed=self.name,
                requirement="a nonempty plain string",
            )
        if (
            type(self.tickers) is not tuple
            or not self.tickers
            or self.tickers != tuple(sorted(set(self.tickers)))
            or any(ticker not in PROJECT_AS_SUBMITTED_TICKERS for ticker in self.tickers)
        ):
            raise PublicationInferenceSpecValueError(
                field="tickers",
                observed=self.tickers,
                requirement=(
                    f"a sorted unique nonempty tuple drawn from {PROJECT_AS_SUBMITTED_TICKERS!r}"
                ),
            )
        if (
            self.quote_date_end_inclusive is not None
            and type(self.quote_date_end_inclusive) is not date
        ):
            raise PublicationInferenceSpecValueError(
                field="quote_date_end_inclusive",
                observed=self.quote_date_end_inclusive,
                requirement="None or a plain datetime.date",
            )


@dataclass(frozen=True, slots=True)
class PublicationVariantContrast:
    """One canonical left-minus-right paired model contrast."""

    left: PublicationVariant
    right: PublicationVariant

    def __post_init__(self) -> None:
        """Require distinct variants in canonical enumeration order."""
        if type(self.left) is not PublicationVariant or type(self.right) is not PublicationVariant:
            raise PublicationInferenceSpecValueError(
                field="contrast variants",
                observed=(self.left, self.right),
                requirement="PublicationVariant values",
            )
        order = {variant: index for index, variant in enumerate(PublicationVariant)}
        if order[self.left] >= order[self.right]:
            raise PublicationInferenceSpecValueError(
                field="contrast order",
                observed=(self.left, self.right),
                requirement="distinct variants in canonical left-before-right order",
            )


@dataclass(frozen=True, slots=True)
class PublicationStationaryBootstrapSpec:
    """Circular stationary bootstrap over complete ordered quote-date panels."""

    replications: int
    seed: int
    domain_start: int
    random_generator: PublicationBootstrapRandomGenerator
    mean_block_length: float
    sensitivity_mean_block_lengths: tuple[float, ...]
    confidence_level: float
    interval: PublicationBootstrapInterval
    quantile: PublicationBootstrapQuantile
    finite_sample_p_value_correction: bool

    def __post_init__(self) -> None:
        """Require fixed resources, geometric-block parameters, and interval semantics."""
        if type(self.replications) is not int or self.replications < _MIN_BOOTSTRAP_REPLICATIONS:
            raise PublicationInferenceSpecValueError(
                field="replications",
                observed=self.replications,
                requirement=f"a plain int of at least {_MIN_BOOTSTRAP_REPLICATIONS}",
            )
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_U64:
            raise PublicationInferenceSpecValueError(
                field="seed",
                observed=self.seed,
                requirement=f"a plain int in [0, {_MAX_U64}]",
            )
        if type(self.domain_start) is not int or not 0 <= self.domain_start <= _MAX_U32:
            raise PublicationInferenceSpecValueError(
                field="domain_start",
                observed=self.domain_start,
                requirement="a plain unsigned 32-bit integer",
            )
        if self.random_generator is not PublicationBootstrapRandomGenerator.PHILOX_4X32_10:
            raise PublicationInferenceSpecValueError(
                field="random_generator",
                observed=self.random_generator,
                requirement="PublicationBootstrapRandomGenerator.PHILOX_4X32_10",
            )
        _validate_block_lengths(self)
        if (
            type(self.confidence_level) is not float
            or not math.isfinite(self.confidence_level)
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise PublicationInferenceSpecValueError(
                field="confidence_level",
                observed=self.confidence_level,
                requirement="a finite plain float strictly between zero and one",
            )
        if type(self.interval) is not PublicationBootstrapInterval:
            raise PublicationInferenceSpecValueError(
                field="interval",
                observed=self.interval,
                requirement="PublicationBootstrapInterval",
            )
        if self.quantile is not PublicationBootstrapQuantile.LINEAR_TYPE_7:
            raise PublicationInferenceSpecValueError(
                field="quantile",
                observed=self.quantile,
                requirement="PublicationBootstrapQuantile.LINEAR_TYPE_7",
            )
        if type(self.finite_sample_p_value_correction) is not bool:
            raise PublicationInferenceSpecValueError(
                field="finite_sample_p_value_correction",
                observed=self.finite_sample_p_value_correction,
                requirement="a plain bool",
            )


def _validate_block_lengths(spec: PublicationStationaryBootstrapSpec) -> None:
    if (
        type(spec.mean_block_length) is not float
        or not math.isfinite(spec.mean_block_length)
        or spec.mean_block_length < _MIN_BLOCK_LENGTH
    ):
        raise PublicationInferenceSpecValueError(
            field="mean_block_length",
            observed=spec.mean_block_length,
            requirement=f"a finite plain float of at least {_MIN_BLOCK_LENGTH}",
        )
    values = spec.sensitivity_mean_block_lengths
    if (
        type(values) is not tuple
        or not values
        or values != tuple(sorted(set(values)))
        or spec.mean_block_length not in values
        or any(
            type(value) is not float or not math.isfinite(value) or value < _MIN_BLOCK_LENGTH
            for value in values
        )
    ):
        raise PublicationInferenceSpecValueError(
            field="sensitivity_mean_block_lengths",
            observed=values,
            requirement=("a sorted unique tuple of admitted floats containing mean_block_length"),
        )


@dataclass(frozen=True, slots=True)
class PublicationEmpiricalInferenceSpec:
    """Complete pre-result empirical comparison and uncertainty identity."""

    name: str
    cohorts: tuple[PublicationCohortSpec, ...]
    primary_metrics: tuple[PublicationEmpiricalMetric, ...]
    diagnostic_metrics: tuple[PublicationEmpiricalMetric, ...]
    contrasts: tuple[PublicationVariantContrast, ...]
    bootstrap: PublicationStationaryBootstrapSpec
    multiplicity: PublicationMultiplicityProcedure
    family_wise_error_rate: float

    def __post_init__(self) -> None:
        """Reject omitted populations, metrics, contrasts, or multiplicity controls."""
        if type(self.name) is not str or not self.name:
            raise PublicationInferenceSpecValueError(
                field="name",
                observed=self.name,
                requirement="a nonempty plain string",
            )
        _validate_cohorts(self.cohorts)
        _validate_metrics(self.primary_metrics, self.diagnostic_metrics)
        if self.contrasts != _all_pairwise_contrasts():
            raise PublicationInferenceSpecValueError(
                field="contrasts",
                observed=self.contrasts,
                requirement="all six canonical PublicationVariant pairs",
            )
        if type(self.bootstrap) is not PublicationStationaryBootstrapSpec:
            raise PublicationInferenceSpecValueError(
                field="bootstrap",
                observed=self.bootstrap,
                requirement="PublicationStationaryBootstrapSpec",
            )
        if self.multiplicity is not PublicationMultiplicityProcedure.HOLM:
            raise PublicationInferenceSpecValueError(
                field="multiplicity",
                observed=self.multiplicity,
                requirement="PublicationMultiplicityProcedure.HOLM",
            )
        if (
            type(self.family_wise_error_rate) is not float
            or not math.isfinite(self.family_wise_error_rate)
            or not 0.0 < self.family_wise_error_rate < 1.0
        ):
            raise PublicationInferenceSpecValueError(
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
            raise PublicationInferenceSpecValueError(
                field="family_wise_error_rate",
                observed=self.family_wise_error_rate,
                requirement="one minus the bootstrap confidence level",
            )


def _validate_cohorts(cohorts: tuple[PublicationCohortSpec, ...]) -> None:
    if type(cohorts) is not tuple or len(cohorts) != _COHORT_COUNT:
        raise PublicationInferenceSpecValueError(
            field="cohorts",
            observed=cohorts,
            requirement="the three predeclared cohort identities",
        )
    if tuple(cohort.name for cohort in cohorts) != (
        "available_case_six_ticker_full_horizon",
        "five_ticker_full_horizon_without_etr",
        "six_ticker_common_horizon_through_etr",
    ):
        raise PublicationInferenceSpecValueError(
            field="cohort names",
            observed=tuple(cohort.name for cohort in cohorts),
            requirement="the canonical primary and two sensitivity names",
        )


def _validate_metrics(
    primary: tuple[PublicationEmpiricalMetric, ...],
    diagnostic: tuple[PublicationEmpiricalMetric, ...],
) -> None:
    if primary != (
        PublicationEmpiricalMetric.MAPE_PERCENT,
        PublicationEmpiricalMetric.RMSE,
    ):
        raise PublicationInferenceSpecValueError(
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
        raise PublicationInferenceSpecValueError(
            field="diagnostic_metrics",
            observed=diagnostic,
            requirement="MAE, bias, relative bias, and SMAPE in canonical order",
        )


def _all_pairwise_contrasts() -> tuple[PublicationVariantContrast, ...]:
    return tuple(
        PublicationVariantContrast(left=left, right=right)
        for left, right in itertools.combinations(PublicationVariant, 2)
    )


PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC: Final = PublicationEmpiricalInferenceSpec(
    name="project-publication-empirical-inference-v1",
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
    bootstrap=PublicationStationaryBootstrapSpec(
        replications=19_999,
        seed=0x13198A2E03707344,
        domain_start=2_000_000,
        random_generator=PublicationBootstrapRandomGenerator.PHILOX_4X32_10,
        mean_block_length=_PRIMARY_BLOCK_LENGTH,
        sensitivity_mean_block_lengths=(5.0, 10.0, _PRIMARY_BLOCK_LENGTH, 40.0, 60.0),
        confidence_level=0.95,
        interval=PublicationBootstrapInterval.BASIC,
        quantile=PublicationBootstrapQuantile.LINEAR_TYPE_7,
        finite_sample_p_value_correction=True,
    ),
    multiplicity=PublicationMultiplicityProcedure.HOLM,
    family_wise_error_rate=0.05,
)
"""Frozen before inspecting any selected-identity publication result."""


def publication_empirical_inference_sha256(spec: PublicationEmpiricalInferenceSpec) -> str:
    """Hash the complete cohort, estimand, resampling, and multiplicity identity."""
    if type(spec) is not PublicationEmpiricalInferenceSpec:
        raise PublicationInferenceSpecValueError(
            field="spec",
            observed=spec,
            requirement="PublicationEmpiricalInferenceSpec",
        )
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
        "bootstrap": {
            "resampling_unit": "complete_ordered_observed_quote_date_panel",
            "stationary_geometric_blocks": True,
            "circular": True,
            "replications": spec.bootstrap.replications,
            "seed": spec.bootstrap.seed,
            "domain_start": spec.bootstrap.domain_start,
            "domain_mapping": "domain_start_plus_cohort_index_times_block_count_plus_block_index",
            "random_generator": spec.bootstrap.random_generator.value,
            "mean_block_length": float(spec.bootstrap.mean_block_length).hex(),
            "sensitivity_mean_block_lengths": [
                float(value).hex() for value in spec.bootstrap.sensitivity_mean_block_lengths
            ],
            "confidence_level": float(spec.bootstrap.confidence_level).hex(),
            "interval": spec.bootstrap.interval.value,
            "quantile": spec.bootstrap.quantile.value,
            "centered_two_sided_p_values": True,
            "finite_sample_p_value_correction": (spec.bootstrap.finite_sample_p_value_correction),
        },
        "multiplicity": spec.multiplicity.value,
        "multiplicity_family": "six_pairwise_contrasts_times_two_primary_metrics",
        "family_wise_error_rate": float(spec.family_wise_error_rate).hex(),
        "metric_weighting": "row_weighted_ratio_over_resampled_date_panels",
        "publication_predictions": "complete_replication_mean_cash_and_deterministic_yields",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC",
    "PublicationBootstrapInterval",
    "PublicationBootstrapQuantile",
    "PublicationBootstrapRandomGenerator",
    "PublicationCohortSpec",
    "PublicationEmpiricalInferenceSpec",
    "PublicationEmpiricalMetric",
    "PublicationInferenceSpecError",
    "PublicationInferenceSpecValueError",
    "PublicationMultiplicityProcedure",
    "PublicationStationaryBootstrapSpec",
    "PublicationVariantContrast",
    "publication_empirical_inference_sha256",
]
