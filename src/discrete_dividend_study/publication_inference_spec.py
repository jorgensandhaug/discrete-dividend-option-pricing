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

_HASH_FORMAT: Final = "project-publication-empirical-inference-v2"
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


class PublicationBootstrapOriginSampler(StrEnum):
    """Exact discrete-uniform mapping used for circular block origins."""

    LEMIRE_U64_REJECTION = "lemire_u64_multiply_high_with_rejection"


class PublicationBootstrapBlockLengthSampler(StrEnum):
    """Exact digital inverse-CDF convention used for geometric block lengths."""

    INVERSE_CDF_OPEN_F64_52_MIDPOINT_LIBM = "inverse_cdf_open_f64_52_midpoint_libm_log1p"


class PublicationBootstrapQuantile(StrEnum):
    """Exact sample-quantile interpolation convention for interval endpoints."""

    LINEAR_TYPE_7 = "linear_hyndman_fan_type_7"


class PublicationBootstrapDesign(StrEnum):
    """Declared treatment of fixed temporal strata in one panel bootstrap."""

    CIRCULAR_STATIONARY = "circular_stationary"
    CALENDAR_YEAR_STRATIFIED_CIRCULAR_STATIONARY = "calendar_year_stratified_circular_stationary"


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
    replication_path_start: int
    random_generator: PublicationBootstrapRandomGenerator
    origin_sampler: PublicationBootstrapOriginSampler
    block_length_sampler: PublicationBootstrapBlockLengthSampler
    mean_block_length: float
    sensitivity_mean_block_lengths: tuple[float, ...]
    stratified_sensitivity_design: PublicationBootstrapDesign
    stratified_sensitivity_domain_start: int
    stratified_sensitivity_mean_block_length: float
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
        _validate_bootstrap_random_mapping(self)
        _validate_block_lengths(self)
        if (
            self.stratified_sensitivity_design
            is not PublicationBootstrapDesign.CALENDAR_YEAR_STRATIFIED_CIRCULAR_STATIONARY
        ):
            raise PublicationInferenceSpecValueError(
                field="stratified_sensitivity_design",
                observed=self.stratified_sensitivity_design,
                requirement=(
                    "PublicationBootstrapDesign.CALENDAR_YEAR_STRATIFIED_CIRCULAR_STATIONARY"
                ),
            )
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
        if self.finite_sample_p_value_correction is not True:
            raise PublicationInferenceSpecValueError(
                field="finite_sample_p_value_correction",
                observed=self.finite_sample_p_value_correction,
                requirement="True for the implemented finite-sample-corrected test",
            )


def _validate_bootstrap_random_mapping(spec: PublicationStationaryBootstrapSpec) -> None:
    """Reject wrapping namespaces or an implementation-selected random mapping."""
    if type(spec.domain_start) is not int or not 0 <= spec.domain_start <= _MAX_U32:
        raise PublicationInferenceSpecValueError(
            field="domain_start",
            observed=spec.domain_start,
            requirement="a plain unsigned 32-bit integer",
        )
    if (
        type(spec.stratified_sensitivity_domain_start) is not int
        or not 0 <= spec.stratified_sensitivity_domain_start <= _MAX_U32
    ):
        raise PublicationInferenceSpecValueError(
            field="stratified_sensitivity_domain_start",
            observed=spec.stratified_sensitivity_domain_start,
            requirement="a plain unsigned 32-bit integer",
        )
    if (
        type(spec.replication_path_start) is not int
        or not 0 <= spec.replication_path_start <= _MAX_U32
        or spec.replication_path_start + spec.replications - 1 > _MAX_U32
    ):
        raise PublicationInferenceSpecValueError(
            field="replication_path_start",
            observed=spec.replication_path_start,
            requirement="a plain unsigned 32-bit start whose complete replication range fits",
        )
    if spec.random_generator is not PublicationBootstrapRandomGenerator.PHILOX_4X32_10:
        raise PublicationInferenceSpecValueError(
            field="random_generator",
            observed=spec.random_generator,
            requirement="PublicationBootstrapRandomGenerator.PHILOX_4X32_10",
        )
    if spec.origin_sampler is not PublicationBootstrapOriginSampler.LEMIRE_U64_REJECTION:
        raise PublicationInferenceSpecValueError(
            field="origin_sampler",
            observed=spec.origin_sampler,
            requirement="PublicationBootstrapOriginSampler.LEMIRE_U64_REJECTION",
        )
    if (
        spec.block_length_sampler
        is not PublicationBootstrapBlockLengthSampler.INVERSE_CDF_OPEN_F64_52_MIDPOINT_LIBM
    ):
        raise PublicationInferenceSpecValueError(
            field="block_length_sampler",
            observed=spec.block_length_sampler,
            requirement=(
                "PublicationBootstrapBlockLengthSampler.INVERSE_CDF_OPEN_F64_52_MIDPOINT_LIBM"
            ),
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
    if spec.stratified_sensitivity_mean_block_length != spec.mean_block_length:
        raise PublicationInferenceSpecValueError(
            field="stratified_sensitivity_mean_block_length",
            observed=spec.stratified_sensitivity_mean_block_length,
            requirement="the primary mean_block_length",
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
        domain_count = len(self.cohorts) * len(self.bootstrap.sensitivity_mean_block_lengths)
        if self.bootstrap.domain_start + domain_count - 1 > _MAX_U32:
            raise PublicationInferenceSpecValueError(
                field="bootstrap domain range",
                observed=(self.bootstrap.domain_start, domain_count),
                requirement="a complete nonwrapping unsigned 32-bit cohort/block domain range",
            )
        stratified_end = self.bootstrap.stratified_sensitivity_domain_start + len(self.cohorts) - 1
        if stratified_end > _MAX_U32:
            raise PublicationInferenceSpecValueError(
                field="stratified bootstrap domain range",
                observed=(self.bootstrap.stratified_sensitivity_domain_start, len(self.cohorts)),
                requirement="a complete nonwrapping unsigned 32-bit cohort domain range",
            )
        ordinary_range = range(
            self.bootstrap.domain_start,
            self.bootstrap.domain_start + domain_count,
        )
        if any(
            domain in ordinary_range
            for domain in range(
                self.bootstrap.stratified_sensitivity_domain_start,
                stratified_end + 1,
            )
        ):
            raise PublicationInferenceSpecValueError(
                field="bootstrap domain ranges",
                observed=(
                    self.bootstrap.domain_start,
                    self.bootstrap.stratified_sensitivity_domain_start,
                ),
                requirement="disjoint ordinary and stratified unsigned 32-bit ranges",
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
    name="project-publication-empirical-inference-v2",
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
        stratified_sensitivity_domain_start=2_100_000,
        stratified_sensitivity_mean_block_length=_PRIMARY_BLOCK_LENGTH,
        confidence_level=0.95,
        interval=PublicationBootstrapInterval.BASIC,
        quantile=PublicationBootstrapQuantile.LINEAR_TYPE_7,
        finite_sample_p_value_correction=True,
    ),
    multiplicity=PublicationMultiplicityProcedure.HOLM,
    family_wise_error_rate=0.05,
)
"""Frozen before inspecting any selected-identity publication result."""


def publication_bootstrap_configurations(
    spec: PublicationEmpiricalInferenceSpec,
) -> tuple[tuple[PublicationBootstrapDesign, float], ...]:
    """Return ordinary block sensitivities followed by one calendar-stratified sensitivity."""
    if type(spec) is not PublicationEmpiricalInferenceSpec:
        raise PublicationInferenceSpecValueError(
            field="spec",
            observed=spec,
            requirement="PublicationEmpiricalInferenceSpec",
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
            "panel_order": "strictly_increasing_observed_quote_date",
            "within_panel_row_order": "stable_publication_source_row_order",
            "panel_additive_reduction": "ordered_float64_numpy_add_reduceat_under_locked_runtime",
            "stationary_geometric_blocks": True,
            "circular": True,
            "replications": spec.bootstrap.replications,
            "seed": spec.bootstrap.seed,
            "domain_start": spec.bootstrap.domain_start,
            "domain_mapping": "domain_start_plus_cohort_index_times_block_count_plus_block_index",
            "replication_path_start": spec.bootstrap.replication_path_start,
            "replication_path_mapping": "replication_path_start_plus_zero_based_replication_index",
            "random_generator": spec.bootstrap.random_generator.value,
            "philox_counter_lanes": "block_low_u32_block_high_u32_path_u32_domain_u32",
            "philox_word_order": "four_u32_words_in_random123_output_order",
            "philox_u64_packing": "first_u32_low_then_second_u32_high",
            "origin_sampler": spec.bootstrap.origin_sampler.value,
            "block_length_sampler": spec.bootstrap.block_length_sampler.value,
            "block_draw_order": "origin_u64_then_block_length_u64",
            "origin_rejection_stream_rule": "consume_next_u64_until_lemire_accepts",
            "block_uniform_open_mapping": (
                "u_equals_top_52_bits_of_u64_plus_one_half_divided_by_two_pow_52"
            ),
            "block_inverse_cdf": (
                "one_plus_floor_libm_log1p_minus_u_divided_by_"
                "libm_log1p_minus_reciprocal_mean_length"
            ),
            "block_length_parameterization": "restart_probability_equals_reciprocal_mean_length",
            "unit_mean_block_stream_rule": "block_length_is_one_without_consuming_a_u64",
            "block_length_terminal_rule": "truncate_final_block_to_exactly_panel_count_draws",
            "mean_block_length": float(spec.bootstrap.mean_block_length).hex(),
            "sensitivity_mean_block_lengths": [
                float(value).hex() for value in spec.bootstrap.sensitivity_mean_block_lengths
            ],
            "stratified_sensitivity_design": (spec.bootstrap.stratified_sensitivity_design.value),
            "stratified_sensitivity_strata": "observed_iso_calendar_year",
            "stratified_sensitivity_offsets": (
                "strictly_increasing_complete_contiguous_panel_partition"
            ),
            "stratified_sensitivity_domain_start": (
                spec.bootstrap.stratified_sensitivity_domain_start
            ),
            "stratified_sensitivity_domain_mapping": ("stratified_domain_start_plus_cohort_index"),
            "stratified_sensitivity_mean_block_length": float(
                spec.bootstrap.stratified_sensitivity_mean_block_length
            ).hex(),
            "stratified_sensitivity_weighting": "preserve_each_calendar_year_panel_count",
            "stratified_sensitivity_stratum_order": "strictly_increasing_iso_calendar_year",
            "stratified_sensitivity_random_draw_order": (
                "one_replication_path_then_strata_in_declared_order"
            ),
            "confidence_level": float(spec.bootstrap.confidence_level).hex(),
            "interval": spec.bootstrap.interval.value,
            "quantile": spec.bootstrap.quantile.value,
            "centered_two_sided_p_values": True,
            "finite_sample_p_value_correction": (spec.bootstrap.finite_sample_p_value_correction),
        },
        "multiplicity": spec.multiplicity.value,
        "multiplicity_family": "six_pairwise_contrasts_times_two_primary_metrics",
        "multiplicity_tie_order": "canonical_contrast_order_then_primary_metric_order",
        "family_wise_error_rate": float(spec.family_wise_error_rate).hex(),
        "metric_weighting": "row_weighted_ratio_over_resampled_date_panels",
        "point_metric_reduction": "source_order_project_float64_chunked_reduction",
        "contrast_direction": "left_metric_minus_right_metric",
        "publication_predictions": "complete_replication_mean_cash_and_deterministic_yields",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC",
    "PublicationBootstrapBlockLengthSampler",
    "PublicationBootstrapDesign",
    "PublicationBootstrapInterval",
    "PublicationBootstrapOriginSampler",
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
    "publication_bootstrap_configurations",
    "publication_empirical_inference_sha256",
]
