"""Paired date-panel empirical inference for the five project neural models."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from finoptions import CpuExecutor, RandomStream, StationaryBootstrap

from .neural_inference_spec import (
    PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC,
    NeuralEmpiricalInferenceSpec,
    NeuralVariantContrast,
    neural_bootstrap_configurations,
)
from .neural_spec import NeuralVariant
from .project_metrics import ProjectPricingMetrics, evaluate_project_pricing
from .publication_inference_spec import (
    PublicationBootstrapDesign,
    PublicationCohortSpec,
    PublicationEmpiricalMetric,
)

Float32Vector = npt.NDArray[np.float32]
Float64Vector = npt.NDArray[np.float64]
Float64Matrix = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]
UInt64Vector = npt.NDArray[np.uint64]
BoolVector = npt.NDArray[np.bool_]

_FLOAT32: Final = np.dtype(np.float32)
_FLOAT64: Final = np.dtype(np.float64)
_PANEL_INPUT_SCHEMA: Final = {
    "ticker": pl.String,
    "quote_date": pl.Date,
    "call_mid_price": pl.Float32,
}
_ADDITIVE_FIELDS: Final = (
    "absolute_error_sum",
    "absolute_relative_error_sum",
    "squared_error_sum",
    "signed_error_sum",
    "relative_signed_error_sum",
    "symmetric_absolute_relative_error_sum",
)
_STATISTICS_PER_VARIANT: Final = len(_ADDITIVE_FIELDS)
_STATISTICS: Final = 1 + len(NeuralVariant) * _STATISTICS_PER_VARIANT
_PERCENT_SCALE: Final = 100.0
_MATRIX_DIMENSIONS: Final = 2
_NONNEGATIVE_ADDITIVE_OFFSETS: Final = (0, 1, 2, 5)
_MAX_ISO_YEAR: Final = 9_999


@dataclass(slots=True)
class NeuralInferenceError(ValueError):
    """Neural inference inputs or derived state violate the frozen contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected neural-inference contract."""
        return f"invalid neural empirical inference: {self.reason}"


@dataclass(frozen=True, slots=True, eq=False)
class NeuralPredictionSet:
    """Canonical target and five source-ordered float32 prediction vectors."""

    target: Float32Vector
    predictions: tuple[Float32Vector, ...]

    def __post_init__(self) -> None:
        """Require complete immutable finite vectors in model-enumeration order."""
        _validate_float32_vector(self.target, field="target")
        if np.any(self.target <= 0.0):
            raise NeuralInferenceError(reason="targets must be strictly positive")
        if type(self.predictions) is not tuple or len(self.predictions) != len(NeuralVariant):
            raise NeuralInferenceError(
                reason="predictions must contain all five variants in canonical order"
            )
        for variant, prediction in zip(NeuralVariant, self.predictions, strict=True):
            _validate_float32_vector(
                prediction,
                field=f"prediction[{variant.value}]",
                length=self.target.size,
            )
            if np.any(prediction < 0.0):
                raise NeuralInferenceError(
                    reason=f"prediction[{variant.value}] must be nonnegative"
                )
        for array in (self.target, *self.predictions):
            if array.flags.writeable:
                raise NeuralInferenceError(reason="prediction-set arrays must be immutable")

    def for_variant(self, variant: NeuralVariant) -> Float32Vector:
        """Return one canonical prediction vector without string aliases."""
        if type(variant) is not NeuralVariant:
            raise NeuralInferenceError(reason="variant must be NeuralVariant")
        return self.predictions[tuple(NeuralVariant).index(variant)]


@dataclass(frozen=True, slots=True, eq=False)
class NeuralCohortPanelStatistics:
    """One cohort's point metrics and ordered additive quote-date panels."""

    cohort: PublicationCohortSpec
    observations: int
    quote_date_panels: int
    point_metrics: tuple[ProjectPricingMetrics, ...]
    panel_years: Int64Vector
    additive_statistics: Float64Matrix

    def __post_init__(self) -> None:
        """Require one complete finite 31-column panel reduction."""
        if type(self.cohort) is not PublicationCohortSpec:
            raise NeuralInferenceError(reason="cohort must be PublicationCohortSpec")
        if type(self.observations) is not int or self.observations <= 0:
            raise NeuralInferenceError(reason="cohort observations must be positive")
        if type(self.quote_date_panels) is not int or self.quote_date_panels <= 0:
            raise NeuralInferenceError(reason="quote-date panel count must be positive")
        if (
            type(self.point_metrics) is not tuple
            or len(self.point_metrics) != len(NeuralVariant)
            or any(
                type(metric) is not ProjectPricingMetrics
                or metric.observations != self.observations
                for metric in self.point_metrics
            )
        ):
            raise NeuralInferenceError(reason="cohort point metrics are incomplete")
        _validate_int64_vector(
            self.panel_years,
            field="panel_years",
            length=self.quote_date_panels,
        )
        if self.panel_years.flags.writeable:
            raise NeuralInferenceError(reason="panel years must be immutable")
        if (
            np.any(self.panel_years < 1)
            or np.any(self.panel_years > _MAX_ISO_YEAR)
            or np.any(np.diff(self.panel_years) < 0)
        ):
            raise NeuralInferenceError(reason="panel years must be sorted ISO calendar years")
        _validate_float64_matrix(
            self.additive_statistics,
            field="additive_statistics",
            shape=(self.quote_date_panels, _STATISTICS),
        )
        if self.additive_statistics.flags.writeable:
            raise NeuralInferenceError(reason="panel statistics must be immutable")
        _validate_additive_content(self.additive_statistics, field="panel statistics")
        observed = float(np.sum(self.additive_statistics[:, 0], dtype=np.float64))
        if observed != float(self.observations):
            raise NeuralInferenceError(reason="panel counts do not sum to observations")


@dataclass(frozen=True, slots=True, eq=False)
class NeuralBootstrapConfigurationResult:
    """One cohort/block configuration's paired additive bootstrap sums."""

    design: PublicationBootstrapDesign
    mean_block_length: float
    domain: int
    additive_sums: Float64Matrix

    def __post_init__(self) -> None:
        """Require one finite immutable native bootstrap matrix."""
        if type(self.design) is not PublicationBootstrapDesign:
            raise NeuralInferenceError(reason="bootstrap design is invalid")
        if (
            type(self.mean_block_length) is not float
            or not math.isfinite(self.mean_block_length)
            or self.mean_block_length < 1.0
        ):
            raise NeuralInferenceError(reason="mean block length must be finite and >= 1")
        if type(self.domain) is not int or not 0 <= self.domain <= (2**32) - 1:
            raise NeuralInferenceError(reason="bootstrap domain must be unsigned 32-bit")
        _validate_float64_matrix(self.additive_sums, field="additive_sums")
        if self.additive_sums.shape[1] != _STATISTICS:
            raise NeuralInferenceError(reason="bootstrap statistic columns are incomplete")
        if self.additive_sums.flags.writeable:
            raise NeuralInferenceError(reason="bootstrap sums must be immutable")
        _validate_additive_content(self.additive_sums, field="bootstrap sums")


@dataclass(frozen=True, slots=True)
class NeuralMetricInterval:
    """Basic bootstrap interval for one neural-model metric."""

    cohort: str
    bootstrap_design: PublicationBootstrapDesign
    mean_block_length: float
    variant: NeuralVariant
    metric: PublicationEmpiricalMetric
    estimate: float
    interval_lower: float
    interval_upper: float

    def __post_init__(self) -> None:
        """Require a finite ordered interval and typed identities."""
        _validate_summary_identity(
            self.cohort,
            self.bootstrap_design,
            self.variant,
            self.metric,
        )
        _validate_finite_summary(self.estimate, self.interval_lower, self.interval_upper)


@dataclass(frozen=True, slots=True)
class NeuralContrastInference:
    """Paired basic interval and centered two-sided bootstrap test."""

    cohort: str
    bootstrap_design: PublicationBootstrapDesign
    mean_block_length: float
    contrast: NeuralVariantContrast
    metric: PublicationEmpiricalMetric
    estimate: float
    interval_lower: float
    interval_upper: float
    p_value: float
    holm_adjusted_p_value: float | None
    reject_family_wise_null: bool | None

    def __post_init__(self) -> None:
        """Require finite inference and all-or-none primary multiplicity state."""
        if type(self.cohort) is not str or not self.cohort:
            raise NeuralInferenceError(reason="contrast cohort must be nonempty")
        if type(self.bootstrap_design) is not PublicationBootstrapDesign:
            raise NeuralInferenceError(reason="contrast bootstrap design is invalid")
        if type(self.contrast) is not NeuralVariantContrast:
            raise NeuralInferenceError(reason="contrast identity is invalid")
        if type(self.metric) is not PublicationEmpiricalMetric:
            raise NeuralInferenceError(reason="contrast metric is invalid")
        _validate_finite_summary(self.estimate, self.interval_lower, self.interval_upper)
        if type(self.p_value) is not float or not 0.0 < self.p_value <= 1.0:
            raise NeuralInferenceError(reason="contrast p-value must be in (0, 1]")
        adjusted_missing = self.holm_adjusted_p_value is None
        decision_missing = self.reject_family_wise_null is None
        if adjusted_missing != decision_missing:
            raise NeuralInferenceError(reason="Holm adjustment and decision must be paired")
        if not adjusted_missing and (
            type(self.holm_adjusted_p_value) is not float
            or not 0.0 < self.holm_adjusted_p_value <= 1.0
            or type(self.reject_family_wise_null) is not bool
        ):
            raise NeuralInferenceError(reason="invalid Holm-adjusted inference")


@dataclass(frozen=True, slots=True, eq=False)
class NeuralCohortInferenceResult:
    """Complete block-length sensitivity inference for one cohort."""

    panels: NeuralCohortPanelStatistics
    bootstrap: tuple[NeuralBootstrapConfigurationResult, ...]

    def __post_init__(self) -> None:
        """Require at least one canonical block result."""
        if type(self.panels) is not NeuralCohortPanelStatistics:
            raise NeuralInferenceError(reason="panels must be prepared neural statistics")
        if type(self.bootstrap) is not tuple or not self.bootstrap:
            raise NeuralInferenceError(reason="cohort bootstrap results cannot be empty")


@dataclass(frozen=True, slots=True, eq=False)
class NeuralEmpiricalInferenceResult:
    """Complete frozen neural inference, including paired raw bootstrap sums."""

    spec: NeuralEmpiricalInferenceSpec
    cohorts: tuple[NeuralCohortInferenceResult, ...]
    model_intervals: tuple[NeuralMetricInterval, ...]
    contrasts: tuple[NeuralContrastInference, ...]

    def __post_init__(self) -> None:
        """Require canonical cohorts and exactly one complete primary Holm family."""
        if type(self.spec) is not NeuralEmpiricalInferenceSpec:
            raise NeuralInferenceError(reason="spec must be NeuralEmpiricalInferenceSpec")
        if tuple(result.panels.cohort for result in self.cohorts) != self.spec.cohorts:
            raise NeuralInferenceError(reason="cohort results are incomplete or reordered")
        primary = tuple(
            result
            for result in self.contrasts
            if result.cohort == self.spec.cohorts[0].name
            and result.bootstrap_design is PublicationBootstrapDesign.CIRCULAR_STATIONARY
            and result.mean_block_length == self.spec.bootstrap.mean_block_length
            and result.contrast in self.spec.primary_contrasts
            and result.metric in self.spec.primary_metrics
        )
        expected = len(self.spec.primary_contrasts) * len(self.spec.primary_metrics)
        if len(primary) != expected or any(
            result.holm_adjusted_p_value is None for result in primary
        ):
            raise NeuralInferenceError(reason="primary Holm family is incomplete")
        if any(
            result.holm_adjusted_p_value is not None and result not in primary
            for result in self.contrasts
        ):
            raise NeuralInferenceError(reason="Holm adjustment leaked into sensitivity output")


def run_neural_empirical_inference(
    frame: pl.DataFrame,
    predictions: NeuralPredictionSet,
    executor: CpuExecutor,
    spec: NeuralEmpiricalInferenceSpec = PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC,
) -> NeuralEmpiricalInferenceResult:
    """Prepare complete date panels, run paired bootstraps, and derive frozen inference."""
    if type(executor) is not CpuExecutor:
        raise NeuralInferenceError(reason="executor must be CpuExecutor")
    if type(spec) is not NeuralEmpiricalInferenceSpec:
        raise NeuralInferenceError(reason="spec must be NeuralEmpiricalInferenceSpec")
    prepared = prepare_neural_inference_panels(frame, predictions, spec)
    cohort_results: list[NeuralCohortInferenceResult] = []
    block_lengths = spec.bootstrap.sensitivity_mean_block_lengths
    for cohort_index, panels in enumerate(prepared):
        configurations: list[NeuralBootstrapConfigurationResult] = []
        for block_index, mean_block_length in enumerate(block_lengths):
            domain = spec.bootstrap.domain_start + cohort_index * len(block_lengths) + block_index
            sums = StationaryBootstrap(
                replications=spec.bootstrap.replications,
                mean_block_length=mean_block_length,
                random_stream=RandomStream(seed=spec.bootstrap.seed, domain=domain),
                replication_path_start=spec.bootstrap.replication_path_start,
            ).sum_panels(panels.additive_statistics, executor=executor)
            _validate_bootstrap_shape(sums, spec.bootstrap.replications)
            configurations.append(
                NeuralBootstrapConfigurationResult(
                    design=PublicationBootstrapDesign.CIRCULAR_STATIONARY,
                    mean_block_length=mean_block_length,
                    domain=domain,
                    additive_sums=sums,
                )
            )
        stratified_mean = spec.bootstrap.stratified_sensitivity_mean_block_length
        stratified_domain = spec.bootstrap.stratified_sensitivity_domain_start + cohort_index
        stratified_sums = StationaryBootstrap(
            replications=spec.bootstrap.replications,
            mean_block_length=stratified_mean,
            random_stream=RandomStream(seed=spec.bootstrap.seed, domain=stratified_domain),
            replication_path_start=spec.bootstrap.replication_path_start,
        ).sum_stratified_panels(
            panels.additive_statistics,
            _calendar_year_stratum_offsets(panels.panel_years),
            executor=executor,
        )
        _validate_bootstrap_shape(stratified_sums, spec.bootstrap.replications)
        configurations.append(
            NeuralBootstrapConfigurationResult(
                design=spec.bootstrap.stratified_sensitivity_design,
                mean_block_length=stratified_mean,
                domain=stratified_domain,
                additive_sums=stratified_sums,
            )
        )
        cohort_results.append(
            NeuralCohortInferenceResult(panels=panels, bootstrap=tuple(configurations))
        )
    cohorts = tuple(cohort_results)
    intervals, contrasts = summarize_neural_empirical_inference(cohorts, spec)
    return NeuralEmpiricalInferenceResult(
        spec=spec,
        cohorts=cohorts,
        model_intervals=intervals,
        contrasts=contrasts,
    )


def prepare_neural_inference_panels(
    frame: pl.DataFrame,
    predictions: NeuralPredictionSet,
    spec: NeuralEmpiricalInferenceSpec = PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC,
) -> tuple[NeuralCohortPanelStatistics, ...]:
    """Reduce source-ordered rows into complete chronological quote-date statistics."""
    _validate_panel_frame(frame, predictions)
    if type(spec) is not NeuralEmpiricalInferenceSpec:
        raise NeuralInferenceError(reason="spec must be NeuralEmpiricalInferenceSpec")
    frame_target = np.ascontiguousarray(frame.get_column("call_mid_price").to_numpy())
    if not np.array_equal(frame_target, predictions.target):
        raise NeuralInferenceError(reason="frame target differs from neural prediction targets")
    quote_dates = frame.get_column("quote_date").to_numpy()
    results = []
    for cohort in spec.cohorts:
        expression = pl.col("ticker").is_in(cohort.tickers)
        if cohort.quote_date_end_inclusive is not None:
            expression &= pl.col("quote_date") <= cohort.quote_date_end_inclusive
        mask = cast(
            "BoolVector",
            np.ascontiguousarray(frame.select(expression.alias("selected"))["selected"].to_numpy()),
        )
        indices = cast("Int64Vector", np.flatnonzero(mask).astype(np.int64, copy=False))
        if indices.size == 0:
            raise NeuralInferenceError(reason=f"cohort {cohort.name!r} is empty")
        point_metrics = tuple(
            evaluate_project_pricing(
                np.ascontiguousarray(predictions.target[indices]),
                np.ascontiguousarray(prediction[indices]),
            )
            for prediction in predictions.predictions
        )
        order = np.argsort(quote_dates[indices], kind="stable")
        ordered_indices = indices[order]
        ordered_dates = quote_dates[ordered_indices]
        starts = cast(
            "Int64Vector",
            np.flatnonzero(
                np.concatenate(
                    (
                        np.array([True], dtype=np.bool_),
                        ordered_dates[1:] != ordered_dates[:-1],
                    )
                )
            ).astype(np.int64, copy=False),
        )
        statistics = _additive_panel_statistics(ordered_indices, starts, predictions)
        statistics.flags.writeable = False
        panel_years = np.ascontiguousarray(
            ordered_dates[starts].astype("datetime64[Y]").astype(np.int64) + 1970,
            dtype=np.int64,
        )
        panel_years.flags.writeable = False
        results.append(
            NeuralCohortPanelStatistics(
                cohort=cohort,
                observations=int(indices.size),
                quote_date_panels=int(starts.size),
                point_metrics=point_metrics,
                panel_years=panel_years,
                additive_statistics=statistics,
            )
        )
    return tuple(results)


def neural_additive_statistic_names() -> tuple[str, ...]:
    """Return the exact column identity of every native bootstrap statistic."""
    return (
        "observation_count",
        *(
            f"{variant.value}.{field}"
            for variant in NeuralVariant
            for field in _ADDITIVE_FIELDS
        ),
    )


def summarize_neural_empirical_inference(
    cohorts: tuple[NeuralCohortInferenceResult, ...],
    spec: NeuralEmpiricalInferenceSpec = PROJECT_NEURAL_EMPIRICAL_INFERENCE_SPEC,
) -> tuple[tuple[NeuralMetricInterval, ...], tuple[NeuralContrastInference, ...]]:
    """Derive intervals, paired tests, and the primary Holm family from fixed sums."""
    if type(spec) is not NeuralEmpiricalInferenceSpec:
        raise NeuralInferenceError(reason="spec must be NeuralEmpiricalInferenceSpec")
    if (
        type(cohorts) is not tuple
        or tuple(result.panels.cohort for result in cohorts) != spec.cohorts
    ):
        raise NeuralInferenceError(reason="cohorts are incomplete or reordered")
    configurations = neural_bootstrap_configurations(spec)
    intervals: list[NeuralMetricInterval] = []
    contrasts: list[NeuralContrastInference] = []
    for cohort in cohorts:
        if (
            tuple((item.design, item.mean_block_length) for item in cohort.bootstrap)
            != configurations
        ):
            raise NeuralInferenceError(reason="bootstrap block results are incomplete")
        if any(
            item.additive_sums.shape != (spec.bootstrap.replications, _STATISTICS)
            for item in cohort.bootstrap
        ):
            raise NeuralInferenceError(reason="bootstrap replication matrices are incomplete")
        for configuration in cohort.bootstrap:
            block_intervals, block_contrasts = _summarize_configuration(
                cohort.panels,
                configuration,
                spec,
            )
            intervals.extend(block_intervals)
            contrasts.extend(block_contrasts)
    return tuple(intervals), _apply_primary_holm(tuple(contrasts), spec)


def _calendar_year_stratum_offsets(panel_years: Int64Vector) -> UInt64Vector:
    starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True], dtype=np.bool_),
                panel_years[1:] != panel_years[:-1],
            )
        )
    )
    offsets = np.ascontiguousarray(
        np.concatenate((starts, np.array([panel_years.size], dtype=np.int64))),
        dtype=np.uint64,
    )
    offsets.flags.writeable = False
    return offsets


def _additive_panel_statistics(
    ordered_indices: Int64Vector,
    starts: Int64Vector,
    predictions: NeuralPredictionSet,
) -> Float64Matrix:
    rows = int(ordered_indices.size)
    panels = int(starts.size)
    statistics = np.empty((panels, _STATISTICS), dtype=np.float64)
    ends = np.concatenate((starts[1:], np.array([rows], dtype=np.int64)))
    statistics[:, 0] = ends - starts
    observed = predictions.target[ordered_indices]
    for variant_index, prediction in enumerate(predictions.predictions):
        fitted = prediction[ordered_indices]
        error = np.subtract(fitted, observed, dtype=np.float64)
        absolute_error = np.abs(error)
        base = 1 + variant_index * _STATISTICS_PER_VARIANT
        statistics[:, base] = np.add.reduceat(absolute_error, starts)
        statistics[:, base + 1] = np.add.reduceat(absolute_error / observed, starts)
        statistics[:, base + 2] = np.add.reduceat(error * error, starts)
        statistics[:, base + 3] = np.add.reduceat(error, starts)
        statistics[:, base + 4] = np.add.reduceat(error / observed, starts)
        symmetric_denominator = np.add(observed, fitted, dtype=np.float64) / 2.0
        statistics[:, base + 5] = np.add.reduceat(
            absolute_error / symmetric_denominator,
            starts,
        )
    if not np.isfinite(statistics).all():
        raise NeuralInferenceError(reason="additive panel statistics became nonfinite")
    return statistics


def _summarize_configuration(
    panels: NeuralCohortPanelStatistics,
    configuration: NeuralBootstrapConfigurationResult,
    spec: NeuralEmpiricalInferenceSpec,
) -> tuple[list[NeuralMetricInterval], list[NeuralContrastInference]]:
    metrics = tuple(PublicationEmpiricalMetric)
    draws = {
        (variant, metric): _metric_draws(configuration.additive_sums, variant, metric)
        for variant in NeuralVariant
        for metric in metrics
    }
    intervals = []
    for variant_index, variant in enumerate(NeuralVariant):
        point = panels.point_metrics[variant_index]
        for metric in metrics:
            estimate = _point_metric(point, metric)
            lower, upper = _basic_interval(
                estimate,
                draws[(variant, metric)],
                spec.bootstrap.confidence_level,
            )
            intervals.append(
                NeuralMetricInterval(
                    cohort=panels.cohort.name,
                    bootstrap_design=configuration.design,
                    mean_block_length=configuration.mean_block_length,
                    variant=variant,
                    metric=metric,
                    estimate=estimate,
                    interval_lower=lower,
                    interval_upper=upper,
                )
            )
    contrasts = []
    for contrast in spec.contrasts:
        left_index = tuple(NeuralVariant).index(contrast.left)
        right_index = tuple(NeuralVariant).index(contrast.right)
        for metric in metrics:
            estimate = _point_metric(panels.point_metrics[left_index], metric) - _point_metric(
                panels.point_metrics[right_index], metric
            )
            contrast_draws = draws[(contrast.left, metric)] - draws[(contrast.right, metric)]
            lower, upper = _basic_interval(
                estimate,
                contrast_draws,
                spec.bootstrap.confidence_level,
            )
            deviations = contrast_draws - estimate
            extreme = int(np.count_nonzero(np.abs(deviations) >= abs(estimate)))
            p_value = float((extreme + 1) / (spec.bootstrap.replications + 1))
            contrasts.append(
                NeuralContrastInference(
                    cohort=panels.cohort.name,
                    bootstrap_design=configuration.design,
                    mean_block_length=configuration.mean_block_length,
                    contrast=contrast,
                    metric=metric,
                    estimate=estimate,
                    interval_lower=lower,
                    interval_upper=upper,
                    p_value=p_value,
                    holm_adjusted_p_value=None,
                    reject_family_wise_null=None,
                )
            )
    return intervals, contrasts


def _metric_draws(
    sums: Float64Matrix,
    variant: NeuralVariant,
    metric: PublicationEmpiricalMetric,
) -> Float64Vector:
    count = sums[:, 0]
    base = 1 + tuple(NeuralVariant).index(variant) * _STATISTICS_PER_VARIANT
    if metric is PublicationEmpiricalMetric.MAE:
        return sums[:, base] / count
    if metric is PublicationEmpiricalMetric.MAPE_PERCENT:
        return (sums[:, base + 1] / count) * _PERCENT_SCALE
    if metric is PublicationEmpiricalMetric.RMSE:
        return cast("Float64Vector", np.sqrt(sums[:, base + 2] / count))
    if metric is PublicationEmpiricalMetric.BIAS:
        return sums[:, base + 3] / count
    if metric is PublicationEmpiricalMetric.RELATIVE_BIAS:
        return sums[:, base + 4] / count
    if metric is PublicationEmpiricalMetric.SMAPE_PERCENT:
        return (sums[:, base + 5] / count) * _PERCENT_SCALE
    raise NeuralInferenceError(reason=f"unsupported metric {metric!r}")


def _point_metric(
    metrics: ProjectPricingMetrics,
    metric: PublicationEmpiricalMetric,
) -> float:
    if metric is PublicationEmpiricalMetric.MAE:
        return metrics.mae
    if metric is PublicationEmpiricalMetric.MAPE_PERCENT:
        return metrics.mape_percent
    if metric is PublicationEmpiricalMetric.RMSE:
        return metrics.rmse
    if metric is PublicationEmpiricalMetric.BIAS:
        return metrics.bias
    if metric is PublicationEmpiricalMetric.RELATIVE_BIAS:
        return metrics.relative_bias
    if metric is PublicationEmpiricalMetric.SMAPE_PERCENT:
        return metrics.smape_percent
    raise NeuralInferenceError(reason=f"unsupported metric {metric!r}")


def _basic_interval(
    estimate: float,
    draws: Float64Vector,
    confidence_level: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    quantiles = np.quantile(draws, (alpha / 2.0, 1.0 - alpha / 2.0), method="linear")
    lower = float(2.0 * estimate - quantiles[1])
    upper = float(2.0 * estimate - quantiles[0])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise NeuralInferenceError(reason="basic bootstrap interval became invalid")
    return lower, upper


def _apply_primary_holm(
    results: tuple[NeuralContrastInference, ...],
    spec: NeuralEmpiricalInferenceSpec,
) -> tuple[NeuralContrastInference, ...]:
    primary_keys = tuple(
        (
            spec.cohorts[0].name,
            PublicationBootstrapDesign.CIRCULAR_STATIONARY,
            spec.bootstrap.mean_block_length,
            contrast,
            metric,
        )
        for contrast in spec.primary_contrasts
        for metric in spec.primary_metrics
    )
    locations = {
        (
            result.cohort,
            result.bootstrap_design,
            result.mean_block_length,
            result.contrast,
            result.metric,
        ): index
        for index, result in enumerate(results)
    }
    try:
        primary_locations = tuple(locations[key] for key in primary_keys)
    except KeyError as error:
        raise NeuralInferenceError(reason="primary Holm family is incomplete") from error
    p_values = tuple(results[index].p_value for index in primary_locations)
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, family_index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[family_index]
        running = max(running, candidate)
        adjusted[family_index] = min(1.0, running)
    output = list(results)
    for family_index, location in enumerate(primary_locations):
        adjusted_p = float(adjusted[family_index])
        output[location] = replace(
            output[location],
            holm_adjusted_p_value=adjusted_p,
            reject_family_wise_null=adjusted_p <= spec.family_wise_error_rate,
        )
    return tuple(output)


def _validate_bootstrap_shape(sums: Float64Matrix, replications: int) -> None:
    if sums.shape != (replications, _STATISTICS):
        raise NeuralInferenceError(reason="native bootstrap returned an invalid shape")
    if np.any(sums[:, 0] <= 0.0):
        raise NeuralInferenceError(reason="bootstrap row counts must be positive")


def _validate_summary_identity(
    cohort: object,
    design: object,
    variant: object,
    metric: object,
) -> None:
    if type(cohort) is not str or not cohort:
        raise NeuralInferenceError(reason="metric interval cohort must be nonempty")
    if type(design) is not PublicationBootstrapDesign:
        raise NeuralInferenceError(reason="metric interval bootstrap design is invalid")
    if type(variant) is not NeuralVariant:
        raise NeuralInferenceError(reason="metric interval variant is invalid")
    if type(metric) is not PublicationEmpiricalMetric:
        raise NeuralInferenceError(reason="metric interval metric is invalid")


def _validate_panel_frame(frame: object, predictions: NeuralPredictionSet) -> None:
    if type(frame) is not pl.DataFrame:
        raise NeuralInferenceError(reason="frame must be polars.DataFrame")
    if type(predictions) is not NeuralPredictionSet:
        raise NeuralInferenceError(reason="predictions must be NeuralPredictionSet")
    schema = frame.schema
    missing = tuple(column for column in _PANEL_INPUT_SCHEMA if column not in schema)
    mismatches = tuple(
        (column, str(schema[column]), str(dtype))
        for column, dtype in _PANEL_INPUT_SCHEMA.items()
        if column in schema and schema[column] != dtype
    )
    if missing or mismatches:
        raise NeuralInferenceError(
            reason=f"panel frame schema differs: missing={missing!r}, mismatches={mismatches!r}"
        )
    if frame.height != predictions.target.size:
        raise NeuralInferenceError(reason="panel frame and predictions have different rows")
    nulls = frame.select(pl.col(column).null_count() for column in _PANEL_INPUT_SCHEMA).row(0)
    if any(nulls):
        raise NeuralInferenceError(reason="panel identity columns cannot contain nulls")


def _validate_float32_vector(
    value: object,
    *,
    field: str,
    length: int | None = None,
) -> None:
    if type(value) is not np.ndarray:
        raise NeuralInferenceError(reason=f"{field} must be a plain numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if array.dtype != _FLOAT32 or array.ndim != 1 or not array.flags.c_contiguous:
        raise NeuralInferenceError(reason=f"{field} must be a C-contiguous Float32 vector")
    if array.size == 0 or (length is not None and array.size != length):
        raise NeuralInferenceError(reason=f"{field} has an invalid length")
    if not np.isfinite(array).all():
        raise NeuralInferenceError(reason=f"{field} must be finite")


def _validate_int64_vector(
    value: object,
    *,
    field: str,
    length: int,
) -> None:
    if type(value) is not np.ndarray:
        raise NeuralInferenceError(reason=f"{field} must be a plain numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if (
        array.dtype != np.dtype(np.int64)
        or array.ndim != 1
        or not array.flags.c_contiguous
        or array.size != length
    ):
        raise NeuralInferenceError(
            reason=f"{field} must be a C-contiguous Int64 vector of length {length}"
        )


def _validate_float64_matrix(
    value: object,
    *,
    field: str,
    shape: tuple[int, int] | None = None,
) -> None:
    if type(value) is not np.ndarray:
        raise NeuralInferenceError(reason=f"{field} must be a plain numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if array.dtype != _FLOAT64 or array.ndim != _MATRIX_DIMENSIONS or not array.flags.c_contiguous:
        raise NeuralInferenceError(reason=f"{field} must be a C-contiguous Float64 matrix")
    if 0 in array.shape or (shape is not None and array.shape != shape):
        raise NeuralInferenceError(reason=f"{field} has an invalid shape")
    if not np.isfinite(array).all():
        raise NeuralInferenceError(reason=f"{field} must be finite")


def _validate_finite_summary(estimate: float, lower: float, upper: float) -> None:
    if any(
        type(value) is not float or not math.isfinite(value) for value in (estimate, lower, upper)
    ):
        raise NeuralInferenceError(reason="inference summary values must be finite floats")
    if lower > upper:
        raise NeuralInferenceError(reason="inference interval is reversed")


def _validate_additive_content(values: Float64Matrix, *, field: str) -> None:
    counts = values[:, 0]
    if np.any(counts <= 0.0) or not np.array_equal(counts, np.floor(counts)):
        raise NeuralInferenceError(reason=f"{field} counts must be positive exact integers")
    for variant_index in range(len(NeuralVariant)):
        base = 1 + variant_index * _STATISTICS_PER_VARIANT
        for offset in _NONNEGATIVE_ADDITIVE_OFFSETS:
            if np.any(values[:, base + offset] < 0.0):
                raise NeuralInferenceError(
                    reason=f"{field} contains a negative nonnegative statistic"
                )


__all__ = [
    "NeuralBootstrapConfigurationResult",
    "NeuralCohortInferenceResult",
    "NeuralCohortPanelStatistics",
    "NeuralContrastInference",
    "NeuralEmpiricalInferenceResult",
    "NeuralInferenceError",
    "NeuralMetricInterval",
    "NeuralPredictionSet",
    "neural_additive_statistic_names",
    "prepare_neural_inference_panels",
    "run_neural_empirical_inference",
    "summarize_neural_empirical_inference",
]
