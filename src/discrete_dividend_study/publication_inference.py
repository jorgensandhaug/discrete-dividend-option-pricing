"""Paired date-panel empirical inference for the frozen publication comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from finoptions import AndersenLakeBatchResult, CpuExecutor, RandomStream, StationaryBootstrap

from .project_metrics import ProjectPricingMetrics, evaluate_project_pricing_f64
from .publication_inference_spec import (
    PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
    PublicationCohortSpec,
    PublicationEmpiricalInferenceSpec,
    PublicationEmpiricalMetric,
    PublicationVariantContrast,
)
from .publication_result_artifact import LoadedPublicationStudyResultArtifact
from .publication_spec import PublicationVariant

Float64Vector = npt.NDArray[np.float64]
Float64Matrix = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]
BoolVector = npt.NDArray[np.bool_]

_FLOAT64: Final = np.dtype(np.float64)
_PANEL_INPUT_SCHEMA: Final = {
    "ticker": pl.String,
    "quote_date": pl.Date,
    "call_mid_price": pl.Float64,
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
_STATISTICS: Final = 1 + len(PublicationVariant) * _STATISTICS_PER_VARIANT
_PERCENT_SCALE: Final = 100.0
_MATRIX_DIMENSIONS: Final = 2


@dataclass(slots=True)
class PublicationInferenceError(ValueError):
    """The empirical inference inputs or derived state violate the frozen contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected inference contract."""
        return f"invalid publication empirical inference: {self.reason}"


@dataclass(frozen=True, slots=True, eq=False)
class PublicationPredictionSet:
    """Canonical target and four source-ordered publication prediction vectors."""

    target: Float64Vector
    predictions: tuple[Float64Vector, ...]

    def __post_init__(self) -> None:
        """Require complete immutable finite vectors in enumeration order."""
        _validate_float64_vector(self.target, field="target")
        if np.any(self.target <= 0.0):
            raise PublicationInferenceError(reason="targets must be strictly positive")
        if type(self.predictions) is not tuple or len(self.predictions) != len(PublicationVariant):
            raise PublicationInferenceError(
                reason="predictions must contain all four variants in canonical order"
            )
        for variant, prediction in zip(PublicationVariant, self.predictions, strict=True):
            _validate_float64_vector(
                prediction,
                field=f"prediction[{variant.value}]",
                length=self.target.size,
            )
            if np.any(prediction < 0.0):
                raise PublicationInferenceError(
                    reason=f"prediction[{variant.value}] must be nonnegative"
                )
        for array in (self.target, *self.predictions):
            if array.flags.writeable:
                raise PublicationInferenceError(reason="prediction-set arrays must be immutable")

    def for_variant(self, variant: PublicationVariant) -> Float64Vector:
        """Return one canonical prediction vector without accepting string aliases."""
        if type(variant) is not PublicationVariant:
            raise PublicationInferenceError(reason="variant must be PublicationVariant")
        return self.predictions[tuple(PublicationVariant).index(variant)]


def publication_prediction_set_from_result(
    loaded: LoadedPublicationStudyResultArtifact,
) -> PublicationPredictionSet:
    """Extract the final cash replication means and deterministic yields in canonical order."""
    if type(loaded) is not LoadedPublicationStudyResultArtifact:
        raise PublicationInferenceError(
            reason="loaded result must be LoadedPublicationStudyResultArtifact"
        )
    predictions: dict[PublicationVariant, Float64Vector] = {
        result.variant: result.mean_prices for result in loaded.result.cash
    }
    for result in loaded.result.yields.variants:
        if type(result.native) is not AndersenLakeBatchResult:
            raise PublicationInferenceError(reason="yield result has an invalid native payload")
        predictions[result.variant] = result.native.prices
    if set(predictions) != set(PublicationVariant):
        raise PublicationInferenceError(reason="publication result variants are incomplete")
    return PublicationPredictionSet(
        target=loaded.target,
        predictions=tuple(predictions[variant] for variant in PublicationVariant),
    )


@dataclass(frozen=True, slots=True, eq=False)
class PublicationCohortPanelStatistics:
    """One cohort's point metrics and ordered additive quote-date panels."""

    cohort: PublicationCohortSpec
    observations: int
    quote_date_panels: int
    point_metrics: tuple[ProjectPricingMetrics, ...]
    additive_statistics: Float64Matrix

    def __post_init__(self) -> None:
        """Require a complete finite 25-column panel reduction."""
        if type(self.cohort) is not PublicationCohortSpec:
            raise PublicationInferenceError(reason="cohort must be PublicationCohortSpec")
        if type(self.observations) is not int or self.observations <= 0:
            raise PublicationInferenceError(reason="cohort observations must be positive")
        if type(self.quote_date_panels) is not int or self.quote_date_panels <= 0:
            raise PublicationInferenceError(reason="quote-date panel count must be positive")
        if (
            type(self.point_metrics) is not tuple
            or len(self.point_metrics) != len(PublicationVariant)
            or any(
                type(metric) is not ProjectPricingMetrics
                or metric.observations != self.observations
                for metric in self.point_metrics
            )
        ):
            raise PublicationInferenceError(reason="cohort point metrics are incomplete")
        _validate_float64_matrix(
            self.additive_statistics,
            field="additive_statistics",
            shape=(self.quote_date_panels, _STATISTICS),
        )
        if self.additive_statistics.flags.writeable:
            raise PublicationInferenceError(reason="panel statistics must be immutable")
        observed = float(np.sum(self.additive_statistics[:, 0], dtype=np.float64))
        if observed != float(self.observations):
            raise PublicationInferenceError(reason="panel counts do not sum to observations")


@dataclass(frozen=True, slots=True, eq=False)
class PublicationBootstrapConfigurationResult:
    """One cohort/block configuration's paired additive bootstrap sums."""

    mean_block_length: float
    domain: int
    additive_sums: Float64Matrix

    def __post_init__(self) -> None:
        """Require one finite immutable native bootstrap matrix."""
        if (
            type(self.mean_block_length) is not float
            or not math.isfinite(self.mean_block_length)
            or self.mean_block_length < 1.0
        ):
            raise PublicationInferenceError(reason="mean block length must be finite and >= 1")
        if type(self.domain) is not int or not 0 <= self.domain <= (2**32) - 1:
            raise PublicationInferenceError(reason="bootstrap domain must be unsigned 32-bit")
        _validate_float64_matrix(self.additive_sums, field="additive_sums")
        if self.additive_sums.shape[1] != _STATISTICS:
            raise PublicationInferenceError(reason="bootstrap statistic columns are incomplete")
        if self.additive_sums.flags.writeable:
            raise PublicationInferenceError(reason="bootstrap sums must be immutable")


@dataclass(frozen=True, slots=True)
class PublicationMetricInterval:
    """Basic bootstrap interval for one model metric."""

    cohort: str
    mean_block_length: float
    variant: PublicationVariant
    metric: PublicationEmpiricalMetric
    estimate: float
    interval_lower: float
    interval_upper: float

    def __post_init__(self) -> None:
        """Require a finite ordered interval and typed identities."""
        if type(self.cohort) is not str or not self.cohort:
            raise PublicationInferenceError(reason="metric interval cohort must be nonempty")
        if type(self.variant) is not PublicationVariant:
            raise PublicationInferenceError(reason="metric interval variant is invalid")
        if type(self.metric) is not PublicationEmpiricalMetric:
            raise PublicationInferenceError(reason="metric interval metric is invalid")
        _validate_finite_summary(self.estimate, self.interval_lower, self.interval_upper)


@dataclass(frozen=True, slots=True)
class PublicationContrastInference:
    """Paired basic interval and centered two-sided bootstrap test."""

    cohort: str
    mean_block_length: float
    contrast: PublicationVariantContrast
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
            raise PublicationInferenceError(reason="contrast cohort must be nonempty")
        if type(self.contrast) is not PublicationVariantContrast:
            raise PublicationInferenceError(reason="contrast identity is invalid")
        if type(self.metric) is not PublicationEmpiricalMetric:
            raise PublicationInferenceError(reason="contrast metric is invalid")
        _validate_finite_summary(self.estimate, self.interval_lower, self.interval_upper)
        if type(self.p_value) is not float or not 0.0 < self.p_value <= 1.0:
            raise PublicationInferenceError(reason="contrast p-value must be in (0, 1]")
        adjusted_missing = self.holm_adjusted_p_value is None
        decision_missing = self.reject_family_wise_null is None
        if adjusted_missing != decision_missing:
            raise PublicationInferenceError(reason="Holm adjustment and decision must be paired")
        if not adjusted_missing and (
            type(self.holm_adjusted_p_value) is not float
            or not 0.0 < self.holm_adjusted_p_value <= 1.0
            or type(self.reject_family_wise_null) is not bool
        ):
            raise PublicationInferenceError(reason="invalid Holm-adjusted inference")


@dataclass(frozen=True, slots=True, eq=False)
class PublicationCohortInferenceResult:
    """Complete block-length sensitivity inference for one cohort."""

    panels: PublicationCohortPanelStatistics
    bootstrap: tuple[PublicationBootstrapConfigurationResult, ...]

    def __post_init__(self) -> None:
        """Require at least one canonical block result."""
        if type(self.panels) is not PublicationCohortPanelStatistics:
            raise PublicationInferenceError(reason="panels must be prepared cohort statistics")
        if type(self.bootstrap) is not tuple or not self.bootstrap:
            raise PublicationInferenceError(reason="cohort bootstrap results cannot be empty")


@dataclass(frozen=True, slots=True, eq=False)
class PublicationEmpiricalInferenceResult:
    """Complete frozen empirical inference, including paired raw bootstrap sums."""

    spec: PublicationEmpiricalInferenceSpec
    cohorts: tuple[PublicationCohortInferenceResult, ...]
    model_intervals: tuple[PublicationMetricInterval, ...]
    contrasts: tuple[PublicationContrastInference, ...]

    def __post_init__(self) -> None:
        """Require canonical cohort order and a complete primary Holm family."""
        if type(self.spec) is not PublicationEmpiricalInferenceSpec:
            raise PublicationInferenceError(reason="spec must be PublicationEmpiricalInferenceSpec")
        if tuple(result.panels.cohort for result in self.cohorts) != self.spec.cohorts:
            raise PublicationInferenceError(reason="cohort results are incomplete or reordered")
        primary = tuple(
            result
            for result in self.contrasts
            if result.cohort == self.spec.cohorts[0].name
            and result.mean_block_length == self.spec.bootstrap.mean_block_length
            and result.metric in self.spec.primary_metrics
        )
        if len(primary) != len(self.spec.contrasts) * len(self.spec.primary_metrics) or any(
            result.holm_adjusted_p_value is None for result in primary
        ):
            raise PublicationInferenceError(reason="primary Holm family is incomplete")
        if any(
            result.holm_adjusted_p_value is not None and result not in primary
            for result in self.contrasts
        ):
            raise PublicationInferenceError(reason="Holm adjustment leaked into sensitivity output")


def run_publication_empirical_inference(
    frame: pl.DataFrame,
    predictions: PublicationPredictionSet,
    executor: CpuExecutor,
    spec: PublicationEmpiricalInferenceSpec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
) -> PublicationEmpiricalInferenceResult:
    """Prepare complete date panels, run all paired bootstraps, and derive frozen inference."""
    if type(executor) is not CpuExecutor:
        raise PublicationInferenceError(reason="executor must be CpuExecutor")
    if type(spec) is not PublicationEmpiricalInferenceSpec:
        raise PublicationInferenceError(reason="spec must be PublicationEmpiricalInferenceSpec")
    prepared = prepare_publication_inference_panels(frame, predictions, spec)
    cohort_results: list[PublicationCohortInferenceResult] = []
    block_lengths = spec.bootstrap.sensitivity_mean_block_lengths
    for cohort_index, panels in enumerate(prepared):
        configurations: list[PublicationBootstrapConfigurationResult] = []
        for block_index, mean_block_length in enumerate(block_lengths):
            domain = spec.bootstrap.domain_start + cohort_index * len(block_lengths) + block_index
            sums = StationaryBootstrap(
                replications=spec.bootstrap.replications,
                mean_block_length=mean_block_length,
                random_stream=RandomStream(seed=spec.bootstrap.seed, domain=domain),
                replication_path_start=spec.bootstrap.replication_path_start,
            ).sum_panels(panels.additive_statistics, executor=executor)
            if sums.shape != (spec.bootstrap.replications, _STATISTICS):
                raise PublicationInferenceError(reason="native bootstrap returned an invalid shape")
            if not np.array_equal(sums[:, 0], np.full(sums.shape[0], panels.observations)):
                raise PublicationInferenceError(
                    reason="bootstrap replications do not preserve the cohort observation count"
                )
            configuration = PublicationBootstrapConfigurationResult(
                mean_block_length=mean_block_length,
                domain=domain,
                additive_sums=sums,
            )
            configurations.append(configuration)
        cohort_results.append(
            PublicationCohortInferenceResult(panels=panels, bootstrap=tuple(configurations))
        )
    cohorts = tuple(cohort_results)
    intervals, contrasts = summarize_publication_empirical_inference(cohorts, spec)
    return PublicationEmpiricalInferenceResult(
        spec=spec,
        cohorts=cohorts,
        model_intervals=intervals,
        contrasts=contrasts,
    )


def prepare_publication_inference_panels(
    frame: pl.DataFrame,
    predictions: PublicationPredictionSet,
    spec: PublicationEmpiricalInferenceSpec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
) -> tuple[PublicationCohortPanelStatistics, ...]:
    """Reduce source-ordered rows into complete chronological quote-date sufficient statistics."""
    _validate_panel_frame(frame, predictions)
    if type(spec) is not PublicationEmpiricalInferenceSpec:
        raise PublicationInferenceError(reason="spec must be PublicationEmpiricalInferenceSpec")
    frame_target = np.ascontiguousarray(frame.get_column("call_mid_price").to_numpy())
    if not np.array_equal(frame_target, predictions.target):
        raise PublicationInferenceError(reason="frame target differs from result-artifact target")
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
            raise PublicationInferenceError(reason=f"cohort {cohort.name!r} is empty")
        point_metrics = tuple(
            evaluate_project_pricing_f64(
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
        results.append(
            PublicationCohortPanelStatistics(
                cohort=cohort,
                observations=int(indices.size),
                quote_date_panels=int(starts.size),
                point_metrics=point_metrics,
                additive_statistics=statistics,
            )
        )
    return tuple(results)


def publication_additive_statistic_names() -> tuple[str, ...]:
    """Return the exact public column identity of every native bootstrap statistic."""
    return (
        "observation_count",
        *(
            f"{variant.value}.{field}"
            for variant in PublicationVariant
            for field in _ADDITIVE_FIELDS
        ),
    )


def summarize_publication_empirical_inference(
    cohorts: tuple[PublicationCohortInferenceResult, ...],
    spec: PublicationEmpiricalInferenceSpec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
) -> tuple[tuple[PublicationMetricInterval, ...], tuple[PublicationContrastInference, ...]]:
    """Derive intervals, paired tests, and the primary Holm family from fixed raw sums."""
    if type(spec) is not PublicationEmpiricalInferenceSpec:
        raise PublicationInferenceError(reason="spec must be PublicationEmpiricalInferenceSpec")
    if (
        type(cohorts) is not tuple
        or tuple(result.panels.cohort for result in cohorts) != spec.cohorts
    ):
        raise PublicationInferenceError(reason="cohorts are incomplete or reordered")
    block_lengths = spec.bootstrap.sensitivity_mean_block_lengths
    intervals: list[PublicationMetricInterval] = []
    contrasts: list[PublicationContrastInference] = []
    for cohort in cohorts:
        if tuple(item.mean_block_length for item in cohort.bootstrap) != block_lengths:
            raise PublicationInferenceError(reason="bootstrap block results are incomplete")
        for configuration in cohort.bootstrap:
            block_intervals, block_contrasts = _summarize_configuration(
                cohort.panels,
                configuration,
                spec,
            )
            intervals.extend(block_intervals)
            contrasts.extend(block_contrasts)
    return tuple(intervals), _apply_primary_holm(tuple(contrasts), spec)


def _additive_panel_statistics(
    ordered_indices: Int64Vector,
    starts: Int64Vector,
    predictions: PublicationPredictionSet,
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
        raise PublicationInferenceError(reason="additive panel statistics became nonfinite")
    return statistics


def _summarize_configuration(
    panels: PublicationCohortPanelStatistics,
    configuration: PublicationBootstrapConfigurationResult,
    spec: PublicationEmpiricalInferenceSpec,
) -> tuple[list[PublicationMetricInterval], list[PublicationContrastInference]]:
    metrics = tuple(PublicationEmpiricalMetric)
    draws = {
        (variant, metric): _metric_draws(configuration.additive_sums, variant, metric)
        for variant in PublicationVariant
        for metric in metrics
    }
    intervals = []
    for variant_index, variant in enumerate(PublicationVariant):
        point = panels.point_metrics[variant_index]
        for metric in metrics:
            estimate = _point_metric(point, metric)
            lower, upper = _basic_interval(
                estimate,
                draws[(variant, metric)],
                spec.bootstrap.confidence_level,
            )
            intervals.append(
                PublicationMetricInterval(
                    cohort=panels.cohort.name,
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
        left_index = tuple(PublicationVariant).index(contrast.left)
        right_index = tuple(PublicationVariant).index(contrast.right)
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
                PublicationContrastInference(
                    cohort=panels.cohort.name,
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
    variant: PublicationVariant,
    metric: PublicationEmpiricalMetric,
) -> Float64Vector:
    count = sums[:, 0]
    base = 1 + tuple(PublicationVariant).index(variant) * _STATISTICS_PER_VARIANT
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
    raise PublicationInferenceError(reason=f"unsupported metric {metric!r}")


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
    raise PublicationInferenceError(reason=f"unsupported metric {metric!r}")


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
        raise PublicationInferenceError(reason="basic bootstrap interval became invalid")
    return lower, upper


def _apply_primary_holm(
    results: tuple[PublicationContrastInference, ...],
    spec: PublicationEmpiricalInferenceSpec,
) -> tuple[PublicationContrastInference, ...]:
    primary_keys = tuple(
        (
            spec.cohorts[0].name,
            spec.bootstrap.mean_block_length,
            contrast,
            metric,
        )
        for contrast in spec.contrasts
        for metric in spec.primary_metrics
    )
    locations = {
        (result.cohort, result.mean_block_length, result.contrast, result.metric): index
        for index, result in enumerate(results)
    }
    try:
        primary_locations = tuple(locations[key] for key in primary_keys)
    except KeyError as error:
        raise PublicationInferenceError(reason="primary Holm family is incomplete") from error
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


def _validate_panel_frame(frame: object, predictions: PublicationPredictionSet) -> None:
    if type(frame) is not pl.DataFrame:
        raise PublicationInferenceError(reason="frame must be polars.DataFrame")
    if type(predictions) is not PublicationPredictionSet:
        raise PublicationInferenceError(reason="predictions must be PublicationPredictionSet")
    schema = frame.schema
    missing = tuple(column for column in _PANEL_INPUT_SCHEMA if column not in schema)
    mismatches = tuple(
        (column, str(schema[column]), str(dtype))
        for column, dtype in _PANEL_INPUT_SCHEMA.items()
        if column in schema and schema[column] != dtype
    )
    if missing or mismatches:
        raise PublicationInferenceError(
            reason=f"panel frame schema differs: missing={missing!r}, mismatches={mismatches!r}"
        )
    if frame.height != predictions.target.size:
        raise PublicationInferenceError(reason="panel frame and predictions have different rows")
    nulls = frame.select(pl.col(column).null_count() for column in _PANEL_INPUT_SCHEMA).row(0)
    if any(nulls):
        raise PublicationInferenceError(reason="panel identity columns cannot contain nulls")


def _validate_float64_vector(
    value: object,
    *,
    field: str,
    length: int | None = None,
) -> None:
    if type(value) is not np.ndarray:
        raise PublicationInferenceError(reason=f"{field} must be a plain numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if array.dtype != _FLOAT64 or array.ndim != 1 or not array.flags.c_contiguous:
        raise PublicationInferenceError(reason=f"{field} must be a C-contiguous Float64 vector")
    if array.size == 0 or (length is not None and array.size != length):
        raise PublicationInferenceError(reason=f"{field} has an invalid length")
    if not np.isfinite(array).all():
        raise PublicationInferenceError(reason=f"{field} must be finite")


def _validate_float64_matrix(
    value: object,
    *,
    field: str,
    shape: tuple[int, int] | None = None,
) -> None:
    if type(value) is not np.ndarray:
        raise PublicationInferenceError(reason=f"{field} must be a plain numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if array.dtype != _FLOAT64 or array.ndim != _MATRIX_DIMENSIONS or not array.flags.c_contiguous:
        raise PublicationInferenceError(reason=f"{field} must be a C-contiguous Float64 matrix")
    if 0 in array.shape or (shape is not None and array.shape != shape):
        raise PublicationInferenceError(reason=f"{field} has an invalid shape")
    if not np.isfinite(array).all():
        raise PublicationInferenceError(reason=f"{field} must be finite")


def _validate_finite_summary(estimate: float, lower: float, upper: float) -> None:
    if any(
        type(value) is not float or not math.isfinite(value) for value in (estimate, lower, upper)
    ):
        raise PublicationInferenceError(reason="inference summary values must be finite floats")
    if lower > upper:
        raise PublicationInferenceError(reason="inference interval is reversed")


__all__ = [
    "PublicationBootstrapConfigurationResult",
    "PublicationCohortInferenceResult",
    "PublicationCohortPanelStatistics",
    "PublicationContrastInference",
    "PublicationEmpiricalInferenceResult",
    "PublicationInferenceError",
    "PublicationMetricInterval",
    "PublicationPredictionSet",
    "prepare_publication_inference_panels",
    "publication_additive_statistic_names",
    "publication_prediction_set_from_result",
    "run_publication_empirical_inference",
    "summarize_publication_empirical_inference",
]
