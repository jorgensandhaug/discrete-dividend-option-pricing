"""Exact pooled pricing-error metrics used by the submitted project analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt

Float32Vector = npt.NDArray[np.float32]

_FLOAT32_DTYPE: Final = np.dtype(np.float32)
_METRIC_CHUNK_ROWS: Final = 1 << 20
_PERCENT_SCALE: Final = 100.0


@dataclass(frozen=True, slots=True)
class ProjectMetricError(ValueError):
    """Targets, predictions, or pooled metric state violate the study contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected metric contract."""
        return f"invalid project pricing metrics: {self.reason}"


@dataclass(frozen=True, slots=True)
class ProjectPricingMetrics:
    """Pooled row-weighted project metrics without per-window averaging.

    ``mape_percent`` and ``smape_percent`` are percentage values. ``bias`` is prediction minus
    market price in price units, while ``relative_bias`` is the unscaled signed relative error.
    ``mse`` is retained so independently evaluated windows can be pooled before taking ``rmse``.
    """

    observations: int
    mae: float
    mape_percent: float
    mse: float
    rmse: float
    bias: float
    relative_bias: float
    smape_percent: float

    def __post_init__(self) -> None:
        """Reject incomplete, nonfinite, or algebraically inconsistent metric state."""
        if type(self.observations) is not int or self.observations <= 0:
            raise ProjectMetricError(reason="observations must be a positive plain int")
        for field, value in (
            ("mae", self.mae),
            ("mape_percent", self.mape_percent),
            ("mse", self.mse),
            ("rmse", self.rmse),
            ("smape_percent", self.smape_percent),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ProjectMetricError(reason=f"{field} must be finite and nonnegative")
        for field, value in (("bias", self.bias), ("relative_bias", self.relative_bias)):
            if type(value) is not float or not math.isfinite(value):
                raise ProjectMetricError(reason=f"{field} must be a finite plain float")
        if not math.isclose(self.rmse, math.sqrt(self.mse), rel_tol=1e-15, abs_tol=0.0):
            raise ProjectMetricError(reason="rmse must equal sqrt(mse)")


def evaluate_project_pricing(
    target: Float32Vector,
    prediction: Float32Vector,
) -> ProjectPricingMetrics:
    """Evaluate the exact submitted formulas with bounded float64 accumulation.

    Targets must be strictly positive, as guaranteed by the project cohort. Predictions must be
    nonnegative. Invalid model output fails instead of reproducing the legacy notebook's implicit
    clipping operation.
    """
    _validate_pricing_vectors(target, prediction)
    rows = int(target.size)
    absolute_error_sum = 0.0
    absolute_percentage_error_sum = 0.0
    squared_error_sum = 0.0
    signed_error_sum = 0.0
    relative_signed_error_sum = 0.0
    symmetric_absolute_percentage_error_sum = 0.0

    for start in range(0, rows, _METRIC_CHUNK_ROWS):
        end = min(start + _METRIC_CHUNK_ROWS, rows)
        observed = target[start:end]
        fitted = prediction[start:end]
        error = np.subtract(fitted, observed, dtype=np.float64)
        absolute_error = np.abs(error)
        absolute_error_sum += float(np.sum(absolute_error, dtype=np.float64))
        squared_error_sum += float(np.sum(error * error, dtype=np.float64))
        signed_error_sum += float(np.sum(error, dtype=np.float64))
        absolute_percentage_error_sum += float(
            np.sum(absolute_error / observed, dtype=np.float64),
        )
        relative_signed_error_sum += float(np.sum(error / observed, dtype=np.float64))
        symmetric_denominator = np.add(observed, fitted, dtype=np.float64) / 2.0
        symmetric_absolute_percentage_error_sum += float(
            np.sum(absolute_error / symmetric_denominator, dtype=np.float64),
        )

    divisor = float(rows)
    mse = squared_error_sum / divisor
    return ProjectPricingMetrics(
        observations=rows,
        mae=absolute_error_sum / divisor,
        mape_percent=(absolute_percentage_error_sum / divisor) * _PERCENT_SCALE,
        mse=mse,
        rmse=math.sqrt(mse),
        bias=signed_error_sum / divisor,
        relative_bias=relative_signed_error_sum / divisor,
        smape_percent=(symmetric_absolute_percentage_error_sum / divisor) * _PERCENT_SCALE,
    )


def aggregate_project_pricing_metrics(
    metrics: tuple[ProjectPricingMetrics, ...],
) -> ProjectPricingMetrics:
    """Pool disjoint partitions by observation count, taking RMSE only after pooled MSE."""
    if (
        type(metrics) is not tuple
        or not metrics
        or any(type(metric) is not ProjectPricingMetrics for metric in metrics)
    ):
        raise ProjectMetricError(
            reason="metrics must be a nonempty tuple[ProjectPricingMetrics, ...]",
        )
    observations = sum(metric.observations for metric in metrics)

    def pooled(field: str) -> float:
        return (
            math.fsum(
                cast("float", getattr(metric, field)) * metric.observations for metric in metrics
            )
            / observations
        )

    mse = pooled("mse")
    return ProjectPricingMetrics(
        observations=observations,
        mae=pooled("mae"),
        mape_percent=pooled("mape_percent"),
        mse=mse,
        rmse=math.sqrt(mse),
        bias=pooled("bias"),
        relative_bias=pooled("relative_bias"),
        smape_percent=pooled("smape_percent"),
    )


def _validate_pricing_vectors(target: object, prediction: object) -> None:
    for field, value in (("target", target), ("prediction", prediction)):
        if type(value) is not np.ndarray:
            raise ProjectMetricError(reason=f"{field} must be a plain numpy.ndarray")
        array = cast("npt.NDArray[np.generic]", value)
        if array.dtype != _FLOAT32_DTYPE or array.ndim != 1 or not array.flags.c_contiguous:
            raise ProjectMetricError(reason=f"{field} must be a C-contiguous float32 vector")
    target_array = cast("Float32Vector", target)
    prediction_array = cast("Float32Vector", prediction)
    if target_array.size == 0 or prediction_array.shape != target_array.shape:
        raise ProjectMetricError(reason="target and prediction must share one nonempty shape")
    if not np.isfinite(target_array).all() or not np.isfinite(prediction_array).all():
        raise ProjectMetricError(reason="target and prediction must be finite")
    if np.any(target_array <= 0.0):
        raise ProjectMetricError(reason="project market-price targets must be strictly positive")
    if np.any(prediction_array < 0.0):
        raise ProjectMetricError(reason="project price predictions must be nonnegative")


__all__ = [
    "ProjectMetricError",
    "ProjectPricingMetrics",
    "aggregate_project_pricing_metrics",
    "evaluate_project_pricing",
]
