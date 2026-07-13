"""Tests for submitted project pricing-error formulas and pooling."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import numpy.typing as npt
import pytest

from discrete_dividend_study.project_metrics import (
    ProjectMetricError,
    aggregate_project_pricing_metrics,
    evaluate_project_pricing,
)

PERCENT_SCALE = 100.0


def test_exact_submitted_formulas_use_prediction_minus_market_bias() -> None:
    """MAPE, RMSE, MAE, bias, relative bias, and sMAPE retain their thesis meanings."""
    target = np.array([2.0, 4.0], dtype=np.float32)
    prediction = np.array([1.0, 6.0], dtype=np.float32)
    observed = evaluate_project_pricing(target, prediction)
    error = prediction.astype(np.float64) - target.astype(np.float64)
    assert observed.observations == target.size
    assert observed.mae == pytest.approx(float(np.mean(np.abs(error))))
    assert observed.mape_percent == pytest.approx(
        float(np.mean(np.abs(error) / target)) * PERCENT_SCALE,
    )
    assert observed.mse == pytest.approx(float(np.mean(error**2)))
    assert observed.rmse == pytest.approx(math.sqrt(float(np.mean(error**2))))
    assert observed.bias == pytest.approx(float(np.mean(error)))
    assert observed.relative_bias == pytest.approx(float(np.mean(error / target)))
    assert observed.smape_percent == pytest.approx(
        float(np.mean(np.abs(error) / ((target + prediction) / 2.0))) * PERCENT_SCALE,
    )


def test_partition_pooling_matches_direct_row_level_evaluation() -> None:
    """Window metrics are observation-weighted and RMSE is never averaged across windows."""
    target = np.array([2.0, 4.0, 5.0, 10.0, 20.0], dtype=np.float32)
    prediction = np.array([1.0, 6.0, 4.0, 13.0, 18.0], dtype=np.float32)
    pooled = aggregate_project_pricing_metrics(
        (
            evaluate_project_pricing(target[:2].copy(), prediction[:2].copy()),
            evaluate_project_pricing(target[2:].copy(), prediction[2:].copy()),
        ),
    )
    direct = evaluate_project_pricing(target, prediction)
    assert asdict(pooled) == pytest.approx(asdict(direct))
    assert pooled.rmse != pytest.approx(
        (
            evaluate_project_pricing(target[:2].copy(), prediction[:2].copy()).rmse
            + evaluate_project_pricing(target[2:].copy(), prediction[2:].copy()).rmse
        )
        / 2.0,
    )


@pytest.mark.parametrize(
    ("target", "prediction", "message"),
    [
        (np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32), "positive"),
        (np.array([1.0], dtype=np.float32), np.array([-0.1], dtype=np.float32), "nonnegative"),
        (np.array([1.0], dtype=np.float64), np.array([1.0], dtype=np.float32), "float32"),
        (np.array([1.0], dtype=np.float32), np.array([np.nan], dtype=np.float32), "finite"),
    ],
)
def test_invalid_financial_or_numeric_domains_fail_without_clipping(
    target: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    message: str,
) -> None:
    """Invalid market prices, model prices, precision, and values are hard errors."""
    with pytest.raises(ProjectMetricError, match=message):
        evaluate_project_pricing(target, prediction)  # ty: ignore[invalid-argument-type]
