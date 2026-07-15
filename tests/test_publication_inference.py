"""Tests for paired complete-date-panel publication inference."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest
from finoptions import CpuExecutor

from discrete_dividend_study.publication_inference import (
    PublicationInferenceError,
    PublicationPredictionSet,
    prepare_publication_inference_panels,
    publication_additive_statistic_names,
    run_publication_empirical_inference,
)
from discrete_dividend_study.publication_inference_spec import (
    PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
    PublicationEmpiricalInferenceSpec,
    PublicationEmpiricalMetric,
)
from discrete_dividend_study.publication_spec import PublicationVariant

_ADDITIVE_STATISTICS = 25
_PRIMARY_TESTS = 12


def _inputs() -> tuple[pl.DataFrame, PublicationPredictionSet]:
    """Build three complete dates spanning the ETR common-horizon boundary."""
    tickers = ("DUK", "EPD", "ETR", "MO", "SO", "VZ")
    dates = (date(2019, 11, 6), date(2019, 11, 7), date(2020, 1, 2))
    rows = [(ticker, quote_date) for ticker in tickers for quote_date in dates]
    target = np.ascontiguousarray(np.linspace(10.0, 27.0, len(rows)), dtype=np.float64)
    predictions = (
        target.copy(),
        np.ascontiguousarray(target + 1.0),
        np.ascontiguousarray(target * 1.1),
        np.ascontiguousarray(target - 1.0),
    )
    for array in (target, *predictions):
        array.flags.writeable = False
    frame = pl.DataFrame(
        {
            "ticker": [row[0] for row in rows],
            "quote_date": [row[1] for row in rows],
            "call_mid_price": target,
        },
        schema={"ticker": pl.String, "quote_date": pl.Date, "call_mid_price": pl.Float64},
    )
    return frame, PublicationPredictionSet(target=target, predictions=predictions)


def _test_spec() -> PublicationEmpiricalInferenceSpec:
    """Retain the frozen estimands while making native test work intentionally small."""
    spec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC
    bootstrap = replace(
        spec.bootstrap,
        replications=999,
        sensitivity_mean_block_lengths=(20.0,),
    )
    return replace(spec, bootstrap=bootstrap)


def test_prepares_exact_available_case_and_sensitivity_panels() -> None:
    """Ticker exclusion and the inclusive ETR cutoff operate before whole-date aggregation."""
    frame, predictions = _inputs()
    panels = prepare_publication_inference_panels(frame, predictions, _test_spec())
    assert tuple((item.observations, item.quote_date_panels) for item in panels) == (
        (18, 3),
        (15, 3),
        (12, 2),
    )
    assert np.array_equal(panels[0].additive_statistics[:, 0], np.full(3, 6.0))
    assert np.array_equal(panels[1].additive_statistics[:, 0], np.full(3, 5.0))
    assert np.array_equal(panels[2].additive_statistics[:, 0], np.full(2, 6.0))
    assert panels[0].point_metrics[0].rmse == 0.0
    assert panels[0].point_metrics[1].mae == 1.0
    assert len(publication_additive_statistic_names()) == _ADDITIVE_STATISTICS


def test_paired_bootstrap_is_thread_invariant_and_holm_is_primary_only() -> None:
    """One Philox path per draw fixes raw sums while multiplicity stays in its declared family."""
    frame, predictions = _inputs()
    spec = _test_spec()
    serial = run_publication_empirical_inference(
        frame,
        predictions,
        CpuExecutor(threads=1),
        spec,
    )
    parallel = run_publication_empirical_inference(
        frame,
        predictions,
        CpuExecutor(threads=4),
        spec,
    )
    for left, right in zip(serial.cohorts, parallel.cohorts, strict=True):
        assert np.array_equal(left.bootstrap[0].additive_sums, right.bootstrap[0].additive_sums)
    primary = [
        result
        for result in serial.contrasts
        if result.cohort == spec.cohorts[0].name
        and result.metric in spec.primary_metrics
        and result.mean_block_length == spec.bootstrap.mean_block_length
    ]
    assert len(primary) == _PRIMARY_TESTS
    assert all(result.holm_adjusted_p_value is not None for result in primary)
    assert (
        sum(result.holm_adjusted_p_value is not None for result in serial.contrasts)
        == _PRIMARY_TESTS
    )
    exact_mape = next(
        result
        for result in serial.model_intervals
        if result.cohort == spec.cohorts[0].name
        and result.variant is PublicationVariant.CASH_JUMP_LIQUIDATOR
        and result.metric is PublicationEmpiricalMetric.MAPE_PERCENT
    )
    assert exact_mape.estimate == exact_mape.interval_lower == exact_mape.interval_upper == 0.0


def test_rejects_mutable_predictions_and_a_different_result_target() -> None:
    """Inference never repairs result identity, vector ownership, or row alignment."""
    frame, predictions = _inputs()
    mutable = predictions.target.copy()
    with pytest.raises(PublicationInferenceError, match="immutable"):
        PublicationPredictionSet(target=mutable, predictions=predictions.predictions)
    different = predictions.target.copy()
    different[0] += 1.0
    different.flags.writeable = False
    mismatched = PublicationPredictionSet(target=different, predictions=predictions.predictions)
    with pytest.raises(PublicationInferenceError, match="frame target differs"):
        prepare_publication_inference_panels(frame, mismatched, _test_spec())
