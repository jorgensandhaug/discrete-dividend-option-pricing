"""Tests for paired complete-date-panel publication inference."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest
from finoptions import CpuExecutor

from discrete_dividend_study.experiments.lsm_validation.inference import (
    PublicationInferenceError,
    PublicationPredictionSet,
    prepare_publication_inference_panels,
    publication_additive_statistic_names,
    run_publication_empirical_inference,
)
from discrete_dividend_study.experiments.lsm_validation.inference_specification import (
    PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
    PublicationBootstrapDesign,
    PublicationEmpiricalInferenceSpec,
    PublicationEmpiricalMetric,
)
from discrete_dividend_study.experiments.lsm_validation.specification import PublicationVariant

_ADDITIVE_STATISTICS = 25
_PRIMARY_TESTS = 12
_TEST_BOOTSTRAP_CONFIGURATIONS = 2


def _inputs() -> tuple[pl.DataFrame, PublicationPredictionSet]:
    """Build unequal complete-date panels spanning the ETR common-horizon boundary."""
    tickers = ("DUK", "EPD", "ETR", "MO", "SO", "VZ")
    dates = (date(2019, 11, 6), date(2019, 11, 7), date(2020, 1, 2))
    admitted_dates = {
        "DUK": dates,
        "EPD": dates,
        "ETR": dates[:2],
        "MO": dates,
        "SO": (dates[0], dates[2]),
        "VZ": dates[:1],
    }
    rows = [(ticker, quote_date) for ticker in tickers for quote_date in admitted_dates[ticker]]
    source_target = np.linspace(10.0, 27.0, len(rows), dtype=np.float32)
    target = np.ascontiguousarray(source_target, dtype=np.float64)
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
            "call_mid_price": source_target,
        },
        schema={"ticker": pl.String, "quote_date": pl.Date, "call_mid_price": pl.Float32},
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
        (14, 3),
        (12, 3),
        (10, 2),
    )
    assert np.array_equal(panels[0].additive_statistics[:, 0], np.array([6.0, 4.0, 4.0]))
    assert np.array_equal(panels[1].additive_statistics[:, 0], np.array([5.0, 3.0, 4.0]))
    assert np.array_equal(panels[2].additive_statistics[:, 0], np.array([6.0, 4.0]))
    assert tuple(panel.panel_years.tolist() for panel in panels) == (
        [2019, 2019, 2020],
        [2019, 2019, 2020],
        [2019, 2019],
    )
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
        assert len(left.bootstrap) == len(right.bootstrap) == _TEST_BOOTSTRAP_CONFIGURATIONS
        for left_configuration, right_configuration in zip(
            left.bootstrap,
            right.bootstrap,
            strict=True,
        ):
            assert np.array_equal(
                left_configuration.additive_sums,
                right_configuration.additive_sums,
            )
        assert left.bootstrap[1].design is (
            PublicationBootstrapDesign.CALENDAR_YEAR_STRATIFIED_CIRCULAR_STATIONARY
        )
    assert set(serial.cohorts[0].bootstrap[1].additive_sums[:, 0]).issubset({12.0, 14.0, 16.0})
    assert np.unique(serial.cohorts[0].bootstrap[0].additive_sums[:, 0]).size > 1
    primary = [
        result
        for result in serial.contrasts
        if result.cohort == spec.cohorts[0].name
        and result.bootstrap_design is PublicationBootstrapDesign.CIRCULAR_STATIONARY
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
        and result.bootstrap_design is PublicationBootstrapDesign.CIRCULAR_STATIONARY
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
