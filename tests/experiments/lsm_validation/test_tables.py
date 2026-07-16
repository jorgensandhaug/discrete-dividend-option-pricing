"""Tests for aggregate-only empirical-inference paper tables."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
from finoptions import CpuExecutor

from discrete_dividend_study.experiments.lsm_validation.inference import (
    run_publication_empirical_inference,
)
from discrete_dividend_study.experiments.lsm_validation.specification import PublicationVariant
from discrete_dividend_study.experiments.lsm_validation.tables import (
    PublicationInferenceTables,
    publication_calendar_year_contrast_frame,
    publication_calendar_year_model_frame,
    publication_primary_contrast_frame,
    publication_primary_contrast_latex,
    publication_primary_contrast_markdown,
    publication_primary_model_frame,
    publication_primary_model_latex,
    publication_primary_model_markdown,
    publication_sensitivity_contrast_frame,
    publication_sensitivity_model_frame,
)
from tests.experiments.lsm_validation.test_inference import _inputs, _test_spec

_PRIMARY_CONTRASTS = 12
_SENSITIVITY_CONTRASTS = 216
_SENSITIVITY_MODELS = 144
_COHORTS = 3
_CALENDAR_YEAR_COHORTS = 5
_METRICS = 6


def _tables() -> PublicationInferenceTables:
    frame, predictions = _inputs()
    result = run_publication_empirical_inference(
        frame,
        predictions,
        CpuExecutor(threads=1),
        _test_spec(),
    )
    return PublicationInferenceTables(artifact_sha256="a" * 64, result=result)


def test_primary_frames_are_complete_typed_and_aggregate_only() -> None:
    """Four model rows and twelve primary contrasts expose no quote-level state."""
    tables = _tables()
    models = publication_primary_model_frame(tables)
    contrasts = publication_primary_contrast_frame(tables)
    sensitivity = publication_sensitivity_contrast_frame(tables)
    sensitivity_models = publication_sensitivity_model_frame(tables)
    calendar_models = publication_calendar_year_model_frame(tables)
    calendar_contrasts = publication_calendar_year_contrast_frame(tables)
    assert models.height == len(PublicationVariant)
    assert models.schema == {
        "variant": pl.String,
        "variant_label": pl.String,
        "observations": pl.UInt64,
        "quote_date_panels": pl.UInt64,
        "mape_percent": pl.Float64,
        "mape_interval_lower": pl.Float64,
        "mape_interval_upper": pl.Float64,
        "rmse": pl.Float64,
        "rmse_interval_lower": pl.Float64,
        "rmse_interval_upper": pl.Float64,
    }
    assert contrasts.height == _PRIMARY_CONTRASTS
    assert contrasts.get_column("holm_adjusted_p_value").null_count() == 0
    assert sensitivity.height == _SENSITIVITY_CONTRASTS
    assert sensitivity_models.height == _SENSITIVITY_MODELS
    assert (
        sensitivity_models.select("cohort", "observations", "quote_date_panels").unique().height
        == _COHORTS
    )
    assert sensitivity.get_column("holm_adjusted_p_value").null_count() == (
        _SENSITIVITY_CONTRASTS - _PRIMARY_CONTRASTS
    )
    assert calendar_models.height == _CALENDAR_YEAR_COHORTS * len(PublicationVariant) * _METRICS
    assert calendar_contrasts.height == _CALENDAR_YEAR_COHORTS * 6 * _METRICS
    assert calendar_models.select("cohort", "year").unique().height == _CALENDAR_YEAR_COHORTS
    exact = calendar_models.filter(
        pl.col("variant") == PublicationVariant.CASH_JUMP_LIQUIDATOR.value
    )
    assert exact.get_column("estimate").max() == 0.0


def test_renderers_are_deterministic_provenance_bound_fragments() -> None:
    """Markdown and booktabs output retain artifact identity but no local paths."""
    tables = _tables()
    renderings = (
        publication_primary_model_markdown(tables),
        publication_primary_model_latex(tables),
        publication_primary_contrast_markdown(tables),
        publication_primary_contrast_latex(tables),
    )
    for rendered in renderings:
        assert f"source_artifact_sha256={tables.artifact_sha256}" in rendered
        assert "/Users/" not in rendered
    assert "Cash jump (liquidator)" in renderings[0]
    assert "Holm p" in renderings[2]
    assert renderings[1].endswith("\\bottomrule\n\\end{tabular}\n")
    assert renderings[3].endswith("\\bottomrule\n\\end{tabular}\n")


def test_renderers_derive_the_confidence_label_from_the_verified_specification() -> None:
    """A nondefault valid confidence level cannot be rendered under a hard-coded 95% label."""
    frame, predictions = _inputs()
    base = _test_spec()
    spec = replace(
        base,
        bootstrap=replace(base.bootstrap, confidence_level=0.90),
        family_wise_error_rate=0.10,
    )
    result = run_publication_empirical_inference(
        frame,
        predictions,
        CpuExecutor(threads=1),
        spec,
    )
    tables = PublicationInferenceTables(artifact_sha256="b" * 64, result=result)
    assert "[90% CI]" in publication_primary_model_markdown(tables)
    assert "[90\\% CI]" in publication_primary_contrast_latex(tables)
