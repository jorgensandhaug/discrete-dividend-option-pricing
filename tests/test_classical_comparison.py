"""Tests for aggregate-only comparison with the submitted project table."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from finoptions import CpuMathBackend

from discrete_dividend_study.classical_artifact import (
    load_classical_study_artifact,
    run_and_write_project_classical_study,
)
from discrete_dividend_study.classical_comparison import (
    PROJECT_SUBMITTED_CLASSICAL_REFERENCES,
    ClassicalPublicationComparison,
    build_classical_publication_comparison,
    classical_publication_frame,
)
from discrete_dividend_study.classical_inputs import prepare_project_classical_inputs
from discrete_dividend_study.classical_spec import ClassicalExecutionSpec, ClassicalVariant
from discrete_dividend_study.neural_run import StudySourceIdentity
from tests.test_classical_inputs import classical_frame

if TYPE_CHECKING:
    from pathlib import Path

_SOURCE = StudySourceIdentity(
    feature_artifact_sha256="1" * 64,
    feature_schema_sha256="2" * 64,
    paper_revision="a" * 40,
    engine_revision="b" * 40,
    dependency_lock_sha256="3" * 64,
)


def _comparison(tmp_path: Path) -> ClassicalPublicationComparison:
    inputs = prepare_project_classical_inputs(classical_frame())
    execution = ClassicalExecutionSpec(
        threads=2,
        math_backend=CpuMathBackend.RUST_STD_F64,
        training_seed=7,
        training_domain_start=100,
        evaluation_seed=11,
        evaluation_domain_start=10_000,
    )
    artifact = run_and_write_project_classical_study(tmp_path, _SOURCE, inputs, execution)
    return build_classical_publication_comparison(load_classical_study_artifact(artifact.path))


def test_submitted_reference_table_is_complete_and_exactly_transcribed() -> None:
    """References retain the four reported names and aggregate metrics without a tolerance rule."""
    assert tuple(
        reference.variant for reference in PROJECT_SUBMITTED_CLASSICAL_REFERENCES
    ) == tuple(ClassicalVariant)
    assert tuple(
        (
            reference.submitted_result_name,
            reference.mape_percent,
            reference.rmse,
        )
        for reference in PROJECT_SUBMITTED_CLASSICAL_REFERENCES
    ) == (
        ("IT_GBM_SPOT", 4.695443, 0.651005),
        ("IT_GBMT_FD_AH", 4.750231, 0.666347),
        ("IT_GBMT_FY", 4.924086, 0.659504),
        ("IT_GBMT_HY", 5.253302, 0.665882),
    )


def test_publication_frame_reports_differences_and_independent_ranks(tmp_path: Path) -> None:
    """The aggregate table describes current and submitted results without asserting equality."""
    comparison = _comparison(tmp_path)
    frame = classical_publication_frame(comparison)
    assert comparison.artifact_sha256 == next(tmp_path.glob("*.safetensors")).stem
    assert frame.schema == {
        "variant": pl.String,
        "submitted_result_name": pl.String,
        "observations": pl.UInt64,
        "current_mape_percent": pl.Float64,
        "submitted_mape_percent": pl.Float64,
        "mape_difference_percentage_points": pl.Float64,
        "current_rmse": pl.Float64,
        "submitted_rmse": pl.Float64,
        "rmse_difference": pl.Float64,
        "current_mape_rank": pl.UInt32,
        "submitted_mape_rank": pl.UInt32,
        "current_rmse_rank": pl.UInt32,
        "submitted_rmse_rank": pl.UInt32,
    }
    assert frame.height == len(ClassicalVariant)
    assert frame.get_column("submitted_mape_rank").to_list() == [1, 2, 3, 4]
    assert frame.get_column("submitted_rmse_rank").to_list() == [1, 4, 2, 3]
    assert frame.select(
        (
            pl.col("current_mape_percent") - pl.col("submitted_mape_percent")
            == pl.col("mape_difference_percentage_points")
        ).all(),
        (pl.col("current_rmse") - pl.col("submitted_rmse") == pl.col("rmse_difference")).all(),
    ).row(0) == (True, True)
