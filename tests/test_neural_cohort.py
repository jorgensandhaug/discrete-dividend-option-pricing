"""Exact aggregate-only admission artifacts for the project neural cohort."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl
import pytest
from finoptions import CpuExecutor

from discrete_dividend_study.neural_cohort import (
    NeuralCohortError,
    build_project_neural_cohort_manifest,
    load_neural_cohort_artifact,
    write_neural_cohort_artifact,
)
from discrete_dividend_study.neural_run import StudySourceIdentity
from discrete_dividend_study.neural_spec import PROJECT_STUDY
from discrete_dividend_study.project_pipeline import PROJECT_AS_SUBMITTED_TICKERS
from discrete_dividend_study.rolling_windows import (
    PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC,
    generate_rolling_windows,
)
from discrete_dividend_study.study_features import (
    StudyFeatureArtifactKind,
    VerifiedStudyFeatureArtifact,
)
from tests.test_model_inputs import model_frame

if TYPE_CHECKING:
    from pathlib import Path

_WINDOW_COUNT = 117


def test_complete_cohort_freezes_all_tickers_windows_and_model_tensors(tmp_path: Path) -> None:
    """The detached manifest identifies every prepared input without licensed rows."""
    feature = _feature_artifact(tmp_path)
    source = _source(feature)
    manifest = build_project_neural_cohort_manifest(feature, source, CpuExecutor(threads=2))

    assert len(manifest.windows) == _WINDOW_COUNT
    assert tuple(item.ticker for item in manifest.tickers) == PROJECT_AS_SUBMITTED_TICKERS
    assert tuple(item.variant for item in manifest.models) == tuple(
        model.variant.value for model in PROJECT_STUDY.models
    )
    assert all(item.rows == manifest.prepared_horizon_rows for item in manifest.models)
    assert sum(item.test_rows for item in manifest.windows) == manifest.test_union_rows
    assert sum(item.test_rows for item in manifest.tickers) == manifest.test_union_rows

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    first = write_neural_cohort_artifact(artifact_root, manifest)
    second = write_neural_cohort_artifact(artifact_root, manifest)
    assert first == second
    assert load_neural_cohort_artifact(first.path) == first
    assert set(first.path.parent.iterdir()) == {first.path}


def test_cross_source_identity_and_tampered_artifact_fail_early(tmp_path: Path) -> None:
    """Neither a rebound source nor changed manifest bytes can enter a study."""
    feature = _feature_artifact(tmp_path)
    with pytest.raises(NeuralCohortError, match="output hashes disagree"):
        build_project_neural_cohort_manifest(
            feature,
            StudySourceIdentity(
                feature_artifact_sha256="a" * 64,
                feature_schema_sha256=feature.feature_schema_sha256,
                paper_revision="b" * 40,
                engine_revision="c" * 40,
                dependency_lock_sha256="d" * 64,
            ),
            CpuExecutor(threads=1),
        )

    manifest = build_project_neural_cohort_manifest(
        feature, _source(feature), CpuExecutor(threads=1)
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact = write_neural_cohort_artifact(artifact_root, manifest)
    artifact.path.write_bytes(artifact.path.read_bytes() + b" ")
    with pytest.raises(NeuralCohortError, match="filename hash"):
        load_neural_cohort_artifact(artifact.path)


def _feature_artifact(root: Path) -> VerifiedStudyFeatureArtifact:
    windows = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    dates = sorted(
        {
            boundary
            for window in windows
            for boundary in (window.train_start, window.validation_start, window.test_start)
        }
    )
    keys = pl.DataFrame(
        {
            "ticker": [ticker for ticker in PROJECT_AS_SUBMITTED_TICKERS for _ in dates],
            "quote_date": dates * len(PROJECT_AS_SUBMITTED_TICKERS),
        },
        schema={"ticker": pl.String, "quote_date": pl.Date},
    )
    frame = keys.join(model_frame().head(1), how="cross")
    path = root / "features.parquet"
    receipt = root / "features.parquet.receipt.json"
    frame.write_parquet(path)
    receipt.write_bytes(b'{"fixture":"verified-neural-cohort"}\n')
    return VerifiedStudyFeatureArtifact(
        kind=StudyFeatureArtifactKind.PROJECT_EXPERIMENT,
        path=path,
        receipt_path=receipt,
        request_sha256="1" * 64,
        pipeline_sha256="2" * 64,
        feature_schema_sha256="3" * 64,
        output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        output_rows=frame.height,
        output_bytes=path.stat().st_size,
        receipt_bytes=receipt.stat().st_size,
    )


def _source(feature: VerifiedStudyFeatureArtifact) -> StudySourceIdentity:
    return StudySourceIdentity(
        feature_artifact_sha256=feature.output_sha256,
        feature_schema_sha256=feature.feature_schema_sha256,
        paper_revision="a" * 40,
        engine_revision="b" * 40,
        dependency_lock_sha256="c" * 64,
    )
