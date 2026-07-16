"""Tests for the closed receipt-verified study feature boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from discrete_dividend_study.artifacts.experiment_features import (
    ProjectExperimentFeatureSources,
    materialize_project_experiment_artifact,
)
from discrete_dividend_study.artifacts.project_features import materialize_project_feature_artifact
from discrete_dividend_study.artifacts.study_features import (
    StudyFeatureArtifactError,
    StudyFeatureArtifactKind,
    load_verified_study_feature_artifact,
    verify_study_feature_artifact_unchanged,
)
from discrete_dividend_study.data.pipeline import (
    project_experiment_feature_schema_sha256,
    project_feature_schema_sha256,
)
from discrete_dividend_study.experiments.neural.run import resolve_study_source_identity
from tests.artifacts.test_project_features import _long_form_options, _sources

if TYPE_CHECKING:
    from pathlib import Path


def test_explicit_kinds_preserve_true_schema_identity_without_fallback(tmp_path: Path) -> None:
    """Both known receipts load explicitly and the wrong producer contract fails closed."""
    sources = _sources(tmp_path)
    full = materialize_project_feature_artifact(sources, tmp_path / "full")
    experiment_sources = ProjectExperimentFeatureSources(
        long_form_options=_long_form_options(tmp_path, sources.options),
        compacted_options=sources.options,
        forecast_dividends=sources.forecast_dividends,
        historical_dividends=sources.historical_dividends,
        underlying_history=sources.underlying_history,
        nss_parameters=sources.nss_parameters,
    )
    experiment = materialize_project_experiment_artifact(
        experiment_sources,
        tmp_path / "experiment",
    )

    verified_full = load_verified_study_feature_artifact(
        full.path,
        StudyFeatureArtifactKind.PROJECT_FULL,
    )
    verified_experiment = load_verified_study_feature_artifact(
        experiment.path,
        StudyFeatureArtifactKind.PROJECT_EXPERIMENT,
    )
    assert verified_full.feature_schema_sha256 == project_feature_schema_sha256()
    assert verified_experiment.feature_schema_sha256 == project_experiment_feature_schema_sha256()
    assert verified_full.feature_schema_sha256 != verified_experiment.feature_schema_sha256
    assert verified_experiment.output_sha256 == experiment.manifest.output_sha256
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"locked")
    identity = resolve_study_source_identity(
        verified_experiment,
        lock,
        "a" * 40,
        "b" * 40,
    )
    assert identity.feature_artifact_sha256 == experiment.manifest.output_sha256
    assert identity.feature_schema_sha256 == project_experiment_feature_schema_sha256()
    verify_study_feature_artifact_unchanged(verified_full)
    verify_study_feature_artifact_unchanged(verified_experiment)

    with pytest.raises(StudyFeatureArtifactError):
        load_verified_study_feature_artifact(
            experiment.path,
            StudyFeatureArtifactKind.PROJECT_FULL,
        )
    with pytest.raises(StudyFeatureArtifactError, match="StudyFeatureArtifactKind"):
        load_verified_study_feature_artifact(
            experiment.path,
            cast("Any", StudyFeatureArtifactKind.PROJECT_EXPERIMENT.value),
        )

    experiment.receipt_path.write_bytes(experiment.receipt_path.read_bytes() + b" ")
    with pytest.raises(StudyFeatureArtifactError, match="receipt byte count changed"):
        verify_study_feature_artifact_unchanged(verified_experiment)
