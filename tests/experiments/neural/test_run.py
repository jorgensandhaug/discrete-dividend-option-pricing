"""End-to-end tests for immutable one-window neural experiment orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from discrete_dividend_study.artifacts.study_features import (
    StudyFeatureArtifactKind,
    VerifiedStudyFeatureArtifact,
)
from discrete_dividend_study.experiments.neural.run import (
    NeuralRunError,
    NeuralWindowRunRequest,
    StudySourceIdentity,
    load_neural_window_run,
    resolve_study_source_identity,
    run_neural_window,
)
from discrete_dividend_study.experiments.neural.window import PreparedNeuralWindow
from discrete_dividend_study.experiments.windows import RollingWindow
from discrete_dividend_study.models.neural.execution import PROJECT_PORTABLE_CPU_EXECUTION_PROFILE
from discrete_dividend_study.models.neural.predictions import NeuralPredictionError
from discrete_dividend_study.models.neural.specification import PROJECT_STUDY, NeuralVariant
from tests.models.neural.test_backends import _prepared_batch as prepared_batch

if TYPE_CHECKING:
    from pathlib import Path

    from discrete_dividend_study.models.neural.inputs import PreparedNeuralBatch

PARAMETER_SEED = 20261209
SHUFFLE_SEED = 20261210
WINDOW = RollingWindow(
    window_id=17,
    train_start=date(2022, 1, 1),
    validation_start=date(2022, 11, 1),
    test_start=date(2022, 12, 1),
    test_end=date(2023, 1, 1),
)
SOURCE = StudySourceIdentity(
    feature_artifact_sha256="a" * 64,
    feature_schema_sha256="b" * 64,
    paper_revision="c" * 40,
    engine_revision="d" * 40,
    dependency_lock_sha256="e" * 64,
)


def test_complete_run_is_deterministic_relocatable_and_cross_referenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One request produces one safe result and resumes without invoking training again."""
    request = _run_request()
    first = run_neural_window(request, tmp_path)

    def unexpected_training(*_args: object, **_kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr(
        "discrete_dividend_study.experiments.neural.run._train_and_predict",
        unexpected_training,
    )
    second = run_neural_window(request, tmp_path)
    assert first == second
    assert tuple((tmp_path / "checkpoints").iterdir()) == (first.checkpoint.path,)
    assert tuple((tmp_path / "predictions").iterdir()) == (first.prediction.path,)
    assert tuple((tmp_path / "runs").iterdir()) == (first.run.path,)
    assert tuple((tmp_path / "receipts").iterdir()) == (first.receipt.path,)

    loaded = load_neural_window_run(first.run.path)
    assert loaded == first.run
    manifest = loaded.manifest
    assert manifest.source == SOURCE
    assert manifest.window.window_id == WINDOW.window_id
    assert manifest.phase_rows.training == request.prepared.training.rows
    assert manifest.phase_rows.validation == request.prepared.validation.rows
    assert manifest.phase_rows.test == request.prepared.test.rows
    assert manifest.checkpoint_sha256 == first.checkpoint.sha256
    assert manifest.prediction_artifact_sha256 == first.prediction.sha256
    assert manifest.metrics == first.prediction.metrics
    assert manifest.execution_profile == PROJECT_PORTABLE_CPU_EXECUTION_PROFILE.name
    assert manifest.inference_batch_size == request.execution.inference_batch_size
    assert manifest.request_sha256 == first.receipt.receipt.request_sha256
    assert first.receipt.receipt.run_sha256 == first.run.sha256


def test_bad_source_revision_and_tampered_run_fail_before_reuse(tmp_path: Path) -> None:
    """Abbreviated revisions and modified manifest bytes cannot identify a production run."""
    with pytest.raises(NeuralRunError, match="full Git commit"):
        replace(SOURCE, paper_revision="abc123")

    result = run_neural_window(_run_request(), tmp_path)
    result.run.path.write_bytes(result.run.path.read_bytes() + b"tampered")
    with pytest.raises(NeuralRunError, match="filename hash"):
        load_neural_window_run(result.run.path)


def test_present_receipt_with_missing_result_fails_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken exact resume chain is an error rather than permission to create another result."""
    request = _run_request()
    result = run_neural_window(request, tmp_path)
    result.prediction.path.unlink()

    def unexpected_training(*_args: object, **_kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr(
        "discrete_dividend_study.experiments.neural.run._train_and_predict",
        unexpected_training,
    )
    with pytest.raises(NeuralPredictionError, match=r"regular \.safetensors"):
        run_neural_window(request, tmp_path)


def test_source_identity_uses_verified_artifact_receipt_and_explicit_lock(tmp_path: Path) -> None:
    """The verified output and schema identities bind the run without serializing local paths."""
    feature = tmp_path / "features.parquet"
    receipt = tmp_path / "features.parquet.receipt.json"
    lock = tmp_path / "uv.lock"
    feature.write_bytes(b"licensed-feature-fixture")
    receipt.write_bytes(b"verified-receipt-fixture")
    lock.write_bytes(b"locked-dependencies")
    artifact = VerifiedStudyFeatureArtifact(
        kind=StudyFeatureArtifactKind.PROJECT_EXPERIMENT,
        path=feature,
        receipt_path=receipt,
        request_sha256="1" * 64,
        pipeline_sha256="2" * 64,
        feature_schema_sha256="3" * 64,
        output_sha256=hashlib.sha256(feature.read_bytes()).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        output_rows=1,
        output_bytes=feature.stat().st_size,
        receipt_bytes=receipt.stat().st_size,
    )
    observed = resolve_study_source_identity(
        artifact,
        lock,
        SOURCE.paper_revision,
        SOURCE.engine_revision,
    )
    assert observed.feature_artifact_sha256 == artifact.output_sha256
    assert observed.feature_schema_sha256 == artifact.feature_schema_sha256
    assert observed.dependency_lock_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()
    with pytest.raises(NeuralRunError, match="VerifiedStudyFeatureArtifact"):
        resolve_study_source_identity(
            cast("Any", feature),
            lock,
            SOURCE.paper_revision,
            SOURCE.engine_revision,
        )


def _run_request() -> NeuralWindowRunRequest:
    base = prepared_batch(NeuralVariant.FORECAST_YIELD)
    training = _with_target_offset(base, 0.0)
    validation = _with_target_offset(base, 0.25)
    test = _with_target_offset(base, 0.5)
    prepared = PreparedNeuralWindow(
        window=WINDOW,
        training=training,
        validation=validation,
        test=test,
    )
    training_spec = replace(
        PROJECT_STUDY.training,
        batch_size=5,
        maximum_epochs=1,
        early_stopping_patience_epochs=1,
    )
    return NeuralWindowRunRequest(
        source=SOURCE,
        prepared=prepared,
        study=replace(PROJECT_STUDY, training=training_spec),
        execution_profile=PROJECT_PORTABLE_CPU_EXECUTION_PROFILE,
        parameter_seed=PARAMETER_SEED,
        shuffle_seed=SHUFFLE_SEED,
    )


def _with_target_offset(
    batch: PreparedNeuralBatch,
    offset: float,
) -> PreparedNeuralBatch:
    target = np.ascontiguousarray(batch.target + np.float32(offset))
    return replace(batch, target=target)
