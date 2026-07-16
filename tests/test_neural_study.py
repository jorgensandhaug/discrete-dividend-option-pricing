"""Tests for complete immutable neural rolling-study orchestration."""

from __future__ import annotations

import hashlib
import runpy
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from finoptions import CpuExecutor

from discrete_dividend_study.neural_execution import PROJECT_PORTABLE_CPU_EXECUTION_PROFILE
from discrete_dividend_study.neural_spec import PROJECT_STUDY, NeuralVariant
from discrete_dividend_study.neural_study import (
    NeuralRollingStudyRequest,
    NeuralSeedRole,
    NeuralStudyRunError,
    derive_neural_window_seed,
    load_neural_rolling_study,
    project_neural_seed_plan,
    run_project_neural_rolling_study,
)
from discrete_dividend_study.rolling_windows import RollingWindow
from discrete_dividend_study.study_features import (
    StudyFeatureArtifactKind,
    VerifiedStudyFeatureArtifact,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


WINDOW = RollingWindow(
    window_id=0,
    train_start=date(2024, 1, 1),
    validation_start=date(2024, 2, 1),
    test_start=date(2024, 3, 1),
    test_end=date(2024, 4, 1),
)
PARAMETER_ROOT_SEED = 2026071401
SHUFFLE_ROOT_SEED = 2026071402
EXPECTED_BIN_WINDOW_17_PARAMETER_SEED = 12_305_562_132_146_074_098


def _study_frame() -> pl.DataFrame:
    helpers = runpy.run_path("tests/test_model_inputs.py")
    factory = cast("Callable[[], pl.DataFrame]", helpers["model_frame"])
    return factory().with_columns(
        pl.Series(
            "quote_date",
            [date(2024, 1, 5), date(2024, 2, 5), date(2024, 3, 5)],
            dtype=pl.Date,
        ),
    )


def _request(tmp_path: Path) -> NeuralRollingStudyRequest:
    feature = tmp_path / "licensed-features.parquet"
    receipt = tmp_path / "licensed-features.parquet.receipt.json"
    lock = tmp_path / "uv.lock"
    _study_frame().write_parquet(feature)
    receipt.write_bytes(b"verified-neural-study-fixture")
    lock.write_text("version = 1\n", encoding="utf-8")
    training = replace(
        PROJECT_STUDY.training,
        batch_size=2,
        maximum_epochs=1,
        early_stopping_patience_epochs=1,
    )
    return NeuralRollingStudyRequest(
        feature_artifact=VerifiedStudyFeatureArtifact(
            kind=StudyFeatureArtifactKind.PROJECT_EXPERIMENT,
            path=feature,
            receipt_path=receipt,
            request_sha256="1" * 64,
            pipeline_sha256="2" * 64,
            feature_schema_sha256="3" * 64,
            output_sha256=hashlib.sha256(feature.read_bytes()).hexdigest(),
            receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            output_rows=_study_frame().height,
            output_bytes=feature.stat().st_size,
            receipt_bytes=receipt.stat().st_size,
        ),
        dependency_lock=lock,
        paper_revision="a" * 40,
        engine_revision="b" * 40,
        windows=(WINDOW,),
        study=replace(PROJECT_STUDY, training=training),
        execution_profile=PROJECT_PORTABLE_CPU_EXECUTION_PROFILE,
        seed_plan=project_neural_seed_plan(PARAMETER_ROOT_SEED, SHUFFLE_ROOT_SEED),
    )


def test_seed_derivation_is_stable_role_model_and_window_separated() -> None:
    """Canonical roots produce stable unsigned seeds without cross-role reuse."""
    plan = project_neural_seed_plan(PARAMETER_ROOT_SEED, SHUFFLE_ROOT_SEED)
    observed = derive_neural_window_seed(
        plan,
        NeuralSeedRole.PARAMETER,
        NeuralVariant.BINS,
        17,
    )
    assert observed == EXPECTED_BIN_WINDOW_17_PARAMETER_SEED
    values = {
        derive_neural_window_seed(plan, role, variant, window_id)
        for role in NeuralSeedRole
        for variant in NeuralVariant
        for window_id in (0, 1)
    }
    assert len(values) == len(NeuralSeedRole) * len(NeuralVariant) * 2


def test_complete_study_runs_resumes_pools_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All five models run once, pool by observations, and then resume without training."""
    request = _request(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    first = run_project_neural_rolling_study(
        request,
        artifact_root,
        CpuExecutor(threads=2),
    )
    assert len(first.runs) == len(NeuralVariant)
    assert len(first.study.manifest.model_results) == len(NeuralVariant)
    assert all(result.completed_windows == 1 for result in first.study.manifest.model_results)
    assert all(result.metrics.observations == 1 for result in first.study.manifest.model_results)
    assert load_neural_rolling_study(first.study.path) == first.study
    assert str(request.feature_artifact.path).encode() not in first.study.path.read_bytes()

    def unexpected_training(*_args: object, **_kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr(
        "discrete_dividend_study.neural_run._train_and_predict",
        unexpected_training,
    )
    second = run_project_neural_rolling_study(
        request,
        artifact_root,
        CpuExecutor(threads=2),
    )
    assert second == first
    assert len(tuple((artifact_root / "receipts").iterdir())) == len(NeuralVariant)
    assert tuple((artifact_root / "studies").iterdir()) == (first.study.path,)


def test_tampered_complete_study_fails_by_content_address(tmp_path: Path) -> None:
    """A modified schedule-level result cannot be loaded under its old content name."""
    request = _request(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    result = run_project_neural_rolling_study(
        request,
        artifact_root,
        CpuExecutor(threads=1),
    )
    result.study.path.write_bytes(result.study.path.read_bytes() + b"tampered")
    with pytest.raises(NeuralStudyRunError, match="filename hash"):
        load_neural_rolling_study(result.study.path)


def test_feature_receipt_mutation_after_verification_fails_before_execution(
    tmp_path: Path,
) -> None:
    """A changed receipt cannot pass preparation into runtime preflight or training."""
    request = _request(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    request.feature_artifact.receipt_path.write_bytes(
        request.feature_artifact.receipt_path.read_bytes() + b"tampered"
    )
    with pytest.raises(NeuralStudyRunError, match="feature receipt byte count changed"):
        run_project_neural_rolling_study(
            request,
            artifact_root,
            CpuExecutor(threads=1),
        )
