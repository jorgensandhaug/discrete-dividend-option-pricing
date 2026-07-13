"""Tests for complete immutable neural rolling-study orchestration."""

from __future__ import annotations

import runpy
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from finoptions import CpuExecutor

from discrete_dividend_study.neural_execution import PROJECT_PORTABLE_CPU_EXECUTION_PROFILE
from discrete_dividend_study.neural_spec import PROJECT_NEURAL_STUDY_SPEC, NeuralVariant
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
    lock = tmp_path / "uv.lock"
    _study_frame().write_parquet(feature)
    lock.write_text("version = 1\n", encoding="utf-8")
    training = replace(
        PROJECT_NEURAL_STUDY_SPEC.training,
        batch_size=2,
        maximum_epochs=1,
        early_stopping_patience_epochs=1,
    )
    return NeuralRollingStudyRequest(
        feature_artifact=feature,
        dependency_lock=lock,
        paper_revision="a" * 40,
        engine_revision="b" * 40,
        windows=(WINDOW,),
        study=replace(PROJECT_NEURAL_STUDY_SPEC, training=training),
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
    assert str(request.feature_artifact).encode() not in first.study.path.read_bytes()

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
