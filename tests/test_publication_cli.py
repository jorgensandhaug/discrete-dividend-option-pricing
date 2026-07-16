"""End-to-end command test for resumable stochastic publication replications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from finoptions import CpuMathBackend

from discrete_dividend_study import publication_artifact
from discrete_dividend_study.cli import main
from discrete_dividend_study.publication_artifact import load_publication_cash_artifact
from discrete_dividend_study.study_features import StudyFeatureArtifactKind
from tests.test_classical_cli import (
    _CLI_TEST_END_EXCLUSIVE,
    _CLI_TEST_ROLLING_SPEC,
    _CLI_TEST_START,
    _CLI_TEST_WINDOWS,
    _verified_feature_handle,
)
from tests.test_publication_artifact import _plan
from tests.test_publication_estimator import study
from tests.test_publication_exercise import schedule_frame

if TYPE_CHECKING:
    from discrete_dividend_study.publication_pricing import PublicationCashReplicationResult


def test_publication_cash_command_runs_and_exactly_resumes_the_frozen_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI prepares once, verifies every replication, and reuses only exact receipts."""
    feature = tmp_path / "features.parquet"
    receipt = tmp_path / "features.parquet.receipt.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    frame = schedule_frame().filter(pl.col("ticker") != "BBB")
    frame.write_parquet(feature)
    receipt.write_bytes(b"verified-publication-cli-fixture")
    verified = _verified_feature_handle(feature, receipt, rows=frame.height)
    paper_revision = "a" * 40
    engine_revision = "b" * 40
    declared = study()
    plan = _plan()

    monkeypatch.setattr(
        "discrete_dividend_study.cli._clean_git_revision",
        lambda _root: paper_revision,
    )
    monkeypatch.setattr(
        "discrete_dividend_study.cli.resolve_project_engine_revision",
        lambda _root: engine_revision,
    )
    monkeypatch.setattr(
        "discrete_dividend_study.cli.PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC",
        _CLI_TEST_ROLLING_SPEC,
    )
    monkeypatch.setattr(
        "discrete_dividend_study.cli.PROJECT_PUBLICATION_STUDY_SPEC",
        declared,
    )
    monkeypatch.setattr(
        "discrete_dividend_study.cli.PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC",
        plan,
    )
    monkeypatch.setattr(
        "discrete_dividend_study.cli.load_verified_study_feature_artifact",
        lambda path, kind: (
            verified
            if Path(path) == feature and kind is StudyFeatureArtifactKind.PROJECT_EXPERIMENT
            else pytest.fail("unexpected feature artifact request")
        ),
    )
    arguments = (
        "run-publication-cash-replications",
        "--features",
        str(feature),
        "--feature-artifact-kind",
        "project-experiment",
        "--artifacts",
        str(artifacts),
        "--threads",
        "2",
        "--math-backend",
        "rust_std_f64",
        "--event-buffer-policy",
        "allow_rechunk",
    )
    assert main(arguments) == 0
    captured = capsys.readouterr()
    summary = cast("dict[str, object]", json.loads(captured.out))
    references = cast("list[dict[str, object]]", summary["artifacts"])
    assert summary["rows"] == frame.height
    assert summary["replications"] == plan.replications
    assert summary["test_windows"] == _CLI_TEST_WINDOWS
    assert summary["test_start"] == _CLI_TEST_START.isoformat()
    assert summary["test_end_exclusive"] == _CLI_TEST_END_EXCLUSIVE.isoformat()
    assert len(captured.err.strip().splitlines()) == plan.replications
    for replication, reference in enumerate(references):
        loaded = load_publication_cash_artifact(
            Path(cast("str", reference["artifact_path"])),
            study=declared,
            replication_plan=plan,
        )
        assert loaded.artifact.manifest.execution.replication == replication
        assert loaded.artifact.manifest.execution.math_backend is CpuMathBackend.RUST_STD_F64
        assert loaded.artifact.manifest.source.paper_revision == paper_revision
        assert loaded.artifact.manifest.source.engine_revision == engine_revision

    def forbidden_replication(*_: object, **__: object) -> PublicationCashReplicationResult:
        pytest.fail("complete CLI rerun did not resume verified requests")

    monkeypatch.setattr(
        publication_artifact,
        "run_publication_cash_replication",
        forbidden_replication,
    )
    assert main(arguments) == 0
    resumed = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert resumed["artifacts"] == summary["artifacts"]
