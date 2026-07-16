"""End-to-end command test for the explicit classical publication workflow."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl
import pytest
from finoptions import CpuMathBackend

from discrete_dividend_study.classical_artifact import load_classical_study_artifact
from discrete_dividend_study.cli import StudyCliError, main
from discrete_dividend_study.local_resources import (
    PROJECT_INTERACTIVE_RESOURCE_BUDGET,
    acquire_local_heavy_process_lease,
)
from discrete_dividend_study.rolling_windows import RollingWindowSpec
from discrete_dividend_study.study_features import (
    StudyFeatureArtifactKind,
    VerifiedStudyFeatureArtifact,
)
from tests.test_classical_groups import grouped_frame
from tests.test_classical_inputs import classical_frame

_GROUPS = 2
_ARTIFACT_FORMAT_VERSION = 4
_CLI_TEST_START = date(2015, 1, 1)
_CLI_TEST_END_EXCLUSIVE = date(2024, 8, 11)
_CLI_TEST_DAYS = (_CLI_TEST_END_EXCLUSIVE - _CLI_TEST_START).days
_CLI_TEST_ROLLING_SPEC = RollingWindowSpec(
    total_lookback_days=60,
    validation_days=30,
    test_days=_CLI_TEST_DAYS,
    step_days=_CLI_TEST_DAYS,
    first_test_start=_CLI_TEST_START,
    latest_test_end=_CLI_TEST_END_EXCLUSIVE,
)
_CLI_TEST_WINDOWS = 1


def test_heavy_command_rejects_overlap_before_reading_features(tmp_path: Path) -> None:
    """A competing local phase fails before source or framework initialization."""
    lock = tmp_path / "heavy-process.lock"
    with (
        acquire_local_heavy_process_lease(lock, PROJECT_INTERACTIVE_RESOURCE_BUDGET),
        pytest.raises(StudyCliError, match="another heavy local process"),
    ):
        main(
            (
                "run-classical-study",
                "--local-resource-lock",
                str(lock),
                "--features",
                str(tmp_path / "must-not-be-read.parquet"),
                "--feature-artifact-kind",
                "project-experiment",
                "--artifacts",
                str(tmp_path / "must-not-be-read"),
                "--threads",
                "1",
                "--math-backend",
                "rust_std_f64",
                "--training-seed",
                "7",
                "--training-domain-start",
                "100",
                "--evaluation-seed",
                "11",
                "--evaluation-domain-start",
                "10000",
                "--population-addressing",
                "contract_rows",
                "--event-buffer-policy",
                "allow_rechunk",
            ),
        )


def test_classical_command_rejects_bare_parquet_without_receipt(tmp_path: Path) -> None:
    """A schema-compatible file is not a study artifact without its producer receipt."""
    feature = tmp_path / "bare.parquet"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    classical_frame().write_parquet(feature)
    with pytest.raises(StudyCliError, match="receipt"):
        main(
            (
                "run-classical-study",
                "--features",
                str(feature),
                "--feature-artifact-kind",
                "project-experiment",
                "--artifacts",
                str(artifacts),
                "--threads",
                "1",
                "--math-backend",
                "rust_std_f64",
                "--training-seed",
                "7",
                "--training-domain-start",
                "100",
                "--evaluation-seed",
                "11",
                "--evaluation-domain-start",
                "10000",
                "--population-addressing",
                "contract_rows",
                "--event-buffer-policy",
                "allow_rechunk",
            )
        )


def test_classical_command_emits_one_verified_complete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI resolves code/source identity, prices all variants, and prints aggregate JSON."""
    feature = tmp_path / "features.parquet"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    in_scope = classical_frame()
    out_of_scope = in_scope.head(1).with_columns(
        pl.lit(date(2014, 12, 31), dtype=pl.Date).alias("quote_date"),
    )
    feature_frame = pl.concat((out_of_scope, in_scope))
    feature_frame.write_parquet(feature)
    receipt = tmp_path / "features.parquet.receipt.json"
    receipt.write_bytes(b"verified-classical-cli-fixture")
    paper_revision = "a" * 40
    engine_revision = "b" * 40

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
    artifact = _verified_feature_handle(feature, receipt, rows=feature_frame.height)
    monkeypatch.setattr(
        "discrete_dividend_study.cli.load_verified_study_feature_artifact",
        lambda path, kind: (
            artifact
            if Path(path) == feature and kind is StudyFeatureArtifactKind.PROJECT_EXPERIMENT
            else (_ for _ in ()).throw(AssertionError)
        ),
    )
    observed = main(
        (
            "run-classical-study",
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
            "--training-seed",
            "7",
            "--training-domain-start",
            "100",
            "--evaluation-seed",
            "11",
            "--evaluation-domain-start",
            "10000",
            "--population-addressing",
            "contract_rows",
            "--event-buffer-policy",
            "allow_rechunk",
        ),
    )
    assert observed == 0
    summary = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    artifact_path = cast("str", summary["artifact_path"])
    loaded = load_classical_study_artifact(Path(artifact_path))
    assert loaded.artifact.manifest.source.paper_revision == paper_revision
    assert loaded.artifact.manifest.source.engine_revision == engine_revision
    assert loaded.artifact.manifest.execution.math_backend is CpuMathBackend.RUST_STD_F64
    assert summary["simulation_units"] == classical_frame().height
    assert summary["population_addressing"] == "contract_rows"
    assert summary["artifact_sha256"] == loaded.artifact.sha256
    assert summary["rows"] == classical_frame().height
    assert summary["test_windows"] == _CLI_TEST_WINDOWS
    assert summary["test_start"] == "2015-01-01"
    assert summary["test_end_exclusive"] == "2024-08-11"
    assert set(cast("dict[str, object]", summary["metrics"])) == set(
        loaded.artifact.manifest.variants
    )

    assert (
        main(
            (
                "render-classical-table",
                "--artifact",
                artifact_path,
                "--format",
                "markdown",
            )
        )
        == 0
    )
    rendered = capsys.readouterr().out
    assert rendered.startswith(f"<!-- source_artifact_sha256={loaded.artifact.sha256} -->\n")
    assert "| Cash jump (liquidator) | IT_GBM_SPOT |" in rendered


def test_classical_command_prices_explicit_natural_strike_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI binds group addressing and simulation-unit count into artifact v3."""
    feature = tmp_path / "grouped-features.parquet"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    grouped_frame().write_parquet(feature)
    receipt = tmp_path / "grouped-features.parquet.receipt.json"
    receipt.write_bytes(b"verified-grouped-classical-cli-fixture")
    paper_revision = "c" * 40
    engine_revision = "d" * 40

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
    artifact = _verified_feature_handle(feature, receipt, rows=grouped_frame().height)
    monkeypatch.setattr(
        "discrete_dividend_study.cli.load_verified_study_feature_artifact",
        lambda path, kind: (
            artifact
            if Path(path) == feature and kind is StudyFeatureArtifactKind.PROJECT_EXPERIMENT
            else (_ for _ in ()).throw(AssertionError)
        ),
    )
    observed = main(
        (
            "run-classical-study",
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
            "--training-seed",
            "7",
            "--training-domain-start",
            "100",
            "--evaluation-seed",
            "11",
            "--evaluation-domain-start",
            "10000",
            "--population-addressing",
            "natural_strike_groups",
            "--event-buffer-policy",
            "allow_rechunk",
        ),
    )
    assert observed == 0
    summary = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert summary["rows"] == grouped_frame().height
    assert summary["simulation_units"] == _GROUPS
    assert summary["population_addressing"] == "natural_strike_groups"
    loaded = load_classical_study_artifact(Path(cast("str", summary["artifact_path"])))
    assert loaded.artifact.manifest.format_version == _ARTIFACT_FORMAT_VERSION
    assert loaded.artifact.manifest.simulation_units == _GROUPS


def _verified_feature_handle(
    feature: Path,
    receipt: Path,
    *,
    rows: int,
) -> VerifiedStudyFeatureArtifact:
    return VerifiedStudyFeatureArtifact(
        kind=StudyFeatureArtifactKind.PROJECT_EXPERIMENT,
        path=feature,
        receipt_path=receipt,
        request_sha256="1" * 64,
        pipeline_sha256="2" * 64,
        feature_schema_sha256="3" * 64,
        output_sha256=hashlib.sha256(feature.read_bytes()).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        output_rows=rows,
        output_bytes=feature.stat().st_size,
        receipt_bytes=receipt.stat().st_size,
    )
