"""End-to-end command test for the explicit classical publication workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from finoptions import CpuMathBackend

from discrete_dividend_study.classical_artifact import load_classical_study_artifact
from discrete_dividend_study.cli import main
from tests.test_classical_inputs import classical_frame

if TYPE_CHECKING:
    import pytest


def test_classical_command_emits_one_verified_complete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI resolves code/source identity, prices all variants, and prints aggregate JSON."""
    feature = tmp_path / "features.parquet"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    classical_frame().write_parquet(feature)
    paper_revision = "a" * 40
    engine_revision = "b" * 40

    def revision(repository: Path) -> str:
        return engine_revision if repository.name == "finoptions-engine" else paper_revision

    monkeypatch.setattr("discrete_dividend_study.cli._clean_git_revision", revision)
    monkeypatch.setattr(
        "discrete_dividend_study.cli._pinned_engine_revision",
        lambda _root: engine_revision,
    )
    observed = main(
        (
            "run-classical-study",
            "--features",
            str(feature),
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
    assert summary["artifact_sha256"] == loaded.artifact.sha256
    assert summary["rows"] == classical_frame().height
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
    assert "| exact_cash | IT_GBM_SPOT |" in rendered
