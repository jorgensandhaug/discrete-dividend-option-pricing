"""Command-level gates for the primary submitted neural-study identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from discrete_dividend_study.cli import _run_neural_study, _StudyCliContext
from discrete_dividend_study.neural_spec import PROJECT_AS_SUBMITTED_NEURAL_STUDY_SPEC
from discrete_dividend_study.rolling_windows import (
    PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC,
    generate_rolling_windows,
)
from discrete_dividend_study.study_features import (
    StudyFeatureArtifactKind,
    VerifiedStudyFeatureArtifact,
)

if TYPE_CHECKING:
    import pytest
    from finoptions import CpuExecutor

    from discrete_dividend_study.neural_study import NeuralRollingStudyRequest

_DATA_THREADS = 4


def test_primary_command_selects_the_exact_submitted_five_model_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public entry point cannot silently run the corrected attention control."""
    observed: list[NeuralRollingStudyRequest] = []

    def capture(
        request: NeuralRollingStudyRequest,
        artifact_root: Path,
        executor: CpuExecutor,
    ) -> object:
        observed.append(request)
        assert artifact_root == tmp_path
        assert executor.threads == _DATA_THREADS
        return SimpleNamespace(
            study=SimpleNamespace(
                manifest=SimpleNamespace(model_results=()),
                path=Path("complete.study.json"),
                sha256="f" * 64,
            ),
            runs=(),
        )

    monkeypatch.setattr(
        "discrete_dividend_study.cli.run_project_neural_rolling_study",
        capture,
    )
    context = _StudyCliContext(
        feature_artifact=_feature_artifact(tmp_path),
        artifact_root=tmp_path,
        paper_root=tmp_path,
        paper_revision="a" * 40,
        engine_revision="b" * 40,
    )
    arguments = argparse.Namespace(
        profile="portable-cpu",
        data_threads=_DATA_THREADS,
        parameter_root_seed=101,
        shuffle_root_seed=202,
    )
    assert _run_neural_study(arguments, context) == 0
    assert len(observed) == 1
    request = observed[0]
    assert request.study is PROJECT_AS_SUBMITTED_NEURAL_STUDY_SPEC
    assert request.windows == generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    assert json.loads(capsys.readouterr().out) == {
        "model_results": [],
        "runs": 0,
        "study_path": "complete.study.json",
        "study_sha256": "f" * 64,
    }


def _feature_artifact(root: Path) -> VerifiedStudyFeatureArtifact:
    return VerifiedStudyFeatureArtifact(
        kind=StudyFeatureArtifactKind.PROJECT_EXPERIMENT,
        path=root / "features.parquet",
        receipt_path=root / "features.parquet.receipt.json",
        request_sha256="1" * 64,
        pipeline_sha256="2" * 64,
        feature_schema_sha256="3" * 64,
        output_sha256="4" * 64,
        receipt_sha256="5" * 64,
        output_rows=1,
        output_bytes=1,
        receipt_bytes=1,
    )
