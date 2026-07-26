"""Tests for the complete aggregate-only project neural paper table."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from discrete_dividend_study.cli import main as cli_main
from discrete_dividend_study.experiments.metrics import (
    ProjectPricingMetrics,
    aggregate_project_pricing_metrics,
)
from discrete_dividend_study.experiments.neural.run import NeuralRunWindow, StudySourceIdentity
from discrete_dividend_study.experiments.neural.study import (
    NeuralModelStudyResult,
    NeuralRollingStudyArtifact,
    NeuralRollingStudyManifest,
    NeuralSeedRole,
    NeuralStudyRunReference,
    derive_neural_window_seed,
    project_neural_seed_plan,
    write_neural_rolling_study,
)
from discrete_dividend_study.experiments.neural.tables import (
    NeuralTableError,
    build_neural_publication_table,
    neural_publication_frame,
    neural_publication_latex,
    neural_publication_markdown,
)
from discrete_dividend_study.experiments.windows import (
    PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC,
    generate_rolling_windows,
)
from discrete_dividend_study.models.neural.specification import (
    NeuralVariant,
    neural_study_sha256,
)
from discrete_dividend_study.paper import PROJECT_STUDY

_OBSERVATIONS_PER_WINDOW = 2
_WINDOWS = len(generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC))
_POOLED_OBSERVATIONS = _WINDOWS * _OBSERVATIONS_PER_WINDOW


def _metrics(
    model_index: int,
    *,
    observations: int = _OBSERVATIONS_PER_WINDOW,
) -> ProjectPricingMetrics:
    rmse = 0.5 + (model_index / 10.0)
    return ProjectPricingMetrics(
        observations=observations,
        mae=0.4 + (model_index / 10.0),
        mape_percent=4.0 + model_index,
        mse=rmse * rmse,
        rmse=rmse,
        bias=-0.1 + (model_index / 100.0),
        relative_bias=-0.01 + (model_index / 1000.0),
        smape_percent=3.8 + model_index,
    )


def _digest(*parts: object) -> str:
    return hashlib.sha256("/".join(str(part) for part in parts).encode()).hexdigest()


def _artifact() -> NeuralRollingStudyArtifact:
    windows = tuple(
        NeuralRunWindow(
            window_id=window.window_id,
            train_start=window.train_start.isoformat(),
            validation_start=window.validation_start.isoformat(),
            test_start=window.test_start.isoformat(),
            test_end=window.test_end.isoformat(),
        )
        for window in generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    )
    seed_plan = project_neural_seed_plan(101, 202)
    runs: list[NeuralStudyRunReference] = []
    results: list[NeuralModelStudyResult] = []
    for model_index, model in enumerate(PROJECT_STUDY.models):
        model_runs = tuple(
            NeuralStudyRunReference(
                variant=model.variant.value,
                submitted_result_name=model.submitted_result_name,
                window_id=window.window_id,
                parameter_seed=derive_neural_window_seed(
                    seed_plan,
                    NeuralSeedRole.PARAMETER,
                    model.variant,
                    window.window_id,
                ),
                shuffle_seed=derive_neural_window_seed(
                    seed_plan,
                    NeuralSeedRole.SHUFFLE,
                    model.variant,
                    window.window_id,
                ),
                request_sha256=_digest(model.variant, window.window_id, "request"),
                run_sha256=_digest(model.variant, window.window_id, "run"),
                checkpoint_sha256=_digest(model.variant, window.window_id, "checkpoint"),
                prediction_sha256=_digest(model.variant, window.window_id, "prediction"),
                test_rows=_OBSERVATIONS_PER_WINDOW,
                metrics=_metrics(model_index),
            )
            for window in windows
        )
        runs.extend(model_runs)
        results.append(
            NeuralModelStudyResult(
                variant=model.variant.value,
                submitted_result_name=model.submitted_result_name,
                completed_windows=len(windows),
                metrics=aggregate_project_pricing_metrics(tuple(run.metrics for run in model_runs)),
            )
        )
    manifest = NeuralRollingStudyManifest(
        format_version=1,
        source=StudySourceIdentity(
            feature_artifact_sha256="1" * 64,
            feature_schema_sha256="2" * 64,
            paper_revision="a" * 40,
            engine_revision="b" * 40,
            dependency_lock_sha256="3" * 64,
        ),
        study_sha256=neural_study_sha256(PROJECT_STUDY),
        execution_profile="test-profile",
        seed_plan=seed_plan,
        windows=windows,
        runs=tuple(runs),
        model_results=tuple(results),
    )
    return NeuralRollingStudyArtifact(
        path=Path("complete.study.json"),
        sha256="f" * 64,
        manifest=manifest,
    )


def test_complete_neural_table_has_canonical_models_and_typed_aggregates() -> None:
    """Only the full submitted identity becomes one ordered five-model paper table."""
    table = build_neural_publication_table(_artifact())
    frame = neural_publication_frame(table)
    assert tuple(row.variant for row in table.rows) == tuple(NeuralVariant)
    assert frame.schema == {
        "variant": pl.String,
        "model": pl.String,
        "submitted_result_name": pl.String,
        "test_windows": pl.UInt32,
        "observations": pl.UInt64,
        "mape_percent": pl.Float64,
        "rmse": pl.Float64,
    }
    assert frame.height == len(NeuralVariant)
    assert frame.get_column("test_windows").unique().item() == _WINDOWS
    assert frame.get_column("observations").unique().item() == _POOLED_OBSERVATIONS


def test_neural_renderers_are_minimal_deterministic_and_aggregate_only() -> None:
    """Paper fragments expose metrics and provenance, not paths or licensed rows."""
    table = build_neural_publication_table(_artifact())
    markdown = neural_publication_markdown(table)
    latex = neural_publication_latex(table)
    provenance = f"source_artifact_sha256={table.artifact_sha256}"
    assert markdown.startswith(f"<!-- {provenance} -->\n")
    assert latex.startswith(f"% {provenance}\n")
    assert "| Direct cash dividends (LSTM) | 117 | 234 | 4.000000 | 0.500000 |" in markdown
    assert r"Direct cash dividends (LSTM) & 117 & 234 & 4.000000 & 0.500000" in latex
    assert latex.endswith("\\bottomrule\n\\end{tabular}\n")
    assert "complete.study.json" not in markdown
    assert "complete.study.json" not in latex


def test_neural_table_rejects_a_different_model_identity() -> None:
    """A complete-looking artifact from another model contract cannot be paper output."""
    artifact = _artifact()
    changed_manifest = replace(artifact.manifest, study_sha256="0" * 64)
    changed_artifact = replace(artifact, manifest=changed_manifest)
    with pytest.raises(NeuralTableError, match="submitted model spec"):
        build_neural_publication_table(changed_artifact)


def test_cli_renders_only_an_explicit_verified_complete_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public command loads by exact path and emits an aggregate-only table."""
    written = write_neural_rolling_study(tmp_path, _artifact().manifest)
    assert (
        cli_main(
            [
                "render-neural-table",
                "--artifact",
                str(written.path),
                "--format",
                "markdown",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert output.startswith(f"<!-- source_artifact_sha256={written.sha256} -->\n")
    assert "Direct cash dividends (attention)" in output
    assert str(tmp_path) not in output
