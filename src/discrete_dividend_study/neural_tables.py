"""Aggregate-only tables for the complete submitted project neural study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from .neural_run import NeuralRunWindow
from .neural_spec import (
    PROJECT_STUDY,
    NeuralVariant,
    neural_study_sha256,
)
from .neural_study import NeuralRollingStudyArtifact
from .project_metrics import ProjectPricingMetrics
from .rolling_windows import PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC, generate_rolling_windows

_SHA256_HEX_LENGTH: Final = 64
_LOWERCASE_HEXADECIMAL: Final = frozenset("0123456789abcdef")
_MODEL_LABELS: Final = {
    NeuralVariant.LSTM: "Direct cash dividends (LSTM)",
    NeuralVariant.EMBEDDING: "Direct cash dividends (attention)",
    NeuralVariant.BINS: "Direct cash dividends (12 bins)",
    NeuralVariant.FORECAST_YIELD: "Forecast continuous yield",
    NeuralVariant.HISTORICAL_YIELD: "Historical continuous yield",
}
_LATEX_ESCAPE_TABLE: Final = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)


@dataclass(slots=True)
class NeuralTableError(ValueError):
    """A neural paper table is incomplete or is not the submitted study."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected table contract."""
        return f"invalid project neural table: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralTableRow:
    """One complete model result with a stable public label."""

    variant: NeuralVariant
    label: str
    submitted_result_name: str
    completed_windows: int
    metrics: ProjectPricingMetrics

    def __post_init__(self) -> None:
        """Reject inferred labels, partial schedules, and mismatched identities."""
        if type(self.variant) is not NeuralVariant:
            raise NeuralTableError(reason="row variant must be NeuralVariant")
        if self.label != _MODEL_LABELS[self.variant]:
            raise NeuralTableError(reason="row label differs from the canonical public label")
        model = PROJECT_STUDY.model(self.variant)
        if self.submitted_result_name != model.submitted_result_name:
            raise NeuralTableError(reason="row result name differs from the submitted identity")
        expected_windows = len(generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC))
        if type(self.completed_windows) is not int or self.completed_windows != expected_windows:
            raise NeuralTableError(reason="row does not contain the complete submitted schedule")
        if type(self.metrics) is not ProjectPricingMetrics:
            raise NeuralTableError(reason="row metrics must be ProjectPricingMetrics")


@dataclass(frozen=True, slots=True)
class NeuralPublicationTable:
    """The complete ordered five-model table bound to one immutable study artifact."""

    artifact_sha256: str
    rows: tuple[NeuralTableRow, ...]

    def __post_init__(self) -> None:
        """Require a valid content identity and the canonical model order."""
        if (
            type(self.artifact_sha256) is not str
            or len(self.artifact_sha256) != _SHA256_HEX_LENGTH
            or not set(self.artifact_sha256) <= _LOWERCASE_HEXADECIMAL
        ):
            raise NeuralTableError(reason="artifact_sha256 must be a lowercase SHA-256 digest")
        if type(self.rows) is not tuple or any(
            type(row) is not NeuralTableRow for row in self.rows
        ):
            raise NeuralTableError(reason="rows must be an immutable tuple of NeuralTableRow")
        if tuple(row.variant for row in self.rows) != tuple(NeuralVariant):
            raise NeuralTableError(reason="rows must contain every model in canonical order")
        observations = {row.metrics.observations for row in self.rows}
        if len(observations) != 1:
            raise NeuralTableError(reason="all models must evaluate the identical test cohort")


def build_neural_publication_table(
    artifact: NeuralRollingStudyArtifact,
) -> NeuralPublicationTable:
    """Validate and project one complete submitted-study artifact into aggregate rows."""
    if type(artifact) is not NeuralRollingStudyArtifact:
        raise NeuralTableError(reason="artifact must be NeuralRollingStudyArtifact")
    manifest = artifact.manifest
    if manifest.study_sha256 != neural_study_sha256(PROJECT_STUDY):
        raise NeuralTableError(
            reason="artifact study identity differs from the submitted model spec"
        )
    if manifest.windows != _submitted_windows():
        raise NeuralTableError(
            reason="artifact rolling schedule differs from the submitted schedule"
        )
    rows = tuple(
        NeuralTableRow(
            variant=variant,
            label=_MODEL_LABELS[variant],
            submitted_result_name=result.submitted_result_name,
            completed_windows=result.completed_windows,
            metrics=result.metrics,
        )
        for variant, result in zip(NeuralVariant, manifest.model_results, strict=True)
    )
    return NeuralPublicationTable(artifact_sha256=artifact.sha256, rows=rows)


def neural_publication_frame(table: NeuralPublicationTable) -> pl.DataFrame:
    """Return an aggregate-only typed Polars table for downstream paper tooling."""
    _require_table(table)
    return pl.DataFrame(
        {
            "variant": [row.variant.value for row in table.rows],
            "model": [row.label for row in table.rows],
            "submitted_result_name": [row.submitted_result_name for row in table.rows],
            "test_windows": [row.completed_windows for row in table.rows],
            "observations": [row.metrics.observations for row in table.rows],
            "mape_percent": [row.metrics.mape_percent for row in table.rows],
            "rmse": [row.metrics.rmse for row in table.rows],
        },
        schema={
            "variant": pl.String,
            "model": pl.String,
            "submitted_result_name": pl.String,
            "test_windows": pl.UInt32,
            "observations": pl.UInt64,
            "mape_percent": pl.Float64,
            "rmse": pl.Float64,
        },
    )


def neural_publication_markdown(table: NeuralPublicationTable) -> str:
    """Render the complete aggregate result as deterministic Markdown."""
    _require_table(table)
    lines = [
        f"<!-- source_artifact_sha256={table.artifact_sha256} -->",
        "| Model | Test windows | N | MAPE (%) | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            (
                row.label,
                str(row.completed_windows),
                str(row.metrics.observations),
                _metric(row.metrics.mape_percent),
                _metric(row.metrics.rmse),
            )
        )
        + " |"
        for row in table.rows
    )
    return "\n".join(lines) + "\n"


def neural_publication_latex(table: NeuralPublicationTable) -> str:
    """Render the complete aggregate result as a deterministic booktabs fragment."""
    _require_table(table)
    lines = [
        f"% source_artifact_sha256={table.artifact_sha256}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Test windows & N & MAPE (\%) & RMSE \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(
            (
                _latex_escape(row.label),
                str(row.completed_windows),
                str(row.metrics.observations),
                _metric(row.metrics.mape_percent),
                _metric(row.metrics.rmse),
            )
        )
        + r" \\"
        for row in table.rows
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _submitted_windows() -> tuple[NeuralRunWindow, ...]:
    return tuple(
        NeuralRunWindow(
            window_id=window.window_id,
            train_start=window.train_start.isoformat(),
            validation_start=window.validation_start.isoformat(),
            test_start=window.test_start.isoformat(),
            test_end=window.test_end.isoformat(),
        )
        for window in generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    )


def _require_table(table: object) -> None:
    if type(table) is not NeuralPublicationTable:
        raise NeuralTableError(reason="table must be NeuralPublicationTable")


def _metric(value: float) -> str:
    return f"{value:.6f}"


def _latex_escape(value: str) -> str:
    return value.translate(_LATEX_ESCAPE_TABLE)


__all__ = [
    "NeuralPublicationTable",
    "NeuralTableError",
    "NeuralTableRow",
    "build_neural_publication_table",
    "neural_publication_frame",
    "neural_publication_latex",
    "neural_publication_markdown",
]
