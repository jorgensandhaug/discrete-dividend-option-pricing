"""Aggregate-only paper tables for verified publication empirical inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from .publication_inference import (
    PublicationContrastInference,
    PublicationEmpiricalInferenceResult,
    PublicationMetricInterval,
)
from .publication_inference_artifact import LoadedPublicationInferenceArtifact
from .publication_inference_spec import PublicationEmpiricalMetric
from .publication_spec import PublicationVariant

_SHA256_LENGTH: Final = 64
_VARIANT_LABELS: Final = {
    PublicationVariant.CASH_JUMP_LIQUIDATOR: "Cash jump (liquidator)",
    PublicationVariant.ESCROWED_CASH_SPOT_MATCHED: "Escrowed cash (spot matched)",
    PublicationVariant.FORECAST_YIELD: "Forecast continuous yield",
    PublicationVariant.HISTORICAL_YIELD: "Historical continuous yield",
}
_METRIC_LABELS: Final = {
    PublicationEmpiricalMetric.MAPE_PERCENT: "MAPE (%)",
    PublicationEmpiricalMetric.RMSE: "RMSE",
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
class PublicationInferenceTableError(ValueError):
    """A requested aggregate inference table is incomplete or misidentified."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected table contract."""
        return f"invalid publication inference table: {self.reason}"


@dataclass(frozen=True, slots=True)
class PublicationInferenceTables:
    """Verified aggregate inference bound to its immutable source artifact."""

    artifact_sha256: str
    result: PublicationEmpiricalInferenceResult

    def __post_init__(self) -> None:
        """Require a lowercase content hash and complete inference result."""
        if (
            type(self.artifact_sha256) is not str
            or len(self.artifact_sha256) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in self.artifact_sha256)
        ):
            raise PublicationInferenceTableError(reason="artifact_sha256 must be lowercase SHA-256")
        if type(self.result) is not PublicationEmpiricalInferenceResult:
            raise PublicationInferenceTableError(
                reason="result must be PublicationEmpiricalInferenceResult"
            )


def build_publication_inference_tables(
    loaded: LoadedPublicationInferenceArtifact,
) -> PublicationInferenceTables:
    """Bind verified inference summaries to one content-addressed artifact."""
    if type(loaded) is not LoadedPublicationInferenceArtifact:
        raise PublicationInferenceTableError(
            reason="loaded must be LoadedPublicationInferenceArtifact"
        )
    return PublicationInferenceTables(
        artifact_sha256=loaded.artifact.sha256,
        result=loaded.result,
    )


def publication_primary_model_frame(tables: PublicationInferenceTables) -> pl.DataFrame:
    """Return four primary-cohort model estimates with paired-panel basic intervals."""
    _validate_tables(tables)
    intervals = _primary_model_intervals(tables.result)
    cohort = tables.result.cohorts[0].panels
    return pl.DataFrame(
        {
            "variant": [variant.value for variant in PublicationVariant],
            "variant_label": [_variant_label(variant) for variant in PublicationVariant],
            "observations": [cohort.observations] * len(PublicationVariant),
            "quote_date_panels": [cohort.quote_date_panels] * len(PublicationVariant),
            "mape_percent": [
                intervals[(variant, PublicationEmpiricalMetric.MAPE_PERCENT)].estimate
                for variant in PublicationVariant
            ],
            "mape_interval_lower": [
                intervals[(variant, PublicationEmpiricalMetric.MAPE_PERCENT)].interval_lower
                for variant in PublicationVariant
            ],
            "mape_interval_upper": [
                intervals[(variant, PublicationEmpiricalMetric.MAPE_PERCENT)].interval_upper
                for variant in PublicationVariant
            ],
            "rmse": [
                intervals[(variant, PublicationEmpiricalMetric.RMSE)].estimate
                for variant in PublicationVariant
            ],
            "rmse_interval_lower": [
                intervals[(variant, PublicationEmpiricalMetric.RMSE)].interval_lower
                for variant in PublicationVariant
            ],
            "rmse_interval_upper": [
                intervals[(variant, PublicationEmpiricalMetric.RMSE)].interval_upper
                for variant in PublicationVariant
            ],
        },
        schema={
            "variant": pl.String,
            "variant_label": pl.String,
            "observations": pl.UInt64,
            "quote_date_panels": pl.UInt64,
            "mape_percent": pl.Float64,
            "mape_interval_lower": pl.Float64,
            "mape_interval_upper": pl.Float64,
            "rmse": pl.Float64,
            "rmse_interval_lower": pl.Float64,
            "rmse_interval_upper": pl.Float64,
        },
    )


def publication_primary_contrast_frame(tables: PublicationInferenceTables) -> pl.DataFrame:
    """Return the twelve Holm-controlled primary left-minus-right comparisons."""
    _validate_tables(tables)
    contrasts = _primary_contrasts(tables.result)
    return pl.DataFrame(
        {
            "left_variant": [item.contrast.left.value for item in contrasts],
            "right_variant": [item.contrast.right.value for item in contrasts],
            "left_label": [_variant_label(item.contrast.left) for item in contrasts],
            "right_label": [_variant_label(item.contrast.right) for item in contrasts],
            "metric": [item.metric.value for item in contrasts],
            "estimate_left_minus_right": [item.estimate for item in contrasts],
            "interval_lower": [item.interval_lower for item in contrasts],
            "interval_upper": [item.interval_upper for item in contrasts],
            "p_value": [item.p_value for item in contrasts],
            "holm_adjusted_p_value": [item.holm_adjusted_p_value for item in contrasts],
            "reject_family_wise_null": [item.reject_family_wise_null for item in contrasts],
        },
        schema={
            "left_variant": pl.String,
            "right_variant": pl.String,
            "left_label": pl.String,
            "right_label": pl.String,
            "metric": pl.String,
            "estimate_left_minus_right": pl.Float64,
            "interval_lower": pl.Float64,
            "interval_upper": pl.Float64,
            "p_value": pl.Float64,
            "holm_adjusted_p_value": pl.Float64,
            "reject_family_wise_null": pl.Boolean,
        },
    )


def publication_sensitivity_contrast_frame(tables: PublicationInferenceTables) -> pl.DataFrame:
    """Return every cohort, block-length, metric, and paired contrast without row-level data."""
    _validate_tables(tables)
    results = tables.result.contrasts
    return pl.DataFrame(
        {
            "cohort": [item.cohort for item in results],
            "mean_block_length": [item.mean_block_length for item in results],
            "left_variant": [item.contrast.left.value for item in results],
            "right_variant": [item.contrast.right.value for item in results],
            "metric": [item.metric.value for item in results],
            "estimate_left_minus_right": [item.estimate for item in results],
            "interval_lower": [item.interval_lower for item in results],
            "interval_upper": [item.interval_upper for item in results],
            "p_value": [item.p_value for item in results],
            "holm_adjusted_p_value": [item.holm_adjusted_p_value for item in results],
            "reject_family_wise_null": [item.reject_family_wise_null for item in results],
        },
        schema={
            "cohort": pl.String,
            "mean_block_length": pl.Float64,
            "left_variant": pl.String,
            "right_variant": pl.String,
            "metric": pl.String,
            "estimate_left_minus_right": pl.Float64,
            "interval_lower": pl.Float64,
            "interval_upper": pl.Float64,
            "p_value": pl.Float64,
            "holm_adjusted_p_value": pl.Float64,
            "reject_family_wise_null": pl.Boolean,
        },
    )


def publication_primary_model_markdown(tables: PublicationInferenceTables) -> str:
    """Render the four-model primary result as deterministic Markdown."""
    frame = publication_primary_model_frame(tables)
    lines = [
        f"<!-- source_artifact_sha256={tables.artifact_sha256} -->",
        "| Variant | N | Date panels | MAPE (%) [95% CI] | RMSE [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            (
                row["variant_label"],
                str(row["observations"]),
                str(row["quote_date_panels"]),
                _estimate_interval(
                    row["mape_percent"],
                    row["mape_interval_lower"],
                    row["mape_interval_upper"],
                ),
                _estimate_interval(
                    row["rmse"],
                    row["rmse_interval_lower"],
                    row["rmse_interval_upper"],
                ),
            )
        )
        + " |"
        for row in frame.iter_rows(named=True)
    )
    return "\n".join(lines) + "\n"


def publication_primary_model_latex(tables: PublicationInferenceTables) -> str:
    """Render the four-model primary result as a deterministic booktabs fragment."""
    frame = publication_primary_model_frame(tables)
    lines = [
        f"% source_artifact_sha256={tables.artifact_sha256}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variant & N & Date panels & MAPE (\%) [95\% CI] & RMSE [95\% CI] \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(
            (
                _latex_escape(row["variant_label"]),
                str(row["observations"]),
                str(row["quote_date_panels"]),
                _estimate_interval(
                    row["mape_percent"],
                    row["mape_interval_lower"],
                    row["mape_interval_upper"],
                ),
                _estimate_interval(
                    row["rmse"],
                    row["rmse_interval_lower"],
                    row["rmse_interval_upper"],
                ),
            )
        )
        + r" \\"
        for row in frame.iter_rows(named=True)
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def publication_primary_contrast_markdown(tables: PublicationInferenceTables) -> str:
    """Render twelve primary paired contrasts with Holm-adjusted decisions."""
    frame = publication_primary_contrast_frame(tables)
    lines = [
        f"<!-- source_artifact_sha256={tables.artifact_sha256} -->",
        "| Left - right | Metric | Estimate [95% CI] | p | Holm p | Reject |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            (
                f"{row['left_label']} - {row['right_label']}",
                _METRIC_LABELS[PublicationEmpiricalMetric(row["metric"])],
                _estimate_interval(
                    row["estimate_left_minus_right"],
                    row["interval_lower"],
                    row["interval_upper"],
                ),
                _metric(row["p_value"]),
                _metric(row["holm_adjusted_p_value"]),
                "yes" if row["reject_family_wise_null"] else "no",
            )
        )
        + " |"
        for row in frame.iter_rows(named=True)
    )
    return "\n".join(lines) + "\n"


def publication_primary_contrast_latex(tables: PublicationInferenceTables) -> str:
    """Render twelve primary paired contrasts as a deterministic booktabs fragment."""
    frame = publication_primary_contrast_frame(tables)
    lines = [
        f"% source_artifact_sha256={tables.artifact_sha256}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Left $-$ right & Metric & Estimate [95\% CI] & $p$ & Holm $p$ & Reject \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(
            (
                _latex_escape(f"{row['left_label']} - {row['right_label']}"),
                _latex_escape(_METRIC_LABELS[PublicationEmpiricalMetric(row["metric"])]),
                _estimate_interval(
                    row["estimate_left_minus_right"],
                    row["interval_lower"],
                    row["interval_upper"],
                ),
                _metric(row["p_value"]),
                _metric(row["holm_adjusted_p_value"]),
                "yes" if row["reject_family_wise_null"] else "no",
            )
        )
        + r" \\"
        for row in frame.iter_rows(named=True)
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _primary_model_intervals(
    result: PublicationEmpiricalInferenceResult,
) -> dict[tuple[PublicationVariant, PublicationEmpiricalMetric], PublicationMetricInterval]:
    spec = result.spec
    selected = tuple(
        item
        for item in result.model_intervals
        if item.cohort == spec.cohorts[0].name
        and item.mean_block_length == spec.bootstrap.mean_block_length
        and item.metric in spec.primary_metrics
    )
    expected = len(PublicationVariant) * len(spec.primary_metrics)
    if len(selected) != expected:
        raise PublicationInferenceTableError(reason="primary model intervals are incomplete")
    return {(item.variant, item.metric): item for item in selected}


def _primary_contrasts(
    result: PublicationEmpiricalInferenceResult,
) -> tuple[PublicationContrastInference, ...]:
    spec = result.spec
    selected = tuple(
        item
        for item in result.contrasts
        if item.cohort == spec.cohorts[0].name
        and item.mean_block_length == spec.bootstrap.mean_block_length
        and item.metric in spec.primary_metrics
    )
    expected = len(spec.contrasts) * len(spec.primary_metrics)
    if len(selected) != expected or any(
        item.holm_adjusted_p_value is None or item.reject_family_wise_null is None
        for item in selected
    ):
        raise PublicationInferenceTableError(reason="primary contrast inference is incomplete")
    return selected


def _validate_tables(value: object) -> None:
    if type(value) is not PublicationInferenceTables:
        raise PublicationInferenceTableError(reason="tables must be PublicationInferenceTables")


def _variant_label(variant: PublicationVariant) -> str:
    return _VARIANT_LABELS[variant]


def _estimate_interval(estimate: float, lower: float, upper: float) -> str:
    return f"{_metric(estimate)} [{_metric(lower)}, {_metric(upper)}]"


def _metric(value: float) -> str:
    return f"{value:.6f}"


def _latex_escape(value: str) -> str:
    return value.translate(_LATEX_ESCAPE_TABLE)


__all__ = [
    "PublicationInferenceTableError",
    "PublicationInferenceTables",
    "build_publication_inference_tables",
    "publication_primary_contrast_frame",
    "publication_primary_contrast_latex",
    "publication_primary_contrast_markdown",
    "publication_primary_model_frame",
    "publication_primary_model_latex",
    "publication_primary_model_markdown",
    "publication_sensitivity_contrast_frame",
]
