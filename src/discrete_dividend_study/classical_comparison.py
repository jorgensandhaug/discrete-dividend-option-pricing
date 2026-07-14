"""Aggregate publication comparison against the submitted classical result table."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import polars as pl

from .classical_artifact import LoadedClassicalStudyArtifact
from .classical_spec import ClassicalVariant

if TYPE_CHECKING:
    from .project_metrics import ProjectPricingMetrics

_SHA256_HEX_LENGTH: Final = 64
_LOWERCASE_HEXADECIMAL: Final = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class ClassicalComparisonError(ValueError):
    """An aggregate publication comparison violates its explicit contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected comparison contract."""
        return f"invalid classical publication comparison: {self.reason}"


@dataclass(frozen=True, slots=True)
class SubmittedClassicalReference:
    """One reported project-thesis classical result used only as a qualitative reference."""

    variant: ClassicalVariant
    submitted_result_name: str
    mape_percent: float
    rmse: float

    def __post_init__(self) -> None:
        """Reject incomplete identities or invalid reported metrics."""
        if type(self.variant) is not ClassicalVariant:
            raise ClassicalComparisonError(reason="variant must be ClassicalVariant")
        if type(self.submitted_result_name) is not str or not self.submitted_result_name:
            raise ClassicalComparisonError(
                reason="submitted_result_name must be a nonempty plain string"
            )
        for field, value in (("mape_percent", self.mape_percent), ("rmse", self.rmse)):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ClassicalComparisonError(
                    reason=f"{field} must be a finite nonnegative plain float"
                )


@dataclass(frozen=True, slots=True)
class ClassicalComparisonRow:
    """One current aggregate result beside its non-binding submitted reference."""

    variant: ClassicalVariant
    submitted_result_name: str
    observations: int
    current: ProjectPricingMetrics
    submitted_mape_percent: float
    submitted_rmse: float
    mape_difference_percentage_points: float
    rmse_difference: float
    current_mape_rank: int
    submitted_mape_rank: int
    current_rmse_rank: int
    submitted_rmse_rank: int


@dataclass(frozen=True, slots=True)
class ClassicalPublicationComparison:
    """A complete ordered aggregate comparison bound to one immutable result artifact."""

    artifact_sha256: str
    rows: tuple[ClassicalComparisonRow, ...]

    def __post_init__(self) -> None:
        """Require every classical variant exactly once in canonical order."""
        if (
            type(self.artifact_sha256) is not str
            or len(self.artifact_sha256) != _SHA256_HEX_LENGTH
            or not set(self.artifact_sha256) <= _LOWERCASE_HEXADECIMAL
        ):
            raise ClassicalComparisonError(
                reason="artifact_sha256 must be a lowercase SHA-256 digest"
            )
        if tuple(row.variant for row in self.rows) != tuple(ClassicalVariant):
            raise ClassicalComparisonError(
                reason="comparison rows must follow the complete canonical variant order"
            )


PROJECT_SUBMITTED_CLASSICAL_REFERENCES: Final = (
    SubmittedClassicalReference(
        variant=ClassicalVariant.EXACT_CASH,
        submitted_result_name="IT_GBM_SPOT",
        mape_percent=4.695443,
        rmse=0.651005,
    ),
    SubmittedClassicalReference(
        variant=ClassicalVariant.ESCROWED_CASH_SPOT_MATCHED,
        submitted_result_name="IT_GBMT_FD_AH",
        mape_percent=4.750231,
        rmse=0.666347,
    ),
    SubmittedClassicalReference(
        variant=ClassicalVariant.FORECAST_YIELD,
        submitted_result_name="IT_GBMT_FY",
        mape_percent=4.924086,
        rmse=0.659504,
    ),
    SubmittedClassicalReference(
        variant=ClassicalVariant.HISTORICAL_YIELD,
        submitted_result_name="IT_GBMT_HY",
        mape_percent=5.253302,
        rmse=0.665882,
    ),
)


@dataclass(frozen=True, slots=True)
class _VariantRanks:
    """Ordinal aggregate ranks for one set of variant metrics."""

    mape: dict[ClassicalVariant, int]
    rmse: dict[ClassicalVariant, int]


def build_classical_publication_comparison(
    loaded: LoadedClassicalStudyArtifact,
) -> ClassicalPublicationComparison:
    """Pair current Float64 metrics with reported references without imposing equality."""
    if type(loaded) is not LoadedClassicalStudyArtifact:
        raise ClassicalComparisonError(reason="loaded must be LoadedClassicalStudyArtifact")
    current_metrics = {
        variant.variant.variant: variant.metrics for variant in loaded.result.variants
    }
    references = {
        reference.variant: reference for reference in PROJECT_SUBMITTED_CLASSICAL_REFERENCES
    }
    current_ranks = _VariantRanks(
        mape=_ranks(
            {variant: metrics.mape_percent for variant, metrics in current_metrics.items()}
        ),
        rmse=_ranks({variant: metrics.rmse for variant, metrics in current_metrics.items()}),
    )
    reference_ranks = _VariantRanks(
        mape=_ranks({variant: reference.mape_percent for variant, reference in references.items()}),
        rmse=_ranks({variant: reference.rmse for variant, reference in references.items()}),
    )
    rows = tuple(
        _comparison_row(
            variant,
            current=current_metrics[variant],
            reference=references[variant],
            current_ranks=current_ranks,
            reference_ranks=reference_ranks,
        )
        for variant in ClassicalVariant
    )
    return ClassicalPublicationComparison(artifact_sha256=loaded.artifact.sha256, rows=rows)


def classical_publication_frame(comparison: ClassicalPublicationComparison) -> pl.DataFrame:
    """Return a small aggregate-only Polars table suitable for paper-table rendering."""
    if type(comparison) is not ClassicalPublicationComparison:
        raise ClassicalComparisonError(reason="comparison must be ClassicalPublicationComparison")
    return pl.DataFrame(
        {
            "variant": [row.variant.value for row in comparison.rows],
            "submitted_result_name": [row.submitted_result_name for row in comparison.rows],
            "observations": [row.observations for row in comparison.rows],
            "current_mape_percent": [row.current.mape_percent for row in comparison.rows],
            "submitted_mape_percent": [row.submitted_mape_percent for row in comparison.rows],
            "mape_difference_percentage_points": [
                row.mape_difference_percentage_points for row in comparison.rows
            ],
            "current_rmse": [row.current.rmse for row in comparison.rows],
            "submitted_rmse": [row.submitted_rmse for row in comparison.rows],
            "rmse_difference": [row.rmse_difference for row in comparison.rows],
            "current_mape_rank": [row.current_mape_rank for row in comparison.rows],
            "submitted_mape_rank": [row.submitted_mape_rank for row in comparison.rows],
            "current_rmse_rank": [row.current_rmse_rank for row in comparison.rows],
            "submitted_rmse_rank": [row.submitted_rmse_rank for row in comparison.rows],
        },
        schema={
            "variant": pl.String,
            "submitted_result_name": pl.String,
            "observations": pl.UInt64,
            "current_mape_percent": pl.Float64,
            "submitted_mape_percent": pl.Float64,
            "mape_difference_percentage_points": pl.Float64,
            "current_rmse": pl.Float64,
            "submitted_rmse": pl.Float64,
            "rmse_difference": pl.Float64,
            "current_mape_rank": pl.UInt32,
            "submitted_mape_rank": pl.UInt32,
            "current_rmse_rank": pl.UInt32,
            "submitted_rmse_rank": pl.UInt32,
        },
    )


def _comparison_row(
    variant: ClassicalVariant,
    *,
    current: ProjectPricingMetrics,
    reference: SubmittedClassicalReference,
    current_ranks: _VariantRanks,
    reference_ranks: _VariantRanks,
) -> ClassicalComparisonRow:
    return ClassicalComparisonRow(
        variant=variant,
        submitted_result_name=reference.submitted_result_name,
        observations=current.observations,
        current=current,
        submitted_mape_percent=reference.mape_percent,
        submitted_rmse=reference.rmse,
        mape_difference_percentage_points=current.mape_percent - reference.mape_percent,
        rmse_difference=current.rmse - reference.rmse,
        current_mape_rank=current_ranks.mape[variant],
        submitted_mape_rank=reference_ranks.mape[variant],
        current_rmse_rank=current_ranks.rmse[variant],
        submitted_rmse_rank=reference_ranks.rmse[variant],
    )


def _ranks(values: dict[ClassicalVariant, float]) -> dict[ClassicalVariant, int]:
    return {
        variant: 1 + sum(other < value for other in values.values())
        for variant, value in values.items()
    }


__all__ = [
    "PROJECT_SUBMITTED_CLASSICAL_REFERENCES",
    "ClassicalComparisonError",
    "ClassicalComparisonRow",
    "ClassicalPublicationComparison",
    "SubmittedClassicalReference",
    "build_classical_publication_comparison",
    "classical_publication_frame",
]
