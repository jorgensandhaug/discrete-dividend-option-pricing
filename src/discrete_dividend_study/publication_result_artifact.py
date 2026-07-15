"""Complete immutable aggregate for the frozen four-variant publication study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Protocol, cast

import numpy as np
import numpy.typing as npt
from finoptions import AndersenLakeBatchResult, CpuMathBackend, build_info
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

from .neural_run import StudySourceIdentity
from .project_metrics import ProjectMetricError, ProjectPricingMetrics, evaluate_project_pricing_f64
from .publication_artifact import (
    LoadedPublicationCashArtifact,
    PublicationCashArtifact,
    load_publication_cash_artifact,
)
from .publication_groups import PreparedPublicationPartitions, publication_partitions_sha256
from .publication_pricing import (
    PublicationVariantReplicationResult,
    PublicationYieldPricingResult,
    run_publication_yield_pricing,
)
from .publication_spec import (
    PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
    PROJECT_PUBLICATION_STUDY_SPEC,
    PublicationDeterministicExecutionSpec,
    PublicationReplicationPlanSpec,
    PublicationStudySpec,
    PublicationVariant,
    publication_deterministic_execution_sha256,
    publication_replication_plan_sha256,
    publication_study_sha256,
)

Float64Vector = npt.NDArray[np.float64]
BoolVector = npt.NDArray[np.bool_]

_FORMAT: Final = "discrete-dividend-publication-study-result-v1"
_FORMAT_VERSION: Final = 1
_PAPER_PACKAGE: Final = "discrete-dividend-option-pricing"
_METADATA_KEY: Final = "publication_study_result"
_ENVELOPE_KEYS: Final = frozenset({"format", "manifest", "metrics"})
_REQUEST_DIRECTORY: Final = "publication-study-requests"
_RECEIPT_FORMAT: Final = "discrete-dividend-publication-study-request-receipt-v1"
_RECEIPT_KEYS: Final = frozenset({"format", "request_sha256", "artifact_sha256"})
_SHA256_HEX_LENGTH: Final = 64
_MIN_REPLICATIONS: Final = 2
_FLOAT64_DTYPE: Final = np.dtype(np.float64)
_BOOL_DTYPE: Final = np.dtype(np.bool_)
_CASH_VARIANTS: Final = tuple(
    variant for variant in PublicationVariant if variant.uses_cash_schedule
)
_YIELD_VARIANTS: Final = tuple(
    variant for variant in PublicationVariant if not variant.uses_cash_schedule
)
_METRIC_FIELDS: Final = (
    "mae",
    "mape_percent",
    "mse",
    "rmse",
    "bias",
    "relative_bias",
    "smape_percent",
)
_YIELD_FIELDS: Final = (
    "prices",
    "exercised_at_origin",
    "maximum_relative_boundary_residuals",
    "maximum_relative_obstacle_projections",
    "used_european_identity",
)
_YIELD_DTYPES: Final = {
    "prices": _FLOAT64_DTYPE,
    "exercised_at_origin": _BOOL_DTYPE,
    "maximum_relative_boundary_residuals": _FLOAT64_DTYPE,
    "maximum_relative_obstacle_projections": _FLOAT64_DTYPE,
    "used_european_identity": _BOOL_DTYPE,
}


class _HashWriter(Protocol):
    """Streaming byte interface shared by hashlib implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(slots=True)
class PublicationResultArtifactError(RuntimeError):
    """A complete study result or its prerequisite identity is invalid."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected final-result contract."""
        return f"invalid publication study result artifact: {self.reason}"


@dataclass(frozen=True, slots=True)
class PublicationMonteCarloSummary:
    """Two-sided Student-t interval over independently seeded estimator replications."""

    replications: int
    mean: float
    sample_standard_deviation: float
    standard_error: float
    interval_lower: float
    interval_upper: float

    def __post_init__(self) -> None:
        """Require finite ordered state and the exact sample-SD/SE identity."""
        if type(self.replications) is not int or self.replications < _MIN_REPLICATIONS:
            raise PublicationResultArtifactError(
                reason="Monte Carlo replications must be a plain int of at least two"
            )
        for field in (
            "mean",
            "sample_standard_deviation",
            "standard_error",
            "interval_lower",
            "interval_upper",
        ):
            value = getattr(self, field)
            if type(value) is not float or not math.isfinite(value):
                raise PublicationResultArtifactError(reason=f"{field} must be a finite float")
        if self.sample_standard_deviation < 0.0 or self.standard_error < 0.0:
            raise PublicationResultArtifactError(reason="Monte Carlo dispersion cannot be negative")
        expected = self.sample_standard_deviation / math.sqrt(self.replications)
        if not math.isclose(self.standard_error, expected, rel_tol=1e-15, abs_tol=0.0):
            raise PublicationResultArtifactError(reason="standard_error disagrees with sample SD")
        if not self.interval_lower <= self.mean <= self.interval_upper:
            raise PublicationResultArtifactError(reason="Monte Carlo interval is not ordered")


@dataclass(frozen=True, slots=True)
class PublicationNamedMonteCarloSummary:
    """One named project metric's numerical replication uncertainty."""

    metric: str
    estimate: PublicationMonteCarloSummary

    def __post_init__(self) -> None:
        """Require one canonical metric name and typed estimate."""
        if self.metric not in _METRIC_FIELDS:
            raise PublicationResultArtifactError(reason="metric name is unsupported")
        if type(self.estimate) is not PublicationMonteCarloSummary:
            raise PublicationResultArtifactError(
                reason="estimate must be PublicationMonteCarloSummary"
            )


@dataclass(frozen=True, slots=True)
class PublicationCashAggregateResult:
    """Replication-mean price vector, per-row dispersion, and metric uncertainty."""

    variant: PublicationVariant
    mean_prices: Float64Vector
    replication_sample_standard_deviations: Float64Vector
    mean_prediction_metrics: ProjectPricingMetrics
    replication_metrics: tuple[ProjectPricingMetrics, ...]
    metric_summaries: tuple[PublicationNamedMonteCarloSummary, ...]

    def __post_init__(self) -> None:
        """Require one complete immutable cash aggregate."""
        if type(self.variant) is not PublicationVariant or not self.variant.uses_cash_schedule:
            raise PublicationResultArtifactError(reason="variant must be a cash PublicationVariant")
        observations = self.mean_prediction_metrics.observations
        _validate_vector(
            self.mean_prices,
            field="mean_prices",
            dtype=_FLOAT64_DTYPE,
            length=observations,
        )
        _validate_vector(
            self.replication_sample_standard_deviations,
            field="replication_sample_standard_deviations",
            dtype=_FLOAT64_DTYPE,
            length=observations,
        )
        if (
            self.mean_prices.flags.writeable
            or self.replication_sample_standard_deviations.flags.writeable
        ):
            raise PublicationResultArtifactError(reason="cash aggregate arrays must be immutable")
        if np.any(self.replication_sample_standard_deviations < 0.0):
            raise PublicationResultArtifactError(reason="per-row replication SD cannot be negative")
        if len(self.replication_metrics) < _MIN_REPLICATIONS or any(
            metric.observations != observations for metric in self.replication_metrics
        ):
            raise PublicationResultArtifactError(reason="cash replication metrics are incomplete")
        if tuple(summary.metric for summary in self.metric_summaries) != _METRIC_FIELDS:
            raise PublicationResultArtifactError(
                reason="metric summaries are incomplete or reordered"
            )


@dataclass(frozen=True, slots=True)
class PublicationStudyAggregateResult:
    """Both stochastic cash aggregates and both deterministic yield results."""

    study: PublicationStudySpec
    cash: tuple[PublicationCashAggregateResult, ...]
    yields: PublicationYieldPricingResult

    def __post_init__(self) -> None:
        """Require the complete canonical four-variant result."""
        if type(self.study) is not PublicationStudySpec or self.yields.study != self.study:
            raise PublicationResultArtifactError(reason="study identities differ")
        if tuple(result.variant for result in self.cash) != _CASH_VARIANTS:
            raise PublicationResultArtifactError(
                reason="cash aggregates are incomplete or reordered"
            )
        if tuple(result.variant for result in self.yields.variants) != _YIELD_VARIANTS:
            raise PublicationResultArtifactError(reason="yield results are incomplete or reordered")
        if any(
            result.mean_prediction_metrics.observations != self.yields.rows for result in self.cash
        ):
            raise PublicationResultArtifactError(reason="variant row populations differ")


@dataclass(frozen=True, slots=True)
class PublicationStudyResultManifest:
    """Complete final identity, including every prerequisite cash artifact."""

    format_version: int
    request_sha256: str
    source: StudySourceIdentity
    study_sha256: str
    replication_plan_sha256: str
    deterministic_execution: PublicationDeterministicExecutionSpec
    deterministic_execution_sha256: str
    partitions_sha256: str
    observations: int
    simulation_units: int
    exact_groups: int
    general_groups: int
    cash_artifact_sha256s: tuple[str, ...]
    cash_request_sha256s: tuple[str, ...]
    cash_execution_sha256s: tuple[str, ...]
    variants: tuple[str, ...]
    confidence_level: float
    replication_critical_value: float
    paper_package_version: str
    engine_package_version: str
    rustc_version: str
    tensors_sha256: str
    metrics_sha256: str

    def __post_init__(self) -> None:
        """Reject incomplete, aliased, or internally inconsistent final identity."""
        _validate_manifest_structure(self)
        _validate_manifest_population(self)
        _validate_manifest_references(self)
        _validate_manifest_versions(self)
        if self.request_sha256 != _request_sha256_from_manifest(self):
            raise PublicationResultArtifactError(
                reason="request_sha256 disagrees with the final request identity"
            )


@dataclass(frozen=True, slots=True)
class PublicationStudyResultArtifact:
    """One verified content-addressed complete publication result."""

    path: Path
    sha256: str
    manifest: PublicationStudyResultManifest

    def __post_init__(self) -> None:
        """Require a content-addressed path and typed manifest."""
        if not isinstance(self.path, Path):
            raise PublicationResultArtifactError(reason="path must be pathlib.Path")
        _require_sha256(self.sha256, field="sha256")
        if self.path.name != f"{self.sha256}.safetensors":
            raise PublicationResultArtifactError(reason="artifact path and content hash differ")
        if type(self.manifest) is not PublicationStudyResultManifest:
            raise PublicationResultArtifactError(
                reason="manifest must be PublicationStudyResultManifest"
            )


@dataclass(frozen=True, slots=True)
class LoadedPublicationStudyResultArtifact:
    """Verified target, aggregate results, and final artifact identity."""

    artifact: PublicationStudyResultArtifact
    target: Float64Vector
    result: PublicationStudyAggregateResult

    def __post_init__(self) -> None:
        """Require immutable row agreement."""
        if type(self.artifact) is not PublicationStudyResultArtifact:
            raise PublicationResultArtifactError(
                reason="artifact must be PublicationStudyResultArtifact"
            )
        _validate_vector(
            self.target,
            field="target",
            dtype=_FLOAT64_DTYPE,
            length=self.artifact.manifest.observations,
        )
        if self.target.flags.writeable:
            raise PublicationResultArtifactError(reason="target must be immutable")
        if type(self.result) is not PublicationStudyAggregateResult:
            raise PublicationResultArtifactError(
                reason="result must be PublicationStudyAggregateResult"
            )


def run_or_load_publication_study_result(  # noqa: PLR0913 - identities stay explicit
    directory: Path,
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    cash_artifacts: tuple[Path, ...],
    deterministic_execution: PublicationDeterministicExecutionSpec,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    replication_plan: PublicationReplicationPlanSpec = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
) -> LoadedPublicationStudyResultArtifact:
    """Exactly resume a final request or stream, aggregate, price yields, and persist it."""
    request_sha256 = publication_study_result_request_sha256(
        source,
        inputs,
        cash_artifacts,
        deterministic_execution,
        study=study,
        replication_plan=replication_plan,
    )
    loaded = load_publication_study_result_request(
        directory,
        request_sha256,
        study=study,
        replication_plan=replication_plan,
    )
    if loaded is not None:
        _require_resumed_result(loaded, source, inputs, cash_artifacts)
        return loaded
    cash, references = _aggregate_cash_artifacts(
        cash_artifacts,
        source,
        inputs,
        study,
        replication_plan,
    )
    yields = run_publication_yield_pricing(inputs, deterministic_execution, study)
    result = PublicationStudyAggregateResult(study=study, cash=cash, yields=yields)
    return _write_publication_study_result(
        directory,
        source,
        inputs,
        replication_plan,
        references,
        result,
    )


def publication_study_result_request_sha256(  # noqa: PLR0913 - identities stay explicit
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    cash_artifacts: tuple[Path, ...],
    deterministic_execution: PublicationDeterministicExecutionSpec,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    replication_plan: PublicationReplicationPlanSpec = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
) -> str:
    """Hash the complete final request known before loading replication payloads."""
    _validate_final_request(
        source,
        inputs,
        cash_artifacts,
        deterministic_execution,
        study,
        replication_plan,
    )
    software = _software_identity()
    return _request_sha256(
        _request_payload(
            source=source,
            study_sha256=publication_study_sha256(study),
            replication_plan_sha256=publication_replication_plan_sha256(replication_plan),
            deterministic_execution=deterministic_execution,
            deterministic_execution_sha256=publication_deterministic_execution_sha256(
                deterministic_execution
            ),
            partitions_sha256=publication_partitions_sha256(inputs),
            observations=inputs.original_rows,
            simulation_units=inputs.simulation_units,
            exact_groups=inputs.exact.groups,
            general_groups=inputs.general.groups,
            cash_artifact_sha256s=tuple(path.stem for path in cash_artifacts),
            variants=tuple(variant.value for variant in PublicationVariant),
            confidence_level=replication_plan.confidence_level,
            replication_critical_value=replication_plan.replication_critical_value,
            paper_package_version=software[0],
            engine_package_version=software[1],
            rustc_version=software[2],
        )
    )


def load_publication_study_result_request(
    directory: Path,
    request_sha256: str,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    replication_plan: PublicationReplicationPlanSpec = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
) -> LoadedPublicationStudyResultArtifact | None:
    """Load one immutable final request, returning ``None`` only when absent."""
    _validate_directory(directory)
    _require_sha256(request_sha256, field="request_sha256")
    receipt = directory / _REQUEST_DIRECTORY / f"{request_sha256}.json"
    if not receipt.exists():
        return None
    payload = _read_receipt(receipt)
    if payload["request_sha256"] != request_sha256:
        raise PublicationResultArtifactError(reason="request receipt filename and identity differ")
    loaded = load_publication_study_result_artifact(
        directory / f"{payload['artifact_sha256']}.safetensors",
        study=study,
        replication_plan=replication_plan,
    )
    if loaded.artifact.manifest.request_sha256 != request_sha256:
        raise PublicationResultArtifactError(reason="request receipt references another request")
    return loaded


def load_publication_study_result_artifact(
    path: Path,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    replication_plan: PublicationReplicationPlanSpec = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
) -> LoadedPublicationStudyResultArtifact:
    """Load and verify one self-contained complete publication result."""
    file_sha256 = _validate_load_request(path, study, replication_plan)
    metadata, tensors = _read_payload(path)
    manifest = _decode_manifest(metadata["manifest"])
    _validate_loaded_manifest(manifest, study, replication_plan)
    if set(tensors) != _expected_tensor_keys():
        raise PublicationResultArtifactError(reason="result tensor keys are missing or unexpected")
    if _tensors_sha256(tensors) != manifest.tensors_sha256:
        raise PublicationResultArtifactError(reason="result tensors disagree with the manifest")
    if _text_sha256(metadata["metrics"]) != manifest.metrics_sha256:
        raise PublicationResultArtifactError(reason="metrics disagree with the manifest")
    target, result = _decode_result(tensors, metadata["metrics"], manifest, study)
    artifact = PublicationStudyResultArtifact(path=path, sha256=file_sha256, manifest=manifest)
    return LoadedPublicationStudyResultArtifact(artifact=artifact, target=target, result=result)


@dataclass(frozen=True, slots=True)
class _CashArtifactReferences:
    artifact_sha256s: tuple[str, ...]
    request_sha256s: tuple[str, ...]
    execution_sha256s: tuple[str, ...]


def _aggregate_cash_artifacts(
    paths: tuple[Path, ...],
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    study: PublicationStudySpec,
    plan: PublicationReplicationPlanSpec,
) -> tuple[tuple[PublicationCashAggregateResult, ...], _CashArtifactReferences]:
    means = {
        variant: np.zeros(inputs.original_rows, dtype=np.float64) for variant in _CASH_VARIANTS
    }
    m2 = {variant: np.zeros(inputs.original_rows, dtype=np.float64) for variant in _CASH_VARIANTS}
    metrics = {variant: [] for variant in _CASH_VARIANTS}
    artifacts: list[PublicationCashArtifact] = []
    profile: tuple[int, CpuMathBackend] | None = None
    for count, path in enumerate(paths, start=1):
        loaded = load_publication_cash_artifact(path, study=study, replication_plan=plan)
        _validate_cash_prerequisite(
            loaded,
            source,
            inputs,
            expected_replication=count - 1,
        )
        observed_profile = (
            loaded.artifact.manifest.execution.threads,
            loaded.artifact.manifest.execution.math_backend,
        )
        if profile is None:
            profile = observed_profile
        elif profile != observed_profile:
            raise PublicationResultArtifactError(
                reason="cash artifacts do not share one execution profile"
            )
        artifacts.append(loaded.artifact)
        for variant in _CASH_VARIANTS:
            result = loaded.result.variant(variant)
            delta = result.native.prices - means[variant]
            means[variant] += delta / count
            m2[variant] += delta * (result.native.prices - means[variant])
            metrics[variant].append(result.metrics)
    cash = tuple(
        _cash_aggregate(
            variant,
            inputs.targets,
            means[variant],
            m2[variant],
            tuple(metrics[variant]),
            plan,
        )
        for variant in _CASH_VARIANTS
    )
    return cash, _CashArtifactReferences(
        artifact_sha256s=tuple(artifact.sha256 for artifact in artifacts),
        request_sha256s=tuple(artifact.manifest.request_sha256 for artifact in artifacts),
        execution_sha256s=tuple(artifact.manifest.execution_sha256 for artifact in artifacts),
    )


def _cash_aggregate(  # noqa: PLR0913 - aggregate inputs remain explicit
    variant: PublicationVariant,
    target: Float64Vector,
    mean: Float64Vector,
    m2: Float64Vector,
    metrics: tuple[ProjectPricingMetrics, ...],
    plan: PublicationReplicationPlanSpec,
) -> PublicationCashAggregateResult:
    if len(metrics) != plan.replications:
        raise PublicationResultArtifactError(reason="cash replication metrics are incomplete")
    sample_sd = np.sqrt(m2 / (plan.replications - 1), dtype=np.float64)
    mean.flags.writeable = False
    sample_sd.flags.writeable = False
    try:
        mean_metrics = evaluate_project_pricing_f64(target, mean)
    except ProjectMetricError as error:
        raise PublicationResultArtifactError(reason=str(error)) from error
    summaries = tuple(
        PublicationNamedMonteCarloSummary(
            metric=field,
            estimate=_monte_carlo_summary(
                tuple(float(getattr(metric, field)) for metric in metrics),
                plan,
            ),
        )
        for field in _METRIC_FIELDS
    )
    return PublicationCashAggregateResult(
        variant=variant,
        mean_prices=mean,
        replication_sample_standard_deviations=sample_sd,
        mean_prediction_metrics=mean_metrics,
        replication_metrics=metrics,
        metric_summaries=summaries,
    )


def _monte_carlo_summary(
    values: tuple[float, ...],
    plan: PublicationReplicationPlanSpec,
) -> PublicationMonteCarloSummary:
    if len(values) != plan.replications or any(not math.isfinite(value) for value in values):
        raise PublicationResultArtifactError(reason="Monte Carlo metric values are incomplete")
    mean = math.fsum(values) / len(values)
    squared_deviations = math.fsum((value - mean) ** 2 for value in values)
    sample_sd = math.sqrt(squared_deviations / (len(values) - 1))
    standard_error = sample_sd / math.sqrt(len(values))
    half_width = plan.replication_critical_value * standard_error
    return PublicationMonteCarloSummary(
        replications=len(values),
        mean=mean,
        sample_standard_deviation=sample_sd,
        standard_error=standard_error,
        interval_lower=mean - half_width,
        interval_upper=mean + half_width,
    )


def _write_publication_study_result(  # noqa: PLR0913 - identities stay explicit
    directory: Path,
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    plan: PublicationReplicationPlanSpec,
    references: _CashArtifactReferences,
    result: PublicationStudyAggregateResult,
) -> LoadedPublicationStudyResultArtifact:
    tensors = _result_tensors(inputs.targets, result)
    metrics_json = _metrics_json(result)
    software = _software_identity()
    execution = result.yields.execution
    request_payload = _request_payload(
        source=source,
        study_sha256=publication_study_sha256(result.study),
        replication_plan_sha256=publication_replication_plan_sha256(plan),
        deterministic_execution=execution,
        deterministic_execution_sha256=publication_deterministic_execution_sha256(execution),
        partitions_sha256=result.yields.partitions_sha256,
        observations=result.yields.rows,
        simulation_units=inputs.simulation_units,
        exact_groups=inputs.exact.groups,
        general_groups=inputs.general.groups,
        cash_artifact_sha256s=references.artifact_sha256s,
        variants=tuple(variant.value for variant in PublicationVariant),
        confidence_level=plan.confidence_level,
        replication_critical_value=plan.replication_critical_value,
        paper_package_version=software[0],
        engine_package_version=software[1],
        rustc_version=software[2],
    )
    manifest = PublicationStudyResultManifest(
        format_version=_FORMAT_VERSION,
        request_sha256=_request_sha256(request_payload),
        source=source,
        study_sha256=publication_study_sha256(result.study),
        replication_plan_sha256=publication_replication_plan_sha256(plan),
        deterministic_execution=execution,
        deterministic_execution_sha256=publication_deterministic_execution_sha256(execution),
        partitions_sha256=result.yields.partitions_sha256,
        observations=result.yields.rows,
        simulation_units=inputs.simulation_units,
        exact_groups=inputs.exact.groups,
        general_groups=inputs.general.groups,
        cash_artifact_sha256s=references.artifact_sha256s,
        cash_request_sha256s=references.request_sha256s,
        cash_execution_sha256s=references.execution_sha256s,
        variants=tuple(variant.value for variant in PublicationVariant),
        confidence_level=plan.confidence_level,
        replication_critical_value=plan.replication_critical_value,
        paper_package_version=software[0],
        engine_package_version=software[1],
        rustc_version=software[2],
        tensors_sha256=_tensors_sha256(tensors),
        metrics_sha256=_text_sha256(metrics_json),
    )
    envelope = {
        "format": _FORMAT,
        "manifest": _canonical_json(asdict(manifest)),
        "metrics": metrics_json,
    }
    destination = _persist_artifact(
        directory,
        tensors=tensors,
        metadata={_METADATA_KEY: _canonical_json(envelope)},
    )
    _persist_receipt(directory, manifest.request_sha256, destination.stem)
    loaded = load_publication_study_result_artifact(
        destination,
        study=result.study,
        replication_plan=plan,
    )
    _require_aggregate_exact(loaded.result, result)
    return loaded


def _result_tensors(
    target: Float64Vector,
    result: PublicationStudyAggregateResult,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {"target": target}
    for cash in result.cash:
        tensors[_tensor_key(cash.variant, "replication_mean_prices")] = cash.mean_prices
        tensors[_tensor_key(cash.variant, "replication_sample_standard_deviations")] = (
            cash.replication_sample_standard_deviations
        )
    for variant_result in result.yields.variants:
        native = cast("AndersenLakeBatchResult", variant_result.native)
        for field in _YIELD_FIELDS:
            tensors[_tensor_key(variant_result.variant, field)] = cast(
                "np.ndarray", getattr(native, field)
            )
    return tensors


def _metrics_json(result: PublicationStudyAggregateResult) -> str:
    payload = {
        "cash": {
            cash.variant.value: {
                "mean_prediction_metrics": asdict(cash.mean_prediction_metrics),
                "replication_metrics": tuple(asdict(metric) for metric in cash.replication_metrics),
                "metric_summaries": {
                    summary.metric: asdict(summary.estimate) for summary in cash.metric_summaries
                },
            }
            for cash in result.cash
        },
        "yield": {
            variant.variant.value: asdict(variant.metrics) for variant in result.yields.variants
        },
    }
    return _canonical_json(payload)


def _decode_result(
    tensors: dict[str, np.ndarray],
    metrics_json: str,
    manifest: PublicationStudyResultManifest,
    study: PublicationStudySpec,
) -> tuple[Float64Vector, PublicationStudyAggregateResult]:
    metrics = _decode_canonical_json(metrics_json, field="metrics")
    if type(metrics) is not dict or set(metrics) != {"cash", "yield"}:
        raise PublicationResultArtifactError(reason="metrics sections are missing or unexpected")
    sections = cast("dict[str, object]", metrics)
    target = cast(
        "Float64Vector",
        _loaded_vector(
            tensors["target"],
            field="target",
            dtype=_FLOAT64_DTYPE,
            length=manifest.observations,
        ),
    )
    cash = _decode_cash_results(tensors, target, sections["cash"], manifest)
    yields = _decode_yield_result(tensors, target, sections["yield"], manifest, study)
    return target, PublicationStudyAggregateResult(study=study, cash=cash, yields=yields)


def _decode_cash_results(
    tensors: dict[str, np.ndarray],
    target: Float64Vector,
    payload: object,
    manifest: PublicationStudyResultManifest,
) -> tuple[PublicationCashAggregateResult, ...]:
    if type(payload) is not dict or set(payload) != {variant.value for variant in _CASH_VARIANTS}:
        raise PublicationResultArtifactError(
            reason="cash metric variants are missing or unexpected"
        )
    values = cast("dict[str, object]", payload)
    return tuple(
        _decode_cash_result(tensors, target, values[variant.value], manifest, variant)
        for variant in _CASH_VARIANTS
    )


def _decode_cash_result(
    tensors: dict[str, np.ndarray],
    target: Float64Vector,
    payload: object,
    manifest: PublicationStudyResultManifest,
    variant: PublicationVariant,
) -> PublicationCashAggregateResult:
    expected = {"mean_prediction_metrics", "replication_metrics", "metric_summaries"}
    if type(payload) is not dict or set(payload) != expected:
        raise PublicationResultArtifactError(reason="cash metric keys are missing or unexpected")
    values = cast("dict[str, object]", payload)
    mean_metrics = _decode_project_metrics(values["mean_prediction_metrics"])
    raw_replications = values["replication_metrics"]
    if type(raw_replications) is not list or len(raw_replications) != len(
        manifest.cash_artifact_sha256s
    ):
        raise PublicationResultArtifactError(reason="cash replication metrics are incomplete")
    replication_metrics = tuple(_decode_project_metrics(item) for item in raw_replications)
    raw_summaries = values["metric_summaries"]
    if type(raw_summaries) is not dict or set(raw_summaries) != set(_METRIC_FIELDS):
        raise PublicationResultArtifactError(reason="cash metric summaries are incomplete")
    summary_values = cast("dict[str, object]", raw_summaries)
    summaries = tuple(
        PublicationNamedMonteCarloSummary(
            metric=field,
            estimate=_decode_monte_carlo_summary(summary_values[field]),
        )
        for field in _METRIC_FIELDS
    )
    mean = cast(
        "Float64Vector",
        _loaded_vector(
            tensors[_tensor_key(variant, "replication_mean_prices")],
            field="replication_mean_prices",
            dtype=_FLOAT64_DTYPE,
            length=manifest.observations,
        ),
    )
    sample_sd = cast(
        "Float64Vector",
        _loaded_vector(
            tensors[_tensor_key(variant, "replication_sample_standard_deviations")],
            field="replication_sample_standard_deviations",
            dtype=_FLOAT64_DTYPE,
            length=manifest.observations,
        ),
    )
    try:
        recomputed_metrics = evaluate_project_pricing_f64(target, mean)
    except ProjectMetricError as error:
        raise PublicationResultArtifactError(reason=str(error)) from error
    if recomputed_metrics != mean_metrics:
        raise PublicationResultArtifactError(reason="cash mean-prediction metrics disagree")
    plan = PublicationReplicationPlanSpec(
        replications=len(replication_metrics),
        seed_root=0,
        seed_increment=1,
        training_domain_start=0,
        evaluation_domain_start=1,
        confidence_level=manifest.confidence_level,
        replication_critical_value=manifest.replication_critical_value,
    )
    expected_summaries = tuple(
        PublicationNamedMonteCarloSummary(
            metric=field,
            estimate=_monte_carlo_summary(
                tuple(float(getattr(metric, field)) for metric in replication_metrics),
                plan,
            ),
        )
        for field in _METRIC_FIELDS
    )
    if summaries != expected_summaries:
        raise PublicationResultArtifactError(reason="cash Monte Carlo summaries disagree")
    return PublicationCashAggregateResult(
        variant=variant,
        mean_prices=mean,
        replication_sample_standard_deviations=sample_sd,
        mean_prediction_metrics=mean_metrics,
        replication_metrics=replication_metrics,
        metric_summaries=summaries,
    )


def _decode_yield_result(
    tensors: dict[str, np.ndarray],
    target: Float64Vector,
    payload: object,
    manifest: PublicationStudyResultManifest,
    study: PublicationStudySpec,
) -> PublicationYieldPricingResult:
    if type(payload) is not dict or set(payload) != {variant.value for variant in _YIELD_VARIANTS}:
        raise PublicationResultArtifactError(
            reason="yield metric variants are missing or unexpected"
        )
    metric_values = cast("dict[str, object]", payload)
    variants = tuple(
        _decode_yield_variant(tensors, target, metric_values[variant.value], manifest, variant)
        for variant in _YIELD_VARIANTS
    )
    return PublicationYieldPricingResult(
        study=study,
        execution=manifest.deterministic_execution,
        partitions_sha256=manifest.partitions_sha256,
        rows=manifest.observations,
        groups=manifest.simulation_units,
        variants=variants,
    )


def _decode_yield_variant(
    tensors: dict[str, np.ndarray],
    target: Float64Vector,
    metrics_payload: object,
    manifest: PublicationStudyResultManifest,
    variant: PublicationVariant,
) -> PublicationVariantReplicationResult:
    metrics = _decode_project_metrics(metrics_payload)

    def loaded(field: str, length: int) -> np.ndarray:
        return _loaded_vector(
            tensors[_tensor_key(variant, field)],
            field=_tensor_key(variant, field),
            dtype=_YIELD_DTYPES[field],
            length=length,
        )

    native = AndersenLakeBatchResult(
        prices=cast("Float64Vector", loaded("prices", manifest.observations)),
        exercised_at_origin=cast(
            "BoolVector", loaded("exercised_at_origin", manifest.observations)
        ),
        maximum_relative_boundary_residuals=cast(
            "Float64Vector",
            loaded("maximum_relative_boundary_residuals", manifest.simulation_units),
        ),
        maximum_relative_obstacle_projections=cast(
            "Float64Vector",
            loaded("maximum_relative_obstacle_projections", manifest.simulation_units),
        ),
        used_european_identity=cast(
            "BoolVector", loaded("used_european_identity", manifest.simulation_units)
        ),
    )
    try:
        recomputed = evaluate_project_pricing_f64(target, native.prices)
    except ProjectMetricError as error:
        raise PublicationResultArtifactError(reason=str(error)) from error
    if recomputed != metrics:
        raise PublicationResultArtifactError(reason="yield metrics disagree with stored prices")
    return PublicationVariantReplicationResult(variant=variant, native=native, metrics=metrics)


def _decode_project_metrics(payload: object) -> ProjectPricingMetrics:
    _require_exact_keys(payload, ProjectPricingMetrics, field="project metrics")
    try:
        return ProjectPricingMetrics(**cast("dict[str, Any]", payload))
    except (TypeError, ProjectMetricError) as error:
        raise PublicationResultArtifactError(
            reason=f"project metrics are invalid: {error}"
        ) from error


def _decode_monte_carlo_summary(payload: object) -> PublicationMonteCarloSummary:
    _require_exact_keys(payload, PublicationMonteCarloSummary, field="Monte Carlo summary")
    try:
        return PublicationMonteCarloSummary(**cast("dict[str, Any]", payload))
    except (TypeError, PublicationResultArtifactError) as error:
        raise PublicationResultArtifactError(
            reason=f"Monte Carlo summary is invalid: {error}"
        ) from error


def _validate_final_request(  # noqa: PLR0913 - independent identities stay explicit
    source: object,
    inputs: object,
    cash_artifacts: object,
    deterministic_execution: object,
    study: object,
    plan: object,
) -> None:
    if type(source) is not StudySourceIdentity:
        raise PublicationResultArtifactError(reason="source must be StudySourceIdentity")
    if type(inputs) is not PreparedPublicationPartitions:
        raise PublicationResultArtifactError(reason="inputs must be PreparedPublicationPartitions")
    if type(cash_artifacts) is not tuple:
        raise PublicationResultArtifactError(reason="cash_artifacts must be a tuple")
    if type(deterministic_execution) is not PublicationDeterministicExecutionSpec:
        raise PublicationResultArtifactError(
            reason="deterministic_execution must be PublicationDeterministicExecutionSpec"
        )
    if type(study) is not PublicationStudySpec:
        raise PublicationResultArtifactError(reason="study must be PublicationStudySpec")
    if type(plan) is not PublicationReplicationPlanSpec:
        raise PublicationResultArtifactError(reason="plan must be PublicationReplicationPlanSpec")
    _validate_final_request_values(inputs, cash_artifacts, study, plan)


def _validate_final_request_values(
    inputs: PreparedPublicationPartitions,
    cash_artifacts: tuple[object, ...],
    study: PublicationStudySpec,
    plan: PublicationReplicationPlanSpec,
) -> None:
    if inputs.study != study:
        raise PublicationResultArtifactError(reason="partition and requested studies differ")
    if len(cash_artifacts) != plan.replications:
        raise PublicationResultArtifactError(reason="cash artifact count differs from the plan")
    for path in cash_artifacts:
        if not isinstance(path, Path) or path.suffix != ".safetensors":
            raise PublicationResultArtifactError(reason="cash artifact paths must be .safetensors")
        _require_sha256(path.stem, field="cash artifact filename")
    plan.validate_units(inputs.simulation_units)


def _validate_cash_prerequisite(
    loaded: LoadedPublicationCashArtifact,
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    *,
    expected_replication: int,
) -> None:
    manifest = loaded.artifact.manifest
    if (
        manifest.source != source
        or manifest.partitions_sha256 != publication_partitions_sha256(inputs)
        or manifest.observations != inputs.original_rows
        or manifest.simulation_units != inputs.simulation_units
        or manifest.exact_groups != inputs.exact.groups
        or manifest.general_groups != inputs.general.groups
        or manifest.execution.replication != expected_replication
    ):
        raise PublicationResultArtifactError(reason="cash prerequisite identity differs")
    if not np.array_equal(loaded.target, inputs.targets):
        raise PublicationResultArtifactError(reason="cash prerequisite target differs")


def _request_payload(  # noqa: PLR0913 - complete final request stays explicit
    *,
    source: StudySourceIdentity,
    study_sha256: str,
    replication_plan_sha256: str,
    deterministic_execution: PublicationDeterministicExecutionSpec,
    deterministic_execution_sha256: str,
    partitions_sha256: str,
    observations: int,
    simulation_units: int,
    exact_groups: int,
    general_groups: int,
    cash_artifact_sha256s: tuple[str, ...],
    variants: tuple[str, ...],
    confidence_level: float,
    replication_critical_value: float,
    paper_package_version: str,
    engine_package_version: str,
    rustc_version: str,
) -> dict[str, object]:
    return {
        "source": asdict(source),
        "study_sha256": study_sha256,
        "replication_plan_sha256": replication_plan_sha256,
        "deterministic_execution": {
            "threads": deterministic_execution.threads,
            "math_backend": deterministic_execution.math_backend.value,
        },
        "deterministic_execution_sha256": deterministic_execution_sha256,
        "partitions_sha256": partitions_sha256,
        "observations": observations,
        "simulation_units": simulation_units,
        "exact_groups": exact_groups,
        "general_groups": general_groups,
        "cash_artifact_sha256s": cash_artifact_sha256s,
        "variants": variants,
        "confidence_level": confidence_level,
        "replication_critical_value": replication_critical_value,
        "paper_package_version": paper_package_version,
        "engine_package_version": engine_package_version,
        "rustc_version": rustc_version,
    }


def _request_sha256_from_manifest(manifest: PublicationStudyResultManifest) -> str:
    return _request_sha256(
        _request_payload(
            source=manifest.source,
            study_sha256=manifest.study_sha256,
            replication_plan_sha256=manifest.replication_plan_sha256,
            deterministic_execution=manifest.deterministic_execution,
            deterministic_execution_sha256=manifest.deterministic_execution_sha256,
            partitions_sha256=manifest.partitions_sha256,
            observations=manifest.observations,
            simulation_units=manifest.simulation_units,
            exact_groups=manifest.exact_groups,
            general_groups=manifest.general_groups,
            cash_artifact_sha256s=manifest.cash_artifact_sha256s,
            variants=manifest.variants,
            confidence_level=manifest.confidence_level,
            replication_critical_value=manifest.replication_critical_value,
            paper_package_version=manifest.paper_package_version,
            engine_package_version=manifest.engine_package_version,
            rustc_version=manifest.rustc_version,
        )
    )


def _request_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        _canonical_json({"format": _FORMAT, "request": payload}).encode("ascii")
    ).hexdigest()


def _validate_manifest_structure(manifest: PublicationStudyResultManifest) -> None:
    if type(manifest.format_version) is not int or manifest.format_version != _FORMAT_VERSION:
        raise PublicationResultArtifactError(reason="format_version must be the plain int 1")
    for field in (
        "request_sha256",
        "study_sha256",
        "replication_plan_sha256",
        "deterministic_execution_sha256",
        "partitions_sha256",
        "tensors_sha256",
        "metrics_sha256",
    ):
        _require_sha256(getattr(manifest, field), field=field)
    if type(manifest.source) is not StudySourceIdentity:
        raise PublicationResultArtifactError(reason="source must be StudySourceIdentity")
    if type(manifest.deterministic_execution) is not PublicationDeterministicExecutionSpec:
        raise PublicationResultArtifactError(reason="deterministic execution type is invalid")
    if manifest.deterministic_execution_sha256 != publication_deterministic_execution_sha256(
        manifest.deterministic_execution
    ):
        raise PublicationResultArtifactError(reason="deterministic execution hash disagrees")
    if manifest.variants != tuple(variant.value for variant in PublicationVariant):
        raise PublicationResultArtifactError(reason="variants are incomplete or reordered")


def _validate_manifest_population(manifest: PublicationStudyResultManifest) -> None:
    if type(manifest.observations) is not int or manifest.observations <= 0:
        raise PublicationResultArtifactError(reason="observations must be positive")
    if (
        type(manifest.simulation_units) is not int
        or not 0 < manifest.simulation_units <= manifest.observations
    ):
        raise PublicationResultArtifactError(reason="simulation_units are invalid")
    if (
        type(manifest.exact_groups) is not int
        or manifest.exact_groups < 0
        or type(manifest.general_groups) is not int
        or manifest.general_groups < 0
        or manifest.exact_groups + manifest.general_groups != manifest.simulation_units
    ):
        raise PublicationResultArtifactError(reason="cash domains do not partition groups")
    for field, value in (
        ("confidence_level", manifest.confidence_level),
        ("replication_critical_value", manifest.replication_critical_value),
    ):
        if type(value) is not float or not math.isfinite(value):
            raise PublicationResultArtifactError(reason=f"{field} must be finite")


def _validate_manifest_references(manifest: PublicationStudyResultManifest) -> None:
    lengths = {
        len(manifest.cash_artifact_sha256s),
        len(manifest.cash_request_sha256s),
        len(manifest.cash_execution_sha256s),
    }
    if len(lengths) != 1 or next(iter(lengths)) < _MIN_REPLICATIONS:
        raise PublicationResultArtifactError(reason="cash artifact references are incomplete")
    for field, values in (
        ("cash_artifact_sha256s", manifest.cash_artifact_sha256s),
        ("cash_request_sha256s", manifest.cash_request_sha256s),
        ("cash_execution_sha256s", manifest.cash_execution_sha256s),
    ):
        if type(values) is not tuple:
            raise PublicationResultArtifactError(reason=f"{field} must be a tuple")
        for value in values:
            _require_sha256(value, field=field)
        if len(set(values)) != len(values):
            raise PublicationResultArtifactError(reason=f"{field} contains aliases")


def _validate_manifest_versions(manifest: PublicationStudyResultManifest) -> None:
    for field in ("paper_package_version", "engine_package_version", "rustc_version"):
        value = getattr(manifest, field)
        if type(value) is not str or not value:
            raise PublicationResultArtifactError(reason=f"{field} must be nonempty")


def _validate_loaded_manifest(
    manifest: PublicationStudyResultManifest,
    study: PublicationStudySpec,
    plan: PublicationReplicationPlanSpec,
) -> None:
    if manifest.study_sha256 != publication_study_sha256(study):
        raise PublicationResultArtifactError(reason="artifact and requested studies differ")
    if manifest.replication_plan_sha256 != publication_replication_plan_sha256(plan):
        raise PublicationResultArtifactError(
            reason="artifact and requested replication plans differ"
        )
    if len(manifest.cash_artifact_sha256s) != plan.replications:
        raise PublicationResultArtifactError(reason="artifact replication count differs")
    if (
        manifest.confidence_level != plan.confidence_level
        or manifest.replication_critical_value != plan.replication_critical_value
    ):
        raise PublicationResultArtifactError(reason="artifact uncertainty identity differs")


def _validate_load_request(
    path: Path,
    study: object,
    plan: object,
) -> str:
    if not isinstance(path, Path):
        raise PublicationResultArtifactError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
        raise PublicationResultArtifactError(reason="path must be a regular .safetensors file")
    if type(study) is not PublicationStudySpec:
        raise PublicationResultArtifactError(reason="study must be PublicationStudySpec")
    if type(plan) is not PublicationReplicationPlanSpec:
        raise PublicationResultArtifactError(reason="plan must be PublicationReplicationPlanSpec")
    _require_sha256(path.stem, field="artifact filename")
    observed = _file_sha256(path)
    if observed != path.stem:
        raise PublicationResultArtifactError(reason="artifact bytes do not match filename hash")
    return observed


def _expected_tensor_keys() -> set[str]:
    return {
        "target",
        *(
            _tensor_key(variant, field)
            for variant in _CASH_VARIANTS
            for field in (
                "replication_mean_prices",
                "replication_sample_standard_deviations",
            )
        ),
        *(_tensor_key(variant, field) for variant in _YIELD_VARIANTS for field in _YIELD_FIELDS),
    }


def _tensor_key(variant: PublicationVariant, field: str) -> str:
    return f"{variant.value}.{field}"


def _persist_artifact(
    directory: Path,
    *,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> Path:
    _validate_directory(directory)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".publication-result-", suffix=".safetensors", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        _fsync_file(temporary)
        sha256 = _file_sha256(temporary)
        destination = directory / f"{sha256}.safetensors"
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _file_sha256(destination) != sha256:
                raise PublicationResultArtifactError(
                    reason="content-addressed destination contains different bytes"
                ) from None
        temporary.unlink()
        _fsync_directory(directory)
    except (OSError, SafetensorError) as error:
        raise PublicationResultArtifactError(reason=f"artifact write failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _persist_receipt(directory: Path, request_sha256: str, artifact_sha256: str) -> Path:
    _require_sha256(request_sha256, field="request_sha256")
    _require_sha256(artifact_sha256, field="artifact_sha256")
    receipt_directory = directory / _REQUEST_DIRECTORY
    try:
        receipt_directory.mkdir(mode=0o755, exist_ok=True)
    except OSError as error:
        raise PublicationResultArtifactError(
            reason=f"receipt directory creation failed: {error}"
        ) from error
    if not receipt_directory.is_dir() or receipt_directory.is_symlink():
        raise PublicationResultArtifactError(reason="request receipt directory is invalid")
    payload = _canonical_json(
        {
            "format": _RECEIPT_FORMAT,
            "request_sha256": request_sha256,
            "artifact_sha256": artifact_sha256,
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".publication-result-request-", suffix=".json", dir=receipt_directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        destination = receipt_directory / f"{request_sha256}.json"
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != temporary.read_bytes():
                raise PublicationResultArtifactError(
                    reason="immutable request receipt contains different bytes"
                ) from None
        temporary.unlink()
        _fsync_directory(receipt_directory)
    except OSError as error:
        raise PublicationResultArtifactError(reason=f"receipt write failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_payload(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    try:
        with safe_open(str(path), framework="numpy") as reader:
            raw_metadata = reader.metadata()
        tensors = load_file(path)
    except (OSError, SafetensorError) as error:
        raise PublicationResultArtifactError(
            reason=f"safe-tensors decode failed: {error}"
        ) from error
    if raw_metadata is None or set(raw_metadata) != {_METADATA_KEY}:
        raise PublicationResultArtifactError(reason="artifact metadata keys are unexpected")
    envelope = _decode_canonical_json(raw_metadata[_METADATA_KEY], field=_METADATA_KEY)
    if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
        raise PublicationResultArtifactError(reason="artifact envelope keys are unexpected")
    values = cast("dict[str, object]", envelope)
    if any(type(values[key]) is not str for key in _ENVELOPE_KEYS):
        raise PublicationResultArtifactError(reason="artifact envelope values must be strings")
    decoded = cast("dict[str, str]", values)
    if decoded["format"] != _FORMAT:
        raise PublicationResultArtifactError(reason="artifact format is unsupported")
    return decoded, tensors


def _read_receipt(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise PublicationResultArtifactError(reason="request receipt must be a regular file")
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise PublicationResultArtifactError(reason=f"receipt read failed: {error}") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise PublicationResultArtifactError(reason="receipt must be one canonical JSON line")
    payload = _decode_canonical_json(text[:-1], field="request receipt")
    if type(payload) is not dict or set(payload) != _RECEIPT_KEYS:
        raise PublicationResultArtifactError(reason="receipt keys are missing or unexpected")
    values = cast("dict[str, object]", payload)
    if values["format"] != _RECEIPT_FORMAT:
        raise PublicationResultArtifactError(reason="receipt format is unsupported")
    for field in ("request_sha256", "artifact_sha256"):
        _require_sha256(values[field], field=field)
    return cast("dict[str, str]", values)


def _decode_manifest(value: str) -> PublicationStudyResultManifest:
    payload = _decode_canonical_json(value, field="manifest")
    _require_exact_keys(payload, PublicationStudyResultManifest, field="manifest")
    values = cast("dict[str, Any]", payload)
    try:
        values["source"] = StudySourceIdentity(**cast("dict[str, Any]", values["source"]))
        execution = cast("dict[str, Any]", values["deterministic_execution"])
        execution["math_backend"] = CpuMathBackend(execution["math_backend"])
        values["deterministic_execution"] = PublicationDeterministicExecutionSpec(**execution)
        for field in (
            "cash_artifact_sha256s",
            "cash_request_sha256s",
            "cash_execution_sha256s",
            "variants",
        ):
            values[field] = tuple(cast("list[str]", values[field]))
        return PublicationStudyResultManifest(**values)
    except (TypeError, ValueError) as error:
        raise PublicationResultArtifactError(
            reason=f"manifest fields are invalid: {error}"
        ) from error


def _require_resumed_result(
    loaded: LoadedPublicationStudyResultArtifact,
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    cash_artifacts: tuple[Path, ...],
) -> None:
    manifest = loaded.artifact.manifest
    if (
        manifest.source != source
        or manifest.partitions_sha256 != publication_partitions_sha256(inputs)
        or manifest.cash_artifact_sha256s != tuple(path.stem for path in cash_artifacts)
        or manifest.observations != inputs.original_rows
        or manifest.simulation_units != inputs.simulation_units
    ):
        raise PublicationResultArtifactError(reason="resumed final result differs from request")
    if not np.array_equal(loaded.target, inputs.targets):
        raise PublicationResultArtifactError(reason="resumed target differs from current input")


def _require_aggregate_exact(
    observed: PublicationStudyAggregateResult,
    expected: PublicationStudyAggregateResult,
) -> None:
    if observed.study != expected.study or observed.yields.execution != expected.yields.execution:
        raise PublicationResultArtifactError(reason="persisted aggregate identity differs")
    for left, right in zip(observed.cash, expected.cash, strict=True):
        if (
            left.variant is not right.variant
            or left.mean_prediction_metrics != right.mean_prediction_metrics
            or left.replication_metrics != right.replication_metrics
            or left.metric_summaries != right.metric_summaries
            or not np.array_equal(left.mean_prices, right.mean_prices)
            or not np.array_equal(
                left.replication_sample_standard_deviations,
                right.replication_sample_standard_deviations,
            )
        ):
            raise PublicationResultArtifactError(reason="persisted cash aggregate differs")
    for left, right in zip(observed.yields.variants, expected.yields.variants, strict=True):
        if left.variant is not right.variant or left.metrics != right.metrics:
            raise PublicationResultArtifactError(reason="persisted yield identity differs")
        left_native = cast("AndersenLakeBatchResult", left.native)
        right_native = cast("AndersenLakeBatchResult", right.native)
        for field in _YIELD_FIELDS:
            if not np.array_equal(getattr(left_native, field), getattr(right_native, field)):
                raise PublicationResultArtifactError(reason="persisted yield diagnostics differ")


def _validate_directory(directory: Path) -> None:
    if not isinstance(directory, Path) or not directory.is_dir() or directory.is_symlink():
        raise PublicationResultArtifactError(
            reason="directory must be an existing non-symlink directory"
        )


def _loaded_vector(
    tensor: np.ndarray,
    *,
    field: str,
    dtype: np.dtype[np.generic],
    length: int,
) -> npt.NDArray[np.generic]:
    _validate_vector(tensor, field=field, dtype=dtype, length=length)
    result = np.ascontiguousarray(tensor, dtype=dtype).copy()
    result.flags.writeable = False
    return result


def _validate_vector(
    value: object,
    *,
    field: str,
    dtype: np.dtype[np.generic],
    length: int,
) -> None:
    if type(value) is not np.ndarray:
        raise PublicationResultArtifactError(reason=f"{field} must be a numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if array.dtype != dtype or array.shape != (length,) or not array.flags.c_contiguous:
        raise PublicationResultArtifactError(
            reason=f"{field} must be a C-contiguous {dtype} vector of length {length}"
        )
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise PublicationResultArtifactError(reason=f"{field} must be finite")


def _tensors_sha256(tensors: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    _hash_text(hasher, _FORMAT)
    for key in sorted(tensors):
        array = tensors[key]
        _hash_text(hasher, key)
        _hash_text(hasher, array.dtype.str)
        hasher.update(array.ndim.to_bytes(8, byteorder="little", signed=False))
        for dimension in array.shape:
            hasher.update(dimension.to_bytes(8, byteorder="little", signed=False))
        hasher.update(memoryview(array).cast("B"))
    return hasher.hexdigest()


def _software_identity() -> tuple[str, str, str]:
    try:
        paper_version = version(_PAPER_PACKAGE)
    except PackageNotFoundError as error:
        raise PublicationResultArtifactError(reason="paper package is not installed") from error
    native = build_info()
    if not paper_version or not native.package_version or not native.rustc_version:
        raise PublicationResultArtifactError(reason="software version identity is incomplete")
    return paper_version, native.package_version, native.rustc_version


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as error:
        raise PublicationResultArtifactError(reason=f"artifact hashing failed: {error}") from error
    return hasher.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _decode_canonical_json(value: object, *, field: str) -> object:
    if type(value) is not str:
        raise PublicationResultArtifactError(reason=f"{field} must be a string")
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise PublicationResultArtifactError(reason=f"{field} is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise PublicationResultArtifactError(reason=f"{field} JSON is not canonical")
    return payload


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PublicationResultArtifactError(
            reason=f"value cannot be canonical JSON: {error}"
        ) from error


def _require_exact_keys(value: object, model: type[object], *, field: str) -> None:
    if type(value) is not dict:
        raise PublicationResultArtifactError(reason=f"{field} must be an object")
    expected = {item.name for item in fields(model)}
    if set(cast("dict[object, object]", value)) != expected:
        raise PublicationResultArtifactError(reason=f"{field} keys are missing or unexpected")


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationResultArtifactError(reason=f"{field} must be a lowercase SHA-256 digest")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "LoadedPublicationStudyResultArtifact",
    "PublicationCashAggregateResult",
    "PublicationMonteCarloSummary",
    "PublicationNamedMonteCarloSummary",
    "PublicationResultArtifactError",
    "PublicationStudyAggregateResult",
    "PublicationStudyResultArtifact",
    "PublicationStudyResultManifest",
    "load_publication_study_result_artifact",
    "load_publication_study_result_request",
    "publication_study_result_request_sha256",
    "run_or_load_publication_study_result",
]
