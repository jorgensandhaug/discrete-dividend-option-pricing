"""Immutable artifact orchestration for publication date-panel empirical inference."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import numpy as np
import numpy.typing as npt
from finoptions import CpuExecutor, CpuMathBackend, build_info
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

from discrete_dividend_study.artifacts.study_features import (
    VerifiedStudyFeatureArtifact,
    verify_study_feature_artifact_unchanged,
)
from discrete_dividend_study.experiments.lsm_validation.inference import (
    PublicationBootstrapConfigurationResult,
    PublicationCohortInferenceResult,
    PublicationCohortPanelStatistics,
    PublicationEmpiricalInferenceResult,
    PublicationInferenceError,
    publication_additive_statistic_names,
    publication_prediction_set_from_result,
    run_publication_empirical_inference,
    summarize_publication_empirical_inference,
)
from discrete_dividend_study.experiments.lsm_validation.inference_specification import (
    PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
    PublicationBootstrapDesign,
    PublicationEmpiricalInferenceSpec,
    publication_bootstrap_configurations,
    publication_empirical_inference_sha256,
)
from discrete_dividend_study.experiments.lsm_validation.result_artifacts import (
    LoadedPublicationStudyResultArtifact,
)
from discrete_dividend_study.experiments.lsm_validation.specification import (
    PROJECT_PUBLICATION_STUDY_SPEC,
    PublicationStudySpec,
    PublicationVariant,
    publication_study_sha256,
)
from discrete_dividend_study.experiments.metrics import ProjectPricingMetrics
from discrete_dividend_study.experiments.neural.run import StudySourceIdentity

if TYPE_CHECKING:
    import polars as pl

Float64Matrix = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]

_FORMAT: Final = "discrete-dividend-publication-empirical-inference-v3"
_FORMAT_VERSION: Final = 3
_METADATA_KEY: Final = "publication_empirical_inference"
_ENVELOPE_KEYS: Final = frozenset({"format", "manifest", "summary"})
_REQUEST_DIRECTORY: Final = "publication-inference-requests"
_RECEIPT_FORMAT: Final = "discrete-dividend-publication-inference-request-receipt-v1"
_RECEIPT_KEYS: Final = frozenset({"format", "request_sha256", "artifact_sha256"})
_PAPER_PACKAGE: Final = "discrete-dividend-option-pricing"
_SHA256_LENGTH: Final = 64
_FLOAT64: Final = np.dtype(np.float64)
_INT64: Final = np.dtype(np.int64)
_VECTOR_DIMENSIONS: Final = 1
_MATRIX_DIMENSIONS: Final = 2
_FILE_HASH_BLOCK_BYTES: Final = 1024 * 1024


class _HashWriter(Protocol):
    """Streaming byte interface shared by hashlib implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(slots=True)
class PublicationInferenceArtifactError(RuntimeError):
    """The inference artifact or one of its immutable prerequisites is invalid."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected artifact contract."""
        return f"invalid publication inference artifact: {self.reason}"


@dataclass(frozen=True, slots=True)
class PublicationInferenceCohortManifest:
    """Persisted cohort population dimensions."""

    name: str
    observations: int
    quote_date_panels: int

    def __post_init__(self) -> None:
        """Require a named nonempty empirical population."""
        if type(self.name) is not str or not self.name:
            raise PublicationInferenceArtifactError(reason="cohort name must be nonempty")
        if type(self.observations) is not int or self.observations <= 0:
            raise PublicationInferenceArtifactError(reason="cohort observations must be positive")
        if type(self.quote_date_panels) is not int or self.quote_date_panels <= 0:
            raise PublicationInferenceArtifactError(
                reason="cohort quote-date panels must be positive"
            )


@dataclass(frozen=True, slots=True)
class PublicationInferenceManifest:
    """Complete source, result, method, execution, and payload identity."""

    format_version: int
    request_sha256: str
    source: StudySourceIdentity
    publication_result_source: StudySourceIdentity
    feature_receipt_sha256: str
    publication_result_sha256: str
    publication_result_request_sha256: str
    study_sha256: str
    inference_spec_sha256: str
    partitions_sha256: str
    observations: int
    cohorts: tuple[PublicationInferenceCohortManifest, ...]
    bootstrap_replications: int
    bootstrap_designs: tuple[str, ...]
    mean_block_lengths: tuple[float, ...]
    bootstrap_domains: tuple[int, ...]
    executor_threads: int
    executor_math_backend: str
    paper_package_version: str
    engine_package_version: str
    rustc_version: str
    tensors_sha256: str
    summary_sha256: str

    def __post_init__(self) -> None:
        """Reject incomplete manifests and request hashes that do not reproduce."""
        _validate_manifest(self)
        if self.request_sha256 != _request_sha256(_manifest_request_payload(self)):
            raise PublicationInferenceArtifactError(
                reason="request_sha256 disagrees with manifest request identity"
            )


@dataclass(frozen=True, slots=True)
class PublicationInferenceArtifact:
    """One content-addressed verified inference artifact."""

    path: Path
    sha256: str
    manifest: PublicationInferenceManifest

    def __post_init__(self) -> None:
        """Require a hash-named safe-tensors path and typed manifest."""
        if not isinstance(self.path, Path):
            raise PublicationInferenceArtifactError(reason="artifact path must be pathlib.Path")
        _require_sha256(self.sha256, field="artifact sha256")
        if self.path.name != f"{self.sha256}.safetensors":
            raise PublicationInferenceArtifactError(reason="artifact path and SHA-256 differ")
        if type(self.manifest) is not PublicationInferenceManifest:
            raise PublicationInferenceArtifactError(reason="manifest has an invalid type")


@dataclass(frozen=True, slots=True, eq=False)
class LoadedPublicationInferenceArtifact:
    """Verified raw bootstrap tensors, summaries, and artifact identity."""

    artifact: PublicationInferenceArtifact
    result: PublicationEmpiricalInferenceResult

    def __post_init__(self) -> None:
        """Require typed immutable state."""
        if type(self.artifact) is not PublicationInferenceArtifact:
            raise PublicationInferenceArtifactError(reason="artifact has an invalid type")
        if type(self.result) is not PublicationEmpiricalInferenceResult:
            raise PublicationInferenceArtifactError(reason="result has an invalid type")


def run_or_load_publication_empirical_inference(  # noqa: PLR0913 - identities stay explicit
    directory: Path,
    source: StudySourceIdentity,
    feature: VerifiedStudyFeatureArtifact,
    publication_result: LoadedPublicationStudyResultArtifact,
    frame: pl.DataFrame,
    partitions_sha256: str,
    executor: CpuExecutor,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    spec: PublicationEmpiricalInferenceSpec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
) -> LoadedPublicationInferenceArtifact:
    """Exactly resume an inference request or compute, verify, and persist every block draw."""
    request_sha256 = publication_inference_request_sha256(
        source,
        feature,
        publication_result,
        partitions_sha256,
        executor,
        study=study,
        spec=spec,
    )
    loaded = load_publication_inference_request(directory, request_sha256, study=study, spec=spec)
    if loaded is not None:
        _require_resumed_identity(
            loaded,
            source,
            feature,
            publication_result,
            partitions_sha256,
            executor,
        )
        return loaded
    verify_study_feature_artifact_unchanged(feature)
    _verify_publication_result_unchanged(publication_result)
    predictions = publication_prediction_set_from_result(publication_result)
    try:
        result = run_publication_empirical_inference(frame, predictions, executor, spec)
    except PublicationInferenceError as error:
        raise PublicationInferenceArtifactError(reason=str(error)) from error
    verify_study_feature_artifact_unchanged(feature)
    _verify_publication_result_unchanged(publication_result)
    return _write_inference_artifact(
        directory,
        source,
        feature,
        publication_result,
        partitions_sha256,
        executor,
        result,
        study,
    )


def publication_inference_request_sha256(  # noqa: PLR0913 - identities stay explicit
    source: StudySourceIdentity,
    feature: VerifiedStudyFeatureArtifact,
    publication_result: LoadedPublicationStudyResultArtifact,
    partitions_sha256: str,
    executor: CpuExecutor,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    spec: PublicationEmpiricalInferenceSpec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
) -> str:
    """Hash every request value known before panel aggregation or resampling."""
    _validate_request(
        source,
        feature,
        publication_result,
        partitions_sha256,
        executor,
        study,
        spec,
    )
    software = _software_identity()
    return _request_sha256(
        _request_payload(
            source=source,
            publication_result_source=publication_result.artifact.manifest.source,
            feature_receipt_sha256=feature.receipt_sha256,
            publication_result_sha256=publication_result.artifact.sha256,
            publication_result_request_sha256=(publication_result.artifact.manifest.request_sha256),
            study_sha256=publication_study_sha256(study),
            inference_spec_sha256=publication_empirical_inference_sha256(spec),
            partitions_sha256=partitions_sha256,
            observations=publication_result.artifact.manifest.observations,
            bootstrap_replications=spec.bootstrap.replications,
            bootstrap_designs=tuple(
                design.value for design, _ in publication_bootstrap_configurations(spec)
            ),
            mean_block_lengths=tuple(
                mean for _, mean in publication_bootstrap_configurations(spec)
            ),
            executor_threads=executor.threads,
            executor_math_backend=executor.math_backend.value,
            paper_package_version=software[0],
            engine_package_version=software[1],
            rustc_version=software[2],
        )
    )


def load_publication_inference_request(
    directory: Path,
    request_sha256: str,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    spec: PublicationEmpiricalInferenceSpec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
) -> LoadedPublicationInferenceArtifact | None:
    """Load one request-addressed artifact, returning ``None`` only when absent."""
    _validate_directory(directory)
    _require_sha256(request_sha256, field="request_sha256")
    receipt_path = directory / _REQUEST_DIRECTORY / f"{request_sha256}.json"
    if not receipt_path.exists():
        return None
    receipt = _read_receipt(receipt_path)
    if receipt["request_sha256"] != request_sha256:
        raise PublicationInferenceArtifactError(
            reason="request receipt filename and payload differ"
        )
    loaded = load_publication_inference_artifact(
        directory / f"{receipt['artifact_sha256']}.safetensors",
        study=study,
        spec=spec,
    )
    if loaded.artifact.manifest.request_sha256 != request_sha256:
        raise PublicationInferenceArtifactError(reason="receipt references another request")
    return loaded


def load_publication_inference_artifact(
    path: Path,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    spec: PublicationEmpiricalInferenceSpec = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
) -> LoadedPublicationInferenceArtifact:
    """Load and fully verify a self-contained inference artifact."""
    file_sha256 = _validate_load_request(path, study, spec)
    metadata, tensors = _read_payload(path)
    manifest = _decode_manifest(metadata["manifest"])
    _validate_loaded_manifest(manifest, study, spec)
    if set(tensors) != _expected_tensor_keys(manifest):
        raise PublicationInferenceArtifactError(
            reason="inference tensor keys are missing or unexpected"
        )
    if _tensors_sha256(tensors) != manifest.tensors_sha256:
        raise PublicationInferenceArtifactError(reason="tensor payload hash differs")
    if _text_sha256(metadata["summary"]) != manifest.summary_sha256:
        raise PublicationInferenceArtifactError(reason="summary payload hash differs")
    result = _decode_result(tensors, metadata["summary"], manifest, spec)
    artifact = PublicationInferenceArtifact(path=path, sha256=file_sha256, manifest=manifest)
    return LoadedPublicationInferenceArtifact(artifact=artifact, result=result)


def _write_inference_artifact(  # noqa: PLR0913 - identities stay explicit
    directory: Path,
    source: StudySourceIdentity,
    feature: VerifiedStudyFeatureArtifact,
    publication_result: LoadedPublicationStudyResultArtifact,
    partitions_sha256: str,
    executor: CpuExecutor,
    result: PublicationEmpiricalInferenceResult,
    study: PublicationStudySpec,
) -> LoadedPublicationInferenceArtifact:
    tensors = _result_tensors(result)
    summary = _summary_json(result)
    software = _software_identity()
    configuration_identity = publication_bootstrap_configurations(result.spec)
    base = _request_payload(
        source=source,
        publication_result_source=publication_result.artifact.manifest.source,
        feature_receipt_sha256=feature.receipt_sha256,
        publication_result_sha256=publication_result.artifact.sha256,
        publication_result_request_sha256=publication_result.artifact.manifest.request_sha256,
        study_sha256=publication_study_sha256(study),
        inference_spec_sha256=publication_empirical_inference_sha256(result.spec),
        partitions_sha256=partitions_sha256,
        observations=publication_result.artifact.manifest.observations,
        bootstrap_replications=result.spec.bootstrap.replications,
        bootstrap_designs=tuple(design.value for design, _ in configuration_identity),
        mean_block_lengths=tuple(mean for _, mean in configuration_identity),
        executor_threads=executor.threads,
        executor_math_backend=executor.math_backend.value,
        paper_package_version=software[0],
        engine_package_version=software[1],
        rustc_version=software[2],
    )
    manifest = PublicationInferenceManifest(
        format_version=_FORMAT_VERSION,
        request_sha256=_request_sha256(base),
        source=source,
        publication_result_source=publication_result.artifact.manifest.source,
        feature_receipt_sha256=feature.receipt_sha256,
        publication_result_sha256=publication_result.artifact.sha256,
        publication_result_request_sha256=publication_result.artifact.manifest.request_sha256,
        study_sha256=publication_study_sha256(study),
        inference_spec_sha256=publication_empirical_inference_sha256(result.spec),
        partitions_sha256=partitions_sha256,
        observations=publication_result.artifact.manifest.observations,
        cohorts=tuple(
            PublicationInferenceCohortManifest(
                name=cohort.panels.cohort.name,
                observations=cohort.panels.observations,
                quote_date_panels=cohort.panels.quote_date_panels,
            )
            for cohort in result.cohorts
        ),
        bootstrap_replications=result.spec.bootstrap.replications,
        bootstrap_designs=tuple(design.value for design, _ in configuration_identity),
        mean_block_lengths=tuple(mean for _, mean in configuration_identity),
        bootstrap_domains=tuple(
            configuration.domain for cohort in result.cohorts for configuration in cohort.bootstrap
        ),
        executor_threads=executor.threads,
        executor_math_backend=executor.math_backend.value,
        paper_package_version=software[0],
        engine_package_version=software[1],
        rustc_version=software[2],
        tensors_sha256=_tensors_sha256(tensors),
        summary_sha256=_text_sha256(summary),
    )
    envelope = {
        "format": _FORMAT,
        "manifest": _canonical_json(asdict(manifest)),
        "summary": summary,
    }
    destination = _persist_artifact(
        directory,
        tensors=tensors,
        metadata={_METADATA_KEY: _canonical_json(envelope)},
    )
    _persist_receipt(directory, manifest.request_sha256, destination.stem)
    loaded = load_publication_inference_artifact(destination, study=study, spec=result.spec)
    _require_result_exact(loaded.result, result)
    return loaded


def _result_tensors(result: PublicationEmpiricalInferenceResult) -> dict[str, np.ndarray]:
    tensors = {}
    for cohort_index, cohort in enumerate(result.cohorts):
        tensors[_panel_tensor_key(cohort_index)] = cohort.panels.additive_statistics
        tensors[_panel_year_tensor_key(cohort_index)] = cohort.panels.panel_years
        for block_index, configuration in enumerate(cohort.bootstrap):
            tensors[_bootstrap_tensor_key(cohort_index, block_index)] = configuration.additive_sums
    return tensors


def _summary_json(result: PublicationEmpiricalInferenceResult) -> str:
    payload = {
        "additive_statistic_names": list(publication_additive_statistic_names()),
        "point_metrics": [
            {
                "cohort": cohort.panels.cohort.name,
                "variants": {
                    variant.value: asdict(metrics)
                    for variant, metrics in zip(
                        PublicationVariant,
                        cohort.panels.point_metrics,
                        strict=True,
                    )
                },
            }
            for cohort in result.cohorts
        ],
        "model_intervals": [
            {
                "cohort": item.cohort,
                "bootstrap_design": item.bootstrap_design.value,
                "mean_block_length": item.mean_block_length,
                "variant": item.variant.value,
                "metric": item.metric.value,
                "estimate": item.estimate,
                "interval_lower": item.interval_lower,
                "interval_upper": item.interval_upper,
            }
            for item in result.model_intervals
        ],
        "contrasts": [
            {
                "cohort": item.cohort,
                "bootstrap_design": item.bootstrap_design.value,
                "mean_block_length": item.mean_block_length,
                "left": item.contrast.left.value,
                "right": item.contrast.right.value,
                "metric": item.metric.value,
                "estimate": item.estimate,
                "interval_lower": item.interval_lower,
                "interval_upper": item.interval_upper,
                "p_value": item.p_value,
                "holm_adjusted_p_value": item.holm_adjusted_p_value,
                "reject_family_wise_null": item.reject_family_wise_null,
            }
            for item in result.contrasts
        ],
    }
    return _canonical_json(payload)


def _decode_result(
    tensors: dict[str, np.ndarray],
    summary: str,
    manifest: PublicationInferenceManifest,
    spec: PublicationEmpiricalInferenceSpec,
) -> PublicationEmpiricalInferenceResult:
    payload = _decode_canonical_json(summary, field="summary")
    if type(payload) is not dict or set(payload) != {
        "additive_statistic_names",
        "point_metrics",
        "model_intervals",
        "contrasts",
    }:
        raise PublicationInferenceArtifactError(reason="summary sections are invalid")
    values = cast("dict[str, object]", payload)
    if values["additive_statistic_names"] != list(publication_additive_statistic_names()):
        raise PublicationInferenceArtifactError(reason="additive statistic identity differs")
    point_metrics = _decode_point_metrics(values["point_metrics"], manifest, spec)
    cohort_results = []
    domains = iter(manifest.bootstrap_domains)
    for cohort_index, (cohort_manifest, cohort_spec, metrics) in enumerate(
        zip(manifest.cohorts, spec.cohorts, point_metrics, strict=True)
    ):
        panels = _loaded_matrix(
            tensors[_panel_tensor_key(cohort_index)],
            field="panel statistics",
            shape=(cohort_manifest.quote_date_panels, len(publication_additive_statistic_names())),
        )
        prepared = PublicationCohortPanelStatistics(
            cohort=cohort_spec,
            observations=cohort_manifest.observations,
            quote_date_panels=cohort_manifest.quote_date_panels,
            point_metrics=metrics,
            panel_years=_loaded_int64_vector(
                tensors[_panel_year_tensor_key(cohort_index)],
                field="panel years",
                length=cohort_manifest.quote_date_panels,
            ),
            additive_statistics=panels,
        )
        configurations = tuple(
            PublicationBootstrapConfigurationResult(
                design=PublicationBootstrapDesign(design),
                mean_block_length=mean,
                domain=next(domains),
                additive_sums=_loaded_matrix(
                    tensors[_bootstrap_tensor_key(cohort_index, block_index)],
                    field="bootstrap sums",
                    shape=(
                        manifest.bootstrap_replications,
                        len(publication_additive_statistic_names()),
                    ),
                ),
            )
            for block_index, (design, mean) in enumerate(
                zip(manifest.bootstrap_designs, manifest.mean_block_lengths, strict=True)
            )
        )
        cohort_results.append(
            PublicationCohortInferenceResult(panels=prepared, bootstrap=configurations)
        )
    try:
        next(domains)
    except StopIteration:
        pass
    else:
        raise PublicationInferenceArtifactError(reason="bootstrap domains contain extra values")
    cohorts = tuple(cohort_results)
    intervals, contrasts = summarize_publication_empirical_inference(cohorts, spec)
    result = PublicationEmpiricalInferenceResult(
        spec=spec,
        cohorts=cohorts,
        model_intervals=intervals,
        contrasts=contrasts,
    )
    if _summary_json(result) != summary:
        raise PublicationInferenceArtifactError(
            reason="stored summaries differ from raw bootstrap tensors"
        )
    return result


def _decode_point_metrics(
    payload: object,
    manifest: PublicationInferenceManifest,
    spec: PublicationEmpiricalInferenceSpec,
) -> tuple[tuple[ProjectPricingMetrics, ...], ...]:
    if type(payload) is not list or len(payload) != len(spec.cohorts):
        raise PublicationInferenceArtifactError(reason="point metric cohorts are incomplete")
    decoded = []
    for item, cohort_manifest, cohort_spec in zip(
        payload, manifest.cohorts, spec.cohorts, strict=True
    ):
        if type(item) is not dict or set(item) != {"cohort", "variants"}:
            raise PublicationInferenceArtifactError(reason="point metric cohort keys are invalid")
        values = cast("dict[str, object]", item)
        if values["cohort"] != cohort_spec.name or cohort_manifest.name != cohort_spec.name:
            raise PublicationInferenceArtifactError(reason="point metric cohort identity differs")
        variants = values["variants"]
        if type(variants) is not dict or set(variants) != {
            variant.value for variant in PublicationVariant
        }:
            raise PublicationInferenceArtifactError(reason="point metric variants are incomplete")
        variant_values = cast("dict[str, object]", variants)
        decoded.append(
            tuple(
                _decode_project_metrics(variant_values[variant.value])
                for variant in PublicationVariant
            )
        )
    return tuple(decoded)


def _decode_project_metrics(value: object) -> ProjectPricingMetrics:
    _require_exact_keys(value, ProjectPricingMetrics, field="point metrics")
    try:
        return ProjectPricingMetrics(**cast("dict[str, Any]", value))
    except (TypeError, ValueError) as error:
        raise PublicationInferenceArtifactError(
            reason=f"point metrics are invalid: {error}"
        ) from error


def _validate_request(  # noqa: C901, PLR0913 - precedence remains auditable
    source: object,
    feature: object,
    publication_result: object,
    partitions_sha256: object,
    executor: object,
    study: object,
    spec: object,
) -> None:
    if type(source) is not StudySourceIdentity:
        raise PublicationInferenceArtifactError(reason="source must be StudySourceIdentity")
    if type(feature) is not VerifiedStudyFeatureArtifact:
        raise PublicationInferenceArtifactError(reason="feature handle has an invalid type")
    if type(publication_result) is not LoadedPublicationStudyResultArtifact:
        raise PublicationInferenceArtifactError(reason="publication result has an invalid type")
    _require_sha256(partitions_sha256, field="partitions_sha256")
    if type(executor) is not CpuExecutor:
        raise PublicationInferenceArtifactError(reason="executor must be CpuExecutor")
    if type(study) is not PublicationStudySpec:
        raise PublicationInferenceArtifactError(reason="study must be PublicationStudySpec")
    if type(spec) is not PublicationEmpiricalInferenceSpec:
        raise PublicationInferenceArtifactError(
            reason="spec must be PublicationEmpiricalInferenceSpec"
        )
    result_manifest = publication_result.artifact.manifest
    if result_manifest.study_sha256 != publication_study_sha256(study):
        raise PublicationInferenceArtifactError(reason="publication study identity differs")
    if result_manifest.partitions_sha256 != partitions_sha256:
        raise PublicationInferenceArtifactError(reason="publication partition identity differs")
    for observed_source in (source, result_manifest.source):
        if (
            observed_source.feature_artifact_sha256 != feature.output_sha256
            or observed_source.feature_schema_sha256 != feature.feature_schema_sha256
        ):
            raise PublicationInferenceArtifactError(reason="feature source identity differs")
    _verify_publication_result_unchanged(publication_result)


def _validate_manifest(  # noqa: C901, PLR0912 - manifest fields remain locally auditable
    manifest: PublicationInferenceManifest,
) -> None:
    if manifest.format_version != _FORMAT_VERSION:
        raise PublicationInferenceArtifactError(reason="manifest format version is unsupported")
    for field in (
        "request_sha256",
        "feature_receipt_sha256",
        "publication_result_sha256",
        "publication_result_request_sha256",
        "study_sha256",
        "inference_spec_sha256",
        "partitions_sha256",
        "tensors_sha256",
        "summary_sha256",
    ):
        _require_sha256(getattr(manifest, field), field=field)
    if (
        type(manifest.source) is not StudySourceIdentity
        or type(manifest.publication_result_source) is not StudySourceIdentity
    ):
        raise PublicationInferenceArtifactError(reason="manifest sources are invalid")
    if type(manifest.observations) is not int or manifest.observations <= 0:
        raise PublicationInferenceArtifactError(reason="manifest observations must be positive")
    if type(manifest.cohorts) is not tuple or not manifest.cohorts:
        raise PublicationInferenceArtifactError(reason="manifest cohorts cannot be empty")
    if type(manifest.bootstrap_replications) is not int or manifest.bootstrap_replications <= 0:
        raise PublicationInferenceArtifactError(
            reason="manifest bootstrap replications must be positive"
        )
    if (
        type(manifest.bootstrap_designs) is not tuple
        or not manifest.bootstrap_designs
        or len(manifest.bootstrap_designs) != len(manifest.mean_block_lengths)
    ):
        raise PublicationInferenceArtifactError(reason="manifest bootstrap designs are invalid")
    try:
        tuple(PublicationBootstrapDesign(value) for value in manifest.bootstrap_designs)
    except ValueError as error:
        raise PublicationInferenceArtifactError(
            reason="manifest bootstrap design value is invalid"
        ) from error
    if (
        type(manifest.mean_block_lengths) is not tuple
        or not manifest.mean_block_lengths
        or any(
            type(value) is not float or not math.isfinite(value) or value < 1.0
            for value in manifest.mean_block_lengths
        )
    ):
        raise PublicationInferenceArtifactError(reason="manifest block lengths are invalid")
    expected_domains = len(manifest.cohorts) * len(manifest.mean_block_lengths)
    if (
        type(manifest.bootstrap_domains) is not tuple
        or len(manifest.bootstrap_domains) != expected_domains
        or any(
            type(value) is not int or not 0 <= value <= (2**32) - 1
            for value in manifest.bootstrap_domains
        )
        or len(set(manifest.bootstrap_domains)) != expected_domains
    ):
        raise PublicationInferenceArtifactError(reason="manifest bootstrap domains are invalid")
    if type(manifest.executor_threads) is not int or manifest.executor_threads <= 0:
        raise PublicationInferenceArtifactError(reason="executor threads must be positive")
    try:
        CpuMathBackend(manifest.executor_math_backend)
    except ValueError as error:
        raise PublicationInferenceArtifactError(
            reason="executor math backend is invalid"
        ) from error
    for field in ("paper_package_version", "engine_package_version", "rustc_version"):
        if type(getattr(manifest, field)) is not str or not getattr(manifest, field):
            raise PublicationInferenceArtifactError(reason=f"{field} must be nonempty")


def _validate_loaded_manifest(
    manifest: PublicationInferenceManifest,
    study: PublicationStudySpec,
    spec: PublicationEmpiricalInferenceSpec,
) -> None:
    if manifest.study_sha256 != publication_study_sha256(study):
        raise PublicationInferenceArtifactError(reason="loaded study identity differs")
    if manifest.inference_spec_sha256 != publication_empirical_inference_sha256(spec):
        raise PublicationInferenceArtifactError(reason="loaded inference specification differs")
    if manifest.bootstrap_replications != spec.bootstrap.replications:
        raise PublicationInferenceArtifactError(reason="loaded replication count differs")
    configuration_identity = publication_bootstrap_configurations(spec)
    if manifest.bootstrap_designs != tuple(
        design.value for design, _ in configuration_identity
    ) or manifest.mean_block_lengths != tuple(mean for _, mean in configuration_identity):
        raise PublicationInferenceArtifactError(reason="loaded bootstrap configurations differ")
    if tuple(item.name for item in manifest.cohorts) != tuple(
        cohort.name for cohort in spec.cohorts
    ):
        raise PublicationInferenceArtifactError(reason="loaded cohorts differ")
    ordinary_blocks = len(spec.bootstrap.sensitivity_mean_block_lengths)
    expected_domains = tuple(
        domain
        for cohort_index in range(len(spec.cohorts))
        for domain in (
            *(
                spec.bootstrap.domain_start + cohort_index * ordinary_blocks + block_index
                for block_index in range(ordinary_blocks)
            ),
            spec.bootstrap.stratified_sensitivity_domain_start + cohort_index,
        )
    )
    if manifest.bootstrap_domains != expected_domains:
        raise PublicationInferenceArtifactError(reason="loaded bootstrap domain mapping differs")


def _request_payload(  # noqa: PLR0913 - immutable identity remains explicit
    *,
    source: StudySourceIdentity,
    publication_result_source: StudySourceIdentity,
    feature_receipt_sha256: str,
    publication_result_sha256: str,
    publication_result_request_sha256: str,
    study_sha256: str,
    inference_spec_sha256: str,
    partitions_sha256: str,
    observations: int,
    bootstrap_replications: int,
    bootstrap_designs: tuple[str, ...],
    mean_block_lengths: tuple[float, ...],
    executor_threads: int,
    executor_math_backend: str,
    paper_package_version: str,
    engine_package_version: str,
    rustc_version: str,
) -> dict[str, object]:
    return {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "source": asdict(source),
        "publication_result_source": asdict(publication_result_source),
        "feature_receipt_sha256": feature_receipt_sha256,
        "publication_result_sha256": publication_result_sha256,
        "publication_result_request_sha256": publication_result_request_sha256,
        "study_sha256": study_sha256,
        "inference_spec_sha256": inference_spec_sha256,
        "partitions_sha256": partitions_sha256,
        "observations": observations,
        "bootstrap_replications": bootstrap_replications,
        "bootstrap_designs": list(bootstrap_designs),
        "mean_block_lengths": list(mean_block_lengths),
        "executor_threads": executor_threads,
        "executor_math_backend": executor_math_backend,
        "paper_package_version": paper_package_version,
        "engine_package_version": engine_package_version,
        "rustc_version": rustc_version,
    }


def _manifest_request_payload(manifest: PublicationInferenceManifest) -> dict[str, object]:
    return _request_payload(
        source=manifest.source,
        publication_result_source=manifest.publication_result_source,
        feature_receipt_sha256=manifest.feature_receipt_sha256,
        publication_result_sha256=manifest.publication_result_sha256,
        publication_result_request_sha256=manifest.publication_result_request_sha256,
        study_sha256=manifest.study_sha256,
        inference_spec_sha256=manifest.inference_spec_sha256,
        partitions_sha256=manifest.partitions_sha256,
        observations=manifest.observations,
        bootstrap_replications=manifest.bootstrap_replications,
        bootstrap_designs=manifest.bootstrap_designs,
        mean_block_lengths=manifest.mean_block_lengths,
        executor_threads=manifest.executor_threads,
        executor_math_backend=manifest.executor_math_backend,
        paper_package_version=manifest.paper_package_version,
        engine_package_version=manifest.engine_package_version,
        rustc_version=manifest.rustc_version,
    )


def _request_sha256(payload: dict[str, object]) -> str:
    return _text_sha256(_canonical_json(payload))


def _expected_tensor_keys(manifest: PublicationInferenceManifest) -> set[str]:
    return {
        *(_panel_tensor_key(index) for index in range(len(manifest.cohorts))),
        *(_panel_year_tensor_key(index) for index in range(len(manifest.cohorts))),
        *(
            _bootstrap_tensor_key(cohort_index, block_index)
            for cohort_index in range(len(manifest.cohorts))
            for block_index in range(len(manifest.mean_block_lengths))
        ),
    }


def _panel_tensor_key(cohort_index: int) -> str:
    return f"cohort_{cohort_index}.panel_statistics"


def _panel_year_tensor_key(cohort_index: int) -> str:
    return f"cohort_{cohort_index}.panel_years"


def _bootstrap_tensor_key(cohort_index: int, block_index: int) -> str:
    return f"cohort_{cohort_index}.block_{block_index}.bootstrap_sums"


def _decode_manifest(value: str) -> PublicationInferenceManifest:
    payload = _decode_canonical_json(value, field="manifest")
    _require_exact_keys(payload, PublicationInferenceManifest, field="manifest")
    values = cast("dict[str, Any]", payload)
    try:
        values["source"] = StudySourceIdentity(**cast("dict[str, Any]", values["source"]))
        values["publication_result_source"] = StudySourceIdentity(
            **cast("dict[str, Any]", values["publication_result_source"])
        )
        values["cohorts"] = tuple(
            PublicationInferenceCohortManifest(**item)
            for item in cast("list[dict[str, Any]]", values["cohorts"])
        )
        values["bootstrap_designs"] = tuple(cast("list[str]", values["bootstrap_designs"]))
        values["mean_block_lengths"] = tuple(cast("list[float]", values["mean_block_lengths"]))
        values["bootstrap_domains"] = tuple(cast("list[int]", values["bootstrap_domains"]))
        return PublicationInferenceManifest(**values)
    except (TypeError, ValueError) as error:
        raise PublicationInferenceArtifactError(
            reason=f"manifest fields are invalid: {error}"
        ) from error


def _persist_artifact(
    directory: Path,
    *,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> Path:
    _validate_directory(directory)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".publication-inference-",
        suffix=".safetensors",
        dir=directory,
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
                raise PublicationInferenceArtifactError(
                    reason="content-addressed destination contains different bytes"
                ) from None
        temporary.unlink()
        _fsync_directory(directory)
    except (OSError, SafetensorError) as error:
        raise PublicationInferenceArtifactError(reason=f"artifact write failed: {error}") from error
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
        raise PublicationInferenceArtifactError(
            reason=f"receipt directory creation failed: {error}"
        ) from error
    if not receipt_directory.is_dir() or receipt_directory.is_symlink():
        raise PublicationInferenceArtifactError(reason="receipt directory is invalid")
    payload = _canonical_json(
        {
            "format": _RECEIPT_FORMAT,
            "request_sha256": request_sha256,
            "artifact_sha256": artifact_sha256,
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".publication-inference-request-",
        suffix=".json",
        dir=receipt_directory,
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
                raise PublicationInferenceArtifactError(
                    reason="immutable request receipt contains different bytes"
                ) from None
        temporary.unlink()
        _fsync_directory(receipt_directory)
    except OSError as error:
        raise PublicationInferenceArtifactError(reason=f"receipt write failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_payload(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    try:
        with safe_open(str(path), framework="numpy") as reader:
            raw_metadata = reader.metadata()
        tensors = load_file(path)
    except (OSError, SafetensorError) as error:
        raise PublicationInferenceArtifactError(
            reason=f"safe-tensors decode failed: {error}"
        ) from error
    if raw_metadata is None or set(raw_metadata) != {_METADATA_KEY}:
        raise PublicationInferenceArtifactError(reason="artifact metadata keys are unexpected")
    envelope = _decode_canonical_json(raw_metadata[_METADATA_KEY], field=_METADATA_KEY)
    if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
        raise PublicationInferenceArtifactError(reason="artifact envelope keys are unexpected")
    values = cast("dict[str, object]", envelope)
    if any(type(values[key]) is not str for key in _ENVELOPE_KEYS):
        raise PublicationInferenceArtifactError(reason="artifact envelope values must be strings")
    decoded = cast("dict[str, str]", values)
    if decoded["format"] != _FORMAT:
        raise PublicationInferenceArtifactError(reason="artifact format is unsupported")
    return decoded, tensors


def _read_receipt(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise PublicationInferenceArtifactError(reason="request receipt must be a regular file")
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise PublicationInferenceArtifactError(reason=f"receipt read failed: {error}") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise PublicationInferenceArtifactError(reason="receipt must be one canonical JSON line")
    payload = _decode_canonical_json(text[:-1], field="request receipt")
    if type(payload) is not dict or set(payload) != _RECEIPT_KEYS:
        raise PublicationInferenceArtifactError(reason="receipt keys are unexpected")
    values = cast("dict[str, object]", payload)
    if values["format"] != _RECEIPT_FORMAT:
        raise PublicationInferenceArtifactError(reason="receipt format is unsupported")
    for field in ("request_sha256", "artifact_sha256"):
        _require_sha256(values[field], field=field)
    return cast("dict[str, str]", values)


def _validate_load_request(
    path: Path,
    study: object,
    spec: object,
) -> str:
    if not isinstance(path, Path):
        raise PublicationInferenceArtifactError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
        raise PublicationInferenceArtifactError(reason="path must be a regular .safetensors file")
    if type(study) is not PublicationStudySpec:
        raise PublicationInferenceArtifactError(reason="study must be PublicationStudySpec")
    if type(spec) is not PublicationEmpiricalInferenceSpec:
        raise PublicationInferenceArtifactError(
            reason="spec must be PublicationEmpiricalInferenceSpec"
        )
    _require_sha256(path.stem, field="artifact filename")
    observed = _file_sha256(path)
    if observed != path.stem:
        raise PublicationInferenceArtifactError(reason="artifact bytes differ from filename hash")
    return observed


def _require_resumed_identity(  # noqa: PLR0913 - identities stay explicit
    loaded: LoadedPublicationInferenceArtifact,
    source: StudySourceIdentity,
    feature: VerifiedStudyFeatureArtifact,
    publication_result: LoadedPublicationStudyResultArtifact,
    partitions_sha256: str,
    executor: CpuExecutor,
) -> None:
    manifest = loaded.artifact.manifest
    if (
        manifest.source != source
        or manifest.feature_receipt_sha256 != feature.receipt_sha256
        or manifest.publication_result_sha256 != publication_result.artifact.sha256
        or manifest.publication_result_request_sha256
        != publication_result.artifact.manifest.request_sha256
        or manifest.partitions_sha256 != partitions_sha256
        or manifest.executor_threads != executor.threads
        or manifest.executor_math_backend != executor.math_backend.value
    ):
        raise PublicationInferenceArtifactError(reason="resumed artifact differs from request")
    verify_study_feature_artifact_unchanged(feature)
    _verify_publication_result_unchanged(publication_result)


def _verify_publication_result_unchanged(
    publication_result: LoadedPublicationStudyResultArtifact,
) -> None:
    path = publication_result.artifact.path
    if (
        not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != publication_result.artifact.sha256
    ):
        raise PublicationInferenceArtifactError(reason="publication result changed after loading")


def _require_result_exact(
    observed: PublicationEmpiricalInferenceResult,
    expected: PublicationEmpiricalInferenceResult,
) -> None:
    if (
        observed.spec != expected.spec
        or observed.model_intervals != expected.model_intervals
        or observed.contrasts != expected.contrasts
    ):
        raise PublicationInferenceArtifactError(reason="persisted inference summaries differ")
    for left, right in zip(observed.cohorts, expected.cohorts, strict=True):
        if (
            left.panels.cohort != right.panels.cohort
            or left.panels.point_metrics != right.panels.point_metrics
            or not np.array_equal(left.panels.panel_years, right.panels.panel_years)
            or not np.array_equal(
                left.panels.additive_statistics,
                right.panels.additive_statistics,
            )
        ):
            raise PublicationInferenceArtifactError(reason="persisted panel state differs")
        for left_bootstrap, right_bootstrap in zip(left.bootstrap, right.bootstrap, strict=True):
            if (
                left_bootstrap.design is not right_bootstrap.design
                or left_bootstrap.mean_block_length != right_bootstrap.mean_block_length
                or left_bootstrap.domain != right_bootstrap.domain
                or not np.array_equal(left_bootstrap.additive_sums, right_bootstrap.additive_sums)
            ):
                raise PublicationInferenceArtifactError(reason="persisted bootstrap sums differ")


def _loaded_matrix(
    tensor: np.ndarray,
    *,
    field: str,
    shape: tuple[int, int],
) -> Float64Matrix:
    if (
        type(tensor) is not np.ndarray
        or tensor.dtype != _FLOAT64
        or tensor.ndim != _MATRIX_DIMENSIONS
        or tensor.shape != shape
        or not tensor.flags.c_contiguous
        or not np.isfinite(tensor).all()
    ):
        raise PublicationInferenceArtifactError(
            reason=f"{field} must be a finite C-contiguous Float64 matrix of shape {shape!r}"
        )
    result = np.ascontiguousarray(tensor, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


def _loaded_int64_vector(
    tensor: np.ndarray,
    *,
    field: str,
    length: int,
) -> Int64Vector:
    if (
        type(tensor) is not np.ndarray
        or tensor.dtype != _INT64
        or tensor.ndim != _VECTOR_DIMENSIONS
        or tensor.shape != (length,)
        or not tensor.flags.c_contiguous
    ):
        raise PublicationInferenceArtifactError(
            reason=f"{field} must be a C-contiguous Int64 vector of length {length}"
        )
    result = np.ascontiguousarray(tensor, dtype=np.int64).copy()
    result.flags.writeable = False
    return result


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
        raise PublicationInferenceArtifactError(reason="paper package is not installed") from error
    native = build_info()
    if not paper_version or not native.package_version or not native.rustc_version:
        raise PublicationInferenceArtifactError(reason="software identity is incomplete")
    return paper_version, native.package_version, native.rustc_version


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(_FILE_HASH_BLOCK_BYTES):
                hasher.update(block)
    except OSError as error:
        raise PublicationInferenceArtifactError(reason=f"file hashing failed: {error}") from error
    return hasher.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _decode_canonical_json(value: object, *, field: str) -> object:
    if type(value) is not str:
        raise PublicationInferenceArtifactError(reason=f"{field} must be a string")
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise PublicationInferenceArtifactError(reason=f"{field} is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise PublicationInferenceArtifactError(reason=f"{field} JSON is not canonical")
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
        raise PublicationInferenceArtifactError(
            reason=f"value cannot be canonical JSON: {error}"
        ) from error


def _require_exact_keys(value: object, model: type[object], *, field: str) -> None:
    if type(value) is not dict:
        raise PublicationInferenceArtifactError(reason=f"{field} must be an object")
    expected = {item.name for item in fields(model)}
    if set(cast("dict[object, object]", value)) != expected:
        raise PublicationInferenceArtifactError(reason=f"{field} keys are unexpected")


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationInferenceArtifactError(reason=f"{field} must be lowercase SHA-256")


def _validate_directory(directory: Path) -> None:
    if not isinstance(directory, Path) or not directory.is_dir() or directory.is_symlink():
        raise PublicationInferenceArtifactError(
            reason="directory must be an existing non-symlink directory"
        )


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
    "LoadedPublicationInferenceArtifact",
    "PublicationInferenceArtifact",
    "PublicationInferenceArtifactError",
    "PublicationInferenceCohortManifest",
    "PublicationInferenceManifest",
    "load_publication_inference_artifact",
    "load_publication_inference_request",
    "publication_inference_request_sha256",
    "run_or_load_publication_empirical_inference",
]
