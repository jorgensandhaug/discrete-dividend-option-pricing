"""Immutable, resumable artifacts for stochastic publication replications."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Protocol, cast

import numpy as np
import numpy.typing as npt
from finoptions import CpuMathBackend, LsmBatchResult, build_info
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

from discrete_dividend_study.experiments.lsm_validation.partitions import (
    PreparedPublicationPartitions,
    publication_partitions_sha256,
)
from discrete_dividend_study.experiments.lsm_validation.pricing import (
    PublicationCashReplicationResult,
    PublicationVariantReplicationResult,
    run_publication_cash_replication,
)
from discrete_dividend_study.experiments.lsm_validation.specification import (
    PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
    PROJECT_PUBLICATION_STUDY_SPEC,
    PublicationReplicationExecutionSpec,
    PublicationReplicationPlanSpec,
    PublicationStudySpec,
    PublicationVariant,
    publication_execution_sha256,
    publication_replication_plan_sha256,
    publication_study_sha256,
)
from discrete_dividend_study.experiments.metrics import (
    ProjectMetricError,
    ProjectPricingMetrics,
    evaluate_project_pricing_f64,
)
from discrete_dividend_study.experiments.neural.run import StudySourceIdentity

Float64Vector = npt.NDArray[np.float64]
UInt32Vector = npt.NDArray[np.uint32]
BoolVector = npt.NDArray[np.bool_]

_FORMAT: Final = "discrete-dividend-publication-cash-replication-v1"
_FORMAT_VERSION: Final = 1
_PAPER_PACKAGE: Final = "discrete-dividend-option-pricing"
_METADATA_KEY: Final = "publication_cash_replication"
_ENVELOPE_KEYS: Final = frozenset({"format", "manifest", "metrics"})
_REQUEST_DIRECTORY: Final = "cash-requests"
_RECEIPT_FORMAT: Final = "discrete-dividend-publication-cash-request-receipt-v1"
_RECEIPT_KEYS: Final = frozenset({"format", "request_sha256", "artifact_sha256"})
_SHA256_HEX_LENGTH: Final = 64
_FLOAT64_DTYPE: Final = np.dtype(np.float64)
_CASH_VARIANTS: Final = tuple(
    variant for variant in PublicationVariant if variant.uses_cash_schedule
)
_NATIVE_FIELDS: Final = (
    "prices",
    "sample_standard_deviations",
    "standard_errors",
    "origin_decisions_fitted",
    "origin_exercised",
    "regression_nodes",
    "no_training_support_nodes",
    "insufficient_exercise_support_nodes",
    "minimum_in_the_money_training_paths",
    "maximum_regression_condition_numbers",
    "early_exercise_paths",
    "expiry_paths",
)
_NATIVE_DTYPES: Final = {
    "prices": np.dtype(np.float64),
    "sample_standard_deviations": np.dtype(np.float64),
    "standard_errors": np.dtype(np.float64),
    "origin_decisions_fitted": np.dtype(np.bool_),
    "origin_exercised": np.dtype(np.bool_),
    "regression_nodes": np.dtype(np.uint32),
    "no_training_support_nodes": np.dtype(np.uint32),
    "insufficient_exercise_support_nodes": np.dtype(np.uint32),
    "minimum_in_the_money_training_paths": np.dtype(np.uint32),
    "maximum_regression_condition_numbers": np.dtype(np.float64),
    "early_exercise_paths": np.dtype(np.uint32),
    "expiry_paths": np.dtype(np.uint32),
}


class _HashWriter(Protocol):
    """Streaming byte interface shared by hashlib implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(slots=True)
class PublicationArtifactError(RuntimeError):
    """A cash-replication artifact or resume receipt violates its contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected artifact contract."""
        return f"invalid publication cash artifact: {self.reason}"


@dataclass(frozen=True, slots=True)
class PublicationCashArtifactManifest:
    """Complete source, estimator, runtime, partition, and software identity."""

    format_version: int
    request_sha256: str
    source: StudySourceIdentity
    study_sha256: str
    replication_plan_sha256: str
    execution: PublicationReplicationExecutionSpec
    execution_sha256: str
    partitions_sha256: str
    observations: int
    simulation_units: int
    exact_groups: int
    general_groups: int
    variants: tuple[str, ...]
    paper_package_version: str
    engine_package_version: str
    rustc_version: str
    tensors_sha256: str
    metrics_sha256: str

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent serialized identity."""
        _validate_manifest_structure(self)
        _validate_manifest_population(self)
        _validate_manifest_execution(self)
        _validate_versions(self)
        if self.request_sha256 != _request_sha256_from_manifest(self):
            raise PublicationArtifactError(reason="request_sha256 disagrees with manifest identity")


def _validate_manifest_structure(manifest: PublicationCashArtifactManifest) -> None:
    if type(manifest.format_version) is not int or manifest.format_version != _FORMAT_VERSION:
        raise PublicationArtifactError(reason="format_version must be the plain int 1")
    _validate_manifest_hashes(manifest)
    if type(manifest.source) is not StudySourceIdentity:
        raise PublicationArtifactError(reason="source must be StudySourceIdentity")
    if type(manifest.execution) is not PublicationReplicationExecutionSpec:
        raise PublicationArtifactError(
            reason="execution must be PublicationReplicationExecutionSpec"
        )
    if manifest.variants != tuple(variant.value for variant in _CASH_VARIANTS):
        raise PublicationArtifactError(reason="variants must be the canonical cash subset")


def _validate_manifest_population(manifest: PublicationCashArtifactManifest) -> None:
    if type(manifest.observations) is not int or manifest.observations <= 0:
        raise PublicationArtifactError(reason="observations must be a positive plain int")
    if (
        type(manifest.simulation_units) is not int
        or not 0 < manifest.simulation_units <= manifest.observations
    ):
        raise PublicationArtifactError(
            reason="simulation_units must be positive and no greater than observations"
        )
    for field, value in (
        ("exact_groups", manifest.exact_groups),
        ("general_groups", manifest.general_groups),
    ):
        if type(value) is not int or value < 0:
            raise PublicationArtifactError(reason=f"{field} must be a nonnegative plain int")
    if manifest.exact_groups + manifest.general_groups != manifest.simulation_units:
        raise PublicationArtifactError(
            reason="exact and general groups do not partition simulation_units"
        )


def _validate_manifest_execution(manifest: PublicationCashArtifactManifest) -> None:
    if manifest.execution_sha256 != publication_execution_sha256(manifest.execution):
        raise PublicationArtifactError(reason="execution_sha256 disagrees with execution")
    manifest.execution.validate_units(manifest.simulation_units)


@dataclass(frozen=True, slots=True)
class PublicationCashArtifact:
    """One verified content-addressed stochastic replication artifact."""

    path: Path
    sha256: str
    manifest: PublicationCashArtifactManifest

    def __post_init__(self) -> None:
        """Require a content-addressed path and complete manifest."""
        if not isinstance(self.path, Path):
            raise PublicationArtifactError(reason="path must be pathlib.Path")
        _require_sha256(self.sha256, field="sha256")
        if self.path.name != f"{self.sha256}.safetensors":
            raise PublicationArtifactError(reason="artifact path does not match its content hash")
        if type(self.manifest) is not PublicationCashArtifactManifest:
            raise PublicationArtifactError(
                reason="manifest must be PublicationCashArtifactManifest"
            )


@dataclass(frozen=True, slots=True)
class LoadedPublicationCashArtifact:
    """Verified target, native diagnostics, metrics, and artifact identity."""

    artifact: PublicationCashArtifact
    target: Float64Vector
    result: PublicationCashReplicationResult

    def __post_init__(self) -> None:
        """Require immutable complete row agreement after decoding."""
        if type(self.artifact) is not PublicationCashArtifact:
            raise PublicationArtifactError(reason="artifact must be PublicationCashArtifact")
        _validate_vector(
            self.target,
            field="target",
            dtype=_FLOAT64_DTYPE,
            observations=self.artifact.manifest.observations,
        )
        if self.target.flags.writeable:
            raise PublicationArtifactError(reason="loaded target must be immutable")
        if type(self.result) is not PublicationCashReplicationResult:
            raise PublicationArtifactError(reason="result must be PublicationCashReplicationResult")
        if self.result.rows != self.artifact.manifest.observations:
            raise PublicationArtifactError(reason="result rows disagree with the manifest")


def publication_cash_request_sha256(
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    execution: PublicationReplicationExecutionSpec,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    replication_plan: PublicationReplicationPlanSpec = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
) -> str:
    """Hash every identity known before stochastic pricing begins."""
    _validate_prepricing_request(source, inputs, execution, study, replication_plan)
    software = _software_identity()
    payload = _request_payload(
        source=source,
        study_sha256=publication_study_sha256(study),
        replication_plan_sha256=publication_replication_plan_sha256(replication_plan),
        execution=execution,
        execution_sha256=publication_execution_sha256(execution),
        partitions_sha256=publication_partitions_sha256(inputs),
        observations=inputs.original_rows,
        simulation_units=inputs.simulation_units,
        exact_groups=inputs.exact.groups,
        general_groups=inputs.general.groups,
        variants=tuple(variant.value for variant in _CASH_VARIANTS),
        paper_package_version=software[0],
        engine_package_version=software[1],
        rustc_version=software[2],
    )
    return _request_sha256(payload)


def run_or_load_publication_cash_replication(  # noqa: PLR0913 - identities remain explicit
    directory: Path,
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    execution: PublicationReplicationExecutionSpec,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    replication_plan: PublicationReplicationPlanSpec = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
) -> LoadedPublicationCashArtifact:
    """Exactly resume a verified request or run, persist, and reload it once."""
    request_sha256 = publication_cash_request_sha256(
        source,
        inputs,
        execution,
        study=study,
        replication_plan=replication_plan,
    )
    loaded = load_publication_cash_request(
        directory,
        request_sha256,
        study=study,
        replication_plan=replication_plan,
    )
    if loaded is not None:
        _require_loaded_request(loaded, source, inputs, execution)
        return loaded
    result = run_publication_cash_replication(inputs, execution, study)
    return write_publication_cash_artifact(
        directory,
        source,
        inputs,
        replication_plan,
        result,
    )


def write_publication_cash_artifact(
    directory: Path,
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    replication_plan: PublicationReplicationPlanSpec,
    result: PublicationCashReplicationResult,
) -> LoadedPublicationCashArtifact:
    """Atomically persist one complete cash replication and its immutable request receipt."""
    _validate_write_request(directory, source, inputs, replication_plan, result)
    execution = result.execution
    tensors = _result_tensors(inputs.targets, result)
    metrics_json = _canonical_json(
        {variant.variant.value: asdict(variant.metrics) for variant in result.variants}
    )
    software = _software_identity()
    request_payload = _request_payload(
        source=source,
        study_sha256=publication_study_sha256(result.study),
        replication_plan_sha256=publication_replication_plan_sha256(replication_plan),
        execution=execution,
        execution_sha256=publication_execution_sha256(execution),
        partitions_sha256=result.partitions_sha256,
        observations=result.rows,
        simulation_units=inputs.simulation_units,
        exact_groups=result.exact_groups,
        general_groups=result.general_groups,
        variants=tuple(variant.variant.value for variant in result.variants),
        paper_package_version=software[0],
        engine_package_version=software[1],
        rustc_version=software[2],
    )
    manifest = PublicationCashArtifactManifest(
        format_version=_FORMAT_VERSION,
        request_sha256=_request_sha256(request_payload),
        source=source,
        study_sha256=publication_study_sha256(result.study),
        replication_plan_sha256=publication_replication_plan_sha256(replication_plan),
        execution=execution,
        execution_sha256=publication_execution_sha256(execution),
        partitions_sha256=result.partitions_sha256,
        observations=result.rows,
        simulation_units=inputs.simulation_units,
        exact_groups=result.exact_groups,
        general_groups=result.general_groups,
        variants=tuple(variant.variant.value for variant in result.variants),
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
    _persist_request_receipt(directory, manifest.request_sha256, destination.stem)
    loaded = load_publication_cash_artifact(
        destination,
        study=result.study,
        replication_plan=replication_plan,
    )
    _require_results_exact(loaded.result, result)
    return loaded


def load_publication_cash_request(
    directory: Path,
    request_sha256: str,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    replication_plan: PublicationReplicationPlanSpec = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
) -> LoadedPublicationCashArtifact | None:
    """Load one immutable request receipt, or return ``None`` only when it is absent."""
    _validate_artifact_directory(directory)
    _require_sha256(request_sha256, field="request_sha256")
    receipt_path = directory / _REQUEST_DIRECTORY / f"{request_sha256}.json"
    if not receipt_path.exists():
        return None
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise PublicationArtifactError(reason="request receipt must be a regular file")
    payload = _read_receipt(receipt_path)
    if payload["request_sha256"] != request_sha256:
        raise PublicationArtifactError(reason="request receipt filename and identity differ")
    artifact_path = directory / f"{payload['artifact_sha256']}.safetensors"
    loaded = load_publication_cash_artifact(
        artifact_path,
        study=study,
        replication_plan=replication_plan,
    )
    if loaded.artifact.manifest.request_sha256 != request_sha256:
        raise PublicationArtifactError(reason="request receipt references another request")
    return loaded


def load_publication_cash_artifact(
    path: Path,
    *,
    study: PublicationStudySpec = PROJECT_PUBLICATION_STUDY_SPEC,
    replication_plan: PublicationReplicationPlanSpec = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
) -> LoadedPublicationCashArtifact:
    """Load only a self-consistent content-addressed cash-replication artifact."""
    file_sha256 = _validate_load_request(path, study, replication_plan)
    metadata, tensors = _read_payload(path)
    manifest = _decode_manifest(metadata["manifest"])
    _validate_loaded_manifest(manifest, study, replication_plan)
    if set(tensors) != _expected_tensor_keys():
        raise PublicationArtifactError(reason="artifact tensor keys are missing or unexpected")
    if _tensors_sha256(tensors) != manifest.tensors_sha256:
        raise PublicationArtifactError(reason="result tensors disagree with the manifest hash")
    if _text_sha256(metadata["metrics"]) != manifest.metrics_sha256:
        raise PublicationArtifactError(reason="metrics disagree with the manifest hash")
    metrics = _decode_metrics(metadata["metrics"])
    target = cast(
        "Float64Vector",
        _loaded_vector(
            tensors["target"],
            field="target",
            dtype=_FLOAT64_DTYPE,
            observations=manifest.observations,
        ),
    )
    variants = tuple(
        _loaded_variant(tensors, target, metrics, variant, manifest.observations)
        for variant in _CASH_VARIANTS
    )
    result = PublicationCashReplicationResult(
        study=study,
        execution=manifest.execution,
        partitions_sha256=manifest.partitions_sha256,
        rows=manifest.observations,
        exact_groups=manifest.exact_groups,
        general_groups=manifest.general_groups,
        variants=variants,
    )
    artifact = PublicationCashArtifact(path=path, sha256=file_sha256, manifest=manifest)
    return LoadedPublicationCashArtifact(artifact=artifact, target=target, result=result)


def _validate_load_request(
    path: Path,
    study: object,
    replication_plan: object,
) -> str:
    if not isinstance(path, Path):
        raise PublicationArtifactError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
        raise PublicationArtifactError(reason="path must be a regular .safetensors file")
    if type(study) is not PublicationStudySpec:
        raise PublicationArtifactError(reason="study must be PublicationStudySpec")
    if type(replication_plan) is not PublicationReplicationPlanSpec:
        raise PublicationArtifactError(
            reason="replication_plan must be PublicationReplicationPlanSpec"
        )
    _require_sha256(path.stem, field="artifact filename")
    file_sha256 = _file_sha256(path)
    if file_sha256 != path.stem:
        raise PublicationArtifactError(reason="artifact bytes do not match the filename hash")
    return file_sha256


def _validate_loaded_manifest(
    manifest: PublicationCashArtifactManifest,
    study: PublicationStudySpec,
    replication_plan: PublicationReplicationPlanSpec,
) -> None:
    if manifest.study_sha256 != publication_study_sha256(study):
        raise PublicationArtifactError(reason="artifact and requested study identities differ")
    if manifest.replication_plan_sha256 != publication_replication_plan_sha256(replication_plan):
        raise PublicationArtifactError(
            reason="artifact and requested replication-plan identities differ"
        )
    expected_execution = replication_plan.execution(
        manifest.execution.replication,
        threads=manifest.execution.threads,
        math_backend=manifest.execution.math_backend,
    )
    if manifest.execution != expected_execution:
        raise PublicationArtifactError(reason="execution is not declared by the replication plan")


def _validate_prepricing_request(
    source: object,
    inputs: object,
    execution: object,
    study: object,
    replication_plan: object,
) -> None:
    if type(source) is not StudySourceIdentity:
        raise PublicationArtifactError(reason="source must be StudySourceIdentity")
    if type(inputs) is not PreparedPublicationPartitions:
        raise PublicationArtifactError(reason="inputs must be PreparedPublicationPartitions")
    if type(execution) is not PublicationReplicationExecutionSpec:
        raise PublicationArtifactError(
            reason="execution must be PublicationReplicationExecutionSpec"
        )
    if type(study) is not PublicationStudySpec:
        raise PublicationArtifactError(reason="study must be PublicationStudySpec")
    if type(replication_plan) is not PublicationReplicationPlanSpec:
        raise PublicationArtifactError(
            reason="replication_plan must be PublicationReplicationPlanSpec"
        )
    if inputs.study != study:
        raise PublicationArtifactError(reason="partition and requested studies differ")
    expected = replication_plan.execution(
        execution.replication,
        threads=execution.threads,
        math_backend=execution.math_backend,
    )
    if execution != expected:
        raise PublicationArtifactError(reason="execution is not declared by the replication plan")
    replication_plan.validate_units(inputs.simulation_units)


def _validate_write_request(
    directory: Path,
    source: object,
    inputs: object,
    replication_plan: object,
    result: object,
) -> None:
    _validate_artifact_directory(directory)
    if type(result) is not PublicationCashReplicationResult:
        raise PublicationArtifactError(reason="result must be PublicationCashReplicationResult")
    _validate_prepricing_request(
        source,
        inputs,
        result.execution,
        result.study,
        replication_plan,
    )
    typed_inputs = cast("PreparedPublicationPartitions", inputs)
    if (
        result.rows != typed_inputs.original_rows
        or result.exact_groups != typed_inputs.exact.groups
        or result.general_groups != typed_inputs.general.groups
        or result.partitions_sha256 != publication_partitions_sha256(typed_inputs)
    ):
        raise PublicationArtifactError(reason="result and prepared partition identities differ")


def _validate_artifact_directory(directory: Path) -> None:
    if not isinstance(directory, Path) or not directory.is_dir() or directory.is_symlink():
        raise PublicationArtifactError(reason="directory must be an existing non-symlink directory")


def _request_payload(  # noqa: PLR0913 - complete pre-pricing identity is explicit
    *,
    source: StudySourceIdentity,
    study_sha256: str,
    replication_plan_sha256: str,
    execution: PublicationReplicationExecutionSpec,
    execution_sha256: str,
    partitions_sha256: str,
    observations: int,
    simulation_units: int,
    exact_groups: int,
    general_groups: int,
    variants: tuple[str, ...],
    paper_package_version: str,
    engine_package_version: str,
    rustc_version: str,
) -> dict[str, object]:
    return {
        "source": asdict(source),
        "study_sha256": study_sha256,
        "replication_plan_sha256": replication_plan_sha256,
        "execution": _execution_payload(execution),
        "execution_sha256": execution_sha256,
        "partitions_sha256": partitions_sha256,
        "observations": observations,
        "simulation_units": simulation_units,
        "exact_groups": exact_groups,
        "general_groups": general_groups,
        "variants": variants,
        "paper_package_version": paper_package_version,
        "engine_package_version": engine_package_version,
        "rustc_version": rustc_version,
    }


def _request_sha256_from_manifest(manifest: PublicationCashArtifactManifest) -> str:
    return _request_sha256(
        _request_payload(
            source=manifest.source,
            study_sha256=manifest.study_sha256,
            replication_plan_sha256=manifest.replication_plan_sha256,
            execution=manifest.execution,
            execution_sha256=manifest.execution_sha256,
            partitions_sha256=manifest.partitions_sha256,
            observations=manifest.observations,
            simulation_units=manifest.simulation_units,
            exact_groups=manifest.exact_groups,
            general_groups=manifest.general_groups,
            variants=manifest.variants,
            paper_package_version=manifest.paper_package_version,
            engine_package_version=manifest.engine_package_version,
            rustc_version=manifest.rustc_version,
        )
    )


def _request_sha256(payload: dict[str, object]) -> str:
    request = {"format": _FORMAT, "request": payload}
    return hashlib.sha256(_canonical_json(request).encode("ascii")).hexdigest()


def _execution_payload(execution: PublicationReplicationExecutionSpec) -> dict[str, object]:
    return {
        "replication": execution.replication,
        "threads": execution.threads,
        "math_backend": execution.math_backend.value,
        "training_seed": execution.training_seed,
        "training_domain_start": execution.training_domain_start,
        "evaluation_seed": execution.evaluation_seed,
        "evaluation_domain_start": execution.evaluation_domain_start,
    }


def _result_tensors(
    target: Float64Vector,
    result: PublicationCashReplicationResult,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {"target": target}
    for variant in result.variants:
        native = cast("LsmBatchResult", variant.native)
        for field in _NATIVE_FIELDS:
            tensors[_tensor_key(variant.variant, field)] = cast(
                "np.ndarray", getattr(native, field)
            )
    return tensors


def _loaded_variant(
    tensors: dict[str, np.ndarray],
    target: Float64Vector,
    metrics: dict[PublicationVariant, ProjectPricingMetrics],
    variant: PublicationVariant,
    observations: int,
) -> PublicationVariantReplicationResult:
    def loaded(field: str) -> np.ndarray:
        key = _tensor_key(variant, field)
        return _loaded_vector(
            tensors[key],
            field=key,
            dtype=_NATIVE_DTYPES[field],
            observations=observations,
        )

    native = LsmBatchResult(
        prices=cast("Float64Vector", loaded("prices")),
        sample_standard_deviations=cast("Float64Vector", loaded("sample_standard_deviations")),
        standard_errors=cast("Float64Vector", loaded("standard_errors")),
        origin_decisions_fitted=cast("BoolVector", loaded("origin_decisions_fitted")),
        origin_exercised=cast("BoolVector", loaded("origin_exercised")),
        regression_nodes=cast("UInt32Vector", loaded("regression_nodes")),
        no_training_support_nodes=cast("UInt32Vector", loaded("no_training_support_nodes")),
        insufficient_exercise_support_nodes=cast(
            "UInt32Vector", loaded("insufficient_exercise_support_nodes")
        ),
        minimum_in_the_money_training_paths=cast(
            "UInt32Vector", loaded("minimum_in_the_money_training_paths")
        ),
        maximum_regression_condition_numbers=cast(
            "Float64Vector", loaded("maximum_regression_condition_numbers")
        ),
        early_exercise_paths=cast("UInt32Vector", loaded("early_exercise_paths")),
        expiry_paths=cast("UInt32Vector", loaded("expiry_paths")),
    )
    stored_metrics = metrics[variant]
    try:
        recomputed = evaluate_project_pricing_f64(target, native.prices)
    except ProjectMetricError as error:
        raise PublicationArtifactError(reason=str(error)) from error
    if stored_metrics != recomputed:
        raise PublicationArtifactError(reason=f"stored metrics disagree for {variant.value!r}")
    return PublicationVariantReplicationResult(
        variant=variant,
        native=native,
        metrics=stored_metrics,
    )


def _expected_tensor_keys() -> set[str]:
    return {
        "target",
        *(_tensor_key(variant, field) for variant in _CASH_VARIANTS for field in _NATIVE_FIELDS),
    }


def _tensor_key(variant: PublicationVariant, field: str) -> str:
    return f"{variant.value}.{field}"


def _persist_artifact(
    directory: Path,
    *,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".publication-cash-",
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
                raise PublicationArtifactError(
                    reason="content-addressed destination contains different bytes"
                ) from None
        temporary.unlink()
        _fsync_directory(directory)
    except (OSError, SafetensorError) as error:
        raise PublicationArtifactError(reason=f"artifact write failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _persist_request_receipt(
    directory: Path,
    request_sha256: str,
    artifact_sha256: str,
) -> Path:
    _require_sha256(request_sha256, field="request_sha256")
    _require_sha256(artifact_sha256, field="artifact_sha256")
    receipt_directory = directory / _REQUEST_DIRECTORY
    try:
        receipt_directory.mkdir(mode=0o755, exist_ok=True)
    except OSError as error:
        raise PublicationArtifactError(
            reason=f"receipt directory creation failed: {error}"
        ) from error
    if not receipt_directory.is_dir() or receipt_directory.is_symlink():
        raise PublicationArtifactError(reason="request receipt directory is invalid")
    payload = _canonical_json(
        {
            "format": _RECEIPT_FORMAT,
            "request_sha256": request_sha256,
            "artifact_sha256": artifact_sha256,
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".publication-request-",
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
                raise PublicationArtifactError(
                    reason="immutable request receipt already references different bytes"
                ) from None
        temporary.unlink()
        _fsync_directory(receipt_directory)
    except OSError as error:
        raise PublicationArtifactError(reason=f"request receipt write failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_payload(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    try:
        with safe_open(str(path), framework="numpy") as reader:
            raw_metadata = reader.metadata()
        tensors = load_file(path)
    except (OSError, SafetensorError) as error:
        raise PublicationArtifactError(reason=f"safe-tensors decode failed: {error}") from error
    if raw_metadata is None or set(raw_metadata) != {_METADATA_KEY}:
        raise PublicationArtifactError(reason="artifact metadata keys are missing or unexpected")
    envelope = _decode_canonical_json(raw_metadata[_METADATA_KEY], field=_METADATA_KEY)
    if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
        raise PublicationArtifactError(reason="artifact envelope keys are missing or unexpected")
    values = cast("dict[str, object]", envelope)
    if any(type(values[key]) is not str for key in _ENVELOPE_KEYS):
        raise PublicationArtifactError(reason="artifact envelope values must be plain strings")
    decoded = cast("dict[str, str]", values)
    if decoded["format"] != _FORMAT:
        raise PublicationArtifactError(reason="artifact format identifier is unsupported")
    return decoded, tensors


def _read_receipt(path: Path) -> dict[str, str]:
    try:
        value = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise PublicationArtifactError(reason=f"request receipt read failed: {error}") from error
    if not value.endswith("\n") or value.count("\n") != 1:
        raise PublicationArtifactError(
            reason="request receipt must contain one canonical JSON line"
        )
    payload = _decode_canonical_json(value[:-1], field="request receipt")
    if type(payload) is not dict or set(payload) != _RECEIPT_KEYS:
        raise PublicationArtifactError(reason="request receipt keys are missing or unexpected")
    values = cast("dict[str, object]", payload)
    if values["format"] != _RECEIPT_FORMAT:
        raise PublicationArtifactError(reason="request receipt format is unsupported")
    for field in ("request_sha256", "artifact_sha256"):
        _require_sha256(values[field], field=field)
    return cast("dict[str, str]", values)


def _decode_manifest(value: str) -> PublicationCashArtifactManifest:
    payload = _decode_canonical_json(value, field="manifest")
    _require_exact_keys(payload, PublicationCashArtifactManifest, field="manifest")
    values = cast("dict[str, Any]", payload)
    try:
        values["source"] = StudySourceIdentity(**cast("dict[str, Any]", values["source"]))
        execution = cast("dict[str, Any]", values["execution"])
        execution["math_backend"] = CpuMathBackend(execution["math_backend"])
        values["execution"] = PublicationReplicationExecutionSpec(**execution)
        values["variants"] = tuple(cast("list[str]", values["variants"]))
        return PublicationCashArtifactManifest(**values)
    except (TypeError, ValueError) as error:
        raise PublicationArtifactError(reason=f"manifest fields are invalid: {error}") from error


def _decode_metrics(value: str) -> dict[PublicationVariant, ProjectPricingMetrics]:
    payload = _decode_canonical_json(value, field="metrics")
    expected = {variant.value for variant in _CASH_VARIANTS}
    if type(payload) is not dict or set(payload) != expected:
        raise PublicationArtifactError(reason="metrics variants are missing or unexpected")
    decoded = cast("dict[str, object]", payload)
    metrics: dict[PublicationVariant, ProjectPricingMetrics] = {}
    try:
        for variant in _CASH_VARIANTS:
            metric_payload = decoded[variant.value]
            _require_exact_keys(metric_payload, ProjectPricingMetrics, field="variant metrics")
            metrics[variant] = ProjectPricingMetrics(**cast("dict[str, Any]", metric_payload))
    except (TypeError, ProjectMetricError) as error:
        raise PublicationArtifactError(reason=f"metrics fields are invalid: {error}") from error
    return metrics


def _loaded_vector(
    tensor: np.ndarray,
    *,
    field: str,
    dtype: np.dtype[np.generic],
    observations: int,
) -> npt.NDArray[np.generic]:
    _validate_vector(tensor, field=field, dtype=dtype, observations=observations)
    result = np.ascontiguousarray(tensor, dtype=dtype).copy()
    result.flags.writeable = False
    return result


def _validate_vector(
    value: object,
    *,
    field: str,
    dtype: np.dtype[np.generic],
    observations: int,
) -> None:
    if type(value) is not np.ndarray:
        raise PublicationArtifactError(reason=f"{field} must be a plain numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if array.dtype != dtype or array.shape != (observations,) or not array.flags.c_contiguous:
        raise PublicationArtifactError(
            reason=f"{field} must be a C-contiguous {dtype} vector of length {observations}"
        )
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise PublicationArtifactError(reason=f"{field} must be finite")


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


def _require_loaded_request(
    loaded: LoadedPublicationCashArtifact,
    source: StudySourceIdentity,
    inputs: PreparedPublicationPartitions,
    execution: PublicationReplicationExecutionSpec,
) -> None:
    manifest = loaded.artifact.manifest
    if (
        manifest.source != source
        or manifest.execution != execution
        or manifest.partitions_sha256 != publication_partitions_sha256(inputs)
        or manifest.observations != inputs.original_rows
        or manifest.simulation_units != inputs.simulation_units
    ):
        raise PublicationArtifactError(reason="resumed artifact differs from the current request")
    if not np.array_equal(loaded.target, inputs.targets):
        raise PublicationArtifactError(
            reason="resumed artifact target differs from the current input"
        )


def _require_results_exact(
    observed: PublicationCashReplicationResult,
    expected: PublicationCashReplicationResult,
) -> None:
    if (
        observed.study != expected.study
        or observed.execution != expected.execution
        or observed.partitions_sha256 != expected.partitions_sha256
        or observed.rows != expected.rows
        or observed.exact_groups != expected.exact_groups
        or observed.general_groups != expected.general_groups
    ):
        raise PublicationArtifactError(reason="persisted replication identity differs")
    for left, right in zip(observed.variants, expected.variants, strict=True):
        if left.variant is not right.variant or left.metrics != right.metrics:
            raise PublicationArtifactError(reason="persisted variant identity or metrics differ")
        left_native = cast("LsmBatchResult", left.native)
        right_native = cast("LsmBatchResult", right.native)
        for field in _NATIVE_FIELDS:
            if not np.array_equal(getattr(left_native, field), getattr(right_native, field)):
                raise PublicationArtifactError(
                    reason=f"persisted {field!r} differs for {left.variant.value!r}"
                )


def _validate_manifest_hashes(manifest: PublicationCashArtifactManifest) -> None:
    for field, value in (
        ("request_sha256", manifest.request_sha256),
        ("study_sha256", manifest.study_sha256),
        ("replication_plan_sha256", manifest.replication_plan_sha256),
        ("execution_sha256", manifest.execution_sha256),
        ("partitions_sha256", manifest.partitions_sha256),
        ("tensors_sha256", manifest.tensors_sha256),
        ("metrics_sha256", manifest.metrics_sha256),
    ):
        _require_sha256(value, field=field)


def _validate_versions(manifest: PublicationCashArtifactManifest) -> None:
    for field, value in (
        ("paper_package_version", manifest.paper_package_version),
        ("engine_package_version", manifest.engine_package_version),
        ("rustc_version", manifest.rustc_version),
    ):
        if type(value) is not str or not value:
            raise PublicationArtifactError(reason=f"{field} must be a nonempty plain string")


def _software_identity() -> tuple[str, str, str]:
    try:
        paper_version = version(_PAPER_PACKAGE)
    except PackageNotFoundError as error:
        raise PublicationArtifactError(reason="paper package is not installed") from error
    native = build_info()
    if not paper_version or not native.package_version or not native.rustc_version:
        raise PublicationArtifactError(reason="software version identity is incomplete")
    return paper_version, native.package_version, native.rustc_version


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as error:
        raise PublicationArtifactError(reason=f"artifact hashing failed: {error}") from error
    return hasher.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _decode_canonical_json(value: object, *, field: str) -> object:
    if type(value) is not str:
        raise PublicationArtifactError(reason=f"{field} metadata must be a plain string")
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise PublicationArtifactError(reason=f"{field} is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise PublicationArtifactError(reason=f"{field} JSON is not canonical")
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
        raise PublicationArtifactError(reason=f"value cannot be canonical JSON: {error}") from error


def _require_exact_keys(value: object, model: type[object], *, field: str) -> None:
    if type(value) is not dict:
        raise PublicationArtifactError(reason=f"{field} must be an object")
    expected = {item.name for item in fields(model)}
    if set(cast("dict[object, object]", value)) != expected:
        raise PublicationArtifactError(reason=f"{field} keys are missing or unexpected")


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationArtifactError(reason=f"{field} must be a lowercase SHA-256 hex digest")


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
    "LoadedPublicationCashArtifact",
    "PublicationArtifactError",
    "PublicationCashArtifact",
    "PublicationCashArtifactManifest",
    "load_publication_cash_artifact",
    "load_publication_cash_request",
    "publication_cash_request_sha256",
    "run_or_load_publication_cash_replication",
    "write_publication_cash_artifact",
]
