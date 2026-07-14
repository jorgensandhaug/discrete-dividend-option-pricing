"""Content-addressed complete artifacts for the four-way classical study."""

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

from .classical_inputs import PreparedClassicalInputs, prepared_classical_inputs_sha256
from .classical_pricing import (
    ClassicalStudyPricingResult,
    ClassicalVariantPricingResult,
    run_project_classical_study,
)
from .classical_spec import (
    PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    ClassicalExecutionSpec,
    ClassicalStudySpec,
    ClassicalVariant,
    ClassicalVariantSpec,
    classical_execution_sha256,
    classical_study_sha256,
)
from .neural_run import StudySourceIdentity
from .project_metrics import (
    ProjectMetricError,
    ProjectPricingMetrics,
    evaluate_project_pricing_f64,
)

Float64Vector = npt.NDArray[np.float64]
UInt32Vector = npt.NDArray[np.uint32]
BoolVector = npt.NDArray[np.bool_]

_FORMAT: Final = "discrete-dividend-classical-study-artifact-v2"
_FORMAT_VERSION: Final = 2
_PAPER_PACKAGE: Final = "discrete-dividend-option-pricing"
_METADATA_KEY: Final = "classical_study"
_ENVELOPE_KEYS: Final = frozenset({"format", "manifest", "metrics"})
_SHA256_HEX_LENGTH: Final = 64
_FLOAT64_DTYPE: Final = np.dtype(np.float64)
_NATIVE_FIELDS: Final = (
    "prices",
    "sample_standard_deviations",
    "standard_errors",
    "origin_decisions_fitted",
    "origin_exercised",
    "regression_nodes",
    "no_training_support_nodes",
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
    "minimum_in_the_money_training_paths": np.dtype(np.uint32),
    "maximum_regression_condition_numbers": np.dtype(np.float64),
    "early_exercise_paths": np.dtype(np.uint32),
    "expiry_paths": np.dtype(np.uint32),
}


class _HashWriter(Protocol):
    """The streaming byte interface shared by hashlib implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(slots=True)
class ClassicalArtifactError(RuntimeError):
    """A classical result artifact violates identity, storage, or numerical contracts."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected artifact contract."""
        return f"invalid classical study artifact: {self.reason}"


@dataclass(frozen=True, slots=True)
class ClassicalStudyArtifactManifest:
    """Complete relocatable run identity embedded beside every result tensor."""

    format_version: int
    request_sha256: str
    source: StudySourceIdentity
    study_sha256: str
    execution: ClassicalExecutionSpec
    execution_sha256: str
    prepared_inputs_sha256: str
    observations: int
    variants: tuple[str, ...]
    submitted_result_names: tuple[str, ...]
    paper_package_version: str
    engine_package_version: str
    rustc_version: str
    tensors_sha256: str
    metrics_sha256: str

    def __post_init__(self) -> None:
        """Reject incomplete, inconsistent, or unsupported serialized run identity."""
        if type(self.format_version) is not int or self.format_version != _FORMAT_VERSION:
            raise ClassicalArtifactError(reason="format_version must be the plain int 2")
        _validate_manifest_hashes(self)
        if type(self.source) is not StudySourceIdentity:
            raise ClassicalArtifactError(reason="source must be StudySourceIdentity")
        if type(self.execution) is not ClassicalExecutionSpec:
            raise ClassicalArtifactError(reason="execution must be ClassicalExecutionSpec")
        if self.execution_sha256 != classical_execution_sha256(self.execution):
            raise ClassicalArtifactError(reason="execution_sha256 disagrees with execution")
        if type(self.observations) is not int or self.observations <= 0:
            raise ClassicalArtifactError(reason="observations must be a positive plain int")
        _validate_manifest_variants(self)
        _validate_manifest_versions(self)
        if self.request_sha256 != _request_sha256_from_manifest(self):
            raise ClassicalArtifactError(
                reason="request_sha256 disagrees with pre-pricing identity"
            )


@dataclass(frozen=True, slots=True)
class ClassicalStudyArtifact:
    """One verified content-addressed complete result file."""

    path: Path
    sha256: str
    manifest: ClassicalStudyArtifactManifest

    def __post_init__(self) -> None:
        """Reject incomplete handles and filename/content disagreement."""
        if not isinstance(self.path, Path):
            raise ClassicalArtifactError(reason="path must be pathlib.Path")
        _require_sha256(self.sha256, field="sha256")
        if self.path.name != f"{self.sha256}.safetensors":
            raise ClassicalArtifactError(reason="artifact path does not match its content hash")
        if type(self.manifest) is not ClassicalStudyArtifactManifest:
            raise ClassicalArtifactError(reason="manifest must be ClassicalStudyArtifactManifest")


@dataclass(frozen=True, slots=True)
class LoadedClassicalStudyArtifact:
    """Verified manifest, target, native diagnostics, and metrics from one artifact."""

    artifact: ClassicalStudyArtifact
    target: Float64Vector
    result: ClassicalStudyPricingResult

    def __post_init__(self) -> None:
        """Require complete immutable row agreement after decoding."""
        if type(self.artifact) is not ClassicalStudyArtifact:
            raise ClassicalArtifactError(reason="artifact must be ClassicalStudyArtifact")
        _validate_vector(
            self.target,
            field="target",
            dtype=_FLOAT64_DTYPE,
            observations=self.artifact.manifest.observations,
        )
        if self.target.flags.writeable:
            raise ClassicalArtifactError(reason="loaded target must be immutable")
        if type(self.result) is not ClassicalStudyPricingResult:
            raise ClassicalArtifactError(reason="result must be ClassicalStudyPricingResult")
        if self.result.rows != self.artifact.manifest.observations:
            raise ClassicalArtifactError(reason="loaded result rows disagree with the manifest")


def run_and_write_project_classical_study(
    directory: Path,
    source: StudySourceIdentity,
    inputs: PreparedClassicalInputs,
    execution: ClassicalExecutionSpec,
) -> ClassicalStudyArtifact:
    """Run the complete paired study, atomically persist it, and verify every loaded tensor."""
    result = run_project_classical_study(inputs, execution)
    return write_classical_study_artifact(directory, source, inputs, execution, result)


def write_classical_study_artifact(
    directory: Path,
    source: StudySourceIdentity,
    inputs: PreparedClassicalInputs,
    execution: ClassicalExecutionSpec,
    result: ClassicalStudyPricingResult,
) -> ClassicalStudyArtifact:
    """Atomically store one complete result and verify it through the public loader."""
    _validate_write_request(directory, source, inputs, execution, result)
    tensors = _result_tensors(inputs.targets, result)
    metrics_payload = {
        variant.variant.variant.value: asdict(variant.metrics) for variant in result.variants
    }
    metrics_json = _canonical_json(metrics_payload)
    native = build_info()
    paper_version = _paper_package_version()
    request_payload: dict[str, object] = {
        "source": asdict(source),
        "study_sha256": classical_study_sha256(result.study),
        "execution": asdict(execution),
        "execution_sha256": classical_execution_sha256(execution),
        "prepared_inputs_sha256": prepared_classical_inputs_sha256(inputs),
        "observations": result.rows,
        "variants": tuple(variant.variant.value for variant in result.study.variants),
        "submitted_result_names": tuple(
            variant.submitted_result_name for variant in result.study.variants
        ),
        "paper_package_version": paper_version,
        "engine_package_version": native.package_version,
        "rustc_version": native.rustc_version,
    }
    request_sha256 = _request_sha256(request_payload)
    manifest = ClassicalStudyArtifactManifest(
        format_version=_FORMAT_VERSION,
        request_sha256=request_sha256,
        source=source,
        study_sha256=classical_study_sha256(result.study),
        execution=execution,
        execution_sha256=classical_execution_sha256(execution),
        prepared_inputs_sha256=prepared_classical_inputs_sha256(inputs),
        observations=result.rows,
        variants=tuple(variant.variant.value for variant in result.study.variants),
        submitted_result_names=tuple(
            variant.submitted_result_name for variant in result.study.variants
        ),
        paper_package_version=paper_version,
        engine_package_version=native.package_version,
        rustc_version=native.rustc_version,
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
    loaded = load_classical_study_artifact(destination, study=result.study)
    if loaded.artifact.manifest != manifest or loaded.result != result:
        _require_results_exact(loaded.result, result)
    return loaded.artifact


def load_classical_study_artifact(
    path: Path,
    *,
    study: ClassicalStudySpec = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
) -> LoadedClassicalStudyArtifact:
    """Load only a self-consistent content-addressed complete classical artifact."""
    if not isinstance(path, Path):
        raise ClassicalArtifactError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
        raise ClassicalArtifactError(reason="path must be a regular .safetensors file")
    if type(study) is not ClassicalStudySpec:
        raise ClassicalArtifactError(reason="study must be ClassicalStudySpec")
    _require_sha256(path.stem, field="artifact filename")
    file_sha256 = _file_sha256(path)
    if file_sha256 != path.stem:
        raise ClassicalArtifactError(reason="artifact bytes do not match the filename hash")
    metadata, tensors = _read_payload(path, study)
    manifest = _decode_manifest(metadata["manifest"])
    if manifest.study_sha256 != classical_study_sha256(study):
        raise ClassicalArtifactError(reason="artifact and requested study identities differ")
    if set(tensors) != _expected_tensor_keys(study):
        raise ClassicalArtifactError(reason="artifact tensor keys are missing or unexpected")
    if _tensors_sha256(tensors) != manifest.tensors_sha256:
        raise ClassicalArtifactError(reason="result tensors disagree with the manifest hash")
    metrics = _decode_metrics(metadata["metrics"], study)
    if _text_sha256(metadata["metrics"]) != manifest.metrics_sha256:
        raise ClassicalArtifactError(reason="metrics disagree with the manifest hash")
    target = _loaded_float64_vector(
        tensors["target"],
        field="target",
        observations=manifest.observations,
    )
    variants = tuple(
        _loaded_variant(tensors, target, metrics, variant, manifest.observations)
        for variant in study.variants
    )
    result = ClassicalStudyPricingResult(
        study=study,
        rows=manifest.observations,
        variants=variants,
    )
    artifact = ClassicalStudyArtifact(path=path, sha256=file_sha256, manifest=manifest)
    return LoadedClassicalStudyArtifact(artifact=artifact, target=target, result=result)


def _validate_write_request(
    directory: Path,
    source: object,
    inputs: object,
    execution: object,
    result: object,
) -> None:
    if not isinstance(directory, Path) or not directory.is_dir() or directory.is_symlink():
        raise ClassicalArtifactError(reason="directory must be an existing non-symlink directory")
    if type(source) is not StudySourceIdentity:
        raise ClassicalArtifactError(reason="source must be StudySourceIdentity")
    if type(inputs) is not PreparedClassicalInputs:
        raise ClassicalArtifactError(reason="inputs must be PreparedClassicalInputs")
    if type(execution) is not ClassicalExecutionSpec:
        raise ClassicalArtifactError(reason="execution must be ClassicalExecutionSpec")
    if type(result) is not ClassicalStudyPricingResult:
        raise ClassicalArtifactError(reason="result must be ClassicalStudyPricingResult")
    if result.study != inputs.study or result.rows != inputs.rows:
        raise ClassicalArtifactError(reason="result and prepared input identities differ")
    execution.validate_rows(result.rows)


def _result_tensors(
    target: Float64Vector,
    result: ClassicalStudyPricingResult,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {"target": target}
    for variant in result.variants:
        for field in _NATIVE_FIELDS:
            tensors[_tensor_key(variant.variant.variant, field)] = cast(
                "np.ndarray",
                getattr(variant.native, field),
            )
    return tensors


def _loaded_variant(
    tensors: dict[str, np.ndarray],
    target: Float64Vector,
    metrics: dict[ClassicalVariant, ProjectPricingMetrics],
    variant: ClassicalVariantSpec,
    observations: int,
) -> ClassicalVariantPricingResult:
    identity = variant.variant

    def key(field: str) -> str:
        return _tensor_key(identity, field)

    native = LsmBatchResult(
        prices=_loaded_float64_vector(
            tensors[key("prices")], field=key("prices"), observations=observations
        ),
        sample_standard_deviations=_loaded_float64_vector(
            tensors[key("sample_standard_deviations")],
            field=key("sample_standard_deviations"),
            observations=observations,
        ),
        standard_errors=_loaded_float64_vector(
            tensors[key("standard_errors")],
            field=key("standard_errors"),
            observations=observations,
        ),
        origin_decisions_fitted=_loaded_bool_vector(
            tensors[key("origin_decisions_fitted")],
            field=key("origin_decisions_fitted"),
            observations=observations,
        ),
        origin_exercised=_loaded_bool_vector(
            tensors[key("origin_exercised")],
            field=key("origin_exercised"),
            observations=observations,
        ),
        regression_nodes=_loaded_uint32_vector(
            tensors[key("regression_nodes")],
            field=key("regression_nodes"),
            observations=observations,
        ),
        no_training_support_nodes=_loaded_uint32_vector(
            tensors[key("no_training_support_nodes")],
            field=key("no_training_support_nodes"),
            observations=observations,
        ),
        minimum_in_the_money_training_paths=_loaded_uint32_vector(
            tensors[key("minimum_in_the_money_training_paths")],
            field=key("minimum_in_the_money_training_paths"),
            observations=observations,
        ),
        maximum_regression_condition_numbers=_loaded_float64_vector(
            tensors[key("maximum_regression_condition_numbers")],
            field=key("maximum_regression_condition_numbers"),
            observations=observations,
        ),
        early_exercise_paths=_loaded_uint32_vector(
            tensors[key("early_exercise_paths")],
            field=key("early_exercise_paths"),
            observations=observations,
        ),
        expiry_paths=_loaded_uint32_vector(
            tensors[key("expiry_paths")],
            field=key("expiry_paths"),
            observations=observations,
        ),
    )
    stored_metrics = metrics[identity]
    try:
        recomputed = evaluate_project_pricing_f64(target, native.prices)
    except ProjectMetricError as error:
        raise ClassicalArtifactError(reason=str(error)) from error
    if stored_metrics != recomputed:
        raise ClassicalArtifactError(reason=f"stored metrics disagree for {identity.value!r}")
    return ClassicalVariantPricingResult(variant=variant, native=native, metrics=stored_metrics)


def _persist_artifact(
    directory: Path,
    *,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".classical-study-",
        suffix=".safetensors",
        dir=directory,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, temporary, metadata=metadata)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        sha256 = _file_sha256(temporary)
        destination = directory / f"{sha256}.safetensors"
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _file_sha256(destination) != sha256:
                raise ClassicalArtifactError(
                    reason="content-addressed destination contains different bytes"
                ) from None
        temporary.unlink()
    except (OSError, SafetensorError) as error:
        raise ClassicalArtifactError(reason=f"artifact write failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_payload(
    path: Path,
    study: ClassicalStudySpec,
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    try:
        with safe_open(str(path), framework="numpy") as reader:
            raw_metadata = reader.metadata()
        tensors = load_file(path)
    except (OSError, SafetensorError) as error:
        raise ClassicalArtifactError(reason=f"safe-tensors decode failed: {error}") from error
    if raw_metadata is None or set(raw_metadata) != {_METADATA_KEY}:
        raise ClassicalArtifactError(reason="artifact metadata keys are missing or unexpected")
    envelope = _decode_canonical_json(raw_metadata[_METADATA_KEY], field=_METADATA_KEY)
    if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
        raise ClassicalArtifactError(reason="artifact envelope keys are missing or unexpected")
    values = cast("dict[str, object]", envelope)
    if any(type(values[key]) is not str for key in _ENVELOPE_KEYS):
        raise ClassicalArtifactError(reason="artifact envelope values must be plain strings")
    decoded = cast("dict[str, str]", values)
    if decoded["format"] != _FORMAT:
        raise ClassicalArtifactError(reason="artifact format identifier is unsupported")
    if set(tensors) != _expected_tensor_keys(study):
        raise ClassicalArtifactError(reason="artifact tensor keys are missing or unexpected")
    return decoded, tensors


def _decode_manifest(value: str) -> ClassicalStudyArtifactManifest:
    payload = _decode_canonical_json(value, field="manifest")
    _require_exact_keys(payload, ClassicalStudyArtifactManifest, field="manifest")
    values = cast("dict[str, Any]", payload)
    try:
        values["source"] = StudySourceIdentity(**cast("dict[str, Any]", values["source"]))
        execution = cast("dict[str, Any]", values["execution"])
        execution["math_backend"] = CpuMathBackend(execution["math_backend"])
        values["execution"] = ClassicalExecutionSpec(**execution)
        values["variants"] = tuple(cast("list[str]", values["variants"]))
        values["submitted_result_names"] = tuple(
            cast("list[str]", values["submitted_result_names"])
        )
        return ClassicalStudyArtifactManifest(**values)
    except (TypeError, ValueError) as error:
        raise ClassicalArtifactError(reason=f"manifest fields are invalid: {error}") from error


def _decode_metrics(
    value: str,
    study: ClassicalStudySpec,
) -> dict[ClassicalVariant, ProjectPricingMetrics]:
    payload = _decode_canonical_json(value, field="metrics")
    expected = {variant.variant.value for variant in study.variants}
    if type(payload) is not dict or set(payload) != expected:
        raise ClassicalArtifactError(reason="metrics variants are missing or unexpected")
    decoded = cast("dict[str, object]", payload)
    metrics: dict[ClassicalVariant, ProjectPricingMetrics] = {}
    try:
        for variant in study.variants:
            metric_payload = decoded[variant.variant.value]
            _require_exact_keys(metric_payload, ProjectPricingMetrics, field="variant metrics")
            metrics[variant.variant] = ProjectPricingMetrics(
                **cast("dict[str, Any]", metric_payload)
            )
    except (TypeError, ProjectMetricError) as error:
        raise ClassicalArtifactError(reason=f"metrics fields are invalid: {error}") from error
    return metrics


def _loaded_float64_vector(
    tensor: np.ndarray,
    *,
    field: str,
    observations: int,
) -> Float64Vector:
    _validate_vector(tensor, field=field, dtype=_FLOAT64_DTYPE, observations=observations)
    result = cast("Float64Vector", np.ascontiguousarray(tensor, dtype=np.float64).copy())
    result.flags.writeable = False
    return result


def _loaded_uint32_vector(
    tensor: np.ndarray,
    *,
    field: str,
    observations: int,
) -> UInt32Vector:
    dtype = _NATIVE_DTYPES["regression_nodes"]
    _validate_vector(tensor, field=field, dtype=dtype, observations=observations)
    result = cast("UInt32Vector", np.ascontiguousarray(tensor, dtype=np.uint32).copy())
    result.flags.writeable = False
    return result


def _loaded_bool_vector(
    tensor: np.ndarray,
    *,
    field: str,
    observations: int,
) -> BoolVector:
    dtype = _NATIVE_DTYPES["origin_exercised"]
    _validate_vector(tensor, field=field, dtype=dtype, observations=observations)
    result = cast("BoolVector", np.ascontiguousarray(tensor, dtype=np.bool_).copy())
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
        raise ClassicalArtifactError(reason=f"{field} must be a plain numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if array.dtype != dtype or array.shape != (observations,) or not array.flags.c_contiguous:
        raise ClassicalArtifactError(
            reason=f"{field} must be a C-contiguous {dtype} vector of length {observations}"
        )
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise ClassicalArtifactError(reason=f"{field} must be finite")


def _expected_tensor_keys(study: ClassicalStudySpec) -> set[str]:
    return {
        "target",
        *(
            _tensor_key(variant.variant, field)
            for variant in study.variants
            for field in _NATIVE_FIELDS
        ),
    }


def _tensor_key(variant: ClassicalVariant, field: str) -> str:
    return f"{variant.value}.{field}"


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


def _request_sha256_from_manifest(manifest: ClassicalStudyArtifactManifest) -> str:
    return _request_sha256(
        {
            "source": asdict(manifest.source),
            "study_sha256": manifest.study_sha256,
            "execution": asdict(manifest.execution),
            "execution_sha256": manifest.execution_sha256,
            "prepared_inputs_sha256": manifest.prepared_inputs_sha256,
            "observations": manifest.observations,
            "variants": manifest.variants,
            "submitted_result_names": manifest.submitted_result_names,
            "paper_package_version": manifest.paper_package_version,
            "engine_package_version": manifest.engine_package_version,
            "rustc_version": manifest.rustc_version,
        }
    )


def _request_sha256(payload: dict[str, object]) -> str:
    request = {"format": _FORMAT, "request": payload}
    return hashlib.sha256(_canonical_json(request).encode("ascii")).hexdigest()


def _require_results_exact(
    observed: ClassicalStudyPricingResult,
    expected: ClassicalStudyPricingResult,
) -> None:
    if observed.study != expected.study or observed.rows != expected.rows:
        raise ClassicalArtifactError(reason="persisted study identity differs from requested state")
    for left, right in zip(observed.variants, expected.variants, strict=True):
        if left.variant != right.variant or left.metrics != right.metrics:
            raise ClassicalArtifactError(reason="persisted variant identity or metrics differ")
        for field in _NATIVE_FIELDS:
            if not np.array_equal(getattr(left.native, field), getattr(right.native, field)):
                raise ClassicalArtifactError(
                    reason=(
                        f"persisted native field {field!r} differs for "
                        f"{left.variant.variant.value!r}"
                    )
                )


def _validate_manifest_hashes(manifest: ClassicalStudyArtifactManifest) -> None:
    for field, value in (
        ("request_sha256", manifest.request_sha256),
        ("study_sha256", manifest.study_sha256),
        ("execution_sha256", manifest.execution_sha256),
        ("prepared_inputs_sha256", manifest.prepared_inputs_sha256),
        ("tensors_sha256", manifest.tensors_sha256),
        ("metrics_sha256", manifest.metrics_sha256),
    ):
        _require_sha256(value, field=field)


def _validate_manifest_variants(manifest: ClassicalStudyArtifactManifest) -> None:
    expected_variants = tuple(variant.value for variant in ClassicalVariant)
    if manifest.variants != expected_variants:
        raise ClassicalArtifactError(reason="variants must contain the complete canonical order")
    if (
        type(manifest.submitted_result_names) is not tuple
        or len(manifest.submitted_result_names) != len(manifest.variants)
        or any(type(value) is not str or not value for value in manifest.submitted_result_names)
        or len(set(manifest.submitted_result_names)) != len(manifest.submitted_result_names)
    ):
        raise ClassicalArtifactError(reason="submitted result names are incomplete or aliased")


def _validate_manifest_versions(manifest: ClassicalStudyArtifactManifest) -> None:
    for field, value in (
        ("paper_package_version", manifest.paper_package_version),
        ("engine_package_version", manifest.engine_package_version),
        ("rustc_version", manifest.rustc_version),
    ):
        if type(value) is not str or not value:
            raise ClassicalArtifactError(reason=f"{field} must be a nonempty plain string")


def _paper_package_version() -> str:
    try:
        observed = version(_PAPER_PACKAGE)
    except PackageNotFoundError as error:
        raise ClassicalArtifactError(reason="paper package is not installed") from error
    if not observed:
        raise ClassicalArtifactError(reason="paper package version must be nonempty")
    return observed


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as error:
        raise ClassicalArtifactError(reason=f"artifact hashing failed: {error}") from error
    return hasher.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _decode_canonical_json(value: object, *, field: str) -> object:
    if type(value) is not str:
        raise ClassicalArtifactError(reason=f"{field} metadata must be a plain string")
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise ClassicalArtifactError(reason=f"{field} is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise ClassicalArtifactError(reason=f"{field} JSON is not canonical")
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
        raise ClassicalArtifactError(reason=f"value cannot be canonical JSON: {error}") from error


def _require_exact_keys(value: object, model: type[object], *, field: str) -> None:
    if type(value) is not dict:
        raise ClassicalArtifactError(reason=f"{field} must be an object")
    expected = {item.name for item in fields(model)}
    if set(cast("dict[object, object]", value)) != expected:
        raise ClassicalArtifactError(reason=f"{field} keys are missing or unexpected")


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ClassicalArtifactError(reason=f"{field} must be a lowercase SHA-256 hex digest")


__all__ = [
    "ClassicalArtifactError",
    "ClassicalStudyArtifact",
    "ClassicalStudyArtifactManifest",
    "LoadedClassicalStudyArtifact",
    "load_classical_study_artifact",
    "run_and_write_project_classical_study",
    "write_classical_study_artifact",
]
