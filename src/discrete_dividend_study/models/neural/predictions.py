"""Safe, content-addressed neural test predictions and exact project metrics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

import numpy as np
import numpy.typing as npt
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

from discrete_dividend_study.core.serialization import dataclass_field_names
from discrete_dividend_study.experiments.metrics import (
    ProjectMetricError,
    ProjectPricingMetrics,
    evaluate_project_pricing,
)
from discrete_dividend_study.experiments.windows import RollingPhase, RollingWindow
from discrete_dividend_study.models.neural.checkpoints import NeuralCheckpointArtifact
from discrete_dividend_study.models.neural.inputs import PreparedNeuralBatch
from discrete_dividend_study.models.neural.specification import (
    NeuralStudySpec,
    NeuralVariant,
    neural_study_sha256,
)

Float32Vector = npt.NDArray[np.float32]

_FORMAT: Final = "discrete-dividend-neural-prediction-v1"
_METADATA_KEYS: Final = frozenset({"prediction"})
_ENVELOPE_KEYS: Final = frozenset({"format", "manifest", "metrics"})
_TENSOR_KEYS: Final = frozenset({"prediction", "target"})
_SHA256_HEX_LENGTH: Final = 64
_VARIANT_VALUES: Final = frozenset(variant.value for variant in NeuralVariant)
_FLOAT32_DTYPE: Final = np.dtype(np.float32)


class _HashWriter(Protocol):
    """The buffer-writing subset shared by hashlib hash implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(slots=True)
class NeuralPredictionError(RuntimeError):
    """A prediction violates numerical, identity, or immutable-storage contracts."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected prediction contract."""
        return f"invalid neural prediction artifact: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralPredictionManifest:
    """Complete relocatable identity embedded beside target and prediction tensors."""

    format_version: int
    study_sha256: str
    variant: str
    submitted_result_name: str
    window_id: int
    phase: str
    test_data_sha256: str
    checkpoint_sha256: str
    observations: int
    target_sha256: str
    prediction_sha256: str
    metrics_sha256: str

    def __post_init__(self) -> None:
        """Reject malformed serialized identity without coercion."""
        if type(self.format_version) is not int or self.format_version != 1:
            raise NeuralPredictionError(reason="format_version must be the plain int 1")
        for field, value in (
            ("study_sha256", self.study_sha256),
            ("test_data_sha256", self.test_data_sha256),
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("target_sha256", self.target_sha256),
            ("prediction_sha256", self.prediction_sha256),
            ("metrics_sha256", self.metrics_sha256),
        ):
            _require_sha256(value, field=field)
        if type(self.variant) is not str or self.variant not in _VARIANT_VALUES:
            raise NeuralPredictionError(reason="variant is not a supported serialized value")
        if type(self.submitted_result_name) is not str or not self.submitted_result_name:
            raise NeuralPredictionError(reason="submitted_result_name must be a nonempty plain str")
        if type(self.window_id) is not int or self.window_id < 0:
            raise NeuralPredictionError(reason="window_id must be a nonnegative plain int")
        if self.phase != RollingPhase.TEST.value or type(self.phase) is not str:
            raise NeuralPredictionError(reason="phase must be the serialized test phase")
        if type(self.observations) is not int or self.observations <= 0:
            raise NeuralPredictionError(reason="observations must be a positive plain int")


@dataclass(frozen=True, slots=True)
class NeuralPredictionArtifact:
    """One verified content-addressed prediction file."""

    path: Path
    sha256: str
    manifest: NeuralPredictionManifest
    metrics: ProjectPricingMetrics

    def __post_init__(self) -> None:
        """Reject incomplete artifact handles."""
        if not isinstance(self.path, Path):
            raise NeuralPredictionError(reason="path must be pathlib.Path")
        _require_sha256(self.sha256, field="sha256")
        if type(self.manifest) is not NeuralPredictionManifest:
            raise NeuralPredictionError(reason="manifest must be NeuralPredictionManifest")
        if type(self.metrics) is not ProjectPricingMetrics:
            raise NeuralPredictionError(reason="metrics must be ProjectPricingMetrics")


@dataclass(frozen=True, slots=True)
class LoadedNeuralPrediction:
    """Verified prediction metadata and read-only float32 vectors."""

    artifact: NeuralPredictionArtifact
    target: Float32Vector
    prediction: Float32Vector

    def __post_init__(self) -> None:
        """Reject partially decoded prediction state."""
        if type(self.artifact) is not NeuralPredictionArtifact:
            raise NeuralPredictionError(reason="artifact must be NeuralPredictionArtifact")
        _validate_vector(
            self.target, field="target", observations=self.artifact.manifest.observations
        )
        _validate_vector(
            self.prediction,
            field="prediction",
            observations=self.artifact.manifest.observations,
        )
        if self.target.flags.writeable or self.prediction.flags.writeable:
            raise NeuralPredictionError(reason="loaded prediction vectors must be read-only")


@dataclass(frozen=True, slots=True)
class NeuralPredictionRequest:
    """One complete model, test phase, checkpoint, and prediction write identity."""

    study: NeuralStudySpec
    window: RollingWindow
    test_batch: PreparedNeuralBatch
    test_data_sha256: str
    checkpoint: NeuralCheckpointArtifact
    prediction: Float32Vector

    def __post_init__(self) -> None:
        """Reject incomplete or mutable prediction state before artifact I/O."""
        if type(self.study) is not NeuralStudySpec:
            raise NeuralPredictionError(reason="study must be NeuralStudySpec")
        if type(self.window) is not RollingWindow:
            raise NeuralPredictionError(reason="window must be RollingWindow")
        if type(self.test_batch) is not PreparedNeuralBatch:
            raise NeuralPredictionError(reason="test_batch must be PreparedNeuralBatch")
        if type(self.checkpoint) is not NeuralCheckpointArtifact:
            raise NeuralPredictionError(reason="checkpoint must be NeuralCheckpointArtifact")
        _require_sha256(self.test_data_sha256, field="test_data_sha256")
        _validate_vector(
            self.prediction,
            field="prediction",
            observations=self.test_batch.rows,
        )
        if self.prediction.flags.writeable:
            raise NeuralPredictionError(reason="prediction storage must be read-only")
        if np.any(self.prediction < 0.0):
            raise NeuralPredictionError(reason="project price predictions must be nonnegative")
        _validate_checkpoint_identity(
            self.study,
            self.test_batch,
            self.test_data_sha256,
            self.checkpoint,
        )


def write_neural_prediction(
    directory: Path,
    request: NeuralPredictionRequest,
) -> NeuralPredictionArtifact:
    """Atomically persist one test phase and verify it by loading the complete file."""
    _validate_write_request(directory, request)
    try:
        metrics = evaluate_project_pricing(request.test_batch.target, request.prediction)
    except ProjectMetricError as error:
        raise NeuralPredictionError(reason=str(error)) from error
    metrics_json = _canonical_json(asdict(metrics))
    manifest = NeuralPredictionManifest(
        format_version=1,
        study_sha256=neural_study_sha256(request.study),
        variant=request.test_batch.model.variant.value,
        submitted_result_name=request.test_batch.model.submitted_result_name,
        window_id=request.window.window_id,
        phase=RollingPhase.TEST.value,
        test_data_sha256=request.test_data_sha256,
        checkpoint_sha256=request.checkpoint.sha256,
        observations=request.test_batch.rows,
        target_sha256=_vector_sha256("target", request.test_batch.target),
        prediction_sha256=_vector_sha256("prediction", request.prediction),
        metrics_sha256=_text_sha256(metrics_json),
    )
    envelope = {
        "format": _FORMAT,
        "manifest": _canonical_json(asdict(manifest)),
        "metrics": metrics_json,
    }
    destination = _persist_prediction(
        directory,
        tensors={
            "prediction": request.prediction,
            "target": request.test_batch.target,
        },
        metadata={"prediction": _canonical_json(envelope)},
    )
    loaded = load_neural_prediction(destination)
    if loaded.artifact.manifest != manifest or loaded.artifact.metrics != metrics:
        raise NeuralPredictionError(reason="persisted prediction differs from requested state")
    return loaded.artifact


def load_neural_prediction(path: Path) -> LoadedNeuralPrediction:
    """Load only self-consistent content-addressed safe-tensors predictions."""
    if not isinstance(path, Path):
        raise NeuralPredictionError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
        raise NeuralPredictionError(reason="path must be a regular .safetensors file")
    _require_sha256(path.stem, field="prediction filename")
    file_sha256 = _file_sha256(path)
    if file_sha256 != path.stem:
        raise NeuralPredictionError(reason="prediction bytes do not match the filename hash")
    metadata, tensors = _read_payload(path)
    manifest = _decode_manifest(metadata["manifest"])
    metrics = _decode_metrics(metadata["metrics"])
    target = _loaded_vector(tensors["target"], field="target", observations=manifest.observations)
    prediction = _loaded_vector(
        tensors["prediction"],
        field="prediction",
        observations=manifest.observations,
    )
    _validate_loaded_payload(manifest, metrics, metadata, target, prediction)
    artifact = NeuralPredictionArtifact(
        path=path,
        sha256=file_sha256,
        manifest=manifest,
        metrics=metrics,
    )
    return LoadedNeuralPrediction(artifact=artifact, target=target, prediction=prediction)


def _validate_write_request(
    directory: Path,
    request: object,
) -> None:
    if not isinstance(directory, Path) or not directory.is_dir() or directory.is_symlink():
        raise NeuralPredictionError(reason="directory must be an existing non-symlink directory")
    if type(request) is not NeuralPredictionRequest:
        raise NeuralPredictionError(reason="request must be NeuralPredictionRequest")


def _validate_checkpoint_identity(
    study: NeuralStudySpec,
    test_batch: PreparedNeuralBatch,
    test_data_sha256: str,
    checkpoint: NeuralCheckpointArtifact,
) -> None:
    expected_study = neural_study_sha256(study)
    manifest = checkpoint.manifest
    if (
        manifest.study_sha256 != expected_study
        or manifest.variant != test_batch.model.variant.value
        or manifest.submitted_result_name != test_batch.model.submitted_result_name
    ):
        raise NeuralPredictionError(reason="checkpoint, study, and test model identities differ")
    if test_data_sha256 in {manifest.training_data_sha256, manifest.validation_data_sha256}:
        raise NeuralPredictionError(reason="test data hash must differ from fitted phase hashes")


def _persist_prediction(
    directory: Path,
    *,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".neural-prediction-",
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
                raise NeuralPredictionError(
                    reason="content-addressed destination contains different bytes",
                ) from None
        temporary.unlink()
    except (OSError, SafetensorError) as error:
        raise NeuralPredictionError(reason=f"prediction write failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_payload(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    try:
        with safe_open(str(path), framework="numpy") as reader:
            raw_metadata = reader.metadata()
        tensors = load_file(path)
    except (OSError, SafetensorError) as error:
        raise NeuralPredictionError(reason=f"safe-tensors decode failed: {error}") from error
    if raw_metadata is None or set(raw_metadata) != _METADATA_KEYS:
        raise NeuralPredictionError(reason="prediction metadata keys are missing or unexpected")
    if set(tensors) != _TENSOR_KEYS:
        raise NeuralPredictionError(reason="prediction tensor keys are missing or unexpected")
    envelope = _decode_canonical_json(raw_metadata["prediction"], field="prediction")
    if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
        raise NeuralPredictionError(reason="prediction envelope keys are missing or unexpected")
    values = cast("dict[str, object]", envelope)
    if any(type(values[key]) is not str for key in _ENVELOPE_KEYS):
        raise NeuralPredictionError(reason="prediction envelope values must be plain strings")
    decoded = cast("dict[str, str]", values)
    if decoded["format"] != _FORMAT:
        raise NeuralPredictionError(reason="prediction format identifier is unsupported")
    return decoded, tensors


def _decode_manifest(value: str) -> NeuralPredictionManifest:
    payload = _decode_canonical_json(value, field="manifest")
    _require_exact_keys(payload, NeuralPredictionManifest, field="manifest")
    try:
        return NeuralPredictionManifest(**cast("dict[str, Any]", payload))
    except TypeError as error:
        raise NeuralPredictionError(reason=f"manifest fields are invalid: {error}") from error


def _decode_metrics(value: str) -> ProjectPricingMetrics:
    payload = _decode_canonical_json(value, field="metrics")
    _require_exact_keys(payload, ProjectPricingMetrics, field="metrics")
    try:
        return ProjectPricingMetrics(**cast("dict[str, Any]", payload))
    except (TypeError, ProjectMetricError) as error:
        raise NeuralPredictionError(reason=f"metrics fields are invalid: {error}") from error


def _loaded_vector(tensor: np.ndarray, *, field: str, observations: int) -> Float32Vector:
    _validate_vector(tensor, field=field, observations=observations)
    result = np.ascontiguousarray(tensor, dtype=np.float32).copy()
    result.flags.writeable = False
    return result


def _validate_loaded_payload(
    manifest: NeuralPredictionManifest,
    metrics: ProjectPricingMetrics,
    metadata: dict[str, str],
    target: Float32Vector,
    prediction: Float32Vector,
) -> None:
    if _text_sha256(metadata["metrics"]) != manifest.metrics_sha256:
        raise NeuralPredictionError(reason="metrics hash does not match the manifest")
    if _vector_sha256("target", target) != manifest.target_sha256:
        raise NeuralPredictionError(reason="target hash does not match the manifest")
    if _vector_sha256("prediction", prediction) != manifest.prediction_sha256:
        raise NeuralPredictionError(reason="prediction hash does not match the manifest")
    try:
        recomputed = evaluate_project_pricing(target, prediction)
    except ProjectMetricError as error:
        raise NeuralPredictionError(reason=str(error)) from error
    if not _metrics_close(metrics, recomputed):
        raise NeuralPredictionError(reason="stored metrics disagree with prediction tensors")


def _metrics_close(left: ProjectPricingMetrics, right: ProjectPricingMetrics) -> bool:
    if left.observations != right.observations:
        return False
    return all(
        math.isclose(
            cast("float", getattr(left, field)),
            cast("float", getattr(right, field)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for field in (
            "mae",
            "mape_percent",
            "mse",
            "rmse",
            "bias",
            "relative_bias",
            "smape_percent",
        )
    )


def _validate_vector(value: object, *, field: str, observations: int) -> None:
    if type(value) is not np.ndarray:
        raise NeuralPredictionError(reason=f"{field} must be a plain numpy.ndarray")
    array = cast("npt.NDArray[np.generic]", value)
    if (
        array.dtype != _FLOAT32_DTYPE
        or array.shape != (observations,)
        or not array.flags.c_contiguous
    ):
        raise NeuralPredictionError(
            reason=f"{field} must be a C-contiguous float32 vector of length {observations}",
        )
    if not np.isfinite(array).all():
        raise NeuralPredictionError(reason=f"{field} must be finite")


def _vector_sha256(label: str, array: Float32Vector) -> str:
    hasher = hashlib.sha256()
    _hash_text(hasher, _FORMAT)
    _hash_text(hasher, label)
    _hash_text(hasher, array.dtype.str)
    hasher.update(array.size.to_bytes(8, byteorder="little", signed=False))
    hasher.update(memoryview(array).cast("B"))
    return hasher.hexdigest()


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _decode_canonical_json(value: object, *, field: str) -> object:
    if type(value) is not str:
        raise NeuralPredictionError(reason=f"{field} metadata must be a plain str")
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise NeuralPredictionError(reason=f"{field} is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise NeuralPredictionError(reason=f"{field} JSON is not canonical")
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
        raise NeuralPredictionError(reason=f"value cannot be canonical JSON: {error}") from error


def _require_exact_keys(value: object, model: type[object], *, field: str) -> None:
    if type(value) is not dict:
        raise NeuralPredictionError(reason=f"{field} must be an object")
    expected = dataclass_field_names(model)
    if set(cast("dict[object, object]", value)) != expected:
        raise NeuralPredictionError(reason=f"{field} keys are missing or unexpected")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as error:
        raise NeuralPredictionError(reason=f"prediction hashing failed: {error}") from error
    return hasher.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NeuralPredictionError(reason=f"{field} must be a lowercase SHA-256 hex digest")


__all__ = [
    "LoadedNeuralPrediction",
    "NeuralPredictionArtifact",
    "NeuralPredictionError",
    "NeuralPredictionManifest",
    "NeuralPredictionRequest",
    "load_neural_prediction",
    "write_neural_prediction",
]
