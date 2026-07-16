"""Safe, immutable, content-addressed neural model checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file, save_file

from discrete_dividend_study.models.neural.execution import (
    CompilationMode,
    NeuralBackend,
    NeuralDevice,
    NeuralRuntimeIdentity,
)
from discrete_dividend_study.models.neural.parameters import NeuralParameter
from discrete_dividend_study.models.neural.specification import (
    ModelPrecision,
    NeuralSpecValueError,
    NeuralVariant,
    neural_study_sha256,
)
from discrete_dividend_study.models.neural.training import (
    NeuralEpochMetrics,
    NeuralTrainingError,
    NeuralTrainingRequest,
    NeuralTrainingSummary,
    validate_neural_training_summary,
)
from discrete_dividend_study.models.neural.training_control import (
    NeuralControlError,
    NeuralControlState,
)

_FORMAT: Final = "discrete-dividend-neural-v1"
_METADATA_KEYS: Final = frozenset({"checkpoint"})
_ENVELOPE_KEYS: Final = frozenset({"format", "manifest", "training_summary"})
_MAX_SEED: Final = (2**64) - 1
_SHA256_HEX_LENGTH: Final = 64
_VARIANT_VALUES: Final = frozenset(variant.value for variant in NeuralVariant)
_BACKEND_VALUES: Final = frozenset(backend.value for backend in NeuralBackend)
_DEVICE_VALUES: Final = frozenset(device.value for device in NeuralDevice)
_PRECISION_VALUES: Final = frozenset(precision.value for precision in ModelPrecision)
_COMPILATION_VALUES: Final = frozenset(mode.value for mode in CompilationMode)


@dataclass(slots=True)
class NeuralCheckpointError(RuntimeError):
    """A checkpoint violates safe storage or immutable identity contracts."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected checkpoint contract."""
        return f"invalid neural checkpoint: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralCheckpointManifest:
    """Small complete identity embedded beside the safe tensor payload."""

    format_version: int
    study_sha256: str
    variant: str
    submitted_result_name: str
    training_data_sha256: str
    validation_data_sha256: str
    initial_parameters_sha256: str
    parameter_seed: int
    shuffle_seed: int
    backend: str
    backend_version: str
    device: str
    precision: str
    compilation: str
    cpu_threads: int | None
    operating_system: str
    machine: str
    completed_epochs: int
    best_epoch: int
    best_validation_loss: float
    training_summary_sha256: str
    trained_parameters_sha256: str
    parameter_count: int

    def __post_init__(self) -> None:
        """Reject malformed identities without coercing serialized values."""
        _validate_manifest_identity(self)


def _validate_manifest_identity(manifest: NeuralCheckpointManifest) -> None:
    if manifest.format_version != 1 or type(manifest.format_version) is not int:
        raise NeuralCheckpointError(reason="format_version must be the plain int 1")
    _validate_manifest_hashes(manifest)
    _validate_manifest_strings(manifest)
    _validate_manifest_numbers(manifest)
    _validate_manifest_runtime(manifest)


def _validate_manifest_hashes(manifest: NeuralCheckpointManifest) -> None:
    for field, value in (
        ("study_sha256", manifest.study_sha256),
        ("training_data_sha256", manifest.training_data_sha256),
        ("validation_data_sha256", manifest.validation_data_sha256),
        ("initial_parameters_sha256", manifest.initial_parameters_sha256),
        ("training_summary_sha256", manifest.training_summary_sha256),
        ("trained_parameters_sha256", manifest.trained_parameters_sha256),
    ):
        _require_sha256(value, field=field)
    if manifest.training_data_sha256 == manifest.validation_data_sha256:
        raise NeuralCheckpointError(reason="training and validation data hashes must differ")


def _validate_manifest_strings(manifest: NeuralCheckpointManifest) -> None:
    _require_enum_value(manifest.variant, _VARIANT_VALUES, field="variant")
    _require_enum_value(manifest.backend, _BACKEND_VALUES, field="backend")
    _require_enum_value(manifest.device, _DEVICE_VALUES, field="device")
    _require_enum_value(manifest.precision, _PRECISION_VALUES, field="precision")
    _require_enum_value(manifest.compilation, _COMPILATION_VALUES, field="compilation")
    for field, value in (
        ("submitted_result_name", manifest.submitted_result_name),
        ("backend_version", manifest.backend_version),
        ("operating_system", manifest.operating_system),
        ("machine", manifest.machine),
    ):
        if type(value) is not str or not value:
            raise NeuralCheckpointError(reason=f"{field} must be a nonempty plain str")


def _validate_manifest_numbers(manifest: NeuralCheckpointManifest) -> None:
    for field, value in (
        ("parameter_seed", manifest.parameter_seed),
        ("shuffle_seed", manifest.shuffle_seed),
    ):
        if type(value) is not int or not 0 <= value <= _MAX_SEED:
            raise NeuralCheckpointError(reason=f"{field} must be an unsigned 64-bit plain int")
    if manifest.cpu_threads is not None and (
        type(manifest.cpu_threads) is not int or manifest.cpu_threads <= 0
    ):
        raise NeuralCheckpointError(reason="cpu_threads must be None or a positive plain int")
    if type(manifest.completed_epochs) is not int or manifest.completed_epochs <= 0:
        raise NeuralCheckpointError(reason="completed_epochs must be a positive plain int")
    if (
        type(manifest.best_epoch) is not int
        or not 0 <= manifest.best_epoch < manifest.completed_epochs
    ):
        raise NeuralCheckpointError(reason="best_epoch must identify a completed epoch")
    if (
        type(manifest.best_validation_loss) is not float
        or not math.isfinite(manifest.best_validation_loss)
        or manifest.best_validation_loss < 0.0
    ):
        raise NeuralCheckpointError(reason="best_validation_loss must be finite and nonnegative")
    if type(manifest.parameter_count) is not int or manifest.parameter_count <= 0:
        raise NeuralCheckpointError(reason="parameter_count must be a positive plain int")


def _validate_manifest_runtime(manifest: NeuralCheckpointManifest) -> None:
    allowed_devices = {
        NeuralBackend.MLX.value: {NeuralDevice.CPU.value, NeuralDevice.MLX_GPU.value},
        NeuralBackend.PYTORCH.value: {
            NeuralDevice.CPU.value,
            NeuralDevice.MPS.value,
            NeuralDevice.CUDA.value,
        },
    }
    if manifest.device not in allowed_devices[manifest.backend]:
        raise NeuralCheckpointError(reason="serialized backend and device are incompatible")
    if (
        manifest.backend == NeuralBackend.PYTORCH.value
        and manifest.compilation != CompilationMode.EAGER.value
    ):
        raise NeuralCheckpointError(reason="serialized PyTorch execution must be eager")
    is_torch_cpu = (
        manifest.backend == NeuralBackend.PYTORCH.value
        and manifest.device == NeuralDevice.CPU.value
    )
    if is_torch_cpu != (manifest.cpu_threads is not None):
        raise NeuralCheckpointError(
            reason="cpu_threads must be present exactly for PyTorch CPU execution"
        )


@dataclass(frozen=True, slots=True)
class NeuralCheckpointArtifact:
    """One verified content-addressed checkpoint written to local storage."""

    path: Path
    sha256: str
    manifest: NeuralCheckpointManifest

    def __post_init__(self) -> None:
        """Reject untyped artifact handles."""
        if not isinstance(self.path, Path):
            raise NeuralCheckpointError(reason="path must be pathlib.Path")
        _require_sha256(self.sha256, field="sha256")
        if type(self.manifest) is not NeuralCheckpointManifest:
            raise NeuralCheckpointError(reason="manifest must be NeuralCheckpointManifest")


@dataclass(frozen=True, slots=True)
class LoadedNeuralCheckpoint:
    """Fully verified metadata, metric history, and immutable NumPy tensors."""

    artifact: NeuralCheckpointArtifact
    summary: NeuralTrainingSummary
    parameters: tuple[NeuralParameter, ...]

    def __post_init__(self) -> None:
        """Reject partially decoded checkpoint state."""
        if type(self.artifact) is not NeuralCheckpointArtifact:
            raise NeuralCheckpointError(reason="artifact must be NeuralCheckpointArtifact")
        if type(self.summary) is not NeuralTrainingSummary:
            raise NeuralCheckpointError(reason="summary must be NeuralTrainingSummary")
        if (
            type(self.parameters) is not tuple
            or not self.parameters
            or any(type(parameter) is not NeuralParameter for parameter in self.parameters)
        ):
            raise NeuralCheckpointError(
                reason="parameters must be a nonempty tuple[NeuralParameter, ...]"
            )


def write_neural_checkpoint(
    directory: Path,
    *,
    request: NeuralTrainingRequest,
    runtime: NeuralRuntimeIdentity,
    summary: NeuralTrainingSummary,
    parameters: tuple[NeuralParameter, ...],
) -> NeuralCheckpointArtifact:
    """Atomically persist a best-restored model without pickle or mutable aliases."""
    _validate_checkpoint_write_request(directory, request, runtime, summary)
    _require_runtime_matches_request(runtime, request)
    validate_neural_training_summary(summary, request.study.training)
    tensors = _validated_parameter_tensors(parameters, request)
    summary_json = _canonical_json(asdict(summary))
    trained_sha256 = _parameter_sha256(parameters)
    control = summary.control
    if control.best_epoch is None or control.best_validation_loss is None:
        raise NeuralCheckpointError(reason="summary has no best validation checkpoint")
    manifest = NeuralCheckpointManifest(
        format_version=1,
        study_sha256=neural_study_sha256(request.study),
        variant=request.parameters.model.variant.value,
        submitted_result_name=request.parameters.model.submitted_result_name,
        training_data_sha256=request.training_data_sha256,
        validation_data_sha256=request.validation_data_sha256,
        initial_parameters_sha256=request.parameters.sha256,
        parameter_seed=request.parameters.seed,
        shuffle_seed=request.shuffle_seed,
        backend=runtime.backend.value,
        backend_version=runtime.backend_version,
        device=runtime.device.value,
        precision=runtime.precision.value,
        compilation=runtime.compilation.value,
        cpu_threads=runtime.cpu_threads,
        operating_system=runtime.operating_system,
        machine=runtime.machine,
        completed_epochs=control.completed_epochs,
        best_epoch=control.best_epoch,
        best_validation_loss=control.best_validation_loss,
        training_summary_sha256=_text_sha256(summary_json),
        trained_parameters_sha256=trained_sha256,
        parameter_count=sum(parameter.values.size for parameter in parameters),
    )
    envelope = {
        "format": _FORMAT,
        "manifest": _canonical_json(asdict(manifest)),
        "training_summary": summary_json,
    }
    metadata = {"checkpoint": _canonical_json(envelope)}
    destination = _persist_checkpoint(directory, tensors=tensors, metadata=metadata)
    loaded = load_neural_checkpoint(destination)
    if loaded.artifact.manifest != manifest:
        raise NeuralCheckpointError(reason="persisted manifest differs from requested manifest")
    return loaded.artifact


def _persist_checkpoint(
    directory: Path,
    *,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".neural-checkpoint-",
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
                raise NeuralCheckpointError(
                    reason="content-addressed destination contains different bytes"
                ) from None
        temporary.unlink()
    except (OSError, SafetensorError) as error:
        raise NeuralCheckpointError(reason=f"checkpoint write failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_neural_checkpoint(path: Path) -> LoadedNeuralCheckpoint:
    """Load only a self-consistent content-addressed safe-tensors checkpoint."""
    if not isinstance(path, Path):
        raise NeuralCheckpointError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
        raise NeuralCheckpointError(reason="path must be a regular .safetensors file")
    _require_sha256(path.stem, field="checkpoint filename")
    file_sha256, metadata, tensors = _read_checkpoint_payload(path)
    manifest = _decode_manifest(metadata)
    summary = _decode_training_summary(metadata["training_summary"])
    parameters = _loaded_parameters(tensors)
    _validate_loaded_payload(manifest, metadata, summary, parameters)
    artifact = NeuralCheckpointArtifact(path=path, sha256=file_sha256, manifest=manifest)
    return LoadedNeuralCheckpoint(
        artifact=artifact,
        summary=summary,
        parameters=parameters,
    )


def _read_checkpoint_payload(
    path: Path,
) -> tuple[str, dict[str, str], dict[str, np.ndarray]]:
    file_sha256 = _file_sha256(path)
    if file_sha256 != path.stem:
        raise NeuralCheckpointError(reason="checkpoint bytes do not match the filename hash")
    try:
        with safe_open(str(path), framework="numpy") as reader:
            metadata = reader.metadata()
        tensors = load_file(path)
    except (OSError, SafetensorError) as error:
        raise NeuralCheckpointError(reason=f"safe-tensors decode failed: {error}") from error
    if metadata is None or set(metadata) != _METADATA_KEYS:
        raise NeuralCheckpointError(reason="checkpoint metadata keys are missing or unexpected")
    envelope = _decode_canonical_json(metadata["checkpoint"], field="checkpoint")
    if type(envelope) is not dict or set(envelope) != _ENVELOPE_KEYS:
        raise NeuralCheckpointError(reason="checkpoint envelope keys are missing or unexpected")
    values = cast("dict[str, object]", envelope)
    if any(type(values[key]) is not str for key in _ENVELOPE_KEYS):
        raise NeuralCheckpointError(reason="checkpoint envelope values must be plain strings")
    decoded = cast("dict[str, str]", values)
    if decoded["format"] != _FORMAT:
        raise NeuralCheckpointError(reason="checkpoint format identifier is unsupported")
    return file_sha256, decoded, tensors


def _decode_manifest(metadata: dict[str, str]) -> NeuralCheckpointManifest:
    manifest_payload = _decode_canonical_json(metadata["manifest"], field="manifest")
    if type(manifest_payload) is not dict:
        raise NeuralCheckpointError(reason="manifest must decode to an object")
    _require_exact_keys(manifest_payload, NeuralCheckpointManifest, field="manifest")
    try:
        manifest_values = cast("dict[str, Any]", manifest_payload)
        return NeuralCheckpointManifest(**manifest_values)
    except TypeError as error:
        raise NeuralCheckpointError(reason=f"manifest fields are invalid: {error}") from error


def _validate_loaded_payload(
    manifest: NeuralCheckpointManifest,
    metadata: dict[str, str],
    summary: NeuralTrainingSummary,
    parameters: tuple[NeuralParameter, ...],
) -> None:
    if _text_sha256(metadata["training_summary"]) != manifest.training_summary_sha256:
        raise NeuralCheckpointError(reason="training summary hash does not match the manifest")
    if _parameter_sha256(parameters) != manifest.trained_parameters_sha256:
        raise NeuralCheckpointError(reason="trained parameter hash does not match the manifest")
    if sum(parameter.values.size for parameter in parameters) != manifest.parameter_count:
        raise NeuralCheckpointError(reason="trained parameter count does not match the manifest")
    if (
        summary.control.completed_epochs != manifest.completed_epochs
        or summary.control.best_epoch != manifest.best_epoch
        or summary.control.best_validation_loss != manifest.best_validation_loss
    ):
        raise NeuralCheckpointError(reason="training summary disagrees with the manifest")


def _validate_checkpoint_write_request(
    directory: Path,
    request: NeuralTrainingRequest,
    runtime: NeuralRuntimeIdentity,
    summary: NeuralTrainingSummary,
) -> None:
    if not isinstance(directory, Path):
        raise NeuralCheckpointError(reason="directory must be pathlib.Path")
    if not directory.is_dir() or directory.is_symlink():
        raise NeuralCheckpointError(reason="directory must be an existing non-symlink directory")
    if type(request) is not NeuralTrainingRequest:
        raise NeuralCheckpointError(reason="request must be NeuralTrainingRequest")
    if type(runtime) is not NeuralRuntimeIdentity:
        raise NeuralCheckpointError(reason="runtime must be NeuralRuntimeIdentity")
    if type(summary) is not NeuralTrainingSummary:
        raise NeuralCheckpointError(reason="summary must be NeuralTrainingSummary")


def _require_runtime_matches_request(
    runtime: NeuralRuntimeIdentity,
    request: NeuralTrainingRequest,
) -> None:
    execution = request.execution
    if (
        runtime.backend is not execution.backend
        or runtime.device is not execution.device
        or runtime.precision is not execution.precision
        or runtime.compilation is not execution.compilation
        or runtime.cpu_threads != execution.cpu_threads
    ):
        raise NeuralCheckpointError(reason="resolved runtime does not match the execution request")
    for field, value in (
        ("backend_version", runtime.backend_version),
        ("operating_system", runtime.operating_system),
        ("machine", runtime.machine),
    ):
        if type(value) is not str or not value:
            raise NeuralCheckpointError(reason=f"runtime {field} must be a nonempty plain str")


def _validated_parameter_tensors(
    parameters: tuple[NeuralParameter, ...],
    request: NeuralTrainingRequest,
) -> dict[str, np.ndarray]:
    if (
        type(parameters) is not tuple
        or not parameters
        or any(type(parameter) is not NeuralParameter for parameter in parameters)
    ):
        raise NeuralCheckpointError(
            reason="parameters must be a nonempty tuple[NeuralParameter, ...]"
        )
    expected = {
        parameter.name: parameter.values.shape for parameter in request.parameters.parameters
    }
    observed = {parameter.name: parameter.values.shape for parameter in parameters}
    if observed != expected or len(observed) != len(parameters):
        raise NeuralCheckpointError(
            reason="trained parameter names and shapes must match the initialization"
        )
    return {parameter.name: parameter.values for parameter in parameters}


def _loaded_parameters(tensors: dict[str, np.ndarray]) -> tuple[NeuralParameter, ...]:
    if not tensors:
        raise NeuralCheckpointError(reason="checkpoint must contain trained tensors")
    parameters: list[NeuralParameter] = []
    try:
        for name in sorted(tensors):
            values = np.ascontiguousarray(tensors[name], dtype=np.float32).copy()
            if tensors[name].dtype != np.dtype(np.float32):
                raise NeuralCheckpointError(reason=f"tensor {name!r} must be float32")
            values.flags.writeable = False
            parameters.append(NeuralParameter(name=name, values=values))
    except NeuralSpecValueError as error:
        raise NeuralCheckpointError(reason=f"invalid trained tensor: {error}") from error
    return tuple(parameters)


def _parameter_sha256(parameters: tuple[NeuralParameter, ...]) -> str:
    hasher = hashlib.sha256()
    for parameter in sorted(parameters, key=lambda candidate: candidate.name):
        name = parameter.name.encode()
        hasher.update(len(name).to_bytes(8, byteorder="little", signed=False))
        hasher.update(name)
        hasher.update(parameter.values.ndim.to_bytes(8, byteorder="little", signed=False))
        for dimension in parameter.values.shape:
            hasher.update(dimension.to_bytes(8, byteorder="little", signed=False))
        hasher.update(parameter.values.tobytes(order="C"))
    return hasher.hexdigest()


def _decode_training_summary(value: str) -> NeuralTrainingSummary:
    payload = _decode_canonical_json(value, field="training_summary")
    if type(payload) is not dict or set(payload) != {"control", "epochs"}:
        raise NeuralCheckpointError(reason="training_summary must contain control and epochs")
    summary_values = cast("dict[str, object]", payload)
    control_payload = summary_values["control"]
    epochs_payload = summary_values["epochs"]
    if type(control_payload) is not dict or type(epochs_payload) is not list:
        raise NeuralCheckpointError(reason="training_summary fields have invalid container types")
    _require_exact_keys(control_payload, NeuralControlState, field="training_summary.control")
    try:
        control = NeuralControlState(**cast("dict[str, Any]", control_payload))
        epochs: list[NeuralEpochMetrics] = []
        for index, epoch_payload in enumerate(cast("list[object]", epochs_payload)):
            if type(epoch_payload) is not dict:
                raise NeuralCheckpointError(reason=f"epoch {index} must be an object")
            _require_exact_keys(
                epoch_payload,
                NeuralEpochMetrics,
                field=f"training_summary.epochs[{index}]",
            )
            epochs.append(NeuralEpochMetrics(**cast("dict[str, Any]", epoch_payload)))
        return NeuralTrainingSummary(control=control, epochs=tuple(epochs))
    except (TypeError, NeuralControlError, NeuralTrainingError) as error:
        raise NeuralCheckpointError(
            reason=f"training_summary fields are invalid: {error}"
        ) from error


def _decode_canonical_json(value: object, *, field: str) -> object:
    if type(value) is not str:
        raise NeuralCheckpointError(reason=f"{field} metadata must be a plain str")
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise NeuralCheckpointError(reason=f"{field} is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise NeuralCheckpointError(reason=f"{field} JSON is not canonical")
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
        raise NeuralCheckpointError(reason=f"value cannot be canonical JSON: {error}") from error


def _require_exact_keys(value: object, model: type[object], *, field: str) -> None:
    if type(value) is not dict:
        raise NeuralCheckpointError(reason=f"{field} must be an object")
    expected = {item.name for item in fields(model)}
    if set(cast("dict[object, object]", value)) != expected:
        raise NeuralCheckpointError(reason=f"{field} keys are missing or unexpected")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as error:
        raise NeuralCheckpointError(reason=f"checkpoint hashing failed: {error}") from error
    return hasher.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NeuralCheckpointError(reason=f"{field} must be a lowercase SHA-256 hex digest")


def _require_enum_value(value: object, allowed: frozenset[str], *, field: str) -> None:
    if type(value) is not str or value not in allowed:
        raise NeuralCheckpointError(reason=f"{field} is not a supported serialized value")


__all__ = [
    "LoadedNeuralCheckpoint",
    "NeuralCheckpointArtifact",
    "NeuralCheckpointError",
    "NeuralCheckpointManifest",
    "load_neural_checkpoint",
    "write_neural_checkpoint",
]
