"""Immutable execution and manifest orchestration for one neural rolling window."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from finoptions import build_info

from discrete_dividend_study.artifacts.study_features import VerifiedStudyFeatureArtifact
from discrete_dividend_study.core.serialization import dataclass_field_names
from discrete_dividend_study.experiments.metrics import ProjectMetricError, ProjectPricingMetrics
from discrete_dividend_study.experiments.neural.window import PreparedNeuralWindow
from discrete_dividend_study.models.neural.checkpoints import (
    NeuralCheckpointArtifact,
    load_neural_checkpoint,
    write_neural_checkpoint,
)
from discrete_dividend_study.models.neural.execution import (
    CompilationMode,
    NeuralBackend,
    NeuralDevice,
    NeuralExecutionProfile,
    NeuralExecutionSpec,
    NeuralRuntimeIdentity,
    require_neural_runtime,
)
from discrete_dividend_study.models.neural.parameters import initialize_neural_parameters
from discrete_dividend_study.models.neural.predictions import (
    NeuralPredictionArtifact,
    NeuralPredictionRequest,
    load_neural_prediction,
    write_neural_prediction,
)
from discrete_dividend_study.models.neural.receipts import (
    NeuralRunReceipt,
    NeuralRunReceiptArtifact,
    load_neural_run_receipt,
    write_neural_run_receipt,
)
from discrete_dividend_study.models.neural.specification import (
    ModelPrecision,
    NeuralStudySpec,
    NeuralVariant,
    neural_study_sha256,
)
from discrete_dividend_study.models.neural.training import (
    NeuralTrainingRequest,
    NeuralTrainingSummary,
)

if TYPE_CHECKING:
    from discrete_dividend_study.models.neural.inputs import Float32Vector
    from discrete_dividend_study.models.neural.parameters import NeuralParameter

_REQUEST_FORMAT: Final = "discrete-dividend-neural-window-request-v1"
_REQUEST_FORMAT_VERSION: Final = 1
_RUN_FORMAT_VERSION: Final = 2
_PAPER_PACKAGE: Final = "discrete-dividend-option-pricing"
_SHA256_HEX_LENGTH: Final = 64
_GIT_HEX_LENGTH: Final = 40
_MAX_SEED: Final = (2**64) - 1
_PHASE_COUNT: Final = 3
_VARIANT_VALUES: Final = frozenset(variant.value for variant in NeuralVariant)


@dataclass(slots=True)
class NeuralRunError(RuntimeError):
    """A run request, referenced artifact, or manifest violates its identity contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected run contract."""
        return f"invalid neural rolling run: {self.reason}"


@dataclass(frozen=True, slots=True)
class StudySourceIdentity:
    """Caller-resolved source, schema, code, engine, and dependency-lock identity."""

    feature_artifact_sha256: str
    feature_schema_sha256: str
    paper_revision: str
    engine_revision: str
    dependency_lock_sha256: str

    def __post_init__(self) -> None:
        """Require content hashes and full Git revisions without path discovery."""
        for field, value in (
            ("feature_artifact_sha256", self.feature_artifact_sha256),
            ("feature_schema_sha256", self.feature_schema_sha256),
            ("dependency_lock_sha256", self.dependency_lock_sha256),
        ):
            _require_sha256(value, field=field)
        for field, value in (
            ("paper_revision", self.paper_revision),
            ("engine_revision", self.engine_revision),
        ):
            _require_git_revision(value, field=field)


@dataclass(frozen=True, slots=True)
class NeuralRunWindow:
    """Serialized half-open rolling-window identity."""

    window_id: int
    train_start: str
    validation_start: str
    test_start: str
    test_end: str

    def __post_init__(self) -> None:
        """Require canonical ISO dates and strictly adjacent phase ordering."""
        if type(self.window_id) is not int or self.window_id < 0:
            raise NeuralRunError(reason="window_id must be a nonnegative plain int")
        parsed = tuple(
            _require_iso_date(value, field=field)
            for field, value in (
                ("train_start", self.train_start),
                ("validation_start", self.validation_start),
                ("test_start", self.test_start),
                ("test_end", self.test_end),
            )
        )
        if not parsed[0] < parsed[1] < parsed[2] < parsed[3]:
            raise NeuralRunError(reason="serialized window bounds must be strictly increasing")


@dataclass(frozen=True, slots=True)
class NeuralRunPhaseRows:
    """Exact nonempty materialized row counts for one rolling run."""

    training: int
    validation: int
    test: int

    def __post_init__(self) -> None:
        """Reject skipped, empty, or inferred phase sizes."""
        for field, value in (
            ("training", self.training),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if type(value) is not int or value <= 0:
                raise NeuralRunError(reason=f"{field} rows must be a positive plain int")


@dataclass(frozen=True, slots=True)
class NeuralWindowRequestIdentity:
    """Every value known before training that can affect one exact window result."""

    format_version: int
    source: StudySourceIdentity
    study_sha256: str
    variant: str
    submitted_result_name: str
    execution_profile: str
    inference_batch_size: int
    window: NeuralRunWindow
    phase_rows: NeuralRunPhaseRows
    training_data_sha256: str
    validation_data_sha256: str
    test_data_sha256: str
    parameter_seed: int
    shuffle_seed: int
    initial_parameters_sha256: str
    runtime: NeuralRuntimeIdentity
    paper_package_version: str
    engine_package_version: str
    rustc_version: str

    def __post_init__(self) -> None:
        """Reject any incomplete or internally inconsistent pre-training identity."""
        _validate_request_identity(self)


@dataclass(frozen=True, slots=True)
class _SoftwareIdentity:
    paper_package_version: str
    engine_package_version: str
    rustc_version: str


@dataclass(frozen=True, slots=True)
class NeuralWindowRunRequest:
    """Complete prepared data, study, execution profile, source, and seed identity."""

    source: StudySourceIdentity
    prepared: PreparedNeuralWindow
    study: NeuralStudySpec
    execution_profile: NeuralExecutionProfile
    parameter_seed: int
    shuffle_seed: int

    def __post_init__(self) -> None:
        """Reject cross-study models and unresolved run identity."""
        if type(self.source) is not StudySourceIdentity:
            raise NeuralRunError(reason="source must be StudySourceIdentity")
        if type(self.prepared) is not PreparedNeuralWindow:
            raise NeuralRunError(reason="prepared must be PreparedNeuralWindow")
        if type(self.study) is not NeuralStudySpec:
            raise NeuralRunError(reason="study must be NeuralStudySpec")
        if self.prepared.model not in self.study.models:
            raise NeuralRunError(reason="prepared model is not owned by the requested study")
        if type(self.execution_profile) is not NeuralExecutionProfile:
            raise NeuralRunError(reason="execution_profile must be NeuralExecutionProfile")
        for field, value in (
            ("parameter_seed", self.parameter_seed),
            ("shuffle_seed", self.shuffle_seed),
        ):
            if type(value) is not int or not 0 <= value <= _MAX_SEED:
                raise NeuralRunError(reason=f"{field} must be an unsigned 64-bit plain int")

    @property
    def execution(self) -> NeuralExecutionSpec:
        """Resolve the exact model execution request from the closed profile."""
        return self.execution_profile.for_variant(self.prepared.model.variant)


@dataclass(frozen=True, slots=True)
class NeuralWindowRunManifest:
    """Complete deterministic run identity without local artifact paths."""

    format_version: int
    request_sha256: str
    source: StudySourceIdentity
    study_sha256: str
    variant: str
    submitted_result_name: str
    execution_profile: str
    inference_batch_size: int
    window: NeuralRunWindow
    phase_rows: NeuralRunPhaseRows
    training_data_sha256: str
    validation_data_sha256: str
    test_data_sha256: str
    parameter_seed: int
    shuffle_seed: int
    initial_parameters_sha256: str
    runtime: NeuralRuntimeIdentity
    paper_package_version: str
    engine_package_version: str
    rustc_version: str
    completed_epochs: int
    best_epoch: int
    best_validation_loss: float
    training_summary_sha256: str
    trained_parameters_sha256: str
    checkpoint_sha256: str
    prediction_artifact_sha256: str
    metrics: ProjectPricingMetrics

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent serialized run state."""
        if type(self.format_version) is not int or self.format_version != _RUN_FORMAT_VERSION:
            raise NeuralRunError(reason="format_version must be the plain int 2")
        if type(self.source) is not StudySourceIdentity:
            raise NeuralRunError(reason="source must be StudySourceIdentity")
        if type(self.window) is not NeuralRunWindow:
            raise NeuralRunError(reason="window must be NeuralRunWindow")
        if type(self.phase_rows) is not NeuralRunPhaseRows:
            raise NeuralRunError(reason="phase_rows must be NeuralRunPhaseRows")
        if type(self.runtime) is not NeuralRuntimeIdentity:
            raise NeuralRunError(reason="runtime must be NeuralRuntimeIdentity")
        if type(self.metrics) is not ProjectPricingMetrics:
            raise NeuralRunError(reason="metrics must be ProjectPricingMetrics")
        if self.metrics.observations != self.phase_rows.test:
            raise NeuralRunError(reason="metrics observations must equal test phase rows")
        _validate_manifest_strings(self)
        _validate_manifest_hashes(self)
        _validate_manifest_numbers(self)
        _validate_runtime(self.runtime, self.inference_batch_size)
        expected_request = neural_window_request_sha256(_request_identity_from_manifest(self))
        if self.request_sha256 != expected_request:
            raise NeuralRunError(reason="request_sha256 does not match the serialized request")


@dataclass(frozen=True, slots=True)
class NeuralWindowRunArtifact:
    """One verified content-addressed canonical JSON run manifest."""

    path: Path
    sha256: str
    manifest: NeuralWindowRunManifest

    def __post_init__(self) -> None:
        """Reject incomplete artifact handles."""
        if not isinstance(self.path, Path):
            raise NeuralRunError(reason="path must be pathlib.Path")
        _require_sha256(self.sha256, field="sha256")
        if type(self.manifest) is not NeuralWindowRunManifest:
            raise NeuralRunError(reason="manifest must be NeuralWindowRunManifest")


@dataclass(frozen=True, slots=True)
class NeuralWindowRunResult:
    """Relocatable handles for every verified artifact emitted by one completed run."""

    run: NeuralWindowRunArtifact
    checkpoint: NeuralCheckpointArtifact
    prediction: NeuralPredictionArtifact
    receipt: NeuralRunReceiptArtifact

    def __post_init__(self) -> None:
        """Require complete and cross-referenced artifact identities."""
        if type(self.run) is not NeuralWindowRunArtifact:
            raise NeuralRunError(reason="run must be NeuralWindowRunArtifact")
        if type(self.checkpoint) is not NeuralCheckpointArtifact:
            raise NeuralRunError(reason="checkpoint must be NeuralCheckpointArtifact")
        if type(self.prediction) is not NeuralPredictionArtifact:
            raise NeuralRunError(reason="prediction must be NeuralPredictionArtifact")
        if type(self.receipt) is not NeuralRunReceiptArtifact:
            raise NeuralRunError(reason="receipt must be NeuralRunReceiptArtifact")
        _validate_result_artifacts(self)


def resolve_study_source_identity(
    feature_artifact: VerifiedStudyFeatureArtifact,
    dependency_lock: Path,
    paper_revision: str,
    engine_revision: str,
) -> StudySourceIdentity:
    """Bind a verified feature receipt, dependency lock, and explicit full revisions."""
    if type(feature_artifact) is not VerifiedStudyFeatureArtifact:
        raise NeuralRunError(reason="feature_artifact must be VerifiedStudyFeatureArtifact")
    return StudySourceIdentity(
        feature_artifact_sha256=feature_artifact.output_sha256,
        feature_schema_sha256=feature_artifact.feature_schema_sha256,
        paper_revision=paper_revision,
        engine_revision=engine_revision,
        dependency_lock_sha256=_source_file_sha256(
            dependency_lock,
            field="dependency_lock",
        ),
    )


def neural_window_request_sha256(identity: NeuralWindowRequestIdentity) -> str:
    """Hash one complete pre-training identity as canonical ASCII JSON."""
    if type(identity) is not NeuralWindowRequestIdentity:
        raise NeuralRunError(reason="identity must be NeuralWindowRequestIdentity")
    payload = {
        "format": _REQUEST_FORMAT,
        "identity": asdict(identity),
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _build_request_identity(
    request: NeuralWindowRunRequest,
    *,
    runtime: NeuralRuntimeIdentity,
    initial_parameters_sha256: str,
    software: _SoftwareIdentity,
) -> NeuralWindowRequestIdentity:
    window = request.prepared.window
    return NeuralWindowRequestIdentity(
        format_version=_REQUEST_FORMAT_VERSION,
        source=request.source,
        study_sha256=neural_study_sha256(request.study),
        variant=request.prepared.model.variant.value,
        submitted_result_name=request.prepared.model.submitted_result_name,
        execution_profile=request.execution_profile.name,
        inference_batch_size=request.execution.inference_batch_size,
        window=NeuralRunWindow(
            window_id=window.window_id,
            train_start=window.train_start.isoformat(),
            validation_start=window.validation_start.isoformat(),
            test_start=window.test_start.isoformat(),
            test_end=window.test_end.isoformat(),
        ),
        phase_rows=NeuralRunPhaseRows(
            training=request.prepared.training.rows,
            validation=request.prepared.validation.rows,
            test=request.prepared.test.rows,
        ),
        training_data_sha256=request.prepared.training_data_sha256,
        validation_data_sha256=request.prepared.validation_data_sha256,
        test_data_sha256=request.prepared.test_data_sha256,
        parameter_seed=request.parameter_seed,
        shuffle_seed=request.shuffle_seed,
        initial_parameters_sha256=initial_parameters_sha256,
        runtime=runtime,
        paper_package_version=software.paper_package_version,
        engine_package_version=software.engine_package_version,
        rustc_version=software.rustc_version,
    )


def run_neural_window(
    request: NeuralWindowRunRequest,
    artifact_root: Path,
) -> NeuralWindowRunResult:
    """Train, predict, evaluate, and atomically manifest one exact rolling window."""
    if type(request) is not NeuralWindowRunRequest:
        raise NeuralRunError(reason="request must be NeuralWindowRunRequest")
    if (
        not isinstance(artifact_root, Path)
        or not artifact_root.is_dir()
        or artifact_root.is_symlink()
    ):
        raise NeuralRunError(reason="artifact_root must be an existing non-symlink directory")
    execution = request.execution
    resolved_runtime = require_neural_runtime(execution)
    directories = _artifact_directories(artifact_root)
    parameters = initialize_neural_parameters(
        request.study,
        request.prepared.model,
        request.parameter_seed,
    )
    engine = build_info()
    software = _SoftwareIdentity(
        paper_package_version=_package_version(),
        engine_package_version=engine.package_version,
        rustc_version=engine.rustc_version,
    )
    identity = _build_request_identity(
        request,
        runtime=resolved_runtime,
        initial_parameters_sha256=parameters.sha256,
        software=software,
    )
    request_sha256 = neural_window_request_sha256(identity)
    receipt_path = directories.receipts / f"{request_sha256}.receipt.json"
    if receipt_path.is_symlink() or receipt_path.exists():
        return _load_neural_window_result(directories, receipt_path, request_sha256)
    training_request = NeuralTrainingRequest(
        study=request.study,
        parameters=parameters,
        training_batch=request.prepared.training,
        validation_batch=request.prepared.validation,
        training_data_sha256=request.prepared.training_data_sha256,
        validation_data_sha256=request.prepared.validation_data_sha256,
        execution=execution,
        shuffle_seed=request.shuffle_seed,
    )
    summary, trained_parameters, prediction, runtime = _train_and_predict(
        training_request,
        request.prepared,
    )
    if runtime != resolved_runtime:
        raise NeuralRunError(reason="training resolved a different runtime identity")
    checkpoint = write_neural_checkpoint(
        directories.checkpoints,
        request=training_request,
        runtime=runtime,
        summary=summary,
        parameters=trained_parameters,
    )
    prediction_artifact = write_neural_prediction(
        directories.predictions,
        NeuralPredictionRequest(
            study=request.study,
            window=request.prepared.window,
            test_batch=request.prepared.test,
            test_data_sha256=request.prepared.test_data_sha256,
            checkpoint=checkpoint,
            prediction=prediction,
        ),
    )
    manifest = _build_run_manifest(
        identity,
        request_sha256=request_sha256,
        summary=summary,
        checkpoint=checkpoint,
        prediction=prediction_artifact,
    )
    run = write_neural_window_run(directories.runs, manifest)
    receipt = write_neural_run_receipt(
        directories.receipts,
        NeuralRunReceipt(
            format_version=1,
            request_sha256=request_sha256,
            run_sha256=run.sha256,
            checkpoint_sha256=checkpoint.sha256,
            prediction_sha256=prediction_artifact.sha256,
        ),
    )
    return NeuralWindowRunResult(
        run=run,
        checkpoint=checkpoint,
        prediction=prediction_artifact,
        receipt=receipt,
    )


def write_neural_window_run(
    directory: Path,
    manifest: NeuralWindowRunManifest,
) -> NeuralWindowRunArtifact:
    """Atomically persist and reload one canonical content-addressed run manifest."""
    if not isinstance(directory, Path) or not directory.is_dir() or directory.is_symlink():
        raise NeuralRunError(reason="directory must be an existing non-symlink directory")
    if type(manifest) is not NeuralWindowRunManifest:
        raise NeuralRunError(reason="manifest must be NeuralWindowRunManifest")
    payload = (_canonical_json(asdict(manifest)) + "\n").encode("ascii")
    sha256 = hashlib.sha256(payload).hexdigest()
    destination = _persist_bytes(
        directory,
        prefix=".neural-window-run-",
        suffix=".run.json",
        destination_name=f"{sha256}.run.json",
        payload=payload,
    )
    loaded = load_neural_window_run(destination)
    if loaded.manifest != manifest:
        raise NeuralRunError(reason="persisted run manifest differs from requested state")
    return loaded


def load_neural_window_run(path: Path) -> NeuralWindowRunArtifact:
    """Load only a canonical JSON manifest whose bytes match its content-addressed name."""
    if not isinstance(path, Path):
        raise NeuralRunError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or not path.name.endswith(".run.json"):
        raise NeuralRunError(reason="path must be a regular .run.json file")
    filename_sha256 = path.name.removesuffix(".run.json")
    _require_sha256(filename_sha256, field="run filename")
    payload = _read_bytes(path)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != filename_sha256:
        raise NeuralRunError(reason="run manifest bytes do not match the filename hash")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise NeuralRunError(reason="run manifest must be ASCII canonical JSON") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise NeuralRunError(reason="run manifest must end with one newline")
    decoded = _decode_canonical_json(text[:-1], field="run manifest")
    manifest = _decode_run_manifest(decoded)
    return NeuralWindowRunArtifact(path=path, sha256=observed_sha256, manifest=manifest)


@dataclass(frozen=True, slots=True)
class _ArtifactDirectories:
    checkpoints: Path
    predictions: Path
    runs: Path
    receipts: Path


def _artifact_directories(root: Path) -> _ArtifactDirectories:
    resolved: list[Path] = []
    for name in ("checkpoints", "predictions", "runs", "receipts"):
        candidate = root / name
        try:
            candidate.mkdir(exist_ok=True)
        except OSError as error:
            raise NeuralRunError(
                reason=f"cannot create artifact directory {name!r}: {error}"
            ) from error
        if not candidate.is_dir() or candidate.is_symlink():
            raise NeuralRunError(reason=f"artifact directory {name!r} must not be a symlink")
        resolved.append(candidate)
    return _ArtifactDirectories(*resolved)


def _load_neural_window_result(
    directories: _ArtifactDirectories,
    receipt_path: Path,
    expected_request_sha256: str,
) -> NeuralWindowRunResult:
    receipt = load_neural_run_receipt(receipt_path)
    if receipt.receipt.request_sha256 != expected_request_sha256:
        raise NeuralRunError(reason="receipt does not identify the exact current request")
    run = load_neural_window_run(
        directories.runs / f"{receipt.receipt.run_sha256}.run.json",
    )
    checkpoint = load_neural_checkpoint(
        directories.checkpoints / f"{receipt.receipt.checkpoint_sha256}.safetensors",
    ).artifact
    prediction = load_neural_prediction(
        directories.predictions / f"{receipt.receipt.prediction_sha256}.safetensors",
    ).artifact
    return NeuralWindowRunResult(
        run=run,
        checkpoint=checkpoint,
        prediction=prediction,
        receipt=receipt,
    )


def _train_and_predict(
    request: NeuralTrainingRequest,
    prepared: PreparedNeuralWindow,
) -> tuple[
    NeuralTrainingSummary,
    tuple[NeuralParameter, ...],
    Float32Vector,
    NeuralRuntimeIdentity,
]:
    batch_size = request.execution.inference_batch_size
    if request.execution.backend is NeuralBackend.PYTORCH:
        from discrete_dividend_study.models.neural.backends.torch_training import (  # noqa: PLC0415 - PyTorch is an optional runtime.
            predict_torch_project_model,
            train_torch_project_model,
        )

        result = train_torch_project_model(request)
        prediction = predict_torch_project_model(
            result.network,
            prepared.test,
            batch_size=batch_size,
        )
        return result.summary, result.parameters, prediction, result.network.runtime
    if request.execution.backend is NeuralBackend.MLX:
        from discrete_dividend_study.models.neural.backends.mlx_training import (  # noqa: PLC0415 - MLX is an optional runtime.
            predict_mlx_project_model,
            train_mlx_project_model,
        )

        result = train_mlx_project_model(request)
        prediction = predict_mlx_project_model(
            result.network,
            prepared.test,
            batch_size=batch_size,
        )
        return result.summary, result.parameters, prediction, result.network.runtime
    raise NeuralRunError(reason="execution backend is unsupported")


def _build_run_manifest(
    identity: NeuralWindowRequestIdentity,
    *,
    request_sha256: str,
    summary: NeuralTrainingSummary,
    checkpoint: NeuralCheckpointArtifact,
    prediction: NeuralPredictionArtifact,
) -> NeuralWindowRunManifest:
    control = summary.control
    checkpoint_manifest = checkpoint.manifest
    if control.best_epoch is None or control.best_validation_loss is None:
        raise NeuralRunError(reason="training summary has no best completed epoch")
    return NeuralWindowRunManifest(
        format_version=_RUN_FORMAT_VERSION,
        request_sha256=request_sha256,
        source=identity.source,
        study_sha256=identity.study_sha256,
        variant=identity.variant,
        submitted_result_name=identity.submitted_result_name,
        execution_profile=identity.execution_profile,
        inference_batch_size=identity.inference_batch_size,
        window=identity.window,
        phase_rows=identity.phase_rows,
        training_data_sha256=identity.training_data_sha256,
        validation_data_sha256=identity.validation_data_sha256,
        test_data_sha256=identity.test_data_sha256,
        parameter_seed=identity.parameter_seed,
        shuffle_seed=identity.shuffle_seed,
        initial_parameters_sha256=checkpoint_manifest.initial_parameters_sha256,
        runtime=identity.runtime,
        paper_package_version=identity.paper_package_version,
        engine_package_version=identity.engine_package_version,
        rustc_version=identity.rustc_version,
        completed_epochs=control.completed_epochs,
        best_epoch=control.best_epoch,
        best_validation_loss=control.best_validation_loss,
        training_summary_sha256=checkpoint_manifest.training_summary_sha256,
        trained_parameters_sha256=checkpoint_manifest.trained_parameters_sha256,
        checkpoint_sha256=checkpoint.sha256,
        prediction_artifact_sha256=prediction.sha256,
        metrics=prediction.metrics,
    )


def _validate_request_identity(identity: NeuralWindowRequestIdentity) -> None:
    _validate_request_identity_types(identity)
    _validate_request_identity_strings(identity)
    _validate_request_identity_hashes(identity)
    _validate_request_identity_numbers(identity)
    _validate_runtime(identity.runtime, identity.inference_batch_size)


def _validate_request_identity_types(identity: NeuralWindowRequestIdentity) -> None:
    if (
        type(identity.format_version) is not int
        or identity.format_version != _REQUEST_FORMAT_VERSION
    ):
        raise NeuralRunError(reason="request format_version must be the plain int 1")
    if type(identity.source) is not StudySourceIdentity:
        raise NeuralRunError(reason="request source must be StudySourceIdentity")
    if type(identity.window) is not NeuralRunWindow:
        raise NeuralRunError(reason="request window must be NeuralRunWindow")
    if type(identity.phase_rows) is not NeuralRunPhaseRows:
        raise NeuralRunError(reason="request phase_rows must be NeuralRunPhaseRows")
    if type(identity.runtime) is not NeuralRuntimeIdentity:
        raise NeuralRunError(reason="request runtime must be NeuralRuntimeIdentity")


def _validate_request_identity_strings(identity: NeuralWindowRequestIdentity) -> None:
    if type(identity.variant) is not str or identity.variant not in _VARIANT_VALUES:
        raise NeuralRunError(reason="request variant is not a supported serialized value")
    for field, value in (
        ("submitted_result_name", identity.submitted_result_name),
        ("execution_profile", identity.execution_profile),
        ("paper_package_version", identity.paper_package_version),
        ("engine_package_version", identity.engine_package_version),
        ("rustc_version", identity.rustc_version),
    ):
        if type(value) is not str or not value:
            raise NeuralRunError(reason=f"request {field} must be a nonempty plain str")


def _validate_request_identity_hashes(identity: NeuralWindowRequestIdentity) -> None:
    for field, value in (
        ("study_sha256", identity.study_sha256),
        ("training_data_sha256", identity.training_data_sha256),
        ("validation_data_sha256", identity.validation_data_sha256),
        ("test_data_sha256", identity.test_data_sha256),
        ("initial_parameters_sha256", identity.initial_parameters_sha256),
    ):
        _require_sha256(value, field=f"request {field}")
    if (
        len(
            {
                identity.training_data_sha256,
                identity.validation_data_sha256,
                identity.test_data_sha256,
            },
        )
        != _PHASE_COUNT
    ):
        raise NeuralRunError(reason="request phase hashes must differ")


def _validate_request_identity_numbers(identity: NeuralWindowRequestIdentity) -> None:
    for field, value in (
        ("parameter_seed", identity.parameter_seed),
        ("shuffle_seed", identity.shuffle_seed),
    ):
        if type(value) is not int or not 0 <= value <= _MAX_SEED:
            raise NeuralRunError(reason=f"request {field} must be an unsigned 64-bit plain int")
    if type(identity.inference_batch_size) is not int or identity.inference_batch_size <= 0:
        raise NeuralRunError(reason="request inference_batch_size must be a positive plain int")


def _request_identity_from_manifest(
    manifest: NeuralWindowRunManifest,
) -> NeuralWindowRequestIdentity:
    return NeuralWindowRequestIdentity(
        format_version=_REQUEST_FORMAT_VERSION,
        source=manifest.source,
        study_sha256=manifest.study_sha256,
        variant=manifest.variant,
        submitted_result_name=manifest.submitted_result_name,
        execution_profile=manifest.execution_profile,
        inference_batch_size=manifest.inference_batch_size,
        window=manifest.window,
        phase_rows=manifest.phase_rows,
        training_data_sha256=manifest.training_data_sha256,
        validation_data_sha256=manifest.validation_data_sha256,
        test_data_sha256=manifest.test_data_sha256,
        parameter_seed=manifest.parameter_seed,
        shuffle_seed=manifest.shuffle_seed,
        initial_parameters_sha256=manifest.initial_parameters_sha256,
        runtime=manifest.runtime,
        paper_package_version=manifest.paper_package_version,
        engine_package_version=manifest.engine_package_version,
        rustc_version=manifest.rustc_version,
    )


def _validate_result_artifacts(result: NeuralWindowRunResult) -> None:
    manifest = result.run.manifest
    receipt = result.receipt.receipt
    checkpoint = result.checkpoint.manifest
    prediction = result.prediction.manifest
    if (
        receipt.request_sha256 != manifest.request_sha256
        or receipt.run_sha256 != result.run.sha256
        or receipt.checkpoint_sha256 != result.checkpoint.sha256
        or receipt.prediction_sha256 != result.prediction.sha256
    ):
        raise NeuralRunError(reason="receipt does not cross-reference the exact run artifacts")
    if (
        manifest.checkpoint_sha256 != result.checkpoint.sha256
        or manifest.prediction_artifact_sha256 != result.prediction.sha256
        or manifest.metrics != result.prediction.metrics
    ):
        raise NeuralRunError(reason="run manifest does not cross-reference its result artifacts")
    if (
        checkpoint.study_sha256 != manifest.study_sha256
        or checkpoint.variant != manifest.variant
        or checkpoint.submitted_result_name != manifest.submitted_result_name
        or checkpoint.training_data_sha256 != manifest.training_data_sha256
        or checkpoint.validation_data_sha256 != manifest.validation_data_sha256
        or checkpoint.initial_parameters_sha256 != manifest.initial_parameters_sha256
        or checkpoint.parameter_seed != manifest.parameter_seed
        or checkpoint.shuffle_seed != manifest.shuffle_seed
        or checkpoint.backend != manifest.runtime.backend.value
        or checkpoint.backend_version != manifest.runtime.backend_version
        or checkpoint.device != manifest.runtime.device.value
        or checkpoint.precision != manifest.runtime.precision.value
        or checkpoint.compilation != manifest.runtime.compilation.value
        or checkpoint.cpu_threads != manifest.runtime.cpu_threads
        or checkpoint.operating_system != manifest.runtime.operating_system
        or checkpoint.machine != manifest.runtime.machine
        or checkpoint.completed_epochs != manifest.completed_epochs
        or checkpoint.best_epoch != manifest.best_epoch
        or checkpoint.best_validation_loss != manifest.best_validation_loss
        or checkpoint.training_summary_sha256 != manifest.training_summary_sha256
        or checkpoint.trained_parameters_sha256 != manifest.trained_parameters_sha256
    ):
        raise NeuralRunError(reason="checkpoint identity differs from the run manifest")
    if (
        prediction.study_sha256 != manifest.study_sha256
        or prediction.variant != manifest.variant
        or prediction.submitted_result_name != manifest.submitted_result_name
        or prediction.window_id != manifest.window.window_id
        or prediction.test_data_sha256 != manifest.test_data_sha256
        or prediction.checkpoint_sha256 != result.checkpoint.sha256
        or prediction.observations != manifest.phase_rows.test
    ):
        raise NeuralRunError(reason="prediction identity differs from the run manifest")


def _validate_manifest_strings(manifest: NeuralWindowRunManifest) -> None:
    if type(manifest.variant) is not str or manifest.variant not in _VARIANT_VALUES:
        raise NeuralRunError(reason="variant is not a supported serialized value")
    for field, value in (
        ("submitted_result_name", manifest.submitted_result_name),
        ("execution_profile", manifest.execution_profile),
        ("paper_package_version", manifest.paper_package_version),
        ("engine_package_version", manifest.engine_package_version),
        ("rustc_version", manifest.rustc_version),
    ):
        if type(value) is not str or not value:
            raise NeuralRunError(reason=f"{field} must be a nonempty plain str")


def _validate_manifest_hashes(manifest: NeuralWindowRunManifest) -> None:
    values = (
        ("request_sha256", manifest.request_sha256),
        ("study_sha256", manifest.study_sha256),
        ("training_data_sha256", manifest.training_data_sha256),
        ("validation_data_sha256", manifest.validation_data_sha256),
        ("test_data_sha256", manifest.test_data_sha256),
        ("initial_parameters_sha256", manifest.initial_parameters_sha256),
        ("training_summary_sha256", manifest.training_summary_sha256),
        ("trained_parameters_sha256", manifest.trained_parameters_sha256),
        ("checkpoint_sha256", manifest.checkpoint_sha256),
        ("prediction_artifact_sha256", manifest.prediction_artifact_sha256),
    )
    for field, value in values:
        _require_sha256(value, field=field)
    phase_hashes = {
        manifest.training_data_sha256,
        manifest.validation_data_sha256,
        manifest.test_data_sha256,
    }
    if len(phase_hashes) != _PHASE_COUNT:
        raise NeuralRunError(reason="training, validation, and test hashes must differ")
    if manifest.checkpoint_sha256 == manifest.prediction_artifact_sha256:
        raise NeuralRunError(reason="checkpoint and prediction artifact hashes must differ")


def _validate_manifest_numbers(manifest: NeuralWindowRunManifest) -> None:
    for field, value in (
        ("parameter_seed", manifest.parameter_seed),
        ("shuffle_seed", manifest.shuffle_seed),
    ):
        if type(value) is not int or not 0 <= value <= _MAX_SEED:
            raise NeuralRunError(reason=f"{field} must be an unsigned 64-bit plain int")
    if type(manifest.completed_epochs) is not int or manifest.completed_epochs <= 0:
        raise NeuralRunError(reason="completed_epochs must be a positive plain int")
    if type(manifest.inference_batch_size) is not int or manifest.inference_batch_size <= 0:
        raise NeuralRunError(reason="inference_batch_size must be a positive plain int")
    if (
        type(manifest.best_epoch) is not int
        or not 0 <= manifest.best_epoch < manifest.completed_epochs
    ):
        raise NeuralRunError(reason="best_epoch must identify a completed epoch")
    if (
        type(manifest.best_validation_loss) is not float
        or not math.isfinite(manifest.best_validation_loss)
        or manifest.best_validation_loss < 0.0
    ):
        raise NeuralRunError(reason="best_validation_loss must be finite and nonnegative")


def _validate_runtime(runtime: NeuralRuntimeIdentity, inference_batch_size: int) -> None:
    try:
        NeuralExecutionSpec(
            backend=runtime.backend,
            device=runtime.device,
            precision=runtime.precision,
            compilation=runtime.compilation,
            cpu_threads=runtime.cpu_threads,
            inference_batch_size=inference_batch_size,
        )
    except (TypeError, ValueError) as error:
        raise NeuralRunError(reason=f"runtime execution fields are invalid: {error}") from error
    for field, value in (
        ("backend_version", runtime.backend_version),
        ("operating_system", runtime.operating_system),
        ("machine", runtime.machine),
    ):
        if type(value) is not str or not value:
            raise NeuralRunError(reason=f"runtime {field} must be a nonempty plain str")


def _decode_run_manifest(value: object) -> NeuralWindowRunManifest:
    _require_exact_keys(value, NeuralWindowRunManifest, field="run manifest")
    payload = cast("dict[str, Any]", value)
    source = _decode_dataclass(payload["source"], StudySourceIdentity, field="source")
    window = _decode_dataclass(payload["window"], NeuralRunWindow, field="window")
    phase_rows = _decode_dataclass(payload["phase_rows"], NeuralRunPhaseRows, field="phase_rows")
    metrics = _decode_dataclass(payload["metrics"], ProjectPricingMetrics, field="metrics")
    runtime = _decode_runtime(payload["runtime"])
    try:
        return NeuralWindowRunManifest(
            format_version=cast("int", payload["format_version"]),
            request_sha256=cast("str", payload["request_sha256"]),
            source=source,
            study_sha256=cast("str", payload["study_sha256"]),
            variant=cast("str", payload["variant"]),
            submitted_result_name=cast("str", payload["submitted_result_name"]),
            execution_profile=cast("str", payload["execution_profile"]),
            inference_batch_size=cast("int", payload["inference_batch_size"]),
            window=window,
            phase_rows=phase_rows,
            training_data_sha256=cast("str", payload["training_data_sha256"]),
            validation_data_sha256=cast("str", payload["validation_data_sha256"]),
            test_data_sha256=cast("str", payload["test_data_sha256"]),
            parameter_seed=cast("int", payload["parameter_seed"]),
            shuffle_seed=cast("int", payload["shuffle_seed"]),
            initial_parameters_sha256=cast("str", payload["initial_parameters_sha256"]),
            runtime=runtime,
            paper_package_version=cast("str", payload["paper_package_version"]),
            engine_package_version=cast("str", payload["engine_package_version"]),
            rustc_version=cast("str", payload["rustc_version"]),
            completed_epochs=cast("int", payload["completed_epochs"]),
            best_epoch=cast("int", payload["best_epoch"]),
            best_validation_loss=cast("float", payload["best_validation_loss"]),
            training_summary_sha256=cast("str", payload["training_summary_sha256"]),
            trained_parameters_sha256=cast("str", payload["trained_parameters_sha256"]),
            checkpoint_sha256=cast("str", payload["checkpoint_sha256"]),
            prediction_artifact_sha256=cast(
                "str",
                payload["prediction_artifact_sha256"],
            ),
            metrics=metrics,
        )
    except (TypeError, NeuralRunError, ProjectMetricError) as error:
        raise NeuralRunError(reason=f"run manifest fields are invalid: {error}") from error


def _decode_dataclass[T](value: object, model: type[T], *, field: str) -> T:
    _require_exact_keys(value, model, field=field)
    try:
        return model(**cast("dict[str, Any]", value))
    except (TypeError, ValueError) as error:
        raise NeuralRunError(reason=f"{field} fields are invalid: {error}") from error


def _decode_runtime(value: object) -> NeuralRuntimeIdentity:
    _require_exact_keys(value, NeuralRuntimeIdentity, field="runtime")
    payload = cast("dict[str, Any]", value)
    try:
        return NeuralRuntimeIdentity(
            backend=NeuralBackend(payload["backend"]),
            backend_version=payload["backend_version"],
            device=NeuralDevice(payload["device"]),
            precision=ModelPrecision(payload["precision"]),
            compilation=CompilationMode(payload["compilation"]),
            cpu_threads=payload["cpu_threads"],
            operating_system=payload["operating_system"],
            machine=payload["machine"],
        )
    except (TypeError, ValueError) as error:
        raise NeuralRunError(reason=f"runtime fields are invalid: {error}") from error


def _persist_bytes(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
    destination_name: str,
    payload: bytes,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        destination = directory / destination_name
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _read_bytes(destination) != payload:
                raise NeuralRunError(
                    reason="content-addressed destination contains different bytes",
                ) from None
        temporary.unlink()
    except OSError as error:
        raise NeuralRunError(reason=f"run manifest write failed: {error}") from error
    finally:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise NeuralRunError(reason=f"run manifest read failed: {error}") from error


def _source_file_sha256(path: Path, *, field: str) -> str:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise NeuralRunError(reason=f"{field} must be an explicit regular non-symlink file")
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as error:
        raise NeuralRunError(reason=f"cannot hash {field}: {error}") from error
    return hasher.hexdigest()


def _decode_canonical_json(value: str, *, field: str) -> object:
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise NeuralRunError(reason=f"{field} is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise NeuralRunError(reason=f"{field} JSON is not canonical")
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
        raise NeuralRunError(reason=f"value cannot be canonical JSON: {error}") from error


def _require_exact_keys(value: object, model: type[object], *, field: str) -> None:
    if type(value) is not dict:
        raise NeuralRunError(reason=f"{field} must be an object")
    expected = dataclass_field_names(model)
    if set(cast("dict[object, object]", value)) != expected:
        raise NeuralRunError(reason=f"{field} keys are missing or unexpected")


def _require_iso_date(value: object, *, field: str) -> date:
    if type(value) is not str:
        raise NeuralRunError(reason=f"{field} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise NeuralRunError(reason=f"{field} must be a canonical ISO date") from error
    if parsed.isoformat() != value:
        raise NeuralRunError(reason=f"{field} must be a canonical ISO date")
    return parsed


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NeuralRunError(reason=f"{field} must be a lowercase SHA-256 hex digest")


def _require_git_revision(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _GIT_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NeuralRunError(reason=f"{field} must be a lowercase full Git commit")


def _package_version() -> str:
    try:
        observed = version(_PAPER_PACKAGE)
    except PackageNotFoundError as error:
        raise NeuralRunError(
            reason=f"installed package {_PAPER_PACKAGE!r} is unavailable"
        ) from error
    if not observed:
        raise NeuralRunError(reason="paper package version must be nonempty")
    return observed


__all__ = [
    "NeuralRunError",
    "NeuralRunPhaseRows",
    "NeuralRunWindow",
    "NeuralWindowRequestIdentity",
    "NeuralWindowRunArtifact",
    "NeuralWindowRunManifest",
    "NeuralWindowRunRequest",
    "NeuralWindowRunResult",
    "StudySourceIdentity",
    "load_neural_window_run",
    "neural_window_request_sha256",
    "resolve_study_source_identity",
    "run_neural_window",
    "write_neural_window_run",
]
