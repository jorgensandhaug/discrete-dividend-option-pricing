"""Complete immutable orchestration for the project neural rolling study."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import polars as pl

from .neural_execution import NeuralExecutionProfile, require_neural_runtime
from .neural_rolling import (
    PROJECT_NEURAL_ROLLING_PREPARATION_SPEC,
    PreparedNeuralRollingSource,
    prepare_project_neural_rolling_source,
)
from .neural_run import (
    NeuralRunWindow,
    NeuralWindowRunRequest,
    NeuralWindowRunResult,
    StudySourceIdentity,
    resolve_study_source_identity,
    run_neural_window,
)
from .neural_spec import NeuralStudySpec, NeuralVariant, neural_study_sha256
from .neural_window import NeuralWindowPreparationSpec, PreparedNeuralWindow
from .project_metrics import (
    ProjectMetricError,
    ProjectPricingMetrics,
    aggregate_project_pricing_metrics,
)
from .rolling_windows import RollingWindow

if TYPE_CHECKING:
    from finoptions import CpuExecutor

_FORMAT_VERSION: Final = 1
_SEED_DERIVATION: Final = "sha256-domain-separated-u64-big-endian-v1"
_SEED_FORMAT: Final = "discrete-dividend-neural-window-seed-v1"
_SHA256_HEX_LENGTH: Final = 64
_MAX_SEED: Final = (2**64) - 1
_VARIANT_VALUES: Final = frozenset(variant.value for variant in NeuralVariant)


@dataclass(frozen=True, slots=True)
class NeuralStudyRunError(RuntimeError):
    """A complete neural-study request, result, or artifact violates its contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected complete-study contract."""
        return f"invalid neural rolling study: {self.reason}"


class NeuralSeedRole(StrEnum):
    """Independent stochastic roles admitted by the closed seed derivation."""

    PARAMETER = "parameter"
    SHUFFLE = "shuffle"


@dataclass(frozen=True, slots=True)
class NeuralSeedPlan:
    """Two independent unsigned roots and an explicit deterministic derivation identity."""

    derivation: str
    parameter_root_seed: int
    shuffle_root_seed: int

    def __post_init__(self) -> None:
        """Reject implicit algorithms, coerced roots, and role aliasing."""
        if type(self.derivation) is not str or self.derivation != _SEED_DERIVATION:
            raise NeuralStudyRunError(reason=f"seed derivation must be {_SEED_DERIVATION!r}")
        for field, value in (
            ("parameter_root_seed", self.parameter_root_seed),
            ("shuffle_root_seed", self.shuffle_root_seed),
        ):
            if type(value) is not int or not 0 <= value <= _MAX_SEED:
                raise NeuralStudyRunError(reason=f"{field} must be an unsigned 64-bit plain int")
        if self.parameter_root_seed == self.shuffle_root_seed:
            raise NeuralStudyRunError(reason="parameter and shuffle root seeds must differ")


def project_neural_seed_plan(parameter_root_seed: int, shuffle_root_seed: int) -> NeuralSeedPlan:
    """Construct the only approved project seed-derivation plan."""
    return NeuralSeedPlan(
        derivation=_SEED_DERIVATION,
        parameter_root_seed=parameter_root_seed,
        shuffle_root_seed=shuffle_root_seed,
    )


def derive_neural_window_seed(
    plan: NeuralSeedPlan,
    role: NeuralSeedRole,
    variant: NeuralVariant,
    window_id: int,
) -> int:
    """Derive one role/model/window seed by canonical domain-separated SHA-256."""
    if type(plan) is not NeuralSeedPlan:
        raise NeuralStudyRunError(reason="plan must be NeuralSeedPlan")
    if type(role) is not NeuralSeedRole:
        raise NeuralStudyRunError(reason="role must be NeuralSeedRole")
    if type(variant) is not NeuralVariant:
        raise NeuralStudyRunError(reason="variant must be NeuralVariant")
    if type(window_id) is not int or window_id < 0:
        raise NeuralStudyRunError(reason="window_id must be a nonnegative plain int")
    root_seed = (
        plan.parameter_root_seed if role is NeuralSeedRole.PARAMETER else plan.shuffle_root_seed
    )
    payload = _canonical_json(
        {
            "format": _SEED_FORMAT,
            "role": role.value,
            "root_seed": root_seed,
            "variant": variant.value,
            "window_id": window_id,
        },
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


@dataclass(frozen=True, slots=True)
class NeuralRollingStudyRequest:
    """Explicit licensed source, code identity, schedule, study, runtime, and seed request."""

    feature_artifact: Path
    dependency_lock: Path
    paper_revision: str
    engine_revision: str
    windows: tuple[RollingWindow, ...]
    study: NeuralStudySpec
    execution_profile: NeuralExecutionProfile
    seed_plan: NeuralSeedPlan
    preparation: NeuralWindowPreparationSpec = PROJECT_NEURAL_ROLLING_PREPARATION_SPEC

    def __post_init__(self) -> None:
        """Reject untyped high-level inputs before reading licensed data."""
        for field, value in (
            ("feature_artifact", self.feature_artifact),
            ("dependency_lock", self.dependency_lock),
        ):
            if not isinstance(value, Path):
                raise NeuralStudyRunError(reason=f"{field} must be pathlib.Path")
        if type(self.paper_revision) is not str or type(self.engine_revision) is not str:
            raise NeuralStudyRunError(reason="Git revisions must be plain strings")
        if (
            type(self.windows) is not tuple
            or not self.windows
            or any(type(window) is not RollingWindow for window in self.windows)
        ):
            raise NeuralStudyRunError(reason="windows must be a nonempty tuple[RollingWindow, ...]")
        if type(self.study) is not NeuralStudySpec:
            raise NeuralStudyRunError(reason="study must be NeuralStudySpec")
        if type(self.execution_profile) is not NeuralExecutionProfile:
            raise NeuralStudyRunError(reason="execution_profile must be NeuralExecutionProfile")
        if type(self.seed_plan) is not NeuralSeedPlan:
            raise NeuralStudyRunError(reason="seed_plan must be NeuralSeedPlan")
        if type(self.preparation) is not NeuralWindowPreparationSpec:
            raise NeuralStudyRunError(reason="preparation must be NeuralWindowPreparationSpec")


@dataclass(frozen=True, slots=True)
class NeuralStudyRunReference:
    """One model/window result identity and exact test metric contribution."""

    variant: str
    submitted_result_name: str
    window_id: int
    parameter_seed: int
    shuffle_seed: int
    request_sha256: str
    run_sha256: str
    checkpoint_sha256: str
    prediction_sha256: str
    test_rows: int
    metrics: ProjectPricingMetrics

    def __post_init__(self) -> None:
        """Reject malformed or internally inconsistent result references."""
        _validate_study_run_reference(self)


@dataclass(frozen=True, slots=True)
class NeuralModelStudyResult:
    """Observation-pooled metrics for one model over the complete rolling schedule."""

    variant: str
    submitted_result_name: str
    completed_windows: int
    metrics: ProjectPricingMetrics

    def __post_init__(self) -> None:
        """Reject incomplete model-level publication results."""
        if type(self.variant) is not str or self.variant not in _VARIANT_VALUES:
            raise NeuralStudyRunError(reason="model result variant is unsupported")
        if type(self.submitted_result_name) is not str or not self.submitted_result_name:
            raise NeuralStudyRunError(reason="submitted_result_name must be a nonempty plain str")
        if type(self.completed_windows) is not int or self.completed_windows <= 0:
            raise NeuralStudyRunError(reason="completed_windows must be a positive plain int")
        if type(self.metrics) is not ProjectPricingMetrics:
            raise NeuralStudyRunError(reason="model result metrics must be ProjectPricingMetrics")


@dataclass(frozen=True, slots=True)
class NeuralRollingStudyManifest:
    """Canonical complete-study identity with every immutable run cross-reference."""

    format_version: int
    source: StudySourceIdentity
    study_sha256: str
    execution_profile: str
    seed_plan: NeuralSeedPlan
    windows: tuple[NeuralRunWindow, ...]
    runs: tuple[NeuralStudyRunReference, ...]
    model_results: tuple[NeuralModelStudyResult, ...]

    def __post_init__(self) -> None:
        """Require the exact Cartesian model/window result set and correctly pooled metrics."""
        _validate_study_manifest(self)


@dataclass(frozen=True, slots=True)
class NeuralRollingStudyArtifact:
    """One verified content-addressed complete-study JSON manifest."""

    path: Path
    sha256: str
    manifest: NeuralRollingStudyManifest

    def __post_init__(self) -> None:
        """Reject incomplete study artifact handles."""
        if not isinstance(self.path, Path):
            raise NeuralStudyRunError(reason="path must be pathlib.Path")
        _require_sha256(self.sha256, field="sha256")
        if type(self.manifest) is not NeuralRollingStudyManifest:
            raise NeuralStudyRunError(reason="manifest must be NeuralRollingStudyManifest")


@dataclass(frozen=True, slots=True)
class NeuralRollingStudyResult:
    """Verified complete-study manifest and its ordered immutable window results."""

    study: NeuralRollingStudyArtifact
    runs: tuple[NeuralWindowRunResult, ...]

    def __post_init__(self) -> None:
        """Require exact ordered agreement between the study and loaded run artifacts."""
        if type(self.study) is not NeuralRollingStudyArtifact:
            raise NeuralStudyRunError(reason="study must be NeuralRollingStudyArtifact")
        if type(self.runs) is not tuple or any(
            type(result) is not NeuralWindowRunResult for result in self.runs
        ):
            raise NeuralStudyRunError(reason="runs must be tuple[NeuralWindowRunResult, ...]")
        observed = tuple(_run_reference(result) for result in self.runs)
        if observed != self.study.manifest.runs:
            raise NeuralStudyRunError(reason="study references differ from loaded run artifacts")


@dataclass(frozen=True, slots=True)
class _PreparedStudyRun:
    prepared: PreparedNeuralWindow
    parameter_seed: int
    shuffle_seed: int


def run_project_neural_rolling_study(
    request: NeuralRollingStudyRequest,
    artifact_root: Path,
    executor: CpuExecutor,
) -> NeuralRollingStudyResult:
    """Preflight, execute/resume, pool, and atomically manifest the complete neural study."""
    _validate_study_execution_request(request, artifact_root)
    source = resolve_study_source_identity(
        request.feature_artifact,
        request.dependency_lock,
        request.paper_revision,
        request.engine_revision,
    )
    prepared_source = prepare_project_neural_rolling_source(
        pl.scan_parquet(request.feature_artifact),
        request.windows,
        request.study,
        executor,
        request.preparation,
    )
    resolved_after_scan = resolve_study_source_identity(
        request.feature_artifact,
        request.dependency_lock,
        request.paper_revision,
        request.engine_revision,
    )
    if resolved_after_scan != source:
        raise NeuralStudyRunError(reason="source or dependency lock changed during preparation")
    _preflight_runtimes(request)
    prepared_runs = _prepare_complete_schedule(request, prepared_source)
    results = tuple(
        run_neural_window(
            NeuralWindowRunRequest(
                source=source,
                prepared=item.prepared,
                study=request.study,
                execution_profile=request.execution_profile,
                parameter_seed=item.parameter_seed,
                shuffle_seed=item.shuffle_seed,
            ),
            artifact_root,
        )
        for item in prepared_runs
    )
    manifest = _study_manifest(request, source, results)
    study_directory = _study_directory(artifact_root)
    artifact = write_neural_rolling_study(study_directory, manifest)
    return NeuralRollingStudyResult(study=artifact, runs=results)


def write_neural_rolling_study(
    directory: Path,
    manifest: NeuralRollingStudyManifest,
) -> NeuralRollingStudyArtifact:
    """Atomically persist and reload one canonical content-addressed study manifest."""
    _validate_directory(directory)
    if type(manifest) is not NeuralRollingStudyManifest:
        raise NeuralStudyRunError(reason="manifest must be NeuralRollingStudyManifest")
    payload = (_canonical_json(asdict(manifest)) + "\n").encode("ascii")
    sha256 = hashlib.sha256(payload).hexdigest()
    destination = _persist_bytes(
        directory,
        destination_name=f"{sha256}.study.json",
        payload=payload,
    )
    loaded = load_neural_rolling_study(destination)
    if loaded.manifest != manifest:
        raise NeuralStudyRunError(reason="persisted study differs from requested state")
    return loaded


def load_neural_rolling_study(path: Path) -> NeuralRollingStudyArtifact:
    """Load one explicit canonical study manifest whose bytes match its content address."""
    if not isinstance(path, Path):
        raise NeuralStudyRunError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or not path.name.endswith(".study.json"):
        raise NeuralStudyRunError(reason="path must be a regular .study.json file")
    filename_sha256 = path.name.removesuffix(".study.json")
    _require_sha256(filename_sha256, field="study filename")
    payload = _read_bytes(path)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != filename_sha256:
        raise NeuralStudyRunError(reason="study bytes do not match the filename hash")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise NeuralStudyRunError(reason="study must be ASCII canonical JSON") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise NeuralStudyRunError(reason="study must end with one newline")
    decoded = _decode_canonical_json(text[:-1])
    manifest = _decode_study_manifest(decoded)
    return NeuralRollingStudyArtifact(
        path=path,
        sha256=observed_sha256,
        manifest=manifest,
    )


def _validate_study_execution_request(
    request: object,
    artifact_root: Path,
) -> None:
    if type(request) is not NeuralRollingStudyRequest:
        raise NeuralStudyRunError(reason="request must be NeuralRollingStudyRequest")
    _validate_directory(artifact_root)


def _preflight_runtimes(request: NeuralRollingStudyRequest) -> None:
    seen: set[object] = set()
    for model in request.study.models:
        execution = request.execution_profile.for_variant(model.variant)
        if execution not in seen:
            require_neural_runtime(execution)
            seen.add(execution)


def _prepare_complete_schedule(
    request: NeuralRollingStudyRequest,
    source: PreparedNeuralRollingSource,
) -> tuple[_PreparedStudyRun, ...]:
    return tuple(
        _PreparedStudyRun(
            prepared=source.window(window.window_id, model),
            parameter_seed=derive_neural_window_seed(
                request.seed_plan,
                NeuralSeedRole.PARAMETER,
                model.variant,
                window.window_id,
            ),
            shuffle_seed=derive_neural_window_seed(
                request.seed_plan,
                NeuralSeedRole.SHUFFLE,
                model.variant,
                window.window_id,
            ),
        )
        for model in request.study.models
        for window in request.windows
    )


def _study_manifest(
    request: NeuralRollingStudyRequest,
    source: StudySourceIdentity,
    results: tuple[NeuralWindowRunResult, ...],
) -> NeuralRollingStudyManifest:
    references = tuple(_run_reference(result) for result in results)
    model_results = tuple(
        NeuralModelStudyResult(
            variant=model.variant.value,
            submitted_result_name=model.submitted_result_name,
            completed_windows=len(request.windows),
            metrics=aggregate_project_pricing_metrics(
                tuple(
                    reference.metrics
                    for reference in references
                    if reference.variant == model.variant.value
                ),
            ),
        )
        for model in request.study.models
    )
    return NeuralRollingStudyManifest(
        format_version=_FORMAT_VERSION,
        source=source,
        study_sha256=neural_study_sha256(request.study),
        execution_profile=request.execution_profile.name,
        seed_plan=request.seed_plan,
        windows=tuple(_serialized_window(window) for window in request.windows),
        runs=references,
        model_results=model_results,
    )


def _run_reference(result: NeuralWindowRunResult) -> NeuralStudyRunReference:
    manifest = result.run.manifest
    return NeuralStudyRunReference(
        variant=manifest.variant,
        submitted_result_name=manifest.submitted_result_name,
        window_id=manifest.window.window_id,
        parameter_seed=manifest.parameter_seed,
        shuffle_seed=manifest.shuffle_seed,
        request_sha256=manifest.request_sha256,
        run_sha256=result.run.sha256,
        checkpoint_sha256=result.checkpoint.sha256,
        prediction_sha256=result.prediction.sha256,
        test_rows=manifest.phase_rows.test,
        metrics=manifest.metrics,
    )


def _serialized_window(window: RollingWindow) -> NeuralRunWindow:
    return NeuralRunWindow(
        window_id=window.window_id,
        train_start=window.train_start.isoformat(),
        validation_start=window.validation_start.isoformat(),
        test_start=window.test_start.isoformat(),
        test_end=window.test_end.isoformat(),
    )


def _validate_study_run_reference(reference: NeuralStudyRunReference) -> None:
    if type(reference.variant) is not str or reference.variant not in _VARIANT_VALUES:
        raise NeuralStudyRunError(reason="run reference variant is unsupported")
    if type(reference.submitted_result_name) is not str or not reference.submitted_result_name:
        raise NeuralStudyRunError(reason="run result name must be a nonempty plain str")
    for field, value in (
        ("window_id", reference.window_id),
        ("test_rows", reference.test_rows),
    ):
        minimum = 0 if field == "window_id" else 1
        if type(value) is not int or value < minimum:
            raise NeuralStudyRunError(reason=f"{field} is outside its plain-int domain")
    for field, value in (
        ("parameter_seed", reference.parameter_seed),
        ("shuffle_seed", reference.shuffle_seed),
    ):
        if type(value) is not int or not 0 <= value <= _MAX_SEED:
            raise NeuralStudyRunError(reason=f"{field} must be an unsigned 64-bit plain int")
    for field, value in (
        ("request_sha256", reference.request_sha256),
        ("run_sha256", reference.run_sha256),
        ("checkpoint_sha256", reference.checkpoint_sha256),
        ("prediction_sha256", reference.prediction_sha256),
    ):
        _require_sha256(value, field=field)
    if type(reference.metrics) is not ProjectPricingMetrics:
        raise NeuralStudyRunError(reason="run metrics must be ProjectPricingMetrics")
    if reference.metrics.observations != reference.test_rows:
        raise NeuralStudyRunError(reason="run metrics observations must equal test_rows")


def _validate_study_manifest(manifest: NeuralRollingStudyManifest) -> None:
    _validate_study_manifest_types(manifest)
    _validate_study_manifest_windows(manifest)
    _validate_study_manifest_runs(manifest)
    _validate_study_manifest_results(manifest)


def _validate_study_manifest_types(manifest: NeuralRollingStudyManifest) -> None:
    if type(manifest.format_version) is not int or manifest.format_version != _FORMAT_VERSION:
        raise NeuralStudyRunError(reason="format_version must be the plain int 1")
    if type(manifest.source) is not StudySourceIdentity:
        raise NeuralStudyRunError(reason="source must be StudySourceIdentity")
    _require_sha256(manifest.study_sha256, field="study_sha256")
    if type(manifest.execution_profile) is not str or not manifest.execution_profile:
        raise NeuralStudyRunError(reason="execution_profile must be a nonempty plain str")
    if type(manifest.seed_plan) is not NeuralSeedPlan:
        raise NeuralStudyRunError(reason="seed_plan must be NeuralSeedPlan")


def _validate_study_manifest_windows(manifest: NeuralRollingStudyManifest) -> None:
    if (
        type(manifest.windows) is not tuple
        or not manifest.windows
        or any(type(window) is not NeuralRunWindow for window in manifest.windows)
    ):
        raise NeuralStudyRunError(reason="windows must be nonempty NeuralRunWindow values")
    if tuple(window.window_id for window in manifest.windows) != tuple(
        range(len(manifest.windows))
    ):
        raise NeuralStudyRunError(
            reason="study window ids must be contiguous and ordered from zero"
        )


def _validate_study_manifest_runs(manifest: NeuralRollingStudyManifest) -> None:
    expected_count = len(manifest.windows) * len(NeuralVariant)
    if (
        type(manifest.runs) is not tuple
        or len(manifest.runs) != expected_count
        or any(type(run) is not NeuralStudyRunReference for run in manifest.runs)
    ):
        raise NeuralStudyRunError(reason="runs must contain every model/window result exactly once")
    observed_pairs = tuple((run.variant, run.window_id) for run in manifest.runs)
    expected_pairs = tuple(
        (variant.value, window.window_id)
        for variant in NeuralVariant
        for window in manifest.windows
    )
    if observed_pairs != expected_pairs:
        raise NeuralStudyRunError(reason="runs must use canonical model-major/window-minor order")
    for reference in manifest.runs:
        variant = NeuralVariant(reference.variant)
        if reference.parameter_seed != derive_neural_window_seed(
            manifest.seed_plan,
            NeuralSeedRole.PARAMETER,
            variant,
            reference.window_id,
        ) or reference.shuffle_seed != derive_neural_window_seed(
            manifest.seed_plan,
            NeuralSeedRole.SHUFFLE,
            variant,
            reference.window_id,
        ):
            raise NeuralStudyRunError(reason="run seeds differ from the declared seed plan")


def _validate_study_manifest_results(manifest: NeuralRollingStudyManifest) -> None:
    if (
        type(manifest.model_results) is not tuple
        or len(manifest.model_results) != len(NeuralVariant)
        or any(type(result) is not NeuralModelStudyResult for result in manifest.model_results)
    ):
        raise NeuralStudyRunError(reason="one model result is required for every neural variant")
    if tuple(result.variant for result in manifest.model_results) != tuple(
        variant.value for variant in NeuralVariant
    ):
        raise NeuralStudyRunError(reason="model results must use canonical variant order")
    for result in manifest.model_results:
        references = tuple(run for run in manifest.runs if run.variant == result.variant)
        try:
            expected = aggregate_project_pricing_metrics(
                tuple(reference.metrics for reference in references),
            )
        except ProjectMetricError as error:
            raise NeuralStudyRunError(reason=f"cannot pool model metrics: {error}") from error
        if (
            result.completed_windows != len(manifest.windows)
            or result.submitted_result_name != references[0].submitted_result_name
            or result.metrics != expected
        ):
            raise NeuralStudyRunError(reason="model result does not equal its pooled run metrics")


def _study_directory(root: Path) -> Path:
    candidate = root / "studies"
    try:
        candidate.mkdir(exist_ok=True)
    except OSError as error:
        raise NeuralStudyRunError(reason=f"cannot create studies directory: {error}") from error
    _validate_directory(candidate)
    return candidate


def _persist_bytes(directory: Path, *, destination_name: str, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".neural-rolling-study-",
        suffix=".study.json",
        dir=directory,
    )
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
                raise NeuralStudyRunError(
                    reason="content-addressed study destination contains different bytes",
                ) from None
        temporary.unlink()
    except OSError as error:
        raise NeuralStudyRunError(reason=f"study write failed: {error}") from error
    finally:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def _decode_study_manifest(value: object) -> NeuralRollingStudyManifest:
    _require_exact_keys(value, NeuralRollingStudyManifest, field="study manifest")
    payload = cast("dict[str, Any]", value)
    source = _decode_dataclass(payload["source"], StudySourceIdentity, field="source")
    seed_plan = _decode_dataclass(payload["seed_plan"], NeuralSeedPlan, field="seed_plan")
    windows = _decode_tuple(payload["windows"], NeuralRunWindow, field="windows")
    runs = _decode_run_references(payload["runs"])
    results = _decode_model_results(payload["model_results"])
    try:
        return NeuralRollingStudyManifest(
            format_version=cast("int", payload["format_version"]),
            source=source,
            study_sha256=cast("str", payload["study_sha256"]),
            execution_profile=cast("str", payload["execution_profile"]),
            seed_plan=seed_plan,
            windows=windows,
            runs=runs,
            model_results=results,
        )
    except (TypeError, NeuralStudyRunError) as error:
        raise NeuralStudyRunError(reason=f"study manifest fields are invalid: {error}") from error


def _decode_run_references(value: object) -> tuple[NeuralStudyRunReference, ...]:
    if type(value) is not list:
        raise NeuralStudyRunError(reason="runs must be an array")
    decoded: list[NeuralStudyRunReference] = []
    for item in cast("list[object]", value):
        _require_exact_keys(item, NeuralStudyRunReference, field="run reference")
        payload = cast("dict[str, Any]", item)
        metrics = _decode_dataclass(payload["metrics"], ProjectPricingMetrics, field="run metrics")
        try:
            decoded.append(
                NeuralStudyRunReference(
                    variant=payload["variant"],
                    submitted_result_name=payload["submitted_result_name"],
                    window_id=payload["window_id"],
                    parameter_seed=payload["parameter_seed"],
                    shuffle_seed=payload["shuffle_seed"],
                    request_sha256=payload["request_sha256"],
                    run_sha256=payload["run_sha256"],
                    checkpoint_sha256=payload["checkpoint_sha256"],
                    prediction_sha256=payload["prediction_sha256"],
                    test_rows=payload["test_rows"],
                    metrics=metrics,
                ),
            )
        except (TypeError, NeuralStudyRunError) as error:
            raise NeuralStudyRunError(
                reason=f"run reference fields are invalid: {error}"
            ) from error
    return tuple(decoded)


def _decode_model_results(value: object) -> tuple[NeuralModelStudyResult, ...]:
    if type(value) is not list:
        raise NeuralStudyRunError(reason="model_results must be an array")
    decoded: list[NeuralModelStudyResult] = []
    for item in cast("list[object]", value):
        _require_exact_keys(item, NeuralModelStudyResult, field="model result")
        payload = cast("dict[str, Any]", item)
        metrics = _decode_dataclass(
            payload["metrics"], ProjectPricingMetrics, field="model metrics"
        )
        try:
            decoded.append(
                NeuralModelStudyResult(
                    variant=payload["variant"],
                    submitted_result_name=payload["submitted_result_name"],
                    completed_windows=payload["completed_windows"],
                    metrics=metrics,
                ),
            )
        except (TypeError, NeuralStudyRunError) as error:
            raise NeuralStudyRunError(reason=f"model result fields are invalid: {error}") from error
    return tuple(decoded)


def _decode_tuple[T](value: object, model: type[T], *, field: str) -> tuple[T, ...]:
    if type(value) is not list:
        raise NeuralStudyRunError(reason=f"{field} must be an array")
    return tuple(
        _decode_dataclass(item, model, field=f"{field} item")
        for item in cast("list[object]", value)
    )


def _decode_dataclass[T](value: object, model: type[T], *, field: str) -> T:
    _require_exact_keys(value, model, field=field)
    try:
        return model(**cast("dict[str, Any]", value))
    except (TypeError, ValueError) as error:
        raise NeuralStudyRunError(reason=f"{field} fields are invalid: {error}") from error


def _decode_canonical_json(value: str) -> object:
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise NeuralStudyRunError(reason="study is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise NeuralStudyRunError(reason="study JSON is not canonical")
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
        raise NeuralStudyRunError(reason=f"value cannot be canonical JSON: {error}") from error


def _require_exact_keys(value: object, model: type[object], *, field: str) -> None:
    if type(value) is not dict:
        raise NeuralStudyRunError(reason=f"{field} must be an object")
    expected = {item.name for item in fields(model)}
    if set(cast("dict[object, object]", value)) != expected:
        raise NeuralStudyRunError(reason=f"{field} keys are missing or unexpected")


def _validate_directory(directory: Path) -> None:
    if not isinstance(directory, Path) or not directory.is_dir() or directory.is_symlink():
        raise NeuralStudyRunError(reason="directory must be an existing non-symlink directory")


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise NeuralStudyRunError(reason=f"study read failed: {error}") from error


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NeuralStudyRunError(reason=f"{field} must be a lowercase SHA-256 hex digest")


__all__ = [
    "NeuralModelStudyResult",
    "NeuralRollingStudyArtifact",
    "NeuralRollingStudyManifest",
    "NeuralRollingStudyRequest",
    "NeuralRollingStudyResult",
    "NeuralSeedPlan",
    "NeuralSeedRole",
    "NeuralStudyRunError",
    "NeuralStudyRunReference",
    "derive_neural_window_seed",
    "load_neural_rolling_study",
    "project_neural_seed_plan",
    "run_project_neural_rolling_study",
    "write_neural_rolling_study",
]
