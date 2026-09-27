"""Immutable aggregate-only admission artifact for the exact project neural cohort."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import polars as pl
from finoptions import BinnedCashDividends, PaddedCashDividends

from discrete_dividend_study.artifacts.study_features import (
    StudyFeatureArtifactError,
    VerifiedStudyFeatureArtifact,
    verify_study_feature_artifact_unchanged,
)
from discrete_dividend_study.core.serialization import dataclass_field_names
from discrete_dividend_study.data.pipeline import PROJECT_AS_SUBMITTED_TICKERS
from discrete_dividend_study.experiments.neural.rolling import (
    PROJECT_NEURAL_ROLLING_PREPARATION_SPEC,
    prepare_project_neural_rolling_source,
)
from discrete_dividend_study.experiments.neural.run import StudySourceIdentity
from discrete_dividend_study.experiments.windows import (
    PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC,
    generate_rolling_windows,
)
from discrete_dividend_study.models.neural.inputs import prepared_neural_batch_sha256
from discrete_dividend_study.models.neural.specification import (
    BinnedDividendInput,
    DirectDividendLstmInput,
    DividendYieldInput,
    EventAttentionInput,
    neural_study_sha256,
)
from discrete_dividend_study.paper import PROJECT_STUDY

if TYPE_CHECKING:
    from finoptions import CpuExecutor

    from discrete_dividend_study.experiments.windows import RollingWindow
    from discrete_dividend_study.models.neural.inputs import PreparedNeuralBatch

_FORMAT_VERSION: Final = 1
_PREPARATION_HASH_FORMAT: Final = "project-neural-cohort-preparation-v1"
_SHA256_LENGTH: Final = 64
_PADDED_KIND: Final = "padded_cash_dividends"
_BINNED_KIND: Final = "binned_cash_dividends"
_SCALAR_KIND: Final = "scalar_dividend_yield"


@dataclass(slots=True)
class NeuralCohortError(RuntimeError):
    """The project cohort, prepared tensors, or detached artifact is invalid."""

    reason: str

    def __str__(self) -> str:
        """Describe the failed aggregate-only cohort contract."""
        return f"neural cohort failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralCohortWindow:
    """Exact half-open schedule bounds and prepared row counts for one window."""

    window_id: int
    train_start: str
    validation_start: str
    test_start: str
    test_end: str
    training_rows: int
    validation_rows: int
    test_rows: int

    def __post_init__(self) -> None:
        """Reject malformed bounds, identifiers, or empty model phases."""
        if type(self.window_id) is not int or self.window_id < 0:
            raise NeuralCohortError(reason="window_id must be a nonnegative plain int")
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
            raise NeuralCohortError(reason="cohort window bounds must be strictly increasing")
        for field, value in (
            ("training_rows", self.training_rows),
            ("validation_rows", self.validation_rows),
            ("test_rows", self.test_rows),
        ):
            if type(value) is not int or value <= 0:
                raise NeuralCohortError(reason=f"{field} must be a positive plain int")


@dataclass(frozen=True, slots=True)
class NeuralCohortTickerCoverage:
    """Aggregate row and date coverage for one submitted ticker."""

    ticker: str
    artifact_rows: int
    horizon_rows: int
    test_rows: int
    artifact_date_min: str
    artifact_date_max: str
    test_date_min: str
    test_date_max: str

    def __post_init__(self) -> None:
        """Reject incomplete universes and internally inconsistent coverage."""
        if type(self.ticker) is not str or self.ticker not in PROJECT_AS_SUBMITTED_TICKERS:
            raise NeuralCohortError(reason="ticker coverage is outside the submitted universe")
        if (
            type(self.artifact_rows) is not int
            or type(self.horizon_rows) is not int
            or type(self.test_rows) is not int
            or not 0 < self.test_rows <= self.horizon_rows <= self.artifact_rows
        ):
            raise NeuralCohortError(reason="ticker row coverage is malformed")
        artifact_min = _require_iso_date(self.artifact_date_min, field="artifact_date_min")
        artifact_max = _require_iso_date(self.artifact_date_max, field="artifact_date_max")
        test_min = _require_iso_date(self.test_date_min, field="test_date_min")
        test_max = _require_iso_date(self.test_date_max, field="test_date_max")
        if not artifact_min <= test_min <= test_max <= artifact_max:
            raise NeuralCohortError(reason="ticker date coverage is inconsistent")


@dataclass(frozen=True, slots=True)
class NeuralCohortModelInput:
    """Exact full-horizon prepared tensor identity for one paper model."""

    variant: str
    submitted_result_name: str
    rows: int
    scalar_width: int
    dividend_kind: str
    dividend_width: int
    prepared_batch_sha256: str

    def __post_init__(self) -> None:
        """Reject unowned models, shapes, and incomplete tensor hashes."""
        if type(self.variant) is not str or type(self.submitted_result_name) is not str:
            raise NeuralCohortError(reason="model identity fields must be plain strings")
        if type(self.rows) is not int or self.rows <= 0:
            raise NeuralCohortError(reason="model rows must be a positive plain int")
        if type(self.scalar_width) is not int or self.scalar_width <= 0:
            raise NeuralCohortError(reason="model scalar width must be a positive plain int")
        if self.dividend_kind not in {_PADDED_KIND, _BINNED_KIND, _SCALAR_KIND}:
            raise NeuralCohortError(reason="model dividend representation is unsupported")
        if type(self.dividend_width) is not int or self.dividend_width < 0:
            raise NeuralCohortError(reason="model dividend width must be a nonnegative plain int")
        _require_sha256(self.prepared_batch_sha256, field="prepared_batch_sha256")


@dataclass(frozen=True, slots=True)
class NeuralCohortManifest:
    """Complete deterministic identity of the admitted project neural inputs."""

    format_version: int
    source: StudySourceIdentity
    feature_request_sha256: str
    feature_pipeline_sha256: str
    feature_receipt_sha256: str
    study_sha256: str
    preparation_sha256: str
    artifact_rows: int
    prepared_horizon_rows: int
    test_union_rows: int
    windows: tuple[NeuralCohortWindow, ...]
    tickers: tuple[NeuralCohortTickerCoverage, ...]
    models: tuple[NeuralCohortModelInput, ...]

    def __post_init__(self) -> None:
        """Require the exact schedule, ticker universe, and five prepared model views."""
        _validate_manifest(self)


@dataclass(frozen=True, slots=True)
class NeuralCohortArtifact:
    """One verified content-addressed cohort manifest."""

    path: Path
    sha256: str
    manifest: NeuralCohortManifest

    def __post_init__(self) -> None:
        """Reject incomplete detached handles."""
        if not isinstance(self.path, Path):
            raise NeuralCohortError(reason="cohort artifact path must be pathlib.Path")
        _require_sha256(self.sha256, field="cohort artifact SHA-256")
        if type(self.manifest) is not NeuralCohortManifest:
            raise NeuralCohortError(reason="cohort manifest has an incorrect runtime type")


def project_neural_preparation_sha256() -> str:
    """Hash the only admitted event-buffer and semantic-column preparation policy."""
    preparation = PROJECT_NEURAL_ROLLING_PREPARATION_SPEC
    columns = preparation.dividend_columns
    payload = {
        "dividend_columns": {
            "amounts": columns.amounts,
            "maturity": columns.maturity,
            "spot": columns.spot,
            "times": columns.times,
        },
        "event_buffer_policy": preparation.event_buffer_policy.value,
        "format": _PREPARATION_HASH_FORMAT,
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def build_project_neural_cohort_manifest(
    feature_artifact: VerifiedStudyFeatureArtifact,
    source: StudySourceIdentity,
    executor: CpuExecutor,
) -> NeuralCohortManifest:
    """Prepare and hash every canonical model input before any training is admitted."""
    if type(feature_artifact) is not VerifiedStudyFeatureArtifact:
        raise NeuralCohortError(reason="feature artifact must be verified before cohort admission")
    if type(source) is not StudySourceIdentity:
        raise NeuralCohortError(reason="source must be a resolved StudySourceIdentity")
    if source.feature_artifact_sha256 != feature_artifact.output_sha256:
        raise NeuralCohortError(reason="source and feature output hashes disagree")
    if source.feature_schema_sha256 != feature_artifact.feature_schema_sha256:
        raise NeuralCohortError(reason="source and feature schema hashes disagree")

    windows = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    scan = feature_artifact.scan()
    ticker_coverage = _ticker_coverage(scan, windows)
    prepared = prepare_project_neural_rolling_source(
        scan,
        windows,
        PROJECT_STUDY,
        executor,
        PROJECT_NEURAL_ROLLING_PREPARATION_SPEC,
    )
    try:
        verify_study_feature_artifact_unchanged(feature_artifact)
    except StudyFeatureArtifactError as error:
        raise NeuralCohortError(reason=str(error)) from error
    serialized_windows = tuple(
        NeuralCohortWindow(
            window_id=window.window_id,
            train_start=window.train_start.isoformat(),
            validation_start=window.validation_start.isoformat(),
            test_start=window.test_start.isoformat(),
            test_end=window.test_end.isoformat(),
            training_rows=slices.training.rows,
            validation_rows=slices.validation.rows,
            test_rows=slices.test.rows,
        )
        for window, slices in zip(windows, prepared.slices, strict=True)
    )
    models = tuple(_model_input(batch) for batch in prepared.full_batches)
    return NeuralCohortManifest(
        format_version=_FORMAT_VERSION,
        source=source,
        feature_request_sha256=feature_artifact.request_sha256,
        feature_pipeline_sha256=feature_artifact.pipeline_sha256,
        feature_receipt_sha256=feature_artifact.receipt_sha256,
        study_sha256=neural_study_sha256(PROJECT_STUDY),
        preparation_sha256=project_neural_preparation_sha256(),
        artifact_rows=feature_artifact.output_rows,
        prepared_horizon_rows=prepared.rows,
        test_union_rows=sum(item.test_rows for item in serialized_windows),
        windows=serialized_windows,
        tickers=ticker_coverage,
        models=models,
    )


def write_neural_cohort_artifact(
    artifact_root: Path,
    manifest: NeuralCohortManifest,
) -> NeuralCohortArtifact:
    """Atomically persist or exactly reuse one content-addressed cohort artifact."""
    if (
        not isinstance(artifact_root, Path)
        or not artifact_root.is_dir()
        or artifact_root.is_symlink()
    ):
        raise NeuralCohortError(reason="artifact root must be an existing non-symlink directory")
    if type(manifest) is not NeuralCohortManifest:
        raise NeuralCohortError(reason="manifest must be NeuralCohortManifest")
    directory = artifact_root / "cohorts"
    try:
        directory.mkdir(exist_ok=True)
    except OSError as error:
        raise NeuralCohortError(reason=f"cannot create cohort directory: {error}") from error
    if not directory.is_dir() or directory.is_symlink():
        raise NeuralCohortError(reason="cohort directory must be a non-symlink directory")
    payload = (_canonical_json(asdict(manifest)) + "\n").encode("ascii")
    sha256 = hashlib.sha256(payload).hexdigest()
    destination = directory / f"{sha256}.cohort.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".neural-cohort-",
        suffix=".json",
        dir=directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _read_bytes(destination) != payload:
                raise NeuralCohortError(
                    reason="cohort destination contains different bytes for the same identity"
                ) from None
        temporary.unlink()
    except OSError as error:
        raise NeuralCohortError(reason=f"cohort artifact write failed: {error}") from error
    finally:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return load_neural_cohort_artifact(destination)


def load_neural_cohort_artifact(path: Path) -> NeuralCohortArtifact:
    """Load and fully validate one canonical content-addressed cohort manifest."""
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise NeuralCohortError(reason="cohort path must be a regular non-symlink file")
    if not path.name.endswith(".cohort.json"):
        raise NeuralCohortError(reason="cohort filename must end with .cohort.json")
    filename_hash = path.name.removesuffix(".cohort.json")
    _require_sha256(filename_hash, field="cohort filename")
    payload = _read_bytes(path)
    if hashlib.sha256(payload).hexdigest() != filename_hash:
        raise NeuralCohortError(reason="cohort bytes do not match the filename hash")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise NeuralCohortError(reason="cohort artifact must be ASCII JSON") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise NeuralCohortError(reason="cohort artifact must end with exactly one newline")
    try:
        decoded: object = json.loads(text[:-1])
    except json.JSONDecodeError as error:
        raise NeuralCohortError(reason="cohort artifact is not valid JSON") from error
    if _canonical_json(decoded) != text[:-1]:
        raise NeuralCohortError(reason="cohort artifact JSON is not canonical")
    manifest = _decode_manifest(decoded)
    return NeuralCohortArtifact(path=path, sha256=filename_hash, manifest=manifest)


def _ticker_coverage(
    frame: pl.LazyFrame,
    windows: tuple[RollingWindow, ...],
) -> tuple[NeuralCohortTickerCoverage, ...]:
    horizon_start = windows[0].train_start
    horizon_end = windows[-1].test_end
    test_start = windows[0].test_start
    within_horizon = (pl.col("quote_date") >= horizon_start) & (pl.col("quote_date") < horizon_end)
    within_test = (pl.col("quote_date") >= test_start) & (pl.col("quote_date") < horizon_end)
    try:
        grouped = (
            frame.group_by("ticker")
            .agg(
                pl.len().alias("artifact_rows"),
                within_horizon.sum().alias("horizon_rows"),
                within_test.sum().alias("test_rows"),
                pl.col("quote_date").min().alias("artifact_date_min"),
                pl.col("quote_date").max().alias("artifact_date_max"),
                pl.col("quote_date").filter(within_test).min().alias("test_date_min"),
                pl.col("quote_date").filter(within_test).max().alias("test_date_max"),
            )
            .sort("ticker")
            .collect(engine="streaming")
        )
    except pl.exceptions.PolarsError as error:
        raise NeuralCohortError(reason=f"cannot aggregate ticker coverage: {error}") from error
    if grouped.get_column("ticker").to_list() != list(PROJECT_AS_SUBMITTED_TICKERS):
        raise NeuralCohortError(reason="feature artifact lacks the exact submitted ticker universe")
    return tuple(
        NeuralCohortTickerCoverage(
            ticker=ticker,
            artifact_rows=artifact_rows,
            horizon_rows=horizon_rows,
            test_rows=test_rows,
            artifact_date_min=artifact_min.isoformat(),
            artifact_date_max=artifact_max.isoformat(),
            test_date_min=test_min.isoformat(),
            test_date_max=test_max.isoformat(),
        )
        for (
            ticker,
            artifact_rows,
            horizon_rows,
            test_rows,
            artifact_min,
            artifact_max,
            test_min,
            test_max,
        ) in grouped.iter_rows()
    )


def _model_input(batch: PreparedNeuralBatch) -> NeuralCohortModelInput:
    dividends = batch.dividends
    if type(dividends) is PaddedCashDividends:
        kind = _PADDED_KIND
        width = dividends.capacity
    elif type(dividends) is BinnedCashDividends:
        kind = _BINNED_KIND
        width = dividends.bins
    elif dividends is None:
        kind = _SCALAR_KIND
        width = 0
    else:  # pragma: no cover - PreparedNeuralBatch already closes this union.
        raise NeuralCohortError(reason="prepared model has an unknown dividend tensor")
    return NeuralCohortModelInput(
        variant=batch.model.variant.value,
        submitted_result_name=batch.model.submitted_result_name,
        rows=batch.rows,
        scalar_width=len(batch.model.scalar_columns),
        dividend_kind=kind,
        dividend_width=width,
        prepared_batch_sha256=prepared_neural_batch_sha256(batch),
    )


def _validate_manifest(manifest: NeuralCohortManifest) -> None:
    _validate_manifest_identity(manifest)
    _validate_manifest_windows(manifest)
    _validate_manifest_tickers(manifest)
    _validate_manifest_models(manifest)


def _validate_manifest_identity(manifest: NeuralCohortManifest) -> None:
    if manifest.format_version != _FORMAT_VERSION:
        raise NeuralCohortError(reason="cohort format version is unsupported")
    if type(manifest.source) is not StudySourceIdentity:
        raise NeuralCohortError(reason="cohort source must be StudySourceIdentity")
    for field, value in (
        ("feature_request_sha256", manifest.feature_request_sha256),
        ("feature_pipeline_sha256", manifest.feature_pipeline_sha256),
        ("feature_receipt_sha256", manifest.feature_receipt_sha256),
        ("study_sha256", manifest.study_sha256),
        ("preparation_sha256", manifest.preparation_sha256),
    ):
        _require_sha256(value, field=field)
    if manifest.study_sha256 != neural_study_sha256(PROJECT_STUDY):
        raise NeuralCohortError(reason="cohort does not identify the canonical project study")
    if manifest.preparation_sha256 != project_neural_preparation_sha256():
        raise NeuralCohortError(reason="cohort preparation identity is unsupported")
    if any(
        type(value) is not int or value <= 0
        for value in (
            manifest.artifact_rows,
            manifest.prepared_horizon_rows,
            manifest.test_union_rows,
        )
    ):
        raise NeuralCohortError(reason="cohort aggregate row counts must be positive plain ints")


def _validate_manifest_windows(manifest: NeuralCohortManifest) -> None:
    expected_windows = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    if (
        type(manifest.windows) is not tuple
        or len(manifest.windows) != len(expected_windows)
        or any(type(item) is not NeuralCohortWindow for item in manifest.windows)
    ):
        raise NeuralCohortError(reason="cohort must contain the complete submitted schedule")
    for observed, expected in zip(manifest.windows, expected_windows, strict=True):
        if (
            observed.window_id,
            observed.train_start,
            observed.validation_start,
            observed.test_start,
            observed.test_end,
        ) != (
            expected.window_id,
            expected.train_start.isoformat(),
            expected.validation_start.isoformat(),
            expected.test_start.isoformat(),
            expected.test_end.isoformat(),
        ):
            raise NeuralCohortError(
                reason="cohort window bounds differ from the submitted schedule"
            )
    if sum(item.test_rows for item in manifest.windows) != manifest.test_union_rows:
        raise NeuralCohortError(reason="cohort window test rows do not sum to the test union")


def _validate_manifest_tickers(manifest: NeuralCohortManifest) -> None:
    if (
        type(manifest.tickers) is not tuple
        or any(type(item) is not NeuralCohortTickerCoverage for item in manifest.tickers)
        or tuple(item.ticker for item in manifest.tickers) != PROJECT_AS_SUBMITTED_TICKERS
        or sum(item.artifact_rows for item in manifest.tickers) != manifest.artifact_rows
        or sum(item.horizon_rows for item in manifest.tickers) != manifest.prepared_horizon_rows
        or sum(item.test_rows for item in manifest.tickers) != manifest.test_union_rows
    ):
        raise NeuralCohortError(reason="cohort ticker coverage is incomplete or inconsistent")


def _validate_manifest_models(manifest: NeuralCohortManifest) -> None:
    expected_models = PROJECT_STUDY.models
    if (
        type(manifest.models) is not tuple
        or len(manifest.models) != len(expected_models)
        or any(type(item) is not NeuralCohortModelInput for item in manifest.models)
    ):
        raise NeuralCohortError(reason="cohort must contain all five prepared model inputs")
    for observed, expected in zip(manifest.models, expected_models, strict=True):
        kind, width = _expected_dividend_layout(expected.dividend_input)
        if (
            observed.variant != expected.variant.value
            or observed.submitted_result_name != expected.submitted_result_name
            or observed.rows != manifest.prepared_horizon_rows
            or observed.scalar_width != len(expected.scalar_columns)
            or observed.dividend_kind != kind
            or observed.dividend_width != width
        ):
            raise NeuralCohortError(
                reason="prepared model identity or tensor shape is inconsistent"
            )


def _expected_dividend_layout(
    value: DirectDividendLstmInput | EventAttentionInput | BinnedDividendInput | DividendYieldInput,
) -> tuple[str, int]:
    if type(value) in {DirectDividendLstmInput, EventAttentionInput}:
        sequence = cast("DirectDividendLstmInput | EventAttentionInput", value)
        return _PADDED_KIND, sequence.layout.capacity
    if type(value) is BinnedDividendInput:
        return _BINNED_KIND, value.layout.bins
    if type(value) is DividendYieldInput:
        return _SCALAR_KIND, 0
    raise NeuralCohortError(reason="canonical model has an unsupported dividend input")


def _decode_manifest(value: object) -> NeuralCohortManifest:
    mapping = _exact_dataclass_mapping(value, NeuralCohortManifest, field="cohort manifest")
    source_mapping = _exact_dataclass_mapping(
        mapping["source"],
        StudySourceIdentity,
        field="cohort source",
    )
    try:
        return NeuralCohortManifest(
            format_version=cast("int", mapping["format_version"]),
            source=StudySourceIdentity(**cast("dict[str, Any]", source_mapping)),
            feature_request_sha256=cast("str", mapping["feature_request_sha256"]),
            feature_pipeline_sha256=cast("str", mapping["feature_pipeline_sha256"]),
            feature_receipt_sha256=cast("str", mapping["feature_receipt_sha256"]),
            study_sha256=cast("str", mapping["study_sha256"]),
            preparation_sha256=cast("str", mapping["preparation_sha256"]),
            artifact_rows=cast("int", mapping["artifact_rows"]),
            prepared_horizon_rows=cast("int", mapping["prepared_horizon_rows"]),
            test_union_rows=cast("int", mapping["test_union_rows"]),
            windows=_decode_tuple(mapping["windows"], NeuralCohortWindow, field="windows"),
            tickers=_decode_tuple(mapping["tickers"], NeuralCohortTickerCoverage, field="tickers"),
            models=_decode_tuple(mapping["models"], NeuralCohortModelInput, field="models"),
        )
    except (TypeError, ValueError, NeuralCohortError) as error:
        raise NeuralCohortError(reason=f"cohort manifest fields are invalid: {error}") from error


def _decode_tuple[T](value: object, model: type[T], *, field: str) -> tuple[T, ...]:
    if type(value) is not list:
        raise NeuralCohortError(reason=f"{field} must be an array")
    decoded: list[T] = []
    for item in cast("list[object]", value):
        mapping = _exact_dataclass_mapping(item, model, field=f"{field} item")
        try:
            decoded.append(model(**cast("dict[str, Any]", mapping)))
        except (TypeError, ValueError, NeuralCohortError) as error:
            raise NeuralCohortError(reason=f"{field} item is invalid: {error}") from error
    return tuple(decoded)


def _exact_dataclass_mapping(
    value: object,
    model: type[object],
    *,
    field: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise NeuralCohortError(reason=f"{field} must be an object")
    mapping = cast("dict[str, object]", value)
    if set(mapping) != dataclass_field_names(model):
        raise NeuralCohortError(reason=f"{field} keys are missing or unexpected")
    return mapping


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
        raise NeuralCohortError(reason=f"cohort value is not canonical JSON: {error}") from error


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise NeuralCohortError(reason=f"cohort artifact cannot be read: {error}") from error


def _require_iso_date(value: object, *, field: str) -> date:
    if type(value) is not str:
        raise NeuralCohortError(reason=f"{field} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise NeuralCohortError(reason=f"{field} must use canonical ISO date syntax") from error
    if parsed.isoformat() != value:
        raise NeuralCohortError(reason=f"{field} must use canonical ISO date syntax")
    return parsed


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NeuralCohortError(reason=f"{field} must be a lowercase SHA-256")


__all__ = [
    "NeuralCohortArtifact",
    "NeuralCohortError",
    "NeuralCohortManifest",
    "NeuralCohortModelInput",
    "NeuralCohortTickerCoverage",
    "NeuralCohortWindow",
    "build_project_neural_cohort_manifest",
    "load_neural_cohort_artifact",
    "project_neural_preparation_sha256",
    "write_neural_cohort_artifact",
]
