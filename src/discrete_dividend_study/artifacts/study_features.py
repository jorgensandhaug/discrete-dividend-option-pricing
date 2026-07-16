"""Explicit receipt-verified feature artifacts admitted by study runners."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import polars as pl

from discrete_dividend_study.artifacts.experiment_features import (
    MaterializedProjectExperimentArtifact,
    ProjectExperimentArtifactError,
    load_verified_project_experiment_artifact,
)
from discrete_dividend_study.artifacts.project_features import (
    MaterializedProjectFeatureArtifact,
    ProjectFeatureArtifactError,
    load_verified_project_feature_artifact,
)

_SHA256_LENGTH: Final = 64
_HASH_BLOCK_BYTES: Final = 8 * 1024 * 1024


@dataclass(slots=True)
class StudyFeatureArtifactError(RuntimeError):
    """A study feature kind, receipt, or immutable file identity is invalid."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the rejected study input without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"study feature artifact failed{location}: {self.reason}"


class StudyFeatureArtifactKind(StrEnum):
    """Closed producer contracts accepted by the project study runners."""

    PROJECT_FULL = "project-full"
    PROJECT_EXPERIMENT = "project-experiment"


@dataclass(frozen=True, slots=True)
class VerifiedStudyFeatureArtifact:
    """Unified immutable handle produced only after kind-specific receipt verification."""

    kind: StudyFeatureArtifactKind
    path: Path
    receipt_path: Path
    request_sha256: str
    pipeline_sha256: str
    feature_schema_sha256: str
    output_sha256: str
    receipt_sha256: str
    output_rows: int
    output_bytes: int
    receipt_bytes: int

    def __post_init__(self) -> None:
        """Reject malformed manually assembled handles before any study preparation."""
        if type(self.kind) is not StudyFeatureArtifactKind:
            raise StudyFeatureArtifactError(reason="kind must be StudyFeatureArtifactKind")
        if not isinstance(self.path, Path) or not isinstance(self.receipt_path, Path):
            raise StudyFeatureArtifactError(reason="artifact paths must be pathlib.Path values")
        for field, value in (
            ("request_sha256", self.request_sha256),
            ("pipeline_sha256", self.pipeline_sha256),
            ("feature_schema_sha256", self.feature_schema_sha256),
            ("output_sha256", self.output_sha256),
            ("receipt_sha256", self.receipt_sha256),
        ):
            if not _is_sha256(value):
                raise StudyFeatureArtifactError(reason=f"{field} is not a lowercase SHA-256")
        if type(self.output_rows) is not int or self.output_rows <= 0:
            raise StudyFeatureArtifactError(reason="output_rows must be a positive plain int")
        if type(self.output_bytes) is not int or self.output_bytes <= 0:
            raise StudyFeatureArtifactError(reason="output_bytes must be a positive plain int")
        if type(self.receipt_bytes) is not int or self.receipt_bytes <= 0:
            raise StudyFeatureArtifactError(reason="receipt_bytes must be a positive plain int")

    def scan(self) -> pl.LazyFrame:
        """Lazily scan the exact receipt-verified Parquet without eager rechunking."""
        return pl.scan_parquet(self.path, rechunk=False)


def load_verified_study_feature_artifact(
    path: str | Path,
    kind: StudyFeatureArtifactKind,
) -> VerifiedStudyFeatureArtifact:
    """Verify exactly one explicitly selected producer contract without inference or fallback."""
    if type(kind) is not StudyFeatureArtifactKind:
        raise StudyFeatureArtifactError(reason="kind must be StudyFeatureArtifactKind")
    output = Path(path).expanduser()
    receipt_path = output.with_name(f"{output.name}.receipt.json")
    receipt_payload = _read_receipt_snapshot(receipt_path)
    try:
        if kind is StudyFeatureArtifactKind.PROJECT_FULL:
            artifact = load_verified_project_feature_artifact(output)
        elif kind is StudyFeatureArtifactKind.PROJECT_EXPERIMENT:
            artifact = load_verified_project_experiment_artifact(output)
        else:  # pragma: no cover - exhaustive guard against a future enum member
            raise StudyFeatureArtifactError(reason="artifact kind is unsupported")
    except (ProjectFeatureArtifactError, ProjectExperimentArtifactError) as error:
        raise StudyFeatureArtifactError(
            reason=str(error), path=str(Path(path).expanduser())
        ) from error
    if _read_receipt_snapshot(receipt_path) != receipt_payload:
        raise StudyFeatureArtifactError(
            reason="feature receipt changed during verification",
            path=str(receipt_path),
        )
    return _unify_artifact(kind, artifact, receipt_payload)


def verify_study_feature_artifact_unchanged(
    artifact: VerifiedStudyFeatureArtifact,
) -> None:
    """Rehash the verified output and receipt after a consuming operation."""
    if type(artifact) is not VerifiedStudyFeatureArtifact:
        raise StudyFeatureArtifactError(reason="artifact must be VerifiedStudyFeatureArtifact")
    _verify_file_unchanged(
        artifact.path,
        expected_bytes=artifact.output_bytes,
        expected_sha256=artifact.output_sha256,
        field="feature output",
    )
    _verify_file_unchanged(
        artifact.receipt_path,
        expected_bytes=artifact.receipt_bytes,
        expected_sha256=artifact.receipt_sha256,
        field="feature receipt",
    )


def _unify_artifact(
    kind: StudyFeatureArtifactKind,
    artifact: MaterializedProjectFeatureArtifact | MaterializedProjectExperimentArtifact,
    receipt_payload: bytes,
) -> VerifiedStudyFeatureArtifact:
    expected_type = (
        MaterializedProjectFeatureArtifact
        if kind is StudyFeatureArtifactKind.PROJECT_FULL
        else MaterializedProjectExperimentArtifact
    )
    if type(artifact) is not expected_type:
        raise StudyFeatureArtifactError(reason="artifact loader returned an unexpected handle type")
    manifest = artifact.manifest
    return VerifiedStudyFeatureArtifact(
        kind=kind,
        path=artifact.path,
        receipt_path=artifact.receipt_path,
        request_sha256=manifest.request_sha256,
        pipeline_sha256=manifest.pipeline_sha256,
        feature_schema_sha256=manifest.feature_schema_sha256,
        output_sha256=manifest.output_sha256,
        receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
        output_rows=manifest.output_rows,
        output_bytes=manifest.output_bytes,
        receipt_bytes=len(receipt_payload),
    )


def _verify_file_unchanged(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    field: str,
) -> None:
    try:
        is_regular = path.is_file() and not path.is_symlink()
        observed_bytes = path.stat().st_size if is_regular else None
        observed_sha256 = _file_sha256(path) if observed_bytes == expected_bytes else None
    except OSError as error:
        raise StudyFeatureArtifactError(
            reason=f"{field} cannot be reverified: {error}",
            path=str(path),
        ) from error
    if not is_regular:
        raise StudyFeatureArtifactError(
            reason=f"{field} changed after verification",
            path=str(path),
        )
    if observed_bytes != expected_bytes:
        raise StudyFeatureArtifactError(
            reason=f"{field} byte count changed after verification",
            path=str(path),
        )
    if observed_sha256 != expected_sha256:
        raise StudyFeatureArtifactError(
            reason=f"{field} SHA-256 changed after verification",
            path=str(path),
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _read_receipt_snapshot(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise StudyFeatureArtifactError(
            reason="feature receipt must be an existing non-symlink file",
            path=str(path),
        )
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise StudyFeatureArtifactError(
            reason=f"feature receipt cannot be read: {error}",
            path=str(path),
        ) from error
    if not payload:
        raise StudyFeatureArtifactError(reason="feature receipt is empty", path=str(path))
    return payload


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "StudyFeatureArtifactError",
    "StudyFeatureArtifactKind",
    "VerifiedStudyFeatureArtifact",
    "load_verified_study_feature_artifact",
    "verify_study_feature_artifact_unchanged",
]
