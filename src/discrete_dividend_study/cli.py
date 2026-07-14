"""Fail-fast command-line entry points for reproducible project study runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl
from finoptions import CpuExecutor

from .classical_artifact import write_classical_study_artifact
from .classical_inputs import prepare_project_classical_inputs
from .classical_pricing import run_project_classical_study
from .classical_spec import (
    PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    ClassicalExecutionSpec,
)
from .event_buffers import EventBufferPolicy
from .neural_execution import (
    PROJECT_APPLE_SILICON_EXECUTION_PROFILE,
    PROJECT_PORTABLE_CPU_EXECUTION_PROFILE,
    NeuralExecutionProfile,
)
from .neural_run import resolve_study_source_identity
from .neural_spec import PROJECT_NEURAL_STUDY_SPEC
from .neural_study import (
    NeuralRollingStudyRequest,
    project_neural_seed_plan,
    run_project_neural_rolling_study,
)
from .rolling_windows import PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC, generate_rolling_windows

if TYPE_CHECKING:
    from collections.abc import Sequence

_ENGINE_REPOSITORY_NAME: Final = "finoptions-engine"
_PROFILE_APPLE: Final = "apple-silicon"
_PROFILE_PORTABLE: Final = "portable-cpu"
_SHA1_HEX_LENGTH: Final = 40
_POSITIVE_INTEGER_MESSAGE: Final = "value must be a positive integer"
_SEED_MESSAGE: Final = "seed must be an unsigned 64-bit integer"
_DOMAIN_MESSAGE: Final = "domain must be an unsigned 32-bit integer"


@dataclass(frozen=True, slots=True)
class StudyCliError(RuntimeError):
    """A local publication-run precondition is not satisfied."""

    reason: str

    def __str__(self) -> str:
        """Return the failed command-line precondition."""
        return f"study command failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class _StudyCliContext:
    feature_artifact: Path
    artifact_root: Path
    paper_root: Path
    paper_revision: str
    engine_revision: str


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit command and return zero only after verified artifact completion."""
    arguments = _parser().parse_args(argv)
    feature_artifact = _required_path(arguments.features, field="features", kind="file")
    artifact_root = _required_path(arguments.artifacts, field="artifacts", kind="directory")
    paper_root = Path(__file__).resolve().parents[2]
    engine_root = paper_root.parent / _ENGINE_REPOSITORY_NAME
    paper_revision = _clean_git_revision(paper_root)
    engine_revision = _clean_git_revision(engine_root)
    pinned_engine_revision = _pinned_engine_revision(paper_root)
    if engine_revision != pinned_engine_revision:
        raise StudyCliError(
            reason="checked-out engine revision differs from config/engine-revision.txt",
        )
    context = _StudyCliContext(
        feature_artifact=feature_artifact,
        artifact_root=artifact_root,
        paper_root=paper_root,
        paper_revision=paper_revision,
        engine_revision=engine_revision,
    )
    if arguments.command == "run-neural-study":
        return _run_neural_study(arguments, context)
    if arguments.command == "run-classical-study":
        return _run_classical_study(arguments, context)
    raise StudyCliError(reason="an explicit supported command is required")


def _run_neural_study(
    arguments: argparse.Namespace,
    context: _StudyCliContext,
) -> int:
    profile = _execution_profile(arguments.profile)
    result = run_project_neural_rolling_study(
        NeuralRollingStudyRequest(
            feature_artifact=context.feature_artifact,
            dependency_lock=context.paper_root / "uv.lock",
            paper_revision=context.paper_revision,
            engine_revision=context.engine_revision,
            windows=generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC),
            study=PROJECT_NEURAL_STUDY_SPEC,
            execution_profile=profile,
            seed_plan=project_neural_seed_plan(
                arguments.parameter_root_seed,
                arguments.shuffle_root_seed,
            ),
        ),
        context.artifact_root,
        CpuExecutor(threads=arguments.data_threads),
    )
    summary = {
        "model_results": tuple(asdict(item) for item in result.study.manifest.model_results),
        "runs": len(result.runs),
        "study_path": str(result.study.path),
        "study_sha256": result.study.sha256,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _run_classical_study(
    arguments: argparse.Namespace,
    context: _StudyCliContext,
) -> int:
    study = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC
    execution = ClassicalExecutionSpec(
        threads=arguments.threads,
        training_seed=arguments.training_seed,
        training_domain_start=arguments.training_domain_start,
        evaluation_seed=arguments.evaluation_seed,
        evaluation_domain_start=arguments.evaluation_domain_start,
    )
    source = resolve_study_source_identity(
        context.feature_artifact,
        context.paper_root / "uv.lock",
        context.paper_revision,
        context.engine_revision,
    )
    try:
        frame = (
            pl.scan_parquet(context.feature_artifact, rechunk=False)
            .select(study.input_columns())
            .collect()
        )
    except pl.exceptions.PolarsError as error:
        raise StudyCliError(
            reason=f"cannot materialize classical input columns: {error}"
        ) from error
    resolved_after_scan = resolve_study_source_identity(
        context.feature_artifact,
        context.paper_root / "uv.lock",
        context.paper_revision,
        context.engine_revision,
    )
    if resolved_after_scan != source:
        raise StudyCliError(reason="source or dependency lock changed during preparation")
    inputs = prepare_project_classical_inputs(
        frame,
        study,
        event_buffer_policy=EventBufferPolicy(arguments.event_buffer_policy),
    )
    result = run_project_classical_study(inputs, execution, study)
    artifact = write_classical_study_artifact(
        context.artifact_root,
        source,
        inputs,
        execution,
        result,
    )
    summary = {
        "artifact_path": str(artifact.path),
        "artifact_sha256": artifact.sha256,
        "request_sha256": artifact.manifest.request_sha256,
        "rows": artifact.manifest.observations,
        "metrics": {
            variant.variant.variant.value: asdict(variant.metrics) for variant in result.variants
        },
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discrete-dividend-study",
        description="Run immutable discrete-dividend option-pricing studies.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    neural = commands.add_parser(
        "run-neural-study",
        help="run or exactly resume the complete five-model rolling neural study",
    )
    neural.add_argument("--features", type=Path, required=True)
    neural.add_argument("--artifacts", type=Path, required=True)
    neural.add_argument(
        "--profile",
        choices=(_PROFILE_APPLE, _PROFILE_PORTABLE),
        required=True,
    )
    neural.add_argument("--data-threads", type=_positive_int, required=True)
    neural.add_argument("--parameter-root-seed", type=_unsigned_seed, required=True)
    neural.add_argument("--shuffle-root-seed", type=_unsigned_seed, required=True)
    classical = commands.add_parser(
        "run-classical-study",
        help="run the complete paired four-baseline classical reconstruction",
    )
    classical.add_argument("--features", type=Path, required=True)
    classical.add_argument("--artifacts", type=Path, required=True)
    classical.add_argument("--threads", type=_positive_int, required=True)
    classical.add_argument("--training-seed", type=_unsigned_seed, required=True)
    classical.add_argument("--training-domain-start", type=_unsigned_domain, required=True)
    classical.add_argument("--evaluation-seed", type=_unsigned_seed, required=True)
    classical.add_argument("--evaluation-domain-start", type=_unsigned_domain, required=True)
    classical.add_argument(
        "--event-buffer-policy",
        choices=tuple(policy.value for policy in EventBufferPolicy),
        required=True,
    )
    return parser


def _execution_profile(value: str) -> NeuralExecutionProfile:
    profiles = {
        _PROFILE_APPLE: PROJECT_APPLE_SILICON_EXECUTION_PROFILE,
        _PROFILE_PORTABLE: PROJECT_PORTABLE_CPU_EXECUTION_PROFILE,
    }
    try:
        return profiles[value]
    except KeyError as error:
        raise StudyCliError(
            reason="profile is not an approved explicit execution profile"
        ) from error


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(_POSITIVE_INTEGER_MESSAGE) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(_POSITIVE_INTEGER_MESSAGE)
    return parsed


def _unsigned_seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(_SEED_MESSAGE) from error
    if not 0 <= parsed <= (2**64) - 1:
        raise argparse.ArgumentTypeError(_SEED_MESSAGE)
    return parsed


def _unsigned_domain(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(_DOMAIN_MESSAGE) from error
    if not 0 <= parsed <= (2**32) - 1:
        raise argparse.ArgumentTypeError(_DOMAIN_MESSAGE)
    return parsed


def _required_path(value: object, *, field: str, kind: str) -> Path:
    if not isinstance(value, Path):
        raise StudyCliError(reason=f"{field} must be an explicit path")
    valid = value.is_file() if kind == "file" else value.is_dir()
    if not valid or value.is_symlink():
        raise StudyCliError(reason=f"{field} must be an existing non-symlink {kind}")
    return value


def _clean_git_revision(repository: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise StudyCliError(reason="git executable is unavailable")
    status = _git_output(
        git,
        repository,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    )
    if status:
        raise StudyCliError(reason=f"repository {repository.name!r} has uncommitted files")
    revision = _git_output(git, repository, ("rev-parse", "HEAD"))
    if len(revision) != _SHA1_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise StudyCliError(
            reason=f"repository {repository.name!r} returned an invalid full commit"
        )
    return revision


def _git_output(git: str, repository: Path, arguments: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            (git, "-C", str(repository), *arguments),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise StudyCliError(reason=f"cannot invoke git for {repository.name!r}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git command failed without diagnostics"
        raise StudyCliError(reason=f"cannot inspect repository {repository.name!r}: {detail}")
    return completed.stdout.strip()


def _pinned_engine_revision(paper_root: Path) -> str:
    path = paper_root / "config" / "engine-revision.txt"
    try:
        revision = path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise StudyCliError(reason=f"cannot read pinned engine revision: {error}") from error
    if len(revision) != _SHA1_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise StudyCliError(reason="pinned engine revision is not a full lowercase Git commit")
    return revision


__all__ = ["StudyCliError", "main"]
