"""Fail-fast command-line entry points for reproducible project study runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl
from finoptions import CpuExecutor, CpuMathBackend

from .classical_artifact import load_classical_study_artifact, write_classical_study_artifact
from .classical_comparison import (
    build_classical_publication_comparison,
    classical_publication_latex,
    classical_publication_markdown,
)
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
from .orats_archive import (
    ORATS_DEFAULT_FTPS_HOST,
    OratsArchiveInventory,
    OratsFtpsClient,
    OratsFtpsCredentials,
    OratsFtpsSpec,
    load_verified_orats_archive,
    orats_archive_name,
    read_orats_archive_inventory,
    write_orats_archive_inventory,
)
from .orats_option_projection import (
    OratsOptionProjectionSpec,
    materialize_orats_option_projection,
    orats_option_projection_spec_sha256,
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
_ISO_DATE_MESSAGE: Final = "date must use exact ISO YYYY-MM-DD syntax"


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
    if arguments.command == "inventory-orats-archives":
        return _inventory_orats_archives(arguments)
    if arguments.command == "download-orats-archives":
        return _download_orats_archives(arguments)
    if arguments.command == "materialize-orats-options":
        return _materialize_orats_options(arguments)
    if arguments.command == "render-classical-table":
        return _render_classical_table(arguments)
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


def _inventory_orats_archives(arguments: argparse.Namespace) -> int:
    manifest = _new_output_path(arguments.manifest, field="manifest")
    credentials = OratsFtpsCredentials.from_environment()
    spec = OratsFtpsSpec(host=arguments.host)
    with OratsFtpsClient(spec, credentials) as client:
        inventory = client.inventory(
            start_date=arguments.start_date,
            end_date=arguments.end_date,
        )
    write_orats_archive_inventory(manifest, inventory)
    _write_inventory_summary(inventory, manifest)
    return 0


def _download_orats_archives(arguments: argparse.Namespace) -> int:
    manifest = _required_path(arguments.manifest, field="manifest", kind="file")
    inventory = read_orats_archive_inventory(manifest)
    if arguments.confirm_entry_count != len(inventory.entries):
        raise StudyCliError(
            reason="confirm-entry-count must exactly equal the immutable manifest entry count",
        )
    destination = _destination_directory(arguments.destination)
    credentials = OratsFtpsCredentials.from_environment()
    spec = OratsFtpsSpec(host=inventory.host)
    receipts = []
    with OratsFtpsClient(spec, credentials) as client:
        for entry in inventory.entries:
            receipt = client.download(entry, destination)
            receipts.append(receipt)
            progress = {
                "archive": receipt.local_name,
                "sha256": receipt.sha256,
                "trade_date": receipt.entry.trade_date.isoformat(),
            }
            sys.stderr.write(
                json.dumps(progress, sort_keys=True, separators=(",", ":")) + "\n",
            )
    summary = {
        "destination": str(destination),
        "entries": len(receipts),
        "entries_sha256": inventory.entries_sha256,
        "manifest": str(manifest),
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _materialize_orats_options(arguments: argparse.Namespace) -> int:
    manifest = _required_path(arguments.manifest, field="manifest", kind="file")
    inventory = read_orats_archive_inventory(manifest)
    if arguments.confirm_entry_count != len(inventory.entries):
        raise StudyCliError(
            reason="confirm-entry-count must exactly equal the immutable manifest entry count",
        )
    raw_directory = _required_path(arguments.raw_directory, field="raw-directory", kind="directory")
    scratch = _required_path(arguments.scratch, field="scratch", kind="directory")
    output_root = _destination_directory(arguments.output_root)
    spec = OratsOptionProjectionSpec(tickers=tuple(arguments.ticker))
    entries_completed = 0
    output_bytes = 0
    output_rows = 0
    for entry in inventory.entries:
        archive_path = raw_directory / orats_archive_name(entry.trade_date)
        source = load_verified_orats_archive(archive_path)
        if source.receipt.entry != entry:
            raise StudyCliError(
                reason="raw archive receipt disagrees with its immutable inventory entry",
            )
        output = materialize_orats_option_projection(source, spec, output_root, scratch)
        entries_completed += 1
        output_bytes += output.receipt.output_bytes
        output_rows += output.receipt.output_rows
        progress = {
            "output": str(output.path),
            "output_sha256": output.receipt.output_sha256,
            "rows": output.receipt.output_rows,
            "trade_date": entry.trade_date.isoformat(),
        }
        sys.stderr.write(json.dumps(progress, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "entries": entries_completed,
        "entries_sha256": inventory.entries_sha256,
        "output_bytes": output_bytes,
        "output_root": str(output_root),
        "projection_spec_sha256": orats_option_projection_spec_sha256(spec),
        "rows": output_rows,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _write_inventory_summary(inventory: OratsArchiveInventory, manifest: Path) -> None:
    summary = {
        "compressed_bytes": sum(entry.size_bytes for entry in inventory.entries),
        "end_date": inventory.end_date.isoformat(),
        "entries": len(inventory.entries),
        "entries_sha256": inventory.entries_sha256,
        "host": inventory.host,
        "manifest": str(manifest),
        "start_date": inventory.start_date.isoformat(),
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )


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


def _render_classical_table(arguments: argparse.Namespace) -> int:
    artifact = _required_path(arguments.artifact, field="artifact", kind="file")
    comparison = build_classical_publication_comparison(load_classical_study_artifact(artifact))
    if arguments.format == "latex":
        output = classical_publication_latex(comparison)
    elif arguments.format == "markdown":
        output = classical_publication_markdown(comparison)
    else:
        raise StudyCliError(reason="classical table format is unsupported")
    sys.stdout.write(output)
    return 0


def _run_classical_study(
    arguments: argparse.Namespace,
    context: _StudyCliContext,
) -> int:
    study = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC
    execution = ClassicalExecutionSpec(
        threads=arguments.threads,
        math_backend=CpuMathBackend(arguments.math_backend),
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
    inventory = commands.add_parser(
        "inventory-orats-archives",
        help="write an immutable MLSD inventory over explicit TLS without downloading archives",
    )
    inventory.add_argument("--start-date", type=_iso_date, required=True)
    inventory.add_argument("--end-date", type=_iso_date, required=True)
    inventory.add_argument("--manifest", type=Path, required=True)
    inventory.add_argument("--host", default=ORATS_DEFAULT_FTPS_HOST)
    download = commands.add_parser(
        "download-orats-archives",
        help="resume every archive in a verified inventory under cryptographic receipts",
    )
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--destination", type=Path, required=True)
    download.add_argument("--confirm-entry-count", type=_positive_int, required=True)
    materialize = commands.add_parser(
        "materialize-orats-options",
        help="verify and project every inventoried option ZIP into immutable daily Parquet",
    )
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--raw-directory", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--scratch", type=Path, required=True)
    materialize.add_argument("--ticker", action="append", required=True)
    materialize.add_argument("--confirm-entry-count", type=_positive_int, required=True)
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
    classical.add_argument(
        "--math-backend",
        choices=tuple(backend.value for backend in CpuMathBackend),
        required=True,
    )
    classical.add_argument("--training-seed", type=_unsigned_seed, required=True)
    classical.add_argument("--training-domain-start", type=_unsigned_domain, required=True)
    classical.add_argument("--evaluation-seed", type=_unsigned_seed, required=True)
    classical.add_argument("--evaluation-domain-start", type=_unsigned_domain, required=True)
    classical.add_argument(
        "--event-buffer-policy",
        choices=tuple(policy.value for policy in EventBufferPolicy),
        required=True,
    )
    table = commands.add_parser(
        "render-classical-table",
        help="render a verified aggregate-only classical artifact as a publication table",
    )
    table.add_argument("--artifact", type=Path, required=True)
    table.add_argument("--format", choices=("latex", "markdown"), required=True)
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


def _iso_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(_ISO_DATE_MESSAGE) from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError(_ISO_DATE_MESSAGE)
    return parsed


def _required_path(value: object, *, field: str, kind: str) -> Path:
    if not isinstance(value, Path):
        raise StudyCliError(reason=f"{field} must be an explicit path")
    valid = value.is_file() if kind == "file" else value.is_dir()
    if not valid or value.is_symlink():
        raise StudyCliError(reason=f"{field} must be an existing non-symlink {kind}")
    return value


def _new_output_path(value: object, *, field: str) -> Path:
    if not isinstance(value, Path) or value.is_symlink() or value.exists():
        raise StudyCliError(reason=f"{field} must be a new non-symlink path")
    parent = value.expanduser().parent
    if not parent.is_dir() or parent.is_symlink():
        raise StudyCliError(reason=f"{field} parent must be an existing non-symlink directory")
    return value.expanduser()


def _destination_directory(value: object) -> Path:
    if not isinstance(value, Path) or value.is_symlink():
        raise StudyCliError(reason="destination must be an explicit non-symlink path")
    destination = value.expanduser()
    if destination.exists() and not destination.is_dir():
        raise StudyCliError(reason="destination must be absent or an existing directory")
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise StudyCliError(
            reason="destination parent must be an existing non-symlink directory",
        )
    return destination


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
