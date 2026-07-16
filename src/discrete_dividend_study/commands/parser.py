"""Declarative command-line interface for the study workflows."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Final

from finoptions import CpuMathBackend

from discrete_dividend_study.artifacts.study_features import StudyFeatureArtifactKind
from discrete_dividend_study.data.providers.orats.archive import ORATS_DEFAULT_FTPS_HOST
from discrete_dividend_study.models.dividend_events import EventBufferPolicy
from discrete_dividend_study.models.lsm.specification import ClassicalPopulationAddressing

APPLE_SILICON_PROFILE: Final = "apple-silicon"
PORTABLE_CPU_PROFILE: Final = "portable-cpu"

_POSITIVE_INTEGER_MESSAGE: Final = "value must be a positive integer"
_SEED_MESSAGE: Final = "seed must be an unsigned 64-bit integer"
_DOMAIN_MESSAGE: Final = "domain must be an unsigned 32-bit integer"
_ISO_DATE_MESSAGE: Final = "date must use exact ISO YYYY-MM-DD syntax"
_MINIMUM_ARCHIVE_YEAR: Final = 2007
_MAXIMUM_ARCHIVE_YEAR: Final = 9999


def build_parser() -> argparse.ArgumentParser:
    """Build the complete command tree without executing a workflow."""
    parser = argparse.ArgumentParser(
        prog="discrete-dividend-study",
        description="Run immutable discrete-dividend option-pricing studies.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_option_data_commands(commands)
    _add_dividend_data_commands(commands)
    _add_curve_data_command(commands)
    _add_project_feature_commands(commands)

    heavy_parent = argparse.ArgumentParser(add_help=False)
    _add_local_resource_lock_argument(heavy_parent)
    _add_neural_commands(commands, heavy_parent)
    _add_lsm_commands(commands, heavy_parent)
    _add_table_commands(commands)
    return parser


def _add_neural_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    heavy_parent: argparse.ArgumentParser,
) -> None:
    cohort = commands.add_parser(
        "freeze-neural-cohort",
        help="verify and freeze the aggregate-only five-model cohort identity",
        parents=(heavy_parent,),
    )
    _add_study_artifact_arguments(cohort)
    cohort.add_argument("--data-threads", type=_positive_int, required=True)

    neural = commands.add_parser(
        "run-neural-study",
        help="run or exactly resume the complete five-model rolling neural study",
        parents=(heavy_parent,),
    )
    _add_study_artifact_arguments(neural)
    neural.add_argument(
        "--profile",
        choices=(APPLE_SILICON_PROFILE, PORTABLE_CPU_PROFILE),
        required=True,
    )
    neural.add_argument("--data-threads", type=_positive_int, required=True)
    neural.add_argument("--parameter-root-seed", type=_unsigned_seed, required=True)
    neural.add_argument("--shuffle-root-seed", type=_unsigned_seed, required=True)


def _add_lsm_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    heavy_parent: argparse.ArgumentParser,
) -> None:
    classical = commands.add_parser(
        "run-classical-study",
        help="run the complete paired four-baseline classical reconstruction",
        parents=(heavy_parent,),
    )
    _add_study_artifact_arguments(classical)
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
        "--population-addressing",
        choices=tuple(addressing.value for addressing in ClassicalPopulationAddressing),
        required=True,
    )
    _add_event_buffer_policy_argument(classical)

    publication = commands.add_parser(
        "run-publication-cash-replications",
        help="run or exactly resume all frozen stochastic publication replications",
        parents=(heavy_parent,),
    )
    _add_study_artifact_arguments(publication)
    publication.add_argument("--threads", type=_positive_int, required=True)
    publication.add_argument(
        "--math-backend",
        choices=tuple(backend.value for backend in CpuMathBackend),
        required=True,
    )
    _add_event_buffer_policy_argument(publication)

    inference = commands.add_parser(
        "run-publication-empirical-inference",
        help="run or exactly resume frozen paired date-panel publication inference",
        parents=(heavy_parent,),
    )
    _add_study_artifact_arguments(inference)
    inference.add_argument("--publication-result", type=Path, required=True)
    inference.add_argument("--threads", type=_positive_int, required=True)
    _add_event_buffer_policy_argument(inference)


def _add_study_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--feature-artifact-kind",
        choices=tuple(kind.value for kind in StudyFeatureArtifactKind),
        required=True,
    )
    parser.add_argument("--artifacts", type=Path, required=True)


def _add_event_buffer_policy_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--event-buffer-policy",
        choices=tuple(policy.value for policy in EventBufferPolicy),
        required=True,
    )


def _add_local_resource_lock_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--local-resource-lock",
        type=Path,
        default=None,
        help="absolute shared local lock path; defaults to the current user's temporary directory",
    )


def _add_table_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    classical = commands.add_parser(
        "render-classical-table",
        help="render a verified aggregate-only classical artifact as a publication table",
    )
    classical.add_argument("--artifact", type=Path, required=True)
    classical.add_argument("--format", choices=("latex", "markdown"), required=True)

    neural = commands.add_parser(
        "render-neural-table",
        help="render the complete submitted neural study as an aggregate-only paper table",
    )
    neural.add_argument("--artifact", type=Path, required=True)
    neural.add_argument("--format", choices=("latex", "markdown"), required=True)

    inference = commands.add_parser(
        "render-publication-inference-table",
        help="render verified primary empirical inference as an aggregate-only paper table",
    )
    inference.add_argument("--artifact", type=Path, required=True)
    inference.add_argument(
        "--table",
        choices=("primary-models", "primary-contrasts"),
        required=True,
    )
    inference.add_argument("--format", choices=("latex", "markdown"), required=True)


def _add_option_data_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
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
    materialize.add_argument("--projection-manifest", type=Path, required=True)
    materialize.add_argument("--scratch", type=Path, required=True)
    materialize.add_argument("--ticker", action="append", required=True)
    materialize.add_argument("--confirm-entry-count", type=_positive_int, required=True)

    compact = commands.add_parser(
        "compact-orats-options",
        help="stream a verified daily projection set into immutable annual Parquet files",
    )
    compact.add_argument("--projection-manifest", type=Path, required=True)
    compact.add_argument("--inventory", type=Path, required=True)
    compact.add_argument("--projection-root", type=Path, required=True)
    compact.add_argument("--output-root", type=Path, required=True)
    compact.add_argument("--compaction-manifest", type=Path, required=True)
    compact.add_argument("--start-date", type=_iso_date, required=True)
    compact.add_argument("--end-date-exclusive", type=_iso_date, required=True)
    compact.add_argument("--confirm-source-entry-count", type=_positive_int, required=True)


def _add_dividend_data_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    inventory = commands.add_parser(
        "inventory-orats-dividend-archives",
        help="freeze annual ORATS dividend archive identities through strict HTTPS HEAD",
    )
    inventory.add_argument("--start-year", type=_archive_year, required=True)
    inventory.add_argument("--end-year", type=_archive_year, required=True)
    inventory.add_argument("--manifest", type=Path, required=True)

    download = commands.add_parser(
        "download-orats-dividend-archives",
        help="resume every inventoried annual ORATS dividend archive with SHA-256 receipts",
    )
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--destination", type=Path, required=True)
    download.add_argument("--confirm-entry-count", type=_positive_int, required=True)

    materialize = commands.add_parser(
        "materialize-orats-dividends",
        help="verify and project annual ORATS dividend ZIPs into immutable ticker Parquet",
    )
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--raw-directory", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--projection-manifest", type=Path, required=True)
    materialize.add_argument("--scratch", type=Path, required=True)
    materialize.add_argument("--ticker", action="append", required=True)
    materialize.add_argument(
        "--unidentified-row-policy",
        choices=("reject", "record-and-exclude"),
        default="reject",
    )
    materialize.add_argument("--confirm-entry-count", type=_positive_int, required=True)

    historical = commands.add_parser(
        "materialize-historical-dividends",
        help="pin and project one licensed historical-dividend Parquet into a study artifact",
    )
    historical.add_argument("--source", type=Path, required=True)
    historical.add_argument("--expected-source-sha256", required=True)
    historical.add_argument("--expected-source-rows", type=_positive_int, required=True)
    historical.add_argument("--maximum-ex-date", type=_iso_date, required=True)
    historical.add_argument("--output-root", type=Path, required=True)
    historical.add_argument("--ticker", action="append", required=True)


def _add_project_feature_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    materialize = commands.add_parser(
        "materialize-project-features",
        help="materialize one closed content-addressed project feature Parquet",
    )
    _add_common_project_sources(materialize)

    experiment = commands.add_parser(
        "materialize-project-experiment-features",
        help="materialize the precision-safe mixed-source project experiment artifact",
    )
    experiment.add_argument("--long-form-options", type=Path, required=True)
    experiment.add_argument("--long-form-sha256", required=True)
    experiment.add_argument("--long-form-rows", type=_positive_int, required=True)
    _add_common_project_sources(experiment)


def _add_common_project_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--option-compaction-manifest", type=Path, required=True)
    parser.add_argument("--option-compaction-root", type=Path, required=True)
    parser.add_argument("--forecast-manifest", type=Path, required=True)
    parser.add_argument("--forecast-inventory", type=Path, required=True)
    parser.add_argument("--forecast-root", type=Path, required=True)
    parser.add_argument("--historical-dividends", type=Path, required=True)
    parser.add_argument("--underlying-history", type=Path, required=True)
    parser.add_argument("--underlying-sha256", required=True)
    parser.add_argument("--underlying-rows", type=_positive_int, required=True)
    parser.add_argument("--nss-parameters", type=Path, required=True)
    parser.add_argument("--nss-sha256", required=True)
    parser.add_argument("--nss-rows", type=_positive_int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def _add_curve_data_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    materialize = commands.add_parser(
        "materialize-fed-nominal-curve",
        help="pin one official Federal Reserve nominal Svensson CSV vintage",
    )
    materialize.add_argument("--source", type=Path, required=True)
    materialize.add_argument("--expected-source-sha256", required=True)
    materialize.add_argument("--expected-source-size-bytes", type=_positive_int, required=True)
    materialize.add_argument("--start-date", type=_iso_date, required=True)
    materialize.add_argument("--end-date-exclusive", type=_iso_date, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(_POSITIVE_INTEGER_MESSAGE) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(_POSITIVE_INTEGER_MESSAGE)
    return parsed


def _archive_year(value: str) -> int:
    parsed = _positive_int(value)
    if not _MINIMUM_ARCHIVE_YEAR <= parsed <= _MAXIMUM_ARCHIVE_YEAR:
        msg = "archive year must be in [2007, 9999]"
        raise argparse.ArgumentTypeError(msg)
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


__all__ = ["APPLE_SILICON_PROFILE", "PORTABLE_CPU_PROFILE", "build_parser"]
