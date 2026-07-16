"""Fail-fast command-line entry points for reproducible project study runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl
from finoptions import CpuExecutor, CpuMathBackend

from discrete_dividend_study.artifacts.experiment_features import (
    ProjectExperimentFeatureSources,
    materialize_project_experiment_artifact,
)
from discrete_dividend_study.artifacts.project_features import (
    ProjectFeatureSources,
    materialize_project_feature_artifact,
)
from discrete_dividend_study.artifacts.study_features import (
    StudyFeatureArtifactError,
    StudyFeatureArtifactKind,
    VerifiedStudyFeatureArtifact,
    load_verified_study_feature_artifact,
    verify_study_feature_artifact_unchanged,
)
from discrete_dividend_study.commands.parser import (
    APPLE_SILICON_PROFILE as _PROFILE_APPLE,
)
from discrete_dividend_study.commands.parser import (
    PORTABLE_CPU_PROFILE as _PROFILE_PORTABLE,
)
from discrete_dividend_study.commands.parser import build_parser as _parser
from discrete_dividend_study.core.calendar import load_project_xnys_calendar
from discrete_dividend_study.core.engine import EngineIdentityError, resolve_project_engine_revision
from discrete_dividend_study.core.parquet import verify_pinned_parquet
from discrete_dividend_study.core.resources import (
    PROJECT_INTERACTIVE_RESOURCE_BUDGET,
    LocalResourceBudgetError,
    acquire_local_heavy_process_lease,
)
from discrete_dividend_study.data.dividends.historical_projection import (
    HistoricalDividendProjectionSpec,
    historical_dividend_projection_spec_sha256,
    load_verified_historical_dividend_projection,
    materialize_historical_dividend_projection,
)
from discrete_dividend_study.data.options.quotes import LONG_FORM_OPTION_QUOTE_DTYPES
from discrete_dividend_study.data.providers.federal_reserve import (
    FederalReserveCurveProjectionSpec,
    materialize_federal_reserve_curve,
)
from discrete_dividend_study.data.providers.orats.archive import (
    OratsArchiveInventory,
    OratsFtpsClient,
    OratsFtpsCredentials,
    OratsFtpsSpec,
    load_verified_orats_archive,
    orats_archive_name,
    read_orats_archive_inventory,
    write_orats_archive_inventory,
)
from discrete_dividend_study.data.providers.orats.dividend_archive import (
    OratsDividendArchiveInventory,
    OratsDividendHttpsClient,
    OratsDividendHttpsSpec,
    load_verified_orats_dividend_archive,
    orats_dividend_archive_name,
    read_orats_dividend_archive_inventory,
    write_orats_dividend_archive_inventory,
)
from discrete_dividend_study.data.providers.orats.dividend_projection import (
    MaterializedOratsDividendProjection,
    OratsDividendProjectionSpec,
    materialize_orats_dividend_projection,
    orats_dividend_projection_spec_sha256,
)
from discrete_dividend_study.data.providers.orats.dividend_set import (
    load_verified_orats_dividend_projection_set,
    write_orats_dividend_projection_set_manifest,
)
from discrete_dividend_study.data.providers.orats.option_compaction import (
    OratsOptionCompactionSpec,
    load_verified_orats_option_compaction,
    materialize_orats_option_compaction,
)
from discrete_dividend_study.data.providers.orats.option_projection import (
    MaterializedOratsOptionProjection,
    OratsOptionProjectionSpec,
    load_verified_orats_option_projection,
    materialize_orats_option_projection,
    orats_option_projection_spec_sha256,
)
from discrete_dividend_study.data.providers.orats.option_set import (
    load_verified_orats_option_projection_set,
    write_orats_option_projection_set_manifest,
)
from discrete_dividend_study.data.rates import NSS_PARAMETER_DTYPES
from discrete_dividend_study.data.underlyings import UNDERLYING_HISTORY_DTYPES
from discrete_dividend_study.experiments.lsm_validation.cash_artifacts import (
    run_or_load_publication_cash_replication,
)
from discrete_dividend_study.experiments.lsm_validation.exercise import (
    prepare_publication_exercise_schedules,
)
from discrete_dividend_study.experiments.lsm_validation.inference_artifacts import (
    load_publication_inference_artifact,
    run_or_load_publication_empirical_inference,
)
from discrete_dividend_study.experiments.lsm_validation.partitions import (
    prepare_publication_partitions,
    publication_partitions_sha256,
)
from discrete_dividend_study.experiments.lsm_validation.result_artifacts import (
    load_publication_study_result_artifact,
    run_or_load_publication_study_result,
)
from discrete_dividend_study.experiments.lsm_validation.specification import (
    PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
    PROJECT_PUBLICATION_STUDY_SPEC,
    PublicationDeterministicExecutionSpec,
)
from discrete_dividend_study.experiments.lsm_validation.tables import (
    build_publication_inference_tables,
    publication_primary_contrast_latex,
    publication_primary_contrast_markdown,
    publication_primary_model_latex,
    publication_primary_model_markdown,
)
from discrete_dividend_study.experiments.neural.cohort import (
    NeuralCohortError,
    build_project_neural_cohort_manifest,
    write_neural_cohort_artifact,
)
from discrete_dividend_study.experiments.neural.run import resolve_study_source_identity
from discrete_dividend_study.experiments.neural.study import (
    NeuralRollingStudyRequest,
    load_neural_rolling_study,
    project_neural_seed_plan,
    run_project_neural_rolling_study,
)
from discrete_dividend_study.experiments.neural.tables import (
    build_neural_publication_table,
    neural_publication_latex,
    neural_publication_markdown,
)
from discrete_dividend_study.experiments.windows import (
    PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC,
    RollingWindowError,
    generate_rolling_windows,
    select_rolling_test_union,
    validate_rolling_test_coverage,
)
from discrete_dividend_study.models.dividend_events import EventBufferPolicy
from discrete_dividend_study.models.lsm.artifacts import (
    load_classical_study_artifact,
    write_classical_grouped_study_artifact,
    write_classical_study_artifact,
)
from discrete_dividend_study.models.lsm.comparison import (
    build_classical_publication_comparison,
    classical_publication_latex,
    classical_publication_markdown,
)
from discrete_dividend_study.models.lsm.groups import (
    build_project_classical_grouped_frame,
    prepare_project_classical_strike_groups,
)
from discrete_dividend_study.models.lsm.inputs import prepare_project_classical_inputs
from discrete_dividend_study.models.lsm.pricing import (
    run_project_classical_grouped_study,
    run_project_classical_study,
)
from discrete_dividend_study.models.lsm.specification import (
    PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    ClassicalExecutionSpec,
    ClassicalPopulationAddressing,
)
from discrete_dividend_study.models.neural.execution import (
    PROJECT_APPLE_SILICON_EXECUTION_PROFILE,
    PROJECT_PORTABLE_CPU_EXECUTION_PROFILE,
    NeuralExecutionProfile,
)
from discrete_dividend_study.models.neural.specification import PROJECT_STUDY

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

    from discrete_dividend_study.experiments.lsm_validation.partitions import (
        PreparedPublicationPartitions,
    )
    from discrete_dividend_study.experiments.neural.run import StudySourceIdentity
    from discrete_dividend_study.experiments.windows import RollingTestWindowCount, RollingWindow

_SHA1_HEX_LENGTH: Final = 40
_HEAVY_STUDY_COMMANDS: Final = frozenset(
    {
        "freeze-neural-cohort",
        "run-classical-study",
        "run-neural-study",
        "run-publication-cash-replications",
        "run-publication-empirical-inference",
    },
)


@dataclass(slots=True)
class StudyCliError(RuntimeError):
    """A local publication-run precondition is not satisfied."""

    reason: str

    def __str__(self) -> str:
        """Return the failed command-line precondition."""
        return f"study command failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class _StudyCliContext:
    feature_artifact: VerifiedStudyFeatureArtifact
    artifact_root: Path
    paper_root: Path
    paper_revision: str
    engine_revision: str


@dataclass(frozen=True, slots=True)
class _PreparedPublicationContext:
    """One source-verified reconstruction of the submitted publication test union."""

    source: StudySourceIdentity
    windows: tuple[RollingWindow, ...]
    test_counts: tuple[RollingTestWindowCount, ...]
    frame: pl.DataFrame
    inputs: PreparedPublicationPartitions


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command directly; installed heavy commands use the pre-import launcher instead."""
    arguments = _parser().parse_args(argv)
    data_result = _run_data_command(arguments)
    if data_result is not None:
        return data_result
    if arguments.command not in _HEAVY_STUDY_COMMANDS:
        raise StudyCliError(reason="an explicit supported command is required")
    lock = _local_resource_lock(arguments.local_resource_lock)
    try:
        with acquire_local_heavy_process_lease(
            lock,
            PROJECT_INTERACTIVE_RESOURCE_BUDGET,
        ):
            return _run_heavy_study_command(arguments)
    except LocalResourceBudgetError as error:
        raise StudyCliError(reason=str(error)) from error


def _main_after_local_resource_admission(argv: Sequence[str] | None = None) -> int:
    """Run one heavy command whose launcher already owns the lease and process budget."""
    arguments = _parser().parse_args(argv)
    if arguments.command not in _HEAVY_STUDY_COMMANDS:
        raise StudyCliError(reason="pre-import admission is valid only for a heavy study command")
    return _run_heavy_study_command(arguments)


def _run_heavy_study_command(arguments: argparse.Namespace) -> int:
    """Resolve and execute one heavy command while its caller owns local admission."""
    try:
        feature_artifact = load_verified_study_feature_artifact(
            arguments.features,
            StudyFeatureArtifactKind(arguments.feature_artifact_kind),
        )
    except (StudyFeatureArtifactError, ValueError) as error:
        raise StudyCliError(reason=str(error)) from error
    artifact_root = _required_path(arguments.artifacts, field="artifacts", kind="directory")
    paper_root = Path(__file__).resolve().parents[2]
    paper_revision = _clean_git_revision(paper_root)
    try:
        engine_revision = resolve_project_engine_revision(paper_root)
    except EngineIdentityError as error:
        raise StudyCliError(reason=str(error)) from error
    context = _StudyCliContext(
        feature_artifact=feature_artifact,
        artifact_root=artifact_root,
        paper_root=paper_root,
        paper_revision=paper_revision,
        engine_revision=engine_revision,
    )
    if arguments.command == "freeze-neural-cohort":
        return _freeze_neural_cohort(arguments, context)
    if arguments.command == "run-neural-study":
        return _run_neural_study(arguments, context)
    if arguments.command == "run-classical-study":
        return _run_classical_study(arguments, context)
    if arguments.command == "run-publication-cash-replications":
        return _run_publication_cash_replications(arguments, context)
    if arguments.command == "run-publication-empirical-inference":
        return _run_publication_empirical_inference(arguments, context)
    raise StudyCliError(reason="an explicit supported command is required")


def _local_resource_lock(value: object) -> Path:
    if value is None:
        user_id = os.getuid()
        return Path(tempfile.gettempdir()) / f"discrete-dividend-study-{user_id}.lock"
    if not isinstance(value, Path):
        raise StudyCliError(reason="local-resource-lock must be pathlib.Path")
    return value


def _run_data_command(arguments: argparse.Namespace) -> int | None:
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "compact-orats-options": _compact_orats_options,
        "download-orats-archives": _download_orats_archives,
        "download-orats-dividend-archives": _download_orats_dividend_archives,
        "inventory-orats-archives": _inventory_orats_archives,
        "inventory-orats-dividend-archives": _inventory_orats_dividend_archives,
        "materialize-fed-nominal-curve": _materialize_fed_nominal_curve,
        "materialize-orats-options": _materialize_orats_options,
        "materialize-orats-dividends": _materialize_orats_dividends,
        "materialize-historical-dividends": _materialize_historical_dividends,
        "materialize-project-features": _materialize_project_features,
        "materialize-project-experiment-features": (_materialize_project_experiment_features),
        "render-classical-table": _render_classical_table,
        "render-neural-table": _render_neural_table,
        "render-publication-inference-table": _render_publication_inference_table,
    }
    handler = handlers.get(arguments.command)
    return None if handler is None else handler(arguments)


def _render_publication_inference_table(arguments: argparse.Namespace) -> int:
    """Render one verified aggregate-only inference artifact without licensed rows."""
    loaded = load_publication_inference_artifact(
        _required_path(arguments.artifact, field="artifact", kind="file")
    )
    tables = build_publication_inference_tables(loaded)
    renderers = {
        ("primary-models", "markdown"): publication_primary_model_markdown,
        ("primary-models", "latex"): publication_primary_model_latex,
        ("primary-contrasts", "markdown"): publication_primary_contrast_markdown,
        ("primary-contrasts", "latex"): publication_primary_contrast_latex,
    }
    try:
        renderer = renderers[(arguments.table, arguments.format)]
    except KeyError as error:  # pragma: no cover - argparse owns the closed product
        raise StudyCliError(reason="unsupported inference table renderer") from error
    sys.stdout.write(renderer(tables))
    return 0


def _render_neural_table(arguments: argparse.Namespace) -> int:
    """Render only a complete artifact for the submitted five-model neural study."""
    artifact = load_neural_rolling_study(
        _required_path(arguments.artifact, field="artifact", kind="file")
    )
    table = build_neural_publication_table(artifact)
    if arguments.format == "latex":
        output = neural_publication_latex(table)
    elif arguments.format == "markdown":
        output = neural_publication_markdown(table)
    else:  # pragma: no cover - argparse owns the closed format domain
        raise StudyCliError(reason="neural table format is unsupported")
    sys.stdout.write(output)
    return 0


def _materialize_fed_nominal_curve(arguments: argparse.Namespace) -> int:
    result = materialize_federal_reserve_curve(
        _required_path(arguments.source, field="source", kind="file"),
        expected_source_sha256=arguments.expected_source_sha256,
        expected_source_size_bytes=arguments.expected_source_size_bytes,
        spec=FederalReserveCurveProjectionSpec(
            start_date=arguments.start_date,
            end_date_exclusive=arguments.end_date_exclusive,
        ),
        output_root=_destination_directory(arguments.output_root),
    )
    receipt = result.receipt
    summary = {
        "excluded_unavailable_rows": receipt.excluded_unavailable_rows,
        "output_path": str(result.path),
        "output_sha256": receipt.output_sha256,
        "request_sha256": receipt.request_sha256,
        "rows": receipt.output_rows,
        "selected_source_rows": receipt.selected_source_rows,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


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
    projection_manifest = _reusable_output_path(
        arguments.projection_manifest,
        field="projection-manifest",
    )
    spec = OratsOptionProjectionSpec(tickers=tuple(arguments.ticker))
    entries_completed = 0
    output_bytes = 0
    output_rows = 0
    projections: list[MaterializedOratsOptionProjection] = []
    projection_spec_sha256 = orats_option_projection_spec_sha256(spec)
    for entry in inventory.entries:
        output_path = (
            output_root
            / projection_spec_sha256
            / str(entry.trade_date.year)
            / f"{entry.trade_date.isoformat()}.parquet"
        )
        receipt_path = output_path.with_name(f"{output_path.name}.receipt.json")
        reused = output_path.exists() or output_path.is_symlink() or receipt_path.exists()
        if reused:
            output = load_verified_orats_option_projection(output_path)
        else:
            archive_path = raw_directory / orats_archive_name(entry.trade_date)
            source = load_verified_orats_archive(archive_path)
            if source.receipt.entry != entry:
                raise StudyCliError(
                    reason="raw archive receipt disagrees with its immutable inventory entry",
                )
            output = materialize_orats_option_projection(source, spec, output_root, scratch)
        if output.receipt.source_entry != entry:
            raise StudyCliError(
                reason="daily projection source disagrees with its immutable inventory entry",
            )
        if output.receipt.projection_spec != spec:
            raise StudyCliError(
                reason="daily projection receipt disagrees with the requested specification",
            )
        projections.append(output)
        entries_completed += 1
        output_bytes += output.receipt.output_bytes
        output_rows += output.receipt.output_rows
        if not reused:
            progress = {
                "output": str(output.path),
                "output_sha256": output.receipt.output_sha256,
                "rows": output.receipt.output_rows,
                "trade_date": entry.trade_date.isoformat(),
            }
            sys.stderr.write(json.dumps(progress, sort_keys=True, separators=(",", ":")) + "\n")
    verified_set = write_orats_option_projection_set_manifest(
        projection_manifest,
        manifest,
        inventory,
        output_root,
        projections,
    )
    summary = {
        "entries": entries_completed,
        "entries_sha256": inventory.entries_sha256,
        "output_bytes": output_bytes,
        "output_root": str(output_root),
        "projection_manifest": str(projection_manifest),
        "projection_spec_sha256": projection_spec_sha256,
        "projection_set_entries_sha256": verified_set.manifest.entries_sha256,
        "rows": output_rows,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _compact_orats_options(arguments: argparse.Namespace) -> int:
    projection_manifest = _required_path(
        arguments.projection_manifest,
        field="projection-manifest",
        kind="file",
    )
    inventory = _required_path(arguments.inventory, field="inventory", kind="file")
    projection_root = _required_path(
        arguments.projection_root,
        field="projection-root",
        kind="directory",
    )
    source = load_verified_orats_option_projection_set(
        projection_manifest,
        inventory,
        projection_root,
    )
    if arguments.confirm_source_entry_count != len(source.manifest.entries):
        raise StudyCliError(
            reason=(
                "confirm-source-entry-count must exactly equal the verified projection-set "
                "entry count"
            ),
        )
    output_root = _destination_directory(arguments.output_root)
    compaction_manifest = _reusable_output_path(
        arguments.compaction_manifest,
        field="compaction-manifest",
    )
    spec = OratsOptionCompactionSpec(
        start_date=arguments.start_date,
        end_date_exclusive=arguments.end_date_exclusive,
    )
    compacted = materialize_orats_option_compaction(
        source,
        spec,
        output_root,
        compaction_manifest,
    )
    summary = {
        "annual_outputs": len(compacted.manifest.entries),
        "compaction_manifest": str(compaction_manifest),
        "compaction_spec_sha256": compacted.manifest.compaction_spec_sha256,
        "entries_sha256": compacted.manifest.entries_sha256,
        "output_bytes": compacted.manifest.total_output_bytes,
        "output_root": str(output_root),
        "rows": compacted.manifest.total_output_rows,
        "selected_source_entries": compacted.manifest.selected_source_entry_count,
        "selected_source_entries_sha256": (compacted.manifest.selected_source_entries_sha256),
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _materialize_project_features(arguments: argparse.Namespace) -> int:
    options = load_verified_orats_option_compaction(
        _required_path(
            arguments.option_compaction_manifest,
            field="option-compaction-manifest",
            kind="file",
        ),
        _required_path(
            arguments.option_compaction_root,
            field="option-compaction-root",
            kind="directory",
        ),
    )
    forecasts = load_verified_orats_dividend_projection_set(
        _required_path(
            arguments.forecast_manifest,
            field="forecast-manifest",
            kind="file",
        ),
        _required_path(
            arguments.forecast_inventory,
            field="forecast-inventory",
            kind="file",
        ),
        _required_path(
            arguments.forecast_root,
            field="forecast-root",
            kind="directory",
        ),
    )
    historical = load_verified_historical_dividend_projection(
        _required_path(
            arguments.historical_dividends,
            field="historical-dividends",
            kind="file",
        ),
    )
    underlying = verify_pinned_parquet(
        _required_path(
            arguments.underlying_history,
            field="underlying-history",
            kind="file",
        ),
        expected_sha256=arguments.underlying_sha256,
        expected_rows=arguments.underlying_rows,
        expected_schema=UNDERLYING_HISTORY_DTYPES,
    )
    nss = verify_pinned_parquet(
        _required_path(arguments.nss_parameters, field="nss-parameters", kind="file"),
        expected_sha256=arguments.nss_sha256,
        expected_rows=arguments.nss_rows,
        expected_schema=NSS_PARAMETER_DTYPES,
    )
    artifact = materialize_project_feature_artifact(
        ProjectFeatureSources(options, forecasts, historical, underlying, nss),
        _destination_directory(arguments.output_root),
    )
    summary = {
        "feature_schema_sha256": artifact.manifest.feature_schema_sha256,
        "output_bytes": artifact.manifest.output_bytes,
        "output_path": str(artifact.path),
        "pipeline_sha256": artifact.manifest.pipeline_sha256,
        "quote_date_max": artifact.manifest.quote_date_max.isoformat(),
        "quote_date_min": artifact.manifest.quote_date_min.isoformat(),
        "request_sha256": artifact.manifest.request_sha256,
        "rows": artifact.manifest.output_rows,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _materialize_project_experiment_features(arguments: argparse.Namespace) -> int:
    long_form = verify_pinned_parquet(
        _required_path(
            arguments.long_form_options,
            field="long-form-options",
            kind="file",
        ),
        expected_sha256=arguments.long_form_sha256,
        expected_rows=arguments.long_form_rows,
        expected_schema=LONG_FORM_OPTION_QUOTE_DTYPES,
    )
    compacted = load_verified_orats_option_compaction(
        _required_path(
            arguments.option_compaction_manifest,
            field="option-compaction-manifest",
            kind="file",
        ),
        _required_path(
            arguments.option_compaction_root,
            field="option-compaction-root",
            kind="directory",
        ),
    )
    forecasts = load_verified_orats_dividend_projection_set(
        _required_path(
            arguments.forecast_manifest,
            field="forecast-manifest",
            kind="file",
        ),
        _required_path(
            arguments.forecast_inventory,
            field="forecast-inventory",
            kind="file",
        ),
        _required_path(
            arguments.forecast_root,
            field="forecast-root",
            kind="directory",
        ),
    )
    historical = load_verified_historical_dividend_projection(
        _required_path(
            arguments.historical_dividends,
            field="historical-dividends",
            kind="file",
        ),
    )
    underlying = verify_pinned_parquet(
        _required_path(
            arguments.underlying_history,
            field="underlying-history",
            kind="file",
        ),
        expected_sha256=arguments.underlying_sha256,
        expected_rows=arguments.underlying_rows,
        expected_schema=UNDERLYING_HISTORY_DTYPES,
    )
    nss = verify_pinned_parquet(
        _required_path(arguments.nss_parameters, field="nss-parameters", kind="file"),
        expected_sha256=arguments.nss_sha256,
        expected_rows=arguments.nss_rows,
        expected_schema=NSS_PARAMETER_DTYPES,
    )
    artifact = materialize_project_experiment_artifact(
        ProjectExperimentFeatureSources(
            long_form,
            compacted,
            forecasts,
            historical,
            underlying,
            nss,
        ),
        _destination_directory(arguments.output_root),
    )
    summary = {
        "descriptive_rows": sum(item.descriptive_rows for item in artifact.manifest.coverage),
        "feature_schema_sha256": artifact.manifest.feature_schema_sha256,
        "output_bytes": artifact.manifest.output_bytes,
        "output_path": str(artifact.path),
        "output_sha256": artifact.manifest.output_sha256,
        "pipeline_sha256": artifact.manifest.pipeline_sha256,
        "post_2015_rows": sum(item.post_2015_rows for item in artifact.manifest.coverage),
        "request_sha256": artifact.manifest.request_sha256,
        "rows": artifact.manifest.output_rows,
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


def _inventory_orats_dividend_archives(arguments: argparse.Namespace) -> int:
    manifest = _new_output_path(arguments.manifest, field="manifest")
    client = OratsDividendHttpsClient(OratsDividendHttpsSpec())
    inventory = client.inventory(start_year=arguments.start_year, end_year=arguments.end_year)
    write_orats_dividend_archive_inventory(manifest, inventory)
    _write_dividend_inventory_summary(inventory, manifest)
    return 0


def _download_orats_dividend_archives(arguments: argparse.Namespace) -> int:
    manifest = _required_path(arguments.manifest, field="manifest", kind="file")
    inventory = read_orats_dividend_archive_inventory(manifest)
    if arguments.confirm_entry_count != len(inventory.entries):
        raise StudyCliError(
            reason="confirm-entry-count must exactly equal the immutable manifest entry count",
        )
    destination = _destination_directory(arguments.destination)
    client = OratsDividendHttpsClient(OratsDividendHttpsSpec(base_url=inventory.base_url))
    completed = 0
    for entry in inventory.entries:
        receipt = client.download(entry, destination)
        completed += 1
        progress = {
            "archive": receipt.local_name,
            "sha256": receipt.sha256,
            "year": receipt.entry.year,
        }
        sys.stderr.write(json.dumps(progress, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "destination": str(destination),
        "entries": completed,
        "entries_sha256": inventory.entries_sha256,
        "manifest": str(manifest),
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _write_dividend_inventory_summary(
    inventory: OratsDividendArchiveInventory,
    manifest: Path,
) -> None:
    summary = {
        "base_url": inventory.base_url,
        "compressed_bytes": sum(entry.size_bytes for entry in inventory.entries),
        "end_year": inventory.end_year,
        "entries": len(inventory.entries),
        "entries_sha256": inventory.entries_sha256,
        "manifest": str(manifest),
        "start_year": inventory.start_year,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )


def _materialize_orats_dividends(arguments: argparse.Namespace) -> int:
    manifest = _required_path(arguments.manifest, field="manifest", kind="file")
    inventory = read_orats_dividend_archive_inventory(manifest)
    if arguments.confirm_entry_count != len(inventory.entries):
        raise StudyCliError(
            reason="confirm-entry-count must exactly equal the immutable manifest entry count",
        )
    raw_directory = _required_path(arguments.raw_directory, field="raw-directory", kind="directory")
    scratch = _required_path(arguments.scratch, field="scratch", kind="directory")
    output_root = _destination_directory(arguments.output_root)
    projection_manifest = _reusable_output_path(
        arguments.projection_manifest,
        field="projection-manifest",
    )
    spec = OratsDividendProjectionSpec(
        tickers=tuple(arguments.ticker),
        unidentified_row_policy=arguments.unidentified_row_policy,
    )
    completed = 0
    output_bytes = 0
    output_duplicate_rows = 0
    output_rows = 0
    source_unidentified_rows = 0
    snapshots = 0
    projections: list[MaterializedOratsDividendProjection] = []
    for entry in inventory.entries:
        source = load_verified_orats_dividend_archive(
            raw_directory / orats_dividend_archive_name(entry.year),
        )
        if source.receipt.entry != entry:
            raise StudyCliError(
                reason="annual dividend archive receipt disagrees with its inventory entry",
            )
        output = materialize_orats_dividend_projection(source, spec, output_root, scratch)
        projections.append(output)
        completed += 1
        output_bytes += output.receipt.output_bytes
        output_duplicate_rows += output.receipt.output_duplicate_rows
        output_rows += output.receipt.output_rows
        source_unidentified_rows += output.receipt.source_unidentified_rows
        snapshots += len(output.receipt.snapshots)
        progress = {
            "output": str(output.path),
            "output_sha256": output.receipt.output_sha256,
            "duplicate_rows": output.receipt.output_duplicate_rows,
            "rows": output.receipt.output_rows,
            "source_unidentified_rows": output.receipt.source_unidentified_rows,
            "snapshots": len(output.receipt.snapshots),
            "year": entry.year,
        }
        sys.stderr.write(json.dumps(progress, sort_keys=True, separators=(",", ":")) + "\n")
    verified_set = write_orats_dividend_projection_set_manifest(
        projection_manifest,
        manifest,
        inventory,
        output_root,
        projections,
    )
    summary = {
        "entries": completed,
        "entries_sha256": inventory.entries_sha256,
        "output_bytes": output_bytes,
        "output_duplicate_rows": output_duplicate_rows,
        "output_root": str(output_root),
        "projection_manifest": str(projection_manifest),
        "projection_spec_sha256": orats_dividend_projection_spec_sha256(spec),
        "projection_set_entries_sha256": verified_set.manifest.entries_sha256,
        "rows": output_rows,
        "snapshots": snapshots,
        "source_unidentified_rows": source_unidentified_rows,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _materialize_historical_dividends(arguments: argparse.Namespace) -> int:
    source = _required_path(arguments.source, field="source", kind="file")
    output_root = _destination_directory(arguments.output_root)
    spec = HistoricalDividendProjectionSpec(
        tickers=tuple(arguments.ticker),
        maximum_ex_date=arguments.maximum_ex_date,
    )
    output = materialize_historical_dividend_projection(
        source,
        expected_source_sha256=arguments.expected_source_sha256,
        expected_source_rows=arguments.expected_source_rows,
        spec=spec,
        output_root=output_root,
    )
    summary = {
        "observed_tickers": output.receipt.observed_tickers,
        "output": str(output.path),
        "output_bytes": output.receipt.output_bytes,
        "output_sha256": output.receipt.output_sha256,
        "projection_spec_sha256": historical_dividend_projection_spec_sha256(spec),
        "rows": output.receipt.output_rows,
        "source_sha256": output.receipt.source.sha256,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


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
            study=PROJECT_STUDY,
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


def _freeze_neural_cohort(
    arguments: argparse.Namespace,
    context: _StudyCliContext,
) -> int:
    """Prepare every model view and persist one aggregate-only cohort identity."""
    source = resolve_study_source_identity(
        context.feature_artifact,
        context.paper_root / "uv.lock",
        context.paper_revision,
        context.engine_revision,
    )
    try:
        manifest = build_project_neural_cohort_manifest(
            context.feature_artifact,
            source,
            CpuExecutor(threads=arguments.data_threads),
        )
    except NeuralCohortError as error:
        raise StudyCliError(reason=str(error)) from error
    source_after = resolve_study_source_identity(
        context.feature_artifact,
        context.paper_root / "uv.lock",
        context.paper_revision,
        context.engine_revision,
    )
    if source_after != source:
        raise StudyCliError(reason="source or dependency lock changed during cohort preparation")
    try:
        artifact = write_neural_cohort_artifact(context.artifact_root, manifest)
    except NeuralCohortError as error:
        raise StudyCliError(reason=str(error)) from error
    summary = {
        "artifact_path": str(artifact.path),
        "artifact_rows": manifest.artifact_rows,
        "cohort_sha256": artifact.sha256,
        "models": tuple(
            {
                "prepared_batch_sha256": model.prepared_batch_sha256,
                "variant": model.variant,
            }
            for model in manifest.models
        ),
        "prepared_horizon_rows": manifest.prepared_horizon_rows,
        "test_union_rows": manifest.test_union_rows,
        "tickers": tuple(
            {"test_rows": item.test_rows, "ticker": item.ticker} for item in manifest.tickers
        ),
        "windows": len(manifest.windows),
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
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
        population_addressing=ClassicalPopulationAddressing(arguments.population_addressing),
    )
    source = resolve_study_source_identity(
        context.feature_artifact,
        context.paper_root / "uv.lock",
        context.paper_revision,
        context.engine_revision,
    )
    windows = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    try:
        source_frame = select_rolling_test_union(context.feature_artifact.scan(), windows)
        if execution.population_addressing is ClassicalPopulationAddressing.CONTRACT_ROWS:
            frame = source_frame.select("quote_date", *study.input_columns()).collect()
        else:
            frame = build_project_classical_grouped_frame(source_frame, study).collect()
    except pl.exceptions.PolarsError as error:
        raise StudyCliError(
            reason=f"cannot materialize classical input columns: {error}"
        ) from error
    try:
        verify_study_feature_artifact_unchanged(context.feature_artifact)
    except StudyFeatureArtifactError as error:
        raise StudyCliError(reason=str(error)) from error
    resolved_after_scan = resolve_study_source_identity(
        context.feature_artifact,
        context.paper_root / "uv.lock",
        context.paper_revision,
        context.engine_revision,
    )
    if resolved_after_scan != source:
        raise StudyCliError(reason="source or dependency lock changed during preparation")
    try:
        test_counts = validate_rolling_test_coverage(frame.lazy(), windows)
    except (pl.exceptions.PolarsError, RollingWindowError) as error:
        raise StudyCliError(reason=f"invalid classical test cohort: {error}") from error
    event_buffer_policy = EventBufferPolicy(arguments.event_buffer_policy)
    if execution.population_addressing is ClassicalPopulationAddressing.CONTRACT_ROWS:
        row_inputs = prepare_project_classical_inputs(
            frame.drop("quote_date"),
            study,
            event_buffer_policy=event_buffer_policy,
        )
        result = run_project_classical_study(row_inputs, execution, study)
        artifact = write_classical_study_artifact(
            context.artifact_root,
            source,
            row_inputs,
            execution,
            result,
        )
    else:
        grouped_inputs = prepare_project_classical_strike_groups(
            frame,
            study,
            event_buffer_policy=event_buffer_policy,
        )
        result = run_project_classical_grouped_study(grouped_inputs, execution, study)
        artifact = write_classical_grouped_study_artifact(
            context.artifact_root,
            source,
            grouped_inputs,
            execution,
            result,
        )
    summary = {
        "artifact_path": str(artifact.path),
        "artifact_sha256": artifact.sha256,
        "request_sha256": artifact.manifest.request_sha256,
        "rows": artifact.manifest.observations,
        "simulation_units": artifact.manifest.simulation_units,
        "population_addressing": artifact.manifest.execution.population_addressing.value,
        "test_windows": len(test_counts),
        "test_start": windows[0].test_start.isoformat(),
        "test_end_exclusive": windows[-1].test_end.isoformat(),
        "metrics": {
            variant.variant.variant.value: asdict(variant.metrics) for variant in result.variants
        },
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )
    return 0


def _run_publication_cash_replications(
    arguments: argparse.Namespace,
    context: _StudyCliContext,
) -> int:
    """Run or exactly resume every frozen stochastic publication replication."""
    study = PROJECT_PUBLICATION_STUDY_SPEC
    plan = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC
    prepared = _prepare_publication_context(arguments, context)
    source = prepared.source
    windows = prepared.windows
    test_counts = prepared.test_counts
    inputs = prepared.inputs
    plan.validate_units(inputs.simulation_units)

    artifacts = []
    for replication in range(plan.replications):
        execution = plan.execution(
            replication,
            threads=arguments.threads,
            math_backend=CpuMathBackend(arguments.math_backend),
        )
        loaded = run_or_load_publication_cash_replication(
            context.artifact_root,
            source,
            inputs,
            execution,
            study=study,
            replication_plan=plan,
        )
        artifacts.append(loaded.artifact)
        progress = {
            "artifact_sha256": loaded.artifact.sha256,
            "replication": replication,
            "request_sha256": loaded.artifact.manifest.request_sha256,
            "status": "verified",
        }
        sys.stderr.write(
            json.dumps(progress, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        sys.stderr.flush()

    final = run_or_load_publication_study_result(
        context.artifact_root,
        source,
        inputs,
        tuple(artifact.path for artifact in artifacts),
        PublicationDeterministicExecutionSpec(
            threads=arguments.threads,
            math_backend=CpuMathBackend(arguments.math_backend),
        ),
        study=study,
        replication_plan=plan,
    )
    summary = {
        "artifacts": tuple(
            {
                "artifact_path": str(artifact.path),
                "artifact_sha256": artifact.sha256,
                "replication": artifact.manifest.execution.replication,
                "request_sha256": artifact.manifest.request_sha256,
            }
            for artifact in artifacts
        ),
        "exact_groups": inputs.exact.groups,
        "general_groups": inputs.general.groups,
        "replications": len(artifacts),
        "result_artifact_path": str(final.artifact.path),
        "result_artifact_sha256": final.artifact.sha256,
        "result_request_sha256": final.artifact.manifest.request_sha256,
        "rows": inputs.original_rows,
        "test_end_exclusive": windows[-1].test_end.isoformat(),
        "test_start": windows[0].test_start.isoformat(),
        "test_windows": len(test_counts),
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    return 0


def _run_publication_empirical_inference(
    arguments: argparse.Namespace,
    context: _StudyCliContext,
) -> int:
    """Run or exactly resume the frozen paired date-panel empirical inference."""
    prepared = _prepare_publication_context(arguments, context)
    publication_result_path = _required_path(
        arguments.publication_result,
        field="publication-result",
        kind="file",
    )
    publication_result = load_publication_study_result_artifact(publication_result_path)
    loaded = run_or_load_publication_empirical_inference(
        context.artifact_root,
        prepared.source,
        context.feature_artifact,
        publication_result,
        prepared.frame,
        publication_partitions_sha256(prepared.inputs),
        CpuExecutor(threads=arguments.threads, math_backend=CpuMathBackend.RUST_STD_F64),
    )
    summary = {
        "artifact_path": str(loaded.artifact.path),
        "artifact_sha256": loaded.artifact.sha256,
        "bootstrap_replications": loaded.artifact.manifest.bootstrap_replications,
        "cohorts": tuple(
            {
                "name": cohort.name,
                "observations": cohort.observations,
                "quote_date_panels": cohort.quote_date_panels,
            }
            for cohort in loaded.artifact.manifest.cohorts
        ),
        "bootstrap_designs": loaded.artifact.manifest.bootstrap_designs,
        "mean_block_lengths": loaded.artifact.manifest.mean_block_lengths,
        "publication_result_sha256": loaded.artifact.manifest.publication_result_sha256,
        "request_sha256": loaded.artifact.manifest.request_sha256,
    }
    sys.stdout.write(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    return 0


def _prepare_publication_context(
    arguments: argparse.Namespace,
    context: _StudyCliContext,
) -> _PreparedPublicationContext:
    """Verify, materialize, group, calendar-map, and partition the exact test union once."""
    source = resolve_study_source_identity(
        context.feature_artifact,
        context.paper_root / "uv.lock",
        context.paper_revision,
        context.engine_revision,
    )
    windows = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    try:
        selected = select_rolling_test_union(context.feature_artifact.scan(), windows)
        frame = build_project_classical_grouped_frame(selected).collect()
    except pl.exceptions.PolarsError as error:
        raise StudyCliError(
            reason=f"cannot materialize publication input columns: {error}"
        ) from error
    try:
        verify_study_feature_artifact_unchanged(context.feature_artifact)
    except StudyFeatureArtifactError as error:
        raise StudyCliError(reason=str(error)) from error
    resolved_after_scan = resolve_study_source_identity(
        context.feature_artifact,
        context.paper_root / "uv.lock",
        context.paper_revision,
        context.engine_revision,
    )
    if resolved_after_scan != source:
        raise StudyCliError(reason="source or dependency lock changed during preparation")
    try:
        test_counts = validate_rolling_test_coverage(frame.lazy(), windows)
    except (pl.exceptions.PolarsError, RollingWindowError) as error:
        raise StudyCliError(reason=f"invalid publication test cohort: {error}") from error
    grouped = prepare_project_classical_strike_groups(
        frame,
        event_buffer_policy=EventBufferPolicy(arguments.event_buffer_policy),
    )
    schedules = prepare_publication_exercise_schedules(
        frame,
        grouped,
        load_project_xnys_calendar(),
    )
    inputs = prepare_publication_partitions(grouped, schedules, PROJECT_PUBLICATION_STUDY_SPEC)
    return _PreparedPublicationContext(
        source=source,
        windows=windows,
        test_counts=test_counts,
        frame=frame,
        inputs=inputs,
    )


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


def _reusable_output_path(value: object, *, field: str) -> Path:
    if not isinstance(value, Path) or value.is_symlink():
        raise StudyCliError(reason=f"{field} must be an explicit non-symlink path")
    path = value.expanduser()
    if path.exists() and not path.is_file():
        raise StudyCliError(reason=f"{field} must be absent or an existing file")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise StudyCliError(reason=f"{field} parent must be an existing non-symlink directory")
    return path


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


__all__ = ["StudyCliError", "main"]
