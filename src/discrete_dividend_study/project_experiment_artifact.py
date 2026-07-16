"""Content-addressed mixed-source artifact for the recoverable project experiments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Self, cast

import polars as pl

from .historical_dividend_projection import (
    MaterializedHistoricalDividendProjection,
    load_verified_historical_dividend_projection,
)
from .nss_rates import NSS_PARAMETER_DTYPES
from .option_quotes import (
    FLOAT32_PAIRED_OPTION_QUOTE_DTYPES,
    LONG_FORM_OPTION_QUOTE_DTYPES,
    LongFormOptionSelection,
    OptionQuotePartition,
    adapt_long_form_option_quotes,
    combine_disjoint_option_quote_partitions,
)
from .orats_dividend_set import (
    VerifiedOratsDividendProjectionSet,
    load_verified_orats_dividend_projection_set,
)
from .orats_option_compaction import (
    VerifiedOratsOptionCompaction,
    load_verified_orats_option_compaction,
    verify_orats_option_compaction_unchanged,
)
from .pinned_parquet import (
    PinnedParquetIdentity,
    VerifiedPinnedParquet,
    verify_pinned_parquet,
    verify_pinned_parquet_unchanged,
)
from .project_feature_artifact import (
    PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE,
    PROJECT_FEATURE_OPTION_START_DATE,
    PROJECT_FEATURE_WRITE_SPEC,
    ArtifactFileIdentity,
    ProjectFeatureWriteSpec,
)
from .project_pipeline import (
    PROJECT_AS_SUBMITTED_TICKERS,
    PROJECT_EXPERIMENT_FEATURE_DTYPES,
    PROJECT_EXPERIMENT_QUOTE_COHORT_DTYPES,
    build_project_experiment_quote_partition,
    enrich_project_experiment_quote_cohort,
    project_experiment_feature_pipeline_sha256,
    project_experiment_feature_schema_sha256,
)
from .quote_filters import QuoteNumericPrecision
from .underlying_history import UNDERLYING_HISTORY_DTYPES

PROJECT_EXPERIMENT_ARTIFACT_VERSION: Final = 2
PROJECT_EXPERIMENT_ARTIFACT_NAME: Final = "project_experiment_features.parquet"
PROJECT_EXPERIMENT_DESCRIPTIVE_START_DATE: Final = date(2012, 1, 1)
PROJECT_EXPERIMENT_POST_2015_START_DATE: Final = date(2015, 1, 1)
PROJECT_EXPERIMENT_LONG_FORM_TICKERS: Final = ("DUK", "EPD", "MO", "SO", "VZ")
PROJECT_EXPERIMENT_COMPACTED_TICKERS: Final = ("ETR",)
_PIPELINE_FORMAT: Final = "project-experiment-mixed-source-pipeline-v1"
_SHA256_LENGTH: Final = 64
_BLOCK_BYTES: Final = 8 * 1024 * 1024


class ProjectExperimentArtifactError(RuntimeError):
    """Base class for an invalid mixed-source experiment artifact."""


@dataclass(slots=True)
class ProjectExperimentArtifactContentError(ProjectExperimentArtifactError):
    """A source, request, receipt, or output violates the closed contract."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"project experiment artifact failed{location}: {self.reason}"


@dataclass(slots=True)
class ProjectExperimentArtifactSpecError(ProjectExperimentArtifactError):
    """A source allocation or writer runtime is unsupported."""

    reason: str

    def __str__(self) -> str:
        """Describe the invalid static request."""
        return self.reason


@dataclass(frozen=True, slots=True)
class ProjectExperimentOptionPlan:
    """Exact ticker-disjoint allocation of the submitted option horizon to two sources."""

    start_date: date
    end_date_exclusive: date
    long_form_tickers: tuple[str, ...]
    compacted_tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject alternate horizons, overlapping tickers, and incomplete universes."""
        if (
            type(self.start_date) is not date
            or type(self.end_date_exclusive) is not date
            or self.start_date != PROJECT_FEATURE_OPTION_START_DATE
            or self.end_date_exclusive != PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE
        ):
            msg = "experiment option plan must use the exact submitted quote horizon"
            raise ProjectExperimentArtifactSpecError(msg)
        for name, tickers in (
            ("long-form", self.long_form_tickers),
            ("compacted", self.compacted_tickers),
        ):
            if (
                type(tickers) is not tuple
                or not tickers
                or tickers != tuple(sorted(set(tickers)))
                or any(type(ticker) is not str for ticker in tickers)
            ):
                msg = f"{name} ticker allocation must be a sorted unique nonempty tuple"
                raise ProjectExperimentArtifactSpecError(msg)
        allocated = (*self.long_form_tickers, *self.compacted_tickers)
        if len(allocated) != len(set(allocated)):
            msg = "experiment option ticker allocations must be pairwise disjoint"
            raise ProjectExperimentArtifactSpecError(msg)
        if tuple(sorted(allocated)) != PROJECT_AS_SUBMITTED_TICKERS:
            msg = "experiment option plan must allocate the exact submitted ticker universe"
            raise ProjectExperimentArtifactSpecError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical source-allocation mapping."""
        return {
            "compacted_tickers": list(self.compacted_tickers),
            "end_date_exclusive": self.end_date_exclusive.isoformat(),
            "long_form_tickers": list(self.long_form_tickers),
            "start_date": self.start_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse one exact option allocation without coercion."""
        mapping = _exact_mapping(
            raw,
            {"compacted_tickers", "end_date_exclusive", "long_form_tickers", "start_date"},
            "experiment option plan",
        )
        long_raw = mapping["long_form_tickers"]
        compacted_raw = mapping["compacted_tickers"]
        if (
            type(mapping["start_date"]) is not str
            or type(mapping["end_date_exclusive"]) is not str
            or type(long_raw) is not list
            or type(compacted_raw) is not list
            or any(type(value) is not str for value in (*long_raw, *compacted_raw))
        ):
            msg = "experiment option plan fields are malformed"
            raise ProjectExperimentArtifactContentError(msg)
        try:
            start = date.fromisoformat(mapping["start_date"])
            end = date.fromisoformat(mapping["end_date_exclusive"])
        except ValueError as error:
            msg = "experiment option plan dates are malformed"
            raise ProjectExperimentArtifactContentError(msg) from error
        return cls(
            start_date=start,
            end_date_exclusive=end,
            long_form_tickers=tuple(cast("list[str]", long_raw)),
            compacted_tickers=tuple(cast("list[str]", compacted_raw)),
        )


PROJECT_EXPERIMENT_OPTION_PLAN: Final = ProjectExperimentOptionPlan(
    start_date=PROJECT_FEATURE_OPTION_START_DATE,
    end_date_exclusive=PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE,
    long_form_tickers=PROJECT_EXPERIMENT_LONG_FORM_TICKERS,
    compacted_tickers=PROJECT_EXPERIMENT_COMPACTED_TICKERS,
)


@dataclass(frozen=True, slots=True)
class ProjectExperimentFeatureSources:
    """Verified handles for both option partitions and all four support sources."""

    long_form_options: VerifiedPinnedParquet
    compacted_options: VerifiedOratsOptionCompaction
    forecast_dividends: VerifiedOratsDividendProjectionSet
    historical_dividends: MaterializedHistoricalDividendProjection
    underlying_history: VerifiedPinnedParquet
    nss_parameters: VerifiedPinnedParquet

    def __post_init__(self) -> None:
        """Require loader-produced source handles rather than frames or inferred paths."""
        expected = (
            (self.long_form_options, VerifiedPinnedParquet),
            (self.compacted_options, VerifiedOratsOptionCompaction),
            (self.forecast_dividends, VerifiedOratsDividendProjectionSet),
            (self.historical_dividends, MaterializedHistoricalDividendProjection),
            (self.underlying_history, VerifiedPinnedParquet),
            (self.nss_parameters, VerifiedPinnedParquet),
        )
        if any(type(value) is not kind for value, kind in expected):
            msg = "experiment feature sources must be exact verified handle types"
            raise ProjectExperimentArtifactContentError(msg)


@dataclass(frozen=True, slots=True)
class ProjectExperimentSourceIdentity:
    """Relocatable byte identities for every input that can affect the artifact."""

    long_form_options: PinnedParquetIdentity
    option_compaction_manifest: ArtifactFileIdentity
    forecast_dividend_manifest: ArtifactFileIdentity
    historical_dividend_output: ArtifactFileIdentity
    historical_dividend_receipt: ArtifactFileIdentity
    underlying_history: PinnedParquetIdentity
    nss_parameters: PinnedParquetIdentity

    def __post_init__(self) -> None:
        """Reject manually substituted identity types."""
        if any(
            type(value) is not ArtifactFileIdentity
            for value in (
                self.option_compaction_manifest,
                self.forecast_dividend_manifest,
                self.historical_dividend_output,
                self.historical_dividend_receipt,
            )
        ):
            msg = "experiment source file identities are malformed"
            raise ProjectExperimentArtifactContentError(msg)
        if any(
            type(value) is not PinnedParquetIdentity
            for value in (
                self.long_form_options,
                self.underlying_history,
                self.nss_parameters,
            )
        ):
            msg = "experiment source Parquet identities are malformed"
            raise ProjectExperimentArtifactContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical closed source mapping."""
        return {
            "forecast_dividend_manifest": self.forecast_dividend_manifest.as_dict(),
            "historical_dividend_output": self.historical_dividend_output.as_dict(),
            "historical_dividend_receipt": self.historical_dividend_receipt.as_dict(),
            "long_form_options": self.long_form_options.as_dict(),
            "nss_parameters": self.nss_parameters.as_dict(),
            "option_compaction_manifest": self.option_compaction_manifest.as_dict(),
            "underlying_history": self.underlying_history.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and validate all seven detached identities."""
        mapping = _exact_mapping(
            raw,
            {
                "forecast_dividend_manifest",
                "historical_dividend_output",
                "historical_dividend_receipt",
                "long_form_options",
                "nss_parameters",
                "option_compaction_manifest",
                "underlying_history",
            },
            "experiment source identity",
        )
        return cls(
            long_form_options=PinnedParquetIdentity.from_dict(mapping["long_form_options"]),
            option_compaction_manifest=ArtifactFileIdentity.from_dict(
                mapping["option_compaction_manifest"]
            ),
            forecast_dividend_manifest=ArtifactFileIdentity.from_dict(
                mapping["forecast_dividend_manifest"]
            ),
            historical_dividend_output=ArtifactFileIdentity.from_dict(
                mapping["historical_dividend_output"]
            ),
            historical_dividend_receipt=ArtifactFileIdentity.from_dict(
                mapping["historical_dividend_receipt"]
            ),
            underlying_history=PinnedParquetIdentity.from_dict(mapping["underlying_history"]),
            nss_parameters=PinnedParquetIdentity.from_dict(mapping["nss_parameters"]),
        )


@dataclass(frozen=True, slots=True)
class ProjectExperimentTickerCoverage:
    """Aggregate-only source coverage for one paper ticker."""

    ticker: str
    rows: int
    descriptive_rows: int
    post_2015_rows: int
    quote_date_min: date
    quote_date_max: date

    def __post_init__(self) -> None:
        """Reject malformed, empty, or internally inconsistent coverage."""
        if type(self.ticker) is not str or self.ticker not in PROJECT_AS_SUBMITTED_TICKERS:
            msg = "experiment coverage ticker is outside the submitted universe"
            raise ProjectExperimentArtifactContentError(msg)
        if (
            type(self.rows) is not int
            or type(self.descriptive_rows) is not int
            or type(self.post_2015_rows) is not int
            or not 0 <= self.post_2015_rows <= self.descriptive_rows <= self.rows
            or self.rows <= 0
        ):
            msg = "experiment coverage row counts are malformed"
            raise ProjectExperimentArtifactContentError(msg)
        if (
            type(self.quote_date_min) is not date
            or type(self.quote_date_max) is not date
            or self.quote_date_min > self.quote_date_max
            or self.quote_date_min < PROJECT_FEATURE_OPTION_START_DATE
            or self.quote_date_max >= PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE
        ):
            msg = "experiment coverage date bounds are malformed"
            raise ProjectExperimentArtifactContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical aggregate coverage mapping."""
        return {
            "descriptive_rows": self.descriptive_rows,
            "post_2015_rows": self.post_2015_rows,
            "quote_date_max": self.quote_date_max.isoformat(),
            "quote_date_min": self.quote_date_min.isoformat(),
            "rows": self.rows,
            "ticker": self.ticker,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse one exact coverage record."""
        mapping = _exact_mapping(
            raw,
            {
                "descriptive_rows",
                "post_2015_rows",
                "quote_date_max",
                "quote_date_min",
                "rows",
                "ticker",
            },
            "experiment ticker coverage",
        )
        if (
            type(mapping["ticker"]) is not str
            or type(mapping["quote_date_min"]) is not str
            or type(mapping["quote_date_max"]) is not str
            or any(
                type(mapping[name]) is not int
                for name in ("descriptive_rows", "post_2015_rows", "rows")
            )
        ):
            msg = "experiment ticker coverage fields are malformed"
            raise ProjectExperimentArtifactContentError(msg)
        try:
            minimum = date.fromisoformat(mapping["quote_date_min"])
            maximum = date.fromisoformat(mapping["quote_date_max"])
        except ValueError as error:
            msg = "experiment ticker coverage dates are malformed"
            raise ProjectExperimentArtifactContentError(msg) from error
        return cls(
            ticker=mapping["ticker"],
            rows=cast("int", mapping["rows"]),
            descriptive_rows=cast("int", mapping["descriptive_rows"]),
            post_2015_rows=cast("int", mapping["post_2015_rows"]),
            quote_date_min=minimum,
            quote_date_max=maximum,
        )


@dataclass(frozen=True, slots=True)
class ProjectExperimentArtifactManifest:
    """Complete request identity and verified output summary."""

    request_sha256: str
    sources: ProjectExperimentSourceIdentity
    option_plan: ProjectExperimentOptionPlan
    pipeline_sha256: str
    feature_schema_sha256: str
    writer: ProjectFeatureWriteSpec
    output_name: str
    output_rows: int
    output_bytes: int
    output_sha256: str
    coverage: tuple[ProjectExperimentTickerCoverage, ...]
    schema_version: int = PROJECT_EXPERIMENT_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        """Validate every request cross-identity and output aggregate."""
        if self.schema_version != PROJECT_EXPERIMENT_ARTIFACT_VERSION:
            msg = "unsupported project experiment artifact version"
            raise ProjectExperimentArtifactContentError(msg)
        if (
            type(self.sources) is not ProjectExperimentSourceIdentity
            or type(self.option_plan) is not ProjectExperimentOptionPlan
            or type(self.writer) is not ProjectFeatureWriteSpec
        ):
            msg = "project experiment request fields are malformed"
            raise ProjectExperimentArtifactContentError(msg)
        expected_pipeline = project_experiment_mixed_pipeline_sha256(self.option_plan)
        expected_schema = project_experiment_feature_schema_sha256()
        if (
            self.pipeline_sha256 != expected_pipeline
            or self.feature_schema_sha256 != expected_schema
        ):
            msg = "project experiment pipeline or schema identity is unsupported"
            raise ProjectExperimentArtifactContentError(msg)
        expected_request = project_experiment_request_sha256(
            self.sources,
            self.option_plan,
            self.writer,
        )
        if self.request_sha256 != expected_request:
            msg = "project experiment request hash disagrees with its inputs"
            raise ProjectExperimentArtifactContentError(msg)
        if (
            self.output_name != PROJECT_EXPERIMENT_ARTIFACT_NAME
            or type(self.output_rows) is not int
            or self.output_rows <= 0
            or type(self.output_bytes) is not int
            or self.output_bytes <= 0
            or type(self.output_sha256) is not str
            or not _is_sha256(self.output_sha256)
        ):
            msg = "project experiment output identity is malformed"
            raise ProjectExperimentArtifactContentError(msg)
        if (
            type(self.coverage) is not tuple
            or any(type(item) is not ProjectExperimentTickerCoverage for item in self.coverage)
            or tuple(item.ticker for item in self.coverage) != PROJECT_AS_SUBMITTED_TICKERS
            or sum(item.rows for item in self.coverage) != self.output_rows
        ):
            msg = "project experiment output coverage is incomplete or inconsistent"
            raise ProjectExperimentArtifactContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical detached receipt mapping."""
        return {
            "coverage": [item.as_dict() for item in self.coverage],
            "feature_schema_sha256": self.feature_schema_sha256,
            "option_plan": self.option_plan.as_dict(),
            "output_bytes": self.output_bytes,
            "output_name": self.output_name,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "pipeline_sha256": self.pipeline_sha256,
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "sources": self.sources.as_dict(),
            "writer": self.writer.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully revalidate one artifact receipt."""
        mapping = _exact_mapping(
            raw,
            {
                "coverage",
                "feature_schema_sha256",
                "option_plan",
                "output_bytes",
                "output_name",
                "output_rows",
                "output_sha256",
                "pipeline_sha256",
                "request_sha256",
                "schema_version",
                "sources",
                "writer",
            },
            "project experiment artifact receipt",
        )
        string_fields = (
            "feature_schema_sha256",
            "output_name",
            "output_sha256",
            "pipeline_sha256",
            "request_sha256",
        )
        coverage_raw = mapping["coverage"]
        if (
            any(type(mapping[name]) is not str for name in string_fields)
            or any(
                type(mapping[name]) is not int
                for name in ("output_bytes", "output_rows", "schema_version")
            )
            or type(coverage_raw) is not list
        ):
            msg = "project experiment artifact receipt fields are malformed"
            raise ProjectExperimentArtifactContentError(msg)
        return cls(
            request_sha256=cast("str", mapping["request_sha256"]),
            sources=ProjectExperimentSourceIdentity.from_dict(mapping["sources"]),
            option_plan=ProjectExperimentOptionPlan.from_dict(mapping["option_plan"]),
            pipeline_sha256=cast("str", mapping["pipeline_sha256"]),
            feature_schema_sha256=cast("str", mapping["feature_schema_sha256"]),
            writer=ProjectFeatureWriteSpec.from_dict(mapping["writer"]),
            output_name=cast("str", mapping["output_name"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            coverage=tuple(
                ProjectExperimentTickerCoverage.from_dict(item)
                for item in cast("list[object]", coverage_raw)
            ),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class MaterializedProjectExperimentArtifact:
    """One verified content-addressed experiment feature Parquet and receipt."""

    path: Path
    receipt_path: Path
    manifest: ProjectExperimentArtifactManifest

    def scan(self) -> pl.LazyFrame:
        """Lazily scan the exact verified feature artifact."""
        return pl.scan_parquet(self.path, rechunk=False)


def project_experiment_mixed_pipeline_sha256(plan: ProjectExperimentOptionPlan) -> str:
    """Hash the two precision-specific quote paths and their exact ticker allocation."""
    if type(plan) is not ProjectExperimentOptionPlan:
        msg = "plan must be a ProjectExperimentOptionPlan value"
        raise ProjectExperimentArtifactSpecError(msg)
    payload = {
        "artifact_version": PROJECT_EXPERIMENT_ARTIFACT_VERSION,
        "feature_schema_sha256": project_experiment_feature_schema_sha256(),
        "format": _PIPELINE_FORMAT,
        "option_plan": plan.as_dict(),
        "partitions": {
            "compacted": {
                "pipeline_sha256": project_experiment_feature_pipeline_sha256(
                    QuoteNumericPrecision.FLOAT32
                ),
                "precision": QuoteNumericPrecision.FLOAT32.value,
                "tickers": list(plan.compacted_tickers),
            },
            "long_form": {
                "pipeline_sha256": project_experiment_feature_pipeline_sha256(
                    QuoteNumericPrecision.FLOAT64
                ),
                "precision": QuoteNumericPrecision.FLOAT64.value,
                "tickers": list(plan.long_form_tickers),
            },
        },
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def project_experiment_request_sha256(
    sources: ProjectExperimentSourceIdentity,
    plan: ProjectExperimentOptionPlan,
    writer: ProjectFeatureWriteSpec,
) -> str:
    """Hash every source, financial transformation, allocation, and writer identity."""
    if (
        type(sources) is not ProjectExperimentSourceIdentity
        or type(plan) is not ProjectExperimentOptionPlan
        or type(writer) is not ProjectFeatureWriteSpec
    ):
        msg = "experiment artifact request values have incorrect runtime types"
        raise ProjectExperimentArtifactSpecError(msg)
    payload = {
        "feature_schema_sha256": project_experiment_feature_schema_sha256(),
        "option_plan": plan.as_dict(),
        "pipeline_sha256": project_experiment_mixed_pipeline_sha256(plan),
        "sources": sources.as_dict(),
        "writer": writer.as_dict(),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def materialize_project_experiment_artifact(
    sources: ProjectExperimentFeatureSources,
    output_root: str | Path,
    *,
    plan: ProjectExperimentOptionPlan = PROJECT_EXPERIMENT_OPTION_PLAN,
    writer: ProjectFeatureWriteSpec = PROJECT_FEATURE_WRITE_SPEC,
) -> MaterializedProjectExperimentArtifact:
    """Verify six sources, preserve source precision through filtering, and write one artifact."""
    _require_writer_runtime(writer)
    verified = _require_verified_sources(sources, plan)
    source_identity = _source_identity(verified)
    request_sha256 = project_experiment_request_sha256(source_identity, plan, writer)
    root = _output_root(output_root)
    output_directory = root / request_sha256
    output = output_directory / PROJECT_EXPERIMENT_ARTIFACT_NAME
    receipt = output.with_name(f"{output.name}.receipt.json")
    if output.exists() or receipt.exists():
        existing = load_verified_project_experiment_artifact(output)
        if (
            existing.manifest.sources != source_identity
            or existing.manifest.option_plan != plan
            or existing.manifest.writer != writer
        ):
            msg = "existing experiment artifact disagrees with the requested inputs"
            raise ProjectExperimentArtifactContentError(msg, str(output))
        return existing
    output_directory.mkdir(parents=True, exist_ok=False)
    if output_directory.is_symlink():
        msg = "experiment artifact request directory must not be a symlink"
        raise ProjectExperimentArtifactContentError(msg, str(output_directory))
    partial = output.with_name(f".{output.name}.part")
    partial_receipt = receipt.with_name(f".{receipt.name}.part")
    quote_stage = output.with_name(".project_experiment_quotes.parquet.part")
    try:
        quotes = _quote_frame(verified, plan)
        quotes.sink_parquet(
            quote_stage,
            compression=writer.quote_stage_compression,
            statistics=writer.quote_stage_statistics,
            row_group_size=writer.quote_stage_row_group_size,
            maintain_order=writer.quote_stage_maintain_order,
            sync_on_close=None,
            mkdir=False,
            engine=writer.engine,
        )
        enriched = enrich_project_experiment_quote_cohort(
            pl.scan_parquet(quote_stage, rechunk=False),
            verified.underlying_history.scan(),
            verified.forecast_dividends.scan(),
            pl.scan_parquet(verified.historical_dividends.path, rechunk=False),
            verified.nss_parameters.scan(),
        )
        enriched.sink_parquet(
            partial,
            compression=writer.compression,
            compression_level=writer.compression_level,
            statistics=writer.statistics,
            row_group_size=writer.row_group_size,
            maintain_order=writer.maintain_order,
            sync_on_close=writer.sync_on_close,
            mkdir=False,
            engine=writer.engine,
        )
        _fsync_file(partial)
        _require_sources_unchanged(verified)
        coverage = _inspect_output(partial)
        manifest = ProjectExperimentArtifactManifest(
            request_sha256=request_sha256,
            sources=source_identity,
            option_plan=plan,
            pipeline_sha256=project_experiment_mixed_pipeline_sha256(plan),
            feature_schema_sha256=project_experiment_feature_schema_sha256(),
            writer=writer,
            output_name=PROJECT_EXPERIMENT_ARTIFACT_NAME,
            output_rows=sum(item.rows for item in coverage),
            output_bytes=partial.stat().st_size,
            output_sha256=_file_sha256(partial),
            coverage=coverage,
        )
        _write_new_json(partial_receipt, manifest.as_dict())
        partial.replace(output)
        try:
            partial_receipt.replace(receipt)
        except BaseException:
            output.replace(partial)
            raise
    except pl.exceptions.PolarsError as error:
        msg = "streaming project experiment materialization failed"
        raise ProjectExperimentArtifactContentError(msg, str(output)) from error
    finally:
        partial.unlink(missing_ok=True)
        partial_receipt.unlink(missing_ok=True)
        quote_stage.unlink(missing_ok=True)
        if output_directory.is_dir() and not any(output_directory.iterdir()):
            output_directory.rmdir()
    return load_verified_project_experiment_artifact(output)


def load_verified_project_experiment_artifact(
    path: str | Path,
) -> MaterializedProjectExperimentArtifact:
    """Verify a detached output, receipt, schema, content hash, and aggregate coverage."""
    output = _regular_file(path, "project experiment output")
    receipt = _regular_file(
        output.with_name(f"{output.name}.receipt.json"),
        "project experiment receipt",
    )
    manifest = _read_manifest(receipt)
    if output.parent.name != manifest.request_sha256:
        msg = "experiment request directory disagrees with its receipt"
        raise ProjectExperimentArtifactContentError(msg, str(output.parent))
    if output.name != manifest.output_name or output.stat().st_size != manifest.output_bytes:
        msg = "experiment output name or byte count disagrees with its receipt"
        raise ProjectExperimentArtifactContentError(msg, str(output))
    if _file_sha256(output) != manifest.output_sha256:
        msg = "experiment output bytes disagree with its receipt SHA-256"
        raise ProjectExperimentArtifactContentError(msg, str(output))
    coverage = _inspect_output(output)
    if coverage != manifest.coverage or sum(item.rows for item in coverage) != manifest.output_rows:
        msg = "experiment output coverage disagrees with its receipt"
        raise ProjectExperimentArtifactContentError(msg, str(output))
    return MaterializedProjectExperimentArtifact(output, receipt, manifest)


def _quote_frame(
    sources: ProjectExperimentFeatureSources,
    plan: ProjectExperimentOptionPlan,
) -> pl.LazyFrame:
    long_form = adapt_long_form_option_quotes(
        sources.long_form_options.scan(),
        LongFormOptionSelection(
            tickers=plan.long_form_tickers,
            start_date=plan.start_date,
            end_date_exclusive=plan.end_date_exclusive,
        ),
    )
    long_partition = build_project_experiment_quote_partition(
        long_form.frame,
        sources.nss_parameters.scan(),
        tickers=plan.long_form_tickers,
        source_precision=QuoteNumericPrecision.FLOAT64,
    )
    compacted = sources.compacted_options.scan().filter(
        pl.col("ticker").is_in(plan.compacted_tickers)
        & (pl.col("quote_date") >= plan.start_date)
        & (pl.col("quote_date") < plan.end_date_exclusive)
    )
    compacted_partition = build_project_experiment_quote_partition(
        compacted.select(tuple(FLOAT32_PAIRED_OPTION_QUOTE_DTYPES)),
        sources.nss_parameters.scan(),
        tickers=plan.compacted_tickers,
        source_precision=QuoteNumericPrecision.FLOAT32,
    )
    return combine_disjoint_option_quote_partitions(
        (
            OptionQuotePartition("long_form_float64", long_partition, plan.long_form_tickers),
            OptionQuotePartition("compacted_float32", compacted_partition, plan.compacted_tickers),
        ),
        expected_schema=PROJECT_EXPERIMENT_QUOTE_COHORT_DTYPES,
        required_tickers=PROJECT_AS_SUBMITTED_TICKERS,
    ).frame


def _require_verified_sources(
    sources: ProjectExperimentFeatureSources,
    plan: ProjectExperimentOptionPlan,
) -> ProjectExperimentFeatureSources:
    if type(sources) is not ProjectExperimentFeatureSources:
        msg = "sources must be a ProjectExperimentFeatureSources value"
        raise ProjectExperimentArtifactContentError(msg)
    if type(plan) is not ProjectExperimentOptionPlan:
        msg = "plan must be a ProjectExperimentOptionPlan value"
        raise ProjectExperimentArtifactContentError(msg)
    long_form = verify_pinned_parquet(
        sources.long_form_options.path,
        expected_sha256=sources.long_form_options.identity.sha256,
        expected_rows=sources.long_form_options.identity.rows,
        expected_schema=LONG_FORM_OPTION_QUOTE_DTYPES,
    )
    compacted = load_verified_orats_option_compaction(
        sources.compacted_options.manifest_path,
        sources.compacted_options.output_root,
    )
    forecasts = load_verified_orats_dividend_projection_set(
        sources.forecast_dividends.manifest_path,
        sources.forecast_dividends.inventory_path,
        sources.forecast_dividends.output_root,
    )
    historical = load_verified_historical_dividend_projection(sources.historical_dividends.path)
    underlying = verify_pinned_parquet(
        sources.underlying_history.path,
        expected_sha256=sources.underlying_history.identity.sha256,
        expected_rows=sources.underlying_history.identity.rows,
        expected_schema=UNDERLYING_HISTORY_DTYPES,
    )
    nss = verify_pinned_parquet(
        sources.nss_parameters.path,
        expected_sha256=sources.nss_parameters.identity.sha256,
        expected_rows=sources.nss_parameters.identity.rows,
        expected_schema=NSS_PARAMETER_DTYPES,
    )
    verified = ProjectExperimentFeatureSources(
        long_form,
        compacted,
        forecasts,
        historical,
        underlying,
        nss,
    )
    if verified != sources:
        msg = "supplied experiment handles disagree with their verified source artifacts"
        raise ProjectExperimentArtifactContentError(msg)
    _validate_source_universes(verified, plan)
    return verified


def _validate_source_universes(
    sources: ProjectExperimentFeatureSources,
    plan: ProjectExperimentOptionPlan,
) -> None:
    compacted = sources.compacted_options.manifest
    if compacted.projected_tickers != PROJECT_AS_SUBMITTED_TICKERS:
        msg = "compacted option source must declare the exact submitted ticker universe"
        raise ProjectExperimentArtifactContentError(
            msg, str(sources.compacted_options.manifest_path)
        )
    if (
        compacted.compaction_spec.start_date != plan.start_date
        or compacted.compaction_spec.end_date_exclusive > plan.end_date_exclusive
        or compacted.selected_first_trade_date < plan.start_date
        or compacted.selected_last_trade_date >= plan.end_date_exclusive
    ):
        msg = "compacted option coverage lies outside the declared experiment horizon"
        raise ProjectExperimentArtifactContentError(
            msg, str(sources.compacted_options.manifest_path)
        )
    if sources.forecast_dividends.manifest.projection_spec.tickers != PROJECT_AS_SUBMITTED_TICKERS:
        msg = "forecast-dividend source must declare the exact submitted ticker universe"
        raise ProjectExperimentArtifactContentError(
            msg, str(sources.forecast_dividends.manifest_path)
        )
    historical_spec = sources.historical_dividends.receipt.projection_spec
    if (
        historical_spec.tickers != PROJECT_AS_SUBMITTED_TICKERS
        or historical_spec.maximum_ex_date != plan.end_date_exclusive - date.resolution
    ):
        msg = "historical-dividend source must match the exact universe and ex-date horizon"
        raise ProjectExperimentArtifactContentError(msg, str(sources.historical_dividends.path))


def _source_identity(sources: ProjectExperimentFeatureSources) -> ProjectExperimentSourceIdentity:
    historical_receipt = sources.historical_dividends.path.with_name(
        f"{sources.historical_dividends.path.name}.receipt.json"
    )
    return ProjectExperimentSourceIdentity(
        long_form_options=sources.long_form_options.identity,
        option_compaction_manifest=_artifact_file_identity(sources.compacted_options.manifest_path),
        forecast_dividend_manifest=_artifact_file_identity(
            sources.forecast_dividends.manifest_path
        ),
        historical_dividend_output=_artifact_file_identity(sources.historical_dividends.path),
        historical_dividend_receipt=_artifact_file_identity(historical_receipt),
        underlying_history=sources.underlying_history.identity,
        nss_parameters=sources.nss_parameters.identity,
    )


def _require_sources_unchanged(sources: ProjectExperimentFeatureSources) -> None:
    verify_pinned_parquet_unchanged(sources.long_form_options)
    verify_orats_option_compaction_unchanged(sources.compacted_options)
    forecasts = load_verified_orats_dividend_projection_set(
        sources.forecast_dividends.manifest_path,
        sources.forecast_dividends.inventory_path,
        sources.forecast_dividends.output_root,
    )
    if forecasts != sources.forecast_dividends:
        msg = "forecast-dividend source changed during experiment materialization"
        raise ProjectExperimentArtifactContentError(msg)
    historical = load_verified_historical_dividend_projection(sources.historical_dividends.path)
    if historical != sources.historical_dividends:
        msg = "historical-dividend source changed during experiment materialization"
        raise ProjectExperimentArtifactContentError(msg)
    verify_pinned_parquet_unchanged(sources.underlying_history)
    verify_pinned_parquet_unchanged(sources.nss_parameters)


def _inspect_output(path: Path) -> tuple[ProjectExperimentTickerCoverage, ...]:
    try:
        scan = pl.scan_parquet(path, rechunk=False)
        if scan.collect_schema() != pl.Schema(PROJECT_EXPERIMENT_FEATURE_DTYPES):
            msg = "experiment output ordered schema is unsupported"
            raise ProjectExperimentArtifactContentError(msg, str(path))
        scalar_numeric = tuple(
            name for name, dtype in PROJECT_EXPERIMENT_FEATURE_DTYPES.items() if dtype.is_numeric()
        )
        diagnostics = scan.select(
            pl.any_horizontal(pl.all().is_null()).sum().alias("rows_with_nulls"),
            pl.struct(("ticker", "quote_date", "expiry_date", "strike"))
            .is_duplicated()
            .sum()
            .alias("duplicate_contract_keys"),
            pl.any_horizontal(*(~pl.col(name).is_finite() for name in scalar_numeric))
            .sum()
            .alias("nonfinite_scalar_rows"),
            pl.any_horizontal(
                ~pl.col("forecasted_discrete_amounts")
                .list.eval(pl.element().is_finite())
                .list.all(),
                ~pl.col("forecasted_years_to_discrete_ex_dates")
                .list.eval(pl.element().is_finite())
                .list.all(),
            )
            .sum()
            .alias("nonfinite_list_rows"),
            pl.any_horizontal(
                pl.col("forecasted_discrete_ex_dates").list.eval(pl.element().is_null()).list.any(),
                pl.col("forecasted_discrete_amounts").list.eval(pl.element().is_null()).list.any(),
                pl.col("forecasted_years_to_discrete_ex_dates")
                .list.eval(pl.element().is_null())
                .list.any(),
            )
            .sum()
            .alias("rows_with_null_list_elements"),
            pl.any_horizontal(
                pl.col("forecasted_discrete_ex_dates").list.len()
                != pl.col("forecasted_dividend_count"),
                pl.col("forecasted_discrete_amounts").list.len()
                != pl.col("forecasted_dividend_count"),
                pl.col("forecasted_years_to_discrete_ex_dates").list.len()
                != pl.col("forecasted_dividend_count"),
            )
            .sum()
            .alias("inconsistent_forecast_list_lengths"),
        ).collect()
        if diagnostics.row(0) != (0, 0, 0, 0, 0, 0):
            msg = f"experiment output contains invalid rows: {diagnostics.row(0)!r}"
            raise ProjectExperimentArtifactContentError(msg, str(path))
        grouped = (
            scan.group_by("ticker")
            .agg(
                pl.len().alias("rows"),
                (pl.col("quote_date") >= PROJECT_EXPERIMENT_DESCRIPTIVE_START_DATE)
                .sum()
                .alias("descriptive_rows"),
                (pl.col("quote_date") >= PROJECT_EXPERIMENT_POST_2015_START_DATE)
                .sum()
                .alias("post_2015_rows"),
                pl.col("quote_date").min().alias("quote_date_min"),
                pl.col("quote_date").max().alias("quote_date_max"),
            )
            .sort("ticker")
            .collect(engine="streaming")
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        msg = "experiment output is unreadable under the exact Parquet contract"
        raise ProjectExperimentArtifactContentError(msg, str(path)) from error
    coverage = tuple(
        ProjectExperimentTickerCoverage(
            ticker=ticker,
            rows=rows,
            descriptive_rows=descriptive,
            post_2015_rows=post_2015,
            quote_date_min=minimum,
            quote_date_max=maximum,
        )
        for ticker, rows, descriptive, post_2015, minimum, maximum in grouped.iter_rows()
    )
    if tuple(item.ticker for item in coverage) != PROJECT_AS_SUBMITTED_TICKERS:
        msg = "experiment output lacks the exact submitted ticker universe"
        raise ProjectExperimentArtifactContentError(msg, str(path))
    return coverage


def _require_writer_runtime(writer: ProjectFeatureWriteSpec) -> None:
    if type(writer) is not ProjectFeatureWriteSpec:
        msg = "writer must be a ProjectFeatureWriteSpec value"
        raise ProjectExperimentArtifactSpecError(msg)
    if writer.polars_version != pl.__version__:
        msg = "experiment writer Polars version disagrees with the active runtime"
        raise ProjectExperimentArtifactSpecError(msg)
    if writer.polars_thread_pool_size != pl.thread_pool_size():
        msg = "experiment writer thread-pool size disagrees with the active runtime"
        raise ProjectExperimentArtifactSpecError(msg)


def _read_manifest(path: Path) -> ProjectExperimentArtifactManifest:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "experiment artifact receipt is unreadable"
        raise ProjectExperimentArtifactContentError(msg, str(path)) from error
    return ProjectExperimentArtifactManifest.from_dict(raw)


def _artifact_file_identity(path: Path) -> ArtifactFileIdentity:
    regular = _regular_file(path, "artifact source")
    return ArtifactFileIdentity(
        name=regular.name,
        size_bytes=regular.stat().st_size,
        sha256=_file_sha256(regular),
    )


def _regular_file(path: str | Path, field: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.is_symlink() or candidate.stat().st_size <= 0:
        msg = f"{field} must be an existing nonempty non-symlink file"
        raise ProjectExperimentArtifactContentError(msg, str(candidate))
    return candidate


def _output_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        msg = "experiment output root must be a non-symlink directory"
        raise ProjectExperimentArtifactContentError(msg, str(root))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_BLOCK_BYTES):
            hasher.update(block)
    return hasher.hexdigest()


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    encoded = _canonical_json_bytes(payload) + b"\n"
    try:
        with path.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        msg = "experiment artifact receipt write failed"
        raise ProjectExperimentArtifactContentError(msg, str(path)) from error


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as source:
            os.fsync(source.fileno())
    except OSError as error:
        msg = "experiment output durability sync failed"
        raise ProjectExperimentArtifactContentError(msg, str(path)) from error


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _exact_mapping(raw: object, keys: set[str], field: str) -> dict[str, object]:
    if type(raw) is not dict or set(raw) != keys or any(type(key) is not str for key in raw):
        msg = f"{field} must contain exactly {tuple(sorted(keys))!r}"
        raise ProjectExperimentArtifactContentError(msg)
    return cast("dict[str, object]", raw)


def _is_sha256(value: str) -> bool:
    return (
        len(value) == _SHA256_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "PROJECT_EXPERIMENT_ARTIFACT_NAME",
    "PROJECT_EXPERIMENT_ARTIFACT_VERSION",
    "PROJECT_EXPERIMENT_COMPACTED_TICKERS",
    "PROJECT_EXPERIMENT_DESCRIPTIVE_START_DATE",
    "PROJECT_EXPERIMENT_LONG_FORM_TICKERS",
    "PROJECT_EXPERIMENT_OPTION_PLAN",
    "PROJECT_EXPERIMENT_POST_2015_START_DATE",
    "MaterializedProjectExperimentArtifact",
    "ProjectExperimentArtifactContentError",
    "ProjectExperimentArtifactError",
    "ProjectExperimentArtifactManifest",
    "ProjectExperimentArtifactSpecError",
    "ProjectExperimentFeatureSources",
    "ProjectExperimentOptionPlan",
    "ProjectExperimentSourceIdentity",
    "ProjectExperimentTickerCoverage",
    "load_verified_project_experiment_artifact",
    "materialize_project_experiment_artifact",
    "project_experiment_mixed_pipeline_sha256",
    "project_experiment_request_sha256",
]
