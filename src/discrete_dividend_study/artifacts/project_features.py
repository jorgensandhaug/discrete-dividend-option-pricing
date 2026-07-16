"""Content-addressed materialization of the complete project feature dataset."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal, Self, cast

import polars as pl

from discrete_dividend_study.core.parquet import (
    PinnedParquetIdentity,
    VerifiedPinnedParquet,
    verify_pinned_parquet,
    verify_pinned_parquet_unchanged,
)
from discrete_dividend_study.data.dividends.historical_projection import (
    MaterializedHistoricalDividendProjection,
    load_verified_historical_dividend_projection,
)
from discrete_dividend_study.data.pipeline import (
    PROJECT_AS_SUBMITTED_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_TICKERS,
    build_project_quote_cohort,
    enrich_project_quote_cohort,
    project_feature_pipeline_sha256,
    project_feature_schema_sha256,
)
from discrete_dividend_study.data.providers.orats.dividend_set import (
    VerifiedOratsDividendProjectionSet,
    load_verified_orats_dividend_projection_set,
)
from discrete_dividend_study.data.providers.orats.option_compaction import (
    VerifiedOratsOptionCompaction,
    load_verified_orats_option_compaction,
    verify_orats_option_compaction_unchanged,
)
from discrete_dividend_study.data.rates import NSS_PARAMETER_DTYPES
from discrete_dividend_study.data.underlyings import UNDERLYING_HISTORY_DTYPES

PROJECT_FEATURE_ARTIFACT_VERSION: Final = 2
PROJECT_FEATURE_ARTIFACT_RECEIPT_VERSION: Final = 2
PROJECT_FEATURE_ARTIFACT_COMPRESSION: Final = "zstd"
PROJECT_FEATURE_ARTIFACT_COMPRESSION_LEVEL: Final = 3
PROJECT_FEATURE_ARTIFACT_ROW_GROUP_SIZE: Final = 131_072
PROJECT_FEATURE_ARTIFACT_BLOCK_BYTES: Final = 8 * 1024 * 1024
PROJECT_FEATURE_ARTIFACT_NAME: Final = "project_features.parquet"
PROJECT_FEATURE_OPTION_START_DATE: Final = date(2011, 1, 1)
PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE: Final = date(2024, 9, 28)
_SHA256_LENGTH: Final = 64
_QUOTE_STAGE_COMPRESSION: Final = "uncompressed"
_QUOTE_STAGE_ROW_GROUP_SIZE: Final = 131_072


class ProjectFeatureArtifactError(RuntimeError):
    """Base class for an invalid feature materialization or stored artifact."""


@dataclass(slots=True)
class ProjectFeatureArtifactContentError(ProjectFeatureArtifactError):
    """A source, request, output, or receipt violates the closed artifact contract."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"project feature artifact failed{location}: {self.reason}"


@dataclass(slots=True)
class ProjectFeatureArtifactSpecError(ProjectFeatureArtifactError):
    """A physical writer setting is unsupported by the frozen artifact contract."""

    reason: str

    def __str__(self) -> str:
        """Describe the invalid static setting."""
        return self.reason


@dataclass(frozen=True, slots=True)
class ArtifactFileIdentity:
    """Relocatable byte identity for one upstream manifest, receipt, or output."""

    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """Reject unsafe names, coerced counts, and malformed hashes."""
        if type(self.name) is not str or not self.name or Path(self.name).name != self.name:
            msg = "artifact source filename is malformed"
            raise ProjectFeatureArtifactContentError(msg)
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            msg = "artifact source byte count must be a positive plain int"
            raise ProjectFeatureArtifactContentError(msg)
        if type(self.sha256) is not str or not _is_sha256(self.sha256):
            msg = "artifact source SHA-256 is malformed"
            raise ProjectFeatureArtifactContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical file identity mapping."""
        return {"name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse one exact file identity without coercion."""
        mapping = _exact_mapping(raw, {"name", "sha256", "size_bytes"}, "artifact file identity")
        return cls(
            name=cast("str", mapping["name"]),
            size_bytes=cast("int", mapping["size_bytes"]),
            sha256=cast("str", mapping["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ProjectFeatureSourceIdentity:
    """Complete relocatable identities of the five canonical feature sources."""

    option_compaction_manifest: ArtifactFileIdentity
    forecast_dividend_manifest: ArtifactFileIdentity
    historical_dividend_output: ArtifactFileIdentity
    historical_dividend_receipt: ArtifactFileIdentity
    underlying_history: PinnedParquetIdentity
    nss_parameters: PinnedParquetIdentity

    def __post_init__(self) -> None:
        """Reject fabricated values of the wrong identity type."""
        if any(
            type(value) is not ArtifactFileIdentity
            for value in (
                self.option_compaction_manifest,
                self.forecast_dividend_manifest,
                self.historical_dividend_output,
                self.historical_dividend_receipt,
            )
        ):
            msg = "feature source file identities are malformed"
            raise ProjectFeatureArtifactContentError(msg)
        if (
            type(self.underlying_history) is not PinnedParquetIdentity
            or type(self.nss_parameters) is not PinnedParquetIdentity
        ):
            msg = "feature support Parquet identities are malformed"
            raise ProjectFeatureArtifactContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical closed source mapping."""
        return {
            "forecast_dividend_manifest": self.forecast_dividend_manifest.as_dict(),
            "historical_dividend_output": self.historical_dividend_output.as_dict(),
            "historical_dividend_receipt": self.historical_dividend_receipt.as_dict(),
            "nss_parameters": self.nss_parameters.as_dict(),
            "option_compaction_manifest": self.option_compaction_manifest.as_dict(),
            "underlying_history": self.underlying_history.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse all six exact source artifacts and revalidate their identities."""
        mapping = _exact_mapping(
            raw,
            {
                "forecast_dividend_manifest",
                "historical_dividend_output",
                "historical_dividend_receipt",
                "nss_parameters",
                "option_compaction_manifest",
                "underlying_history",
            },
            "project feature source identity",
        )
        return cls(
            option_compaction_manifest=ArtifactFileIdentity.from_dict(
                mapping["option_compaction_manifest"],
            ),
            forecast_dividend_manifest=ArtifactFileIdentity.from_dict(
                mapping["forecast_dividend_manifest"],
            ),
            historical_dividend_output=ArtifactFileIdentity.from_dict(
                mapping["historical_dividend_output"],
            ),
            historical_dividend_receipt=ArtifactFileIdentity.from_dict(
                mapping["historical_dividend_receipt"],
            ),
            underlying_history=PinnedParquetIdentity.from_dict(mapping["underlying_history"]),
            nss_parameters=PinnedParquetIdentity.from_dict(mapping["nss_parameters"]),
        )


@dataclass(frozen=True, slots=True)
class ProjectFeatureSources:
    """Verified handles required by one project feature materialization."""

    options: VerifiedOratsOptionCompaction
    forecast_dividends: VerifiedOratsDividendProjectionSet
    historical_dividends: MaterializedHistoricalDividendProjection
    underlying_history: VerifiedPinnedParquet
    nss_parameters: VerifiedPinnedParquet

    def __post_init__(self) -> None:
        """Require loader-produced handles rather than bare frames or inferred paths."""
        expected = (
            (self.options, VerifiedOratsOptionCompaction),
            (self.forecast_dividends, VerifiedOratsDividendProjectionSet),
            (self.historical_dividends, MaterializedHistoricalDividendProjection),
            (self.underlying_history, VerifiedPinnedParquet),
            (self.nss_parameters, VerifiedPinnedParquet),
        )
        if any(type(value) is not kind for value, kind in expected):
            msg = "project feature sources must be exact verified handle types"
            raise ProjectFeatureArtifactContentError(msg)


@dataclass(frozen=True, slots=True)
class ProjectFeatureWriteSpec:
    """Exact physical identity of the feature Parquet sink."""

    polars_version: str = pl.__version__
    polars_thread_pool_size: int = pl.thread_pool_size()
    compression: Literal["zstd"] = PROJECT_FEATURE_ARTIFACT_COMPRESSION
    compression_level: int = PROJECT_FEATURE_ARTIFACT_COMPRESSION_LEVEL
    row_group_size: int = PROJECT_FEATURE_ARTIFACT_ROW_GROUP_SIZE
    statistics: bool = True
    maintain_order: bool = True
    sync_on_close: Literal["all"] = "all"
    engine: Literal["streaming"] = "streaming"
    quote_stage_compression: Literal["uncompressed"] = _QUOTE_STAGE_COMPRESSION
    quote_stage_row_group_size: int = _QUOTE_STAGE_ROW_GROUP_SIZE
    quote_stage_statistics: bool = False
    quote_stage_maintain_order: bool = True
    schema_version: int = PROJECT_FEATURE_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        """Reject runtime substitution or a changed physical output contract."""
        expected = (
            type(self.schema_version) is int
            and self.schema_version == PROJECT_FEATURE_ARTIFACT_VERSION,
            type(self.polars_version) is str and bool(self.polars_version),
            type(self.polars_thread_pool_size) is int and self.polars_thread_pool_size > 0,
            self.compression == PROJECT_FEATURE_ARTIFACT_COMPRESSION,
            type(self.compression_level) is int
            and self.compression_level == PROJECT_FEATURE_ARTIFACT_COMPRESSION_LEVEL,
            type(self.row_group_size) is int
            and self.row_group_size == PROJECT_FEATURE_ARTIFACT_ROW_GROUP_SIZE,
            type(self.statistics) is bool and self.statistics,
            type(self.maintain_order) is bool and self.maintain_order,
            self.sync_on_close == "all",
            self.engine == "streaming",
            self.quote_stage_compression == _QUOTE_STAGE_COMPRESSION,
            type(self.quote_stage_row_group_size) is int
            and self.quote_stage_row_group_size == _QUOTE_STAGE_ROW_GROUP_SIZE,
            type(self.quote_stage_statistics) is bool and not self.quote_stage_statistics,
            type(self.quote_stage_maintain_order) is bool and self.quote_stage_maintain_order,
        )
        if not all(expected):
            msg = "feature writer must use the frozen durable streaming Parquet contract"
            raise ProjectFeatureArtifactSpecError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical physical writer mapping."""
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
            "engine": self.engine,
            "maintain_order": self.maintain_order,
            "polars_version": self.polars_version,
            "polars_thread_pool_size": self.polars_thread_pool_size,
            "quote_stage_compression": self.quote_stage_compression,
            "quote_stage_maintain_order": self.quote_stage_maintain_order,
            "quote_stage_row_group_size": self.quote_stage_row_group_size,
            "quote_stage_statistics": self.quote_stage_statistics,
            "row_group_size": self.row_group_size,
            "schema_version": self.schema_version,
            "statistics": self.statistics,
            "sync_on_close": self.sync_on_close,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse an exact physical writer mapping without coercion."""
        mapping = _exact_mapping(
            raw,
            {
                "compression",
                "compression_level",
                "engine",
                "maintain_order",
                "polars_version",
                "polars_thread_pool_size",
                "quote_stage_compression",
                "quote_stage_maintain_order",
                "quote_stage_row_group_size",
                "quote_stage_statistics",
                "row_group_size",
                "schema_version",
                "statistics",
                "sync_on_close",
            },
            "project feature writer",
        )
        return cls(
            polars_version=cast("str", mapping["polars_version"]),
            polars_thread_pool_size=cast("int", mapping["polars_thread_pool_size"]),
            compression=cast("Literal['zstd']", mapping["compression"]),
            compression_level=cast("int", mapping["compression_level"]),
            quote_stage_compression=cast(
                "Literal['uncompressed']", mapping["quote_stage_compression"]
            ),
            quote_stage_row_group_size=cast("int", mapping["quote_stage_row_group_size"]),
            quote_stage_statistics=cast("bool", mapping["quote_stage_statistics"]),
            quote_stage_maintain_order=cast("bool", mapping["quote_stage_maintain_order"]),
            row_group_size=cast("int", mapping["row_group_size"]),
            statistics=cast("bool", mapping["statistics"]),
            maintain_order=cast("bool", mapping["maintain_order"]),
            sync_on_close=cast("Literal['all']", mapping["sync_on_close"]),
            engine=cast("Literal['streaming']", mapping["engine"]),
            schema_version=cast("int", mapping["schema_version"]),
        )


PROJECT_FEATURE_WRITE_SPEC: Final = ProjectFeatureWriteSpec()


@dataclass(frozen=True, slots=True)
class ProjectFeatureArtifactManifest:
    """Complete request, source, physical, and output identity."""

    request_sha256: str
    sources: ProjectFeatureSourceIdentity
    pipeline_sha256: str
    feature_schema_sha256: str
    writer: ProjectFeatureWriteSpec
    output_name: str
    output_rows: int
    output_bytes: int
    output_sha256: str
    quote_date_min: date
    quote_date_max: date
    observed_tickers: tuple[str, ...]
    schema_version: int = PROJECT_FEATURE_ARTIFACT_RECEIPT_VERSION

    def __post_init__(self) -> None:
        """Validate the request root and all output summary fields."""
        self._validate_request()
        self._validate_output()

    def _validate_request(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != PROJECT_FEATURE_ARTIFACT_RECEIPT_VERSION
        ):
            msg = "unsupported project feature receipt version"
            raise ProjectFeatureArtifactContentError(msg)
        if (
            type(self.sources) is not ProjectFeatureSourceIdentity
            or type(self.writer) is not ProjectFeatureWriteSpec
        ):
            msg = "project feature request sources or writer are malformed"
            raise ProjectFeatureArtifactContentError(msg)
        for value in (self.request_sha256, self.pipeline_sha256, self.feature_schema_sha256):
            if type(value) is not str or not _is_sha256(value):
                msg = "project feature request SHA-256 is malformed"
                raise ProjectFeatureArtifactContentError(msg)
        if self.feature_schema_sha256 != project_feature_schema_sha256():
            msg = "project feature schema identity is unsupported"
            raise ProjectFeatureArtifactContentError(msg)
        if self.request_sha256 != _request_sha256(
            self.sources,
            self.writer,
            self.pipeline_sha256,
            self.feature_schema_sha256,
        ):
            msg = "project feature request hash disagrees with its sources and writer"
            raise ProjectFeatureArtifactContentError(msg)

    def _validate_output(self) -> None:
        if self.output_name != PROJECT_FEATURE_ARTIFACT_NAME:
            msg = "project feature output name is unsupported"
            raise ProjectFeatureArtifactContentError(msg)
        if type(self.output_rows) is not int or type(self.output_bytes) is not int:
            msg = "project feature output counts must be plain ints"
            raise ProjectFeatureArtifactContentError(msg)
        if (
            self.output_rows <= 0
            or self.output_bytes <= 0
            or type(self.output_sha256) is not str
            or not _is_sha256(self.output_sha256)
        ):
            msg = "project feature output identity is malformed"
            raise ProjectFeatureArtifactContentError(msg)
        if (
            type(self.quote_date_min) is not date
            or type(self.quote_date_max) is not date
            or self.quote_date_min > self.quote_date_max
        ):
            msg = "project feature quote-date bounds are malformed"
            raise ProjectFeatureArtifactContentError(msg)
        if (
            self.quote_date_min < PROJECT_FEATURE_OPTION_START_DATE
            or self.quote_date_max >= PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE
        ):
            msg = "project feature quote dates exceed the exact project horizon"
            raise ProjectFeatureArtifactContentError(msg)
        if self.observed_tickers != PROJECT_AS_SUBMITTED_TICKERS:
            msg = "project feature output must contain the exact submitted ticker universe"
            raise ProjectFeatureArtifactContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical artifact receipt mapping."""
        return {
            "feature_schema_sha256": self.feature_schema_sha256,
            "observed_tickers": list(self.observed_tickers),
            "output_bytes": self.output_bytes,
            "output_name": self.output_name,
            "output_rows": self.output_rows,
            "output_sha256": self.output_sha256,
            "pipeline_sha256": self.pipeline_sha256,
            "quote_date_max": self.quote_date_max.isoformat(),
            "quote_date_min": self.quote_date_min.isoformat(),
            "request_sha256": self.request_sha256,
            "schema_version": self.schema_version,
            "sources": self.sources.as_dict(),
            "writer": self.writer.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully revalidate one exact artifact receipt."""
        mapping = _exact_mapping(
            raw,
            {
                "feature_schema_sha256",
                "observed_tickers",
                "output_bytes",
                "output_name",
                "output_rows",
                "output_sha256",
                "pipeline_sha256",
                "quote_date_max",
                "quote_date_min",
                "request_sha256",
                "schema_version",
                "sources",
                "writer",
            },
            "project feature artifact receipt",
        )
        string_fields = (
            "feature_schema_sha256",
            "output_name",
            "output_sha256",
            "pipeline_sha256",
            "quote_date_max",
            "quote_date_min",
            "request_sha256",
        )
        integer_fields = ("output_bytes", "output_rows", "schema_version")
        observed_raw = mapping["observed_tickers"]
        if (
            any(type(mapping[name]) is not str for name in string_fields)
            or any(type(mapping[name]) is not int for name in integer_fields)
            or type(observed_raw) is not list
            or any(type(value) is not str for value in observed_raw)
        ):
            msg = "project feature artifact receipt fields are malformed"
            raise ProjectFeatureArtifactContentError(msg)
        try:
            quote_date_min = date.fromisoformat(cast("str", mapping["quote_date_min"]))
            quote_date_max = date.fromisoformat(cast("str", mapping["quote_date_max"]))
        except ValueError as error:
            msg = "project feature artifact receipt dates are malformed"
            raise ProjectFeatureArtifactContentError(msg) from error
        return cls(
            request_sha256=cast("str", mapping["request_sha256"]),
            sources=ProjectFeatureSourceIdentity.from_dict(mapping["sources"]),
            pipeline_sha256=cast("str", mapping["pipeline_sha256"]),
            feature_schema_sha256=cast("str", mapping["feature_schema_sha256"]),
            writer=ProjectFeatureWriteSpec.from_dict(mapping["writer"]),
            output_name=cast("str", mapping["output_name"]),
            output_rows=cast("int", mapping["output_rows"]),
            output_bytes=cast("int", mapping["output_bytes"]),
            output_sha256=cast("str", mapping["output_sha256"]),
            quote_date_min=quote_date_min,
            quote_date_max=quote_date_max,
            observed_tickers=tuple(cast("list[str]", observed_raw)),
            schema_version=cast("int", mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class MaterializedProjectFeatureArtifact:
    """One verified content-addressed feature Parquet and receipt."""

    path: Path
    receipt_path: Path
    manifest: ProjectFeatureArtifactManifest

    def scan(self) -> pl.LazyFrame:
        """Lazily scan the exact verified feature artifact."""
        return pl.scan_parquet(self.path, rechunk=False)


def project_feature_request_sha256(
    sources: ProjectFeatureSourceIdentity,
    writer: ProjectFeatureWriteSpec,
) -> str:
    """Hash every pre-materialization source, transformation, and writer identity."""
    return _request_sha256(
        sources,
        writer,
        project_feature_pipeline_sha256(),
        project_feature_schema_sha256(),
    )


def _request_sha256(
    sources: ProjectFeatureSourceIdentity,
    writer: ProjectFeatureWriteSpec,
    pipeline_sha256: str,
    feature_schema_sha256: str,
) -> str:
    payload = {
        "feature_schema_sha256": feature_schema_sha256,
        "pipeline_sha256": pipeline_sha256,
        "sources": sources.as_dict(),
        "writer": writer.as_dict(),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def materialize_project_feature_artifact(
    sources: ProjectFeatureSources,
    output_root: str | Path,
    *,
    writer: ProjectFeatureWriteSpec = PROJECT_FEATURE_WRITE_SPEC,
) -> MaterializedProjectFeatureArtifact:
    """Verify five sources, stream the complete project pipeline, and bind one receipt."""
    _require_writer_runtime(writer)
    verified = _require_verified_sources(sources)
    source_identity = _source_identity(verified)
    request_sha256 = project_feature_request_sha256(source_identity, writer)
    root = _output_root(output_root)
    output_directory = root / request_sha256
    output = output_directory / PROJECT_FEATURE_ARTIFACT_NAME
    receipt_path = output.with_name(f"{output.name}.receipt.json")
    if output.exists() or receipt_path.exists():
        existing = load_verified_project_feature_artifact(output)
        if existing.manifest.sources != source_identity or existing.manifest.writer != writer:
            msg = "existing feature artifact disagrees with the requested sources or writer"
            raise ProjectFeatureArtifactContentError(msg, str(output))
        return existing
    output_directory.mkdir(parents=True, exist_ok=False)
    if output_directory.is_symlink():
        msg = "feature artifact request directory must not be a symlink"
        raise ProjectFeatureArtifactContentError(msg, str(output_directory))
    partial = output.with_name(f".{output.name}.part")
    partial_receipt = receipt_path.with_name(f".{receipt_path.name}.part")
    quote_stage = output.with_name(".project_quote_cohort.parquet.part")
    try:
        build_project_quote_cohort(
            verified.options.scan(),
            verified.nss_parameters.scan(),
        ).sink_parquet(
            quote_stage,
            compression=writer.quote_stage_compression,
            statistics=writer.quote_stage_statistics,
            row_group_size=writer.quote_stage_row_group_size,
            maintain_order=writer.quote_stage_maintain_order,
            sync_on_close=None,
            mkdir=False,
            engine=writer.engine,
        )
        feature_frame = _feature_frame(
            verified,
            pl.scan_parquet(quote_stage, rechunk=False),
        )
        feature_frame.sink_parquet(
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
        summary = _inspect_output(partial)
        manifest = ProjectFeatureArtifactManifest(
            request_sha256=request_sha256,
            sources=source_identity,
            pipeline_sha256=project_feature_pipeline_sha256(),
            feature_schema_sha256=project_feature_schema_sha256(),
            writer=writer,
            output_name=PROJECT_FEATURE_ARTIFACT_NAME,
            output_rows=summary.rows,
            output_bytes=partial.stat().st_size,
            output_sha256=_file_sha256(partial),
            quote_date_min=summary.quote_date_min,
            quote_date_max=summary.quote_date_max,
            observed_tickers=summary.tickers,
        )
        _write_new_json(partial_receipt, manifest.as_dict())
        partial.replace(output)
        try:
            partial_receipt.replace(receipt_path)
        except BaseException:
            output.replace(partial)
            raise
    except pl.exceptions.PolarsError as error:
        msg = "streaming project feature materialization failed"
        raise ProjectFeatureArtifactContentError(msg, str(output)) from error
    finally:
        partial.unlink(missing_ok=True)
        partial_receipt.unlink(missing_ok=True)
        quote_stage.unlink(missing_ok=True)
        if output_directory.is_dir() and not any(output_directory.iterdir()):
            output_directory.rmdir()
    return load_verified_project_feature_artifact(output)


def load_verified_project_feature_artifact(
    path: str | Path,
) -> MaterializedProjectFeatureArtifact:
    """Load a detached feature artifact and verify its receipt, bytes, schema, and summary."""
    output = _regular_file(path, "project feature output")
    receipt_path = _regular_file(
        output.with_name(f"{output.name}.receipt.json"),
        "project feature receipt",
    )
    manifest = _read_manifest(receipt_path)
    if output.parent.name != manifest.request_sha256:
        msg = "feature artifact request directory disagrees with its receipt"
        raise ProjectFeatureArtifactContentError(msg, str(output.parent))
    if output.name != manifest.output_name or output.stat().st_size != manifest.output_bytes:
        msg = "feature output name or byte count disagrees with its receipt"
        raise ProjectFeatureArtifactContentError(msg, str(output))
    if _file_sha256(output) != manifest.output_sha256:
        msg = "feature output bytes disagree with its receipt SHA-256"
        raise ProjectFeatureArtifactContentError(msg, str(output))
    summary = _inspect_output(output)
    if (
        summary.rows != manifest.output_rows
        or summary.quote_date_min != manifest.quote_date_min
        or summary.quote_date_max != manifest.quote_date_max
        or summary.tickers != manifest.observed_tickers
    ):
        msg = "feature output content summary disagrees with its receipt"
        raise ProjectFeatureArtifactContentError(msg, str(output))
    return MaterializedProjectFeatureArtifact(
        path=output,
        receipt_path=receipt_path,
        manifest=manifest,
    )


@dataclass(frozen=True, slots=True)
class _OutputSummary:
    rows: int
    quote_date_min: date
    quote_date_max: date
    tickers: tuple[str, ...]


def _require_verified_sources(sources: ProjectFeatureSources) -> ProjectFeatureSources:
    if type(sources) is not ProjectFeatureSources:
        msg = "sources must be a ProjectFeatureSources value"
        raise ProjectFeatureArtifactContentError(msg)
    options = load_verified_orats_option_compaction(
        sources.options.manifest_path,
        sources.options.output_root,
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
    verified = ProjectFeatureSources(options, forecasts, historical, underlying, nss)
    if verified != sources:
        msg = "supplied feature handles disagree with their verified source artifacts"
        raise ProjectFeatureArtifactContentError(msg)
    _validate_source_universes(verified)
    return verified


def _validate_source_universes(sources: ProjectFeatureSources) -> None:
    if sources.options.manifest.projected_tickers != PROJECT_AS_SUBMITTED_TICKERS:
        msg = "option compaction source must declare the exact project ticker universe"
        raise ProjectFeatureArtifactContentError(msg, str(sources.options.manifest_path))
    option_spec = sources.options.manifest.compaction_spec
    if (
        option_spec.start_date != PROJECT_FEATURE_OPTION_START_DATE
        or option_spec.end_date_exclusive != PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE
    ):
        msg = "option compaction source must cover the exact project quote horizon"
        raise ProjectFeatureArtifactContentError(msg, str(sources.options.manifest_path))
    if sources.forecast_dividends.manifest.projection_spec.tickers != (
        PROJECT_AS_SUBMITTED_TICKERS
    ):
        msg = "forecast-dividend source must declare the exact project ticker universe"
        raise ProjectFeatureArtifactContentError(
            msg,
            str(sources.forecast_dividends.manifest_path),
        )
    if sources.historical_dividends.receipt.projection_spec.tickers != (
        PROJECT_AS_SUBMITTED_TICKERS
    ):
        msg = "historical-dividend source must be projected to the exact project universe"
        raise ProjectFeatureArtifactContentError(
            msg,
            str(sources.historical_dividends.path),
        )
    if sources.historical_dividends.receipt.projection_spec.maximum_ex_date != (
        PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE - date.resolution
    ):
        msg = "historical-dividend source must use the exact project ex-date horizon"
        raise ProjectFeatureArtifactContentError(msg, str(sources.historical_dividends.path))


def _source_identity(sources: ProjectFeatureSources) -> ProjectFeatureSourceIdentity:
    historical_receipt = sources.historical_dividends.path.with_name(
        f"{sources.historical_dividends.path.name}.receipt.json",
    )
    return ProjectFeatureSourceIdentity(
        option_compaction_manifest=_artifact_file_identity(sources.options.manifest_path),
        forecast_dividend_manifest=_artifact_file_identity(
            sources.forecast_dividends.manifest_path,
        ),
        historical_dividend_output=_artifact_file_identity(sources.historical_dividends.path),
        historical_dividend_receipt=_artifact_file_identity(historical_receipt),
        underlying_history=sources.underlying_history.identity,
        nss_parameters=sources.nss_parameters.identity,
    )


def _feature_frame(
    sources: ProjectFeatureSources,
    quote_cohort: pl.LazyFrame,
) -> pl.LazyFrame:
    return enrich_project_quote_cohort(
        quote_cohort,
        sources.underlying_history.scan(),
        sources.forecast_dividends.scan(),
        pl.scan_parquet(sources.historical_dividends.path, rechunk=False),
        sources.nss_parameters.scan(),
    )


def _require_sources_unchanged(sources: ProjectFeatureSources) -> None:
    verify_orats_option_compaction_unchanged(sources.options)
    forecasts = load_verified_orats_dividend_projection_set(
        sources.forecast_dividends.manifest_path,
        sources.forecast_dividends.inventory_path,
        sources.forecast_dividends.output_root,
    )
    if forecasts != sources.forecast_dividends:
        msg = "forecast-dividend source changed during feature materialization"
        raise ProjectFeatureArtifactContentError(msg)
    historical = load_verified_historical_dividend_projection(sources.historical_dividends.path)
    if historical != sources.historical_dividends:
        msg = "historical-dividend source changed during feature materialization"
        raise ProjectFeatureArtifactContentError(msg)
    verify_pinned_parquet_unchanged(sources.underlying_history)
    verify_pinned_parquet_unchanged(sources.nss_parameters)


def _inspect_output(path: Path) -> _OutputSummary:
    try:
        scan = pl.scan_parquet(path, rechunk=False)
        if scan.collect_schema() != pl.Schema(PROJECT_AS_SUBMITTED_FEATURE_DTYPES):
            msg = "feature output ordered schema is unsupported"
            raise ProjectFeatureArtifactContentError(msg, str(path))
        row = (
            scan.select(
                pl.len().alias("rows"),
                pl.any_horizontal(pl.all().is_null()).sum().alias("rows_with_nulls"),
                pl.col("quote_date").min().alias("quote_date_min"),
                pl.col("quote_date").max().alias("quote_date_max"),
                pl.col("ticker").unique().sort().implode().alias("tickers"),
            )
            .collect(engine="streaming")
            .row(0, named=True)
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        msg = "feature output is not readable under the exact Parquet contract"
        raise ProjectFeatureArtifactContentError(msg, str(path)) from error
    rows = row["rows"]
    rows_with_nulls = row["rows_with_nulls"]
    quote_date_min = row["quote_date_min"]
    quote_date_max = row["quote_date_max"]
    tickers = row["tickers"]
    if (
        type(rows) is not int
        or rows <= 0
        or type(rows_with_nulls) is not int
        or rows_with_nulls != 0
        or type(quote_date_min) is not date
        or type(quote_date_max) is not date
        or type(tickers) is not list
        or any(type(value) is not str for value in tickers)
    ):
        msg = "feature output aggregate summary is invalid"
        raise ProjectFeatureArtifactContentError(msg, str(path))
    return _OutputSummary(
        rows=rows,
        quote_date_min=quote_date_min,
        quote_date_max=quote_date_max,
        tickers=tuple(cast("list[str]", tickers)),
    )


def _artifact_file_identity(path: Path) -> ArtifactFileIdentity:
    regular = _regular_file(path, "upstream artifact")
    return ArtifactFileIdentity(
        name=regular.name,
        size_bytes=regular.stat().st_size,
        sha256=_file_sha256(regular),
    )


def _require_writer_runtime(writer: ProjectFeatureWriteSpec) -> None:
    if writer.polars_version != pl.__version__:
        msg = (
            "feature writer Polars version disagrees with the active runtime: "
            f"requested {writer.polars_version}, active {pl.__version__}"
        )
        raise ProjectFeatureArtifactSpecError(msg)
    if writer.polars_thread_pool_size != pl.thread_pool_size():
        msg = (
            "feature writer Polars thread-pool size disagrees with the active runtime: "
            f"requested {writer.polars_thread_pool_size}, active {pl.thread_pool_size()}"
        )
        raise ProjectFeatureArtifactSpecError(msg)


def _output_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        msg = "feature output root must be absent or a non-symlink directory"
        raise ProjectFeatureArtifactContentError(msg, str(root))
    if not root.parent.is_dir() or root.parent.is_symlink():
        msg = "feature output parent must be an existing non-symlink directory"
        raise ProjectFeatureArtifactContentError(msg, str(root.parent))
    root.mkdir(exist_ok=True)
    return root


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.is_symlink():
        msg = f"{label} must be an existing non-symlink file"
        raise ProjectFeatureArtifactContentError(msg, str(candidate))
    return candidate


def _read_manifest(path: Path) -> ProjectFeatureArtifactManifest:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "project feature receipt is unreadable"
        raise ProjectFeatureArtifactContentError(msg, str(path)) from error
    return ProjectFeatureArtifactManifest.from_dict(raw)


def _write_new_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        msg = "project feature receipt already exists"
        raise ProjectFeatureArtifactContentError(msg, str(path))
    payload = _canonical_json_bytes(value) + b"\n"
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(PROJECT_FEATURE_ARTIFACT_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _exact_mapping(raw: object, expected: set[str], label: str) -> dict[str, object]:
    if type(raw) is not dict:
        msg = f"{label} must be a JSON object"
        raise ProjectFeatureArtifactContentError(msg)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected or any(type(key) is not str for key in mapping):
        msg = f"{label} keys must be exactly {sorted(expected)!r}"
        raise ProjectFeatureArtifactContentError(msg)
    return cast("dict[str, object]", mapping)


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "PROJECT_FEATURE_ARTIFACT_BLOCK_BYTES",
    "PROJECT_FEATURE_ARTIFACT_COMPRESSION",
    "PROJECT_FEATURE_ARTIFACT_COMPRESSION_LEVEL",
    "PROJECT_FEATURE_ARTIFACT_NAME",
    "PROJECT_FEATURE_ARTIFACT_RECEIPT_VERSION",
    "PROJECT_FEATURE_ARTIFACT_ROW_GROUP_SIZE",
    "PROJECT_FEATURE_ARTIFACT_VERSION",
    "PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE",
    "PROJECT_FEATURE_OPTION_START_DATE",
    "PROJECT_FEATURE_WRITE_SPEC",
    "ArtifactFileIdentity",
    "MaterializedProjectFeatureArtifact",
    "ProjectFeatureArtifactContentError",
    "ProjectFeatureArtifactError",
    "ProjectFeatureArtifactManifest",
    "ProjectFeatureArtifactSpecError",
    "ProjectFeatureSourceIdentity",
    "ProjectFeatureSources",
    "ProjectFeatureWriteSpec",
    "load_verified_project_feature_artifact",
    "materialize_project_feature_artifact",
    "project_feature_request_sha256",
]
