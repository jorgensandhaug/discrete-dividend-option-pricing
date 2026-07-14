"""End-to-end tests for the closed project feature artifact."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import pytest

from discrete_dividend_study import (
    HISTORICAL_DIVIDEND_SOURCE_DTYPES,
    NSS_PARAMETER_DTYPES,
    ORATS_DIVIDEND_ARCHIVE_BASE_URL,
    PROJECT_AS_SUBMITTED_FEATURE_DTYPES,
    PROJECT_AS_SUBMITTED_TICKERS,
    PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE,
    PROJECT_FEATURE_OPTION_START_DATE,
    UNDERLYING_HISTORY_DTYPES,
    HistoricalDividendProjectionSpec,
    MaterializedHistoricalDividendProjection,
    OratsArchiveEntry,
    OratsArchiveInventory,
    OratsArchiveReceipt,
    OratsDividendArchiveEntry,
    OratsDividendArchiveInventory,
    OratsDividendArchiveReceipt,
    OratsDividendProjectionSpec,
    OratsOptionProjectionSpec,
    PinnedParquetContentError,
    VerifiedOratsArchive,
    VerifiedOratsDividendArchive,
    VerifiedOratsDividendProjectionSet,
    VerifiedPinnedParquet,
    archive_entries_sha256,
    materialize_historical_dividend_projection,
    materialize_orats_dividend_projection,
    materialize_orats_option_projection,
    orats_archive_name,
    orats_dividend_archive_entries_sha256,
    orats_dividend_archive_name,
    orats_dividend_archive_url,
    orats_remote_directory,
    verify_pinned_parquet,
    write_orats_archive_inventory,
    write_orats_dividend_archive_inventory,
    write_orats_dividend_projection_set_manifest,
    write_orats_option_projection_set_manifest,
)
from discrete_dividend_study.cli import main
from discrete_dividend_study.orats_option_compaction import (
    OratsOptionCompactionSpec,
    VerifiedOratsOptionCompaction,
    materialize_orats_option_compaction,
)
from discrete_dividend_study.project_feature_artifact import (
    ProjectFeatureArtifactContentError,
    ProjectFeatureArtifactManifest,
    ProjectFeatureArtifactSpecError,
    ProjectFeatureSources,
    ProjectFeatureWriteSpec,
    load_verified_project_feature_artifact,
    materialize_project_feature_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path

INVENTORY_FIRST_DATE = date(2010, 12, 31)
QUOTE_DATE = date(2024, 9, 27)
EXPIRY_DATE = date(2024, 11, 1)
FORECAST_EX_DATE = date(2024, 10, 15)
HISTORICAL_EX_DATE = date(2024, 9, 1)
HISTORY_ROWS = 430
OPTION_HEADER = (
    "ticker,trade_date,expirDate,strike,cAskPx,cBidPx,pAskPx,pBidPx,stkPx,"
    "cVolu,pVolu,cOi,pOi,cValue,pValue,cMidIv,pMidIv,yte,iRate\n"
)


def _option_compaction(
    tmp_path: Path,
    *,
    start_date: date = PROJECT_FEATURE_OPTION_START_DATE,
    end_date_exclusive: date = PROJECT_FEATURE_OPTION_END_DATE_EXCLUSIVE,
) -> VerifiedOratsOptionCompaction:
    raw = tmp_path / "option-raw"
    raw.mkdir()
    entries: list[OratsArchiveEntry] = []
    sources: list[VerifiedOratsArchive] = []
    for trade_date in (INVENTORY_FIRST_DATE, QUOTE_DATE):
        archive_path = raw / orats_archive_name(trade_date)
        expiry_date = EXPIRY_DATE if trade_date == QUOTE_DATE else date(2011, 1, 31)
        rows = "".join(
            f"{ticker},{trade_date:%m/%d/%Y},{expiry_date:%m/%d/%Y},"
            "80,21,20.5,1,0.9,100,10,5,100,50,20.8,0.95,0.2,0.21,0.0849,0.04\n"
            for ticker in PROJECT_AS_SUBMITTED_TICKERS
        )
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                f"ORATS_SMV_Strikes_{trade_date:%Y%m%d}.csv",
                OPTION_HEADER + rows,
            )
        payload = archive_path.read_bytes()
        entry = OratsArchiveEntry(
            trade_date=trade_date,
            remote_path=f"{orats_remote_directory(trade_date.year)}/{archive_path.name}",
            size_bytes=len(payload),
            modified_at_utc=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        )
        entries.append(entry)
        sources.append(
            VerifiedOratsArchive(
                path=archive_path,
                receipt=OratsArchiveReceipt(
                    entry=entry,
                    local_name=archive_path.name,
                    sha256=hashlib.sha256(payload).hexdigest(),
                ),
            ),
        )
    frozen_entries = tuple(entries)
    inventory = OratsArchiveInventory(
        host="de1.hostedftp.com",
        start_date=INVENTORY_FIRST_DATE,
        end_date=QUOTE_DATE,
        observed_at_utc=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
        entries=frozen_entries,
        entries_sha256=archive_entries_sha256(frozen_entries),
    )
    inventory_path = tmp_path / "option-inventory.json"
    write_orats_archive_inventory(inventory_path, inventory)
    daily_root = tmp_path / "option-daily"
    projection_spec = OratsOptionProjectionSpec(tickers=PROJECT_AS_SUBMITTED_TICKERS)
    projections = tuple(
        materialize_orats_option_projection(source, projection_spec, daily_root, tmp_path)
        for source in sources
    )
    daily_set = write_orats_option_projection_set_manifest(
        tmp_path / "option-daily-set.json",
        inventory_path,
        inventory,
        daily_root,
        projections,
    )
    return materialize_orats_option_compaction(
        daily_set,
        OratsOptionCompactionSpec(start_date, end_date_exclusive),
        tmp_path / "option-annual",
        tmp_path / "option-annual-set.json",
    )


def _forecast_dividends(tmp_path: Path) -> VerifiedOratsDividendProjectionSet:
    raw = tmp_path / "forecast-raw"
    raw.mkdir()
    archive_path = raw / orats_dividend_archive_name(QUOTE_DATE.year)
    year_directory = f"{QUOTE_DATE.year}ORATSStockDivForecasts"
    rows = "".join(
        f"{ticker} {FORECAST_EX_DATE.isoformat()} 0.5 4\n"
        for ticker in PROJECT_AS_SUBMITTED_TICKERS
    )
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{year_directory}/{QUOTE_DATE:%Y%m%d}ORATSStockDivForecasts.txt",
            rows,
        )
    payload = archive_path.read_bytes()
    entry = OratsDividendArchiveEntry(
        year=QUOTE_DATE.year,
        url=orats_dividend_archive_url(QUOTE_DATE.year),
        size_bytes=len(payload),
        modified_at_utc=datetime(2025, 1, 2, 3, 21, tzinfo=UTC),
        etag='"0123456789abcdef0123456789abcdef-2"',
    )
    source = VerifiedOratsDividendArchive(
        path=archive_path,
        receipt=OratsDividendArchiveReceipt(
            entry=entry,
            local_name=archive_path.name,
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )
    inventory = OratsDividendArchiveInventory(
        base_url=ORATS_DIVIDEND_ARCHIVE_BASE_URL,
        start_year=QUOTE_DATE.year,
        end_year=QUOTE_DATE.year,
        observed_at_utc=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
        entries=(entry,),
        entries_sha256=orats_dividend_archive_entries_sha256((entry,)),
    )
    inventory_path = tmp_path / "forecast-inventory.json"
    write_orats_dividend_archive_inventory(inventory_path, inventory)
    output_root = tmp_path / "forecast-projected"
    projection = materialize_orats_dividend_projection(
        source,
        OratsDividendProjectionSpec(tickers=PROJECT_AS_SUBMITTED_TICKERS),
        output_root,
        tmp_path,
    )
    return write_orats_dividend_projection_set_manifest(
        tmp_path / "forecast-set.json",
        inventory_path,
        inventory,
        output_root,
        (projection,),
    )


def _historical_dividends(tmp_path: Path) -> MaterializedHistoricalDividendProjection:
    source = tmp_path / "historical-source.parquet"
    pl.DataFrame(
        {
            "ticker": list(PROJECT_AS_SUBMITTED_TICKERS),
            "ex_date": [HISTORICAL_EX_DATE] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "amount": [0.4] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "freq": [4] * len(PROJECT_AS_SUBMITTED_TICKERS),
            "pay_date": [HISTORICAL_EX_DATE + timedelta(days=14)]
            * len(PROJECT_AS_SUBMITTED_TICKERS),
        },
        schema=HISTORICAL_DIVIDEND_SOURCE_DTYPES,
    ).write_parquet(source)
    return materialize_historical_dividend_projection(
        source,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_source_rows=len(PROJECT_AS_SUBMITTED_TICKERS),
        spec=HistoricalDividendProjectionSpec(
            tickers=PROJECT_AS_SUBMITTED_TICKERS,
            maximum_ex_date=QUOTE_DATE,
        ),
        output_root=tmp_path / "historical-projected",
    )


def _underlying_history(tmp_path: Path) -> VerifiedPinnedParquet:
    path = tmp_path / "underlying.parquet"
    origin = QUOTE_DATE - timedelta(days=HISTORY_ROWS - 1)
    dates = [origin + timedelta(days=index) for index in range(HISTORY_ROWS)]
    values = [90.0 + index * 0.01 for index in range(HISTORY_ROWS)]
    frame = pl.DataFrame(
        {
            "ticker": [
                ticker for ticker in PROJECT_AS_SUBMITTED_TICKERS for _ in range(HISTORY_ROWS)
            ],
            "date": dates * len(PROJECT_AS_SUBMITTED_TICKERS),
            "split_adjusted_close": values * len(PROJECT_AS_SUBMITTED_TICKERS),
            "dividend_and_split_adjusted_close": values * len(PROJECT_AS_SUBMITTED_TICKERS),
        },
        schema=UNDERLYING_HISTORY_DTYPES,
    )
    frame.write_parquet(path)
    return verify_pinned_parquet(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_rows=frame.height,
        expected_schema=UNDERLYING_HISTORY_DTYPES,
    )


def _nss_parameters(tmp_path: Path) -> VerifiedPinnedParquet:
    path = tmp_path / "nss.parquet"
    frame = pl.DataFrame(
        {
            "date": [QUOTE_DATE],
            "beta0": [0.03],
            "beta1": [-0.01],
            "beta2": [0.02],
            "beta3": [0.005],
            "tau1": [1.5],
            "tau2": [3.0],
            "shortest_maturity": [0.25],
            "shortest_rate": [0.019],
        },
        schema=NSS_PARAMETER_DTYPES,
    )
    frame.write_parquet(path)
    return verify_pinned_parquet(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_rows=frame.height,
        expected_schema=NSS_PARAMETER_DTYPES,
    )


def _sources(tmp_path: Path) -> ProjectFeatureSources:
    return ProjectFeatureSources(
        options=_option_compaction(tmp_path),
        forecast_dividends=_forecast_dividends(tmp_path),
        historical_dividends=_historical_dividends(tmp_path),
        underlying_history=_underlying_history(tmp_path),
        nss_parameters=_nss_parameters(tmp_path),
    )


def test_feature_artifact_binds_all_sources_and_is_exactly_reusable(tmp_path: Path) -> None:
    """Five verified sources become one detached, typed, content-addressed Parquet."""
    sources = _sources(tmp_path)
    output_root = tmp_path / "features"
    first = materialize_project_feature_artifact(sources, output_root)
    second = materialize_project_feature_artifact(sources, output_root)

    assert first == second
    assert first.path.parent.name == first.manifest.request_sha256
    assert first.scan().collect().shape == (
        len(PROJECT_AS_SUBMITTED_TICKERS),
        len(PROJECT_AS_SUBMITTED_FEATURE_DTYPES),
    )
    assert ProjectFeatureArtifactManifest.from_dict(first.manifest.as_dict()) == first.manifest
    assert json.loads(first.receipt_path.read_bytes()) == first.manifest.as_dict()

    upstream_paths = (
        *sources.options.paths,
        sources.options.manifest_path,
        *sources.forecast_dividends.paths,
        sources.forecast_dividends.manifest_path,
        sources.forecast_dividends.inventory_path,
        sources.historical_dividends.path,
        sources.historical_dividends.path.with_name(
            f"{sources.historical_dividends.path.name}.receipt.json",
        ),
        sources.underlying_history.path,
        sources.nss_parameters.path,
    )
    for path in upstream_paths:
        path.unlink()
    assert load_verified_project_feature_artifact(first.path) == first

    first.path.write_bytes(b"x" * first.path.stat().st_size)
    with pytest.raises(ProjectFeatureArtifactContentError, match="receipt SHA-256"):
        load_verified_project_feature_artifact(first.path)


def test_feature_materialization_rejects_runtime_and_source_drift(tmp_path: Path) -> None:
    """A changed support source or byte-producing runtime fails without fallback."""
    sources = _sources(tmp_path)
    with pytest.raises(ProjectFeatureArtifactSpecError, match="active runtime"):
        materialize_project_feature_artifact(
            sources,
            tmp_path / "wrong-runtime",
            writer=ProjectFeatureWriteSpec(polars_version="0.0.0"),
        )

    sources.underlying_history.path.write_bytes(
        b"x" * sources.underlying_history.path.stat().st_size,
    )
    with pytest.raises(PinnedParquetContentError, match="SHA-256"):
        materialize_project_feature_artifact(sources, tmp_path / "changed-source")


def test_feature_materialization_rejects_a_benchmark_option_horizon(tmp_path: Path) -> None:
    """A valid benchmark subset cannot masquerade as the complete paper feature source."""
    sources = _sources(tmp_path)
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    benchmark_options = _option_compaction(
        benchmark_root,
        start_date=QUOTE_DATE,
        end_date_exclusive=QUOTE_DATE + timedelta(days=1),
    )
    benchmark_sources = ProjectFeatureSources(
        benchmark_options,
        sources.forecast_dividends,
        sources.historical_dividends,
        sources.underlying_history,
        sources.nss_parameters,
    )
    with pytest.raises(ProjectFeatureArtifactContentError, match="exact project quote horizon"):
        materialize_project_feature_artifact(benchmark_sources, tmp_path / "benchmark-features")


def test_feature_cli_requires_every_explicit_verified_source_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public command exposes no path discovery, schema coercion, or inferred hash."""
    sources = _sources(tmp_path)
    arguments = (
        "materialize-project-features",
        "--option-compaction-manifest",
        str(sources.options.manifest_path),
        "--option-compaction-root",
        str(sources.options.output_root),
        "--forecast-manifest",
        str(sources.forecast_dividends.manifest_path),
        "--forecast-inventory",
        str(sources.forecast_dividends.inventory_path),
        "--forecast-root",
        str(sources.forecast_dividends.output_root),
        "--historical-dividends",
        str(sources.historical_dividends.path),
        "--underlying-history",
        str(sources.underlying_history.path),
        "--underlying-sha256",
        sources.underlying_history.identity.sha256,
        "--underlying-rows",
        str(sources.underlying_history.identity.rows),
        "--nss-parameters",
        str(sources.nss_parameters.path),
        "--nss-sha256",
        sources.nss_parameters.identity.sha256,
        "--nss-rows",
        str(sources.nss_parameters.identity.rows),
        "--output-root",
        str(tmp_path / "features-cli"),
    )
    assert main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == len(PROJECT_AS_SUBMITTED_TICKERS)
    assert main(arguments) == 0
