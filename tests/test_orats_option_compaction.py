"""Tests for verified annual compaction of daily ORATS option projections."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import pytest

from discrete_dividend_study import (
    OratsArchiveEntry,
    OratsArchiveInventory,
    OratsArchiveReceipt,
    OratsOptionProjectionSpec,
    VerifiedOratsArchive,
    VerifiedOratsOptionProjectionSet,
    archive_entries_sha256,
    materialize_orats_option_projection,
    orats_archive_name,
    orats_remote_directory,
    write_orats_archive_inventory,
    write_orats_option_projection_set_manifest,
)
from discrete_dividend_study.orats_option_compaction import (
    OratsOptionCompactionContentError,
    OratsOptionCompactionManifest,
    OratsOptionCompactionReceipt,
    OratsOptionCompactionSpec,
    OratsOptionCompactionSpecError,
    ParquetCompactionWriteSpec,
    load_verified_orats_option_compaction,
    materialize_orats_option_compaction,
    verify_orats_option_compaction_unchanged,
)

if TYPE_CHECKING:
    from pathlib import Path

FIRST_DATE = date(2023, 12, 29)
SECOND_DATE = date(2024, 1, 2)
END_EXCLUSIVE = date(2024, 1, 3)
EXPECTED_ROWS = 3
HEADER = (
    "ticker,trade_date,expirDate,strike,cAskPx,cBidPx,pAskPx,pBidPx,stkPx,"
    "cVolu,pVolu,cOi,pOi,cValue,pValue,cMidIv,pMidIv,yte,iRate\n"
)


def _verified_archive(
    root: Path,
    trade_date: date,
    tickers: tuple[str, ...],
) -> VerifiedOratsArchive:
    archive_path = root / orats_archive_name(trade_date)
    expiration = trade_date + timedelta(days=90)
    rows = "".join(
        f"{ticker},{trade_date.month}/{trade_date.day}/{trade_date.year},"
        f"{expiration.month}/{expiration.day}/{expiration.year},80,21,20.5,1,0.9,100,"
        "10,5,100,50,20.8,0.95,0.2,0.21,0.2466,0.04\n"
        for ticker in tickers
    )
    member = f"ORATS_SMV_Strikes_{trade_date:%Y%m%d}.csv"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, HEADER + rows)
    payload = archive_path.read_bytes()
    entry = OratsArchiveEntry(
        trade_date=trade_date,
        remote_path=f"{orats_remote_directory(trade_date.year)}/{archive_path.name}",
        size_bytes=len(payload),
        modified_at_utc=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
    )
    return VerifiedOratsArchive(
        path=archive_path,
        receipt=OratsArchiveReceipt(
            entry=entry,
            local_name=archive_path.name,
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )


def _source_projection_set(tmp_path: Path) -> VerifiedOratsOptionProjectionSet:
    raw = tmp_path / "raw"
    raw.mkdir()
    archives = (
        _verified_archive(raw, FIRST_DATE, ("DUK",)),
        _verified_archive(raw, SECOND_DATE, ("DUK", "VZ")),
    )
    entries = tuple(archive.receipt.entry for archive in archives)
    inventory = OratsArchiveInventory(
        host="de1.hostedftp.com",
        start_date=FIRST_DATE,
        end_date=SECOND_DATE,
        observed_at_utc=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
        entries=entries,
        entries_sha256=archive_entries_sha256(entries),
    )
    inventory_path = tmp_path / "inventory.json"
    write_orats_archive_inventory(inventory_path, inventory)
    projection_root = tmp_path / "daily"
    spec = OratsOptionProjectionSpec(tickers=("DUK", "VZ"))
    projections = tuple(
        materialize_orats_option_projection(archive, spec, projection_root, tmp_path)
        for archive in archives
    )
    projection_manifest = tmp_path / "daily-set.json"
    return write_orats_option_projection_set_manifest(
        projection_manifest,
        inventory_path,
        inventory,
        projection_root,
        projections,
    )


def test_annual_compaction_is_restart_safe_and_detached_from_daily_sources(
    tmp_path: Path,
) -> None:
    """Two annual files preserve all rows even when historical ticker coverage differs."""
    source = _source_projection_set(tmp_path)
    spec = OratsOptionCompactionSpec(
        start_date=FIRST_DATE,
        end_date_exclusive=END_EXCLUSIVE,
    )
    output_root = tmp_path / "annual"
    manifest_path = tmp_path / "annual-set.json"
    first = materialize_orats_option_compaction(source, spec, output_root, manifest_path)
    second = materialize_orats_option_compaction(source, spec, output_root, manifest_path)

    assert first == second
    assert tuple(entry.year for entry in first.manifest.entries) == (2023, 2024)
    assert tuple(entry.observed_tickers for entry in first.manifest.entries) == (
        ("DUK",),
        ("DUK", "VZ"),
    )
    assert first.scan().collect().height == EXPECTED_ROWS
    assert OratsOptionCompactionManifest.from_dict(first.manifest.as_dict()) == first.manifest
    for entry in first.manifest.entries:
        receipt_path = output_root / entry.receipt_relative_path
        receipt = OratsOptionCompactionReceipt.from_dict(json.loads(receipt_path.read_bytes()))
        assert receipt.output_rows == entry.output_rows

    for path in (*source.paths, source.manifest_path, source.inventory_path):
        path.unlink()
    detached = load_verified_orats_option_compaction(manifest_path, output_root)
    assert detached == first
    verify_orats_option_compaction_unchanged(detached)

    corrupted = detached.paths[0]
    corrupted.write_bytes(b"x" * corrupted.stat().st_size)
    with pytest.raises(OratsOptionCompactionContentError, match="receipt SHA-256"):
        load_verified_orats_option_compaction(manifest_path, output_root)


def test_compaction_rejects_uncovered_dates_and_wrong_writer_runtime(tmp_path: Path) -> None:
    """Selection and byte-producing runtime identities never receive a fallback."""
    source = _source_projection_set(tmp_path)
    with pytest.raises(OratsOptionCompactionContentError, match="fully covered"):
        materialize_orats_option_compaction(
            source,
            OratsOptionCompactionSpec(
                start_date=FIRST_DATE - timedelta(days=1),
                end_date_exclusive=END_EXCLUSIVE,
            ),
            tmp_path / "outside",
            tmp_path / "outside.json",
        )

    wrong_runtime = OratsOptionCompactionSpec(
        start_date=FIRST_DATE,
        end_date_exclusive=END_EXCLUSIVE,
        writer=ParquetCompactionWriteSpec(polars_version="0.0.0"),
    )
    with pytest.raises(OratsOptionCompactionSpecError, match="active runtime"):
        materialize_orats_option_compaction(
            source,
            wrong_runtime,
            tmp_path / "wrong-runtime",
            tmp_path / "wrong-runtime.json",
        )


def test_compaction_writer_spec_round_trips_exactly() -> None:
    """The physical Parquet identity is explicit and strictly serializable."""
    writer = ParquetCompactionWriteSpec()
    assert ParquetCompactionWriteSpec.from_dict(writer.as_dict()) == writer
    assert writer.polars_version == pl.__version__
