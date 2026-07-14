"""Tests for immutable projection of verified ORATS option archives."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest

from discrete_dividend_study import (
    ORATS_NORMALIZED_OPTION_DTYPES,
    OratsArchiveEntry,
    OratsArchiveInventory,
    OratsArchiveReceipt,
    OratsOptionProjectionContentError,
    OratsOptionProjectionReceipt,
    OratsOptionProjectionSpec,
    OratsOptionProjectionSpecError,
    VerifiedOratsArchive,
    archive_entries_sha256,
    materialize_orats_option_projection,
    orats_archive_name,
    orats_option_projection_spec_sha256,
    orats_remote_directory,
    write_orats_archive_inventory,
)
from discrete_dividend_study.cli import StudyCliError, main

if TYPE_CHECKING:
    from pathlib import Path


TRADE_DATE = date(2024, 9, 27)
EXPECTED_MEMBER = "ORATS_SMV_Strikes_20240927.csv"
HEADER = (
    "ticker,trade_date,expirDate,strike,cAskPx,cBidPx,pAskPx,pBidPx,stkPx,"
    "cVolu,pVolu,cOi,pOi,cValue,pValue,cMidIv,pMidIv,yte,iRate,unused\n"
)
ROWS = (
    "DUK,9/27/2024,12/20/2024,115,2.2,2.1,1.2,1.1,116,20,10,200,100,"
    "2.15,1.15,0.2,0.21,0.2301,0.041,kept-as-text\n"
    "VZ,9/27/2024,12/20/2024,45,2.3,2.2,0.8,0.7,44,30,15,300,150,"
    "2.25,0.75,0.3,0.31,0.2301,0.041,kept-as-text\n"
    "AAPL,9/27/2024,12/20/2024,225,9.3,9.2,8.8,8.7,224,50,40,500,400,"
    "9.25,8.75,0.25,0.26,0.2301,0.041,excluded\n"
)


def _verified_archive(
    root: Path,
    *,
    member_name: str = EXPECTED_MEMBER,
    csv: str = HEADER + ROWS,
    extra_member: str | None = None,
) -> VerifiedOratsArchive:
    archive_path = root / orats_archive_name(TRADE_DATE)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, csv)
        if extra_member is not None:
            archive.writestr(extra_member, "ambiguous delivery")
    payload = archive_path.read_bytes()
    entry = OratsArchiveEntry(
        trade_date=TRADE_DATE,
        remote_path=f"{orats_remote_directory(TRADE_DATE.year)}/{archive_path.name}",
        size_bytes=len(payload),
        modified_at_utc=datetime(2024, 9, 28, 3, 21, tzinfo=UTC),
    )
    return VerifiedOratsArchive(
        path=archive_path,
        receipt=OratsArchiveReceipt(
            entry=entry,
            local_name=archive_path.name,
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )


def _spec() -> OratsOptionProjectionSpec:
    return OratsOptionProjectionSpec(tickers=("DUK", "VZ"))


def test_projection_is_filtered_typed_receipted_and_exactly_reusable(tmp_path: Path) -> None:
    """One verified ZIP becomes a deterministic-location, hash-receipted daily Parquet."""
    source = _verified_archive(tmp_path)
    output_root = tmp_path / "projected"
    first = materialize_orats_option_projection(source, _spec(), output_root, tmp_path)

    expected = (
        output_root / orats_option_projection_spec_sha256(_spec()) / "2024" / "2024-09-27.parquet"
    )
    assert first.path == expected
    assert first.receipt.output_rows == len(_spec().tickers)
    assert first.receipt.observed_tickers == ("DUK", "VZ")
    frame = pl.read_parquet(expected)
    assert frame.schema == pl.Schema(ORATS_NORMALIZED_OPTION_DTYPES)
    assert frame.get_column("ticker").to_list() == ["DUK", "VZ"]
    assert frame.get_column("quote_date").unique().to_list() == [TRADE_DATE]
    assert not tuple(tmp_path.glob("*.scratch.csv"))

    receipt_path = expected.with_name(f"{expected.name}.receipt.json")
    receipt_json = json.loads(receipt_path.read_bytes())
    assert receipt_json == first.receipt.as_dict()
    second = materialize_orats_option_projection(source, _spec(), output_root, tmp_path)
    assert second == first

    duplicate_tickers = first.receipt.as_dict()
    duplicate_tickers["observed_tickers"] = ["DUK", "DUK"]
    with pytest.raises(OratsOptionProjectionContentError, match="sorted and unique"):
        OratsOptionProjectionReceipt.from_dict(duplicate_tickers)

    impossible_compression = first.receipt.as_dict()
    impossible_compression["member_compressed_bytes"] = source.receipt.entry.size_bytes + 1
    with pytest.raises(OratsOptionProjectionContentError, match="exceed the source archive"):
        OratsOptionProjectionReceipt.from_dict(impossible_compression)


def test_existing_output_is_never_reused_after_same_size_tampering(tmp_path: Path) -> None:
    """A matching path and byte count cannot substitute for the receipt's exact bytes."""
    source = _verified_archive(tmp_path)
    materialized = materialize_orats_option_projection(
        source,
        _spec(),
        tmp_path / "projected",
        tmp_path,
    )
    size = materialized.path.stat().st_size
    materialized.path.write_bytes(b"x" * size)
    with pytest.raises(OratsOptionProjectionContentError, match="receipt SHA-256"):
        materialize_orats_option_projection(
            source,
            _spec(),
            tmp_path / "projected",
            tmp_path,
        )


@pytest.mark.parametrize(
    ("member_name", "extra_member", "message"),
    [
        ("nested/ORATS_SMV_Strikes_20240927.csv", None, "exact expected daily CSV"),
        (EXPECTED_MEMBER, "README.txt", "exactly one non-directory member"),
    ],
)
def test_projection_rejects_ambiguous_archive_shapes(
    tmp_path: Path,
    member_name: str,
    extra_member: str | None,
    message: str,
) -> None:
    """Wrong paths and multi-file ZIPs fail instead of choosing a plausible member."""
    source = _verified_archive(tmp_path, member_name=member_name, extra_member=extra_member)
    with pytest.raises(OratsOptionProjectionContentError, match=message):
        materialize_orats_option_projection(source, _spec(), tmp_path / "out", tmp_path)


def test_projection_rejects_wrong_dates_empty_universes_and_size_policy(tmp_path: Path) -> None:
    """Content identity and declared resource bounds are mandatory, never inferred."""
    wrong_date = _verified_archive(tmp_path, csv=(HEADER + ROWS).replace("9/27/2024", "9/26/2024"))
    with pytest.raises(OratsOptionProjectionContentError, match="quote dates disagree"):
        materialize_orats_option_projection(wrong_date, _spec(), tmp_path / "dates", tmp_path)

    with pytest.raises(OratsOptionProjectionContentError, match="no rows"):
        materialize_orats_option_projection(
            wrong_date,
            OratsOptionProjectionSpec(tickers=("IBM",)),
            tmp_path / "empty",
            tmp_path,
        )

    bounded = _verified_archive(tmp_path)
    with pytest.raises(OratsOptionProjectionContentError, match="byte bound"):
        materialize_orats_option_projection(
            bounded,
            OratsOptionProjectionSpec(tickers=("DUK",), max_member_bytes=10),
            tmp_path / "bounded",
            tmp_path,
        )


def test_projection_spec_refuses_implicit_canonicalization() -> None:
    """Ticker order, duplication, case, and empty universes are caller-owned errors."""
    for tickers in ((), ("VZ", "DUK"), ("DUK", "DUK"), ("duk",)):
        with pytest.raises(OratsOptionProjectionSpecError):
            OratsOptionProjectionSpec(tickers=tickers)


def test_cli_requires_complete_inventory_confirmation_and_projects_every_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public command binds a complete manifest to verified raw and projected outputs."""
    raw = tmp_path / "raw"
    raw.mkdir()
    source = _verified_archive(raw)
    source.path.with_name(f"{source.path.name}.receipt.json").write_text(
        json.dumps(source.receipt.as_dict()),
    )
    entry = source.receipt.entry
    inventory = OratsArchiveInventory(
        host="de1.hostedftp.com",
        start_date=TRADE_DATE,
        end_date=TRADE_DATE,
        observed_at_utc=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
        entries=(entry,),
        entries_sha256=archive_entries_sha256((entry,)),
    )
    manifest = tmp_path / "inventory.json"
    write_orats_archive_inventory(manifest, inventory)
    arguments = (
        "materialize-orats-options",
        "--manifest",
        str(manifest),
        "--raw-directory",
        str(raw),
        "--output-root",
        str(tmp_path / "projected"),
        "--scratch",
        str(tmp_path),
        "--ticker",
        "DUK",
        "--ticker",
        "VZ",
        "--confirm-entry-count",
        "1",
    )
    assert main(arguments) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["entries"] == len(inventory.entries)
    assert summary["rows"] == len(_spec().tickers)
    assert TRADE_DATE.isoformat() in captured.err
    assert main(arguments) == 0

    with pytest.raises(StudyCliError, match="confirm-entry-count"):
        main((*arguments[:-1], "2"))
