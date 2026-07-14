"""Tests for immutable annual ORATS dividend ticker projection."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest

from discrete_dividend_study import (
    ORATS_DIVIDEND_ARCHIVE_BASE_URL,
    ORATS_DIVIDEND_DTYPES,
    OratsDividendArchiveEntry,
    OratsDividendArchiveInventory,
    OratsDividendArchiveReceipt,
    VerifiedOratsDividendArchive,
    orats_dividend_archive_entries_sha256,
    orats_dividend_archive_name,
    orats_dividend_archive_url,
    write_orats_dividend_archive_inventory,
)
from discrete_dividend_study.cli import StudyCliError, main
from discrete_dividend_study.orats_dividend_projection import (
    OratsDividendProjectionContentError,
    OratsDividendProjectionReceipt,
    OratsDividendProjectionSpec,
    OratsDividendProjectionSpecError,
    materialize_orats_dividend_projection,
    orats_dividend_projection_spec_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path


YEAR = 2024
MODIFIED = datetime(2025, 1, 2, 3, 21, tzinfo=UTC)
ETAG = '"0123456789abcdef0123456789abcdef-2"'
FIRST_NAME = "202401020610ORATSStockDivForecasts.txt"
SECOND_NAME = "20240103ORATSStockDivForecasts.txt"
FIRST_ROWS = "DUK 2024-02-01 0.25 4\nVZ 2024-03-01 0.66 4 \nAAPL 2024-02-09 0.24 4\n"
SECOND_ROWS = "DUK 2024-05-01 0.27 4\nAAPL 2024-05-10 0.25 4\n"
PROJECTED_ROW_COUNT = 3


def _verified_archive(
    root: Path,
    *,
    first_rows: str = FIRST_ROWS,
    second_name: str = SECOND_NAME,
    include_unknown: bool = False,
    symlink_member: bool = False,
) -> VerifiedOratsDividendArchive:
    path = root / orats_dividend_archive_name(YEAR)
    year_directory = f"{YEAR}ORATSStockDivForecasts"
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{year_directory}/{FIRST_NAME}", first_rows)
        archive.writestr(f"{year_directory}/{second_name}", SECOND_ROWS)
        archive.writestr(
            f"{year_directory}/20240102ORATSDividends.txt",
            "DUK 2024-02-01 0.25 E\n",
        )
        archive.writestr("__MACOSX/._2024ORATSStockDivForecasts", "metadata")
        if include_unknown:
            archive.writestr(f"{year_directory}/unexpected.txt", "unsupported")
        if symlink_member:
            info = zipfile.ZipInfo(f"{year_directory}/20240104ORATSDividends.txt")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "target")
    payload = path.read_bytes()
    entry = OratsDividendArchiveEntry(
        year=YEAR,
        url=orats_dividend_archive_url(YEAR),
        size_bytes=len(payload),
        modified_at_utc=MODIFIED,
        etag=ETAG,
    )
    return VerifiedOratsDividendArchive(
        path=path,
        receipt=OratsDividendArchiveReceipt(
            entry=entry,
            local_name=path.name,
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )


def _spec() -> OratsDividendProjectionSpec:
    return OratsDividendProjectionSpec(tickers=("DUK", "VZ"))


def test_projection_tracks_every_snapshot_and_is_exactly_reusable(tmp_path: Path) -> None:
    """Known layout variants become one sorted, typed, hash-receipted annual Parquet."""
    source = _verified_archive(tmp_path)
    output_root = tmp_path / "projected"
    first = materialize_orats_dividend_projection(source, _spec(), output_root, tmp_path)
    expected = output_root / orats_dividend_projection_spec_sha256(_spec()) / "2024.parquet"
    assert first.path == expected
    assert first.receipt.output_rows == PROJECTED_ROW_COUNT
    assert first.receipt.observed_tickers == ("DUK", "VZ")
    assert tuple(snapshot.raw_rows for snapshot in first.receipt.snapshots) == (3, 2)
    assert tuple(snapshot.selected_rows for snapshot in first.receipt.snapshots) == (2, 1)
    frame = pl.read_parquet(expected)
    assert frame.schema == pl.Schema(ORATS_DIVIDEND_DTYPES)
    assert frame.get_column("ticker").to_list() == ["DUK", "VZ", "DUK"]
    assert not tuple(tmp_path.glob("*.scratch"))

    receipt_path = expected.with_name(f"{expected.name}.receipt.json")
    assert json.loads(receipt_path.read_bytes()) == first.receipt.as_dict()
    assert OratsDividendProjectionReceipt.from_dict(first.receipt.as_dict()) == first.receipt
    assert materialize_orats_dividend_projection(source, _spec(), output_root, tmp_path) == first


def test_projection_never_reuses_same_size_corrupted_output(tmp_path: Path) -> None:
    """Output paths and byte counts never substitute for exact Parquet bytes."""
    source = _verified_archive(tmp_path)
    projected = materialize_orats_dividend_projection(
        source,
        _spec(),
        tmp_path / "projected",
        tmp_path,
    )
    projected.path.write_bytes(b"x" * projected.path.stat().st_size)
    with pytest.raises(OratsDividendProjectionContentError, match="receipt SHA-256"):
        materialize_orats_dividend_projection(
            source,
            _spec(),
            tmp_path / "projected",
            tmp_path,
        )


@pytest.mark.parametrize(
    ("first_rows", "message"),
    [
        ("DUK 2024-02-01 0.25 4 unexpected\n", "exactly four"),
        ("DUK 2024-02-01 0.25\n", "exactly four"),
        ("IBM 2024-02-01 0.25 4\n", "no rows"),
    ],
)
def test_projection_rejects_malformed_or_empty_selected_content(
    tmp_path: Path,
    first_rows: str,
    message: str,
) -> None:
    """Structural violations and an absent declared universe fail without a partial artifact."""
    source = _verified_archive(
        tmp_path,
        first_rows=first_rows,
    )
    spec = (
        OratsDividendProjectionSpec(tickers=("WYNN",)) if first_rows.startswith("IBM") else _spec()
    )
    with pytest.raises(OratsDividendProjectionContentError, match=message):
        materialize_orats_dividend_projection(source, spec, tmp_path / "out", tmp_path)


def test_projection_rejects_unknown_symlink_duplicate_and_bounded_members(tmp_path: Path) -> None:
    """The ZIP member allowlist and resource identity never guess around unsafe shapes."""
    unknown = _verified_archive(tmp_path, include_unknown=True)
    with pytest.raises(OratsDividendProjectionContentError, match="unsupported provider feed"):
        materialize_orats_dividend_projection(unknown, _spec(), tmp_path / "unknown", tmp_path)

    symlink = _verified_archive(tmp_path, symlink_member=True)
    with pytest.raises(OratsDividendProjectionContentError, match="symbolic-link"):
        materialize_orats_dividend_projection(symlink, _spec(), tmp_path / "symlink", tmp_path)

    duplicate = _verified_archive(
        tmp_path,
        second_name="202401020700ORATSStockDivForecasts.txt",
    )
    with pytest.raises(OratsDividendProjectionContentError, match="unique regular snapshot"):
        materialize_orats_dividend_projection(duplicate, _spec(), tmp_path / "duplicate", tmp_path)

    bounded = _verified_archive(tmp_path)
    with pytest.raises(OratsDividendProjectionContentError, match="per-member byte bound"):
        materialize_orats_dividend_projection(
            bounded,
            OratsDividendProjectionSpec(
                tickers=("DUK",),
                max_member_bytes=10,
                max_total_uncompressed_bytes=1000,
            ),
            tmp_path / "bounded",
            tmp_path,
        )


def test_projection_spec_refuses_implicit_ticker_normalization() -> None:
    """Empty, unsorted, duplicate, and lowercase universes remain caller errors."""
    for tickers in ((), ("VZ", "DUK"), ("DUK", "DUK"), ("duk",)):
        with pytest.raises(OratsDividendProjectionSpecError):
            OratsDividendProjectionSpec(tickers=tickers)


def test_cli_binds_complete_inventory_raw_receipts_and_projection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public command confirms and projects every exact annual inventory entry."""
    raw = tmp_path / "raw"
    raw.mkdir()
    source = _verified_archive(raw)
    source.path.with_name(f"{source.path.name}.receipt.json").write_text(
        json.dumps(source.receipt.as_dict()),
    )
    entry = source.receipt.entry
    inventory = OratsDividendArchiveInventory(
        base_url=ORATS_DIVIDEND_ARCHIVE_BASE_URL,
        start_year=YEAR,
        end_year=YEAR,
        observed_at_utc=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
        entries=(entry,),
        entries_sha256=orats_dividend_archive_entries_sha256((entry,)),
    )
    manifest = tmp_path / "inventory.json"
    write_orats_dividend_archive_inventory(manifest, inventory)
    arguments = (
        "materialize-orats-dividends",
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
    assert summary["rows"] == PROJECTED_ROW_COUNT
    assert summary["snapshots"] == len((FIRST_NAME, SECOND_NAME))
    assert str(YEAR) in captured.err
    assert main(arguments) == 0

    with pytest.raises(StudyCliError, match="confirm-entry-count"):
        main((*arguments[:-1], "2"))
