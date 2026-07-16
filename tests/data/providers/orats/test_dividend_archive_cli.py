"""Command tests for annual ORATS dividend archive acquisition."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from discrete_dividend_study.cli import StudyCliError, main
from discrete_dividend_study.data.providers.orats.dividend_archive import (
    ORATS_DIVIDEND_ARCHIVE_BASE_URL,
    OratsDividendArchiveEntry,
    OratsDividendArchiveInventory,
    OratsDividendArchiveReceipt,
    orats_dividend_archive_entries_sha256,
    orats_dividend_archive_name,
    orats_dividend_archive_url,
    read_orats_dividend_archive_inventory,
)

if TYPE_CHECKING:
    from pathlib import Path


YEAR = 2024
PAYLOAD = b"synthetic-annual-dividend-archive"
MODIFIED = datetime(2024, 12, 31, 3, 21, tzinfo=UTC)
OBSERVED = datetime(2026, 7, 14, 12, 30, tzinfo=UTC)


def _entry() -> OratsDividendArchiveEntry:
    return OratsDividendArchiveEntry(
        year=YEAR,
        url=orats_dividend_archive_url(YEAR),
        size_bytes=len(PAYLOAD),
        modified_at_utc=MODIFIED,
        etag='"0123456789abcdef0123456789abcdef"',
    )


def _inventory() -> OratsDividendArchiveInventory:
    entry = _entry()
    return OratsDividendArchiveInventory(
        base_url=ORATS_DIVIDEND_ARCHIVE_BASE_URL,
        start_year=YEAR,
        end_year=YEAR,
        observed_at_utc=OBSERVED,
        entries=(entry,),
        entries_sha256=orats_dividend_archive_entries_sha256((entry,)),
    )


class _FakeClient:
    def __init__(self, _spec: object) -> None:
        pass

    def inventory(self, *, start_year: int, end_year: int) -> OratsDividendArchiveInventory:
        assert start_year == YEAR
        assert end_year == YEAR
        return _inventory()

    def download(
        self,
        entry: OratsDividendArchiveEntry,
        destination: Path,
    ) -> OratsDividendArchiveReceipt:
        assert entry == _entry()
        destination.mkdir(parents=True, exist_ok=True)
        return OratsDividendArchiveReceipt(
            entry=entry,
            local_name=orats_dividend_archive_name(YEAR),
            sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        )


def test_dividend_archive_commands_freeze_then_confirm_complete_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The two commands expose only aggregate identities and require exact entry confirmation."""
    monkeypatch.setattr("discrete_dividend_study.cli.OratsDividendHttpsClient", _FakeClient)
    manifest = tmp_path / "dividend-inventory.json"
    assert (
        main(
            (
                "inventory-orats-dividend-archives",
                "--start-year",
                str(YEAR),
                "--end-year",
                str(YEAR),
                "--manifest",
                str(manifest),
            ),
        )
        == 0
    )
    inventory_output = json.loads(capsys.readouterr().out)
    assert inventory_output["entries"] == len(_inventory().entries)
    assert inventory_output["compressed_bytes"] == len(PAYLOAD)
    assert read_orats_dividend_archive_inventory(manifest) == _inventory()

    with pytest.raises(StudyCliError, match="confirm-entry-count"):
        main(
            (
                "download-orats-dividend-archives",
                "--manifest",
                str(manifest),
                "--destination",
                str(tmp_path / "raw"),
                "--confirm-entry-count",
                "2",
            ),
        )

    assert (
        main(
            (
                "download-orats-dividend-archives",
                "--manifest",
                str(manifest),
                "--destination",
                str(tmp_path / "raw"),
                "--confirm-entry-count",
                "1",
            ),
        )
        == 0
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["entries"] == len(_inventory().entries)
    assert str(YEAR) in captured.err
