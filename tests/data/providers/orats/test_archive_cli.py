"""Command tests for bounded ORATS archive inventory and acquisition."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Self, cast

import pytest

from discrete_dividend_study.cli import StudyCliError, main
from discrete_dividend_study.data.providers.orats.archive import (
    ORATS_FTPS_PASSWORD_ENV,
    ORATS_FTPS_USER_ENV,
    OratsArchiveEntry,
    OratsArchiveInventory,
    OratsArchiveReceipt,
    archive_entries_sha256,
    read_orats_archive_inventory,
)

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


PAYLOAD = b"synthetic-licensed-archive"
TRADE_DATE = date(2024, 9, 27)
MODIFIED = datetime(2024, 9, 28, 3, 21, tzinfo=UTC)
OBSERVED = datetime(2026, 7, 14, 12, 30, tzinfo=UTC)


def _entry() -> OratsArchiveEntry:
    return OratsArchiveEntry(
        trade_date=TRADE_DATE,
        remote_path="/smvstrikes/2024/ORATS_SMV_Strikes_20240927.zip",
        size_bytes=len(PAYLOAD),
        modified_at_utc=MODIFIED,
    )


def _inventory() -> OratsArchiveInventory:
    entry = _entry()
    return OratsArchiveInventory(
        host="de1.hostedftp.com",
        start_date=TRADE_DATE,
        end_date=TRADE_DATE,
        observed_at_utc=OBSERVED,
        entries=(entry,),
        entries_sha256=archive_entries_sha256((entry,)),
    )


class _FakeClient:
    def __init__(self, *_arguments: object, **_keywords: object) -> None:
        self.inventory_value = _inventory()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def inventory(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> OratsArchiveInventory:
        assert start_date == TRADE_DATE
        assert end_date == TRADE_DATE
        return self.inventory_value

    def download(self, entry: OratsArchiveEntry, destination: Path) -> OratsArchiveReceipt:
        assert entry == _entry()
        destination.mkdir(parents=True, exist_ok=True)
        return OratsArchiveReceipt(
            entry=entry,
            local_name="ORATS_SMV_Strikes_20240927.zip",
            sha256=hashlib.sha256(PAYLOAD).hexdigest(),
        )


def test_archive_commands_write_safe_manifest_and_require_exact_entry_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Inventory and download commands expose no credential and reject a stale row count."""
    user = "researcher"
    credential = "synthetic-credential"
    monkeypatch.setenv(ORATS_FTPS_USER_ENV, user)
    monkeypatch.setenv(ORATS_FTPS_PASSWORD_ENV, credential)
    monkeypatch.setattr("discrete_dividend_study.cli.OratsFtpsClient", _FakeClient)
    manifest = tmp_path / "inventory.json"

    assert (
        main(
            (
                "inventory-orats-archives",
                "--start-date",
                TRADE_DATE.isoformat(),
                "--end-date",
                TRADE_DATE.isoformat(),
                "--manifest",
                str(manifest),
            ),
        )
        == 0
    )
    listed = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert listed["entries"] == 1
    assert listed["compressed_bytes"] == len(PAYLOAD)
    assert read_orats_archive_inventory(manifest) == _inventory()

    with pytest.raises(StudyCliError, match="confirm-entry-count"):
        main(
            (
                "download-orats-archives",
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
                "download-orats-archives",
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
    summary = cast("dict[str, object]", json.loads(captured.out))
    assert summary["entries"] == 1
    assert user not in captured.out + captured.err
    assert credential not in captured.out + captured.err
    assert TRADE_DATE.isoformat() in captured.err
