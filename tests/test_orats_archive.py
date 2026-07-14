"""Tests for encrypted, manifest-first ORATS archive acquisition."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, cast

import pytest

import discrete_dividend_study.orats_archive as archive_module
from discrete_dividend_study import (
    ORATS_FTPS_PASSWORD_ENV,
    ORATS_FTPS_USER_ENV,
    OratsArchiveConfigurationError,
    OratsArchiveEntry,
    OratsArchiveIntegrityError,
    OratsArchiveInventory,
    OratsFtpsClient,
    OratsFtpsCredentials,
    OratsFtpsSpec,
    archive_entries_sha256,
    load_verified_orats_archive,
    orats_archive_name,
    orats_remote_directory,
    read_orats_archive_inventory,
    write_orats_archive_inventory,
)

if TYPE_CHECKING:
    import ftplib
    from collections.abc import Callable, Iterator
    from pathlib import Path


OBSERVED = datetime(2026, 7, 14, 12, 30, tzinfo=UTC)
MODIFIED = datetime(2024, 9, 28, 3, 21, 0, 333_000, tzinfo=UTC)
PAYLOAD = b"licensed-archive-bytes"
NEW_TRANSFER_MLST_CHECKS = 2


def archive_entry(*, size_bytes: int = len(PAYLOAD)) -> OratsArchiveEntry:
    """Build the canonical synthetic 2024-09-27 archive identity."""
    trade_date = date(2024, 9, 27)
    return OratsArchiveEntry(
        trade_date=trade_date,
        remote_path=f"{orats_remote_directory(2024)}/{orats_archive_name(trade_date)}",
        size_bytes=size_bytes,
        modified_at_utc=MODIFIED,
    )


class _FakeFtps:
    def __init__(self, *, payload: bytes = PAYLOAD, modified: str = "20240928032100.333") -> None:
        self.payload = payload
        self.modified = modified
        self.retrievals: list[tuple[str, int, int | None]] = []
        self.mlst_commands: list[str] = []

    def mlsd(
        self,
        path: str,
        *,
        facts: list[str],
    ) -> Iterator[tuple[str, dict[str, str]]]:
        assert path == "/smvstrikes/2024"
        assert facts == ["type", "size", "modify"]
        yield (
            "ORATS_SMV_Strikes_20240927.zip",
            {
                "type": "file",
                "size": str(len(self.payload)),
                "modify": self.modified,
            },
        )
        yield ("notes.txt", {"type": "file", "size": "5", "modify": self.modified})

    def retrbinary(
        self,
        command: str,
        callback: Callable[[bytes], object],
        blocksize: int,
        rest: int | None = None,
    ) -> str:
        self.retrievals.append((command, blocksize, rest))
        callback(self.payload[rest or 0 :])
        return "226 transfer complete"

    def sendcmd(self, command: str) -> str:
        self.mlst_commands.append(command)
        assert command == "MLST /smvstrikes/2024/ORATS_SMV_Strikes_20240927.zip"
        return (
            "250-\n"
            f"Size={len(self.payload)};Modify={self.modified};Type=file; "
            "ORATS_SMV_Strikes_20240927.zip\n\n"
            "250 Requested file action okay, completed."
        )


def _client_with_fake(fake: _FakeFtps) -> OratsFtpsClient:
    dummy_secret = "synthetic-credential"  # noqa: S105 - non-secret test value
    client = OratsFtpsClient(
        OratsFtpsSpec(transfer_block_bytes=64 * 1024),
        OratsFtpsCredentials(username="researcher", password=dummy_secret),
    )
    client._ftp = cast("ftplib.FTP_TLS", fake)  # noqa: SLF001 - isolated transport fake
    return client


def test_credentials_are_environment_only_and_do_not_repr_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing variables fail and an admitted password never appears in repr."""
    monkeypatch.delenv(ORATS_FTPS_USER_ENV, raising=False)
    monkeypatch.delenv(ORATS_FTPS_PASSWORD_ENV, raising=False)
    with pytest.raises(OratsArchiveConfigurationError, match="missing required"):
        OratsFtpsCredentials.from_environment()

    monkeypatch.setenv(ORATS_FTPS_USER_ENV, "researcher")
    dummy_secret = "do-not-print"  # noqa: S105 - non-secret test value
    monkeypatch.setenv(ORATS_FTPS_PASSWORD_ENV, dummy_secret)
    credentials = OratsFtpsCredentials.from_environment()
    assert credentials.username == "researcher"
    assert credentials.password == dummy_secret
    assert dummy_secret not in repr(credentials)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2007, "/smvstrikes_2007_2012/2007"),
        (2012, "/smvstrikes_2007_2012/2012"),
        (2013, "/smvstrikes/2013"),
        (2024, "/smvstrikes/2024"),
    ],
)
def test_remote_directory_has_one_explicit_year_boundary(year: int, expected: str) -> None:
    """The historical and current vendor roots meet only at the declared 2012 boundary."""
    assert orats_remote_directory(year) == expected


def test_entry_rejects_wrong_directory_size_and_timezone() -> None:
    """Remote entry identity cannot be repaired by path or timestamp coercion."""
    with pytest.raises(OratsArchiveIntegrityError, match="expected remote directory"):
        OratsArchiveEntry(
            trade_date=date(2024, 9, 27),
            remote_path="/wrong/ORATS_SMV_Strikes_20240927.zip",
            size_bytes=1,
            modified_at_utc=MODIFIED,
        )
    with pytest.raises(OratsArchiveIntegrityError, match="size must be positive"):
        archive_entry(size_bytes=0)
    with pytest.raises(OratsArchiveIntegrityError, match="timezone-aware UTC"):
        OratsArchiveEntry(
            trade_date=date(2024, 9, 27),
            remote_path="/smvstrikes/2024/ORATS_SMV_Strikes_20240927.zip",
            size_bytes=1,
            modified_at_utc=MODIFIED.replace(tzinfo=None),
        )


def test_listing_builds_a_hashed_bounded_inventory() -> None:
    """MLSD facts become one exact entry while unrelated server files are ignored."""
    client = _client_with_fake(_FakeFtps())
    inventory = client.inventory(
        start_date=date(2024, 9, 27),
        end_date=date(2024, 9, 27),
        observed_at_utc=OBSERVED,
    )
    assert inventory.entries == (archive_entry(),)
    assert inventory.entries_sha256 == archive_entries_sha256(inventory.entries)
    assert inventory.observed_at_utc == OBSERVED


@pytest.mark.parametrize(
    ("size", "modified", "message"),
    [
        ("not-an-integer", "20240928032100", "MLSD size is malformed"),
        (str(len(PAYLOAD)), "2024-09-28", "modification time is malformed"),
    ],
)
def test_listing_rejects_malformed_required_facts(
    size: str,
    modified: str,
    message: str,
) -> None:
    """A matching filename with ambiguous identity stops the listing."""
    facts = {"type": "file", "size": size, "modify": modified}
    with pytest.raises(OratsArchiveIntegrityError, match=message):
        archive_module._entry_from_mlsd(  # noqa: SLF001 - pure parser boundary
            directory="/smvstrikes/2024",
            year=2024,
            name="ORATS_SMV_Strikes_20240927.zip",
            facts=facts,
        )


def test_inventory_round_trip_rehashes_entries_and_refuses_overwrite(tmp_path: Path) -> None:
    """A manifest is immutable and every read revalidates its content hash."""
    entry = archive_entry()
    inventory = OratsArchiveInventory(
        host="de1.hostedftp.com",
        start_date=entry.trade_date,
        end_date=entry.trade_date,
        observed_at_utc=OBSERVED,
        entries=(entry,),
        entries_sha256=archive_entries_sha256((entry,)),
    )
    path = tmp_path / "inventory.json"
    write_orats_archive_inventory(path, inventory)
    assert read_orats_archive_inventory(path) == inventory
    with pytest.raises(OratsArchiveIntegrityError, match="already exists"):
        write_orats_archive_inventory(path, inventory)

    raw = json.loads(path.read_bytes())
    raw["entries"][0]["size_bytes"] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(OratsArchiveIntegrityError, match="entry hash mismatch"):
        read_orats_archive_inventory(path)


def test_download_resumes_hashes_and_reuses_only_verified_receipt(tmp_path: Path) -> None:
    """A partial transfer resumes exactly and a completed transfer is byte-reverified."""
    fake = _FakeFtps()
    client = _client_with_fake(fake)
    entry = archive_entry()
    partial = tmp_path / f".{orats_archive_name(entry.trade_date)}.part"
    partial.write_bytes(PAYLOAD[:8])

    first = client.download(entry, tmp_path)
    final = tmp_path / orats_archive_name(entry.trade_date)
    assert final.read_bytes() == PAYLOAD
    assert first.sha256 == "fd03e1d0712c6f2f52de1a60729b007af9ab6d8fbf10522d86ca3fdf2f6a9fc7"
    assert fake.retrievals == [(f"RETR {entry.remote_path}", 64 * 1024, 8)]

    second = client.download(entry, tmp_path)
    assert second == first
    assert len(fake.retrievals) == 1
    assert len(fake.mlst_commands) == NEW_TRANSFER_MLST_CHECKS
    client._ftp = None  # noqa: SLF001 - prove completed receipts are server-detached
    assert client.download(entry, tmp_path) == first
    verified = load_verified_orats_archive(final)
    assert verified.path == final
    assert verified.receipt == first

    final.write_bytes(b"x" * entry.size_bytes)
    assert final.stat().st_size == entry.size_bytes
    with pytest.raises(OratsArchiveIntegrityError, match="receipt SHA-256"):
        client.download(entry, tmp_path)


def test_download_resumes_an_existing_zero_byte_partial(tmp_path: Path) -> None:
    """An interruption before the first payload byte remains a valid exact resume point."""
    fake = _FakeFtps()
    client = _client_with_fake(fake)
    entry = archive_entry()
    partial = tmp_path / f".{orats_archive_name(entry.trade_date)}.part"
    partial.touch()
    receipt = client.download(entry, tmp_path)
    assert receipt.sha256 == "fd03e1d0712c6f2f52de1a60729b007af9ab6d8fbf10522d86ca3fdf2f6a9fc7"
    assert fake.retrievals == [(f"RETR {entry.remote_path}", 64 * 1024, None)]


def test_download_rejects_remote_identity_change_before_transfer(tmp_path: Path) -> None:
    """A changed MLSD timestamp invalidates a previously observed entry."""
    fake = _FakeFtps(modified="20240929032100")
    client = _client_with_fake(fake)
    with pytest.raises(OratsArchiveIntegrityError, match="changed after inventory"):
        client.download(archive_entry(), tmp_path)
    assert fake.retrievals == []


def test_client_has_no_unencrypted_listing_fallback() -> None:
    """Using a client outside its explicit TLS context fails before any operation."""
    dummy_secret = "synthetic-credential"  # noqa: S105 - non-secret test value
    client = OratsFtpsClient(
        OratsFtpsSpec(),
        OratsFtpsCredentials(username="researcher", password=dummy_secret),
    )
    with pytest.raises(OratsArchiveConfigurationError, match="context manager"):
        client.inventory(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2))
