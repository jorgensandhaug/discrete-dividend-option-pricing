"""Fail-fast, manifest-first access to licensed ORATS daily option archives."""

from __future__ import annotations

import contextlib
import ftplib
import hashlib
import json
import os
import re
import ssl
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Self, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

ORATS_ARCHIVE_INVENTORY_VERSION = 1
ORATS_ARCHIVE_RECEIPT_VERSION = 1
ORATS_DEFAULT_FTPS_HOST = "de1.hostedftp.com"
ORATS_DEFAULT_FTPS_PORT = 21
ORATS_DEFAULT_FTPS_TIMEOUT_SECONDS = 60.0
ORATS_DEFAULT_TRANSFER_BLOCK_BYTES = 8 * 1024 * 1024
ORATS_FTPS_USER_ENV = "ORATS_FTP_USER"
ORATS_FTPS_PASSWORD_ENV = "ORATS_FTP_PASSWORD"  # noqa: S105 - environment-variable name

_ARCHIVE_NAME_PATTERN = re.compile(r"ORATS_SMV_Strikes_(?P<date>[0-9]{8})[.]zip")
_MLSD_MODIFY_PATTERN = re.compile(r"[0-9]{14}(?:[.][0-9]{1,6})?")
_FIRST_ARCHIVE_YEAR = 2007
_LAST_LEGACY_ARCHIVE_YEAR = 2012
_MAXIMUM_ARCHIVE_YEAR = 9999
_MINIMUM_TCP_PORT = 1
_MAXIMUM_TCP_PORT = 65_535
_MINIMUM_TRANSFER_BLOCK_BYTES = 64 * 1024
_SHA256_HEX_LENGTH = 64


class OratsArchiveError(RuntimeError):
    """Base class for an invalid archive request, transport, or artifact."""


@dataclass(slots=True)
class OratsArchiveConfigurationError(OratsArchiveError):
    """Static archive settings violate the explicit transport contract."""

    reason: str

    def __str__(self) -> str:
        """Describe the invalid setting without disclosing credentials."""
        return self.reason


@dataclass(slots=True)
class OratsArchiveTransportError(OratsArchiveError):
    """An encrypted remote operation failed."""

    operation: str
    remote_path: str | None
    reason: str

    def __str__(self) -> str:
        """Describe the failed operation without credentials or licensed rows."""
        location = "" if self.remote_path is None else f" for {self.remote_path}"
        return f"ORATS FTPS {self.operation} failed{location}: {self.reason}"


@dataclass(slots=True)
class OratsArchiveIntegrityError(OratsArchiveError):
    """A listing, manifest, or downloaded artifact violates its identity."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant."""
        location = "" if self.path is None else f" ({self.path})"
        return f"ORATS archive integrity failure{location}: {self.reason}"


def _exact_json_object(
    raw: object,
    *,
    expected_keys: set[str],
    label: str,
) -> dict[str, object]:
    if type(raw) is not dict:
        msg = f"{label} must be a JSON object"
        raise OratsArchiveIntegrityError(msg)
    mapping = cast("dict[object, object]", raw)
    if set(mapping) != expected_keys or any(type(key) is not str for key in mapping):
        msg = f"{label} keys must be exactly {sorted(expected_keys)!r}"
        raise OratsArchiveIntegrityError(msg)
    return cast("dict[str, object]", mapping)


@dataclass(frozen=True, slots=True)
class OratsFtpsCredentials:
    """In-memory ORATS credentials whose password is excluded from representations."""

    username: str
    password: str = field(repr=False)

    def __post_init__(self) -> None:
        """Reject empty credentials before opening a socket."""
        if not self.username:
            msg = "ORATS FTPS username must be nonempty"
            raise OratsArchiveConfigurationError(msg)
        if not self.password:
            msg = "ORATS FTPS password must be nonempty"
            raise OratsArchiveConfigurationError(msg)

    @classmethod
    def from_environment(
        cls,
        *,
        username_variable: str = ORATS_FTPS_USER_ENV,
        password_variable: str = ORATS_FTPS_PASSWORD_ENV,
    ) -> Self:
        """Read credentials from two explicit environment variables.

        No dotenv file, keychain search, prompt, or alternate variable is attempted.
        """
        username = os.environ.get(username_variable)
        password = os.environ.get(password_variable)
        missing = tuple(
            name
            for name, value in (
                (username_variable, username),
                (password_variable, password),
            )
            if value is None
        )
        if missing:
            msg = f"missing required ORATS credential environment variables: {missing!r}"
            raise OratsArchiveConfigurationError(
                msg,
            )
        return cls(username=username or "", password=password or "")


@dataclass(frozen=True, slots=True)
class OratsFtpsSpec:
    """Explicit encrypted transport settings for one archive endpoint."""

    host: str = ORATS_DEFAULT_FTPS_HOST
    port: int = ORATS_DEFAULT_FTPS_PORT
    timeout_seconds: float = ORATS_DEFAULT_FTPS_TIMEOUT_SECONDS
    transfer_block_bytes: int = ORATS_DEFAULT_TRANSFER_BLOCK_BYTES

    def __post_init__(self) -> None:
        """Validate values before connecting."""
        if not self.host or self.host.strip() != self.host:
            msg = "ORATS FTPS host must be nonempty and have no surrounding whitespace"
            raise OratsArchiveConfigurationError(
                msg,
            )
        if not _MINIMUM_TCP_PORT <= self.port <= _MAXIMUM_TCP_PORT:
            msg = "ORATS FTPS port must be in [1, 65535]"
            raise OratsArchiveConfigurationError(msg)
        if not self.timeout_seconds > 0.0:
            msg = "ORATS FTPS timeout must be positive"
            raise OratsArchiveConfigurationError(msg)
        if self.transfer_block_bytes < _MINIMUM_TRANSFER_BLOCK_BYTES:
            msg = "ORATS transfer block must be at least 65536 bytes"
            raise OratsArchiveConfigurationError(
                msg,
            )


@dataclass(frozen=True, slots=True, order=True)
class OratsArchiveEntry:
    """One immutable remote daily archive identity observed through MLSD."""

    trade_date: date
    remote_path: str
    size_bytes: int
    modified_at_utc: datetime

    def __post_init__(self) -> None:
        """Reject ambiguous or internally inconsistent remote identities."""
        expected_name = orats_archive_name(self.trade_date)
        path = PurePosixPath(self.remote_path)
        if not path.is_absolute() or path.name != expected_name:
            raise OratsArchiveIntegrityError(
                reason=f"expected absolute remote path ending in {expected_name!r}",
                path=self.remote_path,
            )
        expected_directory = orats_remote_directory(self.trade_date.year)
        if str(path.parent) != expected_directory:
            raise OratsArchiveIntegrityError(
                reason=f"expected remote directory {expected_directory!r}",
                path=self.remote_path,
            )
        if self.size_bytes <= 0:
            raise OratsArchiveIntegrityError(
                reason="remote archive size must be positive",
                path=self.remote_path,
            )
        if self.modified_at_utc.tzinfo != UTC:
            raise OratsArchiveIntegrityError(
                reason="MLSD modification time must be timezone-aware UTC",
                path=self.remote_path,
            )

    def as_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible identity mapping."""
        return {
            "modified_at_utc": self.modified_at_utc.isoformat().replace("+00:00", "Z"),
            "remote_path": self.remote_path,
            "size_bytes": self.size_bytes,
            "trade_date": self.trade_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse one exact manifest entry without coercing unknown shapes."""
        expected_keys = {"modified_at_utc", "remote_path", "size_bytes", "trade_date"}
        mapping = _exact_json_object(raw, expected_keys=expected_keys, label="archive entry")
        trade_date_raw = mapping["trade_date"]
        remote_path = mapping["remote_path"]
        size_bytes = mapping["size_bytes"]
        modified_raw = mapping["modified_at_utc"]
        if type(trade_date_raw) is not str or type(remote_path) is not str:
            msg = "archive entry fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        if type(size_bytes) is not int or type(modified_raw) is not str:
            msg = "archive entry fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        if not modified_raw.endswith("Z"):
            msg = "archive entry fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        try:
            trade_date = date.fromisoformat(trade_date_raw)
            modified = datetime.fromisoformat(f"{modified_raw[:-1]}+00:00")
        except ValueError as error:
            msg = "archive entry fields are malformed"
            raise OratsArchiveIntegrityError(msg) from error
        return cls(
            trade_date=trade_date,
            remote_path=remote_path,
            size_bytes=size_bytes,
            modified_at_utc=modified,
        )


@dataclass(frozen=True, slots=True)
class OratsArchiveInventory:
    """A bounded remote listing with a content identity independent of observation time."""

    host: str
    start_date: date
    end_date: date
    observed_at_utc: datetime
    entries: tuple[OratsArchiveEntry, ...]
    entries_sha256: str
    schema_version: int = ORATS_ARCHIVE_INVENTORY_VERSION

    def __post_init__(self) -> None:
        """Verify ordering, range, uniqueness, and the declared entry hash."""
        if self.schema_version != ORATS_ARCHIVE_INVENTORY_VERSION:
            msg = "unsupported archive inventory schema version"
            raise OratsArchiveIntegrityError(msg)
        if not self.host:
            msg = "archive inventory host must be nonempty"
            raise OratsArchiveIntegrityError(msg)
        if self.start_date > self.end_date:
            msg = "inventory start date must not exceed end date"
            raise OratsArchiveIntegrityError(msg)
        if self.observed_at_utc.tzinfo != UTC:
            msg = "inventory observation time must be UTC"
            raise OratsArchiveIntegrityError(msg)
        if not self.entries:
            msg = "archive inventory must contain at least one entry"
            raise OratsArchiveIntegrityError(msg)
        dates = tuple(entry.trade_date for entry in self.entries)
        if dates != tuple(sorted(dates)) or len(dates) != len(set(dates)):
            msg = "archive inventory entries must be strictly date ordered and unique"
            raise OratsArchiveIntegrityError(
                msg,
            )
        if dates[0] < self.start_date or dates[-1] > self.end_date:
            msg = "archive inventory entry lies outside its date range"
            raise OratsArchiveIntegrityError(msg)
        actual_hash = archive_entries_sha256(self.entries)
        if self.entries_sha256 != actual_hash:
            msg = (
                f"archive entry hash mismatch: declared={self.entries_sha256}, actual={actual_hash}"
            )
            raise OratsArchiveIntegrityError(
                msg,
            )

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical manifest mapping."""
        return {
            "end_date": self.end_date.isoformat(),
            "entries": [entry.as_dict() for entry in self.entries],
            "entries_sha256": self.entries_sha256,
            "host": self.host,
            "observed_at_utc": self.observed_at_utc.isoformat().replace("+00:00", "Z"),
            "schema_version": self.schema_version,
            "start_date": self.start_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and fully revalidate a manifest mapping."""
        expected_keys = {
            "end_date",
            "entries",
            "entries_sha256",
            "host",
            "observed_at_utc",
            "schema_version",
            "start_date",
        }
        mapping = _exact_json_object(raw, expected_keys=expected_keys, label="archive inventory")
        schema_version = mapping["schema_version"]
        host = mapping["host"]
        start_raw = mapping["start_date"]
        end_raw = mapping["end_date"]
        observed_raw = mapping["observed_at_utc"]
        entries_hash = mapping["entries_sha256"]
        entries_raw = mapping["entries"]
        if type(schema_version) is not int or type(host) is not str:
            msg = "archive inventory fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        if type(start_raw) is not str or type(end_raw) is not str:
            msg = "archive inventory fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        if type(observed_raw) is not str or not observed_raw.endswith("Z"):
            msg = "archive inventory fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        if type(entries_hash) is not str or type(entries_raw) is not list:
            msg = "archive inventory fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        try:
            start_date = date.fromisoformat(start_raw)
            end_date = date.fromisoformat(end_raw)
            observed = datetime.fromisoformat(f"{observed_raw[:-1]}+00:00")
        except ValueError as error:
            msg = "archive inventory fields are malformed"
            raise OratsArchiveIntegrityError(msg) from error
        return cls(
            schema_version=schema_version,
            host=host,
            start_date=start_date,
            end_date=end_date,
            observed_at_utc=observed,
            entries=tuple(
                OratsArchiveEntry.from_dict(entry) for entry in cast("list[object]", entries_raw)
            ),
            entries_sha256=entries_hash,
        )


@dataclass(frozen=True, slots=True)
class OratsArchiveReceipt:
    """Cryptographic receipt for one completed local raw archive."""

    entry: OratsArchiveEntry
    local_name: str
    sha256: str
    schema_version: int = ORATS_ARCHIVE_RECEIPT_VERSION

    def __post_init__(self) -> None:
        """Validate local name and digest syntax."""
        if self.schema_version != ORATS_ARCHIVE_RECEIPT_VERSION:
            msg = "unsupported archive receipt schema version"
            raise OratsArchiveIntegrityError(msg)
        if self.local_name != orats_archive_name(self.entry.trade_date):
            msg = "archive receipt local name is not canonical"
            raise OratsArchiveIntegrityError(msg)
        if not _is_sha256(self.sha256):
            msg = "archive receipt SHA-256 is malformed"
            raise OratsArchiveIntegrityError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical receipt mapping."""
        return {
            "entry": self.entry.as_dict(),
            "local_name": self.local_name,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse and validate a receipt mapping."""
        expected_keys = {"entry", "local_name", "schema_version", "sha256"}
        mapping = _exact_json_object(raw, expected_keys=expected_keys, label="archive receipt")
        local_name = mapping["local_name"]
        schema_version = mapping["schema_version"]
        sha256 = mapping["sha256"]
        if type(local_name) is not str or type(schema_version) is not int:
            msg = "archive receipt fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        if type(sha256) is not str:
            msg = "archive receipt fields are malformed"
            raise OratsArchiveIntegrityError(msg)
        return cls(
            entry=OratsArchiveEntry.from_dict(mapping["entry"]),
            local_name=local_name,
            schema_version=schema_version,
            sha256=sha256,
        )


@dataclass(frozen=True, slots=True)
class VerifiedOratsArchive:
    """A local raw archive whose size and bytes match its immutable sidecar receipt."""

    path: Path
    receipt: OratsArchiveReceipt


def orats_archive_name(trade_date: date) -> str:
    """Return the exact remote daily ZIP filename."""
    return f"ORATS_SMV_Strikes_{trade_date:%Y%m%d}.zip"


def orats_remote_directory(year: int) -> str:
    """Return the year-specific ORATS archive directory without fallback."""
    if not _FIRST_ARCHIVE_YEAR <= year <= _MAXIMUM_ARCHIVE_YEAR:
        msg = "ORATS archive year must be in [2007, 9999]"
        raise OratsArchiveConfigurationError(msg)
    root = "/smvstrikes_2007_2012" if year <= _LAST_LEGACY_ARCHIVE_YEAR else "/smvstrikes"
    return f"{root}/{year}"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def archive_entries_sha256(entries: Iterable[OratsArchiveEntry]) -> str:
    """Hash only ordered remote identities, excluding observation time."""
    payload = [entry.as_dict() for entry in entries]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _entry_from_mlsd(
    *,
    directory: str,
    year: int,
    name: str,
    facts: dict[str, str],
) -> OratsArchiveEntry | None:
    match = _ARCHIVE_NAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    date_digits = match.group("date")
    try:
        trade_date = date.fromisoformat(
            f"{date_digits[:4]}-{date_digits[4:6]}-{date_digits[6:]}",
        )
    except ValueError as error:
        msg = "archive filename contains an invalid date"
        raise OratsArchiveIntegrityError(msg, name) from error
    remote_path = str(PurePosixPath(directory) / name)
    if trade_date.year != year:
        raise OratsArchiveIntegrityError(
            reason=f"archive filename year {trade_date.year} disagrees with directory year {year}",
            path=remote_path,
        )
    if facts.get("type") != "file":
        msg = "matching archive entry is not a file"
        raise OratsArchiveIntegrityError(msg, remote_path)
    size_raw = facts.get("size")
    modify_raw = facts.get("modify")
    if size_raw is None or modify_raw is None:
        msg = "matching archive entry lacks required MLSD size/modify facts"
        raise OratsArchiveIntegrityError(
            msg,
            remote_path,
        )
    if not size_raw.isascii() or not size_raw.isdecimal():
        msg = "MLSD size is malformed"
        raise OratsArchiveIntegrityError(msg, remote_path)
    if _MLSD_MODIFY_PATTERN.fullmatch(modify_raw) is None:
        msg = "MLSD modification time is malformed"
        raise OratsArchiveIntegrityError(msg, remote_path)
    modified = datetime.fromisoformat(
        f"{modify_raw[:4]}-{modify_raw[4:6]}-{modify_raw[6:8]}T"
        f"{modify_raw[8:10]}:{modify_raw[10:12]}:{modify_raw[12:]}+00:00",
    )
    return OratsArchiveEntry(
        trade_date=trade_date,
        remote_path=remote_path,
        size_bytes=int(size_raw),
        modified_at_utc=modified,
    )


def _entry_from_mlst_response(remote_path: str, response: str) -> OratsArchiveEntry:
    path = PurePosixPath(remote_path)
    fact_lines = tuple(
        line.strip() for line in response.splitlines() if "=" in line and ";" in line
    )
    if len(fact_lines) != 1:
        msg = "MLST response must contain exactly one machine-readable fact line"
        raise OratsArchiveIntegrityError(msg, remote_path)
    try:
        facts_text, returned_path = fact_lines[0].split(maxsplit=1)
        fact_items = tuple(item for item in facts_text.split(";") if item)
        fact_pairs = tuple(item.split("=", maxsplit=1) for item in fact_items)
    except (TypeError, ValueError) as error:
        msg = "MLST response facts are malformed"
        raise OratsArchiveIntegrityError(msg, remote_path) from error
    normalized_pairs = tuple((key.lower(), value) for key, value in fact_pairs)
    facts = dict(normalized_pairs)
    if len(facts) != len(normalized_pairs):
        msg = "MLST response contains duplicate fact names"
        raise OratsArchiveIntegrityError(msg, remote_path)
    if PurePosixPath(returned_path).name != path.name:
        msg = "MLST response filename disagrees with the requested archive"
        raise OratsArchiveIntegrityError(msg, remote_path)
    entry = _entry_from_mlsd(
        directory=str(path.parent),
        year=int(path.parent.name),
        name=path.name,
        facts=facts,
    )
    if entry is None:
        msg = "MLST response path is not a canonical ORATS archive"
        raise OratsArchiveIntegrityError(msg, remote_path)
    return entry


def _file_sha256(path: Path, block_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        msg = "temporary manifest already exists"
        raise OratsArchiveIntegrityError(msg, str(temporary))
    payload = _canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_orats_archive_inventory(path: str | Path, inventory: OratsArchiveInventory) -> None:
    """Atomically write a new inventory; existing files are never overwritten."""
    destination = Path(path).expanduser()
    if destination.exists():
        msg = "archive inventory already exists"
        raise OratsArchiveIntegrityError(msg, str(destination))
    _write_json_atomic(destination, inventory.as_dict())


def read_orats_archive_inventory(path: str | Path) -> OratsArchiveInventory:
    """Read and fully revalidate one local inventory."""
    source = Path(path).expanduser()
    if not source.is_file():
        msg = "archive inventory does not exist"
        raise OratsArchiveIntegrityError(msg, str(source))
    try:
        raw = json.loads(source.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "archive inventory is unreadable"
        raise OratsArchiveIntegrityError(msg, str(source)) from error
    return OratsArchiveInventory.from_dict(raw)


def load_verified_orats_archive(path: str | Path) -> VerifiedOratsArchive:
    """Load and cryptographically verify one completed raw archive and sidecar receipt."""
    archive = Path(path).expanduser()
    if not archive.is_file() or archive.is_symlink():
        msg = "raw archive must be an existing non-symlink file"
        raise OratsArchiveIntegrityError(msg, str(archive))
    receipt_path = archive.with_name(f"{archive.name}.receipt.json")
    if not receipt_path.is_file() or receipt_path.is_symlink():
        msg = "raw archive sidecar receipt must be an existing non-symlink file"
        raise OratsArchiveIntegrityError(msg, str(receipt_path))
    receipt = _read_orats_archive_receipt(receipt_path)
    if receipt.local_name != archive.name:
        msg = "raw archive filename disagrees with its sidecar receipt"
        raise OratsArchiveIntegrityError(msg, str(archive))
    if archive.stat().st_size != receipt.entry.size_bytes:
        msg = "raw archive byte count disagrees with its sidecar receipt"
        raise OratsArchiveIntegrityError(msg, str(archive))
    actual_hash = _file_sha256(archive, ORATS_DEFAULT_TRANSFER_BLOCK_BYTES)
    if actual_hash != receipt.sha256:
        msg = "raw archive bytes disagree with its sidecar receipt SHA-256"
        raise OratsArchiveIntegrityError(msg, str(archive))
    return VerifiedOratsArchive(path=archive, receipt=receipt)


class OratsFtpsClient:
    """One explicit-TLS ORATS session with no plaintext or listing fallback."""

    def __init__(self, spec: OratsFtpsSpec, credentials: OratsFtpsCredentials) -> None:
        """Store validated connection inputs without opening a socket."""
        self._spec = spec
        self._credentials = credentials
        self._ftp: ftplib.FTP_TLS | None = None

    def __enter__(self) -> Self:
        """Open, authenticate, and protect both FTPS channels."""
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        # Ruff's generic FTP warning includes FTP_TLS; plaintext FTP is never constructed here.
        ftp = ftplib.FTP_TLS(  # noqa: S321
            context=context,
            timeout=self._spec.timeout_seconds,
        )
        try:
            ftp.connect(self._spec.host, self._spec.port, timeout=self._spec.timeout_seconds)
            ftp.login(self._credentials.username, self._credentials.password)
            ftp.prot_p()
            ftp.set_pasv(True)
        except (OSError, ftplib.Error, ssl.SSLError) as error:
            with contextlib.suppress(OSError):
                ftp.close()
            msg = "connect"
            raise OratsArchiveTransportError(msg, None, str(error)) from error
        if not isinstance(ftp.sock, ssl.SSLSocket):
            ftp.close()
            msg = "connect"
            raise OratsArchiveTransportError(
                msg,
                None,
                "control channel is not TLS protected",
            )
        self._ftp = ftp
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Close the session without hiding the active exception."""
        ftp = self._ftp
        self._ftp = None
        if ftp is None:
            return
        try:
            ftp.quit()
        except (OSError, ftplib.Error, ssl.SSLError):
            ftp.close()

    def _session(self) -> ftplib.FTP_TLS:
        if self._ftp is None:
            msg = "ORATS FTPS client must be used inside its context manager"
            raise OratsArchiveConfigurationError(
                msg,
            )
        return self._ftp

    def _list_year(self, year: int) -> tuple[OratsArchiveEntry, ...]:
        directory = orats_remote_directory(year)
        try:
            listed = tuple(
                self._session().mlsd(directory, facts=["type", "size", "modify"]),
            )
        except (OSError, ftplib.Error, ssl.SSLError) as error:
            msg = "MLSD"
            raise OratsArchiveTransportError(msg, directory, str(error)) from error
        entries = tuple(
            entry
            for name, facts in listed
            if (
                entry := _entry_from_mlsd(
                    directory=directory,
                    year=year,
                    name=name,
                    facts=facts,
                )
            )
            is not None
        )
        if not entries:
            msg = "year contains no matching archives"
            raise OratsArchiveIntegrityError(msg, directory)
        dates = tuple(entry.trade_date for entry in entries)
        if len(dates) != len(set(dates)):
            msg = "year contains duplicate archive dates"
            raise OratsArchiveIntegrityError(msg, directory)
        return tuple(sorted(entries))

    def inventory(
        self,
        *,
        start_date: date,
        end_date: date,
        observed_at_utc: datetime | None = None,
    ) -> OratsArchiveInventory:
        """List and hash all matching daily archives in a closed civil-date interval."""
        if start_date > end_date:
            msg = "inventory start date must not exceed end date"
            raise OratsArchiveConfigurationError(msg)
        observed = observed_at_utc or datetime.now(UTC)
        if observed.tzinfo != UTC:
            msg = "inventory observation time must be UTC"
            raise OratsArchiveConfigurationError(msg)
        entries = tuple(
            entry
            for year in range(start_date.year, end_date.year + 1)
            for entry in self._list_year(year)
            if start_date <= entry.trade_date <= end_date
        )
        if not entries:
            msg = "requested date range contains no matching archives"
            raise OratsArchiveIntegrityError(msg)
        return OratsArchiveInventory(
            host=self._spec.host,
            start_date=start_date,
            end_date=end_date,
            observed_at_utc=observed,
            entries=entries,
            entries_sha256=archive_entries_sha256(entries),
        )

    def _verify_remote_entry(self, expected: OratsArchiveEntry) -> None:
        try:
            response = self._session().sendcmd(f"MLST {expected.remote_path}")
        except (OSError, ftplib.Error, ssl.SSLError) as error:
            operation = "MLST"
            raise OratsArchiveTransportError(
                operation,
                expected.remote_path,
                str(error),
            ) from error
        observed = _entry_from_mlst_response(expected.remote_path, response)
        if observed != expected:
            msg = "remote archive identity changed after inventory"
            raise OratsArchiveIntegrityError(
                msg,
                expected.remote_path,
            )

    def _verified_existing_receipt(
        self,
        *,
        entry: OratsArchiveEntry,
        final_path: Path,
        receipt_path: Path,
    ) -> OratsArchiveReceipt | None:
        if not final_path.exists() and not receipt_path.exists():
            return None
        if not final_path.is_file() or not receipt_path.is_file():
            msg = "completed archive and receipt must either both be regular files or both absent"
            raise OratsArchiveIntegrityError(msg, str(final_path))
        receipt = _read_orats_archive_receipt(receipt_path)
        if receipt.entry != entry or receipt.local_name != final_path.name:
            msg = "existing archive receipt disagrees with requested remote identity"
            raise OratsArchiveIntegrityError(msg, str(receipt_path))
        if final_path.stat().st_size != entry.size_bytes:
            msg = "existing archive size disagrees with receipt"
            raise OratsArchiveIntegrityError(msg, str(final_path))
        actual_hash = _file_sha256(final_path, self._spec.transfer_block_bytes)
        if actual_hash != receipt.sha256:
            msg = "existing archive bytes disagree with receipt SHA-256"
            raise OratsArchiveIntegrityError(msg, str(final_path))
        return receipt

    @staticmethod
    def _partial_offset(partial_path: Path, entry: OratsArchiveEntry) -> int:
        if not partial_path.exists():
            return 0
        if not partial_path.is_file():
            msg = "partial archive is not a regular file"
            raise OratsArchiveIntegrityError(msg, str(partial_path))
        offset = partial_path.stat().st_size
        if offset > entry.size_bytes:
            msg = "partial archive is larger than the remote archive"
            raise OratsArchiveIntegrityError(msg, str(partial_path))
        return offset

    def _transfer_partial(
        self,
        *,
        entry: OratsArchiveEntry,
        partial_path: Path,
        offset: int,
    ) -> None:
        if offset == entry.size_bytes:
            return
        mode = "ab" if partial_path.exists() else "xb"
        try:
            with partial_path.open(mode) as destination:
                self._session().retrbinary(
                    f"RETR {entry.remote_path}",
                    destination.write,
                    blocksize=self._spec.transfer_block_bytes,
                    rest=offset or None,
                )
                destination.flush()
                os.fsync(destination.fileno())
        except (OSError, ftplib.Error, ssl.SSLError) as error:
            operation = "RETR"
            raise OratsArchiveTransportError(operation, entry.remote_path, str(error)) from error

    def _finalize_download(
        self,
        *,
        entry: OratsArchiveEntry,
        partial_path: Path,
        final_path: Path,
        receipt_path: Path,
    ) -> OratsArchiveReceipt:
        actual_size = partial_path.stat().st_size
        if actual_size != entry.size_bytes:
            msg = f"downloaded size {actual_size} disagrees with remote size {entry.size_bytes}"
            raise OratsArchiveIntegrityError(msg, str(partial_path))
        receipt = OratsArchiveReceipt(
            entry=entry,
            local_name=final_path.name,
            sha256=_file_sha256(partial_path, self._spec.transfer_block_bytes),
        )
        partial_path.replace(final_path)
        try:
            _write_json_atomic(receipt_path, receipt.as_dict())
        except BaseException:
            final_path.replace(partial_path)
            raise
        return receipt

    def download(
        self,
        entry: OratsArchiveEntry,
        destination_directory: str | Path,
    ) -> OratsArchiveReceipt:
        """Resume and verify one archive under an immutable sidecar receipt.

        A complete archive is reused only when its sidecar exactly matches the requested remote
        identity and its bytes still match the recorded SHA-256. A partial archive resumes from its
        exact byte count. If the server rejects REST, the operation fails and preserves the partial
        file; it never restarts or overwrites silently.
        """
        self._verify_remote_entry(entry)
        directory = Path(destination_directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / orats_archive_name(entry.trade_date)
        partial_path = directory / f".{final_path.name}.part"
        receipt_path = directory / f"{final_path.name}.receipt.json"

        existing = self._verified_existing_receipt(
            entry=entry,
            final_path=final_path,
            receipt_path=receipt_path,
        )
        if existing is not None:
            return existing
        offset = self._partial_offset(partial_path, entry)
        self._transfer_partial(entry=entry, partial_path=partial_path, offset=offset)
        self._verify_remote_entry(entry)
        return self._finalize_download(
            entry=entry,
            partial_path=partial_path,
            final_path=final_path,
            receipt_path=receipt_path,
        )


def _read_orats_archive_receipt(path: Path) -> OratsArchiveReceipt:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        msg = "archive receipt is unreadable"
        raise OratsArchiveIntegrityError(msg, str(path)) from error
    return OratsArchiveReceipt.from_dict(raw)


__all__ = [
    "ORATS_ARCHIVE_INVENTORY_VERSION",
    "ORATS_ARCHIVE_RECEIPT_VERSION",
    "ORATS_DEFAULT_FTPS_HOST",
    "ORATS_DEFAULT_FTPS_PORT",
    "ORATS_FTPS_PASSWORD_ENV",
    "ORATS_FTPS_USER_ENV",
    "OratsArchiveConfigurationError",
    "OratsArchiveEntry",
    "OratsArchiveError",
    "OratsArchiveIntegrityError",
    "OratsArchiveInventory",
    "OratsArchiveReceipt",
    "OratsArchiveTransportError",
    "OratsFtpsClient",
    "OratsFtpsCredentials",
    "OratsFtpsSpec",
    "VerifiedOratsArchive",
    "archive_entries_sha256",
    "load_verified_orats_archive",
    "orats_archive_name",
    "orats_remote_directory",
    "read_orats_archive_inventory",
    "write_orats_archive_inventory",
]
