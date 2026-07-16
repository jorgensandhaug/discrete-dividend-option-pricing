"""Strict byte, row, and ordered-schema identities for external Parquet sources."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

PINNED_PARQUET_HASH_BLOCK_BYTES: Final = 8 * 1024 * 1024
_SHA256_LENGTH: Final = 64


class PinnedParquetError(RuntimeError):
    """Base class for an invalid externally supplied Parquet source."""


@dataclass(slots=True)
class PinnedParquetContentError(PinnedParquetError):
    """A source path, byte identity, row count, or ordered schema is invalid."""

    reason: str
    path: str | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing source rows."""
        location = "" if self.path is None else f" ({self.path})"
        return f"pinned Parquet verification failed{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class PinnedParquetIdentity:
    """Relocatable byte, row, and ordered-schema identity for one Parquet file."""

    name: str
    size_bytes: int
    sha256: str
    rows: int
    schema_sha256: str

    def __post_init__(self) -> None:
        """Reject malformed serialized identities."""
        if type(self.name) is not str or not self.name or Path(self.name).name != self.name:
            msg = "pinned Parquet source name is malformed"
            raise PinnedParquetContentError(msg)
        if (
            type(self.size_bytes) is not int
            or type(self.rows) is not int
            or self.size_bytes <= 0
            or self.rows <= 0
        ):
            msg = "pinned Parquet byte and row counts must be positive"
            raise PinnedParquetContentError(msg)
        if (
            type(self.sha256) is not str
            or type(self.schema_sha256) is not str
            or not _is_sha256(self.sha256)
            or not _is_sha256(self.schema_sha256)
        ):
            msg = "pinned Parquet SHA-256 is malformed"
            raise PinnedParquetContentError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical identity mapping."""
        return {
            "name": self.name,
            "rows": self.rows,
            "schema_sha256": self.schema_sha256,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Self:
        """Parse an exact canonical identity mapping without coercion."""
        if type(raw) is not dict:
            msg = "pinned Parquet identity must be a JSON object"
            raise PinnedParquetContentError(msg)
        mapping = cast("dict[object, object]", raw)
        expected = {"name", "rows", "schema_sha256", "sha256", "size_bytes"}
        if set(mapping) != expected or any(type(key) is not str for key in mapping):
            msg = f"pinned Parquet identity keys must be exactly {sorted(expected)!r}"
            raise PinnedParquetContentError(msg)
        typed = cast("dict[str, object]", mapping)
        return cls(
            name=cast("str", typed["name"]),
            size_bytes=cast("int", typed["size_bytes"]),
            sha256=cast("str", typed["sha256"]),
            rows=cast("int", typed["rows"]),
            schema_sha256=cast("str", typed["schema_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class VerifiedPinnedParquet:
    """One regular Parquet file verified against an explicit expected identity."""

    path: Path
    identity: PinnedParquetIdentity

    def scan(self) -> pl.LazyFrame:
        """Lazily scan the exact verified file."""
        return pl.scan_parquet(self.path, rechunk=False)


def parquet_schema_sha256(
    schema: Mapping[str, pl.DataType | type[pl.DataType]],
) -> str:
    """Hash an exact ordered mapping of column names and Polars dtypes."""
    payload = [(name, str(dtype)) for name, dtype in schema.items()]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def verify_pinned_parquet(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_schema: Mapping[str, pl.DataType | type[pl.DataType]],
) -> VerifiedPinnedParquet:
    """Verify one exact regular Parquet source without coercion or extra columns."""
    source = _validate_request(path, expected_sha256, expected_rows, expected_schema)
    initial = source.stat()
    observed_sha256 = _file_sha256(source)
    if observed_sha256 != expected_sha256:
        msg = "source bytes disagree with the expected SHA-256"
        raise PinnedParquetContentError(msg, str(source))
    try:
        scan = pl.scan_parquet(source)
        observed_schema = scan.collect_schema()
        observed_rows = scan.select(pl.len()).collect().item()
    except pl.exceptions.PolarsError as error:
        msg = "source is not a readable Parquet file"
        raise PinnedParquetContentError(msg, str(source)) from error
    expected_polars_schema = pl.Schema(expected_schema)
    if observed_schema != expected_polars_schema:
        msg = "source ordered schema disagrees with the expected schema"
        raise PinnedParquetContentError(msg, str(source))
    if type(observed_rows) is not int or observed_rows != expected_rows:
        msg = "source row count disagrees with the expected row count"
        raise PinnedParquetContentError(msg, str(source))
    final = source.stat()
    if (
        final.st_dev != initial.st_dev
        or final.st_ino != initial.st_ino
        or final.st_size != initial.st_size
        or final.st_mtime_ns != initial.st_mtime_ns
    ):
        msg = "source changed during verification"
        raise PinnedParquetContentError(msg, str(source))
    identity = PinnedParquetIdentity(
        name=source.name,
        size_bytes=initial.st_size,
        sha256=observed_sha256,
        rows=observed_rows,
        schema_sha256=parquet_schema_sha256(expected_schema),
    )
    return VerifiedPinnedParquet(path=source, identity=identity)


def _validate_request(
    path: str | Path,
    expected_sha256: str,
    expected_rows: int,
    expected_schema: Mapping[str, pl.DataType | type[pl.DataType]],
) -> Path:
    source = Path(path).expanduser()
    if not source.is_file() or source.is_symlink():
        msg = "source must be an existing non-symlink file"
        raise PinnedParquetContentError(msg, str(source))
    if source.stat().st_size <= 0:
        msg = "source must be nonempty"
        raise PinnedParquetContentError(msg, str(source))
    if not _is_sha256(expected_sha256):
        msg = "expected source SHA-256 is malformed"
        raise PinnedParquetContentError(msg)
    if type(expected_rows) is not int or expected_rows <= 0:
        msg = "expected source rows must be a positive plain int"
        raise PinnedParquetContentError(msg)
    if not expected_schema:
        msg = "expected source schema must be nonempty"
        raise PinnedParquetContentError(msg)
    return source


def verify_pinned_parquet_unchanged(source: VerifiedPinnedParquet) -> None:
    """Rehash a previously verified source after a consuming operation."""
    path = source.path
    if not path.is_file() or path.is_symlink():
        msg = "source changed after verification"
        raise PinnedParquetContentError(msg, str(path))
    if path.stat().st_size != source.identity.size_bytes:
        msg = "source byte count changed after verification"
        raise PinnedParquetContentError(msg, str(path))
    if _file_sha256(path) != source.identity.sha256:
        msg = "source SHA-256 changed after verification"
        raise PinnedParquetContentError(msg, str(path))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(PINNED_PARQUET_HASH_BLOCK_BYTES):
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


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "PINNED_PARQUET_HASH_BLOCK_BYTES",
    "PinnedParquetContentError",
    "PinnedParquetError",
    "PinnedParquetIdentity",
    "VerifiedPinnedParquet",
    "parquet_schema_sha256",
    "verify_pinned_parquet",
    "verify_pinned_parquet_unchanged",
]
