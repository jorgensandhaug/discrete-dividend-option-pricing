"""Tests for strict identities of externally supplied Parquet sources."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import TYPE_CHECKING

import polars as pl
import pytest

from discrete_dividend_study.pinned_parquet import (
    PinnedParquetContentError,
    PinnedParquetIdentity,
    parquet_schema_sha256,
    verify_pinned_parquet,
    verify_pinned_parquet_unchanged,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = {"date": pl.Date, "value": pl.Float64}


def _write_source(path: Path) -> tuple[str, int]:
    frame = pl.DataFrame(
        {"date": [date(2024, 1, 2), date(2024, 1, 3)], "value": [0.01, 0.02]},
        schema=SCHEMA,
    )
    frame.write_parquet(path)
    return hashlib.sha256(path.read_bytes()).hexdigest(), frame.height


def test_exact_source_is_relocatable_typed_and_rescannable(tmp_path: Path) -> None:
    """An exact byte/row/schema identity exposes only the already verified file."""
    path = tmp_path / "source.parquet"
    sha256, rows = _write_source(path)
    verified = verify_pinned_parquet(
        path,
        expected_sha256=sha256,
        expected_rows=rows,
        expected_schema=SCHEMA,
    )
    assert verified.identity.name == path.name
    assert verified.identity.schema_sha256 == parquet_schema_sha256(SCHEMA)
    assert PinnedParquetIdentity.from_dict(verified.identity.as_dict()) == verified.identity
    assert verified.scan().collect().height == rows
    verify_pinned_parquet_unchanged(verified)


def test_wrong_bytes_rows_schema_and_later_mutation_fail(tmp_path: Path) -> None:
    """No caller expectation is inferred, coerced, or silently repaired."""
    path = tmp_path / "source.parquet"
    sha256, rows = _write_source(path)
    with pytest.raises(PinnedParquetContentError, match="expected SHA-256"):
        verify_pinned_parquet(
            path,
            expected_sha256="0" * 64,
            expected_rows=rows,
            expected_schema=SCHEMA,
        )
    with pytest.raises(PinnedParquetContentError, match="row count"):
        verify_pinned_parquet(
            path,
            expected_sha256=sha256,
            expected_rows=rows + 1,
            expected_schema=SCHEMA,
        )
    with pytest.raises(PinnedParquetContentError, match="ordered schema"):
        verify_pinned_parquet(
            path,
            expected_sha256=sha256,
            expected_rows=rows,
            expected_schema={"value": pl.Float64, "date": pl.Date},
        )

    verified = verify_pinned_parquet(
        path,
        expected_sha256=sha256,
        expected_rows=rows,
        expected_schema=SCHEMA,
    )
    size = path.stat().st_size
    path.write_bytes(b"x" * size)
    with pytest.raises(PinnedParquetContentError, match="SHA-256 changed"):
        verify_pinned_parquet_unchanged(verified)


def test_symlink_sources_are_rejected(tmp_path: Path) -> None:
    """A symlink cannot redirect a pinned source after path validation."""
    source = tmp_path / "source.parquet"
    sha256, rows = _write_source(source)
    link = tmp_path / "link.parquet"
    link.symlink_to(source)
    with pytest.raises(PinnedParquetContentError, match="non-symlink"):
        verify_pinned_parquet(
            link,
            expected_sha256=sha256,
            expected_rows=rows,
            expected_schema=SCHEMA,
        )


@pytest.mark.parametrize("value", [True, 1.0, "2"])
def test_serialized_identity_rejects_non_plain_integer_counts(value: object) -> None:
    """JSON-like values never receive Python's implicit numeric coercions."""
    raw = {
        "name": "source.parquet",
        "rows": value,
        "schema_sha256": "1" * 64,
        "sha256": "2" * 64,
        "size_bytes": 10,
    }
    with pytest.raises(PinnedParquetContentError, match="byte and row counts"):
        PinnedParquetIdentity.from_dict(raw)
