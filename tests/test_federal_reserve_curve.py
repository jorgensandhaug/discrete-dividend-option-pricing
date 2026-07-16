"""Tests for the pinned Federal Reserve nominal Svensson curve projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING, Literal, cast

import polars as pl
import pytest

from discrete_dividend_study.cli import main
from discrete_dividend_study.federal_reserve_curve import (
    FederalReserveCurveContentError,
    FederalReserveCurveProjectionSpec,
    FederalReserveCurveReceipt,
    FederalReserveCurveSpecError,
    FederalReserveCurveWriteSpec,
    load_verified_federal_reserve_curve,
    materialize_federal_reserve_curve,
)
from discrete_dividend_study.nss_rates import NSS_PARAMETER_DTYPES

if TYPE_CHECKING:
    from pathlib import Path


START_DATE = date(2011, 1, 1)
END_DATE_EXCLUSIVE = date(2011, 1, 6)
SELECTED_SOURCE_ROWS = 3
OUTPUT_ROWS = 2
SOURCE_HEADER = "Date,BETA0,BETA1,BETA2,BETA3,SVENY01,TAU1,TAU2"
SOURCE_ROWS = (
    "2010-12-31,4.0,-1.0,2.0,-0.5,3.7300,1.2,2.4",
    "2011-01-03,4.1,-1.1,2.1,-0.6,3.7322,1.3,2.5",
    "2011-01-04,NA,NA,NA,NA,NA,NA,NA",
    "2011-01-05,4.2,-1.2,2.2,-0.7,3.7330,1.4,2.6",
    "2011-01-06,4.3,-1.3,2.3,-0.8,3.7328,1.5,2.7",
)


def _spec() -> FederalReserveCurveProjectionSpec:
    return FederalReserveCurveProjectionSpec(
        start_date=START_DATE,
        end_date_exclusive=END_DATE_EXCLUSIVE,
    )


def _write_source(path: Path, rows: tuple[str, ...] = SOURCE_ROWS) -> tuple[str, int]:
    preamble = tuple(f"Federal Reserve metadata line {index}" for index in range(9))
    payload = "\n".join((*preamble, SOURCE_HEADER, *rows, "")).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def test_projection_is_unit_explicit_receipted_and_exactly_reusable(tmp_path: Path) -> None:
    """Published percentage values become pinned decimal continuously compounded rates."""
    source = tmp_path / "feds200628.csv"
    source_sha256, source_size = _write_source(source)
    first = materialize_federal_reserve_curve(
        source,
        expected_source_sha256=source_sha256,
        expected_source_size_bytes=source_size,
        spec=_spec(),
        output_root=tmp_path / "projected",
    )
    frame = pl.read_parquet(first.path)
    assert frame.schema == pl.Schema(NSS_PARAMETER_DTYPES)
    assert frame.get_column("date").to_list() == [date(2011, 1, 3), date(2011, 1, 5)]
    assert frame.get_column("beta0").to_list() == pytest.approx([0.041, 0.042])
    assert frame.get_column("shortest_rate").to_list() == pytest.approx([0.03, 0.03])
    assert frame.get_column("shortest_maturity").to_list() == [1.0e-8, 1.0e-8]
    assert frame.get_column("tau1").to_list() == [1.3, 1.4]
    assert first.receipt.selected_source_rows == SELECTED_SOURCE_ROWS
    assert first.receipt.excluded_unavailable_rows == 1
    assert first.receipt.output_rows == OUTPUT_ROWS
    assert FederalReserveCurveReceipt.from_dict(first.receipt.as_dict()) == first.receipt
    assert load_verified_federal_reserve_curve(first.path) == first

    second = materialize_federal_reserve_curve(
        source,
        expected_source_sha256=source_sha256,
        expected_source_size_bytes=source_size,
        spec=_spec(),
        output_root=tmp_path / "projected",
    )
    assert second == first


def test_source_bytes_and_partial_rows_are_never_repaired_or_inferred(tmp_path: Path) -> None:
    """Source identity drift and partial publication rows fail before an artifact is trusted."""
    source = tmp_path / "feds200628.csv"
    source_sha256, source_size = _write_source(source)
    with pytest.raises(FederalReserveCurveContentError, match="SHA-256"):
        materialize_federal_reserve_curve(
            source,
            expected_source_sha256="0" * 64,
            expected_source_size_bytes=source_size,
            spec=_spec(),
            output_root=tmp_path / "wrong-hash",
        )
    with pytest.raises(FederalReserveCurveContentError, match="byte count"):
        materialize_federal_reserve_curve(
            source,
            expected_source_sha256=source_sha256,
            expected_source_size_bytes=source_size + 1,
            spec=_spec(),
            output_root=tmp_path / "wrong-size",
        )

    partial = tmp_path / "partial.csv"
    partial_rows = (*SOURCE_ROWS[:2], "2011-01-04,4.0,NA,2.0,-0.5,3.7300,1.2,2.4")
    partial_sha256, partial_size = _write_source(partial, partial_rows)
    with pytest.raises(FederalReserveCurveContentError, match="partial_rows"):
        materialize_federal_reserve_curve(
            partial,
            expected_source_sha256=partial_sha256,
            expected_source_size_bytes=partial_size,
            spec=_spec(),
            output_root=tmp_path / "partial-output",
        )

    inconsistent = tmp_path / "inconsistent.csv"
    inconsistent_rows = (
        SOURCE_ROWS[0],
        "2011-01-03,4.1,-1.1,2.1,-0.6,9.9999,1.3,2.5",
    )
    inconsistent_sha256, inconsistent_size = _write_source(inconsistent, inconsistent_rows)
    with pytest.raises(FederalReserveCurveContentError, match="inconsistent_published_yields"):
        materialize_federal_reserve_curve(
            inconsistent,
            expected_source_sha256=inconsistent_sha256,
            expected_source_size_bytes=inconsistent_size,
            spec=_spec(),
            output_root=tmp_path / "inconsistent-output",
        )


def test_projection_specs_reject_unit_or_writer_substitution() -> None:
    """The financial units, one-year observation, and physical writer have no fallback."""
    with pytest.raises(FederalReserveCurveSpecError, match="scale"):
        replace(_spec(), rate_scale=1.0)
    with pytest.raises(FederalReserveCurveSpecError, match="numerical floor"):
        replace(_spec(), shortest_maturity_years=0.5)
    with pytest.raises(FederalReserveCurveSpecError, match="continuous"):
        replace(
            _spec(),
            source_zero_yield_compounding=cast("Literal['continuous']", "annual"),
        )
    with pytest.raises(FederalReserveCurveSpecError, match="durable streaming"):
        FederalReserveCurveWriteSpec(compression_level=9)


def test_cli_materializes_an_explicit_vintage_and_horizon(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command line binds source bytes and both boundaries into the request hash."""
    source = tmp_path / "feds200628.csv"
    source_sha256, source_size = _write_source(source)
    assert (
        main(
            (
                "materialize-fed-nominal-curve",
                "--source",
                str(source),
                "--expected-source-sha256",
                source_sha256,
                "--expected-source-size-bytes",
                str(source_size),
                "--start-date",
                START_DATE.isoformat(),
                "--end-date-exclusive",
                END_DATE_EXCLUSIVE.isoformat(),
                "--output-root",
                str(tmp_path / "projected"),
            )
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == OUTPUT_ROWS
    assert summary["selected_source_rows"] == SELECTED_SOURCE_ROWS
    assert summary["excluded_unavailable_rows"] == 1
    assert (
        load_verified_federal_reserve_curve(summary["output_path"]).receipt.output_sha256
        == (summary["output_sha256"])
    )
