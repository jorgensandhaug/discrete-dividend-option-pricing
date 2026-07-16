"""Strict ORATS option-normalization tests using synthetic licensed-data shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from discrete_dividend_study.data.providers.orats.options import (
    ORATS_NORMALIZED_OPTION_DTYPES,
    ORATS_RAW_OPTION_DTYPES,
    OratsOptionSchemaError,
    OratsOptionSourceError,
    canonicalize_raw_orats_options,
    scan_raw_orats_csv,
)

if TYPE_CHECKING:
    from pathlib import Path


def raw_option_frame() -> pl.LazyFrame:
    """Return two raw rows with the exact declared scan types."""
    return pl.LazyFrame(
        {
            "ticker": ["DUK", "VZ"],
            "trade_date": ["1/2/2015", "9/27/2024"],
            "expirDate": ["2/20/2015", "12/20/2024"],
            "strike": [80.0, 45.0],
            "cAskPx": [4.2, 2.3],
            "cBidPx": [4.1, 2.2],
            "pAskPx": [1.2, 0.8],
            "pBidPx": [1.1, 0.7],
            "stkPx": [82.0, 44.0],
            "cVolu": [20.0, 30.0],
            "pVolu": [10.0, 15.0],
            "cOi": [200.0, 300.0],
            "pOi": [100.0, 150.0],
            "cValue": [4.15, 2.25],
            "pValue": [1.15, 0.75],
            "cMidIv": [0.2, 0.3],
            "pMidIv": [0.21, 0.31],
        },
        schema=ORATS_RAW_OPTION_DTYPES,
    )


def test_normalization_is_lazy_exactly_typed_and_order_preserving() -> None:
    """The shared thesis projection renames only declared columns and preserves rows."""
    normalized = canonicalize_raw_orats_options(raw_option_frame())
    assert dict(normalized.collect_schema()) == ORATS_NORMALIZED_OPTION_DTYPES
    collected = normalized.collect()
    assert collected.get_column("ticker").to_list() == ["DUK", "VZ"]
    assert collected.get_column("quote_date").dt.to_string().to_list() == [
        "2015-01-02",
        "2024-09-27",
    ]
    assert collected.get_column("call_ask").to_list() == pytest.approx([4.2, 2.3])


def test_normalization_rejects_dtype_substitution_before_execution() -> None:
    """A Float64 quote column is not silently narrowed into the as-submitted representation."""
    wrong = raw_option_frame().with_columns(pl.col("cAskPx").cast(pl.Float64))
    with pytest.raises(OratsOptionSchemaError, match="dtype mismatches"):
        canonicalize_raw_orats_options(wrong)


def test_scan_contract_parses_owned_columns_without_inferring_financial_types(
    tmp_path: Path,
) -> None:
    """The local CSV scanner owns option Float32 and curve Float64 parse identities."""
    source = tmp_path / "ORATS_SMV_Strikes_20150102.csv"
    source.write_text(
        "ticker,trade_date,expirDate,strike,cAskPx,cBidPx,pAskPx,pBidPx,stkPx,"
        "cVolu,pVolu,cOi,pOi,cValue,pValue,cMidIv,pMidIv,yte,iRate,unused\n"
        "DUK,1/2/2015,2/20/2015,80,4.2,4.1,1.2,1.1,82,20,10,200,100,"
        "4.15,1.15,0.2,0.21,0.1342,0.0015,kept-as-text\n",
    )
    raw = scan_raw_orats_csv(source)
    schema = raw.collect_schema()
    assert schema["cAskPx"] == pl.Float32
    assert schema["yte"] == pl.Float64
    assert schema["iRate"] == pl.Float64
    assert schema["unused"] == pl.String
    assert canonicalize_raw_orats_options(raw).collect().height == 1


def test_scan_contract_fails_on_bad_sources_and_malformed_numbers(tmp_path: Path) -> None:
    """Unsupported paths and malformed owned numeric fields do not produce partial rows."""
    with pytest.raises(OratsOptionSourceError, match=r"expected a \.csv"):
        scan_raw_orats_csv(tmp_path / "delivery.zip")
    with pytest.raises(OratsOptionSourceError, match="does not exist"):
        scan_raw_orats_csv(tmp_path / "missing.csv")

    malformed = tmp_path / "malformed.csv"
    malformed.write_text(
        "ticker,trade_date,expirDate,strike,cAskPx,cBidPx,pAskPx,pBidPx,stkPx,"
        "cVolu,pVolu,cOi,pOi,cValue,pValue,cMidIv,pMidIv,yte,iRate\n"
        "DUK,1/2/2015,2/20/2015,80,not-a-number,4.1,1.2,1.1,82,20,10,200,"
        "100,4.15,1.15,0.2,0.21,0.1342,0.0015\n",
    )
    with pytest.raises(pl.exceptions.PolarsError):
        canonicalize_raw_orats_options(scan_raw_orats_csv(malformed)).collect()
