"""Lazy, strict normalization of licensed ORATS option-quote deliveries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .schema import require_columns

ORATS_RAW_OPTION_COLUMNS = (
    "ticker",
    "trade_date",
    "expirDate",
    "strike",
    "cAskPx",
    "cBidPx",
    "pAskPx",
    "pBidPx",
    "stkPx",
    "cVolu",
    "pVolu",
    "cOi",
    "pOi",
    "cValue",
    "pValue",
    "cMidIv",
    "pMidIv",
)
ORATS_RAW_OPTION_DTYPES = {
    "ticker": pl.String,
    "trade_date": pl.String,
    "expirDate": pl.String,
    "strike": pl.Float32,
    "cAskPx": pl.Float32,
    "cBidPx": pl.Float32,
    "pAskPx": pl.Float32,
    "pBidPx": pl.Float32,
    "stkPx": pl.Float32,
    "cVolu": pl.Float32,
    "pVolu": pl.Float32,
    "cOi": pl.Float32,
    "pOi": pl.Float32,
    "cValue": pl.Float32,
    "pValue": pl.Float32,
    "cMidIv": pl.Float32,
    "pMidIv": pl.Float32,
}
ORATS_RAW_SCAN_DTYPES = {
    **ORATS_RAW_OPTION_DTYPES,
    "yte": pl.Float64,
    "iRate": pl.Float64,
}
ORATS_NORMALIZED_OPTION_DTYPES = {
    "ticker": pl.String,
    "quote_date": pl.Date,
    "expiry_date": pl.Date,
    "strike": pl.Float32,
    "call_ask": pl.Float32,
    "call_bid": pl.Float32,
    "put_ask": pl.Float32,
    "put_bid": pl.Float32,
    "underlying_price": pl.Float32,
    "call_volume": pl.Float32,
    "put_volume": pl.Float32,
    "call_open_interest": pl.Float32,
    "put_open_interest": pl.Float32,
    "call_theoretical_value": pl.Float32,
    "put_theoretical_value": pl.Float32,
    "call_iv": pl.Float32,
    "put_iv": pl.Float32,
}


class OratsOptionError(ValueError):
    """Base class for an invalid ORATS option source or schema."""


@dataclass(slots=True)
class OratsOptionSourceError(OratsOptionError):
    """A local licensed delivery is absent or is not a supported CSV source."""

    source: Path
    reason: str

    def __str__(self) -> str:
        """Describe the source contract failure without inspecting rows."""
        return f"invalid ORATS CSV source {self.source}: {self.reason}"


@dataclass(slots=True)
class OratsOptionSchemaError(OratsOptionError):
    """Required raw option columns have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed dtypes without exposing licensed rows."""
        return f"ORATS option column dtype mismatches: {self.mismatches!r}"


def scan_raw_orats_csv(source: str | Path) -> pl.LazyFrame:
    """Lazily scan one uncompressed ORATS delivery under an explicit parse contract.

    The columns required by option normalization are parsed directly as the historical `Float32`
    representation. `yte` and `iRate` retain `Float64` for the independent curve adapter. Other
    vendor columns remain strings unless a downstream adapter explicitly owns their semantics.

    Raises:
        OratsOptionSourceError: If `source` is absent, not a file, or not a CSV path.

    """
    source = Path(source).expanduser()
    if source.suffix.lower() != ".csv":
        raise OratsOptionSourceError(source=source, reason="expected a .csv file")
    if not source.is_file():
        raise OratsOptionSourceError(source=source, reason="file does not exist")
    return pl.scan_csv(
        source,
        has_header=True,
        separator=",",
        schema_overrides=ORATS_RAW_SCAN_DTYPES,
        null_values="NULL",
        ignore_errors=False,
        infer_schema=False,
        encoding="utf8",
        low_memory=False,
        rechunk=False,
        try_parse_dates=False,
        glob=False,
        raise_if_empty=True,
        missing_columns="raise",
    )


def canonicalize_raw_orats_options(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Project raw vendor columns into the exact shared 17-column thesis representation.

    The transformation stays lazy, preserves row order and nulls, and never filters quotes. Date
    strings are parsed strictly from the observed `M/D/YYYY` delivery format. Numeric inputs must
    already have the declared `Float32` parse identity; no implicit coercion is performed here.

    Raises:
        MissingColumnsError: If a required raw source column is absent.
        OratsOptionSchemaError: If a required raw source column has the wrong dtype.

    """
    validated = require_columns(frame, ORATS_RAW_OPTION_COLUMNS)
    schema = validated.collect_schema()
    mismatches = tuple(
        (name, str(expected), str(schema[name]))
        for name, expected in ORATS_RAW_OPTION_DTYPES.items()
        if schema[name] != expected
    )
    if mismatches:
        raise OratsOptionSchemaError(mismatches=mismatches)

    return validated.select(
        pl.col("ticker"),
        pl.col("trade_date")
        .str.strptime(pl.Date, format="%m/%d/%Y", strict=True, exact=True, cache=True)
        .alias("quote_date"),
        pl.col("expirDate")
        .str.strptime(pl.Date, format="%m/%d/%Y", strict=True, exact=True, cache=True)
        .alias("expiry_date"),
        pl.col("strike"),
        pl.col("cAskPx").alias("call_ask"),
        pl.col("cBidPx").alias("call_bid"),
        pl.col("pAskPx").alias("put_ask"),
        pl.col("pBidPx").alias("put_bid"),
        pl.col("stkPx").alias("underlying_price"),
        pl.col("cVolu").alias("call_volume"),
        pl.col("pVolu").alias("put_volume"),
        pl.col("cOi").alias("call_open_interest"),
        pl.col("pOi").alias("put_open_interest"),
        pl.col("cValue").alias("call_theoretical_value"),
        pl.col("pValue").alias("put_theoretical_value"),
        pl.col("cMidIv").alias("call_iv"),
        pl.col("pMidIv").alias("put_iv"),
    )


__all__ = [
    "ORATS_NORMALIZED_OPTION_DTYPES",
    "ORATS_RAW_OPTION_COLUMNS",
    "ORATS_RAW_OPTION_DTYPES",
    "ORATS_RAW_SCAN_DTYPES",
    "OratsOptionError",
    "OratsOptionSchemaError",
    "OratsOptionSourceError",
    "canonicalize_raw_orats_options",
    "scan_raw_orats_csv",
]
