"""Strict adapters and unions for provider-neutral paired option quotes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

FLOAT32_PAIRED_OPTION_QUOTE_DTYPES: Final = {
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
}
FLOAT64_PAIRED_OPTION_QUOTE_DTYPES: Final = {
    name: dtype if dtype in {pl.String, pl.Date} else pl.Float64
    for name, dtype in FLOAT32_PAIRED_OPTION_QUOTE_DTYPES.items()
}
LONG_FORM_OPTION_QUOTE_DTYPES: Final = {
    "ticker": pl.String,
    "quote_date": pl.Date,
    "expiry_date": pl.Date,
    "strike": pl.Float64,
    "underlying_price": pl.Float64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "volume": pl.Float64,
    "option_type": pl.String,
}
LONG_FORM_CALL: Final = "call"
LONG_FORM_PUT: Final = "put"
_LONG_FORM_TYPES: Final = (LONG_FORM_CALL, LONG_FORM_PUT)
_CONTRACT_KEY: Final = ("ticker", "quote_date", "expiry_date", "strike")
_SIDE_KEY: Final = (*_CONTRACT_KEY, "option_type")
_LONG_FORM_NUMERIC_COLUMNS: Final = (
    "strike",
    "underlying_price",
    "bid",
    "ask",
    "volume",
)


class OptionQuoteError(ValueError):
    """Base class for invalid provider-neutral option quotes."""


@dataclass(slots=True)
class OptionQuoteSpecError(OptionQuoteError):
    """A long-form selection or partition declaration is malformed."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid declaration without inspecting licensed rows."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(slots=True)
class OptionQuoteSchemaError(OptionQuoteError):
    """An option-quote frame has incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe exact schema disagreements without exposing values."""
        return f"option quote dtype mismatches: {self.mismatches!r}"


@dataclass(slots=True)
class OptionQuoteContentError(OptionQuoteError):
    """Selected quotes cannot satisfy the declared paired contract."""

    source: str
    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing market rows."""
        return f"invalid option quotes from {self.source!r}: {self.violations!r}"


@dataclass(frozen=True, slots=True)
class LongFormOptionSelection:
    """Explicit ticker and half-open date selection for a long-form quote source."""

    tickers: tuple[str, ...]
    start_date: date
    end_date_exclusive: date

    def __post_init__(self) -> None:
        """Reject inferred order, duplicate symbols, and ambiguous date bounds."""
        _validate_tickers(self.tickers, field="selection.tickers")
        if self.tickers != tuple(sorted(self.tickers)):
            raise OptionQuoteSpecError(
                field="selection.tickers",
                observed=self.tickers,
                requirement="a sorted unique nonempty tuple of symbols",
            )
        if type(self.start_date) is not date or type(self.end_date_exclusive) is not date:
            raise OptionQuoteSpecError(
                field="selection date bounds",
                observed=(self.start_date, self.end_date_exclusive),
                requirement="plain datetime.date values",
            )
        if self.start_date >= self.end_date_exclusive:
            raise OptionQuoteSpecError(
                field="selection date bounds",
                observed=(self.start_date, self.end_date_exclusive),
                requirement="a nonempty half-open interval",
            )


@dataclass(frozen=True, slots=True)
class OptionQuoteSummary:
    """Aggregate, non-row-level identity of one validated quote partition."""

    rows: int
    quote_date_min: date
    quote_date_max: date
    tickers: tuple[str, ...]
    unique_quote_dates: int

    def __post_init__(self) -> None:
        """Reject manually fabricated empty or inconsistent summaries."""
        if type(self.rows) is not int or self.rows <= 0:
            raise OptionQuoteSpecError(
                field="summary.rows",
                observed=self.rows,
                requirement="a positive plain int",
            )
        if (
            type(self.quote_date_min) is not date
            or type(self.quote_date_max) is not date
            or self.quote_date_min > self.quote_date_max
        ):
            raise OptionQuoteSpecError(
                field="summary quote-date bounds",
                observed=(self.quote_date_min, self.quote_date_max),
                requirement="ordered plain datetime.date values",
            )
        _validate_tickers(self.tickers, field="summary.tickers")
        if self.tickers != tuple(sorted(self.tickers)):
            raise OptionQuoteSpecError(
                field="summary.tickers",
                observed=self.tickers,
                requirement="a sorted unique nonempty tuple of symbols",
            )
        if type(self.unique_quote_dates) is not int or self.unique_quote_dates <= 0:
            raise OptionQuoteSpecError(
                field="summary.unique_quote_dates",
                observed=self.unique_quote_dates,
                requirement="a positive plain int",
            )

    def as_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible aggregate mapping."""
        return {
            "quote_date_max": self.quote_date_max.isoformat(),
            "quote_date_min": self.quote_date_min.isoformat(),
            "rows": self.rows,
            "tickers": list(self.tickers),
            "unique_quote_dates": self.unique_quote_dates,
        }


@dataclass(frozen=True, slots=True)
class AdaptedLongFormOptionQuotes:
    """One validated source-precision paired view and its aggregate identity."""

    frame: pl.LazyFrame
    selection: LongFormOptionSelection
    source_rows: int
    summary: OptionQuoteSummary

    def __post_init__(self) -> None:
        """Require the exact two-sides-per-contract relationship."""
        if type(self.frame) is not pl.LazyFrame:
            raise OptionQuoteSpecError(
                field="adapted frame",
                observed=type(self.frame).__name__,
                requirement="a polars.LazyFrame",
            )
        if type(self.selection) is not LongFormOptionSelection:
            raise OptionQuoteSpecError(
                field="adapted selection",
                observed=type(self.selection).__name__,
                requirement="LongFormOptionSelection",
            )
        if type(self.source_rows) is not int or self.source_rows != self.summary.rows * 2:
            raise OptionQuoteSpecError(
                field="adapted source rows",
                observed=self.source_rows,
                requirement="exactly twice the paired output rows",
            )


@dataclass(frozen=True, slots=True)
class OptionQuotePartition:
    """One named, ticker-disjoint canonical quote partition."""

    name: str
    frame: pl.LazyFrame
    expected_tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject ambiguous names, runtime types, and ticker declarations."""
        if type(self.name) is not str or not self.name or self.name.strip() != self.name:
            raise OptionQuoteSpecError(
                field="partition.name",
                observed=self.name,
                requirement="a nonempty trimmed plain str",
            )
        if type(self.frame) is not pl.LazyFrame:
            raise OptionQuoteSpecError(
                field=f"partition {self.name!r} frame",
                observed=type(self.frame).__name__,
                requirement="a polars.LazyFrame",
            )
        _validate_tickers(self.expected_tickers, field=f"partition {self.name!r} tickers")
        if self.expected_tickers != tuple(sorted(self.expected_tickers)):
            raise OptionQuoteSpecError(
                field=f"partition {self.name!r} tickers",
                observed=self.expected_tickers,
                requirement="a sorted unique nonempty tuple of symbols",
            )


@dataclass(frozen=True, slots=True)
class PartitionedOptionQuotes:
    """Ticker-disjoint strict union with one summary per source partition."""

    frame: pl.LazyFrame
    summaries: tuple[tuple[str, OptionQuoteSummary], ...]

    def __post_init__(self) -> None:
        """Reject duplicate source names and invalid lazy frames."""
        if type(self.frame) is not pl.LazyFrame:
            raise OptionQuoteSpecError(
                field="partitioned frame",
                observed=type(self.frame).__name__,
                requirement="a polars.LazyFrame",
            )
        names = tuple(name for name, _ in self.summaries)
        if not names or len(names) != len(set(names)):
            raise OptionQuoteSpecError(
                field="partition summaries",
                observed=names,
                requirement="distinct nonempty source names",
            )


def adapt_long_form_option_quotes(
    frame: pl.LazyFrame,
    selection: LongFormOptionSelection,
) -> AdaptedLongFormOptionQuotes:
    """Pair one call and one put per contract while preserving source Float64 values.

    The source is selected before validation. Every selected row must be consumed exactly once by
    a one-call/one-put contract, paired spot values must agree exactly in source precision, and all
    numeric values must be finite. Precision narrowing belongs after any source-sensitive cohort
    predicate; no absent vendor field is fabricated.
    """
    if type(frame) is not pl.LazyFrame:
        raise OptionQuoteSpecError(
            field="long-form frame",
            observed=type(frame).__name__,
            requirement="a polars.LazyFrame",
        )
    if type(selection) is not LongFormOptionSelection:
        raise OptionQuoteSpecError(
            field="selection",
            observed=type(selection).__name__,
            requirement="LongFormOptionSelection",
        )
    validated = _validate_exact_schema(frame, LONG_FORM_OPTION_QUOTE_DTYPES)
    selected = validated.filter(
        pl.col("ticker").is_in(selection.tickers)
        & (pl.col("quote_date") >= selection.start_date)
        & (pl.col("quote_date") < selection.end_date_exclusive),
    ).select(tuple(LONG_FORM_OPTION_QUOTE_DTYPES))
    source = _long_form_summary(selected, selection)
    calls = _long_form_side(selected, LONG_FORM_CALL)
    puts = _long_form_side(selected, LONG_FORM_PUT)
    paired = calls.join(
        puts,
        on=_CONTRACT_KEY,
        how="inner",
        validate="1:1",
        coalesce=True,
        maintain_order="left",
    )
    spot_disagreements = (
        paired.select(
            pl.col("_call_underlying_price")
            .ne(pl.col("_put_underlying_price"))
            .sum()
            .alias("spot_disagreements"),
        )
        .collect()
        .item()
    )
    if type(spot_disagreements) is not int or spot_disagreements != 0:
        count = 1 if type(spot_disagreements) is not int else spot_disagreements
        raise OptionQuoteContentError(
            source="long-form selection",
            violations=(("paired_spot_disagreements", count),),
        )
    output = paired.select(
        pl.col("ticker"),
        pl.col("quote_date"),
        pl.col("expiry_date"),
        pl.col("strike"),
        pl.col("_call_ask").alias("call_ask"),
        pl.col("_call_bid").alias("call_bid"),
        pl.col("_put_ask").alias("put_ask"),
        pl.col("_put_bid").alias("put_bid"),
        pl.col("_call_underlying_price").alias("underlying_price"),
        pl.col("_call_volume").alias("call_volume"),
        pl.col("_put_volume").alias("put_volume"),
    )
    if output.collect_schema() != pl.Schema(FLOAT64_PAIRED_OPTION_QUOTE_DTYPES):
        raise OptionQuoteSchemaError(
            mismatches=_schema_mismatches(output, FLOAT64_PAIRED_OPTION_QUOTE_DTYPES),
        )
    output_summary = _paired_summary(
        output,
        source="adapted long-form output",
        expected_dtypes=FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
    )
    if output_summary != source:
        raise OptionQuoteContentError(
            source="adapted long-form output",
            violations=(("output_summary_disagrees_with_source", 1),),
        )
    return AdaptedLongFormOptionQuotes(
        frame=output,
        selection=selection,
        source_rows=source.rows * 2,
        summary=output_summary,
    )


def combine_disjoint_option_quote_partitions(
    partitions: tuple[OptionQuotePartition, ...],
    *,
    expected_schema: Mapping[str, type[pl.DataType] | pl.DataType] = (
        FLOAT32_PAIRED_OPTION_QUOTE_DTYPES
    ),
    required_tickers: tuple[str, ...] | None = None,
) -> PartitionedOptionQuotes:
    """Validate and concatenate ticker-disjoint canonical quote sources.

    Overlapping ticker partitions are deliberately unsupported: the operation has no source
    precedence, row repair, date fallback, or deduplication semantics. A caller that needs a
    same-ticker temporal splice must first define and validate that separate financial-data
    contract.
    """
    if type(expected_schema) is not dict or not expected_schema:
        raise OptionQuoteSpecError(
            field="expected_schema",
            observed=expected_schema,
            requirement="a nonempty plain ordered dtype mapping",
        )
    if required_tickers is not None:
        _validate_tickers(required_tickers, field="required_tickers")
        if required_tickers != tuple(sorted(required_tickers)):
            raise OptionQuoteSpecError(
                field="required_tickers",
                observed=required_tickers,
                requirement="a sorted unique nonempty tuple of symbols",
            )
    if (
        type(partitions) is not tuple
        or not partitions
        or any(type(partition) is not OptionQuotePartition for partition in partitions)
    ):
        raise OptionQuoteSpecError(
            field="partitions",
            observed=partitions,
            requirement="a nonempty tuple[OptionQuotePartition, ...]",
        )
    names = tuple(partition.name for partition in partitions)
    if len(names) != len(set(names)):
        raise OptionQuoteSpecError(
            field="partition names",
            observed=names,
            requirement="distinct names",
        )
    declared_tickers = tuple(
        ticker for partition in partitions for ticker in partition.expected_tickers
    )
    if len(declared_tickers) != len(set(declared_tickers)):
        raise OptionQuoteSpecError(
            field="partition ticker sets",
            observed=declared_tickers,
            requirement="pairwise-disjoint symbols with no source precedence",
        )
    if required_tickers is not None and tuple(sorted(declared_tickers)) != required_tickers:
        raise OptionQuoteSpecError(
            field="partition ticker union",
            observed=tuple(sorted(declared_tickers)),
            requirement=f"exactly {required_tickers!r}",
        )
    frames: list[pl.LazyFrame] = []
    summaries: list[tuple[str, OptionQuoteSummary]] = []
    for partition in partitions:
        frame = _validate_exact_schema(partition.frame, expected_schema)
        summary = _paired_summary(
            frame,
            source=partition.name,
            expected_dtypes=expected_schema,
        )
        if summary.tickers != partition.expected_tickers:
            missing = len(set(partition.expected_tickers) - set(summary.tickers))
            unexpected = len(set(summary.tickers) - set(partition.expected_tickers))
            raise OptionQuoteContentError(
                source=partition.name,
                violations=(
                    ("missing_declared_tickers", missing),
                    ("unexpected_tickers", unexpected),
                ),
            )
        frames.append(frame)
        summaries.append((partition.name, summary))
    return PartitionedOptionQuotes(
        frame=pl.concat(frames, how="vertical"),
        summaries=tuple(summaries),
    )


def _schema_mismatches(
    frame: pl.LazyFrame,
    expected: Mapping[str, type[pl.DataType] | pl.DataType],
) -> tuple[tuple[str, str, str], ...]:
    schema = frame.collect_schema()
    return tuple(
        (name, str(dtype), str(schema[name]))
        for name, dtype in expected.items()
        if schema[name] != dtype
    )


def _validate_exact_schema(
    frame: pl.LazyFrame,
    expected_dtypes: Mapping[str, type[pl.DataType] | pl.DataType],
) -> pl.LazyFrame:
    schema = frame.collect_schema()
    expected = pl.Schema(expected_dtypes)
    if schema != expected:
        mismatches = tuple(
            (name, str(dtype), str(schema.get(name, "<missing>")))
            for name, dtype in expected_dtypes.items()
            if schema.get(name) != dtype
        )
        extras = tuple(name for name in schema if name not in expected_dtypes)
        if extras:
            mismatches = (*mismatches, *((name, "<absent>", str(schema[name])) for name in extras))
        raise OptionQuoteSchemaError(mismatches=mismatches)
    return frame.select(tuple(expected_dtypes))


def _long_form_summary(
    selected: pl.LazyFrame,
    selection: LongFormOptionSelection,
) -> OptionQuoteSummary:
    row = (
        selected.select(
            pl.len().alias("rows"),
            pl.any_horizontal(pl.all().is_null()).sum().alias("rows_with_nulls"),
            (~pl.col("option_type").is_in(_LONG_FORM_TYPES))
            .fill_null(value=True)
            .sum()
            .alias("unsupported_option_types"),
            pl.struct(_SIDE_KEY).n_unique().alias("unique_side_keys"),
            pl.struct(_CONTRACT_KEY).n_unique().alias("unique_contract_keys"),
            (pl.col("option_type") == LONG_FORM_CALL).sum().alias("call_rows"),
            (pl.col("option_type") == LONG_FORM_PUT).sum().alias("put_rows"),
            pl.any_horizontal(
                *(~pl.col(name).is_finite() for name in _LONG_FORM_NUMERIC_COLUMNS),
            )
            .fill_null(value=True)
            .sum()
            .alias("nonfinite_numeric_rows"),
            pl.col("quote_date").min().alias("quote_date_min"),
            pl.col("quote_date").max().alias("quote_date_max"),
            pl.col("quote_date").n_unique().alias("unique_quote_dates"),
            pl.col("ticker").unique().sort().implode().alias("tickers"),
        )
        .collect()
        .row(0, named=True)
    )
    rows = _plain_int(row["rows"])
    unique_sides = _plain_int(row["unique_side_keys"])
    contracts = _plain_int(row["unique_contract_keys"])
    calls = _plain_int(row["call_rows"])
    puts = _plain_int(row["put_rows"])
    violations = _nonzero_counts(
        row,
        names=(
            "rows_with_nulls",
            "unsupported_option_types",
            "nonfinite_numeric_rows",
        ),
    )
    if rows == 0:
        violations.append(("empty_selection", 1))
    if unique_sides != rows:
        violations.append(("duplicate_contract_sides", rows - unique_sides))
    if rows != contracts * 2:
        violations.append(("contracts_without_exactly_two_sides", abs(rows - contracts * 2)))
    if calls != contracts:
        violations.append(("contracts_without_one_call", abs(calls - contracts)))
    if puts != contracts:
        violations.append(("contracts_without_one_put", abs(puts - contracts)))
    if violations:
        raise OptionQuoteContentError(
            source="long-form selection",
            violations=tuple(violations),
        )
    summary = _summary_from_row(row, rows=contracts, source="long-form selection")
    if summary.tickers != selection.tickers:
        missing = len(set(selection.tickers) - set(summary.tickers))
        unexpected = len(set(summary.tickers) - set(selection.tickers))
        raise OptionQuoteContentError(
            source="long-form selection",
            violations=(
                ("missing_selected_tickers", missing),
                ("unexpected_tickers", unexpected),
            ),
        )
    return summary


def _long_form_side(frame: pl.LazyFrame, option_type: str) -> pl.LazyFrame:
    prefix = "_call" if option_type == LONG_FORM_CALL else "_put"
    return frame.filter(pl.col("option_type") == option_type).select(
        *_CONTRACT_KEY,
        pl.col("underlying_price").alias(f"{prefix}_underlying_price"),
        pl.col("bid").alias(f"{prefix}_bid"),
        pl.col("ask").alias(f"{prefix}_ask"),
        pl.col("volume").alias(f"{prefix}_volume"),
    )


def _paired_summary(
    frame: pl.LazyFrame,
    *,
    source: str,
    expected_dtypes: Mapping[str, type[pl.DataType] | pl.DataType],
) -> OptionQuoteSummary:
    numeric_columns = tuple(name for name, dtype in expected_dtypes.items() if dtype.is_numeric())
    row = (
        frame.select(
            pl.len().alias("rows"),
            pl.any_horizontal(pl.all().is_null()).sum().alias("rows_with_nulls"),
            pl.struct(_CONTRACT_KEY).is_duplicated().sum().alias("duplicate_contract_keys"),
            pl.any_horizontal(
                *(~pl.col(name).is_finite() for name in numeric_columns),
            )
            .fill_null(value=True)
            .sum()
            .alias("nonfinite_numeric_rows"),
            pl.col("quote_date").min().alias("quote_date_min"),
            pl.col("quote_date").max().alias("quote_date_max"),
            pl.col("quote_date").n_unique().alias("unique_quote_dates"),
            pl.col("ticker").unique().sort().implode().alias("tickers"),
        )
        .collect()
        .row(0, named=True)
    )
    rows = _plain_int(row["rows"])
    violations = _nonzero_counts(
        row,
        names=("rows_with_nulls", "duplicate_contract_keys", "nonfinite_numeric_rows"),
    )
    if rows == 0:
        violations.append(("empty_partition", 1))
    if violations:
        raise OptionQuoteContentError(source=source, violations=tuple(violations))
    return _summary_from_row(row, rows=rows, source=source)


def _summary_from_row(row: dict[str, object], *, rows: int, source: str) -> OptionQuoteSummary:
    quote_date_min = row["quote_date_min"]
    quote_date_max = row["quote_date_max"]
    unique_quote_dates = row["unique_quote_dates"]
    tickers = row["tickers"]
    if (
        type(quote_date_min) is not date
        or type(quote_date_max) is not date
        or type(unique_quote_dates) is not int
        or type(tickers) is not list
        or any(type(ticker) is not str for ticker in tickers)
    ):
        raise OptionQuoteContentError(
            source=source,
            violations=(("invalid_aggregate_summary", 1),),
        )
    return OptionQuoteSummary(
        rows=rows,
        quote_date_min=quote_date_min,
        quote_date_max=quote_date_max,
        tickers=tuple(cast("list[str]", tickers)),
        unique_quote_dates=unique_quote_dates,
    )


def _plain_int(value: object) -> int:
    return value if type(value) is int else -1


def _nonzero_counts(
    row: dict[str, object],
    *,
    names: tuple[str, ...],
) -> list[tuple[str, int]]:
    return [
        (name, value if type(value) is int else 1)
        for name in names
        if type(value := row[name]) is not int or value != 0
    ]


def _validate_tickers(tickers: object, *, field: str) -> None:
    if (
        type(tickers) is not tuple
        or not tickers
        or any(
            type(ticker) is not str
            or not ticker
            or ticker.strip() != ticker
            or ticker.upper() != ticker
            for ticker in tickers
        )
        or len(tickers) != len(set(tickers))
    ):
        raise OptionQuoteSpecError(
            field=field,
            observed=tickers,
            requirement="a unique nonempty tuple of uppercase trimmed symbols",
        )


__all__ = [
    "FLOAT32_PAIRED_OPTION_QUOTE_DTYPES",
    "FLOAT64_PAIRED_OPTION_QUOTE_DTYPES",
    "LONG_FORM_CALL",
    "LONG_FORM_OPTION_QUOTE_DTYPES",
    "LONG_FORM_PUT",
    "AdaptedLongFormOptionQuotes",
    "LongFormOptionSelection",
    "OptionQuoteContentError",
    "OptionQuoteError",
    "OptionQuotePartition",
    "OptionQuoteSchemaError",
    "OptionQuoteSpecError",
    "OptionQuoteSummary",
    "PartitionedOptionQuotes",
    "adapt_long_form_option_quotes",
    "combine_disjoint_option_quote_partitions",
]
