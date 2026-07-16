"""Provider-neutral long-form pairing and fail-fast partition tests."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from discrete_dividend_study.data.options.quotes import (
    FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
    LONG_FORM_OPTION_QUOTE_DTYPES,
    LongFormOptionSelection,
    OptionQuoteContentError,
    OptionQuotePartition,
    OptionQuoteSchemaError,
    OptionQuoteSpecError,
    adapt_long_form_option_quotes,
    combine_disjoint_option_quote_partitions,
)

PAIRED_ROWS = 2
SOURCE_ROWS = PAIRED_ROWS * 2


def _long_form() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "ticker": ["DUK", "DUK", "VZ", "VZ", "IGNORED", "IGNORED"],
            "quote_date": [date(2024, 1, 2)] * 6,
            "expiry_date": [date(2024, 2, 2)] * 6,
            "strike": [90.0, 90.0, 35.0, 35.0, 1.0, 1.0],
            "underlying_price": [100.125, 100.125, 40.25, 40.25, 1.0, 1.0],
            "bid": [10.0, 0.8, 5.0, 0.4, 1.0, 1.0],
            "ask": [10.5, 0.9, 5.2, 0.5, 1.0, 1.0],
            "volume": [12.0, 3.0, 7.0, 4.0, 1.0, 1.0],
            "option_type": ["call", "put", "call", "put", "call", "put"],
        },
        schema=LONG_FORM_OPTION_QUOTE_DTYPES,
    )


def _selection() -> LongFormOptionSelection:
    return LongFormOptionSelection(
        tickers=("DUK", "VZ"),
        start_date=date(2024, 1, 1),
        end_date_exclusive=date(2024, 1, 3),
    )


def test_long_form_adapter_consumes_every_selected_side_once() -> None:
    """The explicit adapter pairs exact keys, preserves Float64, and records aggregates."""
    adapted = adapt_long_form_option_quotes(_long_form(), _selection())
    observed = adapted.frame.collect()
    assert observed.schema == pl.Schema(FLOAT64_PAIRED_OPTION_QUOTE_DTYPES)
    assert observed.shape == (PAIRED_ROWS, len(FLOAT64_PAIRED_OPTION_QUOTE_DTYPES))
    assert observed["ticker"].to_list() == ["DUK", "VZ"]
    assert observed["call_bid"].to_list() == [10.0, 5.0]
    assert observed["put_ask"].to_list() == pytest.approx([0.9, 0.5])
    assert adapted.source_rows == SOURCE_ROWS
    assert adapted.summary.rows == PAIRED_ROWS
    assert adapted.summary.tickers == ("DUK", "VZ")
    assert adapted.summary.quote_date_min == date(2024, 1, 2)
    assert adapted.summary.quote_date_max == date(2024, 1, 2)
    assert adapted.summary.unique_quote_dates == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.filter(
                ~((pl.col("ticker") == "DUK") & (pl.col("option_type") == "put"))
            ),
            "exactly_two_sides|one_put",
        ),
        (
            lambda frame: pl.concat(
                (
                    frame,
                    frame.filter((pl.col("ticker") == "DUK") & (pl.col("option_type") == "call")),
                )
            ),
            "duplicate_contract_sides",
        ),
        (
            lambda frame: frame.with_columns(
                pl.when((pl.col("ticker") == "DUK") & (pl.col("option_type") == "put"))
                .then(999.0)
                .otherwise(pl.col("underlying_price"))
                .alias("underlying_price")
            ),
            "paired_spot_disagreements",
        ),
    ],
)
def test_long_form_adapter_rejects_unpaired_or_conflicting_contracts(
    mutate: Callable[[pl.LazyFrame], pl.LazyFrame],
    message: str,
) -> None:
    """Missing, duplicated, and state-inconsistent sides never become partial quotes."""
    with pytest.raises(OptionQuoteContentError, match=message):
        adapt_long_form_option_quotes(mutate(_long_form()), _selection())


def test_long_form_adapter_rejects_wrong_precision_and_missing_ticker() -> None:
    """Source precision and the complete declared ticker selection are exact contracts."""
    wrong_precision = _long_form().with_columns(pl.col("strike").cast(pl.Float32))
    with pytest.raises(OptionQuoteSchemaError, match="strike"):
        adapt_long_form_option_quotes(wrong_precision, _selection())
    with pytest.raises(OptionQuoteSchemaError, match="unexpected"):
        adapt_long_form_option_quotes(
            _long_form().with_columns(pl.lit("x").alias("unexpected")),
            _selection(),
        )
    with pytest.raises(OptionQuoteContentError, match="missing_selected_tickers"):
        adapt_long_form_option_quotes(
            _long_form(),
            LongFormOptionSelection(
                tickers=("DUK", "ETR", "VZ"),
                start_date=date(2024, 1, 1),
                end_date_exclusive=date(2024, 1, 3),
            ),
        )


def test_disjoint_partition_union_has_no_precedence_or_fallback() -> None:
    """Canonical partitions concatenate only when their ticker domains are disjoint and exact."""
    paired = adapt_long_form_option_quotes(_long_form(), _selection()).frame
    combined = combine_disjoint_option_quote_partitions(
        (
            OptionQuotePartition(
                name="duke",
                frame=paired.filter(pl.col("ticker") == "DUK"),
                expected_tickers=("DUK",),
            ),
            OptionQuotePartition(
                name="verizon",
                frame=paired.filter(pl.col("ticker") == "VZ"),
                expected_tickers=("VZ",),
            ),
        ),
        expected_schema=FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
    )
    assert combined.frame.collect()["ticker"].to_list() == ["DUK", "VZ"]
    assert tuple(name for name, _ in combined.summaries) == ("duke", "verizon")

    with pytest.raises(OptionQuoteSpecError, match="pairwise-disjoint"):
        combine_disjoint_option_quote_partitions(
            (
                OptionQuotePartition("first", paired, ("DUK", "VZ")),
                OptionQuotePartition(
                    "second",
                    paired.filter(pl.col("ticker") == "VZ"),
                    ("VZ",),
                ),
            ),
            expected_schema=FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
        )


def test_partition_union_rejects_extra_columns_and_duplicate_keys() -> None:
    """A partition cannot smuggle provenance columns or silently deduplicate contracts."""
    paired = adapt_long_form_option_quotes(_long_form(), _selection()).frame.filter(
        pl.col("ticker") == "DUK"
    )
    with pytest.raises(OptionQuoteSchemaError, match="source"):
        combine_disjoint_option_quote_partitions(
            (
                OptionQuotePartition(
                    "extra", paired.with_columns(pl.lit("x").alias("source")), ("DUK",)
                ),
            ),
            expected_schema=FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
        )
    with pytest.raises(OptionQuoteContentError, match="duplicate_contract_keys"):
        combine_disjoint_option_quote_partitions(
            (OptionQuotePartition("duplicates", pl.concat((paired, paired)), ("DUK",)),),
            expected_schema=FLOAT64_PAIRED_OPTION_QUOTE_DTYPES,
        )
