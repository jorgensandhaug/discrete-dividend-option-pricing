"""Strict ORATS-to-engine curve adapter tests using synthetic rows."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from finoptions import EventSchedule

from discrete_dividend_study.data.providers.orats.curves import (
    OratsCurveContentError,
    OratsCurveDomainError,
    OratsCurveSchemaError,
    adapt_orats_curve,
    adapt_raw_orats_curve,
    canonicalize_orats_curve_frame,
)

QUOTE_DATE = date(2024, 9, 27)
FIRST_EXPIRY = date(2025, 3, 28)
SECOND_EXPIRY = date(2025, 9, 26)


def schedule(*times: float) -> EventSchedule:
    """Construct a terminal-only synthetic event schedule."""
    return EventSchedule(times=times, dividends=(), exercise_nodes=(), expiry_node=len(times) - 1)


def canonical_frame() -> pl.LazyFrame:
    """Return repeated strike-style rows with one rate per maturity."""
    return pl.LazyFrame(
        {
            "quote_date": [QUOTE_DATE] * 4,
            "expiration": [FIRST_EXPIRY, FIRST_EXPIRY, SECOND_EXPIRY, SECOND_EXPIRY],
            "yte": [0.5, 0.5, 1.0, 1.0],
            "iRate": [0.02, 0.02, 0.03, 0.03],
        },
        schema={
            "quote_date": pl.Date,
            "expiration": pl.Date,
            "yte": pl.Float64,
            "iRate": pl.Float64,
        },
    )


def test_interpolates_log_discounts_and_retains_source_nodes() -> None:
    """The adapter inserts origin and interpolates log discounts, not rates."""
    adapted = adapt_orats_curve(canonical_frame(), schedule(0.0, 0.25, 0.75, 1.0))
    expected = (0.0, -0.005, -0.02, -0.03)
    assert adapted.curve.log_discounts == pytest.approx(expected, abs=1e-15)
    assert adapted.provenance.quote_date == QUOTE_DATE
    assert adapted.provenance.source_times == (0.5, 1.0)
    assert adapted.provenance.source_rates == (0.02, 0.03)
    assert adapted.provenance.origin_inserted


def test_normalizes_observed_same_day_node_to_exact_origin() -> None:
    """An observed yte-zero rate retains provenance but cannot change P(0,0)=1."""
    same_day_rate = 0.05
    frame = pl.LazyFrame(
        {
            "quote_date": [QUOTE_DATE, QUOTE_DATE],
            "expiration": [QUOTE_DATE, SECOND_EXPIRY],
            "yte": [0.0, 1.0],
            "iRate": [same_day_rate, 0.03],
        },
        schema={
            "quote_date": pl.Date,
            "expiration": pl.Date,
            "yte": pl.Float64,
            "iRate": pl.Float64,
        },
    )
    adapted = adapt_orats_curve(frame, schedule(0.0, 1.0))
    assert adapted.curve.log_discounts[0] == 0.0
    assert adapted.provenance.source_rates[0] == same_day_rate
    assert not adapted.provenance.origin_inserted


def test_rejects_inconsistent_rate_at_one_maturity() -> None:
    """Cross-sectional rate disagreement never selects first/mean/last silently."""
    frame = canonical_frame().with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.lit(0.021))
        .otherwise(pl.col("iRate"))
        .alias("iRate"),
    )
    with pytest.raises(OratsCurveContentError, match="exactly one iRate"):
        adapt_orats_curve(frame, schedule(0.0, 1.0))


def test_rejects_mixed_dates_wrong_dtype_and_extrapolation() -> None:
    """Source identity, canonical dtypes, and domain limits all fail explicitly."""
    mixed = canonical_frame().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(date(2024, 9, 26)))
        .otherwise(pl.col("quote_date"))
        .alias("quote_date"),
    )
    with pytest.raises(OratsCurveContentError, match="exactly one quote date"):
        adapt_orats_curve(mixed, schedule(0.0, 1.0))

    wrong_dtype = canonical_frame().with_columns(pl.col("yte").cast(pl.Float32))
    with pytest.raises(OratsCurveSchemaError, match="dtype mismatches"):
        adapt_orats_curve(wrong_dtype, schedule(0.0, 1.0))

    with pytest.raises(OratsCurveDomainError, match="forbids extrapolation"):
        adapt_orats_curve(canonical_frame(), schedule(0.0, 1.01))


@pytest.mark.parametrize(
    ("times", "expirations", "message"),
    [
        ((0.0, 1.0), (FIRST_EXPIRY, SECOND_EXPIRY), "zero-maturity node"),
        ((0.5, 1.0), (QUOTE_DATE, SECOND_EXPIRY), "positive ORATS maturity"),
        ((0.5, 1.0), (SECOND_EXPIRY, FIRST_EXPIRY), "strictly increasing with maturity"),
    ],
)
def test_rejects_incoherent_maturity_expiration_axis(
    times: tuple[float, float],
    expirations: tuple[date, date],
    message: str,
) -> None:
    """Calendar expirations cannot contradict their ordered year fractions."""
    frame = pl.LazyFrame(
        {
            "quote_date": [QUOTE_DATE, QUOTE_DATE],
            "expiration": expirations,
            "yte": times,
            "iRate": [0.02, 0.03],
        },
        schema={
            "quote_date": pl.Date,
            "expiration": pl.Date,
            "yte": pl.Float64,
            "iRate": pl.Float64,
        },
    )
    with pytest.raises(OratsCurveContentError, match=message):
        adapt_orats_curve(frame, schedule(0.0, 1.0))


@pytest.mark.parametrize("maturity", [-0.0, float("nan"), float("inf")])
def test_rejects_noncanonical_or_nonfinite_maturity(maturity: float) -> None:
    """Invalid year fractions never enter interpolation or node construction."""
    frame = canonical_frame().with_columns(pl.lit(maturity).alias("yte"))
    with pytest.raises(OratsCurveContentError, match="finite, nonnegative, and canonical"):
        adapt_orats_curve(frame, schedule(0.0, 1.0))


def test_raw_delivery_mapping_remains_lazy_and_strict() -> None:
    """Raw source names/dates map through built-in lazy expressions without numeric coercion."""
    raw = pl.LazyFrame(
        {
            "trade_date": ["9/27/2024", "9/27/2024"],
            "expirDate": ["3/28/2025", "9/26/2025"],
            "yte": [0.5, 1.0],
            "iRate": [0.02, 0.03],
        },
        schema={
            "trade_date": pl.String,
            "expirDate": pl.String,
            "yte": pl.Float64,
            "iRate": pl.Float64,
        },
    )
    canonical = canonicalize_orats_curve_frame(raw)
    assert isinstance(canonical, pl.LazyFrame)
    assert canonical.collect_schema() == pl.Schema(
        {
            "quote_date": pl.Date,
            "expiration": pl.Date,
            "yte": pl.Float64,
            "iRate": pl.Float64,
        },
    )
    adapted = adapt_raw_orats_curve(raw, schedule(0.0, 1.0))
    assert adapted.provenance.quote_date == QUOTE_DATE

    malformed = raw.with_columns(pl.lit("2024-09-27").alias("trade_date"))
    with pytest.raises(OratsCurveContentError, match="date parsing failed"):
        adapt_raw_orats_curve(malformed, schedule(0.0, 1.0))
