"""Strict adapters from canonical ORATS columns to engine financial inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

import numpy as np
import polars as pl
from finoptions import EventSchedule, LogDiscountCurve

from .schema import require_columns

ORATS_CURVE_COLUMNS = ("quote_date", "expiration", "yte", "iRate")
ORATS_CURVE_DTYPES = {
    "quote_date": pl.Date,
    "expiration": pl.Date,
    "yte": pl.Float64,
    "iRate": pl.Float64,
}
ORATS_RAW_CURVE_COLUMNS = ("trade_date", "expirDate", "yte", "iRate")
ORATS_RAW_CURVE_DTYPES = {
    "trade_date": pl.String,
    "expirDate": pl.String,
    "yte": pl.Float64,
    "iRate": pl.Float64,
}


class OratsCurveError(ValueError):
    """Base class for an invalid canonical ORATS curve input."""


@dataclass(slots=True)
class OratsCurveSchemaError(OratsCurveError):
    """Canonical ORATS curve columns have incorrect runtime dtypes."""

    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe expected and observed dtypes without row-level data."""
        return f"ORATS curve column dtype mismatches: {self.mismatches!r}"


@dataclass(slots=True)
class OratsCurveContentError(OratsCurveError):
    """Canonical ORATS maturity groups violate source consistency."""

    reason: str
    maturity: float | None = None

    def __str__(self) -> str:
        """Describe the failed invariant without exposing licensed values beyond maturity."""
        if self.maturity is None:
            return self.reason
        return f"{self.reason}; maturity={self.maturity!r}"


@dataclass(slots=True)
class OratsCurveDomainError(OratsCurveError):
    """The requested event grid lies outside the observed curve domain."""

    requested_max: float
    observed_max: float

    def __str__(self) -> str:
        """Describe the forbidden extrapolation."""
        return (
            "ORATS log-discount interpolation forbids extrapolation; "
            f"requested maximum {self.requested_max!r}, observed maximum {self.observed_max!r}"
        )


@dataclass(frozen=True, slots=True)
class OratsCurveProvenance:
    """Exact vendor nodes retained beside an adapted model curve."""

    quote_date: date
    source_times: tuple[float, ...]
    source_rates: tuple[float, ...]
    source_expirations: tuple[date, ...]
    origin_inserted: bool


@dataclass(frozen=True, slots=True)
class AdaptedOratsCurve:
    """Validated engine curve plus its source-node provenance."""

    curve: LogDiscountCurve
    provenance: OratsCurveProvenance


def _validate_curve_schema(frame: pl.LazyFrame) -> pl.LazyFrame:
    validated = require_columns(frame, ORATS_CURVE_COLUMNS)
    schema = validated.collect_schema()
    mismatches = tuple(
        (name, str(expected), str(schema[name]))
        for name, expected in ORATS_CURVE_DTYPES.items()
        if schema[name] != expected
    )
    if mismatches:
        raise OratsCurveSchemaError(mismatches=mismatches)
    return validated


def canonicalize_orats_curve_frame(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Map strict raw ORATS delivery columns to the canonical lazy curve schema.

    Date strings must use the observed delivery format `M/D/YYYY`; parsing remains lazy and strict.
    No financial numeric column is cast or inferred.

    Raises:
        MissingColumnsError: If a required raw source column is absent.
        OratsCurveSchemaError: If a raw source column has the wrong dtype.

    """
    validated = require_columns(frame, ORATS_RAW_CURVE_COLUMNS)
    schema = validated.collect_schema()
    mismatches = tuple(
        (name, str(expected), str(schema[name]))
        for name, expected in ORATS_RAW_CURVE_DTYPES.items()
        if schema[name] != expected
    )
    if mismatches:
        raise OratsCurveSchemaError(mismatches=mismatches)
    return validated.select(
        pl.col("trade_date")
        .str.strptime(pl.Date, format="%m/%d/%Y", strict=True, exact=True, cache=True)
        .alias("quote_date"),
        pl.col("expirDate")
        .str.strptime(pl.Date, format="%m/%d/%Y", strict=True, exact=True, cache=True)
        .alias("expiration"),
        pl.col("yte"),
        pl.col("iRate"),
    )


def _collect_maturity_groups(frame: pl.LazyFrame) -> pl.DataFrame:
    return (
        _validate_curve_schema(frame)
        .select(ORATS_CURVE_COLUMNS)
        .group_by("quote_date", "yte")
        .agg(
            pl.len().alias("_rows"),
            pl.col("iRate").null_count().alias("_rate_nulls"),
            (~pl.col("iRate").is_finite()).sum().alias("_rate_nonfinite"),
            pl.col("iRate").n_unique().alias("_rate_count"),
            pl.col("iRate").first().alias("iRate"),
            pl.col("expiration").null_count().alias("_expiration_nulls"),
            pl.col("expiration").n_unique().alias("_expiration_count"),
            pl.col("expiration").first().alias("expiration"),
        )
        .sort("quote_date", "yte", nulls_last=False)
        .collect()
    )


def _extract_group(row: dict[str, object]) -> tuple[float, float, date]:
    maturity = row["yte"]
    if (
        type(maturity) is not float
        or not math.isfinite(maturity)
        or maturity < 0.0
        or (maturity == 0.0 and math.copysign(1.0, maturity) < 0.0)
    ):
        raise OratsCurveContentError(
            reason="ORATS maturity must be finite, nonnegative, and canonical",
            maturity=maturity if type(maturity) is float else None,
        )
    if row["_rate_nulls"] != 0 or row["_rate_nonfinite"] != 0:
        raise OratsCurveContentError(
            reason="ORATS maturity contains null or non-finite iRate",
            maturity=maturity,
        )
    if row["_rate_count"] != 1:
        raise OratsCurveContentError(
            reason="ORATS maturity must contain exactly one iRate",
            maturity=maturity,
        )
    if row["_expiration_nulls"] != 0 or row["_expiration_count"] != 1:
        raise OratsCurveContentError(
            reason="ORATS maturity must contain exactly one non-null expiration",
            maturity=maturity,
        )
    rate = row["iRate"]
    expiration = row["expiration"]
    if type(rate) is not float or not math.isfinite(rate):
        raise OratsCurveContentError(
            reason="ORATS maturity did not materialize a finite float iRate",
            maturity=maturity,
        )
    if type(expiration) is not date:
        raise OratsCurveContentError(
            reason="ORATS expiration did not materialize as datetime.date",
            maturity=maturity,
        )
    return maturity, rate, expiration


def _extract_source_nodes(
    groups: pl.DataFrame,
) -> tuple[date, tuple[float, ...], tuple[float, ...], tuple[date, ...]]:
    if groups.is_empty():
        raise OratsCurveContentError(reason="ORATS curve input contains no rows")

    quote_dates = groups.get_column("quote_date").unique().to_list()
    if len(quote_dates) != 1 or quote_dates[0] is None:
        raise OratsCurveContentError(reason="ORATS curve input must contain exactly one quote date")
    quote_date = quote_dates[0]
    if type(quote_date) is not date:
        raise OratsCurveContentError(reason="ORATS quote date did not materialize as datetime.date")

    source_times: list[float] = []
    source_rates: list[float] = []
    source_expirations: list[date] = []
    for row in groups.iter_rows(named=True):
        maturity, rate, expiration = _extract_group(row)
        source_times.append(maturity)
        source_rates.append(rate)
        source_expirations.append(expiration)

    times = tuple(source_times)
    rates = tuple(source_rates)
    expirations = tuple(source_expirations)
    _validate_source_axis(quote_date, times, expirations)
    return quote_date, times, rates, expirations


def _validate_source_axis(
    quote_date: date,
    source_times: tuple[float, ...],
    source_expirations: tuple[date, ...],
) -> None:
    if any(right <= left for left, right in pairwise(source_times)):
        raise OratsCurveContentError(reason="ORATS maturities must be strictly increasing")

    for maturity, expiration in zip(source_times, source_expirations, strict=True):
        if maturity == 0.0 and expiration != quote_date:
            raise OratsCurveContentError(
                reason="an ORATS zero-maturity node must expire on its quote date",
                maturity=maturity,
            )
        if maturity > 0.0 and expiration <= quote_date:
            raise OratsCurveContentError(
                reason="a positive ORATS maturity must expire after its quote date",
                maturity=maturity,
            )

    if any(right <= left for left, right in pairwise(source_expirations)):
        raise OratsCurveContentError(
            reason="ORATS expirations must be strictly increasing with maturity",
        )


def _model_curve_nodes(
    source_times: tuple[float, ...],
    source_rates: tuple[float, ...],
) -> tuple[list[float], list[float], bool]:
    model_times = list(source_times)
    model_log_discounts = [
        -rate * maturity for maturity, rate in zip(source_times, source_rates, strict=True)
    ]
    origin_inserted = model_times[0] != 0.0
    if origin_inserted:
        model_times.insert(0, 0.0)
        model_log_discounts.insert(0, 0.0)
    else:
        model_log_discounts[0] = 0.0
    return model_times, model_log_discounts, origin_inserted


def adapt_orats_curve(frame: pl.LazyFrame, schedule: EventSchedule) -> AdaptedOratsCurve:
    """Reduce canonical ORATS rows to a no-extrapolation event-aligned log-discount curve.

    The input may contain many strikes/tickers but must represent exactly one quote date. Every
    observed maturity must have one finite continuous decimal `iRate` and one expiration. Linear
    interpolation in log-discount space implies piecewise-constant instantaneous forwards.

    Raises:
        MissingColumnsError: If a required canonical source column is absent.
        OratsCurveSchemaError: If canonical columns have the wrong dtype.
        OratsCurveContentError: If rows are empty, null/non-finite, mixed-date, or inconsistent.
        OratsCurveDomainError: If the schedule requires extrapolation.

    """
    quote_date, source_times, source_rates, source_expirations = _extract_source_nodes(
        _collect_maturity_groups(frame),
    )
    model_times, model_log_discounts, origin_inserted = _model_curve_nodes(
        source_times,
        source_rates,
    )

    requested_max = schedule.times[-1]
    observed_max = model_times[-1]
    if requested_max > observed_max:
        raise OratsCurveDomainError(requested_max=requested_max, observed_max=observed_max)

    interpolated = np.interp(
        np.asarray(schedule.times, dtype=np.float64),
        np.asarray(model_times, dtype=np.float64),
        np.asarray(model_log_discounts, dtype=np.float64),
    )
    curve = LogDiscountCurve(
        times=schedule.times,
        log_discounts=tuple(float(value) for value in interpolated),
    )
    provenance = OratsCurveProvenance(
        quote_date=quote_date,
        source_times=source_times,
        source_rates=source_rates,
        source_expirations=source_expirations,
        origin_inserted=origin_inserted,
    )
    return AdaptedOratsCurve(curve=curve, provenance=provenance)


def adapt_raw_orats_curve(frame: pl.LazyFrame, schedule: EventSchedule) -> AdaptedOratsCurve:
    """Strictly parse raw ORATS delivery columns and adapt them to an engine curve.

    Raises:
        MissingColumnsError: If a required raw source column is absent.
        OratsCurveSchemaError: If a raw source column has the wrong dtype.
        OratsCurveContentError: If strict date parsing or canonical content validation fails.
        OratsCurveDomainError: If the schedule requires extrapolation.

    """
    try:
        return adapt_orats_curve(canonicalize_orats_curve_frame(frame), schedule)
    except pl.exceptions.InvalidOperationError as error:
        raise OratsCurveContentError(reason="strict ORATS delivery date parsing failed") from error
