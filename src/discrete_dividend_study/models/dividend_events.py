"""Strict Polars list-buffer extraction shared by financial model boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from finoptions import MaturityBoundary, RaggedCashDividends

Float64Vector = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]

_FLOAT64_DTYPE: Final = np.dtype(np.float64)


class EventBufferPolicy(StrEnum):
    """Explicit handling of fragmented Polars list and child buffers."""

    REQUIRE_ZERO_COPY = "require_zero_copy"
    ALLOW_RECHUNK = "allow_rechunk"


@dataclass(slots=True)
class EventBufferError(ValueError):
    """A list child buffer cannot satisfy the declared copy policy."""

    column: str
    policy: EventBufferPolicy
    reason: str

    def __str__(self) -> str:
        """Describe the failed buffer contract without exposing row values."""
        return f"event column {self.column!r} violates {self.policy.value}: {self.reason}"


@dataclass(slots=True)
class EventBufferTypeError(TypeError):
    """The extractor received an unsupported runtime boundary value."""

    field: str
    observed: str
    expected: str

    def __str__(self) -> str:
        """Describe the required runtime value without coercing it."""
        return f"{self.field} must be {self.expected}; observed {self.observed}"


@dataclass(slots=True)
class EventBufferContentError(ValueError):
    """Paired event lists cannot form one exact Arrow-compatible ragged view."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing licensed rows."""
        return f"invalid ragged event content: {self.violations!r}"


def extract_ragged_cash_dividends(
    frame: pl.DataFrame,
    *,
    times_column: str,
    amounts_column: str,
    policy: EventBufferPolicy,
) -> RaggedCashDividends:
    """Borrow paired Float64 list children and allocate only their required offsets.

    ``REQUIRE_ZERO_COPY`` rejects fragmented Polars storage. ``ALLOW_RECHUNK`` permits only the
    explicitly requested consolidation; it never casts values, changes event order, or repairs
    mismatched lists.

    Raises:
        EventBufferTypeError: If the boundary objects or policy have unsupported runtime types.
        EventBufferError: If storage cannot satisfy the declared copy policy.
        EventBufferContentError: If nulls, lengths, or exact event counts disagree.

    """
    _validate_extract_request(frame, times_column, amounts_column, policy)

    times = _event_list_series(frame.get_column(times_column), column=times_column, policy=policy)
    amounts = _event_list_series(
        frame.get_column(amounts_column),
        column=amounts_column,
        policy=policy,
    )
    violations: list[tuple[str, int]] = []
    if times.null_count() != 0:
        violations.append(("null_dividend_time_lists", times.null_count()))
    if amounts.null_count() != 0:
        violations.append(("null_dividend_amount_lists", amounts.null_count()))
    time_lengths = times.list.len()
    amount_lengths = amounts.list.len()
    paired_length_mismatches = cast("int", (time_lengths != amount_lengths).sum() or 0)
    if paired_length_mismatches != 0:
        violations.append(("paired_dividend_length_mismatches", paired_length_mismatches))

    exploded_times = times.explode(empty_as_null=False, keep_nulls=True)
    exploded_amounts = amounts.explode(empty_as_null=False, keep_nulls=True)
    if exploded_times.null_count() != 0:
        violations.append(("null_dividend_times", exploded_times.null_count()))
    if exploded_amounts.null_count() != 0:
        violations.append(("null_dividend_amounts", exploded_amounts.null_count()))
    if violations:
        raise EventBufferContentError(violations=tuple(violations))

    event_count = cast("int", time_lengths.cast(pl.UInt64).sum() or 0)
    lengths = cast(
        "Int64Vector",
        time_lengths.cast(pl.Int64).to_numpy(writable=False, allow_copy=True),
    )
    offsets = np.empty(frame.height + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    if int(offsets[-1]) != event_count:
        raise EventBufferContentError(violations=(("ragged_offset_overflow", 1),))

    time_values = _event_child_array(exploded_times, column=times_column, policy=policy)
    amount_values = _event_child_array(exploded_amounts, column=amounts_column, policy=policy)
    if time_values.shape != (event_count,) or amount_values.shape != (event_count,):
        raise EventBufferContentError(violations=(("ragged_event_count_mismatch", 1),))
    return RaggedCashDividends(offsets=offsets, times=time_values, amounts=amount_values)


def cash_event_violations(
    schedule: RaggedCashDividends,
    maturities: Float64Vector,
    maturity_boundary: MaturityBoundary,
) -> tuple[tuple[str, int], ...]:
    """Report exact cash-event domain violations without mutating source schedules.

    Callers validate the typed native buffers and row counts before this shared numerical check.
    Equal terminal times are admitted only by ``AT_OR_BEFORE``; events after maturity always fail.
    """
    times = schedule.times
    amounts = schedule.amounts
    violations: list[tuple[str, int]] = []
    for field, array in (("cash_dividend_times", times), ("cash_dividend_amounts", amounts)):
        count = int(np.count_nonzero(~np.isfinite(array)))
        if count:
            violations.append((f"nonfinite_{field}", count))
    nonpositive_times = int(np.count_nonzero(times <= 0.0))
    if nonpositive_times:
        violations.append(("nonpositive_cash_dividend_times", nonpositive_times))
    negative_amounts = int(np.count_nonzero(np.signbit(amounts)))
    if negative_amounts:
        violations.append(("negative_cash_dividend_amounts", negative_amounts))
    if times.size > 1:
        nonincreasing = times[1:] <= times[:-1]
        boundaries = schedule.offsets[1:-1]
        internal = boundaries[(boundaries > 0) & (boundaries < times.size)] - 1
        nonincreasing[internal] = False
        count = int(np.count_nonzero(nonincreasing))
        if count:
            violations.append(("nonincreasing_cash_dividend_times", count))
    violations.extend(_cash_event_maturity_violations(schedule, maturities, maturity_boundary))
    return tuple(violations)


def _cash_event_maturity_violations(
    schedule: RaggedCashDividends,
    maturities: Float64Vector,
    maturity_boundary: MaturityBoundary,
) -> tuple[tuple[str, int], ...]:
    nonempty = schedule.offsets[1:] > schedule.offsets[:-1]
    if not np.any(nonempty):
        return ()
    last_events = schedule.offsets[1:][nonempty] - 1
    terminal_times = schedule.times[last_events]
    contract_maturities = maturities[nonempty]
    violations: list[tuple[str, int]] = []
    after_maturity = int(np.count_nonzero(terminal_times > contract_maturities))
    if after_maturity:
        violations.append(("cash_dividend_schedules_after_maturity", after_maturity))
    if maturity_boundary is MaturityBoundary.STRICTLY_BEFORE:
        at_maturity = int(np.count_nonzero(terminal_times == contract_maturities))
        if at_maturity:
            violations.append(("cash_dividend_schedules_at_excluded_maturity", at_maturity))
    return tuple(violations)


def _validate_extract_request(
    frame: object,
    times_column: object,
    amounts_column: object,
    policy: object,
) -> None:
    if type(frame) is not pl.DataFrame:
        raise EventBufferTypeError(
            field="frame",
            observed=type(frame).__name__,
            expected="a materialized polars.DataFrame",
        )
    if type(times_column) is not str or not times_column:
        raise EventBufferTypeError(
            field="times_column",
            observed=repr(times_column),
            expected="a nonempty plain string",
        )
    if type(amounts_column) is not str or not amounts_column or amounts_column == times_column:
        raise EventBufferTypeError(
            field="amounts_column",
            observed=repr(amounts_column),
            expected="a distinct nonempty plain string",
        )
    if type(policy) is not EventBufferPolicy:
        raise EventBufferTypeError(
            field="event_buffer_policy",
            observed=repr(policy),
            expected="EventBufferPolicy",
        )


def _event_list_series(
    series: pl.Series,
    *,
    column: str,
    policy: EventBufferPolicy,
) -> pl.Series:
    if series.n_chunks() == 1:
        return series
    if policy is EventBufferPolicy.REQUIRE_ZERO_COPY:
        raise EventBufferError(
            column=column,
            policy=policy,
            reason="fragmented list storage requires an explicit rechunk",
        )
    return series.rechunk()


def _event_child_array(
    series: pl.Series,
    *,
    column: str,
    policy: EventBufferPolicy,
) -> Float64Vector:
    candidate = series
    try:
        converted = candidate.to_numpy(writable=False, allow_copy=False)
    except RuntimeError as error:
        if policy is EventBufferPolicy.REQUIRE_ZERO_COPY:
            raise EventBufferError(
                column=column,
                policy=policy,
                reason="fragmented child buffer requires an explicit rechunk",
            ) from error
        candidate = candidate.rechunk()
        try:
            converted = candidate.to_numpy(writable=False, allow_copy=False)
        except RuntimeError as rechunk_error:
            raise EventBufferError(
                column=column,
                policy=policy,
                reason="rechunked child buffer still cannot be borrowed by NumPy",
            ) from rechunk_error
    result = cast("Float64Vector", converted)
    if (
        type(result) is not np.ndarray
        or result.dtype != _FLOAT64_DTYPE
        or result.ndim != 1
        or not result.flags.c_contiguous
    ):
        raise EventBufferTypeError(
            field=column,
            observed=f"{type(result).__name__}, {getattr(result, 'dtype', None)}, "
            f"{getattr(result, 'shape', None)}",
            expected="a C-contiguous Float64 child buffer",
        )
    return result


__all__ = [
    "EventBufferContentError",
    "EventBufferError",
    "EventBufferPolicy",
    "EventBufferTypeError",
    "cash_event_violations",
    "extract_ragged_cash_dividends",
]
