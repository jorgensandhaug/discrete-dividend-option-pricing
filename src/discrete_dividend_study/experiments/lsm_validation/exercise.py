"""Calendar-owned exact cash-call exercise schedules for publication pricing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from finoptions import RaggedExerciseTimes

from discrete_dividend_study.core.calendar import TradingSessionCalendar
from discrete_dividend_study.models.lsm.groups import (
    PreparedClassicalStrikeGroups,
    prepared_classical_strike_groups_sha256,
)

BoolVector = npt.NDArray[np.bool_]

_FORMAT: Final = "project-publication-exercise-schedules-v1"
_GROUP_KEYS: Final = ("ticker", "quote_date", "expiry_date")
_EX_DATE_COLUMN: Final = "forecasted_discrete_ex_dates"
_MATURITY_COLUMN: Final = "yte"
_DAYS_PER_YEAR: Final = 365.0
_DAY_COUNT_NAME: Final = "act_365_fixed"
_MAX_SOURCE_DAY_COUNT_ULPS: Final = 1
_SHA256_HEX_LENGTH: Final = 64
_DATE_ORDINAL_DTYPE: Final = np.dtype(np.int32)


class _HashWriter(Protocol):
    """Minimal streaming hash interface."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(slots=True)
class PublicationExerciseScheduleError(ValueError):
    """Source dates cannot define the declared exact exercise schedule."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected schedule contract."""
        return f"invalid publication exercise schedule: {self.reason}"


@dataclass(frozen=True, slots=True, eq=False)
class PreparedPublicationExerciseSchedules:
    """Immutable group-aligned opportunities and their complete calendar provenance."""

    source_prepared_sha256: str
    source_group_keys_sha256: str
    calendar_name: str
    calendar_source: str
    calendar_source_version: str
    calendar_asset_sha256: str
    day_count: str
    schedules: RaggedExerciseTimes
    terminal_cash_groups: BoolVector

    def __post_init__(self) -> None:
        """Reject incomplete provenance, mutable masks, or group-axis disagreement."""
        for field, value in (
            ("source_prepared_sha256", self.source_prepared_sha256),
            ("source_group_keys_sha256", self.source_group_keys_sha256),
            ("calendar_asset_sha256", self.calendar_asset_sha256),
        ):
            if (
                type(value) is not str
                or len(value) != _SHA256_HEX_LENGTH
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise PublicationExerciseScheduleError(reason=f"{field} must be lowercase SHA-256")
        for field, value in (
            ("calendar_name", self.calendar_name),
            ("calendar_source", self.calendar_source),
            ("calendar_source_version", self.calendar_source_version),
            ("day_count", self.day_count),
        ):
            if type(value) is not str or not value:
                raise PublicationExerciseScheduleError(
                    reason=f"{field} must be a nonempty plain string"
                )
        if self.day_count != _DAY_COUNT_NAME:
            raise PublicationExerciseScheduleError(reason="day_count must be act_365_fixed")
        if type(self.schedules) is not RaggedExerciseTimes:
            raise PublicationExerciseScheduleError(reason="schedules must be RaggedExerciseTimes")
        if type(self.terminal_cash_groups) is not np.ndarray:
            raise PublicationExerciseScheduleError(
                reason="terminal_cash_groups must be a plain numpy.ndarray"
            )
        terminal = cast("npt.NDArray[np.generic]", self.terminal_cash_groups)
        if (
            terminal.dtype != np.dtype(np.bool_)
            or terminal.ndim != 1
            or not terminal.flags.c_contiguous
            or terminal.size != self.schedules.rows
        ):
            raise PublicationExerciseScheduleError(
                reason="terminal_cash_groups must be one contiguous Bool per schedule"
            )
        terminal.flags.writeable = False

    @property
    def groups(self) -> int:
        """Return the exact source-group count."""
        return self.schedules.rows

    @property
    def terminal_groups(self) -> int:
        """Return the number of groups whose final cash event occurs at expiry."""
        return int(np.count_nonzero(self.terminal_cash_groups))


def prepare_publication_exercise_schedules(
    frame: pl.DataFrame,
    inputs: PreparedClassicalStrikeGroups,
    calendar: TradingSessionCalendar,
) -> PreparedPublicationExerciseSchedules:
    """Map source civil dates to exact group schedules without inferring missing sessions."""
    _validate_request_types(frame, inputs, calendar)
    grouped, quote_ordinals, expiry_ordinals = _aligned_group_dates(
        frame,
        inputs,
        calendar,
    )
    event_ordinals = _validate_event_dates(grouped, inputs, quote_ordinals, expiry_ordinals)
    schedules, terminal_groups = _build_calendar_owned_schedules(
        inputs,
        calendar,
        quote_ordinals,
        expiry_ordinals,
        event_ordinals,
    )
    return PreparedPublicationExerciseSchedules(
        source_prepared_sha256=prepared_classical_strike_groups_sha256(inputs),
        source_group_keys_sha256=_group_keys_sha256(grouped),
        calendar_name=calendar.name,
        calendar_source=calendar.source,
        calendar_source_version=calendar.source_version,
        calendar_asset_sha256=calendar.asset_sha256,
        day_count=_DAY_COUNT_NAME,
        schedules=schedules,
        terminal_cash_groups=terminal_groups,
    )


def _validate_request_types(frame: object, inputs: object, calendar: object) -> None:
    if type(frame) is not pl.DataFrame:
        raise PublicationExerciseScheduleError(
            reason="frame must be a materialized Polars DataFrame"
        )
    if type(inputs) is not PreparedClassicalStrikeGroups:
        raise PublicationExerciseScheduleError(
            reason="inputs must be PreparedClassicalStrikeGroups"
        )
    if type(calendar) is not TradingSessionCalendar:
        raise PublicationExerciseScheduleError(reason="calendar must be TradingSessionCalendar")


def _aligned_group_dates(
    frame: pl.DataFrame,
    inputs: PreparedClassicalStrikeGroups,
    calendar: TradingSessionCalendar,
) -> tuple[
    pl.DataFrame,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    _validate_schema(frame)
    grouped = frame.group_by(*_GROUP_KEYS, maintain_order=True).agg(
        pl.len().alias("_group_size"),
        pl.col(_MATURITY_COLUMN).first(),
        pl.col(_EX_DATE_COLUMN).first(),
    )
    if grouped.height != inputs.groups:
        raise PublicationExerciseScheduleError(reason="frame and prepared group counts differ")
    group_sizes = _int64_vector(grouped["_group_size"])
    if not np.array_equal(group_sizes, np.diff(inputs.group_offsets).astype(np.int64, copy=False)):
        raise PublicationExerciseScheduleError(reason="frame and prepared group boundaries differ")
    quote_ordinals = _ordinals(_plain_dates(grouped["quote_date"], field="quote_date"))
    expiry_ordinals = _ordinals(_plain_dates(grouped["expiry_date"], field="expiry_date"))
    if np.any(expiry_ordinals <= quote_ordinals):
        raise PublicationExerciseScheduleError(reason="expiry dates must follow quote dates")
    _require_sessions(calendar, quote_ordinals, field="quote_date")
    grouped_maturities = _float64_vector(grouped[_MATURITY_COLUMN])
    if not _float_bits_equal(grouped_maturities, inputs.maturities) or not _act365f_matches(
        inputs.maturities,
        expiry_ordinals - quote_ordinals,
    ):
        raise PublicationExerciseScheduleError(
            reason="maturity differs from exact quote-to-expiry ACT/365F"
        )
    return grouped, quote_ordinals, expiry_ordinals


def _validate_event_dates(
    grouped: pl.DataFrame,
    inputs: PreparedClassicalStrikeGroups,
    quote_ordinals: npt.NDArray[np.int32],
    expiry_ordinals: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    cash_sizes = np.diff(inputs.cash_dividends.offsets).astype(np.int64, copy=False)
    event_ordinals = _flatten_event_ordinals(grouped[_EX_DATE_COLUMN], cash_sizes)
    repeated_quote = np.repeat(quote_ordinals, cash_sizes)
    repeated_expiry = np.repeat(expiry_ordinals, cash_sizes)
    if np.any(event_ordinals <= repeated_quote) or np.any(event_ordinals > repeated_expiry):
        raise PublicationExerciseScheduleError(
            reason="cash ex-dates must lie strictly after quote and at or before expiry"
        )
    if not _act365f_matches(inputs.cash_dividends.times, event_ordinals - repeated_quote):
        raise PublicationExerciseScheduleError(
            reason="cash times differ from exact quote-to-ex-date ACT/365F"
        )
    return event_ordinals


def _build_calendar_owned_schedules(
    inputs: PreparedClassicalStrikeGroups,
    calendar: TradingSessionCalendar,
    quote_ordinals: npt.NDArray[np.int32],
    expiry_ordinals: npt.NDArray[np.int32],
    event_ordinals: npt.NDArray[np.int32],
) -> tuple[RaggedExerciseTimes, BoolVector]:
    terminal_groups = _terminal_cash_groups(inputs, event_ordinals, expiry_ordinals)
    exercise_times = np.ascontiguousarray(inputs.cash_dividends.times.copy())
    offset_base = int(inputs.cash_dividends.offsets[0])
    terminal_indices = np.flatnonzero(terminal_groups)
    terminal_expiry_indices = _require_sessions(
        calendar,
        expiry_ordinals[terminal_indices],
        field="terminal cash expiry_date",
    )
    for group, expiry_session_index in zip(
        terminal_indices,
        terminal_expiry_indices,
        strict=True,
    ):
        start = int(inputs.cash_dividends.offsets[group]) - offset_base
        end = int(inputs.cash_dividends.offsets[group + 1]) - offset_base
        if expiry_session_index == 0:
            raise PublicationExerciseScheduleError(
                reason="calendar does not retain the terminal-cash predecessor"
            )
        predecessor_ordinal = int(calendar.session_ordinals[expiry_session_index - 1])
        if predecessor_ordinal <= int(quote_ordinals[group]):
            raise PublicationExerciseScheduleError(
                reason="terminal-cash predecessor is not strictly after quote_date"
            )
        predecessor_time = float(predecessor_ordinal - int(quote_ordinals[group])) / _DAYS_PER_YEAR
        if end < 1:
            raise PublicationExerciseScheduleError(
                reason="terminal-cash group has no replaceable terminal event"
            )
        if end - start > 1 and not exercise_times[end - 2] < predecessor_time:
            raise PublicationExerciseScheduleError(
                reason="terminal predecessor collides with or precedes another cash opportunity"
            )
        exercise_times[end - 1] = predecessor_time
    exercise_offsets = inputs.cash_dividends.offsets.view()
    exercise_offsets.flags.writeable = False
    exercise_times.flags.writeable = False
    terminal_groups = np.ascontiguousarray(terminal_groups)
    terminal_groups.flags.writeable = False
    return RaggedExerciseTimes(offsets=exercise_offsets, times=exercise_times), terminal_groups


def publication_exercise_schedules_sha256(
    prepared: PreparedPublicationExerciseSchedules,
) -> str:
    """Hash source identity, calendar provenance, terminal mask, and exact schedule bytes."""
    if type(prepared) is not PreparedPublicationExerciseSchedules:
        raise PublicationExerciseScheduleError(
            reason="prepared must be PreparedPublicationExerciseSchedules"
        )
    hasher = hashlib.sha256()
    for value in (
        _FORMAT,
        prepared.source_prepared_sha256,
        prepared.source_group_keys_sha256,
        prepared.calendar_name,
        prepared.calendar_source,
        prepared.calendar_source_version,
        prepared.calendar_asset_sha256,
        prepared.day_count,
    ):
        _hash_text(hasher, value)
    _hash_array(hasher, "exercise_offsets", prepared.schedules.offsets)
    _hash_array(hasher, "exercise_times", prepared.schedules.times)
    _hash_array(hasher, "terminal_cash_groups", prepared.terminal_cash_groups)
    return hasher.hexdigest()


def _validate_schema(frame: pl.DataFrame) -> None:
    expected = {
        "ticker": pl.String,
        "quote_date": pl.Date,
        "expiry_date": pl.Date,
        _MATURITY_COLUMN: pl.Float64,
        _EX_DATE_COLUMN: pl.List(pl.Date),
    }
    missing = tuple(field for field in expected if field not in frame.schema)
    mismatches = tuple(
        (field, str(dtype), str(frame.schema[field]))
        for field, dtype in expected.items()
        if field in frame.schema and frame.schema[field] != dtype
    )
    if missing or mismatches:
        raise PublicationExerciseScheduleError(
            reason=f"schema differs; missing={missing!r}, mismatches={mismatches!r}"
        )


def _plain_dates(series: pl.Series, *, field: str) -> tuple[date, ...]:
    values = series.to_list()
    if any(type(value) is not date for value in values):
        raise PublicationExerciseScheduleError(reason=f"{field} contains a null or non-date")
    return cast("tuple[date, ...]", tuple(values))


def _ordinals(values: tuple[date, ...]) -> npt.NDArray[np.int32]:
    return np.fromiter(
        (value.toordinal() for value in values),
        dtype=np.int32,
        count=len(values),
    )


def _require_sessions(
    calendar: TradingSessionCalendar,
    ordinals: npt.NDArray[np.int32],
    *,
    field: str,
) -> npt.NDArray[np.int64]:
    indices = np.searchsorted(calendar.session_ordinals, ordinals, side="left")
    within = indices < calendar.sessions
    safe = np.minimum(indices, calendar.sessions - 1)
    exact = calendar.session_ordinals[safe] == ordinals
    invalid = np.flatnonzero(~within | ~exact)
    if invalid.size:
        observed = date.fromordinal(int(ordinals[int(invalid[0])]))
        raise PublicationExerciseScheduleError(
            reason=f"{field} {observed.isoformat()} is not in calendar {calendar.name}"
        )
    return indices.astype(np.int64, copy=False)


def _flatten_event_ordinals(
    series: pl.Series,
    expected_sizes: npt.NDArray[np.int64],
) -> npt.NDArray[np.int32]:
    rows = series.to_list()
    if len(rows) != expected_sizes.size:
        raise PublicationExerciseScheduleError(reason="cash date-list row count differs")
    flattened: list[int] = []
    for group, (values, expected_size) in enumerate(zip(rows, expected_sizes, strict=True)):
        if type(values) is not list or len(values) != int(expected_size):
            raise PublicationExerciseScheduleError(
                reason=f"cash date-list length differs at group {group}"
            )
        previous = 0
        for value in values:
            if type(value) is not date:
                raise PublicationExerciseScheduleError(
                    reason=f"cash date-list contains a null or non-date at group {group}"
                )
            ordinal = value.toordinal()
            if ordinal <= previous:
                raise PublicationExerciseScheduleError(
                    reason=f"cash ex-dates are not strictly increasing at group {group}"
                )
            flattened.append(ordinal)
            previous = ordinal
    return np.asarray(flattened, dtype=_DATE_ORDINAL_DTYPE)


def _terminal_cash_groups(
    inputs: PreparedClassicalStrikeGroups,
    event_ordinals: npt.NDArray[np.int32],
    expiry_ordinals: npt.NDArray[np.int32],
) -> BoolVector:
    """Classify terminal events from exact civil dates, never floating-point equality."""
    terminal = np.zeros(inputs.groups, dtype=np.bool_)
    offset_base = int(inputs.cash_dividends.offsets[0])
    for group in range(inputs.groups):
        end = int(inputs.cash_dividends.offsets[group + 1]) - offset_base
        start = int(inputs.cash_dividends.offsets[group]) - offset_base
        if end > start:
            terminal[group] = event_ordinals[end - 1] == expiry_ordinals[group]
    return terminal


def _group_keys_sha256(grouped: pl.DataFrame) -> str:
    hasher = hashlib.sha256()
    _hash_text(hasher, "project-publication-group-keys-v1")
    tickers = grouped.get_column("ticker").to_list()
    quote_dates = _plain_dates(grouped.get_column("quote_date"), field="quote_date")
    expiry_dates = _plain_dates(grouped.get_column("expiry_date"), field="expiry_date")
    for ticker, quote_date, expiry_date in zip(tickers, quote_dates, expiry_dates, strict=True):
        if type(ticker) is not str or not ticker:
            raise PublicationExerciseScheduleError(reason="ticker is null or blank")
        _hash_text(hasher, ticker)
        hasher.update(quote_date.toordinal().to_bytes(4, byteorder="little", signed=False))
        hasher.update(expiry_date.toordinal().to_bytes(4, byteorder="little", signed=False))
    return hasher.hexdigest()


def _int64_vector(series: pl.Series) -> npt.NDArray[np.int64]:
    return np.asarray(series.to_numpy(writable=False, allow_copy=True), dtype=np.int64)


def _float64_vector(series: pl.Series) -> npt.NDArray[np.float64]:
    return np.asarray(series.to_numpy(writable=False, allow_copy=True), dtype=np.float64)


def _float_bits_equal(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> bool:
    return left.shape == right.shape and bool(
        np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _act365f_matches(
    observed: npt.NDArray[np.float64],
    calendar_days: npt.NDArray[np.int32],
) -> bool:
    """Admit only the two adjacent encodings produced by scalar and vector ACT/365F division."""
    if observed.shape != calendar_days.shape:
        return False
    expected = calendar_days.astype(np.float64) / _DAYS_PER_YEAR
    observed_bits = observed.view(np.uint64)
    expected_bits = expected.view(np.uint64)
    ulps = np.maximum(observed_bits, expected_bits) - np.minimum(observed_bits, expected_bits)
    return bool(np.all(ulps <= _MAX_SOURCE_DAY_COUNT_ULPS))


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _hash_array(hasher: _HashWriter, field: str, array: npt.NDArray[np.generic]) -> None:
    _hash_text(hasher, field)
    _hash_text(hasher, array.dtype.str)
    hasher.update(array.size.to_bytes(8, byteorder="little", signed=False))
    hasher.update(memoryview(array).cast("B"))


__all__ = [
    "PreparedPublicationExerciseSchedules",
    "PublicationExerciseScheduleError",
    "prepare_publication_exercise_schedules",
    "publication_exercise_schedules_sha256",
]
