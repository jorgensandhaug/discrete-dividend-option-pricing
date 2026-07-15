"""Immutable trading-session inputs for calendar-owned exercise schedules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from functools import cache
from importlib import resources
from typing import Final

import numpy as np
import numpy.typing as npt

SessionOrdinals = npt.NDArray[np.int32]

_ASSET_NAME: Final = "xnys_sessions_2009_2025.txt"
_ASSET_SHA256: Final = "19683a870cf0eb1983b1727c0c62fded512c1b7440c0812e2984ea666311cd3b"
_CALENDAR_NAME: Final = "XNYS"
_SOURCE_NAME: Final = "exchange_calendars"
_SOURCE_VERSION: Final = "4.13.2"
_START_DATE: Final = date(2009, 1, 1)
_END_DATE: Final = date(2025, 12, 31)
_EXPECTED_SESSIONS: Final = 4_276
_FIRST_WEEKEND_DAY: Final = 5
_MINIMUM_SESSIONS: Final = 2
_SHA256_HEX_LENGTH: Final = 64
_HEADER: Final = (
    "# generated; do not edit",
    f"# calendar={_CALENDAR_NAME}",
    f"# source={_SOURCE_NAME}",
    f"# source_version={_SOURCE_VERSION}",
    f"# requested_start={_START_DATE.isoformat()}",
    f"# requested_end={_END_DATE.isoformat()}",
)


@dataclass(slots=True)
class MarketCalendarError(ValueError):
    """A calendar asset or requested session violates its exact identity."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected calendar contract."""
        return f"invalid market calendar: {self.reason}"


@dataclass(frozen=True, slots=True, eq=False)
class TradingSessionCalendar:
    """One versioned, immutable sequence of eligible market sessions."""

    name: str
    source: str
    source_version: str
    asset_sha256: str
    session_ordinals: SessionOrdinals

    def __post_init__(self) -> None:
        """Reject mutated metadata or malformed session axes."""
        if type(self.name) is not str or not self.name:
            raise MarketCalendarError(reason="name must be a nonempty plain string")
        if type(self.source) is not str or not self.source:
            raise MarketCalendarError(reason="source must be a nonempty plain string")
        if type(self.source_version) is not str or not self.source_version:
            raise MarketCalendarError(reason="source_version must be a nonempty plain string")
        if (
            type(self.asset_sha256) is not str
            or len(self.asset_sha256) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in self.asset_sha256)
        ):
            raise MarketCalendarError(reason="asset_sha256 must be lowercase SHA-256")
        if type(self.session_ordinals) is not np.ndarray:
            raise MarketCalendarError(reason="session_ordinals must be a plain numpy.ndarray")
        sessions = self.session_ordinals
        if (
            sessions.dtype != np.dtype(np.int32)
            or sessions.ndim != 1
            or not sessions.flags.c_contiguous
            or sessions.size < _MINIMUM_SESSIONS
        ):
            raise MarketCalendarError(
                reason="session_ordinals must be a contiguous one-dimensional Int32 vector"
            )
        if np.any(sessions[1:] <= sessions[:-1]):
            raise MarketCalendarError(reason="sessions must be strictly increasing")
        if any(
            date.fromordinal(int(ordinal)).weekday() >= _FIRST_WEEKEND_DAY for ordinal in sessions
        ):
            raise MarketCalendarError(reason="a declared session falls on a weekend")
        sessions.flags.writeable = False

    @property
    def sessions(self) -> int:
        """Return the exact retained session count."""
        return int(self.session_ordinals.size)

    @property
    def first_session(self) -> date:
        """Return the first available session."""
        return date.fromordinal(int(self.session_ordinals[0]))

    @property
    def last_session(self) -> date:
        """Return the last available session."""
        return date.fromordinal(int(self.session_ordinals[-1]))

    def is_session(self, value: date) -> bool:
        """Return whether one plain date is represented exactly."""
        _require_plain_date(value)
        ordinal = value.toordinal()
        index = int(np.searchsorted(self.session_ordinals, ordinal, side="left"))
        return index < self.sessions and int(self.session_ordinals[index]) == ordinal

    def previous_session(self, value: date) -> date:
        """Return the prior session, requiring `value` itself to be a retained session."""
        _require_plain_date(value)
        ordinal = value.toordinal()
        index = int(np.searchsorted(self.session_ordinals, ordinal, side="left"))
        if index >= self.sessions or int(self.session_ordinals[index]) != ordinal:
            raise MarketCalendarError(reason=f"{value.isoformat()} is not a retained session")
        if index == 0:
            raise MarketCalendarError(
                reason=f"no prior session is retained for {value.isoformat()}"
            )
        return date.fromordinal(int(self.session_ordinals[index - 1]))


@cache
def load_project_xnys_calendar() -> TradingSessionCalendar:
    """Load and verify the checked-in 2009--2025 XNYS session asset."""
    asset = resources.files("discrete_dividend_study").joinpath("resources", _ASSET_NAME)
    payload = asset.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != _ASSET_SHA256:
        raise MarketCalendarError(
            reason=f"asset SHA-256 differs: {observed_sha256} != {_ASSET_SHA256}"
        )
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise MarketCalendarError(reason="asset is not canonical ASCII") from error
    if tuple(lines[: len(_HEADER)]) != _HEADER:
        raise MarketCalendarError(reason="asset metadata header differs")
    raw_sessions = lines[len(_HEADER) :]
    if len(raw_sessions) != _EXPECTED_SESSIONS or any(not value for value in raw_sessions):
        raise MarketCalendarError(reason="asset session count or blank-line contract differs")
    try:
        sessions = np.fromiter(
            (date.fromisoformat(value).toordinal() for value in raw_sessions),
            dtype=np.int32,
            count=len(raw_sessions),
        )
    except ValueError as error:
        raise MarketCalendarError(reason="asset contains a malformed ISO session date") from error
    calendar = TradingSessionCalendar(
        name=_CALENDAR_NAME,
        source=_SOURCE_NAME,
        source_version=_SOURCE_VERSION,
        asset_sha256=observed_sha256,
        session_ordinals=sessions,
    )
    if calendar.first_session < _START_DATE or calendar.last_session > _END_DATE:
        raise MarketCalendarError(reason="asset sessions exceed the requested source interval")
    return calendar


def _require_plain_date(value: object) -> None:
    if type(value) is not date:
        raise MarketCalendarError(reason=f"session must be a plain date; observed {value!r}")


__all__ = [
    "MarketCalendarError",
    "TradingSessionCalendar",
    "load_project_xnys_calendar",
]
