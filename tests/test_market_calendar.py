"""Tests for immutable market-session inputs."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from discrete_dividend_study.market_calendar import (
    MarketCalendarError,
    load_project_xnys_calendar,
)

_EXPECTED_SESSIONS = 4_276


def test_project_calendar_has_frozen_identity_and_historical_closures() -> None:
    """The checked-in source captures ordinary holidays and exceptional full closures."""
    calendar = load_project_xnys_calendar()
    assert calendar.name == "XNYS"
    assert calendar.source == "exchange_calendars"
    assert calendar.source_version == "4.13.2"
    assert (
        calendar.asset_sha256 == "19683a870cf0eb1983b1727c0c62fded512c1b7440c0812e2984ea666311cd3b"
    )
    assert calendar.sessions == _EXPECTED_SESSIONS
    assert calendar.first_session == date(2009, 1, 2)
    assert calendar.last_session == date(2025, 12, 31)
    assert not calendar.session_ordinals.flags.writeable

    assert not calendar.is_session(date(2012, 10, 29))
    assert not calendar.is_session(date(2012, 10, 30))
    assert not calendar.is_session(date(2018, 12, 5))
    assert calendar.is_session(date(2021, 6, 18))
    assert not calendar.is_session(date(2022, 6, 20))
    assert not calendar.is_session(date(2024, 3, 29))


def test_previous_session_requires_an_exact_retained_session() -> None:
    """No weekend, holiday, range extension, or datetime is repaired implicitly."""
    calendar = load_project_xnys_calendar()
    assert calendar.previous_session(date(2014, 12, 26)) == date(2014, 12, 24)
    assert calendar.previous_session(date(2024, 11, 15)) == date(2024, 11, 14)
    with pytest.raises(MarketCalendarError, match="not a retained session"):
        calendar.previous_session(date(2014, 12, 25))
    with pytest.raises(MarketCalendarError, match="no prior session"):
        calendar.previous_session(calendar.first_session)
    with pytest.raises(MarketCalendarError, match="plain date"):
        calendar.is_session(datetime(2024, 1, 2, tzinfo=UTC))
