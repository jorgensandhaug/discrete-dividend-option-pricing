"""Tests for calendar-owned exact publication exercise schedules."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from discrete_dividend_study.classical_groups import (
    build_project_classical_grouped_frame,
    prepare_project_classical_strike_groups,
)
from discrete_dividend_study.market_calendar import load_project_xnys_calendar
from discrete_dividend_study.publication_exercise import (
    PublicationExerciseScheduleError,
    prepare_publication_exercise_schedules,
    publication_exercise_schedules_sha256,
)

_SHA256_HEX_LENGTH = 64
_DAYS_PER_YEAR = 365.0
_GROUPS = 3


def schedule_frame() -> pl.DataFrame:
    """Return pre-expiry, terminal, and empty cash groups with exact ACT/365F dates."""
    rows = (
        ("AAA", date(2024, 1, 2), date(2024, 4, 19), 90.0, 95.0),
        ("AAA", date(2024, 1, 2), date(2024, 4, 19), 100.0, 95.0),
        ("BBB", date(2014, 11, 12), date(2014, 12, 26), 90.0, 100.0),
        ("CCC", date(2024, 1, 2), date(2024, 2, 16), 100.0, 105.0),
    )
    ex_dates = {
        "AAA": [date(2024, 2, 1), date(2024, 3, 1)],
        "BBB": [date(2014, 12, 26)],
        "CCC": [],
    }
    amounts = {"AAA": [1.0, 1.0], "BBB": [0.75], "CCC": []}
    return pl.DataFrame(
        {
            "ticker": [row[0] for row in rows],
            "quote_date": [row[1] for row in rows],
            "expiry_date": [row[2] for row in rows],
            "underlying_price": [row[4] for row in rows],
            "strike": [row[3] for row in rows],
            "yte": [(row[2] - row[1]).days / _DAYS_PER_YEAR for row in rows],
            "risk_free_rate": [0.03, 0.03, 0.02, 0.01],
            "historical_volatility_90": [0.20, 0.20, 0.25, 0.22],
            "forecasted_dividend_yield": [0.02, 0.02, 0.01, 0.0],
            "historical_dividend_yield": [0.01, 0.01, 0.02, 0.0],
            "call_mid_price": [10.0, 7.0, 12.0, 8.0],
            "forecasted_discrete_ex_dates": [ex_dates[row[0]] for row in rows],
            "forecasted_years_to_discrete_ex_dates": [
                [(event - row[1]).days / _DAYS_PER_YEAR for event in ex_dates[row[0]]]
                for row in rows
            ],
            "forecasted_discrete_amounts": [amounts[row[0]] for row in rows],
        },
        schema={
            "ticker": pl.String,
            "quote_date": pl.Date,
            "expiry_date": pl.Date,
            "underlying_price": pl.Float32,
            "strike": pl.Float32,
            "yte": pl.Float64,
            "risk_free_rate": pl.Float64,
            "historical_volatility_90": pl.Float64,
            "forecasted_dividend_yield": pl.Float64,
            "historical_dividend_yield": pl.Float64,
            "call_mid_price": pl.Float32,
            "forecasted_discrete_ex_dates": pl.List(pl.Date),
            "forecasted_years_to_discrete_ex_dates": pl.List(pl.Float64),
            "forecasted_discrete_amounts": pl.List(pl.Float64),
        },
    )


def test_calendar_owned_schedules_replace_only_terminal_cash_time() -> None:
    """Pre-expiry cash stays pre-jump, terminal cash receives the exact prior session."""
    materialized = build_project_classical_grouped_frame(schedule_frame().lazy()).collect()
    inputs = prepare_project_classical_strike_groups(materialized)
    prepared = prepare_publication_exercise_schedules(
        materialized,
        inputs,
        load_project_xnys_calendar(),
    )
    assert prepared.groups == _GROUPS
    assert prepared.terminal_groups == 1
    assert prepared.terminal_cash_groups.tolist() == [False, True, False]
    assert prepared.schedules.offsets.tolist() == [0, 2, 3, 3]
    assert prepared.schedules.times.tolist() == [
        30 / _DAYS_PER_YEAR,
        59 / _DAYS_PER_YEAR,
        42 / _DAYS_PER_YEAR,
    ]
    assert not prepared.schedules.offsets.flags.writeable
    assert not prepared.schedules.times.flags.writeable
    assert len(publication_exercise_schedules_sha256(prepared)) == _SHA256_HEX_LENGTH


def test_schedule_materialization_rejects_date_time_or_calendar_disagreement() -> None:
    """No malformed source date is shifted, snapped, inferred, or accepted by float proximity."""
    calendar = load_project_xnys_calendar()
    valid = build_project_classical_grouped_frame(schedule_frame().lazy()).collect()
    inputs = prepare_project_classical_strike_groups(valid)

    wrong_time = valid.with_columns(
        pl.when(pl.col("ticker") == "AAA")
        .then(pl.lit([0.1, 59 / _DAYS_PER_YEAR], dtype=pl.List(pl.Float64)))
        .otherwise(pl.col("forecasted_years_to_discrete_ex_dates"))
        .alias("forecasted_years_to_discrete_ex_dates")
    )
    wrong_time_inputs = prepare_project_classical_strike_groups(wrong_time)
    with pytest.raises(PublicationExerciseScheduleError, match="cash times differ"):
        prepare_publication_exercise_schedules(wrong_time, wrong_time_inputs, calendar)

    weekend_quote = valid.with_columns(
        pl.when(pl.col("ticker") == "CCC")
        .then(pl.lit(date(2024, 1, 7), dtype=pl.Date))
        .otherwise(pl.col("quote_date"))
        .alias("quote_date"),
        pl.when(pl.col("ticker") == "CCC")
        .then(pl.lit(40 / _DAYS_PER_YEAR, dtype=pl.Float64))
        .otherwise(pl.col("yte"))
        .alias("yte"),
    )
    weekend_inputs = prepare_project_classical_strike_groups(weekend_quote)
    with pytest.raises(PublicationExerciseScheduleError, match="not in calendar XNYS"):
        prepare_publication_exercise_schedules(weekend_quote, weekend_inputs, calendar)

    with pytest.raises(PublicationExerciseScheduleError, match="prepared group counts differ"):
        prepare_publication_exercise_schedules(valid.head(3), inputs, calendar)


def test_terminal_event_identity_is_owned_by_dates_not_float_equality() -> None:
    """An admitted one-ULP source encoding cannot change a terminal event's meaning."""
    terminal_time = np.nextafter(44 / _DAYS_PER_YEAR, -np.inf)
    source = schedule_frame().with_columns(
        pl.when(pl.col("ticker") == "BBB")
        .then(pl.lit([terminal_time], dtype=pl.List(pl.Float64)))
        .otherwise(pl.col("forecasted_years_to_discrete_ex_dates"))
        .alias("forecasted_years_to_discrete_ex_dates")
    )
    materialized = build_project_classical_grouped_frame(source.lazy()).collect()
    inputs = prepare_project_classical_strike_groups(materialized)
    prepared = prepare_publication_exercise_schedules(
        materialized,
        inputs,
        load_project_xnys_calendar(),
    )
    assert prepared.terminal_cash_groups.tolist() == [False, True, False]
    assert prepared.schedules.times.tolist()[-1] == 42 / _DAYS_PER_YEAR


def test_forecast_event_and_legal_expiry_dates_are_not_silently_reinterpreted() -> None:
    """Non-session forecasts remain model times; only terminal cash requires a session expiry."""
    source = schedule_frame().with_columns(
        pl.when(pl.col("ticker") == "AAA")
        .then(pl.lit([date(2024, 2, 3), date(2024, 3, 1)], dtype=pl.List(pl.Date)))
        .otherwise(pl.col("forecasted_discrete_ex_dates"))
        .alias("forecasted_discrete_ex_dates"),
        pl.when(pl.col("ticker") == "AAA")
        .then(pl.lit([32 / _DAYS_PER_YEAR, 59 / _DAYS_PER_YEAR], dtype=pl.List(pl.Float64)))
        .otherwise(pl.col("forecasted_years_to_discrete_ex_dates"))
        .alias("forecasted_years_to_discrete_ex_dates"),
        pl.when(pl.col("ticker") == "CCC")
        .then(pl.lit(date(2024, 2, 17), dtype=pl.Date))
        .otherwise(pl.col("expiry_date"))
        .alias("expiry_date"),
        pl.when(pl.col("ticker") == "CCC")
        .then(pl.lit(46 / _DAYS_PER_YEAR, dtype=pl.Float64))
        .otherwise(pl.col("yte"))
        .alias("yte"),
    )
    materialized = build_project_classical_grouped_frame(source.lazy()).collect()
    inputs = prepare_project_classical_strike_groups(materialized)
    prepared = prepare_publication_exercise_schedules(
        materialized,
        inputs,
        load_project_xnys_calendar(),
    )
    assert prepared.schedules.times.tolist()[:2] == [
        32 / _DAYS_PER_YEAR,
        59 / _DAYS_PER_YEAR,
    ]

    invalid_terminal = schedule_frame().with_columns(
        pl.when(pl.col("ticker") == "BBB")
        .then(pl.lit(date(2014, 12, 27), dtype=pl.Date))
        .otherwise(pl.col("expiry_date"))
        .alias("expiry_date"),
        pl.when(pl.col("ticker") == "BBB")
        .then(pl.lit(45 / _DAYS_PER_YEAR, dtype=pl.Float64))
        .otherwise(pl.col("yte"))
        .alias("yte"),
        pl.when(pl.col("ticker") == "BBB")
        .then(pl.lit([date(2014, 12, 27)], dtype=pl.List(pl.Date)))
        .otherwise(pl.col("forecasted_discrete_ex_dates"))
        .alias("forecasted_discrete_ex_dates"),
        pl.when(pl.col("ticker") == "BBB")
        .then(pl.lit([45 / _DAYS_PER_YEAR], dtype=pl.List(pl.Float64)))
        .otherwise(pl.col("forecasted_years_to_discrete_ex_dates"))
        .alias("forecasted_years_to_discrete_ex_dates"),
    )
    invalid_materialized = build_project_classical_grouped_frame(invalid_terminal.lazy()).collect()
    invalid_inputs = prepare_project_classical_strike_groups(invalid_materialized)
    with pytest.raises(PublicationExerciseScheduleError, match="terminal cash expiry_date"):
        prepare_publication_exercise_schedules(
            invalid_materialized,
            invalid_inputs,
            load_project_xnys_calendar(),
        )
