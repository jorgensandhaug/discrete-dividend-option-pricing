"""Rolling experiment schedule and fail-fast coverage tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from discrete_dividend_study.rolling_windows import (
    PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC,
    RollingPhase,
    RollingTestWindowCount,
    RollingWindow,
    RollingWindowContentError,
    RollingWindowSchemaError,
    RollingWindowSpec,
    RollingWindowSpecError,
    generate_rolling_windows,
    rolling_window_manifest,
    select_rolling_phase,
    select_rolling_test_union,
    validate_rolling_test_coverage,
    validate_rolling_window_coverage,
)

PROJECT_WINDOW_COUNT = 117
PROJECT_TRAINING_DAYS = 1_065
ROWS_PER_BOUNDARY_PHASE = 2


def test_submitted_schedule_has_exact_executable_boundaries() -> None:
    """The paper's 30-day stepping identity resolves to 117 immutable windows."""
    windows = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    assert len(windows) == PROJECT_WINDOW_COUNT
    assert windows[0].window_id == 0
    assert windows[0].train_start == date(2012, 1, 2)
    assert windows[0].validation_start == date(2014, 12, 2)
    assert windows[0].test_start == date(2015, 1, 1)
    assert windows[0].test_end == date(2015, 1, 31)
    assert windows[-1].window_id == PROJECT_WINDOW_COUNT - 1
    assert windows[-1].test_start == date(2024, 7, 12)
    assert windows[-1].test_end == date(2024, 8, 11)


def test_manifest_is_small_ordered_and_strictly_typed() -> None:
    """The resolved schedule is directly persistable as run-manifest data."""
    observed = rolling_window_manifest(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    assert observed.shape == (PROJECT_WINDOW_COUNT, 5)
    assert observed.schema == pl.Schema(
        {
            "window_id": pl.UInt32,
            "train_start": pl.Date,
            "validation_start": pl.Date,
            "test_start": pl.Date,
            "test_end": pl.Date,
        },
    )
    assert observed["window_id"].to_list() == list(range(PROJECT_WINDOW_COUNT))


def test_half_open_phase_selection_assigns_boundaries_once() -> None:
    """Adjacent train, validation, and test intervals neither overlap nor leave gaps."""
    window = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)[0]
    source = pl.LazyFrame(
        {
            "quote_date": [
                window.train_start,
                window.validation_start - timedelta(days=1),
                window.validation_start,
                window.test_start - timedelta(days=1),
                window.test_start,
                window.test_end - timedelta(days=1),
                window.test_end,
            ],
        },
        schema={"quote_date": pl.Date},
    )
    assert (
        select_rolling_phase(source, window, RollingPhase.TRAIN).collect().height
        == ROWS_PER_BOUNDARY_PHASE
    )
    assert (
        select_rolling_phase(source, window, RollingPhase.VALIDATION).collect().height
        == ROWS_PER_BOUNDARY_PHASE
    )
    assert (
        select_rolling_phase(source, window, RollingPhase.TEST).collect().height
        == ROWS_PER_BOUNDARY_PHASE
    )


def test_adjacent_test_union_projects_each_submitted_date_once() -> None:
    """One lazy range is exactly the union of the submitted adjacent test partitions."""
    windows = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    source = pl.LazyFrame(
        {
            "quote_date": [
                windows[0].test_start - timedelta(days=1),
                windows[0].test_start,
                windows[0].test_end - timedelta(days=1),
                windows[1].test_start,
                windows[-1].test_end - timedelta(days=1),
                windows[-1].test_end,
            ],
            "value": list(range(6)),
        },
        schema={"quote_date": pl.Date, "value": pl.Int64},
    )
    observed = select_rolling_test_union(source, windows).collect()
    assert observed["value"].to_list() == [1, 2, 3, 4]


def test_test_only_coverage_counts_and_rejects_schedule_gaps() -> None:
    """Classical pooled evaluation validates every test window without phase expansion."""
    spec = RollingWindowSpec(
        total_lookback_days=60,
        validation_days=10,
        test_days=10,
        step_days=10,
        first_test_start=date(2024, 3, 1),
        latest_test_end=date(2024, 3, 21),
    )
    windows = generate_rolling_windows(spec)
    source = pl.LazyFrame(
        {
            "quote_date": [
                date(2024, 3, 1),
                date(2024, 3, 2),
                date(2024, 3, 11),
            ],
        },
        schema={"quote_date": pl.Date},
    )
    assert validate_rolling_test_coverage(source, windows) == (
        RollingTestWindowCount(window_id=0, test_rows=2),
        RollingTestWindowCount(window_id=1, test_rows=1),
    )

    gapped = (windows[0], replace(windows[1], test_start=date(2024, 3, 12)))
    with pytest.raises(RollingWindowSpecError, match="adjacent"):
        select_rolling_test_union(source, gapped)


def test_one_date_count_scan_validates_every_required_partition() -> None:
    """Coverage counts preserve the exact calendar-day phase lengths without row expansion."""
    windows = generate_rolling_windows(PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC)
    source = pl.LazyFrame(
        {
            "quote_date": pl.date_range(
                windows[0].train_start,
                windows[-1].test_end - timedelta(days=1),
                interval="1d",
                eager=True,
            ),
        },
    )
    counts = validate_rolling_window_coverage(source, windows)
    assert len(counts) == PROJECT_WINDOW_COUNT
    assert counts[0].train_rows == PROJECT_TRAINING_DAYS
    assert counts[0].validation_rows == PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC.validation_days
    assert counts[0].test_rows == PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC.test_days
    assert counts[-1].train_rows == PROJECT_TRAINING_DAYS


def test_empty_partition_and_invalid_inputs_fail_instead_of_skipping() -> None:
    """Incomplete runs, inferred values, nullable dates, and schema drift are explicit errors."""
    spec = RollingWindowSpec(
        total_lookback_days=60,
        validation_days=10,
        test_days=10,
        step_days=10,
        first_test_start=date(2024, 3, 1),
        latest_test_end=date(2024, 3, 11),
    )
    windows = generate_rolling_windows(spec)
    incomplete = pl.LazyFrame(
        {"quote_date": [date(2024, 1, 2), date(2024, 3, 1)]},
        schema={"quote_date": pl.Date},
    )
    with pytest.raises(RollingWindowContentError, match="empty_validation_partitions"):
        validate_rolling_window_coverage(incomplete, windows)

    nullable = pl.LazyFrame(
        {"quote_date": [date(2024, 1, 2), None]},
        schema={"quote_date": pl.Date},
    )
    with pytest.raises(RollingWindowContentError, match="null_quote_dates"):
        validate_rolling_window_coverage(nullable, windows)

    with pytest.raises(RollingWindowSchemaError, match="Date"):
        select_rolling_phase(
            pl.LazyFrame({"quote_date": ["2024-03-01"]}),
            windows[0],
            RollingPhase.TEST,
        )

    invalid_days: object = True
    with pytest.raises(RollingWindowSpecError, match="positive plain int"):
        RollingWindowSpec(
            total_lookback_days=60,
            validation_days=10,
            test_days=10,
            step_days=invalid_days,  # type: ignore[arg-type]
            first_test_start=date(2024, 3, 1),
            latest_test_end=date(2024, 3, 11),
        )
    with pytest.raises(RollingWindowSpecError, match=r"plain datetime\.date"):
        RollingWindowSpec(
            total_lookback_days=60,
            validation_days=10,
            test_days=10,
            step_days=10,
            first_test_start=datetime(2024, 3, 1, tzinfo=UTC),
            latest_test_end=date(2024, 3, 11),
        )
    with pytest.raises(RollingWindowSpecError, match="strictly increasing"):
        RollingWindow(
            window_id=0,
            train_start=date(2024, 1, 1),
            validation_start=date(2024, 2, 1),
            test_start=date(2024, 2, 1),
            test_end=date(2024, 3, 1),
        )
