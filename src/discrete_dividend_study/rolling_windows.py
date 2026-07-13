"""Typed half-open rolling-window identities for project experiments."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Final

import polars as pl

from .schema import require_columns


class RollingWindowError(ValueError):
    """Base class for invalid rolling specifications, inputs, or coverage."""


@dataclass(frozen=True, slots=True)
class RollingWindowPhaseError(TypeError):
    """A phase selector is not a member of the closed phase enum."""

    observed: object

    def __str__(self) -> str:
        """Describe the invalid selector without coercing it."""
        return f"phase must be a RollingPhase; observed {self.observed!r}"


@dataclass(frozen=True, slots=True)
class RollingWindowSpecError(RollingWindowError):
    """A rolling-window specification is outside its declared domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the invalid value and required domain."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


@dataclass(frozen=True, slots=True)
class RollingWindowSchemaError(RollingWindowError):
    """The rolling source has an incorrect quote-date dtype."""

    observed: str

    def __str__(self) -> str:
        """Describe the strict date schema contract."""
        return f"quote_date must have dtype Date; observed {self.observed}"


@dataclass(frozen=True, slots=True)
class RollingWindowContentError(RollingWindowError):
    """The rolling source contains invalid dates or empty required partitions."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing licensed rows."""
        return f"invalid rolling-window content: {self.violations!r}"


class RollingPhase(StrEnum):
    """A non-overlapping phase inside one rolling experiment window."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class RollingWindowSpec:
    """Calendar-day identity for a sequence of half-open rolling windows."""

    total_lookback_days: int
    validation_days: int
    test_days: int
    step_days: int
    first_test_start: date
    latest_test_end: date

    def __post_init__(self) -> None:
        """Reject inferred types, empty phases, and impossible bounds."""
        for field in (
            "total_lookback_days",
            "validation_days",
            "test_days",
            "step_days",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise RollingWindowSpecError(
                    field=field,
                    observed=value,
                    requirement="a positive plain int",
                )
        if self.total_lookback_days <= self.validation_days:
            raise RollingWindowSpecError(
                field="total_lookback_days",
                observed=self.total_lookback_days,
                requirement="greater than validation_days",
            )
        for field in ("first_test_start", "latest_test_end"):
            value = getattr(self, field)
            if type(value) is not date:
                raise RollingWindowSpecError(
                    field=field,
                    observed=value,
                    requirement="a plain datetime.date",
                )
        first_test_end = self.first_test_start + timedelta(days=self.test_days)
        if first_test_end > self.latest_test_end:
            raise RollingWindowSpecError(
                field="latest_test_end",
                observed=self.latest_test_end,
                requirement="on or after the first test interval end",
            )


PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC: Final = RollingWindowSpec(
    total_lookback_days=3 * 365,
    validation_days=30,
    test_days=30,
    step_days=30,
    first_test_start=date(2015, 1, 1),
    latest_test_end=date(2024, 9, 1),
)


@dataclass(frozen=True, slots=True)
class RollingWindow:
    """One immutable set of adjacent half-open train, validation, and test bounds."""

    window_id: int
    train_start: date
    validation_start: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        """Reject coerced identities and overlapping or empty phase bounds."""
        if type(self.window_id) is not int or self.window_id < 0:
            raise RollingWindowSpecError(
                field="window_id",
                observed=self.window_id,
                requirement="a nonnegative plain int",
            )
        for field in ("train_start", "validation_start", "test_start", "test_end"):
            value = getattr(self, field)
            if type(value) is not date:
                raise RollingWindowSpecError(
                    field=field,
                    observed=value,
                    requirement="a plain datetime.date",
                )
        if not self.train_start < self.validation_start < self.test_start < self.test_end:
            raise RollingWindowSpecError(
                field="phase_bounds",
                observed=(
                    self.train_start,
                    self.validation_start,
                    self.test_start,
                    self.test_end,
                ),
                requirement="strictly increasing train/validation/test half-open bounds",
            )

    def bounds(self, phase: RollingPhase) -> tuple[date, date]:
        """Return the half-open bounds for one explicit phase."""
        if phase is RollingPhase.TRAIN:
            return self.train_start, self.validation_start
        if phase is RollingPhase.VALIDATION:
            return self.validation_start, self.test_start
        if phase is RollingPhase.TEST:
            return self.test_start, self.test_end
        raise RollingWindowPhaseError(observed=phase)


@dataclass(frozen=True, slots=True)
class RollingWindowCounts:
    """Validated row counts for every phase in one rolling window."""

    window_id: int
    train_rows: int
    validation_rows: int
    test_rows: int


def generate_rolling_windows(spec: RollingWindowSpec) -> tuple[RollingWindow, ...]:
    """Generate the complete deterministic schedule encoded by ``spec``."""
    windows: list[RollingWindow] = []
    test_start = spec.first_test_start
    while test_start + timedelta(days=spec.test_days) <= spec.latest_test_end:
        windows.append(
            RollingWindow(
                window_id=len(windows),
                train_start=test_start - timedelta(days=spec.total_lookback_days),
                validation_start=test_start - timedelta(days=spec.validation_days),
                test_start=test_start,
                test_end=test_start + timedelta(days=spec.test_days),
            ),
        )
        test_start += timedelta(days=spec.step_days)
    return tuple(windows)


def rolling_window_manifest(spec: RollingWindowSpec) -> pl.DataFrame:
    """Materialize the small, typed schedule suitable for a resolved run manifest."""
    windows = generate_rolling_windows(spec)
    return pl.DataFrame(
        {
            "window_id": [window.window_id for window in windows],
            "train_start": [window.train_start for window in windows],
            "validation_start": [window.validation_start for window in windows],
            "test_start": [window.test_start for window in windows],
            "test_end": [window.test_end for window in windows],
        },
        schema={
            "window_id": pl.UInt32,
            "train_start": pl.Date,
            "validation_start": pl.Date,
            "test_start": pl.Date,
            "test_end": pl.Date,
        },
    )


def _validated_quote_dates(frame: pl.LazyFrame) -> pl.LazyFrame:
    validated = require_columns(frame, ("quote_date",))
    observed = validated.collect_schema()["quote_date"]
    if observed != pl.Date:
        raise RollingWindowSchemaError(observed=str(observed))
    return validated


def select_rolling_phase(
    frame: pl.LazyFrame,
    window: RollingWindow,
    phase: RollingPhase,
) -> pl.LazyFrame:
    """Select one phase lazily using its exact half-open date interval."""
    validated = _validated_quote_dates(frame)
    start, end = window.bounds(phase)
    return validated.filter(
        (pl.col("quote_date") >= start) & (pl.col("quote_date") < end),
    )


def validate_rolling_window_coverage(
    frame: pl.LazyFrame,
    windows: tuple[RollingWindow, ...],
) -> tuple[RollingWindowCounts, ...]:
    """Validate every required partition from one compact date-count scan.

    Empty train, validation, or test phases are errors, so incomplete experiments fail before any
    model is trained.

    """
    if not windows:
        raise RollingWindowContentError(violations=(("empty_window_schedule", 1),))
    date_counts = (
        _validated_quote_dates(frame)
        .group_by("quote_date")
        .len(name="rows")
        .sort("quote_date")
        .collect()
    )
    null_dates = date_counts["quote_date"].null_count()
    if null_dates:
        raise RollingWindowContentError(violations=(("null_quote_dates", null_dates),))
    dates = date_counts["quote_date"].to_list()
    counts = date_counts["rows"].to_list()
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)

    def rows_between(start: date, end: date) -> int:
        left = bisect_left(dates, start)
        right = bisect_left(dates, end)
        return prefix[right] - prefix[left]

    resolved: list[RollingWindowCounts] = []
    empty_train = 0
    empty_validation = 0
    empty_test = 0
    for window in windows:
        train_rows = rows_between(*window.bounds(RollingPhase.TRAIN))
        validation_rows = rows_between(*window.bounds(RollingPhase.VALIDATION))
        test_rows = rows_between(*window.bounds(RollingPhase.TEST))
        empty_train += train_rows == 0
        empty_validation += validation_rows == 0
        empty_test += test_rows == 0
        resolved.append(
            RollingWindowCounts(
                window_id=window.window_id,
                train_rows=train_rows,
                validation_rows=validation_rows,
                test_rows=test_rows,
            ),
        )
    violations = tuple(
        (name, value)
        for name, value in (
            ("empty_train_partitions", empty_train),
            ("empty_validation_partitions", empty_validation),
            ("empty_test_partitions", empty_test),
        )
        if value
    )
    if violations:
        raise RollingWindowContentError(violations=violations)
    return tuple(resolved)


__all__ = [
    "PROJECT_AS_SUBMITTED_ROLLING_WINDOW_SPEC",
    "RollingPhase",
    "RollingWindow",
    "RollingWindowContentError",
    "RollingWindowCounts",
    "RollingWindowError",
    "RollingWindowPhaseError",
    "RollingWindowSchemaError",
    "RollingWindowSpec",
    "RollingWindowSpecError",
    "generate_rolling_windows",
    "rolling_window_manifest",
    "select_rolling_phase",
    "validate_rolling_window_coverage",
]
