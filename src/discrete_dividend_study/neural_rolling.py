"""One-collection, zero-copy rolling preparation for complete neural studies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final, cast

import polars as pl

from .model_inputs import (
    EventBufferPolicy,
    ModelInputTypeError,
    PreparedNeuralBatch,
    prepare_project_neural_batches,
    project_neural_input_columns,
    slice_prepared_neural_batch,
)
from .neural_spec import NeuralModelSpec, NeuralStudySpec
from .neural_window import (
    PROJECT_NEURAL_WINDOW_PREPARATION_SPEC,
    NeuralWindowPreparationSpec,
    PreparedNeuralWindow,
)
from .rolling_windows import (
    RollingWindow,
    RollingWindowContentError,
    RollingWindowSchemaError,
)
from .schema import require_columns

if TYPE_CHECKING:
    from datetime import date

    from finoptions import CpuExecutor

_QUOTE_DATE: Final = "quote_date"

PROJECT_NEURAL_ROLLING_PREPARATION_SPEC: Final = NeuralWindowPreparationSpec(
    event_buffer_policy=EventBufferPolicy.ALLOW_RECHUNK,
    dividend_columns=PROJECT_NEURAL_WINDOW_PREPARATION_SPEC.dividend_columns,
)


@dataclass(slots=True)
class NeuralRollingPreparationError(RuntimeError):
    """A complete rolling source violates its typed preparation contract."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected rolling preparation contract."""
        return f"neural rolling preparation failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class PreparedRowSlice:
    """One nonempty half-open contiguous row interval in the full prepared horizon."""

    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject coercion, empty phases, and reversed bounds."""
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise NeuralRollingPreparationError(
                reason="row slices must be plain ints satisfying 0 <= start < end",
            )

    @property
    def rows(self) -> int:
        """Return the exact number of rows in the interval."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class PreparedRollingWindowSlices:
    """Exact adjacent row intervals for one date-defined rolling window."""

    window_id: int
    training: PreparedRowSlice
    validation: PreparedRowSlice
    test: PreparedRowSlice

    def __post_init__(self) -> None:
        """Require a nonnegative identity and exact phase adjacency."""
        if type(self.window_id) is not int or self.window_id < 0:
            raise NeuralRollingPreparationError(reason="window_id must be a nonnegative plain int")
        phases = (self.training, self.validation, self.test)
        if any(type(phase) is not PreparedRowSlice for phase in phases):
            raise NeuralRollingPreparationError(
                reason="rolling phases must be PreparedRowSlice values",
            )
        if self.training.end != self.validation.start or self.validation.end != self.test.start:
            raise NeuralRollingPreparationError(reason="rolling phase row slices must be adjacent")


@dataclass(frozen=True, slots=True)
class PreparedNeuralRollingSource:
    """Full-horizon model tensors plus exact zero-copy phase views for every window."""

    windows: tuple[RollingWindow, ...]
    full_batches: tuple[PreparedNeuralBatch, ...]
    slices: tuple[PreparedRollingWindowSlices, ...]

    def __post_init__(self) -> None:
        """Reject incomplete schedules, cross-model rows, and out-of-range slices."""
        _validate_prepared_rolling_source(self)

    @property
    def rows(self) -> int:
        """Return the number of observations materialized across the full date horizon."""
        return self.full_batches[0].rows

    @property
    def models(self) -> tuple[NeuralModelSpec, ...]:
        """Return the exact model order prepared for every rolling window."""
        return tuple(batch.model for batch in self.full_batches)

    def window(self, window_id: int, model: NeuralModelSpec) -> PreparedNeuralWindow:
        """Build one immutable window from contiguous views without copying tensor payloads."""
        if type(window_id) is not int or window_id < 0:
            raise NeuralRollingPreparationError(reason="window_id must be a nonnegative plain int")
        if type(model) is not NeuralModelSpec:
            raise ModelInputTypeError(
                field="model",
                observed=type(model).__name__,
                expected="NeuralModelSpec",
            )
        matching_windows = tuple(window for window in self.windows if window.window_id == window_id)
        matching_slices = tuple(item for item in self.slices if item.window_id == window_id)
        matching_batches = tuple(batch for batch in self.full_batches if batch.model == model)
        if len(matching_windows) != 1 or len(matching_slices) != 1:
            raise NeuralRollingPreparationError(
                reason="window_id is absent from the prepared schedule"
            )
        if len(matching_batches) != 1:
            raise NeuralRollingPreparationError(reason="model is absent from the prepared source")
        row_slices = matching_slices[0]
        batch = matching_batches[0]
        return PreparedNeuralWindow(
            window=matching_windows[0],
            training=_slice_batch(batch, row_slices.training),
            validation=_slice_batch(batch, row_slices.validation),
            test=_slice_batch(batch, row_slices.test),
        )


def prepare_project_neural_rolling_source(
    frame: pl.LazyFrame,
    windows: tuple[RollingWindow, ...],
    study: NeuralStudySpec,
    executor: CpuExecutor,
    preparation: NeuralWindowPreparationSpec = PROJECT_NEURAL_ROLLING_PREPARATION_SPEC,
) -> PreparedNeuralRollingSource:
    """Collect the full required horizon once and prepare every model representation once.

    The plan projects the union of required model columns, retains null dates so they cannot be
    filtered away silently, restricts rows to the complete schedule horizon, and performs a stable
    date sort. Polars executes that plan once. Every later phase is a contiguous read-only NumPy
    view, so overlapping three-year windows do not repeat scans, casts, ragged validation, padding,
    binning, or full-phase allocations.
    """
    validated = _validate_rolling_request(frame, windows, study, preparation)
    horizon_start = min(window.train_start for window in windows)
    horizon_end = max(window.test_end for window in windows)
    columns = _rolling_input_columns(study)
    within_horizon = (pl.col(_QUOTE_DATE) >= horizon_start) & (pl.col(_QUOTE_DATE) < horizon_end)
    collected = (
        validated.filter(pl.col(_QUOTE_DATE).is_null() | within_horizon)
        .select(*columns)
        .sort(_QUOTE_DATE, maintain_order=True)
        .collect()
    )
    null_dates = collected.get_column(_QUOTE_DATE).null_count()
    if null_dates:
        raise RollingWindowContentError(violations=(("null_quote_dates", null_dates),))
    if collected.height == 0:
        raise RollingWindowContentError(violations=(("empty_schedule_horizon", 1),))
    row_slices = _resolve_row_slices(collected.get_column(_QUOTE_DATE), windows)
    full_batches = prepare_project_neural_batches(
        collected.drop(_QUOTE_DATE),
        study.models,
        executor,
        event_buffer_policy=preparation.event_buffer_policy,
        dividend_columns=preparation.dividend_columns,
    )
    return PreparedNeuralRollingSource(
        windows=windows,
        full_batches=full_batches,
        slices=row_slices,
    )


def _validate_rolling_request(
    frame: object,
    windows: object,
    study: object,
    preparation: object,
) -> pl.LazyFrame:
    if type(frame) is not pl.LazyFrame:
        raise ModelInputTypeError(
            field="frame",
            observed=type(frame).__name__,
            expected="polars.LazyFrame",
        )
    if (
        type(windows) is not tuple
        or not windows
        or any(type(window) is not RollingWindow for window in windows)
    ):
        raise NeuralRollingPreparationError(
            reason="windows must be a nonempty tuple[RollingWindow, ...]",
        )
    resolved_windows = cast("tuple[RollingWindow, ...]", windows)
    if tuple(window.window_id for window in resolved_windows) != tuple(
        range(len(resolved_windows))
    ):
        raise NeuralRollingPreparationError(
            reason="window ids must be contiguous, unique, and ordered from zero",
        )
    if any(
        previous.test_start >= current.test_start
        for previous, current in pairwise(resolved_windows)
    ):
        raise NeuralRollingPreparationError(reason="windows must have increasing test starts")
    if type(study) is not NeuralStudySpec:
        raise ModelInputTypeError(
            field="study",
            observed=type(study).__name__,
            expected="NeuralStudySpec",
        )
    if type(preparation) is not NeuralWindowPreparationSpec:
        raise ModelInputTypeError(
            field="preparation",
            observed=type(preparation).__name__,
            expected="NeuralWindowPreparationSpec",
        )
    validated = require_columns(frame, (_QUOTE_DATE,))
    observed = validated.collect_schema()[_QUOTE_DATE]
    if observed != pl.Date:
        raise RollingWindowSchemaError(observed=str(observed))
    return validated


def _rolling_input_columns(study: NeuralStudySpec) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                _QUOTE_DATE,
                *(
                    column
                    for model in study.models
                    for column in project_neural_input_columns(model)
                ),
            ),
        ),
    )


def _resolve_row_slices(
    quote_dates: pl.Series,
    windows: tuple[RollingWindow, ...],
) -> tuple[PreparedRollingWindowSlices, ...]:
    raw = tuple(_raw_window_slices(quote_dates, window) for window in windows)
    violations = tuple(
        (name, count)
        for name, count in (
            ("empty_train_partitions", sum(item[0][0] == item[0][1] for item in raw)),
            ("empty_validation_partitions", sum(item[1][0] == item[1][1] for item in raw)),
            ("empty_test_partitions", sum(item[2][0] == item[2][1] for item in raw)),
        )
        if count
    )
    if violations:
        raise RollingWindowContentError(violations=violations)
    return tuple(
        PreparedRollingWindowSlices(
            window_id=window.window_id,
            training=PreparedRowSlice(*bounds[0]),
            validation=PreparedRowSlice(*bounds[1]),
            test=PreparedRowSlice(*bounds[2]),
        )
        for window, bounds in zip(windows, raw, strict=True)
    )


def _raw_window_slices(
    quote_dates: pl.Series,
    window: RollingWindow,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    train_start = _search_date(quote_dates, window.train_start)
    validation_start = _search_date(quote_dates, window.validation_start)
    test_start = _search_date(quote_dates, window.test_start)
    test_end = _search_date(quote_dates, window.test_end)
    return (
        (train_start, validation_start),
        (validation_start, test_start),
        (test_start, test_end),
    )


def _search_date(quote_dates: pl.Series, value: date) -> int:
    observed = quote_dates.search_sorted(value, side="left")
    if type(observed) is not int:
        raise NeuralRollingPreparationError(reason="Polars returned a non-integer date boundary")
    return observed


def _slice_batch(batch: PreparedNeuralBatch, row_slice: PreparedRowSlice) -> PreparedNeuralBatch:
    return slice_prepared_neural_batch(batch, row_slice.start, row_slice.end)


def _validate_prepared_rolling_source(source: PreparedNeuralRollingSource) -> None:
    if (
        type(source.windows) is not tuple
        or not source.windows
        or any(type(window) is not RollingWindow for window in source.windows)
    ):
        raise NeuralRollingPreparationError(reason="windows must be a complete rolling schedule")
    if (
        type(source.full_batches) is not tuple
        or not source.full_batches
        or any(type(batch) is not PreparedNeuralBatch for batch in source.full_batches)
    ):
        raise NeuralRollingPreparationError(reason="full_batches must be prepared neural batches")
    if len({batch.model.variant for batch in source.full_batches}) != len(source.full_batches):
        raise NeuralRollingPreparationError(reason="full_batches contain duplicate models")
    row_counts = {batch.rows for batch in source.full_batches}
    if len(row_counts) != 1:
        raise NeuralRollingPreparationError(reason="full model batches must share one row count")
    if (
        type(source.slices) is not tuple
        or len(source.slices) != len(source.windows)
        or any(type(item) is not PreparedRollingWindowSlices for item in source.slices)
    ):
        raise NeuralRollingPreparationError(
            reason="one prepared slice identity is required per window"
        )
    if tuple(item.window_id for item in source.slices) != tuple(
        window.window_id for window in source.windows
    ):
        raise NeuralRollingPreparationError(reason="window and row-slice identities differ")
    rows = next(iter(row_counts))
    if any(item.test.end > rows for item in source.slices):
        raise NeuralRollingPreparationError(reason="rolling row slice exceeds the prepared horizon")


__all__ = [
    "PROJECT_NEURAL_ROLLING_PREPARATION_SPEC",
    "NeuralRollingPreparationError",
    "PreparedNeuralRollingSource",
    "PreparedRollingWindowSlices",
    "PreparedRowSlice",
    "prepare_project_neural_rolling_source",
]
