"""One-scan preparation of exact train, validation, and test neural window phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import polars as pl

from .model_inputs import (
    PROJECT_RAGGED_DIVIDEND_COLUMNS,
    EventBufferPolicy,
    ModelInputTypeError,
    PreparedNeuralBatch,
    RaggedDividendColumns,
    prepare_project_neural_batch,
    prepared_neural_batch_sha256,
    project_neural_input_columns,
)
from .rolling_windows import RollingPhase, RollingWindow, select_rolling_phase

if TYPE_CHECKING:
    from finoptions import CpuExecutor

    from .neural_spec import NeuralModelSpec


@dataclass(frozen=True, slots=True)
class NeuralWindowPreparationError(RuntimeError):
    """Polars failed to resolve the complete declared phase request."""

    reason: str

    def __str__(self) -> str:
        """Return the failed preparation contract."""
        return f"neural window preparation failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralWindowPreparationSpec:
    """Explicit event-buffer behavior and semantic event-column mapping."""

    event_buffer_policy: EventBufferPolicy
    dividend_columns: RaggedDividendColumns

    def __post_init__(self) -> None:
        """Reject strings and dictionary-shaped configuration at the stateful boundary."""
        if type(self.event_buffer_policy) is not EventBufferPolicy:
            raise ModelInputTypeError(
                field="event_buffer_policy",
                observed=type(self.event_buffer_policy).__name__,
                expected="EventBufferPolicy",
            )
        if type(self.dividend_columns) is not RaggedDividendColumns:
            raise ModelInputTypeError(
                field="dividend_columns",
                observed=type(self.dividend_columns).__name__,
                expected="RaggedDividendColumns",
            )


PROJECT_NEURAL_WINDOW_PREPARATION_SPEC: Final = NeuralWindowPreparationSpec(
    event_buffer_policy=EventBufferPolicy.REQUIRE_ZERO_COPY,
    dividend_columns=PROJECT_RAGGED_DIVIDEND_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class PreparedNeuralWindow:
    """Three disjoint materialized phases and their exact model-boundary hashes."""

    window: RollingWindow
    training: PreparedNeuralBatch
    validation: PreparedNeuralBatch
    test: PreparedNeuralBatch
    training_data_sha256: str = field(init=False)
    validation_data_sha256: str = field(init=False)
    test_data_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Reject cross-model, aliased, or content-identical phase state."""
        if type(self.window) is not RollingWindow:
            raise ModelInputTypeError(
                field="window",
                observed=type(self.window).__name__,
                expected="RollingWindow",
            )
        phases = (self.training, self.validation, self.test)
        if any(type(phase) is not PreparedNeuralBatch for phase in phases):
            raise ModelInputTypeError(
                field="phases",
                observed=repr(tuple(type(phase).__name__ for phase in phases)),
                expected="three PreparedNeuralBatch values",
            )
        if len({id(phase) for phase in phases}) != len(phases):
            raise ModelInputTypeError(
                field="phases",
                observed="aliased batch objects",
                expected="three distinct PreparedNeuralBatch values",
            )
        models = {phase.model for phase in phases}
        if len(models) != 1:
            raise ModelInputTypeError(
                field="phases",
                observed="different model identities",
                expected="one shared NeuralModelSpec",
            )
        hashes = tuple(prepared_neural_batch_sha256(phase) for phase in phases)
        if len(set(hashes)) != len(hashes):
            raise ModelInputTypeError(
                field="phases",
                observed="identical phase content hashes",
                expected="three distinct model-input contents",
            )
        object.__setattr__(self, "training_data_sha256", hashes[0])
        object.__setattr__(self, "validation_data_sha256", hashes[1])
        object.__setattr__(self, "test_data_sha256", hashes[2])

    @property
    def model(self) -> NeuralModelSpec:
        """Return the single model identity shared by every phase."""
        return self.training.model


def prepare_project_neural_window(
    frame: pl.LazyFrame,
    window: RollingWindow,
    model: NeuralModelSpec,
    executor: CpuExecutor,
    preparation: NeuralWindowPreparationSpec = PROJECT_NEURAL_WINDOW_PREPARATION_SPEC,
) -> PreparedNeuralWindow:
    """Collect three projected phases together, then build exact model tensors.

    ``polars.collect_all`` enables common-subplan elimination across the three half-open filters,
    while projection pushdown limits materialization to the selected model's columns. No phase is
    skipped or substituted when its content is invalid.
    """
    if type(frame) is not pl.LazyFrame:
        raise ModelInputTypeError(
            field="frame",
            observed=type(frame).__name__,
            expected="polars.LazyFrame",
        )
    if type(window) is not RollingWindow:
        raise ModelInputTypeError(
            field="window",
            observed=type(window).__name__,
            expected="RollingWindow",
        )
    if type(preparation) is not NeuralWindowPreparationSpec:
        raise ModelInputTypeError(
            field="preparation",
            observed=type(preparation).__name__,
            expected="NeuralWindowPreparationSpec",
        )
    columns = project_neural_input_columns(model)
    plans = tuple(
        select_rolling_phase(frame, window, phase).select(*columns) for phase in RollingPhase
    )
    collected = pl.collect_all(plans)
    if len(collected) != len(RollingPhase):
        raise NeuralWindowPreparationError(
            reason="Polars did not return every requested rolling phase",
        )
    training, validation, test = (
        prepare_project_neural_batch(
            phase,
            model,
            executor,
            event_buffer_policy=preparation.event_buffer_policy,
            dividend_columns=preparation.dividend_columns,
        )
        for phase in collected
    )
    return PreparedNeuralWindow(
        window=window,
        training=training,
        validation=validation,
        test=test,
    )


__all__ = [
    "PROJECT_NEURAL_WINDOW_PREPARATION_SPEC",
    "NeuralWindowPreparationError",
    "NeuralWindowPreparationSpec",
    "PreparedNeuralWindow",
    "prepare_project_neural_window",
]
