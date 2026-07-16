"""Tests for one-collection, full-horizon neural rolling preparation."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from finoptions import CpuExecutor, PaddedCashDividends

from discrete_dividend_study.experiments.neural.rolling import (
    NeuralRollingPreparationError,
    prepare_project_neural_rolling_source,
)
from discrete_dividend_study.experiments.windows import RollingWindow, RollingWindowContentError
from discrete_dividend_study.models.neural.specification import (
    PROJECT_STUDY,
    NeuralVariant,
)
from tests.models.neural.test_inputs import model_frame

WINDOWS = (
    RollingWindow(
        window_id=0,
        train_start=date(2024, 1, 1),
        validation_start=date(2024, 2, 1),
        test_start=date(2024, 3, 1),
        test_end=date(2024, 4, 1),
    ),
    RollingWindow(
        window_id=1,
        train_start=date(2024, 2, 1),
        validation_start=date(2024, 3, 1),
        test_start=date(2024, 4, 1),
        test_end=date(2024, 5, 1),
    ),
)
FULL_HORIZON_ROWS = 4


def _rolling_frame() -> pl.DataFrame:
    base = model_frame().slice(0, 1)
    rows = tuple(
        base.with_columns(
            pl.lit(12.0 + offset, dtype=pl.Float32).alias("call_mid_price"),
            pl.lit(quote_date, dtype=pl.Date).alias("quote_date"),
        )
        for offset, quote_date in enumerate(
            (
                date(2024, 3, 5),
                date(2024, 1, 5),
                date(2024, 4, 5),
                date(2024, 2, 5),
            ),
        )
    )
    return pl.concat(rows, rechunk=True)


def test_full_schedule_collects_once_and_exposes_zero_copy_phase_views() -> None:
    """Every model/window is built from one sorted full-horizon allocation."""
    calls = 0

    def observe_batch(batch: pl.DataFrame) -> pl.DataFrame:
        nonlocal calls
        calls += 1
        return batch

    source_frame = _rolling_frame()
    source = source_frame.lazy().map_batches(
        observe_batch,
        schema=source_frame.schema,
        predicate_pushdown=False,
        projection_pushdown=False,
        slice_pushdown=False,
        streamable=False,
    )
    prepared = prepare_project_neural_rolling_source(
        source,
        WINDOWS,
        PROJECT_STUDY,
        CpuExecutor(threads=2),
    )
    assert calls == 1
    assert prepared.rows == FULL_HORIZON_ROWS
    assert prepared.models == PROJECT_STUDY.models
    assert tuple(
        (item.training.start, item.validation.start, item.test.start, item.test.end)
        for item in prepared.slices
    ) == ((0, 1, 2, 3), (1, 2, 3, 4))

    base_batches = prepared.full_batches[:3]
    assert base_batches[0].scalar_features is base_batches[1].scalar_features
    assert base_batches[1].scalar_features is base_batches[2].scalar_features
    assert type(base_batches[0].dividends) is PaddedCashDividends
    assert base_batches[0].dividends is base_batches[1].dividends
    assert all(batch.target is base_batches[0].target for batch in prepared.full_batches)

    for model, full_batch in zip(prepared.models, prepared.full_batches, strict=True):
        for window in WINDOWS:
            phase = prepared.window(window.window_id, model)
            assert phase.training.rows == phase.validation.rows == phase.test.rows == 1
            assert np.shares_memory(phase.training.scalar_features, full_batch.scalar_features)
            assert np.shares_memory(phase.validation.target, full_batch.target)
            assert np.shares_memory(phase.test.target, full_batch.target)


def test_embedding_study_has_an_explicit_input_boundary() -> None:
    """The paper embedding identity is accepted without admitting arbitrary model specs."""
    prepared = prepare_project_neural_rolling_source(
        _rolling_frame().lazy(),
        WINDOWS,
        PROJECT_STUDY,
        CpuExecutor(threads=1),
    )
    embedding = PROJECT_STUDY.model(NeuralVariant.EMBEDDING)
    assert prepared.models == PROJECT_STUDY.models
    assert prepared.window(0, embedding).training.model == embedding


def test_null_dates_and_empty_partitions_fail_before_any_model_run() -> None:
    """The horizon filter cannot hide null dates or permit a skipped schedule phase."""
    executor = CpuExecutor(threads=1)
    null_row = (
        _rolling_frame()
        .slice(0, 1)
        .with_columns(
            pl.lit(None, dtype=pl.Date).alias("quote_date"),
        )
    )
    with pytest.raises(RollingWindowContentError, match="null_quote_dates"):
        prepare_project_neural_rolling_source(
            pl.concat((_rolling_frame(), null_row), rechunk=True).lazy(),
            WINDOWS,
            PROJECT_STUDY,
            executor,
        )

    without_april = _rolling_frame().filter(pl.col("quote_date") < date(2024, 4, 1))
    with pytest.raises(RollingWindowContentError, match="empty_test_partitions"):
        prepare_project_neural_rolling_source(
            without_april.lazy(),
            WINDOWS,
            PROJECT_STUDY,
            executor,
        )


def test_schedule_identity_must_be_complete_and_ordered() -> None:
    """Preparation rejects ad hoc identifiers before collecting licensed rows."""
    invalid = (WINDOWS[1],)
    with pytest.raises(NeuralRollingPreparationError, match="ordered from zero"):
        prepare_project_neural_rolling_source(
            _rolling_frame().lazy(),
            invalid,
            PROJECT_STUDY,
            CpuExecutor(threads=1),
        )

    prepared = prepare_project_neural_rolling_source(
        _rolling_frame().lazy(),
        WINDOWS,
        PROJECT_STUDY,
        CpuExecutor(threads=1),
    )
    with pytest.raises(NeuralRollingPreparationError, match="absent"):
        prepared.window(99, PROJECT_STUDY.model(NeuralVariant.BINS))
