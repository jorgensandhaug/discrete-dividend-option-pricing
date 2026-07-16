"""Tests for one-scan materialization of complete neural rolling windows."""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from finoptions import CpuExecutor

from discrete_dividend_study.experiments.neural.window import prepare_project_neural_window
from discrete_dividend_study.experiments.windows import RollingWindow
from discrete_dividend_study.models.neural.inputs import ModelInputContentError, ModelInputTypeError
from discrete_dividend_study.models.neural.specification import PROJECT_STUDY, NeuralVariant
from tests.models.neural.test_inputs import model_frame

WINDOW = RollingWindow(
    window_id=3,
    train_start=date(2024, 1, 1),
    validation_start=date(2024, 2, 1),
    test_start=date(2024, 3, 1),
    test_end=date(2024, 4, 1),
)
PHASE_COUNT = 3


def _model_frame() -> pl.DataFrame:
    return model_frame().with_columns(
        pl.Series(
            "quote_date",
            [date(2024, 1, 5), date(2024, 2, 5), date(2024, 3, 5)],
            dtype=pl.Date,
        ),
    )


@pytest.mark.parametrize("variant", list(NeuralVariant))
def test_all_three_phases_share_one_model_and_distinct_content_hashes(
    variant: NeuralVariant,
) -> None:
    """The one collection request returns one exact nonempty batch per half-open phase."""
    model = PROJECT_STUDY.model(variant)
    prepared = prepare_project_neural_window(
        _model_frame().lazy(),
        WINDOW,
        model,
        CpuExecutor(threads=2),
    )
    assert prepared.window == WINDOW
    assert prepared.model == model
    assert prepared.training.rows == 1
    assert prepared.validation.rows == 1
    assert prepared.test.rows == 1
    assert (
        len(
            {
                prepared.training_data_sha256,
                prepared.validation_data_sha256,
                prepared.test_data_sha256,
            },
        )
        == PHASE_COUNT
    )


def test_empty_or_eager_phase_source_fails_without_skipping() -> None:
    """The preparation boundary never coerces an eager source or omits an empty phase."""
    model = PROJECT_STUDY.model(NeuralVariant.FORECAST_YIELD)
    executor = CpuExecutor(threads=1)
    with pytest.raises(ModelInputTypeError, match="LazyFrame"):
        prepare_project_neural_window(
            cast("pl.LazyFrame", _model_frame()),
            WINDOW,
            model,
            executor,
        )
    with pytest.raises(ModelInputContentError, match="empty_frame"):
        prepare_project_neural_window(
            _model_frame().filter(pl.col("quote_date") < date(2024, 3, 1)).lazy(),
            WINDOW,
            model,
            executor,
        )
