"""Tests for one-scan materialization of complete neural rolling windows."""

from __future__ import annotations

import runpy
from datetime import date
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest
from finoptions import CpuExecutor

from discrete_dividend_study.model_inputs import ModelInputContentError, ModelInputTypeError
from discrete_dividend_study.neural_spec import PROJECT_NEURAL_STUDY_SPEC, NeuralVariant
from discrete_dividend_study.neural_window import prepare_project_neural_window
from discrete_dividend_study.rolling_windows import RollingWindow

if TYPE_CHECKING:
    from collections.abc import Callable as CallableType

WINDOW = RollingWindow(
    window_id=3,
    train_start=date(2024, 1, 1),
    validation_start=date(2024, 2, 1),
    test_start=date(2024, 3, 1),
    test_end=date(2024, 4, 1),
)
PHASE_COUNT = 3


def _model_frame() -> pl.DataFrame:
    helpers = runpy.run_path("tests/test_model_inputs.py")
    factory = cast("CallableType[[], pl.DataFrame]", helpers["model_frame"])
    return factory().with_columns(
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
    model = PROJECT_NEURAL_STUDY_SPEC.model(variant)
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
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.FORECAST_YIELD)
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
