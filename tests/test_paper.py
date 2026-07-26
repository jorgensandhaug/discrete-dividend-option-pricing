"""Public paper-surface tests."""

from discrete_dividend_study.paper import (
    DIRECT_LSTM,
    EVENT_ATTENTION,
    FORECAST_YIELD,
    HISTORICAL_YIELD,
    PROJECT_MODELS,
    PROJECT_STUDY,
    TWELVE_BINS,
)

_EXPECTED_MODEL_COUNT = 5


def test_public_paper_surface_owns_the_complete_frozen_study() -> None:
    """The readable paper module owns one complete scientific identity."""
    assert PROJECT_STUDY.models == PROJECT_MODELS
    assert PROJECT_MODELS == (
        DIRECT_LSTM,
        EVENT_ATTENTION,
        TWELVE_BINS,
        FORECAST_YIELD,
        HISTORICAL_YIELD,
    )
    assert len(PROJECT_MODELS) == _EXPECTED_MODEL_COUNT
