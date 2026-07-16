"""Small public surface for the project paper's frozen neural experiment."""

from __future__ import annotations

from typing import Final

from discrete_dividend_study.models.neural.specification import PROJECT_STUDY as _PROJECT_STUDY
from discrete_dividend_study.models.neural.specification import (
    NeuralModelSpec,
    NeuralStudySpec,
    NeuralVariant,
)

PROJECT_STUDY: Final[NeuralStudySpec] = _PROJECT_STUDY
"""Complete architecture, optimizer, loss, and five-model study identity."""

DIRECT_LSTM: Final[NeuralModelSpec] = PROJECT_STUDY.model(NeuralVariant.LSTM)
DIRECT_ATTENTION: Final[NeuralModelSpec] = PROJECT_STUDY.model(NeuralVariant.EMBEDDING)
TWELVE_BINS: Final[NeuralModelSpec] = PROJECT_STUDY.model(NeuralVariant.BINS)
FORECAST_YIELD: Final[NeuralModelSpec] = PROJECT_STUDY.model(NeuralVariant.FORECAST_YIELD)
HISTORICAL_YIELD: Final[NeuralModelSpec] = PROJECT_STUDY.model(NeuralVariant.HISTORICAL_YIELD)

PROJECT_MODELS: Final[tuple[NeuralModelSpec, ...]] = (
    DIRECT_LSTM,
    DIRECT_ATTENTION,
    TWELVE_BINS,
    FORECAST_YIELD,
    HISTORICAL_YIELD,
)
"""The five compared models in paper-table order."""

__all__ = [
    "DIRECT_ATTENTION",
    "DIRECT_LSTM",
    "FORECAST_YIELD",
    "HISTORICAL_YIELD",
    "PROJECT_MODELS",
    "PROJECT_STUDY",
    "TWELVE_BINS",
]
