"""Discrete-dividend option-pricing experiments."""

from importlib.metadata import version
from typing import Final

from discrete_dividend_study.paper import (
    DIRECT_ATTENTION,
    DIRECT_LSTM,
    FORECAST_YIELD,
    HISTORICAL_YIELD,
    PROJECT_MODELS,
    PROJECT_STUDY,
    TWELVE_BINS,
)

__version__: Final = version("discrete-dividend-option-pricing")

__all__ = [
    "DIRECT_ATTENTION",
    "DIRECT_LSTM",
    "FORECAST_YIELD",
    "HISTORICAL_YIELD",
    "PROJECT_MODELS",
    "PROJECT_STUDY",
    "TWELVE_BINS",
    "__version__",
]
