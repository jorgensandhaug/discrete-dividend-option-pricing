"""Reproducible discrete-dividend option-pricing study."""

from __future__ import annotations

from typing import Final

from finoptions import BuildInfo, build_info

from .orats import (
    AdaptedOratsCurve,
    OratsCurveContentError,
    OratsCurveDomainError,
    OratsCurveError,
    OratsCurveProvenance,
    OratsCurveSchemaError,
    adapt_orats_curve,
    adapt_raw_orats_curve,
    canonicalize_orats_curve_frame,
)
from .orats_options import (
    ORATS_NORMALIZED_OPTION_DTYPES,
    ORATS_RAW_OPTION_COLUMNS,
    ORATS_RAW_OPTION_DTYPES,
    ORATS_RAW_SCAN_DTYPES,
    OratsOptionError,
    OratsOptionSchemaError,
    OratsOptionSourceError,
    canonicalize_raw_orats_options,
    scan_raw_orats_csv,
)
from .quote_filters import (
    PROJECT_AS_SUBMITTED_FILTER,
    QUOTE_FEATURE_DTYPES,
    QuoteFilter,
    QuoteFilterError,
    QuoteFilterSpecError,
    QuoteSchemaError,
    add_quote_features,
    apply_quote_filter,
)
from .schema import (
    ConflictingColumnsError,
    DuplicateColumnsError,
    MissingColumnsError,
    forbid_columns,
    require_columns,
)
from .underlying_history import (
    PROJECT_AS_SUBMITTED_UNDERLYING_JOIN_SPEC,
    PROJECT_HISTORICAL_VOLATILITY_SPEC,
    UNDERLYING_HISTORY_DTYPES,
    HistoricalVolatilitySpec,
    HistoricalVolatilitySpecError,
    UnderlyingHistoryContentError,
    UnderlyingHistoryError,
    UnderlyingHistoryJoinError,
    UnderlyingHistoryJoinSchemaError,
    UnderlyingHistoryJoinSpec,
    UnderlyingHistoryJoinSpecError,
    UnderlyingHistorySchemaError,
    join_underlying_history,
    prepare_underlying_history,
)

__version__: Final = "0.1.0"


def engine_build_info() -> BuildInfo:
    """Return the loaded native engine identity for run manifests."""
    return build_info()


__all__ = [
    "ORATS_NORMALIZED_OPTION_DTYPES",
    "ORATS_RAW_OPTION_COLUMNS",
    "ORATS_RAW_OPTION_DTYPES",
    "ORATS_RAW_SCAN_DTYPES",
    "PROJECT_AS_SUBMITTED_FILTER",
    "PROJECT_AS_SUBMITTED_UNDERLYING_JOIN_SPEC",
    "PROJECT_HISTORICAL_VOLATILITY_SPEC",
    "QUOTE_FEATURE_DTYPES",
    "UNDERLYING_HISTORY_DTYPES",
    "AdaptedOratsCurve",
    "ConflictingColumnsError",
    "DuplicateColumnsError",
    "HistoricalVolatilitySpec",
    "HistoricalVolatilitySpecError",
    "MissingColumnsError",
    "OratsCurveContentError",
    "OratsCurveDomainError",
    "OratsCurveError",
    "OratsCurveProvenance",
    "OratsCurveSchemaError",
    "OratsOptionError",
    "OratsOptionSchemaError",
    "OratsOptionSourceError",
    "QuoteFilter",
    "QuoteFilterError",
    "QuoteFilterSpecError",
    "QuoteSchemaError",
    "UnderlyingHistoryContentError",
    "UnderlyingHistoryError",
    "UnderlyingHistoryJoinError",
    "UnderlyingHistoryJoinSchemaError",
    "UnderlyingHistoryJoinSpec",
    "UnderlyingHistoryJoinSpecError",
    "UnderlyingHistorySchemaError",
    "__version__",
    "adapt_orats_curve",
    "adapt_raw_orats_curve",
    "add_quote_features",
    "apply_quote_filter",
    "canonicalize_orats_curve_frame",
    "canonicalize_raw_orats_options",
    "engine_build_info",
    "forbid_columns",
    "join_underlying_history",
    "prepare_underlying_history",
    "require_columns",
    "scan_raw_orats_csv",
]
