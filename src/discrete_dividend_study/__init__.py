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
from .schema import DuplicateColumnsError, MissingColumnsError, require_columns

__version__: Final = "0.1.0"


def engine_build_info() -> BuildInfo:
    """Return the loaded native engine identity for run manifests."""
    return build_info()


__all__ = [
    "ORATS_NORMALIZED_OPTION_DTYPES",
    "ORATS_RAW_OPTION_COLUMNS",
    "ORATS_RAW_OPTION_DTYPES",
    "ORATS_RAW_SCAN_DTYPES",
    "AdaptedOratsCurve",
    "DuplicateColumnsError",
    "MissingColumnsError",
    "OratsCurveContentError",
    "OratsCurveDomainError",
    "OratsCurveError",
    "OratsCurveProvenance",
    "OratsCurveSchemaError",
    "OratsOptionError",
    "OratsOptionSchemaError",
    "OratsOptionSourceError",
    "__version__",
    "adapt_orats_curve",
    "adapt_raw_orats_curve",
    "canonicalize_orats_curve_frame",
    "canonicalize_raw_orats_options",
    "engine_build_info",
    "require_columns",
    "scan_raw_orats_csv",
]
