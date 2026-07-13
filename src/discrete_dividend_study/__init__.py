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
from .schema import DuplicateColumnsError, MissingColumnsError, require_columns

__version__: Final = "0.1.0"


def engine_build_info() -> BuildInfo:
    """Return the loaded native engine identity for run manifests."""
    return build_info()


__all__ = [
    "AdaptedOratsCurve",
    "DuplicateColumnsError",
    "MissingColumnsError",
    "OratsCurveContentError",
    "OratsCurveDomainError",
    "OratsCurveError",
    "OratsCurveProvenance",
    "OratsCurveSchemaError",
    "__version__",
    "adapt_orats_curve",
    "adapt_raw_orats_curve",
    "canonicalize_orats_curve_frame",
    "engine_build_info",
    "require_columns",
]
