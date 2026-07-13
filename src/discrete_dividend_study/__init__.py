"""Reproducible discrete-dividend option-pricing study."""

from __future__ import annotations

from typing import Final

from finoptions import BuildInfo, build_info

from .schema import DuplicateColumnsError, MissingColumnsError, require_columns

__version__: Final = "0.1.0"


def engine_build_info() -> BuildInfo:
    """Return the loaded native engine identity for run manifests."""
    return build_info()


__all__ = [
    "DuplicateColumnsError",
    "MissingColumnsError",
    "__version__",
    "engine_build_info",
    "require_columns",
]
