"""Strict lazy-schema validation shared by provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True, slots=True)
class MissingColumnsError(ValueError):
    """Required source columns are absent from a lazy input schema."""

    missing: tuple[str, ...]
    available: tuple[str, ...]

    def __str__(self) -> str:
        """Describe the violated schema without exposing row-level data."""
        return (
            f"missing required columns {self.missing!r}; available columns are {self.available!r}"
        )


@dataclass(frozen=True, slots=True)
class DuplicateColumnsError(ValueError):
    """A schema declaration repeats one or more required names."""

    required: tuple[str, ...]

    def __str__(self) -> str:
        """Describe the invalid declaration without inspecting data rows."""
        return f"required column names must be unique; observed {self.required!r}"


def require_columns(frame: pl.LazyFrame, required: tuple[str, ...]) -> pl.LazyFrame:
    """Validate required names without materializing rows and return the same lazy plan.

    Raises:
        MissingColumnsError: If one or more required names are absent.
        DuplicateColumnsError: If the required-name declaration contains duplicates.

    """
    if len(required) != len(set(required)):
        raise DuplicateColumnsError(required=required)
    available = tuple(frame.collect_schema().names())
    available_set = set(available)
    missing = tuple(name for name in required if name not in available_set)
    if missing:
        raise MissingColumnsError(missing=missing, available=available)
    return frame
