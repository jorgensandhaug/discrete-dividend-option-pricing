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


@dataclass(frozen=True, slots=True)
class ConflictingColumnsError(ValueError):
    """A lazy input already contains columns owned by a downstream stage."""

    conflicting: tuple[str, ...]

    def __str__(self) -> str:
        """Describe the conflicting schema names without inspecting rows."""
        return f"downstream-owned columns already exist: {self.conflicting!r}"


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


def forbid_columns(frame: pl.LazyFrame, forbidden: tuple[str, ...]) -> pl.LazyFrame:
    """Reject names owned by a downstream stage and return the same lazy plan.

    Raises:
        DuplicateColumnsError: If the forbidden-name declaration contains duplicates.
        ConflictingColumnsError: If one or more forbidden names already exist.

    """
    if len(forbidden) != len(set(forbidden)):
        raise DuplicateColumnsError(required=forbidden)
    available = set(frame.collect_schema().names())
    conflicting = tuple(name for name in forbidden if name in available)
    if conflicting:
        raise ConflictingColumnsError(conflicting=conflicting)
    return frame
