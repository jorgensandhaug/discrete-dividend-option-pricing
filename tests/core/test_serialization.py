"""Tests for typed dataclass artifact decoding primitives."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from discrete_dividend_study.core.serialization import dataclass_field_names


@dataclass(frozen=True, slots=True)
class _Example:
    first: int
    second: str


def test_dataclass_field_names_preserve_exact_declaration() -> None:
    """Dataclass field names are returned as one exact unordered identity."""
    assert dataclass_field_names(_Example) == frozenset({"first", "second"})


def test_dataclass_field_names_reject_non_dataclass_type() -> None:
    """A decoder cannot silently treat an arbitrary class as a dataclass."""
    with pytest.raises(TypeError, match="dataclass type"):
        dataclass_field_names(object)
