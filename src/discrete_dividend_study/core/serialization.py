"""Small typed primitives shared by strict artifact decoders."""

from __future__ import annotations

from dataclasses import fields, is_dataclass


def dataclass_field_names(model: type[object]) -> frozenset[str]:
    """Return the exact declared field names for a typed dataclass model."""
    if not is_dataclass(model):
        message = "model must be a dataclass type"
        raise TypeError(message)
    return frozenset(field.name for field in fields(model))


__all__ = ["dataclass_field_names"]
