"""Initial repository and sibling-engine contract tests."""

from __future__ import annotations

import polars as pl
import pytest

from discrete_dividend_study import (
    DuplicateColumnsError,
    MissingColumnsError,
    engine_build_info,
    require_columns,
)


def test_pinned_engine_package_is_loaded() -> None:
    """The paper environment loads the expected initial engine package line."""
    info = engine_build_info()
    assert info.package_version == "0.1.0"
    assert info.rustc_version.startswith("rustc 1.97.0")


def test_schema_validation_preserves_lazy_plan() -> None:
    """Schema checks inspect metadata without collecting study rows."""
    frame = pl.LazyFrame({"spot": [100.0], "strike": [105.0]})
    validated = require_columns(frame, ("spot", "strike"))
    assert validated is frame


def test_schema_validation_fails_with_all_missing_names() -> None:
    """Missing columns fail explicitly and retain deterministic context."""
    frame = pl.LazyFrame({"spot": [100.0]})
    with pytest.raises(MissingColumnsError) as caught:
        require_columns(frame, ("strike", "expiration"))
    assert caught.value.missing == ("strike", "expiration")
    assert caught.value.available == ("spot",)


def test_schema_declaration_rejects_duplicates() -> None:
    """An ambiguous validation declaration fails before touching rows."""
    frame = pl.LazyFrame({"spot": [100.0]})
    with pytest.raises(DuplicateColumnsError, match="must be unique"):
        require_columns(frame, ("spot", "spot"))
