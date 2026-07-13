"""Shared quote-feature and immutable project-filter contract tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import cast

import polars as pl
import pytest

from discrete_dividend_study import (
    ORATS_RAW_OPTION_DTYPES,
    PROJECT_AS_SUBMITTED_FILTER,
    QUOTE_FEATURE_DTYPES,
    QuoteFilterSpecError,
    QuoteSchemaError,
    add_quote_features,
    apply_quote_filter,
    canonicalize_raw_orats_options,
)

EXPECTED_DAYS_TO_EXPIRY = 49


def raw_rows() -> pl.LazyFrame:
    """Return three exact-schema rows spanning pass, strict-price fail, and null-support fail."""
    return pl.LazyFrame(
        {
            "ticker": ["PASS", "PRICE_BOUNDARY", "NULL_SUPPORT"],
            "trade_date": ["1/2/2015"] * 3,
            "expirDate": ["2/20/2015"] * 3,
            "strike": [80.0] * 3,
            "cAskPx": [4.2, 0.1, 4.2],
            "cBidPx": [4.1, 0.1, 4.1],
            "pAskPx": [1.2] * 3,
            "pBidPx": [1.1] * 3,
            "stkPx": [82.0] * 3,
            "cVolu": [20.0] * 3,
            "pVolu": [10.0] * 3,
            "cOi": [200.0] * 3,
            "pOi": [100.0] * 3,
            "cValue": [4.15] * 3,
            "pValue": [1.15] * 3,
            "cMidIv": [0.2] * 3,
            "pMidIv": [0.21] * 3,
            "support": [1.0, 1.0, None],
        },
        schema={**ORATS_RAW_OPTION_DTYPES, "support": pl.Float64},
    )


def test_shared_features_preserve_exact_dtypes_and_formulas() -> None:
    """Feature names, precision, signed intrinsic values, and both spreads are explicit."""
    featured = add_quote_features(canonicalize_raw_orats_options(raw_rows())).collect()
    for name, dtype in QUOTE_FEATURE_DTYPES.items():
        assert featured.schema[name] == dtype
    first = featured.row(0, named=True)
    assert first["days_to_expiry"] == EXPECTED_DAYS_TO_EXPIRY
    assert first["yte"] == pytest.approx(EXPECTED_DAYS_TO_EXPIRY / 365)
    assert first["call_moneyness"] == pytest.approx(82 / 80)
    assert first["put_moneyness"] == pytest.approx(80 / 82)
    assert first["call_intrinsic"] == pytest.approx(2.0)
    assert first["put_intrinsic"] == pytest.approx(-2.0)
    assert first["project_call_spread"] == pytest.approx((4.2 - 4.15) / 4.15, rel=1e-5)
    assert first["call_spread"] == pytest.approx((4.2 - 4.1) / 4.15, rel=1e-5)


def test_project_filter_preserves_strict_price_and_complete_row_semantics() -> None:
    """Only the row above strict quote bounds and complete after support joins survives."""
    featured = add_quote_features(canonicalize_raw_orats_options(raw_rows())).with_columns(
        pl.Series("support", [1.0, 1.0, None], dtype=pl.Float64),
    )
    observed = apply_quote_filter(featured, PROJECT_AS_SUBMITTED_FILTER).collect()
    assert observed.get_column("ticker").to_list() == ["PASS"]
    assert "support" in observed.columns


def test_filter_configuration_rejects_inferred_types_and_reversed_domains() -> None:
    """Estimator-defining bounds cannot be coerced or reordered silently."""
    invalid_integer: object = True
    with pytest.raises(QuoteFilterSpecError, match="plain int"):
        replace(
            PROJECT_AS_SUBMITTED_FILTER,
            min_days_to_expiry=cast("int", invalid_integer),
        )
    with pytest.raises(QuoteFilterSpecError, match="at least min_call_moneyness"):
        replace(
            PROJECT_AS_SUBMITTED_FILTER,
            min_call_moneyness=2.0,
            max_call_moneyness=1.0,
        )
    with pytest.raises(QuoteFilterSpecError, match="on or after min_quote_date"):
        replace(
            PROJECT_AS_SUBMITTED_FILTER,
            min_quote_date=date(2020, 1, 1),
            max_quote_date=date(2019, 1, 1),
        )


def test_feature_and_filter_schema_mismatches_fail_before_collection() -> None:
    """Neither normalized inputs nor derived filter fields are silently narrowed."""
    normalized = canonicalize_raw_orats_options(raw_rows())
    with pytest.raises(QuoteSchemaError, match="dtype mismatches"):
        add_quote_features(normalized.with_columns(pl.col("strike").cast(pl.Float64)))

    featured = add_quote_features(normalized)
    with pytest.raises(QuoteSchemaError, match="dtype mismatches"):
        apply_quote_filter(
            featured.with_columns(pl.col("call_mid_price").cast(pl.Float64)),
            PROJECT_AS_SUBMITTED_FILTER,
        )
