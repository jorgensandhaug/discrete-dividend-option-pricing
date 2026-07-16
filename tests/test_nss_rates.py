"""Strict NSS scalar-rate feature tests."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from typing import cast

import polars as pl
import pytest

from discrete_dividend_study import (
    NSS_PARAMETER_DTYPES,
    PROJECT_AS_SUBMITTED_NSS_RATE_SPEC,
    ConflictingColumnsError,
    NssRateContentError,
    NssRateFeatureSpec,
    NssRateSchemaError,
    NssRateSpecError,
    add_nss_rate_feature,
)

SHORTEST_RATE = 0.019


def quote_rows() -> pl.LazyFrame:
    """Return one short-end and one NSS-evaluated maturity in deliberate order."""
    return pl.LazyFrame(
        {
            "quote_date": [date(2024, 1, 2)] * 2,
            "yte": [0.25, 2.0],
            "contract_id": [2, 1],
        },
        schema={"quote_date": pl.Date, "yte": pl.Float64, "contract_id": pl.Int64},
    )


def parameter_rows() -> pl.LazyFrame:
    """Return one finite positive-scale NSS parameter row."""
    return pl.LazyFrame(
        {
            "date": [date(2024, 1, 2)],
            "beta0": [0.03],
            "beta1": [-0.01],
            "beta2": [0.02],
            "beta3": [0.005],
            "tau1": [1.5],
            "tau2": [3.0],
            "shortest_maturity": [0.25],
            "shortest_rate": [0.019],
        },
        schema=NSS_PARAMETER_DTYPES,
    )


def python_nss_rate(maturity: float) -> float:
    """Independent direct-form reference at a well-conditioned maturity."""
    x1 = maturity / 1.5
    x2 = maturity / 3.0
    level1 = (1.0 - math.exp(-x1)) / x1
    curvature1 = level1 - math.exp(-x1)
    curvature2 = (1.0 - math.exp(-x2)) / x2 - math.exp(-x2)
    return 0.03 - 0.01 * level1 + 0.02 * curvature1 + 0.005 * curvature2


def test_submitted_short_end_and_nss_formula_preserve_quote_order() -> None:
    """Observed shortest rate and an independent NSS calculation both match."""
    observed = add_nss_rate_feature(
        quote_rows(), parameter_rows(), PROJECT_AS_SUBMITTED_NSS_RATE_SPEC
    ).collect()
    assert observed.get_column("contract_id").to_list() == [2, 1]
    assert observed.schema["risk_free_rate"] == pl.Float64
    assert observed["risk_free_rate"][0] == SHORTEST_RATE
    assert observed["risk_free_rate"][1] == pytest.approx(python_nss_rate(2.0))
    assert not any(name.startswith(("beta", "tau")) for name in observed.columns)


def test_nss_only_policy_is_stable_near_zero() -> None:
    """The algebraically equivalent loading series remains finite near zero maturity."""
    nss_only = NssRateFeatureSpec(
        minimum_nss_maturity=1.0e-12,
        use_observed_shortest_rate=False,
    )
    tiny = quote_rows().slice(0, 1).with_columns(pl.lit(1.0e-12).alias("yte"))
    observed = (
        add_nss_rate_feature(tiny, parameter_rows(), nss_only).collect().item(0, "risk_free_rate")
    )
    assert math.isfinite(observed)
    assert observed == pytest.approx(0.02, abs=1.0e-12)


def test_missing_dates_invalid_scales_and_precision_fail_before_output() -> None:
    """No parameter carry-forward, coercion, or invalid decay scale reaches evaluation."""
    with pytest.raises(NssRateContentError, match="missing_parameter_dates"):
        add_nss_rate_feature(
            quote_rows().with_columns(pl.lit(date(2024, 1, 3)).alias("quote_date")),
            parameter_rows(),
            PROJECT_AS_SUBMITTED_NSS_RATE_SPEC,
        )
    with pytest.raises(NssRateContentError, match="nonpositive_parameter_scales"):
        add_nss_rate_feature(
            quote_rows(),
            parameter_rows().with_columns(pl.lit(0.0).alias("tau1")),
            PROJECT_AS_SUBMITTED_NSS_RATE_SPEC,
        )
    with pytest.raises(NssRateSchemaError, match="dtype mismatches"):
        add_nss_rate_feature(
            quote_rows(),
            parameter_rows().with_columns(pl.col("beta0").cast(pl.Float32)),
            PROJECT_AS_SUBMITTED_NSS_RATE_SPEC,
        )
    with pytest.raises(ConflictingColumnsError, match="already exist"):
        add_nss_rate_feature(
            quote_rows().with_columns(pl.lit(0.01).alias("risk_free_rate")),
            parameter_rows(),
            PROJECT_AS_SUBMITTED_NSS_RATE_SPEC,
        )


def test_rate_spec_rejects_coercions_and_invalid_minimum_maturity() -> None:
    """Numerical domain and short-end policy are explicit typed values."""
    invalid_flag: object = 1
    with pytest.raises(NssRateSpecError, match="plain bool"):
        replace(
            PROJECT_AS_SUBMITTED_NSS_RATE_SPEC,
            use_observed_shortest_rate=cast("bool", invalid_flag),
        )
    with pytest.raises(NssRateSpecError, match="finite positive"):
        replace(PROJECT_AS_SUBMITTED_NSS_RATE_SPEC, minimum_nss_maturity=0.0)
