"""Tests for verified natural strike groups and grouped native classical pricing."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from finoptions import (
    CashDividendBatch,
    ContinuousDividendYieldBatch,
    CpuMathBackend,
    EscrowedCashDividendBatch,
    MaturityBoundary,
)

from discrete_dividend_study.classical_artifact import (
    load_classical_study_artifact,
    run_and_write_project_classical_grouped_study,
)
from discrete_dividend_study.classical_groups import (
    build_project_classical_grouped_frame,
    prepare_project_classical_strike_groups,
    prepared_classical_strike_groups_sha256,
)
from discrete_dividend_study.classical_inputs import ClassicalInputContentError
from discrete_dividend_study.classical_pricing import run_project_classical_grouped_study
from discrete_dividend_study.classical_spec import (
    PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    ClassicalExecutionSpec,
    ClassicalPopulationAddressing,
    ClassicalVariant,
)
from discrete_dividend_study.neural_run import StudySourceIdentity

if TYPE_CHECKING:
    from pathlib import Path

_ROWS = 5
_GROUPS = 2
_SHA256_HEX_LENGTH = 64
_ARTIFACT_FORMAT_VERSION = 4
_SOURCE = StudySourceIdentity(
    feature_artifact_sha256="1" * 64,
    feature_schema_sha256="2" * 64,
    paper_revision="a" * 40,
    engine_revision="b" * 40,
    dependency_lock_sha256="3" * 64,
)


def grouped_frame() -> pl.DataFrame:
    """Return two exact quote/expiry states with five distinct row-ordered strikes."""
    return pl.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "quote_date": [date(2020, 1, 2)] * 3 + [date(2020, 2, 3)] * 2,
            "expiry_date": [date(2021, 1, 2)] * 3 + [date(2020, 11, 3)] * 2,
            "underlying_price": [100.0] * 3 + [120.0] * 2,
            "strike": [70.0, 80.0, 90.0, 80.0, 90.0],
            "yte": [1.0] * 3 + [0.75] * 2,
            "risk_free_rate": [0.03] * 3 + [0.02] * 2,
            "historical_volatility_90": [0.20] * 3 + [0.25] * 2,
            "forecasted_dividend_yield": [0.02] * 3 + [0.01] * 2,
            "historical_dividend_yield": [0.01] * 3 + [0.02] * 2,
            "call_mid_price": [31.0, 22.0, 14.0, 41.0, 32.0],
            "forecasted_years_to_discrete_ex_dates": [[0.25, 0.75]] * 3 + [[0.25]] * 2,
            "forecasted_discrete_amounts": [[1.0, 1.0]] * 3 + [[0.5]] * 2,
        },
        schema={
            "ticker": pl.String,
            "quote_date": pl.Date,
            "expiry_date": pl.Date,
            "underlying_price": pl.Float32,
            "strike": pl.Float32,
            "yte": pl.Float64,
            "risk_free_rate": pl.Float64,
            "historical_volatility_90": pl.Float64,
            "forecasted_dividend_yield": pl.Float64,
            "historical_dividend_yield": pl.Float64,
            "call_mid_price": pl.Float32,
            "forecasted_years_to_discrete_ex_dates": pl.List(pl.Float64),
            "forecasted_discrete_amounts": pl.List(pl.Float64),
        },
    )


def execution() -> ClassicalExecutionSpec:
    """Return a small explicit portable execution identity."""
    return ClassicalExecutionSpec(
        threads=2,
        math_backend=CpuMathBackend.RUST_STD_F64,
        training_seed=7,
        training_domain_start=100,
        evaluation_seed=11,
        evaluation_domain_start=10_000,
        population_addressing=ClassicalPopulationAddressing.NATURAL_STRIKE_GROUPS,
    )


def test_lazy_projection_and_materialized_compaction_preserve_exact_row_order() -> None:
    """The lazy sort creates contiguous keys and compaction stores state exactly once per group."""
    shuffled = grouped_frame()[[3, 1, 4, 0, 2]]
    lazy = build_project_classical_grouped_frame(shuffled.lazy())
    assert type(lazy) is pl.LazyFrame
    materialized = lazy.collect()
    assert materialized.select("ticker", "strike").rows() == [
        ("AAA", 70.0),
        ("AAA", 80.0),
        ("AAA", 90.0),
        ("BBB", 80.0),
        ("BBB", 90.0),
    ]

    prepared = prepare_project_classical_strike_groups(materialized)
    assert prepared.groups == _GROUPS
    assert prepared.rows == _ROWS
    assert prepared.group_offsets.tolist() == [0, 3, 5]
    assert prepared.initial_spots.tolist() == [100.0, 120.0]
    assert prepared.strikes.tolist() == [70.0, 80.0, 90.0, 80.0, 90.0]
    assert prepared.targets.tolist() == [31.0, 22.0, 14.0, 41.0, 32.0]
    assert prepared.cash_dividends.rows == _GROUPS
    assert len(prepared_classical_strike_groups_sha256(prepared)) == _SHA256_HEX_LENGTH
    assert prepared_classical_strike_groups_sha256(
        prepare_project_classical_strike_groups(materialized)
    ) == prepared_classical_strike_groups_sha256(prepared)
    for array in (prepared.group_offsets, *prepared.scalar_arrays()):
        assert array.flags.c_contiguous
        assert not array.flags.writeable

    exact = prepared.dividends_for(ClassicalVariant.EXACT_CASH)
    assert type(exact) is CashDividendBatch
    assert exact.maturity_boundary is MaturityBoundary.AT_OR_BEFORE
    assert (
        type(prepared.dividends_for(ClassicalVariant.ESCROWED_CASH_SPOT_MATCHED))
        is EscrowedCashDividendBatch
    )
    assert type(prepared.dividends_for(ClassicalVariant.FORECAST_YIELD)) is (
        ContinuousDividendYieldBatch
    )


def test_grouped_terminal_cash_boundary_admits_equality_and_rejects_later_events() -> None:
    """Group compaction enforces the same explicit terminal ordering as row preparation."""
    at_maturity = grouped_frame().with_columns(
        pl.when(pl.col("ticker") == "AAA")
        .then(pl.lit([0.25, 1.0], dtype=pl.List(pl.Float64)))
        .otherwise(pl.col("forecasted_years_to_discrete_ex_dates"))
        .alias("forecasted_years_to_discrete_ex_dates")
    )
    prepared = prepare_project_classical_strike_groups(at_maturity)
    assert prepared.cash_dividends.times.tolist() == [0.25, 1.0, 0.25]

    strict_study = replace(
        PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
        cash_maturity_boundary=MaturityBoundary.STRICTLY_BEFORE,
    )
    with pytest.raises(
        ClassicalInputContentError,
        match="cash_dividend_schedules_at_excluded_maturity",
    ):
        prepare_project_classical_strike_groups(at_maturity, strict_study)

    after_maturity = at_maturity.with_columns(
        pl.when(pl.col("ticker") == "AAA")
        .then(pl.lit([0.25, 1.01], dtype=pl.List(pl.Float64)))
        .otherwise(pl.col("forecasted_years_to_discrete_ex_dates"))
        .alias("forecasted_years_to_discrete_ex_dates")
    )
    with pytest.raises(ClassicalInputContentError, match="cash_dividend_schedules_after_maturity"):
        prepare_project_classical_strike_groups(after_maturity)


def test_noncontiguous_nonconstant_and_duplicate_contract_groups_fail_without_repair() -> None:
    """Natural keys and every simulated state field are exact materialized invariants."""
    with pytest.raises(ClassicalInputContentError, match="noncontiguous_natural_groups"):
        prepare_project_classical_strike_groups(grouped_frame()[[0, 3, 1, 2, 4]])

    nonconstant = (
        grouped_frame()
        .with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 1)
            .then(pl.lit(0.21, dtype=pl.Float64))
            .otherwise(pl.col("historical_volatility_90"))
            .alias("historical_volatility_90")
        )
        .drop("index")
    )
    with pytest.raises(
        ClassicalInputContentError,
        match="nonconstant_group_historical_volatility_90",
    ):
        prepare_project_classical_strike_groups(nonconstant)

    duplicate = (
        grouped_frame()
        .with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 1)
            .then(pl.lit(70.0, dtype=pl.Float32))
            .otherwise(pl.col("strike"))
            .alias("strike")
        )
        .drop("index")
    )
    with pytest.raises(ClassicalInputContentError, match="duplicate_group_strikes"):
        prepare_project_classical_strike_groups(duplicate)


def test_all_four_variants_price_through_group_addressed_native_lsm() -> None:
    """The publication comparison retains every row while simulating only two path states."""
    prepared = prepare_project_classical_strike_groups(grouped_frame())
    result = run_project_classical_grouped_study(prepared, execution())
    assert result.rows == _ROWS
    assert tuple(item.variant.variant for item in result.variants) == tuple(ClassicalVariant)
    for variant in result.variants:
        assert variant.metrics.observations == _ROWS
        assert np.isfinite(variant.native.prices).all()
        assert np.all(variant.native.prices >= 0.0)
        assert np.array_equal(
            variant.native.early_exercise_paths.astype(np.uint64)
            + variant.native.expiry_paths.astype(np.uint64),
            np.full(_ROWS, prepared.study.estimator.evaluation_paths, dtype=np.uint64),
        )


def test_grouped_artifact_binds_population_units_and_round_trips(tmp_path: Path) -> None:
    """Artifact v3 distinguishes five outputs from two group-addressed path populations."""
    prepared = prepare_project_classical_strike_groups(grouped_frame())
    artifact = run_and_write_project_classical_grouped_study(
        tmp_path,
        _SOURCE,
        prepared,
        execution(),
    )
    assert artifact.manifest.format_version == _ARTIFACT_FORMAT_VERSION
    assert artifact.manifest.observations == _ROWS
    assert artifact.manifest.simulation_units == _GROUPS
    assert (
        artifact.manifest.execution.population_addressing
        is ClassicalPopulationAddressing.NATURAL_STRIKE_GROUPS
    )
    loaded = load_classical_study_artifact(artifact.path)
    np.testing.assert_array_equal(loaded.target, prepared.targets)
    assert loaded.artifact.manifest == artifact.manifest
