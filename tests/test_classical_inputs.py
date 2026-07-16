"""Tests for the four-baseline classical study identity and native input boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest
from finoptions import (
    CashDividendBatch,
    ContinuousDividendYieldBatch,
    CpuMathBackend,
    EscrowedCashDividendBatch,
    EscrowedVolatilityConvention,
    InsufficientExerciseSupportPolicy,
    MaturityBoundary,
)

from discrete_dividend_study.classical_inputs import (
    ClassicalInputContentError,
    ClassicalInputSchemaError,
    prepare_project_classical_inputs,
    prepared_classical_inputs_sha256,
)
from discrete_dividend_study.classical_pricing import run_project_classical_study
from discrete_dividend_study.classical_spec import (
    PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    ClassicalExecutionSpec,
    ClassicalVariant,
    classical_study_sha256,
)
from discrete_dividend_study.event_buffers import EventBufferError, EventBufferPolicy

_ROWS = 3
_SHA256_HEX_LENGTH = 64
_BASE_INTERVALS = 150
_PATHS = 500
_POLYNOMIAL_DEGREE = 4
_FRAGMENT_COUNT = 2


def classical_frame() -> pl.DataFrame:
    """Return valid exact-schema rows spanning cash, empty, and multiple-event schedules."""
    return pl.DataFrame(
        {
            "quote_date": [date(2015, 1, 2), date(2020, 1, 2), date(2024, 8, 9)],
            "underlying_price": [100.0, 120.0, 90.0],
            "strike": [90.0, 105.0, 80.0],
            "yte": [1.0, 0.75, 0.5],
            "risk_free_rate": [0.03, 0.02, 0.01],
            "historical_volatility_90": [0.20, 0.25, 0.30],
            "forecasted_dividend_yield": [0.02, 0.00, 0.03],
            "historical_dividend_yield": [0.01, 0.02, 0.00],
            "call_mid_price": [14.0, 20.0, 13.0],
            "forecasted_years_to_discrete_ex_dates": [[0.25, 0.75], [], [0.25]],
            "forecasted_discrete_amounts": [[1.0, 1.0], [], [0.5]],
        },
        schema={
            "quote_date": pl.Date,
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


def test_reconstructed_classical_identity_pins_all_four_submitted_names() -> None:
    """The frozen study names every paper baseline and its shared LSM estimator."""
    study = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC
    assert tuple(variant.variant for variant in study.variants) == tuple(ClassicalVariant)
    assert tuple(variant.submitted_result_name for variant in study.variants) == (
        "IT_GBM_SPOT",
        "IT_GBMT_FD_AH",
        "IT_GBMT_FY",
        "IT_GBMT_HY",
    )
    assert study.estimator.base_intervals == _BASE_INTERVALS
    assert study.estimator.training_paths == _PATHS
    assert study.estimator.evaluation_paths == _PATHS
    assert study.estimator.polynomial_degree == _POLYNOMIAL_DEGREE
    assert (
        study.estimator.insufficient_exercise_support is InsufficientExerciseSupportPolicy.CONTINUE
    )
    assert study.cash_maturity_boundary is MaturityBoundary.AT_OR_BEFORE
    assert len(classical_study_sha256(study)) == _SHA256_HEX_LENGTH
    assert classical_study_sha256(study) == classical_study_sha256(study)
    assert classical_study_sha256(
        replace(study, cash_maturity_boundary=MaturityBoundary.STRICTLY_BEFORE)
    ) != classical_study_sha256(study)
    assert classical_study_sha256(
        replace(
            study,
            estimator=replace(
                study.estimator,
                insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
            ),
        )
    ) != classical_study_sha256(study)


def test_one_float64_allocation_and_one_ragged_schedule_feed_every_variant() -> None:
    """All models share immutable native columns while retaining distinct dividend semantics."""
    prepared = prepare_project_classical_inputs(classical_frame())
    assert prepared.rows == _ROWS
    arrays = prepared.scalar_arrays()
    assert all(array.dtype == np.dtype(np.float64) for array in arrays)
    assert all(array.flags.c_contiguous and not array.flags.writeable for array in arrays)
    assert all(array.base is arrays[0].base for array in arrays)
    assert prepared.initial_spots.tolist() == [100.0, 120.0, 90.0]
    assert prepared.targets.tolist() == [14.0, 20.0, 13.0]
    assert prepared.cash_dividends.rows == _ROWS
    assert not prepared.cash_dividends.offsets.flags.writeable
    assert not prepared.cash_dividends.times.flags.writeable
    assert not prepared.cash_dividends.amounts.flags.writeable

    exact = prepared.dividends_for(ClassicalVariant.EXACT_CASH)
    escrowed = prepared.dividends_for(ClassicalVariant.ESCROWED_CASH_SPOT_MATCHED)
    forecast = prepared.dividends_for(ClassicalVariant.FORECAST_YIELD)
    historical = prepared.dividends_for(ClassicalVariant.HISTORICAL_YIELD)
    assert type(exact) is CashDividendBatch
    assert exact.schedules is prepared.cash_dividends
    assert exact.maturity_boundary is MaturityBoundary.AT_OR_BEFORE
    assert type(escrowed) is EscrowedCashDividendBatch
    assert escrowed.schedules is prepared.cash_dividends
    assert escrowed.volatility_convention is EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN
    assert escrowed.maturity_boundary is MaturityBoundary.AT_OR_BEFORE
    assert type(forecast) is ContinuousDividendYieldBatch
    assert forecast.annualized_decimals is prepared.forecast_yields
    assert type(historical) is ContinuousDividendYieldBatch
    assert historical.annualized_decimals is prepared.historical_yields


def test_prepared_hash_binds_study_scalars_target_and_cash_events() -> None:
    """Exact model inputs have a stable content identity independent of local paths."""
    first = prepare_project_classical_inputs(classical_frame())
    second = prepare_project_classical_inputs(classical_frame())
    first_hash = prepared_classical_inputs_sha256(first)
    assert len(first_hash) == _SHA256_HEX_LENGTH
    assert first_hash == prepared_classical_inputs_sha256(second)

    changed = prepare_project_classical_inputs(
        classical_frame().with_columns(
            (pl.col("call_mid_price") + pl.lit(0.25, dtype=pl.Float32)).alias("call_mid_price"),
        ),
    )
    assert prepared_classical_inputs_sha256(changed) != first_hash


def test_fragmented_event_storage_requires_an_explicit_rechunk_policy() -> None:
    """Preparation cannot hide an event-child allocation behind a zero-copy request."""
    source = classical_frame()
    fragmented = pl.concat([source.slice(0, 1), source.slice(1)], rechunk=False)
    assert fragmented.get_column("forecasted_discrete_amounts").n_chunks() == _FRAGMENT_COUNT
    with pytest.raises(EventBufferError, match="explicit rechunk"):
        prepare_project_classical_inputs(fragmented)
    prepared = prepare_project_classical_inputs(
        fragmented,
        event_buffer_policy=EventBufferPolicy.ALLOW_RECHUNK,
    )
    assert prepared.rows == _ROWS


def test_schema_and_financial_content_fail_before_native_pricing() -> None:
    """No dtype inference, negative volatility, or invalid target reaches the engine."""
    with pytest.raises(ClassicalInputSchemaError, match="historical_volatility_90"):
        prepare_project_classical_inputs(
            classical_frame().with_columns(pl.col("historical_volatility_90").cast(pl.Float32)),
        )
    with pytest.raises(ClassicalInputContentError, match="nonpositive_volatilities"):
        prepare_project_classical_inputs(
            classical_frame().with_columns(
                pl.lit(0.0, dtype=pl.Float64).alias("historical_volatility_90"),
            ),
        )
    with pytest.raises(ClassicalInputContentError, match="nonpositive_targets"):
        prepare_project_classical_inputs(
            classical_frame().with_columns(
                pl.lit(0.0, dtype=pl.Float32).alias("call_mid_price"),
            ),
        )


def test_cash_event_maturity_boundary_is_explicit_and_never_repairs_source_times() -> None:
    """The study admits declared terminal cash but rejects excluded or post-expiry events."""
    at_maturity = classical_frame().with_columns(
        pl.Series(
            "forecasted_years_to_discrete_ex_dates",
            [[0.25, 1.0], [], [0.25]],
            dtype=pl.List(pl.Float64),
        ),
    )
    admitted = prepare_project_classical_inputs(at_maturity)
    assert admitted.cash_dividends.times.tolist() == [0.25, 1.0, 0.25]

    strict_study = replace(
        PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
        cash_maturity_boundary=MaturityBoundary.STRICTLY_BEFORE,
    )
    with pytest.raises(
        ClassicalInputContentError,
        match="cash_dividend_schedules_at_excluded_maturity",
    ):
        prepare_project_classical_inputs(at_maturity, strict_study)

    after_maturity = at_maturity.with_columns(
        pl.Series(
            "forecasted_years_to_discrete_ex_dates",
            [[0.25, 1.01], [], [0.25]],
            dtype=pl.List(pl.Float64),
        ),
    )
    with pytest.raises(ClassicalInputContentError, match="cash_dividend_schedules_after_maturity"):
        prepare_project_classical_inputs(after_maturity)

    coalesced = classical_frame().with_columns(
        pl.Series(
            "forecasted_years_to_discrete_ex_dates",
            [[0.25, 0.25], [], [0.25]],
            dtype=pl.List(pl.Float64),
        ),
    )
    with pytest.raises(ClassicalInputContentError, match="nonincreasing_cash_dividend_times"):
        prepare_project_classical_inputs(coalesced)


def test_all_four_prepared_variants_execute_through_the_native_batch_contract() -> None:
    """Every typed reconstruction reaches the engine without a dataframe callback or fallback."""
    prepared = prepare_project_classical_inputs(classical_frame())
    execution = ClassicalExecutionSpec(
        threads=2,
        math_backend=CpuMathBackend.RUST_STD_F64,
        training_seed=7,
        training_domain_start=100,
        evaluation_seed=11,
        evaluation_domain_start=10_000,
    )
    result = run_project_classical_study(prepared, execution)
    assert result.rows == _ROWS
    assert tuple(item.variant.variant for item in result.variants) == tuple(ClassicalVariant)
    for variant in result.variants:
        assert variant.metrics.observations == _ROWS
        assert np.isfinite(variant.native.prices).all()
        assert np.all(variant.native.prices >= 0.0)
        assert not variant.native.insufficient_exercise_support_nodes.flags.writeable
        assert np.array_equal(
            variant.native.early_exercise_paths.astype(np.uint64)
            + variant.native.expiry_paths.astype(np.uint64),
            np.full(_ROWS, prepared.study.estimator.evaluation_paths, dtype=np.uint64),
        )
