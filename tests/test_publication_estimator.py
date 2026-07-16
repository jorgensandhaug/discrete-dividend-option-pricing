"""Tests for the separate fail-fast publication estimator and cash-domain partition."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from finoptions import (
    AndersenLake,
    AndersenLakeBatchResult,
    CpuExecutor,
    CpuMathBackend,
    DividendPolicy,
    ExerciseMesh,
    ExerciseTieRule,
    ExplicitExerciseMesh,
    InsufficientExerciseSupportPolicy,
    Lsm,
    LsmBatchResult,
    LsmStrikeGroups,
    MaturityBoundary,
    OriginExercise,
    UniformExerciseMesh,
)

from discrete_dividend_study import publication_pricing
from discrete_dividend_study.classical_groups import (
    PreparedClassicalStrikeGroups,
    build_project_classical_grouped_frame,
    prepare_project_classical_strike_groups,
)
from discrete_dividend_study.market_calendar import load_project_xnys_calendar
from discrete_dividend_study.publication_exercise import (
    PreparedPublicationExerciseSchedules,
    prepare_publication_exercise_schedules,
)
from discrete_dividend_study.publication_groups import (
    PublicationCashDomain,
    PublicationPartitionError,
    prepare_publication_partitions,
    publication_partitions_sha256,
)
from discrete_dividend_study.publication_spec import (
    PROJECT_PUBLICATION_CASH_LSM_SPEC,
    PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC,
    PROJECT_PUBLICATION_STUDY_SPEC,
    PROJECT_PUBLICATION_YIELD_BOUNDARY_SPEC,
    PublicationCashLsmSpec,
    PublicationCashPartitionRule,
    PublicationDeterministicExecutionSpec,
    PublicationReplicationExecutionSpec,
    PublicationSpecValueError,
    PublicationStudySpec,
    PublicationVariant,
    publication_deterministic_execution_sha256,
    publication_execution_sha256,
    publication_replication_plan_sha256,
    publication_study_sha256,
)
from tests.test_publication_exercise import schedule_frame

_ROWS = 4
_GROUPS = 3
_PATHS = 2_048
_SHA256_HEX_LENGTH = 64
_EXACT_GROUPS = 2
_GENERAL_GROUPS = 1
_EXACT_MESH_CALLS = 2
_UNIFORM_MESH_CALLS = 2
_YIELD_PARTITION_CALLS = 4
_PRODUCTION_GENERAL_INTERVALS = 128
_PRODUCTION_PATHS = 32_768
_PRODUCTION_POLYNOMIAL_DEGREE = 4
_PRODUCTION_REPLICATIONS = 12
_PRODUCTION_CONFIDENCE_LEVEL = 0.99
_PRODUCTION_REPLICATION_CRITICAL_VALUE = 3.1058065155392804
_PRODUCTION_TRAINING_DOMAIN = 10_000
_PRODUCTION_EVALUATION_DOMAIN = 1_000_000


def study() -> PublicationStudySpec:
    """Return a small cash test identity with the selected deterministic yield profile."""
    return PublicationStudySpec(
        name="publication-test-fixture",
        variants=tuple(PublicationVariant),
        cash_lsm=PublicationCashLsmSpec(
            general_base_intervals=2,
            general_origin_exercise=OriginExercise.INCLUDED,
            training_paths=_PATHS,
            evaluation_paths=_PATHS,
            polynomial_degree=0,
            rank_tolerance=1.0e-12,
            condition_limit=1.0e8,
            tie_rule=ExerciseTieRule.STRICTLY_GREATER,
            insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
            cash_partition_rule=PublicationCashPartitionRule.NONNEGATIVE_RATE_EXACT_CALENDAR,
        ),
        yield_boundary=PROJECT_PUBLICATION_YIELD_BOUNDARY_SPEC,
        exact_cash_policy=DividendPolicy.LIQUIDATOR,
        cash_maturity_boundary=MaturityBoundary.AT_OR_BEFORE,
    )


def execution() -> PublicationReplicationExecutionSpec:
    """Return one complete portable replication identity."""
    return PublicationReplicationExecutionSpec(
        replication=0,
        threads=2,
        math_backend=CpuMathBackend.RUST_STD_F64,
        training_seed=7,
        training_domain_start=100,
        evaluation_seed=11,
        evaluation_domain_start=10_000,
    )


def publication_inputs(
    frame: pl.DataFrame | None = None,
) -> tuple[PreparedClassicalStrikeGroups, PreparedPublicationExerciseSchedules]:
    """Return positive pre-expiry/terminal and negative empty-cash natural groups."""
    source = (
        schedule_frame().with_columns(
            pl.when(pl.col("ticker") == "CCC")
            .then(pl.lit(-0.001, dtype=pl.Float64))
            .otherwise(pl.col("risk_free_rate"))
            .alias("risk_free_rate")
        )
        if frame is None
        else frame
    )
    materialized = build_project_classical_grouped_frame(source.lazy()).collect()
    inputs = prepare_project_classical_strike_groups(materialized)
    schedules = prepare_publication_exercise_schedules(
        materialized,
        inputs,
        load_project_xnys_calendar(),
    )
    return inputs, schedules


def test_publication_identity_has_no_implicit_budget_or_sparse_support() -> None:
    """The new study is separately hashed and forbids reconstruction-only continuation."""
    declared = study()
    assert len(publication_study_sha256(declared)) == _SHA256_HEX_LENGTH
    assert len(publication_execution_sha256(execution())) == _SHA256_HEX_LENGTH
    assert publication_study_sha256(declared) != publication_study_sha256(
        replace(
            declared,
            cash_lsm=replace(declared.cash_lsm, training_paths=_PATHS * 2),
        )
    )
    assert publication_study_sha256(declared) != publication_study_sha256(
        replace(
            declared,
            yield_boundary=replace(declared.yield_boundary, iterations=17),
        )
    )
    with pytest.raises(PublicationSpecValueError, match=r"OriginExercise\.INCLUDED"):
        replace(
            declared.cash_lsm,
            general_origin_exercise=OriginExercise.EXCLUDED,
        )
    with pytest.raises(
        PublicationSpecValueError,
        match=r"InsufficientExerciseSupportPolicy\.FAIL",
    ):
        replace(
            declared.cash_lsm,
            insufficient_exercise_support=InsufficientExerciseSupportPolicy.CONTINUE,
        )
    with pytest.raises(PublicationSpecValueError, match=r"DividendPolicy\.LIQUIDATOR"):
        replace(declared, exact_cash_policy=DividendPolicy.SURVIVOR)
    with pytest.raises(PublicationSpecValueError, match="plain int"):
        replace(declared.yield_boundary, iterations=True)
    with pytest.raises(PublicationSpecValueError, match="yield_boundary_method"):
        replace(declared.yield_boundary, collocation_nodes=1)
    with pytest.raises(PublicationSpecValueError, match="yield_boundary_method"):
        replace(declared.yield_boundary, boundary_quadrature_order=-1)


def test_frozen_publication_study_and_replication_plan_are_complete() -> None:
    """Production controls, seeds, domains, and inference critical value are explicit."""
    cash = PROJECT_PUBLICATION_CASH_LSM_SPEC
    assert cash.general_base_intervals == _PRODUCTION_GENERAL_INTERVALS
    assert cash.training_paths == _PRODUCTION_PATHS
    assert cash.evaluation_paths == _PRODUCTION_PATHS
    assert cash.polynomial_degree == _PRODUCTION_POLYNOMIAL_DEGREE
    assert cash.insufficient_exercise_support is InsufficientExerciseSupportPolicy.FAIL
    assert PROJECT_PUBLICATION_STUDY_SPEC.cash_lsm is cash
    assert PROJECT_PUBLICATION_STUDY_SPEC.yield_boundary is PROJECT_PUBLICATION_YIELD_BOUNDARY_SPEC
    assert len(publication_study_sha256(PROJECT_PUBLICATION_STUDY_SPEC)) == _SHA256_HEX_LENGTH

    plan = PROJECT_PUBLICATION_REPLICATION_PLAN_SPEC
    assert plan.replications == _PRODUCTION_REPLICATIONS
    assert plan.confidence_level == _PRODUCTION_CONFIDENCE_LEVEL
    assert plan.replication_critical_value == _PRODUCTION_REPLICATION_CRITICAL_VALUE
    executions = tuple(
        plan.execution(
            replication,
            threads=14,
            math_backend=CpuMathBackend.APPLE_ACCELERATE,
        )
        for replication in range(plan.replications)
    )
    assert len({execution.training_seed for execution in executions}) == plan.replications
    assert all(execution.training_seed == execution.evaluation_seed for execution in executions)
    assert all(
        execution.training_domain_start == _PRODUCTION_TRAINING_DOMAIN for execution in executions
    )
    assert all(
        execution.evaluation_domain_start == _PRODUCTION_EVALUATION_DOMAIN
        for execution in executions
    )
    assert len({publication_execution_sha256(item) for item in executions}) == plan.replications
    plan.validate_units(96_666)
    assert len(publication_replication_plan_sha256(plan)) == _SHA256_HEX_LENGTH
    assert publication_replication_plan_sha256(plan) != publication_replication_plan_sha256(
        replace(plan, seed_root=plan.seed_root + 1)
    )
    with pytest.raises(PublicationSpecValueError, match="a nonzero plain int"):
        replace(plan, seed_increment=0)
    with pytest.raises(PublicationSpecValueError, match="strictly between zero and one"):
        replace(plan, confidence_level=1.0)
    with pytest.raises(PublicationSpecValueError, match=r"\[0, 12\)"):
        plan.seed(12)
    deterministic = PublicationDeterministicExecutionSpec(
        threads=14,
        math_backend=CpuMathBackend.APPLE_ACCELERATE,
    )
    assert len(publication_deterministic_execution_sha256(deterministic)) == _SHA256_HEX_LENGTH
    assert publication_deterministic_execution_sha256(deterministic) != (
        publication_deterministic_execution_sha256(replace(deterministic, threads=13))
    )


def test_cash_domain_partition_is_complete_disjoint_and_source_ordered() -> None:
    """Calendar-exact and negative-rate groups form a complete, disjoint partition."""
    source, schedules = publication_inputs()
    prepared = prepare_publication_partitions(source, schedules, study())
    assert prepared.original_groups == _GROUPS
    assert prepared.original_rows == _ROWS
    assert prepared.simulation_units == _GROUPS
    assert prepared.exact.domain is PublicationCashDomain.EXACT
    assert prepared.general.domain is PublicationCashDomain.GENERAL
    assert prepared.exact.original_group_indices.tolist() == [0, 1]
    assert prepared.general.original_group_indices.tolist() == [2]
    assert prepared.exact.original_row_indices.tolist() == [0, 1, 2]
    assert prepared.general.original_row_indices.tolist() == [3]
    assert prepared.exact.rates.tolist() == [0.03, 0.02]
    assert prepared.general.rates.tolist() == [-0.001]
    assert prepared.exact.exercise_times is not None
    assert prepared.exact.exercise_times.offsets.tolist() == [0, 2, 3]
    assert prepared.exact.exercise_times.times.tolist() == [30 / 365, 59 / 365, 42 / 365]
    assert prepared.general.exercise_times is None
    assert len(publication_partitions_sha256(prepared)) == _SHA256_HEX_LENGTH
    for partition in prepared.ordered():
        arrays = (
            partition.original_group_indices,
            partition.original_row_indices,
            partition.group_offsets,
            partition.initial_spots,
            partition.strikes,
        )
        assert partition.cash_dividends is not None
        arrays = (
            *arrays,
            partition.cash_dividends.offsets,
            partition.cash_dividends.times,
            partition.cash_dividends.amounts,
        )
        if partition.exercise_times is not None:
            arrays = (*arrays, partition.exercise_times.offsets, partition.exercise_times.times)
        for array in arrays:
            assert not array.flags.writeable


def test_partition_rejects_a_schedule_from_another_prepared_source() -> None:
    """Schedule/source identity disagreement is fatal before domain construction."""
    source, schedules = publication_inputs()
    foreign = replace(schedules, source_prepared_sha256="0" * _SHA256_HEX_LENGTH)
    with pytest.raises(
        PublicationPartitionError, match="exercise schedules belong to another source"
    ):
        prepare_publication_partitions(source, foreign, study())


def test_publication_replication_dispatches_explicit_meshes_and_merges_original_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact cash meshes and general meshes dispatch without fallback or row reordering."""
    source, schedules = publication_inputs()
    prepared = prepare_publication_partitions(source, schedules, study())
    observed_lsm: list[tuple[type[object], int]] = []
    observed_yield: list[tuple[int, int]] = []
    native_lsm_price = publication_pricing.price_lsm_call_grouped_batch
    native_yield_price = publication_pricing.price_andersen_lake_call_grouped

    def recording_price(
        *,
        groups: LsmStrikeGroups,
        mesh: ExerciseMesh,
        executor: CpuExecutor,
        method: Lsm,
    ) -> LsmBatchResult:
        observed_lsm.append((type(mesh), method.training_stream.domain))
        return native_lsm_price(groups=groups, mesh=mesh, executor=executor, method=method)

    def recording_yield_price(  # noqa: PLR0913 - exact native facade spy
        *,
        group_offsets: npt.NDArray[np.int64],
        group_spots: npt.NDArray[np.float64],
        group_maturities: npt.NDArray[np.float64],
        group_constant_continuously_compounded_rates: npt.NDArray[np.float64],
        group_annualized_dividend_yields: npt.NDArray[np.float64],
        group_constant_volatilities: npt.NDArray[np.float64],
        strikes: npt.NDArray[np.float64],
        executor: CpuExecutor,
        method: AndersenLake,
    ) -> AndersenLakeBatchResult:
        observed_yield.append((group_offsets.size - 1, strikes.size))
        return native_yield_price(
            group_offsets=group_offsets,
            group_spots=group_spots,
            group_maturities=group_maturities,
            group_constant_continuously_compounded_rates=(
                group_constant_continuously_compounded_rates
            ),
            group_annualized_dividend_yields=group_annualized_dividend_yields,
            group_constant_volatilities=group_constant_volatilities,
            strikes=strikes,
            executor=executor,
            method=method,
        )

    monkeypatch.setattr(publication_pricing, "price_lsm_call_grouped_batch", recording_price)
    monkeypatch.setattr(
        publication_pricing,
        "price_andersen_lake_call_grouped",
        recording_yield_price,
    )
    result = publication_pricing.run_publication_replication(prepared, execution(), study())

    assert result.rows == _ROWS
    assert result.exact_groups == _EXACT_GROUPS
    assert result.general_groups == _GENERAL_GROUPS
    assert tuple(item.variant for item in result.variants) == tuple(PublicationVariant)
    assert [mesh for mesh, _ in observed_lsm].count(ExplicitExerciseMesh) == _EXACT_MESH_CALLS
    assert [mesh for mesh, _ in observed_lsm].count(UniformExerciseMesh) == _UNIFORM_MESH_CALLS
    assert [domain for _, domain in observed_lsm] == [100, 102] * 2
    assert observed_yield == [(_EXACT_GROUPS, 3), (_GENERAL_GROUPS, 1)] * 2
    assert len(observed_yield) == _YIELD_PARTITION_CALLS
    for variant in result.variants:
        assert variant.native.rows == _ROWS
        assert variant.metrics.observations == _ROWS
        assert np.isfinite(variant.native.prices).all()
        assert np.all(variant.native.prices >= 0.0)
        assert not variant.native.prices.flags.writeable
        if variant.variant.uses_cash_schedule:
            assert type(variant.native) is LsmBatchResult
            assert np.array_equal(
                variant.native.early_exercise_paths.astype(np.uint64)
                + variant.native.expiry_paths.astype(np.uint64),
                np.full(_ROWS, _PATHS, dtype=np.uint64),
            )
        else:
            assert type(variant.native) is AndersenLakeBatchResult
            assert variant.native.groups == _GROUPS
            assert not variant.native.exercised_at_origin.flags.writeable
            assert not variant.native.maximum_relative_boundary_residuals.flags.writeable


def test_cash_replication_never_dispatches_the_deterministic_yield_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resumable stochastic unit prices both cash variants and nothing else."""
    source, schedules = publication_inputs()
    declared = study()
    prepared = prepare_publication_partitions(source, schedules, declared)

    def forbidden_yield_price(**_: object) -> AndersenLakeBatchResult:
        pytest.fail("cash replication dispatched the deterministic yield solver")

    monkeypatch.setattr(
        publication_pricing,
        "price_andersen_lake_call_grouped",
        forbidden_yield_price,
    )
    result = publication_pricing.run_publication_cash_replication(
        prepared,
        execution(),
        declared,
    )

    assert result.rows == _ROWS
    assert result.partitions_sha256 == publication_partitions_sha256(prepared)
    assert tuple(item.variant for item in result.variants) == (
        PublicationVariant.CASH_JUMP_LIQUIDATOR,
        PublicationVariant.ESCROWED_CASH_SPOT_MATCHED,
    )
    assert all(type(item.native) is LsmBatchResult for item in result.variants)
    assert all(item.native.rows == _ROWS for item in result.variants)


def test_yield_pricing_never_dispatches_the_stochastic_cash_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deterministic finalization unit prices both yield variants and nothing else."""
    source, schedules = publication_inputs()
    declared = study()
    prepared = prepare_publication_partitions(source, schedules, declared)

    def forbidden_cash_price(**_: object) -> LsmBatchResult:
        pytest.fail("yield pricing dispatched the stochastic cash estimator")

    monkeypatch.setattr(
        publication_pricing,
        "price_lsm_call_grouped_batch",
        forbidden_cash_price,
    )
    result = publication_pricing.run_publication_yield_pricing(
        prepared,
        PublicationDeterministicExecutionSpec(
            threads=2,
            math_backend=CpuMathBackend.RUST_STD_F64,
        ),
        declared,
    )

    assert result.rows == _ROWS
    assert result.groups == _GROUPS
    assert result.partitions_sha256 == publication_partitions_sha256(prepared)
    assert tuple(item.variant for item in result.variants) == (
        PublicationVariant.FORECAST_YIELD,
        PublicationVariant.HISTORICAL_YIELD,
    )
    assert all(type(item.native) is AndersenLakeBatchResult for item in result.variants)


def test_empty_domain_partitions_are_skipped_without_changing_complete_output() -> None:
    """All-exact and all-general cohorts never emit an invalid empty native request."""
    declared = study()
    positive = schedule_frame().with_columns(pl.lit(0.01, dtype=pl.Float64).alias("risk_free_rate"))
    all_exact_source, all_exact_schedules = publication_inputs(positive)
    all_exact = prepare_publication_partitions(all_exact_source, all_exact_schedules, declared)
    assert all_exact.general.groups == 0
    exact_result = publication_pricing.run_publication_replication(
        all_exact,
        execution(),
        declared,
    )
    assert exact_result.rows == all_exact_source.rows

    negative = schedule_frame().with_columns(
        pl.lit(-0.001, dtype=pl.Float64).alias("risk_free_rate")
    )
    all_general_source, all_general_schedules = publication_inputs(negative)
    all_general = prepare_publication_partitions(
        all_general_source,
        all_general_schedules,
        declared,
    )
    assert all_general.exact.groups == 0
    general_result = publication_pricing.run_publication_replication(
        all_general,
        execution(),
        declared,
    )
    assert general_result.rows == all_general_source.rows
