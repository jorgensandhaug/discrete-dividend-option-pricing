"""Tests for the separate fail-fast publication estimator and cash-domain partition."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from finoptions import (
    CashDividendExerciseMesh,
    CpuExecutor,
    CpuMathBackend,
    DividendPolicy,
    ExerciseMesh,
    ExerciseTieRule,
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
from discrete_dividend_study.publication_groups import (
    PublicationCashDomain,
    prepare_publication_partitions,
    publication_partitions_sha256,
)
from discrete_dividend_study.publication_spec import (
    PublicationCashPartitionRule,
    PublicationEstimatorSpec,
    PublicationReplicationExecutionSpec,
    PublicationSpecValueError,
    PublicationStudySpec,
    PublicationVariant,
    publication_execution_sha256,
    publication_study_sha256,
)
from tests.test_classical_groups import grouped_frame

_ROWS = 7
_GROUPS = 3
_PATHS = 2_048
_SHA256_HEX_LENGTH = 64
_GENERAL_GROUPS = 2
_SPECIALIZED_MESH_CALLS = 2
_UNIFORM_MESH_CALLS = 6


def study() -> PublicationStudySpec:
    """Return a small explicit test identity, never a production budget default."""
    return PublicationStudySpec(
        name="publication-test-fixture",
        variants=tuple(PublicationVariant),
        estimator=PublicationEstimatorSpec(
            general_base_intervals=2,
            general_origin_exercise=OriginExercise.INCLUDED,
            training_paths=_PATHS,
            evaluation_paths=_PATHS,
            polynomial_degree=0,
            rank_tolerance=1.0e-12,
            condition_limit=1.0e8,
            tie_rule=ExerciseTieRule.STRICTLY_GREATER,
            insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
            cash_partition_rule=(PublicationCashPartitionRule.NONNEGATIVE_RATE_STRICTLY_PRE_EXPIRY),
        ),
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


def publication_source() -> PreparedClassicalStrikeGroups:
    """Return positive/pre-expiry, negative-rate, and terminal-cash natural groups."""
    base = grouped_frame().with_columns(
        pl.when(pl.col("ticker") == "BBB")
        .then(pl.lit(-0.001, dtype=pl.Float64))
        .otherwise(pl.col("risk_free_rate"))
        .alias("risk_free_rate")
    )
    terminal = (
        grouped_frame()
        .filter(pl.col("ticker") == "BBB")
        .with_columns(
            pl.lit("CCC").alias("ticker"),
            pl.lit(0.5, dtype=pl.Float64).alias("yte"),
            pl.lit(0.01, dtype=pl.Float64).alias("risk_free_rate"),
            pl.lit([0.5], dtype=pl.List(pl.Float64)).alias("forecasted_years_to_discrete_ex_dates"),
            pl.lit([0.75], dtype=pl.List(pl.Float64)).alias("forecasted_discrete_amounts"),
        )
    )
    frame = pl.concat((base, terminal))
    materialized = build_project_classical_grouped_frame(frame.lazy()).collect()
    return prepare_project_classical_strike_groups(materialized)


def test_publication_identity_has_no_implicit_budget_or_sparse_support() -> None:
    """The new study is separately hashed and forbids reconstruction-only continuation."""
    declared = study()
    assert len(publication_study_sha256(declared)) == _SHA256_HEX_LENGTH
    assert len(publication_execution_sha256(execution())) == _SHA256_HEX_LENGTH
    assert publication_study_sha256(declared) != publication_study_sha256(
        replace(
            declared,
            estimator=replace(declared.estimator, training_paths=_PATHS * 2),
        )
    )
    with pytest.raises(PublicationSpecValueError, match=r"OriginExercise\.INCLUDED"):
        replace(
            declared.estimator,
            general_origin_exercise=OriginExercise.EXCLUDED,
        )
    with pytest.raises(
        PublicationSpecValueError,
        match=r"InsufficientExerciseSupportPolicy\.FAIL",
    ):
        replace(
            declared.estimator,
            insufficient_exercise_support=InsufficientExerciseSupportPolicy.CONTINUE,
        )
    with pytest.raises(PublicationSpecValueError, match=r"DividendPolicy\.LIQUIDATOR"):
        replace(declared, exact_cash_policy=DividendPolicy.SURVIVOR)


def test_cash_domain_partition_is_complete_disjoint_and_source_ordered() -> None:
    """Negative rate and terminal cash enter only the named general partition."""
    prepared = prepare_publication_partitions(publication_source(), study())
    assert prepared.original_groups == _GROUPS
    assert prepared.original_rows == _ROWS
    assert prepared.simulation_units == _GROUPS
    assert prepared.specialized.domain is PublicationCashDomain.SPECIALIZED
    assert prepared.general.domain is PublicationCashDomain.GENERAL
    assert prepared.specialized.original_group_indices.tolist() == [0]
    assert prepared.general.original_group_indices.tolist() == [1, 2]
    assert prepared.specialized.original_row_indices.tolist() == [0, 1, 2]
    assert prepared.general.original_row_indices.tolist() == [3, 4, 5, 6]
    assert prepared.specialized.rates.tolist() == [0.03]
    assert prepared.general.rates.tolist() == [-0.001, 0.01]
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
        for array in arrays:
            assert not array.flags.writeable


def test_publication_replication_dispatches_explicit_meshes_and_merges_original_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cash specialization never receives negative/terminal rows and never becomes a fallback."""
    prepared = prepare_publication_partitions(publication_source(), study())
    observed: list[tuple[type[object], int]] = []
    native_price = publication_pricing.price_lsm_call_grouped_batch

    def recording_price(
        *,
        groups: LsmStrikeGroups,
        mesh: ExerciseMesh,
        executor: CpuExecutor,
        method: Lsm,
    ) -> LsmBatchResult:
        observed.append((type(mesh), method.training_stream.domain))
        return native_price(groups=groups, mesh=mesh, executor=executor, method=method)

    monkeypatch.setattr(publication_pricing, "price_lsm_call_grouped_batch", recording_price)
    result = publication_pricing.run_publication_replication(prepared, execution(), study())

    assert result.rows == _ROWS
    assert result.specialized_groups == 1
    assert result.general_groups == _GENERAL_GROUPS
    assert tuple(item.variant for item in result.variants) == tuple(PublicationVariant)
    assert [mesh for mesh, _ in observed].count(CashDividendExerciseMesh) == _SPECIALIZED_MESH_CALLS
    assert [mesh for mesh, _ in observed].count(UniformExerciseMesh) == _UNIFORM_MESH_CALLS
    assert [domain for _, domain in observed] == [100, 101] * len(PublicationVariant)
    for variant in result.variants:
        assert variant.native.rows == _ROWS
        assert variant.metrics.observations == _ROWS
        assert np.isfinite(variant.native.prices).all()
        assert np.all(variant.native.prices >= 0.0)
        assert not variant.native.prices.flags.writeable
        assert np.array_equal(
            variant.native.early_exercise_paths.astype(np.uint64)
            + variant.native.expiry_paths.astype(np.uint64),
            np.full(_ROWS, _PATHS, dtype=np.uint64),
        )


def test_empty_domain_partitions_are_skipped_without_changing_complete_output() -> None:
    """All-specialized and all-general cohorts never emit an invalid empty native request."""
    declared = study()
    all_specialized_source = prepare_project_classical_strike_groups(grouped_frame())
    all_specialized = prepare_publication_partitions(all_specialized_source, declared)
    assert all_specialized.general.groups == 0
    specialized_result = publication_pricing.run_publication_replication(
        all_specialized,
        execution(),
        declared,
    )
    assert specialized_result.rows == all_specialized_source.rows

    all_general_frame = grouped_frame().with_columns(
        pl.lit(-0.001, dtype=pl.Float64).alias("risk_free_rate")
    )
    all_general_source = prepare_project_classical_strike_groups(all_general_frame)
    all_general = prepare_publication_partitions(all_general_source, declared)
    assert all_general.specialized.groups == 0
    general_result = publication_pricing.run_publication_replication(
        all_general,
        execution(),
        declared,
    )
    assert general_result.rows == all_general_source.rows
