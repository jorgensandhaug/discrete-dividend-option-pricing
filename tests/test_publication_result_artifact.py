"""Tests for the complete four-variant publication result artifact."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from finoptions import CpuMathBackend

from discrete_dividend_study import publication_result_artifact
from discrete_dividend_study.publication_artifact import (
    load_publication_cash_artifact,
    run_or_load_publication_cash_replication,
)
from discrete_dividend_study.publication_groups import prepare_publication_partitions
from discrete_dividend_study.publication_result_artifact import (
    PublicationResultArtifactError,
    load_publication_study_result_artifact,
    run_or_load_publication_study_result,
)
from discrete_dividend_study.publication_spec import (
    PublicationDeterministicExecutionSpec,
    PublicationVariant,
)
from tests.test_publication_artifact import _SOURCE, _plan
from tests.test_publication_estimator import publication_inputs, study

if TYPE_CHECKING:
    from pathlib import Path

    from discrete_dividend_study.publication_pricing import PublicationYieldPricingResult


def test_complete_result_streams_cash_replications_prices_yields_and_exactly_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final artifact binds all prerequisites and preserves numerical uncertainty."""
    declared = study()
    plan = _plan()
    source, schedules = publication_inputs()
    inputs = prepare_publication_partitions(source, schedules, declared)
    cash_paths = tuple(
        run_or_load_publication_cash_replication(
            tmp_path,
            _SOURCE,
            inputs,
            plan.execution(
                replication,
                threads=2,
                math_backend=CpuMathBackend.RUST_STD_F64,
            ),
            study=declared,
            replication_plan=plan,
        ).artifact.path
        for replication in range(plan.replications)
    )
    deterministic_execution = PublicationDeterministicExecutionSpec(
        threads=2,
        math_backend=CpuMathBackend.RUST_STD_F64,
    )
    first = run_or_load_publication_study_result(
        tmp_path,
        _SOURCE,
        inputs,
        cash_paths,
        deterministic_execution,
        study=declared,
        replication_plan=plan,
    )

    assert first.artifact.path.name == f"{first.artifact.sha256}.safetensors"
    assert first.artifact.manifest.source == _SOURCE
    assert first.artifact.manifest.cash_artifact_sha256s == tuple(path.stem for path in cash_paths)
    assert first.artifact.manifest.confidence_level == plan.confidence_level
    assert first.artifact.manifest.replication_critical_value == (plan.replication_critical_value)
    assert tuple(result.variant for result in first.result.cash) == (
        PublicationVariant.CASH_JUMP_LIQUIDATOR,
        PublicationVariant.ESCROWED_CASH_SPOT_MATCHED,
    )
    assert tuple(result.variant for result in first.result.yields.variants) == (
        PublicationVariant.FORECAST_YIELD,
        PublicationVariant.HISTORICAL_YIELD,
    )
    np.testing.assert_array_equal(first.target, inputs.targets)

    raw = tuple(
        load_publication_cash_artifact(path, study=declared, replication_plan=plan)
        for path in cash_paths
    )
    for aggregate in first.result.cash:
        prices = tuple(item.result.variant(aggregate.variant).native.prices for item in raw)
        np.testing.assert_array_equal(
            aggregate.mean_prices,
            np.mean(np.stack(prices), axis=0),
        )
        np.testing.assert_allclose(
            aggregate.replication_sample_standard_deviations,
            np.std(np.stack(prices), axis=0, ddof=1),
            rtol=1.0e-14,
            atol=0.0,
        )
        for summary in aggregate.metric_summaries:
            values = tuple(
                float(getattr(item.result.variant(aggregate.variant).metrics, summary.metric))
                for item in raw
            )
            expected_mean = math.fsum(values) / len(values)
            assert summary.estimate.mean == expected_mean
            assert summary.estimate.replications == plan.replications
            assert summary.estimate.interval_lower <= expected_mean
            assert summary.estimate.interval_upper >= expected_mean

    loaded = load_publication_study_result_artifact(
        first.artifact.path,
        study=declared,
        replication_plan=plan,
    )
    assert loaded.artifact == first.artifact

    def forbidden_yields(*_: object, **__: object) -> PublicationYieldPricingResult:
        pytest.fail("verified final request reran deterministic pricing")

    monkeypatch.setattr(
        publication_result_artifact,
        "run_publication_yield_pricing",
        forbidden_yields,
    )
    second = run_or_load_publication_study_result(
        tmp_path,
        _SOURCE,
        inputs,
        cash_paths,
        deterministic_execution,
        study=declared,
        replication_plan=plan,
    )
    assert second.artifact == first.artifact


def test_final_result_rejects_reordered_replication_prerequisites(tmp_path: Path) -> None:
    """Replication order is part of the paired numerical experiment, never inferred."""
    declared = study()
    plan = _plan()
    source, schedules = publication_inputs()
    inputs = prepare_publication_partitions(source, schedules, declared)
    paths = tuple(
        run_or_load_publication_cash_replication(
            tmp_path,
            _SOURCE,
            inputs,
            plan.execution(
                replication,
                threads=2,
                math_backend=CpuMathBackend.RUST_STD_F64,
            ),
            study=declared,
            replication_plan=plan,
        ).artifact.path
        for replication in range(plan.replications)
    )
    with pytest.raises(PublicationResultArtifactError, match="prerequisite identity"):
        run_or_load_publication_study_result(
            tmp_path,
            _SOURCE,
            inputs,
            tuple(reversed(paths)),
            PublicationDeterministicExecutionSpec(
                threads=2,
                math_backend=CpuMathBackend.RUST_STD_F64,
            ),
            study=declared,
            replication_plan=plan,
        )
