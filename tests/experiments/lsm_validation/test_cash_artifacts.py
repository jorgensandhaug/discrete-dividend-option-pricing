"""Tests for immutable, exactly resumable publication cash replications."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from finoptions import CpuMathBackend

from discrete_dividend_study.experiments.lsm_validation import (
    cash_artifacts as publication_artifact,
)
from discrete_dividend_study.experiments.lsm_validation.cash_artifacts import (
    PublicationArtifactError,
    load_publication_cash_artifact,
    load_publication_cash_request,
    publication_cash_request_sha256,
    run_or_load_publication_cash_replication,
)
from discrete_dividend_study.experiments.lsm_validation.partitions import (
    prepare_publication_partitions,
)
from discrete_dividend_study.experiments.lsm_validation.specification import (
    PublicationReplicationPlanSpec,
    PublicationVariant,
    publication_execution_sha256,
    publication_replication_plan_sha256,
    publication_study_sha256,
)
from discrete_dividend_study.experiments.neural.run import StudySourceIdentity
from tests.experiments.lsm_validation.test_pricing import publication_inputs, study

if TYPE_CHECKING:
    from pathlib import Path

    from discrete_dividend_study.experiments.lsm_validation.pricing import (
        PublicationCashReplicationResult,
    )

_SOURCE = StudySourceIdentity(
    feature_artifact_sha256="1" * 64,
    feature_schema_sha256="2" * 64,
    paper_revision="a" * 40,
    engine_revision="b" * 40,
    dependency_lock_sha256="3" * 64,
)
_FORMAT_VERSION = 1


def _plan() -> PublicationReplicationPlanSpec:
    return PublicationReplicationPlanSpec(
        replications=2,
        seed_root=7,
        seed_increment=11,
        training_domain_start=100,
        evaluation_domain_start=10_000,
        confidence_level=0.95,
        replication_critical_value=1.96,
    )


def test_cash_artifact_round_trip_is_exact_content_addressed_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified request is loaded without running the stochastic estimator twice."""
    declared = study()
    source, schedules = publication_inputs()
    inputs = prepare_publication_partitions(source, schedules, declared)
    plan = _plan()
    execution = plan.execution(
        0,
        threads=2,
        math_backend=CpuMathBackend.RUST_STD_F64,
    )
    request_sha256 = publication_cash_request_sha256(
        _SOURCE,
        inputs,
        execution,
        study=declared,
        replication_plan=plan,
    )
    first = run_or_load_publication_cash_replication(
        tmp_path,
        _SOURCE,
        inputs,
        execution,
        study=declared,
        replication_plan=plan,
    )

    def forbidden_replication(*_: object, **__: object) -> PublicationCashReplicationResult:
        pytest.fail("verified request receipt was not resumed")

    monkeypatch.setattr(
        publication_artifact,
        "run_publication_cash_replication",
        forbidden_replication,
    )
    second = run_or_load_publication_cash_replication(
        tmp_path,
        _SOURCE,
        inputs,
        execution,
        study=declared,
        replication_plan=plan,
    )

    assert second.artifact.path == first.artifact.path
    assert first.artifact.path.name == f"{first.artifact.sha256}.safetensors"
    assert first.artifact.manifest.format_version == _FORMAT_VERSION
    assert first.artifact.manifest.request_sha256 == request_sha256
    assert first.artifact.manifest.source == _SOURCE
    assert first.artifact.manifest.study_sha256 == publication_study_sha256(declared)
    assert first.artifact.manifest.replication_plan_sha256 == (
        publication_replication_plan_sha256(plan)
    )
    assert first.artifact.manifest.execution_sha256 == publication_execution_sha256(execution)
    assert first.artifact.manifest.observations == inputs.original_rows
    assert first.artifact.manifest.simulation_units == inputs.simulation_units
    assert first.artifact.manifest.exact_groups == inputs.exact.groups
    assert first.artifact.manifest.general_groups == inputs.general.groups
    assert (tmp_path / "cash-requests" / f"{request_sha256}.json").is_file()
    assert str(tmp_path).encode() not in first.artifact.path.read_bytes()
    np.testing.assert_array_equal(first.target, inputs.targets)
    assert not first.target.flags.writeable
    for variant in (
        PublicationVariant.CASH_JUMP_LIQUIDATOR,
        PublicationVariant.ESCROWED_CASH_SPOT_MATCHED,
    ):
        observed = first.result.variant(variant)
        assert observed.metrics.observations == inputs.original_rows
        assert np.isfinite(observed.native.prices).all()
        assert not observed.native.prices.flags.writeable


def test_artifact_and_receipt_reject_changed_or_corrupt_identity(tmp_path: Path) -> None:
    """A renamed artifact, altered plan, or malformed receipt fails instead of recomputing."""
    declared = study()
    source, schedules = publication_inputs()
    inputs = prepare_publication_partitions(source, schedules, declared)
    plan = _plan()
    execution = plan.execution(
        0,
        threads=2,
        math_backend=CpuMathBackend.RUST_STD_F64,
    )
    loaded = run_or_load_publication_cash_replication(
        tmp_path,
        _SOURCE,
        inputs,
        execution,
        study=declared,
        replication_plan=plan,
    )
    renamed = tmp_path / f"{'0' * 64}.safetensors"
    renamed.write_bytes(loaded.artifact.path.read_bytes())
    with pytest.raises(PublicationArtifactError, match="filename hash"):
        load_publication_cash_artifact(
            renamed,
            study=declared,
            replication_plan=plan,
        )
    with pytest.raises(PublicationArtifactError, match="replication-plan identities"):
        load_publication_cash_artifact(
            loaded.artifact.path,
            study=declared,
            replication_plan=replace(plan, replication_critical_value=2.0),
        )

    request_sha256 = loaded.artifact.manifest.request_sha256
    receipt = tmp_path / "cash-requests" / f"{request_sha256}.json"
    receipt.write_text("{}\n", encoding="ascii")
    with pytest.raises(PublicationArtifactError, match="keys are missing"):
        load_publication_cash_request(
            tmp_path,
            request_sha256,
            study=declared,
            replication_plan=plan,
        )
