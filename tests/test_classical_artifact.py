"""Tests for complete content-addressed classical study artifacts."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from finoptions import CpuMathBackend

from discrete_dividend_study.classical_artifact import (
    ClassicalArtifactError,
    load_classical_study_artifact,
    run_and_write_project_classical_study,
)
from discrete_dividend_study.classical_inputs import prepare_project_classical_inputs
from discrete_dividend_study.classical_spec import (
    ClassicalExecutionSpec,
    ClassicalPopulationAddressing,
    ClassicalVariant,
    classical_execution_sha256,
    classical_study_sha256,
)
from discrete_dividend_study.neural_run import StudySourceIdentity
from tests.test_classical_inputs import classical_frame

if TYPE_CHECKING:
    from pathlib import Path

_ARTIFACT_FORMAT_VERSION = 4
_SOURCE = StudySourceIdentity(
    feature_artifact_sha256="1" * 64,
    feature_schema_sha256="2" * 64,
    paper_revision="a" * 40,
    engine_revision="b" * 40,
    dependency_lock_sha256="3" * 64,
)


def _execution() -> ClassicalExecutionSpec:
    return ClassicalExecutionSpec(
        threads=2,
        math_backend=CpuMathBackend.RUST_STD_F64,
        training_seed=7,
        training_domain_start=100,
        evaluation_seed=11,
        evaluation_domain_start=10_000,
    )


def test_complete_artifact_round_trip_is_exact_relocatable_and_idempotent(tmp_path: Path) -> None:
    """One file binds all four native outputs, metrics, source, code, and execution identities."""
    inputs = prepare_project_classical_inputs(classical_frame())
    execution = _execution()
    first = run_and_write_project_classical_study(tmp_path, _SOURCE, inputs, execution)
    second = run_and_write_project_classical_study(tmp_path, _SOURCE, inputs, execution)
    assert first.path == second.path
    assert first.sha256 == first.path.stem
    assert first.manifest.source == _SOURCE
    assert first.manifest.study_sha256 == classical_study_sha256(inputs.study)
    assert first.manifest.execution_sha256 == classical_execution_sha256(execution)
    assert first.manifest.format_version == _ARTIFACT_FORMAT_VERSION
    assert first.manifest.observations == inputs.rows
    assert first.manifest.simulation_units == inputs.rows
    assert (
        first.manifest.execution.population_addressing
        is ClassicalPopulationAddressing.CONTRACT_ROWS
    )
    assert first.manifest.execution.math_backend is CpuMathBackend.RUST_STD_F64
    assert tuple(first.manifest.variants) == tuple(variant.value for variant in ClassicalVariant)
    assert str(tmp_path).encode() not in first.path.read_bytes()

    loaded = load_classical_study_artifact(first.path)
    np.testing.assert_array_equal(loaded.target, inputs.targets)
    assert not loaded.target.flags.writeable
    for variant in ClassicalVariant:
        observed = loaded.result.variant(variant)
        assert observed.metrics.observations == inputs.rows
        assert np.isfinite(observed.native.prices).all()
        assert not observed.native.prices.flags.writeable

    apple_execution = replace(
        execution,
        math_backend=CpuMathBackend.APPLE_ACCELERATE,
    )
    assert classical_execution_sha256(apple_execution) != classical_execution_sha256(execution)
    grouped_execution = replace(
        execution,
        population_addressing=ClassicalPopulationAddressing.NATURAL_STRIKE_GROUPS,
    )
    assert classical_execution_sha256(grouped_execution) != classical_execution_sha256(execution)


def test_artifact_filename_and_bytes_must_share_one_content_hash(tmp_path: Path) -> None:
    """Renaming valid bytes cannot manufacture a second artifact identity."""
    inputs = prepare_project_classical_inputs(classical_frame())
    artifact = run_and_write_project_classical_study(tmp_path, _SOURCE, inputs, _execution())
    renamed = tmp_path / f"{'0' * 64}.safetensors"
    renamed.write_bytes(artifact.path.read_bytes())
    with pytest.raises(ClassicalArtifactError, match="filename hash"):
        load_classical_study_artifact(renamed)
