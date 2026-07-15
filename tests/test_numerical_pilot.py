"""Tests for the target-free deterministic/LSM publication pilot."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import pytest
from finoptions import CpuMathBackend

from discrete_dividend_study import numerical_pilot
from discrete_dividend_study.numerical_pilot import (
    LsmPilotLevel,
    NumericalPilotError,
    NumericalPilotExecution,
    NumericalPilotTolerances,
    NumericalPilotValueError,
    PilotCashDividend,
    QuadraturePilotLevel,
    numerical_pilot_spec_sha256,
    project_publication_numerical_pilot_v1,
    project_publication_numerical_pilot_v2,
    project_publication_numerical_pilot_v3,
    run_numerical_pilot,
    write_numerical_pilot_receipt,
)

if TYPE_CHECKING:
    from pathlib import Path

_REVISION_A = "0" * 40
_REVISION_B = "1" * 40
_SHA256_LENGTH = 64


def small_spec() -> numerical_pilot.NumericalPilotSpec:
    """Return a fast numerical contract with deliberately broad test-only gates."""
    declared = project_publication_numerical_pilot_v1()
    return replace(
        declared,
        name="numerical-pilot-test",
        cases=declared.cases[:1],
        quadrature_levels=(
            QuadraturePilotLevel("q1", 300.0, 1_001, 32),
            QuadraturePilotLevel("q2", 400.0, 2_001, 64),
        ),
        general_mesh_intervals=(2, 4),
        lsm_levels=(
            LsmPilotLevel("p1", 512, 512),
            LsmPilotLevel("p2", 2_048, 2_048),
        ),
        replications=3,
        replication_critical_value=4.0,
        polynomial_degree=0,
        tolerances=NumericalPilotTolerances(
            deterministic_absolute=1.0,
            deterministic_relative=1.0,
            maximum_boundary_gap=1.0,
            mesh_absolute=1.0,
            mesh_relative=1.0,
            lsm_bias_absolute=1.0,
            lsm_bias_relative=1.0,
            lsm_path_drift_absolute=1.0,
            lsm_path_drift_relative=1.0,
            lsm_native_standard_error_absolute=1.0,
            lsm_native_standard_error_relative=1.0,
        ),
    )


def test_project_pilot_is_target_free_and_hashes_every_declared_sequence() -> None:
    """The frozen panel contains no price targets and every refinement changes identity."""
    declared = project_publication_numerical_pilot_v1()
    payload = json.dumps(asdict(declared))
    assert "target_price" not in payload
    assert "market_price" not in payload
    identity = numerical_pilot_spec_sha256(declared)
    assert len(identity) == _SHA256_LENGTH
    assert identity == "e8680f30b61c68b170ec5080f8bbc98aac1a9571f17b612a187ecc075feff2fa"
    assert identity != numerical_pilot_spec_sha256(
        replace(declared, general_mesh_intervals=(*declared.general_mesh_intervals, 256))
    )


def test_v2_changes_only_the_failed_general_mesh_dimension() -> None:
    """V2 retains every v1 identity except its name and appended 256-interval mesh."""
    v1 = project_publication_numerical_pilot_v1()
    v2 = project_publication_numerical_pilot_v2()
    assert v2 == replace(
        v1,
        name="project-publication-target-free-v2",
        general_mesh_intervals=(*v1.general_mesh_intervals, 256),
    )
    assert (
        numerical_pilot_spec_sha256(v2)
        == "4f04f0530dcb1df484492ed9ae56ea658dc18811c8d50d06513ad1e1be71fc73"
    )
    assert numerical_pilot_spec_sha256(v2) != numerical_pilot_spec_sha256(v1)


def test_v3_changes_only_the_failed_lsm_path_dimension() -> None:
    """V3 retains v2 and appends one geometric path refinement."""
    v2 = project_publication_numerical_pilot_v2()
    v3 = project_publication_numerical_pilot_v3()
    assert v3 == replace(
        v2,
        name="project-publication-target-free-v3",
        lsm_levels=(
            *v2.lsm_levels,
            LsmPilotLevel(name="p4", training_paths=131_072, evaluation_paths=131_072),
        ),
    )
    assert (
        numerical_pilot_spec_sha256(v3)
        == "6f7843bde65ec6629027b3f98b5f6f113f717be92697828467f9ea49c0d4cf27"
    )
    assert numerical_pilot_spec_sha256(v3) != numerical_pilot_spec_sha256(v2)


def test_cash_event_theorem_domain_rejects_terminal_cash() -> None:
    """A theorem-specialized case cannot silently enter a terminal-cash approximation."""
    case = project_publication_numerical_pilot_v1().cases[0]
    with pytest.raises(NumericalPilotValueError, match="strictly pre-expiry"):
        replace(
            case,
            cash_dividends=(PilotCashDividend(time=case.maturity, amount=1.0),),
        )


def test_schedule_builders_preserve_specialized_and_general_event_order() -> None:
    """Specialized and uniform cases expose exactly their declared pre-cash decisions."""
    declared = project_publication_numerical_pilot_v1()
    specialized = numerical_pilot._event_schedule(  # noqa: SLF001 - verify exact internal mesh
        declared.cases[0], base_intervals=None
    )
    assert specialized.times == (0.0, 0.5, 1.0)
    assert specialized.exercise_nodes == (1,)
    assert specialized.dividends[0].node == 1

    terminal = next(case for case in declared.cases if case.name == "terminal_cash_atm")
    general = numerical_pilot._event_schedule(  # noqa: SLF001 - verify exact internal mesh
        terminal, base_intervals=4
    )
    assert general.times == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert general.exercise_nodes == (0, 1, 2, 3)
    assert general.dividends[0].node == general.expiry_node


def test_small_pilot_retains_all_levels_replications_and_random_identities() -> None:
    """A successful receipt retains complete nested paths and independent role domains."""
    declared = small_spec()
    receipt = run_numerical_pilot(
        declared,
        NumericalPilotExecution(threads=2, math_backend=CpuMathBackend.RUST_STD_F64),
        paper_revision=_REVISION_A,
        engine_revision=_REVISION_B,
    )
    assert receipt.passed
    assert len(receipt.cases) == 1
    result = receipt.cases[0]
    assert len(result.deterministic) == len(declared.quadrature_levels)
    assert result.mesh == ()
    assert len(result.lsm) == len(declared.lsm_levels)
    assert all(len(level.replications) == declared.replications for level in result.lsm)
    for prior, finer in zip(result.lsm[0].replications, result.lsm[1].replications, strict=True):
        assert prior.seed == finer.seed
        assert prior.training_domain == finer.training_domain
        assert prior.evaluation_domain == finer.evaluation_domain
        assert prior.training_domain != prior.evaluation_domain
    assert all(gate.passed for gate in result.gates)


def test_receipt_writer_is_atomic_and_never_overwrites(
    tmp_path: Path,
) -> None:
    """A completed JSON receipt is canonical and immutable at its declared path."""
    receipt = run_numerical_pilot(
        small_spec(),
        NumericalPilotExecution(threads=1, math_backend=CpuMathBackend.RUST_STD_F64),
        paper_revision=_REVISION_A,
        engine_revision=_REVISION_B,
    )
    path = tmp_path / "pilot.json"
    write_numerical_pilot_receipt(path, receipt)
    payload = json.loads(path.read_text())
    assert payload["format"] == "publication-numerical-pilot-v1"
    assert payload["passed"] is True
    with pytest.raises(NumericalPilotError, match="already exists"):
        write_numerical_pilot_receipt(path, receipt)
