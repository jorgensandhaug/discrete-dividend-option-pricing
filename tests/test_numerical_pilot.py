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
    project_publication_numerical_pilot_v4,
    project_publication_numerical_pilot_v5,
    run_numerical_pilot,
    terminal_cash_compact_predecessor_diagnostic_v1,
    terminal_cash_polynomial_degree_diagnostic_v1,
    terminal_cash_predecessor_diagnostic_v1,
    write_numerical_pilot_receipt,
)

if TYPE_CHECKING:
    from pathlib import Path

_REVISION_A = "0" * 40
_REVISION_B = "1" * 40
_SHA256_LENGTH = 64
_POLYNOMIAL_DEGREES = 5
_TERMINAL_DEGREE_HASHES = (
    "77bffc33ae67049cf764429e9cb1a1ae07c0ad7781334cd8b00fc6df2b9f7ad2",
    "b0e643d387d98bf07c149de6090567ead88eee4cea7fc4b3d76612539d5313f4",
    "17408240538036511a07a1e867f392fb961639e7d881abac799072175a4dc746",
    "33515f847f52dd19c3dbb7f1f9650b3f5fba6024bd035a16a044434ce696d2e6",
    "beba98e74c2392ab0775b05842c1aa2b818701da956a01abf46f570d5d29fdfa",
)


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


def test_v4_changes_only_the_terminal_cash_exercise_domain() -> None:
    """V4 applies the passing predecessor domain without changing any other identity."""
    v3 = project_publication_numerical_pilot_v3()
    v4 = project_publication_numerical_pilot_v4()
    expected_cases = tuple(
        replace(
            case,
            exercise_domain=numerical_pilot.PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR,
        )
        if case.name == "terminal_cash_atm"
        else case
        for case in v3.cases
    )
    assert v4 == replace(v3, name="project-publication-target-free-v4", cases=expected_cases)
    assert (
        numerical_pilot_spec_sha256(v4)
        == "344670190d05856d90e0ab3b2d9a084a4de32cb1c3e68bc224d943edc21e0c32"
    )
    assert numerical_pilot_spec_sha256(v4) != numerical_pilot_spec_sha256(v3)


def test_v5_changes_only_the_terminal_cash_schedule_representation() -> None:
    """V5 adopts the passing compact schedule without changing the other seven cases."""
    v4 = project_publication_numerical_pilot_v4()
    v5 = project_publication_numerical_pilot_v5()
    expected_cases = tuple(
        replace(
            case,
            exercise_domain=(numerical_pilot.PilotExerciseDomain.COMPACT_TERMINAL_CASH_PREDECESSOR),
        )
        if case.name == "terminal_cash_atm"
        else case
        for case in v4.cases
    )
    assert v5 == replace(v4, name="project-publication-target-free-v5", cases=expected_cases)
    assert (
        numerical_pilot_spec_sha256(v5)
        == "8dff50a6cef9c93271c745f2f6cb784552516d739d3e109884b451378493ce64"
    )
    assert numerical_pilot_spec_sha256(v5) != numerical_pilot_spec_sha256(v4)


def test_terminal_degree_diagnostic_preserves_v3_paths_schedule_and_case_domain() -> None:
    """All five degrees use identical terminal-case p3/p4 streams and numerical controls."""
    v3 = project_publication_numerical_pilot_v3()
    terminal_index = next(
        index for index, case in enumerate(v3.cases) if case.name == "terminal_cash_atm"
    )
    diagnostics = tuple(
        terminal_cash_polynomial_degree_diagnostic_v1(degree)
        for degree in range(_POLYNOMIAL_DEGREES)
    )
    for degree, diagnostic in enumerate(diagnostics):
        assert diagnostic == replace(
            v3,
            name=f"terminal-cash-polynomial-degree-{degree}-diagnostic-v1",
            cases=(v3.cases[terminal_index],),
            lsm_levels=v3.lsm_levels[-2:],
            random_domain_root=v3.random_domain_root + (2 * terminal_index),
            polynomial_degree=degree,
        )
    assert (
        len({numerical_pilot_spec_sha256(diagnostic) for diagnostic in diagnostics})
        == _POLYNOMIAL_DEGREES
    )
    assert tuple(numerical_pilot_spec_sha256(diagnostic) for diagnostic in diagnostics) == (
        _TERMINAL_DEGREE_HASHES
    )
    with pytest.raises(NumericalPilotValueError, match=r"\[0, 4\]"):
        terminal_cash_polynomial_degree_diagnostic_v1(5)


def test_terminal_predecessor_diagnostic_changes_only_the_exercise_domain() -> None:
    """The policy diagnostic keeps degree-four paths and makes one schedule change."""
    degree_four = terminal_cash_polynomial_degree_diagnostic_v1(4)
    diagnostic = terminal_cash_predecessor_diagnostic_v1()
    assert diagnostic == replace(
        degree_four,
        name="terminal-cash-predecessor-diagnostic-v1",
        cases=(
            replace(
                degree_four.cases[0],
                exercise_domain=numerical_pilot.PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR,
            ),
        ),
    )
    assert (
        numerical_pilot_spec_sha256(diagnostic)
        == "f4700ffe31903ff810b2e773efe286b2f2850e6ec316ac469b471a57480e494d"
    )
    assert numerical_pilot_spec_sha256(diagnostic) != numerical_pilot_spec_sha256(degree_four)


def test_compact_predecessor_diagnostic_changes_only_the_schedule_domain() -> None:
    """The compact diagnostic preserves all gates and removes only transition-only nodes."""
    predecessor = terminal_cash_predecessor_diagnostic_v1()
    diagnostic = terminal_cash_compact_predecessor_diagnostic_v1()
    assert diagnostic == replace(
        predecessor,
        name="terminal-cash-compact-predecessor-diagnostic-v1",
        cases=(
            replace(
                predecessor.cases[0],
                exercise_domain=(
                    numerical_pilot.PilotExerciseDomain.COMPACT_TERMINAL_CASH_PREDECESSOR
                ),
            ),
        ),
    )
    assert (
        numerical_pilot_spec_sha256(diagnostic)
        == "e2aa1fd264a7521070fa7d693cacb668da4c2f4d486dae0ef146e6a4b3d9ae4f"
    )
    assert numerical_pilot_spec_sha256(diagnostic) != numerical_pilot_spec_sha256(predecessor)


def test_cash_event_theorem_domain_rejects_terminal_cash() -> None:
    """A theorem-specialized case cannot silently enter a terminal-cash approximation."""
    case = project_publication_numerical_pilot_v1().cases[0]
    with pytest.raises(NumericalPilotValueError, match="strictly pre-expiry"):
        replace(
            case,
            cash_dividends=(PilotCashDividend(time=case.maturity, amount=1.0),),
        )


def test_terminal_predecessor_domain_requires_terminal_cash() -> None:
    """The terminal specialization cannot be selected for a merely pre-expiry schedule."""
    case = project_publication_numerical_pilot_v1().cases[0]
    with pytest.raises(NumericalPilotValueError, match="final event exactly at maturity"):
        replace(
            case,
            exercise_domain=numerical_pilot.PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR,
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

    predecessor = numerical_pilot._event_schedule(  # noqa: SLF001 - exact diagnostic identity
        replace(
            terminal,
            exercise_domain=numerical_pilot.PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR,
        ),
        base_intervals=4,
    )
    assert predecessor.times == general.times
    assert predecessor.dividends == general.dividends
    assert predecessor.exercise_nodes == (3,)

    compact = numerical_pilot._event_schedule(  # noqa: SLF001 - exact diagnostic identity
        replace(
            terminal,
            exercise_domain=(numerical_pilot.PilotExerciseDomain.COMPACT_TERMINAL_CASH_PREDECESSOR),
        ),
        base_intervals=4,
    )
    assert compact.times == (0.0, 0.75, 1.0)
    assert compact.dividends[0].node == compact.expiry_node
    assert compact.dividends[0].amount == general.dividends[0].amount
    assert compact.exercise_nodes == (1,)


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
