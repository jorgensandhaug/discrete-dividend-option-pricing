"""Target-free numerical pilot for the publication estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final

from finoptions import (
    CashDividend,
    ConstantVolatility,
    CpuExecutor,
    CpuMathBackend,
    CumulativeDividendYieldCurve,
    DividendPolicy,
    EventAlignedQuadrature,
    EventSchedule,
    ExerciseTieRule,
    InsufficientExerciseSupportPolicy,
    LogDiscountCurve,
    Lsm,
    NoInTheMoneyPathsDiagnostics,
    OriginDecisionDiagnostics,
    RandomStream,
    RegressionFitDiagnostics,
    build_info,
    price_event_aligned_call,
    price_lsm_call,
)

from .engine_identity import EngineIdentityError, resolve_project_engine_revision

if TYPE_CHECKING:
    from collections.abc import Sequence

_FORMAT: Final = "publication-numerical-pilot-v1"
_SPEC_HASH_FORMAT: Final = "publication-numerical-pilot-spec-v1"
_EXECUTION_HASH_FORMAT: Final = "publication-numerical-pilot-execution-v1"
_SUPPORTED_NORMAL_ORDERS: Final = (32, 64, 128, 256, 512, 1_024, 2_048)
_MAX_U32: Final = (2**32) - 1
_MAX_U64: Final = (2**64) - 1
_MAX_POLYNOMIAL_DEGREE: Final = 4
_MINIMUM_LEVELS: Final = 2
_MINIMUM_TERMINAL_PREDECESSOR_INTERVALS: Final = 2
_SEED_INCREMENT: Final = 0x9E3779B97F4A7C15
_SHA1_LENGTH: Final = 40


class NumericalPilotError(RuntimeError):
    """The declared pilot cannot be executed or recorded without changing its identity."""


@dataclass(slots=True)
class NumericalPilotValueError(ValueError):
    """One named pilot value violates its closed domain."""

    field: str
    observed: object
    requirement: str

    def __str__(self) -> str:
        """Describe the rejection without repairing it."""
        return f"{self.field} must be {self.requirement}; observed {self.observed!r}"


class PilotExerciseDomain(StrEnum):
    """The theorem domain that determines a case's finite exercise schedule."""

    CASH_EVENT_THEOREM = "cash_event_theorem"
    COMPACT_TERMINAL_CASH_PREDECESSOR = "compact_terminal_cash_predecessor"
    TERMINAL_CASH_PREDECESSOR = "terminal_cash_predecessor"
    UNIFORM_BERMUDAN = "uniform_bermudan"


@dataclass(frozen=True, slots=True)
class PilotCashDividend:
    """One exact deterministic cash event in year-fraction time."""

    time: float
    amount: float

    def __post_init__(self) -> None:
        """Require a finite positive time and amount."""
        _positive_float(self.time, field="cash dividend time")
        _positive_float(self.amount, field="cash dividend amount")


@dataclass(frozen=True, slots=True)
class NumericalPilotCase:
    """One synthetic financial case with no empirical price target."""

    name: str
    exercise_domain: PilotExerciseDomain
    initial_spot: float
    strike: float
    maturity: float
    rate: float
    continuous_dividend_yield: float
    volatility: float
    cash_dividends: tuple[PilotCashDividend, ...]

    def __post_init__(self) -> None:
        """Validate the financial domain and the specialized-call theorem boundary."""
        _nonempty_string(self.name, field="case name")
        if type(self.exercise_domain) is not PilotExerciseDomain:
            raise NumericalPilotValueError(
                field="exercise_domain",
                observed=self.exercise_domain,
                requirement="PilotExerciseDomain",
            )
        _positive_float(self.initial_spot, field="initial_spot")
        _positive_float(self.strike, field="strike")
        _positive_float(self.maturity, field="maturity")
        _finite_float(self.rate, field="rate")
        _canonical_nonnegative_float(
            self.continuous_dividend_yield,
            field="continuous_dividend_yield",
        )
        _positive_float(self.volatility, field="volatility")
        if type(self.cash_dividends) is not tuple or any(
            type(dividend) is not PilotCashDividend for dividend in self.cash_dividends
        ):
            raise NumericalPilotValueError(
                field="cash_dividends",
                observed=self.cash_dividends,
                requirement="tuple[PilotCashDividend, ...]",
            )
        previous = 0.0
        for dividend in self.cash_dividends:
            if dividend.time <= previous or dividend.time > self.maturity:
                raise NumericalPilotValueError(
                    field="cash_dividends",
                    observed=self.cash_dividends,
                    requirement="strictly increasing times in (0, maturity]",
                )
            previous = dividend.time
        _validate_case_exercise_domain(self)


def _validate_case_exercise_domain(case: NumericalPilotCase) -> None:
    """Validate the closed cash-event theorem variants."""
    if case.exercise_domain is PilotExerciseDomain.UNIFORM_BERMUDAN:
        return
    if not case.cash_dividends:
        raise NumericalPilotValueError(
            field="cash_dividends",
            observed=case.cash_dividends,
            requirement="at least one event for a cash-event theorem domain",
        )
    if case.rate < 0.0:
        raise NumericalPilotValueError(
            field="rate",
            observed=case.rate,
            requirement="nonnegative in a cash-event theorem domain",
        )
    if case.continuous_dividend_yield != 0.0:
        raise NumericalPilotValueError(
            field="continuous_dividend_yield",
            observed=case.continuous_dividend_yield,
            requirement="exactly zero in a cash-event theorem domain",
        )
    terminal_cash = case.cash_dividends[-1].time == case.maturity
    if case.exercise_domain is PilotExerciseDomain.CASH_EVENT_THEOREM and terminal_cash:
        raise NumericalPilotValueError(
            field="cash_dividends",
            observed=case.cash_dividends,
            requirement="strictly pre-expiry events in the cash-event theorem domain",
        )
    terminal_predecessor_domains = (
        PilotExerciseDomain.COMPACT_TERMINAL_CASH_PREDECESSOR,
        PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR,
    )
    if case.exercise_domain in terminal_predecessor_domains and not terminal_cash:
        raise NumericalPilotValueError(
            field="cash_dividends",
            observed=case.cash_dividends,
            requirement="a final event exactly at maturity in the terminal-predecessor domain",
        )


@dataclass(frozen=True, slots=True)
class QuadraturePilotLevel:
    """One completely explicit deterministic numerical level."""

    name: str
    upper_stock: float
    stock_points: int
    normal_order: int

    def __post_init__(self) -> None:
        """Require an admitted method identity without constructing it implicitly."""
        _nonempty_string(self.name, field="quadrature level name")
        _positive_float(self.upper_stock, field="upper_stock")
        _positive_int(self.stock_points, field="stock_points", minimum=1_001)
        if type(self.normal_order) is not int or self.normal_order not in _SUPPORTED_NORMAL_ORDERS:
            raise NumericalPilotValueError(
                field="normal_order",
                observed=self.normal_order,
                requirement=f"one of {_SUPPORTED_NORMAL_ORDERS}",
            )


@dataclass(frozen=True, slots=True)
class LsmPilotLevel:
    """One nested fitting/evaluation path budget."""

    name: str
    training_paths: int
    evaluation_paths: int

    def __post_init__(self) -> None:
        """Require explicit representable populations."""
        _nonempty_string(self.name, field="LSM level name")
        _positive_int(self.training_paths, field="training_paths")
        _positive_int(self.evaluation_paths, field="evaluation_paths", minimum=2)


@dataclass(frozen=True, slots=True)
class NumericalPilotTolerances:
    """Predeclared deterministic, mesh, and stochastic acceptance envelopes."""

    deterministic_absolute: float
    deterministic_relative: float
    maximum_boundary_gap: float
    mesh_absolute: float
    mesh_relative: float
    lsm_bias_absolute: float
    lsm_bias_relative: float
    lsm_path_drift_absolute: float
    lsm_path_drift_relative: float
    lsm_native_standard_error_absolute: float
    lsm_native_standard_error_relative: float

    def __post_init__(self) -> None:
        """Require finite canonical nonnegative tolerances."""
        for field, value in asdict(self).items():
            _canonical_nonnegative_float(value, field=field)


@dataclass(frozen=True, slots=True)
class NumericalPilotSpec:
    """Immutable scientific identity of a target-free numerical pilot."""

    name: str
    cases: tuple[NumericalPilotCase, ...]
    quadrature_levels: tuple[QuadraturePilotLevel, ...]
    general_mesh_intervals: tuple[int, ...]
    lsm_levels: tuple[LsmPilotLevel, ...]
    replications: int
    replication_critical_value: float
    seed_root: int
    random_domain_root: int
    polynomial_degree: int
    rank_tolerance: float
    condition_limit: float
    tie_rule: ExerciseTieRule
    insufficient_exercise_support: InsufficientExerciseSupportPolicy
    tolerances: NumericalPilotTolerances

    def __post_init__(self) -> None:
        """Reject incomplete sequences, aliases, and non-publication LSM semantics."""
        _nonempty_string(self.name, field="pilot name")
        _validate_pilot_sequences(self)
        _validate_pilot_estimator(self)
        _validate_pilot_ranges(self)


def _validate_pilot_sequences(spec: NumericalPilotSpec) -> None:
    """Validate complete ordered refinement sequences and unique case identities."""
    _typed_nonempty_tuple(spec.cases, NumericalPilotCase, field="cases")
    _typed_nonempty_tuple(
        spec.quadrature_levels,
        QuadraturePilotLevel,
        field="quadrature_levels",
        minimum=_MINIMUM_LEVELS,
    )
    _typed_nonempty_tuple(
        spec.lsm_levels,
        LsmPilotLevel,
        field="lsm_levels",
        minimum=_MINIMUM_LEVELS,
    )
    if len({case.name for case in spec.cases}) != len(spec.cases):
        raise NumericalPilotValueError(
            field="case names",
            observed=tuple(case.name for case in spec.cases),
            requirement="unique values",
        )
    _strictly_increasing_names(
        tuple(level.name for level in spec.quadrature_levels),
        field="quadrature level names",
    )
    _strictly_increasing_names(
        tuple(level.name for level in spec.lsm_levels),
        field="LSM level names",
    )
    _strictly_increasing_ints(
        tuple(level.stock_points for level in spec.quadrature_levels),
        field="quadrature stock_points",
    )
    _strictly_increasing_ints(
        tuple(level.normal_order for level in spec.quadrature_levels),
        field="quadrature normal_order",
    )
    _strictly_increasing_floats(
        tuple(level.upper_stock for level in spec.quadrature_levels),
        field="quadrature upper_stock",
    )
    _strictly_increasing_ints(
        tuple(level.training_paths for level in spec.lsm_levels),
        field="LSM training_paths",
    )
    _strictly_increasing_ints(
        tuple(level.evaluation_paths for level in spec.lsm_levels),
        field="LSM evaluation_paths",
    )
    if (
        type(spec.general_mesh_intervals) is not tuple
        or len(spec.general_mesh_intervals) < _MINIMUM_LEVELS
    ):
        raise NumericalPilotValueError(
            field="general_mesh_intervals",
            observed=spec.general_mesh_intervals,
            requirement="tuple of at least two increasing positive plain integers",
        )
    _strictly_increasing_ints(spec.general_mesh_intervals, field="general_mesh_intervals")


def _validate_pilot_estimator(spec: NumericalPilotSpec) -> None:
    """Validate the strict publication LSM estimator identity."""
    if (
        type(spec.polynomial_degree) is not int
        or not 0 <= spec.polynomial_degree <= _MAX_POLYNOMIAL_DEGREE
    ):
        raise NumericalPilotValueError(
            field="polynomial_degree",
            observed=spec.polynomial_degree,
            requirement=f"a plain integer in [0, {_MAX_POLYNOMIAL_DEGREE}]",
        )
    _positive_float(spec.rank_tolerance, field="rank_tolerance")
    if spec.rank_tolerance >= 1.0:
        raise NumericalPilotValueError(
            field="rank_tolerance",
            observed=spec.rank_tolerance,
            requirement="strictly between zero and one",
        )
    _positive_float(spec.condition_limit, field="condition_limit")
    if not 1.0 <= spec.condition_limit <= 1.0 / spec.rank_tolerance:
        raise NumericalPilotValueError(
            field="condition_limit",
            observed=spec.condition_limit,
            requirement="in [1, 1 / rank_tolerance]",
        )
    if spec.tie_rule is not ExerciseTieRule.STRICTLY_GREATER:
        raise NumericalPilotValueError(
            field="tie_rule",
            observed=spec.tie_rule,
            requirement="ExerciseTieRule.STRICTLY_GREATER",
        )
    if spec.insufficient_exercise_support is not InsufficientExerciseSupportPolicy.FAIL:
        raise NumericalPilotValueError(
            field="insufficient_exercise_support",
            observed=spec.insufficient_exercise_support,
            requirement="InsufficientExerciseSupportPolicy.FAIL",
        )


def _validate_pilot_ranges(spec: NumericalPilotSpec) -> None:
    """Validate replication addressing, tolerances, and common stock boundaries."""
    _positive_int(spec.replications, field="replications", minimum=3)
    _positive_float(spec.replication_critical_value, field="replication_critical_value")
    _bounded_int(spec.seed_root, field="seed_root", maximum=_MAX_U64)
    maximum_domain = spec.random_domain_root + (2 * len(spec.cases)) - 1
    _bounded_int(spec.random_domain_root, field="random_domain_root", maximum=_MAX_U32)
    if maximum_domain > _MAX_U32:
        raise NumericalPilotValueError(
            field="case random domains",
            observed=maximum_domain,
            requirement=f"values no greater than {_MAX_U32}",
        )
    if type(spec.tolerances) is not NumericalPilotTolerances:
        raise NumericalPilotValueError(
            field="tolerances",
            observed=spec.tolerances,
            requirement="NumericalPilotTolerances",
        )
    largest_required = max(
        *(case.initial_spot for case in spec.cases),
        *(case.strike for case in spec.cases),
        *(dividend.amount for case in spec.cases for dividend in case.cash_dividends),
    )
    if spec.quadrature_levels[0].upper_stock <= largest_required:
        raise NumericalPilotValueError(
            field="first quadrature upper_stock",
            observed=spec.quadrature_levels[0].upper_stock,
            requirement="strictly above every case spot, strike, and cash amount",
        )


@dataclass(frozen=True, slots=True)
class NumericalPilotExecution:
    """Resource and floating-point implementation for one pilot execution."""

    threads: int
    math_backend: CpuMathBackend

    def __post_init__(self) -> None:
        """Require an explicit native CPU execution identity."""
        _positive_int(self.threads, field="threads")
        if type(self.math_backend) is not CpuMathBackend:
            raise NumericalPilotValueError(
                field="math_backend",
                observed=self.math_backend,
                requirement="CpuMathBackend",
            )


@dataclass(frozen=True, slots=True)
class DeterministicPilotObservation:
    """One complete direct-quadrature observation."""

    level: str
    upper_stock: float
    stock_points: int
    normal_order: int
    price: float
    maximum_upper_boundary_gap: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class MeshPilotObservation:
    """One fine-quadrature value on a declared uniform Bermudan mesh."""

    base_intervals: int
    price: float
    maximum_upper_boundary_gap: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class LsmReplicationObservation:
    """One independently evaluated fitted-policy observation."""

    replication: int
    seed: int
    training_domain: int
    evaluation_domain: int
    price: float
    native_standard_error: float
    regression_nodes: int
    no_in_the_money_nodes: int
    minimum_in_the_money_training_paths: int | None
    maximum_regression_condition_number: float | None
    origin_exercise: bool | None
    early_exercise_paths: int
    expiry_paths: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class LsmLevelObservation:
    """Complete replication vector and its target-free summary."""

    level: str
    training_paths: int
    evaluation_paths: int
    mean_price: float
    replication_standard_deviation: float
    replication_standard_error: float
    mean_native_standard_error: float
    replications: tuple[LsmReplicationObservation, ...]


@dataclass(frozen=True, slots=True)
class NumericalPilotGate:
    """One named predeclared numerical envelope evaluation."""

    name: str
    observed: float
    limit: float
    passed: bool


@dataclass(frozen=True, slots=True)
class NumericalPilotCaseResult:
    """Complete retained numerical sequence for one synthetic case."""

    case: NumericalPilotCase
    reference_mesh_intervals: int | None
    deterministic: tuple[DeterministicPilotObservation, ...]
    mesh: tuple[MeshPilotObservation, ...]
    lsm: tuple[LsmLevelObservation, ...]
    gates: tuple[NumericalPilotGate, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class NumericalPilotReceipt:
    """Immutable output of one complete target-free pilot execution."""

    format: str
    spec_sha256: str
    execution_sha256: str
    paper_revision: str
    engine_revision: str
    package_version: str
    rustc_version: str
    python_version: str
    platform: str
    started_at_utc: str
    wall_seconds: float
    spec: NumericalPilotSpec
    execution: NumericalPilotExecution
    cases: tuple[NumericalPilotCaseResult, ...]
    passed: bool


def project_publication_numerical_pilot_v1() -> NumericalPilotSpec:
    """Return the first predeclared synthetic panel and numerical sequences."""
    cash = PilotExerciseDomain.CASH_EVENT_THEOREM
    general = PilotExerciseDomain.UNIFORM_BERMUDAN
    return NumericalPilotSpec(
        name="project-publication-target-free-v1",
        cases=(
            NumericalPilotCase(
                name="published_one_cash_atm",
                exercise_domain=cash,
                initial_spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=0.06,
                continuous_dividend_yield=0.0,
                volatility=0.30,
                cash_dividends=(PilotCashDividend(time=0.5, amount=7.0),),
            ),
            NumericalPilotCase(
                name="three_cash_events_atm",
                exercise_domain=cash,
                initial_spot=100.0,
                strike=100.0,
                maturity=1.5,
                rate=0.03,
                continuous_dividend_yield=0.0,
                volatility=0.25,
                cash_dividends=(
                    PilotCashDividend(time=0.25, amount=1.5),
                    PilotCashDividend(time=0.75, amount=1.5),
                    PilotCashDividend(time=1.25, amount=1.5),
                ),
            ),
            NumericalPilotCase(
                name="short_cash_call_itm",
                exercise_domain=cash,
                initial_spot=100.0,
                strike=95.0,
                maturity=0.25,
                rate=0.04,
                continuous_dividend_yield=0.0,
                volatility=0.35,
                cash_dividends=(PilotCashDividend(time=0.1, amount=2.0),),
            ),
            NumericalPilotCase(
                name="continuous_yield_atm",
                exercise_domain=general,
                initial_spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=0.02,
                continuous_dividend_yield=0.04,
                volatility=0.25,
                cash_dividends=(),
            ),
            NumericalPilotCase(
                name="negative_rate_cash_atm",
                exercise_domain=general,
                initial_spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=-0.01,
                continuous_dividend_yield=0.0,
                volatility=0.25,
                cash_dividends=(PilotCashDividend(time=0.5, amount=4.0),),
            ),
            NumericalPilotCase(
                name="terminal_cash_atm",
                exercise_domain=general,
                initial_spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=0.03,
                continuous_dividend_yield=0.0,
                volatility=0.25,
                cash_dividends=(PilotCashDividend(time=1.0, amount=4.0),),
            ),
            NumericalPilotCase(
                name="mixed_cash_and_yield_atm",
                exercise_domain=general,
                initial_spot=100.0,
                strike=100.0,
                maturity=1.0,
                rate=0.02,
                continuous_dividend_yield=0.015,
                volatility=0.30,
                cash_dividends=(PilotCashDividend(time=0.5, amount=2.0),),
            ),
            NumericalPilotCase(
                name="long_high_volatility_itm",
                exercise_domain=general,
                initial_spot=110.0,
                strike=100.0,
                maturity=2.0,
                rate=0.01,
                continuous_dividend_yield=0.025,
                volatility=0.40,
                cash_dividends=(
                    PilotCashDividend(time=0.5, amount=2.0),
                    PilotCashDividend(time=1.5, amount=1.0),
                ),
            ),
        ),
        quadrature_levels=(
            QuadraturePilotLevel(
                name="q1",
                upper_stock=400.0,
                stock_points=10_001,
                normal_order=512,
            ),
            QuadraturePilotLevel(
                name="q2",
                upper_stock=600.0,
                stock_points=20_001,
                normal_order=1_024,
            ),
            QuadraturePilotLevel(
                name="q3",
                upper_stock=800.0,
                stock_points=40_001,
                normal_order=2_048,
            ),
        ),
        general_mesh_intervals=(16, 32, 64, 128),
        lsm_levels=(
            LsmPilotLevel(name="p1", training_paths=2_048, evaluation_paths=2_048),
            LsmPilotLevel(name="p2", training_paths=8_192, evaluation_paths=8_192),
            LsmPilotLevel(name="p3", training_paths=32_768, evaluation_paths=32_768),
        ),
        replications=12,
        replication_critical_value=3.1058065155392804,
        seed_root=0x243F6A8885A308D3,
        random_domain_root=10_000,
        polynomial_degree=4,
        rank_tolerance=1.0e-12,
        condition_limit=1.0e8,
        tie_rule=ExerciseTieRule.STRICTLY_GREATER,
        insufficient_exercise_support=InsufficientExerciseSupportPolicy.FAIL,
        tolerances=NumericalPilotTolerances(
            deterministic_absolute=0.003,
            deterministic_relative=0.0003,
            maximum_boundary_gap=0.002,
            mesh_absolute=0.02,
            mesh_relative=0.001,
            lsm_bias_absolute=0.03,
            lsm_bias_relative=0.003,
            lsm_path_drift_absolute=0.02,
            lsm_path_drift_relative=0.002,
            lsm_native_standard_error_absolute=0.08,
            lsm_native_standard_error_relative=0.005,
        ),
    )


def project_publication_numerical_pilot_v2() -> NumericalPilotSpec:
    """Append one general-mesh level after the retained v1 terminal-cash miss."""
    return replace(
        project_publication_numerical_pilot_v1(),
        name="project-publication-target-free-v2",
        general_mesh_intervals=(16, 32, 64, 128, 256),
    )


def project_publication_numerical_pilot_v3() -> NumericalPilotSpec:
    """Append one path level after the retained v2 high-volatility paired-drift miss."""
    v2 = project_publication_numerical_pilot_v2()
    return replace(
        v2,
        name="project-publication-target-free-v3",
        lsm_levels=(
            *v2.lsm_levels,
            LsmPilotLevel(name="p4", training_paths=131_072, evaluation_paths=131_072),
        ),
    )


def project_publication_numerical_pilot_v4() -> NumericalPilotSpec:
    """Apply the passing explicit-predecessor domain to v3's terminal-cash case."""
    v3 = project_publication_numerical_pilot_v3()
    cases = tuple(
        replace(case, exercise_domain=PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR)
        if case.name == "terminal_cash_atm"
        else case
        for case in v3.cases
    )
    return replace(v3, name="project-publication-target-free-v4", cases=cases)


def project_publication_numerical_pilot_v5() -> NumericalPilotSpec:
    """Use the passing compact exact schedule for v4's terminal-cash case."""
    v4 = project_publication_numerical_pilot_v4()
    cases = tuple(
        replace(
            case,
            exercise_domain=PilotExerciseDomain.COMPACT_TERMINAL_CASH_PREDECESSOR,
        )
        if case.name == "terminal_cash_atm"
        else case
        for case in v4.cases
    )
    return replace(v4, name="project-publication-target-free-v5", cases=cases)


def terminal_cash_polynomial_degree_diagnostic_v1(degree: int) -> NumericalPilotSpec:
    """Isolate polynomial-space bias on v3's exact terminal-cash paths and schedule."""
    if type(degree) is not int or not 0 <= degree <= _MAX_POLYNOMIAL_DEGREE:
        raise NumericalPilotValueError(
            field="degree",
            observed=degree,
            requirement=f"a plain integer in [0, {_MAX_POLYNOMIAL_DEGREE}]",
        )
    v3 = project_publication_numerical_pilot_v3()
    terminal_index = next(
        index for index, case in enumerate(v3.cases) if case.name == "terminal_cash_atm"
    )
    return replace(
        v3,
        name=f"terminal-cash-polynomial-degree-{degree}-diagnostic-v1",
        cases=(v3.cases[terminal_index],),
        lsm_levels=v3.lsm_levels[-2:],
        random_domain_root=v3.random_domain_root + (2 * terminal_index),
        polynomial_degree=degree,
    )


def terminal_cash_predecessor_diagnostic_v1() -> NumericalPilotSpec:
    """Test the exact v3 paths with only the declared pre-terminal-cash decision."""
    degree_four = terminal_cash_polynomial_degree_diagnostic_v1(_MAX_POLYNOMIAL_DEGREE)
    case = degree_four.cases[0]
    return replace(
        degree_four,
        name="terminal-cash-predecessor-diagnostic-v1",
        cases=(
            replace(
                case,
                exercise_domain=PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR,
            ),
        ),
    )


def terminal_cash_compact_predecessor_diagnostic_v1() -> NumericalPilotSpec:
    """Remove transition-only nodes from the passing terminal-predecessor diagnostic."""
    predecessor = terminal_cash_predecessor_diagnostic_v1()
    return replace(
        predecessor,
        name="terminal-cash-compact-predecessor-diagnostic-v1",
        cases=(
            replace(
                predecessor.cases[0],
                exercise_domain=PilotExerciseDomain.COMPACT_TERMINAL_CASH_PREDECESSOR,
            ),
        ),
    )


def numerical_pilot_spec_sha256(spec: NumericalPilotSpec) -> str:
    """Hash the complete target-free scientific identity."""
    if type(spec) is not NumericalPilotSpec:
        raise NumericalPilotValueError(
            field="spec",
            observed=spec,
            requirement="NumericalPilotSpec",
        )
    return _canonical_sha256({"format": _SPEC_HASH_FORMAT, "spec": asdict(spec)})


def numerical_pilot_execution_sha256(execution: NumericalPilotExecution) -> str:
    """Hash the resource and native floating-point implementation identity."""
    if type(execution) is not NumericalPilotExecution:
        raise NumericalPilotValueError(
            field="execution",
            observed=execution,
            requirement="NumericalPilotExecution",
        )
    return _canonical_sha256({"execution": asdict(execution), "format": _EXECUTION_HASH_FORMAT})


def run_numerical_pilot(
    spec: NumericalPilotSpec,
    execution: NumericalPilotExecution,
    *,
    paper_revision: str,
    engine_revision: str,
) -> NumericalPilotReceipt:
    """Run every declared synthetic case and retain every numerical level and replication."""
    if type(spec) is not NumericalPilotSpec:
        raise NumericalPilotValueError(
            field="spec", observed=spec, requirement="NumericalPilotSpec"
        )
    if type(execution) is not NumericalPilotExecution:
        raise NumericalPilotValueError(
            field="execution",
            observed=execution,
            requirement="NumericalPilotExecution",
        )
    _revision(paper_revision, field="paper_revision")
    _revision(engine_revision, field="engine_revision")
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    executor = CpuExecutor(threads=execution.threads, math_backend=execution.math_backend)
    cases = tuple(
        _run_case(spec, case, case_index, executor) for case_index, case in enumerate(spec.cases)
    )
    build = build_info()
    return NumericalPilotReceipt(
        format=_FORMAT,
        spec_sha256=numerical_pilot_spec_sha256(spec),
        execution_sha256=numerical_pilot_execution_sha256(execution),
        paper_revision=paper_revision,
        engine_revision=engine_revision,
        package_version=build.package_version,
        rustc_version=build.rustc_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
        started_at_utc=started_at.isoformat(),
        wall_seconds=time.perf_counter() - started,
        spec=spec,
        execution=execution,
        cases=cases,
        passed=all(case.passed for case in cases),
    )


def _run_case(
    spec: NumericalPilotSpec,
    case: NumericalPilotCase,
    case_index: int,
    executor: CpuExecutor,
) -> NumericalPilotCaseResult:
    reference_mesh = (
        None
        if case.exercise_domain is PilotExerciseDomain.CASH_EVENT_THEOREM
        else spec.general_mesh_intervals[-1]
    )
    schedule = _event_schedule(case, base_intervals=reference_mesh)
    deterministic = tuple(
        _price_quadrature(case, schedule, level, executor) for level in spec.quadrature_levels
    )
    finest = deterministic[-1]
    mesh = _mesh_sequence(spec, case, executor, finest)
    lsm = tuple(
        _run_lsm_level(spec, case, case_index, schedule, level, executor)
        for level in spec.lsm_levels
    )
    gates = _case_gates(spec, case, deterministic, mesh, lsm)
    return NumericalPilotCaseResult(
        case=case,
        reference_mesh_intervals=reference_mesh,
        deterministic=deterministic,
        mesh=mesh,
        lsm=lsm,
        gates=gates,
        passed=all(gate.passed for gate in gates),
    )


def _price_quadrature(
    case: NumericalPilotCase,
    schedule: EventSchedule,
    level: QuadraturePilotLevel,
    executor: CpuExecutor,
) -> DeterministicPilotObservation:
    curve, yield_curve = _curves(case, schedule)
    method = EventAlignedQuadrature(
        upper_stock=level.upper_stock,
        stock_points=level.stock_points,
        normal_order=level.normal_order,
    )
    started = time.perf_counter()
    result = price_event_aligned_call(
        initial_spot=case.initial_spot,
        strike=case.strike,
        schedule=schedule,
        curve=curve,
        dividend_yield_curve=yield_curve,
        volatility=ConstantVolatility(case.volatility),
        dividend_policy=DividendPolicy.LIQUIDATOR,
        executor=executor,
        method=method,
    )
    return DeterministicPilotObservation(
        level=level.name,
        upper_stock=level.upper_stock,
        stock_points=level.stock_points,
        normal_order=level.normal_order,
        price=result.price,
        maximum_upper_boundary_gap=result.maximum_upper_boundary_gap,
        wall_seconds=time.perf_counter() - started,
    )


def _mesh_sequence(
    spec: NumericalPilotSpec,
    case: NumericalPilotCase,
    executor: CpuExecutor,
    finest_reference: DeterministicPilotObservation,
) -> tuple[MeshPilotObservation, ...]:
    if case.exercise_domain in (
        PilotExerciseDomain.CASH_EVENT_THEOREM,
        PilotExerciseDomain.COMPACT_TERMINAL_CASH_PREDECESSOR,
    ):
        return ()
    finest_level = spec.quadrature_levels[-1]
    observations: list[MeshPilotObservation] = []
    for base_intervals in spec.general_mesh_intervals:
        if base_intervals == spec.general_mesh_intervals[-1]:
            observation = finest_reference
        else:
            observation = _price_quadrature(
                case,
                _event_schedule(case, base_intervals=base_intervals),
                finest_level,
                executor,
            )
        observations.append(
            MeshPilotObservation(
                base_intervals=base_intervals,
                price=observation.price,
                maximum_upper_boundary_gap=observation.maximum_upper_boundary_gap,
                wall_seconds=observation.wall_seconds,
            )
        )
    return tuple(observations)


def _run_lsm_level(  # noqa: PLR0913 - scientific identities remain separate arguments
    spec: NumericalPilotSpec,
    case: NumericalPilotCase,
    case_index: int,
    schedule: EventSchedule,
    level: LsmPilotLevel,
    executor: CpuExecutor,
) -> LsmLevelObservation:
    curve, yield_curve = _curves(case, schedule)
    training_domain = spec.random_domain_root + (2 * case_index)
    evaluation_domain = training_domain + 1
    observations: list[LsmReplicationObservation] = []
    for replication in range(spec.replications):
        seed = (spec.seed_root + (replication * _SEED_INCREMENT)) & _MAX_U64
        method = Lsm(
            training_paths=level.training_paths,
            evaluation_paths=level.evaluation_paths,
            training_stream=RandomStream(seed=seed, domain=training_domain),
            evaluation_stream=RandomStream(seed=seed, domain=evaluation_domain),
            polynomial_degree=spec.polynomial_degree,
            rank_tolerance=spec.rank_tolerance,
            condition_limit=spec.condition_limit,
            tie_rule=spec.tie_rule,
            insufficient_exercise_support=spec.insufficient_exercise_support,
        )
        started = time.perf_counter()
        result = price_lsm_call(
            initial_spot=case.initial_spot,
            strike=case.strike,
            schedule=schedule,
            curve=curve,
            dividend_yield_curve=yield_curve,
            volatility=ConstantVolatility(case.volatility),
            dividend_policy=DividendPolicy.LIQUIDATOR,
            executor=executor,
            method=method,
        )
        elapsed = time.perf_counter() - started
        if result.standard_error is None:
            message = "LSM omitted standard error for a multi-path evaluation"
            raise NumericalPilotError(message)
        if sum(result.exercise_counts) != level.evaluation_paths:
            message = "LSM stopping counts do not partition evaluation paths"
            raise NumericalPilotError(message)
        regressions = tuple(
            diagnostic
            for diagnostic in result.fit_diagnostics
            if isinstance(diagnostic, RegressionFitDiagnostics)
        )
        origins = tuple(
            diagnostic
            for diagnostic in result.fit_diagnostics
            if isinstance(diagnostic, OriginDecisionDiagnostics)
        )
        no_itm = sum(
            isinstance(diagnostic, NoInTheMoneyPathsDiagnostics)
            for diagnostic in result.fit_diagnostics
        )
        observations.append(
            LsmReplicationObservation(
                replication=replication,
                seed=seed,
                training_domain=training_domain,
                evaluation_domain=evaluation_domain,
                price=result.price,
                native_standard_error=result.standard_error,
                regression_nodes=len(regressions),
                no_in_the_money_nodes=no_itm,
                minimum_in_the_money_training_paths=(
                    min(regression.in_the_money_paths for regression in regressions)
                    if regressions
                    else None
                ),
                maximum_regression_condition_number=(
                    max(regression.condition_number for regression in regressions)
                    if regressions
                    else None
                ),
                origin_exercise=origins[0].exercise if len(origins) == 1 else None,
                early_exercise_paths=sum(result.exercise_counts[:-1]),
                expiry_paths=result.exercise_counts[-1],
                wall_seconds=elapsed,
            )
        )
    prices = tuple(observation.price for observation in observations)
    replication_standard_deviation = statistics.stdev(prices)
    return LsmLevelObservation(
        level=level.name,
        training_paths=level.training_paths,
        evaluation_paths=level.evaluation_paths,
        mean_price=statistics.fmean(prices),
        replication_standard_deviation=replication_standard_deviation,
        replication_standard_error=replication_standard_deviation / math.sqrt(len(prices)),
        mean_native_standard_error=statistics.fmean(
            observation.native_standard_error for observation in observations
        ),
        replications=tuple(observations),
    )


def _case_gates(
    spec: NumericalPilotSpec,
    case: NumericalPilotCase,
    deterministic: tuple[DeterministicPilotObservation, ...],
    mesh: tuple[MeshPilotObservation, ...],
    lsm: tuple[LsmLevelObservation, ...],
) -> tuple[NumericalPilotGate, ...]:
    tolerance = spec.tolerances
    reference = deterministic[-1].price
    deterministic_limit = tolerance.deterministic_absolute + (
        tolerance.deterministic_relative * abs(reference)
    )
    gates = [
        _gate(
            "deterministic_last_level_change",
            abs(deterministic[-1].price - deterministic[-2].price),
            deterministic_limit,
        ),
        _gate(
            "deterministic_upper_boundary_gap",
            deterministic[-1].maximum_upper_boundary_gap,
            tolerance.maximum_boundary_gap,
        ),
        _gate(
            "deterministic_immediate_exercise_lower_bound",
            max(case.initial_spot - case.strike - reference, 0.0),
            deterministic_limit,
        ),
        _gate(
            "deterministic_stock_dominance",
            max(reference - case.initial_spot, 0.0),
            deterministic_limit,
        ),
    ]
    if mesh:
        mesh_limit = tolerance.mesh_absolute + tolerance.mesh_relative * abs(mesh[-1].price)
        gates.append(
            _gate(
                "general_mesh_last_level_change", abs(mesh[-1].price - mesh[-2].price), mesh_limit
            )
        )
    finest_lsm = lsm[-1]
    bias_limit = (
        tolerance.lsm_bias_absolute
        + (tolerance.lsm_bias_relative * abs(reference))
        + (spec.replication_critical_value * finest_lsm.replication_standard_error)
        + deterministic_limit
    )
    gates.append(
        _gate(
            "finest_lsm_mean_reference_envelope", abs(finest_lsm.mean_price - reference), bias_limit
        )
    )
    paired_differences = tuple(
        fine.price - prior.price
        for prior, fine in zip(lsm[-2].replications, finest_lsm.replications, strict=True)
    )
    paired_standard_error = statistics.stdev(paired_differences) / math.sqrt(
        len(paired_differences)
    )
    drift_limit = (
        tolerance.lsm_path_drift_absolute
        + (tolerance.lsm_path_drift_relative * abs(reference))
        + (spec.replication_critical_value * paired_standard_error)
    )
    gates.append(
        _gate(
            "last_lsm_path_level_paired_drift",
            abs(statistics.fmean(paired_differences)),
            drift_limit,
        )
    )
    native_error_limit = tolerance.lsm_native_standard_error_absolute + (
        tolerance.lsm_native_standard_error_relative * abs(reference)
    )
    gates.append(
        _gate(
            "finest_lsm_mean_native_standard_error",
            finest_lsm.mean_native_standard_error,
            native_error_limit,
        )
    )
    return tuple(gates)


def _event_schedule(
    case: NumericalPilotCase,
    *,
    base_intervals: int | None,
) -> EventSchedule:
    if case.exercise_domain is PilotExerciseDomain.CASH_EVENT_THEOREM:
        if base_intervals is not None:
            message = "cash-event theorem schedule received a uniform mesh"
            raise NumericalPilotError(message)
        times = (0.0, *(dividend.time for dividend in case.cash_dividends), case.maturity)
        exercise_nodes = tuple(range(1, len(times) - 1))
    elif case.exercise_domain is PilotExerciseDomain.COMPACT_TERMINAL_CASH_PREDECESSOR:
        if base_intervals is None:
            message = "compact terminal-predecessor schedule omitted its declared time denominator"
            raise NumericalPilotError(message)
        if base_intervals < _MINIMUM_TERMINAL_PREDECESSOR_INTERVALS:
            message = "compact terminal-cash predecessor requires a denominator of at least two"
            raise NumericalPilotError(message)
        predecessor_time = case.maturity * ((base_intervals - 1) / base_intervals)
        times = tuple(
            sorted(
                {
                    0.0,
                    predecessor_time,
                    *(dividend.time for dividend in case.cash_dividends),
                    case.maturity,
                }
            )
        )
        preexpiry_cash_nodes = {
            times.index(dividend.time)
            for dividend in case.cash_dividends
            if dividend.time < case.maturity
        }
        exercise_nodes = tuple(sorted({*preexpiry_cash_nodes, times.index(predecessor_time)}))
    else:
        if base_intervals is None:
            message = "general-domain schedule omitted its uniform mesh"
            raise NumericalPilotError(message)
        if (
            case.exercise_domain is PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR
            and base_intervals < _MINIMUM_TERMINAL_PREDECESSOR_INTERVALS
        ):
            message = "terminal-cash predecessor schedule requires at least two base intervals"
            raise NumericalPilotError(message)
        base = tuple(
            case.maturity if node == base_intervals else case.maturity * (node / base_intervals)
            for node in range(base_intervals + 1)
        )
        times = tuple(sorted({*base, *(dividend.time for dividend in case.cash_dividends)}))
        if case.exercise_domain is PilotExerciseDomain.TERMINAL_CASH_PREDECESSOR:
            preexpiry_cash_nodes = {
                times.index(dividend.time)
                for dividend in case.cash_dividends
                if dividend.time < case.maturity
            }
            predecessor_time = base[-2]
            exercise_nodes = tuple(sorted({*preexpiry_cash_nodes, times.index(predecessor_time)}))
        else:
            exercise_nodes = tuple(range(len(times) - 1))
    dividends = tuple(
        CashDividend(node=times.index(dividend.time), amount=dividend.amount)
        for dividend in case.cash_dividends
    )
    return EventSchedule(
        times=times,
        dividends=dividends,
        exercise_nodes=exercise_nodes,
        expiry_node=len(times) - 1,
    )


def _curves(
    case: NumericalPilotCase,
    schedule: EventSchedule,
) -> tuple[LogDiscountCurve, CumulativeDividendYieldCurve]:
    return (
        LogDiscountCurve(
            times=schedule.times,
            log_discounts=tuple(
                0.0 if node == 0 else -case.rate * time for node, time in enumerate(schedule.times)
            ),
        ),
        CumulativeDividendYieldCurve(
            times=schedule.times,
            cumulative_yields=tuple(
                0.0 if node == 0 else case.continuous_dividend_yield * time
                for node, time in enumerate(schedule.times)
            ),
        ),
    )


def write_numerical_pilot_receipt(path: Path, receipt: NumericalPilotReceipt) -> None:
    """Write one canonical JSON receipt atomically without overwriting prior evidence."""
    if not isinstance(path, Path):
        raise NumericalPilotValueError(field="path", observed=path, requirement="pathlib.Path")
    if type(receipt) is not NumericalPilotReceipt:
        raise NumericalPilotValueError(
            field="receipt",
            observed=receipt,
            requirement="NumericalPilotReceipt",
        )
    if path.exists():
        message = f"pilot receipt already exists: {path}"
        raise NumericalPilotError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(asdict(receipt)) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            path.hardlink_to(temporary)
        except FileExistsError as error:
            message = f"pilot receipt already exists: {path}"
            raise NumericalPilotError(message) from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one frozen pilot version and return nonzero when any numerical gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    pilot_choices = (
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "terminal-degree-0",
        "terminal-degree-1",
        "terminal-degree-2",
        "terminal-degree-3",
        "terminal-degree-4",
        "terminal-predecessor-v1",
        "terminal-compact-predecessor-v1",
    )
    parser.add_argument("--pilot", choices=pilot_choices, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--math-backend",
        choices=tuple(backend.value for backend in CpuMathBackend),
        required=True,
    )
    arguments = parser.parse_args(argv)
    paper_root = Path(__file__).resolve().parents[2]
    paper_revision = _clean_git_revision(paper_root)
    try:
        engine_revision = resolve_project_engine_revision(paper_root)
    except EngineIdentityError as error:
        raise NumericalPilotError(str(error)) from error
    specs = {
        "v1": project_publication_numerical_pilot_v1,
        "v2": project_publication_numerical_pilot_v2,
        "v3": project_publication_numerical_pilot_v3,
        "v4": project_publication_numerical_pilot_v4,
        "v5": project_publication_numerical_pilot_v5,
        **{
            f"terminal-degree-{degree}": (
                lambda degree=degree: terminal_cash_polynomial_degree_diagnostic_v1(degree)
            )
            for degree in range(_MAX_POLYNOMIAL_DEGREE + 1)
        },
        "terminal-predecessor-v1": terminal_cash_predecessor_diagnostic_v1,
        "terminal-compact-predecessor-v1": terminal_cash_compact_predecessor_diagnostic_v1,
    }
    spec = specs[arguments.pilot]()
    receipt = run_numerical_pilot(
        spec,
        NumericalPilotExecution(
            threads=arguments.threads,
            math_backend=CpuMathBackend(arguments.math_backend),
        ),
        paper_revision=paper_revision,
        engine_revision=engine_revision,
    )
    write_numerical_pilot_receipt(arguments.output, receipt)
    summary = {
        "cases": len(receipt.cases),
        "execution_sha256": receipt.execution_sha256,
        "output": str(arguments.output),
        "passed": receipt.passed,
        "spec_sha256": receipt.spec_sha256,
        "wall_seconds": receipt.wall_seconds,
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if receipt.passed else 2


def _clean_git_revision(root: Path) -> str:
    git = shutil.which("git")
    if git is None:
        message = "git executable is unavailable"
        raise NumericalPilotError(message)
    status = subprocess.run(
        [git, "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        message = f"Git worktree is not clean: {root}"
        raise NumericalPilotError(message)
    revision = subprocess.run(
        [git, "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _revision(revision, field=f"revision for {root}")
    return revision


def _gate(name: str, observed: float, limit: float) -> NumericalPilotGate:
    if not math.isfinite(observed) or observed < 0.0:
        message = f"gate {name!r} produced invalid observation {observed!r}"
        raise NumericalPilotError(message)
    if not math.isfinite(limit) or limit < 0.0:
        message = f"gate {name!r} produced invalid limit {limit!r}"
        raise NumericalPilotError(message)
    return NumericalPilotGate(name=name, observed=observed, limit=limit, passed=observed <= limit)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _revision(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA1_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NumericalPilotValueError(
            field=field,
            observed=value,
            requirement="a lowercase 40-character Git object ID",
        )


def _nonempty_string(value: object, *, field: str) -> None:
    if type(value) is not str or not value:
        raise NumericalPilotValueError(field=field, observed=value, requirement="a nonempty string")


def _finite_float(value: object, *, field: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise NumericalPilotValueError(
            field=field, observed=value, requirement="a finite plain float"
        )


def _positive_float(value: object, *, field: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise NumericalPilotValueError(
            field=field,
            observed=value,
            requirement="a finite strictly positive plain float",
        )


def _canonical_nonnegative_float(value: object, *, field: str) -> None:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or (value == 0.0 and math.copysign(1.0, value) < 0.0)
    ):
        raise NumericalPilotValueError(
            field=field,
            observed=value,
            requirement="a finite canonical nonnegative plain float",
        )


def _bounded_int(value: object, *, field: str, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise NumericalPilotValueError(
            field=field,
            observed=value,
            requirement=f"a plain integer in [0, {maximum}]",
        )


def _positive_int(value: object, *, field: str, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= _MAX_U32:
        raise NumericalPilotValueError(
            field=field,
            observed=value,
            requirement=f"a plain integer in [{minimum}, {_MAX_U32}]",
        )


def _typed_nonempty_tuple[T](
    values: object,
    expected: type[T],
    *,
    field: str,
    minimum: int = 1,
) -> None:
    if (
        type(values) is not tuple
        or len(values) < minimum
        or any(type(value) is not expected for value in values)
    ):
        raise NumericalPilotValueError(
            field=field,
            observed=values,
            requirement=f"tuple[{expected.__name__}, ...] with at least {minimum} values",
        )


def _strictly_increasing_ints(values: tuple[int, ...], *, field: str) -> None:
    if any(type(value) is not int or value <= 0 for value in values) or any(
        right <= left for left, right in pairwise(values)
    ):
        raise NumericalPilotValueError(
            field=field,
            observed=values,
            requirement="strictly increasing positive plain integers",
        )


def _strictly_increasing_floats(values: tuple[float, ...], *, field: str) -> None:
    if any(
        type(value) is not float or not math.isfinite(value) or value <= 0.0 for value in values
    ) or any(right <= left for left, right in pairwise(values)):
        raise NumericalPilotValueError(
            field=field,
            observed=values,
            requirement="strictly increasing finite positive plain floats",
        )


def _strictly_increasing_names(values: tuple[str, ...], *, field: str) -> None:
    if len(set(values)) != len(values) or any(
        type(value) is not str or not value for value in values
    ):
        raise NumericalPilotValueError(
            field=field,
            observed=values,
            requirement="unique nonempty plain strings",
        )


__all__ = [
    "LsmPilotLevel",
    "NumericalPilotCase",
    "NumericalPilotError",
    "NumericalPilotExecution",
    "NumericalPilotReceipt",
    "NumericalPilotSpec",
    "NumericalPilotTolerances",
    "NumericalPilotValueError",
    "PilotCashDividend",
    "PilotExerciseDomain",
    "QuadraturePilotLevel",
    "numerical_pilot_execution_sha256",
    "numerical_pilot_spec_sha256",
    "project_publication_numerical_pilot_v1",
    "project_publication_numerical_pilot_v2",
    "project_publication_numerical_pilot_v3",
    "project_publication_numerical_pilot_v4",
    "project_publication_numerical_pilot_v5",
    "run_numerical_pilot",
    "terminal_cash_compact_predecessor_diagnostic_v1",
    "terminal_cash_polynomial_degree_diagnostic_v1",
    "terminal_cash_predecessor_diagnostic_v1",
    "write_numerical_pilot_receipt",
]
