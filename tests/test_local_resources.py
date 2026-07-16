"""Tests for explicit low-impact local execution budgets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from discrete_dividend_study.local_resources import (
    PROJECT_INTERACTIVE_RESOURCE_BUDGET,
    LocalResourceBudget,
    LocalResourceBudgetError,
    acquire_local_heavy_process_lease,
)

if TYPE_CHECKING:
    from pathlib import Path

_PROJECT_DATA_THREADS = 2
_PROJECT_NICE_INCREMENT = 10


def test_project_interactive_budget_is_one_nonrestarting_heavy_phase() -> None:
    """The workstation profile cannot authorize a local process swarm."""
    budget = PROJECT_INTERACTIVE_RESOURCE_BUDGET
    assert budget.max_heavy_processes == 1
    assert budget.data_threads == _PROJECT_DATA_THREADS
    assert budget.hidden_cpu_threads == 1
    assert budget.nice_increment == _PROJECT_NICE_INCREMENT
    assert budget.automatic_restart is False
    assert budget.thread_environment() == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def test_environment_resolution_is_explicit_and_nonmutating() -> None:
    """The child receives exact limits while the caller environment remains unchanged."""
    base = {"PATH": "/bin", "OMP_NUM_THREADS": "64"}
    resolved = PROJECT_INTERACTIVE_RESOURCE_BUDGET.resolve_environment(base)
    assert base["OMP_NUM_THREADS"] == "64"
    assert resolved["OMP_NUM_THREADS"] == "1"
    assert resolved["PATH"] == "/bin"


def test_heavy_process_lease_rejects_overlap_and_releases(tmp_path: Path) -> None:
    """A second heavy phase fails immediately and the lease remains reusable after exit."""
    lock = tmp_path / "heavy-process.lock"
    with (
        acquire_local_heavy_process_lease(lock, PROJECT_INTERACTIVE_RESOURCE_BUDGET),
        pytest.raises(LocalResourceBudgetError, match="another heavy local process"),
        acquire_local_heavy_process_lease(lock, PROJECT_INTERACTIVE_RESOURCE_BUDGET),
    ):
        pass
    with acquire_local_heavy_process_lease(lock, PROJECT_INTERACTIVE_RESOURCE_BUDGET):
        assert lock.is_file()


def test_heavy_process_lease_rejects_symlink(tmp_path: Path) -> None:
    """The operational lock cannot be redirected through a filesystem alias."""
    target = tmp_path / "target.lock"
    target.touch()
    alias = tmp_path / "alias.lock"
    alias.symlink_to(target)
    with (
        pytest.raises(LocalResourceBudgetError, match="must not be a symlink"),
        acquire_local_heavy_process_lease(alias, PROJECT_INTERACTIVE_RESOURCE_BUDGET),
    ):
        pass


def test_heavy_process_lease_requires_exclusive_budget(tmp_path: Path) -> None:
    """The single-file lease cannot misrepresent a multi-process budget."""
    parallel = LocalResourceBudget(
        name="parallel",
        max_heavy_processes=2,
        data_threads=2,
        hidden_cpu_threads=1,
        nice_increment=10,
        automatic_restart=False,
    )
    with (
        pytest.raises(LocalResourceBudgetError, match="max_heavy_processes == 1"),
        acquire_local_heavy_process_lease(tmp_path / "heavy.lock", parallel),
    ):
        pass


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_heavy_processes", 0),
        ("data_threads", True),
        ("hidden_cpu_threads", 0),
        ("nice_increment", 20),
        ("automatic_restart", 0),
    ],
)
def test_invalid_budget_values_fail_without_coercion(field: str, value: object) -> None:
    """Invalid ranges and booleans-as-integers never become nearby valid budgets."""
    arguments: dict[str, object] = {
        "name": "invalid",
        "max_heavy_processes": 1,
        "data_threads": 2,
        "hidden_cpu_threads": 1,
        "nice_increment": 10,
        "automatic_restart": False,
    }
    arguments[field] = value
    with pytest.raises(LocalResourceBudgetError):
        LocalResourceBudget(**arguments)  # type: ignore[arg-type]
