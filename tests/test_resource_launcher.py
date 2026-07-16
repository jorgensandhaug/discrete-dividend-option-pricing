"""Pre-import resource admission tests for the installed study command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import _discrete_dividend_study_launcher as launcher

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_ADMITTED_RETURN = 17
_OBSERVED_NICE = 10


def test_launcher_import_does_not_initialize_numerical_packages() -> None:
    """A fresh bootstrap import stays ahead of Polars, NumPy, the engine and model frameworks."""
    script = (
        "import sys; import _discrete_dividend_study_launcher; "
        "blocked={'numpy','polars','finoptions','torch','mlx'}; "
        "print(','.join(sorted(blocked & set(sys.modules))))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.stdout == "\n"


def test_heavy_launcher_applies_budget_before_import_while_holding_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission order is lease, process budget, numerical CLI import, execution."""
    lock = tmp_path / "heavy.lock"
    events: list[str] = []

    def apply(_budget: launcher.LocalResourceBudget) -> int:
        events.append("budget")
        return _OBSERVED_NICE

    def load(name: str) -> Callable[[Sequence[str] | None], int]:
        events.append(f"load:{name}")

        def admitted(_arguments: Sequence[str] | None) -> int:
            with (
                pytest.raises(
                    launcher.LocalResourceBudgetError,
                    match="another heavy local process",
                ),
                launcher.acquire_local_heavy_process_lease(
                    lock,
                    launcher.PROJECT_INTERACTIVE_RESOURCE_BUDGET,
                ),
            ):
                pytest.fail("launcher did not retain the heavy-process lease")
            events.append("run")
            return _ADMITTED_RETURN

        return admitted

    monkeypatch.setattr(launcher.LocalResourceBudget, "apply_to_current_process", apply)
    monkeypatch.setattr(launcher, "_load_cli_entry", load)

    observed = launcher.main(
        ("run-neural-study", "--local-resource-lock", str(lock), "--unused-test-argument")
    )

    assert observed == _ADMITTED_RETURN
    assert events == ["budget", "load:_main_after_local_resource_admission", "run"]


def test_overlapping_heavy_launcher_fails_before_cli_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A competing phase cannot initialize the numerical CLI or read model inputs."""
    lock = tmp_path / "heavy.lock"

    def forbidden(_name: str) -> object:
        pytest.fail("overlap reached numerical CLI import")

    monkeypatch.setattr(launcher, "_load_cli_entry", forbidden)
    with (
        launcher.acquire_local_heavy_process_lease(
            lock,
            launcher.PROJECT_INTERACTIVE_RESOURCE_BUDGET,
        ),
        pytest.raises(launcher.ResourceLauncherError, match="another heavy local process"),
    ):
        launcher.main(("run-classical-study", "--local-resource-lock", str(lock)))


def test_process_budget_overrides_hidden_pools_and_applies_niceness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-shot bootstrap installs every exact cap before returning to its caller."""
    observed_increment: list[int] = []

    def nice(increment: int) -> int:
        observed_increment.append(increment)
        return _OBSERVED_NICE

    monkeypatch.setattr(os, "nice", nice)
    for variable in launcher.PROJECT_INTERACTIVE_RESOURCE_BUDGET.thread_environment():
        monkeypatch.setenv(variable, "999")

    observed_nice = launcher.PROJECT_INTERACTIVE_RESOURCE_BUDGET.apply_to_current_process()

    assert observed_nice == _OBSERVED_NICE
    assert observed_increment == [10]
    for (
        variable,
        expected,
    ) in launcher.PROJECT_INTERACTIVE_RESOURCE_BUDGET.thread_environment().items():
        assert os.environ[variable] == expected


@pytest.mark.parametrize(
    "arguments",
    [
        ("run-classical-study", "--threads", "9"),
        ("run-neural-study", "--data-threads=5"),
    ],
)
def test_heavy_launcher_rejects_threads_above_local_budget_before_import(
    arguments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Estimator and preparation concurrency are rejected, never silently rewritten."""

    def forbidden(_name: str) -> object:
        pytest.fail("over-budget request reached numerical CLI import")

    monkeypatch.setattr(launcher, "_load_cli_entry", forbidden)

    with pytest.raises(launcher.ResourceLauncherError, match="exceeds local budget"):
        launcher.main(arguments)
