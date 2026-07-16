"""Explicit local process and hidden-thread budgets for heavy study phases."""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator

_MIN_THREAD_COUNT: Final = 1
_MAX_THREAD_COUNT: Final = 64
_MIN_NICE_INCREMENT: Final = 0
_MAX_NICE_INCREMENT: Final = 19
_OPEN_FLAGS: Final = os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR


@dataclass(slots=True)
class LocalResourceBudgetError(ValueError):
    """A local execution budget is invalid or cannot be resolved exactly."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected resource contract."""
        return f"invalid local resource budget: {self.reason}"


@dataclass(frozen=True, slots=True)
class LocalResourceBudget:
    """One explicit upper bound for a local heavy-process phase.

    The budget controls orchestration and hidden CPU thread pools. It is deliberately separate from
    the financial estimator and neural backend identities. A caller must still acquire an exclusive
    heavy-process lease; ``max_heavy_processes`` is recorded here so the resolved operational
    manifest cannot omit that policy.
    """

    name: str
    max_heavy_processes: int
    data_threads: int
    hidden_cpu_threads: int
    nice_increment: int
    automatic_restart: bool

    def __post_init__(self) -> None:
        """Reject ambiguous names, booleans-as-integers and unsupported resource ranges."""
        if type(self.name) is not str or not self.name:
            raise LocalResourceBudgetError(reason="name must be a nonempty plain string")
        for field, value in (
            ("max_heavy_processes", self.max_heavy_processes),
            ("data_threads", self.data_threads),
            ("hidden_cpu_threads", self.hidden_cpu_threads),
        ):
            if type(value) is not int or not _MIN_THREAD_COUNT <= value <= _MAX_THREAD_COUNT:
                raise LocalResourceBudgetError(
                    reason=f"{field} must be a plain integer in [1, 64]",
                )
        if (
            type(self.nice_increment) is not int
            or not _MIN_NICE_INCREMENT <= self.nice_increment <= _MAX_NICE_INCREMENT
        ):
            raise LocalResourceBudgetError(
                reason="nice_increment must be a plain integer in [0, 19]",
            )
        if type(self.automatic_restart) is not bool:
            raise LocalResourceBudgetError(reason="automatic_restart must be a plain boolean")

    def thread_environment(self) -> dict[str, str]:
        """Return the exact environment owned by this hidden-thread budget."""
        hidden = str(self.hidden_cpu_threads)
        return {
            "MKL_NUM_THREADS": hidden,
            "NUMEXPR_NUM_THREADS": hidden,
            "OMP_NUM_THREADS": hidden,
            "OPENBLAS_NUM_THREADS": hidden,
            "POLARS_MAX_THREADS": str(self.data_threads),
            "VECLIB_MAXIMUM_THREADS": hidden,
        }

    def resolve_environment(self, base: Mapping[str, str]) -> dict[str, str]:
        """Create an explicit child environment without mutating the caller's mapping."""
        if not isinstance(base, Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in base.items()
        ):
            raise LocalResourceBudgetError(reason="base must map plain strings to plain strings")
        resolved = dict(base)
        resolved.update(self.thread_environment())
        return resolved

    def manifest(self) -> dict[str, bool | int | str | dict[str, str]]:
        """Return the complete path-free operational identity for an artifact receipt."""
        return {
            "automatic_restart": self.automatic_restart,
            "data_threads": self.data_threads,
            "hidden_cpu_threads": self.hidden_cpu_threads,
            "max_heavy_processes": self.max_heavy_processes,
            "name": self.name,
            "nice_increment": self.nice_increment,
            "thread_environment": self.thread_environment(),
        }


@contextmanager
def acquire_local_heavy_process_lease(
    path: Path,
    budget: LocalResourceBudget,
) -> Iterator[None]:
    """Hold the exclusive local-heavy-process lease for one complete phase.

    Acquisition is nonblocking: overlap is an error, never a reason to wait or select a different
    execution profile. The lock file is local operational state and is intentionally absent from
    the relocatable budget manifest.
    """
    _validate_lease_request(path, budget)
    try:
        descriptor = os.open(path, _OPEN_FLAGS, 0o600)
    except OSError as error:
        raise LocalResourceBudgetError(
            reason=f"heavy-process lease cannot be opened: {error.strerror or error!s}",
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LocalResourceBudgetError(
                reason="heavy-process lease must be a regular file",
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LocalResourceBudgetError(
                reason="another heavy local process owns the lease",
            ) from error
        except OSError as error:
            raise LocalResourceBudgetError(
                reason=f"heavy-process lease cannot be acquired: {error.strerror or error!s}",
            ) from error
        yield
    finally:
        os.close(descriptor)


def _validate_lease_request(path: object, budget: object) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        raise LocalResourceBudgetError(reason="lease path must be an absolute pathlib.Path")
    if type(budget) is not LocalResourceBudget:
        raise LocalResourceBudgetError(reason="lease budget must be LocalResourceBudget")
    if budget.max_heavy_processes != 1:
        raise LocalResourceBudgetError(
            reason="the exclusive local lease requires max_heavy_processes == 1",
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise LocalResourceBudgetError(
            reason="lease parent must be an existing non-symlink directory",
        )
    if path.is_symlink():
        raise LocalResourceBudgetError(reason="lease path must not be a symlink")


PROJECT_INTERACTIVE_RESOURCE_BUDGET: Final = LocalResourceBudget(
    name="project_interactive_one_phase_v1",
    max_heavy_processes=1,
    data_threads=2,
    hidden_cpu_threads=1,
    nice_increment=10,
    automatic_restart=False,
)


__all__ = [
    "PROJECT_INTERACTIVE_RESOURCE_BUDGET",
    "LocalResourceBudget",
    "LocalResourceBudgetError",
    "acquire_local_heavy_process_lease",
]
