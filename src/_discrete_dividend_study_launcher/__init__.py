"""Pre-import resource admission for the installed project-study command."""

from __future__ import annotations

import fcntl
import importlib
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

_MIN_THREAD_COUNT: Final = 1
_MAX_THREAD_COUNT: Final = 64
_MIN_NICE_INCREMENT: Final = 0
_MAX_NICE_INCREMENT: Final = 19
_OPEN_FLAGS: Final = os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR
_LOCK_OPTION: Final = "--local-resource-lock"
_COMPUTE_THREADS_OPTION: Final = "--threads"
_DATA_THREADS_OPTION: Final = "--data-threads"
_HEAVY_STUDY_COMMANDS: Final = frozenset(
    {
        "run-classical-study",
        "run-neural-study",
        "run-publication-cash-replications",
        "run-publication-empirical-inference",
    }
)


@dataclass(slots=True)
class LocalResourceBudgetError(ValueError):
    """A local execution budget is invalid or cannot be resolved exactly."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected resource contract."""
        return f"invalid local resource budget: {self.reason}"


@dataclass(slots=True)
class ResourceLauncherError(RuntimeError):
    """The installed command could not admit a requested local heavy phase."""

    reason: str

    def __str__(self) -> str:
        """Return the failed launch precondition."""
        return f"study launcher failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class LocalResourceBudget:
    """One explicit upper bound for a local heavy-process phase."""

    name: str
    max_heavy_processes: int
    compute_threads: int
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
            ("compute_threads", self.compute_threads),
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

    def apply_to_current_process(self) -> int:
        """Install thread caps and lower priority before any numerical package is imported."""
        for key, value in self.thread_environment().items():
            os.environ[key] = value
        try:
            observed_nice = os.nice(self.nice_increment)
        except OSError as error:
            raise LocalResourceBudgetError(
                reason=f"process niceness cannot be applied: {error.strerror or error!s}",
            ) from error
        for key, value in self.thread_environment().items():
            if os.environ.get(key) != value:
                raise LocalResourceBudgetError(reason=f"thread environment {key} was not retained")
        return observed_nice

    def manifest(self) -> dict[str, bool | int | str | dict[str, str]]:
        """Return the complete path-free operational identity for an artifact receipt."""
        return {
            "automatic_restart": self.automatic_restart,
            "compute_threads": self.compute_threads,
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
    """Hold one nonblocking exclusive lease across a complete local heavy phase."""
    _validate_lease_request(path, budget)
    try:
        descriptor = os.open(path, _OPEN_FLAGS, 0o600)
    except OSError as error:
        raise LocalResourceBudgetError(
            reason=f"heavy-process lease cannot be opened: {error.strerror or error!s}",
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LocalResourceBudgetError(reason="heavy-process lease must be a regular file")
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
    compute_threads=8,
    data_threads=4,
    hidden_cpu_threads=2,
    nice_increment=10,
    automatic_restart=False,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Admit a heavy command before importing the study or any numerical dependency."""
    arguments = _arguments(argv)
    if not arguments or arguments[0] not in _HEAVY_STUDY_COMMANDS:
        return _load_cli_entry("main")(arguments)
    try:
        _validate_requested_threads(arguments, PROJECT_INTERACTIVE_RESOURCE_BUDGET)
        lock = _local_resource_lock(arguments)
        with acquire_local_heavy_process_lease(lock, PROJECT_INTERACTIVE_RESOURCE_BUDGET):
            PROJECT_INTERACTIVE_RESOURCE_BUDGET.apply_to_current_process()
            return _load_cli_entry("_main_after_local_resource_admission")(arguments)
    except LocalResourceBudgetError as error:
        raise ResourceLauncherError(reason=str(error)) from error


def _arguments(argv: Sequence[str] | None) -> tuple[str, ...]:
    source: object = sys.argv[1:] if argv is None else argv
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise LocalResourceBudgetError(reason="argv must be a sequence of plain strings")
    arguments = tuple(source)
    if any(type(value) is not str for value in arguments):
        raise LocalResourceBudgetError(reason="argv must be a sequence of plain strings")
    return arguments


def _local_resource_lock(arguments: tuple[str, ...]) -> Path:
    values: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == _LOCK_OPTION:
            if index + 1 >= len(arguments):
                raise LocalResourceBudgetError(reason=f"{_LOCK_OPTION} requires one path")
            values.append(arguments[index + 1])
            index += 2
            continue
        prefix = f"{_LOCK_OPTION}="
        if value.startswith(prefix):
            values.append(value[len(prefix) :])
        index += 1
    if len(values) > 1:
        raise LocalResourceBudgetError(reason=f"{_LOCK_OPTION} may be specified only once")
    if values:
        path = Path(values[0])
    else:
        path = Path(tempfile.gettempdir()) / f"discrete-dividend-study-{os.getuid()}.lock"
    if not path.is_absolute():
        raise LocalResourceBudgetError(reason="lease path must be an absolute pathlib.Path")
    return path


def _validate_requested_threads(
    arguments: tuple[str, ...],
    budget: LocalResourceBudget,
) -> None:
    for option, maximum in (
        (_COMPUTE_THREADS_OPTION, budget.compute_threads),
        (_DATA_THREADS_OPTION, budget.data_threads),
    ):
        requested = _integer_option(arguments, option)
        if requested is not None and requested > maximum:
            raise LocalResourceBudgetError(
                reason=f"{option}={requested} exceeds local budget maximum {maximum}",
            )


def _integer_option(arguments: tuple[str, ...], option: str) -> int | None:
    values: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == option:
            if index + 1 >= len(arguments):
                raise LocalResourceBudgetError(reason=f"{option} requires one integer")
            values.append(arguments[index + 1])
            index += 2
            continue
        prefix = f"{option}="
        if value.startswith(prefix):
            values.append(value[len(prefix) :])
        index += 1
    if len(values) > 1:
        raise LocalResourceBudgetError(reason=f"{option} may be specified only once")
    if not values:
        return None
    try:
        requested = int(values[0])
    except ValueError as error:
        raise LocalResourceBudgetError(reason=f"{option} must be an integer") from error
    if requested <= 0:
        raise LocalResourceBudgetError(reason=f"{option} must be positive")
    return requested


def _load_cli_entry(name: str) -> Callable[[Sequence[str] | None], int]:
    module: ModuleType = importlib.import_module("discrete_dividend_study.cli")
    entry = getattr(module, name, None)
    if not callable(entry):
        raise LocalResourceBudgetError(reason=f"study CLI entry {name!r} is unavailable")
    return cast("Callable[[Sequence[str] | None], int]", entry)


__all__ = [
    "PROJECT_INTERACTIVE_RESOURCE_BUDGET",
    "LocalResourceBudget",
    "LocalResourceBudgetError",
    "ResourceLauncherError",
    "acquire_local_heavy_process_lease",
    "main",
]
