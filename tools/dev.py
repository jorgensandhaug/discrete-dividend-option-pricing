"""Synchronize the paper environment, rebuild changed native sources, and run one command."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

PAPER_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE_ROOT: Final = PAPER_ROOT.parent / "finoptions-engine"
ENGINE_ROOT: Final = (
    Path(
        os.environ.get("FINOPTIONS_ENGINE_ROOT", DEFAULT_ENGINE_ROOT),
    )
    .expanduser()
    .resolve()
)
VENV_ROOT: Final = PAPER_ROOT / ".venv"
STATE_PATH: Final = PAPER_ROOT / ".cache" / "dev-state.json"


class DeveloperStateError(RuntimeError):
    """The state-aware development cache is malformed."""


class DeveloperLayoutError(RuntimeError):
    """The configured sibling repository layout is incomplete."""


class DeveloperCommandError(RuntimeError):
    """A requested command is absent from the locked paper environment."""


class MissingDeveloperCommandError(DeveloperCommandError):
    """The launcher was invoked without an environment command."""

    def __str__(self) -> str:
        """Return the supported invocation form."""
        return "usage: tools/dev.py -- <environment-command> [arguments...]"


def digest(paths: list[Path]) -> str:
    """Hash path identities and contents in deterministic order."""
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.as_posix().encode())
        hasher.update(b"\0")
        if path.exists():
            hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def sync_inputs() -> list[Path]:
    """Return inputs capable of changing the Python environment resolution."""
    return [
        PAPER_ROOT / "pyproject.toml",
        PAPER_ROOT / "uv.lock",
        PAPER_ROOT / ".python-version",
        ENGINE_ROOT / "pyproject.toml",
    ]


def native_inputs() -> list[Path]:
    """Return inputs capable of changing the loaded native extension."""
    paths = [
        ENGINE_ROOT / "Cargo.toml",
        ENGINE_ROOT / "Cargo.lock",
        ENGINE_ROOT / "pyproject.toml",
        ENGINE_ROOT / "rust-toolchain.toml",
    ]
    paths.extend((ENGINE_ROOT / "crates").rglob("Cargo.toml"))
    paths.extend((ENGINE_ROOT / "crates").rglob("*.rs"))
    return paths


def load_state() -> dict[str, str]:
    """Load the local build state or fail if it is malformed."""
    if not STATE_PATH.exists():
        return {}
    parsed: object = json.loads(STATE_PATH.read_text())
    if not isinstance(parsed, dict):
        raise DeveloperStateError(STATE_PATH)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()):
        raise DeveloperStateError(STATE_PATH)
    return {str(key): str(value) for key, value in parsed.items()}


def save_state(state: dict[str, str]) -> None:
    """Atomically store the successful local build state."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
    temporary.replace(STATE_PATH)


def run(command: list[str], *, cwd: Path = PAPER_ROOT, env: dict[str, str] | None = None) -> None:
    """Run an argv command without a shell and propagate failure."""
    subprocess.run(command, cwd=cwd, env=env, check=True)


def validate_layout() -> None:
    """Fail early when the configured sibling is absent or is not the engine project."""
    required = [
        ENGINE_ROOT / "pyproject.toml",
        ENGINE_ROOT / "Cargo.toml",
        ENGINE_ROOT / "crates" / "finoptions-python" / "Cargo.toml",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise DeveloperLayoutError(tuple(missing))


def ensure_environment(state: dict[str, str]) -> None:
    """Synchronize the exact uv lock only when environment inputs changed."""
    before = digest(sync_inputs())
    if not VENV_ROOT.exists() or state.get("sync") != before:
        run(
            [
                "uv",
                "sync",
                "--all-groups",
                "--extra",
                "torch",
                "--extra",
                "mlx",
                "--locked",
            ],
        )
        state["sync"] = digest(sync_inputs())


def ensure_native(state: dict[str, str]) -> None:
    """Rebuild and install the native sibling only when its inputs changed."""
    current = digest(native_inputs())
    if state.get("native") == current:
        return

    environment = os.environ.copy()
    environment["VIRTUAL_ENV"] = str(VENV_ROOT)
    run(
        [str(VENV_ROOT / "bin" / "maturin"), "develop", "--profile", "dev-fast"],
        cwd=ENGINE_ROOT,
        env=environment,
    )
    state["native"] = current


def project_command(arguments: list[str]) -> list[str]:
    """Resolve the requested tool strictly inside the paper environment."""
    if not arguments:
        raise MissingDeveloperCommandError
    candidate = VENV_ROOT / "bin" / arguments[0]
    if not candidate.is_file():
        raise DeveloperCommandError(candidate)
    return [str(candidate), *arguments[1:]]


def main() -> None:
    """Prepare exact local state and execute the requested command."""
    arguments = sys.argv[1:]
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]

    validate_layout()
    state = load_state()
    ensure_environment(state)
    ensure_native(state)
    save_state(state)
    run(project_command(arguments))


if __name__ == "__main__":
    main()
