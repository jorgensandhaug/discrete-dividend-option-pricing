"""Generate the immutable project XNYS session asset from one pinned source release."""

from __future__ import annotations

import argparse
import datetime
import importlib
import os
import sys
import tempfile
from importlib import metadata
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

_CALENDAR = "XNYS"
_CROSS_CHECK_CALENDAR = "XNAS"
_PACKAGE = "exchange-calendars"
_VERSION = "4.13.2"
_START = datetime.date(2009, 1, 1)
_END = datetime.date(2025, 12, 31)
_EXPECTED_SESSIONS = 4_276
_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "discrete_dividend_study"
    / "resources"
    / "xnys_sessions_2009_2025.txt"
)


class _Timestamp(Protocol):
    def date(self) -> datetime.date:
        """Return the session's civil date."""


class _Calendar(Protocol):
    def sessions_in_range(self, start: str, end: str) -> Iterable[_Timestamp]:
        """Return an iterable session index for the inclusive interval."""


class _ExchangeCalendars(Protocol):
    def get_calendar(self, name: str) -> _Calendar:
        """Return one named exchange calendar."""


def _sessions(module: _ExchangeCalendars, name: str) -> tuple[datetime.date, ...]:
    raw = module.get_calendar(name).sessions_in_range(_START.isoformat(), _END.isoformat())
    return tuple(value.date() for value in raw)


def _render() -> str:
    observed_version = metadata.version(_PACKAGE)
    if observed_version != _VERSION:
        message = f"{_PACKAGE} must be exactly {_VERSION}; observed {observed_version}"
        raise RuntimeError(message)
    module = cast("_ExchangeCalendars", importlib.import_module("exchange_calendars"))
    primary = _sessions(module, _CALENDAR)
    cross_check = _sessions(module, _CROSS_CHECK_CALENDAR)
    if primary != cross_check:
        message = "XNYS and XNAS session dates differ over the frozen project interval"
        raise RuntimeError(message)
    if len(primary) != _EXPECTED_SESSIONS or any(
        later <= prior for prior, later in pairwise(primary)
    ):
        message = "generated session count or order differs from the frozen contract"
        raise RuntimeError(message)
    header = (
        "# generated; do not edit",
        f"# calendar={_CALENDAR}",
        "# source=exchange_calendars",
        f"# source_version={_VERSION}",
        f"# requested_start={_START.isoformat()}",
        f"# requested_end={_END.isoformat()}",
    )
    return "\n".join((*header, *(session.isoformat() for session in primary))) + "\n"


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Write or byte-check the deterministic calendar asset."""
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    payload = _render()
    if arguments.check:
        if (
            not arguments.output.is_file()
            or arguments.output.read_text(encoding="ascii") != payload
        ):
            sys.stderr.write(f"calendar asset differs: {arguments.output}\n")
            return 1
        return 0
    _write_atomic(arguments.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
