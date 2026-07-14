"""Shared strict benchmark measurements."""

from __future__ import annotations

import platform
import resource
from dataclasses import dataclass
from typing import Final

_RSS_DOMAIN_ERROR: Final = "ru_maxrss must be a nonnegative plain int or float"
_OPERATING_SYSTEM_TYPE_ERROR: Final = "operating_system must be a plain str"


@dataclass(frozen=True, slots=True)
class BenchmarkPlatformError(RuntimeError):
    """A benchmark metric has no declared interpretation on this platform."""

    operating_system: str

    def __str__(self) -> str:
        """Describe the unsupported resource-unit convention."""
        return f"ru_maxrss units are undeclared on {self.operating_system!r}"


def max_rss_bytes() -> int:
    """Return peak process RSS in bytes on the two admitted benchmark platforms."""
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return normalize_max_rss_bytes(observed, platform.system())


def normalize_max_rss_bytes(observed: float, operating_system: str) -> int:
    """Normalize Darwin bytes and Linux kibibytes without guessing another unit."""
    if type(observed) not in {int, float} or observed < 0:
        raise ValueError(_RSS_DOMAIN_ERROR)
    if type(operating_system) is not str:
        raise TypeError(_OPERATING_SYSTEM_TYPE_ERROR)
    if operating_system == "Darwin":
        return int(observed)
    if operating_system == "Linux":
        return int(observed * 1024)
    raise BenchmarkPlatformError(operating_system=operating_system)


__all__ = ["BenchmarkPlatformError", "max_rss_bytes", "normalize_max_rss_bytes"]
