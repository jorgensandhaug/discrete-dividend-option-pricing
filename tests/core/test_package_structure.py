"""Guards for the intentionally small public package surface."""

from pathlib import Path
from typing import Final

import discrete_dividend_study

_ROOT_MODULES: Final = frozenset({"__init__.py", "cli.py", "paper.py"})
_PUBLIC_NAMES: Final = frozenset(
    {
        "DIRECT_ATTENTION",
        "DIRECT_LSTM",
        "FORECAST_YIELD",
        "HISTORICAL_YIELD",
        "PROJECT_MODELS",
        "PROJECT_STUDY",
        "TWELVE_BINS",
        "__version__",
    }
)


def test_package_root_contains_only_entry_points() -> None:
    """Implementation modules belong to explicit domain packages."""
    package_file = discrete_dividend_study.__file__
    assert package_file is not None
    package_root = Path(package_file).parent
    root_modules = frozenset(path.name for path in package_root.glob("*.py"))
    assert root_modules == _ROOT_MODULES


def test_public_api_contains_only_paper_identities() -> None:
    """Internal helpers cannot silently become root-level API."""
    assert frozenset(discrete_dividend_study.__all__) == _PUBLIC_NAMES
