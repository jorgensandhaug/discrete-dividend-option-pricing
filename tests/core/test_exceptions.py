"""Python exception mutability invariants."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from discrete_dividend_study.cli import StudyCliError

if TYPE_CHECKING:
    from collections.abc import Iterator


PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "discrete_dividend_study"


def _is_frozen_dataclass(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "dataclass"
        and any(
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
    )


def test_dataclass_domain_errors_are_never_frozen() -> None:
    """Exception traceback state must remain writable on every supported Python runtime."""
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("Error"):
                continue
            if any(_is_frozen_dataclass(decorator) for decorator in node.decorator_list):
                violations.append(f"{path.name}:{node.lineno}:{node.name}")
    assert not violations


@contextmanager
def _generator_scope() -> Iterator[None]:
    yield


def test_dataclass_domain_error_survives_generator_context_unwinding() -> None:
    """Contextlib may restore an exception traceback without changing its domain type."""
    with pytest.raises(StudyCliError, match="expected domain failure"), _generator_scope():
        raise StudyCliError(reason="expected domain failure")
