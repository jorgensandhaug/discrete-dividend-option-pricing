"""Standalone engine identity tests with no sibling source checkout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from discrete_dividend_study.engine_identity import (
    ENGINE_GIT_URL,
    EngineIdentityError,
    resolve_project_engine_revision,
)

if TYPE_CHECKING:
    from pathlib import Path

_REVISION = "0f8754ac099cab6a0e5050eed481058fd7b5752d"
_VERSION = "0.1.0"
_TAG = "v0.1.0"


@dataclass(frozen=True)
class _Distribution:
    version: str = _VERSION
    direct_url: str | None = None

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self.direct_url


def test_identity_resolves_from_pin_lock_and_installed_pep610_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal standalone clone needs no sibling engine repository."""
    root = _paper_root(tmp_path)
    monkeypatch.setattr(
        "discrete_dividend_study.engine_identity.distribution",
        lambda _name: _Distribution(direct_url=_direct_url(_REVISION)),
    )
    assert resolve_project_engine_revision(root) == _REVISION


@pytest.mark.parametrize(
    "case",
    [
        ("a" * 40, _REVISION, _REVISION, "project pin"),
        (_REVISION, "b" * 40, _REVISION, "installed engine distribution"),
        (_REVISION, _REVISION, "c" * 40, "installed engine distribution"),
    ],
)
def test_identity_disagreement_fails_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, str, str, str],
) -> None:
    """No alternate checkout or package source can silently replace the locked engine."""
    pin, lock_revision, installed_revision, message = case
    root = _paper_root(tmp_path, pin=pin, lock_revision=lock_revision)
    monkeypatch.setattr(
        "discrete_dividend_study.engine_identity.distribution",
        lambda _name: _Distribution(direct_url=_direct_url(installed_revision)),
    )
    with pytest.raises(EngineIdentityError, match=message):
        resolve_project_engine_revision(root)


def test_missing_direct_url_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry or manually substituted install cannot impersonate the Git-pinned engine."""
    root = _paper_root(tmp_path)
    monkeypatch.setattr(
        "discrete_dividend_study.engine_identity.distribution",
        lambda _name: _Distribution(direct_url=None),
    )
    with pytest.raises(EngineIdentityError, match="PEP 610"):
        resolve_project_engine_revision(root)


def _paper_root(
    root: Path,
    *,
    pin: str = _REVISION,
    lock_revision: str = _REVISION,
) -> Path:
    config = root / "config"
    config.mkdir()
    (config / "engine-revision.txt").write_text(f"{pin}\n", encoding="ascii")
    (root / "uv.lock").write_text(
        "\n".join(
            (
                "version = 1",
                "revision = 3",
                "",
                "[[package]]",
                'name = "finoptions-engine"',
                f'version = "{_VERSION}"',
                f'source = {{ git = "{ENGINE_GIT_URL}?tag={_TAG}#{lock_revision}" }}',
                "",
            ),
        ),
        encoding="utf-8",
    )
    return root


def _direct_url(revision: str) -> str:
    return json.dumps(
        {
            "url": ENGINE_GIT_URL,
            "vcs_info": {
                "commit_id": revision,
                "requested_revision": _TAG,
                "vcs": "git",
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
