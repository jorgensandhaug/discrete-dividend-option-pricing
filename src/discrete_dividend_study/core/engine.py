"""Fail-fast identity for the exact engine installed from the locked Git source."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Final, cast
from urllib.parse import parse_qs, urlsplit, urlunsplit

from finoptions import BuildInfo, build_info

ENGINE_DISTRIBUTION: Final = "finoptions-engine"
ENGINE_GIT_URL: Final = "https://github.com/jorgensandhaug/finoptions-engine.git"
_GIT_REVISION_LENGTH: Final = 40


@dataclass(slots=True)
class EngineIdentityError(RuntimeError):
    """The pin, lock, or installed engine distribution has an ambiguous identity."""

    reason: str

    def __str__(self) -> str:
        """Describe the rejected engine identity without falling back to a checkout."""
        return f"engine identity failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class EnginePackageIdentity:
    """Exact version, repository, requested Git reference, and resolved Git commit."""

    version: str
    source_url: str
    requested_revision: str
    commit_id: str

    def __post_init__(self) -> None:
        """Reject partial package identities and noncanonical revisions."""
        if type(self.version) is not str or not self.version:
            raise EngineIdentityError(reason="engine version must be a nonempty plain string")
        if self.source_url != ENGINE_GIT_URL:
            raise EngineIdentityError(reason="engine source URL is not the approved repository")
        if type(self.requested_revision) is not str or not self.requested_revision:
            raise EngineIdentityError(reason="engine requested revision must be explicit")
        _require_git_revision(self.commit_id, field="engine commit")


def engine_build_info() -> BuildInfo:
    """Return the loaded native engine identity for run manifests."""
    return build_info()


def resolve_project_engine_revision(paper_root: Path) -> str:
    """Cross-check the project pin, uv lock, and loaded distribution's PEP 610 identity."""
    if not isinstance(paper_root, Path) or not paper_root.is_dir() or paper_root.is_symlink():
        raise EngineIdentityError(reason="paper root must be an existing non-symlink directory")
    pinned = _read_pinned_revision(paper_root / "config" / "engine-revision.txt")
    locked = _read_locked_identity(paper_root / "uv.lock")
    installed = _read_installed_identity()
    if locked != installed:
        raise EngineIdentityError(
            reason="installed engine distribution differs from the exact uv lock identity"
        )
    if locked.commit_id != pinned:
        raise EngineIdentityError(reason="uv lock engine commit differs from the project pin")
    return pinned


def _read_pinned_revision(path: Path) -> str:
    payload = _read_regular_file(path, field="engine revision pin")
    try:
        revision = payload.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise EngineIdentityError(reason="engine revision pin must be ASCII") from error
    _require_git_revision(revision, field="engine revision pin")
    return revision


def _read_locked_identity(path: Path) -> EnginePackageIdentity:
    payload = _read_regular_file(path, field="uv lock")
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise EngineIdentityError(reason="uv lock is not valid UTF-8 TOML") from error
    packages = document.get("package")
    if type(packages) is not list:
        raise EngineIdentityError(reason="uv lock has no package array")
    matches = tuple(
        item
        for item in cast("list[object]", packages)
        if type(item) is dict and item.get("name") == ENGINE_DISTRIBUTION
    )
    if len(matches) != 1:
        raise EngineIdentityError(reason="uv lock must contain exactly one engine package")
    package = cast("dict[str, object]", matches[0])
    version = package.get("version")
    source = package.get("source")
    if type(version) is not str or type(source) is not dict or set(source) != {"git"}:
        raise EngineIdentityError(reason="locked engine package must have one explicit Git source")
    git = cast("dict[str, object]", source).get("git")
    if type(git) is not str:
        raise EngineIdentityError(reason="locked engine Git source is malformed")
    parsed = urlsplit(git)
    try:
        query = parse_qs(parsed.query, strict_parsing=True)
    except ValueError as error:
        raise EngineIdentityError(reason="locked engine Git query is malformed") from error
    requested = query.get("tag")
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.fragment == ""
        or set(query) != {"tag"}
        or requested is None
        or len(requested) != 1
        or not requested[0]
    ):
        raise EngineIdentityError(reason="locked engine must use one explicit HTTPS Git tag")
    source_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return EnginePackageIdentity(
        version=version,
        source_url=source_url,
        requested_revision=requested[0],
        commit_id=parsed.fragment,
    )


def _read_installed_identity() -> EnginePackageIdentity:
    try:
        installed = distribution(ENGINE_DISTRIBUTION)
    except PackageNotFoundError as error:
        raise EngineIdentityError(reason="engine distribution is not installed") from error
    raw = installed.read_text("direct_url.json")
    if raw is None:
        raise EngineIdentityError(reason="installed engine has no PEP 610 direct_url.json")
    try:
        document: object = json.loads(raw)
    except json.JSONDecodeError as error:
        raise EngineIdentityError(reason="installed engine direct_url.json is invalid") from error
    if type(document) is not dict or set(document) != {"url", "vcs_info"}:
        raise EngineIdentityError(reason="installed engine direct URL fields are incomplete")
    mapping = cast("dict[str, object]", document)
    vcs = mapping["vcs_info"]
    if type(vcs) is not dict or set(vcs) != {"commit_id", "requested_revision", "vcs"}:
        raise EngineIdentityError(reason="installed engine VCS identity fields are incomplete")
    vcs_mapping = cast("dict[str, object]", vcs)
    if vcs_mapping["vcs"] != "git":
        raise EngineIdentityError(reason="installed engine is not a Git distribution")
    url = mapping["url"]
    requested = vcs_mapping["requested_revision"]
    commit_id = vcs_mapping["commit_id"]
    if type(url) is not str or type(requested) is not str or type(commit_id) is not str:
        raise EngineIdentityError(reason="installed engine VCS values are malformed")
    return EnginePackageIdentity(
        version=installed.version,
        source_url=url,
        requested_revision=requested,
        commit_id=commit_id,
    )


def _read_regular_file(path: Path, *, field: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise EngineIdentityError(reason=f"{field} must be an existing non-symlink file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise EngineIdentityError(reason=f"{field} cannot be read: {error}") from error


def _require_git_revision(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _GIT_REVISION_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EngineIdentityError(reason=f"{field} must be a full lowercase Git commit")


__all__ = [
    "ENGINE_DISTRIBUTION",
    "ENGINE_GIT_URL",
    "EngineIdentityError",
    "EnginePackageIdentity",
    "engine_build_info",
    "resolve_project_engine_revision",
]
