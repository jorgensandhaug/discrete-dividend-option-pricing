"""Exact immutable request receipts for fail-fast neural-run resume."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final, cast

_FORMAT: Final = "discrete-dividend-neural-run-receipt-v1"
_SHA256_HEX_LENGTH: Final = 64


@dataclass(frozen=True, slots=True)
class NeuralRunReceiptError(RuntimeError):
    """A request receipt violates canonical storage or cross-reference contracts."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected receipt contract."""
        return f"invalid neural run receipt: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralRunReceipt:
    """Exact mapping from one pre-training request to its immutable result artifacts."""

    format_version: int
    request_sha256: str
    run_sha256: str
    checkpoint_sha256: str
    prediction_sha256: str

    def __post_init__(self) -> None:
        """Reject incomplete, malformed, or aliased artifact identities."""
        if type(self.format_version) is not int or self.format_version != 1:
            raise NeuralRunReceiptError(reason="format_version must be the plain int 1")
        for field, value in (
            ("request_sha256", self.request_sha256),
            ("run_sha256", self.run_sha256),
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("prediction_sha256", self.prediction_sha256),
        ):
            _require_sha256(value, field=field)
        if (
            len(
                {
                    self.request_sha256,
                    self.run_sha256,
                    self.checkpoint_sha256,
                    self.prediction_sha256,
                },
            )
            != len(fields(self)) - 1
        ):
            raise NeuralRunReceiptError(reason="request and result hashes must be distinct")


@dataclass(frozen=True, slots=True)
class NeuralRunReceiptArtifact:
    """One verified canonical receipt addressed by its exact request hash."""

    path: Path
    receipt: NeuralRunReceipt

    def __post_init__(self) -> None:
        """Reject incomplete handles and filename/request disagreement."""
        if not isinstance(self.path, Path):
            raise NeuralRunReceiptError(reason="path must be pathlib.Path")
        if type(self.receipt) is not NeuralRunReceipt:
            raise NeuralRunReceiptError(reason="receipt must be NeuralRunReceipt")
        if self.path.name != f"{self.receipt.request_sha256}.receipt.json":
            raise NeuralRunReceiptError(reason="receipt path does not match its request hash")


def write_neural_run_receipt(
    directory: Path,
    receipt: NeuralRunReceipt,
) -> NeuralRunReceiptArtifact:
    """Atomically bind one request hash; a conflicting prior result is an error."""
    _validate_directory(directory)
    if type(receipt) is not NeuralRunReceipt:
        raise NeuralRunReceiptError(reason="receipt must be NeuralRunReceipt")
    payload = (_canonical_json(asdict(receipt)) + "\n").encode("ascii")
    destination = _persist_bytes(
        directory,
        destination_name=f"{receipt.request_sha256}.receipt.json",
        payload=payload,
    )
    loaded = load_neural_run_receipt(destination)
    if loaded.receipt != receipt:
        raise NeuralRunReceiptError(reason="persisted receipt differs from requested state")
    return loaded


def load_neural_run_receipt(path: Path) -> NeuralRunReceiptArtifact:
    """Load one explicitly addressed canonical receipt without directory discovery."""
    if not isinstance(path, Path):
        raise NeuralRunReceiptError(reason="path must be pathlib.Path")
    if not path.is_file() or path.is_symlink() or not path.name.endswith(".receipt.json"):
        raise NeuralRunReceiptError(reason="path must be a regular .receipt.json file")
    filename_sha256 = path.name.removesuffix(".receipt.json")
    _require_sha256(filename_sha256, field="receipt filename")
    payload = _read_bytes(path)
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise NeuralRunReceiptError(reason="receipt must be ASCII canonical JSON") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise NeuralRunReceiptError(reason="receipt must end with one newline")
    decoded = _decode_canonical_json(text[:-1])
    _require_exact_keys(decoded, NeuralRunReceipt)
    try:
        receipt = NeuralRunReceipt(**cast("dict[str, Any]", decoded))
    except (TypeError, NeuralRunReceiptError) as error:
        raise NeuralRunReceiptError(reason=f"receipt fields are invalid: {error}") from error
    if receipt.request_sha256 != filename_sha256:
        raise NeuralRunReceiptError(reason="receipt request hash does not match its filename")
    return NeuralRunReceiptArtifact(path=path, receipt=receipt)


def _persist_bytes(directory: Path, *, destination_name: str, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".neural-run-receipt-",
        suffix=".receipt.json",
        dir=directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        destination = directory / destination_name
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _read_bytes(destination) != payload:
                raise NeuralRunReceiptError(
                    reason="request hash is already bound to a different result",
                ) from None
        temporary.unlink()
    except OSError as error:
        raise NeuralRunReceiptError(reason=f"receipt write failed: {error}") from error
    finally:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def _validate_directory(directory: Path) -> None:
    if not isinstance(directory, Path) or not directory.is_dir() or directory.is_symlink():
        raise NeuralRunReceiptError(reason="directory must be an existing non-symlink directory")


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise NeuralRunReceiptError(reason=f"receipt read failed: {error}") from error


def _decode_canonical_json(value: str) -> object:
    try:
        payload: object = json.loads(value)
    except json.JSONDecodeError as error:
        raise NeuralRunReceiptError(reason="receipt is not valid JSON") from error
    if _canonical_json(payload) != value:
        raise NeuralRunReceiptError(reason="receipt JSON is not canonical")
    return payload


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise NeuralRunReceiptError(reason=f"value cannot be canonical JSON: {error}") from error


def _require_exact_keys(value: object, model: type[object]) -> None:
    if type(value) is not dict:
        raise NeuralRunReceiptError(reason="receipt must be an object")
    expected = {item.name for item in fields(model)}
    if set(cast("dict[object, object]", value)) != expected:
        raise NeuralRunReceiptError(reason="receipt keys are missing or unexpected")


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NeuralRunReceiptError(reason=f"{field} must be a lowercase SHA-256 hex digest")


__all__ = [
    "NeuralRunReceipt",
    "NeuralRunReceiptArtifact",
    "NeuralRunReceiptError",
    "load_neural_run_receipt",
    "write_neural_run_receipt",
]
