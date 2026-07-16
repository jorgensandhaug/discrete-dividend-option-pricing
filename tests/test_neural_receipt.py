"""Tests for exact immutable neural-run resume receipts."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from discrete_dividend_study.neural_receipt import (
    NeuralRunReceipt,
    NeuralRunReceiptError,
    load_neural_run_receipt,
    write_neural_run_receipt,
)

if TYPE_CHECKING:
    from pathlib import Path


RECEIPT = NeuralRunReceipt(
    format_version=1,
    request_sha256="a" * 64,
    run_sha256="b" * 64,
    checkpoint_sha256="c" * 64,
    prediction_sha256="d" * 64,
)


def test_receipt_round_trip_is_exact_idempotent_and_request_addressed(tmp_path: Path) -> None:
    """One request name always resolves to the same canonical cross-reference bytes."""
    first = write_neural_run_receipt(tmp_path, RECEIPT)
    second = write_neural_run_receipt(tmp_path, RECEIPT)
    assert first == second
    assert load_neural_run_receipt(first.path) == first
    assert tuple(tmp_path.iterdir()) == (first.path,)


def test_conflict_tampering_and_wrong_filename_fail_without_discovery(tmp_path: Path) -> None:
    """A request can never be rebound or recovered through an alternate receipt."""
    artifact = write_neural_run_receipt(tmp_path, RECEIPT)
    original = artifact.path.read_bytes()
    with pytest.raises(NeuralRunReceiptError, match="already bound"):
        write_neural_run_receipt(tmp_path, replace(RECEIPT, run_sha256="e" * 64))

    artifact.path.write_bytes(original + b"tampered")
    with pytest.raises(NeuralRunReceiptError, match="one newline"):
        load_neural_run_receipt(artifact.path)

    wrong_name = tmp_path / f"{'f' * 64}.receipt.json"
    wrong_name.write_bytes(original)
    with pytest.raises(NeuralRunReceiptError, match="filename"):
        load_neural_run_receipt(wrong_name)


def test_receipt_hashes_must_be_distinct_and_complete() -> None:
    """The request identity cannot masquerade as one of its result artifacts."""
    with pytest.raises(NeuralRunReceiptError, match="must be distinct"):
        replace(RECEIPT, run_sha256=RECEIPT.request_sha256)
