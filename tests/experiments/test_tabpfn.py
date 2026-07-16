"""Tests for the bounded project cash-bin to TabPFN input boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from finoptions import CpuExecutor

from discrete_dividend_study.experiments.tabpfn import (
    PROJECT_TABPFN_BINNED_INPUT_SPEC,
    PreparedTabPfnBatch,
    TabPfnBinnedInputSpec,
    TabPfnInputError,
    prepare_project_tabpfn_binned_batch,
    prepared_tabpfn_batch_sha256,
    slice_prepared_tabpfn_batch,
)
from discrete_dividend_study.models.neural.inputs import prepare_project_neural_batch
from discrete_dividend_study.models.neural.specification import PROJECT_STUDY, NeuralVariant
from tests.models.neural.test_inputs import model_frame


def _prepared() -> PreparedTabPfnBatch:
    model = PROJECT_STUDY.model(NeuralVariant.BINS)
    neural = prepare_project_neural_batch(model_frame(), model, CpuExecutor(threads=2))
    return prepare_project_tabpfn_binned_batch(neural, PROJECT_TABPFN_BINNED_INPUT_SPEC)


def test_binned_boundary_is_one_exact_immutable_eighteen_column_table() -> None:
    """Scalar order, chronological cash bins, target identity, and storage are unchanged."""
    neural = prepare_project_neural_batch(
        model_frame(),
        PROJECT_STUDY.model(NeuralVariant.BINS),
        CpuExecutor(threads=1),
    )
    prepared = prepare_project_tabpfn_binned_batch(neural, PROJECT_TABPFN_BINNED_INPUT_SPEC)
    assert prepared.spec.feature_names[:6] == prepared.spec.source_model.scalar_columns
    assert prepared.spec.feature_names[6:] == tuple(
        f"forecast_cash_bin_{index:02d}_over_spot" for index in range(1, 13)
    )
    assert prepared.features.shape == (3, 18)
    assert prepared.features.dtype == np.dtype(np.float32)
    assert prepared.features.flags.c_contiguous
    assert not prepared.features.flags.writeable
    assert not prepared.target.flags.writeable
    assert prepared.target is neural.target
    np.testing.assert_array_equal(
        prepared.features[:, :6],
        neural.scalar_features,
    )
    assert neural.dividends is not None
    np.testing.assert_array_equal(prepared.features[:, 6:], neural.dividends.amounts)
    np.testing.assert_array_equal(prepared.target, model_frame()["call_mid_price"].to_numpy())


def test_hash_is_deterministic_and_phase_slices_are_zero_copy() -> None:
    """A rolling phase has its own byte identity while retaining the full-horizon allocation."""
    complete = _prepared()
    repeated = _prepared()
    phase = slice_prepared_tabpfn_batch(complete, 1, 3)
    assert prepared_tabpfn_batch_sha256(complete) == prepared_tabpfn_batch_sha256(repeated)
    assert prepared_tabpfn_batch_sha256(phase) != prepared_tabpfn_batch_sha256(complete)
    assert np.shares_memory(phase.features, complete.features)
    assert np.shares_memory(phase.target, complete.target)
    assert phase.source_prepared_batch_sha256 == complete.source_prepared_batch_sha256
    with pytest.raises(TabPfnInputError, match="row bounds"):
        slice_prepared_tabpfn_batch(complete, 2, 2)


def test_boundary_rejects_other_representations_and_manual_schema_drift() -> None:
    """There is no flattening fallback for sequences, yields, or renamed bins."""
    lstm = prepare_project_neural_batch(
        model_frame(),
        PROJECT_STUDY.model(NeuralVariant.LSTM),
        CpuExecutor(threads=1),
    )
    with pytest.raises(TabPfnInputError, match="batch model disagrees"):
        prepare_project_tabpfn_binned_batch(lstm, PROJECT_TABPFN_BINNED_INPUT_SPEC)
    with pytest.raises(TabPfnInputError, match="feature_names disagree"):
        TabPfnBinnedInputSpec(
            source_model=PROJECT_TABPFN_BINNED_INPUT_SPEC.source_model,
            feature_names=(*PROJECT_TABPFN_BINNED_INPUT_SPEC.feature_names[:-1], "renamed"),
        )
    prepared = _prepared()
    with pytest.raises(TabPfnInputError, match="read-only"):
        replace(prepared, features=prepared.features.copy())
