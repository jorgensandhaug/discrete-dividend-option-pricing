"""Tests for the Polars-to-native neural model-input boundary."""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
import pytest
from finoptions import BatchError, BinnedCashDividends, CpuExecutor, PaddedCashDividends

from discrete_dividend_study.model_inputs import (
    EventBufferError,
    EventBufferPolicy,
    ModelInputContentError,
    ModelInputSchemaError,
    ModelInputTypeError,
    PreparedNeuralBatch,
    prepare_project_neural_batch,
    prepare_project_neural_batches,
    prepared_neural_batch_sha256,
    slice_prepared_neural_batch,
)
from discrete_dividend_study.neural_spec import PROJECT_NEURAL_STUDY_SPEC, NeuralVariant

PROJECT_ROWS = 3
FRAGMENT_COUNT = 2
SHA256_HEX_LENGTH = 64


def model_frame() -> pl.DataFrame:
    """Return a small exact-schema frame with empty and maturity-day schedules."""
    return pl.DataFrame(
        {
            "strike": [90.0, 100.0, 55.0],
            "underlying_price": [100.0, 80.0, 50.0],
            "risk_free_rate": [0.03, 0.02, 0.01],
            "call_moneyness": [100.0 / 90.0, 0.8, 50.0 / 55.0],
            "historical_volatility_90": [0.2, 0.3, 0.4],
            "yte": [0.5, 1.0, 0.75],
            "forecasted_dividend_yield": [0.02, 0.0, 0.03],
            "historical_dividend_yield": [0.01, 0.02, 0.0],
            "call_mid_price": [12.0, 3.0, 2.0],
            "forecasted_years_to_discrete_ex_dates": [
                [0.25, 0.5],
                [],
                [0.25, 0.5, 0.75],
            ],
            "forecasted_discrete_amounts": [[1.0, 2.0], [], [0.5, 1.0, 1.5]],
        },
        schema={
            "strike": pl.Float32,
            "underlying_price": pl.Float32,
            "risk_free_rate": pl.Float64,
            "call_moneyness": pl.Float32,
            "historical_volatility_90": pl.Float64,
            "yte": pl.Float64,
            "forecasted_dividend_yield": pl.Float64,
            "historical_dividend_yield": pl.Float64,
            "call_mid_price": pl.Float32,
            "forecasted_years_to_discrete_ex_dates": pl.List(pl.Float64),
            "forecasted_discrete_amounts": pl.List(pl.Float64),
        },
    )


@pytest.fixture(scope="module")
def executor() -> CpuExecutor:
    """Reuse one explicit native execution resource across tests."""
    return CpuExecutor(threads=2)


@pytest.mark.parametrize(
    ("variant", "result_type"),
    [
        (NeuralVariant.LSTM, PaddedCashDividends),
        (NeuralVariant.EMBEDDING, PaddedCashDividends),
        (NeuralVariant.BINS, BinnedCashDividends),
        (NeuralVariant.FORECAST_YIELD, type(None)),
        (NeuralVariant.HISTORICAL_YIELD, type(None)),
    ],
)
def test_each_variant_allocates_only_its_declared_dividend_input(
    executor: CpuExecutor,
    variant: NeuralVariant,
    result_type: type[object],
) -> None:
    """Sequence, bin, and scalar models cannot receive an accidental substitute."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(variant)
    batch = prepare_project_neural_batch(model_frame(), model, executor)
    assert batch.rows == PROJECT_ROWS
    assert batch.scalar_features.dtype == np.dtype(np.float32)
    assert batch.scalar_features.flags.c_contiguous
    assert batch.scalar_features.shape == (PROJECT_ROWS, len(model.scalar_columns))
    assert batch.target.dtype == np.dtype(np.float32)
    assert type(batch.dividends) is result_type


def test_scalar_feature_order_and_canonical_padded_events(executor: CpuExecutor) -> None:
    """Feature order, chronological events, explicit mask, and lengths survive the boundary."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.LSTM)
    batch = prepare_project_neural_batch(model_frame(), model, executor)
    np.testing.assert_array_equal(
        batch.scalar_features[0],
        np.array([90.0, 100.0, 0.03, 100.0 / 90.0, 0.2, 0.5], dtype=np.float32),
    )
    assert type(batch.dividends) is PaddedCashDividends
    np.testing.assert_array_equal(batch.dividends.lengths, np.array([2, 0, 3], dtype=np.uint32))
    np.testing.assert_array_equal(
        batch.dividends.mask,
        np.array(
            [
                [True, True, False, False],
                [False, False, False, False],
                [True, True, True, False],
            ],
        ),
    )
    np.testing.assert_array_equal(
        batch.dividends.times,
        np.array(
            [
                [0.25, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.25, 0.5, 0.75, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        batch.dividends.amounts,
        np.array(
            [
                [0.01, 0.02, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.01, 0.02, 0.03, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_right_closed_chronological_bins(executor: CpuExecutor) -> None:
    """Boundary events enter the preceding chronological bin and cash is conserved."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.BINS)
    batch = prepare_project_neural_batch(model_frame(), model, executor)
    assert type(batch.dividends) is BinnedCashDividends
    expected = np.zeros((3, 12), dtype=np.float32)
    expected[0, 2] = 0.01
    expected[0, 5] = 0.02
    expected[2, 2] = 0.01
    expected[2, 5] = 0.02
    expected[2, 8] = 0.03
    np.testing.assert_array_equal(batch.dividends.amounts, expected)
    np.testing.assert_allclose(
        batch.dividends.amounts.sum(axis=1),
        np.array([0.03, 0.0, 0.06], dtype=np.float32),
        rtol=0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize("variant", list(NeuralVariant))
def test_prepared_batch_hash_binds_exact_read_only_model_tensors(
    executor: CpuExecutor,
    variant: NeuralVariant,
) -> None:
    """The phase identity is deterministic, model-specific, and cannot omit a tensor family."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(variant)
    first = prepare_project_neural_batch(model_frame(), model, executor)
    second = prepare_project_neural_batch(model_frame(), model, executor)
    assert prepared_neural_batch_sha256(first) == prepared_neural_batch_sha256(second)
    assert len(prepared_neural_batch_sha256(first)) == SHA256_HEX_LENGTH
    arrays = [first.scalar_features, first.target]
    if type(first.dividends) is PaddedCashDividends:
        arrays.extend(
            (
                first.dividends.times,
                first.dividends.amounts,
                first.dividends.mask,
                first.dividends.lengths,
            ),
        )
    elif type(first.dividends) is BinnedCashDividends:
        arrays.append(first.dividends.amounts)
    assert all(not array.flags.writeable for array in arrays)

    changed = prepare_project_neural_batch(
        model_frame().with_columns(
            (pl.col("call_mid_price") + pl.lit(0.25, dtype=pl.Float32)).alias("call_mid_price"),
        ),
        model,
        executor,
    )
    assert prepared_neural_batch_sha256(changed) != prepared_neural_batch_sha256(first)


def test_multi_model_preparation_shares_only_identical_project_allocations(
    executor: CpuExecutor,
) -> None:
    """One source conversion shares targets, base scalars, and the common padded layout."""
    models = PROJECT_NEURAL_STUDY_SPEC.models
    batches = prepare_project_neural_batches(model_frame(), models, executor)
    assert tuple(batch.model for batch in batches) == models
    assert all(batch.target is batches[0].target for batch in batches)
    assert batches[0].scalar_features is batches[1].scalar_features
    assert batches[1].scalar_features is batches[2].scalar_features
    assert batches[3].scalar_features is not batches[0].scalar_features
    assert batches[4].scalar_features is not batches[0].scalar_features
    assert type(batches[0].dividends) is PaddedCashDividends
    assert batches[0].dividends is batches[1].dividends
    assert type(batches[2].dividends) is BinnedCashDividends

    independently_prepared = tuple(
        prepare_project_neural_batch(model_frame(), model, executor) for model in models
    )
    assert tuple(map(prepared_neural_batch_sha256, batches)) == tuple(
        map(prepared_neural_batch_sha256, independently_prepared),
    )


def test_prepared_batch_slices_are_exact_immutable_zero_copy_views(
    executor: CpuExecutor,
) -> None:
    """A contiguous phase slice retains byte identity without allocating tensor payloads."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.LSTM)
    complete = prepare_project_neural_batch(model_frame(), model, executor)
    phase = slice_prepared_neural_batch(complete, 0, 2)
    independent = prepare_project_neural_batch(model_frame().slice(0, 2), model, executor)
    assert prepared_neural_batch_sha256(phase) == prepared_neural_batch_sha256(independent)
    assert np.shares_memory(phase.scalar_features, complete.scalar_features)
    assert np.shares_memory(phase.target, complete.target)
    assert type(phase.dividends) is PaddedCashDividends
    assert type(complete.dividends) is PaddedCashDividends
    assert np.shares_memory(phase.dividends.times, complete.dividends.times)
    assert np.shares_memory(phase.dividends.amounts, complete.dividends.amounts)
    assert not phase.scalar_features.flags.writeable
    assert not phase.target.flags.writeable
    with pytest.raises(ModelInputTypeError, match="row bounds"):
        slice_prepared_neural_batch(complete, 1, 1)


def test_fragmented_event_children_require_explicit_rechunk(executor: CpuExecutor) -> None:
    """A zero-copy request never degrades silently into an allocation."""
    source = model_frame()
    fragmented = pl.concat([source.slice(0, 1), source.slice(1)], rechunk=False)
    assert fragmented.get_column("forecasted_discrete_amounts").n_chunks() == FRAGMENT_COUNT
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.LSTM)
    with pytest.raises(EventBufferError, match="explicit rechunk"):
        prepare_project_neural_batch(fragmented, model, executor)

    batch = prepare_project_neural_batch(
        fragmented,
        model,
        executor,
        event_buffer_policy=EventBufferPolicy.ALLOW_RECHUNK,
    )
    assert type(batch.dividends) is PaddedCashDividends
    np.testing.assert_array_equal(batch.dividends.lengths, np.array([2, 0, 3], dtype=np.uint32))


def test_schema_list_and_numeric_violations_fail_before_training(executor: CpuExecutor) -> None:
    """Missing columns, drift, unpaired lists, null children, and invalid numbers are errors."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.LSTM)
    source = model_frame()
    with pytest.raises(ModelInputSchemaError, match="risk_free_rate"):
        prepare_project_neural_batch(source.drop("risk_free_rate"), model, executor)
    with pytest.raises(ModelInputSchemaError, match="Float64"):
        prepare_project_neural_batch(
            source.with_columns(pl.col("risk_free_rate").cast(pl.Float32)),
            model,
            executor,
        )
    with pytest.raises(ModelInputContentError, match="paired_dividend_length_mismatches"):
        prepare_project_neural_batch(
            source.with_columns(
                pl.Series(
                    "forecasted_discrete_amounts",
                    [[1.0], [], [0.5, 1.0, 1.5]],
                    dtype=pl.List(pl.Float64),
                ),
            ),
            model,
            executor,
        )
    with pytest.raises(ModelInputContentError, match="null_dividend_times"):
        prepare_project_neural_batch(
            source.with_columns(
                pl.Series(
                    "forecasted_years_to_discrete_ex_dates",
                    [[0.25, None], [], [0.25, 0.5, 0.75]],
                    dtype=pl.List(pl.Float64),
                ),
            ),
            model,
            executor,
        )
    with pytest.raises(ModelInputContentError, match="nonpositive_targets"):
        prepare_project_neural_batch(
            source.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("call_mid_price")),
            model,
            executor,
        )
    with pytest.raises(ModelInputContentError, match="nonfinite_float32_scalar_features"):
        prepare_project_neural_batch(
            source.with_columns(pl.lit(1.0e300).alias("risk_free_rate")),
            model,
            executor,
        )


def test_financial_event_domain_is_revalidated_by_native_engine(executor: CpuExecutor) -> None:
    """A nonpositive event time fails; no filtering or replacement path exists."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.LSTM)
    invalid = model_frame().with_columns(
        pl.Series(
            "forecasted_years_to_discrete_ex_dates",
            [[0.0, 0.5], [], [0.25, 0.5, 0.75]],
            dtype=pl.List(pl.Float64),
        ),
    )
    with pytest.raises(BatchError, match="strictly positive"):
        prepare_project_neural_batch(invalid, model, executor)


def test_lazy_frames_and_untyped_policies_are_rejected(executor: CpuExecutor) -> None:
    """The stateful boundary never collects lazily or coerces a policy string."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.LSTM)
    lazy = cast("pl.DataFrame", model_frame().lazy())
    with pytest.raises(ModelInputTypeError, match="materialized"):
        prepare_project_neural_batch(lazy, model, executor)
    policy = cast("EventBufferPolicy", "allow_rechunk")
    with pytest.raises(ModelInputTypeError, match="EventBufferPolicy"):
        prepare_project_neural_batch(
            model_frame(),
            model,
            executor,
            event_buffer_policy=policy,
        )


@pytest.mark.parametrize(
    ("mutation", "violation"),
    [
        ("mask", "event_mask_length_mismatch"),
        ("padding", "nonzero_event_padding"),
        ("maturity", "post_maturity_events"),
        ("order", "nonchronological_events"),
    ],
)
def test_manually_corrupted_prepared_sequences_fail_immediately(
    executor: CpuExecutor,
    mutation: str,
    violation: str,
) -> None:
    """Backend callers cannot bypass mask, padding, maturity, or event-order contracts."""
    model = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.LSTM)
    valid = prepare_project_neural_batch(model_frame(), model, executor)
    assert isinstance(valid.dividends, PaddedCashDividends)
    times = valid.dividends.times.copy()
    amounts = valid.dividends.amounts.copy()
    mask = valid.dividends.mask.copy()
    if mutation == "mask":
        mask[0, 2] = True
    elif mutation == "padding":
        amounts[0, 2] = 0.01
    elif mutation == "maturity":
        times[0, 1] = 0.75
    elif mutation == "order":
        times[2, :3] = np.array([0.5, 0.25, 0.75], dtype=np.float32)
    corrupted = PaddedCashDividends(
        times=times,
        amounts=amounts,
        mask=mask,
        lengths=valid.dividends.lengths.copy(),
    )
    with pytest.raises(ModelInputContentError, match=violation):
        PreparedNeuralBatch(
            model=model,
            scalar_features=valid.scalar_features,
            target=valid.target,
            dividends=corrupted,
        )
