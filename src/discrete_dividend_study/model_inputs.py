"""Strict materialized boundary from project features to neural model arrays."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from finoptions import (
    BinnedCashDividends,
    CpuExecutor,
    PaddedCashDividends,
    RaggedCashDividends,
    bin_cash_dividends,
    pack_padded_cash_dividends,
)

from .neural_spec import (
    PROJECT_NEURAL_STUDY_SPEC,
    BinnedDividendInput,
    DirectDividendLstmInput,
    DividendYieldInput,
    EventAttentionInput,
    NeuralModelSpec,
)
from .project_pipeline import PROJECT_AS_SUBMITTED_FEATURE_DTYPES

Float32Vector = npt.NDArray[np.float32]
Float32Matrix = npt.NDArray[np.float32]
Float64Vector = npt.NDArray[np.float64]
Int64Vector = npt.NDArray[np.int64]
type DividendTensor = PaddedCashDividends | BinnedCashDividends

_FLOAT32_DTYPE: Final = np.dtype(np.float32)
_FLOAT64_DTYPE: Final = np.dtype(np.float64)
_PREPARED_BATCH_HASH_FORMAT: Final = "discrete-dividend-prepared-neural-batch-v1"


class _HashWriter(Protocol):
    """The buffer-writing subset shared by hashlib hash implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


class ModelInputError(ValueError):
    """Base class for an invalid materialized neural-model boundary."""


@dataclass(frozen=True, slots=True)
class ModelInputTypeError(TypeError):
    """The caller passed the wrong runtime boundary type."""

    field: str
    observed: str
    expected: str

    def __str__(self) -> str:
        """Describe the required runtime type without coercion."""
        return f"{self.field} must be {self.expected}; observed {self.observed}"


@dataclass(frozen=True, slots=True)
class ModelInputSchemaError(ModelInputError):
    """Required model columns are missing or have incorrect dtypes."""

    missing: tuple[str, ...]
    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe schema differences without exposing licensed rows."""
        return f"model input schema error: missing={self.missing!r}, mismatches={self.mismatches!r}"


@dataclass(frozen=True, slots=True)
class ModelInputContentError(ModelInputError):
    """Materialized model rows violate numeric or paired-list contracts."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing licensed rows."""
        return f"invalid model input content: {self.violations!r}"


@dataclass(frozen=True, slots=True)
class EventBufferError(ModelInputError):
    """A list child buffer cannot satisfy the declared copy policy."""

    column: str
    policy: EventBufferPolicy
    reason: str

    def __str__(self) -> str:
        """Describe the failed buffer contract."""
        return f"event column {self.column!r} violates {self.policy.value}: {self.reason}"


class EventBufferPolicy(StrEnum):
    """Explicit handling of fragmented Polars event child buffers."""

    REQUIRE_ZERO_COPY = "require_zero_copy"
    ALLOW_RECHUNK = "allow_rechunk"


@dataclass(frozen=True, slots=True)
class RaggedDividendColumns:
    """Semantic columns used to form the Arrow-compatible ragged event view."""

    times: str
    amounts: str
    maturity: str
    spot: str

    def __post_init__(self) -> None:
        """Reject blank or aliased source columns."""
        values = (self.times, self.amounts, self.maturity, self.spot)
        if any(type(value) is not str or not value for value in values) or len(set(values)) != len(
            values
        ):
            raise ModelInputTypeError(
                field="ragged dividend columns",
                observed=repr(values),
                expected="four distinct nonempty plain strings",
            )


PROJECT_RAGGED_DIVIDEND_COLUMNS: Final = RaggedDividendColumns(
    times="forecasted_years_to_discrete_ex_dates",
    amounts="forecasted_discrete_amounts",
    maturity="yte",
    spot="underlying_price",
)


@dataclass(frozen=True, slots=True)
class PreparedNeuralBatch:
    """Owned scalar/target arrays and only the declared dividend tensor."""

    model: NeuralModelSpec
    scalar_features: Float32Matrix
    target: Float32Vector
    dividends: DividendTensor | None

    def __post_init__(self) -> None:
        """Reject manually assembled batches outside the model-input contract."""
        if type(self.model) is not NeuralModelSpec:
            raise ModelInputTypeError(
                field="model",
                observed=type(self.model).__name__,
                expected="NeuralModelSpec",
            )
        _validate_array(
            self.scalar_features,
            field="scalar_features",
            dtype=_FLOAT32_DTYPE,
            dimensions=2,
        )
        _validate_array(self.target, field="target", dtype=_FLOAT32_DTYPE, dimensions=1)
        if self.scalar_features.shape[1] != len(self.model.scalar_columns):
            raise ModelInputContentError(
                violations=(("incorrect_scalar_feature_width", 1),),
            )
        if self.target.shape != (self.scalar_features.shape[0],):
            raise ModelInputContentError(violations=(("target_row_mismatch", 1),))
        if not np.isfinite(self.scalar_features).all() or not np.isfinite(self.target).all():
            raise ModelInputContentError(violations=(("nonfinite_model_arrays", 1),))
        _validate_dividend_tensor(self)
        if self.dividends is not None and self.dividends.rows != self.scalar_features.shape[0]:
            raise ModelInputContentError(violations=(("dividend_tensor_row_mismatch", 1),))
        _freeze_batch_arrays(self)

    @property
    def rows(self) -> int:
        """Return the exact number of model observations."""
        return int(self.scalar_features.shape[0])


def prepared_neural_batch_sha256(batch: PreparedNeuralBatch) -> str:
    """Hash the exact model identity, tensor layouts, dtypes, and materialized bytes.

    The function streams C-contiguous buffers directly into SHA-256; it does not serialize or copy
    a full phase. The result identifies the actual float32 model boundary rather than a mutable
    source path or a provider-specific row representation.
    """
    if type(batch) is not PreparedNeuralBatch:
        raise ModelInputTypeError(
            field="batch",
            observed=type(batch).__name__,
            expected="PreparedNeuralBatch",
        )
    hasher = hashlib.sha256()
    _hash_text(hasher, _PREPARED_BATCH_HASH_FORMAT)
    _hash_text(hasher, batch.model.variant.value)
    _hash_text(hasher, batch.model.submitted_result_name)
    for column in batch.model.scalar_columns:
        _hash_text(hasher, column)
    _hash_text(hasher, batch.model.target_column)
    _hash_array(hasher, "scalar_features", batch.scalar_features)
    _hash_array(hasher, "target", batch.target)
    if type(batch.dividends) is PaddedCashDividends:
        _hash_text(hasher, "padded_cash_dividends")
        _hash_array(hasher, "dividend_times", batch.dividends.times)
        _hash_array(hasher, "dividend_amounts", batch.dividends.amounts)
        _hash_array(hasher, "dividend_mask", batch.dividends.mask)
        _hash_array(hasher, "dividend_lengths", batch.dividends.lengths)
    elif type(batch.dividends) is BinnedCashDividends:
        _hash_text(hasher, "binned_cash_dividends")
        _hash_array(hasher, "binned_amounts", batch.dividends.amounts)
    else:
        _hash_text(hasher, "no_dividend_tensor")
    return hasher.hexdigest()


def prepare_project_neural_batch(
    frame: pl.DataFrame,
    model: NeuralModelSpec,
    executor: CpuExecutor,
    *,
    event_buffer_policy: EventBufferPolicy = EventBufferPolicy.REQUIRE_ZERO_COPY,
    dividend_columns: RaggedDividendColumns = PROJECT_RAGGED_DIVIDEND_COLUMNS,
) -> PreparedNeuralBatch:
    """Materialize one strict project-model batch with bounded, visible allocations.

    Scalar features and targets are deliberately converted to contiguous float32 arrays because
    that is the declared model precision. Ragged offsets and float64 spots are bounded temporary
    allocations. Event time and amount children are borrowed from Polars unless ``ALLOW_RECHUNK``
    explicitly permits consolidation of fragmented chunks. Only sequence models allocate padded
    tensors; only the binned model allocates bins; scalar-yield models allocate neither.

    This is a stateful training boundary and therefore accepts a materialized ``DataFrame`` rather
    than pretending model input construction is a lazy expression.
    """
    _validate_request_types(
        frame=frame,
        model=model,
        executor=executor,
        event_buffer_policy=event_buffer_policy,
        dividend_columns=dividend_columns,
    )
    if frame.height == 0:
        raise ModelInputContentError(violations=(("empty_frame", 1),))

    dividend_input = model.dividend_input
    needs_events = type(dividend_input) is not DividendYieldInput
    required = _required_columns(model, dividend_columns, needs_events=needs_events)
    expected = _expected_dtypes(required, dividend_columns)
    _validate_schema(frame, expected)

    scalar_features = _scalar_matrix(frame, model.scalar_columns)
    target = _float32_vector(frame.get_column(model.target_column), field=model.target_column)
    _validate_model_arrays(model, scalar_features, target)
    dividends: DividendTensor | None = None
    if needs_events:
        ragged = _ragged_dividends(
            frame,
            dividend_columns,
            policy=event_buffer_policy,
        )
        maturities = _float64_vector(
            frame.get_column(dividend_columns.maturity),
            field=dividend_columns.maturity,
        )
        spots = _float64_vector(
            frame.get_column(dividend_columns.spot).cast(pl.Float64),
            field=dividend_columns.spot,
        )
        if type(dividend_input) in {DirectDividendLstmInput, EventAttentionInput}:
            sequence_input = cast("DirectDividendLstmInput | EventAttentionInput", dividend_input)
            dividends = pack_padded_cash_dividends(
                dividends=ragged,
                maturities=maturities,
                spots=spots,
                spec=sequence_input.layout,
                executor=executor,
            )
        elif type(dividend_input) is BinnedDividendInput:
            dividends = bin_cash_dividends(
                dividends=ragged,
                maturities=maturities,
                spots=spots,
                spec=dividend_input.layout,
                executor=executor,
            )

    return PreparedNeuralBatch(
        model=model,
        scalar_features=scalar_features,
        target=target,
        dividends=dividends,
    )


def project_neural_input_columns(model: NeuralModelSpec) -> tuple[str, ...]:
    """Return the minimal ordered materialized column set for one project model."""
    if type(model) is not NeuralModelSpec or model not in PROJECT_NEURAL_STUDY_SPEC.models:
        raise ModelInputTypeError(
            field="model",
            observed=type(model).__name__,
            expected="a model owned by PROJECT_NEURAL_STUDY_SPEC",
        )
    needs_events = type(model.dividend_input) is not DividendYieldInput
    return _required_columns(
        model,
        PROJECT_RAGGED_DIVIDEND_COLUMNS,
        needs_events=needs_events,
    )


def _validate_request_types(
    *,
    frame: object,
    model: object,
    executor: object,
    event_buffer_policy: object,
    dividend_columns: object,
) -> None:
    checks = (
        (frame, pl.DataFrame, "frame", "a materialized polars.DataFrame"),
        (executor, CpuExecutor, "executor", "CpuExecutor"),
        (event_buffer_policy, EventBufferPolicy, "event_buffer_policy", "EventBufferPolicy"),
        (dividend_columns, RaggedDividendColumns, "dividend_columns", "RaggedDividendColumns"),
    )
    for value, expected_type, field, expected in checks:
        if type(value) is not expected_type:
            raise ModelInputTypeError(
                field=field,
                observed=type(value).__name__,
                expected=expected,
            )
    if type(model) is not NeuralModelSpec or model not in PROJECT_NEURAL_STUDY_SPEC.models:
        raise ModelInputTypeError(
            field="model",
            observed=type(model).__name__,
            expected="a model owned by PROJECT_NEURAL_STUDY_SPEC",
        )


def _required_columns(
    model: NeuralModelSpec,
    columns: RaggedDividendColumns,
    *,
    needs_events: bool,
) -> tuple[str, ...]:
    required = (*model.scalar_columns, model.target_column)
    if needs_events:
        required = (*required, columns.times, columns.amounts, columns.maturity, columns.spot)
    return tuple(dict.fromkeys(required))


def _expected_dtypes(
    required: tuple[str, ...],
    columns: RaggedDividendColumns,
) -> dict[str, type[pl.DataType] | pl.DataType]:
    expected: dict[str, type[pl.DataType] | pl.DataType] = {}
    for name in required:
        if name in {columns.times, columns.amounts}:
            expected[name] = pl.List(pl.Float64)
        elif name == columns.maturity:
            expected[name] = pl.Float64
        elif name == columns.spot:
            expected[name] = pl.Float32
        elif name in PROJECT_AS_SUBMITTED_FEATURE_DTYPES:
            expected[name] = PROJECT_AS_SUBMITTED_FEATURE_DTYPES[name]
        else:
            raise ModelInputSchemaError(
                missing=(),
                mismatches=((name, "a declared project feature", "unknown model column"),),
            )
    return expected


def _validate_schema(
    frame: pl.DataFrame,
    expected: dict[str, type[pl.DataType] | pl.DataType],
) -> None:
    schema = frame.schema
    missing = tuple(name for name in expected if name not in schema)
    mismatches = tuple(
        (name, str(dtype), str(schema[name]))
        for name, dtype in expected.items()
        if name in schema and schema[name] != dtype
    )
    if missing or mismatches:
        raise ModelInputSchemaError(missing=missing, mismatches=mismatches)


def _validate_model_arrays(
    model: NeuralModelSpec,
    scalar_features: Float32Matrix,
    target: Float32Vector,
) -> None:
    indices = {column: index for index, column in enumerate(model.scalar_columns)}
    violations: list[tuple[str, int]] = []
    positive_columns = {
        "strike": "nonpositive_strikes",
        "underlying_price": "nonpositive_spots",
        "call_moneyness": "nonpositive_call_moneyness",
        "yte": "nonpositive_maturities",
    }
    for column, violation in positive_columns.items():
        count = int(np.count_nonzero(scalar_features[:, indices[column]] <= 0.0))
        if count != 0:
            violations.append((violation, count))
    nonnegative_columns = {
        "historical_volatility_90": "negative_historical_volatility",
        "forecasted_dividend_yield": "negative_forecasted_dividend_yields",
        "historical_dividend_yield": "negative_historical_dividend_yields",
    }
    for column, violation in nonnegative_columns.items():
        if column in indices:
            count = int(np.count_nonzero(np.signbit(scalar_features[:, indices[column]])))
            if count != 0:
                violations.append((violation, count))
    nonpositive_targets = int(np.count_nonzero(target <= 0.0))
    if nonpositive_targets != 0:
        violations.append(("nonpositive_targets", nonpositive_targets))
    if violations:
        raise ModelInputContentError(violations=tuple(violations))


def _scalar_matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> Float32Matrix:
    converted = frame.select(
        *(pl.col(column).cast(pl.Float32) for column in columns),
    ).to_numpy(order="c", writable=False, allow_copy=True)
    result = cast("Float32Matrix", converted)
    _validate_array(result, field="scalar_features", dtype=_FLOAT32_DTYPE, dimensions=2)
    if not np.isfinite(result).all():
        raise ModelInputContentError(violations=(("nonfinite_float32_scalar_features", 1),))
    return result


def _float32_vector(series: pl.Series, *, field: str) -> Float32Vector:
    converted = series.cast(pl.Float32).to_numpy(writable=False, allow_copy=True)
    result = cast("Float32Vector", converted)
    _validate_array(result, field=field, dtype=_FLOAT32_DTYPE, dimensions=1)
    if not np.isfinite(result).all():
        raise ModelInputContentError(violations=((f"nonfinite_float32_{field}", 1),))
    return result


def _float64_vector(series: pl.Series, *, field: str) -> Float64Vector:
    converted = series.to_numpy(writable=False, allow_copy=True)
    result = cast("Float64Vector", converted)
    _validate_array(result, field=field, dtype=_FLOAT64_DTYPE, dimensions=1)
    return result


def _ragged_dividends(
    frame: pl.DataFrame,
    columns: RaggedDividendColumns,
    *,
    policy: EventBufferPolicy,
) -> RaggedCashDividends:
    times = _event_list_series(frame.get_column(columns.times), column=columns.times, policy=policy)
    amounts = _event_list_series(
        frame.get_column(columns.amounts),
        column=columns.amounts,
        policy=policy,
    )
    violations: list[tuple[str, int]] = []
    if times.null_count() != 0:
        violations.append(("null_dividend_time_lists", times.null_count()))
    if amounts.null_count() != 0:
        violations.append(("null_dividend_amount_lists", amounts.null_count()))
    time_lengths = times.list.len()
    amount_lengths = amounts.list.len()
    paired_length_mismatches = cast("int", (time_lengths != amount_lengths).sum() or 0)
    if paired_length_mismatches != 0:
        violations.append(("paired_dividend_length_mismatches", paired_length_mismatches))

    exploded_times = times.explode(empty_as_null=False, keep_nulls=True)
    exploded_amounts = amounts.explode(empty_as_null=False, keep_nulls=True)
    if exploded_times.null_count() != 0:
        violations.append(("null_dividend_times", exploded_times.null_count()))
    if exploded_amounts.null_count() != 0:
        violations.append(("null_dividend_amounts", exploded_amounts.null_count()))
    if violations:
        raise ModelInputContentError(violations=tuple(violations))

    event_count = cast("int", time_lengths.cast(pl.UInt64).sum() or 0)
    lengths = cast(
        "Int64Vector",
        time_lengths.cast(pl.Int64).to_numpy(writable=False, allow_copy=True),
    )
    offsets = np.empty(frame.height + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    if int(offsets[-1]) != event_count:
        raise ModelInputContentError(violations=(("ragged_offset_overflow", 1),))

    time_values = _event_child_array(exploded_times, column=columns.times, policy=policy)
    amount_values = _event_child_array(exploded_amounts, column=columns.amounts, policy=policy)
    if time_values.shape != (event_count,) or amount_values.shape != (event_count,):
        raise ModelInputContentError(violations=(("ragged_event_count_mismatch", 1),))
    return RaggedCashDividends(offsets=offsets, times=time_values, amounts=amount_values)


def _event_list_series(
    series: pl.Series,
    *,
    column: str,
    policy: EventBufferPolicy,
) -> pl.Series:
    if series.n_chunks() == 1:
        return series
    if policy is EventBufferPolicy.REQUIRE_ZERO_COPY:
        raise EventBufferError(
            column=column,
            policy=policy,
            reason="fragmented list storage requires an explicit rechunk",
        )
    if policy is EventBufferPolicy.ALLOW_RECHUNK:
        return series.rechunk()
    raise ModelInputTypeError(
        field="event_buffer_policy",
        observed=repr(policy),
        expected="EventBufferPolicy",
    )


def _event_child_array(
    series: pl.Series,
    *,
    column: str,
    policy: EventBufferPolicy,
) -> Float64Vector:
    candidate = series
    try:
        converted = candidate.to_numpy(writable=False, allow_copy=False)
    except RuntimeError as error:
        if policy is EventBufferPolicy.REQUIRE_ZERO_COPY:
            raise EventBufferError(
                column=column,
                policy=policy,
                reason="fragmented child buffer requires an explicit rechunk",
            ) from error
        if policy is not EventBufferPolicy.ALLOW_RECHUNK:
            raise ModelInputTypeError(
                field="event_buffer_policy",
                observed=repr(policy),
                expected="EventBufferPolicy",
            ) from error
        candidate = candidate.rechunk()
        try:
            converted = candidate.to_numpy(writable=False, allow_copy=False)
        except RuntimeError as rechunk_error:
            raise EventBufferError(
                column=column,
                policy=policy,
                reason="rechunked child buffer still cannot be borrowed by NumPy",
            ) from rechunk_error
    result = cast("Float64Vector", converted)
    _validate_array(result, field=column, dtype=_FLOAT64_DTYPE, dimensions=1)
    return result


def _validate_array(
    value: object,
    *,
    field: str,
    dtype: np.dtype[np.generic],
    dimensions: int,
) -> None:
    if type(value) is not np.ndarray:
        raise ModelInputTypeError(
            field=field,
            observed=type(value).__name__,
            expected="a plain numpy.ndarray",
        )
    array = cast("npt.NDArray[np.generic]", value)
    if array.dtype != dtype:
        raise ModelInputTypeError(
            field=field,
            observed=str(array.dtype),
            expected=f"dtype {dtype}",
        )
    if array.ndim != dimensions:
        raise ModelInputTypeError(
            field=field,
            observed=repr(array.shape),
            expected=f"{dimensions} dimensions",
        )
    if not array.flags.c_contiguous:
        raise ModelInputTypeError(
            field=field,
            observed="non-C-contiguous ndarray",
            expected="C-contiguous storage",
        )


def _validate_padded_tensor(batch: PreparedNeuralBatch, capacity: int) -> None:
    dividends = cast("PaddedCashDividends", batch.dividends)
    if dividends.capacity != capacity:
        raise ModelInputContentError(violations=(("incorrect_event_capacity", 1),))
    if not np.isfinite(dividends.times).all() or not np.isfinite(dividends.amounts).all():
        raise ModelInputContentError(violations=(("nonfinite_padded_dividends", 1),))
    expected_mask = np.arange(capacity, dtype=np.uint32)[None, :] < dividends.lengths[:, None]
    if not np.array_equal(dividends.mask, expected_mask):
        raise ModelInputContentError(violations=(("event_mask_length_mismatch", 1),))
    if np.any(dividends.times[~dividends.mask] != 0.0) or np.any(
        dividends.amounts[~dividends.mask] != 0.0
    ):
        raise ModelInputContentError(violations=(("nonzero_event_padding", 1),))
    if np.any(dividends.times[dividends.mask] < 0.0):
        raise ModelInputContentError(violations=(("negative_event_times", 1),))
    maturity_index = batch.model.scalar_columns.index("yte")
    if np.any(dividends.times > batch.scalar_features[:, maturity_index, None]):
        raise ModelInputContentError(violations=(("post_maturity_events", 1),))
    adjacent_valid = dividends.mask[:, :-1] & dividends.mask[:, 1:]
    adjacent_decrease = dividends.times[:, 1:] < dividends.times[:, :-1]
    if np.any(adjacent_valid & adjacent_decrease):
        raise ModelInputContentError(violations=(("nonchronological_events", 1),))


def _validate_dividend_tensor(batch: PreparedNeuralBatch) -> None:
    dividend_input = batch.model.dividend_input
    if (
        type(dividend_input) is DirectDividendLstmInput
        or type(dividend_input) is EventAttentionInput
    ):
        if type(batch.dividends) is not PaddedCashDividends:
            raise ModelInputTypeError(
                field="dividends",
                observed=type(batch.dividends).__name__,
                expected="PaddedCashDividends",
            )
        _validate_padded_tensor(batch, dividend_input.layout.capacity)
    elif type(dividend_input) is BinnedDividendInput:
        if type(batch.dividends) is not BinnedCashDividends:
            raise ModelInputTypeError(
                field="dividends",
                observed=type(batch.dividends).__name__,
                expected="BinnedCashDividends",
            )
        _validate_binned_tensor(batch, dividend_input.layout.bins)
    elif type(dividend_input) is DividendYieldInput and batch.dividends is not None:
        raise ModelInputTypeError(
            field="dividends",
            observed=type(batch.dividends).__name__,
            expected="None for scalar-yield models",
        )


def _validate_binned_tensor(batch: PreparedNeuralBatch, bins: int) -> None:
    dividends = cast("BinnedCashDividends", batch.dividends)
    if dividends.bins != bins:
        raise ModelInputContentError(violations=(("incorrect_dividend_bin_count", 1),))
    if not np.isfinite(dividends.amounts).all():
        raise ModelInputContentError(violations=(("nonfinite_binned_dividends", 1),))


def _freeze_batch_arrays(batch: PreparedNeuralBatch) -> None:
    arrays = [batch.scalar_features, batch.target]
    if type(batch.dividends) is PaddedCashDividends:
        arrays.extend(
            (
                batch.dividends.times,
                batch.dividends.amounts,
                batch.dividends.mask,
                batch.dividends.lengths,
            ),
        )
    elif type(batch.dividends) is BinnedCashDividends:
        arrays.append(batch.dividends.amounts)
    for array in arrays:
        array.flags.writeable = False


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _hash_array(
    hasher: _HashWriter,
    label: str,
    array: npt.NDArray[np.generic],
) -> None:
    _hash_text(hasher, label)
    _hash_text(hasher, array.dtype.str)
    hasher.update(array.ndim.to_bytes(8, byteorder="little", signed=False))
    for dimension in array.shape:
        hasher.update(dimension.to_bytes(8, byteorder="little", signed=False))
    hasher.update(memoryview(array).cast("B"))


__all__ = [
    "PROJECT_RAGGED_DIVIDEND_COLUMNS",
    "EventBufferError",
    "EventBufferPolicy",
    "ModelInputContentError",
    "ModelInputError",
    "ModelInputSchemaError",
    "ModelInputTypeError",
    "PreparedNeuralBatch",
    "RaggedDividendColumns",
    "prepare_project_neural_batch",
    "prepared_neural_batch_sha256",
    "project_neural_input_columns",
]
