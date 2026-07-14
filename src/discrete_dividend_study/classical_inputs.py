"""One-allocation Polars boundary for all four project classical baselines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from finoptions import (
    BatchDividends,
    CashDividendBatch,
    ContinuousDividendYieldBatch,
    EscrowedCashDividendBatch,
    EscrowedVolatilityConvention,
    RaggedCashDividends,
)

from .classical_spec import (
    PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    ClassicalDividendRepresentation,
    ClassicalSpecValueError,
    ClassicalStudySpec,
    ClassicalVariant,
    classical_study_sha256,
)
from .event_buffers import (
    EventBufferContentError,
    EventBufferPolicy,
    EventBufferTypeError,
    extract_ragged_cash_dividends,
)
from .project_pipeline import PROJECT_AS_SUBMITTED_FEATURE_DTYPES

Float64Vector = npt.NDArray[np.float64]
Float64Matrix = npt.NDArray[np.float64]

_FLOAT64_DTYPE: Final = np.dtype(np.float64)
_PREPARED_INPUT_HASH_FORMAT: Final = "project-classical-prepared-input-v1"
_SCALAR_FIELDS: Final = (
    "initial_spots",
    "strikes",
    "maturities",
    "rates",
    "volatilities",
    "forecast_yields",
    "historical_yields",
    "targets",
)


class _HashWriter(Protocol):
    """The streaming byte interface shared by hashlib implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


class ClassicalInputError(ValueError):
    """A materialized classical-pricing boundary violates its declared contract."""


@dataclass(frozen=True, slots=True)
class ClassicalInputTypeError(TypeError):
    """The caller supplied an unsupported runtime boundary type."""

    field: str
    observed: str
    expected: str

    def __str__(self) -> str:
        """Describe the required runtime type without converting the input."""
        return f"{self.field} must be {self.expected}; observed {self.observed}"


@dataclass(frozen=True, slots=True)
class ClassicalInputSchemaError(ClassicalInputError):
    """Required pricing columns are missing or use the wrong source dtype."""

    missing: tuple[str, ...]
    mismatches: tuple[tuple[str, str, str], ...]

    def __str__(self) -> str:
        """Describe schema differences without exposing licensed values."""
        return (
            f"classical input schema error: missing={self.missing!r}, "
            f"mismatches={self.mismatches!r}"
        )


@dataclass(frozen=True, slots=True)
class ClassicalInputContentError(ClassicalInputError):
    """Materialized pricing columns violate a financial or buffer invariant."""

    violations: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        """Describe aggregate violations without exposing licensed rows."""
        return f"invalid classical pricing inputs: {self.violations!r}"


@dataclass(frozen=True, slots=True, eq=False)
class PreparedClassicalInputs:
    """Immutable shared Float64 columns and one exact ragged dividend schedule.

    The eight scalar columns are views into one Fortran-ordered allocation, making every column a
    C-contiguous one-dimensional vector accepted directly by the native engine. The explicit copy
    converts the project's historical Float32 spot, strike, and target representation to pricing
    and metric Float64 once; the four variants do not duplicate those buffers.
    """

    study: ClassicalStudySpec
    initial_spots: Float64Vector
    strikes: Float64Vector
    maturities: Float64Vector
    rates: Float64Vector
    volatilities: Float64Vector
    forecast_yields: Float64Vector
    historical_yields: Float64Vector
    targets: Float64Vector
    cash_dividends: RaggedCashDividends

    def __post_init__(self) -> None:
        """Reject manual construction outside the exact shared column contract."""
        if type(self.study) is not ClassicalStudySpec:
            raise ClassicalInputTypeError(
                field="study",
                observed=type(self.study).__name__,
                expected="ClassicalStudySpec",
            )
        arrays = self.scalar_arrays()
        rows = int(self.initial_spots.size)
        for field, array in zip(_SCALAR_FIELDS, arrays, strict=True):
            _validate_float64_vector(array, field=field)
            if array.shape != (rows,):
                raise ClassicalInputContentError(violations=((f"{field}_row_mismatch", 1),))
        if rows == 0:
            raise ClassicalInputContentError(violations=(("empty_frame", 1),))
        if type(self.cash_dividends) is not RaggedCashDividends:
            raise ClassicalInputTypeError(
                field="cash_dividends",
                observed=type(self.cash_dividends).__name__,
                expected="RaggedCashDividends",
            )
        if self.cash_dividends.rows != rows:
            raise ClassicalInputContentError(violations=(("cash_dividend_row_mismatch", 1),))
        _validate_financial_content(self)
        for array in (
            *arrays,
            self.cash_dividends.offsets,
            self.cash_dividends.times,
            self.cash_dividends.amounts,
        ):
            array.flags.writeable = False

    @property
    def rows(self) -> int:
        """Return the exact contract count shared by every variant."""
        return int(self.initial_spots.size)

    def scalar_arrays(self) -> tuple[Float64Vector, ...]:
        """Return shared pricing and target vectors in their canonical hash order."""
        return (
            self.initial_spots,
            self.strikes,
            self.maturities,
            self.rates,
            self.volatilities,
            self.forecast_yields,
            self.historical_yields,
            self.targets,
        )

    def dividends_for(self, variant: ClassicalVariant) -> BatchDividends:
        """Return the engine dividend model declared by one owned variant."""
        variant_spec = self.study.variant(variant)
        representation = variant_spec.dividend_representation
        if representation is ClassicalDividendRepresentation.EXACT_CASH_JUMP:
            return CashDividendBatch(
                schedules=self.cash_dividends,
                policy=self.study.exact_cash_policy,
            )
        if representation is ClassicalDividendRepresentation.ESCROWED_PREPAID_FORWARD:
            return EscrowedCashDividendBatch(
                schedules=self.cash_dividends,
                volatility_convention=(EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN),
            )
        if representation is ClassicalDividendRepresentation.CONTINUOUS_FORECAST_YIELD:
            return ContinuousDividendYieldBatch(annualized_decimals=self.forecast_yields)
        if representation is ClassicalDividendRepresentation.CONTINUOUS_HISTORICAL_YIELD:
            return ContinuousDividendYieldBatch(annualized_decimals=self.historical_yields)
        raise ClassicalSpecValueError(
            field="dividend_representation",
            observed=representation,
            requirement="a supported ClassicalDividendRepresentation",
        )


def prepare_project_classical_inputs(
    frame: pl.DataFrame,
    study: ClassicalStudySpec = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    *,
    event_buffer_policy: EventBufferPolicy = EventBufferPolicy.REQUIRE_ZERO_COPY,
) -> PreparedClassicalInputs:
    """Prepare all four baselines from one materialized minimal project frame.

    Stateful pricing remains outside Polars. This boundary performs one visible Float64 conversion
    into column-major storage so the native engine can borrow every scalar vector. Event child
    buffers remain zero-copy unless ``ALLOW_RECHUNK`` is explicitly requested.

    Raises:
        ClassicalInputTypeError: If runtime boundary types are unsupported.
        ClassicalInputSchemaError: If source columns or exact dtypes disagree.
        ClassicalInputContentError: If materialized values violate the study contract.

    """
    if type(frame) is not pl.DataFrame:
        raise ClassicalInputTypeError(
            field="frame",
            observed=type(frame).__name__,
            expected="a materialized polars.DataFrame",
        )
    if type(study) is not ClassicalStudySpec:
        raise ClassicalInputTypeError(
            field="study",
            observed=type(study).__name__,
            expected="ClassicalStudySpec",
        )
    if type(event_buffer_policy) is not EventBufferPolicy:
        raise ClassicalInputTypeError(
            field="event_buffer_policy",
            observed=repr(event_buffer_policy),
            expected="EventBufferPolicy",
        )
    _validate_schema(frame, study)
    if frame.height == 0:
        raise ClassicalInputContentError(violations=(("empty_frame", 1),))

    scalar_sources = (
        study.spot_column,
        study.strike_column,
        study.maturity_column,
        study.rate_column,
        study.volatility_column,
        study.forecast_yield_column,
        study.historical_yield_column,
        study.target_column,
    )
    converted = frame.select(
        *(pl.col(column).cast(pl.Float64) for column in scalar_sources),
    ).to_numpy(order="fortran", writable=False, allow_copy=True)
    matrix = cast("Float64Matrix", converted)
    if (
        type(matrix) is not np.ndarray
        or matrix.dtype != _FLOAT64_DTYPE
        or matrix.shape != (frame.height, len(scalar_sources))
        or not matrix.flags.f_contiguous
    ):
        raise ClassicalInputContentError(violations=(("invalid_float64_columnar_matrix", 1),))
    matrix.flags.writeable = False
    columns = tuple(matrix[:, index] for index in range(matrix.shape[1]))
    try:
        cash_dividends = extract_ragged_cash_dividends(
            frame,
            times_column=study.cash_times_column,
            amounts_column=study.cash_amounts_column,
            policy=event_buffer_policy,
        )
    except EventBufferContentError as error:
        raise ClassicalInputContentError(violations=error.violations) from error
    except EventBufferTypeError as error:
        raise ClassicalInputTypeError(
            field=error.field,
            observed=error.observed,
            expected=error.expected,
        ) from error
    return PreparedClassicalInputs(
        study=study,
        initial_spots=columns[0],
        strikes=columns[1],
        maturities=columns[2],
        rates=columns[3],
        volatilities=columns[4],
        forecast_yields=columns[5],
        historical_yields=columns[6],
        targets=columns[7],
        cash_dividends=cash_dividends,
    )


def prepared_classical_inputs_sha256(inputs: PreparedClassicalInputs) -> str:
    """Hash the study identity and exact immutable native input bytes without serialization."""
    if type(inputs) is not PreparedClassicalInputs:
        raise ClassicalInputTypeError(
            field="inputs",
            observed=type(inputs).__name__,
            expected="PreparedClassicalInputs",
        )
    hasher = hashlib.sha256()
    _hash_text(hasher, _PREPARED_INPUT_HASH_FORMAT)
    _hash_text(hasher, classical_study_sha256(inputs.study))
    for field, array in zip(_SCALAR_FIELDS, inputs.scalar_arrays(), strict=True):
        _hash_array(hasher, field, array)
    _hash_array(hasher, "cash_offsets", inputs.cash_dividends.offsets)
    _hash_array(hasher, "cash_times", inputs.cash_dividends.times)
    _hash_array(hasher, "cash_amounts", inputs.cash_dividends.amounts)
    return hasher.hexdigest()


def _validate_schema(frame: pl.DataFrame, study: ClassicalStudySpec) -> None:
    expected = {
        column: PROJECT_AS_SUBMITTED_FEATURE_DTYPES[column] for column in study.input_columns()
    }
    schema = frame.schema
    missing = tuple(column for column in expected if column not in schema)
    mismatches = tuple(
        (column, str(dtype), str(schema[column]))
        for column, dtype in expected.items()
        if column in schema and schema[column] != dtype
    )
    if missing or mismatches:
        raise ClassicalInputSchemaError(missing=missing, mismatches=mismatches)


def _validate_float64_vector(array: object, *, field: str) -> None:
    if type(array) is not np.ndarray:
        raise ClassicalInputTypeError(
            field=field,
            observed=type(array).__name__,
            expected="a plain numpy.ndarray",
        )
    typed = cast("npt.NDArray[np.generic]", array)
    if typed.dtype != _FLOAT64_DTYPE or typed.ndim != 1 or not typed.flags.c_contiguous:
        raise ClassicalInputTypeError(
            field=field,
            observed=f"dtype={typed.dtype}, shape={typed.shape}, strides={typed.strides}",
            expected="a C-contiguous Float64 vector",
        )


def _validate_financial_content(inputs: PreparedClassicalInputs) -> None:
    violations: list[tuple[str, int]] = []
    for field, array in zip(_SCALAR_FIELDS, inputs.scalar_arrays(), strict=True):
        count = int(np.count_nonzero(~np.isfinite(array)))
        if count != 0:
            violations.append((f"nonfinite_{field}", count))
    for field, array in (
        ("initial_spots", inputs.initial_spots),
        ("strikes", inputs.strikes),
        ("maturities", inputs.maturities),
        ("volatilities", inputs.volatilities),
        ("targets", inputs.targets),
    ):
        count = int(np.count_nonzero(array <= 0.0))
        if count != 0:
            violations.append((f"nonpositive_{field}", count))
    for field, array in (
        ("forecast_yields", inputs.forecast_yields),
        ("historical_yields", inputs.historical_yields),
    ):
        count = int(np.count_nonzero(np.signbit(array)))
        if count != 0:
            violations.append((f"negative_{field}", count))
    violations.extend(_cash_event_violations(inputs))
    if violations:
        raise ClassicalInputContentError(violations=tuple(violations))


def _cash_event_violations(inputs: PreparedClassicalInputs) -> list[tuple[str, int]]:
    """Validate exact event domains in linear native-array operations before pricing."""
    schedule = inputs.cash_dividends
    times = schedule.times
    amounts = schedule.amounts
    violations: list[tuple[str, int]] = []
    for field, array in (("cash_dividend_times", times), ("cash_dividend_amounts", amounts)):
        count = int(np.count_nonzero(~np.isfinite(array)))
        if count != 0:
            violations.append((f"nonfinite_{field}", count))
    nonpositive_times = int(np.count_nonzero(times <= 0.0))
    if nonpositive_times != 0:
        violations.append(("nonpositive_cash_dividend_times", nonpositive_times))
    negative_amounts = int(np.count_nonzero(np.signbit(amounts)))
    if negative_amounts != 0:
        violations.append(("negative_cash_dividend_amounts", negative_amounts))

    if times.size > 1:
        nonincreasing = times[1:] <= times[:-1]
        boundaries = schedule.offsets[1:-1]
        internal_boundaries = boundaries[(boundaries > 0) & (boundaries < times.size)] - 1
        nonincreasing[internal_boundaries] = False
        nonincreasing_count = int(np.count_nonzero(nonincreasing))
        if nonincreasing_count != 0:
            violations.append(("nonincreasing_cash_dividend_times", nonincreasing_count))

    nonempty_rows = schedule.offsets[1:] > schedule.offsets[:-1]
    if np.any(nonempty_rows):
        last_event_indices = schedule.offsets[1:][nonempty_rows] - 1
        at_or_after_maturity = int(
            np.count_nonzero(
                times[last_event_indices] >= inputs.maturities[nonempty_rows],
            )
        )
        if at_or_after_maturity != 0:
            violations.append(("cash_dividend_rows_at_or_after_maturity", at_or_after_maturity))
    return violations


def _hash_text(hasher: _HashWriter, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    hasher.update(encoded)


def _hash_array(hasher: _HashWriter, field: str, array: npt.NDArray[np.generic]) -> None:
    _hash_text(hasher, field)
    _hash_text(hasher, array.dtype.str)
    hasher.update(array.ndim.to_bytes(8, byteorder="little", signed=False))
    for dimension in array.shape:
        hasher.update(dimension.to_bytes(8, byteorder="little", signed=False))
    hasher.update(memoryview(array).cast("B"))


__all__ = [
    "ClassicalInputContentError",
    "ClassicalInputError",
    "ClassicalInputSchemaError",
    "ClassicalInputTypeError",
    "PreparedClassicalInputs",
    "prepare_project_classical_inputs",
    "prepared_classical_inputs_sha256",
]
