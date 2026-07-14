"""Verified Polars-to-native strike groups for the reconstructed classical study."""

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
    LsmStrikeGroups,
    RaggedCashDividends,
)

from .classical_inputs import (
    ClassicalInputContentError,
    ClassicalInputSchemaError,
    ClassicalInputTypeError,
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
UInt32Vector = npt.NDArray[np.uint32]

_GROUP_KEYS: Final = ("ticker", "quote_date", "expiry_date")
_GROUP_STATE_FIELDS: Final = (
    "underlying_price",
    "yte",
    "risk_free_rate",
    "historical_volatility_90",
    "forecasted_dividend_yield",
    "historical_dividend_yield",
    "forecasted_years_to_discrete_ex_dates",
    "forecasted_discrete_amounts",
)
_GROUP_SCALAR_FIELDS: Final = (
    "initial_spots",
    "maturities",
    "rates",
    "volatilities",
    "forecast_yields",
    "historical_yields",
)
_GROUP_SOURCE_SCALARS: Final = _GROUP_STATE_FIELDS[:6]
_ROW_FIELDS: Final = ("strikes", "targets")
_ROW_SOURCE_SCALARS: Final = ("strike", "call_mid_price")
_FLOAT64_DTYPE: Final = np.dtype(np.float64)
_UINT32_DTYPE: Final = np.dtype(np.uint32)
_MAX_U32: Final = (2**32) - 1
_PREPARED_GROUP_HASH_FORMAT: Final = "project-classical-prepared-strike-groups-v1"


class _HashWriter(Protocol):
    """The streaming byte interface shared by hashlib implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(frozen=True, slots=True, eq=False)
class PreparedClassicalStrikeGroups:
    """Immutable group-level path states and row-level strikes/targets."""

    study: ClassicalStudySpec
    group_offsets: UInt32Vector
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
        """Reject construction outside the exact compact two-axis contract."""
        if type(self.study) is not ClassicalStudySpec:
            raise ClassicalInputTypeError(
                field="study",
                observed=type(self.study).__name__,
                expected="ClassicalStudySpec",
            )
        _validate_vector(self.group_offsets, field="group_offsets", dtype=_UINT32_DTYPE)
        for field, array in zip(
            (*_GROUP_SCALAR_FIELDS, *_ROW_FIELDS),
            self.scalar_arrays(),
            strict=True,
        ):
            _validate_vector(array, field=field, dtype=_FLOAT64_DTYPE)
        if self.groups == 0 or self.rows == 0:
            raise ClassicalInputContentError(violations=(("empty_grouped_frame", 1),))
        if self.group_offsets.shape != (self.groups + 1,):
            raise ClassicalInputContentError(violations=(("group_offset_count_mismatch", 1),))
        if self.group_offsets[0] != 0 or self.group_offsets[-1] != self.rows:
            raise ClassicalInputContentError(violations=(("group_offset_boundary_mismatch", 1),))
        if np.any(self.group_offsets[1:] <= self.group_offsets[:-1]):
            raise ClassicalInputContentError(violations=(("empty_or_reversed_group", 1),))
        if type(self.cash_dividends) is not RaggedCashDividends:
            raise ClassicalInputTypeError(
                field="cash_dividends",
                observed=type(self.cash_dividends).__name__,
                expected="RaggedCashDividends",
            )
        if self.cash_dividends.rows != self.groups:
            raise ClassicalInputContentError(violations=(("cash_dividend_group_mismatch", 1),))
        _validate_grouped_financial_content(self)
        for array in (
            self.group_offsets,
            *self.scalar_arrays(),
            self.cash_dividends.offsets,
            self.cash_dividends.times,
            self.cash_dividends.amounts,
        ):
            array.flags.writeable = False

    @property
    def groups(self) -> int:
        """Return the exact independently simulated state count."""
        return int(self.initial_spots.size)

    @property
    def rows(self) -> int:
        """Return the exact row-ordered strike/target count."""
        return int(self.strikes.size)

    def scalar_arrays(self) -> tuple[Float64Vector, ...]:
        """Return group then row vectors in canonical hash order."""
        return (
            self.initial_spots,
            self.maturities,
            self.rates,
            self.volatilities,
            self.forecast_yields,
            self.historical_yields,
            self.strikes,
            self.targets,
        )

    def dividends_for(self, variant: ClassicalVariant) -> BatchDividends:
        """Return one compact group-aligned engine dividend representation."""
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
                volatility_convention=EscrowedVolatilityConvention.SPOT_MATCHED_AT_ORIGIN,
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

    def engine_groups_for(self, variant: ClassicalVariant) -> LsmStrikeGroups:
        """Build one zero-copy native facade for a declared study variant."""
        return LsmStrikeGroups(
            group_offsets=self.group_offsets,
            initial_spots=self.initial_spots,
            strikes=self.strikes,
            maturities=self.maturities,
            rates=self.rates,
            volatilities=self.volatilities,
            dividends=self.dividends_for(variant),
        )


def build_project_classical_grouped_frame(
    frame: pl.LazyFrame,
    study: ClassicalStudySpec = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
) -> pl.LazyFrame:
    """Return a deterministic lazy projection with contiguous natural strike groups.

    This is stateless preparation only. It selects the exact classical columns and sorts by the
    public quote identity followed by strike. Constancy, uniqueness, and financial content are
    checked only at the explicit materialized preparation boundary.
    """
    if type(frame) is not pl.LazyFrame:
        raise ClassicalInputTypeError(
            field="frame",
            observed=type(frame).__name__,
            expected="a polars.LazyFrame",
        )
    if type(study) is not ClassicalStudySpec:
        raise ClassicalInputTypeError(
            field="study",
            observed=type(study).__name__,
            expected="ClassicalStudySpec",
        )
    required = (*_GROUP_KEYS, *study.input_columns())
    schema = frame.collect_schema()
    missing = tuple(column for column in required if column not in schema)
    mismatches = tuple(
        (column, str(PROJECT_AS_SUBMITTED_FEATURE_DTYPES[column]), str(schema[column]))
        for column in required
        if column in schema and schema[column] != PROJECT_AS_SUBMITTED_FEATURE_DTYPES[column]
    )
    if missing or mismatches:
        raise ClassicalInputSchemaError(missing=missing, mismatches=mismatches)
    return frame.select(required).sort((*_GROUP_KEYS, study.strike_column))


def prepare_project_classical_strike_groups(
    frame: pl.DataFrame,
    study: ClassicalStudySpec = PROJECT_CLASSICAL_RECONSTRUCTION_SPEC,
    *,
    event_buffer_policy: EventBufferPolicy = EventBufferPolicy.REQUIRE_ZERO_COPY,
) -> PreparedClassicalStrikeGroups:
    """Prove exact natural-group constancy and compact one materialized pricing frame."""
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
    _validate_materialized_schema(frame, study)
    if frame.height == 0:
        raise ClassicalInputContentError(violations=(("empty_frame", 1),))
    _validate_group_keys(frame)
    group_summary = frame.group_by(*_GROUP_KEYS, maintain_order=True).agg(
        pl.len().alias("_group_size"),
        *(pl.col(field).n_unique().alias(field) for field in _GROUP_STATE_FIELDS),
    )
    _validate_contiguous_groups(frame, group_summary.height)
    violations = tuple(
        (f"nonconstant_group_{field}", int((group_summary[field] != 1).sum()))
        for field in _GROUP_STATE_FIELDS
        if (group_summary[field] != 1).any()
    )
    duplicate_strikes = (
        frame.group_by(*_GROUP_KEYS, study.strike_column).len().filter(pl.col("len") != 1).height
    )
    if duplicate_strikes:
        violations = (*violations, ("duplicate_group_strikes", duplicate_strikes))
    if violations:
        raise ClassicalInputContentError(violations=violations)

    grouped = frame.group_by(*_GROUP_KEYS, maintain_order=True).agg(
        pl.len().alias("_group_size"),
        *(pl.col(field).first().alias(field) for field in _GROUP_STATE_FIELDS),
    )
    group_matrix = _float64_matrix(grouped, _GROUP_SOURCE_SCALARS)
    row_matrix = _float64_matrix(frame, _ROW_SOURCE_SCALARS)
    group_columns = tuple(group_matrix[:, index] for index in range(group_matrix.shape[1]))
    row_columns = tuple(row_matrix[:, index] for index in range(row_matrix.shape[1]))
    group_offsets = _group_offsets(grouped.get_column("_group_size"))
    try:
        cash_dividends = extract_ragged_cash_dividends(
            grouped,
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
    return PreparedClassicalStrikeGroups(
        study=study,
        group_offsets=group_offsets,
        initial_spots=group_columns[0],
        strikes=row_columns[0],
        maturities=group_columns[1],
        rates=group_columns[2],
        volatilities=group_columns[3],
        forecast_yields=group_columns[4],
        historical_yields=group_columns[5],
        targets=row_columns[1],
        cash_dividends=cash_dividends,
    )


def prepared_classical_strike_groups_sha256(inputs: PreparedClassicalStrikeGroups) -> str:
    """Hash the exact study, compact boundaries, state, row, target, and event bytes."""
    if type(inputs) is not PreparedClassicalStrikeGroups:
        raise ClassicalInputTypeError(
            field="inputs",
            observed=type(inputs).__name__,
            expected="PreparedClassicalStrikeGroups",
        )
    hasher = hashlib.sha256()
    _hash_text(hasher, _PREPARED_GROUP_HASH_FORMAT)
    _hash_text(hasher, classical_study_sha256(inputs.study))
    _hash_array(hasher, "group_offsets", inputs.group_offsets)
    for field, array in zip(
        (*_GROUP_SCALAR_FIELDS, *_ROW_FIELDS),
        inputs.scalar_arrays(),
        strict=True,
    ):
        _hash_array(hasher, field, array)
    _hash_array(hasher, "cash_offsets", inputs.cash_dividends.offsets)
    _hash_array(hasher, "cash_times", inputs.cash_dividends.times)
    _hash_array(hasher, "cash_amounts", inputs.cash_dividends.amounts)
    return hasher.hexdigest()


def _validate_materialized_schema(frame: pl.DataFrame, study: ClassicalStudySpec) -> None:
    required = (*_GROUP_KEYS, *study.input_columns())
    schema = frame.schema
    missing = tuple(column for column in required if column not in schema)
    mismatches = tuple(
        (column, str(PROJECT_AS_SUBMITTED_FEATURE_DTYPES[column]), str(schema[column]))
        for column in required
        if column in schema and schema[column] != PROJECT_AS_SUBMITTED_FEATURE_DTYPES[column]
    )
    if missing or mismatches:
        raise ClassicalInputSchemaError(missing=missing, mismatches=mismatches)


def _validate_group_keys(frame: pl.DataFrame) -> None:
    violations: list[tuple[str, int]] = []
    for field in _GROUP_KEYS:
        nulls = frame.get_column(field).null_count()
        if nulls:
            violations.append((f"null_group_key_{field}", nulls))
    blank_tickers = frame.get_column("ticker").str.strip_chars().eq("").sum()
    if blank_tickers:
        violations.append(("blank_group_key_ticker", int(blank_tickers)))
    if violations:
        raise ClassicalInputContentError(violations=tuple(violations))


def _validate_contiguous_groups(frame: pl.DataFrame, groups: int) -> None:
    key = pl.struct(*_GROUP_KEYS)
    transitions = frame.select(
        key.ne(key.shift(1)).fill_null(value=True).sum().alias("transitions")
    ).item()
    if int(cast("int", transitions)) != groups:
        raise ClassicalInputContentError(violations=(("noncontiguous_natural_groups", 1),))


def _float64_matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> npt.NDArray[np.float64]:
    converted = frame.select(*(pl.col(column).cast(pl.Float64) for column in columns)).to_numpy(
        order="fortran",
        writable=False,
        allow_copy=True,
    )
    matrix = cast("npt.NDArray[np.float64]", converted)
    if (
        type(matrix) is not np.ndarray
        or matrix.dtype != _FLOAT64_DTYPE
        or matrix.shape != (frame.height, len(columns))
        or not matrix.flags.f_contiguous
    ):
        raise ClassicalInputContentError(violations=(("invalid_float64_columnar_matrix", 1),))
    matrix.flags.writeable = False
    return matrix


def _group_offsets(sizes: pl.Series) -> UInt32Vector:
    size_values = sizes.to_numpy(writable=False, allow_copy=False)
    cumulative = np.cumsum(size_values, dtype=np.uint64)
    if cumulative[-1] > _MAX_U32:
        raise ClassicalInputContentError(violations=(("grouped_row_count_exceeds_uint32", 1),))
    offsets = np.empty(cumulative.size + 1, dtype=np.uint32)
    offsets[0] = 0
    offsets[1:] = cumulative
    offsets.flags.writeable = False
    return offsets


def _validate_vector(array: object, *, field: str, dtype: np.dtype[np.generic]) -> None:
    if type(array) is not np.ndarray:
        raise ClassicalInputTypeError(
            field=field,
            observed=type(array).__name__,
            expected="a plain numpy.ndarray",
        )
    typed = cast("npt.NDArray[np.generic]", array)
    if typed.dtype != dtype or typed.ndim != 1 or not typed.flags.c_contiguous:
        raise ClassicalInputTypeError(
            field=field,
            observed=f"dtype={typed.dtype}, shape={typed.shape}, strides={typed.strides}",
            expected=f"a C-contiguous {dtype.name} vector",
        )


def _validate_grouped_financial_content(inputs: PreparedClassicalStrikeGroups) -> None:
    violations: list[tuple[str, int]] = []
    for field, array in zip(
        (*_GROUP_SCALAR_FIELDS, *_ROW_FIELDS),
        inputs.scalar_arrays(),
        strict=True,
    ):
        count = int(np.count_nonzero(~np.isfinite(array)))
        if count:
            violations.append((f"nonfinite_{field}", count))
    for field, array in (
        ("initial_spots", inputs.initial_spots),
        ("maturities", inputs.maturities),
        ("volatilities", inputs.volatilities),
        ("strikes", inputs.strikes),
        ("targets", inputs.targets),
    ):
        count = int(np.count_nonzero(array <= 0.0))
        if count:
            violations.append((f"nonpositive_{field}", count))
    for field, array in (
        ("forecast_yields", inputs.forecast_yields),
        ("historical_yields", inputs.historical_yields),
    ):
        count = int(np.count_nonzero(np.signbit(array)))
        if count:
            violations.append((f"negative_{field}", count))
    violations.extend(_cash_event_violations(inputs))
    if violations:
        raise ClassicalInputContentError(violations=tuple(violations))


def _cash_event_violations(inputs: PreparedClassicalStrikeGroups) -> list[tuple[str, int]]:
    schedule = inputs.cash_dividends
    times = schedule.times
    amounts = schedule.amounts
    violations: list[tuple[str, int]] = []
    for field, array in (("cash_dividend_times", times), ("cash_dividend_amounts", amounts)):
        count = int(np.count_nonzero(~np.isfinite(array)))
        if count:
            violations.append((f"nonfinite_{field}", count))
    nonpositive_times = int(np.count_nonzero(times <= 0.0))
    if nonpositive_times:
        violations.append(("nonpositive_cash_dividend_times", nonpositive_times))
    negative_amounts = int(np.count_nonzero(np.signbit(amounts)))
    if negative_amounts:
        violations.append(("negative_cash_dividend_amounts", negative_amounts))
    if times.size > 1:
        nonincreasing = times[1:] <= times[:-1]
        boundaries = schedule.offsets[1:-1]
        internal = boundaries[(boundaries > 0) & (boundaries < times.size)] - 1
        nonincreasing[internal] = False
        count = int(np.count_nonzero(nonincreasing))
        if count:
            violations.append(("nonincreasing_cash_dividend_times", count))
    nonempty = schedule.offsets[1:] > schedule.offsets[:-1]
    if np.any(nonempty):
        last_events = schedule.offsets[1:][nonempty] - 1
        count = int(np.count_nonzero(times[last_events] >= inputs.maturities[nonempty]))
        if count:
            violations.append(("cash_dividend_groups_at_or_after_maturity", count))
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
    "PreparedClassicalStrikeGroups",
    "build_project_classical_grouped_frame",
    "prepare_project_classical_strike_groups",
    "prepared_classical_strike_groups_sha256",
]
