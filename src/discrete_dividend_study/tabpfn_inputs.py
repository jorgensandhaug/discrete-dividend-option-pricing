"""Strict project binned-dividend boundary for a future local TabPFN benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
from finoptions import BinnedCashDividends

from .model_inputs import PreparedNeuralBatch, prepared_neural_batch_sha256
from .neural_spec import (
    PROJECT_NEURAL_STUDY_SPEC,
    BinnedDividendInput,
    NeuralModelSpec,
    NeuralVariant,
)

Float32Vector = npt.NDArray[np.float32]
Float32Matrix = npt.NDArray[np.float32]

_FORMAT: Final = "discrete-dividend-tabpfn-binned-input-v1"
_FLOAT32_DTYPE: Final = np.dtype(np.float32)
_SHA256_HEX_LENGTH: Final = 64
_LOWERCASE_HEXADECIMAL: Final = frozenset("0123456789abcdef")


class _HashWriter(Protocol):
    """The streaming byte interface shared by hashlib implementations."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Append bytes to the digest state."""


@dataclass(slots=True)
class TabPfnInputError(ValueError):
    """A future TabPFN input violates the closed project boundary."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected input contract."""
        return f"invalid TabPFN project input: {self.reason}"


@dataclass(frozen=True, slots=True)
class TabPfnBinnedInputSpec:
    """One explicit flat representation derived from the project cash-bin model."""

    source_model: NeuralModelSpec
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject any representation other than the canonical project cash bins."""
        canonical = PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.BINS)
        if type(self.source_model) is not NeuralModelSpec or self.source_model != canonical:
            raise TabPfnInputError(reason="source_model must be the canonical project bin model")
        expected_names = _feature_names(canonical)
        if type(self.feature_names) is not tuple or self.feature_names != expected_names:
            raise TabPfnInputError(reason="feature_names disagree with the canonical bin order")

    @property
    def target_name(self) -> str:
        """Return the raw positive market-price target retained by this first boundary."""
        return self.source_model.target_column


@dataclass(frozen=True, slots=True)
class PreparedTabPfnBatch:
    """One immutable contiguous flat matrix and its unchanged positive market target."""

    spec: TabPfnBinnedInputSpec
    source_prepared_batch_sha256: str
    features: Float32Matrix
    target: Float32Vector

    def __post_init__(self) -> None:
        """Require exact dtype, geometry, finiteness, and immutability."""
        if type(self.spec) is not TabPfnBinnedInputSpec:
            raise TabPfnInputError(reason="spec must be TabPfnBinnedInputSpec")
        _require_sha256(self.source_prepared_batch_sha256)
        _validate_array(self.features, dimensions=2, field="features")
        _validate_array(self.target, dimensions=1, field="target")
        if self.features.shape != (self.target.shape[0], len(self.spec.feature_names)):
            raise TabPfnInputError(reason="feature and target geometry disagrees with the spec")
        if self.features.shape[0] == 0:
            raise TabPfnInputError(reason="a prepared batch cannot be empty")
        if not self.features.flags.c_contiguous or not self.target.flags.c_contiguous:
            raise TabPfnInputError(reason="prepared arrays must be C-contiguous")
        if self.features.flags.writeable or self.target.flags.writeable:
            raise TabPfnInputError(reason="prepared arrays must be read-only")
        if not np.isfinite(self.features).all() or not np.isfinite(self.target).all():
            raise TabPfnInputError(reason="prepared arrays must be finite")
        if np.any(self.target <= 0.0):
            raise TabPfnInputError(reason="raw call-price targets must be strictly positive")

    @property
    def rows(self) -> int:
        """Return the exact number of observations."""
        return int(self.features.shape[0])


def prepare_project_tabpfn_binned_batch(
    batch: PreparedNeuralBatch,
    spec: TabPfnBinnedInputSpec,
) -> PreparedTabPfnBatch:
    """Allocate the one flat matrix required by TabPFN without changing any values.

    The scalar and cash-bin sources are already validated project float32 arrays. They are copied
    once into a row-major matrix because a tabular estimator consumes one physical feature table.
    The positive raw market target remains a zero-copy immutable reference. No scaling, categorical
    inference, target transform, imputation, or prediction policy is introduced here.
    """
    if type(batch) is not PreparedNeuralBatch:
        raise TabPfnInputError(reason="batch must be PreparedNeuralBatch")
    if type(spec) is not TabPfnBinnedInputSpec:
        raise TabPfnInputError(reason="spec must be TabPfnBinnedInputSpec")
    if batch.model != spec.source_model:
        raise TabPfnInputError(reason="batch model disagrees with the TabPFN input spec")
    if type(batch.dividends) is not BinnedCashDividends:
        raise TabPfnInputError(reason="the canonical binned dividend tensor is required")
    scalar_width = batch.scalar_features.shape[1]
    bin_width = batch.dividends.amounts.shape[1]
    if scalar_width + bin_width != len(spec.feature_names):
        raise TabPfnInputError(reason="source tensor widths disagree with feature_names")
    features = np.empty((batch.rows, scalar_width + bin_width), dtype=np.float32, order="C")
    features[:, :scalar_width] = batch.scalar_features
    features[:, scalar_width:] = batch.dividends.amounts
    features.flags.writeable = False
    return PreparedTabPfnBatch(
        spec=spec,
        source_prepared_batch_sha256=prepared_neural_batch_sha256(batch),
        features=features,
        target=batch.target,
    )


def slice_prepared_tabpfn_batch(
    batch: PreparedTabPfnBatch,
    start: int,
    end: int,
) -> PreparedTabPfnBatch:
    """Return one nonempty immutable zero-copy rolling-phase view."""
    if type(batch) is not PreparedTabPfnBatch:
        raise TabPfnInputError(reason="batch must be PreparedTabPfnBatch")
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end > batch.rows
    ):
        raise TabPfnInputError(
            reason="row bounds must be plain ints satisfying 0 <= start < end <= rows"
        )
    return PreparedTabPfnBatch(
        spec=batch.spec,
        source_prepared_batch_sha256=batch.source_prepared_batch_sha256,
        features=batch.features[start:end],
        target=batch.target[start:end],
    )


def prepared_tabpfn_batch_sha256(batch: PreparedTabPfnBatch) -> str:
    """Hash the flat representation, parent prepared input, schema, and exact bytes."""
    if type(batch) is not PreparedTabPfnBatch:
        raise TabPfnInputError(reason="batch must be PreparedTabPfnBatch")
    hasher = hashlib.sha256()
    _hash_text(hasher, _FORMAT)
    _hash_text(hasher, batch.source_prepared_batch_sha256)
    for name in batch.spec.feature_names:
        _hash_text(hasher, name)
    _hash_text(hasher, batch.spec.target_name)
    _hash_array(hasher, "features", batch.features)
    _hash_array(hasher, "target", batch.target)
    return hasher.hexdigest()


def _feature_names(model: NeuralModelSpec) -> tuple[str, ...]:
    dividend_input = model.dividend_input
    if type(dividend_input) is not BinnedDividendInput:
        raise TabPfnInputError(reason="source model must declare BinnedDividendInput")
    bins = dividend_input.layout.bins
    return (
        *model.scalar_columns,
        *(f"forecast_cash_bin_{index:02d}_over_spot" for index in range(1, bins + 1)),
    )


def _validate_array(array: object, *, dimensions: int, field: str) -> None:
    if type(array) is not np.ndarray:
        raise TabPfnInputError(reason=f"{field} must be an exact numpy.ndarray")
    if array.dtype != _FLOAT32_DTYPE or array.ndim != dimensions:
        raise TabPfnInputError(reason=f"{field} must be {dimensions}D float32")


def _require_sha256(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or not set(value) <= _LOWERCASE_HEXADECIMAL
    ):
        raise TabPfnInputError(reason="source_prepared_batch_sha256 must be lowercase SHA-256")


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


PROJECT_TABPFN_BINNED_INPUT_SPEC: Final = TabPfnBinnedInputSpec(
    source_model=PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.BINS),
    feature_names=_feature_names(PROJECT_NEURAL_STUDY_SPEC.model(NeuralVariant.BINS)),
)


__all__ = [
    "PROJECT_TABPFN_BINNED_INPUT_SPEC",
    "PreparedTabPfnBatch",
    "TabPfnBinnedInputSpec",
    "TabPfnInputError",
    "prepare_project_tabpfn_binned_batch",
    "prepared_tabpfn_batch_sha256",
    "slice_prepared_tabpfn_batch",
]
