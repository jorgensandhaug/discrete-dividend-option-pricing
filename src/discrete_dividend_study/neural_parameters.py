"""Backend-independent deterministic parameter initialization."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from .neural_spec import (
    BinnedDividendInput,
    DirectDividendLstmInput,
    DividendYieldInput,
    EventAttentionInput,
    NeuralModelSpec,
    NeuralSpecValueError,
    NeuralStudySpec,
    ParameterInitialization,
    trainable_parameter_count,
)

Float32Array = npt.NDArray[np.float32]
_MAX_SEED: Final = (2**64) - 1


@dataclass(frozen=True, slots=True)
class NeuralParameter:
    """One named immutable C-contiguous float32 parameter array."""

    name: str
    values: Float32Array

    def __post_init__(self) -> None:
        """Reject unsafe names, dtypes, layouts, values, and writable buffers."""
        if type(self.name) is not str or not self.name:
            raise NeuralSpecValueError(
                field="parameter.name",
                observed=self.name,
                requirement="a nonempty plain str",
            )
        if type(self.values) is not np.ndarray:
            raise NeuralSpecValueError(
                field=f"parameter.{self.name}",
                observed=type(self.values).__name__,
                requirement="a plain numpy.ndarray",
            )
        if self.values.dtype != np.dtype(np.float32) or not self.values.flags.c_contiguous:
            raise NeuralSpecValueError(
                field=f"parameter.{self.name}",
                observed=(str(self.values.dtype), self.values.flags.c_contiguous),
                requirement="C-contiguous float32 storage",
            )
        if self.values.ndim not in {1, 2} or self.values.size == 0:
            raise NeuralSpecValueError(
                field=f"parameter.{self.name}.shape",
                observed=self.values.shape,
                requirement="a nonempty vector or matrix",
            )
        if not np.isfinite(self.values).all():
            raise NeuralSpecValueError(
                field=f"parameter.{self.name}",
                observed="nonfinite values",
                requirement="finite values",
            )
        if self.values.flags.writeable:
            raise NeuralSpecValueError(
                field=f"parameter.{self.name}",
                observed="writable storage",
                requirement="read-only storage",
            )


@dataclass(frozen=True, slots=True)
class NeuralParameterSet:
    """One complete immutable canonical initialization for a study model."""

    study: NeuralStudySpec
    model: NeuralModelSpec
    seed: int
    parameters: tuple[NeuralParameter, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate completeness, uniqueness, size, and content identity."""
        if type(self.study) is not NeuralStudySpec:
            raise NeuralSpecValueError(
                field="parameter_set.study",
                observed=self.study,
                requirement="a NeuralStudySpec",
            )
        if type(self.model) is not NeuralModelSpec or self.model not in self.study.models:
            raise NeuralSpecValueError(
                field="parameter_set.model",
                observed=self.model,
                requirement="a model owned by parameter_set.study",
            )
        _validate_seed(self.seed)
        if type(self.parameters) is not tuple or any(
            type(parameter) is not NeuralParameter for parameter in self.parameters
        ):
            raise NeuralSpecValueError(
                field="parameter_set.parameters",
                observed=self.parameters,
                requirement="a tuple[NeuralParameter, ...]",
            )
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise NeuralSpecValueError(
                field="parameter_set.parameters",
                observed=names,
                requirement="unique parameter names",
            )
        observed_count = sum(parameter.values.size for parameter in self.parameters)
        expected_count = trainable_parameter_count(self.study, self.model)
        if observed_count != expected_count:
            raise NeuralSpecValueError(
                field="parameter_set.parameter_count",
                observed=observed_count,
                requirement=str(expected_count),
            )
        object.__setattr__(self, "sha256", _parameter_sha256(self))

    def get(self, name: str) -> Float32Array:
        """Return one read-only array by exact name."""
        if type(name) is not str or not name:
            raise NeuralSpecValueError(
                field="parameter.name",
                observed=name,
                requirement="a nonempty plain str",
            )
        try:
            return next(parameter.values for parameter in self.parameters if parameter.name == name)
        except StopIteration as error:
            raise KeyError(name) from error


def initialize_neural_parameters(
    study: NeuralStudySpec,
    model: NeuralModelSpec,
    seed: int,
) -> NeuralParameterSet:
    """Initialize one model identically for every backend using NumPy PCG64DXSM.

    Linear weights and biases use ``U(-1/sqrt(fan_in), +1/sqrt(fan_in))``. LSTM input and recurrent
    matrices use ``U(-1/sqrt(hidden_size), +1/sqrt(hidden_size))``. Draw order and parameter names
    are explicit, so backend choice cannot alter initialization.
    """
    if type(study) is not NeuralStudySpec:
        raise NeuralSpecValueError(
            field="study",
            observed=study,
            requirement="a NeuralStudySpec",
        )
    if type(model) is not NeuralModelSpec or model not in study.models:
        raise NeuralSpecValueError(
            field="model",
            observed=model,
            requirement="a model owned by study",
        )
    if study.parameter_initialization is not ParameterInitialization.UNIFORM_PCG64DXSM:
        raise NeuralSpecValueError(
            field="study.parameter_initialization",
            observed=study.parameter_initialization,
            requirement="ParameterInitialization.UNIFORM_PCG64DXSM",
        )
    _validate_seed(seed)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    parameters: list[NeuralParameter] = []
    _append_dividend_parameters(parameters, rng, model.dividend_input)

    layer_input = model.trunk_input_width
    for layer in range(study.mlp.hidden_layers):
        _append_linear(
            parameters,
            rng,
            prefix=f"mlp.{layer}",
            inputs=layer_input,
            outputs=study.mlp.hidden_width,
        )
        layer_input = study.mlp.hidden_width
    _append_linear(parameters, rng, prefix="output", inputs=layer_input, outputs=1)
    return NeuralParameterSet(study=study, model=model, seed=seed, parameters=tuple(parameters))


def _append_dividend_parameters(
    parameters: list[NeuralParameter],
    rng: np.random.Generator,
    dividend_input: DirectDividendLstmInput
    | EventAttentionInput
    | BinnedDividendInput
    | DividendYieldInput,
) -> None:
    if type(dividend_input) is DirectDividendLstmInput:
        event_width = 3 if dividend_input.include_option_maturity else 2
        bound = 1.0 / math.sqrt(dividend_input.hidden_size)
        layer_input = event_width
        for layer in range(dividend_input.layers):
            _append_uniform(
                parameters,
                rng,
                name=f"lstm.{layer}.weight_input",
                shape=(4 * dividend_input.hidden_size, layer_input),
                bound=bound,
            )
            _append_uniform(
                parameters,
                rng,
                name=f"lstm.{layer}.weight_hidden",
                shape=(4 * dividend_input.hidden_size, dividend_input.hidden_size),
                bound=bound,
            )
            if dividend_input.bias:
                _append_uniform(
                    parameters,
                    rng,
                    name=f"lstm.{layer}.bias_input",
                    shape=(4 * dividend_input.hidden_size,),
                    bound=bound,
                )
                _append_uniform(
                    parameters,
                    rng,
                    name=f"lstm.{layer}.bias_hidden",
                    shape=(4 * dividend_input.hidden_size,),
                    bound=bound,
                )
            layer_input = dividend_input.hidden_size
    elif type(dividend_input) is EventAttentionInput:
        event_width = 3 if dividend_input.include_option_maturity else 2
        _append_linear(
            parameters,
            rng,
            prefix="embedding.0",
            inputs=event_width,
            outputs=dividend_input.embedding_width,
        )
        for layer in range(1, dividend_input.embedding_layers):
            _append_linear(
                parameters,
                rng,
                prefix=f"embedding.{layer}",
                inputs=dividend_input.embedding_width,
                outputs=dividend_input.embedding_width,
            )
        _append_linear(
            parameters,
            rng,
            prefix="attention",
            inputs=dividend_input.embedding_width,
            outputs=1,
        )
    elif type(dividend_input) not in {BinnedDividendInput, DividendYieldInput}:
        raise NeuralSpecValueError(
            field="model.dividend_input",
            observed=dividend_input,
            requirement="a supported typed dividend input",
        )


def _append_linear(
    parameters: list[NeuralParameter],
    rng: np.random.Generator,
    *,
    prefix: str,
    inputs: int,
    outputs: int,
) -> None:
    bound = 1.0 / math.sqrt(inputs)
    _append_uniform(
        parameters,
        rng,
        name=f"{prefix}.weight",
        shape=(outputs, inputs),
        bound=bound,
    )
    _append_uniform(
        parameters,
        rng,
        name=f"{prefix}.bias",
        shape=(outputs,),
        bound=bound,
    )


def _append_uniform(
    parameters: list[NeuralParameter],
    rng: np.random.Generator,
    *,
    name: str,
    shape: tuple[int, ...],
    bound: float,
) -> None:
    values = np.ascontiguousarray(rng.uniform(-bound, bound, size=shape), dtype=np.float32)
    values.flags.writeable = False
    parameters.append(NeuralParameter(name=name, values=values))


def _validate_seed(seed: object) -> None:
    if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
        raise NeuralSpecValueError(
            field="parameter_set.seed",
            observed=seed,
            requirement=f"a plain int in [0, {_MAX_SEED}]",
        )


def _parameter_sha256(parameter_set: NeuralParameterSet) -> str:
    hasher = hashlib.sha256()
    hasher.update(parameter_set.model.variant.value.encode())
    hasher.update(parameter_set.study.parameter_initialization.value.encode())
    hasher.update(parameter_set.seed.to_bytes(8, byteorder="little", signed=False))
    for parameter in parameter_set.parameters:
        hasher.update(parameter.name.encode())
        hasher.update(b"\0")
        for dimension in parameter.values.shape:
            hasher.update(dimension.to_bytes(8, byteorder="little", signed=False))
        hasher.update(b"\0")
        hasher.update(parameter.values.tobytes(order="C"))
    return hasher.hexdigest()


__all__ = [
    "NeuralParameter",
    "NeuralParameterSet",
    "initialize_neural_parameters",
]
