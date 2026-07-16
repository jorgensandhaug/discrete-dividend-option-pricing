"""Production-geometry neural training benchmark on deterministic synthetic rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from enum import StrEnum
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
from finoptions import BinnedCashDividends, PaddedCashDividends

from benchmarks._support import max_rss_bytes
from discrete_dividend_study.models.neural.backends.mlx_training import train_mlx_project_model
from discrete_dividend_study.models.neural.backends.torch_training import train_torch_project_model
from discrete_dividend_study.models.neural.execution import (
    PROJECT_APPLE_SILICON_EXECUTION_PROFILE,
    CompilationMode,
    NeuralBackend,
    NeuralDevice,
    NeuralExecutionSpec,
)
from discrete_dividend_study.models.neural.inputs import PreparedNeuralBatch
from discrete_dividend_study.models.neural.parameters import (
    NeuralParameter,
    initialize_neural_parameters,
)
from discrete_dividend_study.models.neural.specification import (
    PROJECT_STUDY,
    BinnedDividendInput,
    DirectDividendLstmInput,
    EventAttentionInput,
    ModelPrecision,
    NeuralModelSpec,
    NeuralVariant,
)
from discrete_dividend_study.models.neural.training import NeuralTrainingRequest

Float32Vector = npt.NDArray[np.float32]
Float32Matrix = npt.NDArray[np.float32]
Array = npt.NDArray[np.generic]

PARAMETER_SEED: Final = 20_261_209
SHUFFLE_SEED: Final = 20_261_210
BIN_COUNT: Final = 12
_POSITIVE_ARGUMENTS_ERROR: Final = "rows, epochs, and threads must be positive"
_VARIANT_TYPE_ERROR: Final = "variant must be NeuralVariant"
_SYNTHETIC_DOMAIN_ERROR: Final = "rows must be positive and phase must be nonnegative"


class ExecutionChoice(StrEnum):
    """Explicit benchmark execution selections."""

    PROFILE = "profile"
    TORCH_CPU = "torch_cpu"
    MLX_GPU = "mlx_gpu"


class _Digest(Protocol):
    def update(self, data: bytes, /) -> object:
        """Add bytes to a running digest."""


def main() -> None:
    """Run one isolated timing and emit one canonical JSON object."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=[variant.value for variant in NeuralVariant], required=True
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--validation-rows", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--execution",
        choices=[choice.value for choice in ExecutionChoice],
        default=ExecutionChoice.PROFILE.value,
    )
    parser.add_argument("--threads", type=int, default=1)
    arguments = parser.parse_args()
    if (
        arguments.rows <= 0
        or arguments.validation_rows <= 0
        or arguments.epochs <= 0
        or arguments.threads <= 0
    ):
        raise ValueError(_POSITIVE_ARGUMENTS_ERROR)

    variant = NeuralVariant(arguments.variant)
    training_batch = synthetic_neural_batch(variant, arguments.rows, phase=0)
    validation_batch = synthetic_neural_batch(variant, arguments.validation_rows, phase=1)
    training = replace(
        PROJECT_STUDY.training,
        maximum_epochs=arguments.epochs,
        early_stopping_patience_epochs=arguments.epochs,
    )
    study = replace(PROJECT_STUDY, training=training)
    parameters = initialize_neural_parameters(study, training_batch.model, PARAMETER_SEED)
    execution_choice = ExecutionChoice(arguments.execution)
    execution = resolve_benchmark_execution(
        execution_choice,
        variant,
        arguments.threads,
    )
    request = NeuralTrainingRequest(
        study=study,
        parameters=parameters,
        training_batch=training_batch,
        validation_batch=validation_batch,
        training_data_sha256=_batch_sha256(training_batch),
        validation_data_sha256=_batch_sha256(validation_batch),
        execution=execution,
        shuffle_seed=SHUFFLE_SEED,
    )

    started = time.perf_counter()
    result = (
        train_mlx_project_model(request)
        if execution.backend is NeuralBackend.MLX
        else train_torch_project_model(request)
    )
    elapsed = time.perf_counter() - started
    completed = result.summary.control.completed_epochs
    sys.stdout.write(
        json.dumps(
            {
                "backend": execution.backend.value,
                "compilation": execution.compilation.value,
                "device": execution.device.value,
                "elapsed_seconds": elapsed,
                "epochs": completed,
                "execution_choice": execution_choice.value,
                "execution_profile": (
                    PROJECT_APPLE_SILICON_EXECUTION_PROFILE.name
                    if execution_choice is ExecutionChoice.PROFILE
                    else None
                ),
                "machine": platform.machine(),
                "max_rss_bytes": max_rss_bytes(),
                "rows": arguments.rows,
                "summary_sha256": _canonical_json_sha256(asdict(result.summary)),
                "total_row_epochs_per_second": (
                    (arguments.rows + arguments.validation_rows) * completed / elapsed
                ),
                "trained_parameters_sha256": _parameter_sha256(result.parameters),
                "training_row_epochs_per_second": arguments.rows * completed / elapsed,
                "validation_rows": arguments.validation_rows,
                "variant": variant.value,
            },
            sort_keys=True,
        )
        + "\n",
    )


def synthetic_neural_batch(
    variant: NeuralVariant,
    rows: int,
    *,
    phase: int,
) -> PreparedNeuralBatch:
    """Build finite project-shaped rows without licensed source data."""
    if type(variant) is not NeuralVariant:
        raise TypeError(_VARIANT_TYPE_ERROR)
    if type(rows) is not int or rows <= 0 or type(phase) is not int or phase < 0:
        raise ValueError(_SYNTHETIC_DOMAIN_ERROR)
    model = PROJECT_STUDY.model(variant)
    columns = _synthetic_scalar_columns(rows, phase=phase)
    scalar = np.ascontiguousarray(
        np.column_stack([columns[column] for column in model.scalar_columns]),
        dtype=np.float32,
    )
    target = np.ascontiguousarray(
        1.0
        + np.maximum(columns["underlying_price"] - columns["strike"], 0.0)
        + 2.0 * columns["historical_volatility_90"]
        + 3.0 * columns["yte"],
        dtype=np.float32,
    )
    return PreparedNeuralBatch(
        model=model,
        scalar_features=scalar,
        target=target,
        dividends=_synthetic_dividends(model, columns, rows),
    )


def _synthetic_scalar_columns(rows: int, *, phase: int) -> dict[str, Float32Vector]:
    position = (
        np.arange(rows, dtype=np.float32) + np.float32(phase) + np.float32(1.0)
    ) / np.float32(rows + phase + 1)
    spot = np.float32(90.0) + np.float32(20.0) * position
    strike = np.float32(80.0) + np.float32(40.0) * position
    return {
        "strike": strike,
        "underlying_price": spot,
        "risk_free_rate": np.float32(0.01) + np.float32(0.03) * position,
        "call_moneyness": spot / strike,
        "historical_volatility_90": np.float32(0.15) + np.float32(0.20) * position,
        "yte": np.float32(0.25) + np.float32(0.75) * position,
        "forecasted_dividend_yield": np.float32(0.005) + np.float32(0.035) * position,
        "historical_dividend_yield": np.float32(0.01) + np.float32(0.025) * position,
    }


def _synthetic_dividends(
    model: NeuralModelSpec,
    columns: dict[str, Float32Vector],
    rows: int,
) -> PaddedCashDividends | BinnedCashDividends | None:
    dividend_input = model.dividend_input
    if (
        type(dividend_input) is DirectDividendLstmInput
        or type(dividend_input) is EventAttentionInput
    ):
        capacity = dividend_input.layout.capacity
        lengths = np.ascontiguousarray(np.arange(rows) % (capacity + 1), dtype=np.uint32)
        times = np.zeros((rows, capacity), dtype=np.float32)
        amounts = np.zeros((rows, capacity), dtype=np.float32)
        mask = np.zeros((rows, capacity), dtype=np.bool_)
        denominator = lengths.astype(np.float32) + np.float32(1.0)
        for event in range(capacity):
            active = lengths > event
            mask[:, event] = active
            times[:, event] = np.where(
                active,
                columns["yte"] * np.float32(event + 1) / denominator,
                np.float32(0.0),
            )
            amounts[:, event] = np.where(
                active,
                np.float32(0.004 * (event + 1)),
                np.float32(0.0),
            )
        return PaddedCashDividends(
            times=times,
            amounts=amounts,
            mask=mask,
            lengths=lengths,
        )
    if type(dividend_input) is BinnedDividendInput:
        base = columns["forecasted_dividend_yield"][:, None]
        weights = np.linspace(1.0, 2.0, BIN_COUNT, dtype=np.float32)[None, :]
        return BinnedCashDividends(amounts=np.ascontiguousarray(base * weights))
    return None


def resolve_benchmark_execution(
    choice: ExecutionChoice,
    variant: NeuralVariant,
    threads: int,
) -> NeuralExecutionSpec:
    """Resolve one explicit benchmark backend without automatic fallback."""
    if choice is ExecutionChoice.PROFILE:
        return PROJECT_APPLE_SILICON_EXECUTION_PROFILE.for_variant(variant)
    if choice is ExecutionChoice.TORCH_CPU:
        return NeuralExecutionSpec(
            backend=NeuralBackend.PYTORCH,
            device=NeuralDevice.CPU,
            precision=ModelPrecision.FLOAT32,
            compilation=CompilationMode.EAGER,
            cpu_threads=threads,
            inference_batch_size=8192,
        )
    if choice is ExecutionChoice.MLX_GPU:
        return NeuralExecutionSpec(
            backend=NeuralBackend.MLX,
            device=NeuralDevice.MLX_GPU,
            precision=ModelPrecision.FLOAT32,
            compilation=CompilationMode.COMPILED,
            cpu_threads=None,
            inference_batch_size=8192,
        )
    reason = f"unsupported execution choice {choice!r}"
    raise ValueError(reason)


def _batch_sha256(batch: PreparedNeuralBatch) -> str:
    hasher = hashlib.sha256()
    hasher.update(batch.model.variant.value.encode())
    arrays: list[Array] = [batch.scalar_features, batch.target]
    if isinstance(batch.dividends, PaddedCashDividends):
        arrays.extend(
            [
                batch.dividends.times,
                batch.dividends.amounts,
                batch.dividends.mask,
                batch.dividends.lengths,
            ],
        )
    elif isinstance(batch.dividends, BinnedCashDividends):
        arrays.append(batch.dividends.amounts)
    for values in arrays:
        _update_array_identity(hasher, values)
    return hasher.hexdigest()


def _parameter_sha256(parameters: tuple[NeuralParameter, ...]) -> str:
    hasher = hashlib.sha256()
    for parameter in parameters:
        hasher.update(parameter.name.encode())
        _update_array_identity(hasher, parameter.values)
    return hasher.hexdigest()


def _update_array_identity(hasher: _Digest, values: Array) -> None:
    typed = hashlib.sha256()
    typed.update(values.dtype.str.encode())
    typed.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    typed.update(memoryview(values).cast("B"))
    hasher.update(typed.digest())


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
