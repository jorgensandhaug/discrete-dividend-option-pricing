"""Tests for explicit no-fallback neural execution profiles."""

from __future__ import annotations

from typing import cast

import pytest

from discrete_dividend_study.neural_execution import (
    PROJECT_APPLE_SILICON_EXECUTION_PROFILE,
    PROJECT_PORTABLE_CPU_EXECUTION_PROFILE,
    CompilationMode,
    NeuralBackend,
    NeuralDevice,
    NeuralExecutionSpec,
    NeuralExecutionSpecError,
)
from discrete_dividend_study.neural_spec import ModelPrecision, NeuralVariant


def test_apple_profile_uses_measured_per_model_policies() -> None:
    """The short LSTM remains CPU-only while four models use compiled MLX GPU."""
    profile = PROJECT_APPLE_SILICON_EXECUTION_PROFILE
    for variant in NeuralVariant:
        execution = profile.for_variant(variant)
        if variant is NeuralVariant.LSTM:
            assert execution.backend is NeuralBackend.PYTORCH
            assert execution.device is NeuralDevice.CPU
            assert execution.compilation is CompilationMode.EAGER
            assert execution.cpu_threads == 1
        else:
            assert execution.backend is NeuralBackend.MLX
            assert execution.device is NeuralDevice.MLX_GPU
            assert execution.compilation is CompilationMode.COMPILED
            assert execution.cpu_threads is None


def test_portable_profile_never_infers_an_accelerator() -> None:
    """Every portable request is the same explicit one-thread PyTorch CPU policy."""
    for variant in NeuralVariant:
        execution = PROJECT_PORTABLE_CPU_EXECUTION_PROFILE.for_variant(variant)
        assert execution.backend is NeuralBackend.PYTORCH
        assert execution.device is NeuralDevice.CPU
        assert execution.cpu_threads == 1


@pytest.mark.parametrize(
    ("backend", "device", "compilation", "threads"),
    [
        (NeuralBackend.MLX, NeuralDevice.MPS, CompilationMode.EAGER, None),
        (NeuralBackend.PYTORCH, NeuralDevice.MLX_GPU, CompilationMode.EAGER, None),
        (NeuralBackend.PYTORCH, NeuralDevice.CPU, CompilationMode.EAGER, None),
        (NeuralBackend.PYTORCH, NeuralDevice.MPS, CompilationMode.EAGER, 1),
        (NeuralBackend.PYTORCH, NeuralDevice.CPU, CompilationMode.COMPILED, 1),
    ],
)
def test_invalid_backend_device_thread_and_compilation_combinations_fail(
    backend: NeuralBackend,
    device: NeuralDevice,
    compilation: CompilationMode,
    threads: int | None,
) -> None:
    """A request is rejected instead of being rewritten to something available."""
    with pytest.raises(NeuralExecutionSpecError):
        NeuralExecutionSpec(
            backend=backend,
            device=device,
            precision=ModelPrecision.FLOAT32,
            compilation=compilation,
            cpu_threads=threads,
        )


def test_profile_lookup_rejects_variant_strings() -> None:
    """Profile dispatch cannot coerce a model name."""
    variant = cast("NeuralVariant", "nn_lstm")
    with pytest.raises(NeuralExecutionSpecError, match="NeuralVariant"):
        PROJECT_APPLE_SILICON_EXECUTION_PROFILE.for_variant(variant)
