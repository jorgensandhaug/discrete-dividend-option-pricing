"""Explicit neural backend/device profiles with no runtime fallback."""

from __future__ import annotations

import importlib
import platform
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final

from .neural_spec import ModelPrecision, NeuralVariant

MLX_VERSION: Final = "0.32.0"
PYTORCH_VERSION: Final = "2.13.0"


class NeuralBackend(StrEnum):
    """Production neural framework implementations."""

    MLX = "mlx"
    PYTORCH = "pytorch"


class NeuralDevice(StrEnum):
    """Explicit devices admitted by a backend execution request."""

    CPU = "cpu"
    MLX_GPU = "mlx_gpu"
    MPS = "mps"
    CUDA = "cuda"


class CompilationMode(StrEnum):
    """Whether the complete training step is compiled."""

    EAGER = "eager"
    COMPILED = "compiled"


class NeuralExecutionError(RuntimeError):
    """Base class for invalid or unavailable neural execution."""


@dataclass(slots=True)
class NeuralExecutionSpecError(NeuralExecutionError):
    """A backend/device specification is internally inconsistent."""

    reason: str

    def __str__(self) -> str:
        """Return the rejected execution contract."""
        return f"invalid neural execution specification: {self.reason}"


@dataclass(slots=True)
class NeuralRuntimeUnavailableError(NeuralExecutionError):
    """The exact requested backend, version, or device is unavailable."""

    backend: NeuralBackend
    device: NeuralDevice
    reason: str

    def __str__(self) -> str:
        """Describe the unavailable request without suggesting a fallback."""
        return f"requested {self.backend.value}/{self.device.value} is unavailable: {self.reason}"


@dataclass(frozen=True, slots=True)
class NeuralExecutionSpec:
    """One immutable backend, device, precision, compilation, and batching identity."""

    backend: NeuralBackend
    device: NeuralDevice
    precision: ModelPrecision
    compilation: CompilationMode
    cpu_threads: int | None
    inference_batch_size: int

    def __post_init__(self) -> None:
        """Reject unsupported combinations rather than rewriting them."""
        if type(self.backend) is not NeuralBackend:
            raise NeuralExecutionSpecError(reason="backend must be NeuralBackend")
        if type(self.device) is not NeuralDevice:
            raise NeuralExecutionSpecError(reason="device must be NeuralDevice")
        if self.precision is not ModelPrecision.FLOAT32:
            raise NeuralExecutionSpecError(reason="only explicit float32 is approved")
        if type(self.compilation) is not CompilationMode:
            raise NeuralExecutionSpecError(reason="compilation must be CompilationMode")
        _validate_inference_batch_size(self.inference_batch_size)
        allowed_devices = {
            NeuralBackend.MLX: {NeuralDevice.CPU, NeuralDevice.MLX_GPU},
            NeuralBackend.PYTORCH: {NeuralDevice.CPU, NeuralDevice.MPS, NeuralDevice.CUDA},
        }[self.backend]
        if self.device not in allowed_devices:
            raise NeuralExecutionSpecError(
                reason=f"{self.device.value} is not a {self.backend.value} device",
            )
        if self.backend is NeuralBackend.PYTORCH and self.compilation is not CompilationMode.EAGER:
            raise NeuralExecutionSpecError(
                reason="PyTorch compilation is not approved for the project geometries",
            )
        if self.device is NeuralDevice.CPU and self.backend is NeuralBackend.PYTORCH:
            if type(self.cpu_threads) is not int or self.cpu_threads <= 0:
                raise NeuralExecutionSpecError(
                    reason="PyTorch CPU requires a positive explicit thread count",
                )
        elif self.cpu_threads is not None:
            raise NeuralExecutionSpecError(
                reason="cpu_threads is valid only for PyTorch CPU execution",
            )


@dataclass(frozen=True, slots=True)
class ModelExecutionSpec:
    """One model variant and its exact execution contract."""

    variant: NeuralVariant
    execution: NeuralExecutionSpec

    def __post_init__(self) -> None:
        """Reject string variants and untyped execution values."""
        if type(self.variant) is not NeuralVariant:
            raise NeuralExecutionSpecError(reason="variant must be NeuralVariant")
        if type(self.execution) is not NeuralExecutionSpec:
            raise NeuralExecutionSpecError(reason="execution must be NeuralExecutionSpec")


def _validate_inference_batch_size(value: object) -> None:
    if type(value) is not int or value <= 0:
        raise NeuralExecutionSpecError(
            reason="inference_batch_size must be a positive plain int",
        )


@dataclass(frozen=True, slots=True)
class NeuralExecutionProfile:
    """One complete explicit execution policy for every project neural variant."""

    name: str
    models: tuple[ModelExecutionSpec, ...]

    def __post_init__(self) -> None:
        """Require a stable name and exactly one execution request per variant."""
        if type(self.name) is not str or not self.name:
            raise NeuralExecutionSpecError(reason="profile name must be a nonempty plain str")
        if type(self.models) is not tuple or any(
            type(model) is not ModelExecutionSpec for model in self.models
        ):
            raise NeuralExecutionSpecError(reason="models must be tuple[ModelExecutionSpec, ...]")
        variants = tuple(model.variant for model in self.models)
        if len(variants) != len(NeuralVariant) or set(variants) != set(NeuralVariant):
            raise NeuralExecutionSpecError(reason="profile must contain every variant exactly once")

    def for_variant(self, variant: NeuralVariant) -> NeuralExecutionSpec:
        """Return one execution request without string coercion."""
        if type(variant) is not NeuralVariant:
            raise NeuralExecutionSpecError(reason="variant must be NeuralVariant")
        return next(model.execution for model in self.models if model.variant is variant)


@dataclass(frozen=True, slots=True)
class NeuralRuntimeIdentity:
    """Resolved package and machine identity for a validated execution request."""

    backend: NeuralBackend
    backend_version: str
    device: NeuralDevice
    precision: ModelPrecision
    compilation: CompilationMode
    cpu_threads: int | None
    operating_system: str
    machine: str


def require_neural_runtime(spec: NeuralExecutionSpec) -> NeuralRuntimeIdentity:
    """Validate and configure exactly the requested runtime or fail without fallback."""
    if type(spec) is not NeuralExecutionSpec:
        raise NeuralExecutionSpecError(reason="spec must be NeuralExecutionSpec")
    expected_version = MLX_VERSION if spec.backend is NeuralBackend.MLX else PYTORCH_VERSION
    package = "mlx" if spec.backend is NeuralBackend.MLX else "torch"
    try:
        observed_version = version(package)
    except PackageNotFoundError as error:
        raise NeuralRuntimeUnavailableError(
            backend=spec.backend,
            device=spec.device,
            reason=f"required package {package}=={expected_version} is not installed",
        ) from error
    observed_base_version = observed_version.partition("+")[0]
    version_matches = (
        observed_version == expected_version
        if spec.backend is NeuralBackend.MLX
        else observed_base_version == expected_version
    )
    if not version_matches:
        raise NeuralRuntimeUnavailableError(
            backend=spec.backend,
            device=spec.device,
            reason=f"requires {package}=={expected_version}; observed {observed_version}",
        )
    if spec.backend is NeuralBackend.MLX:
        _configure_mlx(spec)
    else:
        _configure_pytorch(spec)
    return NeuralRuntimeIdentity(
        backend=spec.backend,
        backend_version=observed_version,
        device=spec.device,
        precision=spec.precision,
        compilation=spec.compilation,
        cpu_threads=spec.cpu_threads,
        operating_system=platform.system(),
        machine=platform.machine(),
    )


def _configure_mlx(spec: NeuralExecutionSpec) -> None:
    try:
        core: Any = importlib.import_module("mlx.core")
    except ModuleNotFoundError as error:
        raise NeuralRuntimeUnavailableError(
            backend=spec.backend,
            device=spec.device,
            reason="mlx.core cannot be imported",
        ) from error
    device = core.cpu if spec.device is NeuralDevice.CPU else core.gpu
    if not core.is_available(device):
        raise NeuralRuntimeUnavailableError(
            backend=spec.backend,
            device=spec.device,
            reason="MLX reports the requested device unavailable",
        )
    core.set_default_device(device)
    if core.default_device() != device:
        raise NeuralRuntimeUnavailableError(
            backend=spec.backend,
            device=spec.device,
            reason="MLX did not retain the requested default device",
        )


def _configure_pytorch(spec: NeuralExecutionSpec) -> None:
    try:
        torch: Any = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise NeuralRuntimeUnavailableError(
            backend=spec.backend,
            device=spec.device,
            reason="torch cannot be imported",
        ) from error
    available = {
        NeuralDevice.CPU: True,
        NeuralDevice.MPS: bool(torch.backends.mps.is_available()),
        NeuralDevice.CUDA: bool(torch.cuda.is_available()),
    }[spec.device]
    if not available:
        raise NeuralRuntimeUnavailableError(
            backend=spec.backend,
            device=spec.device,
            reason="PyTorch reports the requested device unavailable",
        )
    if spec.device is NeuralDevice.CPU:
        torch.set_num_threads(spec.cpu_threads)
        if torch.get_num_threads() != spec.cpu_threads:
            raise NeuralRuntimeUnavailableError(
                backend=spec.backend,
                device=spec.device,
                reason="PyTorch did not retain the requested CPU thread count",
            )


def _execution(
    backend: NeuralBackend,
    device: NeuralDevice,
    compilation: CompilationMode,
    *,
    cpu_threads: int | None = None,
    inference_batch_size: int,
) -> NeuralExecutionSpec:
    return NeuralExecutionSpec(
        backend=backend,
        device=device,
        precision=ModelPrecision.FLOAT32,
        compilation=compilation,
        cpu_threads=cpu_threads,
        inference_batch_size=inference_batch_size,
    )


_MLX_GPU_COMPILED: Final = _execution(
    NeuralBackend.MLX,
    NeuralDevice.MLX_GPU,
    CompilationMode.COMPILED,
    inference_batch_size=8192,
)
_PYTORCH_CPU_1: Final = _execution(
    NeuralBackend.PYTORCH,
    NeuralDevice.CPU,
    CompilationMode.EAGER,
    cpu_threads=1,
    inference_batch_size=8192,
)

PROJECT_APPLE_SILICON_EXECUTION_PROFILE: Final = NeuralExecutionProfile(
    name="project_apple_silicon_m4_pro_v4",
    models=tuple(
        ModelExecutionSpec(
            variant=variant,
            execution=(_PYTORCH_CPU_1 if variant is NeuralVariant.LSTM else _MLX_GPU_COMPILED),
        )
        for variant in NeuralVariant
    ),
)
PROJECT_PORTABLE_CPU_EXECUTION_PROFILE: Final = NeuralExecutionProfile(
    name="project_portable_pytorch_cpu_1_v2",
    models=tuple(
        ModelExecutionSpec(variant=variant, execution=_PYTORCH_CPU_1) for variant in NeuralVariant
    ),
)


__all__ = [
    "MLX_VERSION",
    "PROJECT_APPLE_SILICON_EXECUTION_PROFILE",
    "PROJECT_PORTABLE_CPU_EXECUTION_PROFILE",
    "PYTORCH_VERSION",
    "CompilationMode",
    "ModelExecutionSpec",
    "NeuralBackend",
    "NeuralDevice",
    "NeuralExecutionError",
    "NeuralExecutionProfile",
    "NeuralExecutionSpec",
    "NeuralExecutionSpecError",
    "NeuralRuntimeIdentity",
    "NeuralRuntimeUnavailableError",
    "require_neural_runtime",
]
