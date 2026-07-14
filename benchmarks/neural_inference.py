"""Cold one-pass benchmark for explicit neural inference batch geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from typing import Final

from benchmarks._support import max_rss_bytes
from benchmarks.neural_training import PARAMETER_SEED, synthetic_neural_batch
from discrete_dividend_study.mlx_neural import build_mlx_project_model
from discrete_dividend_study.mlx_training import predict_mlx_project_model
from discrete_dividend_study.neural_execution import (
    PROJECT_APPLE_SILICON_EXECUTION_PROFILE,
    NeuralBackend,
)
from discrete_dividend_study.neural_parameters import initialize_neural_parameters
from discrete_dividend_study.neural_spec import PROJECT_NEURAL_STUDY_SPEC, NeuralVariant
from discrete_dividend_study.torch_neural import build_torch_project_model
from discrete_dividend_study.torch_training import predict_torch_project_model

_POSITIVE_ARGUMENTS_ERROR: Final = "rows and batch size must be positive"


def main() -> None:
    """Run one cold isolated prediction pass and emit one canonical JSON object."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in NeuralVariant],
        required=True,
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.rows <= 0 or arguments.batch_size <= 0:
        raise ValueError(_POSITIVE_ARGUMENTS_ERROR)

    variant = NeuralVariant(arguments.variant)
    batch = synthetic_neural_batch(variant, arguments.rows, phase=2)
    execution = PROJECT_APPLE_SILICON_EXECUTION_PROFILE.for_variant(variant)
    parameters = initialize_neural_parameters(
        PROJECT_NEURAL_STUDY_SPEC,
        batch.model,
        PARAMETER_SEED,
    )
    if execution.backend is NeuralBackend.MLX:
        network = build_mlx_project_model(
            study=PROJECT_NEURAL_STUDY_SPEC,
            model=batch.model,
            parameters=parameters,
            execution=execution,
        )
        started = time.perf_counter()
        prediction = predict_mlx_project_model(
            network,
            batch,
            batch_size=arguments.batch_size,
        )
    else:
        network = build_torch_project_model(
            study=PROJECT_NEURAL_STUDY_SPEC,
            model=batch.model,
            parameters=parameters,
            execution=execution,
        )
        started = time.perf_counter()
        prediction = predict_torch_project_model(
            network,
            batch,
            batch_size=arguments.batch_size,
        )
    elapsed = time.perf_counter() - started
    sys.stdout.write(
        json.dumps(
            {
                "backend": execution.backend.value,
                "batch_size": arguments.batch_size,
                "compilation": execution.compilation.value,
                "device": execution.device.value,
                "elapsed_seconds": elapsed,
                "machine": platform.machine(),
                "max_rss_bytes": max_rss_bytes(),
                "prediction_sha256": hashlib.sha256(
                    memoryview(prediction).cast("B"),
                ).hexdigest(),
                "profile": PROJECT_APPLE_SILICON_EXECUTION_PROFILE.name,
                "rows": arguments.rows,
                "rows_per_second": arguments.rows / elapsed,
                "variant": variant.value,
            },
            sort_keys=True,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
