"""Cold benchmark for the bounded project cash-bin to TabPFN table allocation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from typing import Final

from benchmarks._support import max_rss_bytes
from benchmarks.neural_training import synthetic_neural_batch
from discrete_dividend_study.neural_spec import NeuralVariant
from discrete_dividend_study.tabpfn_inputs import (
    PROJECT_TABPFN_BINNED_INPUT_SPEC,
    prepare_project_tabpfn_binned_batch,
    prepared_tabpfn_batch_sha256,
)

_POSITIVE_ROWS_ERROR: Final = "rows must be positive"


def main() -> None:
    """Allocate one deterministic table and emit canonical timing and content identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    arguments = parser.parse_args()
    if arguments.rows <= 0:
        raise ValueError(_POSITIVE_ROWS_ERROR)
    source = synthetic_neural_batch(NeuralVariant.BINS, arguments.rows, phase=0)
    started = time.perf_counter()
    prepared = prepare_project_tabpfn_binned_batch(
        source,
        PROJECT_TABPFN_BINNED_INPUT_SPEC,
    )
    elapsed = time.perf_counter() - started
    sys.stdout.write(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "feature_columns": prepared.features.shape[1],
                "feature_payload_bytes": prepared.features.nbytes,
                "machine": platform.machine(),
                "max_rss_bytes": max_rss_bytes(),
                "prepared_sha256": prepared_tabpfn_batch_sha256(prepared),
                "rows": prepared.rows,
                "rows_per_second": prepared.rows / elapsed,
                "source_prepared_batch_sha256": prepared.source_prepared_batch_sha256,
                "target_is_zero_copy": prepared.target is source.target,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
