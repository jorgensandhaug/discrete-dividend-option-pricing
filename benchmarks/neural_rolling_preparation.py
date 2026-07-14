"""Cold isolated benchmark for full-horizon versus repeated rolling preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from finoptions import CpuExecutor

from benchmarks._support import max_rss_bytes
from discrete_dividend_study.neural_rolling import (
    PROJECT_NEURAL_ROLLING_PREPARATION_SPEC,
    prepare_project_neural_rolling_source,
)
from discrete_dividend_study.neural_spec import PROJECT_NEURAL_STUDY_SPEC
from discrete_dividend_study.neural_window import prepare_project_neural_window
from discrete_dividend_study.rolling_windows import (
    RollingWindow,
    RollingWindowSpec,
    generate_rolling_windows,
)

_DAY_ZERO: Final = date(2020, 1, 2)
_WINDOW_DAYS: Final = 30
_LOOKBACK_DAYS: Final = 90
_POSITIVE_ARGUMENTS_ERROR: Final = "rows, windows, and threads must be positive"


class PreparationStrategy(StrEnum):
    """Explicit isolated preparation strategies admitted by the benchmark."""

    FULL_HORIZON = "full_horizon"
    REPEATED_WINDOWS = "repeated_windows"


def main() -> None:
    """Run one strategy against one synthetic Parquet artifact and emit canonical JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in PreparationStrategy],
        required=True,
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.rows <= 0 or arguments.windows <= 0 or arguments.threads <= 0:
        raise ValueError(_POSITIVE_ARGUMENTS_ERROR)
    windows = _windows(arguments.windows)
    with tempfile.TemporaryDirectory(prefix="neural-rolling-benchmark-") as directory:
        artifact = Path(directory) / "synthetic-features.parquet"
        _synthetic_frame(arguments.rows, windows).write_parquet(artifact)
        executor = CpuExecutor(threads=arguments.threads)
        strategy = PreparationStrategy(arguments.strategy)
        started = time.perf_counter()
        checksum, phase_count = (
            _prepare_full_horizon(artifact, windows, executor)
            if strategy is PreparationStrategy.FULL_HORIZON
            else _prepare_repeated_windows(artifact, windows, executor)
        )
        elapsed = time.perf_counter() - started
    sys.stdout.write(
        json.dumps(
            {
                "checksum": checksum,
                "elapsed_seconds": elapsed,
                "machine": platform.machine(),
                "max_rss_bytes": max_rss_bytes(),
                "model_windows_per_second": (len(PROJECT_NEURAL_STUDY_SPEC.models) * len(windows))
                / elapsed,
                "phase_count": phase_count,
                "rows": arguments.rows,
                "strategy": strategy.value,
                "threads": arguments.threads,
                "windows": len(windows),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
    )


def _prepare_full_horizon(
    artifact: Path,
    windows: tuple[RollingWindow, ...],
    executor: CpuExecutor,
) -> tuple[str, int]:
    source = prepare_project_neural_rolling_source(
        pl.scan_parquet(artifact),
        windows,
        PROJECT_NEURAL_STUDY_SPEC,
        executor,
    )
    hasher = hashlib.sha256()
    phases = 0
    for model in PROJECT_NEURAL_STUDY_SPEC.models:
        for window in windows:
            prepared = source.window(window.window_id, model)
            for digest in (
                prepared.training_data_sha256,
                prepared.validation_data_sha256,
                prepared.test_data_sha256,
            ):
                hasher.update(bytes.fromhex(digest))
                phases += 1
    return hasher.hexdigest(), phases


def _prepare_repeated_windows(
    artifact: Path,
    windows: tuple[RollingWindow, ...],
    executor: CpuExecutor,
) -> tuple[str, int]:
    source = pl.scan_parquet(artifact)
    hasher = hashlib.sha256()
    phases = 0
    for model in PROJECT_NEURAL_STUDY_SPEC.models:
        for window in windows:
            prepared = prepare_project_neural_window(
                source,
                window,
                model,
                executor,
                PROJECT_NEURAL_ROLLING_PREPARATION_SPEC,
            )
            for digest in (
                prepared.training_data_sha256,
                prepared.validation_data_sha256,
                prepared.test_data_sha256,
            ):
                hasher.update(bytes.fromhex(digest))
                phases += 1
    return hasher.hexdigest(), phases


def _windows(count: int) -> tuple[RollingWindow, ...]:
    first_test_start = _DAY_ZERO + timedelta(days=_LOOKBACK_DAYS)
    return generate_rolling_windows(
        RollingWindowSpec(
            total_lookback_days=_LOOKBACK_DAYS,
            validation_days=_WINDOW_DAYS,
            test_days=_WINDOW_DAYS,
            step_days=_WINDOW_DAYS,
            first_test_start=first_test_start,
            latest_test_end=first_test_start + timedelta(days=count * _WINDOW_DAYS),
        ),
    )


def _synthetic_frame(rows: int, windows: tuple[RollingWindow, ...]) -> pl.DataFrame:
    horizon_start = windows[0].train_start
    horizon_end = windows[-1].test_end
    horizon_days = (horizon_end - horizon_start).days
    position = (np.arange(rows, dtype=np.float64) + 0.5) / rows
    spot = 90.0 + 20.0 * position
    strike = 80.0 + 40.0 * position
    day_offsets = np.minimum((position * horizon_days).astype(np.int64), horizon_days - 1)
    quote_dates = np.datetime64(horizon_start, "D") + day_offsets.astype("timedelta64[D]")
    return pl.DataFrame(
        {
            "quote_date": pl.Series(quote_dates).cast(pl.Date),
            "strike": pl.Series(strike).cast(pl.Float32),
            "underlying_price": pl.Series(spot).cast(pl.Float32),
            "risk_free_rate": 0.01 + 0.03 * position,
            "call_moneyness": pl.Series(spot / strike).cast(pl.Float32),
            "historical_volatility_90": 0.15 + 0.20 * position,
            "yte": 0.5 + 0.5 * position,
            "forecasted_dividend_yield": 0.005 + 0.035 * position,
            "historical_dividend_yield": 0.01 + 0.025 * position,
            "call_mid_price": pl.Series(2.0 + np.maximum(spot - strike, 0.0)).cast(pl.Float32),
        },
    ).with_columns(
        pl.lit([0.1, 0.2], dtype=pl.List(pl.Float64)).alias(
            "forecasted_years_to_discrete_ex_dates",
        ),
        pl.lit([0.5, 0.5], dtype=pl.List(pl.Float64)).alias(
            "forecasted_discrete_amounts",
        ),
    )


if __name__ == "__main__":
    main()
