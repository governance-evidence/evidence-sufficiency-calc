"""Micro-benchmark harness for core evidence sufficiency paths.

This script is intentionally lightweight and stdlib-only. It is designed for
local comparative runs while refactoring or changing numerical settings, not as
a statistically rigorous replacement for dedicated benchmarking frameworks.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sufficiency import BlindPeriodSimulator, DriftSpec, DriftType, default_config
from sufficiency.dimensions.reliability import compute_reliability
from sufficiency.dimensions.representativeness import compute_representativeness

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary statistics for one benchmark case."""

    name: str
    parameters: str
    median_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class BenchmarkRunMetadata:
    """Execution metadata for one benchmark harness run."""

    generated_at_utc: str
    sizes: list[int]
    bootstrap: int
    repeat: int
    warmup: int
    daily_days: int
    seed: int


def _time_callable(
    benchmark_fn: Callable[[], object],
    *,
    repeat: int,
    warmup: int,
) -> list[float]:
    """Return elapsed times in milliseconds for a benchmarkable callable."""
    for _ in range(warmup):
        benchmark_fn()

    elapsed_ms: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        benchmark_fn()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    return elapsed_ms


def _summarize(name: str, parameters: str, samples_ms: list[float]) -> BenchmarkResult:
    """Build a compact summary row from timing samples."""
    return BenchmarkResult(
        name=name,
        parameters=parameters,
        median_ms=statistics.median(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
    )


def _benchmark_reliability(
    *,
    size: int,
    bootstrap: int,
    repeat: int,
    warmup: int,
    seed: int,
) -> BenchmarkResult:
    """Benchmark reliability scoring on a synthetic binary classification batch."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=size)
    y_pred = y_true.copy()
    flip_count = max(1, size // 8)
    flip_idx = rng.choice(size, size=flip_count, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    def run() -> object:
        return compute_reliability(
            y_true,
            y_pred,
            n_bootstrap=bootstrap,
            rng_seed=seed,
        )

    samples_ms = _time_callable(run, repeat=repeat, warmup=warmup)
    return _summarize(
        "reliability",
        f"n={size}, bootstrap={bootstrap}",
        samples_ms,
    )


def _benchmark_representativeness(
    *,
    size: int,
    repeat: int,
    warmup: int,
    seed: int,
) -> BenchmarkResult:
    """Benchmark KS-based representativeness scoring on synthetic score arrays."""
    rng = np.random.default_rng(seed)
    reference = rng.normal(0.30, 0.15, size=size)
    production = rng.normal(0.32, 0.15, size=size)

    def run() -> object:
        return compute_representativeness(reference, production)

    samples_ms = _time_callable(run, repeat=repeat, warmup=warmup)
    return _summarize("representativeness", f"n={size}", samples_ms)


def _benchmark_blind_period_checkpoints(
    *,
    repeat: int,
    warmup: int,
) -> BenchmarkResult:
    """Benchmark the standard checkpoint simulation path."""
    simulator = BlindPeriodSimulator(
        config=default_config(),
        drift_specs=[DriftSpec(DriftType.COVARIATE, magnitude=0.6, onset_day=15)],
    )

    def run() -> object:
        return simulator.simulate([30, 60, 90, 180])

    samples_ms = _time_callable(run, repeat=repeat, warmup=warmup)
    return _summarize("blind_period_checkpoints", "days=30,60,90,180", samples_ms)


def _benchmark_blind_period_daily(
    *,
    total_days: int,
    repeat: int,
    warmup: int,
) -> BenchmarkResult:
    """Benchmark the full daily blind-period trajectory path."""
    simulator = BlindPeriodSimulator(
        config=default_config(),
        drift_specs=[DriftSpec(DriftType.REAL_CONCEPT, magnitude=0.6, onset_day=20)],
    )

    def run() -> object:
        return simulator.simulate_daily(total_days)

    samples_ms = _time_callable(run, repeat=repeat, warmup=warmup)
    return _summarize("blind_period_daily", f"days={total_days}", samples_ms)


def _parse_sizes(value: str) -> list[int]:
    """Parse a comma-separated list of positive benchmark sizes."""
    try:
        sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        msg = f"sizes must be a comma-separated list of integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc

    if not sizes or any(size <= 0 for size in sizes):
        msg = f"sizes must contain only positive integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg)

    return sizes


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the benchmark harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=[1_000, 10_000],
        help="Comma-separated sample sizes for array-based benchmarks (default: 1000,10000)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=300,
        help="Bootstrap resamples for reliability (default: 300)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Timed repetitions per case (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per case before timing (default: 1)",
    )
    parser.add_argument(
        "--daily-days",
        type=int,
        default=180,
        help="Daily blind-period horizon to benchmark (default: 180)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic benchmark inputs (default: 42)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON file path for benchmark results",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV file path for benchmark results",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate positive CLI numeric arguments."""
    for name in ("bootstrap", "repeat", "warmup", "daily_days"):
        value = getattr(args, name)
        if value < 0 or (name != "warmup" and value == 0):
            msg = f"{name.replace('_', '-')} must be positive"
            raise SystemExit(msg)


def _print_results(results: list[BenchmarkResult]) -> None:
    """Render a compact benchmark summary table."""
    name_width = max(len(result.name) for result in results)
    params_width = max(len(result.parameters) for result in results)

    header = (
        f"{'case':<{name_width}}  "
        f"{'parameters':<{params_width}}  "
        f"{'median ms':>10}  {'min ms':>10}  {'max ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<{name_width}}  "
            f"{result.parameters:<{params_width}}  "
            f"{result.median_ms:>10.3f}  "
            f"{result.min_ms:>10.3f}  "
            f"{result.max_ms:>10.3f}"
        )


def _build_metadata(args: argparse.Namespace) -> BenchmarkRunMetadata:
    """Construct metadata for a benchmark harness invocation."""
    return BenchmarkRunMetadata(
        generated_at_utc=datetime.now(UTC).isoformat(),
        sizes=list(args.sizes),
        bootstrap=args.bootstrap,
        repeat=args.repeat,
        warmup=args.warmup,
        daily_days=args.daily_days,
        seed=args.seed,
    )


def _write_json_results(
    output_path: Path,
    *,
    metadata: BenchmarkRunMetadata,
    results: list[BenchmarkResult],
) -> None:
    """Write benchmark results and run metadata to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "generated_at_utc": metadata.generated_at_utc,
            "sizes": metadata.sizes,
            "bootstrap": metadata.bootstrap,
            "repeat": metadata.repeat,
            "warmup": metadata.warmup,
            "daily_days": metadata.daily_days,
            "seed": metadata.seed,
        },
        "results": [
            {
                "name": result.name,
                "parameters": result.parameters,
                "median_ms": result.median_ms,
                "min_ms": result.min_ms,
                "max_ms": result.max_ms,
            }
            for result in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv_results(output_path: Path, results: list[BenchmarkResult]) -> None:
    """Write benchmark result rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "parameters", "median_ms", "min_ms", "max_ms"])
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.parameters,
                    f"{result.median_ms:.6f}",
                    f"{result.min_ms:.6f}",
                    f"{result.max_ms:.6f}",
                ]
            )


def _write_outputs(
    *,
    metadata: BenchmarkRunMetadata,
    results: list[BenchmarkResult],
    json_out: Path | None,
    csv_out: Path | None,
) -> None:
    """Persist benchmark results to any configured output files."""
    if json_out is not None:
        _write_json_results(json_out, metadata=metadata, results=results)
        print(f"Wrote JSON results to {json_out}")
    if csv_out is not None:
        _write_csv_results(csv_out, results)
        print(f"Wrote CSV results to {csv_out}")


def main() -> None:
    """Run the configured benchmark suite."""
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    metadata = _build_metadata(args)

    results: list[BenchmarkResult] = []
    for size in args.sizes:
        results.append(
            _benchmark_reliability(
                size=size,
                bootstrap=args.bootstrap,
                repeat=args.repeat,
                warmup=args.warmup,
                seed=args.seed,
            )
        )
        results.append(
            _benchmark_representativeness(
                size=size,
                repeat=args.repeat,
                warmup=args.warmup,
                seed=args.seed,
            )
        )

    results.append(
        _benchmark_blind_period_checkpoints(
            repeat=args.repeat,
            warmup=args.warmup,
        )
    )
    results.append(
        _benchmark_blind_period_daily(
            total_days=args.daily_days,
            repeat=args.repeat,
            warmup=args.warmup,
        )
    )

    print(
        "Micro-benchmark harness for comparative local runs. "
        "Prefer comparing deltas on the same machine over treating absolute values as portable."
    )
    print()
    _print_results(results)
    if args.json_out is not None or args.csv_out is not None:
        print()
    _write_outputs(
        metadata=metadata,
        results=results,
        json_out=args.json_out,
        csv_out=args.csv_out,
    )


if __name__ == "__main__":
    main()
