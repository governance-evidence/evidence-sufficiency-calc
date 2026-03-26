"""Compare two persisted benchmark JSON runs and summarize deltas.

This utility is intentionally lightweight and read-only. It compares median
runtime values for cases matched by ``(name, parameters)`` and classifies each
delta as a regression, improvement, or within-threshold noise band.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    """Normalized benchmark result row loaded from a persisted JSON file."""

    name: str
    parameters: str
    median_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class ComparisonRow:
    """Comparison summary for a matched benchmark case."""

    name: str
    parameters: str
    baseline_ms: float
    current_ms: float
    delta_ms: float
    delta_pct: float
    classification: str


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for benchmark result comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_json", type=Path, help="Baseline benchmark JSON file")
    parser.add_argument("current_json", type=Path, help="Current benchmark JSON file")
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Percentage delta threshold below which changes are treated as noise (default: 5.0)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return exit code 1 when one or more benchmark regressions are detected",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON file path for a machine-readable comparison report",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional Markdown file path for a human-readable comparison summary",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before comparison begins."""
    if args.threshold_pct < 0.0:
        raise SystemExit("threshold-pct must be non-negative")


def load_benchmark_results(path: Path) -> dict[tuple[str, str], BenchmarkResult]:
    """Load persisted benchmark results indexed by case identity."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError(f"{path} must contain a top-level 'results' list")

    indexed_results: dict[tuple[str, str], BenchmarkResult] = {}
    for index, raw_result in enumerate(raw_results):
        if not isinstance(raw_result, dict):
            raise ValueError(f"{path} results[{index}] must be an object")

        result = BenchmarkResult(
            name=_require_string(raw_result, "name", path, index),
            parameters=_require_string(raw_result, "parameters", path, index),
            median_ms=_require_finite_float(raw_result, "median_ms", path, index),
            min_ms=_require_finite_float(raw_result, "min_ms", path, index),
            max_ms=_require_finite_float(raw_result, "max_ms", path, index),
        )
        result_key = (result.name, result.parameters)
        if result_key in indexed_results:
            raise ValueError(
                f"{path} contains duplicate benchmark case {result.name!r} / {result.parameters!r}"
            )
        indexed_results[result_key] = result

    return indexed_results


def compare_results(
    baseline_results: dict[tuple[str, str], BenchmarkResult],
    current_results: dict[tuple[str, str], BenchmarkResult],
    *,
    threshold_pct: float,
) -> tuple[list[ComparisonRow], list[BenchmarkResult], list[BenchmarkResult]]:
    """Compare current results against a baseline and classify matched cases."""
    comparison_rows: list[ComparisonRow] = []

    shared_keys = sorted(set(baseline_results) & set(current_results))
    for result_key in shared_keys:
        baseline = baseline_results[result_key]
        current = current_results[result_key]
        delta_ms = current.median_ms - baseline.median_ms
        delta_pct = _delta_pct(baseline.median_ms, current.median_ms)
        comparison_rows.append(
            ComparisonRow(
                name=current.name,
                parameters=current.parameters,
                baseline_ms=baseline.median_ms,
                current_ms=current.median_ms,
                delta_ms=delta_ms,
                delta_pct=delta_pct,
                classification=_classify_delta(delta_ms, delta_pct, threshold_pct=threshold_pct),
            )
        )

    baseline_only = [
        baseline_results[key] for key in sorted(set(baseline_results) - set(current_results))
    ]
    current_only = [
        current_results[key] for key in sorted(set(current_results) - set(baseline_results))
    ]
    return comparison_rows, baseline_only, current_only


def _require_string(
    raw_result: dict[str, object],
    field_name: str,
    path: Path,
    index: int,
) -> str:
    """Require a non-empty string field in a persisted benchmark record."""
    value = raw_result.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} results[{index}].{field_name} must be a non-empty string")
    return value


def _require_finite_float(
    raw_result: dict[str, object],
    field_name: str,
    path: Path,
    index: int,
) -> float:
    """Require a finite numeric field in a persisted benchmark record."""
    value = raw_result.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{path} results[{index}].{field_name} must be a finite number")
    return float(value)


def _delta_pct(baseline_ms: float, current_ms: float) -> float:
    """Compute percentage change from baseline to current median runtime."""
    if baseline_ms == 0.0:
        if current_ms == 0.0:
            return 0.0
        return math.inf
    return ((current_ms - baseline_ms) / baseline_ms) * 100.0


def _classify_delta(delta_ms: float, delta_pct: float, *, threshold_pct: float) -> str:
    """Classify a delta row using a percentage noise threshold."""
    if abs(delta_pct) < threshold_pct:
        return "within-threshold"
    if delta_ms > 0.0:
        return "regression"
    if delta_ms < 0.0:
        return "improvement"
    return "within-threshold"


def _print_rows(rows: list[ComparisonRow]) -> None:
    """Render comparison rows as a compact aligned table."""
    if not rows:
        print("No shared benchmark cases to compare.")
        return

    case_labels = [f"{row.name} [{row.parameters}]" for row in rows]
    case_width = max(len("case"), *(len(label) for label in case_labels))
    status_width = max(len("classification"), *(len(row.classification) for row in rows))
    header = (
        f"{'case':<{case_width}}  {'baseline ms':>11}  {'current ms':>10}  "
        f"{'delta ms':>10}  {'delta %':>9}  {'classification':<{status_width}}"
    )
    print(header)
    print("-" * len(header))
    for label, row in zip(case_labels, rows, strict=True):
        print(
            f"{label:<{case_width}}  {row.baseline_ms:>11.3f}  {row.current_ms:>10.3f}  "
            f"{row.delta_ms:>+10.3f}  {_format_pct(row.delta_pct):>9}  "
            f"{row.classification:<{status_width}}"
        )


def _format_pct(delta_pct: float) -> str:
    """Render a delta percentage for terminal display."""
    if math.isinf(delta_pct):
        return "+inf%"
    return f"{delta_pct:+.1f}%"


def _print_unmatched(title: str, results: list[BenchmarkResult]) -> None:
    """Render benchmark cases that appear only in one side of the comparison."""
    if not results:
        return
    print()
    print(title)
    for result in results:
        print(f"- {result.name} [{result.parameters}]")


def _print_summary(rows: list[ComparisonRow]) -> None:
    """Render a small summary count by classification."""
    regressions = sum(row.classification == "regression" for row in rows)
    improvements = sum(row.classification == "improvement" for row in rows)
    within_threshold = sum(row.classification == "within-threshold" for row in rows)
    print()
    print(
        "Summary: "
        f"{regressions} regression(s), "
        f"{improvements} improvement(s), "
        f"{within_threshold} within-threshold"
    )


def _summary_counts(rows: list[ComparisonRow]) -> dict[str, int]:
    """Return summary counts by comparison classification."""
    return {
        "regressions": sum(row.classification == "regression" for row in rows),
        "improvements": sum(row.classification == "improvement" for row in rows),
        "within_threshold": sum(row.classification == "within-threshold" for row in rows),
    }


def _has_regressions(rows: list[ComparisonRow]) -> bool:
    """Return whether the comparison contains one or more regression rows."""
    return any(row.classification == "regression" for row in rows)


def _comparison_row_payload(row: ComparisonRow) -> dict[str, object]:
    """Convert a comparison row to a JSON-serializable payload."""
    return {
        "name": row.name,
        "parameters": row.parameters,
        "baseline_ms": row.baseline_ms,
        "current_ms": row.current_ms,
        "delta_ms": row.delta_ms,
        "delta_pct": row.delta_pct,
        "classification": row.classification,
    }


def _benchmark_result_payload(result: BenchmarkResult) -> dict[str, object]:
    """Convert a benchmark result to a JSON-serializable payload."""
    return {
        "name": result.name,
        "parameters": result.parameters,
        "median_ms": result.median_ms,
        "min_ms": result.min_ms,
        "max_ms": result.max_ms,
    }


def _build_json_report(
    *,
    baseline_json: Path,
    current_json: Path,
    threshold_pct: float,
    fail_on_regression: bool,
    rows: list[ComparisonRow],
    baseline_only: list[BenchmarkResult],
    current_only: list[BenchmarkResult],
) -> dict[str, object]:
    """Build a machine-readable comparison report payload."""
    summary = _summary_counts(rows)
    return {
        "inputs": {
            "baseline_json": str(baseline_json),
            "current_json": str(current_json),
            "threshold_pct": threshold_pct,
            "fail_on_regression": fail_on_regression,
        },
        "summary": {
            **summary,
            "shared_cases": len(rows),
            "baseline_only": len(baseline_only),
            "current_only": len(current_only),
            "has_regressions": summary["regressions"] > 0,
        },
        "rows": [_comparison_row_payload(row) for row in rows],
        "baseline_only": [_benchmark_result_payload(result) for result in baseline_only],
        "current_only": [_benchmark_result_payload(result) for result in current_only],
    }


def _write_json_report(output_path: Path, payload: dict[str, object]) -> None:
    """Persist a machine-readable benchmark comparison report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _markdown_summary_line(rows: list[ComparisonRow]) -> str:
    """Build a compact markdown summary line from comparison rows."""
    summary = _summary_counts(rows)
    return (
        f"- Regressions: {summary['regressions']}\n"
        f"- Improvements: {summary['improvements']}\n"
        f"- Within-threshold: {summary['within_threshold']}"
    )


def _markdown_table(rows: list[ComparisonRow]) -> str:
    """Render matched comparison rows as a markdown table."""
    if not rows:
        return "No shared benchmark cases to compare."

    lines = [
        "| Case | Baseline ms | Current ms | Delta ms | Delta % | Classification |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        [
            "| "
            f"{row.name} [{row.parameters}] | "
            f"{row.baseline_ms:.3f} | "
            f"{row.current_ms:.3f} | "
            f"{row.delta_ms:+.3f} | "
            f"{_format_pct(row.delta_pct)} | "
            f"{row.classification} |"
            for row in rows
        ]
    )
    return "\n".join(lines)


def _markdown_unmatched(title: str, results: list[BenchmarkResult]) -> str:
    """Render unmatched benchmark cases as a markdown section."""
    if not results:
        return f"## {title}\n\nNone."

    lines = [f"## {title}", ""]
    lines.extend(f"- {result.name} [{result.parameters}]" for result in results)
    return "\n".join(lines)


def _build_markdown_report(
    *,
    baseline_json: Path,
    current_json: Path,
    threshold_pct: float,
    fail_on_regression: bool,
    rows: list[ComparisonRow],
    baseline_only: list[BenchmarkResult],
    current_only: list[BenchmarkResult],
) -> str:
    """Build a human-readable markdown comparison report."""
    return "\n\n".join(
        [
            "# Benchmark Comparison Report",
            "## Inputs\n\n"
            f"- Baseline JSON: {baseline_json}\n"
            f"- Current JSON: {current_json}\n"
            f"- Threshold pct: {threshold_pct}\n"
            f"- Fail on regression: {fail_on_regression}",
            f"## Summary\n\n{_markdown_summary_line(rows)}",
            f"## Matched Cases\n\n{_markdown_table(rows)}",
            _markdown_unmatched("Only in Baseline", baseline_only),
            _markdown_unmatched("Only in Current", current_only),
        ]
    )


def _write_markdown_report(output_path: Path, report: str) -> None:
    """Persist a human-readable markdown comparison report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Compare two persisted benchmark JSON files and print a delta report."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    baseline_results = load_benchmark_results(args.baseline_json)
    current_results = load_benchmark_results(args.current_json)
    rows, baseline_only, current_only = compare_results(
        baseline_results,
        current_results,
        threshold_pct=args.threshold_pct,
    )

    print(
        "Benchmark comparison report. "
        "Treat small single-run deltas as noisy unless they persist across repeated runs."
    )
    print()
    _print_rows(rows)
    _print_summary(rows)
    _print_unmatched("Only in baseline:", baseline_only)
    _print_unmatched("Only in current:", current_only)
    if args.json_out is not None:
        payload = _build_json_report(
            baseline_json=args.baseline_json,
            current_json=args.current_json,
            threshold_pct=args.threshold_pct,
            fail_on_regression=args.fail_on_regression,
            rows=rows,
            baseline_only=baseline_only,
            current_only=current_only,
        )
        _write_json_report(args.json_out, payload)
        print()
        print(f"Wrote JSON comparison report to {args.json_out}")
    if args.markdown_out is not None:
        report = _build_markdown_report(
            baseline_json=args.baseline_json,
            current_json=args.current_json,
            threshold_pct=args.threshold_pct,
            fail_on_regression=args.fail_on_regression,
            rows=rows,
            baseline_only=baseline_only,
            current_only=current_only,
        )
        _write_markdown_report(args.markdown_out, report)
        print()
        print(f"Wrote Markdown comparison report to {args.markdown_out}")
    if args.fail_on_regression and _has_regressions(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
