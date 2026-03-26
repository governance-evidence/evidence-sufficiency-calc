"""Tests for persisted benchmark result comparison utilities."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from benchmarks.compare_results import (
    _delta_pct,
    _format_pct,
    _validate_args,
    compare_results,
    load_benchmark_results,
    main,
)


def _write_results(path: Path, results: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"metadata": {}, "results": results}) + "\n", encoding="utf-8")


def test_load_benchmark_results_rejects_duplicate_cases(tmp_path: Path) -> None:
    payload_path = tmp_path / "duplicate.json"
    _write_results(
        payload_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            },
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.2,
                "min_ms": 1.1,
                "max_ms": 1.3,
            },
        ],
    )

    with pytest.raises(ValueError, match="duplicate benchmark case"):
        load_benchmark_results(payload_path)


def test_compare_results_classifies_rows_and_unmatched_cases(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            },
            {
                "name": "representativeness",
                "parameters": "n=1000",
                "median_ms": 2.0,
                "min_ms": 1.8,
                "max_ms": 2.2,
            },
            {
                "name": "blind_period_daily",
                "parameters": "days=90",
                "median_ms": 4.0,
                "min_ms": 3.8,
                "max_ms": 4.2,
            },
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.2,
                "min_ms": 1.1,
                "max_ms": 1.3,
            },
            {
                "name": "representativeness",
                "parameters": "n=1000",
                "median_ms": 1.7,
                "min_ms": 1.6,
                "max_ms": 1.8,
            },
            {
                "name": "blind_period_checkpoints",
                "parameters": "days=30,60,90,180",
                "median_ms": 0.1,
                "min_ms": 0.1,
                "max_ms": 0.1,
            },
        ],
    )

    rows, baseline_only, current_only = compare_results(
        load_benchmark_results(baseline_path),
        load_benchmark_results(current_path),
        threshold_pct=5.0,
    )

    assert [row.classification for row in rows] == ["regression", "improvement"]
    assert rows[0].name == "reliability"
    assert rows[1].name == "representativeness"
    assert [result.name for result in baseline_only] == ["blind_period_daily"]
    assert [result.name for result in current_only] == ["blind_period_checkpoints"]


def test_compare_results_marks_small_changes_within_threshold(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.03,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )

    rows, _, _ = compare_results(
        load_benchmark_results(baseline_path),
        load_benchmark_results(current_path),
        threshold_pct=5.0,
    )

    assert rows[0].classification == "within-threshold"


def test_main_prints_summary_and_unmatched_sections(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.2,
                "min_ms": 1.1,
                "max_ms": 1.3,
            },
            {
                "name": "blind_period_daily",
                "parameters": "days=90",
                "median_ms": 4.0,
                "min_ms": 3.8,
                "max_ms": 4.2,
            },
        ],
    )

    exit_code = main([str(baseline_path), str(current_path), "--threshold-pct", "5.0"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Summary: 1 regression(s), 0 improvement(s), 0 within-threshold" in captured
    assert "Only in current:" in captured
    assert "blind_period_daily [days=90]" in captured


def test_main_fails_when_regressions_exist_and_flag_is_enabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.2,
                "min_ms": 1.1,
                "max_ms": 1.3,
            }
        ],
    )

    exit_code = main(
        [
            str(baseline_path),
            str(current_path),
            "--threshold-pct",
            "5.0",
            "--fail-on-regression",
        ]
    )
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "Summary: 1 regression(s), 0 improvement(s), 0 within-threshold" in captured


def test_main_succeeds_with_fail_flag_when_no_regressions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 0.8,
                "min_ms": 0.7,
                "max_ms": 0.9,
            }
        ],
    )

    exit_code = main(
        [
            str(baseline_path),
            str(current_path),
            "--threshold-pct",
            "5.0",
            "--fail-on-regression",
        ]
    )
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Summary: 0 regression(s), 1 improvement(s), 0 within-threshold" in captured


def test_main_writes_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    report_path = tmp_path / "report.json"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.2,
                "min_ms": 1.1,
                "max_ms": 1.3,
            },
            {
                "name": "blind_period_daily",
                "parameters": "days=90",
                "median_ms": 4.0,
                "min_ms": 3.8,
                "max_ms": 4.2,
            },
        ],
    )

    exit_code = main(
        [
            str(baseline_path),
            str(current_path),
            "--threshold-pct",
            "5.0",
            "--json-out",
            str(report_path),
        ]
    )
    captured = capsys.readouterr().out
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "Wrote JSON comparison report to" in captured
    assert payload["inputs"]["baseline_json"] == str(baseline_path)
    assert payload["inputs"]["current_json"] == str(current_path)
    assert payload["summary"] == {
        "regressions": 1,
        "improvements": 0,
        "within_threshold": 0,
        "shared_cases": 1,
        "baseline_only": 0,
        "current_only": 1,
        "has_regressions": True,
    }
    assert payload["rows"][0]["classification"] == "regression"
    assert payload["current_only"][0]["name"] == "blind_period_daily"


def test_main_writes_markdown_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    report_path = tmp_path / "report.md"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.2,
                "min_ms": 1.1,
                "max_ms": 1.3,
            },
            {
                "name": "blind_period_daily",
                "parameters": "days=90",
                "median_ms": 4.0,
                "min_ms": 3.8,
                "max_ms": 4.2,
            },
        ],
    )

    exit_code = main(
        [
            str(baseline_path),
            str(current_path),
            "--threshold-pct",
            "5.0",
            "--markdown-out",
            str(report_path),
        ]
    )
    captured = capsys.readouterr().out
    report = report_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert "Wrote Markdown comparison report to" in captured
    assert "# Benchmark Comparison Report" in report
    assert "## Summary" in report
    assert "- Regressions: 1" in report
    assert "| reliability [n=1000] | 1.000 | 1.200 | +0.200 | +20.0% | regression |" in report
    assert "## Only in Current" in report
    assert "- blind_period_daily [days=90]" in report


def test_main_writes_markdown_report_without_shared_cases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    report_path = tmp_path / "report.md"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "blind_period_daily",
                "parameters": "days=90",
                "median_ms": 4.0,
                "min_ms": 3.8,
                "max_ms": 4.2,
            }
        ],
    )

    exit_code = main(
        [
            str(baseline_path),
            str(current_path),
            "--markdown-out",
            str(report_path),
        ]
    )
    capsys.readouterr()
    report = report_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert "No shared benchmark cases to compare." in report
    assert "## Only in Baseline" in report
    assert "## Only in Current" in report


def test_validate_args_rejects_negative_threshold() -> None:
    class Args:
        threshold_pct = -1.0

    with pytest.raises(SystemExit, match="threshold-pct must be non-negative"):
        _validate_args(Args())


def test_load_benchmark_results_rejects_missing_results_list(tmp_path: Path) -> None:
    payload_path = tmp_path / "missing-results.json"
    payload_path.write_text(json.dumps({"metadata": {}}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level 'results' list"):
        load_benchmark_results(payload_path)


def test_load_benchmark_results_rejects_non_object_row(tmp_path: Path) -> None:
    payload_path = tmp_path / "bad-row.json"
    _write_results(payload_path, [123])

    with pytest.raises(ValueError, match=r"results\[0\] must be an object"):
        load_benchmark_results(payload_path)


def test_load_benchmark_results_rejects_empty_name(tmp_path: Path) -> None:
    payload_path = tmp_path / "bad-name.json"
    _write_results(
        payload_path,
        [
            {
                "name": "",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )

    with pytest.raises(ValueError, match=r"results\[0\]\.name must be a non-empty string"):
        load_benchmark_results(payload_path)


def test_load_benchmark_results_rejects_non_finite_numeric_field(tmp_path: Path) -> None:
    payload_path = tmp_path / "bad-number.json"
    _write_results(
        payload_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": float("inf"),
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )

    with pytest.raises(ValueError, match=r"results\[0\]\.median_ms must be a finite number"):
        load_benchmark_results(payload_path)


def test_delta_pct_handles_zero_baseline() -> None:
    assert _delta_pct(0.0, 0.0) == 0.0
    assert _delta_pct(0.0, 1.0) == float("inf")


def test_format_pct_renders_infinite_delta() -> None:
    assert _format_pct(float("inf")) == "+inf%"


def test_compare_results_marks_exact_zero_delta_with_zero_threshold(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )

    rows, _, _ = compare_results(
        load_benchmark_results(baseline_path),
        load_benchmark_results(current_path),
        threshold_pct=0.0,
    )

    assert rows[0].classification == "within-threshold"


def test_main_prints_no_shared_case_message(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_results(
        baseline_path,
        [
            {
                "name": "reliability",
                "parameters": "n=1000",
                "median_ms": 1.0,
                "min_ms": 0.9,
                "max_ms": 1.1,
            }
        ],
    )
    _write_results(
        current_path,
        [
            {
                "name": "blind_period_daily",
                "parameters": "days=90",
                "median_ms": 4.0,
                "min_ms": 3.8,
                "max_ms": 4.2,
            }
        ],
    )

    exit_code = main([str(baseline_path), str(current_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "No shared benchmark cases to compare." in captured
    assert "Only in baseline:" in captured
    assert "Only in current:" in captured
