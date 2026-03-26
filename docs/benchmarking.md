# Benchmarking

For local performance comparisons, run:

```bash
make bench
```

Or call the harness directly with custom parameters:

```bash
python benchmarks/core_paths.py --sizes 1000,5000,20000 --bootstrap 500 --repeat 7
```

To persist a run for later comparison:

```bash
python benchmarks/core_paths.py \
    --sizes 1000,5000,20000 \
    --json-out benchmarks/results/latest.json \
    --csv-out benchmarks/results/latest.csv
```

Or via `make`:

```bash
make bench BENCH_ARGS="--json-out benchmarks/results/latest.json --csv-out benchmarks/results/latest.csv"
```

To refresh the canonical latest run in the standard path:

```bash
make bench-latest
```

Default latest output paths:

- latest JSON: `benchmarks/results/latest.json`
- latest CSV: `benchmarks/results/latest.csv`

Override them when you want to keep multiple latest snapshots:

```bash
make bench-latest \
    BENCH_LATEST_JSON=benchmarks/results/branch-latest.json \
    BENCH_LATEST_CSV=benchmarks/results/branch-latest.csv \
    BENCH_ARGS="--sizes 1000,5000 --repeat 7"
```

To refresh the canonical local baseline in the standard path:

```bash
make bench-baseline
```

Default baseline output paths:

- baseline JSON: `benchmarks/results/baseline.json`
- baseline CSV: `benchmarks/results/baseline.csv`

Override them when you want to maintain multiple baselines, for example per machine or per branch:

```bash
make bench-baseline \
    BENCH_BASELINE=benchmarks/results/m2-baseline.json \
    BENCH_BASELINE_CSV=benchmarks/results/m2-baseline.csv \
    BENCH_ARGS="--sizes 1000,5000 --repeat 7"
```

To run a fresh benchmark, compare it against a baseline, and emit both machine-readable and Markdown reports in one step:

```bash
make bench-report
```

If the configured baseline file does not exist, `make bench-report` stops
immediately with a message telling you to run `make bench-baseline` first
or override `BENCH_BASELINE`. Internally, `make bench-report` reuses
`make bench-latest` for the fresh run step, so the latest-run contract is
consistent across commands.

Default output paths:

- baseline JSON: `benchmarks/results/baseline.json`
- baseline CSV: `benchmarks/results/baseline.csv`
- raw run JSON: `benchmarks/results/latest.json`
- raw run CSV: `benchmarks/results/latest.csv`
- compare JSON: `benchmarks/results/compare.json`
- compare Markdown: `benchmarks/results/compare.md`

Override baseline path, benchmark parameters, threshold, or failure behavior as needed:

```bash
make bench-report \
    BENCH_BASELINE=benchmarks/results/baseline.json \
    BENCH_ARGS="--sizes 1000,5000 --repeat 7" \
    BENCH_COMPARE_THRESHOLD=3.0 \
    BENCH_COMPARE_FAIL=1
```

To compare two persisted JSON runs:

```bash
python benchmarks/compare_results.py \
    benchmarks/results/baseline.json \
    benchmarks/results/latest.json
```

Or via `make`:

```bash
make bench-compare
```

`make bench-compare` uses the standard paths by default:

- baseline input: `benchmarks/results/baseline.json`
- current input: `benchmarks/results/latest.json`
- compare JSON: `benchmarks/results/compare.json`
- compare Markdown: `benchmarks/results/compare.md`

Override any of them when you want to compare custom files:

```bash
make bench-compare \
    BENCH_COMPARE_BASELINE=benchmarks/results/m2-baseline.json \
    BENCH_COMPARE_CURRENT=benchmarks/results/branch-latest.json \
    BENCH_COMPARE_JSON=benchmarks/results/branch-compare.json \
    BENCH_COMPARE_MD=benchmarks/results/branch-compare.md \
    BENCH_COMPARE_THRESHOLD=3.0 \
    BENCH_COMPARE_FAIL=1
```

If either configured input file is missing, `make bench-compare` stops
immediately with a message explaining which file is absent and how to
generate or override it.

To remove only persisted benchmark artifacts without touching caches, coverage, or other build outputs:

```bash
make bench-clean-results
```

This deletes `benchmarks/results/` and leaves the broader `clean` target unchanged.

By default, deltas smaller than `5%` are treated as within-threshold noise rather than a real regression or improvement:

```bash
python benchmarks/compare_results.py \
    benchmarks/results/baseline.json \
    benchmarks/results/latest.json \
    --threshold-pct 3.0
```

To make the compare step fail when any regression is detected:

```bash
python benchmarks/compare_results.py \
    benchmarks/results/baseline.json \
    benchmarks/results/latest.json \
    --fail-on-regression
```

This is useful for CI-style guardrails around benchmark baselines.

To persist a machine-readable comparison report:

```bash
python benchmarks/compare_results.py \
    benchmarks/results/baseline.json \
    benchmarks/results/latest.json \
    --json-out benchmarks/results/compare.json
```

This report includes summary counts, matched-case deltas, and unmatched baseline/current cases.

To persist a Markdown summary suitable for CI artifacts or PR comments:

```bash
python benchmarks/compare_results.py \
    benchmarks/results/baseline.json \
    benchmarks/results/latest.json \
    --markdown-out benchmarks/results/compare.md
```

This report includes a compact summary, a markdown table for matched cases, and sections for unmatched cases.

The harness currently covers:

- `compute_reliability(...)`
- `compute_representativeness(...)`
- `BlindPeriodSimulator.simulate(...)`
- `BlindPeriodSimulator.simulate_daily(...)`

Treat the output as comparative local telemetry, not as a cross-machine performance claim.
