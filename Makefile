.PHONY: install lint format test cov typecheck check bench bench-baseline bench-latest bench-compare bench-report bench-clean-results precommit-manual demo-ieee clean

PYTHON ?= ./.venv/bin/python
PIP ?= $(PYTHON) -m pip
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
PYTEST ?= $(PYTHON) -m pytest
PRE_COMMIT ?= $(PYTHON) -m pre_commit

BENCH_BASELINE ?= benchmarks/results/baseline.json
BENCH_BASELINE_CSV ?= benchmarks/results/baseline.csv
BENCH_LATEST_JSON ?= benchmarks/results/latest.json
BENCH_LATEST_CSV ?= benchmarks/results/latest.csv
BENCH_COMPARE_BASELINE ?= $(BENCH_BASELINE)
BENCH_COMPARE_CURRENT ?= $(BENCH_LATEST_JSON)
BENCH_COMPARE_JSON ?= benchmarks/results/compare.json
BENCH_COMPARE_MD ?= benchmarks/results/compare.md
BENCH_COMPARE_THRESHOLD ?= 5.0
BENCH_COMPARE_FAIL ?= 0

install:
	$(PIP) install -e ".[dev]"

lint:
	$(RUFF) check src/ tests/ examples/ benchmarks/
	$(RUFF) format --check src/ tests/ examples/ benchmarks/

format:
	$(RUFF) check --fix src/ tests/ examples/ benchmarks/
	$(RUFF) format src/ tests/ examples/ benchmarks/

typecheck:
	$(MYPY) src/

test:
	$(PYTEST) --cov=sufficiency --cov-report=term-missing

cov:
	$(PYTEST) --cov=sufficiency --cov-report=term-missing --cov-report=html

check: lint typecheck test

bench:
	$(PYTHON) benchmarks/core_paths.py $(BENCH_ARGS)

bench-baseline:
	$(PYTHON) benchmarks/core_paths.py $(BENCH_ARGS) --json-out $(BENCH_BASELINE) --csv-out $(BENCH_BASELINE_CSV)

bench-latest:
	$(PYTHON) benchmarks/core_paths.py $(BENCH_ARGS) --json-out $(BENCH_LATEST_JSON) --csv-out $(BENCH_LATEST_CSV)

bench-compare:
	@test -f "$(BENCH_COMPARE_BASELINE)" || { echo "Missing comparison baseline: $(BENCH_COMPARE_BASELINE). Run 'make bench-baseline' first or override BENCH_COMPARE_BASELINE=..." >&2; exit 1; }
	@test -f "$(BENCH_COMPARE_CURRENT)" || { echo "Missing comparison current run: $(BENCH_COMPARE_CURRENT). Generate it with 'make bench' or 'make bench-report', or override BENCH_COMPARE_CURRENT=..." >&2; exit 1; }
	$(PYTHON) benchmarks/compare_results.py $(BENCH_COMPARE_BASELINE) $(BENCH_COMPARE_CURRENT) --threshold-pct $(BENCH_COMPARE_THRESHOLD) --json-out $(BENCH_COMPARE_JSON) --markdown-out $(BENCH_COMPARE_MD) $(if $(filter 1 true TRUE yes YES,$(BENCH_COMPARE_FAIL)),--fail-on-regression,)

bench-report:
	@test -f "$(BENCH_BASELINE)" || { echo "Missing benchmark baseline: $(BENCH_BASELINE). Run 'make bench-baseline' first or override BENCH_BASELINE=..." >&2; exit 1; }
	$(MAKE) bench-latest BENCH_ARGS='$(BENCH_ARGS)' BENCH_LATEST_JSON='$(BENCH_LATEST_JSON)' BENCH_LATEST_CSV='$(BENCH_LATEST_CSV)'
	$(PYTHON) benchmarks/compare_results.py $(BENCH_BASELINE) $(BENCH_LATEST_JSON) --threshold-pct $(BENCH_COMPARE_THRESHOLD) --json-out $(BENCH_COMPARE_JSON) --markdown-out $(BENCH_COMPARE_MD) $(if $(filter 1 true TRUE yes YES,$(BENCH_COMPARE_FAIL)),--fail-on-regression,)

bench-clean-results:
	rm -rf benchmarks/results

demo-ieee:
	$(PYTHON) examples/ieee_cis_demo.py

precommit-manual:
	$(PRE_COMMIT) run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
