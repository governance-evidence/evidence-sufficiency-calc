# Evidence Sufficiency Calculator

[![CI](https://github.com/governance-evidence/evidence-sufficiency-calc/actions/workflows/ci.yml/badge.svg)](https://github.com/governance-evidence/evidence-sufficiency-calc/actions/workflows/ci.yml)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)
![Python: 3.11-3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A Python library that computes evidence sufficiency scores for governance
assessments in risk decision systems. Answers: *"Is there enough defensible
operational proof to govern this system right now?"*

The package combines four evidence quality dimensions, a decision-readiness
gate, blind-period simulation, and an optional adapter for Decision Event
Schema records.

## For Users

### Install

```bash
pip install -e .
```

Installation modes:

| Mode | Command | Use when |
| --- | --- | --- |
| Base | `pip install -e .` | You only need the core sufficiency scoring library. |
| DES adapter | `pip install -e '.[des]'` | You need Decision Event Schema validation and extraction helpers. |
| Development | `pip install -e '.[dev]'` | You are contributing and need lint, typecheck, test, and pre-commit tooling. |

If you need the Decision Event Schema adapter layer, install the optional `des`
extra:

```bash
pip install -e '.[des]'
```

The JSON Schema used by the adapter ships with this package; no sibling schema
repository checkout is required at runtime.

## For Developers

### Developer Workflow

Quality checks are runnable both locally and in CI.

Run the full local quality gate with:

```bash
make precommit-manual
```

Common development commands:

```bash
make install            # install package with development dependencies
make lint               # Ruff lint + format check
make typecheck          # mypy on src/
make test               # pytest with terminal coverage report
make cov                # pytest with HTML coverage report
make bench              # local micro-benchmark harness for core paths
make bench-baseline     # capture a fresh baseline run in the standard path
make bench-latest       # capture a fresh latest run in the standard path
make bench-clean-results # remove persisted benchmark artifacts only
make check              # lint + typecheck + test
```

Detailed maintainer guidance lives in the docs set:

- [docs/benchmarking.md](docs/benchmarking.md) for benchmark commands, baselines, and comparison reports.
- [docs/development.md](docs/development.md) for contribution and release workflow.

### Quick Start

```python
import numpy as np
from sufficiency import compute_sufficiency, fraud_detection_config
from sufficiency.dimensions.completeness import compute_completeness
from sufficiency.dimensions.freshness import compute_freshness
from sufficiency.dimensions.reliability import compute_reliability
from sufficiency.dimensions.representativeness import compute_representativeness

config = fraud_detection_config()
rng = np.random.default_rng(42)

y_true = rng.integers(0, 2, size=500)
y_pred = y_true.copy()
flip_idx = rng.choice(len(y_true), size=60, replace=False)
y_pred[flip_idx] = 1 - y_pred[flip_idx]

ref_scores = rng.normal(0.30, 0.15, size=1000)
prod_scores = rng.normal(0.32, 0.15, size=1000)

dimensions = {
    "completeness": compute_completeness(labeled_count=8500, total_count=10000),
    "freshness": compute_freshness(delta_t_days=7.0, lambda_rate=config.lambda_freshness),
    "reliability": compute_reliability(y_true, y_pred),
    "representativeness": compute_representativeness(ref_scores, prod_scores),
}

result = compute_sufficiency(dimensions, config)
print(f"S(t) = {result.composite:.3f}  [{result.status.value}]")
```

Additional user-facing reference material:

- `docs/configuration.md` for preset and custom governance configurations.
- `docs/dimensions.md` for the four scoring dimensions and gate semantics.
- `docs/blind_period.md` for blind-period modeling assumptions and drift effects.
- `docs/api_notes.md` for API layers, edge-case contracts, and internal layout notes.

### Examples

Runnable examples are available in `examples/`:

- `examples/fraud_detection.py` shows an end-to-end fraud detection
    assessment from raw signals through composite sufficiency scoring, then
    simulates blind-period degradation under covariate drift.
- `examples/credit_scoring.py` compares blind-period trajectories across
    multiple drift types for a credit scoring policy setting and prints a
    compact scenario table.
- `examples/des_adapter.py` validates a small batch of Decision Event Schema
    records, extracts adapter inputs, and computes an end-to-end sufficiency
    result using explicit proxy scores for the remaining dimensions.
- `examples/lending_club_demo.py` runs the calculator on the Lending Club
    credit dataset. The raw CSV is not bundled here: clone the separate
    `governance-drift-toolkit` repository and prepare its demo data there,
    then this example will read
    `../governance-drift-toolkit/data/lending_club/accepted_2007_to_2018Q4.csv`.
- `examples/ieee_cis_demo.py` runs the calculator on the IEEE-CIS fraud
    dataset. The raw CSV is not bundled here: clone the separate
    `governance-drift-toolkit` repository and prepare its demo data there,
    then this example will read
    `../governance-drift-toolkit/data/ieee_cis/train_transaction.csv`.

Run them with:

```bash
python examples/fraud_detection.py
python examples/credit_scoring.py
python examples/des_adapter.py
python examples/lending_club_demo.py
python examples/ieee_cis_demo.py
```

### Scoring Model

```text
S(t) = A(t) * [w_c*C(t) + w_f*F(t) + w_r*R(t) + w_p*P(t)]
A(t) = min(1, C/tau_c) * min(1, R/tau_r)
```

Four evidence quality dimensions, weighted and modulated by a decision-readiness gate:

| Dimension | Formula | What it measures |
| --- | --- | --- |
| Completeness C(t) | labeled / total | Label coverage |
| Freshness F(t) | exp(-lambda * dt) | Evidence age |
| Reliability R(t) | F1(y_true, y_pred) | Prediction accuracy |
| Representativeness P(t) | max(0, 1 - KS/cap) | Distribution coverage |

The gate A(t) prevents high freshness/representativeness from masking inadequate completeness or reliability.

### Blind Period Simulation

```python
from sufficiency import BlindPeriodSimulator, DriftSpec, DriftType

sim = BlindPeriodSimulator(
    config=config,
    drift_specs=[DriftSpec(DriftType.COVARIATE, magnitude=0.6)],
)
for result in sim.simulate([30, 60, 90, 180]):
    print(f"Day {(result.timestamp - sim.start_time).days}: S={result.composite:.3f}")
```

## Citation

Citation metadata for this project is maintained in `CITATION.cff` at the
repository root.

Until a Zenodo DOI is minted, use the repository URL and release tag as the
canonical software reference.

## Related Projects

This calculator is part of the [governance-evidence](https://github.com/governance-evidence) toolkit:

| Repository | Role | DOI |
| ---------- | ---- | --- |
| [decision-event-schema](https://github.com/governance-evidence/decision-event-schema) | Schema this calculator validates against (bundled copy included) | [10.5281/zenodo.18923178](https://doi.org/10.5281/zenodo.18923178) |
| [governance-drift-toolkit](https://github.com/governance-evidence/governance-drift-toolkit) | Drift monitoring — feeds proxy values to this calculator | Pending |
| [evidence-collector-sdk](https://github.com/governance-evidence/evidence-collector-sdk) | Collects decision events that this calculator scores | Pending |

## License

Apache-2.0
