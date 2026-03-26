"""Example: Evidence sufficiency assessment for a fraud detection system.

Demonstrates computing sufficiency scores from production-like data,
then simulating blind period degradation with covariate drift.
"""

from __future__ import annotations

import numpy as np

from sufficiency import (
    BlindPeriodSimulator,
    DriftSpec,
    DriftType,
    compute_sufficiency,
    fraud_detection_config,
)
from sufficiency.dimensions.completeness import compute_completeness
from sufficiency.dimensions.freshness import compute_freshness
from sufficiency.dimensions.reliability import compute_reliability
from sufficiency.dimensions.representativeness import compute_representativeness


def main() -> None:
    config = fraud_detection_config()
    rng = np.random.default_rng(42)

    # --- Current state assessment ---
    print("=== Fraud Detection: Current Evidence Sufficiency ===\n")

    # Simulate production data
    n_decisions = 10_000
    n_labeled = 8_500  # 85% labeled (some still in blind period)
    y_true = rng.integers(0, 2, size=n_labeled)
    y_pred = y_true.copy()
    noise_idx = rng.choice(n_labeled, size=int(n_labeled * 0.12), replace=False)
    y_pred[noise_idx] = 1 - y_pred[noise_idx]

    ref_scores = rng.normal(0.3, 0.15, size=5000)
    prod_scores = rng.normal(0.32, 0.15, size=5000)

    dimensions = {
        "completeness": compute_completeness(n_labeled, n_decisions),
        "freshness": compute_freshness(7.0, config.lambda_freshness),
        "reliability": compute_reliability(y_true, y_pred, rng_seed=42),
        "representativeness": compute_representativeness(ref_scores, prod_scores),
    }

    result = compute_sufficiency(dimensions, config)

    print(f"Composite score: {result.composite:.3f}")
    print(f"Gate A(t):       {result.gate:.3f}")
    print(f"Status:          {result.status.value}")
    print()
    for name, dim in result.dimensions.items():
        print(
            f"  {name:22s}: {dim.value:.3f}  [{dim.confidence_low:.3f}, {dim.confidence_high:.3f}]"
        )

    # --- Blind period simulation ---
    print("\n=== Blind Period Simulation (Covariate Drift) ===\n")

    sim = BlindPeriodSimulator(
        initial_completeness=dimensions["completeness"].value,
        initial_reliability=dimensions["reliability"].value,
        initial_representativeness=dimensions["representativeness"].value,
        config=config,
        drift_specs=[DriftSpec(DriftType.COVARIATE, magnitude=0.6, onset_day=15)],
    )

    for r in sim.simulate([30, 60, 90, 180]):
        print(
            f"  Day {(r.timestamp - sim.start_time).days:3d}: "
            f"S={r.composite:.3f}  A={r.gate:.3f}  "
            f"status={r.status.value}"
        )


if __name__ == "__main__":
    main()
