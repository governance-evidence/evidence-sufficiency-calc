"""Example: Blind period analysis for a credit scoring system.

Compares degradation trajectories under three drift types to show
how the sufficiency model differentiates governance risk.
"""

from __future__ import annotations

from sufficiency import BlindPeriodSimulator, DriftSpec, DriftType, credit_scoring_config


def main() -> None:
    config = credit_scoring_config()

    print("=== Credit Scoring: Drift-Type Comparison ===\n")
    print(f"Config: lambda={config.lambda_freshness}, tau_c={config.tau_c}, tau_r={config.tau_r}")
    print()

    drift_scenarios = [
        ("No drift", []),
        ("Covariate P(X)", [DriftSpec(DriftType.COVARIATE, magnitude=0.6)]),
        ("Real concept P(Y|X)", [DriftSpec(DriftType.REAL_CONCEPT, magnitude=0.6)]),
        ("Prior probability P(Y)", [DriftSpec(DriftType.PRIOR_PROBABILITY, magnitude=0.6)]),
    ]

    days = [30, 60, 90, 180]

    header = f"{'Scenario':25s} | " + " | ".join(f"Day {d:3d}" for d in days)
    print(header)
    print("-" * len(header))

    for name, drifts in drift_scenarios:
        sim = BlindPeriodSimulator(config=config, drift_specs=drifts)
        results = sim.simulate(days)
        scores = " | ".join(f"{r.composite:7.3f}" for r in results)
        print(f"{name:25s} | {scores}")

    # Detailed dimension breakdown for concept drift at 90 days
    print("\n=== Dimension Breakdown: Real Concept Drift at Day 90 ===\n")

    sim = BlindPeriodSimulator(
        config=config,
        drift_specs=[DriftSpec(DriftType.REAL_CONCEPT, magnitude=0.6)],
    )
    results = sim.simulate([90])
    r = results[0]

    print(f"Composite: {r.composite:.3f}  (gate: {r.gate:.3f})")
    for name, dim in r.dimensions.items():
        print(f"  {name:22s}: {dim.value:.3f}")


if __name__ == "__main__":
    main()
