"""Governance context configuration factories.

Provides preset configurations for common deployment contexts.
"""

from __future__ import annotations

from sufficiency._dimensions import COMPLETENESS, FRESHNESS, RELIABILITY, REPRESENTATIVENESS
from sufficiency.types import GovernanceConfig, SufficiencyThresholds


def default_config() -> GovernanceConfig:
    """Equal-weight configuration used in Paper 14 evaluation baseline."""
    return GovernanceConfig()


def fraud_detection_config() -> GovernanceConfig:
    """Fraud detection: fast decay, emphasis on freshness and reliability.

    Fraud patterns evolve rapidly; stale evidence is dangerous.
    Reliability is critical because adversarial drift degrades it silently.
    """
    return GovernanceConfig(
        weights={
            COMPLETENESS: 0.20,
            FRESHNESS: 0.30,
            RELIABILITY: 0.30,
            REPRESENTATIVENESS: 0.20,
        },
        tau_c=0.6,
        tau_r=0.7,
        lambda_freshness=0.02,
        ks_cap=0.30,
        thresholds=SufficiencyThresholds(sufficient=0.8, degraded=0.5),
    )


def credit_scoring_config() -> GovernanceConfig:
    """Credit scoring: slower decay, emphasis on representativeness.

    Borrower behavior changes slowly; representativeness across segments
    matters more than raw freshness. Lambda is lower because credit
    outcomes mature over months, not days.
    """
    return GovernanceConfig(
        weights={
            COMPLETENESS: 0.25,
            FRESHNESS: 0.20,
            RELIABILITY: 0.25,
            REPRESENTATIVENESS: 0.30,
        },
        tau_c=0.6,
        tau_r=0.7,
        lambda_freshness=0.005,
        ks_cap=0.30,
        thresholds=SufficiencyThresholds(sufficient=0.8, degraded=0.5),
    )
