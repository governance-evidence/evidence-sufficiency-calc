# Configuration Guide

Governance contexts are defined as `GovernanceConfig` dataclasses. The library
provides preset factories for common domains.

After validation, `weights` are stored as an immutable mapping snapshot. Pass a
normal mapping at construction time, but do not rely on mutating
`config.weights` in place later.

## Preset Configurations

### Default (Paper 14 baseline)

```python
from sufficiency import default_config
config = default_config()
# Equal weights (0.25 each), lambda=0.02, tau_c=0.6, tau_r=0.7
```

### Fraud Detection

```python
from sufficiency import fraud_detection_config
config = fraud_detection_config()
# Higher freshness/reliability weights (0.30), lambda=0.02
```

### Credit Scoring

```python
from sufficiency import credit_scoring_config
config = credit_scoring_config()
# Higher representativeness weight (0.30), lambda=0.005 (slower decay)
```

## Custom Configuration

```python
from sufficiency import GovernanceConfig, SufficiencyThresholds

config = GovernanceConfig(
    weights={
        "completeness": 0.20,
        "freshness": 0.25,
        "reliability": 0.35,
        "representativeness": 0.20,
    },
    tau_c=0.5,
    tau_r=0.65,
    lambda_freshness=0.01,
    ks_cap=0.25,
    thresholds=SufficiencyThresholds(sufficient=0.75, degraded=0.45),
)
```

## Parameter Descriptions

| Parameter | Description | Default |
|---|---|---|
| `weights` | Per-dimension weights (must sum to 1.0; stored immutably after validation) | 0.25 each |
| `tau_c` | Completeness threshold for gate | 0.6 |
| `tau_r` | Reliability threshold for gate | 0.7 |
| `lambda_freshness` | Exponential decay rate (per day) | 0.02 |
| `ks_cap` | KS divergence normalization cap | 0.30 |
| `thresholds.sufficient` | S(t) >= this = sufficient | 0.8 |
| `thresholds.degraded` | S(t) >= this = degraded | 0.5 |
