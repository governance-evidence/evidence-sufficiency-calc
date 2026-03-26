# Blind Period Modeling

The blind period is the time between a decision and the arrival of its
outcome label. During this period, governance evidence degrades through
mechanisms that depend on drift type.

## Drift-Type Impact Matrix

| Dimension | Covariate P(X) | Real Concept P(Y\|X) | Prior Probability P(Y) |
|---|---|---|---|
| Completeness | Unchanged | Unchanged | Degraded |
| Freshness | Decays (always) | Decays (always) | Decays (always) |
| Reliability | Mildly degraded | Severely degraded | Moderately degraded |
| Representativeness | Severely degraded | Unchanged | Moderately degraded |
| Observable without labels | Yes | No | Partially |

## Simulation Protocol

The `BlindPeriodSimulator` models degradation by:

1. Starting from initial dimension scores (pre-blind-period values)
2. Applying exponential freshness decay at every time step
3. Applying linear completeness decay (unlabeled decisions accumulate)
4. Applying drift-specific degradation based on the impact matrix above
5. Computing the composite sufficiency score at checkpoint days

Input contract notes:

- `days`, `total_days`, and `DriftSpec.onset_day` must be non-negative integer-like values.
- NumPy integer scalars are accepted; floats and booleans are rejected.

## Drift Progression

Drift effects ramp up over 90 days from onset (linear progression).
The magnitude parameter (0.0-1.0) scales the maximum impact.

```python
progress = min(1.0, drift_days / 90.0)
degradation = impact * magnitude * progress
```

## Standard Evaluation Windows

Paper 14 evaluates blind periods of 30, 60, 90, and 180 days,
matching realistic label latency ranges in fraud detection (7-180 days).
