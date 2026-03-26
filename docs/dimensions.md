# Evidence Quality Dimensions

Four dimensions measure evidence quality; a decision-readiness gate modulates the composite.

## Completeness C(t)

Fraction of decisions with confirmed outcome labels.

```text
C(t) = labeled_count / total_count
```text

Threshold: tau_c = 0.6. Below this, the decision-readiness gate suppresses the composite score.

Confidence interval: Wilson score interval for binomial proportions.

`labeled_count` and `total_count` are integer-like counts. Plain `int` and NumPy integer scalars are accepted; floats and booleans are rejected.

## Freshness F(t)

Temporal relevance of evidence, modeled as exponential decay of label age.

```text
F(t) = exp(-lambda * delta_t)
```text

- `delta_t`: days since the reference point for the most recent confirmed labels
- `lambda`: domain-specific decay rate (fraud: 0.02/day, credit: 0.005/day)
- At lambda=0.02: F(30) = 0.55, F(60) = 0.30, F(90) = 0.17, F(180) = 0.03

When using the DES adapter, `extract_freshness_inputs(...)` defaults the reference point to the latest decision timestamp in the batch and returns the median age of the labeled evidence relative to that point. If no valid reference-time age can be formed, it falls back to median observed label latency.

## Reliability R(t)

Accuracy of evidence against ground truth, measured as F1-score on retroactively labeled data.

```text
R(t) = F1(y_true, y_pred)
```text

Threshold: tau_r = 0.7. Below this, the decision-readiness gate activates.

Confidence interval: bootstrap resampling (default 1000 iterations).

Degrades sharply under real concept drift P(Y|X) but is unobservable without labels.

## Representativeness P(t)

Coverage of the operational distribution, measured via two-sample KS test.

```text
P(t) = max(0, 1 - KS_statistic / KS_cap)
```text

- `KS_cap`: normalization cap (default 0.30, calibrated to 99th percentile of observed range)
- KS at or above cap maps to P = 0.0
- Degrades sharply under covariate drift P(X)

## Decision-Readiness Gate A(t)

Not a dimension but a structural safeguard. Multiplicative gate derived from completeness and reliability.

```text
A(t) = min(1, C/tau_c) * min(1, R/tau_r)
```text

- Above thresholds: A(t) = 1.0 (no suppression)
- Below thresholds: proportional suppression
- Simultaneous degradation of C and R produces compounding penalty
