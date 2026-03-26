# API Notes

## API Layers

- Core API: `sufficiency`
  Use this for scoring, configuration, blind-period simulation, and threshold
  monitoring.
- Experimental API: `sufficiency.experimental.monitoring`
  Use this only for exploratory monitoring helpers that are not part of the
  stable root contract.
- Adapter API: `sufficiency.adapters.des`
  Use this for Decision Event Schema validation and transformation when your
  scoring inputs come from DES-shaped records rather than from direct numeric
  measurements.

## API Notes

- Count-like inputs such as `labeled_count`, `total_count`, and
  `DriftSpec.onset_day` must be integer-like values. Plain `int` and NumPy
  integer scalars are accepted; floats and booleans are rejected.
- `compute_reliability(...)` accepts only binary labels (`0`/`1`) or boolean
  equivalents. Non-binary numeric scores such as probabilities are rejected.
- `BlindPeriodSimulator.simulate(...)` and `simulate_daily(...)` require
  non-negative integer day values.
- `compute_sufficiency(..., timestamp=...)` requires a timezone-aware `datetime`.
- `sufficiency.adapters.des` requires timezone-aware ISO 8601 timestamps with a
  `Z` suffix or explicit UTC offset.
- `compute_representativeness(...)` expects one-dimensional numeric samples for
  both reference and production inputs.
- `result.dimensions` is returned as an immutable mapping snapshot. Iterate
  over it normally, but do not expect in-place mutation to be supported.
- `EValueAccumulator.e_value` may saturate to `inf` for long-running sequences;
  `log_e_value` remains the numerically stable internal accumulator.

## Monitoring

- `ThresholdMonitor` in `sufficiency` is the policy-authoritative monitoring
  entry point for classifying composite scores against configured thresholds.
- `EValueAccumulator` in `sufficiency.experimental.monitoring` is a heuristic
  alerting signal for repeated below-threshold behavior. It is not documented
  as a formally anytime-valid e-process.

## API Stability

The stable compatibility contract is the root `sufficiency` namespace.
Experimental helpers stay under `sufficiency.experimental.*`, and integration
utilities stay under `sufficiency.adapters.*`.

## Internal Layout

The package keeps its public contract small, but the implementation is split
internally for maintainability:

- `sufficiency._validation`
  Shared internal input validation helpers. Not a public API surface.
- `sufficiency._dimensions`
  Canonical internal dimension-name registry and default weight mapping.
- `sufficiency.sequential`
  Stable threshold-monitoring entry point. Legacy imports for
  `EValueAccumulator` remain supported there for compatibility.
- `sufficiency.experimental.evalue`
  Experimental implementation of the heuristic e-value accumulator.
- `sufficiency.adapters.des`
  Public DES adapter facade. Internal schema and extraction responsibilities
  are split into private helper modules.

These internal modules may evolve more freely than the documented root API.
