# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-09

Zenodo release: [10.5281/zenodo.19479120](https://doi.org/10.5281/zenodo.19479120).

Detailed change notes pending; see GitHub release notes for the interim summary.

## [0.1.2] - 2026-03-28

Zenodo release: [10.5281/zenodo.19270499](https://doi.org/10.5281/zenodo.19270499).

Detailed change notes pending; see GitHub release notes for the interim summary.

## [0.1.1] - 2026-03-27

Zenodo release: [10.5281/zenodo.19245277](https://doi.org/10.5281/zenodo.19245277).

Detailed change notes pending; see GitHub release notes for the interim summary.

## [0.1.0] - 2026-03-26

Initial public release. Zenodo: [10.5281/zenodo.19233931](https://doi.org/10.5281/zenodo.19233931).

### Added

- Core sufficiency scoring: completeness, freshness, reliability, representativeness
- Decision-readiness gate A(t) with configurable thresholds
- Weighted composite score S(t) computation
- Blind period simulator for temporal degradation modeling
- Threshold monitor for sequential alerting
- E-value accumulator (experimental) for sequential hypothesis testing
- DES adapter (`sufficiency.adapters.des`) with JSON Schema validation
- Packaged DES schema resource for wheel distribution
- Domain-specific config factories: `default_config`, `fraud_detection_config`, `credit_scoring_config`
- Benchmark harness with regression comparison (`benchmarks/`)
- 100% test coverage enforcement (208 tests, branch coverage)
- Pre-commit hooks: ruff, mypy strict, markdownlint, yamllint, toml-sort, typos, detect-secrets
- CI pipeline: quality gate, Python 3.11-3.14 matrix, wheel smoke test
- Apache-2.0 license
- CITATION.cff for academic citation
