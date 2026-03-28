"""Tests for the package-level public API."""

from __future__ import annotations

import importlib
import importlib.metadata
from importlib.metadata import version

import sufficiency
from sufficiency import sequential
from sufficiency.adapters import des
from sufficiency.experimental import monitoring


class TestPackageApi:
    def test_version_exposed(self) -> None:
        assert sufficiency.__version__ == version("evidence-sufficiency-calc")

    def test_version_falls_back_when_package_metadata_is_unavailable(self, monkeypatch) -> None:
        def fake_version(_: str) -> str:
            raise importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(importlib.metadata, "version", fake_version)

        reloaded = importlib.reload(sufficiency)

        assert reloaded.__version__ == "0+unknown"

        importlib.reload(sufficiency)

    def test_core_symbols_reexported(self) -> None:
        expected = {
            "BlindPeriodSimulator",
            "DimensionScore",
            "DriftSpec",
            "DriftType",
            "GovernanceConfig",
            "SufficiencyResult",
            "SufficiencyStatus",
            "SufficiencyThresholds",
            "ThresholdMonitor",
            "compute_gate",
            "compute_sufficiency",
            "credit_scoring_config",
            "default_config",
            "fraud_detection_config",
        }

        assert expected == set(sufficiency.__all__)

        for name in expected:
            assert hasattr(sufficiency, name)

    def test_root_api_does_not_export_experimental_symbols(self) -> None:
        assert "EValueAccumulator" not in sufficiency.__all__
        assert not hasattr(sufficiency, "EValueAccumulator")

    def test_reexported_config_factories_are_callable(self) -> None:
        assert abs(sum(sufficiency.default_config().weights.values()) - 1.0) < 1e-6
        assert abs(sum(sufficiency.fraud_detection_config().weights.values()) - 1.0) < 1e-6
        assert abs(sum(sufficiency.credit_scoring_config().weights.values()) - 1.0) < 1e-6

    def test_root_api_policy_markers_exposed(self) -> None:
        assert sufficiency.ROOT_API_STABILITY == "stable"

    def test_threshold_monitor_marked_stable(self) -> None:
        assert sequential.THRESHOLD_MONITOR_STABILITY == "stable"

    def test_evalue_accumulator_marked_experimental(self) -> None:
        assert sequential.EVALUE_ACCUMULATOR_STABILITY == "experimental"

    def test_experimental_namespace_exposes_canonical_monitoring_api(self) -> None:
        assert monitoring.API_STABILITY == "experimental"
        assert monitoring.CANONICAL_NAMESPACE == "sufficiency.experimental.monitoring"
        assert monitoring.EValueAccumulator is sequential.EValueAccumulator

    def test_evalue_accumulator_implementation_lives_under_experimental_module(self) -> None:
        assert monitoring.EValueAccumulator.__module__ == "sufficiency.experimental.evalue"

    def test_adapter_namespace_exposes_canonical_des_api(self) -> None:
        assert des.MODULE_LAYER == "adapter"
        assert des.ADAPTER_NAME == "des"
        assert des.CANONICAL_NAMESPACE == "sufficiency.adapters.des"
        assert des.extract_freshness_inputs.__module__ == "sufficiency.adapters.des"
        assert des.extract_completeness_inputs.__module__ == "sufficiency.adapters.des"
        assert des.validate_events.__module__ == "sufficiency.adapters.des"
