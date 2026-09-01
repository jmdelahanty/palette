from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import fisheye.analytics_exports.validated_behavior_adapters as adapters
import fisheye.analytics_exports.validated_behavior_phase_a_contracts as contracts
import fisheye.analytics_exports.validated_behavior_profiles as profiles
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_TABLE_NAMES,
    validate_table_specs,
)

EXPECTED_SCIENTIFIC_TABLES = {
    "body_alignment_distance_bins",
    "bout_chaser_associations",
    "bout_response_distance_bins",
    "canonical_swim_bouts",
    "chaser_occurrences",
    "controller_trials",
    "epoch_behavior_summary",
    "position_providers",
    "radial_near_field_density_bins",
    "radial_near_field_distance_cdf",
    "radial_near_field_summary",
    "recording_source_bindings",
    "same_quadrant_occupancy",
    "semantic_epochs",
    "spatial_occupancy_bins",
    "spatial_occupancy_support",
    "trial_escape_freeze_events",
    "trial_escape_freeze_summaries",
    "trial_escape_freeze_threshold_sweeps",
}


def test_phase_a_profile_closes_exact_contract_and_extractor_rosters() -> None:
    profile = profiles.resolve_validated_behavior_profile(contracts.PHASE_A_PROFILE_ID)

    assert profile.profile_id == contracts.PHASE_A_PROFILE_ID
    assert set(validate_table_specs(profile.table_specs)) == (
        set(CORE_TABLE_NAMES) | EXPECTED_SCIENTIFIC_TABLES
    )
    assert set(profile.row_extractors()) == EXPECTED_SCIENTIFIC_TABLES
    assert len(profile.table_specs) == 22
    assert (
        profile.table_specs["body_alignment_distance_bins"].capability_policy
        == "optional_explicit_coverage"
    )
    assert all(
        spec.capability_policy == "required_all_admitted"
        for name, spec in profile.table_specs.items()
        if name not in CORE_TABLE_NAMES and name != "body_alignment_distance_bins"
    )


def test_profile_routing_reads_only_an_installed_exact_profile_id(
    tmp_path: Path,
) -> None:
    record = tmp_path / "plan.json"
    record.write_text(
        json.dumps({"export_profile": {"profile_id": contracts.PHASE_A_PROFILE_ID}}),
        encoding="utf-8",
    )

    assert (
        profiles.profile_id_from_record(record, record_kind="export plan")
        == contracts.PHASE_A_PROFILE_ID
    )
    with pytest.raises(
        profiles.ValidatedBehaviorProfileError, match="is not installed"
    ):
        profiles.resolve_validated_behavior_profile("uninstalled_profile_v1")


def test_phase_a_adapter_does_not_encode_dataset_or_protocol_selection() -> None:
    source = Path(adapters.__file__).read_text(encoding="utf-8").casefold()

    assert "goodbatbadbat" not in source
    assert "goodcopbadcop" not in source
    assert "protocol_name" not in source
    assert "protocol_hash" not in source


def test_unregistered_zero_semantic_role_is_explicitly_null() -> None:
    assert (
        adapters._decode_optional_zero(  # noqa: SLF001
            {1: "pre", 2: "active", 3: "post"},
            np.int32(0),
            field="semantic role",
        )
        is None
    )
    with pytest.raises(
        adapters.ValidatedBehaviorAdapterError,
        match="absent from its sealed registry",
    ):
        adapters._decode_optional_zero(  # noqa: SLF001
            {1: "pre"}, np.int32(4), field="semantic role"
        )


class _BoutContext:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        run_path = "swim_bout_runs/canonical_fixture"
        lineage = "a" * 64
        self.bundle = {
            "source_bindings": {
                "canonical_swim_bouts": {
                    "source": {
                        "run_path": run_path,
                        "lineage_hash": lineage,
                        "default_signal_id": 2,
                        "track_id": 7,
                    }
                }
            }
        }
        self.handle = SimpleNamespace(
            arrays=arrays,
            scientific_manifest={
                "dimensions": {"n_bouts": 2, "n_chasers": 2},
                "identity_registries": {"semantic_role": {"1": "active"}},
                "sources": {
                    "swim_bouts": {
                        "run_path": run_path,
                        "lineage_sha256": lineage,
                        "signal_id": 2,
                    }
                },
            },
        )

    def require_capability(self, capability: str) -> None:
        assert capability == "canonical_swim_bouts"

    def composable_child(self, key: str) -> Any:
        assert key == "generalized_bout_response"
        return self.handle

    def chaser_identity_maps(self) -> tuple[dict[int, str], dict[int, str]]:
        return {1: "first", 2: "second"}, {1: "pursuer", 2: "control"}

    def child_common(self, key: str) -> dict[str, Any]:
        assert key == "generalized_bout_response"
        return {
            "export_run_id": "fixture-export",
            "recording_id": "fixture-recording",
            "membership_member_sha256": "1" * 64,
            "bundle_set_member_sha256": "2" * 64,
            "bundle_record_sha256": "3" * 64,
            "source_child_key": key,
            "source_run_path": "generalized_chaser_bout_response_runs/fixture",
            "source_manifest_sha256": "4" * 64,
            "source_payload_sha256": "5" * 64,
            "source_receipt_sha256": "6" * 64,
        }


def _bout_arrays() -> dict[str, np.ndarray]:
    arrays = {
        name: np.zeros(4, dtype=np.float64)
        for name in adapters._BOUT_ASSOCIATION_ARRAYS  # noqa: SLF001
    }
    arrays.update(
        {
            "bout_row_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "bout_id": np.asarray([10, 10, 11, 11], dtype=np.int64),
            "source_signal_id": np.asarray([2, 2, 2, 2], dtype=np.int32),
            "start_acquisition_frame_id": np.asarray(
                [100, 100, 200, 200], dtype=np.int64
            ),
            "end_acquisition_frame_id": np.asarray(
                [110, 110, 212, 212], dtype=np.int64
            ),
            "bout_duration_s": np.asarray([0.1, 0.1, 0.12, 0.12]),
            "bout_path_length_mm": np.asarray([1.0, 1.0, 1.5, 1.5]),
            "bout_net_displacement_mm": np.asarray([0.8, 0.8, 1.1, 1.1]),
            "bout_mean_speed_mm_s": np.asarray([8.0, 8.0, 9.0, 9.0]),
            "bout_peak_speed_mm_s": np.asarray([12.0, 12.0, 13.0, 13.0]),
            "bout_tortuosity": np.asarray([1.25, 1.25, 1.36, 1.36]),
        }
    )
    return arrays


def test_canonical_bouts_are_deduplicated_from_equal_chaser_associations() -> None:
    rows, zero_reason = adapters._canonical_swim_bouts(  # noqa: SLF001
        _BoutContext(_bout_arrays())
    )

    assert zero_reason is None
    assert [row["bout_id"] for row in rows] == [10, 11]
    assert [row["bout_row_id"] for row in rows] == [0, 1]
    assert all(row["track_id"] == 7 for row in rows)
    assert set(rows[0]) == {item.name for item in contracts.CANONICAL_SWIM_BOUTS.fields}


def test_canonical_bout_copy_fails_when_repeated_chaser_facts_diverge() -> None:
    arrays = _bout_arrays()
    arrays["bout_peak_speed_mm_s"][1] = 99.0

    with pytest.raises(
        adapters.ValidatedBehaviorAdapterError,
        match="Repeated bout facts differ",
    ):
        adapters._canonical_swim_bouts(_BoutContext(arrays))  # noqa: SLF001
