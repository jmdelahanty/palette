from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fisheye.analytics_exports.validated_behavior_bout_kinematics import (
    BoundBoutKinematicsMetricsSource,
)
from fisheye.analytics_exports.validated_behavior_bout_kinematics_contracts import (
    BOUT_EYE_GAZE_TABLE,
    BOUT_HEADING_TABLE,
    BOUT_KINEMATICS_CAPABILITY_KEYS,
    BOUT_KINEMATICS_EXPORT_PROFILE_ID,
    BOUT_MOVEMENT_TABLE,
    source_dtype,
)
from fisheye.analytics_exports.validated_behavior_contracts import (
    validate_table_specs,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CORE_BEHAVIOR_TABLE_SPECS,
)
from fisheye.analytics_exports.validated_behavior_profiles import (
    resolve_validated_behavior_profile,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_adapters import (
    _bout_eye_gaze_metrics,
    _bout_heading_metrics,
    _bout_movement_metrics,
)


def test_frozen_export_fields_match_the_native_source_schema() -> None:
    from fisheye.analysis import bout_kinematics as native

    expected = {
        "movement": native._movement_metrics_dtype(),
        "heading": native._metrics_dtype(),
        "eye_gaze": native._eye_gaze_metrics_dtype(),
    }
    for level, dtype in expected.items():
        assert source_dtype(level) == dtype


def test_new_profile_adds_three_grains_without_changing_the_v005_profile() -> None:
    old = resolve_validated_behavior_profile(CORE_BEHAVIOR_EXPORT_PROFILE_ID)
    new = resolve_validated_behavior_profile(BOUT_KINEMATICS_EXPORT_PROFILE_ID)
    assert old.table_specs is CORE_BEHAVIOR_TABLE_SPECS
    assert len(CORE_BEHAVIOR_CAPABILITY_KEYS) == 6
    assert len(BOUT_KINEMATICS_CAPABILITY_KEYS) == 7
    assert set(new.table_specs) == set(old.table_specs) | {
        BOUT_MOVEMENT_TABLE,
        BOUT_HEADING_TABLE,
        BOUT_EYE_GAZE_TABLE,
    }
    assert set(new.row_extractors()) == set(old.row_extractors()) | {
        BOUT_MOVEMENT_TABLE,
        BOUT_HEADING_TABLE,
        BOUT_EYE_GAZE_TABLE,
    }
    assert len(validate_table_specs(new.table_specs)) == 11
    for table_name in (BOUT_MOVEMENT_TABLE, BOUT_HEADING_TABLE, BOUT_EYE_GAZE_TABLE):
        spec = new.table_specs[table_name]
        assert any(
            target == "canonical_swim_bouts" for _, target, _ in spec.foreign_keys
        )
        assert "source_binding_sha256" in {field.name for field in spec.contract.fields}


class _Context:
    row_group_rows = 1

    def __init__(self) -> None:
        records = {}
        for level in ("movement", "heading_raw", "heading_smoothed", "eye_gaze"):
            native_level = "heading" if level.startswith("heading_") else level
            arr = np.zeros(2, dtype=source_dtype(native_level))
            arr["bout_id"] = [1, 2]
            arr["source_start_frame"] = [10, 30]
            arr["source_end_frame"] = [20, 40]
            arr["source_core_start_frame"] = [12, 32]
            arr["source_core_end_frame"] = [18, 38]
            arr["failure_reason_bytes"] = [b"", b"bad source"]
            if level.startswith("heading_"):
                arr["net_delta_heading_deg"] = (
                    [1.0, 2.0] if level == "heading_raw" else [3.0, 4.0]
                )
            records[level] = arr
        binding = {
            "payload_sha256": "a" * 64,
            "run_name": "bout-run",
            "run_path": "analysis/bout_kinematics_runs/bout-run",
            "source_array_manifest_sha256": "b" * 64,
            "content_sha256_by_level": {level: level for level in records},
            "source_track_id": 0,
            "source_signal_id": 4,
        }
        self.bound = SimpleNamespace(
            bout_kinematics=BoundBoutKinematicsMetricsSource(binding, records)
        )
        self._capability = {
            "source_binding": binding,
            "projection_contract": {"payload_sha256": "c" * 64},
        }

    def capability_binding(self, _capability_id: str) -> dict[str, object]:
        return self._capability

    def common_columns(self, count: int) -> dict[str, list[str]]:
        return {
            name: [name] * count
            for name in (
                "export_run_id",
                "recording_id",
                "membership_member_sha256",
                "bundle_set_member_sha256",
                "bundle_record_sha256",
                "cross_grain_join_authority_sha256",
            )
        }


def test_bout_projection_keeps_all_levels_native_values_and_validity_columns() -> None:
    context = _Context()
    movement = list(_bout_movement_metrics(context).batches)
    heading = list(_bout_heading_metrics(context).batches)
    eye = list(_bout_eye_gaze_metrics(context).batches)
    assert len(movement) == 2
    assert [batch["heading_level"][0] for batch in heading] == [
        "heading_raw",
        "heading_raw",
        "heading_smoothed",
        "heading_smoothed",
    ]
    assert [batch["net_delta_heading_deg"][0] for batch in heading] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert movement[1]["failure_reason"] == ["bad source"]
    assert eye[0]["within_eye_window_valid"].dtype == np.dtype("bool")
    assert set(movement[0]) == {
        field.name
        for field in resolve_validated_behavior_profile(
            BOUT_KINEMATICS_EXPORT_PROFILE_ID
        )
        .table_specs[BOUT_MOVEMENT_TABLE]
        .contract.fields
    }
