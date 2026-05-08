from __future__ import annotations

import json
import math

import pytest

from fisheye.shared.run_lineage_fingerprint import (
    RunLineageFingerprintError,
    build_run_lineage_attrs,
    build_run_lineage_payload,
    canonical_lineage_json,
    compute_run_lineage_hash,
)


def test_run_lineage_hash_is_deterministic_for_key_order() -> None:
    left = build_run_lineage_payload(
        run_family="swim_bout_run",
        analysis_schema={"schema_version": 6, "schema_id": "analysis.swim_bout_runs"},
        method="peak_event",
        method_version="detect_bouts_multi_level.v7",
        source_refs={"source_track_kinematics_run": "tk_1", "track_id": 0},
        parameters={"min_peak_prominence_mm_s": 4.0, "window_lengths_s": [10.0, 30.0]},
    )
    right = build_run_lineage_payload(
        run_family="swim_bout_run",
        analysis_schema={"schema_id": "analysis.swim_bout_runs", "schema_version": 6},
        method="peak_event",
        method_version="detect_bouts_multi_level.v7",
        source_refs={"track_id": 0, "source_track_kinematics_run": "tk_1"},
        parameters={"window_lengths_s": [10.0, 30.0], "min_peak_prominence_mm_s": 4.0},
    )

    assert canonical_lineage_json(left) == canonical_lineage_json(right)
    assert compute_run_lineage_hash(left) == compute_run_lineage_hash(right)


def test_run_lineage_attrs_are_strict_json_safe() -> None:
    payload = build_run_lineage_payload(
        run_family="stimulus_response_run",
        parameters={
            "threshold": math.nan,
            "valid": True,
            "label": b"omr\x00\x00",
        },
    )

    attrs = build_run_lineage_attrs(payload, fingerprint_status="best_effort")
    loaded = json.loads(attrs["lineage_payload_json"])

    assert loaded["parameters"]["threshold"] is None
    assert loaded["parameters"]["label"] == "omr"
    assert attrs["source_fingerprint"] == attrs["source_lineage_hash"]
    assert attrs["source_fingerprint"] == attrs["lineage_hash"]
    assert len(attrs["source_fingerprint"]) == 64


def test_run_lineage_attrs_reject_unknown_status() -> None:
    payload = build_run_lineage_payload(run_family="track_kinematics_run")

    with pytest.raises(RunLineageFingerprintError):
        build_run_lineage_attrs(payload, fingerprint_status="probably")
