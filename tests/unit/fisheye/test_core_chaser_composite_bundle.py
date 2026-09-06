from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fisheye.analysis_workflows.core_chaser_composite_bundle as subject
from fisheye.analysis_workflows.core_authority_roster import (
    build_core_authority_roster,
)
from fisheye.analysis_workflows.validated_behavior_source_admission import (
    CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
    CORE_BEHAVIOR_EXECUTION_SCHEMA_ID,
    CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION,
    EXACT_CHASER_ADMISSION_ROLE,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_CAPABILITY_KEYS,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _sealed(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def _record_sealed(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "record_sha256": canonical_json_sha256(body)}


def _receipt(
    role: str, path: Path, *, record: str, schema_id: str, schema_version: int
) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "file_sha256": canonical_json_sha256({"file": str(path)}),
        "record_sha256": record,
        "schema_id": schema_id,
        "schema_version": schema_version,
    }


def _core_roster(tmp_path: Path) -> dict[str, Any]:
    archive = (tmp_path / "recording-a.zarr").resolve()
    report = _receipt(
        CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
        tmp_path / "core-report.json",
        record="a" * 64,
        schema_id=CORE_BEHAVIOR_EXECUTION_SCHEMA_ID,
        schema_version=CORE_BEHAVIOR_EXECUTION_SCHEMA_VERSION,
    )
    join = _sealed(
        {
            "recording_id": "recording-a",
            "camera_id": "camera-a",
            "source_total_frames": 10,
            "source_sample_rate_hz": 30.0,
        }
    )
    bindings: dict[str, Any] = {"cross_grain_join_authority": join}
    for capability in CORE_BEHAVIOR_CAPABILITY_KEYS:
        if capability == "cross_grain_join_authority":
            continue
        source_body: dict[str, Any] = {
            "recording_id": "recording-a",
            "zarr_path": str(archive),
            "run_path": f"analysis/{capability}/run-a",
        }
        if capability == "kinematics_samples":
            source_body["tracks"] = [{"track_id": 0}]
        bindings[capability] = {
            "profile_id": f"{capability}_fixture_v1",
            "source_binding": _sealed(source_body),
            "projection_contract": _sealed(
                {
                    "profile_id": f"{capability}_projection_fixture_v1",
                    "sampling_stride_frames": 1,
                }
            ),
            "join_authority_sha256": join["payload_sha256"],
        }
    return build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=archive,
        execution_report_binding=report,
        capability_bindings=bindings,
    )


def _child_binding(tmp_path: Path, key: str) -> dict[str, Any]:
    return {
        "receipt_path": str((tmp_path / f"{key}.receipt.json").resolve()),
        "receipt_sha256": canonical_json_sha256({"receipt": key}),
        "run_path": f"analysis/{key}_runs/{key}-a",
        "manifest_sha256": canonical_json_sha256({"manifest": key}),
        "payload_digest": canonical_json_sha256({"payload": key}),
    }


def _content(tmp_path: Path) -> dict[str, Any]:
    archive = (tmp_path / "recording-a.zarr").resolve()
    roster = _core_roster(tmp_path)
    projection_receipt = _receipt(
        EXACT_CHASER_ADMISSION_ROLE,
        tmp_path / "chaser-projection.json",
        record="b" * 64,
        schema_id="palette.analysis.exact_chaser.projection_receipt",
        schema_version=8,
    )
    receipts = sorted(
        [roster["execution_report_binding"], projection_receipt],
        key=lambda item: (item["role"], item["path"]),
    )
    children = {
        key: _child_binding(tmp_path, key)
        for key in (
            *subject.BASE_SCIENTIFIC_CHILD_KEYS,
            "body_alignment_by_distance",
            "gaze",
        )
    }
    internal = {
        key: subject._complete_internal("source_bindings", key)  # noqa: SLF001
        for key in subject._INTERNAL_SOURCE_CAPABILITIES  # noqa: SLF001
    }
    internal.update(
        {
            key: subject._complete_internal(
                "scientific_child_bindings", key
            )  # noqa: SLF001
            for key in children
        }
    )
    lineage = _record_sealed(
        {
            "policy_id": "exact_chaser_children_share_one_core_roster_v1",
            "core_authority_roster_sha256": roster["record_sha256"],
            "relative_dependencies": {},
            "motion_dependencies": {},
            "controller_payload_sha256": "c" * 64,
            "bout_payload_sha256": "d" * 64,
            "spatial_core_authority": {},
        }
    )
    capabilities: dict[str, Any] = {
        key: {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": roster["capability_bindings"][key],
        }
        for key in CORE_BEHAVIOR_CAPABILITY_KEYS
    }
    for key in subject._CHASER_EXTENSION_CAPABILITY_KEYS:  # noqa: SLF001
        capabilities[key] = {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": _record_sealed(
                {
                    "profile_id": f"{key}_fixture_v1",
                    "core_authority_roster_sha256": roster["record_sha256"],
                }
            ),
        }
    core = roster["capability_bindings"]
    motion = core["kinematics_samples"]["source_binding"]
    body = core["subject_body_frame_samples"]["source_binding"]
    bouts = core["canonical_swim_bouts"]["source_binding"]
    child_seal = {
        "run_path": "analysis/chaser_relative_frame_runs/fixture",
        "manifest_sha256": "f" * 64,
    }
    source_bindings = {
        "fish_position_keypoint": {
            "binding_type": "core_motion_on_exact_chaser_carrier_v1",
            "authority": {"source_authority_id": motion["run_path"]},
            "sealed_by": child_seal,
        },
        "fish_position_detection": {
            "binding_type": "core_motion_on_exact_chaser_carrier_v1",
            "authority": {"source_authority_id": motion["run_path"]},
            "sealed_by": child_seal,
        },
        "chaser_observations_keypoint_projection": {
            "binding_type": "relative_frame_source_authority_v1",
            "authority": {"fixture": "keypoint"},
            "sealed_by": child_seal,
        },
        "chaser_observations_detection_projection": {
            "binding_type": "relative_frame_source_authority_v1",
            "authority": {"fixture": "detection"},
            "sealed_by": child_seal,
        },
        "anatomical_body_frame": {
            "binding_type": "selected_core_subject_body_frame_v1",
            "authority": {"fixture": "body"},
            "source": body,
            "sealed_by": child_seal,
        },
        "row_axis_timing_and_scale": {
            "binding_type": "paired_relative_frame_consensus_v1",
            "authority": {"fixture": "axis"},
            "sealed_by": {"keypoint": child_seal, "detection": child_seal},
        },
        "provider_motion": {
            "binding_type": "selected_core_motion_authority_v1",
            "source": motion,
            "authority": {
                "provider_id": motion["run_path"],
                "provider_digest": motion["payload_sha256"],
            },
            "sealed_by": roster["record_sha256"],
        },
        "canonical_swim_bouts": {
            "binding_type": "selected_core_swim_bout_authority_v1",
            "source": bouts,
            "sealed_by": roster["record_sha256"],
        },
        "semantic_epochs": {
            "binding_type": "exact_protocol_semantic_selection_v1",
            "source": {"fixture": "semantic"},
            "sealed_by": child_seal,
        },
        "reviewed_arena_and_scale": {
            "binding_type": "spatial_radial_consensus_v1",
            "authority": {"fixture": "geometry"},
            "sealed_by": {"spatial_occupancy": child_seal},
        },
    }
    return {
        "analysis_zarr": str(archive),
        "recording_id": "recording-a",
        "source_admission_receipts": receipts,
        "core_authority_roster": roster,
        "chaser_projection": {
            "receipt_path": projection_receipt["path"],
            "receipt_sha256": projection_receipt["record_sha256"],
            "schema_id": projection_receipt["schema_id"],
            "schema_version": projection_receipt["schema_version"],
        },
        "source_bindings": source_bindings,
        "scientific_child_bindings": children,
        "internal_capabilities": internal,
        "capabilities": capabilities,
        "compatibility_proofs": {
            "extension_core_lineage": lineage,
            "paired_relative_frame_axis": _record_sealed(
                {
                    "policy_id": "exact_keypoint_detection_axis_consensus_v1",
                    "evidence": {},
                }
            ),
            "spatial_radial_composition": _record_sealed(
                {
                    "policy_id": "exact_spatial_radial_consensus_v1",
                    "evidence": {},
                }
            ),
        },
        "validation_policy": dict(subject.VALIDATION_POLICY),
        "safety": dict(subject.SAFETY),
    }


def test_capability_contract_matches_composed_table_dependencies() -> None:
    contract = subject.core_chaser_capability_contract()

    assert set(contract["keys"]) == set(subject.CORE_CHASER_CAPABILITY_KEYS)
    assert "cross_grain_join_authority" in contract["keys"]
    assert "provider_motion" not in contract["keys"]


def test_bundle_binds_exactly_one_core_and_one_chaser_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _content(tmp_path)
    monkeypatch.setattr(subject, "_resolve_content", lambda *_args, **_kwargs: content)

    bundle = subject.build_core_chaser_composite_bundle(
        tmp_path / "core-report.json",
        tmp_path / "chaser-projection.json",
        palette_commit="e" * 40,
        expected_analysis_zarr=tmp_path / "recording-a.zarr",
        expected_recording_id="recording-a",
        created_at_utc="2026-09-05T12:00:00+00:00",
    )

    validated = subject.validate_core_chaser_composite_bundle(bundle)
    assert validated["record_sha256"] == bundle["record_sha256"]
    assert {item["role"] for item in validated["source_admission_receipts"]} == {
        CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLE,
        EXACT_CHASER_ADMISSION_ROLE,
    }
    assert validated["safety"]["selector_eligible"] is False


def test_bundle_rejects_redigested_duplicate_core_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _content(tmp_path)
    monkeypatch.setattr(subject, "_resolve_content", lambda *_args, **_kwargs: content)
    bundle = subject.build_core_chaser_composite_bundle(
        tmp_path / "core-report.json",
        tmp_path / "chaser-projection.json",
        palette_commit="e" * 40,
        expected_analysis_zarr=tmp_path / "recording-a.zarr",
        expected_recording_id="recording-a",
        created_at_utc="2026-09-05T12:00:00+00:00",
    )
    tampered = dict(bundle)
    body = {key: value for key, value in tampered.items() if key != "record_sha256"}
    receipts = [dict(item) for item in body["source_admission_receipts"]]
    receipts[1] = dict(receipts[0])
    body["source_admission_receipts"] = receipts
    tampered = {**body, "record_sha256": canonical_json_sha256(body)}

    with pytest.raises(
        subject.CoreChaserCompositeBundleError,
        match="receipt roles or ordering",
    ):
        subject.validate_core_chaser_composite_bundle(tampered)


def test_bundle_rejects_nested_core_capability_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _content(tmp_path)
    monkeypatch.setattr(subject, "_resolve_content", lambda *_args, **_kwargs: content)
    bundle = subject.build_core_chaser_composite_bundle(
        tmp_path / "core-report.json",
        tmp_path / "chaser-projection.json",
        palette_commit="e" * 40,
        expected_analysis_zarr=tmp_path / "recording-a.zarr",
        expected_recording_id="recording-a",
        created_at_utc="2026-09-05T12:00:00+00:00",
    )
    body = {key: value for key, value in bundle.items() if key != "record_sha256"}
    capabilities = {key: dict(value) for key, value in body["capabilities"].items()}
    capabilities["kinematics_samples"]["binding"] = capabilities["eye_trace_samples"][
        "binding"
    ]
    body["capabilities"] = capabilities
    tampered = {**body, "record_sha256": canonical_json_sha256(body)}

    with pytest.raises(
        subject.CoreChaserCompositeBundleError,
        match="Core capability 'kinematics_samples' differs",
    ):
        subject.validate_core_chaser_composite_bundle(tampered)


def test_bundle_keeps_absent_nonexported_gaze_as_typed_internal_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _content(tmp_path)
    content["scientific_child_bindings"].pop("gaze")
    content["internal_capabilities"][
        "gaze"
    ] = subject._missing_internal()  # noqa: SLF001
    monkeypatch.setattr(subject, "_resolve_content", lambda *_args, **_kwargs: content)

    bundle = subject.build_core_chaser_composite_bundle(
        tmp_path / "core-report.json",
        tmp_path / "chaser-projection.json",
        palette_commit="e" * 40,
        expected_analysis_zarr=tmp_path / "recording-a.zarr",
        expected_recording_id="recording-a",
        created_at_utc="2026-09-05T12:00:00+00:00",
    )

    assert bundle["internal_capabilities"]["gaze"]["state"] == "unavailable"


def test_composite_resolution_rejects_a_sampled_core_motion_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster = _core_roster(tmp_path)
    bindings = {
        key: dict(value) for key, value in roster["capability_bindings"].items()
    }
    motion = dict(bindings["kinematics_samples"])
    projection_body = {
        key: value
        for key, value in motion["projection_contract"].items()
        if key != "payload_sha256"
    }
    projection_body["sampling_stride_frames"] = 3
    motion["projection_contract"] = _sealed(projection_body)
    bindings["kinematics_samples"] = motion
    sampled = build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        execution_report_binding=roster["execution_report_binding"],
        capability_bindings=bindings,
    )
    monkeypatch.setattr(
        subject,
        "bind_core_behavior_cohort_sources",
        lambda *_args, **_kwargs: SimpleNamespace(core_authority_roster=sampled),
    )

    with pytest.raises(
        subject.CoreChaserCompositeBundleError,
        match="full-rate core-motion projection",
    ):
        subject._resolve_content(  # noqa: SLF001
            tmp_path / "core-report.json",
            tmp_path / "projection.json",
            expected_analysis_zarr=tmp_path / "recording-a.zarr",
            expected_recording_id="recording-a",
        )
