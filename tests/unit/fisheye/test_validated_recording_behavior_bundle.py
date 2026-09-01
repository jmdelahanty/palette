from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import fisheye.analysis_workflows.validated_recording_behavior_bundle as subject
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _digest(label: str) -> str:
    return canonical_json_sha256({"fixture": label})


def _scientific(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "payload_digest": canonical_json_sha256(body)}


def _outer(
    *, key: str, run_path: str, recording_id: str, scientific: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_id": "palette.analysis.composable_chaser_successor.run",
        "schema_version": 1,
        "successor_kind": subject._SUCCESSOR_KINDS[key],
        "run_name": run_path.rsplit("/", 1)[-1],
        "run_path": run_path,
        "recording_id": recording_id,
        "scientific_manifest": scientific,
        "scientific_payload_sha256": scientific["payload_digest"],
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
    }


def _receipt(
    *, key: str, run_path: str, manifest: dict[str, Any], recording_id: str
) -> dict[str, Any]:
    return {
        "record_sha256": _digest(f"receipt:{key}"),
        "run_path": run_path,
        "recording_id": recording_id,
        "manifest_sha256": _digest(f"manifest:{key}"),
        "payload_digest": _digest(f"payload:{key}"),
        "manifest": manifest,
    }


def _disposition(state: str, reason: str) -> dict[str, Any]:
    return {"state": state, "reason_code": reason, "detail": None}


def _base_dispositions() -> dict[str, dict[str, Any]]:
    return {
        "gaze": _disposition("unavailable", "upstream_segmentation_quality"),
        "subject_shape": _disposition("unavailable", "upstream_segmentation_quality"),
        "eye_angles": _disposition("unavailable", "upstream_segmentation_quality"),
        "tail_kinematics": _disposition("unavailable", "not_persisted"),
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    archive = (tmp_path / "recording.analysis.zarr").resolve()
    archive.mkdir()
    recording_id = "recording-1"
    semantic_run = (
        "analysis/protocol_semantic_chaser_selection_runs/semantic-selection-v2"
    )
    relative_paths = {
        "keypoint": "analysis/chaser_relative_frame_runs/keypoint-relative-v1",
        "detection": "analysis/chaser_relative_frame_runs/detection-relative-v1",
    }
    exact_paths = {
        "semantic_selection": semantic_run,
        "keypoint_radial": "analysis/chaser_radial_near_field_runs/keypoint-radial-v1",
        "detection_radial": (
            "analysis/chaser_radial_near_field_runs/detection-radial-v1"
        ),
        "controller": "analysis/controller_chase_trial_runs/controller-v1",
        "bout": "analysis/generalized_chaser_bout_response_runs/bout-v1",
        "escape": "analysis/chaser_escape_freeze_runs/escape-v1",
        "spatial_occupancy": ("analysis/chaser_spatial_occupancy_runs/spatial-v1"),
        "epoch_behavior": (
            "analysis/stimulus_epoch_behavior_summary_runs/epoch-behavior-v2"
        ),
        "body_alignment_by_distance": (
            "analysis/chaser_body_alignment_by_distance_runs/alignment-v1"
        ),
    }
    semantic_binding = {
        "run_path": semantic_run,
        "manifest_sha256": _digest("manifest:semantic_selection"),
    }
    common_axis = {
        "coordinate_authority_id": "camera-coordinate-v1",
        "scale_authority_id": "reviewed-scale-v1",
        "timing_authority_id": "session-timing-v1",
        "row_axis_authority_id": "acquisition-rows-v1",
        "row_axis_authority_digest": _digest("row-axis"),
    }

    def authority(
        role: str, *, source: str, provider: str, provider_digest: str
    ) -> dict[str, Any]:
        return {
            "recording_id": recording_id,
            "source_authority_id": source,
            "source_digest": _digest(f"source:{role}"),
            "provider_id": provider,
            "provider_digest": provider_digest,
            **common_axis,
        }

    keypoint_fish = authority(
        "keypoint-fish",
        source="analysis/keypoints_coordinate_successor_runs/keypoint-position-v1",
        provider="keypoint_triad_centroid.v1",
        provider_digest=_digest("keypoint-provider"),
    )
    detection_fish = authority(
        "detection-fish",
        source="analysis/detection_position_runs/detection-position-v1",
        provider="detection_bbox_centroid.v1",
        provider_digest=_digest("detection-provider"),
    )
    body = authority(
        "body-frame",
        source="analysis/body_frame_runs/body-frame-v1",
        provider="exact_keypoint_body_frame_projection.v1",
        provider_digest=_digest("position-body-composition"),
    )
    chaser_keypoint = authority(
        "chaser-keypoint",
        source="analysis/chaser_input_provenance_proxy_runs/keypoint-proxy-v1",
        provider="sealed_chaser_input_proxy.v1",
        provider_digest=_digest("keypoint-proxy"),
    )
    chaser_detection = authority(
        "chaser-detection",
        source="analysis/chaser_input_provenance_proxy_runs/detection-proxy-v1",
        provider="sealed_chaser_input_proxy.v1",
        provider_digest=_digest("detection-proxy"),
    )
    dimensions = {"n_frames": 10, "n_chasers": 2, "n_rows": 20}
    coordinate = {
        "policy_id": "source-camera-v1",
        "coordinate_authority_id": common_axis["coordinate_authority_id"],
        "coordinate_frame": "source_camera_continuous_pixel_xy",
    }
    scale = {
        "policy_id": "reviewed-scale-v1",
        "scale_authority_id": common_axis["scale_authority_id"],
        "scale_digest": _digest("scale"),
        "pixels_per_unit": 10.0,
        "unit": "mm",
    }
    timing = {
        "policy_id": "session-timestamp-v1",
        "timing_authority_id": common_axis["timing_authority_id"],
        "timing_digest": _digest("timing"),
        "frame_key_name": "acquisition_frame_id",
        "track_sample_key_name": "track_sample_id",
        "timestamp_field": "timestamp_ns_session",
    }
    context = {
        "temporal_selection": {
            "record": {"selection_id": "all-chaser-frames-v1"},
            "sha256": "",
        },
        "chaser_occurrence": {
            "record": {"occurrence_policy_id": "logged-occurrence-v1"},
            "sha256": "",
        },
    }
    for envelope in context.values():
        envelope["sha256"] = canonical_json_sha256(envelope["record"])
    registries = {
        "fish": {"1": "fish-1"},
        "chaser": {"1": "blue", "2": "red"},
        "behavior_role": {"1": "baseline", "2": "treatment"},
    }

    relative_receipts: dict[str, dict[str, Any]] = {}
    shared_shapes = {
        "base/acquisition_frame_id": [20],
        "base/timestamp_ns": [20],
        "base/timestamp_valid": [20],
        "base/selection_member": [20],
        "base/chaser_identity_code": [20],
        "base/chaser_behavior_role_code": [20],
        "base/chaser_occurrence_member": [20],
        "base/chaser_position_xy_px": [20, 2],
        "base/chaser_position_valid": [20],
    }
    shared_declarations = [
        {
            "path": path,
            "dtype": "<f4" if path.endswith("_xy_px") else "<i8",
            "shape": shape,
            "content_sha256": _digest(f"shared-array:{path}"),
        }
        for path, shape in shared_shapes.items()
    ]
    for role, fish, chaser in (
        ("keypoint", keypoint_fish, chaser_keypoint),
        ("detection", detection_fish, chaser_detection),
    ):
        manifest = {
            "recording_id": recording_id,
            "run_path": relative_paths[role],
            "dimensions": deepcopy(dimensions),
            "schema_binding": {"body_extension_present": role == "keypoint"},
            "source_authorities": {
                "fish_position": deepcopy(fish),
                "chaser_position": deepcopy(chaser),
                "body_frame": deepcopy(body) if role == "keypoint" else None,
            },
            "coordinate_policy": deepcopy(coordinate),
            "scale_policy": deepcopy(scale),
            "timing_policy": deepcopy(timing),
            "context": deepcopy(context),
            "identity_registries": deepcopy(registries),
        }
        relative_receipts[role] = {
            "record_sha256": _digest(f"relative-receipt:{role}"),
            "run_path": relative_paths[role],
            "recording_id": recording_id,
            "manifest_sha256": _digest(f"relative-manifest:{role}"),
            "payload_digest": _digest(f"relative-payload:{role}"),
            "run_manifest": manifest,
            "array_declarations": deepcopy(shared_declarations),
        }

    geometry = {
        "geometry_selection_run_path": "analysis/arena_geometry_selection/reviewed-v1",
        "geometry_selection_manifest_sha256": _digest("geometry"),
        "scale_authority_id": "reviewed-scale-v1",
        "scale_digest": _digest("scale"),
    }
    epoch_records = [
        {"analysis_role": "chaser_pre", "start_frame": 0, "end_frame": 3},
        {"analysis_role": "chaser_training", "start_frame": 3, "end_frame": 7},
        {"analysis_role": "chaser_post", "start_frame": 7, "end_frame": 10},
    ]
    arena = {"center_x_px": 100.0, "center_y_px": 100.0, "radius_mm": 10.0}
    radial_scientific: dict[str, dict[str, Any]] = {}
    for role, exact_key, fish in (
        ("keypoint", "keypoint_radial", keypoint_fish),
        ("detection", "detection_radial", detection_fish),
    ):
        radial_scientific[exact_key] = _scientific(
            {
                "recording_id": recording_id,
                "sources": {
                    "relative_frame": {
                        "run_path": relative_paths[role],
                        "manifest_sha256": _digest(f"relative-manifest:{role}"),
                    },
                    "protocol_semantic_selection": deepcopy(semantic_binding),
                    "fish_position": deepcopy(fish),
                    "timing": deepcopy(timing),
                    "arena_geometry_and_scale": deepcopy(geometry),
                },
                "position_provider": {
                    "provider_id": fish["provider_id"],
                    "provider_digest": fish["provider_digest"],
                    "status": "first_class_explicit_authority",
                },
                "epoch_records": deepcopy(epoch_records),
                "arena": deepcopy(arena),
                "selector_eligible": False,
                "production_authority": False,
                "registry_update": False,
            }
        )
    provider_records = []
    for role, exact_key, fish in (
        ("keypoint", "keypoint_radial", keypoint_fish),
        ("detection", "detection_radial", detection_fish),
    ):
        provider_records.append(
            {
                "provider_role": role,
                "provider_id": fish["provider_id"],
                "provider_digest": fish["provider_digest"],
                "fish_position_authority": deepcopy(fish),
                "relative_frame": {
                    "run_path": relative_paths[role],
                    "manifest_sha256": _digest(f"relative-manifest:{role}"),
                    "verification_mode": "receipt_bound",
                    "validation_receipt_sha256": _digest(f"relative-receipt:{role}"),
                },
                "radial_near_field": {
                    "run_path": exact_paths[exact_key],
                    "manifest_sha256": _digest(f"manifest:{exact_key}"),
                },
            }
        )
    spatial = _scientific(
        {
            "recording_id": recording_id,
            "sources": {
                "protocol_semantic_selection": deepcopy(semantic_binding),
                "arena_geometry_and_scale": deepcopy(geometry),
                "position_providers": provider_records,
            },
            "epoch_records": deepcopy(epoch_records),
            "arena": deepcopy(arena),
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
        }
    )
    motion = {
        "run_path": "analysis/track_kinematics_runs/provider/motion-v1",
        "manifest_sha256": _digest("motion-manifest"),
        "verification_digest": _digest("motion-verification"),
        "track_id": 0,
        "track_row_start": 0,
        "track_row_stop": 10,
    }
    bouts = {
        "schema_id": "palette.selector_ineligible_swim_bout_binding.v1",
        "run_name": "bouts-v1",
        "run_path": "analysis/swim_bout_runs/bouts-v1",
        "lineage_hash": _digest("bout-lineage"),
        "frame_axis_sha256": _digest("bout-frame-axis"),
        "source_track_motion_manifest_sha256": motion["manifest_sha256"],
        "source_track_motion_verification_digest": motion["verification_digest"],
        "track_id": 0,
        "track_row_start": 0,
        "track_row_stop": 10,
        "default_candidate_id": 1,
        "default_signal_id": 2,
        "default_signal_level": "filtered",
    }
    bouts["sha256"] = canonical_json_sha256(bouts)
    semantic_source = {
        **semantic_binding,
        "roles": ["chaser_pre", "chaser_training", "chaser_post"],
        "selector_eligible": False,
        "production_authority": False,
    }
    epoch = {
        "recording_id": recording_id,
        "sources": {
            "provider_motion": motion,
            "swim_bouts": bouts,
            "protocol_semantic_selection": semantic_source,
        },
        "parameters": {"track_id": 0, "physical_speed_level": "filtered"},
    }
    controller = _scientific(
        {
            "recording_id": recording_id,
            "source_relative_frame": {
                "run_path": relative_paths["keypoint"],
                "manifest_sha256": _digest("relative-manifest:keypoint"),
            },
            "semantic_selection": semantic_source,
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
        }
    )
    projection_record = {
        "schema_id": "palette.provider_motion.relative_frame_projection",
        "schema_version": 1,
        "join_policy": "left_join_missing_provider_rows_invalid_no_interpolation",
        "relative_frame_count": 10,
        "fallback": "prohibited",
    }
    bout = _scientific(
        {
            "recording_id": recording_id,
            "sources": {
                "relative_frame": {
                    "run_path": relative_paths["keypoint"],
                    "manifest_sha256": _digest("relative-manifest:keypoint"),
                },
                "motion": {
                    "run_path": motion["run_path"],
                    "manifest_sha256": motion["manifest_sha256"],
                    "relative_frame_projection": projection_record,
                },
                "swim_bouts": {
                    "run_path": bouts["run_path"],
                    "lineage_sha256": bouts["lineage_hash"],
                },
                "semantic_selection_manifest_sha256": semantic_binding[
                    "manifest_sha256"
                ],
                "controller_trial_payload_sha256": controller["payload_digest"],
            },
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
        }
    )
    escape = _scientific(
        {
            "recording_id": recording_id,
            "sources": {
                "motion": {
                    "run_path": motion["run_path"],
                    "manifest_sha256": motion["manifest_sha256"],
                    "relative_frame_projection": projection_record,
                },
                "controller_trial_payload_sha256": controller["payload_digest"],
                "bout_response_payload_sha256": bout["payload_digest"],
            },
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
        }
    )
    alignment = _scientific(
        {
            "recording_id": recording_id,
            "sources": {
                "relative_frame": {
                    "run_path": relative_paths["keypoint"],
                    "manifest_sha256": _digest("relative-manifest:keypoint"),
                },
                "protocol_semantic_selection": deepcopy(semantic_binding),
            },
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
        }
    )
    scientific_by_key = {
        **radial_scientific,
        "controller": controller,
        "bout": bout,
        "escape": escape,
        "spatial_occupancy": spatial,
        "body_alignment_by_distance": alignment,
    }
    exact_receipts: dict[str, dict[str, Any]] = {
        "semantic_selection": _receipt(
            key="semantic_selection",
            run_path=exact_paths["semantic_selection"],
            manifest={"recording_id": recording_id},
            recording_id=recording_id,
        ),
        "epoch_behavior": _receipt(
            key="epoch_behavior",
            run_path=exact_paths["epoch_behavior"],
            manifest=epoch,
            recording_id=recording_id,
        ),
    }
    for key, scientific_manifest in scientific_by_key.items():
        exact_receipts[key] = _receipt(
            key=key,
            run_path=exact_paths[key],
            manifest=_outer(
                key=key,
                run_path=exact_paths[key],
                recording_id=recording_id,
                scientific=scientific_manifest,
            ),
            recording_id=recording_id,
        )

    projection_path = (tmp_path / "projection.json").resolve()
    projection_path.write_text("{}", encoding="utf-8")
    exact_bindings: dict[str, dict[str, Any]] = {}
    exact_by_path: dict[Path, dict[str, Any]] = {}
    for key, receipt in exact_receipts.items():
        path = (tmp_path / f"{key}.receipt.json").resolve()
        path.write_text("{}", encoding="utf-8")
        exact_bindings[key] = subject._binding_from_receipt(receipt, path)
        exact_by_path[path] = receipt
    relative_bindings: dict[str, dict[str, Any]] = {}
    relative_by_path: dict[Path, dict[str, Any]] = {}
    for key, receipt in relative_receipts.items():
        path = (tmp_path / f"relative-{key}.receipt.json").resolve()
        path.write_text("{}", encoding="utf-8")
        relative_bindings[key] = subject._binding_from_receipt(receipt, path)
        relative_by_path[path] = receipt
    projection = {
        "schema_id": subject.PROJECTION_RECEIPT_SCHEMA_ID,
        "schema_version": 7,
        "analysis_zarr": str(archive),
        "recording_id": recording_id,
        "record_sha256": _digest("projection"),
        "exact_children": exact_bindings,
        "relative_frame_children": relative_bindings,
    }

    def read_projection(path: str | Path, **_kwargs: Any) -> dict[str, Any]:
        assert Path(path).resolve() == projection_path
        return projection

    def read_exact(path: str | Path, **_kwargs: Any) -> dict[str, Any]:
        return exact_by_path[Path(path).resolve()]

    def read_relative(path: str | Path, **_kwargs: Any) -> dict[str, Any]:
        return relative_by_path[Path(path).resolve()]

    motion_authority_record = {
        "analysis_zarr_path": str(archive),
        "position_source": {
            "run_path": keypoint_fish["source_authority_id"],
            "manifest_sha256": keypoint_fish["source_digest"],
        },
        "body_frame_source": {
            "run_path": body["source_authority_id"],
            "manifest_sha256": body["source_digest"],
        },
    }
    motion_authority = {
        "record": motion_authority_record,
        "sha256": body["provider_digest"],
    }

    def provider_manifest(
        _archive: Path, _source: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (
            {"payload_digest": motion["manifest_sha256"]},
            motion_authority,
            {
                "schema_id": "palette.zarr.metadata_equivalence",
                "schema_version": 1,
                "subtree_path": motion["run_path"],
                "node_count": 12,
                "group_count": 3,
                "array_count": 9,
                "declarations_sha256": _digest("motion-published-metadata"),
            },
        )

    monkeypatch.setattr(
        subject, "read_exact_chaser_projection_receipt", read_projection
    )
    monkeypatch.setattr(
        subject, "read_exact_immutable_child_validation_receipt", read_exact
    )
    monkeypatch.setattr(
        subject, "read_chaser_relative_frame_validation_receipt", read_relative
    )
    monkeypatch.setattr(subject, "_provider_motion_manifest", provider_manifest)
    return {
        "archive": archive,
        "recording_id": recording_id,
        "projection_path": projection_path,
        "projection": projection,
        "exact_receipts": exact_receipts,
        "relative_receipts": relative_receipts,
        "motion_authority": motion_authority,
        "dispositions": _base_dispositions(),
    }


def _build(fixture: dict[str, Any]) -> dict[str, Any]:
    return subject.build_validated_recording_behavior_bundle(
        fixture["projection_path"],
        absent_capability_dispositions=fixture["dispositions"],
        palette_commit="a" * 40,
        expected_analysis_zarr=fixture["archive"],
        expected_recording_id=fixture["recording_id"],
    )


def _redigest(bundle: dict[str, Any]) -> None:
    body = {key: value for key, value in bundle.items() if key != "record_sha256"}
    bundle["record_sha256"] = canonical_json_sha256(body)


def test_builds_one_closed_no_gaze_recording_behavior_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    bundle = _build(fixture)
    validated = subject.validate_validated_recording_behavior_bundle(
        bundle,
        expected_analysis_zarr=fixture["archive"],
        expected_recording_id="recording-1",
    )

    assert validated["capabilities"]["provider_motion"]["state"] == "complete"
    assert validated["capabilities"]["gaze"] == {
        "state": "unavailable",
        "reason_code": "upstream_segmentation_quality",
        "detail": None,
        "binding_scope": None,
        "binding_key": None,
    }
    assert validated["capabilities"]["fish_position_detection"]["state"] == ("complete")
    assert (
        validated["source_bindings"]["fish_position_detection"]["authority"][
            "provider_id"
        ]
        == "detection_bbox_centroid.v1"
    )
    assert (
        validated["source_bindings"]["provider_motion"]["published_metadata"][
            "subtree_path"
        ]
        == "analysis/track_kinematics_runs/provider/motion-v1"
    )
    assert validated["safety"]["zarr_mutation"] is False
    assert (
        "provider_motion_swim_bout_body_frame_composition"
        in validated["compatibility_proofs"]
    )


def test_requires_an_explicit_state_for_every_absent_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["dispositions"].pop("gaze")

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="Absent capability dispositions are inexact",
    ):
        _build(fixture)


def test_rejects_same_shape_relative_children_with_different_axis_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    declarations = fixture["relative_receipts"]["detection"]["array_declarations"]
    timestamp = next(
        item for item in declarations if item["path"] == "base/timestamp_ns"
    )
    timestamp["content_sha256"] = _digest("another-timestamp-axis")

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="differ at 'shared_array_declarations'",
    ):
        _build(fixture)


def test_rejects_motion_and_bouts_from_different_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    epoch = fixture["exact_receipts"]["epoch_behavior"]["manifest"]
    epoch["sources"]["swim_bouts"]["track_id"] = 1

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="not one exact track",
    ):
        _build(fixture)


def test_rejects_provider_motion_from_another_body_frame_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["motion_authority"]["record"]["body_frame_source"]["manifest_sha256"] = (
        _digest("other-body-frame")
    )

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="do not share the exact position/body-frame authority",
    ):
        _build(fixture)


def test_provider_motion_rejects_stale_published_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = (tmp_path / "recording.analysis.zarr").resolve()
    archive.mkdir()

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise subject.ZarrMetadataEquivalenceError("stale consolidated generation")

    monkeypatch.setattr(subject, "validate_direct_consolidated_subtree", reject)

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="published metadata is absent, stale, or inconsistent",
    ):
        subject._provider_motion_manifest(
            archive,
            {
                "run_path": "analysis/track_kinematics_runs/provider/motion-v1",
                "manifest_sha256": _digest("motion-manifest"),
            },
        )


def test_rejects_semantic_mixing_inside_controller_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    controller_outer = fixture["exact_receipts"]["controller"]["manifest"]
    scientific = controller_outer["scientific_manifest"]
    body = deepcopy(
        {key: value for key, value in scientific.items() if key != "payload_digest"}
    )
    body["semantic_selection"]["manifest_sha256"] = _digest("other-semantic")
    controller_outer["scientific_manifest"] = _scientific(body)
    controller_outer["scientific_payload_sha256"] = controller_outer[
        "scientific_manifest"
    ]["payload_digest"]

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="Controller-trial child binds another",
    ):
        _build(fixture)


def test_self_digest_tamper_fails_without_reopening_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    bundle = _build(fixture)
    bundle["recording_id"] = "other-recording"

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError, match="digest is stale"
    ):
        subject.validate_validated_recording_behavior_bundle(
            bundle, validate_current_sources=False
        )


def test_redigested_source_substitution_fails_current_composition_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    bundle = _build(fixture)
    changed = json.loads(json.dumps(bundle))
    changed["source_bindings"]["provider_motion"]["source"]["verification_digest"] = (
        _digest("forged-verification")
    )
    _redigest(changed)

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="changed at 'source_bindings'",
    ):
        subject.validate_validated_recording_behavior_bundle(changed)


def test_offline_validation_rejects_capability_bound_to_another_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    bundle = _build(fixture)
    bundle["capabilities"]["provider_motion"]["binding_key"] = "fish_position_keypoint"
    _redigest(bundle)

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="lacks one exact binding",
    ):
        subject.validate_validated_recording_behavior_bundle(
            bundle, validate_current_sources=False
        )


def test_offline_validation_rejects_noncomplete_capability_with_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    bundle = _build(fixture)
    bundle["capabilities"]["gaze"]["binding_scope"] = "source_bindings"
    bundle["capabilities"]["gaze"]["binding_key"] = "gaze"
    _redigest(bundle)

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="must not name a binding",
    ):
        subject.validate_validated_recording_behavior_bundle(
            bundle, validate_current_sources=False
        )


def test_offline_validation_rejects_open_source_binding_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    bundle = _build(fixture)
    bundle["source_bindings"]["provider_motion"]["selector"] = "latest"
    _redigest(bundle)

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="field set is inexact",
    ):
        subject.validate_validated_recording_behavior_bundle(
            bundle, validate_current_sources=False
        )


def test_existing_bundle_is_not_reused_for_another_projection_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "bundle.json"
    created = subject.ensure_validated_recording_behavior_bundle(
        fixture["projection_path"],
        absent_capability_dispositions=fixture["dispositions"],
        palette_commit="a" * 40,
        output_json=output,
        expected_analysis_zarr=fixture["archive"],
        expected_recording_id=fixture["recording_id"],
    )
    assert created["mode"] == "created"
    other_projection = (tmp_path / "other-projection.json").resolve()
    other_projection.write_text("{}", encoding="utf-8")

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="belongs to another projection receipt",
    ):
        subject.ensure_validated_recording_behavior_bundle(
            other_projection,
            absent_capability_dispositions=fixture["dispositions"],
            palette_commit="a" * 40,
            output_json=output,
            expected_analysis_zarr=fixture["archive"],
            expected_recording_id=fixture["recording_id"],
        )


def test_selector_named_motion_source_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    epoch = fixture["exact_receipts"]["epoch_behavior"]["manifest"]
    epoch["sources"]["provider_motion"][
        "run_path"
    ] = "analysis/track_kinematics_runs/provider/latest"

    with pytest.raises(
        subject.ValidatedRecordingBehaviorBundleError,
        match="exact non-selector child",
    ):
        _build(fixture)
