from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import hashlib

import pytest

from fisheye.analysis_workflows.eye_gaze_source_handle import (
    build_gaze_convention_review_receipt,
)
from fisheye.analysis_workflows import core_behavior_cohort_adapter as core_adapter
from fisheye.analysis_workflows.core_behavior_cohort_adapter import (
    CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
    CORE_BEHAVIOR_BUNDLE_METHOD_ID,
    CORE_BEHAVIOR_BUNDLE_STATUS,
    core_behavior_capability_contract,
)
from fisheye.analysis_workflows.validated_behavior_cohort import (
    build_validated_behavior_bundle_set,
    build_validated_behavior_cohort_membership,
    policy_envelope,
)
from fisheye.analytics_exports.kinematics_samples import (
    CORE_MOTION_SOURCE_SURFACE_PROFILE_ID,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CANONICAL_SWIM_BOUTS_CAPABILITY,
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CROSS_GRAIN_JOIN_AUTHORITY,
    KINEMATICS_SAMPLES_CAPABILITY,
    SUBJECT_BODY_FRAME_CAPABILITY,
    SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils import materialize_composable_chaser_successor_cohort as cohort

REPO = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = REPO / "scripts/submit_composable_chaser_successors_bsub.sh"


def _write_group(path: Path, attrs: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _record(attrs: dict[str, object], key: str, value: dict[str, object]) -> None:
    attrs[key] = value
    attrs[f"{key}_sha256"] = canonical_json_sha256(value)


def _self_digested(body: dict[str, object]) -> dict[str, object]:
    return {**body, "payload_digest": canonical_json_sha256(body)}


def _sealed(**body: object) -> dict[str, object]:
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_core_bundle_set(
    tmp_path: Path,
    *,
    archive: Path,
    recording_id: str,
) -> Path:
    receipt_root = tmp_path / "core-receipts"
    bundle_root = tmp_path / "core-bundles"
    receipt_root.mkdir()
    bundle_root.mkdir()
    source_path = tmp_path / "core-source-membership.json"
    source_member = {"recording_id": recording_id}
    source_path.write_text(json.dumps({"members": [source_member]}), encoding="utf-8")
    report_path = receipt_root / "execution-report.json"
    report_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    decision_path = tmp_path / "core-analysis-unit-decision.md"
    decision_path.write_text("one recording is one analysis unit\n", encoding="utf-8")
    report_binding = {
        "role": "core_behavior_workflow_execution",
        "path": str(report_path.resolve()),
        "file_sha256": _file_sha256(report_path),
        "record_sha256": canonical_json_sha256({"fixture": "execution-report"}),
        "schema_id": "palette.analysis_workflow_execution",
        "schema_version": 3,
    }
    membership = build_validated_behavior_cohort_membership(
        membership_id="chaser-core-authority-fixture-v1",
        source_membership={
            "adapter_id": "chaser_core_authority_fixture_v1",
            "schema_id": "fixture.chaser_membership",
            "schema_version": 1,
            "profile": "chaser_core_authority_fixture_v1",
            "path": str(source_path.resolve()),
            "file_sha256": _file_sha256(source_path),
            "record_sha256": canonical_json_sha256({"fixture": "membership"}),
            "member_count": 1,
            "source_members_sha256": canonical_json_sha256(
                [canonical_json_sha256(source_member)]
            ),
        },
        members=[
            {
                "source_ordinal": 1,
                "dataset_id": "fixture-dataset",
                "recording_id": recording_id,
                "analysis_zarr": str(archive.resolve()),
                "protocol_names": ["goodbatbadbat"],
                "protocol_hashes": ["a" * 64],
                "source_member_sha256": canonical_json_sha256(source_member),
                "source_subject_ids": ["capture-subject-fixture"],
                "source_subject_identity_status": "capture_time_non_authoritative",
                "acquisition_batch_id": None,
                "acquisition_batch_identity_status": "missing_historical_not_inferred",
                "analysis_unit_kind": "recording",
                "analysis_unit_id": recording_id,
                "membership_state": "admitted",
                "reason_code": None,
                "disposition_evidence": {
                    "evidence_type": "fixture_decision_v1",
                    "detail": "exact fixture authority accepted",
                    "path": None,
                    "file_sha256": None,
                    "record_sha256": None,
                },
                "admission_receipts": [report_binding],
            }
        ],
        analysis_zarr_root=tmp_path,
        admission_receipt_root=receipt_root,
        analysis_unit_policy=policy_envelope(
            {
                "policy_id": "recording_scoped_fixture_v1",
                "analysis_unit_kind": "recording",
                "member_id_field": "recording_id",
                "decision_evidence": {
                    "path": str(decision_path.resolve()),
                    "file_sha256": _file_sha256(decision_path),
                },
            }
        ),
        acquisition_batch_policy=policy_envelope(
            {
                "policy_id": "missing_batch_fixture_v1",
                "missing_identity_status": "missing_historical_not_inferred",
                "authoritative_identity_status": "authoritative",
                "inference_allowed": False,
            }
        ),
        temporal_alignment_policy=policy_envelope(
            {
                "policy_id": "exact_frame_fixture_v1",
                "temporal_alignment_requirement": "exact_acquisition_frame",
            }
        ),
        palette_commit="b" * 40,
        created_at_utc="2026-09-05T12:00:00Z",
    )
    membership_path = tmp_path / "core-membership.json"
    membership_path.write_text(json.dumps(membership, sort_keys=True), encoding="utf-8")

    join = _sealed(
        schema_id="palette.validated_behavior.cross_grain_join_authority",
        schema_version=1,
        recording_id=recording_id,
        camera_id="2010093",
        source_total_frames=100,
        source_sample_rate_hz=30.0,
        acquisition_camera_frame_ref="/metadata/acquisition_camera_frame",
        acquisition_camera_frame_sha256="c" * 64,
        source_video_metadata_sha256="d" * 64,
    )
    join_sha = join["payload_sha256"]
    capabilities: dict[str, object] = {}
    for capability in CORE_BEHAVIOR_CAPABILITY_KEYS:
        if capability == CROSS_GRAIN_JOIN_AUTHORITY:
            binding: object = join
        else:
            profile_id = f"{capability}_v1"
            source_values: dict[str, object] = {
                "schema_id": f"fixture.{capability}.source",
                "schema_version": 1,
                "recording_id": recording_id,
                "zarr_path": str(archive.resolve()),
            }
            if capability == KINEMATICS_SAMPLES_CAPABILITY:
                profile_id = CORE_MOTION_SOURCE_SURFACE_PROFILE_ID
                source_values.update(
                    {
                        "schema_version": 2,
                        "run_path": "analysis/track_kinematics_runs/offline/core-motion",
                        "tracks": [{"track_id": 7}],
                    }
                )
            elif capability == SUBJECT_BODY_FRAME_CAPABILITY:
                profile_id = SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID
                source_values["run_path"] = "analysis/subject_shape_runs/core-body"
            elif capability == CANONICAL_SWIM_BOUTS_CAPABILITY:
                source_values.update(
                    {
                        "run_path": "analysis/swim_bout_runs/core-bouts",
                        "track_id": 7,
                    }
                )
            binding = {
                "profile_id": profile_id,
                "source_binding": _sealed(**source_values),
                "projection_contract": _sealed(
                    schema_id=f"fixture.{capability}.projection",
                    schema_version=1,
                ),
                "join_authority_sha256": join_sha,
            }
        capabilities[capability] = {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": binding,
        }
    capability_bindings = {key: value["binding"] for key, value in capabilities.items()}
    inventory = {
        "execution_report": report_binding,
        "capability_bindings": capability_bindings,
    }
    bundle_path = bundle_root / "recording.bundle.json"
    bundle_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    contract = core_behavior_capability_contract()
    bundle_set = build_validated_behavior_bundle_set(
        bundle_set_id="chaser-core-authority-fixture-v1",
        membership=membership,
        membership_path=membership_path,
        membership_file_sha256=_file_sha256(membership_path),
        bundle_root=bundle_root,
        bundle_profile=core_adapter._bundle_profile(
            contract,
            export_profile_id=CORE_BEHAVIOR_EXPORT_PROFILE_ID,
        ),
        capability_contract=contract,
        members=[
            {
                "recording_id": recording_id,
                "bundle_state": "complete",
                "reason_code": None,
                "bundle": {
                    "adapter_id": CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
                    "path": str(bundle_path.resolve()),
                    "file_sha256": _file_sha256(bundle_path),
                    "record_sha256": canonical_json_sha256({"fixture": "bundle"}),
                    "schema_id": "palette.analysis_workflow_execution",
                    "schema_version": 3,
                    "method_id": CORE_BEHAVIOR_BUNDLE_METHOD_ID,
                    "status": CORE_BEHAVIOR_BUNDLE_STATUS,
                    "receipt_bindings": [report_binding],
                    "binding_inventory_sha256": canonical_json_sha256(inventory),
                },
                "capabilities": capabilities,
            }
        ],
        palette_commit="b" * 40,
        created_at_utc="2026-09-05T12:00:00Z",
    )
    bundle_set_path = tmp_path / "core-bundle-set.json"
    bundle_set_path.write_text(
        json.dumps(bundle_set, sort_keys=True),
        encoding="utf-8",
    )
    return bundle_set_path


def _composable_manifest(
    *,
    successor_kind: str,
    run_name: str,
    run_path: str,
    recording_id: str,
    scientific: dict[str, object],
) -> dict[str, object]:
    return _self_digested(
        {
            "successor_kind": successor_kind,
            "run_name": run_name,
            "run_path": run_path,
            "recording_id": recording_id,
            "scientific_manifest": scientific,
            "scientific_payload_sha256": scientific["payload_digest"],
            "selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "production_selector_activation": False,
            "registry_update": False,
        }
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    recording_id = "2026-08-10T17-20-55Z_arena_1_goodbatbadbat"
    recording = tmp_path / recording_id
    archive = recording / "zarr" / f"{recording_id}_analysis.zarr"
    raw_h5 = recording / "raw" / f"{recording_id}.h5"
    raw_h5.parent.mkdir(parents=True)
    raw_h5.write_bytes(b"sealed raw h5 fixture")
    stat = raw_h5.stat()
    _write_group(
        archive,
        {
            "recording_id": recording_id,
            "source_h5_path": str(raw_h5),
            "source_h5_size_bytes": stat.st_size,
            "source_h5_fingerprint": "fixture-stat",
            "source_h5_fingerprint_strategy": "stat_v1",
        },
    )
    _write_group(
        archive / "analysis/stimulus_runs/stimulus_canonical_v1_fixture",
        {"schema_id": "palette.stimulus_run"},
    )

    geometry_run = "arena_geometry_selection_fixture"
    _write_group(
        archive / "analysis/arena_geometry_selection",
        {"latest": geometry_run, "latest_complete": geometry_run},
    )
    geometry_attrs: dict[str, object] = {"selection_id": geometry_run}
    _record(
        geometry_attrs,
        "selection_record",
        {
            "selected_candidate": {
                "arena_binding": {
                    "arena_id": "arena_1",
                    "camera_serial": "2010093",
                }
            }
        },
    )
    _write_group(
        archive / "analysis/arena_geometry_selection" / geometry_run,
        geometry_attrs,
    )
    physical_attrs: dict[str, object] = {}
    _record(
        physical_attrs,
        "source_camera_physical_authority",
        {"camera_id": "2010093", "pixels_per_mm": 50.0},
    )
    _write_group(
        archive / "analysis/calibration/coordinate_frames",
        physical_attrs,
    )

    for run_name, provider in (
        (cohort.KEYPOINT_PROXY_RUN, "keypoint"),
        (cohort.DETECTION_PROXY_RUN, "detection"),
    ):
        attrs: dict[str, object] = {}
        _record(
            attrs,
            "chaser_input_provenance_proxy_manifest",
            {
                "recording_id": recording_id,
                "run_name": run_name,
                "provider": provider,
            },
        )
        _write_group(
            archive / "analysis/chaser_input_provenance_proxy_runs" / run_name,
            attrs,
        )

    snapshot = tmp_path / "registry.json"
    snapshot.write_text(
        json.dumps(
            [
                {
                    "dataset_id": "fixture-dataset",
                    "recording_id": recording_id,
                    "zarr_path": str(archive),
                    "protocol_name": "goodbatbadbat",
                    "protocol_hash": "a" * 64,
                    "arena_id": "arena_1",
                    "camera_id": "2010093",
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_core_bundle_set(
        tmp_path,
        archive=archive,
        recording_id=recording_id,
    )
    return archive, raw_h5, snapshot


def _plan(snapshot: Path, *, operations_root: Path) -> dict[str, object]:
    return cohort.plan_cohort_task(
        snapshot,
        operations_root=operations_root,
        core_bundle_set=snapshot.parent / "core-bundle-set.json",
    )


def _clean_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "palette-deployment"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    py = scripts / "py"
    shutil.copy2(REPO / "scripts/py", py)
    os.chmod(py, 0o755)
    (repo / "src").symlink_to(REPO / "src", target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Palette Tests",
            "-c",
            "user.email=palette-tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return repo, commit


def _eye_gaze_bindings(
    tmp_path: Path,
    archive: Path,
    *,
    recording_id: str,
) -> tuple[Path, Path]:
    run_name = "eye-angle-reviewed-v7"
    run_path = f"analysis/eye_angle_runs/{run_name}"
    _write_group(archive / run_path, {"stage_selector_eligible": False})
    numeric_validation = {
        "schema_id": "palette.gaze_convention_validation.v1",
        "schema_version": 1,
        "created_at_utc": "2026-08-31T12:00:00+00:00",
        "status": "pass",
        "zarr_path": str(archive),
        "eye_angle_run": run_name,
        "eye_angle_run_path": run_path,
        "read_only": True,
        "sampling": {"sample_rows": 2},
        "comparison_contract": {
            "object_angle_field": "egocentric_bearing/per_chaser/bearing_deg",
            "eye_angle_fields": [
                "left_gaze_signed_deg",
                "right_gaze_signed_deg",
            ],
            "coordinate_frame": "fish_body_frame",
            "zero": "fish_forward",
            "positive": "anatomical_left",
            "explicitly_not_comparable_fields": [
                "left_eye_angle_deg",
                "right_eye_angle_deg",
            ],
        },
        "checks": [{"name": "all_numeric_identities", "passed": True}],
        "direction_assumption": {
            "name": "ellipse_axis_direction_assumption",
            "passed": None,
            "review_required": True,
        },
        "review_png": str(tmp_path / "review.png"),
        "review_mask_source_path": "analysis/refined_subject_masks_runs/masks",
        "review_row_indices": [0, 1],
    }
    receipt = build_gaze_convention_review_receipt(
        numeric_validation=numeric_validation,
        source_eye_logical_sha256="b" * 64,
        reviewer="reviewer@example.org",
        reviewed_at_utc="2026-08-31T12:30:00+00:00",
        review_artifact_sha256="a" * 64,
    )
    receipt_path = tmp_path / "gaze_convention_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    bindings_path = tmp_path / "eye_gaze_bindings.json"
    bindings_path.write_text(
        json.dumps(
            [
                {
                    "recording_id": recording_id,
                    "analysis_zarr": str(archive),
                    "eye_run_name": run_name,
                    "eye_channel_variant": "smoothed",
                    "eye_convention_receipt": str(receipt_path),
                }
            ]
        ),
        encoding="utf-8",
    )
    return bindings_path, receipt_path


def test_plan_freezes_exact_recording_inputs(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = _plan(
        snapshot,
        operations_root=tmp_path / "operations",
    )

    assert task["recording_count"] == 1
    assert task["schema_version"] == 8
    assert task["status_counts"] == {"ready": 1}
    assert task["runnable_task_indices"] == [1]
    assert cohort.load_cohort_task(task)["task_sha256"] == task["task_sha256"]
    entry = task["entries"][0]
    assert entry["canonical_stimulus_run"] == "stimulus_canonical_v1_fixture"
    assert entry["keypoint_proxy"]["run_name"] == cohort.KEYPOINT_PROXY_RUN
    assert entry["detection_proxy"]["run_name"] == cohort.DETECTION_PROXY_RUN
    assert entry["core_authority"]["admission_state"] == ("static_capability_admitted")
    assert entry["core_authority"]["selected_track_id"] == 7
    assert (
        entry["core_authority"]["core_authority_roster_sha256"]
        == entry["core_authority"]["core_authority_roster"]["record_sha256"]
    )
    assert "motion_and_bouts" not in entry
    assert entry["output_run_names"]["epoch_behavior"] == cohort.EPOCH_BEHAVIOR_RUN
    assert entry["output_run_names"]["body_alignment_by_distance"] == (
        cohort.BODY_ALIGNMENT_RUN
    )
    assert entry["output_run_names"]["body_alignment_plot_bundle"] == (
        cohort.BODY_ALIGNMENT_RECIPE_BUNDLE_NAME
    )
    assert entry["output_run_names"]["body_alignment_plot_bundle"].endswith("recipe_v1")
    assert entry["output_run_names"]["keypoint_near_field_visits"] == (
        cohort.KEYPOINT_NEAR_FIELD_VISIT_RUN
    )
    assert entry["output_run_names"]["detection_near_field_visits"] == (
        cohort.DETECTION_NEAR_FIELD_VISIT_RUN
    )
    assert entry["near_field_visit_successor"] == {
        "provider_policy": cohort.NEAR_FIELD_VISIT_PROVIDER_POLICY,
        "receipt_policy": cohort.NEAR_FIELD_VISIT_RECEIPT_POLICY,
        "minimum_quality_sample_count": (
            cohort.NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
        ),
    }
    assert (
        f"analysis/chaser_near_field_visits_runs/"
        f"{cohort.KEYPOINT_NEAR_FIELD_VISIT_RUN}" in entry["output_group_paths"]
    )
    assert task["selection_policy"]["near_field_visit_provider_policy"] == (
        cohort.NEAR_FIELD_VISIT_PROVIDER_POLICY
    )
    assert task["selection_policy"]["near_field_visit_receipt_policy"] == (
        cohort.NEAR_FIELD_VISIT_RECEIPT_POLICY
    )
    assert len(entry["input_group_bindings"]) == 5
    assert task["selection_policy"]["core_authority_resolution"] == (
        cohort.CORE_BUNDLE_SELECTION_POLICY
    )
    assert entry["existing_output_group_paths"] == []
    assert task["safety"] == cohort.EXPECTED_SAFETY


def test_existing_outputs_remain_runnable_for_dynamic_admission(
    tmp_path: Path,
) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    operations_root = tmp_path / "operations"
    initial = _plan(snapshot, operations_root=operations_root)
    entry = initial["entries"][0]
    recording_id = entry["recording_id"]
    roster_sha256 = entry["core_authority"]["core_authority_roster_sha256"]

    for group_path in cohort._output_groups():
        run_name = Path(group_path).name
        attrs: dict[str, object] = {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "selection": "none",
            "palette_run_name": run_name,
            "recording_id": recording_id,
        }
        _record(
            attrs,
            "fixture_manifest",
            {
                "recording_id": recording_id,
                "run_name": run_name,
                "core_authority": {
                    "core_authority_roster_sha256": roster_sha256,
                },
            },
        )
        _write_group(archive / group_path, attrs)

    plot_dir = Path(entry["plot_output_dir"])
    plot_receipts = {
        plot_dir / f"{cohort.SUCCESSOR_RUN}_plot_receipt.json",
        plot_dir / f"{cohort.DASHBOARD_RECIPE_BUNDLE_NAME}_plot_receipt.json",
        plot_dir / "detailed" / f"{cohort.DETAILED_BUNDLE_NAME}_receipt.json",
        plot_dir / "detailed" / f"{cohort.DETAILED_RECIPE_BUNDLE_NAME}_receipt.json",
        plot_dir
        / "spatial_occupancy"
        / (f"{cohort.SPATIAL_OCCUPANCY_RUN}_" "spatial_occupancy_plot_receipt.json"),
        plot_dir
        / "spatial_occupancy"
        / (
            f"{cohort.SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME}_"
            "spatial_occupancy_plot_receipt.json"
        ),
        plot_dir
        / "body_alignment_by_distance"
        / (
            f"{cohort.BODY_ALIGNMENT_RECIPE_BUNDLE_NAME}_"
            "body_alignment_plot_receipt.json"
        ),
        plot_dir
        / "near_field_visits"
        / (
            f"{cohort.KEYPOINT_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME}_"
            "near_field_visit_plot_receipt.json"
        ),
        plot_dir
        / "near_field_visits"
        / (
            f"{cohort.DETECTION_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME}_"
            "near_field_visit_plot_receipt.json"
        ),
    }
    for receipt in plot_receipts:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{}", encoding="utf-8")

    replanned = _plan(snapshot, operations_root=operations_root)
    assert replanned["status_counts"] == {"validation_only": 1}
    assert replanned["runnable_task_indices"] == [1]

    successor = cohort.successor_cohort_task(replanned)
    assert successor["status_counts"] == {"validation_only": 1}
    assert successor["runnable_task_indices"] == [1]


def test_task_digest_rejects_mutation(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = _plan(snapshot, operations_root=tmp_path / "operations")
    task["entries"][0]["recording_id"] = "changed"

    with pytest.raises(cohort.ComposableChaserCohortError, match="digest is stale"):
        cohort.load_cohort_task(task)


def test_execution_revalidates_core_bundle_before_scratch_creation(
    tmp_path: Path,
) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = _plan(snapshot, operations_root=tmp_path / "operations")
    bundle_set = snapshot.parent / "core-bundle-set.json"
    bundle_set.write_text(
        bundle_set.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    repo, commit = _clean_repo(tmp_path)

    with pytest.raises(
        cohort.ComposableChaserCohortError,
        match="bundle-set or membership file changed",
    ):
        cohort.run_one(
            task,
            task_index=1,
            palette_repo=repo,
            palette_commit=commit,
            scratch_root=tmp_path / "scratch",
            receipt_root=tmp_path / "receipts",
            apply=True,
        )

    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "receipts").exists()


def test_replan_freezes_versioned_body_successor_from_prior_recording_set(
    tmp_path: Path,
) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = _plan(snapshot, operations_root=tmp_path / "old-operations")

    replanned = cohort.replan_cohort_task(
        original,
        operations_root=tmp_path / "body-operations",
    )

    assert replanned["recording_count"] == original["recording_count"]
    assert (
        replanned["selection_policy"]["successor_of_task_sha256"]
        == original["task_sha256"]
    )
    assert replanned["selection_policy"]["core_authority_resolution"] == (
        cohort.CORE_BUNDLE_SELECTION_POLICY
    )
    entry = replanned["entries"][0]
    assert entry["output_run_names"]["keypoint_relative"] == (
        cohort.KEYPOINT_RELATIVE_RUN
    )
    assert entry["core_authority"]["selected_track_id"] == 7
    assert len(entry["input_group_bindings"]) == 5
    assert cohort.load_cohort_task(replanned)["task_sha256"] == replanned["task_sha256"]


def test_task_successor_freezes_receipt_bound_plot_recipes(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = _plan(snapshot, operations_root=tmp_path / "operations")
    successor = cohort.successor_cohort_task(original)

    assert successor["schema_version"] == cohort.TASK_SCHEMA_VERSION
    assert (
        successor["selection_policy"]["successor_of_task_sha256"]
        == original["task_sha256"]
    )
    entry = successor["entries"][0]
    assert entry["relative_frame_validation"]["mode"] == (
        cohort.RELATIVE_FRAME_VALIDATION_MODE
    )
    assert entry["output_run_names"]["spatial_occupancy"] == (
        cohort.SPATIAL_OCCUPANCY_RECEIPT_BOUND_RUN
    )
    assert entry["output_run_names"]["spatial_plot_bundle"] == (
        cohort.SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME
    )
    assert entry["spatial_occupancy_successor"]["mode"] == (
        "materialize_missing_receipt_bound_v2"
    )
    assert entry["output_run_names"]["dashboard_bundle"] == (
        cohort.DASHBOARD_RECIPE_BUNDLE_NAME
    )
    assert entry["output_run_names"]["detailed_bundle"] == (
        cohort.DETAILED_RECIPE_BUNDLE_NAME
    )
    assert entry["output_run_names"]["detailed_bundle"].endswith("recipe_v1")
    assert entry["output_run_names"]["epoch_behavior"] == cohort.EPOCH_BEHAVIOR_RUN
    assert entry["output_run_names"]["body_alignment_plot_bundle"] == (
        cohort.BODY_ALIGNMENT_RECIPE_BUNDLE_NAME
    )
    assert entry["output_run_names"]["keypoint_near_field_visit_plot_bundle"] == (
        cohort.KEYPOINT_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME
    )
    assert (
        entry["output_run_names"]["detection_near_field_visit_plot_bundle"]
        == cohort.DETECTION_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME
    )
    assert successor["selection_policy"]["plot_recipe_provenance"] == (
        "self_contained_exact_parameters_v5"
    )
    assert successor["selection_policy"]["near_field_visit_provider_policy"] == (
        cohort.NEAR_FIELD_VISIT_PROVIDER_POLICY
    )
    assert cohort.load_cohort_task(successor)["task_sha256"] == successor["task_sha256"]


def test_task_rejects_partial_dual_provider_near_field_visit_outputs(
    tmp_path: Path,
) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = _plan(snapshot, operations_root=tmp_path / "operations")
    task["entries"][0]["output_run_names"].pop("detection_near_field_visit_plot_bundle")
    task["task_sha256"] = cohort._task_digest(task)

    with pytest.raises(
        cohort.ComposableChaserCohortError,
        match="must bind both providers and plot bundles",
    ):
        cohort.load_cohort_task(task)


def test_task_successor_reuses_existing_exact_spatial_science(tmp_path: Path) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = _plan(snapshot, operations_root=tmp_path / "operations")
    recording_id = original["entries"][0]["recording_id"]
    core_roster_sha256 = original["entries"][0]["core_authority"][
        "core_authority_roster_sha256"
    ]
    attrs: dict[str, object] = {
        "palette_run_completion_status": "complete",
        "stage_selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "selection": "none",
        "palette_run_name": cohort.SPATIAL_OCCUPANCY_RUN,
        "recording_id": recording_id,
    }
    _record(
        attrs,
        "composable_chaser_successor_manifest",
        {
            "recording_id": recording_id,
            "run_name": cohort.SPATIAL_OCCUPANCY_RUN,
            "core_authority": {
                "core_authority_roster_sha256": core_roster_sha256,
            },
        },
    )
    _write_group(
        archive
        / "analysis/chaser_spatial_occupancy_runs"
        / cohort.SPATIAL_OCCUPANCY_RUN,
        attrs,
    )

    successor = cohort.successor_cohort_task(original)

    entry = successor["entries"][0]
    assert entry["output_run_names"]["spatial_occupancy"] == (
        cohort.SPATIAL_OCCUPANCY_RUN
    )
    assert entry["spatial_occupancy_successor"] == {
        "mode": "reuse_existing_exact_complete_v1",
        "exact_run_name": cohort.SPATIAL_OCCUPANCY_RUN,
        "plot_bundle": cohort.SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME,
    }


def test_task_successor_rejects_completed_spatial_science_without_exact_core_roster(
    tmp_path: Path,
) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = _plan(snapshot, operations_root=tmp_path / "operations")
    recording_id = original["entries"][0]["recording_id"]
    attrs: dict[str, object] = {
        "palette_run_completion_status": "complete",
        "stage_selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "selection": "none",
        "palette_run_name": cohort.SPATIAL_OCCUPANCY_RUN,
        "recording_id": recording_id,
    }
    _record(
        attrs,
        "composable_chaser_successor_manifest",
        {
            "recording_id": recording_id,
            "run_name": cohort.SPATIAL_OCCUPANCY_RUN,
        },
    )
    _write_group(
        archive
        / "analysis/chaser_spatial_occupancy_runs"
        / cohort.SPATIAL_OCCUPANCY_RUN,
        attrs,
    )

    with pytest.raises(
        cohort.ComposableChaserCohortError,
        match="no sealed core-authority roster",
    ):
        cohort.successor_cohort_task(original)

    _record(
        attrs,
        "composable_chaser_successor_manifest",
        {
            "recording_id": recording_id,
            "run_name": cohort.SPATIAL_OCCUPANCY_RUN,
            "core_authority": {
                "core_authority_roster_sha256": "f" * 64,
            },
        },
    )
    _write_group(
        archive
        / "analysis/chaser_spatial_occupancy_runs"
        / cohort.SPATIAL_OCCUPANCY_RUN,
        attrs,
    )
    with pytest.raises(
        cohort.ComposableChaserCohortError,
        match="binds another core-authority roster",
    ):
        cohort.successor_cohort_task(original)


def test_task_successor_freezes_reviewed_gaze_and_plans_exact_projection(
    tmp_path: Path,
) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = _plan(snapshot, operations_root=tmp_path / "operations")
    recording_id = original["entries"][0]["recording_id"]
    bindings, _receipt = _eye_gaze_bindings(
        tmp_path,
        archive,
        recording_id=recording_id,
    )
    task = cohort.successor_cohort_task(
        original,
        eye_gaze_bindings=bindings,
    )

    entry = task["entries"][0]
    assert entry["eye_gaze"]["run_name"] == "eye-angle-reviewed-v7"
    assert entry["eye_gaze"]["channel_variant"] == "smoothed"
    assert entry["output_run_names"]["gaze_tracking"] == cohort.SUCCESSOR_RUN
    assert (
        f"analysis/chaser_gaze_tracking_runs/{cohort.SUCCESSOR_RUN}"
        in entry["output_group_paths"]
    )
    assert len(entry["input_group_bindings"]) == 6
    assert task["selection_policy"]["eye_gaze_resolution"] == (
        cohort.EYE_GAZE_BINDING_RESOLUTION
    )
    assert task["selection_policy"]["eye_gaze_binding_source"]["sha256"]
    assert cohort.load_cohort_task(task)["task_sha256"] == task["task_sha256"]

    repo, commit = _clean_repo(tmp_path)
    result = cohort.run_one(
        task,
        task_index=1,
        palette_repo=repo,
        palette_commit=commit,
        scratch_root=tmp_path / "scratch",
        receipt_root=tmp_path / "receipts",
        apply=False,
    )
    gaze = next(
        stage for stage in result["stages"] if stage["stage"] == "gaze_tracking"
    )
    assert gaze["command"][gaze["command"].index("--eye-run-name") + 1] == (
        "eye-angle-reviewed-v7"
    )
    assert "--eye-convention-receipt" in gaze["command"]
    assert "--radial-run-name" in gaze["command"]
    assert "gaze_exact_child_validation_receipt" in {
        stage["stage"] for stage in result["stages"]
    }
    projection = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "exact_chaser_projection_receipt"
    )
    assert "--gaze-receipt" in projection["command"]
    assert (
        Path(
            projection["command"][projection["command"].index("--output-json") + 1]
        ).name
        == cohort.GAZE_EPOCH_ALIGNMENT_PROJECTION_RECEIPT_NAME
    )


def test_gaze_successor_rejects_changed_convention_receipt(tmp_path: Path) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = _plan(snapshot, operations_root=tmp_path / "operations")
    recording_id = original["entries"][0]["recording_id"]
    bindings, receipt = _eye_gaze_bindings(
        tmp_path,
        archive,
        recording_id=recording_id,
    )
    task = cohort.successor_cohort_task(
        original,
        eye_gaze_bindings=bindings,
    )
    receipt.write_text("{}", encoding="utf-8")

    with pytest.raises(
        cohort.ComposableChaserCohortError,
        match="receipt file has changed",
    ):
        cohort.run_one(
            task,
            task_index=1,
            palette_repo=tmp_path / "unused",
            palette_commit="0" * 40,
            scratch_root=tmp_path / "scratch",
            receipt_root=tmp_path / "receipts",
        )


def test_gaze_task_rejects_missing_cohort_resolution_policy(tmp_path: Path) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = _plan(snapshot, operations_root=tmp_path / "operations")
    recording_id = original["entries"][0]["recording_id"]
    bindings, _receipt = _eye_gaze_bindings(
        tmp_path,
        archive,
        recording_id=recording_id,
    )
    task = cohort.successor_cohort_task(
        original,
        eye_gaze_bindings=bindings,
    )
    task["selection_policy"].pop("eye_gaze_resolution")
    task["task_sha256"] = cohort._task_digest(task)

    with pytest.raises(
        cohort.ComposableChaserCohortError,
        match="without a resolution policy",
    ):
        cohort.load_cohort_task(task)


def test_run_one_rejects_changed_frozen_input_metadata(tmp_path: Path) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = _plan(snapshot, operations_root=tmp_path / "operations")
    proxy_path = (
        archive
        / "analysis/chaser_input_provenance_proxy_runs"
        / cohort.KEYPOINT_PROXY_RUN
        / "zarr.json"
    )
    proxy_path.write_text("{}", encoding="utf-8")

    with pytest.raises(cohort.ComposableChaserCohortError, match="metadata changed"):
        cohort.run_one(
            task,
            task_index=1,
            palette_repo=tmp_path / "unused",
            palette_commit="0" * 40,
            scratch_root=tmp_path / "scratch",
            receipt_root=tmp_path / "receipts",
        )


def test_run_one_dry_run_renders_complete_serial_chain(tmp_path: Path) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = _plan(snapshot, operations_root=tmp_path / "operations")
    repo, commit = _clean_repo(tmp_path)

    result = cohort.run_one(
        task,
        task_index=1,
        palette_repo=repo,
        palette_commit=commit,
        scratch_root=tmp_path / "scratch",
        receipt_root=tmp_path / "receipts",
        apply=False,
    )

    assert result["status"] == "planned_no_writes"
    assert result["palette_commit"] == commit
    assert [stage["stage"] for stage in result["stages"]] == [
        "semantic_stimulus",
        "semantic_epoch_v1",
        "semantic_epoch_v2",
        "semantic_selection",
        "epoch_behavior",
        "keypoint_relative_frame",
        "detection_relative_frame",
        "composable_successors",
        "keypoint_radial_near_field",
        "detection_radial_near_field",
        "keypoint_near_field_visits",
        "detection_near_field_visits",
        "body_alignment_by_distance",
        "spatial_occupancy",
        "keypoint_near_field_visit_plots",
        "detection_near_field_visit_plots",
        "body_alignment_by_distance_plots",
        "spatial_occupancy_plots",
        "dashboard_plots",
        "detailed_plots",
    ]
    assert all(stage["mode"] == "planned_no_write" for stage in result["stages"])
    keypoint_relative = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "keypoint_relative_frame"
    )
    detection_relative = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "detection_relative_frame"
    )
    successors = next(
        stage for stage in result["stages"] if stage["stage"] == "composable_successors"
    )
    assert "--body-frame-run" not in keypoint_relative["command"]
    assert "--body-frame-run" not in detection_relative["command"]
    assert "--core-authority-roster" in keypoint_relative["command"]
    assert "--core-authority-roster" in detection_relative["command"]
    assert "--core-track-id" in successors["command"]
    assert "--provider-motion-run-path" not in successors["command"]
    assert "--swim-bout-run-name" not in successors["command"]
    assert "--track-id" not in successors["command"]
    assert "--no-body-extension" not in successors["command"]
    assert result["safety"] == cohort.EXPECTED_SAFETY
    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "receipts").exists()


def test_receipt_bound_successor_dry_run_passes_targeted_receipts(
    tmp_path: Path,
) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    original = _plan(snapshot, operations_root=tmp_path / "operations")
    task = cohort.successor_cohort_task(original)
    entry = task["entries"][0]
    repo, commit = _clean_repo(tmp_path)

    result = cohort.run_one(
        task,
        task_index=1,
        palette_repo=repo,
        palette_commit=commit,
        scratch_root=tmp_path / "scratch",
        receipt_root=tmp_path / "receipts",
        apply=False,
    )

    names = [stage["stage"] for stage in result["stages"]]
    assert names[7:9] == [
        "keypoint_relative_frame_validation_receipt",
        "detection_relative_frame_validation_receipt",
    ]
    for source in (
        "semantic_selection",
        "epoch_behavior",
        "keypoint_radial",
        "detection_radial",
        "controller",
        "bout",
        "escape",
        "body_alignment_by_distance",
        "spatial_occupancy",
        "keypoint_near_field_visits",
        "detection_near_field_visits",
    ):
        assert f"{source}_exact_child_validation_receipt" in names
    projection_receipt = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "exact_chaser_projection_receipt"
    )
    assert "--epoch-behavior-receipt" in projection_receipt["command"]
    assert "--body-alignment-by-distance-receipt" in projection_receipt["command"]
    assert "--keypoint-near-field-visits-receipt" not in projection_receipt["command"]
    assert "--detection-near-field-visits-receipt" not in projection_receipt["command"]
    assert (
        Path(
            projection_receipt["command"][
                projection_receipt["command"].index("--output-json") + 1
            ]
        ).name
        == cohort.EPOCH_ALIGNMENT_PROJECTION_RECEIPT_NAME
    )
    alignment = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "body_alignment_by_distance"
    )
    assert "--relative-frame-receipt" in alignment["command"]
    assert "--semantic-selection-receipt" in alignment["command"]
    spatial = next(
        stage for stage in result["stages"] if stage["stage"] == "spatial_occupancy"
    )
    detailed = next(
        stage for stage in result["stages"] if stage["stage"] == "detailed_plots"
    )
    dashboard = next(
        stage for stage in result["stages"] if stage["stage"] == "dashboard_plots"
    )
    for stage in (spatial, detailed):
        assert "--keypoint-relative-frame-receipt" in stage["command"]
        assert "--detection-relative-frame-receipt" in stage["command"]
        keypoint_receipt = stage["command"][
            stage["command"].index("--keypoint-relative-frame-receipt") + 1
        ]
        assert commit in keypoint_receipt
    for option in (
        "--semantic-selection-receipt",
        "--keypoint-radial-receipt",
        "--detection-radial-receipt",
    ):
        assert option in spatial["command"]
        assert option not in detailed["command"]
    for option in (
        "--controller-validation-receipt",
        "--bout-validation-receipt",
        "--escape-validation-receipt",
    ):
        assert option in dashboard["command"]
        assert option in detailed["command"]
    assert (
        dashboard["command"][dashboard["command"].index("--bundle-name") + 1]
        == cohort.DASHBOARD_RECIPE_BUNDLE_NAME
    )
    occupancy_plot = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "spatial_occupancy_plots"
    )
    assert (
        occupancy_plot["command"][occupancy_plot["command"].index("--bundle-name") + 1]
        == cohort.SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME
    )
    assert "--source-validation-receipt" in occupancy_plot["command"]
    alignment_plot = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "body_alignment_by_distance_plots"
    )
    assert (
        alignment_plot["command"][alignment_plot["command"].index("--bundle-name") + 1]
        == cohort.BODY_ALIGNMENT_RECIPE_BUNDLE_NAME
    )
    assert "--source-validation-receipt" in alignment_plot["command"]

    for provider in ("keypoint", "detection"):
        visit = next(
            stage
            for stage in result["stages"]
            if stage["stage"] == f"{provider}_near_field_visits"
        )
        for option in (
            "--relative-frame-validation-receipt",
            "--semantic-selection-validation-receipt",
            "--radial-validation-receipt",
            "--expected-recording-id",
            "--minimum-quality-sample-count",
        ):
            assert option in visit["command"]
        assert (
            int(
                visit["command"][
                    visit["command"].index("--minimum-quality-sample-count") + 1
                ]
            )
            == cohort.NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
        )
        visit_plot = next(
            stage
            for stage in result["stages"]
            if stage["stage"] == f"{provider}_near_field_visit_plots"
        )
        assert "--source-validation-receipt" in visit_plot["command"]
        assert (
            visit_plot["command"][visit_plot["command"].index("--bundle-name") + 1]
            == entry["output_run_names"][f"{provider}_near_field_visit_plot_bundle"]
        )


def test_reused_plot_receipt_is_content_verified(tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    output.write_bytes(b"figure bytes")
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "recording_id": "recording-1",
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "outputs": [
            {
                "path": str(output),
                "sha256": cohort._sha256_file(output),
                "size_bytes": output.stat().st_size,
            }
        ],
    }
    receipt["payload_sha256"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert cohort._validated_plot_receipt(receipt_path, recording_id="recording-1")
    output.write_bytes(b"changed")
    with pytest.raises(cohort.ComposableChaserCohortError, match="differs"):
        cohort._validated_plot_receipt(receipt_path, recording_id="recording-1")


def test_reused_visit_plot_receipt_requires_exact_source_run(tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    output.write_bytes(b"figure bytes")
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "recording_id": "recording-1",
        "run_name": "visits-a",
        "source_binding": {
            "run_path": "analysis/chaser_near_field_visits_runs/visits-a"
        },
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "outputs": [
            {
                "path": str(output),
                "sha256": cohort._sha256_file(output),
                "size_bytes": output.stat().st_size,
            }
        ],
    }
    receipt["payload_sha256"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert cohort._validated_plot_receipt(
        receipt_path,
        recording_id="recording-1",
        expected_source_run_path=("analysis/chaser_near_field_visits_runs/visits-a"),
    )
    with pytest.raises(cohort.ComposableChaserCohortError, match="source run mismatch"):
        cohort._validated_plot_receipt(
            receipt_path,
            recording_id="recording-1",
            expected_source_run_path=(
                "analysis/chaser_near_field_visits_runs/visits-b"
            ),
        )


def test_existing_visit_science_requires_current_exact_source_bindings(
    tmp_path: Path,
) -> None:
    archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = _plan(snapshot, operations_root=tmp_path / "operations")
    entry = task["entries"][0]
    recording_id = entry["recording_id"]
    outputs = entry["output_run_names"]
    relative_path = (
        f"analysis/chaser_relative_frame_runs/{outputs['keypoint_relative']}"
    )
    semantic_path = (
        "analysis/protocol_semantic_chaser_selection_runs/"
        f"{outputs['semantic_selection']}"
    )
    radial_path = f"analysis/chaser_radial_near_field_runs/{outputs['keypoint_radial']}"
    visit_run = outputs["keypoint_near_field_visits"]
    visit_path = f"analysis/chaser_near_field_visits_runs/{visit_run}"

    relative_attrs: dict[str, object] = {}
    _record(relative_attrs, "chaser_relative_frame_manifest", {"source": "relative"})
    _write_group(archive / relative_path, relative_attrs)
    semantic_attrs: dict[str, object] = {}
    _record(
        semantic_attrs,
        "protocol_semantic_chaser_selection_manifest",
        {"source": "semantic"},
    )
    _write_group(archive / semantic_path, semantic_attrs)
    relative_sha = relative_attrs["chaser_relative_frame_manifest_sha256"]
    semantic_sha = semantic_attrs["protocol_semantic_chaser_selection_manifest_sha256"]
    provider = {
        "provider_id": "keypoint_triad.v1",
        "provider_digest": "a" * 64,
    }
    fish_position = {
        **provider,
        "coordinate_authority_id": "/coordinate@pixel_frame",
    }
    radial_scientific = _self_digested(
        {
            "position_provider": provider,
            "sources": {
                "relative_frame": {
                    "run_path": relative_path,
                    "manifest_sha256": relative_sha,
                },
                "protocol_semantic_selection": {
                    "run_path": semantic_path,
                    "manifest_sha256": semantic_sha,
                },
                "fish_position": fish_position,
            },
        }
    )
    radial_manifest = _composable_manifest(
        successor_kind="chaser_radial_near_field",
        run_name=outputs["keypoint_radial"],
        run_path=radial_path,
        recording_id=recording_id,
        scientific=radial_scientific,
    )
    radial_attrs: dict[str, object] = {}
    _record(radial_attrs, "composable_chaser_successor_manifest", radial_manifest)
    _write_group(archive / radial_path, radial_attrs)

    radial_sha = radial_attrs["composable_chaser_successor_manifest_sha256"]

    def write_visit(relative_source_path: str) -> None:
        scientific = _self_digested(
            {
                "scientific_schema": {
                    "schema_id": cohort.NEAR_FIELD_VISIT_SCIENTIFIC_SCHEMA_ID,
                    "schema_version": 1,
                },
                "recording_id": recording_id,
                "sources": {
                    "relative_frame": {
                        "run_path": relative_source_path,
                        "manifest_sha256": relative_sha,
                    },
                    "protocol_semantic_selection": {
                        "run_path": semantic_path,
                        "manifest_sha256": semantic_sha,
                    },
                    "radial_near_field": {
                        "run_path": radial_path,
                        "manifest_sha256": radial_sha,
                        "scientific_payload_sha256": radial_scientific[
                            "payload_digest"
                        ],
                    },
                    "fish_position": fish_position,
                },
                "position_provider": provider,
                "config": {
                    "minimum_quality_sample_count": (
                        cohort.NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
                    )
                },
                "selector_eligible": False,
                "selection": "none",
                "production_authority": False,
                "registry_update": False,
            }
        )
        publication = _composable_manifest(
            successor_kind="chaser_near_field_visits",
            run_name=visit_run,
            run_path=visit_path,
            recording_id=recording_id,
            scientific=scientific,
        )
        attrs: dict[str, object] = {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "selection": "none",
            "palette_run_name": visit_run,
            "recording_id": recording_id,
        }
        _record(attrs, "composable_chaser_successor_manifest", publication)
        _write_group(archive / visit_path, attrs)

    binding = {
        "recording_id": recording_id,
        "visit_run": visit_run,
        "relative_frame_run": outputs["keypoint_relative"],
        "semantic_selection_run": outputs["semantic_selection"],
        "radial_near_field_run": outputs["keypoint_radial"],
        "minimum_quality_sample_count": (
            cohort.NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
        ),
    }
    write_visit(relative_path)
    assert cohort._existing_near_field_visit_output(archive, **binding)

    write_visit("analysis/chaser_relative_frame_runs/wrong-provider")
    with pytest.raises(
        cohort.ComposableChaserCohortError,
        match="relative-frame binding is incompatible",
    ):
        cohort._existing_near_field_visit_output(archive, **binding)


def test_reused_detailed_receipt_requires_exact_recipe_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "figure.png"
    output.write_bytes(b"figure bytes")
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "recording_id": "recording-1",
        "plot_recipe_id": "sealed_chaser_detailed_plot_bundle_v2",
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "outputs": [
            {
                "path": str(output),
                "sha256": cohort._sha256_file(output),
                "size_bytes": output.stat().st_size,
            }
        ],
    }
    receipt["payload_sha256"] = canonical_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        cohort.ComposableChaserCohortError, match="recipe identity mismatch"
    ):
        cohort._validated_plot_receipt(
            receipt_path,
            recording_id="recording-1",
            expected_plot_recipe_id=cohort.DETAILED_PLOT_RECIPE_ID,
        )


def test_bsub_submitter_renders_pinned_array_without_submission(
    tmp_path: Path,
) -> None:
    _archive, _raw_h5, snapshot = _fixture(tmp_path)
    task = _plan(snapshot, operations_root=tmp_path / "operations")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    repo, commit = _clean_repo(tmp_path)
    run_root = tmp_path / "runs"

    completed = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--task",
            str(task_path),
            "--palette-repo",
            str(repo),
            "--palette-commit",
            commit,
            "--run-root",
            str(run_root),
            "--run-id",
            "fixture",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PALETTE_PYTHON": os.environ.get("PALETTE_PYTHON", "")},
    )

    assert "mode=dry_run_no_submission" in completed.stdout
    assert "recording_count=1" in completed.stdout
    assert "array_indices=1-1" in completed.stdout
    assert "selected_recording_count=1" in completed.stdout
    run_dir = run_root / "composable_chaser_successors_fixture"
    job_script = (run_dir / "run_one_recording.sh").read_text(encoding="utf-8")
    assert "LSB_JOBINDEX" in job_script
    assert "materialize_composable_chaser_successor_cohort run-one" in job_script
    assert f"--palette-commit {commit}" in job_script
    submission = (run_dir / "submission.env").read_text(encoding="utf-8")
    assert "selector_eligible=false" in submission
    assert "production_authority=false" in submission
    assert "registry_update=false" in submission
    assert "submit_requested=0" in submission
