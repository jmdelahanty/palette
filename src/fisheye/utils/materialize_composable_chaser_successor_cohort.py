"""Plan or run the selector-ineligible GoodBatBadBat successor cohort.

The planner consumes a frozen registry JSON export and resolves every mutable
recording-local authority exactly once.  The resulting task contains exact run
names, manifest digests, metadata-file digests, raw-H5 stat identities, and a
canonical task digest.  ``run-one`` accepts only that task, revalidates all
frozen inputs, and executes one recording serially so no two processes write
the same analysis Zarr.

Neither command resolves a scientific selector while executing a task.  Every
publication remains selector-ineligible and no command writes the registry.
Task schema v7 adds exact keypoint- and detection-provider near-field visit
children, independent validation receipts, and receipt-bound static plots.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from fisheye.analysis_workflows.chaser_near_field_visit_successor import (
    MIN_VISIT_SAMPLE_COUNT as NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT,
    SCHEMA_ID as NEAR_FIELD_VISIT_SCIENTIFIC_SCHEMA_ID,
    SCHEMA_VERSION as NEAR_FIELD_VISIT_SCIENTIFIC_SCHEMA_VERSION,
)
from fisheye.analysis_workflows.eye_gaze_source_handle import (
    validate_gaze_convention_review_receipt,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.chaser_spatial_occupancy_display import (
    DISPLAY_RECIPE_ID as SPATIAL_OCCUPANCY_PLOT_RECIPE_ID,
)


TASK_SCHEMA_ID = "palette.composable_chaser_successor_cohort_task"
TASK_SCHEMA_VERSION = 7
RECEIPT_SCHEMA_ID = "palette.composable_chaser_successor_cohort_receipt"
RECEIPT_SCHEMA_VERSION = 1

SEMANTIC_STIMULUS_RUN = "stimulus_semantic_goodbatbadbat_historical_20260825_v2"
SEMANTIC_EPOCH_V1_RUN = "stimulus_epochs_semantic_goodbatbadbat_20260825_r2_v1"
SEMANTIC_EPOCH_V2_RUN = "stimulus_epochs_semantic_goodbatbadbat_20260825_r2_v2"
SEMANTIC_SELECTION_RUN = "protocol_semantic_chaser_goodbatbadbat_historical_20260825_v2"
EPOCH_BEHAVIOR_RUN = "epoch_behavior_goodbatbadbat_keypoint_triad_semantic_20260830_v2"
BODY_ALIGNMENT_RUN = (
    "goodbatbadbat_chaser_body_alignment_by_distance_keypoint_semantic_20260830_v1"
)
BODY_ALIGNMENT_RECIPE_BUNDLE_NAME = (
    "goodbatbadbat_chaser_body_alignment_by_distance_keypoint_semantic_20260830_"
    "recipe_v2"
)
KEYPOINT_PROXY_RUN = "chaser_input_provenance_proxy_keypoint_triad_cohort_20260821_v2"
DETECTION_PROXY_RUN = (
    "chaser_input_provenance_proxy_detection_bbox_centroid_cohort_20260821_v2"
)
KEYPOINT_RELATIVE_RUN = (
    "chaser_relative_frame_keypoint_triad_cohort_20260827_"
    "exact_body_frame_projection_v4"
)
DETECTION_RELATIVE_RUN = (
    "chaser_relative_frame_detection_bbox_centroid_cohort_20260825_"
    "exact_trials_session_time_activity_orthogonal_v3"
)
SUCCESSOR_RUN = "goodbatbadbat_chaser_successors_20260827_body_frame_projection_v4"
KEYPOINT_RADIAL_RUN = (
    "goodbatbadbat_chaser_radial_near_field_20260827_body_frame_projection_v3"
)
DETECTION_RADIAL_RUN = (
    "goodbatbadbat_chaser_radial_near_field_detection_bbox_centroid_20260825_"
    "exact_session_time_activity_orthogonal_v2"
)
KEYPOINT_NEAR_FIELD_VISIT_RUN = (
    "goodbatbadbat_chaser_near_field_visits_keypoint_body_frame_20260903_v1"
)
DETECTION_NEAR_FIELD_VISIT_RUN = (
    "goodbatbadbat_chaser_near_field_visits_detection_bbox_centroid_20260903_v1"
)
KEYPOINT_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME = (
    "goodbatbadbat_chaser_near_field_visits_keypoint_body_frame_20260903_recipe_v1"
)
DETECTION_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME = (
    "goodbatbadbat_chaser_near_field_visits_detection_bbox_centroid_20260903_recipe_v1"
)
NEAR_FIELD_VISIT_PROVIDER_POLICY = "both_exact_first_class_position_providers_v1"
NEAR_FIELD_VISIT_RECEIPT_POLICY = (
    "independent_exact_child_receipts_existing_projection_unchanged_v1"
)
NEAR_FIELD_VISIT_PLOT_RECIPE_ID = "persisted_exact_near_field_visit_trajectories_v1"
SPATIAL_OCCUPANCY_RUN = (
    "goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260827_"
    "body_frame_projection_v3"
)
DETAILED_BUNDLE_NAME = "goodbatbadbat_chaser_detailed_body_frame_v5"
SPATIAL_OCCUPANCY_RECEIPT_BOUND_RUN = (
    "goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260827_"
    "body_frame_projection_receipt_bound_v4"
)
SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME = (
    "goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260827_"
    "body_frame_projection_recipe_v4"
)
DASHBOARD_RECIPE_BUNDLE_NAME = "goodbatbadbat_chaser_dashboard_body_frame_recipe_v3"
DETAILED_RECIPE_BUNDLE_NAME = "goodbatbadbat_chaser_detailed_body_frame_recipe_v7"
DETAILED_PLOT_RECIPE_ID = "sealed_chaser_detailed_plot_bundle_v5"
BODY_ALIGNMENT_PLOT_RECIPE_ID = "persisted_anatomical_alignment_distance_bins_static_v2"
RELATIVE_FRAME_VALIDATION_MODE = "reusable_direct_subtree_receipt_v1"
EPOCH_ALIGNMENT_PROJECTION_RECEIPT_NAME = (
    "exact_chaser.epoch_alignment.projection_receipt.v7.json"
)
GAZE_EPOCH_ALIGNMENT_PROJECTION_RECEIPT_NAME = (
    "exact_chaser.gaze_epoch_alignment.projection_receipt.v8.json"
)
EYE_GAZE_BINDING_RESOLUTION = (
    "exact_eye_run_and_reviewed_convention_receipt_no_selector_v1"
)

MOTION_BOUT_PAIRS = (
    (
        "provider_motion_goodbatbadbat_keypoint_triad_talk_20260818_v2",
        "swim_bouts_goodbatbadbat_keypoint_triad_talk_20260818_v2",
    ),
    (
        "provider_motion_goodbatbadbat_keypoint_triad_talk_20260818_v1",
        "swim_bouts_goodbatbadbat_keypoint_triad_talk_20260818_v1",
    ),
    (
        "provider_motion_goodbatbadbat_keypoint_triad_canary_20260818_v1",
        "swim_bouts_goodbatbadbat_keypoint_triad_canary_20260818_v1",
    ),
)

EXPECTED_SAFETY = {
    "selector_eligible": False,
    "production_authority": False,
    "registry_update": False,
    "selector_activation": False,
    "one_writer_per_analysis_zarr": True,
}


class ComposableChaserCohortError(ValueError):
    """Raised when a cohort plan or execution cannot remain fail-closed."""


def _fail(message: str) -> None:
    raise ComposableChaserCohortError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be non-empty normalized text.")
    return value


def _exact_name(value: object, *, field: str) -> str:
    name = _text(value, field=field)
    if (
        name in {"latest", "latest_complete", "selected", "current", ".", ".."}
        or "/" in name
        or "\\" in name
        or any(character.isspace() for character in name)
    ):
        _fail(f"{field} must be one exact immutable run name.")
    return name


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_eye_gaze_bindings(
    source: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Freeze explicit reviewed eye sources without resolving any selector."""

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Eye-gaze binding file does not exist: {path}")
    source_bytes = path.read_bytes()
    try:
        rows = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ComposableChaserCohortError(
            f"Eye-gaze binding file is not strict JSON: {path}"
        ) from exc
    if not isinstance(rows, list) or not rows:
        _fail("Eye-gaze bindings must be one non-empty JSON row list.")
    expected_keys = {
        "recording_id",
        "analysis_zarr",
        "eye_run_name",
        "eye_channel_variant",
        "eye_convention_receipt",
    }
    frozen: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, field=f"eye-gaze binding row {index}")
        if set(row) != expected_keys:
            _fail(f"Eye-gaze binding row {index} has missing or unexpected fields.")
        recording_id = _text(
            row.get("recording_id"), field=f"eye-gaze row {index} recording_id"
        )
        if recording_id in frozen:
            _fail(f"Eye-gaze bindings duplicate recording {recording_id!r}.")
        archive = (
            Path(
                _text(
                    row.get("analysis_zarr"),
                    field=f"eye-gaze row {index} analysis_zarr",
                )
            )
            .expanduser()
            .resolve()
        )
        run_name = _exact_name(
            row.get("eye_run_name"), field=f"eye-gaze row {index} eye_run_name"
        )
        channel_variant = _text(
            row.get("eye_channel_variant"),
            field=f"eye-gaze row {index} eye_channel_variant",
        )
        if channel_variant not in {"raw", "smoothed"}:
            _fail("Eye-gaze channel variant must be exact 'raw' or 'smoothed'.")
        run_path = f"analysis/eye_angle_runs/{run_name}"
        _attrs, metadata_sha256 = _zarr_attrs(
            archive / run_path,
            field=f"eye-gaze row {index} exact eye-angle run",
        )
        receipt_path = (
            Path(
                _text(
                    row.get("eye_convention_receipt"),
                    field=f"eye-gaze row {index} convention receipt",
                )
            )
            .expanduser()
            .resolve()
        )
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"Eye-gaze convention receipt does not exist: {receipt_path}"
            )
        try:
            receipt = _mapping(
                json.loads(receipt_path.read_bytes()),
                field=f"eye-gaze row {index} convention receipt",
            )
        except json.JSONDecodeError as exc:
            raise ComposableChaserCohortError(
                f"Eye-gaze convention receipt is not strict JSON: {receipt_path}"
            ) from exc
        logical_sha256 = _digest(
            receipt.get("source_eye_logical_sha256"),
            field=f"eye-gaze row {index} source logical digest",
        )
        validated_receipt = validate_gaze_convention_review_receipt(
            receipt,
            expected_run_path=run_path,
            expected_logical_sha256=logical_sha256,
        )
        frozen[recording_id] = {
            "analysis_zarr": str(archive),
            "run_name": run_name,
            "run_path": run_path,
            "channel_variant": channel_variant,
            "run_metadata_sha256": metadata_sha256,
            "source_eye_logical_sha256": logical_sha256,
            "convention_receipt_path": str(receipt_path),
            "convention_receipt_file_sha256": _sha256_file(receipt_path),
            "convention_receipt_sha256": _digest(
                validated_receipt.get("receipt_sha256"),
                field=f"eye-gaze row {index} convention receipt digest",
            ),
        }
    return frozen, {
        "path": str(path),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "row_count": len(rows),
    }


def _zarr_attrs(group: Path, *, field: str) -> tuple[dict[str, Any], str]:
    metadata = group / "zarr.json"
    if not metadata.is_file():
        _fail(f"{field} metadata is absent: {metadata}")
    try:
        payload = json.loads(metadata.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ComposableChaserCohortError(
            f"{field} metadata cannot be read: {metadata}"
        ) from exc
    attrs = _mapping(payload.get("attributes"), field=f"{field} attributes")
    return dict(attrs), _sha256_file(metadata)


def _verified_record(
    attrs: Mapping[str, Any],
    *,
    record_key: str,
    digest_key: str,
    field: str,
) -> tuple[dict[str, Any], str]:
    record = dict(_mapping(attrs.get(record_key), field=field))
    persisted = _digest(attrs.get(digest_key), field=f"{field} digest")
    observed = canonical_json_sha256(record)
    if observed != persisted:
        _fail(f"{field} persisted digest is stale.")
    return record, observed


def _task_digest(task: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in task.items() if key != "task_sha256"}
    )


def _run_children(parent: Path) -> tuple[str, ...]:
    if not parent.is_dir():
        return ()
    return tuple(
        sorted(
            child.name
            for child in parent.iterdir()
            if child.is_dir() and (child / "zarr.json").is_file()
        )
    )


def _one_prefixed_run(parent: Path, *, prefix: str, field: str) -> str:
    names = tuple(name for name in _run_children(parent) if name.startswith(prefix))
    if len(names) != 1:
        _fail(f"{field} requires exactly one {prefix!r} run; observed {names!r}.")
    return _exact_name(names[0], field=field)


def _recording_identity(root_attrs: Mapping[str, Any], archive: Path) -> str:
    recording_id = _text(root_attrs.get("recording_id"), field="recording identity")
    if archive.name != f"{recording_id}_analysis.zarr":
        _fail(f"Analysis Zarr basename disagrees with {recording_id!r}.")
    return recording_id


def _raw_h5_binding(
    root_attrs: Mapping[str, Any], *, recording_id: str
) -> dict[str, Any]:
    raw_h5 = (
        Path(_text(root_attrs.get("source_h5_path"), field="source raw-H5 path"))
        .expanduser()
        .resolve()
    )
    if not raw_h5.is_file():
        _fail(f"Raw H5 is absent for {recording_id!r}: {raw_h5}")
    stat = raw_h5.stat()
    expected_size = root_attrs.get("source_h5_size_bytes")
    if expected_size is not None and int(expected_size) != stat.st_size:
        _fail(f"Raw-H5 size differs from imported evidence for {recording_id!r}.")
    return {
        "path": str(raw_h5),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "imported_fingerprint": root_attrs.get("source_h5_fingerprint"),
        "imported_fingerprint_strategy": root_attrs.get(
            "source_h5_fingerprint_strategy"
        ),
    }


def _input_group_binding(
    archive: Path, relative_path: str, *, field: str
) -> dict[str, Any]:
    _attrs, metadata_sha = _zarr_attrs(archive / relative_path, field=field)
    return {
        "group_path": relative_path,
        "metadata_sha256": metadata_sha,
    }


def _resolve_geometry(archive: Path, *, recording_id: str) -> dict[str, Any]:
    parent_path = archive / "analysis" / "arena_geometry_selection"
    parent_attrs, _parent_sha = _zarr_attrs(
        parent_path, field="arena geometry selection parent"
    )
    latest = parent_attrs.get("latest")
    latest_complete = parent_attrs.get("latest_complete")
    if latest != latest_complete:
        _fail(f"Geometry selectors disagree for {recording_id!r}.")
    run_name = _exact_name(latest_complete, field="geometry selection run")
    run_attrs, run_metadata_sha = _zarr_attrs(
        parent_path / run_name, field="arena geometry selection"
    )
    selection_record, selection_sha = _verified_record(
        run_attrs,
        record_key="selection_record",
        digest_key="selection_record_sha256",
        field="arena geometry selection record",
    )
    if run_attrs.get("selection_id") != run_name:
        _fail(f"Geometry selection run identity mismatch for {recording_id!r}.")
    selected_candidate = _mapping(
        selection_record.get("selected_candidate"), field="selected geometry candidate"
    )
    arena_binding = _mapping(
        selected_candidate.get("arena_binding"), field="selected arena binding"
    )
    camera_serial = _text(
        arena_binding.get("camera_serial"), field="selected camera serial"
    )

    physical_path = archive / "analysis" / "calibration" / "coordinate_frames"
    physical_attrs, physical_metadata_sha = _zarr_attrs(
        physical_path, field="source-camera physical authority"
    )
    physical_record, physical_sha = _verified_record(
        physical_attrs,
        record_key="source_camera_physical_authority",
        digest_key="source_camera_physical_authority_sha256",
        field="source-camera physical authority",
    )
    if str(physical_record.get("camera_id")) != camera_serial:
        _fail(f"Physical authority camera mismatch for {recording_id!r}.")
    return {
        "selection_run": run_name,
        "selection_record_sha256": selection_sha,
        "selection_metadata_sha256": run_metadata_sha,
        "physical_authority_sha256": physical_sha,
        "physical_metadata_sha256": physical_metadata_sha,
        "camera_serial": camera_serial,
        "arena_id": arena_binding.get("arena_id"),
    }


def _resolve_motion_bouts(archive: Path, *, recording_id: str) -> dict[str, Any]:
    motion_parent = archive / "analysis" / "track_kinematics_runs" / "provider"
    bout_parent = archive / "analysis" / "swim_bout_runs"
    motion_names = set(_run_children(motion_parent))
    bout_names = set(_run_children(bout_parent))
    matches = [
        (motion, bouts)
        for motion, bouts in MOTION_BOUT_PAIRS
        if motion in motion_names and bouts in bout_names
    ]
    if not matches:
        _fail(f"No supported exact motion/bout pair exists for {recording_id!r}.")
    motion, bouts = matches[0]
    motion_run_path = f"analysis/track_kinematics_runs/provider/{motion}"
    motion_attrs, motion_metadata_sha = _zarr_attrs(
        archive / motion_run_path,
        field="provider motion input",
    )
    motion_manifest = _mapping(
        motion_attrs.get("provider_track_motion_manifest"),
        field="provider motion manifest",
    )
    if set(motion_manifest) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        _fail(f"Provider motion manifest envelope is not exact for {recording_id!r}.")
    payload = _mapping(
        motion_manifest.get("payload"), field="provider motion manifest payload"
    )
    motion_manifest_sha = _digest(
        motion_manifest.get("payload_digest"),
        field="provider motion manifest payload digest",
    )
    persisted_motion_manifest_sha = _digest(
        motion_attrs.get("provider_track_motion_manifest_sha256"),
        field="persisted provider motion manifest digest",
    )
    if (
        canonical_json_sha256(payload) != motion_manifest_sha
        or persisted_motion_manifest_sha != motion_manifest_sha
    ):
        _fail(
            f"Provider motion manifest persisted digest is stale for {recording_id!r}."
        )
    authority_envelope = _mapping(
        payload.get("source_authority"), field="provider motion source authority"
    )
    authority_record = _mapping(
        authority_envelope.get("record"), field="provider motion authority record"
    )
    authority_sha = _digest(
        authority_envelope.get("sha256"), field="provider motion authority digest"
    )
    if canonical_json_sha256(authority_record) != authority_sha:
        _fail(f"Provider motion authority digest is stale for {recording_id!r}.")
    body_source = _mapping(
        authority_record.get("body_frame_source"),
        field="provider motion body-frame source",
    )
    body_run_path = _text(
        body_source.get("run_path"), field="body-frame source run path"
    )
    body_run_name = _exact_name(
        body_run_path.removeprefix("analysis/body_frame_runs/"),
        field="body-frame source run name",
    )
    if body_run_path != f"analysis/body_frame_runs/{body_run_name}":
        _fail(f"Body-frame source path is not an exact run for {recording_id!r}.")
    body_manifest_sha = _digest(
        body_source.get("manifest_sha256"), field="body-frame source manifest digest"
    )
    body_attrs, body_metadata_sha = _zarr_attrs(
        archive / body_run_path,
        field="body-frame source input",
    )
    body_manifest = _mapping(
        body_attrs.get("run_manifest"), field="body-frame run manifest"
    )
    observed_body_manifest_sha = _digest(
        body_manifest.get("payload_digest"),
        field="body-frame run manifest payload digest",
    )
    body_payload = _mapping(
        body_manifest.get("payload"), field="body-frame run manifest payload"
    )
    if canonical_json_sha256(body_payload) != observed_body_manifest_sha:
        _fail(f"Body-frame run manifest digest is stale for {recording_id!r}.")
    if observed_body_manifest_sha != body_manifest_sha:
        _fail(f"Provider motion and body-frame run disagree for {recording_id!r}.")
    publication = _mapping(
        body_payload.get("publication"), field="body-frame publication disposition"
    )
    if (
        publication.get("completion_status") != "complete"
        or publication.get("stage_selector_eligible") is not False
    ):
        _fail(f"Body-frame source is not exact complete evidence for {recording_id!r}.")
    return {
        "selection_policy": "ordered_exact_compatible_pair_v1",
        "motion_run_path": motion_run_path,
        "motion_manifest_sha256": motion_manifest_sha,
        "motion_metadata_sha256": motion_metadata_sha,
        "swim_bout_run": bouts,
        "body_frame_run_path": body_run_path,
        "body_frame_run_name": body_run_name,
        "body_frame_manifest_sha256": body_manifest_sha,
        "body_frame_metadata_sha256": body_metadata_sha,
        "body_frame_resolution": "exact_provider_motion_authority_v1",
    }


def _output_groups() -> tuple[str, ...]:
    return (
        f"analysis/stimulus_runs/{SEMANTIC_STIMULUS_RUN}",
        f"analysis/stimulus_epoch_runs/{SEMANTIC_EPOCH_V1_RUN}",
        f"analysis/stimulus_epoch_runs/{SEMANTIC_EPOCH_V2_RUN}",
        f"analysis/protocol_semantic_chaser_selection_runs/{SEMANTIC_SELECTION_RUN}",
        f"analysis/stimulus_epoch_behavior_summary_runs/{EPOCH_BEHAVIOR_RUN}",
        f"analysis/chaser_body_alignment_by_distance_runs/{BODY_ALIGNMENT_RUN}",
        f"analysis/chaser_relative_frame_runs/{KEYPOINT_RELATIVE_RUN}",
        f"analysis/chaser_relative_frame_runs/{DETECTION_RELATIVE_RUN}",
        f"analysis/controller_chase_trial_runs/{SUCCESSOR_RUN}",
        f"analysis/generalized_chaser_bout_response_runs/{SUCCESSOR_RUN}",
        f"analysis/chaser_escape_freeze_runs/{SUCCESSOR_RUN}",
        f"analysis/chaser_radial_near_field_runs/{KEYPOINT_RADIAL_RUN}",
        f"analysis/chaser_radial_near_field_runs/{DETECTION_RADIAL_RUN}",
        (f"analysis/chaser_near_field_visits_runs/{KEYPOINT_NEAR_FIELD_VISIT_RUN}"),
        (f"analysis/chaser_near_field_visits_runs/{DETECTION_NEAR_FIELD_VISIT_RUN}"),
        f"analysis/chaser_spatial_occupancy_runs/{SPATIAL_OCCUPANCY_RUN}",
    )


def _plan_entry(
    row: Mapping[str, Any], *, task_index: int, operations_root: Path
) -> dict[str, Any]:
    archive = Path(_text(row.get("zarr_path"), field="registry Zarr path")).resolve()
    if not archive.is_dir():
        _fail(f"Analysis Zarr is absent: {archive}")
    root_attrs, _root_metadata_sha = _zarr_attrs(archive, field="analysis Zarr root")
    recording_id = _recording_identity(root_attrs, archive)
    if str(row.get("protocol_name") or "").casefold() != "goodbatbadbat":
        _fail(f"Registry protocol is not GoodBatBadBat for {recording_id!r}.")

    canonical_run = _one_prefixed_run(
        archive / "analysis" / "stimulus_runs",
        prefix="stimulus_canonical_v1_",
        field="canonical stimulus run",
    )
    geometry = _resolve_geometry(archive, recording_id=recording_id)
    motion_bouts = _resolve_motion_bouts(archive, recording_id=recording_id)

    keypoint_proxy_attrs, keypoint_proxy_metadata_sha = _zarr_attrs(
        archive
        / "analysis"
        / "chaser_input_provenance_proxy_runs"
        / KEYPOINT_PROXY_RUN,
        field="keypoint input-provenance proxy",
    )
    _keypoint_proxy_manifest, keypoint_proxy_sha = _verified_record(
        keypoint_proxy_attrs,
        record_key="chaser_input_provenance_proxy_manifest",
        digest_key="chaser_input_provenance_proxy_manifest_sha256",
        field="keypoint input-provenance proxy manifest",
    )
    detection_proxy_attrs, detection_proxy_metadata_sha = _zarr_attrs(
        archive
        / "analysis"
        / "chaser_input_provenance_proxy_runs"
        / DETECTION_PROXY_RUN,
        field="detection input-provenance proxy",
    )
    _detection_proxy_manifest, detection_proxy_sha = _verified_record(
        detection_proxy_attrs,
        record_key="chaser_input_provenance_proxy_manifest",
        digest_key="chaser_input_provenance_proxy_manifest_sha256",
        field="detection input-provenance proxy manifest",
    )

    input_groups = [
        _input_group_binding(
            archive,
            f"analysis/stimulus_runs/{canonical_run}",
            field="canonical stimulus input",
        ),
        {
            "group_path": (
                f"analysis/chaser_input_provenance_proxy_runs/{KEYPOINT_PROXY_RUN}"
            ),
            "metadata_sha256": keypoint_proxy_metadata_sha,
        },
        {
            "group_path": (
                f"analysis/chaser_input_provenance_proxy_runs/{DETECTION_PROXY_RUN}"
            ),
            "metadata_sha256": detection_proxy_metadata_sha,
        },
        {
            "group_path": motion_bouts["motion_run_path"],
            "metadata_sha256": motion_bouts["motion_metadata_sha256"],
        },
        _input_group_binding(
            archive,
            f"analysis/swim_bout_runs/{motion_bouts['swim_bout_run']}",
            field="swim-bout input",
        ),
        {
            "group_path": motion_bouts["body_frame_run_path"],
            "metadata_sha256": motion_bouts["body_frame_metadata_sha256"],
        },
        {
            "group_path": (
                f"analysis/arena_geometry_selection/{geometry['selection_run']}"
            ),
            "metadata_sha256": geometry["selection_metadata_sha256"],
        },
        {
            "group_path": "analysis/calibration/coordinate_frames",
            "metadata_sha256": geometry["physical_metadata_sha256"],
        },
    ]

    output_groups = _output_groups()
    existing_outputs = [
        group_path for group_path in output_groups if (archive / group_path).is_dir()
    ]
    plot_dir = (
        operations_root / SUCCESSOR_RUN / "cohort_plots" / recording_id
    ).resolve()
    detailed_receipt = plot_dir / "detailed" / f"{DETAILED_BUNDLE_NAME}_receipt.json"
    dashboard_receipt = plot_dir / f"{SUCCESSOR_RUN}_plot_receipt.json"
    spatial_receipt = (
        plot_dir
        / "spatial_occupancy"
        / f"{SPATIAL_OCCUPANCY_RUN}_spatial_occupancy_plot_receipt.json"
    )
    alignment_receipt = (
        plot_dir
        / "body_alignment_by_distance"
        / f"{BODY_ALIGNMENT_RECIPE_BUNDLE_NAME}_body_alignment_plot_receipt.json"
    )
    visit_plot_dir = plot_dir / "near_field_visits"
    near_field_visit_receipts = (
        visit_plot_dir
        / (
            f"{KEYPOINT_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME}_"
            "near_field_visit_plot_receipt.json"
        ),
        visit_plot_dir
        / (
            f"{DETECTION_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME}_"
            "near_field_visit_plot_receipt.json"
        ),
    )
    if len(existing_outputs) == len(output_groups):
        status = (
            "complete"
            if (
                detailed_receipt.is_file()
                and dashboard_receipt.is_file()
                and spatial_receipt.is_file()
                and alignment_receipt.is_file()
                and all(path.is_file() for path in near_field_visit_receipts)
            )
            else "plot_only"
        )
    elif existing_outputs:
        status = "resume"
    else:
        status = "ready"

    return json_attr_safe(
        {
            "task_index": task_index,
            "status": status,
            "dataset_id": row.get("dataset_id"),
            "recording_id": recording_id,
            "analysis_zarr": str(archive),
            "protocol_name": str(row.get("protocol_name")),
            "protocol_hash": row.get("protocol_hash"),
            "arena_id": row.get("arena_id"),
            "camera_id": row.get("camera_id"),
            "raw_h5": _raw_h5_binding(root_attrs, recording_id=recording_id),
            "canonical_stimulus_run": canonical_run,
            "geometry": geometry,
            "keypoint_proxy": {
                "run_name": KEYPOINT_PROXY_RUN,
                "manifest_sha256": keypoint_proxy_sha,
            },
            "detection_proxy": {
                "run_name": DETECTION_PROXY_RUN,
                "manifest_sha256": detection_proxy_sha,
            },
            "motion_and_bouts": motion_bouts,
            "input_group_bindings": input_groups,
            "output_run_names": {
                "semantic_stimulus": SEMANTIC_STIMULUS_RUN,
                "semantic_epoch_v1": SEMANTIC_EPOCH_V1_RUN,
                "semantic_epoch_v2": SEMANTIC_EPOCH_V2_RUN,
                "semantic_selection": SEMANTIC_SELECTION_RUN,
                "epoch_behavior": EPOCH_BEHAVIOR_RUN,
                "body_alignment_by_distance": BODY_ALIGNMENT_RUN,
                "body_alignment_plot_bundle": BODY_ALIGNMENT_RECIPE_BUNDLE_NAME,
                "keypoint_relative": KEYPOINT_RELATIVE_RUN,
                "detection_relative": DETECTION_RELATIVE_RUN,
                "successors": SUCCESSOR_RUN,
                "keypoint_radial": KEYPOINT_RADIAL_RUN,
                "detection_radial": DETECTION_RADIAL_RUN,
                "keypoint_near_field_visits": KEYPOINT_NEAR_FIELD_VISIT_RUN,
                "detection_near_field_visits": DETECTION_NEAR_FIELD_VISIT_RUN,
                "keypoint_near_field_visit_plot_bundle": (
                    KEYPOINT_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME
                ),
                "detection_near_field_visit_plot_bundle": (
                    DETECTION_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME
                ),
                "spatial_occupancy": SPATIAL_OCCUPANCY_RUN,
                "detailed_bundle": DETAILED_BUNDLE_NAME,
            },
            "output_group_paths": list(output_groups),
            "existing_output_group_paths": existing_outputs,
            "plot_output_dir": str(plot_dir),
            "near_field_visit_successor": {
                "provider_policy": NEAR_FIELD_VISIT_PROVIDER_POLICY,
                "receipt_policy": NEAR_FIELD_VISIT_RECEIPT_POLICY,
                "minimum_quality_sample_count": (
                    NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
                ),
            },
        }
    )


def _build_cohort_task(
    normalized_rows: Sequence[Mapping[str, Any]],
    *,
    operations_root: str | Path,
    source_registry_snapshot: Mapping[str, Any],
    successor_of_task_sha256: str | None = None,
) -> dict[str, Any]:
    if not normalized_rows:
        _fail("Cohort planning requires at least one recording row.")
    normalized_rows = list(normalized_rows)
    normalized_rows.sort(key=lambda row: str(row.get("zarr_path") or ""))
    operations = Path(operations_root).expanduser().resolve()
    entries = [
        _plan_entry(row, task_index=index, operations_root=operations)
        for index, row in enumerate(normalized_rows, start=1)
    ]
    recording_ids = [str(entry["recording_id"]) for entry in entries]
    if len(recording_ids) != len(set(recording_ids)):
        _fail("Frozen registry snapshot resolves duplicate recording identities.")
    status_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    runnable_indices = [
        int(entry["task_index"]) for entry in entries if entry["status"] != "complete"
    ]
    selection_policy: dict[str, Any] = {
        "protocol_name": "goodbatbadbat",
        "recording_order": "lexicographic_absolute_analysis_zarr_path_v1",
        "motion_bout_resolution": "ordered_exact_compatible_pair_v1",
        "body_frame_resolution": "exact_provider_motion_authority_v1",
        "near_field_visit_provider_policy": NEAR_FIELD_VISIT_PROVIDER_POLICY,
        "near_field_visit_receipt_policy": NEAR_FIELD_VISIT_RECEIPT_POLICY,
        "near_field_visit_minimum_quality_sample_count": (
            NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
        ),
        "selector_resolution_time": "planning_only",
        "execution_selector_resolution": False,
    }
    if successor_of_task_sha256 is not None:
        selection_policy["successor_of_task_sha256"] = _digest(
            successor_of_task_sha256, field="predecessor cohort task digest"
        )
    task = json_attr_safe(
        {
            "schema_id": TASK_SCHEMA_ID,
            "schema_version": TASK_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_registry_snapshot": dict(source_registry_snapshot),
            "selection_policy": selection_policy,
            "recording_count": len(entries),
            "status_counts": status_counts,
            "runnable_task_indices": runnable_indices,
            "operations_root": str(operations),
            "entries": entries,
            "safety": EXPECTED_SAFETY,
        }
    )
    task["task_sha256"] = _task_digest(task)
    return task


def plan_cohort_task(
    registry_snapshot: str | Path,
    *,
    operations_root: str | Path,
) -> dict[str, Any]:
    """Build one frozen task from a read-only registry JSON export."""

    snapshot = Path(registry_snapshot).expanduser().resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(f"Registry snapshot does not exist: {snapshot}")
    source_bytes = snapshot.read_bytes()
    rows = json.loads(source_bytes)
    if not isinstance(rows, list) or not rows:
        _fail("Registry snapshot must be a non-empty JSON row list.")
    normalized_rows = [_mapping(row, field="registry snapshot row") for row in rows]
    return _build_cohort_task(
        normalized_rows,
        operations_root=operations_root,
        source_registry_snapshot={
            "path": str(snapshot),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "row_count": len(normalized_rows),
        },
    )


def replan_cohort_task(
    source: str | Path | Mapping[str, Any],
    *,
    operations_root: str | Path,
) -> dict[str, Any]:
    """Freeze a new versioned cohort from one prior task's exact recording set."""

    previous = load_cohort_task(source)
    rows = []
    for raw_entry in previous["entries"]:
        entry = _mapping(raw_entry, field="predecessor cohort entry")
        rows.append(
            {
                "dataset_id": entry.get("dataset_id"),
                "recording_id": entry.get("recording_id"),
                "zarr_path": entry.get("analysis_zarr"),
                "protocol_name": entry.get("protocol_name"),
                "protocol_hash": entry.get("protocol_hash"),
                "arena_id": entry.get("arena_id"),
                "camera_id": entry.get("camera_id"),
            }
        )
    source_snapshot = dict(
        _mapping(
            previous.get("source_registry_snapshot"),
            field="predecessor registry snapshot binding",
        )
    )
    if int(source_snapshot.get("row_count", -1)) != len(rows):
        _fail("Predecessor registry snapshot row count is stale.")
    return _build_cohort_task(
        rows,
        operations_root=operations_root,
        source_registry_snapshot=source_snapshot,
        successor_of_task_sha256=str(previous["task_sha256"]),
    )


def _near_field_visit_configuration(
    entry: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    raw_outputs = entry.get("output_run_names")
    if raw_outputs is None and not required:
        return None
    outputs = _mapping(raw_outputs, field="output run names")
    keys = (
        "keypoint_near_field_visits",
        "detection_near_field_visits",
        "keypoint_near_field_visit_plot_bundle",
        "detection_near_field_visit_plot_bundle",
    )
    present = tuple(key in outputs for key in keys)
    if not any(present):
        if required:
            _fail("Cohort task lacks its required dual-provider near-field visits.")
        return None
    if not all(present):
        _fail("Near-field visit outputs must bind both providers and plot bundles.")
    policy = _mapping(
        entry.get("near_field_visit_successor"),
        field="near-field visit successor policy",
    )
    if policy.get("provider_policy") != NEAR_FIELD_VISIT_PROVIDER_POLICY:
        _fail("Near-field visit provider policy is unsupported.")
    if policy.get("receipt_policy") != NEAR_FIELD_VISIT_RECEIPT_POLICY:
        _fail("Near-field visit receipt policy is unsupported.")
    minimum = policy.get("minimum_quality_sample_count")
    if minimum != NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT:
        _fail("Near-field visit minimum quality sample count is unsupported.")
    output_paths = entry.get("output_group_paths")
    if not isinstance(output_paths, list) or any(
        type(path) is not str for path in output_paths
    ):
        _fail("Cohort task output group paths are invalid.")
    resolved: dict[str, Any] = {
        "provider_policy": NEAR_FIELD_VISIT_PROVIDER_POLICY,
        "receipt_policy": NEAR_FIELD_VISIT_RECEIPT_POLICY,
        "minimum_quality_sample_count": minimum,
    }
    for provider in ("keypoint", "detection"):
        run_name = _exact_name(
            outputs[f"{provider}_near_field_visits"],
            field=f"{provider} near-field visit run",
        )
        plot_bundle = _exact_name(
            outputs[f"{provider}_near_field_visit_plot_bundle"],
            field=f"{provider} near-field visit plot bundle",
        )
        resolved[f"{provider}_run"] = run_name
        resolved[f"{provider}_plot_bundle"] = plot_bundle
    if resolved["keypoint_run"] == resolved["detection_run"]:
        _fail("Near-field visit provider runs must remain distinct.")
    if resolved["keypoint_plot_bundle"] == resolved["detection_plot_bundle"]:
        _fail("Near-field visit provider plot bundles must remain distinct.")
    expected_visit_paths = {
        (f"analysis/chaser_near_field_visits_runs/{resolved['keypoint_run']}"),
        (f"analysis/chaser_near_field_visits_runs/{resolved['detection_run']}"),
    }
    observed_visit_paths = [
        path
        for path in output_paths
        if path.startswith("analysis/chaser_near_field_visits_runs/")
    ]
    if (
        len(observed_visit_paths) != len(expected_visit_paths)
        or set(observed_visit_paths) != expected_visit_paths
    ):
        _fail("Cohort task near-field visit output paths are not exact.")
    return resolved


def load_cohort_task(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load and strictly validate one frozen task document."""

    if isinstance(source, Mapping):
        task = dict(source)
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cohort task does not exist: {path}")
        task = dict(_mapping(json.loads(path.read_bytes()), field="cohort task"))
    if task.get("schema_id") != TASK_SCHEMA_ID or task.get("schema_version") not in {
        1,
        2,
        3,
        4,
        5,
        6,
        TASK_SCHEMA_VERSION,
    }:
        _fail("Cohort task schema is unsupported.")
    persisted = _digest(task.get("task_sha256"), field="cohort task digest")
    if _task_digest(task) != persisted:
        _fail("Cohort task digest is stale.")
    if task.get("safety") != EXPECTED_SAFETY:
        _fail("Cohort task does not retain the required safety state.")
    entries = task.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("Cohort task has no entries.")
    indices = [
        int(_mapping(entry, field="cohort entry").get("task_index"))
        for entry in entries
    ]
    if indices != list(range(1, len(entries) + 1)):
        _fail("Cohort task indices are not a contiguous one-based axis.")
    if int(task.get("recording_count", -1)) != len(entries):
        _fail("Cohort task recording count is stale.")
    entry_records = [_mapping(entry, field="cohort entry") for entry in entries]
    selection_policy = _mapping(
        task.get("selection_policy"), field="cohort selection policy"
    )
    schema_version = int(task["schema_version"])
    if schema_version >= 7:
        if (
            selection_policy.get("near_field_visit_provider_policy")
            != NEAR_FIELD_VISIT_PROVIDER_POLICY
            or selection_policy.get("near_field_visit_receipt_policy")
            != NEAR_FIELD_VISIT_RECEIPT_POLICY
            or selection_policy.get("near_field_visit_minimum_quality_sample_count")
            != NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
        ):
            _fail("Cohort task near-field visit selection policy is invalid.")
    for entry in entry_records:
        _near_field_visit_configuration(entry, required=schema_version >= 7)
    gaze_resolution = selection_policy.get("eye_gaze_resolution")
    gaze_entries = [entry.get("eye_gaze") is not None for entry in entry_records]
    if gaze_resolution is None:
        if any(gaze_entries):
            _fail("Cohort task has eye-gaze entries without a resolution policy.")
    else:
        if gaze_resolution != EYE_GAZE_BINDING_RESOLUTION or not all(gaze_entries):
            _fail("Eye-gaze task must bind every frozen cohort entry exactly.")
        source_binding = _mapping(
            selection_policy.get("eye_gaze_binding_source"),
            field="eye-gaze binding source",
        )
        if int(source_binding.get("row_count", -1)) != len(entry_records):
            _fail("Eye-gaze binding-source row count is stale.")
        _text(source_binding.get("path"), field="eye-gaze binding-source path")
        _digest(source_binding.get("sha256"), field="eye-gaze binding-source digest")
        for entry in entry_records:
            eye_gaze = _mapping(entry.get("eye_gaze"), field="eye-gaze binding")
            output_names = _mapping(
                entry.get("output_run_names"), field="output run names"
            )
            gaze_run = _exact_name(
                output_names.get("gaze_tracking"), field="gaze successor run"
            )
            expected_path = f"analysis/chaser_gaze_tracking_runs/{gaze_run}"
            output_paths = entry.get("output_group_paths")
            if not isinstance(output_paths, list) or expected_path not in output_paths:
                _fail("Eye-gaze task lacks its exact successor output path.")
            eye_run = _exact_name(eye_gaze.get("run_name"), field="eye-gaze input run")
            if eye_gaze.get("run_path") != f"analysis/eye_angle_runs/{eye_run}":
                _fail("Eye-gaze task input run path is inconsistent.")
    return task


def _entry(task: Mapping[str, Any], task_index: int) -> Mapping[str, Any]:
    entries = task["entries"]
    if task_index < 1 or task_index > len(entries):
        _fail(f"Task index {task_index} is outside 1..{len(entries)}.")
    entry = _mapping(entries[task_index - 1], field="cohort entry")
    if int(entry.get("task_index", -1)) != task_index:
        _fail("Cohort entry index does not match its row position.")
    return entry


def _revalidate_entry(entry: Mapping[str, Any]) -> None:
    archive = Path(_text(entry.get("analysis_zarr"), field="analysis Zarr"))
    if not archive.is_dir():
        _fail(f"Frozen analysis Zarr is absent: {archive}")
    recording_id = _text(entry.get("recording_id"), field="recording identity")
    if archive.name != f"{recording_id}_analysis.zarr":
        _fail("Frozen recording identity disagrees with the archive basename.")
    raw = _mapping(entry.get("raw_h5"), field="raw-H5 binding")
    raw_path = Path(_text(raw.get("path"), field="raw-H5 path"))
    if not raw_path.is_file():
        _fail("Frozen raw H5 is absent.")
    stat = raw_path.stat()
    if int(raw.get("size_bytes", -1)) != stat.st_size:
        _fail("Frozen raw-H5 size has changed.")
    if int(raw.get("mtime_ns", -1)) != stat.st_mtime_ns:
        _fail("Frozen raw-H5 modification time has changed.")
    bindings = entry.get("input_group_bindings")
    if not isinstance(bindings, list) or not bindings:
        _fail("Frozen input group bindings are absent.")
    for raw_binding in bindings:
        binding = _mapping(raw_binding, field="input group binding")
        group_path = _text(binding.get("group_path"), field="input group path")
        expected = _digest(
            binding.get("metadata_sha256"), field="input metadata digest"
        )
        metadata = archive / group_path / "zarr.json"
        if not metadata.is_file() or _sha256_file(metadata) != expected:
            _fail(f"Frozen input metadata changed: {group_path}")
    eye_gaze_raw = entry.get("eye_gaze")
    if eye_gaze_raw is not None:
        eye_gaze = _mapping(eye_gaze_raw, field="eye-gaze binding")
        if (
            Path(_text(eye_gaze.get("analysis_zarr"), field="eye-gaze analysis Zarr"))
            .expanduser()
            .resolve()
            != archive.resolve()
        ):
            _fail("Frozen eye-gaze archive differs from the cohort archive.")
        run_name = _exact_name(eye_gaze.get("run_name"), field="eye-gaze run name")
        run_path = _text(eye_gaze.get("run_path"), field="eye-gaze run path")
        if run_path != f"analysis/eye_angle_runs/{run_name}":
            _fail("Frozen eye-gaze run path does not match its exact run name.")
        if eye_gaze.get("channel_variant") not in {"raw", "smoothed"}:
            _fail("Frozen eye-gaze channel variant is unsupported.")
        receipt_path = (
            Path(
                _text(
                    eye_gaze.get("convention_receipt_path"),
                    field="eye-gaze convention receipt path",
                )
            )
            .expanduser()
            .resolve()
        )
        if not receipt_path.is_file():
            _fail("Frozen eye-gaze convention receipt is absent.")
        expected_file_sha256 = _digest(
            eye_gaze.get("convention_receipt_file_sha256"),
            field="eye-gaze convention receipt file digest",
        )
        if _sha256_file(receipt_path) != expected_file_sha256:
            _fail("Frozen eye-gaze convention receipt file has changed.")
        try:
            receipt = _mapping(
                json.loads(receipt_path.read_bytes()),
                field="eye-gaze convention receipt",
            )
        except json.JSONDecodeError as exc:
            raise ComposableChaserCohortError(
                "Frozen eye-gaze convention receipt is not strict JSON."
            ) from exc
        logical_sha256 = _digest(
            eye_gaze.get("source_eye_logical_sha256"),
            field="eye-gaze source logical digest",
        )
        validated = validate_gaze_convention_review_receipt(
            receipt,
            expected_run_path=run_path,
            expected_logical_sha256=logical_sha256,
        )
        if validated.get("receipt_sha256") != _digest(
            eye_gaze.get("convention_receipt_sha256"),
            field="eye-gaze convention receipt digest",
        ):
            _fail("Frozen eye-gaze convention receipt identity has changed.")


def successor_cohort_task(
    source: str | Path | Mapping[str, Any],
    *,
    eye_gaze_bindings: str | Path | None = None,
) -> dict[str, Any]:
    """Create a receipt-bound plotting/visit successor with optional gaze."""

    previous = load_cohort_task(source)
    previous_digest = previous["task_sha256"]
    frozen_eye_gaze: dict[str, dict[str, Any]] = {}
    eye_gaze_binding_source: dict[str, Any] | None = None
    if eye_gaze_bindings is not None:
        frozen_eye_gaze, eye_gaze_binding_source = _load_eye_gaze_bindings(
            eye_gaze_bindings
        )
        expected_recordings = {
            _text(
                _mapping(raw_entry, field="cohort entry").get("recording_id"),
                field="cohort recording identity",
            )
            for raw_entry in previous["entries"]
        }
        if set(frozen_eye_gaze) != expected_recordings:
            missing = sorted(expected_recordings.difference(frozen_eye_gaze))
            unexpected = sorted(set(frozen_eye_gaze).difference(expected_recordings))
            _fail(
                "Eye-gaze bindings must cover the exact frozen recording set; "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )
    entries = []
    for raw_entry in previous["entries"]:
        entry = dict(_mapping(raw_entry, field="cohort entry"))
        output_names = dict(
            _mapping(entry["output_run_names"], field="output run names")
        )
        output_names["epoch_behavior"] = EPOCH_BEHAVIOR_RUN
        output_names["body_alignment_by_distance"] = BODY_ALIGNMENT_RUN
        output_names["body_alignment_plot_bundle"] = BODY_ALIGNMENT_RECIPE_BUNDLE_NAME
        output_names.update(
            {
                "keypoint_near_field_visits": KEYPOINT_NEAR_FIELD_VISIT_RUN,
                "detection_near_field_visits": DETECTION_NEAR_FIELD_VISIT_RUN,
                "keypoint_near_field_visit_plot_bundle": (
                    KEYPOINT_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME
                ),
                "detection_near_field_visit_plot_bundle": (
                    DETECTION_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME
                ),
            }
        )
        archive = Path(_text(entry["analysis_zarr"], field="analysis Zarr"))
        recording_id = _text(entry["recording_id"], field="recording identity")
        if frozen_eye_gaze:
            eye_gaze = dict(frozen_eye_gaze[recording_id])
            source_archive = (
                Path(_text(eye_gaze["analysis_zarr"], field="eye-gaze source archive"))
                .expanduser()
                .resolve()
            )
            if source_archive != archive.resolve():
                _fail(f"Eye-gaze binding archive differs for {recording_id!r}.")
            entry["eye_gaze"] = eye_gaze
        elif "eye_gaze" in entry:
            eye_gaze = dict(_mapping(entry["eye_gaze"], field="eye-gaze binding"))
        else:
            eye_gaze = None
        existing_spatial_path = (
            f"analysis/chaser_spatial_occupancy_runs/{SPATIAL_OCCUPANCY_RUN}"
        )
        if _existing_complete_output(
            archive,
            existing_spatial_path,
            recording_id,
        ):
            spatial_occupancy_run = SPATIAL_OCCUPANCY_RUN
            spatial_occupancy_mode = "reuse_existing_exact_complete_v1"
        else:
            spatial_occupancy_run = SPATIAL_OCCUPANCY_RECEIPT_BOUND_RUN
            spatial_occupancy_mode = "materialize_missing_receipt_bound_v2"
        output_names.update(
            {
                "spatial_occupancy": spatial_occupancy_run,
                "spatial_plot_bundle": SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME,
                "dashboard_bundle": DASHBOARD_RECIPE_BUNDLE_NAME,
                "detailed_bundle": DETAILED_RECIPE_BUNDLE_NAME,
            }
        )
        if eye_gaze is not None:
            output_names["gaze_tracking"] = SUCCESSOR_RUN
        spatial_path = f"analysis/chaser_spatial_occupancy_runs/{spatial_occupancy_run}"
        output_paths = [
            str(path)
            for path in entry.get("output_group_paths", [])
            if not str(path).startswith(
                (
                    "analysis/chaser_spatial_occupancy_runs/",
                    "analysis/chaser_near_field_visits_runs/",
                )
            )
        ]
        output_paths.append(spatial_path)
        output_paths.extend(
            (
                (
                    "analysis/chaser_near_field_visits_runs/"
                    f"{KEYPOINT_NEAR_FIELD_VISIT_RUN}"
                ),
                (
                    "analysis/chaser_near_field_visits_runs/"
                    f"{DETECTION_NEAR_FIELD_VISIT_RUN}"
                ),
            )
        )
        if eye_gaze is not None:
            gaze_path = (
                f"analysis/chaser_gaze_tracking_runs/{output_names['gaze_tracking']}"
            )
            if gaze_path not in output_paths:
                output_paths.append(gaze_path)
        alignment_path = (
            f"analysis/chaser_body_alignment_by_distance_runs/{BODY_ALIGNMENT_RUN}"
        )
        if alignment_path not in output_paths:
            output_paths.append(alignment_path)
        existing_paths = [
            path for path in output_paths if (archive / path / "zarr.json").is_file()
        ]
        plot_dir = Path(_text(entry["plot_output_dir"], field="plot output dir"))
        expected_plot_receipts = (
            plot_dir / f"{DASHBOARD_RECIPE_BUNDLE_NAME}_plot_receipt.json",
            plot_dir / "detailed" / f"{DETAILED_RECIPE_BUNDLE_NAME}_receipt.json",
            plot_dir
            / "spatial_occupancy"
            / (
                f"{SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME}_"
                "spatial_occupancy_plot_receipt.json"
            ),
            plot_dir
            / "body_alignment_by_distance"
            / (f"{BODY_ALIGNMENT_RECIPE_BUNDLE_NAME}_body_alignment_plot_receipt.json"),
            plot_dir
            / "near_field_visits"
            / (
                f"{KEYPOINT_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME}_"
                "near_field_visit_plot_receipt.json"
            ),
            plot_dir
            / "near_field_visits"
            / (
                f"{DETECTION_NEAR_FIELD_VISIT_PLOT_BUNDLE_NAME}_"
                "near_field_visit_plot_receipt.json"
            ),
        )
        if len(existing_paths) == len(output_paths):
            status = (
                "complete"
                if all(path.is_file() for path in expected_plot_receipts)
                else "plot_only"
            )
        elif existing_paths:
            status = "resume"
        else:
            status = "ready"
        if eye_gaze is not None and status == "complete":
            # The deployment commit is not known while planning, so an exact
            # gaze child and v8 projection receipt must still be sealed by
            # run-one even when every immutable science/plot output exists.
            status = "plot_only"
        entry.update(
            {
                "status": status,
                "output_run_names": output_names,
                "output_group_paths": output_paths,
                "existing_output_group_paths": existing_paths,
                "relative_frame_validation": {
                    "mode": RELATIVE_FRAME_VALIDATION_MODE,
                    "receipt_directory": "source_validation_receipts",
                },
                "spatial_occupancy_successor": {
                    "mode": spatial_occupancy_mode,
                    "exact_run_name": spatial_occupancy_run,
                    "plot_bundle": SPATIAL_OCCUPANCY_RECIPE_BUNDLE_NAME,
                },
                "near_field_visit_successor": {
                    "provider_policy": NEAR_FIELD_VISIT_PROVIDER_POLICY,
                    "receipt_policy": NEAR_FIELD_VISIT_RECEIPT_POLICY,
                    "minimum_quality_sample_count": (
                        NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
                    ),
                },
                "successor_of_entry_task_sha256": previous_digest,
            }
        )
        if eye_gaze is not None:
            input_bindings = []
            for value in entry["input_group_bindings"]:
                binding = dict(_mapping(value, field="input group binding"))
                if str(binding.get("group_path")).startswith(
                    "analysis/eye_angle_runs/"
                ):
                    continue
                input_bindings.append(binding)
            input_bindings.append(
                {
                    "group_path": eye_gaze["run_path"],
                    "metadata_sha256": eye_gaze["run_metadata_sha256"],
                }
            )
            entry["input_group_bindings"] = input_bindings
        entries.append(json_attr_safe(entry))

    status_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    task = json_attr_safe(
        {
            **{
                key: value
                for key, value in previous.items()
                if key
                not in {
                    "task_sha256",
                    "created_at_utc",
                    "schema_version",
                    "entries",
                    "status_counts",
                    "runnable_task_indices",
                }
            },
            "schema_version": TASK_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_policy": {
                **dict(previous["selection_policy"]),
                "successor_of_task_sha256": previous_digest,
                "relative_frame_validation": RELATIVE_FRAME_VALIDATION_MODE,
                "plot_recipe_provenance": "self_contained_exact_parameters_v5",
                "near_field_visit_provider_policy": (NEAR_FIELD_VISIT_PROVIDER_POLICY),
                "near_field_visit_receipt_policy": NEAR_FIELD_VISIT_RECEIPT_POLICY,
                "near_field_visit_minimum_quality_sample_count": (
                    NEAR_FIELD_VISIT_MINIMUM_QUALITY_SAMPLE_COUNT
                ),
                **(
                    {
                        "eye_gaze_resolution": EYE_GAZE_BINDING_RESOLUTION,
                        "eye_gaze_binding_source": eye_gaze_binding_source,
                    }
                    if eye_gaze_binding_source is not None
                    else {}
                ),
            },
            "status_counts": status_counts,
            "runnable_task_indices": [
                int(entry["task_index"])
                for entry in entries
                if entry["status"] != "complete"
            ],
            "entries": entries,
        }
    )
    task["task_sha256"] = _task_digest(task)
    return task


def _existing_complete_output(
    archive: Path, group_path: str, recording_id: str
) -> bool:
    target = archive / group_path
    if not target.exists():
        return False
    attrs, _metadata_sha = _zarr_attrs(target, field=f"existing output {group_path}")
    if attrs.get("palette_run_completion_status") != "complete":
        _fail(f"Existing output is not complete: {group_path}")
    if attrs.get("stage_selector_eligible") is not False:
        _fail(f"Existing output is not explicitly selector-ineligible: {group_path}")
    if attrs.get("production_authority") not in {None, False}:
        _fail(f"Existing output claims production authority: {group_path}")
    if attrs.get("registry_update") not in {None, False}:
        _fail(f"Existing output claims a registry update: {group_path}")
    selection = attrs.get("selection")
    if selection is not None and selection != "none":
        _fail(f"Existing output claims selector activation: {group_path}")
    expected_run_name = Path(group_path).name
    if attrs.get("palette_run_name") not in {None, expected_run_name}:
        _fail(f"Existing output run identity mismatch: {group_path}")
    observed_recording = attrs.get("recording_id")
    if observed_recording is not None and observed_recording != recording_id:
        _fail(f"Existing output recording identity mismatch: {group_path}")
    for key, value in attrs.items():
        if not key.endswith("_manifest") or not isinstance(value, Mapping):
            continue
        manifest_recording = value.get("recording_id")
        if manifest_recording is not None and manifest_recording != recording_id:
            _fail(f"Existing output manifest recording mismatch: {group_path}")
        manifest_run = value.get("run_name")
        if manifest_run is not None and manifest_run != expected_run_name:
            _fail(f"Existing output manifest run mismatch: {group_path}")
        digest_key = f"{key}_sha256"
        if digest_key in attrs:
            persisted = _digest(attrs[digest_key], field=f"existing {key} digest")
            if canonical_json_sha256(value) != persisted:
                _fail(f"Existing output manifest digest is stale: {group_path}")
    return True


def _existing_near_field_visit_output(
    archive: Path,
    *,
    recording_id: str,
    visit_run: str,
    relative_frame_run: str,
    semantic_selection_run: str,
    radial_near_field_run: str,
    minimum_quality_sample_count: int,
) -> bool:
    visit_path = f"analysis/chaser_near_field_visits_runs/{visit_run}"
    if not _existing_complete_output(archive, visit_path, recording_id):
        return False
    visit_attrs, _metadata_sha = _zarr_attrs(
        archive / visit_path,
        field=f"existing near-field visit output {visit_path}",
    )
    publication, _publication_sha = _verified_record(
        visit_attrs,
        record_key="composable_chaser_successor_manifest",
        digest_key="composable_chaser_successor_manifest_sha256",
        field="near-field visit publication manifest",
    )
    publication_payload_sha = _digest(
        publication.get("payload_digest"),
        field="near-field visit publication payload digest",
    )
    if (
        canonical_json_sha256(
            {
                key: value
                for key, value in publication.items()
                if key != "payload_digest"
            }
        )
        != publication_payload_sha
        or publication.get("successor_kind") != "chaser_near_field_visits"
        or publication.get("run_name") != visit_run
        or publication.get("run_path") != visit_path
        or publication.get("recording_id") != recording_id
        or publication.get("selector_eligible") is not False
        or publication.get("selection") != "none"
        or publication.get("production_authority") is not False
        or publication.get("registry_update") is not False
    ):
        _fail("Existing near-field visit publication identity is incompatible.")
    scientific = _mapping(
        publication.get("scientific_manifest"),
        field="near-field visit scientific manifest",
    )
    scientific_digest = _digest(
        scientific.get("payload_digest"),
        field="near-field visit scientific payload digest",
    )
    scientific_schema = _mapping(
        scientific.get("scientific_schema"),
        field="near-field visit scientific schema",
    )
    if (
        canonical_json_sha256(
            {key: value for key, value in scientific.items() if key != "payload_digest"}
        )
        != scientific_digest
        or publication.get("scientific_payload_sha256") != scientific_digest
        or scientific.get("recording_id") != recording_id
        or scientific_schema
        != {
            "schema_id": NEAR_FIELD_VISIT_SCIENTIFIC_SCHEMA_ID,
            "schema_version": NEAR_FIELD_VISIT_SCIENTIFIC_SCHEMA_VERSION,
        }
        or scientific.get("selector_eligible") is not False
        or scientific.get("selection") != "none"
        or scientific.get("production_authority") is not False
        or scientific.get("registry_update") is not False
    ):
        _fail("Existing near-field visit scientific identity is stale.")

    relative_path = f"analysis/chaser_relative_frame_runs/{relative_frame_run}"
    relative_attrs, _relative_metadata_sha = _zarr_attrs(
        archive / relative_path,
        field="near-field visit relative-frame source",
    )
    _relative_manifest, relative_manifest_sha = _verified_record(
        relative_attrs,
        record_key="chaser_relative_frame_manifest",
        digest_key="chaser_relative_frame_manifest_sha256",
        field="near-field visit relative-frame source manifest",
    )
    semantic_path = (
        f"analysis/protocol_semantic_chaser_selection_runs/{semantic_selection_run}"
    )
    semantic_attrs, _semantic_metadata_sha = _zarr_attrs(
        archive / semantic_path,
        field="near-field visit semantic-selection source",
    )
    _semantic_manifest, semantic_manifest_sha = _verified_record(
        semantic_attrs,
        record_key="protocol_semantic_chaser_selection_manifest",
        digest_key="protocol_semantic_chaser_selection_manifest_sha256",
        field="near-field visit semantic-selection source manifest",
    )
    radial_path = f"analysis/chaser_radial_near_field_runs/{radial_near_field_run}"
    radial_attrs, _radial_metadata_sha = _zarr_attrs(
        archive / radial_path,
        field="near-field visit radial source",
    )
    radial_publication, radial_manifest_sha = _verified_record(
        radial_attrs,
        record_key="composable_chaser_successor_manifest",
        digest_key="composable_chaser_successor_manifest_sha256",
        field="near-field visit radial source manifest",
    )
    radial_publication_payload_sha = _digest(
        radial_publication.get("payload_digest"),
        field="near-field visit radial publication payload digest",
    )
    if (
        canonical_json_sha256(
            {
                key: value
                for key, value in radial_publication.items()
                if key != "payload_digest"
            }
        )
        != radial_publication_payload_sha
    ):
        _fail("Near-field visit radial publication source digest is stale.")
    radial_scientific = _mapping(
        radial_publication.get("scientific_manifest"),
        field="near-field visit radial scientific manifest",
    )
    radial_payload_sha = _digest(
        radial_publication.get("scientific_payload_sha256"),
        field="near-field visit radial scientific payload digest",
    )
    radial_scientific_sha = _digest(
        radial_scientific.get("payload_digest"),
        field="near-field visit radial embedded scientific payload digest",
    )
    if (
        canonical_json_sha256(
            {
                key: value
                for key, value in radial_scientific.items()
                if key != "payload_digest"
            }
        )
        != radial_scientific_sha
        or radial_scientific_sha != radial_payload_sha
    ):
        _fail("Near-field visit radial scientific source digest is stale.")
    radial_sources = _mapping(
        radial_scientific.get("sources"),
        field="near-field visit radial scientific sources",
    )
    if radial_sources.get("relative_frame") != {
        "run_path": relative_path,
        "manifest_sha256": relative_manifest_sha,
    }:
        _fail("Near-field visit radial source uses another relative frame.")
    if radial_sources.get("protocol_semantic_selection") != {
        "run_path": semantic_path,
        "manifest_sha256": semantic_manifest_sha,
    }:
        _fail("Near-field visit radial source uses another semantic selection.")
    sources = _mapping(
        scientific.get("sources"), field="near-field visit scientific sources"
    )
    if sources.get("relative_frame") != {
        "run_path": relative_path,
        "manifest_sha256": relative_manifest_sha,
    }:
        _fail("Existing near-field visit relative-frame binding is incompatible.")
    if sources.get("protocol_semantic_selection") != {
        "run_path": semantic_path,
        "manifest_sha256": semantic_manifest_sha,
    }:
        _fail("Existing near-field visit semantic-selection binding is incompatible.")
    if sources.get("radial_near_field") != {
        "run_path": radial_path,
        "manifest_sha256": radial_manifest_sha,
        "scientific_payload_sha256": radial_payload_sha,
    }:
        _fail("Existing near-field visit radial binding is incompatible.")
    if scientific.get("position_provider") != radial_scientific.get(
        "position_provider"
    ) or sources.get("fish_position") != radial_sources.get("fish_position"):
        _fail("Existing near-field visit position-provider binding is incompatible.")
    config = _mapping(scientific.get("config"), field="near-field visit config")
    if config.get("minimum_quality_sample_count") != minimum_quality_sample_count:
        _fail("Existing near-field visit quality policy is incompatible.")
    return True


def _validated_plot_receipt(
    path: Path,
    *,
    recording_id: str,
    require_self_contained_recipe: bool = False,
    expected_plot_recipe_id: str | None = None,
    expected_source_run_path: str | None = None,
) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        _fail(f"Plot receipt is not a file: {path}")
    try:
        receipt = dict(_mapping(json.loads(path.read_bytes()), field="plot receipt"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComposableChaserCohortError(
            f"Plot receipt cannot be read: {path}"
        ) from exc
    persisted = _digest(
        receipt.pop("payload_sha256", None), field="plot receipt digest"
    )
    if canonical_json_sha256(receipt) != persisted:
        _fail(f"Plot receipt digest is stale: {path}")
    if receipt.get("recording_id") != recording_id:
        _fail(f"Plot receipt recording identity mismatch: {path}")
    if expected_source_run_path is not None:
        source = _mapping(receipt.get("source_binding"), field="plot source binding")
        if source.get("run_path") != expected_source_run_path:
            _fail(f"Plot receipt source run mismatch: {path}")
        if receipt.get("run_name") != Path(expected_source_run_path).name:
            _fail(f"Plot receipt source run-name mismatch: {path}")
    if (
        expected_plot_recipe_id is not None
        and receipt.get("plot_recipe_id") != expected_plot_recipe_id
    ):
        _fail(f"Plot receipt recipe identity mismatch: {path}")
    if require_self_contained_recipe:
        parameters = receipt.get("plot_parameters")
        if (
            not isinstance(receipt.get("plot_recipe_id"), str)
            or not receipt["plot_recipe_id"]
            or not isinstance(parameters, Mapping)
            or canonical_json_sha256(parameters)
            != _digest(
                receipt.get("plot_parameters_sha256"),
                field="plot parameters digest",
            )
        ):
            _fail(f"Plot receipt lacks its exact self-contained recipe: {path}")
    if (
        receipt.get("selector_eligible") is not False
        or receipt.get("production_authority") is not False
        or receipt.get("registry_update") is not False
    ):
        _fail(f"Plot receipt safety state is invalid: {path}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        _fail(f"Plot receipt has no outputs: {path}")
    for raw_output in outputs:
        output = _mapping(raw_output, field="plot output binding")
        output_path = Path(_text(output.get("path"), field="plot output path"))
        expected_sha = _digest(output.get("sha256"), field="plot output digest")
        if not output_path.is_file() or _sha256_file(output_path) != expected_sha:
            _fail(f"Plot output differs from its receipt: {output_path}")
    return True


def _repo_commit(repo: Path, expected_commit: str) -> str:
    if not repo.is_dir() or not (repo / "scripts" / "py").is_file():
        _fail(f"Palette repository is invalid: {repo}")
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        _fail("Expected Palette commit must be one full lowercase Git SHA.")
    observed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if observed != expected_commit:
        _fail("Palette repository HEAD differs from the frozen execution commit.")
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if dirty:
        _fail("Palette execution repository must be clean.")
    return observed


def _invoke(
    *,
    stage: str,
    command: Sequence[str],
    log_dir: Path,
    apply: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage,
        "command": list(command),
        "mode": "apply" if apply else "planned_no_write",
    }
    if not apply:
        return record
    completed = subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
    )
    stdout_path = log_dir / f"{stage}.stdout.log"
    stderr_path = log_dir / f"{stage}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record.update(
        {
            "returncode": completed.returncode,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cohort stage {stage!r} failed with code {completed.returncode}; "
            f"see {stderr_path}."
        )
    return record


def _stage_command(
    py: Path,
    module: str,
    *arguments: object,
) -> list[str]:
    return [str(py), "-m", module, *(str(argument) for argument in arguments)]


def run_one(
    task_source: str | Path | Mapping[str, Any],
    *,
    task_index: int,
    palette_repo: str | Path,
    palette_commit: str,
    scratch_root: str | Path,
    receipt_root: str | Path,
    copy_backend: str = "rsync",
    apply: bool = False,
) -> dict[str, Any]:
    """Validate and optionally execute one frozen recording task."""

    if copy_backend not in {"python", "rsync"}:
        _fail("Copy backend must be python or rsync.")
    task = load_cohort_task(task_source)
    entry = _entry(task, task_index)
    visit_configuration = _near_field_visit_configuration(
        entry,
        required=int(task["schema_version"]) >= 7,
    )
    _revalidate_entry(entry)
    repo = Path(palette_repo).expanduser().resolve()
    commit = _repo_commit(repo, palette_commit)
    py = repo / "scripts" / "py"
    archive = Path(str(entry["analysis_zarr"]))
    recording_id = str(entry["recording_id"])
    scratch = Path(scratch_root).expanduser().resolve() / f"task_{task_index:04d}"
    receipt_dir = Path(receipt_root).expanduser().resolve() / f"task_{task_index:04d}"
    if apply:
        scratch.mkdir(parents=True, exist_ok=True)
        receipt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(str(entry["plot_output_dir"]))
    detailed_dir = plot_dir / "detailed"
    spatial_plot_dir = plot_dir / "spatial_occupancy"
    alignment_plot_dir = plot_dir / "body_alignment_by_distance"
    visit_plot_dir = plot_dir / "near_field_visits"
    if apply:
        plot_dir.mkdir(parents=True, exist_ok=True)
        detailed_dir.mkdir(parents=True, exist_ok=True)
        spatial_plot_dir.mkdir(parents=True, exist_ok=True)
        alignment_plot_dir.mkdir(parents=True, exist_ok=True)
        if visit_configuration is not None:
            visit_plot_dir.mkdir(parents=True, exist_ok=True)

    outputs = _mapping(entry["output_run_names"], field="output run names")
    spatial_occupancy_run = _exact_name(
        outputs.get("spatial_occupancy", SPATIAL_OCCUPANCY_RUN),
        field="spatial occupancy run",
    )
    spatial_plot_bundle = _exact_name(
        outputs.get("spatial_plot_bundle", spatial_occupancy_run),
        field="spatial occupancy plot bundle",
    )
    dashboard_bundle = _exact_name(
        outputs.get("dashboard_bundle", outputs["successors"]),
        field="dashboard bundle",
    )
    alignment_plot_bundle = _exact_name(
        outputs.get(
            "body_alignment_plot_bundle",
            outputs.get("body_alignment_by_distance", BODY_ALIGNMENT_RUN),
        ),
        field="body-alignment plot bundle",
    )
    relative_validation_raw = entry.get("relative_frame_validation")
    if relative_validation_raw is None:
        receipt_bound_relative = False
        relative_receipt_dir = None
    else:
        relative_validation = _mapping(
            relative_validation_raw, field="relative-frame validation policy"
        )
        if relative_validation.get("mode") != RELATIVE_FRAME_VALIDATION_MODE:
            _fail("Relative-frame validation mode is unsupported.")
        receipt_directory = _exact_name(
            relative_validation.get("receipt_directory"),
            field="relative-frame receipt directory",
        )
        receipt_bound_relative = True
        relative_receipt_dir = plot_dir / receipt_directory / commit
        if apply:
            relative_receipt_dir.mkdir(parents=True, exist_ok=True)
    raw_h5 = _mapping(entry["raw_h5"], field="raw-H5 binding")
    geometry = _mapping(entry["geometry"], field="geometry binding")
    keypoint_proxy = _mapping(entry["keypoint_proxy"], field="keypoint proxy")
    detection_proxy = _mapping(entry["detection_proxy"], field="detection proxy")
    motion_bouts = _mapping(entry["motion_and_bouts"], field="motion and bout binding")
    eye_gaze_raw = entry.get("eye_gaze")
    eye_gaze = (
        _mapping(eye_gaze_raw, field="eye-gaze binding")
        if eye_gaze_raw is not None
        else None
    )
    if (eye_gaze is None) != ("gaze_tracking" not in outputs):
        _fail("Eye-gaze input and gaze-successor output bindings must appear together.")
    stages: list[dict[str, Any]] = []

    def execute_if_missing(stage: str, group_path: str, command: Sequence[str]) -> None:
        if _existing_complete_output(archive, group_path, recording_id):
            stages.append({"stage": stage, "mode": "reused_exact_complete_output"})
            return
        stages.append(
            _invoke(stage=stage, command=command, log_dir=receipt_dir, apply=apply)
        )
        if apply and not _existing_complete_output(archive, group_path, recording_id):
            _fail(f"Stage {stage!r} did not produce its exact complete output.")

    execute_if_missing(
        "semantic_stimulus",
        f"analysis/stimulus_runs/{outputs['semantic_stimulus']}",
        _stage_command(
            py,
            "fisheye.utils.materialize_historical_protocol_semantic_stimulus_successor",
            "--analysis-zarr",
            archive,
            "--source-run-name",
            entry["canonical_stimulus_run"],
            "--run-name",
            outputs["semantic_stimulus"],
            "--raw-h5",
            raw_h5["path"],
            "--scratch-root",
            scratch / "semantic_stimulus",
            "--copy-backend",
            copy_backend,
            "--apply",
        ),
    )
    execute_if_missing(
        "semantic_epoch_v1",
        f"analysis/stimulus_epoch_runs/{outputs['semantic_epoch_v1']}",
        _stage_command(
            py,
            "fisheye.analysis.stimulus_epoch_runs",
            archive,
            "--run-name",
            outputs["semantic_epoch_v1"],
            "--stimulus-run",
            outputs["semantic_stimulus"],
            "--protocol-profile",
            repo / "src/fisheye/analysis/profiles/goodcopbadcop_source_v1.yaml",
            "--selector-ineligible",
            "--apply",
            "--json",
        ),
    )
    execute_if_missing(
        "semantic_epoch_v2",
        f"analysis/stimulus_epoch_runs/{outputs['semantic_epoch_v2']}",
        _stage_command(
            py,
            "fisheye.analysis_workflows.materializers.stimulus_epochs",
            archive,
            "--source-run",
            outputs["semantic_epoch_v1"],
            "--run-name",
            outputs["semantic_epoch_v2"],
            "--scratch-root",
            scratch / "semantic_epoch_v2",
            "--allow-selector-ineligible-source",
            "--copy-backend",
            copy_backend,
            "--apply",
            "--json",
        ),
    )

    epoch_group = (
        archive / "analysis" / "stimulus_epoch_runs" / outputs["semantic_epoch_v2"]
    )
    if apply or epoch_group.is_dir():
        epoch_attrs, _epoch_metadata_sha = _zarr_attrs(
            epoch_group, field="semantic epoch v2 output"
        )
        epoch_manifest = _mapping(
            epoch_attrs.get("stimulus_epoch_run_manifest"),
            field="semantic epoch run manifest",
        )
        epoch_manifest_sha = canonical_json_sha256(epoch_manifest)
    else:
        epoch_manifest_sha = "<created-by-semantic-epoch-v2-stage>"
    execute_if_missing(
        "semantic_selection",
        (
            "analysis/protocol_semantic_chaser_selection_runs/"
            f"{outputs['semantic_selection']}"
        ),
        _stage_command(
            py,
            "fisheye.utils.materialize_protocol_semantic_chaser_selection",
            archive,
            "--source-epoch-selection-run",
            outputs["semantic_epoch_v2"],
            "--expected-source-epoch-manifest-sha256",
            epoch_manifest_sha,
            "--run-name",
            outputs["semantic_selection"],
            "--chaser-pre-window-id",
            0,
            "--chaser-training-window-id",
            1,
            "--chaser-post-window-id",
            2,
            "--expected-recording-id",
            recording_id,
            "--scratch-root",
            scratch / "semantic_selection",
            "--copy-backend",
            copy_backend,
            "--apply",
        ),
    )

    if "epoch_behavior" in outputs:
        execute_if_missing(
            "epoch_behavior",
            (
                "analysis/stimulus_epoch_behavior_summary_runs/"
                f"{outputs['epoch_behavior']}"
            ),
            _stage_command(
                py,
                (
                    "fisheye.analysis_workflows.materializers."
                    "provider_epoch_behavior_summary"
                ),
                archive,
                "--run-name",
                outputs["epoch_behavior"],
                "--epoch-run",
                outputs["semantic_epoch_v2"],
                "--protocol-semantic-selection-run",
                outputs["semantic_selection"],
                "--motion-run",
                motion_bouts["motion_run_path"],
                "--swim-bout-run",
                motion_bouts["swim_bout_run"],
                "--track-id",
                0,
                "--speed-level",
                "filtered",
                "--scratch-root",
                scratch / "epoch_behavior",
                "--copy-backend",
                copy_backend,
                "--apply",
                "--json",
            ),
        )

    for provider, proxy, relative_key in (
        ("keypoint", keypoint_proxy, "keypoint_relative"),
        ("detection", detection_proxy, "detection_relative"),
    ):
        relative_name = outputs[relative_key]
        body_frame_flags: tuple[object, ...] = ()
        if provider == "keypoint":
            body_frame_flags = (
                "--body-frame-run",
                motion_bouts["body_frame_run_name"],
            )
        execute_if_missing(
            f"{provider}_relative_frame",
            f"analysis/chaser_relative_frame_runs/{relative_name}",
            _stage_command(
                py,
                "fisheye.utils.materialize_chaser_proxy_relative_frame",
                archive,
                "--proxy-run-name",
                proxy["run_name"],
                "--output-run-name",
                relative_name,
                "--scratch-root",
                scratch / f"{provider}_relative",
                "--analysis-profile",
                repo / "src/fisheye/analysis/profiles/chaser_behavior_full_v3.yaml",
                "--expected-recording-id",
                recording_id,
                "--expected-proxy-manifest-sha256",
                proxy["manifest_sha256"],
                *body_frame_flags,
                "--copy-backend",
                copy_backend,
                "--apply",
                "--json",
            ),
        )

    relative_receipts: dict[str, Path] = {}
    if receipt_bound_relative:
        assert relative_receipt_dir is not None
        for provider, relative_key in (
            ("keypoint", "keypoint_relative"),
            ("detection", "detection_relative"),
        ):
            relative_name = _exact_name(
                outputs[relative_key], field=f"{provider} relative-frame run"
            )
            receipt_path = (
                relative_receipt_dir / f"{relative_name}.validation_receipt.json"
            )
            relative_receipts[provider] = receipt_path
            stages.append(
                _invoke(
                    stage=f"{provider}_relative_frame_validation_receipt",
                    command=_stage_command(
                        py,
                        "fisheye.utils.seal_chaser_relative_frame_validation_receipt",
                        archive,
                        "--run-name",
                        relative_name,
                        "--palette-commit",
                        commit,
                        "--output-json",
                        receipt_path,
                        "--expected-recording-id",
                        recording_id,
                    ),
                    log_dir=receipt_dir,
                    apply=apply,
                )
            )

    successor_groups = (
        f"analysis/controller_chase_trial_runs/{outputs['successors']}",
        f"analysis/generalized_chaser_bout_response_runs/{outputs['successors']}",
        f"analysis/chaser_escape_freeze_runs/{outputs['successors']}",
    )
    successor_existing = [
        _existing_complete_output(archive, group_path, recording_id)
        for group_path in successor_groups
    ]
    if all(successor_existing):
        stages.append(
            {"stage": "composable_successors", "mode": "reused_exact_complete_output"}
        )
    elif any(successor_existing):
        _fail("Composable successor outputs are only partially present.")
    else:
        stages.append(
            _invoke(
                stage="composable_successors",
                command=_stage_command(
                    py,
                    "fisheye.utils.materialize_composable_chaser_successors",
                    archive,
                    "--run-name",
                    outputs["successors"],
                    "--expected-recording-id",
                    recording_id,
                    "--relative-frame-run",
                    outputs["keypoint_relative"],
                    "--semantic-selection-run",
                    outputs["semantic_selection"],
                    "--provider-motion-run-path",
                    motion_bouts["motion_run_path"],
                    "--swim-bout-run-name",
                    motion_bouts["swim_bout_run"],
                    "--track-id",
                    0,
                    "--module",
                    "controller_chase_trials",
                    "--module",
                    "generalized_chaser_bout_response",
                    "--module",
                    "chaser_escape_freeze_v2",
                    "--speed-level",
                    "filtered",
                    "--scratch-root",
                    scratch / "successors",
                    "--copy-backend",
                    copy_backend,
                    "--apply",
                ),
                log_dir=receipt_dir,
                apply=apply,
            )
        )
        if apply:
            for group_path in successor_groups:
                if not _existing_complete_output(archive, group_path, recording_id):
                    _fail("Composable successor publication was incomplete.")

    for provider, relative_key, radial_key in (
        ("keypoint", "keypoint_relative", "keypoint_radial"),
        ("detection", "detection_relative", "detection_radial"),
    ):
        radial_name = outputs[radial_key]
        execute_if_missing(
            f"{provider}_radial_near_field",
            f"analysis/chaser_radial_near_field_runs/{radial_name}",
            _stage_command(
                py,
                "fisheye.utils.materialize_chaser_radial_near_field_successor",
                "--analysis-zarr",
                archive,
                "--run-name",
                radial_name,
                "--relative-frame-run",
                outputs[relative_key],
                "--semantic-selection-run",
                outputs["semantic_selection"],
                "--geometry-selection-run",
                geometry["selection_run"],
                "--expected-selection-record-sha256",
                geometry["selection_record_sha256"],
                "--expected-physical-authority-sha256",
                geometry["physical_authority_sha256"],
                "--scratch-root",
                scratch / f"{provider}_radial",
                "--copy-backend",
                copy_backend,
                "--apply",
            ),
        )

    if eye_gaze is not None:
        execute_if_missing(
            "gaze_tracking",
            f"analysis/chaser_gaze_tracking_runs/{outputs['gaze_tracking']}",
            _stage_command(
                py,
                "fisheye.utils.materialize_composable_chaser_successors",
                archive,
                "--run-name",
                outputs["gaze_tracking"],
                "--expected-recording-id",
                recording_id,
                "--relative-frame-run",
                outputs["keypoint_relative"],
                "--semantic-selection-run",
                outputs["semantic_selection"],
                "--eye-run-name",
                eye_gaze["run_name"],
                "--eye-channel-variant",
                eye_gaze["channel_variant"],
                "--eye-convention-receipt",
                eye_gaze["convention_receipt_path"],
                "--radial-run-name",
                outputs["keypoint_radial"],
                "--module",
                "chaser_gaze_tracking_v3",
                "--scratch-root",
                scratch / "gaze_tracking",
                "--copy-backend",
                copy_backend,
                "--apply",
            ),
        )

    exact_child_receipts: dict[str, Path] = {}
    if receipt_bound_relative:
        assert relative_receipt_dir is not None
        exact_child_specs = [
            (
                "semantic_selection",
                (
                    "analysis/protocol_semantic_chaser_selection_runs/"
                    f"{outputs['semantic_selection']}"
                ),
                "protocol_semantic_chaser_selection_manifest",
                "protocol_semantic_chaser_selection_manifest_sha256",
            ),
            (
                "keypoint_radial",
                f"analysis/chaser_radial_near_field_runs/{outputs['keypoint_radial']}",
                "composable_chaser_successor_manifest",
                "composable_chaser_successor_manifest_sha256",
            ),
            (
                "detection_radial",
                f"analysis/chaser_radial_near_field_runs/{outputs['detection_radial']}",
                "composable_chaser_successor_manifest",
                "composable_chaser_successor_manifest_sha256",
            ),
            (
                "controller",
                f"analysis/controller_chase_trial_runs/{outputs['successors']}",
                "composable_chaser_successor_manifest",
                "composable_chaser_successor_manifest_sha256",
            ),
            (
                "bout",
                (
                    "analysis/generalized_chaser_bout_response_runs/"
                    f"{outputs['successors']}"
                ),
                "composable_chaser_successor_manifest",
                "composable_chaser_successor_manifest_sha256",
            ),
            (
                "escape",
                f"analysis/chaser_escape_freeze_runs/{outputs['successors']}",
                "composable_chaser_successor_manifest",
                "composable_chaser_successor_manifest_sha256",
            ),
        ]
        if "epoch_behavior" in outputs:
            exact_child_specs.insert(
                1,
                (
                    "epoch_behavior",
                    (
                        "analysis/stimulus_epoch_behavior_summary_runs/"
                        f"{outputs['epoch_behavior']}"
                    ),
                    "provider_epoch_behavior_summary_manifest",
                    "provider_epoch_behavior_summary_manifest_sha256",
                ),
            )
        if "gaze_tracking" in outputs:
            exact_child_specs.append(
                (
                    "gaze",
                    (f"analysis/chaser_gaze_tracking_runs/{outputs['gaze_tracking']}"),
                    "composable_chaser_successor_manifest",
                    "composable_chaser_successor_manifest_sha256",
                )
            )
        for key, run_path, manifest_attr, manifest_digest_attr in exact_child_specs:
            receipt_path = (
                relative_receipt_dir / f"{key}.exact_child_validation_receipt.json"
            )
            exact_child_receipts[key] = receipt_path
            stages.append(
                _invoke(
                    stage=f"{key}_exact_child_validation_receipt",
                    command=_stage_command(
                        py,
                        "fisheye.utils.seal_exact_immutable_child_validation_receipt",
                        archive,
                        "--run-path",
                        run_path,
                        "--manifest-attr",
                        manifest_attr,
                        "--manifest-digest-attr",
                        manifest_digest_attr,
                        "--palette-commit",
                        commit,
                        "--output-json",
                        receipt_path,
                        "--expected-recording-id",
                        recording_id,
                    ),
                    log_dir=receipt_dir,
                    apply=apply,
                )
            )

    if visit_configuration is not None:
        for provider, relative_key, radial_key in (
            ("keypoint", "keypoint_relative", "keypoint_radial"),
            ("detection", "detection_relative", "detection_radial"),
        ):
            visit_run = visit_configuration[f"{provider}_run"]
            visit_receipt_flags: tuple[object, ...] = ()
            if receipt_bound_relative:
                visit_receipt_flags = (
                    "--relative-frame-validation-receipt",
                    relative_receipts[provider],
                    "--semantic-selection-validation-receipt",
                    exact_child_receipts["semantic_selection"],
                    "--radial-validation-receipt",
                    exact_child_receipts[radial_key],
                )
            visit_stage = f"{provider}_near_field_visits"
            visit_binding = {
                "recording_id": recording_id,
                "visit_run": visit_run,
                "relative_frame_run": outputs[relative_key],
                "semantic_selection_run": outputs["semantic_selection"],
                "radial_near_field_run": outputs[radial_key],
                "minimum_quality_sample_count": visit_configuration[
                    "minimum_quality_sample_count"
                ],
            }
            if _existing_near_field_visit_output(archive, **visit_binding):
                stages.append(
                    {
                        "stage": visit_stage,
                        "mode": "reused_exact_complete_output",
                    }
                )
            else:
                stages.append(
                    _invoke(
                        stage=visit_stage,
                        command=_stage_command(
                            py,
                            (
                                "fisheye.utils."
                                "materialize_chaser_near_field_visit_successor"
                            ),
                            "--analysis-zarr",
                            archive,
                            "--run-name",
                            visit_run,
                            "--relative-frame-run",
                            outputs[relative_key],
                            "--semantic-selection-run",
                            outputs["semantic_selection"],
                            "--radial-near-field-run",
                            outputs[radial_key],
                            "--expected-recording-id",
                            recording_id,
                            *visit_receipt_flags,
                            "--minimum-quality-sample-count",
                            visit_configuration["minimum_quality_sample_count"],
                            "--scratch-root",
                            scratch / f"{provider}_near_field_visits",
                            "--copy-backend",
                            copy_backend,
                            "--apply",
                        ),
                        log_dir=receipt_dir,
                        apply=apply,
                    )
                )
                if apply and not _existing_near_field_visit_output(
                    archive, **visit_binding
                ):
                    _fail(
                        f"Stage {visit_stage!r} did not produce its exact bound output."
                    )
            if receipt_bound_relative:
                assert relative_receipt_dir is not None
                visit_receipt = (
                    relative_receipt_dir
                    / f"{provider}_near_field_visits.exact_child_validation_receipt.json"
                )
                receipt_key = f"{provider}_near_field_visits"
                exact_child_receipts[receipt_key] = visit_receipt
                stages.append(
                    _invoke(
                        stage=f"{receipt_key}_exact_child_validation_receipt",
                        command=_stage_command(
                            py,
                            (
                                "fisheye.utils."
                                "seal_exact_immutable_child_validation_receipt"
                            ),
                            archive,
                            "--run-path",
                            f"analysis/chaser_near_field_visits_runs/{visit_run}",
                            "--manifest-attr",
                            "composable_chaser_successor_manifest",
                            "--manifest-digest-attr",
                            "composable_chaser_successor_manifest_sha256",
                            "--palette-commit",
                            commit,
                            "--output-json",
                            visit_receipt,
                            "--expected-recording-id",
                            recording_id,
                        ),
                        log_dir=receipt_dir,
                        apply=apply,
                    )
                )

    if "body_alignment_by_distance" in outputs:
        alignment_receipt_flags: tuple[object, ...] = ()
        if receipt_bound_relative:
            alignment_receipt_flags = (
                "--relative-frame-receipt",
                relative_receipts["keypoint"],
                "--semantic-selection-receipt",
                exact_child_receipts["semantic_selection"],
            )
        execute_if_missing(
            "body_alignment_by_distance",
            (
                "analysis/chaser_body_alignment_by_distance_runs/"
                f"{outputs['body_alignment_by_distance']}"
            ),
            _stage_command(
                py,
                (
                    "fisheye.utils."
                    "materialize_chaser_body_alignment_by_distance_successor"
                ),
                "--analysis-zarr",
                archive,
                "--run-name",
                outputs["body_alignment_by_distance"],
                "--relative-frame-run",
                outputs["keypoint_relative"],
                "--semantic-selection-run",
                outputs["semantic_selection"],
                "--expected-recording-id",
                recording_id,
                *alignment_receipt_flags,
                "--distance-bin-width-mm",
                5.0,
                "--scratch-root",
                scratch / "body_alignment_by_distance",
                "--copy-backend",
                copy_backend,
                "--apply",
            ),
        )
        if receipt_bound_relative:
            assert relative_receipt_dir is not None
            alignment_receipt = (
                relative_receipt_dir
                / "body_alignment_by_distance.exact_child_validation_receipt.json"
            )
            exact_child_receipts["body_alignment_by_distance"] = alignment_receipt
            stages.append(
                _invoke(
                    stage=("body_alignment_by_distance_exact_child_validation_receipt"),
                    command=_stage_command(
                        py,
                        ("fisheye.utils.seal_exact_immutable_child_validation_receipt"),
                        archive,
                        "--run-path",
                        (
                            "analysis/chaser_body_alignment_by_distance_runs/"
                            f"{outputs['body_alignment_by_distance']}"
                        ),
                        "--manifest-attr",
                        "composable_chaser_successor_manifest",
                        "--manifest-digest-attr",
                        "composable_chaser_successor_manifest_sha256",
                        "--palette-commit",
                        commit,
                        "--output-json",
                        alignment_receipt,
                        "--expected-recording-id",
                        recording_id,
                    ),
                    log_dir=receipt_dir,
                    apply=apply,
                )
            )

    relative_receipt_flags: tuple[object, ...] = ()
    occupancy_receipt_flags: tuple[object, ...] = ()
    if receipt_bound_relative:
        relative_receipt_flags = (
            "--keypoint-relative-frame-receipt",
            relative_receipts["keypoint"],
            "--detection-relative-frame-receipt",
            relative_receipts["detection"],
        )
        occupancy_receipt_flags = (
            *relative_receipt_flags,
            "--semantic-selection-receipt",
            exact_child_receipts["semantic_selection"],
            "--keypoint-radial-receipt",
            exact_child_receipts["keypoint_radial"],
            "--detection-radial-receipt",
            exact_child_receipts["detection_radial"],
        )
    execute_if_missing(
        "spatial_occupancy",
        f"analysis/chaser_spatial_occupancy_runs/{spatial_occupancy_run}",
        _stage_command(
            py,
            "fisheye.utils.materialize_chaser_spatial_occupancy_successor",
            "--analysis-zarr",
            archive,
            "--run-name",
            spatial_occupancy_run,
            "--keypoint-relative-frame-run",
            outputs["keypoint_relative"],
            "--detection-relative-frame-run",
            outputs["detection_relative"],
            *occupancy_receipt_flags,
            "--semantic-selection-run",
            outputs["semantic_selection"],
            "--keypoint-radial-run",
            outputs["keypoint_radial"],
            "--detection-radial-run",
            outputs["detection_radial"],
            "--expected-recording-id",
            recording_id,
            "--bin-width-mm",
            2.0,
            "--scratch-root",
            scratch / "spatial_occupancy",
            "--copy-backend",
            copy_backend,
            "--apply",
        ),
    )

    if receipt_bound_relative:
        assert relative_receipt_dir is not None
        spatial_source_receipt = (
            relative_receipt_dir
            / "spatial_occupancy.exact_child_validation_receipt.json"
        )
        exact_child_receipts["spatial_occupancy"] = spatial_source_receipt
        stages.append(
            _invoke(
                stage="spatial_occupancy_exact_child_validation_receipt",
                command=_stage_command(
                    py,
                    "fisheye.utils.seal_exact_immutable_child_validation_receipt",
                    archive,
                    "--run-path",
                    (f"analysis/chaser_spatial_occupancy_runs/{spatial_occupancy_run}"),
                    "--manifest-attr",
                    "composable_chaser_successor_manifest",
                    "--manifest-digest-attr",
                    "composable_chaser_successor_manifest_sha256",
                    "--palette-commit",
                    commit,
                    "--output-json",
                    spatial_source_receipt,
                    "--expected-recording-id",
                    recording_id,
                ),
                log_dir=receipt_dir,
                apply=apply,
            )
        )
        projection_receipt = relative_receipt_dir / (
            GAZE_EPOCH_ALIGNMENT_PROJECTION_RECEIPT_NAME
            if "gaze" in exact_child_receipts
            else EPOCH_ALIGNMENT_PROJECTION_RECEIPT_NAME
        )
        # The visit successors retain independent lineage and receipts.  They
        # intentionally do not widen the closed v7/v8 projection grammar.
        stages.append(
            _invoke(
                stage="exact_chaser_projection_receipt",
                command=_stage_command(
                    py,
                    "fisheye.utils.seal_exact_chaser_projection_receipt",
                    "--analysis-zarr",
                    archive,
                    "--palette-commit",
                    commit,
                    "--output-json",
                    projection_receipt,
                    "--expected-recording-id",
                    recording_id,
                    "--semantic-selection-receipt",
                    exact_child_receipts["semantic_selection"],
                    "--keypoint-radial-receipt",
                    exact_child_receipts["keypoint_radial"],
                    "--detection-radial-receipt",
                    exact_child_receipts["detection_radial"],
                    "--controller-receipt",
                    exact_child_receipts["controller"],
                    "--bout-receipt",
                    exact_child_receipts["bout"],
                    "--escape-receipt",
                    exact_child_receipts["escape"],
                    "--spatial-occupancy-receipt",
                    exact_child_receipts["spatial_occupancy"],
                    *(
                        ("--gaze-receipt", exact_child_receipts["gaze"])
                        if "gaze" in exact_child_receipts
                        else ()
                    ),
                    *(
                        (
                            "--epoch-behavior-receipt",
                            exact_child_receipts["epoch_behavior"],
                        )
                        if "epoch_behavior" in exact_child_receipts
                        else ()
                    ),
                    *(
                        (
                            "--body-alignment-by-distance-receipt",
                            exact_child_receipts["body_alignment_by_distance"],
                        )
                        if "body_alignment_by_distance" in exact_child_receipts
                        else ()
                    ),
                    "--keypoint-relative-frame-receipt",
                    relative_receipts["keypoint"],
                    "--detection-relative-frame-receipt",
                    relative_receipts["detection"],
                ),
                log_dir=receipt_dir,
                apply=apply,
            )
        )

    if visit_configuration is not None:
        for provider in ("keypoint", "detection"):
            visit_run = visit_configuration[f"{provider}_run"]
            visit_bundle = visit_configuration[f"{provider}_plot_bundle"]
            visit_plot_receipt = visit_plot_dir / (
                f"{visit_bundle}_near_field_visit_plot_receipt.json"
            )
            if _validated_plot_receipt(
                visit_plot_receipt,
                recording_id=recording_id,
                require_self_contained_recipe=receipt_bound_relative,
                expected_plot_recipe_id=NEAR_FIELD_VISIT_PLOT_RECIPE_ID,
                expected_source_run_path=(
                    f"analysis/chaser_near_field_visits_runs/{visit_run}"
                ),
            ):
                stages.append(
                    {
                        "stage": f"{provider}_near_field_visit_plots",
                        "mode": "reused_exact_receipt",
                    }
                )
            else:
                stages.append(
                    _invoke(
                        stage=f"{provider}_near_field_visit_plots",
                        command=_stage_command(
                            py,
                            "fisheye.utils.plot_chaser_near_field_visit_successor",
                            archive,
                            "--run-name",
                            visit_run,
                            "--bundle-name",
                            visit_bundle,
                            "--expected-recording-id",
                            recording_id,
                            "--output-dir",
                            visit_plot_dir,
                            *(
                                (
                                    "--source-validation-receipt",
                                    exact_child_receipts[
                                        f"{provider}_near_field_visits"
                                    ],
                                )
                                if receipt_bound_relative
                                else ()
                            ),
                        ),
                        log_dir=receipt_dir,
                        apply=apply,
                    )
                )

    if "body_alignment_by_distance" in outputs:
        alignment_plot_receipt = (
            alignment_plot_dir
            / f"{alignment_plot_bundle}_body_alignment_plot_receipt.json"
        )
        if _validated_plot_receipt(
            alignment_plot_receipt,
            recording_id=recording_id,
            require_self_contained_recipe=receipt_bound_relative,
            expected_plot_recipe_id=BODY_ALIGNMENT_PLOT_RECIPE_ID,
        ):
            stages.append(
                {
                    "stage": "body_alignment_by_distance_plots",
                    "mode": "reused_exact_receipt",
                }
            )
        else:
            stages.append(
                _invoke(
                    stage="body_alignment_by_distance_plots",
                    command=_stage_command(
                        py,
                        (
                            "fisheye.utils."
                            "plot_chaser_body_alignment_by_distance_successor"
                        ),
                        archive,
                        "--run-name",
                        outputs["body_alignment_by_distance"],
                        "--bundle-name",
                        alignment_plot_bundle,
                        "--expected-recording-id",
                        recording_id,
                        "--output-dir",
                        alignment_plot_dir,
                        *(
                            (
                                "--source-validation-receipt",
                                exact_child_receipts["body_alignment_by_distance"],
                            )
                            if receipt_bound_relative
                            else ()
                        ),
                    ),
                    log_dir=receipt_dir,
                    apply=apply,
                )
            )

    spatial_receipt = (
        spatial_plot_dir / f"{spatial_plot_bundle}_spatial_occupancy_plot_receipt.json"
    )
    if _validated_plot_receipt(
        spatial_receipt,
        recording_id=recording_id,
        require_self_contained_recipe=receipt_bound_relative,
        expected_plot_recipe_id=SPATIAL_OCCUPANCY_PLOT_RECIPE_ID,
    ):
        stages.append(
            {"stage": "spatial_occupancy_plots", "mode": "reused_exact_receipt"}
        )
    else:
        stages.append(
            _invoke(
                stage="spatial_occupancy_plots",
                command=_stage_command(
                    py,
                    "fisheye.utils.plot_chaser_spatial_occupancy_successor",
                    archive,
                    "--run-name",
                    spatial_occupancy_run,
                    "--bundle-name",
                    spatial_plot_bundle,
                    "--expected-recording-id",
                    recording_id,
                    "--output-dir",
                    spatial_plot_dir,
                    *(
                        (
                            "--source-validation-receipt",
                            exact_child_receipts["spatial_occupancy"],
                        )
                        if receipt_bound_relative
                        else ()
                    ),
                ),
                log_dir=receipt_dir,
                apply=apply,
            )
        )

    dashboard_receipt = plot_dir / f"{dashboard_bundle}_plot_receipt.json"
    if _validated_plot_receipt(
        dashboard_receipt,
        recording_id=recording_id,
        require_self_contained_recipe=receipt_bound_relative,
    ):
        stages.append({"stage": "dashboard_plots", "mode": "reused_exact_receipt"})
    else:
        stages.append(
            _invoke(
                stage="dashboard_plots",
                command=_stage_command(
                    py,
                    "fisheye.utils.plot_composable_chaser_successors",
                    archive,
                    "--run-name",
                    outputs["successors"],
                    "--expected-recording-id",
                    recording_id,
                    "--output-dir",
                    plot_dir,
                    "--bundle-name",
                    dashboard_bundle,
                    *(
                        (
                            "--controller-validation-receipt",
                            exact_child_receipts["controller"],
                            "--bout-validation-receipt",
                            exact_child_receipts["bout"],
                            "--escape-validation-receipt",
                            exact_child_receipts["escape"],
                        )
                        if receipt_bound_relative
                        else ()
                    ),
                ),
                log_dir=receipt_dir,
                apply=apply,
            )
        )
    detailed_receipt = detailed_dir / f"{outputs['detailed_bundle']}_receipt.json"
    if _validated_plot_receipt(
        detailed_receipt,
        recording_id=recording_id,
        require_self_contained_recipe=receipt_bound_relative,
        expected_plot_recipe_id=DETAILED_PLOT_RECIPE_ID,
    ):
        stages.append({"stage": "detailed_plots", "mode": "reused_exact_receipt"})
    else:
        stages.append(
            _invoke(
                stage="detailed_plots",
                command=_stage_command(
                    py,
                    "fisheye.utils.plot_chaser_detailed_successors",
                    archive,
                    "--run-name",
                    outputs["successors"],
                    "--relative-frame-run",
                    outputs["keypoint_relative"],
                    "--detection-relative-frame-run",
                    outputs["detection_relative"],
                    *relative_receipt_flags,
                    *(
                        (
                            "--controller-validation-receipt",
                            exact_child_receipts["controller"],
                            "--bout-validation-receipt",
                            exact_child_receipts["bout"],
                            "--escape-validation-receipt",
                            exact_child_receipts["escape"],
                            "--keypoint-radial-validation-receipt",
                            exact_child_receipts["keypoint_radial"],
                            "--detection-radial-validation-receipt",
                            exact_child_receipts["detection_radial"],
                        )
                        if receipt_bound_relative
                        else ()
                    ),
                    "--keypoint-radial-run",
                    outputs["keypoint_radial"],
                    "--detection-radial-run",
                    outputs["detection_radial"],
                    "--expected-recording-id",
                    recording_id,
                    "--output-dir",
                    detailed_dir,
                    "--bundle-name",
                    outputs["detailed_bundle"],
                ),
                log_dir=receipt_dir,
                apply=apply,
            )
        )

    result = json_attr_safe(
        {
            "schema_id": RECEIPT_SCHEMA_ID,
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": (
                "complete_selector_ineligible" if apply else "planned_no_writes"
            ),
            "task_sha256": task["task_sha256"],
            "task_index": task_index,
            "recording_id": recording_id,
            "analysis_zarr": str(archive),
            "palette_repo": str(repo),
            "palette_commit": commit,
            "stages": stages,
            "plot_output_dir": str(plot_dir),
            "safety": EXPECTED_SAFETY,
        }
    )
    result["payload_sha256"] = canonical_json_sha256(result)
    if apply:
        write_json_atomic(receipt_dir / "receipt.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="freeze exact cohort inputs")
    plan_parser.add_argument("--registry-snapshot", type=Path, required=True)
    plan_parser.add_argument("--operations-root", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    replan_parser = subparsers.add_parser(
        "replan",
        help="freeze versioned successors from a prior task's exact recording set",
    )
    replan_parser.add_argument("task", type=Path)
    replan_parser.add_argument("--operations-root", type=Path, required=True)
    replan_parser.add_argument("--output", type=Path, required=True)
    successor_parser = subparsers.add_parser(
        "successor",
        help="derive the receipt-bound visit and self-contained-plot task successor",
    )
    successor_parser.add_argument("task", type=Path)
    successor_parser.add_argument("--output", type=Path, required=True)
    successor_parser.add_argument(
        "--eye-gaze-bindings",
        type=Path,
        help=(
            "Optional exact JSON row list covering every frozen recording with "
            "analysis_zarr, eye_run_name, eye_channel_variant, and one accepted "
            "eye_convention_receipt. No eye selector is resolved."
        ),
    )
    validate_parser = subparsers.add_parser("validate", help="validate a frozen task")
    validate_parser.add_argument("task", type=Path)
    run_parser = subparsers.add_parser("run-one", help="run one frozen task entry")
    run_parser.add_argument("task", type=Path)
    run_parser.add_argument("--task-index", type=int, required=True)
    run_parser.add_argument("--palette-repo", type=Path, required=True)
    run_parser.add_argument("--palette-commit", required=True)
    run_parser.add_argument("--scratch-root", type=Path, required=True)
    run_parser.add_argument("--receipt-root", type=Path, required=True)
    run_parser.add_argument(
        "--copy-backend", choices=("python", "rsync"), default="rsync"
    )
    run_parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        result = plan_cohort_task(
            args.registry_snapshot,
            operations_root=args.operations_root,
        )
        write_json_atomic(args.output.expanduser().resolve(), result)
    elif args.command == "replan":
        result = replan_cohort_task(
            args.task,
            operations_root=args.operations_root,
        )
        write_json_atomic(args.output.expanduser().resolve(), result)
    elif args.command == "successor":
        result = successor_cohort_task(
            args.task,
            eye_gaze_bindings=args.eye_gaze_bindings,
        )
        write_json_atomic(args.output.expanduser().resolve(), result)
    elif args.command == "validate":
        task = load_cohort_task(args.task)
        result = {
            "status": "valid",
            "task_sha256": task["task_sha256"],
            "recording_count": task["recording_count"],
            "status_counts": task["status_counts"],
            "runnable_task_indices": task["runnable_task_indices"],
        }
    else:
        result = run_one(
            args.task,
            task_index=args.task_index,
            palette_repo=args.palette_repo,
            palette_commit=args.palette_commit,
            scratch_root=args.scratch_root,
            receipt_root=args.receipt_root,
            copy_backend=args.copy_backend,
            apply=args.apply,
        )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ComposableChaserCohortError",
    "load_cohort_task",
    "main",
    "plan_cohort_task",
    "replan_cohort_task",
    "run_one",
    "successor_cohort_task",
]
