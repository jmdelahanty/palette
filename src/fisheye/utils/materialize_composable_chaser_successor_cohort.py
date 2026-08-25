"""Plan or run the selector-ineligible GoodBatBadBat successor cohort.

The planner consumes a frozen registry JSON export and resolves every mutable
recording-local authority exactly once.  The resulting task contains exact run
names, manifest digests, metadata-file digests, raw-H5 stat identities, and a
canonical task digest.  ``run-one`` accepts only that task, revalidates all
frozen inputs, and executes one recording serially so no two processes write
the same analysis Zarr.

Neither command resolves a scientific selector while executing a task.  Every
publication remains selector-ineligible and no command writes the registry.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


TASK_SCHEMA_ID = "palette.composable_chaser_successor_cohort_task"
TASK_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_ID = "palette.composable_chaser_successor_cohort_receipt"
RECEIPT_SCHEMA_VERSION = 1

SEMANTIC_STIMULUS_RUN = (
    "stimulus_semantic_goodbatbadbat_historical_20260825_v2"
)
SEMANTIC_EPOCH_V1_RUN = (
    "stimulus_epochs_semantic_goodbatbadbat_20260825_r2_v1"
)
SEMANTIC_EPOCH_V2_RUN = (
    "stimulus_epochs_semantic_goodbatbadbat_20260825_r2_v2"
)
SEMANTIC_SELECTION_RUN = (
    "protocol_semantic_chaser_goodbatbadbat_historical_20260825_v2"
)
KEYPOINT_PROXY_RUN = (
    "chaser_input_provenance_proxy_keypoint_triad_cohort_20260821_v2"
)
DETECTION_PROXY_RUN = (
    "chaser_input_provenance_proxy_detection_bbox_centroid_cohort_20260821_v2"
)
KEYPOINT_RELATIVE_RUN = (
    "chaser_relative_frame_keypoint_triad_cohort_20260825_"
    "exact_trials_session_time_activity_orthogonal_v3"
)
DETECTION_RELATIVE_RUN = (
    "chaser_relative_frame_detection_bbox_centroid_cohort_20260825_"
    "exact_trials_session_time_activity_orthogonal_v3"
)
SUCCESSOR_RUN = (
    "goodbatbadbat_chaser_successors_20260825_"
    "exact_trials_session_time_activity_orthogonal_v3"
)
KEYPOINT_RADIAL_RUN = (
    "goodbatbadbat_chaser_radial_near_field_20260825_"
    "exact_session_time_activity_orthogonal_v2"
)
DETECTION_RADIAL_RUN = (
    "goodbatbadbat_chaser_radial_near_field_detection_bbox_centroid_20260825_"
    "exact_session_time_activity_orthogonal_v2"
)
SPATIAL_OCCUPANCY_RUN = (
    "goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260825_"
    "exact_epochs_v1"
)
DETAILED_BUNDLE_NAME = "goodbatbadbat_chaser_detailed_activity_orthogonal_v2"

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


def _raw_h5_binding(root_attrs: Mapping[str, Any], *, recording_id: str) -> dict[str, Any]:
    raw_h5 = Path(
        _text(root_attrs.get("source_h5_path"), field="source raw-H5 path")
    ).expanduser().resolve()
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


def _resolve_motion_bouts(archive: Path, *, recording_id: str) -> dict[str, str]:
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
    return {
        "selection_policy": "ordered_exact_compatible_pair_v1",
        "motion_run_path": f"analysis/track_kinematics_runs/provider/{motion}",
        "swim_bout_run": bouts,
    }


def _output_groups() -> tuple[str, ...]:
    return (
        f"analysis/stimulus_runs/{SEMANTIC_STIMULUS_RUN}",
        f"analysis/stimulus_epoch_runs/{SEMANTIC_EPOCH_V1_RUN}",
        f"analysis/stimulus_epoch_runs/{SEMANTIC_EPOCH_V2_RUN}",
        f"analysis/protocol_semantic_chaser_selection_runs/{SEMANTIC_SELECTION_RUN}",
        f"analysis/chaser_relative_frame_runs/{KEYPOINT_RELATIVE_RUN}",
        f"analysis/chaser_relative_frame_runs/{DETECTION_RELATIVE_RUN}",
        f"analysis/controller_chase_trial_runs/{SUCCESSOR_RUN}",
        f"analysis/generalized_chaser_bout_response_runs/{SUCCESSOR_RUN}",
        f"analysis/chaser_escape_freeze_runs/{SUCCESSOR_RUN}",
        f"analysis/chaser_radial_near_field_runs/{KEYPOINT_RADIAL_RUN}",
        f"analysis/chaser_radial_near_field_runs/{DETECTION_RADIAL_RUN}",
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
        _input_group_binding(
            archive,
            motion_bouts["motion_run_path"],
            field="provider motion input",
        ),
        _input_group_binding(
            archive,
            f"analysis/swim_bout_runs/{motion_bouts['swim_bout_run']}",
            field="swim-bout input",
        ),
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
    if len(existing_outputs) == len(output_groups):
        status = (
            "complete"
            if (
                detailed_receipt.is_file()
                and dashboard_receipt.is_file()
                and spatial_receipt.is_file()
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
                "keypoint_relative": KEYPOINT_RELATIVE_RUN,
                "detection_relative": DETECTION_RELATIVE_RUN,
                "successors": SUCCESSOR_RUN,
                "keypoint_radial": KEYPOINT_RADIAL_RUN,
                "detection_radial": DETECTION_RADIAL_RUN,
                "spatial_occupancy": SPATIAL_OCCUPANCY_RUN,
                "detailed_bundle": DETAILED_BUNDLE_NAME,
            },
            "output_group_paths": list(output_groups),
            "existing_output_group_paths": existing_outputs,
            "plot_output_dir": str(plot_dir),
        }
    )


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
    normalized_rows = [
        _mapping(row, field="registry snapshot row") for row in rows
    ]
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
        int(entry["task_index"])
        for entry in entries
        if entry["status"] != "complete"
    ]
    task = json_attr_safe(
        {
            "schema_id": TASK_SCHEMA_ID,
            "schema_version": TASK_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_registry_snapshot": {
                "path": str(snapshot),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "row_count": len(normalized_rows),
            },
            "selection_policy": {
                "protocol_name": "goodbatbadbat",
                "recording_order": "lexicographic_absolute_analysis_zarr_path_v1",
                "motion_bout_resolution": "ordered_exact_compatible_pair_v1",
                "selector_resolution_time": "planning_only",
                "execution_selector_resolution": False,
            },
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


def load_cohort_task(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load and strictly validate one frozen task document."""

    if isinstance(source, Mapping):
        task = dict(source)
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cohort task does not exist: {path}")
        task = dict(_mapping(json.loads(path.read_bytes()), field="cohort task"))
    if task.get("schema_id") != TASK_SCHEMA_ID or task.get("schema_version") != 1:
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
    root_attrs, _root_sha = _zarr_attrs(archive, field="analysis Zarr root")
    recording_id = _recording_identity(root_attrs, archive)
    if recording_id != entry.get("recording_id"):
        _fail("Frozen recording identity has changed.")
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


def _existing_complete_output(archive: Path, group_path: str, recording_id: str) -> bool:
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
            persisted = _digest(
                attrs[digest_key], field=f"existing {key} digest"
            )
            if canonical_json_sha256(value) != persisted:
                _fail(f"Existing output manifest digest is stale: {group_path}")
    return True


def _validated_plot_receipt(path: Path, *, recording_id: str) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        _fail(f"Plot receipt is not a file: {path}")
    try:
        receipt = dict(
            _mapping(json.loads(path.read_bytes()), field="plot receipt")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ComposableChaserCohortError(
            f"Plot receipt cannot be read: {path}"
        ) from exc
    persisted = _digest(receipt.pop("payload_sha256", None), field="plot receipt digest")
    if canonical_json_sha256(receipt) != persisted:
        _fail(f"Plot receipt digest is stale: {path}")
    if receipt.get("recording_id") != recording_id:
        _fail(f"Plot receipt recording identity mismatch: {path}")
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
    if (
        len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
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
    if apply:
        plot_dir.mkdir(parents=True, exist_ok=True)
        detailed_dir.mkdir(parents=True, exist_ok=True)
        spatial_plot_dir.mkdir(parents=True, exist_ok=True)

    outputs = _mapping(entry["output_run_names"], field="output run names")
    spatial_occupancy_run = _exact_name(
        outputs.get("spatial_occupancy", SPATIAL_OCCUPANCY_RUN),
        field="spatial occupancy run",
    )
    raw_h5 = _mapping(entry["raw_h5"], field="raw-H5 binding")
    geometry = _mapping(entry["geometry"], field="geometry binding")
    keypoint_proxy = _mapping(entry["keypoint_proxy"], field="keypoint proxy")
    detection_proxy = _mapping(entry["detection_proxy"], field="detection proxy")
    motion_bouts = _mapping(
        entry["motion_and_bouts"], field="motion and bout binding"
    )
    stages: list[dict[str, Any]] = []

    def execute_if_missing(
        stage: str, group_path: str, command: Sequence[str]
    ) -> None:
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

    for provider, proxy, relative_key in (
        ("keypoint", keypoint_proxy, "keypoint_relative"),
        ("detection", detection_proxy, "detection_relative"),
    ):
        relative_name = outputs[relative_key]
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
                "--copy-backend",
                copy_backend,
                "--apply",
                "--json",
            ),
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
                    "--no-body-extension",
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

    spatial_receipt = (
        spatial_plot_dir
        / f"{spatial_occupancy_run}_spatial_occupancy_plot_receipt.json"
    )
    if _validated_plot_receipt(spatial_receipt, recording_id=recording_id):
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
                    "--expected-recording-id",
                    recording_id,
                    "--output-dir",
                    spatial_plot_dir,
                ),
                log_dir=receipt_dir,
                apply=apply,
            )
        )

    dashboard_receipt = plot_dir / f"{outputs['successors']}_plot_receipt.json"
    if _validated_plot_receipt(dashboard_receipt, recording_id=recording_id):
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
                ),
                log_dir=receipt_dir,
                apply=apply,
            )
        )
    detailed_receipt = detailed_dir / f"{outputs['detailed_bundle']}_receipt.json"
    if _validated_plot_receipt(detailed_receipt, recording_id=recording_id):
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
    validate_parser = subparsers.add_parser("validate", help="validate a frozen task")
    validate_parser.add_argument("task", type=Path)
    run_parser = subparsers.add_parser("run-one", help="run one frozen task entry")
    run_parser.add_argument("task", type=Path)
    run_parser.add_argument("--task-index", type=int, required=True)
    run_parser.add_argument("--palette-repo", type=Path, required=True)
    run_parser.add_argument("--palette-commit", required=True)
    run_parser.add_argument("--scratch-root", type=Path, required=True)
    run_parser.add_argument("--receipt-root", type=Path, required=True)
    run_parser.add_argument("--copy-backend", choices=("python", "rsync"), default="rsync")
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
    "run_one",
]
