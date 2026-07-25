from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from types import MappingProxyType, SimpleNamespace
import uuid

import numpy as np
import pytest

from fisheye.analysis import track_kinematics as mod
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.shared.coordinate_record import stamp_and_bind_persisted_coordinate_record
from fisheye.shared.zarr_payload_receipt import DECODED_PAYLOAD_CANONICALIZATION
from tests.unit.fisheye.test_directed_transform_chain import _world
from tests.unit.fisheye.test_track_kinematics_coordinate_contract import (
    _WritableGroup,
    _canonical_crop_position_surface,
    _selected_stimulus_physical_authority,
)


def _fresh_full_motion_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    physical: bool = False,
    root_auxiliary: bool = False,
    bout_auxiliary: bool = False,
    materializer_metadata: bool = False,
    fps: float = 1.0,
    smooth_seconds: float = 1.0,
    headings_deg: np.ndarray | None = None,
    hysteresis_enabled: bool = False,
    source_rows: np.ndarray | None = None,
    _return_template_source: bool = False,
):
    world = _world(convention="continuous", archive_token=object())
    source = _canonical_crop_position_surface(world)
    root = world["root"]
    source_heading_values = (
        np.zeros(2, dtype=np.float32)
        if headings_deg is None
        else np.asarray(headings_deg)
    )
    crop_parent = _WritableGroup(
        path="crop_runs",
        archive_token=world["archive_token"],
    )
    crop = crop_parent.create_group("c1")
    crop.children["centers_img_xy"] = source.coordinates.coordinate_node
    crop.children["instance_key"] = source.coordinates.row_identity._key_array_node
    root["crop_runs"] = crop_parent

    keypoint_parent = _WritableGroup(
        path="keypoints_runs",
        archive_token=world["archive_token"],
    )
    keypoint = keypoint_parent.create_group("kp_1")
    heading_node = keypoint.create_array(
        "heading",
        data=source_heading_values,
    )
    usability_node = keypoint.create_array(
        "heading_usable",
        data=np.ones(2, dtype=bool),
    )
    keypoint.create_array(
        "instance_key",
        data=np.asarray(source.coordinates.row_identity._key_array_node[:]),
    )
    root["keypoints_runs"] = keypoint_parent

    tracking_parent = _WritableGroup(
        path="tracking_runs",
        archive_token=world["archive_token"],
    )
    tracking = tracking_parent.create_group("trk_1")
    tracking.create_array(
        "track_ids",
        data=np.asarray([7, 7], dtype=np.int32),
    )
    tracking.create_array(
        "arena_ids",
        data=np.asarray([3, 3], dtype=np.int32),
    )
    tracking.create_array(
        "instance_key",
        data=np.asarray(source.coordinates.row_identity._key_array_node[:]),
    )
    tracking.create_array(
        "track_ids_present",
        data=np.asarray([7], dtype=np.int32),
    )
    tracking.create_array(
        "track_arena_ids",
        data=np.asarray([3], dtype=np.int32),
    )
    root["tracking_runs"] = tracking_parent
    input_authority = mod.build_track_motion_input_authority(
        root,
        source_positions=source.coordinates,
        mode="offline_exact_sources_v1",
        heading_node=heading_node,
        keypoint_usability_node=usability_node,
        keypoint_row_key_node=keypoint["instance_key"],
        tracking_group=tracking,
    )
    physical_authority = (
        _selected_stimulus_physical_authority(world) if physical else None
    )
    pixel_to_mm = (
        physical_authority.mm_per_pixel if physical_authority is not None else None
    )
    source_rows = (
        np.asarray([0, 1], dtype=np.int64)
        if source_rows is None
        else np.asarray(source_rows, dtype=np.int64)
    )
    frames = mod.resolve_source_acquisition_frame_indices(
        source.temporal_authority,
        source_rows,
    )
    tracks, summaries = mod.build_track_datasets(
        track_ids=np.full(source_rows.shape, 7, dtype=np.int64),
        frames=frames,
        positions_px=np.asarray(source.coordinates.coordinate_node[:])[source_rows],
        headings_deg=source_heading_values[source_rows],
        keypoint_success=np.ones(source_rows.shape, dtype=bool),
        detection_source=None,
        fps=fps,
        smooth_seconds=smooth_seconds,
        pixel_to_mm=pixel_to_mm,
        hysteresis_high_px=2.0 if hysteresis_enabled else None,
        hysteresis_low_px=1.0 if hysteresis_enabled else None,
        hysteresis_min_frames=3 if hysteresis_enabled else None,
        hysteresis_band_policy="reset",
        smoothing_alignment="centered",
        source_row_index=source_rows,
        source_temporal_authority=source.temporal_authority,
    )
    run_name = "motion_physical" if physical else "motion_pixel"
    runs = _WritableGroup(
        path="analysis/track_kinematics_runs",
        archive_token=world["archive_token"],
    )
    offline = runs.create_group("offline")
    run = offline.create_group(run_name)
    run.attrs[mod.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR] = str(uuid.uuid4())
    root["analysis"]["track_kinematics_runs"] = runs
    mod.save_track_kinematics_tracks(
        run,
        tracks,
        summaries,
        source_temporal_authority=source.temporal_authority,
        positions_px_source=source.coordinates,
        input_authority=input_authority,
        physical_authority=physical_authority,
        track_id_to_arena_id={7: 3},
    )
    if bout_auxiliary:
        bouts = run["tracks/id_7"].create_group("swim_bouts")
        bouts.attrs["source_swim_bout_run"] = "bout_1"
        start = bouts.create_array(
            "start_frame",
            data=np.asarray([10], dtype=np.int32),
        )
        end = bouts.create_array(
            "end_frame",
            data=np.asarray([11], dtype=np.int32),
        )
        mod.stamp_geometry_preload_attrs(start)
        mod.stamp_geometry_preload_attrs(end)
    inputs = {
        "detection_path": "detect_runs/d1",
        "position_source_path": "crop_runs/c1/centers_img_xy",
        "position_source_rowset_path": "crop_runs/c1",
        "position_source_kind": "canonical_crop_rows_source_camera_centers",
        "keypoint_path": "keypoints_runs/kp_1",
        "crop_run": "c1",
        "tracking_path": "tracking_runs/trk_1",
    }
    if bout_auxiliary:
        inputs["swim_bout_run"] = "bout_1"
    if root_auxiliary:
        auxiliary_arrays = {
            "camera_frame_ids": np.asarray([10, 11], dtype=np.int64),
            "stimulus_frame_nums": np.asarray([20, 21], dtype=np.int64),
            "timestamp_ns": np.asarray([100, 200], dtype=np.int64),
            "trial_state": np.asarray([1, 2], dtype=np.int16),
            "has_offline": np.asarray([True, True], dtype=bool),
        }
        for name, values in auxiliary_arrays.items():
            node = run.create_array(name, data=values)
            mod.stamp_geometry_preload_attrs(node)
        inputs["chaser_metrics"] = {
            "metrics_run": "metrics_1",
            "stimulus_run": "stimulus_1",
            "chaser_index": 0,
            "distance_interpolation_seconds": 0.0,
            "coordinate_geometry_status": "not_present",
            "coordinate_geometry_reason_code": "NONE",
            "omitted_coordinate_fields": [],
        }
    parameters = {
        "fps": fps,
        "smoothing_seconds": smooth_seconds,
        "smoothing_method": "moving_average",
        "smoothing_alignment": "centered",
        "savgol_polyorder": None,
        "distance_interpolation_seconds": 0.0,
        "coordinate_space": "source_camera_image_px",
        "hysteresis_enabled": hysteresis_enabled,
        "hysteresis_high_px": 2.0 if hysteresis_enabled else None,
        "hysteresis_low_px": 1.0 if hysteresis_enabled else None,
        "hysteresis_min_frames": 3 if hysteresis_enabled else None,
        # A disabled filter still persists its valid effective policy.  The
        # policy is inactive, not undefined.
        "hysteresis_band_policy": "reset",
    }
    run.attrs.update(
        mod._track_kinematics_contract_attrs(
            run_type="offline",
            method="track_kinematics_offline",
            parameters=parameters,
            inputs=inputs,
        )
    )
    stage_provenance = {
        "stage": "track_kinematics",
        "parameters": copy.deepcopy(parameters),
        "inputs": copy.deepcopy(inputs),
    }
    run_provenance = {
        "schema": "palette.run_provenance.v1",
        "git_sha": "a" * 40,
        "config_hash": mod.sha256_payload(parameters),
        "params": copy.deepcopy(parameters),
        "input_run_ids": copy.deepcopy(inputs),
        "command": "test_track_motion_publication",
        "fisheye_version": None,
    }
    run.attrs.update(
        {
            "inputs": inputs,
            "fps": fps,
            "smoothing_seconds": smooth_seconds,
            "smoothing_method": "moving_average",
            "smoothing_alignment": "centered",
            "savgol_polyorder": None,
            "distance_interpolation_seconds": 0.0,
            "hysteresis_enabled": hysteresis_enabled,
            "hysteresis_high_px": 2.0 if hysteresis_enabled else None,
            "hysteresis_low_px": 1.0 if hysteresis_enabled else None,
            "hysteresis_min_frames": 3 if hysteresis_enabled else None,
            "hysteresis_band_policy": "reset",
            "provenance": stage_provenance,
            "run_provenance": run_provenance,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    if materializer_metadata:
        run.attrs.update(
            {
                "physical_storage_layout": {
                    "authority": "dense_motion_arrays",
                    "version": 1,
                },
                "cluster_output_staging": {
                    "status": "present_during_materialization",
                },
            }
        )
    monkeypatch.setattr(
        mod,
        "load_persisted_source_camera_position_surface",
        lambda _root, path: source if path == "crop_runs/c1" else None,
    )
    sealed = mod._seal_and_load_track_motion_run_before_selection(root, run)
    result = root, run, run["tracks/id_7"], sealed, physical_authority
    if _return_template_source:
        return (*result, source)
    return result


@dataclass(frozen=True)
class _FullMotionRunTemplate:
    """One sealed writer result whose mutable archive graph is never exposed."""

    root: object
    sealed: mod.BoundTrackMotionRun
    source_surface: object


def _motion_run_template(
    monkeypatch: pytest.MonkeyPatch,
    **writer_options: object,
) -> _FullMotionRunTemplate:
    root, _run, _track, sealed, _physical, source = _fresh_full_motion_run(
        monkeypatch,
        _return_template_source=True,
        **writer_options,
    )
    return _FullMotionRunTemplate(
        root=root,
        sealed=sealed,
        source_surface=source,
    )


def _clone_motion_run_template(
    template: _FullMotionRunTemplate,
    monkeypatch: pytest.MonkeyPatch,
):
    # Bound coordinate authorities carry process-local opaque seals.  Preserve
    # those identity-only objects and immutable mapping proxies while copying
    # the source binding together with the archive graph, so its array nodes
    # are redirected to the clone rather than back to the template.
    memo: dict[int, object] = {}
    _seed_template_clone_memo(
        (template.root, template.source_surface),
        memo=memo,
        seen=set(),
    )
    root, source = copy.deepcopy(
        (template.root, template.source_surface),
        memo=memo,
    )
    run = root["analysis"]["track_kinematics_runs"]["offline"][
        template.sealed.position_bindings.run_name
    ]
    monkeypatch.setattr(
        mod,
        "load_persisted_source_camera_position_surface",
        lambda _root, path: source if path == "crop_runs/c1" else None,
    )
    sealed = mod._load_bound_track_motion_run_impl(
        root,
        run,
        expected_selector_eligible=False,
    )
    return (
        root,
        run,
        run["tracks/id_7"],
        sealed,
        sealed.position_bindings.physical_authority,
    )


def _seed_template_clone_memo(
    value: object,
    *,
    memo: dict[int, object],
    seen: set[int],
) -> None:
    """Keep opaque verification seals stable while cloning their bindings."""

    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if type(value) is object or isinstance(value, MappingProxyType):
        memo[identity] = value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _seed_template_clone_memo(key, memo=memo, seen=seen)
            _seed_template_clone_memo(item, memo=memo, seen=seen)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _seed_template_clone_memo(item, memo=memo, seen=seen)
        return
    try:
        attributes = vars(value)
    except TypeError:
        return
    for item in attributes.values():
        _seed_template_clone_memo(item, memo=memo, seen=seen)


@lru_cache(maxsize=2)
def _cached_motion_template(*, physical: bool) -> _FullMotionRunTemplate:
    """Build each meaningful sealed template family at most once per worker."""

    monkeypatch = pytest.MonkeyPatch()
    try:
        return _motion_run_template(monkeypatch, physical=physical)
    finally:
        monkeypatch.undo()


def _clone_full_motion_run(
    monkeypatch: pytest.MonkeyPatch,
):
    return _clone_motion_run_template(
        _cached_motion_template(physical=False),
        monkeypatch,
    )


def _clone_physical_motion_run(
    monkeypatch: pytest.MonkeyPatch,
):
    return _clone_motion_run_template(
        _cached_motion_template(physical=True),
        monkeypatch,
    )


def _restore_publication_attrs(run, manifest, digest, commit) -> None:
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR] = copy.deepcopy(manifest)
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR] = digest
    run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR] = copy.deepcopy(commit)


def _decoded_payload_receipt_fixture(*, root_sha256: str = "8" * 64):
    return {
        "canonicalization": "test.decoded_payload.v1",
        "array_count": 1,
        "decoded_bytes": 16,
        "root_sha256": root_sha256,
        "arrays": [{}],
    }


def test_staged_scientific_validation_binds_full_numeric_check_to_decoded_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_UNBOUND_STAGE_STATUS
    )
    run.attrs[mod.TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR] = "7" * 64

    receipt = mod.build_track_motion_staged_scientific_validation(
        run,
        decoded_payload_receipt=_decoded_payload_receipt_fixture(),
        run_name=str(run.path).rsplit("/", 1)[-1],
    )

    assert receipt["result"] == "valid"
    assert receipt["decoded_payload"]["root_sha256"] == "8" * 64
    assert receipt["validated_tracks"] == [{"track_id": 7, "sample_count": 2}]
    assert len(receipt["record_sha256"]) == 64
    run.attrs[mod.TRACK_MOTION_STAGED_SCIENTIFIC_VALIDATION_ATTR] = receipt
    root_attrs = mod._motion_run_root_attrs_record(run, _sealed.position_bindings)
    assert root_attrs["record"]["immutable_attrs"][
        mod.TRACK_MOTION_STAGED_SCIENTIFIC_VALIDATION_ATTR
    ] == receipt


def test_staged_scientific_validation_v2_binds_manifest_value_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_UNBOUND_STAGE_STATUS
    )
    run.attrs[mod.TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR] = "7" * 64
    staging_manifest = copy.deepcopy(
        run.attrs.get(mod.TRACK_KINEMATICS_STAGING_MANIFEST_ATTR, {})
    )
    staging_manifest["physical_authority"] = None
    run.attrs[mod.TRACK_KINEMATICS_STAGING_MANIFEST_ATTR] = staging_manifest

    live_nodes = {
        str(name): run[str(name)] for name in run.array_keys()
    }
    for track_id, track_group in mod._live_track_groups(run):
        for relative_path, node in mod._iter_track_array_nodes(track_group):
            live_nodes[f"tracks/id_{track_id}/{relative_path}"] = node
    arrays = []
    for path, node in sorted(live_nodes.items()):
        values = np.ascontiguousarray(np.asarray(node[:]))
        arrays.append(
            {
                "path": path,
                "dtype": str(node.dtype),
                "shape": [int(value) for value in node.shape],
                "decoded_bytes": int(values.nbytes),
                "content_sha256": hashlib.sha256(
                    values.tobytes(order="C")
                ).hexdigest(),
            }
        )
    decoded_bytes = sum(int(record["decoded_bytes"]) for record in arrays)
    content_body = {
        "schema_id": "palette.decoded_array_content_inventory",
        "schema_version": 1,
        "canonicalization": DECODED_PAYLOAD_CANONICALIZATION,
        "decoded_payload_root_sha256": "8" * 64,
        "array_count": len(arrays),
        "decoded_bytes": decoded_bytes,
        "arrays": arrays,
        "inventory_sha256": mod._canonical_json_sha256(arrays),
    }
    content_inventory = {
        **content_body,
        "record_sha256": mod._canonical_json_sha256(content_body),
    }
    decoded_receipt = {
        "canonicalization": DECODED_PAYLOAD_CANONICALIZATION,
        "array_count": len(arrays),
        "decoded_bytes": decoded_bytes,
        "root_sha256": "8" * 64,
        "arrays": [{} for _ in arrays],
    }

    receipt = mod.build_track_motion_staged_scientific_validation(
        run,
        decoded_payload_receipt=decoded_receipt,
        decoded_content_inventory=content_inventory,
        run_name=str(run.path).rsplit("/", 1)[-1],
    )

    assert receipt["schema_version"] == 2
    assert receipt["decoded_content_inventory"] == content_inventory
    assert receipt["publication_value_validation"]["result"] == "valid"
    assert receipt["publication_value_validation"]["array_count"] == len(arrays)

    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_BOUND_CANONICAL_STATUS
    )
    run.attrs[mod.TRACK_MOTION_STAGED_SCIENTIFIC_VALIDATION_ATTR] = receipt
    original_payload_sha256 = mod.array_payload_sha256

    def reject_rehash_of_published_motion(node):
        path = str(getattr(node, "path", ""))
        if path == str(run.path) or path.startswith(f"{run.path}/"):
            raise AssertionError(f"unexpected authoritative payload rehash: {path}")
        return original_payload_sha256(node)

    monkeypatch.setattr(
        mod,
        "array_payload_sha256",
        reject_rehash_of_published_motion,
    )
    rebuilt = mod._build_track_motion_publication_manifest(
        root,
        run,
        sealed.position_bindings,
        prevalidated_staged_scientific_validation=receipt,
    )
    assert rebuilt["track_count"] == 1


def test_staged_scientific_validation_rejects_wrong_numeric_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_UNBOUND_STAGE_STATUS
    )
    run.attrs[mod.TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR] = "7" * 64
    track["speed_raw_px"].data[1] += np.float32(5.0)

    with pytest.raises(ValueError, match="numeric derivation invariant"):
        mod.build_track_motion_staged_scientific_validation(
            run,
            decoded_payload_receipt=_decoded_payload_receipt_fixture(),
            run_name=str(run.path).rsplit("/", 1)[-1],
        )


def test_staged_scientific_validation_rejects_another_installed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs[mod.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR] = (
        mod.TRACK_KINEMATICS_UNBOUND_STAGE_STATUS
    )
    run.attrs[mod.TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR] = "7" * 64
    receipt = mod.build_track_motion_staged_scientific_validation(
        run,
        decoded_payload_receipt=_decoded_payload_receipt_fixture(),
        run_name=str(run.path).rsplit("/", 1)[-1],
    )
    run.attrs[mod.TRACK_MOTION_STAGED_SCIENTIFIC_VALIDATION_ATTR] = receipt
    monkeypatch.setattr(
        mod,
        "canonical_payload_integrity_receipt",
        lambda _value: {
            "decoded_payload": _decoded_payload_receipt_fixture(
                root_sha256="6" * 64
            )
        },
    )

    with pytest.raises(ValueError, match="another decoded payload"):
        mod.verify_track_motion_staged_scientific_validation(
            run,
            receipt,
            payload_integrity_receipt={},
        )


def test_seal_reuses_one_position_binding_within_same_publication_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    original = mod._load_bound_track_position_bindings_before_selection
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        mod,
        "_load_bound_track_position_bindings_before_selection",
        counted,
    )

    rebound = mod._seal_and_load_track_motion_run_before_selection(root, run)

    assert calls == 1
    assert rebound.position_bindings.run_group.path == run.path


def _replace_motion_derivation_inputs(run, inputs: dict[str, object]) -> None:
    """Replace every duplicated derivation input copy coherently."""

    copied = copy.deepcopy(inputs)
    run.attrs["inputs"] = copied
    run.attrs["source_refs"] = mod._track_kinematics_source_refs(
        run_type="offline",
        inputs=copied,
        publication_schema_version=(
            mod._track_motion_publication_schema_version(run)
        ),
    )
    stage_provenance = copy.deepcopy(run.attrs["provenance"])
    stage_provenance["inputs"] = copy.deepcopy(copied)
    run.attrs["provenance"] = stage_provenance
    run_provenance = copy.deepcopy(run.attrs["run_provenance"])
    run_provenance["input_run_ids"] = copy.deepcopy(copied)
    run.attrs["run_provenance"] = run_provenance


def _expected_public_pixel_surface_input_refs(track_path: str) -> dict[str, list[str]]:
    def ref(name: str) -> str:
        return f"/{track_path}/{name}"

    run_ref = "#/run_derivation"
    source_position = "#/source_authority/position"
    source_time = "#/source_authority/temporal"
    expected = {
        "frame_indices": [ref("source_acquisition_frame_index")],
        "track_sample_key": [
            ref("source_acquisition_frame_index"),
            f"/{track_path}@track_id",
        ],
        "source_acquisition_frame_index": [ref("source_row_index"), source_time],
        "source_frame_interpolation": [ref("source_row_index"), source_time],
        "source_instance_key": [ref("source_row_index"), source_time],
        "source_row_index": [source_position, source_time],
        "time_seconds": [ref("source_acquisition_frame_index"), run_ref],
        "positions_px": [source_position, ref("source_row_index")],
        "heading_degrees": [
            ref("source_row_index"),
            "#/input_authority/fields/heading_degrees",
        ],
        "heading_radians": [ref("heading_degrees")],
        "smoothed_heading_degrees": [ref("smoothed_heading_radians")],
        "smoothed_heading_radians": [ref("heading_radians"), run_ref],
        "keypoint_success": [
            ref("source_row_index"),
            "#/input_authority/fields/keypoint_success",
        ],
        "detection_source": [
            ref("source_row_index"),
            "#/input_authority/fields/detection_source",
        ],
        "sample_observed": [
            f"/{track_path}@track_id",
            "#/input_authority/fields/track_id",
        ],
        "sample_valid": [
            ref("sample_observed"),
            ref("source_observed"),
            ref("keypoint_usable"),
            ref("position_finite"),
        ],
        "source_observed": [ref("detection_source")],
        "keypoint_usable": [ref("keypoint_success"), ref("heading_degrees")],
        "position_finite": [ref("positions_px")],
        "heading_usable": [ref("heading_degrees"), ref("keypoint_success")],
        "sample_reason_code": [
            ref("sample_valid"),
            ref("sample_observed"),
            ref("source_observed"),
            ref("position_finite"),
            ref("heading_usable"),
            ref("keypoint_success"),
        ],
        "cumulative_path_distance_px": [
            ref("movement/speed/smoothed/frame_path_distance_px")
        ],
        "delta_heading_degrees": [
            ref("heading_degrees"),
            ref("delta_seconds"),
            ref("transition_valid"),
            ref("sample_valid"),
        ],
        "angular_velocity_raw_deg_s": [
            ref("delta_heading_degrees"),
            ref("delta_seconds"),
        ],
        "angular_speed_raw_deg_s": [ref("angular_velocity_raw_deg_s")],
        "delta_heading_smoothed_degrees": [
            ref("smoothed_heading_degrees"),
            ref("delta_seconds"),
            ref("transition_valid"),
            ref("sample_valid"),
        ],
        "angular_velocity_smoothed_deg_s": [
            ref("delta_heading_smoothed_degrees"),
            ref("delta_seconds"),
        ],
        "angular_speed_smoothed_deg_s": [ref("angular_velocity_smoothed_deg_s")],
        "angular_velocity_deg_s": [ref("angular_velocity_raw_deg_s")],
        "delta_frames": [ref("source_acquisition_frame_index")],
        "delta_seconds": [ref("delta_frames"), run_ref],
        "transition_valid": [
            ref("delta_frames"),
            ref("delta_seconds"),
            ref("positions_px"),
        ],
        "transition_reason_code": [
            ref("transition_valid"),
            ref("delta_frames"),
            ref("delta_seconds"),
            ref("positions_px"),
        ],
        "second_indices": [ref("source_acquisition_frame_index"), run_ref],
        "speed_per_second_px": [
            ref("movement/speed/smoothed/frame_path_distance_px"),
            ref("delta_seconds"),
            ref("second_indices"),
            ref("source_acquisition_frame_index"),
            run_ref,
        ],
        "heading_per_second_degrees": [
            ref("heading_radians"),
            ref("second_indices"),
            ref("source_acquisition_frame_index"),
            run_ref,
        ],
        "heading_per_second_resultant": [
            ref("heading_radians"),
            ref("second_indices"),
            ref("source_acquisition_frame_index"),
            run_ref,
        ],
    }
    for source_level, group_level in mod.MOVEMENT_SPEED_LEVEL_NAMES.items():
        grouped = f"movement/speed/{group_level}/px"
        expected[grouped] = (
            [
                ref(f"movement/speed/{group_level}/frame_path_distance_px"),
                ref("delta_seconds"),
            ]
            if group_level != "averaged"
            else [ref("movement/speed/smoothed/px"), run_ref]
        )
        expected[f"{source_level}_px"] = [ref(grouped)]
        derivative = f"speed_derivatives/{source_level}/acceleration_px"
        smoothed_derivative = (
            f"speed_derivatives/{source_level}/smoothed_acceleration_px"
        )
        expected[derivative] = [ref(f"{source_level}_px"), ref("delta_seconds")]
        expected[smoothed_derivative] = [ref(derivative), run_ref]
        expected[f"movement/speed/{group_level}/acceleration_px"] = [ref(derivative)]
        expected[f"movement/speed/{group_level}/smoothed_acceleration_px"] = [
            ref(smoothed_derivative)
        ]
    expected.update(
        {
            "movement/speed/raw/frame_path_distance_px": [
                ref("positions_px"),
                ref("transition_valid"),
            ],
            "movement/speed/filtered/frame_path_distance_px": [
                ref("movement/speed/raw/frame_path_distance_px"),
                run_ref,
            ],
            "movement/speed/smoothed/frame_path_distance_px": [
                ref("movement/speed/filtered/frame_path_distance_px"),
                ref("transition_valid"),
                run_ref,
            ],
            "frame_path_distance_raw_px": [
                ref("movement/speed/raw/frame_path_distance_px")
            ],
            "frame_path_distance_filtered_px": [
                ref("movement/speed/filtered/frame_path_distance_px")
            ],
            "frame_path_distance_smoothed_px": [
                ref("movement/speed/smoothed/frame_path_distance_px")
            ],
            "acceleration_px": [
                ref(
                    "speed_derivatives/"
                    f"{mod.DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/acceleration_px"
                )
            ],
            "smoothed_acceleration_px": [
                ref(
                    "speed_derivatives/"
                    f"{mod.DEFAULT_ACCELERATION_SOURCE_SPEED_LEVEL}/"
                    "smoothed_acceleration_px"
                )
            ],
        }
    )
    return expected


def test_fresh_full_motion_writer_seal_round_trips_domains_and_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track_group, sealed, _physical = _fresh_full_motion_run(monkeypatch)
    template = _cached_motion_template(physical=False)

    # Writer/sealing coverage must remain independent from the cached reader
    # template, including its mutable archive graph and minted authority.
    assert root is not template.root
    assert run is not template.sealed.run_group
    assert sealed is not template.sealed
    assert root._coordinate_archive_token is not template.root._coordinate_archive_token

    assert isinstance(sealed, mod.BoundTrackMotionRun)
    assert isinstance(sealed.manifest, MappingProxyType)
    assert sealed.manifest["schema_version"] == 1
    assert mod.TRACK_MOTION_PUBLICATION_SCHEMA_VERSION_ATTR not in run.attrs
    assert "position_lineage_mode" not in sealed.manifest
    assert "position_lineage" not in sealed.manifest["source_authority"]
    assert "schema_id" not in sealed.manifest["run_derivation"]
    assert "schema_version" not in sealed.manifest["run_derivation"]
    track = sealed.track(7)
    assert track.surface("time_seconds").axis0_domain == (
        mod.TRACK_MOTION_AXIS_TRACK_SAMPLE
    )
    assert track.surface("movement/speed/raw/px").axis0_domain == (
        mod.TRACK_MOTION_AXIS_TRACK_TRANSITION
    )
    assert track.surface("movement/speed/raw/px").units == "px/s"
    assert track.surface("speed_per_second_px").axis0_domain == (
        mod.TRACK_MOTION_AXIS_TRACK_SECOND
    )
    flat = track.surface("speed_raw_px")
    assert flat.operation_id == "exact_alias_v1"
    assert flat.alias_of == f"/{track.track_group.path}/movement/speed/raw/px"
    assert all(
        value["kind"] in {"array", "group_attr", "manifest_record", "external_lineage"}
        for value in flat.input_refs
    )
    with pytest.raises(TypeError):
        sealed.manifest["run_name"] = "forged"  # type: ignore[index]
    with pytest.raises(ValueError, match="minted by the live loader"):
        mod.BoundTrackMotionSurface(
            relative_path=flat.relative_path,
            axis0_domain=flat.axis0_domain,
            units=flat.units,
            semantic_profile=flat.semantic_profile,
            operation_id=flat.operation_id,
            input_refs=flat.input_refs,
            alias_of=flat.alias_of,
            dtype=flat.dtype,
            shape=flat.shape,
            content_sha256=flat.content_sha256,
            node=flat.node,
        )
    with pytest.raises(ValueError, match="minted by the live loader"):
        mod.BoundTrackMotionTrack(
            track_id=track.track_id,
            position_binding=track.position_binding,
            surfaces=track.surfaces,
            track_group=track.track_group,
        )
    with pytest.raises(ValueError, match="minted by the live loader"):
        mod.BoundTrackMotionRun(
            position_bindings=sealed.position_bindings,
            manifest_sha256=sealed.manifest_sha256,
            manifest={},
            tracks=(),
            run_group=run,
            authoritative_root=root,
            expected_selector_eligible=False,
        )

    sealed.assert_verified()
    run.attrs["stage_selector_eligible"] = True
    public = mod.load_bound_track_motion_run(root, run)
    public.assert_verified()


def test_motion_template_clones_isolate_mutations_from_template_and_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, first_run, first_track, first_sealed, _ = _clone_full_motion_run(
        monkeypatch
    )
    second_root, second_run, second_track, second_sealed, _ = _clone_full_motion_run(
        monkeypatch
    )
    template = _cached_motion_template(physical=False)
    template_run = template.sealed.run_group
    template_track = template_run["tracks/id_7"]
    template_speed = template_track["speed_raw_px"].data.copy()
    template_heading = template.root["keypoints_runs"]["kp_1"]["heading"].data.copy()

    assert first_root is not second_root
    assert first_root is not template.root
    assert first_run is not second_run
    assert first_sealed is not second_sealed
    assert first_sealed is not template.sealed
    assert second_sealed is not template.sealed
    assert (
        first_sealed.track(7).surface("speed_raw_px").node
        is first_track["speed_raw_px"]
    )
    assert (
        second_sealed.track(7).surface("speed_raw_px").node
        is second_track["speed_raw_px"]
    )
    assert (
        first_sealed.position_bindings.source_positions.coordinate_node
        is first_root["crop_runs"]["c1"]["centers_img_xy"]
    )
    assert (
        second_sealed.position_bindings.source_positions.coordinate_node
        is second_root["crop_runs"]["c1"]["centers_img_xy"]
    )

    first_run.attrs["inputs"]["crop_run"] = "mutated_clone"
    first_track["speed_raw_px"].data[1] += np.float32(1.0)
    first_root["keypoints_runs"]["kp_1"]["heading"].data[0] = np.float32(17.0)

    assert second_run.attrs["inputs"]["crop_run"] == "c1"
    assert template_run.attrs["inputs"]["crop_run"] == "c1"
    np.testing.assert_array_equal(second_track["speed_raw_px"][:], template_speed)
    np.testing.assert_array_equal(template_track["speed_raw_px"][:], template_speed)
    np.testing.assert_array_equal(
        second_root["keypoints_runs"]["kp_1"]["heading"][:],
        template_heading,
    )
    np.testing.assert_array_equal(
        template.root["keypoints_runs"]["kp_1"]["heading"][:],
        template_heading,
    )


def test_every_public_pixel_surface_has_exact_kernel_input_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, _run, track_group, sealed, _physical = _clone_full_motion_run(monkeypatch)
    records = sealed.manifest["tracks"]["id_7"]["surfaces"]
    expected = _expected_public_pixel_surface_input_refs(track_group.path)
    expected_paths = mod._expected_motion_track_surface_paths(include_physical=False)

    assert set(expected) == expected_paths
    assert set(records) == expected_paths
    for path, exact_refs in expected.items():
        assert [item["ref"] for item in records[path]["input_refs"]] == exact_refs
    assert records["heading_degrees"]["operation_id"] == (
        "float32_source_heading_subset_reorder_v1"
    )


def test_full_motion_loader_rejects_recomputed_manifest_and_live_attacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    baseline_manifest = copy.deepcopy(
        run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR]
    )
    baseline_digest = run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR]
    baseline_commit = copy.deepcopy(run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR])
    mod.load_bound_track_motion_run(root, run)

    canonical = track["movement/speed/raw/px"]
    canonical_data = canonical.data.copy()
    canonical.data[1] = np.float32(canonical.data[1] + 0.25)
    forged = copy.deepcopy(baseline_manifest)
    forged["tracks"]["id_7"]["surfaces"]["movement/speed/raw/px"]["content_sha256"] = (
        mod.array_payload_sha256(canonical)
    )
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR] = forged
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR] = (
        mod._canonical_json_sha256(forged)
    )
    run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR] = (
        mod._track_motion_publication_commit(forged)
    )
    with pytest.raises(ValueError, match="alias target"):
        mod.load_bound_track_motion_run(root, run)
    canonical.data = canonical_data
    _restore_publication_attrs(run, baseline_manifest, baseline_digest, baseline_commit)

    attacks = (
        ("axis0_domain", mod.TRACK_MOTION_AXIS_TRACK_SAMPLE),
        ("units", "mm/s"),
        ("operation_id", "divide_by_pixels_per_mm_v1"),
        (
            "input_refs",
            [
                {
                    "kind": "array",
                    "ref": f"/{track.path}/positions_typo",
                    "dtype": "<f4",
                    "shape": [2, 2],
                    "content_sha256": "0" * 64,
                }
            ],
        ),
    )
    for field, value in attacks:
        forged = copy.deepcopy(baseline_manifest)
        forged["tracks"]["id_7"]["surfaces"]["movement/speed/raw/px"][field] = value
        run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR] = forged
        run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR] = (
            mod._canonical_json_sha256(forged)
        )
        run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR] = (
            mod._track_motion_publication_commit(forged)
        )
        with pytest.raises(ValueError, match="differs from the exact live"):
            mod.load_bound_track_motion_run(root, run)
        _restore_publication_attrs(
            run, baseline_manifest, baseline_digest, baseline_commit
        )

    speed = track["speed_raw_px"]
    speed_original = speed.data.copy()
    speed.data = speed.data[:-1]
    speed.shape = speed.data.shape
    with pytest.raises(ValueError, match="axis-0 length|alias target"):
        mod.load_bound_track_motion_run(root, run)
    speed.data = speed_original
    speed.shape = speed.data.shape

    extra = track.create_array(
        "mystery_speed_px",
        data=np.zeros(2, dtype=np.float32),
    )
    assert extra is track["mystery_speed_px"]
    with pytest.raises(ValueError, match="extra=.*mystery_speed_px"):
        mod.load_bound_track_motion_run(root, run)
    del track.children["mystery_speed_px"]

    missing = track.children.pop("speed_raw_px")
    with pytest.raises(ValueError, match="missing=.*speed_raw_px"):
        mod.load_bound_track_motion_run(root, run)
    track.children["speed_raw_px"] = missing
    mod.load_bound_track_motion_run(root, run)


def test_full_motion_physical_surfaces_use_nonunit_multiply_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track_group, _sealed, physical = _clone_physical_motion_run(monkeypatch)
    assert physical is not None
    assert physical.mm_per_pixel != 1.0
    run.attrs["stage_selector_eligible"] = True
    bound = mod.load_bound_track_motion_run(root, run)
    track = bound.track(7)
    for physical_path in (
        "positions_mm",
        "movement/speed/raw/mm",
        "movement/speed/raw/frame_path_distance_mm",
        "speed_derivatives/speed_raw/acceleration_mm",
        "cumulative_path_distance_mm",
        "speed_per_second_mm",
    ):
        surface = track.surface(physical_path)
        assert surface.units.startswith("mm")
        record = bound.manifest["tracks"]["id_7"]["surfaces"][physical_path]
        assert record["physical_value_comparison"]["rtol"] == 0.0
        assert record["physical_value_comparison"]["atol"] == 0.0

    mm = track_group["movement/speed/raw/mm"]
    original = mm.data.copy()
    px = track_group["movement/speed/raw/px"].data
    mm.data = np.asarray(px / physical.mm_per_pixel, dtype=px.dtype)
    with pytest.raises(ValueError, match="multiplied|physical"):
        mod.load_bound_track_motion_run(root, run)
    mm.data = original


def test_motion_commit_rejects_recomputed_group_and_root_auxiliary_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _fresh_full_motion_run(
        monkeypatch,
        root_auxiliary=True,
    )
    run.attrs["stage_selector_eligible"] = True
    baseline_manifest = copy.deepcopy(
        run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR]
    )
    baseline_digest = run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR]
    baseline_commit = copy.deepcopy(run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR])

    level = track["movement/speed/raw"]
    original_method = level.attrs["derivative_method"]
    level.attrs["derivative_method"] = "divide_by_wrong_input"
    positions = mod.load_bound_track_position_bindings(root, run)
    with pytest.raises(ValueError, match="group attr.*controlled"):
        mod._build_track_motion_publication_manifest(root, run, positions)
    level.attrs["derivative_method"] = original_method
    _restore_publication_attrs(run, baseline_manifest, baseline_digest, baseline_commit)

    trial_state = run["trial_state"]
    original_trial_state = trial_state.data.copy()
    trial_state.data[0] = np.int16(7)
    positions = mod.load_bound_track_position_bindings(root, run)
    forged = mod._build_track_motion_publication_manifest(root, run, positions)
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR] = forged
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR] = (
        mod._canonical_json_sha256(forged)
    )
    with pytest.raises(ValueError, match="publication commit"):
        mod.load_bound_track_motion_run(root, run)
    trial_state.data = original_trial_state
    _restore_publication_attrs(run, baseline_manifest, baseline_digest, baseline_commit)

    del track.attrs["track_id"]
    with pytest.raises(ValueError, match="track-root group attr inventory"):
        mod.load_bound_track_motion_run(root, run)


def test_full_motion_loader_rejects_recomputed_physical_authority_attack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, physical = _clone_physical_motion_run(monkeypatch)
    assert physical is not None
    run.attrs["stage_selector_eligible"] = True
    public = mod.load_bound_track_motion_run(root, run)
    forged = copy.deepcopy(run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR])
    forged["physical_authority"]["mm_per_pixel"] = float(physical.mm_per_pixel) * 2.0
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR] = forged
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR] = (
        mod._canonical_json_sha256(forged)
    )
    run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR] = (
        mod._track_motion_publication_commit(forged)
    )
    with pytest.raises(ValueError, match="exact live|manifest"):
        mod.load_bound_track_motion_run(root, run)

    # A previously returned authority also performs a fresh live check.
    with pytest.raises(ValueError):
        public.assert_verified()


def test_mirrored_bout_payload_is_sealed_but_not_public_motion_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track_group, sealed, _physical = _fresh_full_motion_run(
        monkeypatch,
        bout_auxiliary=True,
    )
    bout_record = sealed.manifest["tracks"]["id_7"]["surfaces"][
        "swim_bouts/start_frame"
    ]
    assert bout_record["authority_scope"] == ("sealed_auxiliary_not_motion_public")
    with pytest.raises(KeyError):
        sealed.track(7).surface("swim_bouts/start_frame")

    run.attrs["stage_selector_eligible"] = True
    baseline_commit = copy.deepcopy(run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR])
    start = track_group["swim_bouts/start_frame"]
    start.data[0] = np.int32(12)
    positions = mod.load_bound_track_position_bindings(root, run)
    forged = mod._build_track_motion_publication_manifest(root, run, positions)
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_ATTR] = forged
    run.attrs[mod.TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR] = (
        mod._canonical_json_sha256(forged)
    )
    run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR] = baseline_commit
    with pytest.raises(ValueError, match="publication commit"):
        mod.load_bound_track_motion_run(root, run)


def test_full_motion_fps_60_publishes_unique_second_bin_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _fresh_full_motion_run(
        monkeypatch,
        fps=60.0,
        smooth_seconds=1.0 / 60.0,
    )
    seconds = np.asarray(track["second_indices"][:])
    assert seconds.dtype == np.dtype("<i8")
    assert seconds.tolist() == [0]
    assert track["speed_per_second_px"].shape == (1,)
    assert track["heading_per_second_degrees"].shape == (1,)
    run.attrs["stage_selector_eligible"] = True
    mod.load_bound_track_motion_run(root, run)


def test_track_builder_fps_60_uses_one_key_per_observed_second() -> None:
    frames = np.asarray([0, 59, 60, 61, 120], dtype=np.int64)
    tracks, _summaries = mod.build_track_datasets(
        track_ids=np.full(frames.shape, 7, dtype=np.int64),
        frames=frames,
        positions_px=np.column_stack(
            [frames.astype(np.float32), np.zeros(frames.shape, dtype=np.float32)]
        ),
        headings_deg=np.asarray([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        keypoint_success=np.ones(frames.shape, dtype=bool),
        detection_source=None,
        fps=60.0,
        smooth_seconds=1.0 / 60.0,
        pixel_to_mm=None,
    )

    track = tracks[7]
    assert track["second_indices"].tolist() == [0, 1, 2]
    assert track["speed_per_second_px"].shape == (3,)
    assert track["heading_per_second_degrees"].shape == (3,)
    assert track["heading_per_second_resultant"].shape == (3,)


@pytest.mark.parametrize("sample_count", (1, 2))
@pytest.mark.parametrize("alignment", ("centered", "causal"))
def test_short_tracks_bound_every_smoothing_window(
    sample_count: int,
    alignment: str,
) -> None:
    frames = np.arange(sample_count, dtype=np.int64)
    tracks, _summaries = mod.build_track_datasets(
        track_ids=np.full(sample_count, 7, dtype=np.int64),
        frames=frames,
        positions_px=np.column_stack(
            (
                frames.astype(np.float64),
                np.zeros(sample_count, dtype=np.float64),
            )
        ),
        headings_deg=np.linspace(
            0.123456789,
            89.987654321,
            sample_count,
            dtype=np.float64,
        ),
        keypoint_success=np.ones(sample_count, dtype=bool),
        detection_source=None,
        fps=60.0,
        smooth_seconds=1.0,
        pixel_to_mm=None,
        smoothing_alignment=alignment,
    )

    track = tracks[7]
    assert track["speed_smoothed_px"].shape == (sample_count,)
    assert track["speed_averaged_px"].shape == (sample_count,)
    assert track["smoothed_acceleration_px"].shape == (sample_count,)
    assert track["smoothed_heading_degrees"].shape == (sample_count,)
    windows = track["motion_smoothing_windows"]
    assert windows["distance_transition"] == {
        "alignment": alignment,
        "requested_frames": 60,
        "effective_frames": max(0, sample_count - 1),
    }
    for domain in ("speed_sample", "acceleration_sample", "heading_sample"):
        assert windows[domain]["requested_frames"] == 60
        assert windows[domain]["effective_frames"] == sample_count


def test_heading_nan_gap_contributes_zero_circular_numerator() -> None:
    tracks, _summaries = mod.build_track_datasets(
        track_ids=np.asarray([7, 7], dtype=np.int64),
        frames=np.asarray([0, 1], dtype=np.int64),
        positions_px=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        headings_deg=np.asarray([90.0, np.nan], dtype=np.float64),
        keypoint_success=np.ones(2, dtype=bool),
        detection_source=None,
        fps=60.0,
        smooth_seconds=1.0,
        pixel_to_mm=None,
        smoothing_alignment="centered",
    )

    np.testing.assert_allclose(
        tracks[7]["smoothed_heading_degrees"],
        np.asarray([90.0, 90.0], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )


def test_float64_heading_input_and_disabled_hysteresis_writer_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _fresh_full_motion_run(
        monkeypatch,
        fps=60.0,
        smooth_seconds=0.2,
        headings_deg=np.asarray(
            [0.123456789012345, 179.98765432109876],
            dtype=np.float64,
        ),
        hysteresis_enabled=False,
    )

    assert track["heading_degrees"].dtype == np.dtype("<f4")
    assert run.attrs["parameters"]["hysteresis_band_policy"] == "reset"
    assert run.attrs["hysteresis_band_policy"] == "reset"
    run.attrs["stage_selector_eligible"] = True
    mod.load_bound_track_motion_run(root, run)


def test_physical_derivative_uses_finalized_pixel_rounding_path() -> None:
    derivative = mod._compute_speed_derivative(
        np.asarray([0.0, -89235796.23955546], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        pixel_to_mm=0.04,
        smooth_seconds=1.0,
        fps=60.0,
    )

    for px_name, mm_name in (
        ("acceleration_px", "acceleration_mm"),
        ("smoothed_acceleration_px", "smoothed_acceleration_mm"),
    ):
        pixel = np.asarray(derivative[px_name])
        physical = np.asarray(derivative[mm_name])
        expected = np.asarray(
            pixel * np.asarray(0.04, dtype=pixel.dtype),
            dtype=pixel.dtype,
        )
        np.testing.assert_array_equal(physical, expected)


def test_full_motion_rejects_unsealed_run_root_child_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    mystery = run.create_group("mystery_public")
    mystery.create_array(
        "positions_px",
        data=np.asarray([[999.0, 999.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="root group inventory is not closed"):
        mod.load_bound_track_motion_run(root, run)


def test_full_motion_rejects_array_directly_under_tracks_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    run["tracks"].create_array(
        "mystery_track_index",
        data=np.asarray([7], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="/tracks array inventory is not closed"):
        mod.load_bound_track_motion_run(root, run)


def test_full_motion_rejects_recomputed_equal_shape_role_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    delta = track["delta_seconds"]
    cumulative = track["cumulative_path_distance_px"]
    delta_data = delta.data.copy()
    cumulative_data = cumulative.data.copy()
    assert delta_data.dtype == cumulative_data.dtype
    assert delta_data.shape == cumulative_data.shape
    delta.data = cumulative_data.copy()
    cumulative.data = delta_data.copy()
    positions = mod.load_bound_track_position_bindings(root, run)
    with pytest.raises(ValueError, match="numeric derivation invariant"):
        mod._build_track_motion_publication_manifest(root, run, positions)


def test_full_motion_rejects_recomputed_equal_shape_second_role_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    heading = track["heading_per_second_degrees"]
    resultant = track["heading_per_second_resultant"]
    assert heading.dtype == resultant.dtype
    assert heading.shape == resultant.shape
    heading_data = heading.data.copy()
    resultant_data = resultant.data.copy()
    heading.data = resultant_data
    resultant.data = heading_data
    positions = mod.load_bound_track_position_bindings(root, run)

    with pytest.raises(ValueError, match="numeric derivation invariant"):
        mod._build_track_motion_publication_manifest(root, run, positions)


def test_full_motion_recomputes_summary_values_and_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    positions = mod.load_bound_track_position_bindings(root, run)

    original_summary = copy.deepcopy(track.attrs["summary"])
    track.attrs["summary"]["mean_speed_raw_px"] += 1.0
    with pytest.raises(ValueError, match="summary field.*numeric derivation"):
        mod._build_track_motion_publication_manifest(root, run, positions)
    track.attrs["summary"] = copy.deepcopy(original_summary)

    track.attrs["summary"]["alternate_speed_px"] = 1.0
    with pytest.raises(ValueError, match="summary inventory"):
        mod._build_track_motion_publication_manifest(root, run, positions)


def test_acceleration_summary_uses_persisted_float32_reduction_domain() -> None:
    values = np.linspace(-1000.0, 1000.0, 1_182_938, dtype=np.float32)

    mean, standard_deviation = mod._acceleration_summary_statistics(values)

    assert mean == float(np.mean(values))
    assert standard_deviation == float(np.std(values))
    assert mean != float(np.mean(values.astype(np.float64)))
    assert standard_deviation != float(np.std(values.astype(np.float64)))


def test_resultant_domain_allows_only_one_float32_boundary_step() -> None:
    one = np.float32(1.0)
    one_step = np.nextafter(one, np.float32(np.inf))
    two_steps = np.nextafter(one_step, np.float32(np.inf))

    assert mod._float32_resultants_within_unit_interval(
        np.asarray([0.0, one, one_step, np.nan], dtype=np.float32)
    )
    assert not mod._float32_resultants_within_unit_interval(
        np.asarray([two_steps], dtype=np.float32)
    )
    assert not mod._float32_resultants_within_unit_interval(
        np.asarray([-np.finfo(np.float32).tiny], dtype=np.float32)
    )


def test_smoothed_turning_uses_persisted_float32_heading_parent() -> None:
    smoothed = np.asarray(
        [45.034366607666016, 142.99696350097656, 99.2468490600586],
        dtype=np.float64,
    )
    delta_seconds = np.asarray([0.0, 1.0 / 30.0, 1.0 / 30.0])
    valid = np.ones(smoothed.shape, dtype=bool)

    observed = mod._compute_turning_from_persisted_smoothed_heading(
        smoothed,
        delta_seconds,
        transition_valid=valid,
        sample_valid=valid,
    )
    expected = mod._compute_heading_turning(
        smoothed.astype(np.float32),
        delta_seconds,
        transition_valid=valid,
        sample_valid=valid,
    )

    for observed_array, expected_array in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(observed_array, expected_array)


def test_motion_seal_accepts_materializer_storage_and_dynamic_staging_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _fresh_full_motion_run(
        monkeypatch,
        materializer_metadata=True,
    )
    run.attrs["stage_selector_eligible"] = True
    mod.load_bound_track_motion_run(root, run)

    # Staging diagnostics are explicitly operational and may change after the
    # immutable motion commit has been minted.
    run.attrs["cluster_output_staging"] = {"status": "cleaned"}
    mod.load_bound_track_motion_run(root, run)

    # The physical layout is immutable semantic/storage metadata and remains
    # part of the closed root attrs record.
    run.attrs["physical_storage_layout"] = {
        "authority": "different_layout",
        "version": 2,
    }
    with pytest.raises(ValueError, match="exact live|publication manifest"):
        mod.load_bound_track_motion_run(root, run)


def test_full_motion_loader_resolves_detached_handle_from_authoritative_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    detached = _WritableGroup(
        path=run.path,
        archive_token=run._coordinate_archive_token,
    )
    detached.attrs["stage_selector_eligible"] = False

    bound = mod.load_bound_track_motion_run(root, detached)

    assert bound.run_group is run


def test_full_motion_rejects_recomputed_validity_and_angle_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    positions = mod.load_bound_track_position_bindings(root, run)

    sample_valid = track["sample_valid"]
    original_valid = sample_valid.data.copy()
    sample_valid.data[0] = False
    with pytest.raises(ValueError, match="numeric derivation invariant"):
        mod._build_track_motion_publication_manifest(root, run, positions)
    sample_valid.data = original_valid

    heading_radians = track["heading_radians"]
    original_radians = heading_radians.data.copy()
    heading_radians.data[1] = np.float32(1.0)
    with pytest.raises(ValueError, match="degree/radian"):
        mod._build_track_motion_publication_manifest(root, run, positions)
    heading_radians.data = original_radians

    angular_speed = track["angular_speed_raw_deg_s"]
    original_speed = angular_speed.data.copy()
    angular_speed.data[1] = np.float32(3.0)
    with pytest.raises(ValueError, match="numeric derivation invariant"):
        mod._build_track_motion_publication_manifest(root, run, positions)
    angular_speed.data = original_speed


def test_full_motion_rejects_conflicting_root_and_array_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True

    run.attrs["coordinate_space"] = "texture"
    with pytest.raises(ValueError, match="unsupported track coordinate space"):
        mod.load_bound_track_motion_run(root, run)
    del run.attrs["coordinate_space"]

    track["delta_seconds"].attrs["units"] = "px"
    with pytest.raises(ValueError, match="array attr inventory is not closed"):
        mod.load_bound_track_motion_run(root, run)
    del track["delta_seconds"].attrs["units"]

    track["delta_seconds"].attrs["scientific_role"] = "position_x"
    with pytest.raises(ValueError, match="array attr inventory is not closed"):
        mod.load_bound_track_motion_run(root, run)
    del track["delta_seconds"].attrs["scientific_role"]

    frame_indices = track["frame_indices"]
    original_role = frame_indices.attrs["semantic_role"]
    frame_indices.attrs["semantic_role"] = "authoritative_acquisition_frame"
    with pytest.raises(ValueError, match="identity attr.*controlled"):
        mod.load_bound_track_motion_run(root, run)
    frame_indices.attrs["semantic_role"] = original_role

    run.attrs["coordinate_guess_from_range"] = "camera"
    with pytest.raises(ValueError, match="unsupported run-root attrs"):
        mod.load_bound_track_motion_run(root, run)


def test_manifest_rejects_position_source_ref_that_conflicts_with_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    source_refs = copy.deepcopy(run.attrs["source_refs"])
    source_refs["source_position_source_path"] = "crop_runs/decoy"
    run.attrs["source_refs"] = source_refs

    with pytest.raises(ValueError, match="mechanical projection of inputs"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_manifest_rejects_self_consistent_decoy_position_source_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs["position_source_path"] = "crop_runs/decoy"
    run.attrs["inputs"] = inputs
    run.attrs["source_refs"] = mod._track_kinematics_source_refs(
        run_type="offline",
        inputs=inputs,
    )
    stage_provenance = copy.deepcopy(run.attrs["provenance"])
    stage_provenance["inputs"] = copy.deepcopy(inputs)
    run.attrs["provenance"] = stage_provenance
    run_provenance = copy.deepcopy(run.attrs["run_provenance"])
    run_provenance["input_run_ids"] = copy.deepcopy(inputs)
    run.attrs["run_provenance"] = run_provenance

    with pytest.raises(ValueError, match="not the exact sealed position"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_manifest_rejects_self_consistent_missing_position_source_rowset_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    del inputs["position_source_rowset_path"]
    run.attrs["inputs"] = inputs
    run.attrs["source_refs"] = mod._track_kinematics_source_refs(
        run_type="offline",
        inputs=inputs,
    )
    stage_provenance = copy.deepcopy(run.attrs["provenance"])
    stage_provenance["inputs"] = copy.deepcopy(inputs)
    run.attrs["provenance"] = stage_provenance
    run_provenance = copy.deepcopy(run.attrs["run_provenance"])
    run_provenance["input_run_ids"] = copy.deepcopy(inputs)
    run.attrs["run_provenance"] = run_provenance

    with pytest.raises(ValueError, match="input inventory is not closed"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_geometry_path", "crop_runs/c1/centers_img_xy"),
        ("detection_run", "d1"),
        ("detection_variant", "raw"),
        ("source_detect_run", "d1"),
        ("source_arena_assignment_run", "arena_1"),
        ("keypoint_variant", "raw"),
        ("base_keypoint_run", "kp_1"),
        ("keypoint_usability_dataset", "heading_usable"),
        ("source_tracking_rowset_fingerprint", "forged"),
        ("tracking_metadata", {"source_rowset_path": "crop_runs/decoy"}),
    ],
)
def test_manifest_rejects_coherent_deprecated_lineage_aliases(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs[field] = value
    _replace_motion_derivation_inputs(run, inputs)

    with pytest.raises(ValueError, match="input inventory is not closed"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_manifest_rejects_coherent_legacy_position_source_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs["position_source_kind"] = "crop_rows_source_image_bbox"
    _replace_motion_derivation_inputs(run, inputs)

    with pytest.raises(ValueError, match="position source kind"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_manifest_rejects_coherent_crop_run_that_is_not_position_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs["crop_run"] = "decoy"
    _replace_motion_derivation_inputs(run, inputs)

    with pytest.raises(ValueError, match="crop_run does not identify"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_manifest_rejects_coherent_detection_path_outside_crop_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs["detection_path"] = "detect_runs/decoy"
    _replace_motion_derivation_inputs(run, inputs)

    with pytest.raises(ValueError, match="detection_path does not identify"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def _configure_v2_successor_run(root, run) -> tuple[str, str]:
    historical_rowset = "crop_runs/historical_collection"
    detection_run_id = "finalized_collection_proxy:collection_1"
    root["keypoints_runs"]["kp_1"].attrs.update(
        {
            "source_crop_run": "historical_collection",
            "source_detect_run": detection_run_id,
        }
    )
    root["tracking_runs"]["trk_1"].attrs.update(
        {
            "source_rowset_path": historical_rowset,
            "source_detect_run": detection_run_id,
        }
    )
    mapping_record = {
        "schema_id": mod.COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_ID,
        "schema_version": mod.COLLECTION_PROXY_SUCCESSOR_MAPPING_SCHEMA_VERSION,
        "operation": mod.COLLECTION_PROXY_SUCCESSOR_MAPPING_OPERATION,
        "historical_source": {"rowset_ref": f"/{historical_rowset}"},
    }
    stamp_and_bind_persisted_coordinate_record(
        root["crop_runs"]["c1"],
        mapping_record,
        attr_name=mod.COLLECTION_PROXY_SUCCESSOR_MAPPING_ATTR,
    )
    run.attrs[mod.TRACK_MOTION_PUBLICATION_SCHEMA_VERSION_ATTR] = (
        mod.TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION_V2
    )
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs.pop("detection_path")
    inputs.update(
        {
            "position_lineage_mode": (
                mod.TRACK_POSITION_LINEAGE_COLLECTION_PROXY_SUCCESSOR_V1
            ),
            "keypoint_source_crop_run": "historical_collection",
            "tracking_source_rowset_path": historical_rowset,
            "source_detection_run_id": detection_run_id,
        }
    )
    _replace_motion_derivation_inputs(run, inputs)
    return historical_rowset, detection_run_id


def _configure_v2_direct_run(root, run, positions) -> None:
    selection_records = [
        record
        for record in positions.source_positions.lineage_records
        if record.record.get("schema_id")
        == mod.CROP_GEOMETRY_SELECTION_SCHEMA_ID
    ]
    assert len(selection_records) == 1
    stamp_and_bind_persisted_coordinate_record(
        root["crop_runs"]["c1"],
        selection_records[0].record,
        attr_name=mod.CROP_GEOMETRY_SELECTION_ATTR,
    )
    run.attrs[mod.TRACK_MOTION_PUBLICATION_SCHEMA_VERSION_ATTR] = (
        mod.TRACK_MOTION_PUBLICATION_MANIFEST_SCHEMA_VERSION_V2
    )
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs["position_lineage_mode"] = mod.TRACK_POSITION_LINEAGE_DIRECT_CROP_V1
    _replace_motion_derivation_inputs(run, inputs)


def test_manifest_accepts_verified_merged_rowset_coordinate_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    historical_rowset, detection_run_id = _configure_v2_successor_run(
        root,
        run,
    )
    monkeypatch.setattr(
        mod,
        "load_collection_proxy_successor_source_rowset",
        lambda candidate_root, rowset_path: historical_rowset
        if candidate_root is root and rowset_path == "crop_runs/c1"
        else None,
    )

    manifest = mod._build_track_motion_publication_manifest(
        root,
        run,
        sealed.position_bindings,
    )

    refs = manifest["run_derivation"]["record"]["source_refs"]
    assert manifest["schema_version"] == 2
    assert manifest["run_derivation"]["schema_version"] == 2
    assert manifest["position_lineage_mode"] == (
        mod.TRACK_POSITION_LINEAGE_COLLECTION_PROXY_SUCCESSOR_V1
    )
    assert refs["source_detection_run_id"] == detection_run_id
    assert "source_detection_path" not in refs
    assert refs["source_tracking_rowset_path"] == historical_rowset
    lineage = manifest["source_authority"]["position_lineage"]
    assert lineage["historical_source_rowset_ref"] == f"/{historical_rowset}"
    assert lineage["successor_mapping"]["record_ref"] == (
        "/crop_runs/c1@collection_proxy_coordinate_successor_mapping"
    )
    assert len(lineage["successor_mapping"]["record_sha256"]) == 64


def test_manifest_accepts_unknown_keypoint_detection_with_exact_tracking_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    historical_rowset, detection_run_id = _configure_v2_successor_run(root, run)
    root["keypoints_runs"]["kp_1"].attrs["source_detect_run"] = "unknown"
    monkeypatch.setattr(
        mod,
        "load_collection_proxy_successor_source_rowset",
        lambda candidate_root, rowset_path: historical_rowset
        if candidate_root is root and rowset_path == "crop_runs/c1"
        else None,
    )

    manifest = mod._build_track_motion_publication_manifest(
        root,
        run,
        sealed.position_bindings,
    )

    assert (
        manifest["source_authority"]["position_lineage"][
            "source_detection_run_id"
        ]
        == detection_run_id
    )


def test_manifest_rejects_successor_tracking_detection_identity_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    historical_rowset, _detection_run_id = _configure_v2_successor_run(root, run)
    root["tracking_runs"]["trk_1"].attrs["source_detect_run"] = "decoy"
    monkeypatch.setattr(
        mod,
        "load_collection_proxy_successor_source_rowset",
        lambda candidate_root, rowset_path: historical_rowset
        if candidate_root is root and rowset_path == "crop_runs/c1"
        else None,
    )

    with pytest.raises(ValueError, match="conflicts with the verified successor"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_v2_direct_crop_seal_round_trips_through_versioned_reader_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, original, _physical = _clone_full_motion_run(monkeypatch)
    _configure_v2_direct_run(root, run, original.position_bindings)

    sealed = mod._seal_and_load_track_motion_run_before_selection(root, run)

    assert sealed.manifest["schema_version"] == 2
    assert sealed.manifest["position_lineage_mode"] == (
        mod.TRACK_POSITION_LINEAGE_DIRECT_CROP_V1
    )
    lineage = sealed.manifest["source_authority"]["position_lineage"]
    assert lineage["crop_selection"]["record_ref"] == (
        "/crop_runs/c1@crop_geometry_selection"
    )
    assert run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR]["schema_version"] == 2
    sealed.assert_verified()


def test_v2_successor_seal_round_trips_through_versioned_reader_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    historical_rowset, _detection_run_id = _configure_v2_successor_run(root, run)
    monkeypatch.setattr(
        mod,
        "load_collection_proxy_successor_source_rowset",
        lambda candidate_root, rowset_path: historical_rowset
        if candidate_root is root and rowset_path == "crop_runs/c1"
        else None,
    )

    sealed = mod._seal_and_load_track_motion_run_before_selection(root, run)

    assert sealed.manifest["schema_version"] == 2
    assert sealed.manifest["position_lineage_mode"] == (
        mod.TRACK_POSITION_LINEAGE_COLLECTION_PROXY_SUCCESSOR_V1
    )
    assert run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR] == (
        mod._track_motion_publication_commit(
            mod._thaw_motion_manifest(sealed.manifest)
        )
    )
    assert run.attrs[mod.TRACK_MOTION_PUBLICATION_COMMIT_ATTR]["schema_version"] == 2
    sealed.assert_verified()


def test_manifest_rejects_successor_that_maps_to_another_historical_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    _configure_v2_successor_run(root, run)
    monkeypatch.setattr(
        mod,
        "load_collection_proxy_successor_source_rowset",
        lambda _root, _rowset_path: "crop_runs/another_collection",
    )

    with pytest.raises(ValueError, match="mapping contract is invalid"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("keypoint_path", "keypoints_runs/kp_decoy", "selected keypoint run"),
        ("tracking_path", "tracking_runs/trk_decoy", "selected tracking run"),
    ],
)
def test_manifest_rejects_coherent_decoy_keypoint_or_tracking_path(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs[field] = value
    _replace_motion_derivation_inputs(run, inputs)

    with pytest.raises(ValueError, match=message):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("keypoint_path", "keypoints_runs/kp_1/heading"),
        ("keypoint_path", "/keypoints_runs/kp_1"),
        ("keypoint_path", "refined_keypoints_runs/kp_1/"),
        ("tracking_path", "tracking_runs/trk_1/track_ids"),
        ("tracking_path", "/tracking_runs/trk_1"),
        ("tracking_path", "tracking_runs/trk_1/"),
    ],
)
def test_offline_source_refs_reject_malformed_run_path_grammar(
    field: str,
    value: str,
) -> None:
    inputs = {
        "detection_path": "detect_runs/d1",
        "position_source_path": "crop_runs/c1/centers_img_xy",
        "position_source_rowset_path": "crop_runs/c1",
        "position_source_kind": "canonical_crop_rows_source_camera_centers",
        "keypoint_path": "keypoints_runs/kp_1",
        "crop_run": "c1",
        "tracking_path": "tracking_runs/trk_1",
    }
    inputs[field] = value

    with pytest.raises(ValueError, match=rf"{field}.*(?:controlled|canonical)"):
        mod._track_kinematics_source_refs(run_type="offline", inputs=inputs)


def test_manifest_rejects_coherent_duplicate_physical_calibration_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    parameters = copy.deepcopy(run.attrs["parameters"])
    parameters["physical_calibration"] = {
        "pixels_per_mm_projector": 5.0,
    }
    run.attrs["parameters"] = copy.deepcopy(parameters)
    stage_provenance = copy.deepcopy(run.attrs["provenance"])
    stage_provenance["parameters"] = copy.deepcopy(parameters)
    run.attrs["provenance"] = stage_provenance
    run_provenance = copy.deepcopy(run.attrs["run_provenance"])
    run_provenance["params"] = copy.deepcopy(parameters)
    run_provenance["config_hash"] = mod.sha256_payload(parameters)
    run.attrs["run_provenance"] = run_provenance

    with pytest.raises(ValueError, match="must not duplicate the typed physical"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


@pytest.mark.parametrize(
    "name",
    ("calibration_used", "texture_to_camera_scale", "tracking_metadata"),
)
def test_manifest_rejects_any_reintroduced_parameter_authority_alias(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    parameters = copy.deepcopy(run.attrs["parameters"])
    parameters[name] = 999
    run.attrs["parameters"] = copy.deepcopy(parameters)
    stage = copy.deepcopy(run.attrs["provenance"])
    stage["parameters"] = copy.deepcopy(parameters)
    run.attrs["provenance"] = stage
    final = copy.deepcopy(run.attrs["run_provenance"])
    final["params"] = copy.deepcopy(parameters)
    final["config_hash"] = mod.sha256_payload(parameters)
    run.attrs["run_provenance"] = final

    with pytest.raises(ValueError, match="parameter inventory is not closed"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_manifest_rejects_uncontrolled_tracks_parent_attrs_even_if_resealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    run["tracks"].attrs["physical_calibration"] = {"mm_per_pixel": 999.0}

    with pytest.raises(ValueError, match="/tracks attr inventory is not closed"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_manifest_rejects_reintroduced_root_physical_calibration_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["physical_calibration"] = {
        "pixels_per_mm_projector": 5.0,
    }

    with pytest.raises(ValueError, match="unsupported run-root attrs"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


@pytest.mark.parametrize(
    "value",
    ("/absolute", "../evil", "nested/evil", "evil/", 7, {"run": "evil"}),
)
def test_offline_swim_bout_input_rejects_nonleaf_values(
    value: object,
) -> None:
    inputs = {
        "detection_path": "detect_runs/d1",
        "position_source_path": "crop_runs/c1/centers_img_xy",
        "position_source_rowset_path": "crop_runs/c1",
        "position_source_kind": "canonical_crop_rows_source_camera_centers",
        "keypoint_path": "keypoints_runs/kp_1",
        "crop_run": "c1",
        "tracking_path": "tracking_runs/trk_1",
        "swim_bout_run": value,
    }

    with pytest.raises(ValueError, match="swim_bout_run.*run-name leaf"):
        mod._track_kinematics_source_refs(run_type="offline", inputs=inputs)


def test_manifest_rejects_chaser_auxiliary_legacy_nested_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs["chaser_metrics"] = {
        "metrics_run": "metrics_1",
        "stimulus_run": "stimulus_1",
        "chaser_index": 0,
        "distance_interpolation_seconds": 0.0,
        "coordinate_geometry_status": "not_present",
        "coordinate_geometry_reason_code": "NONE",
        "omitted_coordinate_fields": [],
        "tracking_metadata": {"physical_calibration": 999},
    }
    _replace_motion_derivation_inputs(run, inputs)

    with pytest.raises(
        ValueError, match="chaser_metrics input inventory is not closed"
    ):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_manifest_rejects_claimed_chaser_auxiliary_without_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs["chaser_metrics"] = {
        "metrics_run": "metrics_1",
        "stimulus_run": "stimulus_1",
        "chaser_index": 0,
        "distance_interpolation_seconds": 0.0,
        "coordinate_geometry_status": "not_present",
        "coordinate_geometry_reason_code": "NONE",
        "omitted_coordinate_fields": [],
    }
    _replace_motion_derivation_inputs(run, inputs)

    with pytest.raises(ValueError, match="without its exact sealed auxiliary arrays"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_online_input_inventory_rejects_legacy_authority_aliases() -> None:
    source_path = (
        "analysis/stimulus_runs/stim_1/tracking_data/chaser_states/target_position_xy"
    )
    inputs = {
        "stimulus_run": "stim_1",
        "chaser_index": 0,
        "positions_px_source_path": source_path,
        "positions_px_coordinate_descriptor_sha256": "a" * 64,
        "base_keypoint_run": "legacy",
        "physical_calibration": {"mm_per_pixel": 999},
    }

    with pytest.raises(ValueError, match="online track input inventory is not closed"):
        mod._validate_online_input_inventory(
            method="track_kinematics_online",
            inputs=inputs,
            source_path=source_path,
        )


def test_manifest_rejects_conflicting_legacy_position_source_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["positions_px_source_path"] = "crop_runs/decoy/centers_img_xy"

    with pytest.raises(ValueError, match="positions_px_source_path conflicts"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_online_derivation_binds_exact_position_path_and_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    source = sealed.position_bindings.source_positions
    online_source_path = (
        "analysis/stimulus_runs/stim_1/tracking_data/chaser_states/target_position_xy"
    )
    online_positions = SimpleNamespace(
        source_positions=SimpleNamespace(
            coordinate_node=SimpleNamespace(path=online_source_path),
            descriptor=source.descriptor,
        )
    )
    online = _WritableGroup(
        path="analysis/track_kinematics_runs/online/motion_online",
        archive_token=root._coordinate_archive_token,
    )
    inputs = {
        "stimulus_run": "stim_1",
        "chaser_index": 0,
        "positions_px_source_path": online_source_path,
        "positions_px_coordinate_descriptor_sha256": source.descriptor.digest(),
    }
    online_parameters = {
        "fps": 30.0,
        "smoothing_seconds": 0.05,
        "smoothing_method": "moving_average",
        "smoothing_alignment": "centered",
        "savgol_polyorder": None,
        "coordinate_space": source.descriptor.space_id,
        "hysteresis_enabled": False,
        "hysteresis_high_px": None,
        "hysteresis_low_px": None,
        "hysteresis_min_frames": None,
        "hysteresis_band_policy": "latch",
    }
    online.attrs.update(
        {
            **mod._track_kinematics_contract_attrs(
                run_type="online",
                method="track_kinematics_online",
                parameters=online_parameters,
                inputs=inputs,
            ),
            "inputs": inputs,
            "provenance": {
                "stage": "track_kinematics",
                "parameters": copy.deepcopy(online_parameters),
                "inputs": copy.deepcopy(inputs),
            },
            "run_provenance": {
                "schema": "palette.run_provenance.v1",
                "git_sha": "a" * 40,
                "config_hash": mod.sha256_payload(online_parameters),
                "params": copy.deepcopy(online_parameters),
                "input_run_ids": copy.deepcopy(inputs),
                "command": "test_online_derivation",
                "fisheye_version": None,
            },
        }
    )
    record = mod._motion_run_derivation_record(
        online,
        online_positions,
    )
    assert record["record"]["source_refs"]["source_positions_px_path"] == (
        online_source_path
    )

    bad_inputs = copy.deepcopy(inputs)
    bad_inputs["positions_px_coordinate_descriptor_sha256"] = "0" * 64
    online.attrs["inputs"] = bad_inputs
    bad_stage_provenance = copy.deepcopy(online.attrs["provenance"])
    bad_stage_provenance["inputs"] = copy.deepcopy(bad_inputs)
    online.attrs["provenance"] = bad_stage_provenance
    bad_run_provenance = copy.deepcopy(online.attrs["run_provenance"])
    bad_run_provenance["input_run_ids"] = copy.deepcopy(bad_inputs)
    online.attrs["run_provenance"] = bad_run_provenance
    online.attrs["source_refs"] = mod._track_kinematics_source_refs(
        run_type="online",
        inputs=bad_inputs,
    )
    with pytest.raises(ValueError, match="descriptor digest differs"):
        mod._motion_run_derivation_record(
            online,
            online_positions,
        )


def test_run_derivation_rejects_all_conflicting_duplicate_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    positions = sealed.position_bindings

    original_stage = copy.deepcopy(run.attrs["provenance"])
    conflicting_stage = copy.deepcopy(original_stage)
    conflicting_stage["parameters"]["fps"] = 2.0
    run.attrs["provenance"] = conflicting_stage
    with pytest.raises(ValueError, match="stage provenance parameters or inputs"):
        mod._motion_run_derivation_record(run, positions)

    conflicting_stage = copy.deepcopy(original_stage)
    conflicting_stage["inputs"]["position_source_path"] = "crop_runs/decoy"
    run.attrs["provenance"] = conflicting_stage
    with pytest.raises(ValueError, match="stage provenance parameters or inputs"):
        mod._motion_run_derivation_record(run, positions)
    run.attrs["provenance"] = original_stage

    original_final = copy.deepcopy(run.attrs["run_provenance"])
    conflicting_final = copy.deepcopy(original_final)
    conflicting_final["params"]["fps"] = 2.0
    run.attrs["run_provenance"] = conflicting_final
    with pytest.raises(ValueError, match="finalization provenance conflicts"):
        mod._motion_run_derivation_record(run, positions)

    conflicting_final = copy.deepcopy(original_final)
    conflicting_final["input_run_ids"]["position_source_path"] = "crop_runs/decoy"
    run.attrs["run_provenance"] = conflicting_final
    with pytest.raises(ValueError, match="finalization provenance conflicts"):
        mod._motion_run_derivation_record(run, positions)
    run.attrs["run_provenance"] = original_final

    run.attrs["smoothing_method"] = "savitzky_golay"
    with pytest.raises(ValueError, match="root smoothing_method conflicts"):
        mod._motion_run_derivation_record(run, positions)


def test_v2_rejects_duplicate_top_level_git_outside_stage_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    _configure_v2_successor_run(root, run)
    run.attrs["git_commit"] = "b" * 40
    run.attrs["git_branch"] = "agent/decoy"

    with pytest.raises(ValueError, match="unsupported run-root attrs"):
        mod._motion_run_root_attrs_record(run, sealed.position_bindings)


def test_frozen_v1_derivation_rejects_successor_only_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    inputs = copy.deepcopy(run.attrs["inputs"])
    inputs.update(
        {
            "position_lineage_mode": (
                mod.TRACK_POSITION_LINEAGE_COLLECTION_PROXY_SUCCESSOR_V1
            ),
            "keypoint_source_crop_run": "c1",
            "tracking_source_rowset_path": "crop_runs/c1",
            "source_detection_run_id": "collection_1",
        }
    )
    _replace_motion_derivation_inputs(run, inputs)

    with pytest.raises(ValueError, match="input inventory is not closed"):
        mod._motion_run_derivation_record(run, sealed.position_bindings)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("keypoints_runs", "kp_1", "heading"), np.float32(17.0)),
        (("keypoints_runs", "kp_1", "heading_usable"), False),
        (("tracking_runs", "trk_1", "track_ids"), np.int32(8)),
        (("tracking_runs", "trk_1", "arena_ids"), np.int32(4)),
        (("tracking_runs", "trk_1", "track_arena_ids"), np.int32(4)),
    ),
)
def test_full_motion_rejects_mutated_exact_upstream_input_arrays(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    node = root
    for part in path:
        node = node[part]
    node.data[0] = replacement

    with pytest.raises(ValueError, match="payload changed|evidence changed"):
        mod.load_bound_track_motion_run(root, run)


def test_full_motion_rejects_equal_payload_from_unselected_keypoint_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    keypoint_runs = root["keypoints_runs"]
    selected = keypoint_runs["kp_1/heading"]
    decoy_run = keypoint_runs.create_group("kp_decoy")
    decoy = decoy_run.create_array(
        "heading",
        data=np.asarray(selected[:]),
    )
    authority = copy.deepcopy(run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR])
    authority["fields"]["heading_degrees"]["array"] = mod._motion_input_array_record(
        decoy
    )
    run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR] = authority
    positions = mod.load_bound_track_position_bindings(root, run)

    with pytest.raises(ValueError, match="not the selected keypoint run"):
        mod._build_track_motion_publication_manifest(root, run, positions)


def test_input_authority_rejects_equal_payload_detection_source_decoy_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    positions = sealed.position_bindings.source_positions
    crop = root["crop_runs"]["c1"]
    decoy = crop.create_array(
        "decoy_detection_source",
        data=np.zeros(2, dtype=np.int8),
    )

    with pytest.raises(ValueError, match="exact detection_source sibling"):
        mod.build_track_motion_input_authority(
            root,
            source_positions=positions,
            mode="offline_exact_sources_v1",
            heading_node=root["keypoints_runs"]["kp_1"]["heading"],
            keypoint_usability_node=root["keypoints_runs"]["kp_1"]["heading_usable"],
            keypoint_row_key_node=root["keypoints_runs"]["kp_1"]["instance_key"],
            tracking_group=root["tracking_runs"]["trk_1"],
            detection_source_node=decoy,
        )


def test_input_authority_rejects_omitted_selected_detection_source_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    crop = root["crop_runs"]["c1"]
    crop.create_array(
        "detection_source",
        data=np.zeros(2, dtype=np.int8),
    )

    with pytest.raises(ValueError, match="cannot be replaced by a generated value"):
        mod.build_track_motion_input_authority(
            root,
            source_positions=sealed.position_bindings.source_positions,
            mode="offline_exact_sources_v1",
            heading_node=root["keypoints_runs"]["kp_1"]["heading"],
            keypoint_usability_node=root["keypoints_runs"]["kp_1"]["heading_usable"],
            keypoint_row_key_node=root["keypoints_runs"]["kp_1"]["instance_key"],
            tracking_group=root["tracking_runs"]["trk_1"],
        )


def test_manifest_rejects_generated_detection_source_after_leaf_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    root["crop_runs"]["c1"].create_array(
        "detection_source",
        data=np.zeros(2, dtype=np.int8),
    )

    with pytest.raises(ValueError, match="omits the selected detection_source leaf"):
        mod._build_track_motion_publication_manifest(
            root,
            run,
            sealed.position_bindings,
        )


def test_input_authority_rejects_uncontrolled_keypoint_usability_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    keypoint = root["keypoints_runs"]["kp_1"]
    decoy = keypoint.create_array(
        "decoy_usability",
        data=np.ones(2, dtype=bool),
    )

    with pytest.raises(ValueError, match="first available controlled usability leaf"):
        mod.build_track_motion_input_authority(
            root,
            source_positions=sealed.position_bindings.source_positions,
            mode="offline_exact_sources_v1",
            heading_node=keypoint["heading"],
            keypoint_usability_node=decoy,
            keypoint_row_key_node=keypoint["instance_key"],
            tracking_group=root["tracking_runs"]["trk_1"],
        )


def test_validator_rejects_equal_payload_uncontrolled_usability_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    decoy = root["keypoints_runs"]["kp_1"].create_array(
        "decoy_usability",
        data=np.ones(2, dtype=bool),
    )
    authority = copy.deepcopy(run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR])
    authority["fields"]["keypoint_success"] = mod._motion_input_array_field(
        decoy,
        row_alignment="keypoint_exact_row_key_equality_v1",
        output_dtype="|b1",
    )
    run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR] = authority

    with pytest.raises(ValueError, match="first available controlled dataset"):
        mod._validate_track_motion_input_authority(
            root,
            run,
            sealed.position_bindings,
            [(7, track)],
        )


def test_validator_rejects_generated_usability_when_controlled_leaf_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    authority = copy.deepcopy(run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR])
    authority["fields"]["keypoint_success"] = mod._motion_generated_field(
        generator_id="all_true_v1",
        row_count=2,
        output_dtype="|b1",
        value=True,
    )
    run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR] = authority

    with pytest.raises(ValueError, match="omits a controlled selected-run leaf"):
        mod._validate_track_motion_input_authority(
            root,
            run,
            sealed.position_bindings,
            [(7, track)],
        )


def test_validator_rejects_omitted_tracking_arena_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    original = copy.deepcopy(run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR])

    missing_rows = copy.deepcopy(original)
    missing_rows["fields"]["arena_id"] = mod._motion_generated_field(
        generator_id="unavailable_v1",
        row_count=2,
        output_dtype="<i8",
    )
    run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR] = missing_rows
    with pytest.raises(ValueError, match="omits the selected tracking arena_ids leaf"):
        mod._validate_track_motion_input_authority(
            root,
            run,
            sealed.position_bindings,
            [(7, track)],
        )


def test_validator_accepts_negative_unassigned_tracking_rows_outside_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, sealed, _physical = _clone_full_motion_run(
        monkeypatch,
        source_rows=np.asarray([0], dtype=np.int64),
    )
    tracking = root["tracking_runs"]["trk_1"]
    tracking["track_ids"].data[1] = np.int32(-1)
    authority = mod.build_track_motion_input_authority(
        root,
        source_positions=sealed.position_bindings.source_positions,
        mode="offline_exact_sources_v1",
        heading_node=root["keypoints_runs"]["kp_1"]["heading"],
        keypoint_usability_node=root["keypoints_runs"]["kp_1"]["heading_usable"],
        keypoint_row_key_node=root["keypoints_runs"]["kp_1"]["instance_key"],
        tracking_group=tracking,
    )
    run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR] = authority.record

    _manifest, values = mod._validate_track_motion_input_authority(
        root,
        run,
        sealed.position_bindings,
        [(7, track)],
    )

    assert values["track_id"].tolist() == [7, -1]


def test_validator_rejects_nonnegative_tracking_row_missing_from_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, sealed, _physical = _clone_full_motion_run(
        monkeypatch,
        source_rows=np.asarray([0], dtype=np.int64),
    )
    tracking = root["tracking_runs"]["trk_1"]
    tracking["track_ids"].data[1] = np.int32(8)
    authority = mod.build_track_motion_input_authority(
        root,
        source_positions=sealed.position_bindings.source_positions,
        mode="offline_exact_sources_v1",
        heading_node=root["keypoints_runs"]["kp_1"]["heading"],
        keypoint_usability_node=root["keypoints_runs"]["kp_1"]["heading_usable"],
        keypoint_row_key_node=root["keypoints_runs"]["kp_1"]["instance_key"],
        tracking_group=tracking,
    )
    run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR] = authority.record

    with pytest.raises(ValueError, match="assigned track IDs disagree"):
        mod._validate_track_motion_input_authority(
            root,
            run,
            sealed.position_bindings,
            [(7, track)],
        )

    missing_inventory = copy.deepcopy(original)
    missing_inventory["arena_inventory"] = None
    run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR] = missing_inventory
    with pytest.raises(ValueError, match="omits the selected tracking arena inventory"):
        mod._validate_track_motion_input_authority(
            root,
            run,
            sealed.position_bindings,
            [(7, track)],
        )


def test_manifest_rejects_forged_equal_payload_detection_source_decoy_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    crop = root["crop_runs"]["c1"]
    decoy = crop.create_array(
        "decoy_detection_source",
        data=np.zeros(2, dtype=np.int8),
    )
    authority = copy.deepcopy(run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR])
    authority["fields"]["detection_source"] = mod._motion_input_array_field(
        decoy,
        row_alignment="selected_position_rowset_sibling_v1",
        output_dtype="|i1",
    )
    run.attrs[mod.TRACK_MOTION_INPUT_AUTHORITY_ATTR] = authority
    positions = mod.load_bound_track_position_bindings(root, run)

    with pytest.raises(ValueError, match="exact selected detection_source leaf"):
        mod._build_track_motion_publication_manifest(root, run, positions)


def test_online_input_authority_accepts_only_visual_angle_heading_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    positions = sealed.position_bindings.source_positions
    crop = root["crop_runs"]["c1"]
    selected = crop.create_array(
        "visual_angle_deg",
        data=np.asarray([10.0, 20.0], dtype=np.float32),
    )
    authority = mod.build_track_motion_input_authority(
        root,
        source_positions=positions,
        mode="online_exact_or_generated_v1",
        heading_node=selected,
        generated_track_id=0,
    )
    assert (
        authority.record["fields"]["heading_degrees"]["array"]["array_ref"]
        == "/crop_runs/c1/visual_angle_deg"
    )

    decoy = crop.create_array(
        "decoy_visual_angle_deg",
        data=np.asarray(selected[:]),
    )
    with pytest.raises(ValueError, match="exact visual_angle_deg sibling"):
        mod.build_track_motion_input_authority(
            root,
            source_positions=positions,
            mode="online_exact_or_generated_v1",
            heading_node=decoy,
            generated_track_id=0,
        )


def test_online_input_authority_rejects_omitted_visual_angle_heading_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _run, _track, sealed, _physical = _clone_full_motion_run(monkeypatch)
    root["crop_runs"]["c1"].create_array(
        "visual_angle_deg",
        data=np.asarray([10.0, 20.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="cannot be replaced by a generated heading"):
        mod.build_track_motion_input_authority(
            root,
            source_positions=sealed.position_bindings.source_positions,
            mode="online_exact_or_generated_v1",
            generated_track_id=0,
        )


def test_full_motion_rejects_forged_arena_identity_at_every_output_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    run["track_arena_ids"].data[0] = np.int32(9)
    track.attrs["arena_id"] = 9
    forged_manifest = copy.deepcopy(run.attrs["track_manifest"])
    forged_manifest[0]["arena_id"] = 9
    run.attrs["track_manifest"] = forged_manifest

    with pytest.raises(ValueError, match="arena identity|track_arena_ids"):
        mod.load_bound_track_motion_run(root, run)


def test_full_motion_rejects_reintroduced_detection_ordinal_even_if_resealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    assert "detection_indices" not in track
    run.attrs["stage_selector_eligible"] = True
    track.create_array(
        "detection_indices",
        data=np.asarray([0, 1], dtype=np.int64),
    )
    positions = mod.load_bound_track_position_bindings(root, run)

    with pytest.raises(ValueError, match="array inventory differs"):
        mod._build_track_motion_publication_manifest(root, run, positions)


@pytest.mark.parametrize("legacy_space", ("camera", "texture"))
def test_full_motion_normal_seal_rejects_legacy_coordinate_space_labels(
    monkeypatch: pytest.MonkeyPatch,
    legacy_space: str,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    parameters = copy.deepcopy(run.attrs["parameters"])
    parameters["coordinate_space"] = legacy_space
    run.attrs["parameters"] = parameters
    stage_provenance = copy.deepcopy(run.attrs["provenance"])
    stage_provenance["parameters"] = copy.deepcopy(parameters)
    run.attrs["provenance"] = stage_provenance
    run_provenance = copy.deepcopy(run.attrs["run_provenance"])
    run_provenance["params"] = copy.deepcopy(parameters)
    run_provenance["config_hash"] = mod.sha256_payload(parameters)
    run.attrs["run_provenance"] = run_provenance

    with pytest.raises(ValueError, match="unsupported track coordinate space"):
        mod.load_bound_track_motion_run(root, run)


def test_one_row_pixel_only_producer_seal_and_strict_reader_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, track, sealed, _physical = _fresh_full_motion_run(
        monkeypatch,
        source_rows=np.asarray([0], dtype=np.int64),
    )
    assert "positions_mm" not in track
    assert "detection_indices" not in track
    assert sealed.track(7).position_binding.positions_mm is None
    run.attrs["stage_selector_eligible"] = True

    tables = load_track_kinematics_track(
        root,
        run_name="motion_pixel",
        scope="offline",
        track_id=7,
    )

    assert tables.positions_px is not None
    assert tables.positions_px.shape == (1, 2)
    assert tables.positions_mm is None
    assert tables.positions_px_descriptor is not None
    assert tables.positions_px_descriptor.geometry_type == "point_xy"
    camera_positions, width, height = tables.require_direct_source_camera_positions_px()
    np.testing.assert_array_equal(camera_positions, tables.positions_px)
    assert (width, height) == (100, 80)


def test_strict_reader_rejects_mutated_detached_surface_after_child_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _track, _sealed, _physical = _clone_full_motion_run(monkeypatch)
    run.attrs["stage_selector_eligible"] = True
    real_loader = mod.load_bound_track_motion_run

    def _replace_after_binding(authoritative_root, candidate):
        bound = real_loader(authoritative_root, candidate)
        old_run = bound.run_group
        replacement = copy.deepcopy(old_run)
        offline = authoritative_root["analysis"]["track_kinematics_runs"]["offline"]
        offline.children[bound.position_bindings.run_name] = replacement
        detached = bound.track(7).surface("positions_px").node
        detached.data[0, 0] += np.asarray(123.0, dtype=detached.dtype)
        return bound

    monkeypatch.setattr(mod, "load_bound_track_motion_run", _replace_after_binding)

    with pytest.raises(ValueError, match="changed payload"):
        load_track_kinematics_track(
            root,
            run_name="motion_pixel",
            scope="offline",
            track_id=7,
        )
