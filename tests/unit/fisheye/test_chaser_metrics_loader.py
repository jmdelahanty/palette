from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pytest
import zarr

import fisheye.analysis.chaser_metrics_loader as mod
from fisheye.analysis import import_stimulus_to_zarr as stimulus_import
from fisheye.shared.coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    COORDINATE_DESCRIPTOR_DIGEST_SUFFIX,
    canonical_coordinate_descriptor_v2_digest,
)
from fisheye.shared.canonical_coordinate_publication import (
    build_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_identity import (
    ROW_IDENTITY_CONTRACT_ATTR,
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_ARRAY_REF,
    build_row_identity_contract,
    identity_array_content_sha256,
    row_identity_contract_attrs,
    row_identity_key_attrs,
    stamp_and_bind_row_identity_contract,
)
from fisheye.shared.coordinate_record import (
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_reference import (
    bind_persisted_record_reference_extent,
)
import fisheye.shared.stimulus_coordinate_contract as stimulus_contract
from fisheye.shared.stimulus_coordinate_contract import (
    ARENA_GEOMETRY_RECORD_ATTR,
    ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
    CAMERA_FRAME_IDS_ARRAY,
    CAMERA_MAPPING_RECORD_ATTR,
    CAMERA_MAPPING_RECORD_DIGEST_ATTR,
    CAMERA_MAPPING_SCHEMA_ID,
    CAMERA_MAPPING_SCHEMA_VERSION,
    COORDINATE_CONTRACT_EPOCH,
    COORDINATE_IMPORT_LINEAGE_ATTR,
    COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR,
    COORDINATE_OUTPUT_MANIFEST_ATTR,
    COORDINATE_OUTPUT_MANIFEST_DIGEST_ATTR,
    COORDINATE_OUTPUT_MANIFEST_SCHEMA_ID,
    COORDINATE_OUTPUT_MANIFEST_SCHEMA_VERSION,
    COORDINATE_SURFACE_MANIFEST_ATTR,
    COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR,
    COORDINATE_SURFACE_MANIFEST_SCHEMA,
    COORDINATE_SURFACE_MANIFEST_VERSION,
    SOURCE_ROW_INDICES_ARRAY,
    SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
    SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
    SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
    SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR,
    STIMULUS_IMPORT_VERSION,
    arena_geometry_record,
    canonical_mapping_digest,
    numpy_content_digest,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.shared.zarr.columnar import (
    store_array,
    write_columnar_dataset,
)
from tests.unit.fisheye.test_import_stimulus_to_zarr_paths import (
    _prepare_acquisition_authority,
    _write_h5_contract_attrs,
    _write_stimulus_h5_with_arena_relative_chaser_states,
)


def _legacy_canonical_root(
    *,
    multi_chaser: bool = False,
) -> tuple[zarr.Group, zarr.Group]:
    root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    analysis = root.create_group("analysis")
    stimulus_parent = analysis.create_group("stimulus_runs")
    stimulus_parent.attrs["latest"] = "stim_1"
    stimulus_parent.attrs["latest_complete"] = "stim_1"
    stim = stimulus_parent.create_group("stim_1")
    stim.attrs.update(
        {
            RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
            "import_version": STIMULUS_IMPORT_VERSION,
            "chaser_states_coordinate_descriptor_status": "canonical",
        }
    )

    frame_dtype = np.dtype(
        [
            ("stimulus_frame_num", np.int64),
            ("triggering_camera_frame_id", np.int64),
            ("timestamp_ns", np.int64),
        ]
    )
    frame_metadata = np.zeros(3, dtype=frame_dtype)
    frame_metadata["stimulus_frame_num"] = [0, 1, 2]
    frame_metadata["triggering_camera_frame_id"] = [10, 11, 12]
    frame_metadata["timestamp_ns"] = [0, 10_000_000, 20_000_000]
    write_columnar_dataset(
        stim.create_group("video_metadata"),
        "frame_metadata",
        frame_metadata,
    )

    calibration = stim.create_group("calibration")
    arena = calibration.create_group("arena_geometry")
    arena.attrs.update(
        {
            "arena_region_width_px": 344,
            "arena_region_height_px": 344,
            "arena_origin_in_canvas_x_px": 270,
            "arena_origin_in_canvas_y_px": 520,
        }
    )
    arena_record = arena_geometry_record(dict(arena.attrs))
    stamp_and_bind_persisted_coordinate_record(
        arena,
        arena_record,
        attr_name=ARENA_GEOMETRY_RECORD_ATTR,
        digest_attr_name=ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
    )
    arena_reference = bind_persisted_record_reference_extent(
        arena,
        record_attr=ARENA_GEOMETRY_RECORD_ATTR,
        digest_attr=ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
        width_field="arena_region_width_px",
        height_field="arena_region_height_px",
        units_field="units",
    )

    row_count = 6 if multi_chaser else 3
    chaser_dtype = np.dtype(
        [
            ("stimulus_frame_num", np.int64),
            ("chaser_index", np.int16),
            ("trial_state", np.int16),
            ("target_pos_x", np.float32),
            ("target_pos_y", np.float32),
            ("distance_to_target_px", np.float32),
        ]
    )
    chaser_states = np.zeros(row_count, dtype=chaser_dtype)
    if multi_chaser:
        chaser_states["stimulus_frame_num"] = [0, 0, 1, 1, 2, 2]
        chaser_states["chaser_index"] = [0, 1, 0, 1, 0, 1]
        target_xy = np.array(
            [
                [1.0, 4.0],
                [101.0, 104.0],
                [2.0, 5.0],
                [102.0, 105.0],
                [3.0, 6.0],
                [103.0, 106.0],
            ],
            dtype=np.float32,
        )
        identity_values = np.column_stack(
            (
                chaser_states["chaser_index"].astype(np.int64),
                chaser_states["stimulus_frame_num"],
            )
        )
        identity_components = ("chaser_index", "stimulus_frame_num")
    else:
        chaser_states["stimulus_frame_num"] = [0, 1, 2]
        chaser_states["chaser_index"] = 0
        target_xy = np.array(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            dtype=np.float32,
        )
        identity_values = chaser_states["stimulus_frame_num"].copy()
        identity_components = ("stimulus_frame_num",)
    chaser_states["trial_state"] = 1
    chaser_states["target_pos_x"] = target_xy[:, 0]
    chaser_states["target_pos_y"] = target_xy[:, 1]
    chaser_states["distance_to_target_px"] = np.arange(row_count) + 0.5

    tracking = stim.create_group("tracking_data")
    chaser_group = write_columnar_dataset(
        tracking,
        "chaser_states",
        chaser_states,
    )
    key_node = store_array(
        chaser_group,
        STIMULUS_STATE_KEY_ARRAY_REF,
        np.asarray(identity_values, dtype=np.int64),
        {},
    )
    identity = build_row_identity_contract(
        domain=STIMULUS_STATE_DOMAIN,
        values=identity_values,
        components=identity_components,
    )
    bound_identity = stamp_and_bind_row_identity_contract(
        chaser_group,
        key_node,
        contract=identity,
    )

    point_node = store_array(chaser_group, "target_position_xy", target_xy, {})
    point_node.attrs.update(
        {
            "semantic_role": "target_position",
            "source_component_fields": ["target_pos_x", "target_pos_y"],
        }
    )
    chaser_group["target_pos_x"].attrs.update(
        {
            "parent_semantic_role": "target_position",
            "coordinate_component": "x",
            "coordinate_surface_array_ref": "target_position_xy",
        }
    )
    chaser_group["target_pos_y"].attrs.update(
        {
            "parent_semantic_role": "target_position",
            "coordinate_component": "y",
            "coordinate_surface_array_ref": "target_position_xy",
        }
    )

    row_fields = list(identity_components)
    classifications = {
        name: (
            "row_identity"
            if name in row_fields
            else "coordinate_component"
            if name in {"target_pos_x", "target_pos_y"}
            else "non_spatial"
        )
        for name in chaser_dtype.names or ()
    }
    surface_manifest = {
        "schema_id": COORDINATE_SURFACE_MANIFEST_SCHEMA,
        "schema_version": COORDINATE_SURFACE_MANIFEST_VERSION,
        "coordinate_fields_complete": True,
        "field_classifications": classifications,
        "row_identity_fields": row_fields,
        "surfaces": [
            {
                "array_name": "target_position_xy",
                "semantic_role": "target_position",
                "component_fields": ["target_pos_x", "target_pos_y"],
            }
        ],
    }
    surface_manifest_record = stamp_and_bind_persisted_coordinate_record(
        chaser_group,
        surface_manifest,
        attr_name=COORDINATE_SURFACE_MANIFEST_ATTR,
        digest_attr_name=COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR,
    )
    camera_frame_ids = np.asarray(
        [10 + int(value) for value in chaser_states["stimulus_frame_num"]],
        dtype=np.int64,
    )
    source_row_indices = np.arange(row_count, dtype=np.int64)
    camera_node = store_array(
        chaser_group,
        CAMERA_FRAME_IDS_ARRAY,
        camera_frame_ids,
        {},
    )
    source_rows_node = store_array(
        chaser_group,
        SOURCE_ROW_INDICES_ARRAY,
        source_row_indices,
        {},
    )
    frame_group = stim["video_metadata"]["frame_metadata"]
    frame_stimulus = np.asarray(frame_group["stimulus_frame_num"][:], dtype=np.int64)
    frame_camera = np.asarray(
        frame_group["triggering_camera_frame_id"][:],
        dtype=np.int64,
    )
    camera_mapping_payload = {
        "schema_id": CAMERA_MAPPING_SCHEMA_ID,
        "schema_version": CAMERA_MAPPING_SCHEMA_VERSION,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "method": "exact_stimulus_frame_to_camera_frame_lookup",
        "row_count": row_count,
        "row_identity_record_ref": bound_identity.record_ref,
        "row_identity_record_sha256": bound_identity.record_sha256,
        "stimulus_state_key_ref": f"/{key_node.path}",
        "stimulus_state_key_sha256": identity_array_content_sha256(identity_values),
        "frame_metadata_rowset_ref": f"/{frame_group.path}",
        "frame_metadata_stimulus_field": "stimulus_frame_num",
        "frame_metadata_stimulus_ref": f"/{frame_group['stimulus_frame_num'].path}",
        "frame_metadata_stimulus_sha256": numpy_content_digest(frame_stimulus),
        "frame_metadata_camera_field": "triggering_camera_frame_id",
        "frame_metadata_camera_ref": f"/{frame_group['triggering_camera_frame_id'].path}",
        "frame_metadata_camera_sha256": numpy_content_digest(frame_camera),
        "stimulus_frame_component": "stimulus_frame_num",
        "camera_frame_ids_ref": f"/{camera_node.path}",
        "camera_frame_ids_sha256": numpy_content_digest(camera_frame_ids),
        "source_row_indices_ref": f"/{source_rows_node.path}",
        "source_row_indices_sha256": numpy_content_digest(source_row_indices),
    }
    camera_mapping_record = stamp_and_bind_persisted_coordinate_record(
        chaser_group,
        camera_mapping_payload,
        attr_name=CAMERA_MAPPING_RECORD_ATTR,
        digest_attr_name=CAMERA_MAPPING_RECORD_DIGEST_ATTR,
    )
    output_entries = stimulus_contract._coordinate_output_array_entries(
        chaser_group,
        manifest=surface_manifest,
    )
    import_payload = {
        "schema_id": "palette.coordinate_import_lineage",
        "schema_version": 1,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "output_row_identity_contract_ref": bound_identity.record_ref,
        "output_row_identity_contract_sha256": bound_identity.record_sha256,
        "selected_arena_geometry_ref": arena_reference.record_ref,
        "selected_arena_geometry_sha256": arena_reference.record_sha256,
        "camera_mapping_record_ref": camera_mapping_record.record_ref,
        "camera_mapping_record_sha256": camera_mapping_record.record_sha256,
        "output_array_sha256": {
            name: entry["content_sha256"]
            for name, entry in output_entries.items()
        },
    }
    import_record = stamp_and_bind_persisted_coordinate_record(
        chaser_group,
        import_payload,
        attr_name=COORDINATE_IMPORT_LINEAGE_ATTR,
        digest_attr_name=COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR,
    )
    output_payload = {
        "schema_id": COORDINATE_OUTPUT_MANIFEST_SCHEMA_ID,
        "schema_version": COORDINATE_OUTPUT_MANIFEST_SCHEMA_VERSION,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "row_count": row_count,
        "row_identity": {
            "record_ref": bound_identity.record_ref,
            "record_sha256": bound_identity.record_sha256,
        },
        "reference_extent": {
            "record_ref": arena_reference.record_ref,
            "record_sha256": arena_reference.record_sha256,
            "selector": arena_reference.selector,
            "width": arena_reference.width,
            "height": arena_reference.height,
            "units": arena_reference.units,
        },
        "records": {
            "surface_manifest": {
                "record_ref": surface_manifest_record.record_ref,
                "record_sha256": surface_manifest_record.record_sha256,
            },
            "camera_mapping": {
                "record_ref": camera_mapping_record.record_ref,
                "record_sha256": camera_mapping_record.record_sha256,
            },
            "import_lineage": {
                "record_ref": import_record.record_ref,
                "record_sha256": import_record.record_sha256,
            },
        },
        "arrays": output_entries,
    }
    output_record = stamp_and_bind_persisted_coordinate_record(
        chaser_group,
        output_payload,
        attr_name=COORDINATE_OUTPUT_MANIFEST_ATTR,
        digest_attr_name=COORDINATE_OUTPUT_MANIFEST_DIGEST_ATTR,
    )
    lineage_records = (
        surface_manifest_record,
        camera_mapping_record,
        import_record,
        output_record,
    )
    descriptor_bindings = [
        build_bound_canonical_coordinate_descriptor(
            point_node,
            profile_id="arena_relative_canvas_px.top_left_y_down.v1",
            geometry_type="points_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            pixel_convention="continuous",
            row_identity=bound_identity,
            reference_extent=arena_reference,
            source_camera_overlay_status="not_suitable",
            lineage_records=lineage_records,
        )
    ]
    for component, field in (("x", "target_pos_x"), ("y", "target_pos_y")):
        descriptor_bindings.append(
            build_bound_canonical_coordinate_descriptor(
                chaser_group[field],
                profile_id="arena_relative_canvas_px.top_left_y_down.v1",
                geometry_type="coordinate_component",
                components=(component,),
                component_units=("px",),
                pixel_convention="continuous",
                row_identity=bound_identity,
                reference_extent=arena_reference,
                source_camera_overlay_status="not_suitable",
                lineage_records=lineage_records,
            )
        )
    stamp_bound_canonical_coordinate_descriptors(descriptor_bindings)
    chaser_group.attrs["coordinate_descriptor_status"] = "canonical"
    return root, chaser_group


_CANONICAL_FIXTURE_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _canonical_root(
    *,
    multi_chaser: bool = False,
) -> tuple[zarr.Group, zarr.Group]:
    """Build the reader fixture through the real future-canonical importer."""

    fixture_dir = tempfile.TemporaryDirectory(prefix="palette_chaser_contract_")
    _CANONICAL_FIXTURE_DIRS.append(fixture_dir)
    base = Path(fixture_dir.name)
    h5_path = base / "stimulus.h5"
    zarr_path = base / "analysis.zarr"
    _write_stimulus_h5_with_arena_relative_chaser_states(
        h5_path,
        multi_chaser=multi_chaser,
    )

    with h5py.File(h5_path, "a") as h5:
        frame_node = h5["/video_metadata/frame_metadata"]
        frame_dtype = frame_node.dtype
        frame_attrs = dict(frame_node.attrs)
        del h5["/video_metadata/frame_metadata"]
        frame_values = np.zeros(3, dtype=frame_dtype)
        frame_values["stimulus_frame_num"] = [0, 1, 2]
        frame_values["triggering_camera_frame_id"] = [10, 11, 12]
        frame_values["timestamp_ns"] = [0, 10_000_000, 20_000_000]
        replacement_frames = h5["/video_metadata"].create_dataset(
            "frame_metadata",
            data=frame_values,
        )
        replacement_frames.attrs.update(frame_attrs)

        old_chaser = h5["/tracking_data/chaser_states"]
        old_attrs = dict(old_chaser.attrs)
        old_manifest = json.loads(old_attrs[COORDINATE_SURFACE_MANIFEST_ATTR])
        base_descr = [
            item
            for item in old_chaser.dtype.descr
            if item[0] != "chaser_index"
        ]
        dtype_descr = (
            [("chaser_index", "<u2"), *base_descr]
            if multi_chaser
            else base_descr
        )
        dtype_descr.extend([("trial_state", "<i2"), ("distance_to_target_px", "<f4")])
        row_count = 6 if multi_chaser else 3
        rows = np.zeros(row_count, dtype=np.dtype(dtype_descr))
        if multi_chaser:
            rows["chaser_index"] = [0, 1, 0, 1, 0, 1]
            rows["stimulus_frame_num"] = [0, 0, 1, 1, 2, 2]
            target = np.asarray(
                [
                    [1.0, 4.0],
                    [101.0, 104.0],
                    [2.0, 5.0],
                    [102.0, 105.0],
                    [3.0, 6.0],
                    [103.0, 106.0],
                ],
                dtype=np.float32,
            )
            components = ("chaser_index", "stimulus_frame_num")
            key_values = np.column_stack(
                (rows["chaser_index"], rows["stimulus_frame_num"])
            ).astype(np.int64)
            acquisition_values = np.asarray([0, 0, 1, 1, 2, 2], dtype="<i8")
        else:
            rows["stimulus_frame_num"] = [0, 1, 2]
            target = np.asarray(
                [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
                dtype=np.float32,
            )
            components = ("stimulus_frame_num",)
            key_values = np.asarray([0, 1, 2], dtype=np.int64)
            acquisition_values = np.asarray([0, 1, 2], dtype="<i8")
        rows["target_pos_x"] = target[:, 0]
        rows["target_pos_y"] = target[:, 1]
        rows["target_clamped_pos_x"] = target[:, 0]
        rows["target_clamped_pos_y"] = target[:, 1]
        rows["chaser_pos_x"] = target[:, 0]
        rows["chaser_pos_y"] = target[:, 1]
        rows["trial_state"] = 1
        rows["distance_to_target_px"] = np.arange(row_count) + 0.5
        old_manifest["field_classifications"]["trial_state"] = "non_spatial"
        old_manifest["field_classifications"]["distance_to_target_px"] = (
            "non_spatial"
        )
        old_manifest["row_identity_fields"] = list(components)
        old_attrs[COORDINATE_SURFACE_MANIFEST_ATTR] = json.dumps(
            old_manifest,
            sort_keys=True,
        )
        old_attrs[COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR] = (
            canonical_mapping_digest(old_manifest)
        )
        identity = build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=key_values,
            components=components,
        )
        descriptor = json.loads(old_attrs[COORDINATE_DESCRIPTOR_ATTR])
        descriptor["row_identity"]["record_sha256"] = identity.digest()
        old_attrs[COORDINATE_DESCRIPTOR_ATTR] = json.dumps(
            descriptor,
            sort_keys=True,
        )
        old_attrs["coordinate_descriptor_sha256"] = (
            canonical_coordinate_descriptor_v2_digest(descriptor)
        )
        del h5["/tracking_data/chaser_states"]
        replacement_chaser = h5["/tracking_data"].create_dataset(
            "chaser_states",
            data=rows,
        )
        replacement_chaser.attrs.update(old_attrs)
        _write_h5_contract_attrs(
            replacement_chaser,
            row_identity_contract_attrs(identity),
        )

        del h5[f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY_REF}"]
        key_node = h5["/tracking_data"].create_dataset(
            STIMULUS_STATE_KEY_ARRAY_REF,
            data=key_values,
            dtype="<i8",
        )
        _write_h5_contract_attrs(key_node, row_identity_key_attrs(identity))

        source_time = h5[SOURCE_ACQUISITION_MAPPING_ARRAY_PATH]
        source_time_record = json.loads(
            source_time.attrs[SOURCE_ACQUISITION_MAPPING_RECORD_ATTR]
        )
        del h5[SOURCE_ACQUISITION_MAPPING_ARRAY_PATH]
        source_time = h5["/tracking_data"].create_dataset(
            SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            data=acquisition_values,
            dtype="<i8",
        )
        source_time_record.update(
            {
                "source_row_identity_sha256": identity_array_content_sha256(
                    key_values
                ),
                "source_row_identity_contract_sha256": identity.digest(),
                "source_total_frames": 3,
                "array_shape": [row_count],
                "array_content_sha256": numpy_content_digest(
                    acquisition_values
                ),
            }
        )
        _write_h5_contract_attrs(
            source_time,
            {
                SOURCE_ACQUISITION_MAPPING_RECORD_ATTR: source_time_record,
                SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR: (
                    canonical_mapping_digest(source_time_record)
                ),
            },
        )

    _prepare_acquisition_authority(zarr_path, total_frames=3)
    run_name = stimulus_import.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stim_1",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    chaser = root["analysis"]["stimulus_runs"][run_name]["tracking_data"][
        "chaser_states"
    ]
    return root, chaser


def _load(
    monkeypatch: pytest.MonkeyPatch,
    root: zarr.Group,
    *,
    chaser_index: int = 0,
) -> mod.ChaserMetricsBundle:
    monkeypatch.setattr(mod.zarr, "open", lambda *args, **kwargs: root)
    return mod.load_chaser_metrics("ignored.zarr", chaser_index=chaser_index)


def test_load_chaser_metrics_returns_exact_canonical_surface_and_identity_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root(multi_chaser=True)

    bundle = _load(monkeypatch, root, chaser_index=1)

    np.testing.assert_array_equal(bundle.camera_frame_ids, [10, 11, 12])
    np.testing.assert_allclose(
        bundle.online["target_position_xy"],
        [[101.0, 104.0], [102.0, 105.0], [103.0, 106.0]],
    )
    np.testing.assert_allclose(bundle.online["target_pos_x"], [101.0, 102.0, 103.0])
    np.testing.assert_allclose(bundle.online["target_pos_y"], [104.0, 105.0, 106.0])
    np.testing.assert_array_equal(bundle.offline["has_offline"], np.zeros(3, dtype=bool))

    metadata = bundle.online_coordinate_metadata
    point_node = chaser_group["target_position_xy"]
    key_node = chaser_group[STIMULUS_STATE_KEY_ARRAY_REF]
    assert metadata["schema_id"] == mod.CANONICAL_ONLINE_COORDINATE_HANDOFF_SCHEMA_ID
    assert metadata["schema_version"] == 1
    assert metadata["semantic_role"] == "target_position"
    assert metadata["source_path"] == point_node.path
    assert metadata["coordinate_descriptor"]["schema_version"] == 2
    assert metadata["coordinate_descriptor_sha256"] == point_node.attrs[
        "coordinate_descriptor_sha256"
    ]
    assert metadata["rowset_path"] == chaser_group.path
    assert metadata["row_identity_contract"] == chaser_group.attrs[
        ROW_IDENTITY_CONTRACT_ATTR
    ]
    assert metadata["row_identity_record_ref"] == (
        f"/{chaser_group.path}@{ROW_IDENTITY_CONTRACT_ATTR}"
    )

    mapping = metadata["stimulus_state_key_mapping"]
    assert mapping["source_path"] == key_node.path
    assert mapping["components"] == ["chaser_index", "stimulus_frame_num"]
    assert mapping["values"] == [
        [0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]
    ]
    assert mapping["selected_source_row_indices"] == [1, 3, 5]
    assert mapping["selected_values"] == [[1, 0], [1, 1], [1, 2]]
    assert mapping["selected_camera_frame_ids"] == [10, 11, 12]
    assert mapping["bundle_row_source_indices"] == [1, 3, 5]
    handoff = bundle.online_coordinate_handoff
    assert handoff is not None
    handoff.assert_verified()
    np.testing.assert_array_equal(
        handoff.stimulus_state_key,
        [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]],
    )
    np.testing.assert_array_equal(handoff.camera_frame_ids, [10, 10, 11, 11, 12, 12])
    assert handoff.import_lineage.attr_name == COORDINATE_IMPORT_LINEAGE_ATTR
    assert handoff.output_manifest.attr_name == COORDINATE_OUTPUT_MANIFEST_ATTR
    assert handoff.camera_mapping.attr_name == CAMERA_MAPPING_RECORD_ATTR
    assert "chaser_fish_metrics" not in root["analysis"]


def test_track_selection_uses_exact_chaser_source_rows_not_bundle_reconstruction(
) -> None:
    from fisheye.analysis.track_kinematics import (
        select_canonical_online_track_rows,
    )

    root, chaser_group = _canonical_root(multi_chaser=True)
    stimulus_parent = root["analysis/stimulus_runs"]
    stimulus_group = stimulus_parent[stimulus_parent.attrs["latest_complete"]]
    _, _, handoff = mod.load_canonical_online_coordinate_surface(
        root,
        stimulus_group,
        chaser_group,
    )

    source_rows, acquisition_frames, positions = (
        select_canonical_online_track_rows(
            handoff,
            chaser_index=1,
        )
    )

    np.testing.assert_array_equal(source_rows, [1, 3, 5])
    np.testing.assert_array_equal(acquisition_frames, [0, 1, 2])
    expected = np.asarray(chaser_group["target_position_xy"][:])[source_rows]
    assert positions.dtype == expected.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(positions, expected)


def test_load_rejects_descriptor_free_array_even_when_group_has_coordinate_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    point_node = chaser_group["target_position_xy"]
    chaser_group.attrs[COORDINATE_DESCRIPTOR_ATTR] = deepcopy(
        point_node.attrs[COORDINATE_DESCRIPTOR_ATTR]
    )
    chaser_group.attrs[
        f"{COORDINATE_DESCRIPTOR_ATTR}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"
    ] = point_node.attrs["coordinate_descriptor_sha256"]
    del point_node.attrs[COORDINATE_DESCRIPTOR_ATTR]
    del point_node.attrs["coordinate_descriptor_sha256"]

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="descriptor_attr_missing",
    ):
        _load(monkeypatch, root)


def test_load_rejects_schema_v1_on_exact_point_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    point_node = chaser_group["target_position_xy"]
    payload = deepcopy(point_node.attrs[COORDINATE_DESCRIPTOR_ATTR])
    payload["schema_version"] = 1
    point_node.attrs[COORDINATE_DESCRIPTOR_ATTR] = payload

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="canonical_schema_version_required",
    ):
        _load(monkeypatch, root)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("stale_descriptor_digest", "descriptor_digest_mismatch"),
        ("wrong_identity_ref", "row_identity_record_ref_mismatch"),
        ("wrong_identity_digest", "row_identity_record_digest_mismatch"),
    ],
)
def test_load_rejects_tampered_descriptor_identity_binding(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    root, chaser_group = _canonical_root()
    point_node = chaser_group["target_position_xy"]
    payload = deepcopy(point_node.attrs[COORDINATE_DESCRIPTOR_ATTR])
    if mutation == "stale_descriptor_digest":
        payload["reference_extent"]["width"] = 345
    elif mutation == "wrong_identity_ref":
        payload["row_identity"]["record_ref"] = (
            "/analysis/stimulus_runs/stim_1/tracking_data/other_rows"
            f"@{ROW_IDENTITY_CONTRACT_ATTR}"
        )
    else:
        payload["row_identity"]["record_sha256"] = "b" * 64
    point_node.attrs[COORDINATE_DESCRIPTOR_ATTR] = payload
    if mutation != "stale_descriptor_digest":
        point_node.attrs["coordinate_descriptor_sha256"] = (
            canonical_coordinate_descriptor_v2_digest(payload)
        )

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match=expected,
    ):
        _load(monkeypatch, root)


def test_load_rejects_ambiguous_target_point_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    duplicate = store_array(
        chaser_group,
        "other_target_position_xy",
        np.asarray(chaser_group["target_position_xy"][:]),
        {},
    )
    duplicate.attrs["semantic_role"] = "target_position"

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="semantic_role 'target_position' is claimed by multiple output arrays",
    ):
        _load(monkeypatch, root)


def test_load_rejects_missing_requested_chaser_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_root()

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="no chaser_index field, so only chaser_index=0 is addressable",
    ):
        _load(monkeypatch, root, chaser_index=1)


def test_load_rejects_point_row_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    old_attrs = dict(chaser_group["target_position_xy"].attrs)
    replacement = store_array(
        chaser_group,
        "target_position_xy",
        np.asarray([[1.0, 4.0], [2.0, 5.0]], dtype=np.float32),
        old_attrs,
    )
    assert replacement.shape == (2, 2)

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="does not exactly equal its scalar component arrays",
    ):
        _load(monkeypatch, root)


def test_load_rejects_key_values_that_disagree_with_rowset_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    chaser_group["stimulus_frame_num"][1] = 99

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="stimulus_state_key differs from its exact owning row fields",
    ):
        _load(monkeypatch, root)


def test_load_rejects_tampered_key_content_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    chaser_group[STIMULUS_STATE_KEY_ARRAY_REF][1] = 99

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="key_content_digest_mismatch",
    ):
        _load(monkeypatch, root)


def test_load_rejects_legacy_structured_rowset_without_canonical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    del chaser_group[STIMULUS_STATE_KEY_ARRAY_REF]

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="missing identity, mapping, or arena nodes",
    ):
        _load(monkeypatch, root)


def test_online_coordinate_metadata_is_serializable_and_live_bindings_are_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _canonical_root()

    bundle = _load(monkeypatch, root)

    json.dumps(bundle.online_coordinate_metadata, allow_nan=False, sort_keys=True)
    assert bundle.online_coordinate_handoff is not None
    assert "coordinate_descriptor" not in bundle.online_coordinate_metadata.get(
        "live_bindings",
        {},
    )


@pytest.mark.parametrize(
    ("attr_name", "value", "expected"),
    [
        (RUN_COMPLETION_STATUS_ATTR, "running", "only be consumed from a complete run"),
        ("coordinate_contract_epoch", 2, "exact canonical coordinate-contract epoch"),
    ],
)
def test_load_rejects_incomplete_or_wrong_epoch_source_run(
    monkeypatch: pytest.MonkeyPatch,
    attr_name: str,
    value: object,
    expected: str,
) -> None:
    root, _ = _canonical_root()
    stim = root["analysis"]["stimulus_runs"]["stim_1"]
    stim.attrs[attr_name] = value

    with pytest.raises(mod.ChaserMetricsCoordinateContractError, match=expected):
        _load(monkeypatch, root)


def test_load_rejects_target_point_component_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    chaser_group["target_pos_x"][1] = 999.0

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="target_position_xy does not exactly equal its scalar component arrays",
    ):
        _load(monkeypatch, root)


def test_load_rejects_forged_nonexistent_descriptor_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    node = chaser_group["target_position_xy"]
    payload = deepcopy(node.attrs[COORDINATE_DESCRIPTOR_ATTR])
    payload["lineage_refs"][1]["record_ref"] = (
        f"/{chaser_group.path}@nonexistent_camera_mapping"
    )
    node.attrs[COORDINATE_DESCRIPTOR_ATTR] = payload
    node.attrs["coordinate_descriptor_sha256"] = (
        canonical_coordinate_descriptor_v2_digest(payload)
    )

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="coordinate_lineage_mismatch",
    ):
        _load(monkeypatch, root)


def test_load_rejects_stale_import_lineage_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    lineage = deepcopy(chaser_group.attrs[COORDINATE_IMPORT_LINEAGE_ATTR])
    lineage["operation"] = "forged"
    chaser_group.attrs[COORDINATE_IMPORT_LINEAGE_ATTR] = lineage

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="coordinate_import_lineage_sha256.*stale",
    ):
        _load(monkeypatch, root)


def test_live_handoff_revalidates_records_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    handoff = _load(monkeypatch, root).online_coordinate_handoff
    assert handoff is not None
    lineage = deepcopy(chaser_group.attrs[COORDINATE_IMPORT_LINEAGE_ATTR])
    lineage["operation"] = "changed_after_load"
    chaser_group.attrs[COORDINATE_IMPORT_LINEAGE_ATTR] = lineage

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="no longer fresh",
    ):
        handoff.assert_verified()


def test_load_rejects_output_content_mutation_even_when_components_still_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, chaser_group = _canonical_root()
    chaser_group["target_pos_x"][1] = 999.0
    chaser_group["target_position_xy"][1, 0] = 999.0

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="Coordinate import lineage does not bind the exact published output",
    ):
        _load(monkeypatch, root)


@pytest.mark.parametrize(
    ("field", "values", "expected"),
    [
        (
            "triggering_camera_frame_id",
            np.array([10.0, 11.5, 12.0], dtype=np.float64),
            "fractional mappings are forbidden",
        ),
        (
            "stimulus_frame_num",
            np.array([0, 0, 2], dtype=np.int64),
            "duplicate stimulus-frame mappings",
        ),
    ],
)
def test_load_rejects_fractional_or_conflicting_camera_mapping_source(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    values: np.ndarray,
    expected: str,
) -> None:
    root, _ = _canonical_root()
    frame_group = root["analysis"]["stimulus_runs"]["stim_1"]["video_metadata"][
        "frame_metadata"
    ]
    store_array(frame_group, field, values, {})

    with pytest.raises(mod.ChaserMetricsCoordinateContractError, match=expected):
        _load(monkeypatch, root)


def test_load_rejects_same_path_evidence_from_another_store() -> None:
    first_root, first_chaser = _canonical_root()
    second_root, second_chaser = _canonical_root()
    first_stim = first_root["analysis"]["stimulus_runs"]["stim_1"]
    second_stim = second_root["analysis"]["stimulus_runs"]["stim_1"]
    first_evidence = stimulus_contract.load_bound_stimulus_coordinate_evidence(
        first_stim,
        first_chaser,
        root_node=first_root,
    )

    with pytest.raises(
        mod.ChaserMetricsCoordinateContractError,
        match="different archive/store",
    ):
        mod._load_canonical_target_surface(
            second_root,
            second_stim,
            second_chaser,
            stimulus_evidence=first_evidence,
        )
