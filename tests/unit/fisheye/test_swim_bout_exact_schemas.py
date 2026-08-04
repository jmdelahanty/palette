from __future__ import annotations

from copy import deepcopy

import numpy as np
import zarr

from fisheye.analysis._exact_tabular_run_schema import MANIFEST_ATTRIBUTE
from fisheye.analysis import bout_kinematics as bout_writer
from fisheye.analysis import detect_bouts_multi_level as swim_writer
from fisheye.analysis import bout_kinematics_schema as bout_schema
from fisheye.analysis import swim_bout_schema as swim_schema
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _create_array(
    group: zarr.Group, path: str, dtype: str | None, axes: tuple[str, ...]
) -> None:
    parent = group
    parts = path.split("/")
    for name in parts[:-1]:
        parent = parent.require_group(name)
    resolved = np.dtype("S64" if dtype is None else dtype)
    first_extent = (
        2 if axes[0] == "detector_signal" else (7 if axes[0] == "frame" else 3)
    )
    shape = (first_extent,) if len(axes) == 1 else (first_extent, 7)
    parent.create_array(parts[-1], data=np.zeros(shape, dtype=resolved))


def _set_columnar_attrs(
    group: zarr.Group, specs: dict[str, object], table_paths: tuple[str, ...]
) -> None:
    for table_path in table_paths:
        prefix = table_path + "/"
        fields = [
            (path[len(prefix) :], spec)
            for path, spec in specs.items()
            if path.startswith(prefix) and "/" not in path[len(prefix) :]
        ]
        if not fields:
            continue
        table = group[table_path]
        table.attrs["storage_layout"] = "columnar"
        table.attrs["field_names"] = [name for name, _spec in fields]
        table.attrs["field_dtypes"] = {
            name: spec.logical_dtype for name, spec in fields
        }


def _swim_group(*, embedded_axis: bool = True) -> zarr.Group:
    group = zarr.group()
    group.attrs.update(
        {
            "schema_id": swim_schema.SWIM_BOUT_RUN_SCHEMA_ID,
            "schema_version": swim_schema.SWIM_BOUT_RUN_SCHEMA_VERSION,
            "layout": swim_schema.SWIM_BOUT_LAYOUT,
        }
    )
    required = swim_schema._required_specs()
    for spec in required.values():
        _create_array(group, spec.path, spec.dtype, spec.axes)
    combined = dict(required)
    if embedded_axis:
        for spec in swim_schema._optional_bundles()["embedded_frame_axis"].values():
            _create_array(group, spec.path, spec.dtype, spec.axes)
            combined[spec.path] = spec
    _set_columnar_attrs(group, combined, swim_schema._COLUMNAR_TABLE_PATHS)
    swim_schema.write_swim_bout_array_manifest(group)
    return group


def _bout_group(*, eye: bool = True) -> zarr.Group:
    group = zarr.group()
    group.attrs.update(
        {
            "schema_id": bout_schema.BOUT_KINEMATICS_RUN_SCHEMA_ID,
            "schema_version": bout_schema.BOUT_KINEMATICS_RUN_SCHEMA_VERSION,
            "layout": bout_schema.BOUT_KINEMATICS_LAYOUT,
        }
    )
    required = bout_schema._required_specs()
    for spec in required.values():
        _create_array(group, spec.path, spec.dtype, spec.axes)
    combined = dict(required)
    if eye:
        for spec in bout_schema._optional_bundles()["eye_gaze_metrics"].values():
            _create_array(group, spec.path, spec.dtype, spec.axes)
            combined[spec.path] = spec
    _set_columnar_attrs(group, combined, bout_schema._COLUMNAR_TABLE_PATHS)
    bout_schema.write_bout_kinematics_array_manifest(group)
    return group


def _delete_array(group: zarr.Group, path: str) -> None:
    parent = group
    parts = path.split("/")
    for name in parts[:-1]:
        parent = parent[name]
    del parent[parts[-1]]


def test_swim_manifest_is_exact_and_supports_external_or_embedded_frame_axis() -> None:
    embedded = _swim_group(embedded_axis=True)
    external = _swim_group(embedded_axis=False)

    assert swim_schema.validate_swim_bout_array_manifest(embedded) == ()
    assert swim_schema.validate_swim_bout_array_manifest(external) == ()
    assert embedded.attrs[MANIFEST_ATTRIBUTE]["payload"][
        "enabled_optional_bundles"
    ] == ["embedded_frame_axis"]
    assert (
        external.attrs[MANIFEST_ATTRIBUTE]["payload"]["enabled_optional_bundles"] == []
    )


def test_swim_columnar_field_dtype_contract_is_public_and_exact() -> None:
    fields = swim_schema.build_swim_bout_columnar_field_dtypes()

    assert fields["indexes/candidates"]["candidate_name"] == "|S256"
    assert fields["indexes/candidates"]["parameters_json"] == "|S8192"
    assert fields["indexes/signal_variants"]["parameters_json"] == "|S8192"
    assert fields["tables/summary_metrics"]["metric_name"] == "|S64"


def test_swim_fixed_text_contract_rejects_silent_truncation() -> None:
    assert (
        swim_writer._fixed_utf8_bytes("candidate", width=16, label="candidate")
        == b"candidate"
    )
    with np.testing.assert_raises_regex(ValueError, "fixed-width UTF-8"):
        swim_writer._fixed_utf8_bytes("x" * 16, width=16, label="candidate")


def test_swim_manifest_rejects_missing_unexpected_wrong_dtype_and_partial_bundle() -> (
    None
):
    missing = _swim_group()
    _delete_array(missing, "tables/bouts/bout_id")
    assert any(
        "Missing required" in error
        for error in swim_schema.validate_swim_bout_array_manifest(missing)
    )

    unexpected = _swim_group()
    unexpected["tables/bouts"].create_array(
        "invented", data=np.zeros(3, dtype=np.int32)
    )
    assert any(
        "Unexpected compact arrays" in error
        for error in swim_schema.validate_swim_bout_array_manifest(unexpected)
    )

    wrong = _swim_group()
    _delete_array(wrong, "tables/bouts/bout_id")
    wrong["tables/bouts"].create_array("bout_id", data=np.zeros(3, dtype=np.float32))
    assert any(
        "dtype mismatch" in error
        for error in swim_schema.validate_swim_bout_array_manifest(wrong)
    )

    wrong_shape = _swim_group()
    _delete_array(wrong_shape, "tables/bouts/bout_id")
    wrong_shape["tables/bouts"].create_array(
        "bout_id", data=np.zeros(4, dtype=np.int32)
    )
    assert any(
        "Shared dimension" in error
        for error in swim_schema.validate_swim_bout_array_manifest(wrong_shape)
    )

    partial = _swim_group(embedded_axis=False)
    partial["signals"].create_array(
        "frame_indices", data=np.zeros((3, 2), dtype=np.int64)
    )
    assert any(
        "rank mismatch" in error
        for error in swim_schema.validate_swim_bout_array_manifest(partial)
    )


def test_swim_manifest_rejects_recomputed_nested_tampering() -> None:
    group = _swim_group()
    tampered = deepcopy(group.attrs[MANIFEST_ATTRIBUTE])
    tampered["payload"]["arrays"][0]["fill_semantics"] = "invented"
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    group.attrs[MANIFEST_ATTRIBUTE] = tampered
    errors = swim_schema.validate_swim_bout_array_manifest(group)
    assert "array_schema_manifest does not equal the executable schema" in errors

    dtype_tampered = _swim_group()
    field_dtypes = dict(dtype_tampered["tables/bouts"].attrs["field_dtypes"])
    field_dtypes["bout_id"] = "float32"
    dtype_tampered["tables/bouts"].attrs["field_dtypes"] = field_dtypes
    assert any(
        "field_dtypes mismatch" in error
        for error in swim_schema.validate_swim_bout_array_manifest(dtype_tampered)
    )


def test_bout_kinematics_manifest_is_exact_with_optional_eye_bundle() -> None:
    with_eye = _bout_group(eye=True)
    without_eye = _bout_group(eye=False)

    assert bout_schema.validate_bout_kinematics_array_manifest(with_eye) == ()
    assert bout_schema.validate_bout_kinematics_array_manifest(without_eye) == ()
    assert with_eye.attrs[MANIFEST_ATTRIBUTE]["payload"][
        "enabled_optional_bundles"
    ] == ["eye_gaze_metrics"]


def test_exact_compact_manifests_bind_byte_planner_adoption_explicitly() -> None:
    swim = _swim_group()
    swim_manifest = swim_schema.build_swim_bout_array_manifest(
        swim,
        byte_planner_adopted=True,
    )
    swim.attrs[MANIFEST_ATTRIBUTE] = swim_manifest
    assert swim_manifest["payload"]["physical_policy_owner"] == (
        "analysis_storage_planning_v1"
    )
    assert swim_manifest["payload"]["byte_planner_adopted"] is True
    assert all(
        declaration["byte_planner_adopted"] is True
        for declaration in swim_manifest["payload"]["arrays"]
    )
    assert (
        swim_schema.validate_swim_bout_array_manifest(
            swim,
            byte_planner_adopted=True,
        )
        == ()
    )
    assert swim_schema.validate_swim_bout_array_manifest(swim)

    bout = _bout_group()
    bout_manifest = bout_schema.build_bout_kinematics_array_manifest(
        bout,
        byte_planner_adopted=True,
    )
    bout.attrs[MANIFEST_ATTRIBUTE] = bout_manifest
    assert bout_manifest["payload"]["physical_policy_owner"] == (
        "analysis_storage_planning_v1"
    )
    assert bout_manifest["payload"]["byte_planner_adopted"] is True
    assert (
        bout_schema.validate_bout_kinematics_array_manifest(
            bout,
            byte_planner_adopted=True,
        )
        == ()
    )
    assert bout_schema.validate_bout_kinematics_array_manifest(bout)


def test_bout_kinematics_manifest_rejects_missing_unexpected_wrong_dtype_and_tampering() -> (
    None
):
    missing = _bout_group()
    _delete_array(missing, "movement_metrics/bout_id")
    assert any(
        "Missing required" in error
        for error in bout_schema.validate_bout_kinematics_array_manifest(missing)
    )

    unexpected = _bout_group()
    unexpected.create_array("frame_counts", data=np.zeros(3, dtype=np.int32))
    assert any(
        "Unexpected compact arrays" in error
        for error in bout_schema.validate_bout_kinematics_array_manifest(unexpected)
    )

    wrong = _bout_group()
    _delete_array(wrong, "heading_metrics/net_delta_heading_deg")
    wrong["heading_metrics"].create_array(
        "net_delta_heading_deg", data=np.zeros(3, dtype=np.float32)
    )
    assert any(
        "dtype mismatch" in error
        for error in bout_schema.validate_bout_kinematics_array_manifest(wrong)
    )

    tampered = _bout_group()
    manifest = deepcopy(tampered.attrs[MANIFEST_ATTRIBUTE])
    manifest["payload"]["arrays"][0]["access_pattern"] = "per_row"
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    tampered.attrs[MANIFEST_ATTRIBUTE] = manifest
    assert "array_schema_manifest does not equal the executable schema" in (
        bout_schema.validate_bout_kinematics_array_manifest(tampered)
    )


def test_bout_kinematics_compact_writer_emits_the_exact_manifest() -> None:
    group = zarr.group()
    group.attrs.update(
        {
            "schema_id": bout_schema.BOUT_KINEMATICS_RUN_SCHEMA_ID,
            "schema_version": bout_schema.BOUT_KINEMATICS_RUN_SCHEMA_VERSION,
            "layout": bout_schema.BOUT_KINEMATICS_LAYOUT,
            "heading_levels": ["heading_smoothed"],
            "analysis_levels": ["movement", "heading_smoothed", "eye_gaze"],
            "default_heading_level": "heading_smoothed",
        }
    )
    movement = np.zeros(2, dtype=bout_writer._movement_metrics_dtype())
    heading = np.zeros(2, dtype=bout_writer._metrics_dtype())
    eye = np.zeros(2, dtype=bout_writer._eye_gaze_metrics_dtype())
    bout_writer._write_compact_bout_kinematics_tables(
        group,
        movement_metrics=movement,
        movement_attrs={},
        metrics_by_level={"heading_smoothed": heading},
        heading_levels=["heading_smoothed"],
        default_heading_level="heading_smoothed",
        heading_table_attrs={},
        eye_gaze_metrics=eye,
        eye_gaze_attrs={},
        output_shard_rows=262_144,
    )

    with np.testing.assert_raises_regex(
        ValueError, "Unmanifested compact bout-kinematics"
    ):
        bout_writer.resolve_bout_kinematics_tables(group)
    legacy_records, _legacy_attrs, _legacy_table_attrs = (
        bout_writer.resolve_bout_kinematics_tables(group, legacy_compatibility=True)
    )
    assert set(legacy_records) == {"movement", "heading_smoothed", "eye_gaze"}

    bout_schema.write_bout_kinematics_array_manifest(group)

    assert bout_schema.validate_bout_kinematics_array_manifest(group) == ()
    records, _attrs, _table_attrs = bout_writer.resolve_bout_kinematics_tables(group)
    assert set(records) == {"movement", "heading_smoothed", "eye_gaze"}
