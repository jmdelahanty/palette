"""Static native geometry consistency, never physical registration admission."""

from __future__ import annotations

import re

import h5py
import numpy as np

from .common import canonical_json, digest, exact_keys, require, same_json, text
from .hdf5_types import dataset_bytes
from .schema import contract, read_json

INPUT = "/geometry/correspondence/input"
AUTHORITY = "/geometry/correspondence/authority_json"
NAMESPACE = "/definitions/source_namespaces/current_geometry_json"
NAMESPACE_BINDINGS = {
    "/calibration_snapshot": "/geometry/calibration",
    "/display_snapshot": "/metadata/display",
    "/presentation_mapping": "/geometry/presentation",
    "/rig_snapshot": "/metadata/rig",
    "/runtime_geometry_contract": "/geometry/runtime",
    "/stimulus_renderer_snapshot": "/geometry/renderer",
}


def _input_fields(h5, definition):
    fields = {}
    for suffix, key in (("", "input_attributes"), ("/renderer", "renderer_attributes")):
        group = h5[INPUT + suffix]
        require(
            set(group.attrs) == set(definition[key]), "geometry_closed_input_attributes"
        )
        for name, expected in definition[key].items():
            attr, value = group.attrs.get_id(name), group.attrs[name]
            require(attr.shape == (), "geometry_attribute_not_scalar")
            field = ("renderer/" if suffix else "") + name
            if expected == "utf8":
                string = h5py.check_string_dtype(attr.dtype)
                require(
                    string is not None and string.encoding == "utf-8",
                    "geometry_attribute_not_utf8",
                )
                fields[field] = {"dtype": expected, "value": text(value, field)}
            else:
                require(attr.dtype == np.dtype(expected), "geometry_attribute_dtype")
                require(np.isfinite(value), "geometry_attribute_nonfinite")
                fields[field] = {"dtype": expected}
                fields[field]["value" if expected == "<i8" else "bits_le_hex"] = (
                    int(value)
                    if expected == "<i8"
                    else np.asarray(value, dtype="<f8").tobytes().hex()
                )
    return fields


def _matrix(h5, path):
    dataset = h5[path]
    require(
        dataset.shape == (3, 3) and dataset.dtype == np.dtype("<f8"),
        "geometry_matrix_type",
    )
    values = dataset[()]
    require(np.isfinite(values).all(), "geometry_matrix_nonfinite")
    a = values.reshape(-1)
    determinant = (
        a[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
    )
    require(
        np.isfinite(determinant) and abs(determinant) > 1e-12,
        "geometry_matrix_singular",
    )
    return values


def _validate_static_inputs(h5, matrices):
    a, r = h5[INPUT].attrs, h5[INPUT + "/renderer"].attrs
    require(
        re.fullmatch(r"arena_[1-9][0-9]*", a["arena_id"]) is not None
        and r["arena_id"] == a["arena_id"]
        and r["texture_origin"] == "top_left"
        and r["evidence_kind"] == "supplied_static_snapshot_claim",
        "geometry_renderer_identity",
    )
    for name in (
        "native_source_width_px",
        "native_source_height_px",
        "final_display_width_px",
        "final_display_height_px",
    ):
        require(a[name] > 0, "geometry_extent_nonpositive")
    for axis, dimension in (("x", "width"), ("y", "height")):
        extent, center = r[f"texture_{dimension}_px"], r[f"texture_center_{axis}"]
        origin = a[f"effective_arena_origin_{axis}_px"]
        require(
            extent > 0
            and 0 <= center <= extent
            and 0 <= origin
            and origin + extent <= a[f"final_display_{dimension}_px"],
            "geometry_renderer_extent",
        )
    forward, inverse = matrices
    for left, right in ((forward, inverse), (inverse, forward)):
        # Match the producer's scalar accumulation and tolerance.
        for row in range(3):
            for column in range(3):
                value = sum(
                    float(left[row, k]) * float(right[k, column]) for k in range(3)
                )
                require(
                    np.isfinite(value) and abs(value - int(row == column)) <= 1e-8,
                    "geometry_inverse_mismatch",
                )


def _validate_geometry_bindings(h5, receipt, matrices, binding):
    a, r, session = (
        h5[INPUT].attrs,
        h5[INPUT + "/renderer"].attrs,
        h5["/metadata/session"].attrs,
    )
    camera, arena = text(a["camera_serial"], "camera_serial"), text(
        a["arena_id"], "arena_id"
    )
    require(
        camera == binding["camera_serial"] and session["arena_id"] == arena,
        "geometry_source_identity",
    )
    calibration = "/geometry/calibration/" + camera + "/homography_matrix"
    require(
        a["calibration_authority_ref"]
        == "/calibration_snapshot/" + camera + "/homography_matrix"
        and a["runtime_geometry_contract_ref"]
        == "/runtime_geometry_contract/contract_json",
        "geometry_source_reference_role",
    )
    require(
        receipt["forward_reuses_calibration"] is True
        and _matrix(h5, calibration).tobytes() == matrices[0].tobytes(),
        "geometry_forward_snapshot_mismatch",
    )
    matrix_digest = digest(canonical_json(matrices[0].tolist()))
    require(
        a["calibration_authority_sha256"] == matrix_digest,
        "geometry_calibration_digest",
    )
    expected_matrix_attrs = {
        "camera_id": camera,
        "arena_config_name": arena,
        "source_frame": "camera_view_px",
        "dest_frame": "final_display_canvas_px",
        "runtime_matrix_checksum_sha256": matrix_digest,
    }
    require(
        all(
            h5[calibration].attrs.get(k) == v for k, v in expected_matrix_attrs.items()
        ),
        "geometry_calibration_attributes",
    )
    config = read_json(h5, receipt["calibration_config_ref"])
    cameras = [
        value for value in config["camera_calibrations"] if value["camera_id"] == camera
    ]
    require(len(cameras) == 1, "geometry_camera_snapshot_not_unique")
    for dimension in ("width", "height"):
        require(
            cameras[0][f"native_{dimension}_px"] == a[f"native_source_{dimension}_px"]
            and session[f"stimulus_output_{dimension}"]
            == a[f"final_display_{dimension}_px"],
            "geometry_source_display_extent",
        )
    runtime = read_json(h5, receipt["runtime_ref"])
    require(
        runtime["schema_id"] == "citrus.session.runtime_geometry_contract"
        and runtime["schema_version"] == 1
        and runtime["identity"]["arena_id"] == arena
        and runtime["identity"]["camera_id"] == camera,
        "geometry_runtime_identity",
    )
    require(
        a["runtime_geometry_contract_sha256"] == receipt["runtime_sha256"]
        and h5[receipt["runtime_ref"]].attrs.get("checksum_sha256")
        == receipt["runtime_sha256"],
        "geometry_runtime_digest",
    )
    region = runtime["effective_geometry"]["arena_region"]
    for axis, dimension in (("x", "width"), ("y", "height")):
        require(
            region["origin_px"][axis] == a[f"effective_arena_origin_{axis}_px"]
            and region["size_px"][dimension] == r[f"texture_{dimension}_px"],
            "geometry_runtime_region",
        )
    require(
        runtime["readiness"]["accepted_commissioned_projection"] is True,
        "geometry_recorded_projection_not_accepted",
    )
    if "/geometry/renderer" in h5:
        observed = h5["/geometry/renderer/" + arena]
        for name in (
            "active_stimulus_mode",
            "texture_origin",
            "texture_width_px",
            "texture_height_px",
        ):
            require(observed.attrs[name] == r[name], "geometry_renderer_snapshot")
        for name in ("texture_center_x", "texture_center_y"):
            require(
                np.asarray(
                    observed["custom_coordinates"].attrs[name], dtype="<f8"
                ).tobytes()
                == np.asarray(r[name], dtype="<f8").tobytes(),
                "geometry_renderer_center_bits",
            )
    return calibration


def validate_geometry(h5, binding, descriptors):
    definition = contract("experimental_h5_geometry_v1.json")
    receipt = read_json(h5, "/geometry/correspondence/receipt_json", canonical=True)
    exact_keys(receipt, definition["receipt_fields"], "geometry_receipt")
    require(
        receipt["schema_id"] == "citrus.experimental_h5.geometry_receipt"
        and type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["status"] == "complete"
        and receipt["reason"] == ""
        and receipt["scope"] == definition["receipt_scope"]
        and receipt["input_encoding"] == definition["input_encoding"]
        and receipt["input_ref"] == INPUT,
        "geometry_receipt_contract",
    )
    fields = _input_fields(h5, definition)
    matrices = [_matrix(h5, INPUT + "/" + name) for name in definition["matrices"]]
    require(
        receipt["input_sha256"]
        == digest(
            b"citrus.geometry_input.v1\n"
            + canonical_json(fields)
            + b"\n"
            + b"".join(value.tobytes() for value in matrices)
        ),
        "geometry_input_digest",
    )
    paths = {
        "authority": AUTHORITY,
        "runtime": "/geometry/runtime/contract_json",
        "presentation": "/geometry/presentation/contract_json",
        "calibration_config": "/geometry/calibration/arena_config_json",
        "source_namespace": NAMESPACE,
    }
    for name, path in paths.items():
        require(
            receipt[name + "_ref"] == path
            and receipt[name + "_sha256"] == digest(dataset_bytes(h5[path])),
            f"geometry_reference_digest:{name}",
        )
    require(
        same_json(
            read_json(h5, NAMESPACE, canonical=True),
            {
                "schema_id": "citrus.experimental_h5.geometry_source_namespaces",
                "schema_version": 1,
                "scope": "captured_geometry_h5_references_not_filesystem_paths_or_asset_acceptance",
                "bindings": NAMESPACE_BINDINGS,
            },
        )
        and h5[NAMESPACE].attrs.get("checksum_sha256")
        == receipt["source_namespace_sha256"],
        "geometry_source_namespace",
    )
    _validate_static_inputs(h5, matrices)
    calibration = _validate_geometry_bindings(h5, receipt, matrices, binding)
    expected = {
        "schema_id": "citrus.experimental_h5.coordinate_authority",
        "schema_version": 1,
        "coordinate_frame": "arena_relative_canvas_px",
        "origin": "top_left_of_active_arena",
        "units": "px",
        "x_axis": "right",
        "y_axis": "down",
        "static_input_ref": INPUT,
        "renderer_claim_ref": INPUT + "/renderer",
        "runtime_geometry_contract_ref": paths["runtime"],
        "calibration_authority_ref": calibration,
        "source_namespace_ref": NAMESPACE,
        "presentation_mapping_ref": paths["presentation"],
        "physical_readiness_source": paths["runtime"] + "#/readiness",
        "transform_roles": {},
    }
    for name, spec in definition["matrices"].items():
        expected[name + "_ref"] = INPUT + "/" + name
        expected["transform_roles"][name + "_ref"] = {
            key: spec[key]
            for key in ("source_space", "destination_space", "transform_direction")
        }
    require(
        same_json(read_json(h5, AUTHORITY, canonical=True), expected),
        "geometry_authority_roles",
    )
    for path, descriptor in descriptors.items():
        if "geometry_ref" in descriptor["references"]:
            require(
                descriptor["references"]["geometry_ref"] == AUTHORITY,
                f"geometry_table_reference:{path}",
            )
    from .presentation import validate_presentation

    validate_presentation(h5, receipt)
    return receipt
