from __future__ import annotations

import copy
import hashlib
import inspect
import json
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from fisheye.shared.directed_transform import (
    DIRECTED_TRANSFORM_ATTR,
    DIRECTED_TRANSFORM_DIGEST_SUFFIX,
    TransformReferenceExtent,
    homography_matrix_sha256,
)
from fisheye.shared.selected_calibration import (
    ACTIVE_CAMERA_CALIBRATION_REF_ATTR,
    ACTIVE_CAMERA_ID_ATTR,
    ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR,
    CAMERA_ID_ATTR,
    CANONICAL_AXES,
    CANONICAL_COORDINATE_ORIGIN,
    CANONICAL_DEST_FRAME,
    CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
    CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
    CANONICAL_IMAGE_SPACE,
    CANONICAL_SOURCE_FRAME,
    NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE,
    PIXEL_TO_MM_DERIVATION,
    SELECTED_CALIBRATION_MANIFEST_ATTR,
    SELECTED_CALIBRATION_MANIFEST_DIGEST_SUFFIX,
    SELECTED_CALIBRATION_TRANSFORM_ATTR,
    SELECTED_CALIBRATION_TRANSFORM_DIGEST_ATTR,
    SELECTED_OUTPUT_GEOMETRY_ATTR,
    SELECTED_OUTPUT_NAME_ATTR,
    SELECTED_OUTPUT_TRANSFORM_TOKEN_ATTR,
    SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION,
    SOURCE_DISPLAY_DATASET_PATH,
    SOURCE_DISPLAY_DATASET_PATH_ATTR,
    SOURCE_DISPLAY_DATASET_SHA256_ATTR,
    SOURCE_DISPLAY_EVIDENCE_ATTR,
    SOURCE_DISPLAY_EVIDENCE_DIGEST_SUFFIX,
    SOURCE_DISPLAY_GROUP_PATH,
    SOURCE_HOMOGRAPHY_EVIDENCE_ATTR,
    SOURCE_HOMOGRAPHY_EVIDENCE_DIGEST_SUFFIX,
    YAML_HOMOGRAPHY_PAYLOAD_SOURCE,
    YAML_HOMOGRAPHY_SERIALIZATION_FORMAT,
    SelectedCalibrationSnapshot,
    SelectedCalibrationError,
    VerifiedSelectedCameraSourceEvidence,
    VerifiedSelectedDisplaySourceEvidence,
    VerifiedSelectedHomographySourceEvidence,
    build_selected_camera_source_evidence_from_h5_values,
    build_selected_display_source_evidence_from_h5_values,
    build_selected_homography_source_evidence_from_h5_values,
    camera_calibration_attrs,
    load_selected_calibration_snapshot,
    parse_selected_camera_source_evidence,
    parse_selected_calibration_manifest,
    parse_selected_display_source_evidence,
    parse_selected_homography_source_evidence,
    require_bound_selected_calibration_snapshot,
    selected_calibration_paths,
    source_arena_config_dataset_sha256,
    source_artifact_from_homography_evidence,
    source_display_dataset_sha256,
    stamp_selected_calibration_snapshot,
)
from fisheye.shared.proof_verification import proof_verification_scope


STIMULUS_RUN = "stim_1"
CAMERA_ID = "2010093"
SOURCE_H5_PATH = "/recording/raw/stimulus.h5"
PATHS = selected_calibration_paths(
    stimulus_run=STIMULUS_RUN,
    camera_id=CAMERA_ID,
)
SOURCE_EXTENT = TransformReferenceExtent(
    width=4512,
    height=4512,
    units="px",
    authority=f"{PATHS.camera_calibration_path}@native_width_px,native_height_px",
)
TARGET_EXTENT = TransformReferenceExtent(
    width=1920,
    height=1080,
    units="px",
    authority=(
        f"{PATHS.stimulus_run_path}/display_snapshot"
        "@selected_output_geometry"
    ),
)
NON_SELF_INVERSE = np.asarray(
    [
        [0.4, 0.02, 25.0],
        [-0.01, 0.35, 70.0],
        [0.0001, -0.0002, 1.0],
    ],
    dtype=np.float64,
)
ARENA_CONFIG = {
    "active_camera_id": CAMERA_ID,
    "calculated_z_eff_mm": 20.0,
    "camera_calibrations": [
        {
            "camera_id": "decoy-first-camera",
            "native_width_px": 640,
            "native_height_px": 480,
            "pixels_per_mm_camera": 12.0,
            "pixels_per_mm_projector": 3.0,
            "real_world_ref_mm": 10.0,
        },
        {
            "camera_id": CAMERA_ID,
            "native_width_px": 4512,
            "native_height_px": 4512,
            "pixels_per_mm_camera": 25.0,
            "pixels_per_mm_projector": 4.0,
            "real_world_ref_mm": 10.0,
        },
    ],
}
CAMERA_GROUP_ATTRS = {
    "pixels_per_mm_camera": 25.0,
    "pixels_per_mm_projector": 4.0,
    "real_world_ref_mm": 10.0,
}
SOURCE_DISPLAY_BLOCK = (
    "DP-3 connected 1920x1080+3840+0 "
    "(normal left inverted right x axis y axis) 0mm x 0mm\n"
    "   1920x1080     60.00*+\n"
).encode("utf-8")
SOURCE_DISPLAY_ATTRS = {
    "selected_output_name": "DP-3",
    "selected_output_connection_state": "connected",
    "selected_output_geometry": "1920x1080+3840+0",
    "selected_output_transform_token": "normal",
    "selected_output_transform_raw": (
        "normal left inverted right x axis y axis"
    ),
}
ARTIFACT_ATTRS = {
    "homography_provenance_schema": "citrus.homography_provenance.v1",
    "homography_artifact_path": "/rig/calibration/homography.yml",
    "homography_artifact_exists": "true",
    "homography_artifact_checksum_algorithm": "fnv1a64",
    "homography_artifact_checksum_fnv1a64": "f999f671b0ebd9fd",
    "homography_artifact_size_bytes": 347,
    "homography_artifact_mtime_unix_ns": 1779901497426733358,
}


def _camera_evidence(
    *,
    source_h5_path: str = SOURCE_H5_PATH,
    arena_config: dict[str, object] | None = None,
    camera_group_attrs: dict[str, object] | None = None,
    expected_camera_id: str = CAMERA_ID,
    camera_group_path: str | None = None,
):
    config = ARENA_CONFIG if arena_config is None else arena_config
    attrs = CAMERA_GROUP_ATTRS if camera_group_attrs is None else camera_group_attrs
    return build_selected_camera_source_evidence_from_h5_values(
        source_h5_path=source_h5_path,
        arena_config_raw=json.dumps(config, separators=(",", ":")).encode("utf-8"),
        camera_group_path=(
            f"/calibration_snapshot/{expected_camera_id}"
            if camera_group_path is None
            else camera_group_path
        ),
        camera_group_attrs=attrs,
        expected_camera_id=expected_camera_id,
    )


def _camera_evidence_from_raw_json(
    arena_config_raw: bytes | str,
):
    return build_selected_camera_source_evidence_from_h5_values(
        source_h5_path=SOURCE_H5_PATH,
        arena_config_raw=arena_config_raw,
        camera_group_path=f"/calibration_snapshot/{CAMERA_ID}",
        camera_group_attrs=CAMERA_GROUP_ATTRS,
        expected_camera_id=CAMERA_ID,
    )


def _display_evidence(
    *,
    source_h5_path: str = SOURCE_H5_PATH,
    display_group_path: str = SOURCE_DISPLAY_GROUP_PATH,
    display_group_attrs: dict[str, object] | None = None,
    selected_output_dataset_path: str = SOURCE_DISPLAY_DATASET_PATH,
    selected_output_block_raw: bytes | str = SOURCE_DISPLAY_BLOCK,
):
    return build_selected_display_source_evidence_from_h5_values(
        source_h5_path=source_h5_path,
        display_group_path=display_group_path,
        display_group_attrs=(
            SOURCE_DISPLAY_ATTRS
            if display_group_attrs is None
            else display_group_attrs
        ),
        selected_output_dataset_path=selected_output_dataset_path,
        selected_output_block_raw=selected_output_block_raw,
    )


def _raw_homography_attrs(
    *,
    kind: str,
    camera_id: str = CAMERA_ID,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    attrs: dict[str, object] = {
        "source_frame": CANONICAL_SOURCE_FRAME,
        "dest_frame": CANONICAL_DEST_FRAME,
        "axes": CANONICAL_AXES,
        "coordinate_origin": CANONICAL_COORDINATE_ORIGIN,
        "image_space": CANONICAL_IMAGE_SPACE,
        "camera_id": camera_id,
        "canvas_name": "shadow",
        **ARTIFACT_ATTRS,
        "homography_payload_source": (
            NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE
            if kind == "numeric"
            else YAML_HOMOGRAPHY_PAYLOAD_SOURCE
        ),
    }
    if kind == "yaml":
        attrs["serialization_format"] = YAML_HOMOGRAPHY_SERIALIZATION_FORMAT
    if overrides:
        attrs.update(overrides)
    return attrs


def _yaml_payload(matrix: np.ndarray = NON_SELF_INVERSE, *, suffix: str = "") -> bytes:
    values = ", ".join(format(float(value), ".17g") for value in matrix.ravel())
    return (
        "%YAML:1.0\n"
        "---\n"
        "homography_matrix: !!opencv-matrix\n"
        "   rows: 3\n"
        "   cols: 3\n"
        "   dt: d\n"
        f"   data: [ {values} ]\n"
        f"{suffix}"
    ).encode("utf-8")


def _homography_evidence(
    *,
    source_h5_path: str = SOURCE_H5_PATH,
    expected_camera_id: str = CAMERA_ID,
    numeric_dataset_path: str | None = None,
    numeric_matrix: np.ndarray = NON_SELF_INVERSE,
    numeric_dataset_attrs: dict[str, object] | None = None,
    yaml_dataset_path: str | None = None,
    yaml_dataset_raw: bytes | str | None = None,
    yaml_dataset_attrs: dict[str, object] | None = None,
):
    prefix = f"/calibration_snapshot/{expected_camera_id}"
    return build_selected_homography_source_evidence_from_h5_values(
        source_h5_path=source_h5_path,
        expected_camera_id=expected_camera_id,
        numeric_dataset_path=(
            f"{prefix}/homography_matrix"
            if numeric_dataset_path is None
            else numeric_dataset_path
        ),
        numeric_matrix=numeric_matrix,
        numeric_dataset_attrs=(
            _raw_homography_attrs(kind="numeric", camera_id=expected_camera_id)
            if numeric_dataset_attrs is None
            else numeric_dataset_attrs
        ),
        yaml_dataset_path=(
            f"{prefix}/homography_matrix_yml"
            if yaml_dataset_path is None
            else yaml_dataset_path
        ),
        yaml_dataset_raw=(
            _yaml_payload(numeric_matrix)
            if yaml_dataset_raw is None
            else yaml_dataset_raw
        ),
        yaml_dataset_attrs=(
            _raw_homography_attrs(kind="yaml", camera_id=expected_camera_id)
            if yaml_dataset_attrs is None
            else yaml_dataset_attrs
        ),
    )


SOURCE_CAMERA = _camera_evidence()
SOURCE_DISPLAY = _display_evidence()
SOURCE_HOMOGRAPHY = _homography_evidence()
SOURCE_ARTIFACT = source_artifact_from_homography_evidence(SOURCE_HOMOGRAPHY)
_ARCHIVE_TOKEN = object()


class FakeGroup(dict[str, object]):
    def __init__(
        self,
        *,
        path: str,
        attrs: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.attrs = attrs if attrs is not None else {}
        self._coordinate_archive_token = _ARCHIVE_TOKEN


class FakeArray:
    def __init__(self, data: np.ndarray, *, path: str) -> None:
        self.data = np.asarray(data, dtype=np.float64).copy()
        self.path = path
        self.attrs: dict[str, object] = {}
        self._coordinate_archive_token = _ARCHIVE_TOKEN

    def __getitem__(self, key):
        return self.data[key]


class FailOnceAttrs(dict[str, object]):
    """Mapping that partially performs one operation and then fails once."""

    def __init__(self, value: dict[str, object], *, operation: str) -> None:
        super().__init__(copy.deepcopy(value))
        self.operation = operation
        self.armed = True

    def update(self, *args: object, **kwargs: object) -> None:
        incoming = dict(*args, **kwargs)
        if self.armed and self.operation == "update":
            self.armed = False
            if incoming:
                name = next(iter(incoming))
                dict.__setitem__(self, name, incoming[name])
            raise RuntimeError("injected partial update failure")
        dict.update(self, incoming)

    def __delitem__(self, name: str) -> None:
        dict.__delitem__(self, name)
        if self.armed and self.operation == "delete":
            self.armed = False
            raise RuntimeError("injected post-delete failure")


class FailEveryUpdateAttrs(dict[str, object]):
    """Mapping used to prove an incomplete rollback is reported explicitly."""

    def update(self, *args: object, **kwargs: object) -> None:
        incoming = dict(*args, **kwargs)
        if incoming:
            name = next(iter(incoming))
            dict.__setitem__(self, name, incoming[name])
        raise RuntimeError("injected persistent update failure")


class ClearingUnrelatedAttrs(dict[str, object]):
    def update(self, *args: object, **kwargs: object) -> None:
        incoming = dict(*args, **kwargs)
        self.pop("unrelated_keep_me", None)
        super().update(incoming)


class CoercingSelectedTransformAttrs(dict[str, object]):
    def __init__(self, value: dict[str, object]) -> None:
        super().__init__(value)
        self.armed = True

    def update(self, *args: object, **kwargs: object) -> None:
        incoming = copy.deepcopy(dict(*args, **kwargs))
        record = incoming.get(SELECTED_CALIBRATION_TRANSFORM_ATTR)
        if self.armed and isinstance(record, dict):
            self.armed = False
            record["schema_version"] = 2.0
        super().update(incoming)


def _build_root(
    *,
    include_scalars: bool = True,
) -> tuple[FakeGroup, FakeGroup, FakeGroup, FakeGroup, FakeArray]:
    matrix = FakeArray(NON_SELF_INVERSE, path=PATHS.homography_array_path)
    if include_scalars:
        source_camera = SOURCE_CAMERA
    else:
        no_scalars_config = copy.deepcopy(ARENA_CONFIG)
        no_scalars_config.pop("calculated_z_eff_mm", None)
        selected_record = no_scalars_config["camera_calibrations"][1]
        assert isinstance(selected_record, dict)
        for name in CAMERA_GROUP_ATTRS:
            selected_record.pop(name, None)
        source_camera = _camera_evidence(
            arena_config=no_scalars_config,
            camera_group_attrs={},
        )
    camera = FakeGroup(
        path=PATHS.camera_calibration_path,
        attrs={"native_width_px": 1, "pixels_per_mm_camera": 999.0},
    )
    camera["homography_matrix"] = matrix
    calibration = FakeGroup(path=PATHS.calibration_path)
    calibration[CAMERA_ID] = camera
    display = FakeGroup(
        path=PATHS.display_snapshot_path,
        attrs={
            SELECTED_OUTPUT_NAME_ATTR: "stale-output",
            SELECTED_OUTPUT_GEOMETRY_ATTR: "640x480+0+0",
            SELECTED_OUTPUT_TRANSFORM_TOKEN_ATTR: "left",
        },
    )
    run = FakeGroup(path=PATHS.stimulus_run_path)
    run["calibration"] = calibration
    run["display_snapshot"] = display
    runs = FakeGroup(path="analysis/stimulus_runs")
    runs[STIMULUS_RUN] = run
    analysis = FakeGroup(path="analysis")
    analysis["stimulus_runs"] = runs
    root = FakeGroup(path="")
    root["analysis"] = analysis
    calibration._root_node = root
    stamp_selected_calibration_snapshot(
        calibration,
        camera,
        display,
        matrix,
        root_node=root,
        stimulus_run=STIMULUS_RUN,
        camera_id=CAMERA_ID,
        source_camera=source_camera,
        source_display=SOURCE_DISPLAY,
        source_homography=SOURCE_HOMOGRAPHY,
    )
    return root, calibration, camera, display, matrix


def _load(root: FakeGroup, **overrides: object):
    kwargs: dict[str, object] = {
        "stimulus_run": STIMULUS_RUN,
        "expected_camera_id": CAMERA_ID,
        "expected_from_space_id": CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
        "expected_to_space_id": CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
        "expected_source_reference_extent": SOURCE_EXTENT,
        "expected_target_reference_extent": TARGET_EXTENT,
    }
    kwargs.update(overrides)
    return load_selected_calibration_snapshot(root, **kwargs)


def _stamp_again(
    calibration: FakeGroup,
    camera: FakeGroup,
    display: FakeGroup,
    matrix: FakeArray,
    **overrides: object,
):
    kwargs: dict[str, object] = {
        "stimulus_run": STIMULUS_RUN,
        "camera_id": CAMERA_ID,
        "source_camera": SOURCE_CAMERA,
        "source_display": SOURCE_DISPLAY,
        "source_homography": SOURCE_HOMOGRAPHY,
    }
    kwargs.update(overrides)
    return stamp_selected_calibration_snapshot(
        calibration,
        camera,
        display,
        matrix,
        root_node=calibration._root_node,
        **kwargs,
    )


def _attrs_snapshot(*nodes: Any) -> list[dict[str, object]]:
    return [copy.deepcopy(dict(node.attrs)) for node in nodes]


def test_loads_one_valid_builder_bound_non_self_inverse_snapshot() -> None:
    root, calibration, camera, display, matrix = _build_root()

    snapshot = _load(root)

    assert snapshot.camera_id == CAMERA_ID
    np.testing.assert_array_equal(snapshot.homography.matrix, NON_SELF_INVERSE)
    assert not np.allclose(NON_SELF_INVERSE, np.linalg.inv(NON_SELF_INVERSE))
    assert snapshot.source_reference_extent == SOURCE_EXTENT
    assert snapshot.target_reference_extent == TARGET_EXTENT
    assert snapshot.display_output_name == "DP-3"
    assert snapshot.display_output_geometry == "1920x1080+3840+0"
    assert snapshot.pixels_per_mm_camera == 25.0
    assert snapshot.pixel_to_mm == 0.04
    assert snapshot.z_eff_mm == 20.0
    assert snapshot.manifest.source_camera.to_dict() == SOURCE_CAMERA.to_dict()
    assert snapshot.manifest.source_display.to_dict() == SOURCE_DISPLAY.to_dict()
    assert (
        snapshot.manifest.source_homography.to_dict()
        == SOURCE_HOMOGRAPHY.to_dict()
    )
    assert snapshot.manifest.source_artifact == SOURCE_ARTIFACT
    assert camera.attrs["native_width_px"] == 4512
    assert display.attrs[SELECTED_OUTPUT_NAME_ATTR] == "DP-3"
    assert display.attrs[SOURCE_DISPLAY_DATASET_SHA256_ATTR] == (
        SOURCE_DISPLAY.selected_output_dataset_sha256
    )
    assert display.attrs["source_display_dataset_digest_canonicalization"] == (
        SOURCE_DISPLAY_DATASET_DIGEST_CANONICALIZATION
    )
    assert matrix.attrs[SELECTED_CALIBRATION_TRANSFORM_ATTR]["from_space_id"] == (
        CANONICAL_HOMOGRAPHY_FROM_SPACE_ID
    )
    assert matrix.attrs[SELECTED_CALIBRATION_TRANSFORM_ATTR]["direction"] == (
        "source_to_target"
    )
    assert DIRECTED_TRANSFORM_ATTR not in matrix.attrs
    assert (
        f"{DIRECTED_TRANSFORM_ATTR}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}"
        not in matrix.attrs
    )
    manifest_digest_name = (
        f"{SELECTED_CALIBRATION_MANIFEST_ATTR}"
        f"{SELECTED_CALIBRATION_MANIFEST_DIGEST_SUFFIX}"
    )
    assert calibration.attrs[manifest_digest_name] == snapshot.manifest_sha256


def test_bound_snapshot_verification_is_reused_only_inside_one_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _calibration, _camera, _display, _matrix = _build_root()
    snapshot = _load(root)
    calls = 0
    original = SelectedCalibrationSnapshot.assert_verified

    def counted(value: SelectedCalibrationSnapshot) -> None:
        nonlocal calls
        calls += 1
        original(value)

    monkeypatch.setattr(SelectedCalibrationSnapshot, "assert_verified", counted)
    with proof_verification_scope():
        require_bound_selected_calibration_snapshot(snapshot)
        require_bound_selected_calibration_snapshot(snapshot)
        require_bound_selected_calibration_snapshot(snapshot)
        assert calls == 1
    assert calls == 2

    require_bound_selected_calibration_snapshot(snapshot)
    assert calls == 3


def test_display_builder_binds_exact_raw_bytes_and_group_attrs() -> None:
    evidence = SOURCE_DISPLAY
    assert evidence.selected_output_first_line == SOURCE_DISPLAY_BLOCK.decode().splitlines()[0]
    assert evidence.selected_output_name == "DP-3"
    assert evidence.width_px == 1920
    assert evidence.height_px == 1080
    assert evidence.offset_x_px == 3840
    assert evidence.offset_y_px == 0
    assert evidence.selected_output_dataset_sha256 == hashlib.sha256(
        SOURCE_DISPLAY_BLOCK
    ).hexdigest()
    assert source_display_dataset_sha256(SOURCE_DISPLAY_BLOCK) == (
        source_display_dataset_sha256(SOURCE_DISPLAY_BLOCK.decode("utf-8"))
    )

    same_fields_different_raw = _display_evidence(
        selected_output_block_raw=SOURCE_DISPLAY_BLOCK + b"   1280x720 59.0\n"
    )
    assert same_fields_different_raw.selected_output_dataset_sha256 != (
        evidence.selected_output_dataset_sha256
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_output_name", "HDMI-0"),
        ("selected_output_connection_state", "disconnected"),
        ("selected_output_geometry", "1920x1080+0+0"),
        ("selected_output_geometry", "1280x720+3840+0"),
        ("selected_output_transform_token", "left"),
        ("selected_output_transform_raw", "normal"),
    ],
)
def test_display_builder_rejects_attr_and_raw_block_disagreement(
    field: str,
    value: str,
) -> None:
    attrs = dict(SOURCE_DISPLAY_ATTRS)
    attrs[field] = value

    with pytest.raises(SelectedCalibrationError):
        _display_evidence(display_group_attrs=attrs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"display_group_path": "/other"},
            "display_group_path",
        ),
        (
            {"selected_output_dataset_path": "/display_snapshot/other"},
            "wrong H5 dataset",
        ),
        (
            {
                "selected_output_block_raw": (
                    SOURCE_DISPLAY_BLOCK
                    + b"   HDMI-0 disconnected (normal)\n"
                )
            },
            "exactly one unambiguous",
        ),
    ],
)
def test_display_builder_rejects_wrong_paths_or_ambiguous_output_block(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SelectedCalibrationError, match=message):
        _display_evidence(**overrides)


def test_homography_builder_binds_independent_numeric_yaml_and_lineage() -> None:
    evidence = SOURCE_HOMOGRAPHY
    assert evidence.numeric_matrix_sha256 == homography_matrix_sha256(
        NON_SELF_INVERSE
    )
    assert evidence.yaml_matrix_sha256 == evidence.numeric_matrix_sha256
    assert evidence.numeric_dataset_attrs["homography_payload_source"] == (
        NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE
    )
    assert evidence.yaml_dataset_attrs["homography_payload_source"] == (
        YAML_HOMOGRAPHY_PAYLOAD_SOURCE
    )
    assert evidence.yaml_dataset_attrs["serialization_format"] == (
        YAML_HOMOGRAPHY_SERIALIZATION_FORMAT
    )
    assert SOURCE_ARTIFACT.homography_artifact_path == (
        ARTIFACT_ATTRS["homography_artifact_path"]
    )

    same_matrix_different_raw = _homography_evidence(
        yaml_dataset_raw=_yaml_payload(suffix="# exact-source comment\n")
    )
    assert same_matrix_different_raw.yaml_dataset_sha256 != (
        evidence.yaml_dataset_sha256
    )


def test_homography_builder_rejects_numeric_yaml_matrix_disagreement() -> None:
    different = NON_SELF_INVERSE.copy()
    different[0, 2] += 1.0

    with pytest.raises(SelectedCalibrationError, match="payloads disagree"):
        _homography_evidence(yaml_dataset_raw=_yaml_payload(different))


@pytest.mark.parametrize(
    ("yaml_raw", "message"),
    [
        (
            _yaml_payload()
            + (
                "homography_matrix:\n"
                "  rows: 3\n"
                "  cols: 3\n"
                "  dt: d\n"
                "  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
            ).encode("utf-8"),
            "invalid",
        ),
        (
            _yaml_payload().replace(b"dt: d", b"dt: f"),
            "float64",
        ),
        (
            _yaml_payload().replace(b"dt: d", b"dt: d\n   extra: 1"),
            "fields must be exactly",
        ),
    ],
)
def test_homography_builder_rejects_ambiguous_or_inexact_opencv_yaml(
    yaml_raw: bytes,
    message: str,
) -> None:
    with pytest.raises(SelectedCalibrationError, match=message):
        _homography_evidence(yaml_dataset_raw=yaml_raw)


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("numeric", "source_frame", "model_input_px"),
        ("yaml", "dest_frame", "camera_view_px"),
        ("numeric", "axes", "x_right_y_up"),
        ("yaml", "coordinate_origin", "bottom_left"),
        ("numeric", "image_space", "crop"),
        ("yaml", "camera_id", "other"),
        ("numeric", "homography_payload_source", "some_other_source"),
        ("yaml", "homography_payload_source", NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE),
        ("yaml", "serialization_format", "json"),
    ],
)
def test_homography_builder_rejects_noncanonical_controlled_semantics(
    kind: str,
    field: str,
    value: str,
) -> None:
    attrs = _raw_homography_attrs(kind=kind, overrides={field: value})
    kwargs = {f"{kind}_dataset_attrs": attrs}

    with pytest.raises(SelectedCalibrationError, match=field):
        _homography_evidence(**kwargs)


def test_homography_builder_rejects_consistently_reversed_h5_semantics() -> None:
    reversed_fields = {
        "source_frame": CANONICAL_DEST_FRAME,
        "dest_frame": CANONICAL_SOURCE_FRAME,
    }

    with pytest.raises(SelectedCalibrationError, match="source_frame"):
        _homography_evidence(
            numeric_dataset_attrs=_raw_homography_attrs(
                kind="numeric",
                overrides=reversed_fields,
            ),
            yaml_dataset_attrs=_raw_homography_attrs(
                kind="yaml",
                overrides=reversed_fields,
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canvas_name", "other_canvas"),
        ("homography_artifact_path", "/rig/calibration/other.yml"),
        ("homography_artifact_checksum_fnv1a64", "a" * 16),
        ("homography_artifact_size_bytes", 348),
        ("homography_artifact_mtime_unix_ns", 1779901497426733359),
    ],
)
def test_homography_builder_rejects_mixed_numeric_yaml_lineage(
    field: str,
    value: object,
) -> None:
    yaml_attrs = _raw_homography_attrs(kind="yaml", overrides={field: value})

    with pytest.raises(SelectedCalibrationError, match="lineage disagree"):
        _homography_evidence(yaml_dataset_attrs=yaml_attrs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"numeric_dataset_path": "/calibration_snapshot/other/homography_matrix"},
            "numeric_dataset_path",
        ),
        (
            {"yaml_dataset_path": "/calibration_snapshot/other/homography_matrix_yml"},
            "yaml_dataset_path",
        ),
    ],
)
def test_homography_builder_rejects_wrong_selected_camera_paths(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SelectedCalibrationError, match=message):
        _homography_evidence(**kwargs)


def test_camera_builder_selects_exact_active_camera_and_binds_scalars() -> None:
    assert SOURCE_CAMERA.active_camera_id == CAMERA_ID
    assert SOURCE_CAMERA.selected_camera_record_index == 1
    assert SOURCE_CAMERA.native_width_px == 4512
    assert SOURCE_CAMERA.pixel_to_mm == 1.0 / 25.0
    assert SOURCE_CAMERA.arena_config_dataset_sha256 == (
        source_arena_config_dataset_sha256(
            json.dumps(ARENA_CONFIG, separators=(",", ":")).encode("utf-8")
        )
    )
    attrs = camera_calibration_attrs(SOURCE_CAMERA)
    assert attrs["pixel_to_mm_derivation"] == PIXEL_TO_MM_DERIVATION
    assert attrs["z_eff_mm"] == 20.0


@pytest.mark.parametrize("failure", ["wrong_active", "missing_active", "duplicate_active"])
def test_camera_builder_never_falls_back_or_ambiguously_selects(
    failure: str,
) -> None:
    config = copy.deepcopy(ARENA_CONFIG)
    if failure == "wrong_active":
        config["active_camera_id"] = "decoy-first-camera"
    elif failure == "missing_active":
        config["camera_calibrations"] = config["camera_calibrations"][:1]
    else:
        config["camera_calibrations"].append(
            copy.deepcopy(config["camera_calibrations"][1])
        )
    message = (
        "active_camera_id does not match"
        if failure == "wrong_active"
        else "exactly one active-camera"
    )

    with pytest.raises(SelectedCalibrationError, match=message):
        _camera_evidence(arena_config=config)


@pytest.mark.parametrize(
    ("case", "needle", "replacement"),
    [
        (
            "top-level active camera",
            '"active_camera_id":"2010093"',
            '"active_camera_id":"decoy","active_camera_id":"2010093"',
        ),
        ("nested active camera", None, None),
        (
            "camera id",
            '"camera_id":"decoy-first-camera"',
            '"camera_id":"wrong","camera_id":"decoy-first-camera"',
        ),
        (
            "native width",
            '"native_width_px":640',
            '"native_width_px":1,"native_width_px":640',
        ),
        (
            "native height",
            '"native_height_px":480',
            '"native_height_px":1,"native_height_px":480',
        ),
        (
            "camera pixels per mm",
            '"pixels_per_mm_camera":12.0',
            '"pixels_per_mm_camera":1.0,"pixels_per_mm_camera":12.0',
        ),
        (
            "projector pixels per mm",
            '"pixels_per_mm_projector":3.0',
            '"pixels_per_mm_projector":1.0,"pixels_per_mm_projector":3.0',
        ),
        (
            "real-world reference",
            '"real_world_ref_mm":10.0',
            '"real_world_ref_mm":1.0,"real_world_ref_mm":10.0',
        ),
    ],
)
def test_camera_builder_rejects_duplicate_json_keys_recursively(
    case: str,
    needle: str | None,
    replacement: str | None,
) -> None:
    raw = json.dumps(ARENA_CONFIG, separators=(",", ":"))
    if case == "nested active camera":
        raw = (
            '{"unused":{"active_camera_id":"one",'
            '"active_camera_id":"two"},'
            f"{raw[1:]}"
        )
    else:
        assert needle is not None and replacement is not None
        assert needle in raw
        raw = raw.replace(needle, replacement, 1)

    with pytest.raises(SelectedCalibrationError, match="duplicate JSON key"):
        _camera_evidence_from_raw_json(raw.encode("utf-8"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pixels_per_mm_camera", 26.0),
        ("pixels_per_mm_projector", 5.0),
        ("real_world_ref_mm", 11.0),
    ],
)
def test_camera_builder_rejects_mixed_record_and_group_scalars(
    field: str,
    value: float,
) -> None:
    attrs = dict(CAMERA_GROUP_ATTRS)
    attrs[field] = value
    with pytest.raises(SelectedCalibrationError, match=field):
        _camera_evidence(camera_group_attrs=attrs)


def test_parsers_reject_tampered_evidence_digests_and_fields() -> None:
    camera_payload = SOURCE_CAMERA.to_dict()
    camera_payload["selected_camera_record"]["native_width_px"] = 640
    with pytest.raises(SelectedCalibrationError, match="record digest"):
        parse_selected_camera_source_evidence(camera_payload)

    display_payload = SOURCE_DISPLAY.to_dict()
    display_payload["display_group_attrs"]["selected_output_name"] = "HDMI-0"
    with pytest.raises(SelectedCalibrationError, match="attrs digest"):
        parse_selected_display_source_evidence(display_payload)

    homography_payload = SOURCE_HOMOGRAPHY.to_dict()
    homography_payload["numeric_matrix"][0][2] += 1.0
    with pytest.raises(SelectedCalibrationError, match="matrix digest"):
        parse_selected_homography_source_evidence(homography_payload)


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            '"camera_id":"2010093"',
            '"camera_id":"wrong","camera_id":"2010093"',
        ),
        (
            '"camera_calibration":{"native_width_px":4512',
            '"camera_calibration":{"native_width_px":640,'
            '"native_width_px":4512',
        ),
        (
            '"selected_output_geometry":"1920x1080+3840+0"',
            '"selected_output_geometry":"640x480+0+0",'
            '"selected_output_geometry":"1920x1080+3840+0"',
        ),
        (
            '"width_px":1920,"height_px":1080',
            '"width_px":640,"width_px":1920,"height_px":1080',
        ),
    ],
)
@pytest.mark.parametrize("as_bytes", [False, True])
def test_manifest_parser_rejects_duplicate_json_camera_display_and_extent_fields(
    needle: str,
    replacement: str,
    as_bytes: bool,
) -> None:
    _root, calibration, _camera, _display, _matrix = _build_root()
    payload = calibration.attrs[SELECTED_CALIBRATION_MANIFEST_ATTR]
    raw = json.dumps(payload, separators=(",", ":"))
    assert needle in raw
    raw = raw.replace(needle, replacement, 1)
    value: str | bytes = raw.encode("utf-8") if as_bytes else raw

    with pytest.raises(SelectedCalibrationError, match="duplicate JSON key"):
        parse_selected_calibration_manifest(value)


def test_stamper_has_only_builder_evidence_not_free_coordinate_arguments() -> None:
    signature = inspect.signature(stamp_selected_calibration_snapshot)
    for forbidden in (
        "native_width_px",
        "native_height_px",
        "pixels_per_mm_camera",
        "pixel_to_mm",
        "pixels_per_mm_projector",
        "z_eff_mm",
        "selected_output_geometry",
        "expected_from_space_id",
        "expected_to_space_id",
        "source_artifact",
    ):
        assert forbidden not in signature.parameters
    for required in ("source_camera", "source_display", "source_homography"):
        assert signature.parameters[required].default is inspect.Parameter.empty


@pytest.mark.parametrize("kind", ["camera", "display", "homography"])
def test_verified_evidence_dataclasses_cannot_be_nominally_forged(kind: str) -> None:
    verified_type, parser, source = {
        "camera": (
            VerifiedSelectedCameraSourceEvidence,
            parse_selected_camera_source_evidence,
            SOURCE_CAMERA,
        ),
        "display": (
            VerifiedSelectedDisplaySourceEvidence,
            parse_selected_display_source_evidence,
            SOURCE_DISPLAY,
        ),
        "homography": (
            VerifiedSelectedHomographySourceEvidence,
            parse_selected_homography_source_evidence,
            SOURCE_HOMOGRAPHY,
        ),
    }[kind]
    parsed_base = parser(source)

    with pytest.raises(SelectedCalibrationError, match="cannot be constructed directly"):
        verified_type(**vars(parsed_base))
    with pytest.raises(SelectedCalibrationError, match="cannot be constructed directly"):
        verified_type(**vars(parsed_base), _seal=object())


@pytest.mark.parametrize("kind", ["camera", "display", "homography"])
@pytest.mark.parametrize("representation", ["mapping", "parsed_base"])
def test_stamper_rejects_non_builder_evidence_without_any_mutation(
    kind: str,
    representation: str,
) -> None:
    _root, calibration, camera, display, matrix = _build_root()
    nodes = (camera, calibration, display, matrix)
    before = _attrs_snapshot(*nodes)
    verified = {
        "camera": SOURCE_CAMERA,
        "display": SOURCE_DISPLAY,
        "homography": SOURCE_HOMOGRAPHY,
    }[kind]
    parser = {
        "camera": parse_selected_camera_source_evidence,
        "display": parse_selected_display_source_evidence,
        "homography": parse_selected_homography_source_evidence,
    }[kind]
    value = verified.to_dict() if representation == "mapping" else parser(verified)

    with pytest.raises(SelectedCalibrationError, match="builder-validated"):
        _stamp_again(
            calibration,
            camera,
            display,
            matrix,
            **{f"source_{kind}": value},
        )

    assert _attrs_snapshot(*nodes) == before


def test_stamper_derives_and_overwrites_stale_display_and_transform_attrs() -> None:
    _root, calibration, camera, display, matrix = _build_root()
    display.attrs.update(
        {
            SELECTED_OUTPUT_NAME_ATTR: "wrong",
            SELECTED_OUTPUT_GEOMETRY_ATTR: "344x344+0+0",
            SELECTED_OUTPUT_TRANSFORM_TOKEN_ATTR: "left",
            SOURCE_DISPLAY_DATASET_PATH_ATTR: "/wrong",
        }
    )
    stale_transform = copy.deepcopy(
        matrix.attrs[SELECTED_CALIBRATION_TRANSFORM_ATTR]
    )
    stale_transform["from_space_id"] = CANONICAL_HOMOGRAPHY_TO_SPACE_ID
    stale_transform["to_space_id"] = CANONICAL_HOMOGRAPHY_FROM_SPACE_ID
    matrix.attrs[SELECTED_CALIBRATION_TRANSFORM_ATTR] = stale_transform
    matrix.attrs[SELECTED_CALIBRATION_TRANSFORM_DIGEST_ATTR] = "f" * 64

    _stamp_again(calibration, camera, display, matrix)

    assert display.attrs[SELECTED_OUTPUT_NAME_ATTR] == "DP-3"
    assert display.attrs[SELECTED_OUTPUT_GEOMETRY_ATTR] == "1920x1080+3840+0"
    assert display.attrs[SOURCE_DISPLAY_DATASET_PATH_ATTR] == (
        SOURCE_DISPLAY_DATASET_PATH
    )
    transform = matrix.attrs[SELECTED_CALIBRATION_TRANSFORM_ATTR]
    assert transform["from_space_id"] == CANONICAL_HOMOGRAPHY_FROM_SPACE_ID
    assert transform["to_space_id"] == CANONICAL_HOMOGRAPHY_TO_SPACE_ID
    assert transform["target_reference_extent"]["width"] == 1920
    assert transform["target_reference_extent"]["height"] == 1080


def test_future_stamper_refuses_historical_directed_transform_attrs() -> None:
    _root, calibration, camera, display, matrix = _build_root()
    matrix.attrs[DIRECTED_TRANSFORM_ATTR] = {"schema_version": 1}
    matrix.attrs[
        f"{DIRECTED_TRANSFORM_ATTR}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}"
    ] = "f" * 64
    before = _attrs_snapshot(camera, calibration, display, matrix)

    with pytest.raises(SelectedCalibrationError, match="explicit migration"):
        _stamp_again(calibration, camera, display, matrix)

    assert _attrs_snapshot(camera, calibration, display, matrix) == before


@pytest.mark.parametrize("hostile", ["clear_unrelated", "coerce_transform"])
def test_stamper_rejects_attr_clear_or_type_coercion_before_write(
    hostile: str,
) -> None:
    _root, calibration, camera, display, matrix = _build_root()
    if hostile == "clear_unrelated":
        display.attrs["unrelated_keep_me"] = {"nested": [1, 2]}
        display.attrs = ClearingUnrelatedAttrs(dict(display.attrs))
    else:
        matrix.attrs = CoercingSelectedTransformAttrs(dict(matrix.attrs))
    before = _attrs_snapshot(camera, calibration, display, matrix)

    with pytest.raises(SelectedCalibrationError, match="exact trusted.*no write"):
        _stamp_again(calibration, camera, display, matrix)

    assert _attrs_snapshot(camera, calibration, display, matrix) == before


@pytest.mark.parametrize(
    "failure",
    [
        "camera_attrs_immutable",
        "calibration_attrs_immutable",
        "display_attrs_immutable",
        "homography_attrs_immutable",
        "wrong_node_path",
        "different_display_h5",
        "different_homography_h5",
        "persisted_matrix_mismatch",
    ],
)
def test_stamper_failure_is_atomic_across_all_four_attr_targets(
    failure: str,
) -> None:
    _root, calibration, camera, display, matrix = _build_root()
    nodes = (camera, calibration, display, matrix)
    before = _attrs_snapshot(*nodes)
    overrides: dict[str, object] = {}
    if failure == "camera_attrs_immutable":
        camera.attrs = MappingProxyType(dict(camera.attrs))
    elif failure == "calibration_attrs_immutable":
        calibration.attrs = MappingProxyType(dict(calibration.attrs))
    elif failure == "display_attrs_immutable":
        display.attrs = MappingProxyType(dict(display.attrs))
    elif failure == "homography_attrs_immutable":
        matrix.attrs = MappingProxyType(dict(matrix.attrs))
    elif failure == "wrong_node_path":
        matrix.path = f"{PATHS.camera_calibration_path}/other"
    elif failure == "different_display_h5":
        overrides["source_display"] = _display_evidence(
            source_h5_path="/recording/raw/different.h5"
        )
    elif failure == "different_homography_h5":
        overrides["source_homography"] = _homography_evidence(
            source_h5_path="/recording/raw/different.h5"
        )
    else:
        matrix.data[0, 2] += 1.0

    with pytest.raises(SelectedCalibrationError):
        _stamp_again(calibration, camera, display, matrix, **overrides)

    assert _attrs_snapshot(*nodes) == before


@pytest.mark.parametrize(
    ("target", "operation"),
    [
        ("camera", "delete"),
        ("display", "update"),
        ("homography", "update"),
        ("calibration", "update"),
    ],
)
def test_stamper_rejects_hostile_attr_write_hooks_and_can_retry_with_plain_dict(
    target: str,
    operation: str,
) -> None:
    root, calibration, camera, display, matrix = _build_root()
    selected = {
        "camera": camera,
        "display": display,
        "homography": matrix,
        "calibration": calibration,
    }[target]
    selected.attrs = FailOnceAttrs(dict(selected.attrs), operation=operation)
    nodes = (camera, calibration, display, matrix)
    before = _attrs_snapshot(*nodes)

    with pytest.raises(SelectedCalibrationError, match="exact trusted.*no write"):
        _stamp_again(calibration, camera, display, matrix)

    assert _attrs_snapshot(*nodes) == before
    selected.attrs = dict(selected.attrs)
    _stamp_again(calibration, camera, display, matrix)
    assert _load(root).manifest.digest() == load_selected_calibration_snapshot(
        root,
        stimulus_run=STIMULUS_RUN,
        expected_camera_id=CAMERA_ID,
        expected_from_space_id=CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
        expected_to_space_id=CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
        expected_source_reference_extent=SOURCE_EXTENT,
        expected_target_reference_extent=TARGET_EXTENT,
    ).manifest.digest()


def test_stamper_rejects_persistently_hostile_attrs_before_write() -> None:
    _root, calibration, camera, display, matrix = _build_root()
    display.attrs = FailEveryUpdateAttrs(copy.deepcopy(dict(display.attrs)))
    before = _attrs_snapshot(camera, calibration, display, matrix)

    with pytest.raises(SelectedCalibrationError, match="exact trusted.*no write"):
        _stamp_again(calibration, camera, display, matrix)

    assert _attrs_snapshot(camera, calibration, display, matrix) == before


def test_stamper_rejects_homography_for_different_camera_without_mutation() -> None:
    _root, calibration, camera, display, matrix = _build_root()
    before = _attrs_snapshot(camera, calibration, display, matrix)
    other_id = "other-camera"
    other = _homography_evidence(
        expected_camera_id=other_id,
        numeric_dataset_attrs=_raw_homography_attrs(
            kind="numeric",
            camera_id=other_id,
        ),
        yaml_dataset_attrs=_raw_homography_attrs(
            kind="yaml",
            camera_id=other_id,
        ),
    )

    with pytest.raises(SelectedCalibrationError, match="requested camera_id"):
        _stamp_again(
            calibration,
            camera,
            display,
            matrix,
            source_homography=other,
        )

    assert _attrs_snapshot(camera, calibration, display, matrix) == before


def test_loader_binds_persisted_display_and_homography_evidence_to_manifest() -> None:
    root, _calibration, _camera, display, _matrix = _build_root()
    display_payload = copy.deepcopy(display.attrs[SOURCE_DISPLAY_EVIDENCE_ATTR])
    display_payload["source_h5_path"] = "/recording/raw/other.h5"
    parsed_display = parse_selected_display_source_evidence(display_payload)
    display.attrs[SOURCE_DISPLAY_EVIDENCE_ATTR] = parsed_display.to_dict()
    display.attrs[
        f"{SOURCE_DISPLAY_EVIDENCE_ATTR}{SOURCE_DISPLAY_EVIDENCE_DIGEST_SUFFIX}"
    ] = parsed_display.digest()
    with pytest.raises(SelectedCalibrationError, match="display evidence"):
        _load(root)

    root, _calibration, _camera, _display, matrix = _build_root()
    homography_payload = copy.deepcopy(matrix.attrs[SOURCE_HOMOGRAPHY_EVIDENCE_ATTR])
    homography_payload["source_h5_path"] = "/recording/raw/other.h5"
    parsed_homography = parse_selected_homography_source_evidence(
        homography_payload
    )
    matrix.attrs[SOURCE_HOMOGRAPHY_EVIDENCE_ATTR] = parsed_homography.to_dict()
    matrix.attrs[
        f"{SOURCE_HOMOGRAPHY_EVIDENCE_ATTR}"
        f"{SOURCE_HOMOGRAPHY_EVIDENCE_DIGEST_SUFFIX}"
    ] = parsed_homography.digest()
    with pytest.raises(SelectedCalibrationError, match="homography evidence"):
        _load(root)


@pytest.mark.parametrize("surface", ["display", "homography"])
def test_loader_rejects_missing_persisted_source_evidence(surface: str) -> None:
    root, _calibration, _camera, display, matrix = _build_root()
    attrs = display.attrs if surface == "display" else matrix.attrs
    name = (
        SOURCE_DISPLAY_EVIDENCE_ATTR
        if surface == "display"
        else SOURCE_HOMOGRAPHY_EVIDENCE_ATTR
    )
    del attrs[name]
    with pytest.raises(SelectedCalibrationError, match=f"source-{surface} evidence"):
        _load(root)


def test_loader_rejects_tampered_display_transform_matrix_and_manifest() -> None:
    root, _calibration, _camera, display, _matrix = _build_root()
    display.attrs[SELECTED_OUTPUT_GEOMETRY_ATTR] = "1280x720+3840+0"
    with pytest.raises(SelectedCalibrationError, match="display snapshot"):
        _load(root)

    root, _calibration, _camera, _display, matrix = _build_root()
    transform = copy.deepcopy(matrix.attrs[SELECTED_CALIBRATION_TRANSFORM_ATTR])
    transform["transform_id"] = "tampered"
    matrix.attrs[SELECTED_CALIBRATION_TRANSFORM_ATTR] = transform
    with pytest.raises(SelectedCalibrationError, match="digest is stale"):
        _load(root)

    root, _calibration, _camera, _display, matrix = _build_root()
    matrix.data[0, 2] += 1.0
    with pytest.raises(SelectedCalibrationError, match="matrix digest"):
        _load(root)


def test_manifest_cannot_rebind_source_artifact_away_from_h5_evidence() -> None:
    root, calibration, _camera, _display, _matrix = _build_root()
    payload = copy.deepcopy(calibration.attrs[SELECTED_CALIBRATION_MANIFEST_ATTR])
    payload["source_artifact"]["homography_artifact_path"] = (
        "/rig/calibration/other.yml"
    )
    calibration.attrs[SELECTED_CALIBRATION_MANIFEST_ATTR] = payload
    digest_name = (
        f"{SELECTED_CALIBRATION_MANIFEST_ATTR}"
        f"{SELECTED_CALIBRATION_MANIFEST_DIGEST_SUFFIX}"
    )
    calibration.attrs[digest_name] = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(SelectedCalibrationError, match="homography evidence"):
        _load(root)


def test_loader_rejects_wrong_direction_extent_and_active_camera_pointers() -> None:
    root, calibration, camera, _display, _matrix = _build_root()
    with pytest.raises(SelectedCalibrationError, match="direction mismatch"):
        _load(
            root,
            expected_from_space_id=CANONICAL_HOMOGRAPHY_TO_SPACE_ID,
            expected_to_space_id=CANONICAL_HOMOGRAPHY_FROM_SPACE_ID,
        )
    wrong_target = TransformReferenceExtent(
        width=344,
        height=344,
        units="px",
        authority=TARGET_EXTENT.authority,
    )
    with pytest.raises(SelectedCalibrationError, match="extents"):
        _load(root, expected_target_reference_extent=wrong_target)

    calibration.attrs[ACTIVE_CAMERA_ID_ATTR] = "other"
    with pytest.raises(SelectedCalibrationError, match="active_camera_id.*mismatch"):
        _load(root)
    calibration.attrs[ACTIVE_CAMERA_ID_ATTR] = CAMERA_ID
    calibration.attrs[ACTIVE_CAMERA_CALIBRATION_REF_ATTR] = "/wrong"
    with pytest.raises(
        SelectedCalibrationError,
        match="active_camera_calibration_ref.*mismatch",
    ):
        _load(root)
    calibration.attrs[ACTIVE_CAMERA_CALIBRATION_REF_ATTR] = (
        PATHS.camera_calibration_path
    )
    calibration.attrs[ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR] = "f" * 64
    with pytest.raises(SelectedCalibrationError, match="transform digest pointer"):
        _load(root)
    manifest = calibration.attrs[SELECTED_CALIBRATION_MANIFEST_ATTR]
    assert isinstance(manifest, dict)
    calibration.attrs[ACTIVE_CAMERA_TRANSFORM_SHA256_ATTR] = manifest[
        "transform_sha256"
    ]
    camera.attrs[CAMERA_ID_ATTR] = "other"
    with pytest.raises(SelectedCalibrationError, match="camera_id.*mismatch"):
        _load(root)


def test_missing_camera_scalars_are_not_reconstructed_from_other_groups() -> None:
    root, _calibration, camera, _display, _matrix = _build_root(
        include_scalars=False
    )
    analysis = root["analysis"]
    assert isinstance(analysis, FakeGroup)
    analysis["calibration"] = FakeGroup(
        path="analysis/calibration",
        attrs={
            "pixels_per_mm_camera": 999.0,
            "pixel_to_mm": 1.0 / 999.0,
            "pixels_per_mm_projector": 888.0,
            "z_eff_mm": 777.0,
        },
    )

    snapshot = _load(root)

    assert snapshot.pixels_per_mm_camera is None
    assert snapshot.pixel_to_mm is None
    assert snapshot.pixels_per_mm_projector is None
    assert snapshot.z_eff_mm is None
    assert "pixels_per_mm_camera" not in camera.attrs
