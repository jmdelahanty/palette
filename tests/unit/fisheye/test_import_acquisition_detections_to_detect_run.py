from __future__ import annotations

import csv
from dataclasses import replace
import json

import numpy as np
import pytest
import zarr

from fisheye.shared import detection_producer_lifecycle as lifecycle_mod
from fisheye.shared.detection_producer_lifecycle import (
    ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR,
    DETECTION_ARTIFACT_FAMILY_CONTRACT,
    EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR,
    STRICT_ARTIFACT_INTEGRITY_CONTRACT,
    UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR,
    UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
    validate_artifact_payload_inventory_seal,
    validate_empty_artifact_observation_proof,
    validate_unbound_artifact_numeric_semantics,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    is_run_selector_eligible,
)
from fisheye.utils import import_acquisition_detections_to_detect_run as import_mod
from fisheye.utils.import_acquisition_detections_to_detect_run import (
    import_acquisition_detections_to_detect_run,
)


def _write_crop_meta(path) -> None:
    path.write_text(
        "\n".join(
            [
                "recording_frame_id,local_frame_id,camera_frame_id,timestamp,timestamp_sys,has_detection,blank_frame,detection_confidence,crop_x,crop_y,crop_w,crop_h,detection_x,detection_y,detection_w,detection_h",
                "1,10,100,0,0,1,0,0.50,0,0,100,100,10,20,30,40",
                "2,11,101,0,0,1,1,0.60,0,0,100,100,20,30,30,40",
                "3,12,102,0,0,0,0,0.00,0,0,100,100,,,,",
                "4,13,103,0,0,1,0,0.90,5,6,100,100,30,40,20,10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(recording_dir, *, include_stream_contract: bool = True) -> None:
    video_streams = {
        "schema_id": "orange_runtime_video_streams_v1",
        "frame_clock": "recording_frame_id",
        "streams": {
            "full": {
                "camera_id": "2010093",
                "frame_clock": "recording_frame_id",
                "coordinate_space": "full_frame_pixels",
                "width": 1000,
                "height": 500,
                "frame_count": 4,
            },
            "crop": {
                "camera_id": "2010093",
                "frame_clock": "recording_frame_id",
                "frame_count": 4,
                "metadata": (
                    "derived/external_crop_recorder/"
                    "Cam2010093_session_crop_meta.csv"
                ),
                "video": (
                    "derived/external_crop_recorder/"
                    "Cam2010093_session_crop_external.mp4"
                ),
                "selection_policy": "largest_detection_by_confidence",
                "blank_frame_policy": "encode_black_frame_when_no_detection",
                "video_pixel_coordinate_space": "crop_frame_pixels",
                "source_geometry_coordinate_space": "full_frame_pixels",
                "geometry_columns": [
                    "crop_x",
                    "crop_y",
                    "crop_w",
                    "crop_h",
                    "detection_x",
                    "detection_y",
                    "detection_w",
                    "detection_h",
                ],
            },
        },
    }
    if not include_stream_contract:
        video_streams.pop("schema_id")
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "recording_id": "recording-test",
                "video_streams": video_streams,
            }
        ),
        encoding="utf-8",
    )


def _rewrite_crop_meta(
    path,
    *,
    frame_ids: list[object] | None = None,
    crop_override: tuple[int, tuple[object, object, object, object]] | None = None,
    field_overrides: tuple[int, dict[str, object]] | None = None,
    drop_last: bool = False,
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if frame_ids is not None:
        for row, value in zip(rows, frame_ids, strict=True):
            row["recording_frame_id"] = str(value)
    if crop_override is not None:
        row_index, values = crop_override
        for name, value in zip(
            ("crop_x", "crop_y", "crop_w", "crop_h"),
            values,
            strict=True,
        ):
            rows[row_index][name] = str(value)
    if field_overrides is not None:
        row_index, values = field_overrides
        for name, value in values.items():
            rows[row_index][name] = str(value)
    if drop_last:
        rows.pop()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _drop_crop_meta_column(path, column: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name for name in (reader.fieldnames or ()) if name != column]
        rows = list(reader)
    for row in rows:
        row.pop(column, None)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_manifest(recording_dir, *path, value) -> None:
    manifest_path = recording_dir / "recording_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = payload
    for name in path[:-1]:
        target = target[name]
    if value is _DELETE:
        target.pop(path[-1], None)
    else:
        target[path[-1]] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


_DELETE = object()


def _prepare_import_fixture(tmp_path):
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    crop_meta = crop_dir / "Cam2010093_session_crop_meta.csv"
    _write_crop_meta(crop_meta)
    _write_manifest(recording_dir)
    zarr.open_group(str(zarr_path), mode="w")
    return recording_dir, zarr_path, crop_meta


def _assert_no_detection_output(zarr_path) -> None:
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_runs" not in reopened
    assert "detection_artifact_runs" not in reopened


def test_import_acquisition_detections_writes_nonselector_artifact(tmp_path) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    crop_meta = crop_dir / "Cam2010093_session_crop_meta.csv"
    _write_crop_meta(crop_meta)
    _write_manifest(recording_dir)
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["recording_id"] = "recording-test"
    root.attrs["camera_id"] = "2010093"

    result = import_acquisition_detections_to_detect_run(
        zarr_path,
        run_name="detect_acquisition_test",
        apply=True,
        artifact_only=True,
    )

    assert result.applied is True
    assert result.total_detections == 2
    assert result.blank_frame_count == 1
    assert result.no_detection_frame_count == 1
    assert result.output_parent == "detection_artifact_runs"
    assert result.stage_selector_eligible is False
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_runs" not in root
    parent = root["detection_artifact_runs"]
    run = parent["detect_acquisition_test"]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert selector not in parent.attrs
    assert (
        parent.attrs["artifact_family_contract"]
        == DETECTION_ARTIFACT_FAMILY_CONTRACT
    )
    assert parent.attrs["stage_selector_eligible"] is False
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert run.attrs["stage_selector_eligible"] is False
    assert (
        run.attrs["coordinate_contract"]
        == UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT
    )
    assert run.attrs["detection_method"] == "acquisition_runtime_import"
    assert run.attrs["detection_source"] == "external_crop_recorder_crop_meta"
    assert len(run.attrs["source_artifact_sha256"]) == 64
    assert len(run.attrs["source_manifest_sha256"]) == 64
    source_frame_sha256 = run.attrs["external_source_frame_evidence_sha256"]
    assert len(source_frame_sha256) == 64
    assert (
        import_mod._canonical_mapping_sha256(
            run.attrs["external_source_frame_evidence"]
        )
        == source_frame_sha256
    )
    source_mapping_sha256 = run.attrs["external_source_frame_evidence"][
        "recording_frame_ids_sha256"
    ]
    assert len(source_mapping_sha256) == 64
    assert run.attrs["external_source_frame_evidence"][
        "manifest_full_stream_ref"
    ] == (
        f"{recording_dir / 'recording_manifest.json'}"
        "#/video_streams/streams/full"
    )
    assert run.attrs["external_source_frame_evidence"]["reference_width"] == 1000
    assert run.attrs["external_source_frame_evidence"]["reference_height"] == 500
    assert "instance_key" not in run
    assert not any(name.startswith("instance_key_") for name in run.attrs)
    assert (
        run.attrs["artifact_integrity_contract"]
        == STRICT_ARTIFACT_INTEGRITY_CONTRACT
    )
    assert run["artifact_row_id"].dtype == np.dtype("uint64")
    assert run["artifact_row_id"][:].tolist() == [0, 1]
    assert run["frame_indices"][:].tolist() == [0, 3]
    assert run["scores"][:].tolist() == [np.float32(0.5), np.float32(0.9)]
    assert run["class_ids"][:].tolist() == [0, 0]
    assert run["frame_counts"][:].tolist() == [1, 0, 0, 1]
    assert run["n_detections"][:].tolist() == [1, 0, 0, 1]
    np.testing.assert_allclose(
        run["bbox_norm_coords"][:],
        np.asarray(
            [
                [0.025, 0.08, 0.03, 0.08],
                [0.04, 0.09, 0.02, 0.02],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        run["bbox_img_xyxy"][:],
        np.asarray([[10.0, 20.0, 40.0, 60.0], [30.0, 40.0, 50.0, 50.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        run["source_crop_xywh"][:],
        np.asarray([[0.0, 0.0, 100.0, 100.0], [5.0, 6.0, 100.0, 100.0]], dtype=np.float64),
    )
    assert run["source_crop_xywh"].dtype == np.dtype("float64")
    assert run["source_recording_frame_ids"][:].tolist() == [1, 4]
    assert run["source_crop_meta_row_indices"][:].tolist() == [0, 3]
    expected_profiles = {
        "artifact_row_id": "import.artifact_row_id.v1",
        "frame_indices": "import.frame_indices.v1",
        "bbox_norm_coords": "import.bbox_norm_cxcywh.v1",
        "bbox_img_xyxy": "import.bbox_img_xyxy.v1",
        "centers_img_xy": "import.centers_img_xy.v1",
        "scores": "import.scores.v1",
        "class_ids": "import.class_ids.v1",
        "frame_counts": "import.frame_counts.v1",
        "n_detections": "import.n_detections.v1",
        "source_crop_xywh": "import.source_crop_xywh.v1",
        "source_crop_meta_row_indices": (
            "import.source_crop_meta_row_indices.v1"
        ),
        "source_recording_frame_ids": "import.source_recording_frame_ids.v1",
    }
    for array_name in run.keys():
        semantics = validate_unbound_artifact_numeric_semantics(run[array_name])
        assert semantics["canonical_binding_status"] == "unbound"
        assert semantics["semantic_profile_id"] == expected_profiles[array_name]
        assert semantics["reference"]["node_path"] == (
            f"{recording_dir / 'recording_manifest.json'}"
            "#/video_streams/streams/full"
        )
        assert semantics["reference"]["width"] == 1000
        assert semantics["reference"]["height"] == 500
        assert semantics["source_sha256"] == source_frame_sha256
    crop_semantics = validate_unbound_artifact_numeric_semantics(
        run["source_crop_xywh"]
    )
    assert crop_semantics["geometry_type"] == "crop_xywh"
    assert crop_semantics["component_units"] == ["px"] * 4
    assert crop_semantics["pixel_convention"] == (
        "source_edge_convention_undeclared"
    )
    assert crop_semantics["source_camera_overlay_suitability"] == "unsupported"
    assert crop_semantics["canonical_promotion_suitability"] == "unsupported"
    assert (
        validate_unbound_artifact_numeric_semantics(run["bbox_img_xyxy"])[
            "semantic_profile_id"
        ]
        != validate_unbound_artifact_numeric_semantics(run["centers_img_xy"])[
            "semantic_profile_id"
        ]
    )
    assert (
        validate_unbound_artifact_numeric_semantics(run["frame_counts"])[
            "axis_0_domain"
        ]
        == "dense_frame_rows"
    )
    payload_seal = validate_artifact_payload_inventory_seal(run)
    assert payload_seal["row_count"] == 2
    assert set(payload_seal["arrays"]) == set(run.keys())
    assert payload_seal["unbound_numeric_manifest_id"] == (
        "acquisition_detection_import.v1"
    )
    assert {
        name: evidence["semantic_profile_id"]
        for name, evidence in payload_seal["arrays"].items()
    } == expected_profiles
    mapping_by_array = {
        name: run[name].attrs[UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR][
            "temporal_evidence"
        ]["source_mapping_sha256"]
        for name in run.keys()
    }
    assert {
        name for name, mapping in mapping_by_array.items() if mapping is not None
    } == {"frame_indices"}
    assert mapping_by_array["frame_indices"] == source_mapping_sha256
    assert payload_seal["source_mapping_sha256"] == source_mapping_sha256
    assert ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR in run.attrs


def test_import_preserves_fractional_float64_geometry_exactly(tmp_path) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    detection_x = 10.1234567890123
    detection_y = 20.2345678901234
    detection_w = 30.3456789012345
    detection_h = 40.4567890123456
    crop_x = 0.1234567890123
    crop_y = 1.2345678901234
    _rewrite_crop_meta(
        crop_meta,
        crop_override=(0, (crop_x, crop_y, 100.5, 101.5)),
    )
    _rewrite_crop_meta(
        crop_meta,
        field_overrides=(
            0,
            {
                "detection_x": detection_x,
                "detection_y": detection_y,
                "detection_w": detection_w,
                "detection_h": detection_h,
            },
        ),
    )

    import_acquisition_detections_to_detect_run(
        zarr_path,
        run_name="fractional_float64",
        apply=True,
        artifact_only=True,
    )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["detection_artifact_runs"]["fractional_float64"]
    np.testing.assert_array_equal(
        run["bbox_img_xyxy"][0],
        np.asarray(
            [
                detection_x,
                detection_y,
                detection_x + detection_w,
                detection_y + detection_h,
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        run["source_crop_xywh"][0],
        np.asarray([crop_x, crop_y, 100.5, 101.5], dtype=np.float64),
    )


def test_import_acquisition_detections_dry_run_does_not_write(tmp_path) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    _write_crop_meta(crop_dir / "Cam2010093_session_crop_meta.csv")
    _write_manifest(recording_dir)
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["source_video_width"] = 17
    root.attrs["source_video_height"] = 19

    result = import_acquisition_detections_to_detect_run(
        zarr_path,
        run_name="detect_acquisition_test",
        apply=False,
    )

    assert result.applied is False
    assert result.run_path == (
        "detection_artifact_runs/detect_acquisition_test"
    )
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_runs" not in root
    assert "detection_artifact_runs" not in root


def test_import_normalizes_run_name_once_for_path_and_child(tmp_path) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)

    result = import_acquisition_detections_to_detect_run(
        zarr_path,
        run_name="  normalized-run  ",
        apply=True,
        artifact_only=True,
    )

    assert result.run_name == "normalized-run"
    assert result.run_path == "detection_artifact_runs/normalized-run"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "normalized-run" in root["detection_artifact_runs"]


def test_import_persists_verified_genuine_zero_observation_proof(tmp_path) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    for row_index in range(4):
        _rewrite_crop_meta(
            crop_meta,
            field_overrides=(
                row_index,
                {"has_detection": 0, "blank_frame": 0},
            ),
        )

    result = import_acquisition_detections_to_detect_run(
        zarr_path,
        run_name="genuine-zero",
        apply=True,
        artifact_only=True,
    )

    assert result.total_detections == 0
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["detection_artifact_runs"]["genuine-zero"]
    proof = run.attrs[EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR]
    assert proof["status"] == "verified_no_observations"
    assert proof["source_frame_count"] == 4
    assert proof["full_domain_evidence"]["eligible_observation_count"] == 0
    assert len(proof["full_domain_evidence"]["raw_selection_sha256"]) == 64
    assert run["artifact_row_id"].dtype == np.dtype("uint64")
    assert run["artifact_row_id"].shape == (0,)
    assert run["frame_indices"].shape == (0,)
    assert run["frame_counts"][:].tolist() == [0, 0, 0, 0]
    assert run["n_detections"][:].tolist() == [0, 0, 0, 0]
    assert set(proof["array_inventory"]) == set(run.keys())
    assert "artifact_row_id" in proof["row_arrays"]
    validate_empty_artifact_observation_proof(run)
    assert validate_artifact_payload_inventory_seal(run)["row_count"] == 0


def test_import_apply_requires_explicit_artifact_mode_before_io(tmp_path) -> None:
    zarr_path = tmp_path / "must_not_exist.zarr"

    with pytest.raises(ValueError, match="artifact_only=True"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="must_not_exist",
            apply=True,
        )

    assert not zarr_path.exists()


def test_import_preflight_rejects_loose_root_dimensions_before_creating_runs(
    tmp_path,
) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    _write_crop_meta(crop_dir / "Cam2010093_session_crop_meta.csv")
    _write_manifest(recording_dir, include_stream_contract=False)
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "source_video_width": 1000,
            "source_video_height": 500,
            "total_frames": 4,
        }
    )

    with pytest.raises(ValueError, match="orange_runtime_video_streams_v1"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="must_not_exist",
            apply=True,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_runs" not in reopened
    assert "detection_artifact_runs" not in reopened


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("non_integral", "exact positive base-10 integers"),
        ("whitespace", "exact positive base-10 integers"),
        ("leading_zero", "exact canonical positive base-10 token"),
        ("permuted", "permuted"),
        ("duplicate", "duplicated"),
        ("incomplete", "incomplete"),
    ],
)
def test_import_preflight_rejects_raw_frame_identity_defects(
    case, match, monkeypatch, tmp_path
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    if case == "non_integral":
        _rewrite_crop_meta(crop_meta, frame_ids=["1.5", 2, 3, 4])
    elif case == "whitespace":
        _rewrite_crop_meta(crop_meta, frame_ids=[" 1 ", 2, 3, 4])
    elif case == "leading_zero":
        _rewrite_crop_meta(crop_meta, frame_ids=["01", 2, 3, 4])
    elif case == "permuted":
        _rewrite_crop_meta(crop_meta, frame_ids=[2, 1, 3, 4])
    elif case == "duplicate":
        _rewrite_crop_meta(crop_meta, frame_ids=[1, 1, 3, 4])
    else:
        _rewrite_crop_meta(crop_meta, drop_last=True)

    def forbidden_normalizing_loader(_path):
        raise AssertionError("Raw frame IDs must fail before shared normalization")

    monkeypatch.setattr(
        import_mod,
        "load_crop_meta_realtime_detection_rows",
        forbidden_normalizing_loader,
    )

    with pytest.raises(ValueError, match=match):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name=f"invalid_{case}",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize("field_name", ["has_detection", "blank_frame"])
@pytest.mark.parametrize(
    "bad_value",
    ["", "1.0", "2", "-1", "garbage", " 1", "1 ", " 0 "],
)
def test_import_preflight_rejects_nonexact_raw_selection_flags_before_filtering(
    field_name, bad_value, monkeypatch, tmp_path
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    _rewrite_crop_meta(
        crop_meta,
        field_overrides=(0, {field_name: bad_value}),
    )

    def forbidden_normalizing_loader(_path):
        raise AssertionError("Raw selection flags must fail before normalization")

    monkeypatch.setattr(
        import_mod,
        "load_crop_meta_realtime_detection_rows",
        forbidden_normalizing_loader,
    )

    with pytest.raises(ValueError, match=f"{field_name} must be exactly integer 0 or 1"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name=f"invalid_{field_name}",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize("field_name", ["has_detection", "blank_frame"])
def test_import_preflight_rejects_missing_raw_selection_column_before_filtering(
    field_name, monkeypatch, tmp_path
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    _drop_crop_meta_column(crop_meta, field_name)

    def forbidden_normalizing_loader(_path):
        raise AssertionError("Missing raw flags must fail before normalization")

    monkeypatch.setattr(
        import_mod,
        "load_crop_meta_realtime_detection_rows",
        forbidden_normalizing_loader,
    )

    with pytest.raises(ValueError, match="missing raw identity/selection columns"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name=f"missing_{field_name}",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize(
    ("manifest_path", "bad_value"),
    [
        (("video_streams", "streams", "full", "width"), 1000.0),
        (("video_streams", "streams", "full", "frame_count"), "4"),
        (("video_streams", "streams", "crop", "frame_count"), 4.0),
    ],
)
def test_import_preflight_requires_exact_manifest_integers(
    manifest_path, bad_value, tmp_path
) -> None:
    recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    _update_manifest(recording_dir, *manifest_path, value=bad_value)

    with pytest.raises(ValueError, match="exact JSON integer"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="invalid_manifest_int",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


def test_import_preflight_requires_equal_full_and_crop_frame_counts(
    tmp_path,
) -> None:
    recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    _update_manifest(
        recording_dir,
        "video_streams",
        "streams",
        "crop",
        "frame_count",
        value=3,
    )

    with pytest.raises(ValueError, match="must exactly equal"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="mismatched_stream_counts",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


def test_import_preflight_requires_stable_manifest_recording_identity(
    tmp_path,
) -> None:
    recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    _update_manifest(recording_dir, "recording_id", value=_DELETE)

    with pytest.raises(ValueError, match="recording_id.*exact unpadded"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="missing_recording_identity",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize(
    ("manifest_path", "bad_value", "match"),
    [
        (("recording_id",), " recording-test", "recording_id.*exact unpadded"),
        (
            ("video_streams", "streams", "full", "camera_id"),
            "2010093 ",
            "camera_id.*exact unpadded",
        ),
    ],
)
def test_import_preflight_rejects_padded_manifest_identity(
    manifest_path, bad_value, match, tmp_path
) -> None:
    recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    _update_manifest(recording_dir, *manifest_path, value=bad_value)

    with pytest.raises(ValueError, match=match):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="padded_manifest_identity",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize(
    ("attr_name", "attr_value", "match"),
    [
        ("recording_id", 123, "recording_id disagrees"),
        ("recording_id", " recording-test", "recording_id disagrees"),
        ("camera_id", 2010093, "camera_id disagrees"),
        ("camera_id", "2010093 ", "camera_id disagrees"),
    ],
)
def test_import_preflight_rejects_coerced_root_identity(
    attr_name, attr_value, match, tmp_path
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root.attrs[attr_name] = attr_value

    with pytest.raises(ValueError, match=match):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="coerced_root_identity",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize(
    ("crop_xywh", "match"),
    [
        (("nan", 0, 100, 100), "finite"),
        ((-1, 0, 100, 100), "nonnegative origin"),
        ((0, -1, 100, 100), "nonnegative origin"),
        ((0, 0, 0, 100), "positive extent"),
        ((0, 0, 100, 0), "positive extent"),
        ((950, 0, 100, 100), "fit inside"),
        ((0, 450, 100, 100), "fit inside"),
    ],
)
def test_import_preflight_rejects_invalid_crop_geometry(
    crop_xywh, match, tmp_path
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    _rewrite_crop_meta(crop_meta, crop_override=(2, crop_xywh))

    with pytest.raises(ValueError, match=match):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="invalid_crop_geometry",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize("row_index", [0, 2])
def test_import_preflight_rejects_zero_crop_sentinel_without_blank_no_detection(
    row_index, tmp_path
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    _rewrite_crop_meta(crop_meta, crop_override=(row_index, (0, 0, 0, 0)))

    with pytest.raises(ValueError, match="permitted only"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="invalid_crop_sentinel",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


def test_import_allows_blank_no_detection_crop_sentinel_without_persisting_it(
    tmp_path,
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    _rewrite_crop_meta(
        crop_meta,
        crop_override=(2, (0, 0, 0, 0)),
        field_overrides=(2, {"has_detection": 0, "blank_frame": 1}),
    )

    result = import_acquisition_detections_to_detect_run(
        zarr_path,
        run_name="valid_blank_sentinel",
        apply=True,
        artifact_only=True,
    )

    assert result.total_detections == 2
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = reopened["detection_artifact_runs"]["valid_blank_sentinel"]
    persisted_crop = run["source_crop_xywh"][:]
    assert persisted_crop.shape == (2, 4)
    assert not np.any(np.all(persisted_crop == 0.0, axis=1))
    assert run["source_crop_meta_row_indices"][:].tolist() == [0, 3]
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE


@pytest.mark.parametrize(
    "case",
    ["rank", "count", "range", "identity"],
)
def test_import_preflight_rejects_invalid_detection_row_indices(
    case, monkeypatch, tmp_path
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    original_loader = import_mod.load_crop_meta_realtime_detection_rows
    invalid_row_indices = {
        "rank": np.asarray([[0, 3]], dtype=np.int64),
        "count": np.asarray([0], dtype=np.int64),
        "range": np.asarray([0, 4], dtype=np.int64),
        "identity": np.asarray([0, 2], dtype=np.int64),
    }[case]

    def load_with_invalid_row_indices(path):
        detections, crop_rows = original_loader(path)
        return replace(detections, row_indices=invalid_row_indices), crop_rows

    monkeypatch.setattr(
        import_mod,
        "load_crop_meta_realtime_detection_rows",
        load_with_invalid_row_indices,
    )

    with pytest.raises(ValueError, match="row_indices"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name=f"invalid_row_indices_{case}",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize("field_name", ["has_detection", "blank_frame"])
def test_import_preflight_binds_normalized_flags_to_exact_raw_flags(
    field_name, monkeypatch, tmp_path
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    original_loader = import_mod.load_crop_meta_realtime_detection_rows

    def load_with_changed_flag(path):
        detections, crop_rows = original_loader(path)
        values = np.asarray(getattr(crop_rows, field_name), dtype=bool).copy()
        values[0] = ~values[0]
        return detections, replace(crop_rows, **{field_name: values})

    monkeypatch.setattr(
        import_mod,
        "load_crop_meta_realtime_detection_rows",
        load_with_changed_flag,
    )

    with pytest.raises(ValueError, match="changed exact raw"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name=f"changed_{field_name}",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("crop_frame_fractional", "without coercion"),
        ("crop_row_fractional", "int64 source row identity"),
        ("crop_float32", "exact float64"),
        ("bbox_float32", "exact float64"),
        ("centers_float32", "exact float64"),
        ("confidence_float32", "exact float64"),
        ("crop_value_changed", "changed exact raw float64 crop placement"),
        ("bbox_value_changed", "changed exact raw float64 detection boxes"),
    ],
)
def test_import_preflight_rejects_dtype_fractional_or_value_corruption(
    case, match, monkeypatch, tmp_path
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    original_loader = import_mod.load_crop_meta_realtime_detection_rows

    def load_with_corruption(path):
        detections, crop_rows = original_loader(path)
        if case == "crop_frame_fractional":
            crop_rows = replace(
                crop_rows,
                frame_indices=crop_rows.frame_indices.astype(np.float64) + 0.25,
            )
        elif case == "crop_row_fractional":
            crop_rows = replace(
                crop_rows,
                row_indices=crop_rows.row_indices.astype(np.float64) + 0.25,
            )
        elif case == "crop_float32":
            crop_rows = replace(
                crop_rows,
                crop_xywh=crop_rows.crop_xywh.astype(np.float32),
            )
        elif case == "bbox_float32":
            detections = replace(
                detections,
                bbox_img_xyxy=detections.bbox_img_xyxy.astype(np.float32),
            )
        elif case == "centers_float32":
            detections = replace(
                detections,
                centers_xy=detections.centers_xy.astype(np.float32),
            )
        elif case == "confidence_float32":
            detections = replace(
                detections,
                confidence=detections.confidence.astype(np.float32),
            )
        elif case == "crop_value_changed":
            changed = crop_rows.crop_xywh.copy()
            changed[0, 0] += 0.125
            crop_rows = replace(crop_rows, crop_xywh=changed)
        else:
            changed = detections.bbox_img_xyxy.copy()
            changed[0, 0] += 0.125
            detections = replace(detections, bbox_img_xyxy=changed)
        return detections, crop_rows

    monkeypatch.setattr(
        import_mod,
        "load_crop_meta_realtime_detection_rows",
        load_with_corruption,
    )

    with pytest.raises(ValueError, match=match):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name=f"corrupt_{case}",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


def test_import_preflight_rejects_omitted_raw_eligible_detection_row(
    tmp_path,
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    _rewrite_crop_meta(
        crop_meta,
        field_overrides=(0, {"detection_x": ""}),
    )

    with pytest.raises(ValueError, match="finite detection geometry.*false-empty"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="omitted_eligible_row",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


def test_import_preflight_rejects_extra_ineligible_detection_row(
    monkeypatch, tmp_path
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    original_loader = import_mod.load_crop_meta_realtime_detection_rows

    def load_with_extra_ineligible_row(path):
        detections, crop_rows = original_loader(path)
        return (
            replace(
                detections,
                frame_indices=np.asarray([0, 2, 3], dtype=np.int64),
                bbox_img_xyxy=np.insert(
                    detections.bbox_img_xyxy,
                    1,
                    np.asarray([100.0, 100.0, 120.0, 120.0]),
                    axis=0,
                ),
                centers_xy=np.insert(
                    detections.centers_xy,
                    1,
                    np.asarray([110.0, 110.0]),
                    axis=0,
                ),
                confidence=np.insert(detections.confidence, 1, 0.75),
                row_indices=np.asarray([0, 2, 3], dtype=np.int64),
            ),
            crop_rows,
        )

    monkeypatch.setattr(
        import_mod,
        "load_crop_meta_realtime_detection_rows",
        load_with_extra_ineligible_row,
    )

    with pytest.raises(ValueError, match="no ineligible row may be added"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="extra_ineligible_row",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


@pytest.mark.parametrize("bad_value", ["", "nan", "inf", "-inf"])
def test_import_preflight_rejects_false_empty_when_all_eligible_geometry_is_invalid(
    bad_value, tmp_path,
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    _rewrite_crop_meta(
        crop_meta,
        field_overrides=(0, {"detection_x": bad_value}),
    )
    _rewrite_crop_meta(
        crop_meta,
        field_overrides=(3, {"detection_x": bad_value}),
    )

    with pytest.raises(ValueError, match="finite detection geometry.*false-empty"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="all_eligible_geometry_invalid",
            apply=True,
            artifact_only=True,
        )

    _assert_no_detection_output(zarr_path)


def test_import_preflight_rejects_source_that_changes_during_read(
    monkeypatch, tmp_path
) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    crop_meta = crop_dir / "Cam2010093_session_crop_meta.csv"
    _write_crop_meta(crop_meta)
    _write_manifest(recording_dir)
    zarr.open_group(str(zarr_path), mode="w")
    original_load = import_mod.load_crop_meta_realtime_detection_rows

    def load_then_mutate(path):
        result = original_load(path)
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        import_mod,
        "load_crop_meta_realtime_detection_rows",
        load_then_mutate,
    )

    with pytest.raises(ValueError, match="changed while"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="unstable_source",
            apply=True,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_runs" not in reopened
    assert "detection_artifact_runs" not in reopened


@pytest.mark.parametrize(
    "tamper_case",
    [
        "missing_semantics",
        "tampered_semantics",
        "wrong_reference_extent",
        "artifact_row_id",
        "array_payload",
        "row_cardinality",
        "frame_counts",
    ],
)
def test_import_rejects_post_seal_artifact_drift(
    tamper_case,
    monkeypatch,
    tmp_path,
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    original_write = import_mod.write_stage_provenance

    def write_then_tamper(run, provenance):
        original_write(run, provenance)
        if tamper_case == "missing_semantics":
            del run["bbox_img_xyxy"].attrs[
                UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
            ]
        elif tamper_case == "tampered_semantics":
            attr = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
            record = json.loads(json.dumps(run["bbox_img_xyxy"].attrs[attr]))
            record["numeric_space_id"] = "forged_coordinate_space"
            run["bbox_img_xyxy"].attrs[attr] = record
        elif tamper_case == "wrong_reference_extent":
            attr = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
            digest_attr = f"{attr}_sha256"
            record = json.loads(json.dumps(run["bbox_img_xyxy"].attrs[attr]))
            record["reference"]["height"] += 1
            run["bbox_img_xyxy"].attrs[attr] = record
            run["bbox_img_xyxy"].attrs[digest_attr] = (
                lifecycle_mod._canonical_sha256(record)
            )
        elif tamper_case == "artifact_row_id":
            run["artifact_row_id"][0] = np.uint64(9)
        elif tamper_case == "array_payload":
            run["scores"][0] = np.float32(0.125)
        elif tamper_case == "row_cardinality":
            old = run["scores"]
            attrs = dict(old.attrs)
            values = np.append(old[:], np.float32(0.5))
            del run["scores"]
            replacement = run.create_array(
                "scores",
                data=values,
                chunks=(values.shape[0],),
            )
            replacement.attrs.update(attrs)
        else:
            run["frame_counts"][0] = np.int32(2)

    monkeypatch.setattr(import_mod, "write_stage_provenance", write_then_tamper)

    with pytest.raises(ValueError):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name=f"tampered_{tamper_case}",
            apply=True,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    failed = reopened["detection_artifact_runs"][f"tampered_{tamper_case}"]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_import_rejects_coherent_all_array_reference_rewrite(
    monkeypatch,
    tmp_path,
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    original_write = import_mod.write_stage_provenance

    def write_then_rewrite_all_reference_evidence(run, provenance):
        original_write(run, provenance)
        evidence_attr = "external_source_frame_evidence"
        evidence = json.loads(json.dumps(run.attrs[evidence_attr]))
        binding = evidence[lifecycle_mod.UNBOUND_ARTIFACT_RUN_BINDING_KEY]
        binding["reference"]["width"] += 1
        source_digest = lifecycle_mod._canonical_sha256(evidence)
        run.attrs[evidence_attr] = evidence
        run.attrs[f"{evidence_attr}_sha256"] = source_digest

        semantics_attr = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
        semantics_digest_attr = f"{semantics_attr}_sha256"
        seal = json.loads(
            json.dumps(run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR])
        )
        seal["run_evidence"]["sha256"] = source_digest
        seal["run_evidence"]["binding"] = binding
        for array_name in run.keys():
            semantics = json.loads(
                json.dumps(run[array_name].attrs[semantics_attr])
            )
            semantics["reference"]["width"] += 1
            semantics["source_sha256"] = source_digest
            semantics_digest = lifecycle_mod._canonical_sha256(semantics)
            run[array_name].attrs[semantics_attr] = semantics
            run[array_name].attrs[semantics_digest_attr] = semantics_digest
            seal["arrays"][array_name][
                "numeric_semantics_sha256"
            ] = semantics_digest
        run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR] = seal
        run.attrs[f"{ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR}_sha256"] = (
            lifecycle_mod._canonical_sha256(seal)
        )

    monkeypatch.setattr(
        import_mod,
        "write_stage_provenance",
        write_then_rewrite_all_reference_evidence,
    )

    with pytest.raises(ValueError, match="Acquisition-import source/reference"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="coherent_reference_rewrite",
            apply=True,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    failed = reopened["detection_artifact_runs"][
        "coherent_reference_rewrite"
    ]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_import_rejects_zero_row_identity_tampering(
    monkeypatch,
    tmp_path,
) -> None:
    _recording_dir, zarr_path, crop_meta = _prepare_import_fixture(tmp_path)
    for row_index in range(4):
        _rewrite_crop_meta(
            crop_meta,
            field_overrides=(
                row_index,
                {"has_detection": 0, "blank_frame": 0},
            ),
        )
    original_write = import_mod.write_stage_provenance

    def write_then_replace_zero_row_id(run, provenance):
        original_write(run, provenance)
        del run["artifact_row_id"]
        run.create_array(
            "artifact_row_id",
            data=np.empty((0,), dtype=np.int64),
            chunks=(1,),
        )

    monkeypatch.setattr(
        import_mod,
        "write_stage_provenance",
        write_then_replace_zero_row_id,
    )

    with pytest.raises(ValueError):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="tampered_zero_row_id",
            apply=True,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    failed = reopened["detection_artifact_runs"]["tampered_zero_row_id"]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_import_base_exception_during_completed_child_validation_fails_closed(
    exception_type,
    monkeypatch,
    tmp_path,
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    original_validate = lifecycle_mod.validate_artifact_payload_inventory_seal
    validation_calls = 0

    def interrupt_completed_child_validation(run):
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 3:
            raise exception_type("injected completed-child validation interruption")
        return original_validate(run)

    monkeypatch.setattr(
        lifecycle_mod,
        "validate_artifact_payload_inventory_seal",
        interrupt_completed_child_validation,
    )

    with pytest.raises(exception_type, match="completed-child validation"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="interrupted_final_validation",
            apply=True,
            artifact_only=True,
        )

    assert validation_calls == 3
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    failed = reopened["detection_artifact_runs"][
        "interrupted_final_validation"
    ]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_import_write_failure_marks_failed_and_restores_exact_artifact_selectors(
    monkeypatch, tmp_path
) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    _write_crop_meta(crop_dir / "Cam2010093_session_crop_meta.csv")
    _write_manifest(recording_dir)
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.require_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )
    original_create = import_mod._create_array

    def fail_on_scores(group, name, data):
        if name == "scores":
            raise RuntimeError("injected ordinary producer failure")
        return original_create(group, name, data)

    monkeypatch.setattr(import_mod, "_create_array", fail_on_scores)

    with pytest.raises(RuntimeError, match="injected ordinary producer failure"):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="failed_import",
            apply=True,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert selector not in parent.attrs
    assert (
        parent["failed_import"].attrs[RUN_COMPLETION_STATUS_ATTR]
        == RUN_STATUS_FAILED
    )
    assert "detect_runs" not in reopened


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_import_base_exception_marks_failed_and_restores_exact_artifact_selectors(
    exception_type, monkeypatch, tmp_path
) -> None:
    _recording_dir, zarr_path, _crop_meta = _prepare_import_fixture(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    parent = root.require_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )
    original_create = import_mod._create_array

    def interrupt_on_scores(group, name, data):
        if name == "scores":
            raise exception_type("injected producer interruption")
        return original_create(group, name, data)

    monkeypatch.setattr(import_mod, "_create_array", interrupt_on_scores)

    with pytest.raises(exception_type):
        import_acquisition_detections_to_detect_run(
            zarr_path,
            run_name="interrupted_import",
            apply=True,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert selector not in parent.attrs
    child = parent["interrupted_import"]
    assert child.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED
    assert child.attrs["stage_selector_eligible"] is False
    assert is_run_selector_eligible(child) is False
    assert not any(
        is_run_selector_eligible(parent[name]) for name in parent.group_keys()
    )
    assert "detect_runs" not in reopened
