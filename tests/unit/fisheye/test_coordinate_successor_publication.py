from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

from fisheye.shared.coordinate_record import (
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.zarr.coordinate_successor_authority import (
    COORDINATE_SUCCESSOR_AUTHORITY_ATTR,
    CoordinateSuccessorAuthorityError,
    KEYPOINT_COORDINATE_SUCCESSOR_KIND,
    build_coordinate_successor_authority,
    load_coordinate_successor_authority,
    stamp_coordinate_successor_authority,
    validate_coordinate_successor_authority,
)
from fisheye.shared.zarr.coordinate_successor_files import (
    copy_metadata_and_link_payload,
    metadata_tree_sha256,
)
from fisheye.shared.model_input_transform import ModelInputTransform
from fisheye.shared.zarr import historical_geometry_only_crop_adapter as historical_crop
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


class _Node:
    _archive = object()

    def __init__(self, *, path: str, children=None) -> None:
        self.path = path
        self._coordinate_archive_token = self._archive
        self.attrs: dict[str, object] = {}
        self.children = dict(children or {})

    def __getitem__(self, key: str):
        return self.children[key]


class _Array:
    def __init__(self, *, path: str, values: object) -> None:
        self.path = path
        self.attrs: dict[str, object] = {}
        self._values = np.asarray(values)
        self.shape = self._values.shape
        self.dtype = self._values.dtype

    def __getitem__(self, key: object) -> np.ndarray:
        return self._values[key]


class _Group:
    def __init__(self, *, path: str, children: dict[str, object]) -> None:
        self.path = path
        self.attrs: dict[str, object] = {
            "status": "complete",
            "stage_selector_eligible": False,
            "artifact_class": "geometry_only_analysis",
        }
        self.children = children

    def __getitem__(self, key: str) -> object:
        return self.children[key]

    def keys(self):
        return self.children.keys()


class _Root:
    def __init__(self, nodes: dict[str, object]) -> None:
        self.nodes = nodes

    def __getitem__(self, key: str) -> object:
        return self.nodes[key]


def _historical_crop_fixture(tmp_path: Path):
    n_frames = 5
    n_instances = 4
    width = height = 100
    row_signature = np.arange(n_instances * 32, dtype=np.uint8).reshape(n_instances, 32)
    crop_arrays = {
        name: _Array(path=f"crop_runs/crop/{name}", values=np.zeros(1, dtype=np.uint8))
        for name in CROP_GEOMETRY_SCHEMA_V1.binding_paths
    }
    crop_arrays["instance_key"] = _Array(
        path="crop_runs/crop/instance_key",
        values=np.asarray([10, 11, 12, 13], dtype="<u8"),
    )
    crop_arrays["source_acquisition_frame_index"] = _Array(
        path="crop_runs/crop/source_acquisition_frame_index",
        values=np.asarray([0, 1, 3, 4], dtype="<i8"),
    )
    crop_arrays["source_row_signature"] = _Array(
        path="crop_runs/crop/source_row_signature", values=row_signature
    )
    crop_arrays["source_crop_xywh"] = _Array(
        path="crop_runs/crop/source_crop_xywh",
        values=np.asarray(
            [[0, 0, 384, 384], [1, 2, 384, 384], [3, 4, 384, 384], [5, 6, 384, 384]],
            dtype="<f4",
        ),
    )
    crop_arrays["roi_sizes_full"] = _Array(
        path="crop_runs/crop/roi_sizes_full",
        values=np.full((n_instances, 2), 384, dtype="<i4"),
    )
    crop_group = _Group(
        path="crop_runs/crop",
        children=dict(crop_arrays),
    )
    logical = {
        "digest_algorithm": "sha256_canonical_json_v1",
        "digest": "b" * 64,
        "document": {
            "arrays": {"source_row_signature": {"sha256": "c" * 64}}
        },
    }
    manifest = {
        "schema_id": "palette.crop_geometry.run_manifest",
        "schema_version": 2,
        "payload_digest": "a" * 64,
        "payload": {
            "run_id": "crop",
            "logical_schema": {
                "dimensions": {
                    "n_frames": n_frames,
                    "n_instances": n_instances,
                    "source_width": width,
                    "source_height": height,
                }
            },
            "logical_content": logical,
            "coordinate_contract": {"digest": "d" * 64},
            "source_pixel_authority": {
                "n_frames": n_frames,
                "source_width": width,
                "source_height": height,
                "camera_identity": "cam",
            },
        },
    }
    crop_group.attrs["run_manifest"] = manifest
    crop_reference = {
        "run_path": "crop_runs/crop",
        "manifest_digest": canonical_json_sha256(manifest),
        "manifest_payload_digest": manifest["payload_digest"],
        "logical_content_digest": logical["digest"],
        "row_signatures_digest": "c" * 64,
        "coordinate_catalog_digest": "d" * 64,
    }
    source_arrays = {
        "source_crop_row_ids": _Array(
            path="source/source_crop_row_ids", values=np.arange(n_instances, dtype="<i8")
        ),
        "instance_key": crop_arrays["instance_key"],
        "source_acquisition_frame_index": crop_arrays["source_acquisition_frame_index"],
        "source_crop_row_signature": crop_arrays["source_row_signature"],
    }
    acquisition = SimpleNamespace(record=SimpleNamespace(camera_id="cam"))
    point = SimpleNamespace(
        pixel_convention="continuous",
        endpoint=SimpleNamespace(width=width, height=height),
        record_ref="/camera/continuous",
        record_sha256="e" * 64,
    )
    bbox = SimpleNamespace(
        pixel_convention="pixel_edge_half_open",
        endpoint=SimpleNamespace(width=width, height=height),
        record_ref="/camera/pixel_edge_half_open",
        record_sha256="f" * 64,
    )
    root = _Root(
        {
            "crop_runs/crop": crop_group,
            "analysis/coordinate_frames/source_camera/cam/continuous": point,
            "analysis/coordinate_frames/source_camera/cam/pixel_edge_half_open": bbox,
        }
    )
    source_manifest = {
        "payload": {"logical_schema": {"dimensions": {"n_frames": n_frames, "n_instances": n_instances}}}
    }
    transform = ModelInputTransform(
        name="identity",
        native_height=384,
        native_width=384,
        model_height=384,
        model_width=384,
    )
    return {
        "archive": tmp_path,
        "root": root,
        "crop_group": crop_group,
        "crop_arrays": crop_arrays,
        "manifest": manifest,
        "crop_reference": crop_reference,
        "source_manifest": source_manifest,
        "source_arrays": source_arrays,
        "acquisition": acquisition,
        "point": point,
        "bbox": bbox,
        "transform": transform,
    }


def _patch_historical_crop_fixture(monkeypatch, fixture) -> None:
    monkeypatch.setattr(
        historical_crop,
        "open_persisted_crop_geometry_publication",
        lambda archive, run_id: SimpleNamespace(
            manifest=fixture["manifest"], arrays=fixture["crop_arrays"]
        ),
    )
    monkeypatch.setattr(historical_crop, "validate_crop_run_manifest", lambda manifest: ())
    monkeypatch.setattr(
        historical_crop,
        "load_persisted_acquisition_camera_authority",
        lambda root: (None, fixture["acquisition"]),
    )
    monkeypatch.setattr(
        historical_crop,
        "load_source_camera_pixel_frame_authority",
        lambda node, *, acquisition_frame: node,
    )


def _source_manifest() -> dict[str, object]:
    return {
        "schema_id": "palette.keypoint.run_manifest",
        "schema_version": 1,
        "payload_digest": "1" * 64,
        "payload": {"logical_content": {"digest": "2" * 64}},
    }


def _authority(run: _Node) -> dict[str, object]:
    coordinate = stamp_and_bind_persisted_coordinate_record(
        run,
        {"schema_id": "coordinate-test", "schema_version": 1},
        attr_name="coordinate_context",
    )
    return build_coordinate_successor_authority(
        kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
        source_family="keypoints_runs",
        source_run_path="keypoints_runs/source",
        source_manifest=_source_manifest(),
        source_authority_kind="sealed_keypoint_bundle",
        source_authority={"schema_id": "source-authority", "schema_version": 1},
        successor_family="keypoints_runs",
        successor_run_path=run.path,
        payload_equivalence={"mode": "hardlink_exact_payload"},
        coordinate_records={
            "context": {
                "record_ref": coordinate.record_ref,
                "record_sha256": coordinate.record_sha256,
            }
        },
    )


def test_coordinate_successor_authority_revalidates_persisted_record() -> None:
    run = _Node(path="keypoints_runs/successor")
    authority = _authority(run)

    assert validate_coordinate_successor_authority(authority) == ()
    stamp_coordinate_successor_authority(run, authority)
    assert load_coordinate_successor_authority(
        run,
        expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
        expected_successor_run_path=run.path,
    ) == authority

    run.attrs["coordinate_context"] = {
        "schema_id": "coordinate-test",
        "schema_version": 2,
    }
    with pytest.raises(
        CoordinateSuccessorAuthorityError,
        match="missing, malformed, or stale",
    ):
        load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run.path,
        )


def test_coordinate_successor_authority_tampering_fails_closed() -> None:
    run = _Node(path="keypoints_runs/successor")
    authority = _authority(run)
    changed = deepcopy(authority)
    changed["payload"]["successor"]["stage_selector_eligible"] = True

    errors = validate_coordinate_successor_authority(changed)

    assert "coordinate successor authority payload digest differs" in errors
    assert "coordinate successor target lifecycle is invalid" in errors

    stamp_coordinate_successor_authority(run, authority)
    persisted = deepcopy(run.attrs[COORDINATE_SUCCESSOR_AUTHORITY_ATTR])
    persisted["payload"]["production_state_changes"] = ["latest"]
    run.attrs[COORDINATE_SUCCESSOR_AUTHORITY_ATTR] = persisted
    with pytest.raises(CoordinateSuccessorAuthorityError, match="payload digest"):
        load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run.path,
        )


def test_successor_copy_separates_metadata_and_hardlinks_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "keypoints" / "c").mkdir(parents=True)
    (source / "zarr.json").write_text('{"node_type":"group"}\n')
    (source / "keypoints" / "zarr.json").write_text(
        '{"node_type":"array"}\n'
    )
    (source / "keypoints" / "c" / "0").write_bytes(b"payload")
    source_metadata = metadata_tree_sha256(source)

    receipt = copy_metadata_and_link_payload(source, target)

    assert receipt == {
        "metadata_files_copied": 2,
        "payload_files_hardlinked": 1,
    }
    assert metadata_tree_sha256(source) == source_metadata
    assert os.stat(source / "zarr.json").st_ino != os.stat(
        target / "zarr.json"
    ).st_ino
    assert os.stat(source / "keypoints" / "c" / "0").st_ino == os.stat(
        target / "keypoints" / "c" / "0"
    ).st_ino

    (target / "zarr.json").write_text('{"node_type":"group","attributes":{}}\n')
    assert metadata_tree_sha256(source) == source_metadata
    assert (source / "keypoints" / "c" / "0").read_bytes() == b"payload"


def test_historical_geometry_only_adapter_proves_keypoint_source_and_records_no_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _historical_crop_fixture(tmp_path)
    _patch_historical_crop_fixture(monkeypatch, fixture)

    binding = historical_crop.bind_historical_geometry_only_crop_source(
        analysis_zarr=fixture["archive"],
        root=fixture["root"],
        crop_reference=fixture["crop_reference"],
        source_manifest=fixture["source_manifest"],
        source_arrays=fixture["source_arrays"],
        source_run_path="keypoints_runs/source",
        model_input_transform=fixture["transform"],
    )

    record = binding.as_record()
    assert record["adapter_kind"] == (
        "sealed_geometry_only_crop_manifest_v2_no_pixel_source"
    )
    assert record["crop_path"] == "crop_runs/crop"
    assert record["crop_manifest_digest"] == canonical_json_sha256(fixture["manifest"])
    assert record["model_input_transform"] == fixture["transform"].to_attrs()
    assert record["pixel_source"] == "none_historical_coordinate_migration_only"
    assert record["ephemeral_flat_pixel_cache"] == "not_an_immutable_source"
    assert "coordinate_contract" not in fixture["crop_group"].attrs
    assert fixture["crop_group"].attrs["stage_selector_eligible"] is False
    durable_ref = "/crop_runs/crop@run_manifest"
    durable_digest = canonical_json_sha256(
        fixture["crop_group"].attrs["run_manifest"]
    )
    evidence = binding.source.crop_geometry.source_geometry.frame_evidence
    refs = [
        binding.source.crop_geometry.selection_derivation,
        binding.source.crop_geometry.row_identity,
        binding.source.roi_geometry.derivation,
        binding.source.roi_frame,
        binding.source.bbox_roi_frame,
        evidence.normalized_frame,
        *evidence.normalized_to_source_camera.transform_records,
    ]
    assert refs
    assert all(item.record_ref == durable_ref for item in refs)
    assert all(item.record_sha256 == durable_digest for item in refs)


def test_historical_geometry_only_adapter_is_shared_by_keypoint_and_raw_mask_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _historical_crop_fixture(tmp_path)
    _patch_historical_crop_fixture(monkeypatch, fixture)
    binding = historical_crop.bind_historical_geometry_only_crop_source(
        analysis_zarr=fixture["archive"],
        root=fixture["root"],
        crop_reference=fixture["crop_reference"],
        source_manifest=fixture["source_manifest"],
        source_arrays=fixture["source_arrays"],
        source_run_path="keypoints_runs/source",
        model_input_transform=fixture["transform"],
    )
    from fisheye.shared import keypoint_coordinate_publication as keypoint_publication
    from fisheye.shared import subject_mask_coordinate_publication as mask_publication

    original_keypoint_loader = keypoint_publication.load_persisted_keypoint_crop_source
    original_mask_loader = mask_publication.load_persisted_subject_mask_crop_source
    with historical_crop.historical_geometry_only_crop_loader(binding):
        assert (
            keypoint_publication.load_persisted_keypoint_crop_source(
                fixture["root"], binding.crop_path
            )
            is binding.source
        )
        assert (
            mask_publication.load_persisted_subject_mask_crop_source(
                fixture["root"], binding.crop_path
            )
            is binding.source
        )
        with pytest.raises(historical_crop.HistoricalGeometryOnlyCropAdapterError):
            keypoint_publication.load_persisted_keypoint_crop_source(
                object(), binding.crop_path
            )
    assert keypoint_publication.load_persisted_keypoint_crop_source is original_keypoint_loader
    assert mask_publication.load_persisted_subject_mask_crop_source is original_mask_loader
    with pytest.raises(RuntimeError, match="restore"):
        with historical_crop.historical_geometry_only_crop_loader(binding):
            raise RuntimeError("restore")
    assert keypoint_publication.load_persisted_keypoint_crop_source is original_keypoint_loader
    assert mask_publication.load_persisted_subject_mask_crop_source is original_mask_loader


def test_ordinary_keypoint_crop_loader_still_rejects_geometry_only_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _historical_crop_fixture(tmp_path)
    from fisheye.shared import keypoint_coordinate_publication as keypoint_publication

    geometry = SimpleNamespace(_rowset_node=fixture["crop_group"])
    monkeypatch.setattr(
        keypoint_publication,
        "load_persisted_crop_observation_geometry",
        lambda root, path: geometry,
    )
    monkeypatch.setattr(
        keypoint_publication,
        "require_bound_crop_observation_geometry",
        lambda value: value,
    )

    with pytest.raises(Exception, match="completion contract|canonical_v2"):
        keypoint_publication._load_persisted_keypoint_crop_source_fresh(
            fixture["root"], "crop_runs/crop"
        )


@pytest.mark.parametrize("mutation", ["manifest_digest", "extent", "missing_array", "model"])
def test_historical_geometry_only_adapter_rejects_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    fixture = _historical_crop_fixture(tmp_path)
    _patch_historical_crop_fixture(monkeypatch, fixture)
    crop_reference = dict(fixture["crop_reference"])
    transform = fixture["transform"]
    if mutation == "manifest_digest":
        crop_reference["manifest_digest"] = "9" * 64
    elif mutation == "extent":
        fixture["point"].endpoint.width = 101
    elif mutation == "missing_array":
        fixture["crop_arrays"] = dict(fixture["crop_arrays"])
        del fixture["crop_arrays"]["source_crop_xywh"]
    elif mutation == "model":
        transform = ModelInputTransform(
            name="identity",
            native_height=385,
            native_width=384,
            model_height=385,
            model_width=384,
        )

    with pytest.raises(historical_crop.HistoricalGeometryOnlyCropAdapterError):
        historical_crop.bind_historical_geometry_only_crop_source(
            analysis_zarr=fixture["archive"],
            root=fixture["root"],
            crop_reference=crop_reference,
            source_manifest=fixture["source_manifest"],
            source_arrays=fixture["source_arrays"],
            source_run_path="keypoints_runs/source",
            model_input_transform=transform,
        )


def test_historical_geometry_only_adapter_rejects_raw_mask_rowset_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _historical_crop_fixture(tmp_path)
    _patch_historical_crop_fixture(monkeypatch, fixture)
    source_arrays = {
        "source_crop_row_ids": fixture["source_arrays"]["source_crop_row_ids"],
        "instance_key": fixture["source_arrays"]["instance_key"],
        "source_acquisition_frame_index": fixture["source_arrays"][
            "source_acquisition_frame_index"
        ],
        "source_crop_xywh": _Array(
            path="source/source_crop_xywh",
            values=np.asarray(fixture["crop_arrays"]["source_crop_xywh"][...] + 1),
        ),
    }
    with pytest.raises(
        historical_crop.HistoricalGeometryOnlyCropAdapterError,
        match="placement",
    ):
        historical_crop.bind_historical_geometry_only_crop_source(
            analysis_zarr=fixture["archive"],
            root=fixture["root"],
            crop_reference=fixture["crop_reference"],
            source_manifest=fixture["source_manifest"],
            source_arrays=source_arrays,
            source_run_path="subject_mask_runs/raw",
            model_input_transform=fixture["transform"],
        )
