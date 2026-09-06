"""Catalog preservation tests exercise the real folder and pixel-frame readers."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import zarr

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    plan_producer_native_acquisition_geometry_candidate,
)
from fisheye.analysis_workflows.materializers import (
    arena_geometry_reference_catalog as catalog,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.recording_geometry import RecordingGeometryError
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_recording_geometry import (
    _json_bytes,
    _sha,
    _write_folder_bundle,
)


def _source(
    root: Path,
    *,
    camera: str = "2010093",
    organized: bool = True,
) -> catalog.GeometryReferenceSource:
    bundle = root / "raw/recording_geometry_bundle" if organized else root
    contract, snapshot = _write_folder_bundle(bundle)
    if camera != "2010093":
        # Re-key every producer camera declaration without numeric conversion.
        for path in (bundle / "recording_geometry_assets").rglob("observation.json"):
            target = Path(str(path).replace("Cam2010093", f"Cam{camera}"))
            target.parent.mkdir(parents=True)
            path.rename(target)
        for path in (bundle / "recording_geometry_assets/manifest.json",):
            value = json.loads(path.read_text().replace("2010093", camera))
            path.write_bytes(_json_bytes(value))
        contract = json.loads(json.dumps(contract).replace("2010093", camera))
        snapshot = json.loads(json.dumps(snapshot).replace("2010093", camera))
        contract["materialized_assets"]["sha256"] = _sha(
            (bundle / "recording_geometry_assets/manifest.json").read_bytes()
        )
        _update_contract(bundle, contract, snapshot)
    archive = root / "zarr/recording_analysis.zarr"
    group = zarr.open_group(
        str(archive), mode="w", zarr_format=3, use_consolidated=False
    )
    group.attrs.update(
        {
            "recording_id": root.name,
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "camera_id": camera,
                "width": 640,
                "height": 480,
                "total_frames": 2,
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": f"cams/{camera}.mp4",
                },
                "file_fingerprint": {
                    "strategy": "size_mtime_sha256_v1",
                    "value": "a" * 64,
                    "size_bytes": 1234,
                    "mtime_ns": 5678,
                    "relocation_stable": False,
                },
            },
        }
    )
    node = group.require_group(f"analysis/acquisition_camera_frames/{camera}")
    ownership = stamp_acquisition_import_ownership(group, node)
    stamp_acquisition_camera_frame(group, node, import_ownership=ownership)
    _, acquisition = load_persisted_acquisition_camera_authority(
        group, expected_camera_id=camera
    )
    frame = group.require_group(
        f"analysis/coordinate_frames/source_camera/{camera}/continuous"
    )
    stamp_source_camera_pixel_frame_authority(
        frame,
        frame_id=f"{camera}_source_camera",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )
    return catalog.GeometryReferenceSource(
        recording_root=root,
        source_zarr=archive,
        rig_id="omnifin0",
        canvas_name="shadow",
        arena_id="arena_1",
        camera_serial=camera,
        applicable_at_utc="2026-07-22T12:00:00Z",
    )


def _update_contract(root: Path, contract: dict, snapshot: dict | None = None) -> None:
    if snapshot is None:
        snapshot = json.loads((root / "recording_snapshot.json").read_text())
    raw = _json_bytes(contract, newline=True)
    (root / "recording_geometry_contract.json").write_bytes(raw)
    snapshot["recording_geometry_contract"]["sha256"] = _sha(raw)
    (root / "recording_snapshot.json").write_bytes(_json_bytes(snapshot))


def _resign(payload: dict) -> None:
    payload["catalog_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "catalog_sha256"}
    )


def test_catalog_deduplicates_deterministically_and_preserves_candidate(tmp_path):
    first = _source(tmp_path / "first")
    second = _source(tmp_path / "second", organized=False)
    result = catalog.build_geometry_reference_catalog([second, first, first])
    assert result == catalog.build_geometry_reference_catalog([first, second])
    assert result["catalog_role"] == "non_authoritative_index"
    assert result["metadata_read_mode"] == "unconsolidated_diagnostic"
    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert len(entry["sources"]) == 2
    assert entry["reference"]["physical_inner_rim"]["geometry"]["radius_px"] == 200.0
    gate = entry["reference"]["valid_detection_region"]
    assert gate["geometry"]["radius_px"] == 205.0
    assert gate["additional_palette_tolerance_px"] == 0.0
    assert entry["reference"]["producer_operator_accepted"] is True
    assert entry["reference"]["producer_quality_flags"] == []
    # The folder supplier is sufficient; missing Citrus application is not a gate.
    assert (
        entry["sources"][0]["acquisition_source"][
            "selected_daily_registration_applied_by_citrus"
        ]
        is None
    )
    assert catalog.validate_geometry_reference_catalog(result) == result
    key = catalog.GeometryReferenceKey(**entry["key"])
    record = catalog.resolve_geometry_reference(
        result, key=key, source_zarr=first.source_zarr
    )
    plan = plan_producer_native_acquisition_geometry_candidate(
        source_zarr=first.source_zarr,
        recording_folder=first.recording_root,
        camera_serial=first.camera_serial,
        arena_id=first.arena_id,
    )
    assert record == plan.candidate_record
    assert result["entries"][0]["sources"][0]["candidate_record_sha256"] == (
        plan.candidate_record_sha256
    )
    assert not plan.target_run_path.exists()


def test_catalog_preserves_leading_zero_camera_text(tmp_path):
    source = _source(tmp_path / "recording", camera="002010093")
    result = catalog.build_geometry_reference_catalog([source])
    assert result["entries"][0]["key"]["camera_serial"] == "002010093"
    with pytest.raises(RecordingGeometryError):
        catalog.build_geometry_reference_catalog(
            [replace(source, camera_serial=2010093)]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("rig_id", "other"),
        ("canvas_name", "other"),
        ("arena_id", "arena_2"),
        ("camera_serial", "01"),
    ],
)
def test_catalog_rejects_wrong_source_scope(tmp_path, field, value):
    source = _source(tmp_path / "recording")
    with pytest.raises(RecordingGeometryError):
        catalog.build_geometry_reference_catalog([replace(source, **{field: value})])


@pytest.mark.parametrize("damage", ["missing", "tampered", "unapproved", "partial"])
def test_catalog_reopens_assets_and_rejects_bad_producer_evidence(tmp_path, damage):
    source = _source(tmp_path / "recording")
    result = catalog.build_geometry_reference_catalog([source])
    bundle = source.recording_root / "raw/recording_geometry_bundle"
    observation = next((bundle / "recording_geometry_assets").rglob("observation.json"))
    if damage == "missing":
        observation.unlink()
    elif damage == "tampered":
        observation.write_bytes(b"tampered")
    else:
        contract = json.loads((bundle / "recording_geometry_contract.json").read_text())
        daily = contract["daily_registration_geometry"]
        if damage == "unapproved":
            daily["cameras"][source.camera_serial]["recording_snapshot_entry"][
                "operator_review"
            ]["accepted"] = False
        else:
            daily["status"] = "selected_partial"
        _update_contract(bundle, contract)
    with pytest.raises(RecordingGeometryError):
        catalog.validate_geometry_reference_catalog(result)


def test_catalog_reopens_live_source_pixel_authority(tmp_path):
    source = _source(tmp_path / "recording")
    result = catalog.build_geometry_reference_catalog([source])
    group = zarr.open_group(str(source.source_zarr), mode="a", use_consolidated=False)
    group["analysis/coordinate_frames/source_camera/2010093/continuous"].attrs[
        "pixel_frame_authority_sha256"
    ] = ("f" * 64)
    with pytest.raises(ValueError):
        catalog.validate_geometry_reference_catalog(result)


def test_catalog_expiry_is_explicit_not_wall_clock_based(tmp_path):
    source = _source(tmp_path / "recording")
    historical = catalog.build_geometry_reference_catalog([source])
    key = catalog.GeometryReferenceKey(**historical["entries"][0]["key"])
    with pytest.raises(RecordingGeometryError, match="expired"):
        catalog.resolve_geometry_reference(
            historical,
            key=key,
            source_zarr=source.source_zarr,
            applicable_at_utc="2026-07-23T04:00:00Z",
        )
    with pytest.raises(RecordingGeometryError, match="timezone"):
        catalog.build_geometry_reference_catalog(
            [replace(source, applicable_at_utc="2026-07-22T12:00:00")]
        )
    unscoped = catalog.build_geometry_reference_catalog(
        [replace(source, applicable_at_utc=None)]
    )
    assert unscoped["entries"][0]["sources"][0]["validity_check"] == "not_requested"


def test_catalog_rejects_conflicting_reference_geometry(tmp_path):
    first = _source(tmp_path / "first")
    second = _source(tmp_path / "second")
    bundle = second.recording_root / "raw/recording_geometry_bundle"
    contract = json.loads((bundle / "recording_geometry_contract.json").read_text())
    contract["daily_registration_geometry"]["cameras"][second.camera_serial][
        "recording_snapshot_entry"
    ]["valid_detection_region"]["geometry"]["radius_px"] = 206.0
    _update_contract(bundle, contract)
    with pytest.raises(RecordingGeometryError, match="conflict"):
        catalog.build_geometry_reference_catalog([first, second])


def test_catalog_rejects_same_registration_id_with_conflicting_digest(tmp_path):
    first = _source(tmp_path / "first")
    second = _source(tmp_path / "second")
    bundle = second.recording_root / "raw/recording_geometry_bundle"
    contract = json.loads((bundle / "recording_geometry_contract.json").read_text())
    contract["daily_registration_geometry"]["registration"]["sha256"] = (
        "sha256:" + "f" * 64
    )
    _update_contract(bundle, contract)
    with pytest.raises(RecordingGeometryError, match="conflict"):
        catalog.build_geometry_reference_catalog([first, second])


@pytest.mark.parametrize("damage", ["digest", "entry", "extra", "empty", "pixel"])
def test_catalog_rejects_tampering_even_with_recomputed_index_digest(tmp_path, damage):
    source = _source(tmp_path / "recording")
    result = copy.deepcopy(catalog.build_geometry_reference_catalog([source]))
    if damage == "digest":
        result["catalog_sha256"] = "f" * 64
    elif damage == "entry":
        result["entries"][0]["reference"]["physical_inner_rim"]["geometry"][
            "radius_px"
        ] = 199.0
    elif damage == "extra":
        result["approval"] = True
    elif damage == "empty":
        result["entries"][0]["sources"] = []
    else:
        result["entries"][0]["sources"][0]["coordinate_binding"][
            "pixel_frame_record_sha256"
        ] = ("f" * 64)
    if damage != "digest":
        _resign(result)
    with pytest.raises(RecordingGeometryError):
        catalog.validate_geometry_reference_catalog(result)


def test_resolver_never_uses_other_recording_or_nearest_reference(tmp_path):
    source = _source(tmp_path / "recording")
    result = catalog.build_geometry_reference_catalog([source])
    key = catalog.GeometryReferenceKey(**result["entries"][0]["key"])
    with pytest.raises(RecordingGeometryError, match="exact"):
        catalog.resolve_geometry_reference(
            result,
            key=replace(key, registration_id="different"),
            source_zarr=source.source_zarr,
        )
    with pytest.raises(RecordingGeometryError, match="exact"):
        catalog.resolve_geometry_reference(
            result, key=key, source_zarr=tmp_path / "other.zarr"
        )


def test_inspection_cli_build_and_validate_are_read_only(tmp_path, capsys):
    from fisheye.utils.inspect_geometry_reference_catalog import main

    source = _source(tmp_path / "recording")
    source_file = tmp_path / "sources.json"
    source_file.write_text(json.dumps([asdict(source)], default=str))
    assert main(["build", "--source-list", str(source_file)]) == 0
    result = json.loads(capsys.readouterr().out)
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(result))
    assert main(["validate", "--catalog", str(catalog_file)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated_index_only"
    assert not (source.source_zarr / "analysis/arena_geometry_runs").exists()
