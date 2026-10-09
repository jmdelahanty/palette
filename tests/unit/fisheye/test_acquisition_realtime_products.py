"""Realtime products recorded at import for one organized recording camera."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import zarr

from fisheye.shared.acquisition_realtime_products import (
    REALTIME_PRODUCTS_GROUP,
    build_realtime_products_record,
    read_realtime_products_record,
    write_realtime_products_record,
)
from fisheye.shared.orange_realtime_products import RealtimeProductsError

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "orange_realtime_products_v2"
CAMERA = "2010093"


def _recording(tmp_path: Path, *, declare: bool = True, change=None) -> tuple[Path, dict]:
    """An organized recording folder with Orange's session, snapshot and logs."""

    rec = tmp_path / "recording"
    acq = rec / "raw/acquisition"
    acq.mkdir(parents=True)
    block = json.loads((FIXTURE / "realtime_products.json").read_text())
    if change is not None:
        change(block)
    session = {"realtime_products": block} if declare else {}
    files = {
        "recording_session.json": json.dumps(session).encode(),
        "recording_snapshot_start.json": json.dumps(
            {"models": json.loads((FIXTURE / "models.json").read_text())}
        ).encode(),
    }
    declared = block["cameras"][CAMERA]
    for group in (declared["detections"]["files"], declared["pose"]["files"],
                  declared["crop_files"], declared["acquisition_files"]):
        for item in group:
            source = FIXTURE / item["path"]
            files[item["path"]] = source.read_bytes() if source.exists() else b"placeholder"
    mapping = []
    for name, data in files.items():
        (acq / name).write_bytes(data)
        sealed = {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        for group in (declared["detections"]["files"], declared["pose"]["files"],
                      declared["crop_files"], declared["acquisition_files"]):
            for item in group:
                if item["path"] == name and not (FIXTURE / name).exists():
                    # Declared digests of files the fixture doesn't carry.
                    sealed.update(sha256=item["sha256"], size_bytes=item["size_bytes"])
        mapping.append({"relative_path": f"raw/acquisition/{name}", "role": "session_context", "source": sealed})
    recording_id = json.loads((FIXTURE / "Cam2010093_yolo_events.jsonl").read_text().splitlines()[0])["recording_id"]
    manifest = {
        "camera_id": CAMERA,
        "session_uuid": recording_id,
        "source_transfer": {"source_to_parent_files": mapping},
    }
    return rec, manifest


def test_the_record_validates_and_describes_both_products(tmp_path):
    rec, manifest = _recording(tmp_path)
    record = build_realtime_products_record(rec, manifest)
    assert record["declared"] is True and record["camera_id"] == CAMERA
    detections = record["products"]["detections"]
    assert (detections["validation"], detections["row_count"]) == ("every_line_v2", 3)
    assert detections["model"]["schema"] == "detect_model_v1"
    assert detections["model"]["engine_manifest"]["precision"] == "int8"
    assert record["products"]["pose"]["model"]["schema"] == "pose_model_v2"
    assert all(f["path"].startswith("raw/acquisition/") for f in detections["files"])
    assert len(record["record_sha256"]) == 64


def test_the_record_is_written_once_and_read_back_verified(tmp_path):
    rec, manifest = _recording(tmp_path)
    record = build_realtime_products_record(rec, manifest)
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w")
    write_realtime_products_record(root, record)
    write_realtime_products_record(root, record)  # identical replay is a no-op
    assert read_realtime_products_record(root) == record
    other = dict(record, camera_id="9999999", record_sha256="0" * 64)
    with pytest.raises(RealtimeProductsError, match="different realtime-products record"):
        write_realtime_products_record(root, other)
    group = root[REALTIME_PRODUCTS_GROUP]
    tampered = dict(group.attrs["record"], camera_id="tampered")
    group.attrs["record"] = tampered
    with pytest.raises(RealtimeProductsError, match="does not match its digest"):
        read_realtime_products_record(root)


def test_a_declared_digest_that_disagrees_with_the_seal_is_refused(tmp_path):
    def change(block):
        block["cameras"][CAMERA]["detections"]["files"][0]["sha256"] = "0" * 64

    rec, manifest = _recording(tmp_path, change=change)
    with pytest.raises(RealtimeProductsError, match="differs from the sealed inventory"):
        build_realtime_products_record(rec, manifest)


def test_a_declaration_that_disagrees_with_its_log_is_refused(tmp_path):
    def change(block):
        block["cameras"][CAMERA]["pose"]["row_count"] = 4

    rec, manifest = _recording(tmp_path, change=change)
    with pytest.raises(RealtimeProductsError, match="declared row_count 4"):
        build_realtime_products_record(rec, manifest)


def test_a_session_before_the_declaration_is_recorded_as_undeclared(tmp_path):
    rec, manifest = _recording(tmp_path, declare=False)
    record = build_realtime_products_record(rec, manifest)
    assert record["declared"] is False and "before de773b1" in record["reason"]


def test_recordings_without_a_transfer_mapping_get_no_record(tmp_path):
    rec, manifest = _recording(tmp_path)
    manifest.pop("source_transfer")
    assert build_realtime_products_record(rec, manifest) is None
