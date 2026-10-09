"""Orange realtime products: declaration, model blocks and v2 event logs.

The fixture is cut from the reference delivery d830510c (camera 2010093): the
session_header plus three frame lines of each v2 log, with its declaration
rewritten for those three frames.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from fisheye.shared import orange_realtime_products as rp

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "orange_realtime_products_v2"
CAMERA = "2010093"
LOGS = {"detections": "Cam2010093_yolo_events.jsonl", "pose": "Cam2010093_pose_events.jsonl"}


def _copy(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURE, tmp_path / "fixture"))


def _block(root: Path) -> dict:
    return json.loads((root / "realtime_products.json").read_text())


def _models(root: Path) -> dict:
    return json.loads((root / "models.json").read_text())[CAMERA]


def _recording_id(root: Path) -> str:
    return json.loads((root / LOGS["detections"]).read_text().splitlines()[0])["recording_id"]


def _check(root: Path, product: str, *, declared: dict | None = None) -> rp.EventLogSummary:
    block = _block(root)
    declared = declared or block["cameras"][CAMERA][product]
    summary = rp.scan_event_log(
        root / LOGS[product], product=product,
        line_schema_version=declared["line_schema"]["schema_version"],
    )
    rp.check_product(
        product, declared, summary, camera_serial=CAMERA, recording_id=_recording_id(root),
        snapshot_model_digests=rp.model_digests(_models(root)[rp.PRODUCT_MODEL_KEY[product]]),
    )
    return summary


def _rewrite_line(path: Path, index: int, change) -> None:
    lines = path.read_text().splitlines()
    line = json.loads(lines[index])
    change(line)
    lines[index] = json.dumps(line, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")


def test_the_reference_declaration_validates_and_names_its_cameras(tmp_path):
    root = _copy(tmp_path)
    session = {"realtime_products": _block(root)}
    assert rp.realtime_products_declaration(session, cameras=[CAMERA])["schema_id"] == rp.REALTIME_PRODUCTS_SCHEMA_ID
    assert rp.realtime_products_declaration({}, cameras=[CAMERA]) is None
    with pytest.raises(rp.RealtimeProductsError, match="declares cameras"):
        rp.realtime_products_declaration(session, cameras=[CAMERA, "2010094"])


def test_a_malformed_declaration_is_refused(tmp_path):
    block = _block(_copy(tmp_path))
    block["cameras"][CAMERA]["detections"]["unexpected"] = 1
    with pytest.raises(rp.RealtimeProductsError, match="violates its schema"):
        rp.realtime_products_declaration({"realtime_products": block}, cameras=[CAMERA])


def test_model_blocks_validate_against_the_newest_pinned_schema(tmp_path):
    models = _models(_copy(tmp_path))
    assert rp.model_block_schema("detect", models["detect"]) == "detect_model_v1"
    assert rp.model_block_schema("pose", models["pose"]) == "pose_model_v2"
    assert rp.model_block_schema("detect", {"enabled": True}) is None
    assert rp.model_digests(models["detect"])["engine_sha256"].startswith("479e82d2")


@pytest.mark.parametrize("product", rp.PRODUCTS)
def test_every_v2_line_validates_and_matches_the_declaration(tmp_path, product):
    summary = _check(_copy(tmp_path), product)
    assert summary.validation == "every_line_v2"
    assert (summary.frame_rows, summary.header_rows) == (3, 1)
    assert (summary.first_recording_frame_id, summary.last_recording_frame_id) == (1, 3)
    assert summary.rows_by_kind == {"result": 3, "no_result": 0, "failed": 0, "other": 0}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("row_count", 4, "declared row_count 4 but the log has 3"),
        ("last_recording_frame_id", 4, "declared last_recording_frame_id"),
        ("rows_by_kind", {"result": 2, "no_result": 0, "failed": 1, "other": 0}, "declared rows_by_kind"),
        ("header_rows", 0, "declared header_rows 0"),
    ],
)
def test_a_declaration_that_disagrees_with_its_log_is_refused(tmp_path, field, value, message):
    root = _copy(tmp_path)
    declared = _block(root)["cameras"][CAMERA]["detections"]
    declared[field] = value
    with pytest.raises(rp.RealtimeProductsError, match=message):
        _check(root, "detections", declared=declared)


def test_a_model_ref_that_disagrees_with_the_snapshot_is_refused(tmp_path):
    root = _copy(tmp_path)
    declared = _block(root)["cameras"][CAMERA]["pose"]
    declared["model_ref"]["engine_sha256"] = "0" * 64
    with pytest.raises(rp.RealtimeProductsError, match="model_ref engine_sha256"):
        _check(root, "pose", declared=declared)


@pytest.mark.parametrize(
    ("index", "change", "message"),
    [
        (2, lambda line: line.update(event_sequence=5), "event_sequence 5 is not 2"),
        (3, lambda line: line["frame"].update(recording_frame_id=2), "does not increase"),
        (1, lambda line: line.update(unexpected=True), "violates yolo_event_v2"),
        (0, lambda line: line.update(event_kind="yolo_result"), "violates yolo_event_v2|first line is not a session_header"),
    ],
)
def test_a_bad_v2_line_is_refused(tmp_path, index, change, message):
    root = _copy(tmp_path)
    _rewrite_line(root / LOGS["detections"], index, change)
    with pytest.raises(rp.RealtimeProductsError, match=message):
        rp.scan_event_log(root / LOGS["detections"], product="detections", line_schema_version=2)


def test_a_header_for_another_camera_is_refused(tmp_path):
    root = _copy(tmp_path)
    block = _block(root)
    summary = rp.scan_event_log(root / LOGS["pose"], product="pose", line_schema_version=2)
    with pytest.raises(rp.RealtimeProductsError, match="another camera or recording"):
        rp.check_product(
            "pose", block["cameras"][CAMERA]["pose"], summary, camera_serial=CAMERA,
            recording_id="some_other_recording",
            snapshot_model_digests=rp.model_digests(_models(root)["pose"]),
        )


def test_a_v1_log_is_counted_not_validated(tmp_path):
    log = tmp_path / "Cam1_yolo_events.jsonl"
    log.write_text('{"anything": 1}\n{"anything": 2}\n')
    summary = rp.scan_event_log(log, product="detections", line_schema_version=1)
    assert (summary.validation, summary.frame_rows, summary.header_rows) == ("file_level_v1", 2, 0)


def test_pinned_schemas_detect_drift(monkeypatch):
    rp._validator.cache_clear()
    monkeypatch.setitem(rp._SCHEMAS, "yolo_event_v2", ("orange_yolo_event_v2.schema.json", "0" * 64))
    try:
        with pytest.raises(rp.RealtimeProductsError, match="packaged_contract_drift"):
            rp._validator("yolo_event_v2")
    finally:
        rp._validator.cache_clear()
