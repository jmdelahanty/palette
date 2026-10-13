"""Registry rows for the realtime-products record in an analysis Zarr."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import zarr

from fisheye.shared.acquisition_realtime_products import read_realtime_products_record


def _int(value: Any) -> Optional[int]:
    return value if type(value) is int else None


def extract_realtime_product_rows(
    root: zarr.Group, *, recording_id: Optional[str]
) -> List[Dict[str, Any]]:
    """One row per product, or one ``undeclared`` row; empty without a record."""

    record = read_realtime_products_record(root)
    if record is None:
        return []
    base = {
        "recording_id": recording_id,
        "camera_id": record.get("camera_id"),
        "declared": 1 if record.get("declared") else 0,
        "record_sha256": record["record_sha256"],
    }
    if not record.get("declared"):
        return [{**base, "product": "undeclared", "status": "undeclared"}]
    rows = []
    for product, entry in sorted((record.get("products") or {}).items()):
        row: Dict[str, Any] = {**base, "product": product, "status": entry.get("status")}
        if entry.get("status") == "present":
            kinds = entry.get("rows_by_kind") or {}
            model = entry.get("model") or {}
            manifest = model.get("engine_manifest") or {}
            events = next((f for f in entry.get("files") or [] if f.get("role") == "events"), {})
            line_schema = entry.get("line_schema") or {}
            row.update(
                line_schema_id=line_schema.get("schema_id"),
                line_schema_version=_int(line_schema.get("schema_version")),
                row_count=_int(entry.get("row_count")),
                header_rows=_int(entry.get("header_rows")),
                result_rows=_int(kinds.get("result")),
                no_result_rows=_int(kinds.get("no_result")),
                failed_rows=_int(kinds.get("failed")),
                other_rows=_int(kinds.get("other")),
                first_recording_frame_id=_int(entry.get("first_recording_frame_id")),
                last_recording_frame_id=_int(entry.get("last_recording_frame_id")),
                validation=entry.get("validation"),
                line_validator=entry.get("line_validator"),
                events_path=events.get("path"),
                events_sha256=events.get("sha256"),
                model_id=model.get("model_id"),
                model_schema=model.get("schema"),
                engine_sha256=model.get("engine_sha256") or None,
                engine_bytes=_int(model.get("engine_bytes")),
                weights_sha256=model.get("weights_sha256") or None,
                onnx_sha256=model.get("onnx_sha256") or None,
                engine_manifest_sha256=manifest.get("sha256") or None,
                engine_manifest_run_id=manifest.get("run_id") or None,
                engine_manifest_status=manifest.get("status") or None,
                engine_precision=manifest.get("precision") or None,
                engine_build_id=manifest.get("build_id") or None,
            )
        rows.append(row)
    return rows


__all__ = ["extract_realtime_product_rows"]
