"""Record Orange's realtime products for one organized recording camera.

Step 1c of intake item 7 (docs/design/2026-10-09-realtime-products-intake).
At import, for a transfer-v2 recording, this validates the camera's
``realtime_products`` declaration, its detection and pose event logs, and the
start snapshot's model blocks, and writes one immutable record into the
analysis Zarr at ``analysis/acquisition_realtime_products``.

Declared file digests are checked against the sealed inventory the parent
manifest carries (``source_transfer.source_to_parent_files``), so only the
event logs are read; their lines are validated as they are scanned. Any
disagreement raises :class:`RealtimeProductsError` (a deterministic refusal).
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import zarr

from fisheye.shared.orange_realtime_products import (
    MODEL_SCHEMAS,
    PRODUCT_MODEL_KEY,
    PRODUCTS,
    RealtimeProductsError,
    check_product,
    model_block_schema,
    model_digests,
    scan_event_log,
    schema_error,
)
from fisheye.shared.recording_transfer_snapshot import strict_json

RECORD_SCHEMA_ID = "palette.acquisition_realtime_products"
RECORD_SCHEMA_VERSION = 1
REALTIME_PRODUCTS_GROUP = "analysis/acquisition_realtime_products"
_ENGINE_MANIFEST_FIELDS = (
    "path", "sha256", "present", "status", "run_id", "set_id", "build_id",
    "precision", "target_hardware_class", "engine_sha256_matches",
)
_RUNTIME_FIELDS = (
    "model_id", "engine_path", "engine_sha256", "engine_bytes", "weights_sha256",
    "onnx_sha256", "gpu_id", "backend", "mode",
)


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sealed_parent_files(manifest: Mapping[str, Any]) -> dict[str, dict] | None:
    """Source path -> {relative_path, sha256, size_bytes} of this parent's files."""

    transfer = manifest.get("source_transfer")
    files = transfer.get("source_to_parent_files") if isinstance(transfer, Mapping) else None
    if not isinstance(files, list):
        return None
    sealed = {}
    for item in files:
        source = item.get("source") if isinstance(item, Mapping) else None
        if isinstance(source, Mapping) and isinstance(source.get("path"), str):
            sealed[source["path"]] = {
                "relative_path": item["relative_path"],
                "sha256": source.get("sha256"),
                "size_bytes": source.get("size_bytes"),
            }
    return sealed


def _declared_file(sealed: Mapping[str, dict], declared: Mapping[str, Any], *, label: str) -> dict:
    found = sealed.get(declared["path"])
    if found is None:
        raise RealtimeProductsError(f"{label}: declared file {declared['path']} is not in this recording")
    if (found["sha256"], found["size_bytes"]) != (declared["sha256"], declared["size_bytes"]):
        raise RealtimeProductsError(
            f"{label}: declared digest or size of {declared['path']} differs from the sealed inventory"
        )
    return {
        "role": declared["role"],
        "path": found["relative_path"],
        "size_bytes": declared["size_bytes"],
        "sha256": declared["sha256"],
    }


def _model(kind: str, block: Mapping[str, Any] | None) -> dict:
    runtime = block.get("runtime") if isinstance(block, Mapping) else None
    runtime = runtime if isinstance(runtime, Mapping) else {}
    manifest = runtime.get("engine_manifest") if isinstance(runtime.get("engine_manifest"), Mapping) else {}
    return {
        "models_key_kind": kind,
        "schema": model_block_schema(kind, block),
        "schemas_tried": list(MODEL_SCHEMAS[kind]),
        **{name: runtime.get(name) for name in _RUNTIME_FIELDS if name in runtime},
        "engine_manifest": {name: manifest.get(name) for name in _ENGINE_MANIFEST_FIELDS if name in manifest},
    }


def build_realtime_products_record(recording_dir: Path, manifest: Mapping[str, Any]) -> dict | None:
    """The validated realtime-products record for this recording, or None.

    None for a recording without a transfer-v2 parent mapping. A transfer-v2
    recording whose session predates the declaration gets a record that says so.
    """

    sealed = _sealed_parent_files(manifest)
    if sealed is None or "recording_session.json" not in sealed:
        return None
    recording_dir = Path(recording_dir)
    camera = str(manifest.get("camera_id") or "")
    session = strict_json(recording_dir / sealed["recording_session.json"]["relative_path"])
    block = session.get("realtime_products")
    record: dict[str, Any] = {
        "schema_id": RECORD_SCHEMA_ID,
        "schema_version": RECORD_SCHEMA_VERSION,
        "camera_id": camera,
        "declared": block is not None,
    }
    if block is None:
        record["reason"] = "recording_session.json declares no realtime_products (Orange before de773b1)"
        record["record_sha256"] = _canonical_sha256(record)
        return record
    error = schema_error("realtime_products_v1", block)
    if error is not None:
        raise RealtimeProductsError(f"realtime_products violates its schema: {error}")
    declared = block["cameras"].get(camera)
    if declared is None:
        raise RealtimeProductsError(f"realtime_products declares nothing for camera {camera}")
    models: Mapping[str, Any] = {}
    if "recording_snapshot_start.json" in sealed:
        snapshot = strict_json(recording_dir / sealed["recording_snapshot_start.json"]["relative_path"])
        models = (snapshot.get("models") or {}).get(camera) or {}
    recording_id = str(manifest.get("session_uuid") or "")
    products: dict[str, Any] = {}
    for product in PRODUCTS:
        item = declared[product]
        label = f"Cam{camera} {product}"
        files = [_declared_file(sealed, f, label=label) for f in item["files"]]
        entry: dict[str, Any] = {"status": item["status"], "files": files}
        if item["status"] == "present":
            events = [f for f in files if f["role"] == "events"]
            if len(events) != 1:
                raise RealtimeProductsError(f"{label}: expected one events file, found {len(events)}")
            kind = PRODUCT_MODEL_KEY[product]
            summary = scan_event_log(
                recording_dir / events[0]["path"], product=product,
                line_schema_version=item["line_schema"]["schema_version"],
            )
            check_product(
                product, item, summary, camera_serial=camera, recording_id=recording_id,
                snapshot_model_digests=model_digests(models.get(kind)),
            )
            entry.update(
                line_schema=item["line_schema"],
                frame_identity_key=item["frame_identity_key"],
                row_count=summary.frame_rows,
                header_rows=summary.header_rows,
                rows_by_kind=item["rows_by_kind"],
                rows_by_status=item["rows_by_status"],
                first_recording_frame_id=item["first_recording_frame_id"],
                last_recording_frame_id=item["last_recording_frame_id"],
                validation=summary.validation,
                line_validator=summary.line_validator,
                model_ref=item["model_ref"],
                model=_model(kind, models.get(kind)),
            )
        products[product] = entry
    record.update(
        producer_schema={"schema_id": block["schema_id"], "schema_version": block["schema_version"]},
        products=products,
        crop_files=[_declared_file(sealed, f, label=f"Cam{camera} crop_files") for f in declared["crop_files"]],
        acquisition_files=[
            _declared_file(sealed, f, label=f"Cam{camera} acquisition_files") for f in declared["acquisition_files"]
        ],
    )
    record["record_sha256"] = _canonical_sha256(record)
    return record


def write_realtime_products_record(root: Any, record: Mapping[str, Any]) -> None:
    """Write the record once; an identical replay is a no-op, a different one refuses."""

    group = root.require_group(REALTIME_PRODUCTS_GROUP)
    try:
        group = zarr.open_group(
            store=group.store_path.store, path=str(group.path), mode="r+", use_consolidated=False
        )
    except (AttributeError, TypeError):
        pass
    existing = group.attrs.get("record_sha256")
    if existing is not None:
        if existing != record["record_sha256"]:
            raise RealtimeProductsError("analysis Zarr already holds a different realtime-products record")
        return
    group.attrs.put({"record": json.loads(json.dumps(record)), "record_sha256": record["record_sha256"]})


def read_realtime_products_record(root: Any) -> dict | None:
    """The stored record, verified against its own digest."""

    group = root.get(REALTIME_PRODUCTS_GROUP)
    if group is None:
        return None
    record = dict(group.attrs.get("record") or {})
    stored = record.pop("record_sha256", None)
    if stored != group.attrs.get("record_sha256") or _canonical_sha256(record) != stored:
        raise RealtimeProductsError("realtime-products record does not match its digest")
    record["record_sha256"] = stored
    return record


__all__ = [
    "REALTIME_PRODUCTS_GROUP",
    "RECORD_SCHEMA_ID",
    "build_realtime_products_record",
    "read_realtime_products_record",
    "write_realtime_products_record",
]
