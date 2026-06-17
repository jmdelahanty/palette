"""Datasette plugin: per-model download bundles for the registry browser.

Adds  GET /download/model/<run_id>  -> a zip with the portable artifacts
(.pt + .onnx when present) and a manifest.json assembled from the registry.
The hardware-specific TensorRT .engine is excluded by default; add
`?include_engine=1` to include it. Either way it is described in the manifest.

Tunnel-only by design: this serves files from disk, so the server must stay
bound to 127.0.0.1 and be reached over SSH. Every file path is looked up in the
registry and confined to an allowlist of model roots — client input (the
run_id) is only ever used as a parameterized lookup key, never as a path.

Loaded via:  datasette ... --plugins-dir docs/registry_browser/plugins
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from datasette import hookimpl
from datasette.utils.asgi import Response

# Allowlisted roots that artifact paths must live under. Override with
# PALETTE_MODEL_ROOTS (os.pathsep-separated absolute paths).
_DEFAULT_ROOTS = (
    "/groups/johnson/johnsonlab/jeremy/models",
    "/nvme1/training/models",
)

# Where each artifact kind is placed inside the zip.
_KIND_DIR = {"training": "weights", "onnx": "onnx", "tensorrt": "tensorrt"}


def model_roots() -> list[Path]:
    env = os.environ.get("PALETTE_MODEL_ROOTS")
    raw = env.split(os.pathsep) if env else list(_DEFAULT_ROOTS)
    out: list[Path] = []
    for p in raw:
        if p.strip():
            try:
                out.append(Path(p).resolve())
            except (OSError, RuntimeError):
                pass
    return out


def resolve_allowlisted(path: str | None, roots: list[Path]) -> Path | None:
    """Real path if it exists, is a file, and lives under an allowlisted root.

    Uses realpath before the containment check, so symlinks that escape the
    allowlist are rejected.
    """
    if not path:
        return None
    try:
        real = Path(path).resolve()
    except (OSError, RuntimeError):
        return None
    if not real.is_file():
        return None
    for root in roots:
        try:
            real.relative_to(root)
            return real
        except ValueError:
            continue
    return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(text: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text or "").strip("_") or "model"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open_ro(registry_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON;")
    return conn


def _one(conn: sqlite3.Connection, sql: str, params: tuple) -> dict | None:
    try:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None


def _assemble_manifest(conn, run_id, set_id, task_type, file_entries, warnings, registry_path):
    onnx = _one(
        conn,
        "SELECT input_shape, input_layout, img_h, img_w, max_batch, dynamic_shapes, opset, "
        "nms_conf, nms_iou, nms_topk, requires_plugins, plugin_ops_json, file_size_bytes, "
        "exporter_torch_version, exporter_cuda_version, skeleton_id "
        "FROM onnx_models WHERE run_id = ?",
        (run_id,),
    )
    mis = _one(
        conn,
        "SELECT input_shape, input_layout, input_channels, input_dtype, input_color_space, "
        "max_batch, dynamic_shapes FROM model_input_shapes "
        "WHERE run_id = ? AND input_shape IS NOT NULL LIMIT 1",
        (run_id,),
    )
    trt = _one(
        conn,
        "SELECT precision, trt_version, cuda_version, compute_capability, gpu_name, sha256, "
        "file_size_bytes, path, requires_plugins, plugin_ops_json "
        "FROM tensorrt_models WHERE run_id = ?",
        (run_id,),
    )
    trun = _one(
        conn,
        "SELECT status, config_path, final_metrics_json, created_utc FROM training_runs WHERE run_id = ?",
        (run_id,),
    )
    tset = _one(
        conn,
        "SELECT name, task_type, query_filter, skeleton_id, created_utc FROM training_sets WHERE set_id = ?",
        (set_id,),
    )

    contract = mis or {}
    detection_params = {}
    if onnx:
        detection_params = {
            k: onnx.get(k)
            for k in ("nms_conf", "nms_iou", "nms_topk", "requires_plugins", "plugin_ops_json")
        }

    skeleton_id = (onnx or {}).get("skeleton_id") or (tset or {}).get("skeleton_id")
    skeleton = _one(conn, "SELECT * FROM pose_skeleton_specs WHERE skeleton_id = ?", (skeleton_id,)) if skeleton_id else None

    engine = None
    if trt:
        engine = {
            "filename": os.path.basename(trt.get("path") or ""),
            "precision": trt.get("precision"),
            "sha256": trt.get("sha256"),
            "file_size_bytes": trt.get("file_size_bytes"),
            "build_env": {
                "trt_version": trt.get("trt_version"),
                "cuda_version": trt.get("cuda_version"),
                "compute_capability": trt.get("compute_capability"),
                "gpu_name": trt.get("gpu_name"),
            },
            "portable": "best-effort",
            "note": (
                "TensorRT engines are tied to the GPU architecture and the TRT/CUDA "
                "versions used at build time (see build_env). The engine may still load "
                "on similar hardware but can fail or run sub-optimally elsewhere; rebuild "
                "from the included .onnx for best performance."
            ),
        }

    return {
        "schema": "palette.model_bundle/v1",
        "generated_utc": _utc_now(),
        "registry_path": str(registry_path),
        "run_id": run_id,
        "set_id": set_id,
        "task_type": task_type,
        "inference_contract": contract,
        "detection_params": detection_params,
        "files": file_entries,
        "tensorrt_engine": engine,
        "pose_skeleton": skeleton,
        "training": {"run": trun, "set": tset},
        "warnings": warnings,
    }


def build_bundle(registry_path: str, run_id: str, include_engine: bool):
    """Build the zip for one model run. Returns (zip_bytes, filename) or None
    if the run_id is unknown. Pure/sync so it can run in a worker thread and be
    unit-tested without the server."""
    roots = model_roots()
    conn = _open_ro(registry_path)
    try:
        arts = [
            dict(r)
            for r in conn.execute(
                "SELECT artifact_kind, artifact_path, artifact_sha256, set_id, task_type "
                "FROM model_input_shapes WHERE run_id = ? ORDER BY artifact_kind",
                (run_id,),
            ).fetchall()
        ]
        if not arts:
            return None

        set_id = next((a.get("set_id") for a in arts if a.get("set_id")), run_id)
        task_type = next((a.get("task_type") for a in arts if a.get("task_type")), None)

        zip_files: list[tuple[str, Path]] = []
        file_entries: list[dict] = []
        warnings: list[str] = []

        for a in arts:
            kind = a["artifact_kind"]
            if kind == "tensorrt" and not include_engine:
                file_entries.append({
                    "kind": kind, "included": False,
                    "reason": "excluded by default (hardware-specific); pass ?include_engine=1",
                })
                continue

            real = resolve_allowlisted(a.get("artifact_path"), roots)
            if real is None:
                warnings.append(f"{kind}: artifact unreachable or outside allowlist: {a.get('artifact_path')}")
                file_entries.append({"kind": kind, "included": False, "reason": "unreachable_or_disallowed"})
                continue

            actual = sha256_of(real)
            expected = a.get("artifact_sha256")
            verified = expected is not None and actual == expected
            if expected is not None and not verified:
                warnings.append(f"{kind}: sha256 mismatch, omitted: {real}")
                file_entries.append({
                    "kind": kind, "included": False, "reason": "sha256_mismatch",
                    "expected_sha256": expected, "actual_sha256": actual,
                })
                continue

            arcname = f"{_KIND_DIR.get(kind, kind)}/{real.name}"
            zip_files.append((arcname, real))
            file_entries.append({
                "kind": kind,
                "arcname": arcname,
                "filename": real.name,
                "sha256": actual,
                "sha256_verified": verified,
                "file_size_bytes": real.stat().st_size,
                "included": True,
                "portable": True if kind in ("training", "onnx") else "best-effort",
            })

        manifest = _assemble_manifest(conn, run_id, set_id, task_type, file_entries, warnings, registry_path)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            for arcname, real in zip_files:
                zf.write(real, arcname)

        filename = f"{_safe_name(set_id)}__{_safe_name(run_id)}.zip"
        return buf.getvalue(), filename
    finally:
        conn.close()


async def download_model(request, datasette):
    run_id = unquote(request.url_vars["run_id"])
    include_engine = (request.args.get("include_engine") or "").lower() in {"1", "true", "yes", "on"}
    registry_path = datasette.get_database("palette_registry").path
    result = await asyncio.to_thread(build_bundle, registry_path, run_id, include_engine)
    if result is None:
        return Response.text(f"Unknown model run_id: {run_id}", status=404)
    body, filename = result
    return Response(
        body=body,
        status=200,
        headers={"content-disposition": f'attachment; filename="{filename}"'},
        content_type="application/zip",
    )


@hookimpl
def register_routes():
    return [(r"^/download/model/(?P<run_id>.+)$", download_model)]
