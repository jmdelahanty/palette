"""Helpers for persisted plot and visualization artifacts.

Palette stores run-local rendered snapshots as compressed PNG byte arrays under
``<run>/visualizations/<artifact_name>``. Interactive plots should additionally
store a small render spec that points back to canonical source arrays, rather
than treating rendered pixels as the source of truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps


PNG_ARTIFACT_SCHEMA_ID = "palette.visualization.png_bytes.v1"
INTERACTIVE_SPEC_SCHEMA_ID = "palette.visualization.interactive_spec.v1"
SPEC_MEDIA_TYPE = "application/vnd.palette.plot-spec+json"
DEFAULT_PNG_CHUNK_BYTES = 1_048_576


@dataclass(frozen=True)
class PlotArtifactResult:
    artifact_name: str
    path: str
    content_sha256: str
    byte_length: int
    artifact_schema_id: str
    created_at_utc: str


_json_safe = json_attr_safe


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_visualizations_group(run_group: zarr.Group) -> zarr.Group:
    return run_group.require_group("visualizations")


def _update_visualization_manifest(
    run_group: zarr.Group,
    artifact_name: str,
    manifest_entry: Mapping[str, Any],
) -> None:
    current = run_group.attrs.get("visualizations")
    manifest = dict(current) if isinstance(current, Mapping) else {}
    manifest[artifact_name] = _json_safe(manifest_entry)
    run_group.attrs["visualizations"] = manifest


def write_png_visualization_artifact(
    run_group: zarr.Group,
    artifact_name: str,
    png_bytes: bytes,
    *,
    description: str,
    created_by: str,
    artifact_signature: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    role: str = "snapshot",
    source_paths: Optional[Mapping[str, Any]] = None,
    source_runs: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    extra_attrs: Optional[Mapping[str, Any]] = None,
    overwrite: bool = True,
) -> PlotArtifactResult:
    """Write a static PNG snapshot artifact into ``run_group/visualizations``.

    The artifact is stored as a one-dimensional ``uint8`` array containing PNG
    bytes. This matches the existing detect/keypoint finalized-artifact pattern.
    """

    if not artifact_name:
        raise ValueError("artifact_name must not be empty")
    if not png_bytes:
        raise ValueError("png_bytes must not be empty")

    vis_group = _require_visualizations_group(run_group)
    if artifact_name in vis_group:
        if not overwrite:
            raise ValueError(f"visualization artifact already exists: {artifact_name}")
        del vis_group[artifact_name]

    created_at = created_at_utc or utc_now()
    content_sha256 = _sha256_bytes(png_bytes)
    data = np.frombuffer(png_bytes, dtype=np.uint8)
    chunk = max(1, min(int(data.shape[0]), DEFAULT_PNG_CHUNK_BYTES))
    dataset = vis_group.create_array(
        artifact_name,
        data=data,
        chunks=(chunk,),
        overwrite=True,
    )

    attrs: dict[str, Any] = {
        "artifact_schema_id": PNG_ARTIFACT_SCHEMA_ID,
        "artifact_type": "visualization",
        "artifact_role": role,
        "media_type": "image/png",
        "mime": "image/png",
        "storage_encoding": "png_bytes_uint8",
        "description": description,
        "created_at_utc": created_at,
        "created_by": created_by,
        "content_sha256": content_sha256,
        "byte_length": int(len(png_bytes)),
    }
    if artifact_signature is not None:
        attrs["artifact_signature"] = str(artifact_signature)
    if source_paths is not None:
        attrs["source_paths"] = _json_safe(source_paths)
    if source_runs is not None:
        attrs["source_runs"] = _json_safe(source_runs)
    if parameters is not None:
        attrs["parameters"] = _json_safe(parameters)
    if extra_attrs:
        attrs.update(_json_safe(dict(extra_attrs)))
    dataset.attrs.update(attrs)

    path = f"visualizations/{artifact_name}"
    _update_visualization_manifest(
        run_group,
        artifact_name,
        {
            "path": path,
            "description": description,
            "artifact_schema_id": PNG_ARTIFACT_SCHEMA_ID,
            "artifact_type": "visualization",
            "artifact_role": role,
            "media_type": "image/png",
            "artifact_signature": artifact_signature,
            "created_at_utc": created_at,
            "created_by": created_by,
            "content_sha256": content_sha256,
            "byte_length": int(len(png_bytes)),
        },
    )
    return PlotArtifactResult(
        artifact_name=artifact_name,
        path=path,
        content_sha256=content_sha256,
        byte_length=int(len(png_bytes)),
        artifact_schema_id=PNG_ARTIFACT_SCHEMA_ID,
        created_at_utc=created_at,
    )


def write_interactive_plot_spec_artifact(
    run_group: zarr.Group,
    artifact_name: str,
    spec: Mapping[str, Any],
    *,
    description: str,
    created_by: str,
    renderer: str,
    artifact_signature: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    snapshot_artifact: Optional[str] = None,
    source_paths: Optional[Mapping[str, Any]] = None,
    source_runs: Optional[Mapping[str, Any]] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    extra_attrs: Optional[Mapping[str, Any]] = None,
    overwrite: bool = True,
) -> PlotArtifactResult:
    """Write a renderer-neutral interactive plot spec artifact.

    The spec should be small JSON that declares how a UI/notebook can read
    canonical source arrays and render hover, zoom, filtering, or overlays.
    Large numeric series should remain in their canonical analysis arrays; if a
    future plot needs decimated arrays, store them as explicit children of this
    artifact group with their own lineage attrs.
    """

    if not artifact_name:
        raise ValueError("artifact_name must not be empty")
    if not renderer:
        raise ValueError("renderer must not be empty")

    vis_group = _require_visualizations_group(run_group)
    if artifact_name in vis_group:
        if not overwrite:
            raise ValueError(f"visualization artifact already exists: {artifact_name}")
        del vis_group[artifact_name]

    created_at = created_at_utc or utc_now()
    safe_spec = _json_safe(spec)
    spec_bytes = strict_json_dumps(safe_spec).encode("utf-8")
    content_sha256 = _sha256_bytes(spec_bytes)

    artifact_group = vis_group.create_group(artifact_name)
    spec_data = np.frombuffer(spec_bytes, dtype=np.uint8)
    chunk = max(1, min(int(spec_data.shape[0]), DEFAULT_PNG_CHUNK_BYTES))
    spec_array = artifact_group.create_array(
        "spec_json",
        data=spec_data,
        chunks=(chunk,),
        overwrite=True,
    )
    spec_array.attrs.update(
        {
            "media_type": "application/json",
            "storage_encoding": "utf8_json_bytes_uint8",
            "content_sha256": content_sha256,
            "byte_length": int(len(spec_bytes)),
        }
    )

    attrs: dict[str, Any] = {
        "artifact_schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
        "artifact_type": "visualization",
        "artifact_role": "interactive_spec",
        "media_type": SPEC_MEDIA_TYPE,
        "description": description,
        "created_at_utc": created_at,
        "created_by": created_by,
        "renderer": renderer,
        "spec_path": "spec_json",
        "content_sha256": content_sha256,
        "byte_length": int(len(spec_bytes)),
    }
    if artifact_signature is not None:
        attrs["artifact_signature"] = str(artifact_signature)
    if snapshot_artifact is not None:
        attrs["snapshot_artifact"] = str(snapshot_artifact)
    if source_paths is not None:
        attrs["source_paths"] = _json_safe(source_paths)
    if source_runs is not None:
        attrs["source_runs"] = _json_safe(source_runs)
    if parameters is not None:
        attrs["parameters"] = _json_safe(parameters)
    if extra_attrs:
        attrs.update(_json_safe(dict(extra_attrs)))
    artifact_group.attrs.update(attrs)

    path = f"visualizations/{artifact_name}"
    _update_visualization_manifest(
        run_group,
        artifact_name,
        {
            "path": path,
            "description": description,
            "artifact_schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
            "artifact_type": "visualization",
            "artifact_role": "interactive_spec",
            "media_type": SPEC_MEDIA_TYPE,
            "renderer": renderer,
            "artifact_signature": artifact_signature,
            "snapshot_artifact": snapshot_artifact,
            "created_at_utc": created_at,
            "created_by": created_by,
            "content_sha256": content_sha256,
            "byte_length": int(len(spec_bytes)),
        },
    )
    return PlotArtifactResult(
        artifact_name=artifact_name,
        path=path,
        content_sha256=content_sha256,
        byte_length=int(len(spec_bytes)),
        artifact_schema_id=INTERACTIVE_SPEC_SCHEMA_ID,
        created_at_utc=created_at,
    )
