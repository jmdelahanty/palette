"""Immutable reference or portable-copy exports for planned dataset reports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from fisheye.shared.batch_logging import utc_now_z
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes

from .manifest import report_plan_sha256, report_plan_to_dict
from .models import PlanStatus, ReportPlan, VisualizationPlanItem


REPORT_EXPORT_SCHEMA_ID = "palette.dataset_report_export.v1"
MATERIALIZATION_POLICIES = ("reference", "copy")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_attr_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return result or "unnamed"


def _entity_filename(item: VisualizationPlanItem) -> str:
    entity = item.entity_id if item.entity_id is not None else "recording"
    return f"{_slug(item.entity_scope)}-{_slug(entity)}.png"


def _selected_items(
    plan: ReportPlan,
    visualization_ids: Sequence[str],
) -> list[tuple[Any, VisualizationPlanItem]]:
    requested = set(visualization_ids)
    return [
        (recording_plan, item)
        for recording_plan in plan.recordings
        for item in recording_plan.items
        if not requested or item.visualization_id in requested
    ]


def _source_collection_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "content_sha256": _sha256_bytes(payload),
        "byte_length": len(payload),
    }


def export_report_bundle(
    *,
    plan: ReportPlan,
    output_dir: Path,
    materialization_policy: str,
    visualization_ids: Sequence[str] = (),
    fail_on_nonready: bool = True,
    source_collection_manifest: Path | None = None,
    report_id: str | None = None,
    analytics_export: Mapping[str, Any] | None = None,
    source_tables: Sequence[str] = (),
) -> dict[str, Any]:
    """Create an immutable report directory and return its manifest payload."""

    policy = str(materialization_policy).strip().lower()
    if policy not in MATERIALIZATION_POLICIES:
        raise ValueError(f"materialization_policy must be one of {MATERIALIZATION_POLICIES}")
    if report_id is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", str(report_id)
    ):
        raise ValueError("Invalid report_id")
    if analytics_export is not None and report_id is None:
        raise ValueError("report_id is required when analytics_export is provided")
    bound_export = dict(analytics_export) if analytics_export is not None else None
    selected_source_tables = sorted({str(value) for value in source_tables})
    if bound_export is None and selected_source_tables:
        raise ValueError("source_tables require an analytics_export binding")
    if bound_export is not None:
        available_tables = {
            str(value) for value in bound_export.get("available_tables", [])
        }
        unknown_tables = sorted(set(selected_source_tables) - available_tables)
        if unknown_tables:
            raise ValueError(
                "Source tables are not present in the bound analytics export: "
                + ", ".join(unknown_tables)
            )
    unknown = sorted(set(visualization_ids) - {
        item.visualization_id
        for recording_plan in plan.recordings
        for item in recording_plan.items
    })
    if unknown:
        raise ValueError(f"Visualization IDs were not planned: {', '.join(unknown)}")

    selected = _selected_items(plan, visualization_ids)
    if not selected:
        raise ValueError("The report selection contains no visualization items.")
    nonready = [
        {
            "dataset_id": recording_plan.recording.dataset_id,
            "recording_id": recording_plan.recording.recording_id,
            "visualization_id": item.visualization_id,
            "entity_scope": item.entity_scope,
            "entity_id": item.entity_id,
            "status": item.status,
            "reason": item.reason,
        }
        for recording_plan, item in selected
        if item.status != PlanStatus.READY.value or item.artifact is None
    ]
    if fail_on_nonready and nonready:
        first = nonready[0]
        raise ValueError(
            f"{first['recording_id']}: {first['visualization_id']} is "
            f"{first['status']}: {first['reason']}"
        )

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Immutable report output already exists: {output_dir}. Choose a new directory."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    artifact_entries: list[dict[str, Any]] = []
    roots: dict[str, Any] = {}
    try:
        for recording_plan, item in selected:
            if item.status != PlanStatus.READY.value or item.artifact is None:
                continue
            recording = recording_plan.recording
            entry: dict[str, Any] = {
                "dataset_id": recording.dataset_id,
                "recording_id": recording.recording_id,
                "visualization_id": item.visualization_id,
                "provider_id": item.provider_id,
                "label": item.label,
                "entity_scope": item.entity_scope,
                "entity_id": item.entity_id,
                "visualization_contract_id": item.artifact.visualization_contract_id,
                "renderer": item.artifact.renderer,
                "renderer_version": item.artifact.renderer_version,
                "artifact_signature": item.artifact.artifact_signature,
                "source": {
                    "zarr_path": recording.zarr_path,
                    "artifact_path": item.artifact.path,
                    "source_run": item.source_run.path if item.source_run else None,
                    "declared_content_sha256": item.artifact.content_sha256,
                },
                "materialization_policy": policy,
                "source_backend": "zarr",
            }
            if policy == "copy":
                root = roots.get(recording.zarr_path)
                if root is None:
                    root = open_zarr_root(Path(recording.zarr_path), mode="r")
                    roots[recording.zarr_path] = root
                resolved_path, png_bytes = load_png_artifact_bytes(
                    root,
                    item.artifact.path,
                )
                actual_sha256 = _sha256_bytes(png_bytes)
                declared = item.artifact.content_sha256
                if declared is not None and declared != actual_sha256:
                    raise ValueError(
                        f"Artifact hash mismatch at {recording.zarr_path}:{resolved_path}: "
                        f"declared {declared}, calculated {actual_sha256}"
                    )
                relative_path = (
                    Path("artifacts")
                    / _slug(recording.dataset_id)
                    / _slug(recording.recording_id)
                    / _slug(item.visualization_id)
                    / _entity_filename(item)
                )
                destination = temporary_dir / relative_path
                if destination.exists():
                    raise ValueError(f"Portable artifact path collision: {relative_path}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".png.tmp")
                temporary.write_bytes(png_bytes)
                os.replace(temporary, destination)
                entry["materialized"] = {
                    "relative_path": relative_path.as_posix(),
                    "content_sha256": actual_sha256,
                    "byte_length": len(png_bytes),
                }
            else:
                entry["materialized"] = {
                    "zarr_path": recording.zarr_path,
                    "artifact_path": item.artifact.path,
                    "content_sha256": item.artifact.content_sha256,
                }
            artifact_entries.append(entry)

        manifest: dict[str, Any] = {
            "schema_id": REPORT_EXPORT_SCHEMA_ID,
            "schema_version": 1,
            "created_at_utc": utc_now_z(),
            "report_id": report_id,
            "analytics_export": bound_export,
            "materialization_policy": policy,
            "source_backends": sorted(
                {"zarr"} | ({"parquet"} if selected_source_tables else set())
            ),
            "source_tables": selected_source_tables,
            "source_report_plan_sha256": report_plan_sha256(plan),
            "source_collection_manifest": _source_collection_reference(
                source_collection_manifest
            ),
            "visualization_ids": sorted(
                set(visualization_ids)
                or {item.visualization_id for _, item in selected}
            ),
            "artifact_count": len(artifact_entries),
            "nonready_count": len(nonready),
            "nonready": nonready,
            "artifacts": artifact_entries,
            "report_plan": report_plan_to_dict(plan),
            "manifest_relative_path": "report_manifest.json",
        }
        manifest["manifest_sha256"] = report_manifest_sha256(manifest)
        manifest_path = temporary_dir / "report_manifest.json"
        manifest_path.write_text(
            json.dumps(json_attr_safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    result = dict(manifest)
    result["manifest_path"] = str(output_dir / "report_manifest.json")
    return result


def verify_report_manifest_sha256(manifest: dict[str, Any]) -> bool:
    """Verify the embedded content hash of a loaded report manifest."""

    expected = manifest.get("manifest_sha256")
    return isinstance(expected, str) and report_manifest_sha256(manifest) == expected


def report_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical digest for a report manifest without its digest field."""

    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return _sha256_bytes(_canonical_bytes(unsigned))


__all__ = [
    "MATERIALIZATION_POLICIES",
    "REPORT_EXPORT_SCHEMA_ID",
    "export_report_bundle",
    "report_manifest_sha256",
    "verify_report_manifest_sha256",
]
