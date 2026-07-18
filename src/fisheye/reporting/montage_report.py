"""Publish completed semantic montages as immutable analytics report artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

from PIL import Image

from fisheye.shared.batch_logging import utc_now_z
from fisheye.shared.json_safety import json_attr_safe

from .catalog import VISUALIZATIONS
from .export import REPORT_EXPORT_SCHEMA_ID, report_manifest_sha256
from .montage import SEMANTIC_MONTAGE_SCHEMA_ID


SEMANTIC_MONTAGE_ARTIFACT_CONTRACT_ID = (
    "palette.reporting.semantic_visualization_montage.png.v1"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def publish_semantic_montage_report(
    *,
    semantic_manifest_path: Path,
    output_dir: Path,
    report_id: str,
    analytics_export: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy one verified montage set into a canonical immutable report bundle."""

    safe_report_id = str(report_id).strip()
    if not _SAFE_ID.fullmatch(safe_report_id):
        raise ValueError(
            "report_id must begin with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    binding = dict(analytics_export)
    export_run_id = str(binding.get("export_run_id", ""))
    if not export_run_id:
        raise ValueError("analytics_export.export_run_id is required")

    source_manifest = Path(semantic_manifest_path).expanduser().resolve()
    semantic = _load_json_object(source_manifest)
    if semantic.get("schema_id") != SEMANTIC_MONTAGE_SCHEMA_ID:
        raise ValueError(
            f"{source_manifest}: expected schema_id={SEMANTIC_MONTAGE_SCHEMA_ID!r}"
        )
    if int(semantic.get("schema_version", 0)) != 1:
        raise ValueError(f"{source_manifest}: unsupported semantic montage schema version")
    outputs = semantic.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"{source_manifest}: semantic montage outputs are required")

    source_dir = source_manifest.parent.resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Immutable report output already exists: {output_dir}. Choose a new report_id."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    artifact_entries: list[dict[str, Any]] = []
    try:
        for output in outputs:
            if not isinstance(output, Mapping):
                raise ValueError(f"{source_manifest}: montage output is not an object")
            visualization_id = str(output.get("visualization_id", ""))
            visualization = VISUALIZATIONS.get(visualization_id)
            if visualization is None:
                raise ValueError(
                    f"{source_manifest}: unknown visualization_id={visualization_id!r}"
                )
            source_path = Path(str(output.get("path", ""))).expanduser().resolve()
            if not _within(source_path, source_dir) or source_path.parent != source_dir:
                raise PermissionError(
                    f"Montage artifact resolves outside its manifest directory: {source_path}"
                )
            if not source_path.is_file():
                raise FileNotFoundError(f"Montage artifact is missing: {source_path}")
            with Image.open(source_path) as image:
                if image.format != "PNG":
                    raise ValueError(f"Montage artifact is not PNG: {source_path}")
                actual_width, actual_height = image.size
                image.verify()
            for key, actual in (("width_px", actual_width), ("height_px", actual_height)):
                declared = output.get(key)
                if declared is not None and int(declared) != int(actual):
                    raise ValueError(
                        f"{source_path}: declared {key}={declared}, actual={actual}"
                    )

            relative_path = Path("artifacts") / source_path.name
            destination = temporary_dir / relative_path
            if destination.exists():
                raise ValueError(f"Montage artifact path collision: {relative_path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            source_sha256 = _file_sha256(source_path)
            copied_sha256 = _file_sha256(destination)
            if copied_sha256 != source_sha256:
                raise ValueError(f"Copied montage hash mismatch: {source_path}")
            artifact_entries.append(
                {
                    "visualization_id": visualization_id,
                    "provider_id": visualization.provider_id,
                    "label": visualization.label,
                    "entity_scope": "cohort",
                    "entity_id": binding.get("collection_id") or safe_report_id,
                    "artifact_role": "cohort_montage",
                    "media_type": "image/png",
                    "visualization_contract_id": SEMANTIC_MONTAGE_ARTIFACT_CONTRACT_ID,
                    "source_visualization_contract_id": output.get(
                        "visualization_contract_id"
                    ),
                    "renderer": "fisheye.reporting.montage.compose_visualization_montage",
                    "renderer_version": "1",
                    "source_backend": "zarr",
                    "source": {
                        "semantic_manifest_path": str(source_manifest),
                        "semantic_manifest_sha256": _file_sha256(source_manifest),
                        "tile_count": int(output.get("tile_count", 0)),
                    },
                    "materialization_policy": "copy",
                    "materialized": {
                        "relative_path": relative_path.as_posix(),
                        "content_sha256": copied_sha256,
                        "byte_length": destination.stat().st_size,
                        "width_px": actual_width,
                        "height_px": actual_height,
                    },
                }
            )

        nonready = [
            dict(item)
            for item in semantic.get("nonready", [])
            if isinstance(item, Mapping)
        ]
        manifest: dict[str, Any] = {
            "schema_id": REPORT_EXPORT_SCHEMA_ID,
            "schema_version": 1,
            "created_at_utc": utc_now_z(),
            "report_id": safe_report_id,
            "analytics_export": binding,
            "materialization_policy": "copy",
            "source_backends": ["zarr"],
            "source_tables": [],
            "source_report_plan_sha256": semantic.get("source_report_plan_sha256"),
            "semantic_montage_source": {
                "schema_id": semantic.get("schema_id"),
                "schema_version": semantic.get("schema_version"),
                "manifest_path": str(source_manifest),
                "manifest_sha256": _file_sha256(source_manifest),
            },
            "visualization_ids": [
                str(item["visualization_id"]) for item in artifact_entries
            ],
            "artifact_count": len(artifact_entries),
            "nonready_count": len(nonready),
            "nonready": nonready,
            "artifacts": artifact_entries,
            "manifest_relative_path": "report_manifest.json",
        }
        manifest["manifest_sha256"] = report_manifest_sha256(manifest)
        (temporary_dir / "report_manifest.json").write_text(
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


__all__ = [
    "SEMANTIC_MONTAGE_ARTIFACT_CONTRACT_ID",
    "publish_semantic_montage_report",
]
