"""Compose cohort montages from semantic visualization plan items."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from fisheye.montage.models import MontageArtifactSpec, MontageLayout, RegistryRecording
from fisheye.montage.render import compose_visualization_montage, load_recording_tiles
from fisheye.shared.json_safety import json_attr_safe

from .catalog import VISUALIZATIONS
from .manifest import report_plan_sha256
from .models import PlanStatus, ReportPlan, VisualizationPlanItem


SEMANTIC_MONTAGE_SCHEMA_ID = "palette.semantic_visualization_montages.v1"


def _output_name(visualization_id: str) -> str:
    return visualization_id.replace(".", "-") + "_montage.png"


def _recording_label_id(recording_id: str, item: VisualizationPlanItem | None) -> str:
    if item is None or item.entity_id is None:
        return recording_id
    return f"{recording_id} [{item.entity_scope} {item.entity_id}]"


def _query_label(plan: ReportPlan) -> str:
    terms: list[str] = []
    for key in ("protocol_name", "recording_id_contains", "path_contains"):
        value = plan.query.get(key)
        if value:
            terms.append(f"{key}={value}")
    return " | ".join(terms) or "registry report selection"


def _items_for_recording(
    recording_plan: Any,
    visualization_id: str,
) -> tuple[VisualizationPlanItem | None, ...]:
    matches = tuple(
        item
        for item in recording_plan.items
        if item.visualization_id == visualization_id
    )
    return matches or (None,)


def _bound_loaded_image(
    image: Image.Image | None,
    *,
    tile_width: int,
    max_image_height: int,
) -> Image.Image | None:
    """Downsample one loaded tile before retaining it for cohort composition."""

    if image is None:
        return None
    if image.width > tile_width or image.height > max_image_height:
        image.thumbnail(
            (tile_width, max_image_height),
            Image.Resampling.LANCZOS,
        )
    return image


def build_semantic_visualization_montages(
    *,
    plan: ReportPlan,
    output_dir: Path,
    visualization_ids: Sequence[str],
    columns: int = 4,
    tile_width: int = 600,
    max_image_height: int = 480,
    fail_on_nonready: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one contract-safe montage per semantic visualization ID."""

    if not visualization_ids:
        raise ValueError("Select at least one semantic visualization ID.")
    unknown = sorted(set(visualization_ids) - set(VISUALIZATIONS))
    if unknown:
        raise ValueError(f"Unknown visualization ID(s): {', '.join(unknown)}")
    if columns < 1 or tile_width < 160 or max_image_height < 120:
        raise ValueError("columns >= 1, tile_width >= 160, and max_image_height >= 120")

    output_dir = Path(output_dir).expanduser().resolve()
    expected_outputs = [output_dir / _output_name(value) for value in visualization_ids]
    expected_outputs.append(output_dir / "semantic_montage_manifest.json")
    if not overwrite:
        existing = [path for path in expected_outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing[0]}")

    outputs: list[dict[str, Any]] = []
    all_nonready: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for visualization_id in visualization_ids:
        visualization = VISUALIZATIONS[visualization_id]
        tile_recordings: list[RegistryRecording] = []
        tile_images = []
        tile_errors: list[str | None] = []
        tile_manifest: list[dict[str, Any]] = []

        for recording_plan in plan.recordings:
            for item in _items_for_recording(recording_plan, visualization_id):
                recording = recording_plan.recording
                tile_recording = RegistryRecording(
                    recording_id=_recording_label_id(recording.recording_id, item),
                    zarr_path=Path(recording.zarr_path),
                    dataset_id=recording.dataset_id,
                    protocol_name=recording.protocol_name,
                    arena_id=recording.arena_id,
                    recording_started_utc=recording.recording_started_utc,
                )
                tile_recordings.append(tile_recording)
                ready = (
                    item is not None
                    and item.status == PlanStatus.READY.value
                    and item.artifact is not None
                )
                if not ready:
                    status = item.status if item is not None else PlanStatus.NOT_APPLICABLE.value
                    reason = item.reason if item is not None else "visualization was not planned"
                    nonready = {
                        "recording_id": recording.recording_id,
                        "visualization_id": visualization_id,
                        "entity_id": item.entity_id if item is not None else None,
                        "status": status,
                        "reason": reason,
                    }
                    all_nonready.append(nonready)
                    if fail_on_nonready:
                        raise ValueError(
                            f"{recording.recording_id}: {visualization_id} is {status}: {reason}"
                        )
                    tile_images.append(None)
                    tile_errors.append(f"{status}: {reason}")
                    tile_manifest.append(nonready)
                    continue

                spec = MontageArtifactSpec(
                    artifact_id=visualization_id,
                    label=visualization.label,
                    path=item.artifact.path,
                    visualization_contract_id=item.expected_visualization_contract_id,
                )
                loaded, missing = load_recording_tiles(
                    tile_recording,
                    (spec,),
                    fail_on_missing=fail_on_nonready,
                )
                tile = loaded[0]
                tile_images.append(
                    _bound_loaded_image(
                        tile.image,
                        tile_width=tile_width,
                        max_image_height=max_image_height,
                    )
                )
                tile_errors.append(tile.error)
                tile_manifest.append(
                    {
                        "recording_id": recording.recording_id,
                        "entity_id": item.entity_id,
                        "status": item.status,
                        "source_run": item.source_run.path if item.source_run else None,
                        "artifact_path": item.artifact.path,
                        "content_sha256": item.artifact.content_sha256,
                    }
                )
                all_nonready.extend(missing)

        layout = MontageLayout(
            columns=columns,
            tile_width=tile_width,
            max_image_height=max_image_height,
        )
        montage = compose_visualization_montage(
            title=visualization.label,
            query_label=_query_label(plan),
            recordings=tile_recordings,
            images=tile_images,
            errors=tile_errors,
            layout=layout,
        )
        output_path = output_dir / _output_name(visualization_id)
        temporary = output_path.with_suffix(".png.tmp")
        montage.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, output_path)
        outputs.append(
            {
                "visualization_id": visualization_id,
                "label": visualization.label,
                "visualization_contract_id": visualization.visualization_contract_id,
                "path": str(output_path),
                "tile_count": len(tile_recordings),
                "columns": columns,
                "rows": math.ceil(len(tile_recordings) / columns),
                "width_px": montage.width,
                "height_px": montage.height,
                "tiles": tile_manifest,
            }
        )

    manifest: dict[str, Any] = {
        "schema_id": SEMANTIC_MONTAGE_SCHEMA_ID,
        "schema_version": 1,
        "source_report_plan_sha256": report_plan_sha256(plan),
        "registry_path": plan.registry_path,
        "query": dict(plan.query),
        "visualization_ids": list(visualization_ids),
        "nonready_count": len(all_nonready),
        "nonready": all_nonready,
        "outputs": outputs,
    }
    manifest_path = output_dir / "semantic_montage_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(json_attr_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
