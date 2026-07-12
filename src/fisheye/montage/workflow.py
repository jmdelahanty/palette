"""Orchestrate registry selection, artifact loading, rendering, and provenance."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from fisheye.shared.batch_logging import utc_now_z
from fisheye.shared.json_safety import json_attr_safe

from .models import MontageLayout
from .profiles import PLOT_PROFILES
from .registry import query_registry_recordings
from .render import compose_visualization_montage, load_recording_tiles


SCHEMA_ID = "palette.registry_visualization_montages.v1"


def build_registry_visualization_montages(
    *,
    registry_path: Path,
    output_dir: Path,
    plot_types: Sequence[str],
    protocol_name: str | None = None,
    recording_ids: Sequence[str] = (),
    recording_id_contains: str | None = None,
    path_contains: str | None = None,
    arena_ids: Sequence[str] = (),
    chaser_behaviors: Sequence[str] = (),
    chaser_count: int | None = None,
    zarr_use: str = "analysis",
    status: str = "active",
    limit: int | None = None,
    all_recordings: bool = False,
    chaser_distance_run: str | None = None,
    detection_occupancy_run: str | None = None,
    egocentric_component: str | None = None,
    escape_freeze_component: str | None = None,
    columns: int = 4,
    tile_width: int = 600,
    max_image_height: int | None = None,
    fail_on_missing: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not plot_types:
        raise ValueError("Select at least one --plot-type.")
    if len(set(plot_types)) != len(plot_types):
        raise ValueError("Each --plot-type may be selected only once.")
    unknown = sorted(set(plot_types) - set(PLOT_PROFILES))
    if unknown:
        raise ValueError(f"Unknown plot type(s): {', '.join(unknown)}")
    if columns < 1 or tile_width < 160:
        raise ValueError("columns must be >= 1 and tile_width must be >= 160")
    if max_image_height is not None and max_image_height < 120:
        raise ValueError("max_image_height must be >= 120")

    registry_path = Path(registry_path).expanduser().resolve(strict=True)
    recordings = query_registry_recordings(
        registry_path,
        protocol_name=protocol_name,
        recording_ids=recording_ids,
        recording_id_contains=recording_id_contains,
        path_contains=path_contains,
        arena_ids=arena_ids,
        chaser_behaviors=chaser_behaviors,
        chaser_count=chaser_count,
        zarr_use=zarr_use,
        status=status,
        limit=limit,
        all_recordings=all_recordings,
    )
    if not recordings:
        raise ValueError("Registry query selected no recordings.")

    run_parameters = {
        "chaser_distance_run": chaser_distance_run,
        "detection_occupancy_run": detection_occupancy_run,
        "egocentric_component": egocentric_component,
        "escape_freeze_component": escape_freeze_component,
    }
    profiles = [PLOT_PROFILES[profile_id] for profile_id in plot_types]
    specs = [profile.artifact_spec(run_parameters) for profile in profiles]
    query = {
        "protocol_name": protocol_name,
        "recording_ids": list(recording_ids),
        "recording_id_contains": recording_id_contains,
        "path_contains": path_contains,
        "arena_ids": list(arena_ids),
        "chaser_behaviors": list(chaser_behaviors),
        "chaser_count": chaser_count,
        "zarr_use": zarr_use,
        "status": status,
        "limit": limit,
        "all_recordings": all_recordings,
        "ordering": ["recording_started_utc_or_recording_id", "arena_id", "recording_id"],
    }
    query_terms = [f"protocol={protocol_name}"] if protocol_name else []
    if chaser_behaviors:
        query_terms.append(f"chaser_behavior={','.join(chaser_behaviors)}")
    if chaser_count is not None:
        query_terms.append(f"chaser_count={chaser_count}")
    query_label = " | ".join(query_terms) or "registry selection"

    output_dir = Path(output_dir).expanduser().resolve()
    if not dry_run and not overwrite:
        expected_outputs = [
            output_dir / f"{profile.profile_id}_montage.png" for profile in profiles
        ] + [output_dir / "montage_manifest.json"]
        existing = [path for path in expected_outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing[0]}")

    images_by_profile: dict[str, list[Image.Image | None]] = {
        profile.profile_id: [] for profile in profiles
    }
    errors_by_profile: dict[str, list[str | None]] = {
        profile.profile_id: [] for profile in profiles
    }
    missing: list[dict[str, str]] = []
    for recording in recordings:
        tiles, recording_missing = load_recording_tiles(
            recording,
            specs,
            fail_on_missing=fail_on_missing,
        )
        missing.extend(recording_missing)
        for profile, tile in zip(profiles, tiles, strict=True):
            images_by_profile[profile.profile_id].append(tile.image)
            errors_by_profile[profile.profile_id].append(tile.error)

    output_files: list[dict[str, Any]] = []
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for profile, spec in zip(profiles, specs, strict=True):
            output_path = output_dir / f"{profile.profile_id}_montage.png"
            layout = MontageLayout(
                columns=columns,
                tile_width=tile_width,
                max_image_height=(
                    max_image_height
                    if max_image_height is not None
                    else profile.default_max_image_height
                ),
            )
            montage = compose_visualization_montage(
                title=profile.title,
                query_label=query_label,
                recordings=recordings,
                images=images_by_profile[profile.profile_id],
                errors=errors_by_profile[profile.profile_id],
                layout=layout,
            )
            tmp_path = output_path.with_suffix(".png.tmp")
            montage.save(tmp_path, format="PNG", optimize=True)
            os.replace(tmp_path, output_path)
            output_files.append(
                {
                    "plot_type": profile.profile_id,
                    "artifact_path": spec.path,
                    "visualization_contract_id": spec.visualization_contract_id,
                    "path": str(output_path),
                    "width_px": montage.width,
                    "height_px": montage.height,
                    "columns": columns,
                    "rows": math.ceil(len(recordings) / columns),
                }
            )

    manifest: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "created_at_utc": utc_now_z(),
        "tool": "fisheye.montage",
        "dry_run": dry_run,
        "registry_path": str(registry_path),
        "query": query,
        "run_parameters": run_parameters,
        "plot_types": list(plot_types),
        "recording_count": len(recordings),
        "recordings": [
            {
                "dataset_id": recording.dataset_id,
                "recording_id": recording.recording_id,
                "zarr_path": str(recording.zarr_path),
                "protocol_name": recording.protocol_name,
                "arena_id": recording.arena_id,
                "recording_started_utc": recording.recording_started_utc,
                "chaser_behaviors": list(recording.chaser_behaviors),
                "chaser_count": recording.chaser_count,
            }
            for recording in recordings
        ],
        "missing_artifact_count": len(missing),
        "missing_artifacts": missing,
        "outputs": output_files,
    }
    if not dry_run:
        manifest_path = output_dir / "montage_manifest.json"
        tmp_manifest_path = manifest_path.with_suffix(".json.tmp")
        tmp_manifest_path.write_text(
            json.dumps(json_attr_safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_manifest_path, manifest_path)
        manifest["manifest_path"] = str(manifest_path)
    return manifest
