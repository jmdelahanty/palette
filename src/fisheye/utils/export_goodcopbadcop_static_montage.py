"""Build scrollable static-plot montage artifacts for GoodCopBadCop exports."""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now_z as _utc_now
import argparse
import json
import math
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.system_metadata import get_git_info
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
from fisheye.utils.zarr_io import open_zarr_root


SCHEMA_ID = "palette.goodcopbadcop.static_montage.v1"
DEFAULT_EXPORT_ROOT = Path("/nvme1/exports/palette_analytics")
DEFAULT_ARTIFACT_SET_ID = "goodcopbadcop_static_plots_v1"
DEFAULT_CHASER_DISTANCE_RUN = "goodcopbadcop_chaser_distance_v1_20260617"
DEFAULT_DETECTION_OCCUPANCY_RUN = "goodcopbadcop_detection_occupancy_v1_20260617"
DEFAULT_TRACK_KINEMATICS_RUN = "goodcopbadcop_tk_hyst4_low2_latch_s005"
DEFAULT_EGOCENTRIC_COMPONENT = "track_offline_goodcopbadcop_tk_hyst4_low2_latch_s005_id_0_smoothed"


@dataclass(frozen=True)
class MontageArtifactSpec:
    artifact_id: str
    label: str
    path: str


@dataclass(frozen=True)
class SourceRecording:
    recording_id: str
    zarr_path: Path


@dataclass(frozen=True)
class LoadedTile:
    artifact_id: str
    label: str
    path: str
    image: Image.Image | None
    error: str | None


def _normalize_path(path: str) -> str:
    return "/".join(part for part in str(path).strip("/").split("/") if part)


def _join_path(*parts: str) -> str:
    return "/".join(_normalize_path(part) for part in parts if _normalize_path(part))


def _recording_id_from_zarr_path(zarr_path: Path) -> str:
    name = zarr_path.name
    if name.endswith(".zarr"):
        name = name[:-5]
    if name.endswith("_analysis"):
        name = name[:-9]
    return name


def _manifest_path(export_root: Path, export_run_id: str) -> Path:
    return export_root / "v1" / "manifests" / f"export_run_id={export_run_id}.json"


def _resolve_export_run_id(export_root: Path, export_run_id: str) -> str:
    if export_run_id != "latest":
        return export_run_id
    manifest_dir = export_root / "v1" / "manifests"
    manifests = sorted(manifest_dir.glob("export_run_id=*.json"))
    if not manifests:
        raise FileNotFoundError(f"No export manifests found under {manifest_dir}")
    return manifests[-1].name.removeprefix("export_run_id=").removesuffix(".json")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return payload


def _collection_manifest_path_from_export(export_manifest: Mapping[str, Any]) -> Path | None:
    collection = export_manifest.get("collection_manifest")
    if not isinstance(collection, Mapping):
        return None
    path = collection.get("path")
    if not isinstance(path, str) or not path:
        return None
    return Path(path).expanduser().resolve()


def _sources_from_collection_manifest(path: Path) -> list[SourceRecording]:
    manifest = _read_json(path)
    sources: list[SourceRecording] = []
    records = manifest.get("records")
    if not isinstance(records, list):
        return sources
    for record in records:
        if not isinstance(record, Mapping):
            continue
        status = record.get("status")
        if not isinstance(status, Mapping) or status.get("included") is not True:
            continue
        locator = record.get("locator_at_selection")
        if not isinstance(locator, Mapping):
            continue
        uri = locator.get("uri")
        if not isinstance(uri, str) or not uri:
            continue
        zarr_path = Path(uri).expanduser().resolve()
        recording_id = record.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            attrs = record.get("recording_attrs")
            if isinstance(attrs, Mapping) and isinstance(attrs.get("recording_id"), str):
                recording_id = str(attrs["recording_id"])
            else:
                recording_id = _recording_id_from_zarr_path(zarr_path)
        sources.append(SourceRecording(recording_id=recording_id, zarr_path=zarr_path))
    return sources


def _sources_from_export_manifest(export_manifest: Mapping[str, Any]) -> list[SourceRecording]:
    raw_sources = export_manifest.get("source_zarrs")
    if not isinstance(raw_sources, list):
        return []
    sources: list[SourceRecording] = []
    for raw_path in raw_sources:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        zarr_path = Path(raw_path).expanduser().resolve()
        sources.append(SourceRecording(recording_id=_recording_id_from_zarr_path(zarr_path), zarr_path=zarr_path))
    return sources


def default_goodcopbadcop_artifact_specs(
    *,
    chaser_distance_run: str = DEFAULT_CHASER_DISTANCE_RUN,
    detection_occupancy_run: str = DEFAULT_DETECTION_OCCUPANCY_RUN,
    track_kinematics_run: str = DEFAULT_TRACK_KINEMATICS_RUN,
    egocentric_component: str = DEFAULT_EGOCENTRIC_COMPONENT,
) -> tuple[MontageArtifactSpec, ...]:
    chaser_root = _join_path("analysis/chaser_distance_runs", chaser_distance_run)
    return (
        MontageArtifactSpec(
            artifact_id="detection_occupancy_overview",
            label="Detection occupancy",
            path=_join_path(
                "analysis/detection_occupancy_runs",
                detection_occupancy_run,
                "visualizations/detection_occupancy_overview_png",
            ),
        ),
        MontageArtifactSpec(
            artifact_id="chaser_distance_timeseries",
            label="Chaser distance timeseries",
            path=_join_path(chaser_root, "visualizations/chaser_distance_timeseries_png"),
        ),
        MontageArtifactSpec(
            artifact_id="chaser_distance_epoch_median",
            label="Chaser distance medians",
            path=_join_path(chaser_root, "visualizations/chaser_distance_epoch_median_png"),
        ),
        MontageArtifactSpec(
            artifact_id="chaser_distance_epoch_distribution",
            label="Chaser distance distributions",
            path=_join_path(chaser_root, "visualizations/chaser_distance_epoch_distribution_png"),
        ),
        MontageArtifactSpec(
            artifact_id="cra_primary_endpoint_overview",
            label="CRA primary endpoint",
            path=_join_path(
                chaser_root,
                "cra_primary_endpoint/object_relative_pre_post_v1",
                "visualizations/cra_primary_endpoint_overview_png",
            ),
        ),
        MontageArtifactSpec(
            artifact_id="cra_near_field_summary",
            label="CRA near-field summary",
            path=_join_path(
                chaser_root,
                "cra_near_field/object_relative_near_field_v1",
                "visualizations/cra_near_field_summary_png",
            ),
        ),
        MontageArtifactSpec(
            artifact_id="cra_near_field_radial_density",
            label="Near-field radial density",
            path=_join_path(
                chaser_root,
                "cra_near_field/object_relative_near_field_v1",
                "visualizations/cra_near_field_radial_density_png",
            ),
        ),
        MontageArtifactSpec(
            artifact_id="cra_near_field_distance_cdf",
            label="Near-field distance CDF",
            path=_join_path(
                chaser_root,
                "cra_near_field/object_relative_near_field_v1",
                "visualizations/cra_near_field_distance_cdf_png",
            ),
        ),
        MontageArtifactSpec(
            artifact_id="egocentric_bearing_polar",
            label="Egocentric polar heatmap",
            path=_join_path(
                chaser_root,
                "egocentric_bearing",
                egocentric_component,
                "visualizations/egocentric_bearing_pre_post_polar_png",
            ),
        ),
        MontageArtifactSpec(
            artifact_id="egocentric_bearing_point_cloud",
            label="Egocentric point cloud",
            path=_join_path(
                chaser_root,
                "egocentric_bearing",
                egocentric_component,
                "visualizations/egocentric_bearing_pre_post_polar_point_cloud_png",
            ),
        ),
        MontageArtifactSpec(
            artifact_id="track_kinematics_summary",
            label="Track kinematics summary",
            path=_join_path(
                "analysis/track_kinematics_runs/offline",
                track_kinematics_run,
                "visualizations/track_kinematics_summary_track_0_png",
            ),
        ),
    )


def _load_recording_tiles(
    source: SourceRecording,
    specs: Sequence[MontageArtifactSpec],
    *,
    fail_on_missing: bool,
) -> tuple[list[LoadedTile], list[dict[str, str]]]:
    tiles: list[LoadedTile] = []
    missing: list[dict[str, str]] = []
    try:
        root = open_zarr_root(source.zarr_path, mode="r")
    except Exception as exc:
        if fail_on_missing:
            raise
        message = f"failed to open zarr: {type(exc).__name__}: {exc}"
        for spec in specs:
            tiles.append(
                LoadedTile(
                    artifact_id=spec.artifact_id,
                    label=spec.label,
                    path=spec.path,
                    image=None,
                    error=message,
                )
            )
            missing.append(
                {
                    "recording_id": source.recording_id,
                    "zarr_path": str(source.zarr_path),
                    "artifact_id": spec.artifact_id,
                    "artifact_path": spec.path,
                    "error": message,
                }
            )
        return tiles, missing
    for spec in specs:
        try:
            _, png_bytes = load_png_artifact_bytes(root, spec.path)
            image = Image.open(BytesIO(png_bytes)).convert("RGB")
            image.load()
            tiles.append(
                LoadedTile(
                    artifact_id=spec.artifact_id,
                    label=spec.label,
                    path=spec.path,
                    image=image,
                    error=None,
                )
            )
        except Exception as exc:
            if fail_on_missing:
                raise
            message = f"{type(exc).__name__}: {exc}"
            tiles.append(
                LoadedTile(
                    artifact_id=spec.artifact_id,
                    label=spec.label,
                    path=spec.path,
                    image=None,
                    error=message,
                )
            )
            missing.append(
                {
                    "recording_id": source.recording_id,
                    "zarr_path": str(source.zarr_path),
                    "artifact_id": spec.artifact_id,
                    "artifact_path": spec.path,
                    "error": message,
                }
            )
    return tiles, missing


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _ellipsize_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if _text_size(draw, text, font)[0] <= max_width:
        return text
    suffix = "..."
    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid] + suffix
        if _text_size(draw, candidate, font)[0] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


def _draw_placeholder(
    *,
    width: int,
    height: int,
    label: str,
    error: str | None,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    title_font = _font(16)
    body_font = _font(12)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(180, 180, 180), width=2)
    draw.line((0, 0, width, height), fill=(210, 210, 210), width=2)
    draw.line((0, height, width, 0), fill=(210, 210, 210), width=2)
    title = _ellipsize_text(draw, label, title_font, width - 32)
    title_w, title_h = _text_size(draw, title, title_font)
    draw.text(((width - title_w) // 2, height // 2 - title_h - 6), title, fill=(70, 70, 70), font=title_font)
    message = _ellipsize_text(draw, error or "missing artifact", body_font, width - 32)
    message_w, _ = _text_size(draw, message, body_font)
    draw.text(((width - message_w) // 2, height // 2 + 8), message, fill=(110, 45, 45), font=body_font)
    return image


def _resize_to_tile(image: Image.Image, *, tile_width: int, max_image_height: int) -> Image.Image:
    scale = tile_width / float(image.width)
    new_height = max(1, int(round(image.height * scale)))
    if new_height > max_image_height:
        scale = max_image_height / float(image.height)
        new_width = max(1, int(round(image.width * scale)))
        new_height = max_image_height
    else:
        new_width = tile_width
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_width, max_image_height), (255, 255, 255))
    x = (tile_width - resized.width) // 2
    canvas.paste(resized, (x, 0))
    return canvas


def _compose_recording_panel(
    *,
    source: SourceRecording,
    tiles: Sequence[LoadedTile],
    columns: int,
    tile_width: int,
    max_image_height: int,
    margin: int,
    gutter: int,
) -> Image.Image:
    title_font = _font(20)
    subtitle_font = _font(12)
    label_font = _font(13)
    label_height = 26
    header_height = 74
    tile_height = label_height + max_image_height
    rows = max(1, math.ceil(len(tiles) / columns))
    width = margin * 2 + columns * tile_width + (columns - 1) * gutter
    height = header_height + rows * tile_height + (rows - 1) * gutter + margin
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(220, 220, 220), width=1)
    draw.rectangle((0, 0, width, header_height - 1), fill=(245, 247, 250))
    title = _ellipsize_text(draw, source.recording_id, title_font, width - margin * 2)
    draw.text((margin, 14), title, fill=(25, 30, 38), font=title_font)
    subtitle = _ellipsize_text(draw, str(source.zarr_path), subtitle_font, width - margin * 2)
    draw.text((margin, 44), subtitle, fill=(90, 96, 106), font=subtitle_font)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        x = margin + col * (tile_width + gutter)
        y = header_height + row * (tile_height + gutter)
        label = _ellipsize_text(draw, tile.label, label_font, tile_width)
        draw.text((x, y), label, fill=(42, 47, 56), font=label_font)
        if tile.image is None:
            image = _draw_placeholder(
                width=tile_width,
                height=max_image_height,
                label=tile.label,
                error=tile.error,
            )
        else:
            image = _resize_to_tile(tile.image, tile_width=tile_width, max_image_height=max_image_height)
        panel.paste(image, (x, y + label_height))
    return panel


def _compose_page(
    *,
    export_run_id: str,
    page_index: int,
    page_count: int,
    page_sources: Sequence[SourceRecording],
    panels: Sequence[Image.Image],
    margin: int,
    gutter: int,
) -> Image.Image:
    if not panels:
        raise ValueError("cannot compose an empty montage page")
    title_font = _font(24)
    subtitle_font = _font(13)
    width = max(panel.width for panel in panels) + 2 * margin
    header_height = 82
    height = header_height + sum(panel.height for panel in panels) + gutter * (len(panels) - 1) + margin
    page = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(page)
    title = f"GoodCopBadCop static plot montage - page {page_index + 1} of {page_count}"
    draw.text((margin, 18), title, fill=(25, 30, 38), font=title_font)
    subtitle = f"{export_run_id} | recordings {page_sources[0].recording_id} through {page_sources[-1].recording_id}"
    subtitle = _ellipsize_text(draw, subtitle, subtitle_font, width - 2 * margin)
    draw.text((margin, 52), subtitle, fill=(82, 89, 100), font=subtitle_font)
    y = header_height
    for panel in panels:
        page.paste(panel, (margin, y))
        y += panel.height + gutter
    return page


def _write_html_index(
    *,
    output_path: Path,
    export_run_id: str,
    page_files: Sequence[Mapping[str, Any]],
    manifest_name: str,
) -> None:
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>GoodCopBadCop Static Montage</title>",
        "<style>",
        "body{margin:0;background:#f4f4f4;color:#222;font-family:Arial,sans-serif;}",
        "header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;z-index:1;}",
        "main{padding:16px 20px 32px;}",
        "img{display:block;max-width:100%;height:auto;margin:0 auto 24px;border:1px solid #ddd;background:#fff;}",
        "code{background:#eee;padding:2px 4px;border-radius:3px;}",
        "</style>",
        "</head>",
        "<body>",
        "<header>",
        f"<strong>GoodCopBadCop static plot montage</strong><br><code>{export_run_id}</code> | manifest <code>{manifest_name}</code>",
        "</header>",
        "<main>",
    ]
    for page in page_files:
        rel = str(page["path"])
        lines.append(f'<img src="{rel}" alt="{rel}">')
    lines.extend(["</main>", "</body>", "</html>"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_goodcopbadcop_static_montage(
    *,
    export_root: Path = DEFAULT_EXPORT_ROOT,
    export_run_id: str,
    collection_manifest: Path | None = None,
    zarr_paths: Sequence[Path] = (),
    output_dir: Path | None = None,
    artifact_set_id: str = DEFAULT_ARTIFACT_SET_ID,
    records_per_page: int = 8,
    columns: int = 3,
    tile_width: int = 440,
    max_image_height: int = 320,
    margin: int = 24,
    gutter: int = 18,
    fail_on_missing: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    export_root = Path(export_root).expanduser().resolve()
    export_run_id = _resolve_export_run_id(export_root, export_run_id)
    if records_per_page < 1:
        raise ValueError("records_per_page must be >= 1")
    if columns < 1:
        raise ValueError("columns must be >= 1")
    if tile_width < 160:
        raise ValueError("tile_width must be >= 160")
    if max_image_height < 120:
        raise ValueError("max_image_height must be >= 120")

    export_manifest_path = _manifest_path(export_root, export_run_id)
    export_manifest = _read_json(export_manifest_path)
    if collection_manifest is None:
        collection_manifest = _collection_manifest_path_from_export(export_manifest)

    sources: list[SourceRecording]
    if zarr_paths:
        sources = [
            SourceRecording(recording_id=_recording_id_from_zarr_path(Path(path)), zarr_path=Path(path).expanduser().resolve())
            for path in zarr_paths
        ]
    elif collection_manifest is not None:
        sources = _sources_from_collection_manifest(Path(collection_manifest).expanduser().resolve())
    else:
        sources = _sources_from_export_manifest(export_manifest)
    if not sources:
        raise ValueError("No source recordings were resolved for the montage.")

    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else export_root
        / "v1"
        / "artifacts"
        / f"export_run_id={export_run_id}"
        / artifact_set_id
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = default_goodcopbadcop_artifact_specs()
    all_missing: list[dict[str, str]] = []
    page_files: list[dict[str, Any]] = []
    page_count = math.ceil(len(sources) / records_per_page)
    for page_index in range(page_count):
        start = page_index * records_per_page
        stop = min(len(sources), start + records_per_page)
        page_sources = sources[start:stop]
        panels: list[Image.Image] = []
        for source in page_sources:
            tiles, missing = _load_recording_tiles(source, specs, fail_on_missing=fail_on_missing)
            all_missing.extend(missing)
            panels.append(
                _compose_recording_panel(
                    source=source,
                    tiles=tiles,
                    columns=columns,
                    tile_width=tile_width,
                    max_image_height=max_image_height,
                    margin=margin,
                    gutter=gutter,
                )
            )
        page = _compose_page(
            export_run_id=export_run_id,
            page_index=page_index,
            page_count=page_count,
            page_sources=page_sources,
            panels=panels,
            margin=margin,
            gutter=gutter,
        )
        page_name = f"{artifact_set_id}_page_{page_index + 1:03d}.png"
        page_path = output_dir / page_name
        page.save(page_path, format="PNG", optimize=True)
        page_files.append(
            {
                "page_index": page_index,
                "path": page_name,
                "absolute_path": str(page_path),
                "recording_start_index": start,
                "recording_stop_index_exclusive": stop,
                "recording_ids": [source.recording_id for source in page_sources],
                "width_px": page.width,
                "height_px": page.height,
            }
        )

    index_path = output_dir / f"{artifact_set_id}_index.html"
    manifest_name = f"{artifact_set_id}_manifest.json"
    _write_html_index(
        output_path=index_path,
        export_run_id=export_run_id,
        page_files=page_files,
        manifest_name=manifest_name,
    )

    git = get_git_info(Path(__file__).resolve().parents[3])
    manifest: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "artifact_set_id": artifact_set_id,
        "created_at_utc": _utc_now(),
        "tool": "fisheye.utils.export_goodcopbadcop_static_montage",
        "hostname": socket.gethostname(),
        "palette_git_commit": git.get("commit_hash"),
        "palette_git_dirty": git.get("is_dirty"),
        "export_root": str(export_root),
        "export_run_id": export_run_id,
        "export_manifest_path": str(export_manifest_path),
        "collection_manifest_path": str(collection_manifest) if collection_manifest is not None else None,
        "output_dir": str(output_dir),
        "index_html": index_path.name,
        "recording_count": len(sources),
        "artifact_count_per_recording": len(specs),
        "records_per_page": records_per_page,
        "columns": columns,
        "tile_width_px": tile_width,
        "max_image_height_px": max_image_height,
        "artifact_specs": [
            {"artifact_id": spec.artifact_id, "label": spec.label, "path": spec.path}
            for spec in specs
        ],
        "source_recordings": [
            {"recording_id": source.recording_id, "zarr_path": str(source.zarr_path)}
            for source in sources
        ],
        "pages": page_files,
        "missing_artifacts": all_missing,
        "missing_artifact_count": len(all_missing),
    }
    manifest_path = output_dir / manifest_name
    tmp_manifest_path = manifest_path.with_suffix(".json.tmp")
    tmp_manifest_path.write_text(json.dumps(json_attr_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_manifest_path, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build GoodCopBadCop static PNG montage pages for a Palette analytics export.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Palette analytics export root.",
    )
    parser.add_argument(
        "--export-run-id",
        default="latest",
        help="Analytics export_run_id to use, or 'latest'.",
    )
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        help="Optional collection manifest. Defaults to the path stored in the export manifest.",
    )
    parser.add_argument(
        "--zarr",
        action="append",
        type=Path,
        default=[],
        help="Override source Zarr path. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output artifact directory. Defaults under <export-root>/v1/artifacts/export_run_id=<id>/.",
    )
    parser.add_argument("--artifact-set-id", default=DEFAULT_ARTIFACT_SET_ID)
    parser.add_argument("--records-per-page", type=int, default=8)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=440)
    parser.add_argument("--max-image-height", type=int, default=320)
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_goodcopbadcop_static_montage(
        export_root=args.export_root,
        export_run_id=str(args.export_run_id),
        collection_manifest=args.collection_manifest,
        zarr_paths=args.zarr,
        output_dir=args.output_dir,
        artifact_set_id=str(args.artifact_set_id),
        records_per_page=int(args.records_per_page),
        columns=int(args.columns),
        tile_width=int(args.tile_width),
        max_image_height=int(args.max_image_height),
        fail_on_missing=bool(args.fail_on_missing),
        overwrite=bool(args.overwrite),
    )
    print(f"export_run_id\t{manifest['export_run_id']}")
    print(f"recording_count\t{manifest['recording_count']}")
    print(f"page_count\t{len(manifest['pages'])}")
    print(f"missing_artifact_count\t{manifest['missing_artifact_count']}")
    print(f"output_dir\t{manifest['output_dir']}")
    print(f"index_html\t{Path(manifest['output_dir']) / manifest['index_html']}")
    print(f"manifest_path\t{manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
