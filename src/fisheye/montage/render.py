"""Load Zarr PNG artifacts and compose labeled montage canvases."""

from __future__ import annotations

import math
from io import BytesIO
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes

from .models import LoadedTile, MontageArtifactSpec, MontageLayout, RegistryRecording


CHASER_DISTANCE_PARENT = "analysis/chaser_distance_runs"


def _chaser_artifact_request(path: str) -> tuple[str, str] | None:
    normalized = "/".join(part for part in str(path).strip("/").split("/") if part)
    prefix = CHASER_DISTANCE_PARENT + "/"
    if not normalized.startswith(prefix):
        return None
    parts = normalized[len(prefix) :].split("/")
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise ChaserDistanceReadError(
            "A chaser montage artifact must identify one exact run child and one "
            "relative artifact path."
        )
    return parts[0], "/".join(parts[1:])


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _ellipsize_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _text_width(draw, text[:mid] + suffix, font) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


def _draw_placeholder(*, width: int, height: int, label: str, error: str | None) -> Image.Image:
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    title_font = _font(16)
    body_font = _font(12)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(180, 180, 180), width=2)
    draw.line((0, 0, width, height), fill=(210, 210, 210), width=2)
    draw.line((0, height, width, 0), fill=(210, 210, 210), width=2)
    title = _ellipsize_text(draw, label, title_font, width - 32)
    message = _ellipsize_text(draw, error or "missing artifact", body_font, width - 32)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    message_bbox = draw.textbbox((0, 0), message, font=body_font)
    draw.text(
        ((width - (title_bbox[2] - title_bbox[0])) // 2, height // 2 - 28),
        title,
        fill=(70, 70, 70),
        font=title_font,
    )
    draw.text(
        ((width - (message_bbox[2] - message_bbox[0])) // 2, height // 2 + 8),
        message,
        fill=(110, 45, 45),
        font=body_font,
    )
    return image


def _resize_to_tile(
    image: Image.Image,
    *,
    tile_width: int,
    max_image_height: int,
) -> Image.Image:
    scale = min(tile_width / float(image.width), max_image_height / float(image.height))
    resized = image.resize(
        (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (tile_width, max_image_height), (255, 255, 255))
    canvas.paste(resized, ((tile_width - resized.width) // 2, 0))
    return canvas


def load_recording_tiles(
    recording: RegistryRecording,
    specs: Sequence[MontageArtifactSpec],
    *,
    fail_on_missing: bool,
) -> tuple[list[LoadedTile], list[dict[str, str]]]:
    tiles: list[LoadedTile] = []
    missing: list[dict[str, str]] = []
    try:
        root = open_zarr_root(recording.zarr_path, mode="r")
    except Exception as exc:
        if fail_on_missing:
            raise
        error = f"failed to open zarr: {type(exc).__name__}: {exc}"
        for spec in specs:
            tiles.append(LoadedTile(spec.artifact_id, spec.label, spec.path, None, error))
            missing.append(
                {
                    "recording_id": recording.recording_id,
                    "zarr_path": str(recording.zarr_path),
                    "artifact_id": spec.artifact_id,
                    "artifact_path": spec.path,
                    "error": error,
                }
            )
        return tiles, missing

    for spec in specs:
        try:
            chaser_request = _chaser_artifact_request(spec.path)
            if chaser_request is not None:
                run_name, relative_path = chaser_request
                distance = load_chaser_distance_run(root, run_name=run_name)
                distance.require_derived_surface_authority(relative_path)
            else:
                resolved_path, png_bytes = load_png_artifact_bytes(root, spec.path)
                if spec.visualization_contract_id is not None:
                    actual_contract = root[resolved_path].attrs.get(
                        "visualization_contract_id"
                    )
                    if actual_contract != spec.visualization_contract_id:
                        raise ValueError(
                            f"visualization contract mismatch: expected "
                            f"{spec.visualization_contract_id!r}, found {actual_contract!r}"
                        )
                image = Image.open(BytesIO(png_bytes)).convert("RGB")
                image.load()
                tiles.append(LoadedTile(spec.artifact_id, spec.label, spec.path, image, None))
        except Exception as exc:
            if fail_on_missing:
                raise
            if isinstance(exc, ChaserDistanceReadError):
                error = f"canonical chaser artifact preflight failed closed: {exc}"
            else:
                error = f"{type(exc).__name__}: {exc}"
            tiles.append(LoadedTile(spec.artifact_id, spec.label, spec.path, None, error))
            missing.append(
                {
                    "recording_id": recording.recording_id,
                    "zarr_path": str(recording.zarr_path),
                    "artifact_id": spec.artifact_id,
                    "artifact_path": spec.path,
                    "visualization_contract_id": spec.visualization_contract_id,
                    "error": error,
                }
            )
    return tiles, missing


def compose_visualization_montage(
    *,
    title: str,
    query_label: str,
    recordings: Sequence[RegistryRecording],
    images: Sequence[Image.Image | None],
    errors: Sequence[str | None],
    layout: MontageLayout,
) -> Image.Image:
    if not recordings:
        raise ValueError("cannot compose an empty montage")
    if len(images) != len(recordings) or len(errors) != len(recordings):
        raise ValueError("recordings, images, and errors must have equal lengths")
    if layout.columns < 1:
        raise ValueError("columns must be >= 1")

    rows = math.ceil(len(recordings) / layout.columns)
    cell_height = layout.label_height + layout.max_image_height
    width = (
        2 * layout.margin
        + layout.columns * layout.tile_width
        + (layout.columns - 1) * layout.gutter
    )
    height = (
        layout.header_height
        + rows * cell_height
        + (rows - 1) * layout.gutter
        + layout.margin
    )
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((layout.margin, 16), title, fill=(25, 30, 38), font=_font(26))
    draw.text(
        (layout.margin, 55),
        f"{query_label} | {len(recordings)} recordings | registry order",
        fill=(82, 89, 100),
        font=_font(14),
    )

    label_font = _font(14)
    for index, (recording, image, error) in enumerate(
        zip(recordings, images, errors, strict=True)
    ):
        row, column = divmod(index, layout.columns)
        x = layout.margin + column * (layout.tile_width + layout.gutter)
        y = layout.header_height + row * (cell_height + layout.gutter)
        behavior_suffix = (
            f" | {', '.join(recording.chaser_behaviors)}"
            if recording.chaser_behaviors
            else ""
        )
        tile_label = _ellipsize_text(
            draw,
            f"{recording.recording_id}{behavior_suffix}",
            label_font,
            layout.tile_width,
        )
        draw.text((x, y + 5), tile_label, fill=(42, 47, 56), font=label_font)
        if image is None:
            rendered = _draw_placeholder(
                width=layout.tile_width,
                height=layout.max_image_height,
                label=recording.recording_id,
                error=error,
            )
        else:
            rendered = _resize_to_tile(
                image,
                tile_width=layout.tile_width,
                max_image_height=layout.max_image_height,
            )
        canvas.paste(rendered, (x, y + layout.label_height))
    return canvas
