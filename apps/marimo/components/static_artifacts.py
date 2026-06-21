"""Static artifact panels shared by Palette marimo explorer components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
from fisheye.utils.zarr_io import open_zarr_root

from .common import join_path, normalize_path, png_bytes_to_markdown_image


def resolve_static_artifact_path(raw_path: object, *, run_path: str) -> Optional[str]:
    if raw_path is None:
        return None
    text = normalize_path(str(raw_path))
    if not text:
        return None
    if text.startswith("visualizations/"):
        return join_path(run_path, text)
    return text


def build_static_artifact_views(
    mo: Any,
    *,
    zarr_path: Path | str,
    run_path: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    root = open_zarr_root(Path(zarr_path), mode="r")
    views: dict[str, Any] = {}
    errors: dict[str, str] = {}
    raw_artifacts = spec.get("static_artifacts", {})
    if not isinstance(raw_artifacts, Mapping):
        return views, {"static_artifacts": "Spec static_artifacts is not a mapping."}
    for key, raw_path in raw_artifacts.items():
        artifact_path = resolve_static_artifact_path(raw_path, run_path=run_path)
        if not artifact_path:
            continue
        label = str(key)
        try:
            resolved_path, png_bytes = load_png_artifact_bytes(root, artifact_path)
        except Exception as exc:
            errors[label] = str(exc)
            continue
        views[label] = mo.vstack(
            [
                mo.md(f"`{resolved_path}`"),
                png_bytes_to_markdown_image(mo, png_bytes, alt_text=label),
            ]
        )
    return views, errors


def build_static_artifacts_panel(
    mo: Any,
    *,
    zarr_path: Path | str,
    run_path: str,
    spec: Mapping[str, Any],
) -> Any:
    views, errors = build_static_artifact_views(mo, zarr_path=zarr_path, run_path=run_path, spec=spec)
    sections: dict[str, Any] = dict(views)
    if errors:
        sections["Unresolved artifacts"] = mo.tree(errors)
    if sections:
        return mo.accordion(sections)
    return mo.md("No static snapshots resolved for this spec.")

