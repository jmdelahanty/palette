"""Renderer registry and generic interactive-spec discovery for marimo apps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import zarr

from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, SPEC_MEDIA_TYPE
from fisheye.utils.view_zarr_visualization import iter_visualization_artifacts
from fisheye.utils.zarr_io import open_zarr_root
from fisheye.visualization.goodcopbadcop_interactive import GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER

from .common import join_path, normalize_path


@dataclass(frozen=True)
class RendererRegistration:
    renderer: str
    label: str
    component_key: str
    description: str


@dataclass(frozen=True)
class InteractiveSpecOption:
    zarr_path: Path
    artifact_path: str
    run_path: str
    artifact_name: str
    renderer: str
    schema_id: Optional[str]
    title: str
    run_name: str
    label: str
    is_supported: bool
    attrs: Mapping[str, Any]
    spec: Mapping[str, Any]


DEFAULT_RENDERER_REGISTRY: dict[str, RendererRegistration] = {
    GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER: RendererRegistration(
        renderer=GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
        label="GoodCopBadCop chaser dashboard",
        component_key="goodcopbadcop_chaser",
        description="Distance traces, selected-window occupancy, and persisted chaser protocol snapshots.",
    )
}


def renderer_registration_for(
    renderer: str,
    *,
    registry: Mapping[str, RendererRegistration] = DEFAULT_RENDERER_REGISTRY,
) -> Optional[RendererRegistration]:
    return registry.get(str(renderer))


def supported_renderer_ids(
    registry: Mapping[str, RendererRegistration] = DEFAULT_RENDERER_REGISTRY,
) -> tuple[str, ...]:
    return tuple(sorted(registry))


def _group_contains(group: object, key: str) -> bool:
    if isinstance(group, zarr.Array) or (hasattr(group, "shape") and hasattr(group, "dtype")):
        return False
    if not (isinstance(group, zarr.Group) or hasattr(group, "group_keys") or hasattr(group, "array_keys")):
        return False
    try:
        return key in group  # type: ignore[operator]
    except Exception:
        return False


def _json_from_uint8_array(array: zarr.Array) -> Mapping[str, Any]:
    payload = np.asarray(array[:], dtype=np.uint8).tobytes().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, Mapping):
        raise ValueError("interactive spec payload must be a JSON object")
    return parsed


def _split_artifact_path(path: str) -> tuple[str, str]:
    normalized = normalize_path(path)
    parts = normalized.split("/") if normalized else []
    try:
        visualizations_index = len(parts) - 1 - parts[::-1].index("visualizations")
    except ValueError:
        parent = "/".join(parts[:-1])
        return parent, parts[-1] if parts else ""
    run_path = "/".join(parts[:visualizations_index])
    artifact_name = parts[visualizations_index + 1] if visualizations_index + 1 < len(parts) else ""
    return run_path, artifact_name


def _is_interactive_spec_candidate(artifact: Any, node: object) -> bool:
    attrs = getattr(node, "attrs", {})
    if str(getattr(artifact, "artifact_role", "") or attrs.get("artifact_role") or "") == "interactive_spec":
        return True
    if str(getattr(artifact, "media_type", "") or attrs.get("media_type") or "") == SPEC_MEDIA_TYPE:
        return True
    if str(attrs.get("artifact_schema_id") or "") == INTERACTIVE_SPEC_SCHEMA_ID:
        return True
    return _group_contains(node, "spec_json")


def _option_label(
    *,
    registration: Optional[RendererRegistration],
    renderer: str,
    run_name: str,
    artifact_name: str,
    title: str,
) -> str:
    renderer_label = registration.label if registration else (renderer or "Unknown renderer")
    detail = run_name or artifact_name or title
    unsupported = "" if registration else " | unsupported"
    return f"{renderer_label} | {detail} | {artifact_name}{unsupported}"


def _read_option(root: zarr.Group, zarr_path: Path, artifact_path: str) -> Optional[InteractiveSpecOption]:
    try:
        node = root[normalize_path(artifact_path)]
    except Exception:
        return None
    if not _is_interactive_spec_candidate(
        type("_Artifact", (), {"artifact_role": None, "media_type": None})(),
        node,
    ):
        return None
    if not _group_contains(node, "spec_json"):
        return None
    try:
        spec = _json_from_uint8_array(node["spec_json"])  # type: ignore[index]
    except Exception:
        return None
    attrs = dict(getattr(node, "attrs", {}))
    renderer = str(spec.get("renderer") or attrs.get("renderer") or "").strip()
    run_path, artifact_name = _split_artifact_path(artifact_path)
    fallback_run_name = normalize_path(run_path).split("/")[-1] if run_path else ""
    run_name = str(spec.get("run_name") or fallback_run_name).strip()
    title = str(spec.get("title") or attrs.get("description") or artifact_name).strip()
    registration = renderer_registration_for(renderer)
    return InteractiveSpecOption(
        zarr_path=zarr_path,
        artifact_path=normalize_path(artifact_path),
        run_path=normalize_path(run_path),
        artifact_name=str(artifact_name),
        renderer=renderer,
        schema_id=str(spec.get("schema_id")) if spec.get("schema_id") is not None else None,
        title=title,
        run_name=run_name,
        label=_option_label(
            registration=registration,
            renderer=renderer,
            run_name=run_name,
            artifact_name=str(artifact_name),
            title=title,
        ),
        is_supported=registration is not None,
        attrs=attrs,
        spec=spec,
    )


def discover_interactive_spec_options(
    zarr_path: Path | str,
    *,
    renderer_filter: Optional[str] = None,
    run_path_filter: Optional[str] = None,
    artifact_filter: Optional[str] = None,
) -> list[InteractiveSpecOption]:
    """Discover persisted interactive visualization specs in a Palette Zarr archive."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    renderer_wanted = str(renderer_filter).strip() if renderer_filter else None
    run_path_wanted = normalize_path(str(run_path_filter)) if run_path_filter else None
    artifact_wanted = normalize_path(str(artifact_filter)) if artifact_filter else None
    options: list[InteractiveSpecOption] = []
    for artifact in iter_visualization_artifacts(root):
        try:
            node = root[normalize_path(artifact.path)]
        except Exception:
            continue
        if not _is_interactive_spec_candidate(artifact, node):
            continue
        option = _read_option(root, archive, artifact.path)
        if option is None:
            continue
        if renderer_wanted and option.renderer != renderer_wanted:
            continue
        if run_path_wanted and option.run_path != run_path_wanted:
            continue
        if artifact_wanted and artifact_wanted not in {option.artifact_name, option.artifact_path}:
            continue
        options.append(option)
    return sorted(options, key=lambda item: (not item.is_supported, item.renderer, item.run_path, item.artifact_name))


def group_options_by_renderer(options: Iterable[InteractiveSpecOption]) -> dict[str, list[InteractiveSpecOption]]:
    grouped: dict[str, list[InteractiveSpecOption]] = {}
    for option in options:
        grouped.setdefault(option.renderer or "unknown", []).append(option)
    return grouped


def artifact_path_for(run_path: str, artifact_name: str) -> str:
    return join_path(run_path, "visualizations", artifact_name)
