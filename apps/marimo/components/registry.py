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
from fisheye.visualization.goodcopbadcop_interactive import (
    DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
    GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
)

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


@dataclass(frozen=True)
class RecordingSpecOption:
    zarr_path: Path
    recording_id: str
    label: str
    interactive_spec_count: int
    supported_spec_count: int
    renderer_counts: Mapping[str, int]


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


def _group_names(group: object) -> list[str]:
    group_keys = getattr(group, "group_keys", None)
    if callable(group_keys):
        try:
            return sorted(str(name) for name in group_keys())
        except Exception:
            return []
    return []


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


def _discover_goodcopbadcop_chaser_specs_fast(
    root: zarr.Group,
    archive: Path,
    *,
    run_path_filter: Optional[str],
    artifact_filter: Optional[str],
) -> list[InteractiveSpecOption]:
    run_path_wanted = normalize_path(str(run_path_filter)) if run_path_filter else None
    artifact_wanted = normalize_path(str(artifact_filter)) if artifact_filter else None
    default_artifact = DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT

    if artifact_wanted and "/" not in artifact_wanted and artifact_wanted != default_artifact:
        return []

    candidate_paths: list[str]
    if artifact_wanted and "/" in artifact_wanted:
        candidate_paths = [artifact_wanted]
    elif run_path_wanted:
        candidate_paths = [artifact_path_for(run_path_wanted, default_artifact)]
    else:
        try:
            parent = root["analysis/chaser_distance_runs"]
        except Exception:
            return []
        candidate_paths = [
            artifact_path_for(f"analysis/chaser_distance_runs/{run_name}", default_artifact)
            for run_name in _group_names(parent)
        ]

    options: list[InteractiveSpecOption] = []
    seen: set[str] = set()
    for artifact_path in candidate_paths:
        normalized_path = normalize_path(artifact_path)
        if not normalized_path or normalized_path in seen:
            continue
        seen.add(normalized_path)
        option = _read_option(root, archive, normalized_path)
        if option is None:
            continue
        if option.renderer != GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER:
            continue
        if run_path_wanted and option.run_path != run_path_wanted:
            continue
        if artifact_wanted and artifact_wanted not in {option.artifact_name, option.artifact_path}:
            continue
        options.append(option)
    return sorted(options, key=lambda item: (not item.is_supported, item.renderer, item.run_path, item.artifact_name))


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
    if renderer_wanted == GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER:
        return _discover_goodcopbadcop_chaser_specs_fast(
            root,
            archive,
            run_path_filter=run_path_wanted,
            artifact_filter=artifact_wanted,
        )
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


def recording_id_from_analysis_zarr(zarr_path: Path | str) -> str:
    archive = Path(zarr_path)
    if archive.parent.name == "zarr" and archive.parent.parent.name:
        return archive.parent.parent.name
    name = archive.name
    if name.endswith(".zarr"):
        name = name[:-5]
    if name.endswith("_analysis"):
        name = name[:-9]
    return name or str(archive)


def infer_recordings_root_from_zarr_path(zarr_path: Path | str) -> Path:
    archive = Path(zarr_path).expanduser()
    if archive.parent.name == "zarr" and archive.parent.parent.parent.name:
        return archive.parent.parent.parent
    if archive.parent.name:
        return archive.parent
    return archive


def _candidate_analysis_zarrs(
    recordings_root: Path,
    *,
    name_contains: Optional[str],
) -> list[Path]:
    root = recordings_root.expanduser()
    if root.name.endswith(".zarr") and root.is_dir():
        return [root]

    needle = str(name_contains or "").strip().lower()
    candidates: set[Path] = set()

    def add_from_zarr_dir(zarr_dir: Path) -> None:
        if zarr_dir.is_dir():
            candidates.update(path for path in zarr_dir.glob("*_analysis.zarr") if path.is_dir())

    if (root / "zarr").is_dir():
        add_from_zarr_dir(root / "zarr")

    try:
        children = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        children = []

    for child in children:
        haystack = f"{child.name} {child}".lower()
        if needle and needle not in haystack:
            continue
        if child.name.endswith(".zarr"):
            candidates.add(child)
            continue
        add_from_zarr_dir(child / "zarr")
        candidates.update(path for path in child.glob("*_analysis.zarr") if path.is_dir())

    if not candidates and root.is_dir():
        candidates.update(
            path
            for path in root.rglob("*_analysis.zarr")
            if path.is_dir() and (not needle or needle in f"{path.name} {path}".lower())
        )

    return sorted(candidates)


def discover_protocol_recording_options(
    seed_zarr_path: Path | str,
    *,
    recordings_root: Optional[Path | str] = None,
    renderer_filter: Optional[str] = None,
    run_path_filter: Optional[str] = None,
    artifact_filter: Optional[str] = None,
    name_contains: Optional[str] = "GoodCopBadCop",
) -> list[RecordingSpecOption]:
    """Find sibling recordings with matching interactive specs.

    The common organized-recording layout is:
    ``<recordings_root>/<recording_id>/zarr/<recording_id>_analysis.zarr``.
    The seed archive is always included as a candidate so explicit Zarr paths
    continue to work even when no sibling root can be inferred.
    """

    seed = Path(seed_zarr_path).expanduser()
    root = Path(recordings_root).expanduser() if recordings_root else infer_recordings_root_from_zarr_path(seed)
    candidates = {seed}
    candidates.update(_candidate_analysis_zarrs(root, name_contains=name_contains))

    options: list[RecordingSpecOption] = []
    for archive in sorted(candidates):
        spec_options = discover_interactive_spec_options(
            archive,
            renderer_filter=renderer_filter,
            run_path_filter=run_path_filter,
            artifact_filter=artifact_filter,
        )
        if not spec_options:
            continue
        renderer_counts: dict[str, int] = {}
        for spec in spec_options:
            renderer_counts[spec.renderer or "unknown"] = renderer_counts.get(spec.renderer or "unknown", 0) + 1
        recording_id = recording_id_from_analysis_zarr(archive)
        supported_count = sum(1 for spec in spec_options if spec.is_supported)
        options.append(
            RecordingSpecOption(
                zarr_path=archive,
                recording_id=recording_id,
                label=f"{recording_id} ({supported_count}/{len(spec_options)} supported specs)",
                interactive_spec_count=len(spec_options),
                supported_spec_count=supported_count,
                renderer_counts=renderer_counts,
            )
        )
    return sorted(options, key=lambda item: item.recording_id)


def group_options_by_renderer(options: Iterable[InteractiveSpecOption]) -> dict[str, list[InteractiveSpecOption]]:
    grouped: dict[str, list[InteractiveSpecOption]] = {}
    for option in options:
        grouped.setdefault(option.renderer or "unknown", []).append(option)
    return grouped


def artifact_path_for(run_path: str, artifact_name: str) -> str:
    return join_path(run_path, "visualizations", artifact_name)
