"""Renderer registry and generic interactive-spec discovery for marimo apps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import zarr

from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.analysis.provider_chaser_distance_candidates import (
    MANIFEST_DIGEST_ATTR as PROVIDER_CHASER_CANDIDATE_MANIFEST_DIGEST_ATTR,
    PARENT_PATH as PROVIDER_CHASER_CANDIDATE_PARENT_PATH,
    SCHEMA_ID as PROVIDER_CHASER_CANDIDATE_SCHEMA_ID,
    validate_provider_chaser_distance_candidate,
)
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    MANIFEST_ATTR as RELATIVE_FRAME_MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR as RELATIVE_FRAME_MANIFEST_DIGEST_ATTR,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    MANIFEST_ATTR as COMPOSABLE_CHASER_MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR as COMPOSABLE_CHASER_MANIFEST_DIGEST_ATTR,
    STORAGE_SCHEMA_ID as COMPOSABLE_CHASER_STORAGE_SCHEMA_ID,
    STORAGE_SCHEMA_VERSION as COMPOSABLE_CHASER_STORAGE_SCHEMA_VERSION,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    require_same_exact_relative_frame_child,
    validate_exact_relative_frame_binding,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, SPEC_MEDIA_TYPE
from fisheye.shared.recording_artifact_inventory import (
    build_recording_artifact_inventory,
)
from fisheye.utils.view_zarr_visualization import iter_visualization_artifacts
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.status_page.query import open_readonly_connection, resolve_registry_path
from fisheye.visualization.goodcopbadcop_interactive import (
    CHASER_DASHBOARD_RENDERER,
    CHASER_DASHBOARD_RENDERERS,
    LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
)
from fisheye.visualization.bout_kinematics_interactive import (
    BOUT_EYE_GAZE_PLOT_RENDERER,
    BOUT_HEADING_PLOT_RENDERER,
    BOUT_MOVEMENT_PLOT_RENDERER,
    BOUT_PLOT_RENDERERS,
    BOUT_PLOT_SPEC_SCHEMA_IDS,
    LEGACY_BOUT_PLOT_RENDERER,
    bout_schema_for_artifact_name,
    effective_bout_renderer,
)
from .common import join_path, normalize_path

TRACK_KINEMATICS_PLOT_RENDERER = "palette-track-kinematics-summary-v1"
TRACK_KINEMATICS_INTERACTIVE_ARTIFACT = "track_kinematics_summary_track_0_interactive"
PROVIDER_CHASER_CANDIDATE_RENDERER = (
    "palette-provider-chaser-distance-candidate-explorer-v1"
)
PROVIDER_CHASER_CANDIDATE_ARTIFACT = "provider_chaser_distance_candidate"
CHASER_EXACT_SUCCESSOR_RENDERER = "palette-chaser-exact-successor-explorer-v1"
CHASER_EXACT_SUCCESSOR_ARTIFACT = "chaser_exact_successor_bundle"
CHASER_EXACT_SUCCESSOR_PARENT_PATH = "analysis/chaser_spatial_occupancy_runs"
_PROVIDER_CHASER_CANDIDATE_FORBIDDEN_SELECTORS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "selected",
        "default",
    }
)


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
    spec_counts_loaded: bool = True


DEFAULT_RENDERER_REGISTRY: dict[str, RendererRegistration] = {
    TRACK_KINEMATICS_PLOT_RENDERER: RendererRegistration(
        renderer=TRACK_KINEMATICS_PLOT_RENDERER,
        label="Core behavior",
        component_key="core_behavior",
        description=(
            "Projected speed, heading, position, and swim-bout views from one "
            "persisted track-kinematics run."
        ),
    ),
    BOUT_HEADING_PLOT_RENDERER: RendererRegistration(
        renderer=BOUT_HEADING_PLOT_RENDERER,
        label="Bout heading kinematics",
        component_key="bout_kinematics",
        description="Persisted per-bout heading-change and angular-motion summaries.",
    ),
    BOUT_MOVEMENT_PLOT_RENDERER: RendererRegistration(
        renderer=BOUT_MOVEMENT_PLOT_RENDERER,
        label="Bout physical movement",
        component_key="bout_kinematics",
        description="Persisted per-bout duration, path-length, and speed summaries.",
    ),
    BOUT_EYE_GAZE_PLOT_RENDERER: RendererRegistration(
        renderer=BOUT_EYE_GAZE_PLOT_RENDERER,
        label="Bout eye gaze",
        component_key="bout_kinematics",
        description="Persisted bout-aligned eye-gaze and convergence summaries.",
    ),
    CHASER_DASHBOARD_RENDERER: RendererRegistration(
        renderer=CHASER_DASHBOARD_RENDERER,
        label="Chaser dashboard",
        component_key="goodcopbadcop_chaser",
        description="Distance traces, selected-window occupancy, and persisted chaser protocol snapshots.",
    ),
    PROVIDER_CHASER_CANDIDATE_RENDERER: RendererRegistration(
        renderer=PROVIDER_CHASER_CANDIDATE_RENDERER,
        label="Chaser provider candidate",
        component_key="provider_chaser_candidate",
        description=(
            "Read-only inspection of an exact, manifest-validated, unpromoted "
            "provider-aware chaser-distance candidate."
        ),
    ),
    CHASER_EXACT_SUCCESSOR_RENDERER: RendererRegistration(
        renderer=CHASER_EXACT_SUCCESSOR_RENDERER,
        label="Exact chaser successors",
        component_key="chaser_exact_successors",
        description=(
            "Read-only paired-provider views from one sealed spatial-occupancy "
            "bundle and its exact relative-frame and radial children."
        ),
    ),
    LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER: RendererRegistration(
        renderer=LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
        label="Chaser dashboard",
        component_key="goodcopbadcop_chaser",
        description="Legacy GoodCopBadCop renderer ID for chaser dashboard artifacts.",
    ),
    "palette-goodcopbadcop-cra-primary-endpoint-v1": RendererRegistration(
        renderer="palette-goodcopbadcop-cra-primary-endpoint-v1",
        label="CRA primary endpoint",
        component_key="goodcopbadcop_chaser",
        description="Object-relative CRA endpoint view within a chaser protocol run.",
    ),
    "palette-goodcopbadcop-cra-near-field-v1": RendererRegistration(
        renderer="palette-goodcopbadcop-cra-near-field-v1",
        label="CRA near-field",
        component_key="goodcopbadcop_chaser",
        description="Near-field avoidance view within a chaser protocol run.",
    ),
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
    if isinstance(group, zarr.Array) or (
        hasattr(group, "shape") and hasattr(group, "dtype")
    ):
        return False
    if not (
        isinstance(group, zarr.Group)
        or hasattr(group, "group_keys")
        or hasattr(group, "array_keys")
    ):
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
    artifact_name = (
        parts[visualizations_index + 1] if visualizations_index + 1 < len(parts) else ""
    )
    return run_path, artifact_name


def _chaser_distance_run_name(path: str | None) -> str | None:
    parts = normalize_path(str(path or "")).split("/")
    for index in range(len(parts) - 2):
        if parts[index : index + 2] == ["analysis", "chaser_distance_runs"]:
            name = parts[index + 2].strip()
            if name and name not in {".", ".."}:
                return name
    return None


def _is_interactive_spec_candidate(artifact: Any, node: object) -> bool:
    attrs = getattr(node, "attrs", {})
    if (
        str(getattr(artifact, "artifact_role", "") or attrs.get("artifact_role") or "")
        == "interactive_spec"
    ):
        return True
    if (
        str(getattr(artifact, "media_type", "") or attrs.get("media_type") or "")
        == SPEC_MEDIA_TYPE
    ):
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
    renderer_label = (
        registration.label if registration else (renderer or "Unknown renderer")
    )
    detail = run_name or artifact_name or title
    unsupported = "" if registration else " | unsupported"
    return f"{renderer_label} | {detail} | {artifact_name}{unsupported}"


def _read_option(
    root: zarr.Group, zarr_path: Path, artifact_path: str
) -> Optional[InteractiveSpecOption]:
    chaser_run_name = _chaser_distance_run_name(artifact_path)
    if chaser_run_name is not None:
        # A verified base run does not seal its dashboard spec. Preflight the
        # exact base publication, but never read or advertise the raw artifact.
        try:
            load_chaser_distance_run(root, run_name=chaser_run_name)
        except ChaserDistanceReadError:
            return None
        return None
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
    schema_id = str(spec.get("schema_id") or attrs.get("plot_schema_id") or "").strip()
    persisted_renderer = str(
        spec.get("renderer") or attrs.get("renderer") or ""
    ).strip()
    renderer = effective_bout_renderer(persisted_renderer, schema_id)
    run_path, artifact_name = _split_artifact_path(artifact_path)
    fallback_run_name = normalize_path(run_path).split("/")[-1] if run_path else ""
    run_name = str(spec.get("run_name") or fallback_run_name).strip()
    if schema_id in BOUT_PLOT_SPEC_SCHEMA_IDS and fallback_run_name:
        # Materialized copies may retain the source spec's run_name. Display
        # the actual owning run while leaving the persisted spec untouched for
        # provenance inspection.
        run_name = fallback_run_name
    title = str(spec.get("title") or attrs.get("description") or artifact_name).strip()
    registration = renderer_registration_for(renderer)
    return InteractiveSpecOption(
        zarr_path=zarr_path,
        artifact_path=normalize_path(artifact_path),
        run_path=normalize_path(run_path),
        artifact_name=str(artifact_name),
        renderer=renderer,
        schema_id=schema_id or None,
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


def _inventory_interactive_artifact_paths(root: zarr.Group) -> list[str]:
    try:
        inventory = build_recording_artifact_inventory(root)
    except Exception:
        return []

    paths: list[str] = []
    families = []
    for key in ("root_run_families", "analysis_run_families", "nested_report_families"):
        value = inventory.get(key)
        if isinstance(value, list):
            families.extend(item for item in value if isinstance(item, Mapping))

    for family in families:
        runs = family.get("runs")
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, Mapping):
                continue
            run_path = normalize_path(str(run.get("path") or ""))
            visualizations = run.get("visualizations")
            if not run_path or not isinstance(visualizations, list):
                continue
            for artifact in visualizations:
                if not isinstance(artifact, Mapping):
                    continue
                role = str(artifact.get("artifact_role") or "")
                media_type = str(artifact.get("media_type") or "")
                schema_id = str(artifact.get("artifact_schema_id") or "")
                if (
                    role != "interactive_spec"
                    and media_type != SPEC_MEDIA_TYPE
                    and schema_id != INTERACTIVE_SPEC_SCHEMA_ID
                ):
                    continue
                relative_path = normalize_path(str(artifact.get("path") or ""))
                if not relative_path:
                    artifact_name = str(artifact.get("artifact_name") or "").strip()
                    if not artifact_name:
                        continue
                    relative_path = artifact_path_for(run_path, artifact_name)
                    paths.append(relative_path)
                else:
                    paths.append(join_path(run_path, relative_path))
    return sorted(set(paths))


def _discover_goodcopbadcop_chaser_specs_fast(
    root: zarr.Group,
    archive: Path,
    *,
    run_path_filter: Optional[str],
    artifact_filter: Optional[str],
) -> list[InteractiveSpecOption]:
    run_path_wanted = normalize_path(str(run_path_filter)) if run_path_filter else None
    artifact_wanted = normalize_path(str(artifact_filter)) if artifact_filter else None
    requested_run = (
        _chaser_distance_run_name(run_path_wanted)
        or _chaser_distance_run_name(artifact_wanted)
        or "latest"
    )
    try:
        load_chaser_distance_run(root, run_name=requested_run)
    except ChaserDistanceReadError:
        return []
    # Dashboard artifacts have no independent payload/semantic seal yet.
    return []


def discover_provider_chaser_candidate_options(
    zarr_path: Path | str,
    *,
    run_path_filter: Optional[str] = None,
    artifact_filter: Optional[str] = None,
) -> list[InteractiveSpecOption]:
    """Discover exact valid candidates without introducing selector semantics."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    run_path_wanted = normalize_path(str(run_path_filter)) if run_path_filter else None
    artifact_wanted = normalize_path(str(artifact_filter)) if artifact_filter else None
    try:
        parent = root[PROVIDER_CHASER_CANDIDATE_PARENT_PATH]
    except Exception:
        return []
    if _PROVIDER_CHASER_CANDIDATE_FORBIDDEN_SELECTORS.intersection(parent.attrs):
        return []

    options: list[InteractiveSpecOption] = []
    for run_name in _group_names(parent):
        run_path = f"{PROVIDER_CHASER_CANDIDATE_PARENT_PATH}/{run_name}"
        if run_path_wanted and run_path != run_path_wanted:
            continue
        run = parent[run_name]
        attrs = dict(getattr(run, "attrs", {}))
        manifest_sha256 = str(
            attrs.get(PROVIDER_CHASER_CANDIDATE_MANIFEST_DIGEST_ATTR) or ""
        ).strip()
        if (
            attrs.get("schema_id") != PROVIDER_CHASER_CANDIDATE_SCHEMA_ID
            or attrs.get("stage_selector_eligible") is not False
            or attrs.get("row_axis") != "stimulus_samples"
            or not manifest_sha256
        ):
            continue
        try:
            validation = validate_provider_chaser_distance_candidate(
                archive / run_path,
                use_consolidated=True,
                archive_path=archive,
                archive_run_path=run_path,
                expected_manifest_sha256=manifest_sha256,
            )
        except Exception:
            continue
        if not validation.get("valid"):
            continue

        static_artifacts = {
            "Distance trace (stimulus samples)": "visualizations/distance_trace_png",
            "Distance histogram (stimulus samples)": (
                "visualizations/distance_histogram_png"
            ),
        }
        accepted_artifacts = {
            PROVIDER_CHASER_CANDIDATE_ARTIFACT,
            *static_artifacts.values(),
            *(join_path(run_path, path) for path in static_artifacts.values()),
        }
        if artifact_wanted and artifact_wanted not in accepted_artifacts:
            continue

        title = f"Unpromoted chaser-distance candidate: {run_name}"
        spec = {
            "schema_id": "palette.provider_chaser_distance_candidate_explorer_spec",
            "schema_version": 1,
            "renderer": PROVIDER_CHASER_CANDIDATE_RENDERER,
            "title": title,
            "run_name": run_name,
            "candidate_status": "unpromoted_selector_ineligible",
            "row_axis": "stimulus_samples",
            "manifest_sha256": manifest_sha256,
            "source_paths": {
                "candidate_run": run_path,
                "position_run": attrs.get("source_position_run_path"),
                "stimulus_run": attrs.get("source_stimulus_run_path"),
                "stimulus_epoch_run": attrs.get("source_stimulus_epoch_run_path"),
            },
            "static_artifacts": static_artifacts,
            "adapter_semantics": (
                "read_only_exact_run_no_selector_no_canonical_promotion"
            ),
        }
        options.append(
            InteractiveSpecOption(
                zarr_path=archive,
                artifact_path=join_path(run_path, "visualizations"),
                run_path=run_path,
                artifact_name=PROVIDER_CHASER_CANDIDATE_ARTIFACT,
                renderer=PROVIDER_CHASER_CANDIDATE_RENDERER,
                schema_id=PROVIDER_CHASER_CANDIDATE_SCHEMA_ID,
                title=title,
                run_name=run_name,
                label=(
                    f"Chaser provider candidate | {run_name} | "
                    "UNPROMOTED · selector-ineligible"
                ),
                is_supported=True,
                attrs=attrs,
                spec=spec,
            )
        )
    return sorted(
        options,
        key=lambda item: (
            str(item.attrs.get("created_at_utc") or ""),
            item.run_name,
        ),
        reverse=True,
    )


def _valid_composable_successor_manifest(
    run: Any,
    *,
    expected_kind: str,
    expected_run_path: str,
    expected_digest: str | None = None,
) -> Mapping[str, Any] | None:
    attrs = dict(getattr(run, "attrs", {}))
    manifest = attrs.get(COMPOSABLE_CHASER_MANIFEST_ATTR)
    digest = str(attrs.get(COMPOSABLE_CHASER_MANIFEST_DIGEST_ATTR) or "")
    if not isinstance(manifest, Mapping) or len(digest) != 64:
        return None
    try:
        current_digest = canonical_json_sha256(dict(manifest))
    except Exception:
        return None
    if (
        current_digest != digest
        or (expected_digest is not None and digest != expected_digest)
        or attrs.get("schema_id") != COMPOSABLE_CHASER_STORAGE_SCHEMA_ID
        or attrs.get("schema_version") != COMPOSABLE_CHASER_STORAGE_SCHEMA_VERSION
        or attrs.get("successor_kind") != expected_kind
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("successor_kind") != expected_kind
        or manifest.get("run_path") != expected_run_path
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        return None
    return manifest


def _exact_bound_child(
    root: Any,
    binding: Any,
    *,
    parent: str,
    kind: str | None,
    recording_id: str,
) -> Mapping[str, Any] | None:
    if not isinstance(binding, Mapping):
        return None
    path = str(binding.get("run_path") or "")
    digest = str(binding.get("manifest_sha256") or "")
    prefix = f"{parent}/"
    name = path.removeprefix(prefix)
    if (
        not path.startswith(prefix)
        or not name
        or "/" in name
        or name in {"latest", "default", "selected", ".", ".."}
        or len(digest) != 64
    ):
        return None
    try:
        run = root[path]
    except Exception:
        return None
    if kind is not None:
        manifest = _valid_composable_successor_manifest(
            run,
            expected_kind=kind,
            expected_run_path=path,
            expected_digest=digest,
        )
        if manifest is None or manifest.get("recording_id") != recording_id:
            return None
        return manifest
    attrs = dict(getattr(run, "attrs", {}))
    manifest = attrs.get(RELATIVE_FRAME_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        return None
    try:
        current_digest = canonical_json_sha256(dict(manifest))
    except Exception:
        return None
    if (
        attrs.get(RELATIVE_FRAME_MANIFEST_DIGEST_ATTR) != digest
        or current_digest != digest
        or attrs.get("schema_id") != "palette.analysis.chaser_relative_frame"
        or attrs.get("schema_version") != 1
        or attrs.get("run_path") != path
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("recording_id") != recording_id
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        return None
    return manifest


def discover_exact_chaser_successor_options(
    zarr_path: Path | str,
    *,
    run_path_filter: Optional[str] = None,
    artifact_filter: Optional[str] = None,
) -> list[InteractiveSpecOption]:
    """Discover complete paired-provider bundles without resolving selectors."""

    archive = Path(zarr_path)
    try:
        root = open_zarr_root(archive, mode="r", use_consolidated=True)
    except (OSError, TypeError, ValueError, RuntimeError):
        # Exact successors are immutable published artifacts. Missing or stale
        # consolidated metadata makes this capability undiscoverable; it is not
        # normalized through an unconsolidated fallback.
        return []
    try:
        parent = root[CHASER_EXACT_SUCCESSOR_PARENT_PATH]
    except Exception:
        return []
    if {
        "latest",
        "latest_complete",
        "selected",
        "authoritative",
        "default",
    }.intersection(getattr(parent, "attrs", {})):
        return []
    run_wanted = normalize_path(str(run_path_filter)) if run_path_filter else None
    artifact_wanted = normalize_path(str(artifact_filter)) if artifact_filter else None
    options: list[InteractiveSpecOption] = []
    for run_name in _group_names(parent):
        run_path = f"{CHASER_EXACT_SUCCESSOR_PARENT_PATH}/{run_name}"
        if run_wanted and run_wanted != run_path:
            continue
        run = parent[run_name]
        manifest = _valid_composable_successor_manifest(
            run,
            expected_kind="chaser_spatial_occupancy",
            expected_run_path=run_path,
        )
        if manifest is None:
            continue
        recording_id = str(manifest.get("recording_id") or "")
        scientific = manifest.get("scientific_manifest")
        sources = scientific.get("sources") if isinstance(scientific, Mapping) else None
        providers = (
            sources.get("position_providers") if isinstance(sources, Mapping) else None
        )
        if not isinstance(providers, list) or len(providers) != 2:
            continue
        roles = tuple(
            value.get("provider_role") if isinstance(value, Mapping) else None
            for value in providers
        )
        if roles != ("keypoint", "detection"):
            continue
        provider_ids: list[str] = []
        relative_binding_proofs: list[dict[str, Any]] = []
        valid = True
        for record in providers:
            if not isinstance(record, Mapping):
                valid = False
                break
            try:
                spatial_relative_binding = validate_exact_relative_frame_binding(
                    record.get("relative_frame"),
                    label="spatial relative-frame binding",
                )
            except ExactRelativeFrameBindingError:
                valid = False
                break
            relative = _exact_bound_child(
                root,
                spatial_relative_binding.normalized_identity,
                parent="analysis/chaser_relative_frame_runs",
                kind=None,
                recording_id=recording_id,
            )
            radial = _exact_bound_child(
                root,
                record.get("radial_near_field"),
                parent="analysis/chaser_radial_near_field_runs",
                kind="chaser_radial_near_field",
                recording_id=recording_id,
            )
            if relative is None or radial is None:
                valid = False
                break
            relative_authorities = relative.get("source_authorities")
            relative_fish = (
                relative_authorities.get("fish_position")
                if isinstance(relative_authorities, Mapping)
                else None
            )
            radial_scientific = radial.get("scientific_manifest")
            radial_provider = (
                radial_scientific.get("position_provider")
                if isinstance(radial_scientific, Mapping)
                else None
            )
            radial_sources = (
                radial_scientific.get("sources")
                if isinstance(radial_scientific, Mapping)
                else None
            )
            radial_relative = (
                radial_sources.get("relative_frame")
                if isinstance(radial_sources, Mapping)
                else None
            )
            try:
                relative_proof = require_same_exact_relative_frame_child(
                    record.get("relative_frame"),
                    radial_relative,
                    expected_label="spatial relative-frame binding",
                    observed_label="radial relative-frame binding",
                )
            except ExactRelativeFrameBindingError:
                valid = False
                break
            provider_id = str(record.get("provider_id") or "")
            provider_digest = str(record.get("provider_digest") or "")
            if (
                not isinstance(relative_fish, Mapping)
                or not isinstance(radial_provider, Mapping)
                or relative_fish.get("provider_id") != provider_id
                or relative_fish.get("provider_digest") != provider_digest
                or radial_provider.get("provider_id") != provider_id
                or radial_provider.get("provider_digest") != provider_digest
                or radial_provider.get("status") != "first_class_explicit_authority"
            ):
                valid = False
                break
            provider_ids.append(provider_id)
            relative_binding_proofs.append(
                {
                    "normalized_identity": dict(relative_proof.normalized_identity),
                    "spatial_binding_profile": relative_proof.expected.profile_id,
                    "radial_binding_profile": relative_proof.observed.profile_id,
                    "validation_receipt_sha256": (
                        relative_proof.expected.validation_receipt_sha256
                    ),
                    "verification_mode": relative_proof.expected.verification_mode,
                    "validation_behavior": (
                        "binding_schema_and_identity_validated_"
                        "receipt_digest_not_reopened"
                    ),
                }
            )
        if not valid or len(set(provider_ids)) != 2:
            continue
        artifact_path = join_path(run_path, "interactive")
        if artifact_wanted and artifact_wanted not in {
            CHASER_EXACT_SUCCESSOR_ARTIFACT,
            artifact_path,
        }:
            continue
        attrs = dict(getattr(run, "attrs", {}))
        digest = str(attrs[COMPOSABLE_CHASER_MANIFEST_DIGEST_ATTR])
        title = f"Exact paired-provider chaser successors: {run_name}"
        spec = {
            "schema_id": "palette.chaser_exact_successor_explorer_spec",
            "schema_version": 2,
            "renderer": CHASER_EXACT_SUCCESSOR_RENDERER,
            "title": title,
            "run_name": run_name,
            "bundle_status": "exact_selector_ineligible",
            "bundle_manifest_sha256": digest,
            "provider_ids": provider_ids,
            "source_paths": {
                "spatial_occupancy": run_path,
                "position_providers": providers,
            },
            "relative_frame_binding_proofs": relative_binding_proofs,
            "adapter_semantics": (
                "read_only_exact_children_no_selector_no_interpolation"
            ),
            "display_parameters": {
                "distance_traces": {
                    "algorithm": (
                        "source_order_bucket_first_last_min_max_missing_break_v1"
                    ),
                    "max_points_per_series": 6000,
                    "connect_missing_gaps": False,
                },
                "trajectory_overlays": {
                    "algorithm": ("source_order_uniform_plus_coordinate_extrema_v1"),
                    "max_points_per_series_per_epoch": 15000,
                },
                "scientific_recomputation": False,
                "interpolation": "prohibited",
            },
        }
        options.append(
            InteractiveSpecOption(
                zarr_path=archive,
                artifact_path=artifact_path,
                run_path=run_path,
                artifact_name=CHASER_EXACT_SUCCESSOR_ARTIFACT,
                renderer=CHASER_EXACT_SUCCESSOR_RENDERER,
                schema_id=str(spec["schema_id"]),
                title=title,
                run_name=run_name,
                label=f"Exact chaser successors | {run_name} | selector-ineligible",
                is_supported=True,
                attrs=attrs,
                spec=spec,
            )
        )
    return sorted(options, key=lambda item: item.run_name, reverse=True)


def _discover_track_kinematics_specs_fast(
    root: zarr.Group,
    archive: Path,
    *,
    run_path_filter: Optional[str],
    artifact_filter: Optional[str],
) -> list[InteractiveSpecOption]:
    run_path_wanted = normalize_path(str(run_path_filter)) if run_path_filter else None
    artifact_wanted = normalize_path(str(artifact_filter)) if artifact_filter else None
    if (
        artifact_wanted
        and "/" not in artifact_wanted
        and artifact_wanted != TRACK_KINEMATICS_INTERACTIVE_ARTIFACT
    ):
        return []

    if artifact_wanted and "/" in artifact_wanted:
        candidate_paths = [artifact_wanted]
    else:
        candidate_paths = []
        try:
            visualization_parent = root["analysis/track_kinematics_visualization_runs"]
        except Exception:
            return []
        for scope in _group_names(visualization_parent):
            scope_group = visualization_parent[scope]
            for source_run_name in _group_names(scope_group):
                source_run = scope_group[source_run_name]
                tracks = source_run.get("tracks")
                if tracks is None:
                    continue
                for track_name in _group_names(tracks):
                    render_parent = tracks[track_name]
                    render_name = str(
                        render_parent.attrs.get("latest_complete")
                        or render_parent.attrs.get("latest")
                        or ""
                    ).strip()
                    if not render_name or render_name not in render_parent:
                        continue
                    render = render_parent[render_name]
                    if (
                        render.attrs.get("palette_run_completion_status") != "complete"
                        or render.attrs.get("stage_selector_eligible") is not True
                    ):
                        continue
                    visualizations = render.get("visualizations")
                    if visualizations is None:
                        continue
                    candidate_paths.extend(
                        f"analysis/track_kinematics_visualization_runs/{scope}/"
                        f"{source_run_name}/tracks/{track_name}/{render_name}/"
                        f"visualizations/{artifact_name}"
                        for artifact_name in _group_names(visualizations)
                    )

    options: list[InteractiveSpecOption] = []
    for artifact_path in candidate_paths:
        option = _read_option(root, archive, artifact_path)
        if option is None or option.renderer != TRACK_KINEMATICS_PLOT_RENDERER:
            continue
        if run_path_wanted:
            source_paths = option.spec.get("source_paths")
            source_run_path = (
                normalize_path(str(source_paths.get("run") or ""))
                if isinstance(source_paths, Mapping)
                else ""
            )
            if run_path_wanted not in {option.run_path, source_run_path}:
                continue
        if artifact_wanted and artifact_wanted not in {
            option.artifact_name,
            option.artifact_path,
        }:
            continue
        options.append(option)
    return sorted(
        options,
        key=lambda item: (not item.is_supported, item.run_path, item.artifact_name),
    )


def _discover_bout_kinematics_specs_fast(
    root: zarr.Group,
    archive: Path,
    *,
    renderer_filter: Optional[str],
    run_path_filter: Optional[str],
    artifact_filter: Optional[str],
) -> list[InteractiveSpecOption]:
    """Discover only exact bout schemas without walking the whole archive."""

    renderer_wanted = str(renderer_filter).strip() if renderer_filter else None
    run_path_wanted = normalize_path(str(run_path_filter)) if run_path_filter else None
    artifact_wanted = normalize_path(str(artifact_filter)) if artifact_filter else None
    if renderer_wanted not in {None, LEGACY_BOUT_PLOT_RENDERER, *BOUT_PLOT_RENDERERS}:
        return []
    if artifact_wanted and "/" not in artifact_wanted:
        if bout_schema_for_artifact_name(artifact_wanted) is None:
            return []

    if artifact_wanted and "/" in artifact_wanted:
        candidate_paths = [artifact_wanted]
    else:
        if run_path_wanted:
            run_paths = [run_path_wanted]
        else:
            try:
                parent = root["analysis/bout_kinematics_runs"]
            except Exception:
                return []
            run_names = _group_names(parent)
            preferred_run_names: list[str] = []
            parent_attrs = getattr(parent, "attrs", {})
            for pointer_name in ("latest_complete", "latest"):
                pointed_run = str(parent_attrs.get(pointer_name) or "").strip()
                if pointed_run in run_names and pointed_run not in preferred_run_names:
                    preferred_run_names.append(pointed_run)
            ordered_run_names = preferred_run_names + [
                run_name
                for run_name in reversed(run_names)
                if run_name not in preferred_run_names
            ]
            run_paths = [
                f"analysis/bout_kinematics_runs/{run_name}"
                for run_name in ordered_run_names
            ]
        candidate_paths = []
        for run_path in run_paths:
            if artifact_wanted:
                candidate_paths.append(artifact_path_for(run_path, artifact_wanted))
                continue
            try:
                visualizations = root[join_path(run_path, "visualizations")]
            except Exception:
                continue
            candidate_paths.extend(
                artifact_path_for(run_path, artifact_name)
                for artifact_name in _group_names(visualizations)
                if bout_schema_for_artifact_name(artifact_name) is not None
            )

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
        if option.schema_id not in BOUT_PLOT_SPEC_SCHEMA_IDS:
            continue
        if option.renderer not in BOUT_PLOT_RENDERERS:
            continue
        if renderer_wanted not in {None, LEGACY_BOUT_PLOT_RENDERER, option.renderer}:
            continue
        if run_path_wanted and option.run_path != run_path_wanted:
            continue
        if artifact_wanted and artifact_wanted not in {
            option.artifact_name,
            option.artifact_path,
        }:
            continue
        options.append(option)
    run_order = {
        run_path: index
        for index, run_path in enumerate(
            dict.fromkeys(option.run_path for option in options)
        )
    }
    return sorted(
        options, key=lambda item: (run_order[item.run_path], item.artifact_name)
    )


def discover_recording_explorer_spec_options(
    zarr_path: Path | str,
    *,
    renderer_filter: Optional[str] = None,
    run_path_filter: Optional[str] = None,
    artifact_filter: Optional[str] = None,
) -> list[InteractiveSpecOption]:
    """Discover only providers mounted by the single-recording explorer.

    This avoids building the broad recording artifact inventory, which is
    useful for audits but expensive over a network filesystem.
    """

    archive = Path(zarr_path)
    renderer_wanted = str(renderer_filter).strip() if renderer_filter else None
    if renderer_wanted and renderer_wanted not in {
        TRACK_KINEMATICS_PLOT_RENDERER,
        PROVIDER_CHASER_CANDIDATE_RENDERER,
        CHASER_EXACT_SUCCESSOR_RENDERER,
        *CHASER_DASHBOARD_RENDERERS,
        *BOUT_PLOT_RENDERERS,
        LEGACY_BOUT_PLOT_RENDERER,
    }:
        return discover_interactive_spec_options(
            archive,
            renderer_filter=renderer_wanted,
            run_path_filter=run_path_filter,
            artifact_filter=artifact_filter,
        )
    root = open_zarr_root(archive, mode="r")
    options: list[InteractiveSpecOption] = []
    if renderer_wanted in {None, TRACK_KINEMATICS_PLOT_RENDERER}:
        options.extend(
            _discover_track_kinematics_specs_fast(
                root,
                archive,
                run_path_filter=run_path_filter,
                artifact_filter=artifact_filter,
            )
        )
    if renderer_wanted in {None, PROVIDER_CHASER_CANDIDATE_RENDERER}:
        options.extend(
            discover_provider_chaser_candidate_options(
                archive,
                run_path_filter=run_path_filter,
                artifact_filter=artifact_filter,
            )
        )
    if renderer_wanted in {None, CHASER_EXACT_SUCCESSOR_RENDERER}:
        options.extend(
            discover_exact_chaser_successor_options(
                archive,
                run_path_filter=run_path_filter,
                artifact_filter=artifact_filter,
            )
        )
    if renderer_wanted is None or renderer_wanted in CHASER_DASHBOARD_RENDERERS:
        options.extend(
            _discover_goodcopbadcop_chaser_specs_fast(
                root,
                archive,
                run_path_filter=run_path_filter,
                artifact_filter=artifact_filter,
            )
        )
    if renderer_wanted in {None, LEGACY_BOUT_PLOT_RENDERER, *BOUT_PLOT_RENDERERS}:
        options.extend(
            _discover_bout_kinematics_specs_fast(
                root,
                archive,
                renderer_filter=renderer_wanted,
                run_path_filter=run_path_filter,
                artifact_filter=artifact_filter,
            )
        )
    return sorted(
        options,
        key=lambda item: (
            not item.is_supported,
            item.renderer,
            item.run_path,
            item.artifact_name,
        ),
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
    if renderer_wanted in CHASER_DASHBOARD_RENDERERS:
        return _discover_goodcopbadcop_chaser_specs_fast(
            root,
            archive,
            run_path_filter=run_path_wanted,
            artifact_filter=artifact_wanted,
        )
    if renderer_wanted == PROVIDER_CHASER_CANDIDATE_RENDERER:
        return discover_provider_chaser_candidate_options(
            archive,
            run_path_filter=run_path_wanted,
            artifact_filter=artifact_wanted,
        )
    if renderer_wanted == CHASER_EXACT_SUCCESSOR_RENDERER:
        return discover_exact_chaser_successor_options(
            archive,
            run_path_filter=run_path_wanted,
            artifact_filter=artifact_wanted,
        )
    if renderer_wanted in {LEGACY_BOUT_PLOT_RENDERER, *BOUT_PLOT_RENDERERS}:
        return _discover_bout_kinematics_specs_fast(
            root,
            archive,
            renderer_filter=renderer_wanted,
            run_path_filter=run_path_wanted,
            artifact_filter=artifact_wanted,
        )
    options: list[InteractiveSpecOption] = []
    seen_paths: set[str] = set()
    inventory_paths = _inventory_interactive_artifact_paths(root)
    for artifact_path in inventory_paths:
        normalized_path = normalize_path(artifact_path)
        if not normalized_path or normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        option = _read_option(root, archive, normalized_path)
        if option is None:
            continue
        if renderer_wanted and option.renderer != renderer_wanted:
            continue
        if run_path_wanted and option.run_path != run_path_wanted:
            continue
        if artifact_wanted and artifact_wanted not in {
            option.artifact_name,
            option.artifact_path,
        }:
            continue
        options.append(option)
    if inventory_paths:
        return sorted(
            options,
            key=lambda item: (
                not item.is_supported,
                item.renderer,
                item.run_path,
                item.artifact_name,
            ),
        )

    for artifact in iter_visualization_artifacts(root):
        if normalize_path(artifact.path) in seen_paths:
            continue
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
        if artifact_wanted and artifact_wanted not in {
            option.artifact_name,
            option.artifact_path,
        }:
            continue
        options.append(option)
    return sorted(
        options,
        key=lambda item: (
            not item.is_supported,
            item.renderer,
            item.run_path,
            item.artifact_name,
        ),
    )


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
            candidates.update(
                path for path in zarr_dir.glob("*_analysis.zarr") if path.is_dir()
            )

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
        candidates.update(
            path for path in child.glob("*_analysis.zarr") if path.is_dir()
        )

    if not candidates and root.is_dir():
        candidates.update(
            path
            for path in root.rglob("*_analysis.zarr")
            if path.is_dir() and (not needle or needle in f"{path.name} {path}".lower())
        )

    return sorted(candidates)


def _registry_analysis_zarrs(
    registry_path: Path | str,
    *,
    name_contains: Optional[str],
    limit: int = 500,
) -> list[Path]:
    needle = str(name_contains or "").strip().lower()
    registry = resolve_registry_path(Path(registry_path), cwd=Path.cwd())
    sql = [
        "SELECT dataset_id, recording_id, zarr_path, zarr_use, status",
        "FROM datasets",
        "WHERE (status IS NULL OR status != 'missing')",
        "  AND (zarr_use = 'analysis' OR zarr_path LIKE '%_analysis.zarr')",
    ]
    params: list[Any] = []
    if needle:
        sql.append(
            "  AND (LOWER(COALESCE(dataset_id, '')) LIKE ? "
            "OR LOWER(COALESCE(recording_id, '')) LIKE ? "
            "OR LOWER(COALESCE(zarr_path, '')) LIKE ?)"
        )
        pattern = f"%{needle}%"
        params.extend([pattern, pattern, pattern])
    sql.append("ORDER BY COALESCE(recording_id, dataset_id), dataset_id")
    sql.append("LIMIT ?")
    params.append(int(limit))
    with open_readonly_connection(registry) as conn:
        rows = conn.execute(" ".join(sql), params).fetchall()

    candidates: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        zarr_path = str(row["zarr_path"] or "").strip()
        if not zarr_path:
            continue
        path = Path(zarr_path).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(path)
    return candidates


def discover_geometry_review_queue(
    registry_path: Path | str,
    *,
    include_inactive: bool = False,
):
    """Build the registered-geometry queue through Palette's read-only registry path."""

    from fisheye.registry.geometry_review import load_geometry_review_queue

    return load_geometry_review_queue(
        registry_path,
        include_inactive=include_inactive,
    )


def discover_protocol_recording_options(
    seed_zarr_path: Path | str,
    *,
    recordings_root: Optional[Path | str] = None,
    registry_path: Optional[Path | str] = None,
    renderer_filter: Optional[str] = None,
    run_path_filter: Optional[str] = None,
    artifact_filter: Optional[str] = None,
    name_contains: Optional[str] = "GoodCopBadCop",
    lazy_registry_specs: bool = True,
    recording_explorer_only: bool = False,
    include_collection: bool = True,
    include_seed_without_specs: bool = False,
) -> list[RecordingSpecOption]:
    """Find sibling recordings with matching interactive specs.

    The common organized-recording layout is:
    ``<recordings_root>/<recording_id>/zarr/<recording_id>_analysis.zarr``.
    The seed archive is always included as a candidate so explicit Zarr paths
    continue to work even when no sibling root can be inferred. Set
    ``include_collection=False`` for a direct single-recording launch; sibling
    discovery then occurs only when a recordings root or registry is supplied.
    """

    seed = Path(seed_zarr_path).expanduser()
    candidates = {seed}
    registry_candidates: set[Path] = set()
    if registry_path is not None:
        registry_candidates.update(
            _registry_analysis_zarrs(registry_path, name_contains=name_contains)
        )
        candidates.update(registry_candidates)
    elif recordings_root is not None or include_collection:
        root = (
            Path(recordings_root).expanduser()
            if recordings_root
            else infer_recordings_root_from_zarr_path(seed)
        )
        candidates.update(_candidate_analysis_zarrs(root, name_contains=name_contains))

    options: list[RecordingSpecOption] = []
    for archive in sorted(candidates):
        if (
            registry_path is not None
            and lazy_registry_specs
            and archive in registry_candidates
        ):
            recording_id = recording_id_from_analysis_zarr(archive)
            options.append(
                RecordingSpecOption(
                    zarr_path=archive,
                    recording_id=recording_id,
                    label=f"{recording_id} (registered; specs loaded on selection)",
                    interactive_spec_count=0,
                    supported_spec_count=0,
                    renderer_counts={},
                    spec_counts_loaded=False,
                )
            )
            continue
        discover_specs = (
            discover_recording_explorer_spec_options
            if recording_explorer_only
            else discover_interactive_spec_options
        )
        spec_options = discover_specs(
            archive,
            renderer_filter=renderer_filter,
            run_path_filter=run_path_filter,
            artifact_filter=artifact_filter,
        )
        if not spec_options:
            if include_seed_without_specs and archive == seed:
                recording_id = recording_id_from_analysis_zarr(archive)
                options.append(
                    RecordingSpecOption(
                        zarr_path=archive,
                        recording_id=recording_id,
                        label=f"{recording_id} (no supported interactive specs)",
                        interactive_spec_count=0,
                        supported_spec_count=0,
                        renderer_counts={},
                    )
                )
            continue
        renderer_counts: dict[str, int] = {}
        for spec in spec_options:
            renderer_counts[spec.renderer or "unknown"] = (
                renderer_counts.get(spec.renderer or "unknown", 0) + 1
            )
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


def group_options_by_renderer(
    options: Iterable[InteractiveSpecOption],
) -> dict[str, list[InteractiveSpecOption]]:
    grouped: dict[str, list[InteractiveSpecOption]] = {}
    for option in options:
        grouped.setdefault(option.renderer or "unknown", []).append(option)
    return grouped


def artifact_path_for(run_path: str, artifact_name: str) -> str:
    return join_path(run_path, "visualizations", artifact_name)
