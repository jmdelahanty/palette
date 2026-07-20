"""Read-only discovery of runs, entities, stimulus steps, and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Iterable, Mapping

from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.shared.zarr_helpers import (
    safe_int,
    zarr_attrs_dict,
    zarr_child_group,
    zarr_group_keys,
)
from fisheye.shared.zarr_run_completion import (
    AUTHORITATIVE_RUN_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_LATEST_COMPLETE_ATTR,
)

from .models import (
    AnalysisFamilySpec,
    ArtifactReference,
    ChaserEntity,
    ResolvedRun,
    StimulusStep,
)


_INCOMPLETE_STATUSES = {"failed", "running", "pending", "incomplete"}


@dataclass(frozen=True)
class RunHandle:
    reference: ResolvedRun
    group: Any
    root: Any
    attrs: Mapping[str, Any]
    parent_order: int
    authoritative: bool


def _array_keys(group: Any | None) -> list[str]:
    if group is None:
        return []
    method = getattr(group, "array_keys", None)
    if callable(method):
        try:
            return sorted(str(value) for value in method())
        except Exception:
            return []
    return []


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_schema_version(value: Any) -> int | None:
    return safe_int(value)


def _first_present(attrs: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = attrs.get(name)
        if value is not None:
            return value
    return None


def _entity_from_attrs(
    attrs: Mapping[str, Any],
    entity_id_attrs: Iterable[str],
) -> str | None:
    for name in entity_id_attrs:
        value = attrs.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parent_authoritative_name(parent: Any) -> str | None:
    attrs = zarr_attrs_dict(parent)
    for key in (AUTHORITATIVE_RUN_ATTR, RUN_LATEST_COMPLETE_ATTR, "latest"):
        value = _as_text(attrs.get(key))
        if value:
            return value
    return None


def list_family_runs(root: Any, spec: AnalysisFamilySpec) -> tuple[RunHandle, ...]:
    """List usable run candidates for a logical family without mutating the store."""

    if spec.family_id == "stimulus.chaser_distance":
        # Reporting is a normal presentation consumer, not an inventory bypass.
        # Pay the full canonical base-publication verification cost, then expose
        # no raw child while its dashboard/component artifacts remain unsealed.
        try:
            load_chaser_distance_run(root, run_name="latest")
        except ChaserDistanceReadError:
            return ()
        return ()

    handles: list[RunHandle] = []
    for parent_order, parent_path in enumerate(spec.run_parent_paths):
        parent = zarr_child_group(root, parent_path)
        if parent is None:
            continue
        authoritative_name = _parent_authoritative_name(parent)
        for run_id in zarr_group_keys(parent):
            group = zarr_child_group(parent, run_id)
            if group is None:
                continue
            attrs = zarr_attrs_dict(group)
            completion = str(attrs.get(RUN_COMPLETION_STATUS_ATTR) or "").strip().lower()
            if completion in _INCOMPLETE_STATUSES:
                continue
            source_fingerprint = _as_text(
                attrs.get("source_fingerprint")
                or attrs.get("source_lineage_hash")
                or attrs.get("lineage_hash")
            )
            entity_id = _entity_from_attrs(attrs, spec.entity_id_attrs)
            authoritative = run_id == authoritative_name
            handles.append(
                RunHandle(
                    reference=ResolvedRun(
                        family_id=spec.family_id,
                        run_id=run_id,
                        path=f"{parent_path}/{run_id}",
                        selection=(
                            "resolved_authoritative"
                            if authoritative
                            else "available_complete"
                        ),
                        schema_id=_as_text(
                            attrs.get("schema_id") or attrs.get("output_schema_id")
                        ),
                        schema_version=_as_schema_version(
                            attrs.get("schema_version")
                            or attrs.get("output_schema_version")
                        ),
                        method=_as_text(attrs.get("method")),
                        method_version=_as_text(attrs.get("method_version")),
                        source_fingerprint=source_fingerprint,
                        lineage_hash=_as_text(
                            attrs.get("lineage_hash")
                            or attrs.get("source_lineage_hash")
                        ),
                        entity_id=entity_id,
                    ),
                    group=group,
                    root=root,
                    attrs=attrs,
                    parent_order=parent_order,
                    authoritative=authoritative,
                )
            )
    return tuple(handles)


def select_family_run(
    handles: Iterable[RunHandle],
    spec: AnalysisFamilySpec,
    *,
    entity_id: str | None = None,
) -> RunHandle | None:
    """Select a deterministic concrete run, optionally for one entity."""

    candidates = list(handles)
    if entity_id is not None and spec.entity_id_attrs:
        candidates = [
            handle
            for handle in candidates
            if handle.reference.entity_id == str(entity_id)
        ]
    if not candidates:
        return None

    def rank(handle: RunHandle) -> tuple[int, int, str, str]:
        created_at = _as_text(
            handle.attrs.get("created_at_utc") or handle.attrs.get("created_utc")
        ) or ""
        return (
            -int(handle.parent_order),
            int(handle.authoritative),
            created_at,
            handle.reference.run_id,
        )

    selected = max(candidates, key=rank)
    if not selected.authoritative:
        reference = ResolvedRun(
            **{
                **selected.reference.__dict__,
                "selection": (
                    "resolved_latest_compatible"
                    if entity_id is not None and spec.entity_id_attrs
                    else "resolved_latest_available"
                ),
            }
        )
        selected = RunHandle(
            reference=reference,
            group=selected.group,
            root=selected.root,
            attrs=selected.attrs,
            parent_order=selected.parent_order,
            authoritative=selected.authoritative,
        )
    return selected


def discover_track_ids(track_run: RunHandle | None) -> tuple[str, ...]:
    if track_run is None:
        return ()
    tracks = zarr_child_group(track_run.group, "tracks")
    values: list[tuple[int, str]] = []
    for name in zarr_group_keys(tracks):
        text = str(name)
        numeric = safe_int(text[3:] if text.startswith("id_") else text)
        if numeric is not None:
            values.append((numeric, str(numeric)))
    return tuple(value for _, value in sorted(values))


def discover_stimulus_steps(stimulus_run: RunHandle | None) -> tuple[StimulusStep, ...]:
    if stimulus_run is None:
        return ()
    steps_group = zarr_child_group(stimulus_run.group, "steps")
    steps: list[StimulusStep] = []
    for ordinal, name in enumerate(zarr_group_keys(steps_group)):
        group = zarr_child_group(steps_group, name)
        attrs = zarr_attrs_dict(group)
        step_index = safe_int(attrs.get("step_index"))
        if step_index is None:
            step_index = safe_int(str(name).replace("step_", "", 1))
        if step_index is None:
            step_index = ordinal
        mode = str(
            attrs.get("stimulus_mode")
            or attrs.get("stimulus_mode_str")
            or "UNKNOWN"
        ).strip().upper()
        steps.append(
            StimulusStep(
                step_index=int(step_index),
                step_name=str(attrs.get("step_name") or name),
                stimulus_mode=mode,
                start_frame=safe_int(
                    _first_present(attrs, "start_camera_frame", "start_frame")
                ),
                end_frame=safe_int(
                    _first_present(attrs, "end_camera_frame", "end_frame")
                ),
                duration_s=(
                    float(attrs["duration_s"])
                    if attrs.get("duration_s") is not None
                    else None
                ),
            )
        )
    return tuple(sorted(steps, key=lambda step: step.step_index))


def discover_chasers(stimulus_run: RunHandle | None) -> tuple[ChaserEntity, ...]:
    """Fail closed until behavior roles have sealed protocol authority.

    Raw ``protocol_json`` can describe authoring intent, but it is not a
    payload-bound scientific identity/behavior contract and reporting must not
    promote it into presentation labels.
    """

    return ()


def missing_source_requirements(root: Any, spec: AnalysisFamilySpec) -> tuple[str, ...]:
    missing: list[str] = []
    for requirement in spec.source_requirements:
        if not any(
            zarr_child_group(root, path) is not None
            for path in requirement.any_group_paths
        ):
            missing.append(requirement.source_id)
    return tuple(missing)


def _iter_artifact_nodes(
    group: Any,
    *,
    relative_path: str = "",
    depth: int = 0,
    max_depth: int = 7,
) -> Iterable[tuple[str, Any]]:
    if depth > max_depth:
        return
    for name in _array_keys(group):
        path = f"{relative_path}/{name}".strip("/")
        try:
            yield path, group[name]
        except Exception:
            continue
    for name in zarr_group_keys(group):
        child = zarr_child_group(group, name)
        if child is None:
            continue
        path = f"{relative_path}/{name}".strip("/")
        attrs = zarr_attrs_dict(child)
        if attrs.get("artifact_type") == "visualization":
            yield path, child
        yield from _iter_artifact_nodes(
            child,
            relative_path=path,
            depth=depth + 1,
            max_depth=max_depth,
        )


def find_artifacts(
    run: RunHandle,
    *,
    path_pattern: str,
    entity_id: str | None,
) -> tuple[ArtifactReference, ...]:
    pattern = path_pattern.format(entity_id=entity_id if entity_id is not None else "*")
    container_group = run.group
    container_path = run.reference.path
    expected_motion_authority: Mapping[str, Any] | None = None
    if (
        run.reference.family_id == "core.track_kinematics"
        and pattern.startswith("visualizations/")
    ):
        parts = run.reference.path.split("/")
        track_id = safe_int(entity_id)
        if (
            len(parts) != 4
            or parts[:2] != ["analysis", "track_kinematics_runs"]
            or track_id is None
            or track_id < 0
        ):
            return ()
        visualization_parent_path = (
            "analysis/track_kinematics_visualization_runs/"
            f"{parts[2]}/{parts[3]}/tracks/id_{track_id}"
        )
        parent = zarr_child_group(run.root, visualization_parent_path)
        if parent is None:
            return ()
        render_name = _parent_authoritative_name(parent)
        render = zarr_child_group(parent, render_name) if render_name else None
        if render is None:
            return ()
        render_attrs = zarr_attrs_dict(render)
        if (
            str(render_attrs.get(RUN_COMPLETION_STATUS_ATTR) or "").lower()
            != "complete"
            or render_attrs.get("stage_selector_eligible") is not True
        ):
            return ()
        authority = render_attrs.get("source_track_motion_authority")
        track_ref = f"/{run.reference.path}/tracks/id_{track_id}"
        run_ref = f"/{run.reference.path}"
        positions_px_ref = f"{track_ref}/positions_px"
        positions_mm_ref = f"{track_ref}/positions_mm"
        track_group = zarr_child_group(
            run.group,
            f"tracks/id_{track_id}",
        )
        positions_px = (
            track_group.get("positions_px") if track_group is not None else None
        )
        positions_px_attrs = zarr_attrs_dict(positions_px)
        positions_mm = (
            track_group.get("positions_mm") if track_group is not None else None
        )
        positions_mm_attrs = zarr_attrs_dict(positions_mm)
        if (
            not isinstance(authority, Mapping)
            or authority.get("run_ref") != run_ref
            or authority.get("track_ref") != track_ref
            or safe_int(authority.get("track_id")) != track_id
            or authority.get("schema_id")
            != "palette.track_motion_read_authority"
            or safe_int(authority.get("schema_version")) != 1
            or authority.get("motion_manifest_ref")
            != f"{run_ref}@track_motion_publication_manifest"
            or authority.get("positions_px_ref") != positions_px_ref
            or authority.get("track_sample_key_ref")
            != f"{track_ref}/track_sample_key"
            or authority.get("source_acquisition_frame_index_ref")
            != f"{track_ref}/source_acquisition_frame_index"
            or run.attrs.get("coordinate_binding_status") != "bound_canonical_v2"
            or str(run.attrs.get(RUN_COMPLETION_STATUS_ATTR) or "").lower()
            != "complete"
            or run.attrs.get("stage_selector_eligible") is not True
            or authority.get("motion_manifest_sha256")
            != run.attrs.get("track_motion_publication_manifest_sha256")
            or authority.get("positions_px_coordinate_descriptor_sha256")
            != positions_px_attrs.get("coordinate_descriptor_sha256")
            or (
                positions_mm is None
                and (
                    authority.get("positions_mm_ref") is not None
                    or authority.get(
                        "positions_mm_coordinate_descriptor_sha256"
                    )
                    is not None
                )
            )
            or (
                positions_mm is not None
                and (
                    authority.get("positions_mm_ref") != positions_mm_ref
                    or authority.get(
                        "positions_mm_coordinate_descriptor_sha256"
                    )
                    != positions_mm_attrs.get("coordinate_descriptor_sha256")
                )
            )
            or safe_int(render_attrs.get("track_id")) != track_id
        ):
            return ()
        expected_motion_authority = authority
        container_group = render
        container_path = f"{visualization_parent_path}/{render_name}"

    found: list[ArtifactReference] = []
    for relative_path, node in _iter_artifact_nodes(container_group):
        if not fnmatch(relative_path, pattern):
            continue
        attrs = zarr_attrs_dict(node)
        if (
            expected_motion_authority is not None
            and attrs.get("track_motion_authority")
            != expected_motion_authority
        ):
            continue
        found.append(
            ArtifactReference(
                path=f"{container_path}/{relative_path}",
                artifact_name=relative_path.rsplit("/", 1)[-1],
                artifact_role=_as_text(attrs.get("artifact_role")),
                visualization_contract_id=_as_text(
                    attrs.get("visualization_contract_id")
                ),
                renderer=_as_text(attrs.get("renderer")),
                renderer_version=_as_text(attrs.get("renderer_version")),
                artifact_signature=_as_text(attrs.get("artifact_signature")),
                content_sha256=_as_text(attrs.get("content_sha256")),
            )
        )
    return tuple(sorted(found, key=lambda artifact: artifact.path))


def choose_artifact(
    artifacts: Iterable[ArtifactReference],
    *,
    expected_contract_id: str | None,
    expected_renderer: str | None,
    expected_renderer_version: str | None,
) -> ArtifactReference | None:
    candidates = list(artifacts)
    if not candidates:
        return None

    def rank(artifact: ArtifactReference) -> tuple[int, int, int, str]:
        return (
            int(
                expected_contract_id is not None
                and artifact.visualization_contract_id == expected_contract_id
            ),
            int(
                expected_renderer is not None
                and artifact.renderer == expected_renderer
            ),
            int(
                expected_renderer_version is not None
                and artifact.renderer_version == expected_renderer_version
            ),
            artifact.path,
        )

    return max(candidates, key=rank)
