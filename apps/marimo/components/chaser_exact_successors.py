"""Read-only Marimo views for paired exact composable chaser successors.

The spatial-occupancy successor is the bundle anchor: its sealed scientific
manifest names the ordered keypoint/detection relative-frame and radial
children.  This consumer resolves no selector and never recomputes a
scientific product.  It deep-audits the small successor tables and content-
hashes only the relative-frame arrays needed by the selected visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    CHASER_RELATIVE_FRAME_RUN_PARENT_PATH,
    MANIFEST_ATTR as RELATIVE_MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR as RELATIVE_MANIFEST_DIGEST_ATTR,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    load_composable_chaser_successor_source_handle,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingProof,
    require_same_exact_relative_frame_child,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

from .common import normalize_path
from .registry import (
    CHASER_EXACT_SUCCESSOR_RENDERER,
    InteractiveSpecOption,
)

ANALYSIS_IDS = (
    "radial_near_field",
    "distance_traces",
    "trajectory_overlays",
    "provenance",
)
PROVIDER_ROLES = ("keypoint", "detection")
RELATIVE_PARENT = CHASER_RELATIVE_FRAME_RUN_PARENT_PATH
RADIAL_PARENT = "analysis/chaser_radial_near_field_runs"
SPATIAL_PARENT = "analysis/chaser_spatial_occupancy_runs"
TRACE_MAX_POINTS = 6_000
TRAJECTORY_MAX_POINTS = 15_000

_FRAME_ARRAY_NAMES = (
    "acquisition_frame_id",
    "timestamp_ns",
    "timestamp_valid",
    "selection_member",
    "fish_position_xy_px",
    "fish_position_valid",
    "chaser_identity_code",
    "chaser_behavior_role_code",
    "chaser_occurrence_member",
    "chaser_position_xy_px",
    "chaser_position_valid",
    "relative_distance_physical",
    "relative_physical_valid",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one object.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _exact_child_path(value: Any, *, parent: str, label: str) -> tuple[str, str]:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ValueError(f"{label} must be one exact child path.")
    prefix = f"{parent}/"
    name = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not name
        or "/" in name
        or name in {".", "..", "latest", "default", "selected"}
    ):
        raise ValueError(f"{label} must be one exact child below {parent!r}.")
    return value, name


def _option_bundle(option: InteractiveSpecOption) -> tuple[str, str]:
    if option.renderer != CHASER_EXACT_SUCCESSOR_RENDERER:
        raise ValueError("Selected source is not an exact chaser-successor bundle.")
    return _exact_child_path(
        normalize_path(option.run_path),
        parent=SPATIAL_PARENT,
        label="spatial bundle run",
    )


@dataclass(frozen=True)
class RelativeFrameProjection:
    run_path: str
    run_name: str
    recording_id: str
    manifest_sha256: str
    n_frames: int
    n_chasers: int
    source_authorities: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    def frame_chaser(self, name: str) -> np.ndarray:
        values = self.arrays[name]
        return values.reshape((self.n_frames, self.n_chasers) + values.shape[1:])

    def collapsed_frame(self, name: str) -> np.ndarray:
        values = self.frame_chaser(name)
        reference = values[:, :1, ...]
        if values.dtype.kind == "f":
            repeated = np.array_equal(
                values, np.broadcast_to(reference, values.shape), equal_nan=True
            )
        else:
            repeated = np.array_equal(values, np.broadcast_to(reference, values.shape))
        if not repeated:
            raise ValueError(
                f"Exact relative-frame array {name!r} differs across chaser rows."
            )
        return values[:, 0, ...]


@dataclass(frozen=True)
class ExactChaserSuccessorProjection:
    analysis_id: str
    recording_id: str
    spatial: Any
    radials: tuple[Any, Any]
    relatives: tuple[RelativeFrameProjection, RelativeFrameProjection] | None
    provider_ids: tuple[str, str]
    epoch_records: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]


def _load_targeted_relative(
    archive: Path,
    *,
    run_path: str,
    expected_manifest_sha256: str,
    expected_recording_id: str,
) -> RelativeFrameProjection:
    """Deep-read only arrays required by these interactive display projections."""

    exact_path, run_name = _exact_child_path(
        run_path, parent=RELATIVE_PARENT, label="relative-frame run"
    )
    validate_direct_consolidated_subtree(archive, subtree_path=exact_path)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[exact_path]
    attrs = dict(getattr(run, "attrs", {}))
    manifest = _mapping(attrs.get(RELATIVE_MANIFEST_ATTR), label="relative manifest")
    manifest_digest = _digest(
        attrs.get(RELATIVE_MANIFEST_DIGEST_ATTR), label="relative manifest digest"
    )
    if manifest_digest != expected_manifest_sha256:
        raise ValueError(
            "Relative-frame manifest digest differs from its bundle binding."
        )
    if canonical_json_sha256(_plain(manifest)) != manifest_digest:
        raise ValueError("Relative-frame manifest digest is stale.")
    if (
        attrs.get("schema_id") != "palette.analysis.chaser_relative_frame"
        or attrs.get("schema_version") != 1
        or attrs.get("run_path") != exact_path
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        raise ValueError("Relative-frame source is not one complete exact candidate.")
    if manifest.get("recording_id") != expected_recording_id:
        raise ValueError("Relative-frame source belongs to another recording.")
    dimensions = _mapping(manifest.get("dimensions"), label="relative dimensions")
    n_frames = int(dimensions.get("n_frames", 0))
    n_chasers = int(dimensions.get("n_chasers", 0))
    n_rows = int(dimensions.get("n_rows", 0))
    if n_frames <= 0 or n_chasers <= 0 or n_rows != n_frames * n_chasers:
        raise ValueError("Relative-frame dimensions are invalid.")
    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list):
        raise ValueError("Relative-frame manifest lacks array declarations.")
    declaration_by_path = {
        item.get("path"): item for item in declarations if isinstance(item, Mapping)
    }
    if len(declaration_by_path) != len(declarations):
        raise ValueError("Relative-frame array declarations are duplicated or invalid.")
    base = run["base"]
    arrays: dict[str, np.ndarray] = {}
    for name in _FRAME_ARRAY_NAMES:
        path = f"base/{name}"
        declaration = _mapping(
            declaration_by_path.get(path), label=f"declaration {path}"
        )
        values = np.asarray(base[name][...])
        if (
            values.dtype.str != declaration.get("dtype")
            or list(values.shape) != declaration.get("shape")
            or values.shape[0] != n_rows
        ):
            raise ValueError(f"Relative-frame array {path!r} metadata changed.")
        if array_values_sha256(values) != declaration.get("content_sha256"):
            raise ValueError(f"Relative-frame array {path!r} content digest changed.")
        copied = np.array(values, copy=True, order="C")
        copied.setflags(write=False)
        arrays[name] = copied
    authorities = _mapping(
        manifest.get("source_authorities"), label="relative source authorities"
    )
    return RelativeFrameProjection(
        run_path=exact_path,
        run_name=run_name,
        recording_id=expected_recording_id,
        manifest_sha256=manifest_digest,
        n_frames=n_frames,
        n_chasers=n_chasers,
        source_authorities=_freeze(authorities),
        arrays=MappingProxyType(arrays),
    )


def _source_records(spatial: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    scientific = spatial.scientific_manifest
    sources = _mapping(scientific.get("sources"), label="spatial sources")
    providers = sources.get("position_providers")
    if not isinstance(providers, (list, tuple)) or len(providers) != 2:
        raise ValueError("Spatial bundle must bind exactly two position providers.")
    records = tuple(_mapping(value, label="spatial provider") for value in providers)
    if tuple(record.get("provider_role") for record in records) != PROVIDER_ROLES:
        raise ValueError(
            "Spatial bundle provider roles are not keypoint then detection."
        )
    return records  # type: ignore[return-value]


def _verify_bundle_children(
    spatial: Any,
    radials: Sequence[Any],
    relative_bindings: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], tuple[ExactRelativeFrameBindingProof, ...]]:
    records = _source_records(spatial)
    if len(radials) != 2 or len(relative_bindings) != 2:
        raise ValueError("Exact chaser bundle is incomplete.")
    provider_ids: list[str] = []
    semantic_binding: Mapping[str, Any] | None = None
    geometry_binding: Mapping[str, Any] | None = None
    epoch_records: Any = None
    arena: Any = None
    relative_binding_proofs: list[ExactRelativeFrameBindingProof] = []
    for record, radial, relative_binding in zip(
        records, radials, relative_bindings, strict=True
    ):
        radial_binding = _mapping(
            record.get("radial_near_field"), label="radial binding"
        )
        expected_relative = _mapping(
            record.get("relative_frame"), label="relative binding"
        )
        require_same_exact_relative_frame_child(
            expected_relative,
            relative_binding,
            expected_label="spatial relative-frame binding",
            observed_label="projection relative-frame binding",
        )
        if radial.run_path != radial_binding.get(
            "run_path"
        ) or radial.manifest_sha256 != radial_binding.get("manifest_sha256"):
            raise ValueError("Spatial bundle child digest binding is stale.")
        scientific = radial.scientific_manifest
        provider = _mapping(
            scientific.get("position_provider"), label="radial provider"
        )
        if (
            provider.get("status") != "first_class_explicit_authority"
            or provider.get("provider_id") != record.get("provider_id")
            or provider.get("provider_digest") != record.get("provider_digest")
        ):
            raise ValueError(
                "Radial successor position authority differs from the bundle."
            )
        radial_sources = _mapping(scientific.get("sources"), label="radial sources")
        relative_binding_proof = require_same_exact_relative_frame_child(
            expected_relative,
            _mapping(
                radial_sources.get("relative_frame"), label="radial relative source"
            ),
            expected_label="spatial relative-frame binding",
            observed_label="radial relative-frame binding",
        )
        local_semantic = _mapping(
            radial_sources.get("protocol_semantic_selection"),
            label="radial semantic selection",
        )
        local_geometry = _mapping(
            radial_sources.get("arena_geometry_and_scale"),
            label="radial arena geometry",
        )
        if semantic_binding is None:
            semantic_binding = local_semantic
            geometry_binding = local_geometry
            epoch_records = _plain(scientific.get("epoch_records"))
            arena = _plain(scientific.get("arena"))
        elif (
            dict(local_semantic) != dict(semantic_binding)
            or dict(local_geometry) != dict(geometry_binding or {})
            or _plain(scientific.get("epoch_records")) != epoch_records
            or _plain(scientific.get("arena")) != arena
        ):
            raise ValueError(
                "Paired radial successors do not share exact epochs and arena."
            )
        provider_ids.append(str(provider["provider_id"]))
        relative_binding_proofs.append(relative_binding_proof)
    if len(set(provider_ids)) != 2:
        raise ValueError("Exact chaser bundle providers are not distinct.")
    spatial_sources = _mapping(
        spatial.scientific_manifest.get("sources"), label="spatial sources"
    )
    spatial_semantic = _mapping(
        spatial_sources.get("protocol_semantic_selection"),
        label="spatial semantic selection",
    )
    spatial_geometry = _mapping(
        spatial_sources.get("arena_geometry_and_scale"),
        label="spatial arena geometry",
    )
    if (
        dict(spatial_semantic) != dict(semantic_binding or {})
        or dict(spatial_geometry) != dict(geometry_binding or {})
        or _plain(spatial.scientific_manifest.get("epoch_records")) != epoch_records
    ):
        raise ValueError(
            "Spatial and radial successors use different semantic epochs or geometry."
        )
    return (
        (provider_ids[0], provider_ids[1]),
        tuple(relative_binding_proofs),
    )


def available_exact_chaser_successor_analysis_ids(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
) -> tuple[str, ...]:
    del zarr_path
    _option_bundle(option)
    if option.spec.get("bundle_status") != "exact_selector_ineligible":
        return ()
    return ANALYSIS_IDS


def load_exact_chaser_successor_projection(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
    *,
    analysis_id: str,
) -> ExactChaserSuccessorProjection:
    """Load one exact, selector-free visualization projection."""

    if analysis_id not in ANALYSIS_IDS:
        raise ValueError(f"Unsupported exact chaser analysis {analysis_id!r}.")
    archive = Path(zarr_path).expanduser().resolve()
    spatial_path, spatial_name = _option_bundle(option)
    spatial = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_spatial_occupancy",
        run_name=spatial_name,
        deep_audit=True,
    )
    option_digest = _digest(
        option.spec.get("bundle_manifest_sha256"), label="bundle option digest"
    )
    if spatial.run_path != spatial_path or spatial.manifest_sha256 != option_digest:
        raise ValueError("Selected exact chaser bundle changed after discovery.")
    records = _source_records(spatial)
    relative_bindings = tuple(
        _mapping(record.get("relative_frame"), label="relative binding")
        for record in records
    )
    radial_bindings = tuple(
        _mapping(record.get("radial_near_field"), label="radial binding")
        for record in records
    )
    radials = tuple(
        load_composable_chaser_successor_source_handle(
            archive,
            successor_kind="chaser_radial_near_field",
            run_name=_exact_child_path(
                binding.get("run_path"), parent=RADIAL_PARENT, label="radial run"
            )[1],
            expected_recording_id=spatial.recording_id,
            deep_audit=True,
        )
        for binding in radial_bindings
    )
    provider_ids, relative_binding_proofs = _verify_bundle_children(
        spatial, radials, relative_bindings
    )
    relatives = None
    if analysis_id in {"distance_traces", "trajectory_overlays"}:
        relatives = tuple(
            _load_targeted_relative(
                archive,
                run_path=str(binding["run_path"]),
                expected_manifest_sha256=_digest(
                    binding.get("manifest_sha256"), label="relative binding digest"
                ),
                expected_recording_id=spatial.recording_id,
            )
            for binding in relative_bindings
        )
        for record, relative in zip(records, relatives, strict=True):
            authority = _mapping(
                relative.source_authorities.get("fish_position"),
                label="relative fish authority",
            )
            if authority.get("provider_id") != record.get(
                "provider_id"
            ) or authority.get("provider_digest") != record.get("provider_digest"):
                raise ValueError(
                    "Relative-frame fish authority differs from the bundle."
                )
        if (
            relatives[0].n_frames != relatives[1].n_frames
            or relatives[0].n_chasers != relatives[1].n_chasers
        ):
            raise ValueError("Paired relative-frame dimensions differ.")
        for name in (
            "acquisition_frame_id",
            "timestamp_ns",
            "timestamp_valid",
            "selection_member",
            "chaser_identity_code",
            "chaser_behavior_role_code",
            "chaser_occurrence_member",
            "chaser_position_xy_px",
            "chaser_position_valid",
        ):
            if not np.array_equal(relatives[0].arrays[name], relatives[1].arrays[name]):
                raise ValueError(f"Paired exact chaser evidence differs for {name!r}.")
    epoch_records = spatial.scientific_manifest.get("epoch_records")
    if not isinstance(epoch_records, (list, tuple)) or not epoch_records:
        raise ValueError("Exact chaser bundle lacks epoch records.")
    return ExactChaserSuccessorProjection(
        analysis_id=analysis_id,
        recording_id=spatial.recording_id,
        spatial=spatial,
        radials=radials,  # type: ignore[arg-type]
        relatives=relatives,  # type: ignore[arg-type]
        provider_ids=provider_ids,
        epoch_records=tuple(_freeze(record) for record in epoch_records),
        provenance=_freeze(
            {
                "bundle_run_path": spatial.run_path,
                "bundle_manifest_sha256": spatial.manifest_sha256,
                "radial_run_paths": [radial.run_path for radial in radials],
                "radial_manifest_sha256": [
                    radial.manifest_sha256 for radial in radials
                ],
                "relative_bindings": [_plain(value) for value in relative_bindings],
                "relative_binding_proofs": [
                    _plain(proof.provenance_record())
                    for proof in relative_binding_proofs
                ],
                "adapter_semantics": (
                    "read_only_exact_children_no_selector_no_interpolation"
                ),
                "display_trace_algorithm": (
                    "source_order_bucket_first_last_min_max_missing_break_v1"
                ),
                "display_trace_max_points_per_series": TRACE_MAX_POINTS,
                "display_trajectory_algorithm": (
                    "source_order_uniform_plus_coordinate_extrema_v1"
                ),
                "display_trajectory_max_points_per_panel": TRAJECTORY_MAX_POINTS,
            }
        ),
    )


def _registry(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    registries = manifest.get("identity_registries")
    if not isinstance(registries, Mapping):
        return {}
    value = registries.get(name)
    return value if isinstance(value, Mapping) else {}


def _metric_rows(handle: Any) -> dict[tuple[int, int, int], dict[str, float]]:
    epoch = np.asarray(handle.array("metric_epoch_role_code"), dtype=np.int64)
    behavior = np.asarray(handle.array("metric_behavior_role_code"), dtype=np.int64)
    chaser = np.asarray(handle.array("metric_chaser_identity_code"), dtype=np.int64)
    names = (
        "distance_p25_mm",
        "distance_p50_mm",
        "distance_p75_mm",
        "near_zone_fraction_valid",
        "near_zone_dwell_s",
        "near_zone_entry_rate_per_min_valid_time",
    )
    columns = {
        name: np.asarray(handle.array(f"metric_{name}"), dtype=np.float64)
        for name in names
    }
    if any(
        values.size != epoch.size for values in (*columns.values(), behavior, chaser)
    ):
        raise ValueError("Radial metric columns have inconsistent lengths.")
    return {
        (int(epoch[index]), int(behavior[index]), int(chaser[index])): {
            name: float(values[index]) for name, values in columns.items()
        }
        for index in range(epoch.size)
    }


def _stratum_label(handle: Any, key: tuple[int, int, int]) -> str:
    epochs = _registry(handle.scientific_manifest, "epoch_role")
    behaviors = _registry(handle.scientific_manifest, "behavior_role")
    return (
        f"{epochs.get(str(key[0]), f'epoch {key[0]}')} · "
        f"{behaviors.get(str(key[1]), f'role {key[1]}')} · chaser {key[2]}"
    )


def _trace_display_projection(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    *,
    max_points: int = TRACE_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound display size while preserving extrema and every observed gap."""

    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    observed = (
        np.asarray(valid, dtype=bool).reshape(-1)
        & np.isfinite(x_values)
        & np.isfinite(y_values)
    )
    if not (x_values.size == y_values.size == observed.size):
        raise ValueError("Trace display vectors have inconsistent lengths.")
    if x_values.size <= max_points:
        output = y_values.copy()
        output[~observed] = np.nan
        return x_values.copy(), output
    bucket_count = max(1, max_points // 4)
    edges = np.linspace(0, x_values.size, bucket_count + 1, dtype=np.int64)
    selected: set[int] = set()
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        candidates = np.flatnonzero(observed[start:end]) + start
        if not candidates.size:
            continue
        selected.update((int(candidates[0]), int(candidates[-1])))
        local = y_values[candidates]
        selected.add(int(candidates[int(np.argmin(local))]))
        selected.add(int(candidates[int(np.argmax(local))]))
    indices = np.asarray(sorted(selected), dtype=np.int64)
    projected_x: list[float] = []
    projected_y: list[float] = []
    previous: int | None = None
    for index in indices.tolist():
        if previous is not None and np.any(~observed[previous + 1 : index]):
            projected_x.append(float(x_values[previous]))
            projected_y.append(float("nan"))
        projected_x.append(float(x_values[index]))
        projected_y.append(float(y_values[index]))
        previous = index
    return np.asarray(projected_x), np.asarray(projected_y)


def _trajectory_display_indices(
    xy: np.ndarray,
    valid: np.ndarray,
    *,
    max_points: int = TRAJECTORY_MAX_POINTS,
) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    observed = np.asarray(valid, dtype=bool).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != observed.size:
        raise ValueError("Trajectory display vectors have inconsistent shapes.")
    candidates = np.flatnonzero(observed & np.all(np.isfinite(points), axis=1))
    if candidates.size <= max_points:
        return candidates
    chosen = set(
        int(value)
        for value in candidates[
            np.linspace(0, candidates.size - 1, max_points, dtype=np.int64)
        ].tolist()
    )
    for column in range(2):
        chosen.add(int(candidates[int(np.argmin(points[candidates, column]))]))
        chosen.add(int(candidates[int(np.argmax(points[candidates, column]))]))
    return np.asarray(sorted(chosen), dtype=np.int64)


def build_exact_radial_near_field_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    handles = projection.radials
    rows = tuple(_metric_rows(handle) for handle in handles)
    if set(rows[0]) != set(rows[1]) or not rows[0]:
        raise ValueError("Paired radial products expose different or empty strata.")
    keys = sorted(rows[0])
    labels = [_stratum_label(handles[0], key) for key in keys]
    colors = ("#1f77b4", "#d95f02")

    distance = go.Figure()
    near = go.Figure()
    visits = go.Figure()
    radial = go.Figure()
    for provider_index, (provider_id, provider_rows, handle) in enumerate(
        zip(projection.provider_ids, rows, handles, strict=True)
    ):
        median = np.asarray([provider_rows[key]["distance_p50_mm"] for key in keys])
        low = np.asarray([provider_rows[key]["distance_p25_mm"] for key in keys])
        high = np.asarray([provider_rows[key]["distance_p75_mm"] for key in keys])
        distance.add_trace(
            go.Scatter(
                x=labels,
                y=median,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": high - median,
                    "arrayminus": median - low,
                },
                mode="markers",
                name=provider_id,
                marker={"color": colors[provider_index], "size": 9},
            )
        )
        near.add_trace(
            go.Bar(
                x=labels,
                y=[provider_rows[key]["near_zone_fraction_valid"] for key in keys],
                name=f"{provider_id} · fraction",
                marker_color=colors[provider_index],
            )
        )
        visits.add_trace(
            go.Bar(
                x=labels,
                y=[provider_rows[key]["near_zone_dwell_s"] for key in keys],
                name=f"{provider_id} · dwell",
                marker_color=colors[provider_index],
            )
        )
        visits.add_trace(
            go.Scatter(
                x=labels,
                y=[
                    provider_rows[key]["near_zone_entry_rate_per_min_valid_time"]
                    for key in keys
                ],
                name=f"{provider_id} · entries/min",
                mode="lines+markers",
                line={
                    "color": colors[provider_index],
                    "dash": ("solid", "dash")[provider_index],
                },
                yaxis="y2",
            )
        )
        epoch = np.asarray(handle.array("radial_epoch_role_code"), dtype=np.int64)
        behavior = np.asarray(handle.array("radial_behavior_role_code"), dtype=np.int64)
        chaser = np.asarray(handle.array("radial_chaser_identity_code"), dtype=np.int64)
        start = np.asarray(handle.array("radial_bin_start_mm"), dtype=np.float64)
        end = np.asarray(handle.array("radial_bin_end_mm"), dtype=np.float64)
        selection = np.asarray(
            handle.array("radial_selection_index_geometric"), dtype=np.float64
        )
        for key, label in zip(keys, labels, strict=True):
            mask = (epoch == key[0]) & (behavior == key[1]) & (chaser == key[2])
            order = np.argsort(start[mask])
            radial.add_trace(
                go.Scatter(
                    x=((start[mask][order] + end[mask][order]) / 2.0),
                    y=selection[mask][order],
                    mode="lines",
                    name=f"{label} · {provider_id}",
                    line={
                        "color": colors[provider_index],
                        "dash": ("solid", "dash")[provider_index],
                    },
                )
            )
    figure_meta = _plain(projection.provenance)
    distance.update_layout(
        title="Fish–chaser distance median and IQR",
        yaxis_title="distance (mm)",
        meta=figure_meta,
    )
    radial.update_layout(
        title="Area-corrected moving-chaser radial selection",
        xaxis_title="distance (mm)",
        yaxis_title="geometric selection index",
        meta=figure_meta,
    )
    near.update_layout(
        title="Near-field occupancy",
        barmode="group",
        yaxis={"title": "fraction of valid rows"},
        meta=figure_meta,
    )
    visits.update_layout(
        title="Exact-session-time near-field dwell and entry rate",
        barmode="group",
        yaxis={"title": "dwell (s)"},
        yaxis2={"title": "entries/min valid time", "overlaying": "y", "side": "right"},
        meta=figure_meta,
    )
    return mo.vstack(
        [
            mo.callout(
                "Persisted paired-provider summaries; exact session time, reviewed moving-reference geometry, no interpolation.",
                kind="info",
            ),
            distance,
            radial,
            near,
            visits,
        ]
    )


def build_exact_distance_traces_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    if projection.relatives is None:
        raise ValueError("Distance traces require exact relative-frame sources.")
    from plotly.subplots import make_subplots

    keypoint = projection.relatives[0]
    frame_id = keypoint.collapsed_frame("acquisition_frame_id").astype(np.int64)
    timestamp = keypoint.collapsed_frame("timestamp_ns").astype(np.int64)
    timestamp_valid = keypoint.collapsed_frame("timestamp_valid").astype(bool)
    selected = keypoint.collapsed_frame("selection_member").astype(bool)
    valid_timestamp_rows = np.flatnonzero(timestamp_valid)
    if not valid_timestamp_rows.size:
        raise ValueError("Exact relative-frame source has no valid session time.")
    time_s = (
        timestamp.astype(np.float64) - float(timestamp[valid_timestamp_rows[0]])
    ) / 1e9
    identities = keypoint.frame_chaser("chaser_identity_code")
    roles = keypoint.frame_chaser("chaser_behavior_role_code")
    if not np.all(identities == identities[:1]) or not np.all(roles == roles[:1]):
        raise ValueError("Exact chaser identity or behavior roles change by frame.")
    registry = _registry(projection.radials[0].scientific_manifest, "behavior_role")
    panels: list[tuple[str, np.ndarray]] = [
        ("full recording", np.ones(frame_id.size, dtype=bool))
    ]
    for record in projection.epoch_records:
        panels.append(
            (
                str(record["analysis_role"]),
                selected
                & (frame_id >= int(record["start_frame"]))
                & (frame_id < int(record["end_frame_exclusive"])),
            )
        )
    titles = [
        f"{panel} · {registry.get(str(int(roles[0, column])), f'role {int(roles[0, column])}')} · chaser {int(identities[0, column])}"
        for panel, _ in panels
        for column in range(keypoint.n_chasers)
    ]
    figure = make_subplots(
        rows=len(panels), cols=keypoint.n_chasers, subplot_titles=titles
    )
    colors = ("#1f77b4", "#d95f02")
    for row_index, (_, panel_mask) in enumerate(panels, start=1):
        indices = np.flatnonzero(panel_mask)
        if not indices.size:
            raise ValueError("An exact distance panel has no source rows.")
        for column in range(keypoint.n_chasers):
            for provider_index, (provider_id, relative) in enumerate(
                zip(projection.provider_ids, projection.relatives, strict=True)
            ):
                values = relative.frame_chaser("relative_distance_physical")[:, column]
                valid = (
                    relative.frame_chaser("relative_physical_valid")[:, column]
                    & relative.frame_chaser("chaser_occurrence_member")[:, column]
                    & timestamp_valid
                )
                display_x, display_y = _trace_display_projection(
                    time_s[indices], values[indices], valid[indices]
                )
                figure.add_trace(
                    go.Scattergl(
                        x=display_x,
                        y=display_y,
                        mode="lines",
                        name=provider_id,
                        legendgroup=provider_id,
                        showlegend=row_index == 1 and column == 0,
                        connectgaps=False,
                        line={"color": colors[provider_index], "width": 1},
                    ),
                    row=row_index,
                    col=column + 1,
                )
    figure.update_xaxes(title_text="session time from first valid timestamp (s)")
    figure.update_yaxes(title_text="distance (mm)")
    figure.update_layout(
        title=f"Full-recording and exact-epoch fish–chaser distance · {projection.recording_id}",
        height=280 * len(panels),
        meta=_plain(projection.provenance),
    )
    return mo.vstack(
        [
            mo.callout(
                f"Display-only extrema-preserving projection, at most {TRACE_MAX_POINTS:,} source points per trace; missing rows always break lines.",
                kind="info",
            ),
            figure,
        ]
    )


def build_exact_trajectory_overlays_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    if projection.relatives is None:
        raise ValueError("Trajectory overlays require exact relative-frame sources.")
    from plotly.subplots import make_subplots

    keypoint = projection.relatives[0]
    frame_id = keypoint.collapsed_frame("acquisition_frame_id").astype(np.int64)
    selected = keypoint.collapsed_frame("selection_member").astype(bool)
    chaser_xy = keypoint.frame_chaser("chaser_position_xy_px")
    chaser_valid = keypoint.frame_chaser(
        "chaser_position_valid"
    ) & keypoint.frame_chaser("chaser_occurrence_member")
    identities = keypoint.frame_chaser("chaser_identity_code")
    roles = keypoint.frame_chaser("chaser_behavior_role_code")
    registry = _registry(projection.radials[0].scientific_manifest, "behavior_role")
    arena = _mapping(
        projection.radials[0].scientific_manifest.get("arena"), label="reviewed arena"
    )
    center_x = float(arena["center_x_px"])
    center_y = float(arena["center_y_px"])
    radius = float(arena["radius_px"])
    titles = [
        f"{record['analysis_role']} · {provider_id}"
        for provider_id in projection.provider_ids
        for record in projection.epoch_records
    ]
    figure = make_subplots(
        rows=2, cols=len(projection.epoch_records), subplot_titles=titles
    )
    chaser_colors = ("#2ca02c", "#9467bd", "#8c564b", "#e377c2")
    for provider_index, (provider_id, relative) in enumerate(
        zip(projection.provider_ids, projection.relatives, strict=True), start=1
    ):
        fish_xy = relative.collapsed_frame("fish_position_xy_px")
        fish_valid = relative.collapsed_frame("fish_position_valid").astype(bool)
        for epoch_index, record in enumerate(projection.epoch_records, start=1):
            mask = (
                selected
                & (frame_id >= int(record["start_frame"]))
                & (frame_id < int(record["end_frame_exclusive"]))
            )
            rows = np.flatnonzero(mask)
            if not rows.size:
                raise ValueError("An exact trajectory panel has no source rows.")
            fish_display = _trajectory_display_indices(fish_xy[rows], fish_valid[rows])
            source_rows = rows[fish_display]
            figure.add_trace(
                go.Scattergl(
                    x=fish_xy[source_rows, 0],
                    y=fish_xy[source_rows, 1],
                    mode="markers",
                    name=f"fish · {provider_id}",
                    legendgroup=f"fish-{provider_id}",
                    showlegend=epoch_index == 1,
                    marker={"color": "#222222", "size": 2, "opacity": 0.25},
                ),
                row=provider_index,
                col=epoch_index,
            )
            for column in range(keypoint.n_chasers):
                local_valid = chaser_valid[rows, column]
                local_xy = chaser_xy[rows, column]
                display = _trajectory_display_indices(local_xy, local_valid)
                chaser_rows = rows[display]
                role = registry.get(
                    str(int(roles[0, column])), f"role {int(roles[0, column])}"
                )
                label = f"{role} · chaser {int(identities[0, column])}"
                figure.add_trace(
                    go.Scattergl(
                        x=chaser_xy[chaser_rows, column, 0],
                        y=chaser_xy[chaser_rows, column, 1],
                        mode="markers",
                        name=label,
                        legendgroup=label,
                        showlegend=provider_index == 1 and epoch_index == 1,
                        marker={
                            "color": chaser_colors[column % len(chaser_colors)],
                            "size": 3,
                            "opacity": 0.55,
                        },
                    ),
                    row=provider_index,
                    col=epoch_index,
                )
            figure.add_shape(
                type="circle",
                x0=center_x - radius,
                x1=center_x + radius,
                y0=center_y - radius,
                y1=center_y + radius,
                line={"color": "#666666", "width": 1},
                row=provider_index,
                col=epoch_index,
            )
            figure.update_xaxes(
                range=[center_x - radius * 1.03, center_x + radius * 1.03],
                row=provider_index,
                col=epoch_index,
            )
            figure.update_yaxes(
                range=[center_y + radius * 1.03, center_y - radius * 1.03],
                scaleanchor=f"x{(provider_index - 1) * len(projection.epoch_records) + epoch_index}",
                scaleratio=1,
                row=provider_index,
                col=epoch_index,
            )
    figure.update_layout(
        title=f"Exact-epoch fish positions with logged chaser overlays · {projection.recording_id}",
        height=820,
        meta=_plain(projection.provenance),
    )
    return mo.vstack(
        [
            mo.callout(
                f"Display-only deterministic source-order projection, at most {TRAJECTORY_MAX_POINTS:,} valid points per series and panel; scientific occupancy remains in the persisted successor.",
                kind="info",
            ),
            figure,
        ]
    )


__all__ = [
    "ANALYSIS_IDS",
    "ExactChaserSuccessorProjection",
    "RelativeFrameProjection",
    "available_exact_chaser_successor_analysis_ids",
    "build_exact_distance_traces_output",
    "build_exact_radial_near_field_output",
    "build_exact_trajectory_overlays_output",
    "load_exact_chaser_successor_projection",
]
