"""Verified logical readers for ``analysis/track_kinematics_runs``.

Normal scientific readers cross the typed full-motion publication boundary in
this module.  Physical array names and ``latest`` pointers are discovery only:
the exact selected child must be complete, selector-eligible, coordinate-bound,
and freshly match its immutable payload/derivation manifest before any values
are returned.

Historical layout fallback remains available only through the explicitly named
inspection loader.  It must not be used by normal future-recording workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import zarr

from fisheye.shared.coordinate_descriptor import CanonicalCoordinateDescriptor
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name

TRACK_KINEMATICS_LAYOUT_HIERARCHICAL_V1 = "hierarchical_v1"
TRACK_KINEMATICS_SPEED_LEVELS = ("raw", "filtered", "smoothed", "averaged")
TRACK_KINEMATICS_SOURCE_SPEED_LEVELS = {
    "raw": "speed_raw",
    "filtered": "speed_filtered",
    "smoothed": "speed_smoothed",
    "averaged": "speed_averaged",
}
TRACK_KINEMATICS_GROUPED_SPEED_LAYOUT = "movement/speed/<level>"


@dataclass(frozen=True)
class TrackKinematicsTrackTables:
    """Logical arrays for one track-kinematics track."""

    run_name: str
    scope: str
    run_path: str
    track_id: int
    track_path: str
    run_attrs: Mapping[str, Any]
    track_attrs: Mapping[str, Any]
    authority_status: str
    motion_manifest_sha256: Optional[str]
    positions_px_descriptor_sha256: Optional[str]
    positions_mm_descriptor_sha256: Optional[str]
    positions_px_descriptor: Optional[CanonicalCoordinateDescriptor]
    positions_mm_descriptor: Optional[CanonicalCoordinateDescriptor]
    track_sample_key: Optional[np.ndarray]
    source_acquisition_frame_index: Optional[np.ndarray]
    source_frame_interpolation: Optional[np.ndarray]
    source_instance_key: Optional[np.ndarray]
    source_row_index: Optional[np.ndarray]
    frame_indices: np.ndarray
    speed_mm_by_level: Mapping[str, np.ndarray]
    speed_px_by_level: Mapping[str, np.ndarray]
    frame_path_distance_mm_by_level: Mapping[str, np.ndarray]
    frame_path_distance_px_by_level: Mapping[str, np.ndarray]
    acceleration_mm_by_level: Mapping[str, np.ndarray]
    acceleration_px_by_level: Mapping[str, np.ndarray]
    smoothed_acceleration_mm_by_level: Mapping[str, np.ndarray]
    smoothed_acceleration_px_by_level: Mapping[str, np.ndarray]
    delta_seconds: Optional[np.ndarray]
    transition_valid: Optional[np.ndarray]
    sample_valid: Optional[np.ndarray]
    time_seconds: Optional[np.ndarray]
    heading_degrees: Optional[np.ndarray]
    heading_radians: Optional[np.ndarray]
    smoothed_heading_degrees: Optional[np.ndarray]
    smoothed_heading_radians: Optional[np.ndarray]
    delta_heading_degrees: Optional[np.ndarray]
    delta_heading_smoothed_degrees: Optional[np.ndarray]
    angular_velocity_deg_s: Optional[np.ndarray]
    angular_velocity_smoothed_deg_s: Optional[np.ndarray]
    angular_speed_raw_deg_s: Optional[np.ndarray]
    angular_speed_smoothed_deg_s: Optional[np.ndarray]
    detection_source: Optional[np.ndarray]
    sample_reason_code: Optional[np.ndarray]
    transition_reason_code: Optional[np.ndarray]
    positions_mm: Optional[np.ndarray]
    positions_px: Optional[np.ndarray]
    cumulative_path_distance_mm: Optional[np.ndarray]
    cumulative_path_distance_px: Optional[np.ndarray]

    def require_direct_source_camera_positions_px(
        self,
    ) -> tuple[np.ndarray, int, int]:
        """Return pixels and their exact camera extent for direct overlays.

        A normal presentation/scientific reader must not infer camera suitability
        from the ``positions_px`` name or from root video dimensions. Texture,
        canvas, projector, and transform-required positions are rejected here.
        """

        descriptor = self.positions_px_descriptor
        if descriptor is None or self.positions_px is None:
            raise ValueError(
                f"{self.track_path} has no verified pixel-position descriptor."
            )
        if (
            descriptor.space_id != "source_camera_image_px"
            or descriptor.source_camera_overlay.status != "direct"
        ):
            raise ValueError(
                f"{self.track_path}/positions_px is not directly suitable for "
                "source-camera presentation."
            )
        if (
            descriptor.geometry_type != "point_xy"
            or descriptor.components != ("x", "y")
            or descriptor.component_units != ("px", "px")
        ):
            raise ValueError(
                f"{self.track_path}/positions_px has unsupported point geometry semantics."
            )
        extent = descriptor.reference_extent
        width = extent.width
        height = extent.height
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, (int, float))
            or not isinstance(height, (int, float))
            or not float(width).is_integer()
            or not float(height).is_integer()
            or int(width) <= 0
            or int(height) <= 0
            or extent.units != "px"
        ):
            raise ValueError(
                f"{self.track_path}/positions_px has no exact positive pixel reference extent."
            )
        return self.positions_px, int(width), int(height)

    def authority_record(self) -> dict[str, Any]:
        """Return the compact immutable authority inherited by derived writers."""

        if (
            self.authority_status != "verified_canonical_track_motion_v1"
            or not self.motion_manifest_sha256
            or not self.positions_px_descriptor_sha256
        ):
            raise ValueError(
                "Only a freshly verified canonical track-motion read can mint "
                "downstream authority."
            )
        run_ref = f"/{self.run_path}"
        track_ref = f"/{self.track_path}"
        return {
            "schema_id": "palette.track_motion_read_authority",
            "schema_version": 1,
            "run_ref": run_ref,
            "track_ref": track_ref,
            "track_id": int(self.track_id),
            "motion_manifest_ref": (
                f"{run_ref}@track_motion_publication_manifest"
            ),
            "motion_manifest_sha256": self.motion_manifest_sha256,
            "positions_px_ref": f"{track_ref}/positions_px",
            "positions_px_coordinate_descriptor_sha256": (
                self.positions_px_descriptor_sha256
            ),
            "positions_mm_ref": (
                f"{track_ref}/positions_mm"
                if self.positions_mm_descriptor_sha256 is not None
                else None
            ),
            "positions_mm_coordinate_descriptor_sha256": (
                self.positions_mm_descriptor_sha256
            ),
            "track_sample_key_ref": f"{track_ref}/track_sample_key",
            "source_acquisition_frame_index_ref": (
                f"{track_ref}/source_acquisition_frame_index"
            ),
        }

    def speed_level_dict(self) -> dict[str, Optional[np.ndarray]]:
        """Return the legacy speed-dict shape expected by bout detectors."""

        out: dict[str, Optional[np.ndarray]] = {"frames": self.frame_indices}
        for level in TRACK_KINEMATICS_SPEED_LEVELS:
            source_level = TRACK_KINEMATICS_SOURCE_SPEED_LEVELS[level]
            out[f"{source_level}_mm"] = self.speed_mm_by_level.get(level)
            out[f"{source_level}_px"] = self.speed_px_by_level.get(level)
            out[f"frame_path_distance_{level}_mm"] = self.frame_path_distance_mm_by_level.get(level)
            out[f"frame_path_distance_{level}_px"] = self.frame_path_distance_px_by_level.get(level)
        out["delta_seconds"] = self.delta_seconds
        out["transition_valid"] = self.transition_valid
        out["sample_valid"] = self.sample_valid
        return out


def _group_attrs(group: zarr.Group) -> dict[str, Any]:
    return dict(group.attrs.asdict() if hasattr(group.attrs, "asdict") else dict(group.attrs))


def _optional_array(group: zarr.Group, name: str) -> Optional[np.ndarray]:
    if name not in group:
        return None
    return np.asarray(group[name][:])


def _require_array(group: zarr.Group, name: str, *, label: str) -> np.ndarray:
    values = _optional_array(group, name)
    if values is None:
        raise ValueError(f"{label} is missing required array '{name}'")
    return values


def _resolve_track_kinematics_parent(root: zarr.Group) -> zarr.Group:
    try:
        return root["analysis"]["track_kinematics_runs"]
    except Exception as exc:
        raise ValueError("No analysis/track_kinematics_runs group found") from exc


def resolve_track_kinematics_run(
    root: zarr.Group,
    *,
    run_name: str = "latest",
    scope: str = "offline",
    historical_inspection: bool = False,
) -> tuple[zarr.Group, str, str]:
    """Resolve an exact track-kinematics child.

    Normal implicit resolution requires the root-qualified selector pair and
    both scope selectors to agree.  The explicit historical inspection path
    retains raw scope-``latest`` behavior without granting coordinate
    authority.
    """

    parent = _resolve_track_kinematics_parent(root)
    if scope not in {"online", "offline"}:
        raise ValueError("Track-kinematics scope must be 'online' or 'offline'.")
    if scope not in parent:
        raise ValueError(f"No {scope!r} track_kinematics_runs group found")
    scope_group = parent[scope]
    resolved_name = run_name
    if run_name == "latest":
        if historical_inspection:
            resolved_name = scope_group.attrs.get("latest")
            if not resolved_name:
                raise ValueError(f"No latest {scope!r} track_kinematics run found")
        else:
            qualified_name = resolve_latest_complete_run_name(
                parent,
                legacy_default=False,
            )
            prefix = f"{scope}/"
            if (
                not qualified_name
                or not str(qualified_name).startswith(prefix)
                or not str(qualified_name)[len(prefix) :]
                or "/" in str(qualified_name)[len(prefix) :]
            ):
                raise ValueError(
                    f"No stable complete selector-eligible {scope!r} track "
                    "kinematics run is selected; root selector activation may "
                    "be in progress, so retry the read."
                )
            resolved_name = str(qualified_name)[len(prefix) :]
            if (
                parent.attrs.get(f"latest_{scope}") != resolved_name
                or scope_group.attrs.get("latest") != resolved_name
            ):
                raise ValueError(
                    f"No stable {scope!r} track-kinematics selector agreement; "
                    "selector activation may be in progress, so retry the read."
                )
    else:
        run_spec = str(run_name).strip()
        prefix = f"analysis/track_kinematics_runs/{scope}/"
        if "/" not in run_spec:
            resolved_name = run_spec
        elif (
            run_spec.startswith(prefix)
            and run_spec[len(prefix) :]
            and "/" not in run_spec[len(prefix) :]
        ):
            resolved_name = run_spec[len(prefix) :]
        else:
            raise ValueError(
                "Track-kinematics run must be a bare child name or the exact "
                f"path {prefix}<run>; got {run_name!r}."
            )
    resolved_name = str(resolved_name)
    if resolved_name not in scope_group:
        raise ValueError(f"Track kinematics run {resolved_name!r} not found under {scope!r}")
    run_path = f"analysis/track_kinematics_runs/{scope}/{resolved_name}"
    run_group = scope_group[resolved_name]
    if run_name == "latest" and not historical_inspection:
        qualified_name = f"{scope}/{resolved_name}"
        if (
            resolve_latest_complete_run_name(parent, legacy_default=False)
            != qualified_name
            or parent.attrs.get(f"latest_{scope}") != resolved_name
            or scope_group.attrs.get("latest") != resolved_name
        ):
            raise ValueError(
                f"The selected {scope!r} track-kinematics publication changed "
                "during resolution; retry the read."
            )
    return run_group, resolved_name, run_path


def list_track_ids(run_group: zarr.Group) -> list[int]:
    """Return available track ids from a current hierarchical run."""

    if "track_ids" in run_group:
        return [int(value) for value in np.asarray(run_group["track_ids"][:]).tolist()]
    if "tracks" not in run_group:
        return []
    ids: list[int] = []
    for key in run_group["tracks"].group_keys():
        key_str = str(key)
        if key_str.startswith("id_"):
            try:
                ids.append(int(key_str.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(ids)


def _load_legacy_speed_level_for_inspection(
    track_group: zarr.Group,
    *,
    level: str,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Load one unverified historical speed level for inspection only.

    Prefer the grouped v2 ``movement/speed/<level>`` surface. Fall back to the
    legacy flat arrays to keep historical runs readable.
    """

    source_level = TRACK_KINEMATICS_SOURCE_SPEED_LEVELS[level]
    grouped = None
    if "movement" in track_group:
        movement = track_group["movement"]
        if "speed" in movement and level in movement["speed"]:
            grouped = movement["speed"][level]

    if grouped is not None:
        speed_mm = _optional_array(grouped, "mm")
        speed_px = _optional_array(grouped, "px")
        path_mm = _optional_array(grouped, "frame_path_distance_mm")
        path_px = _optional_array(grouped, "frame_path_distance_px")
        accel_mm = _optional_array(grouped, "acceleration_mm")
        accel_px = _optional_array(grouped, "acceleration_px")
        smooth_accel_mm = _optional_array(grouped, "smoothed_acceleration_mm")
        smooth_accel_px = _optional_array(grouped, "smoothed_acceleration_px")
    else:
        speed_mm = speed_px = path_mm = path_px = None
        accel_mm = accel_px = smooth_accel_mm = smooth_accel_px = None

    if speed_mm is None:
        speed_mm = _optional_array(track_group, f"{source_level}_mm")
    if speed_px is None:
        speed_px = _optional_array(track_group, f"{source_level}_px")
    if path_mm is None:
        path_mm = _optional_array(track_group, f"frame_path_distance_{level}_mm")
    if path_px is None:
        path_px = _optional_array(track_group, f"frame_path_distance_{level}_px")
    if level == "smoothed":
        if accel_mm is None:
            accel_mm = _optional_array(track_group, "acceleration_mm")
        if accel_px is None:
            accel_px = _optional_array(track_group, "acceleration_px")
        if smooth_accel_mm is None:
            smooth_accel_mm = _optional_array(track_group, "smoothed_acceleration_mm")
        if smooth_accel_px is None:
            smooth_accel_px = _optional_array(track_group, "smoothed_acceleration_px")
    return speed_mm, speed_px, path_mm, path_px, accel_mm, accel_px, smooth_accel_mm, smooth_accel_px


def _validate_requested_speed_levels(
    required_speed_levels: Iterable[str],
) -> tuple[str, ...]:
    requested = tuple(
        dict.fromkeys(str(level).strip() for level in required_speed_levels)
    )
    unsupported = tuple(
        level for level in requested if level not in TRACK_KINEMATICS_SPEED_LEVELS
    )
    if unsupported:
        supported = ", ".join(TRACK_KINEMATICS_SPEED_LEVELS)
        raise ValueError(
            "Unsupported physical track speed level(s): "
            f"{', '.join(unsupported)}. Expected a subset of: {supported}."
        )
    return requested


def _copy_bound_surface(track: Any, relative_path: str) -> np.ndarray:
    """Copy one loader-authorized surface from its exact live node."""

    surface = track.surface(relative_path)
    node = surface.node
    values = np.array(node[:], copy=True, order="C")
    if (
        values.shape != surface.shape
        or values.dtype.str != surface.dtype
        or values.shape != tuple(int(value) for value in node.shape)
        or values.dtype != np.dtype(node.dtype)
        or array_values_sha256(values) != surface.content_sha256
    ):
        raise ValueError(
            f"Verified track-motion surface /{node.path} changed payload, dtype, or shape while reading."
        )
    return values


def _copy_optional_bound_surface(
    track: Any,
    relative_path: str,
) -> Optional[np.ndarray]:
    try:
        return _copy_bound_surface(track, relative_path)
    except KeyError:
        return None


def _manifest_owned_attrs(
    bound_run: Any,
    bound_track: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return immutable attrs from the verified manifest, never live handles."""

    manifest = bound_run.manifest
    root_binding = manifest.get("run_root_attrs")
    root_record = (
        root_binding.get("record")
        if isinstance(root_binding, Mapping)
        else None
    )
    immutable = (
        root_record.get("immutable_attrs")
        if isinstance(root_record, Mapping)
        else None
    )
    legacy_record = (
        root_record.get("legacy_compatibility")
        if isinstance(root_record, Mapping)
        else None
    )
    legacy = (
        legacy_record.get("attrs")
        if isinstance(legacy_record, Mapping)
        else None
    )
    tracks = manifest.get("tracks")
    track_record = (
        tracks.get(f"id_{int(bound_track.track_id)}")
        if isinstance(tracks, Mapping)
        else None
    )
    groups = (
        track_record.get("groups")
        if isinstance(track_record, Mapping)
        else None
    )
    track_root = groups.get(".") if isinstance(groups, Mapping) else None
    track_attrs = (
        track_root.get("attrs")
        if isinstance(track_root, Mapping)
        else None
    )
    if not all(
        isinstance(value, Mapping)
        for value in (immutable, legacy, track_attrs)
    ):
        raise ValueError(
            "Verified track-motion manifest lacks exact run/track attribute snapshots."
        )
    run_attrs = dict(immutable)
    for name, value in legacy.items():
        if name in run_attrs:
            raise ValueError("Track-motion manifest attr partitions overlap.")
        run_attrs[str(name)] = value
    run_attrs["track_motion_publication_manifest_sha256"] = (
        bound_run.manifest_sha256
    )
    run_attrs["coordinate_binding_status"] = manifest.get(
        "coordinate_binding_status"
    )
    return run_attrs, track_attrs


def load_track_kinematics_track(
    root: zarr.Group,
    *,
    run_name: str = "latest",
    scope: str = "offline",
    track_id: int = 0,
    required_speed_levels: Iterable[str] = TRACK_KINEMATICS_SPEED_LEVELS,
) -> TrackKinematicsTrackTables:
    """Load one freshly verified canonical track-motion publication.

    This is the normal reader boundary.  Missing canonical authority is an
    error; callers that are explicitly auditing historical layouts must invoke
    :func:`load_legacy_track_kinematics_track_for_inspection` instead.
    """

    requested_speed_levels = _validate_requested_speed_levels(
        required_speed_levels
    )

    run_group, resolved_name, run_path = resolve_track_kinematics_run(
        root,
        run_name=run_name,
        scope=scope,
    )
    # Import locally so this small reader module remains acyclic for producer
    # tooling that imports logical table types.
    from fisheye.analysis.track_kinematics import load_bound_track_motion_run

    bound_run = load_bound_track_motion_run(root, run_group)
    # All later metadata comes from the freshly resolved bound child or its
    # manifest snapshot, never the discovery handle supplied above.
    run_group = bound_run.run_group
    try:
        bound_track = bound_run.track(int(track_id))
    except KeyError as exc:
        available = [int(value.track_id) for value in bound_run.tracks]
        raise ValueError(
            f"Track id_{int(track_id)} not found in track kinematics run "
            f"{resolved_name!r}; "
            f"available track ids: {available}"
        ) from exc

    track_path = f"{run_path}/tracks/id_{int(track_id)}"
    frame_indices = _copy_bound_surface(bound_track, "frame_indices").astype(
        np.int64, copy=False
    )
    source_acquisition_frame_index = _copy_bound_surface(
        bound_track, "source_acquisition_frame_index"
    ).astype(np.int64, copy=False)
    if not np.array_equal(frame_indices, source_acquisition_frame_index):
        raise ValueError(
            f"{track_path}/frame_indices is not the exact declared alias of "
            "source_acquisition_frame_index."
        )

    speed_mm_by_level: dict[str, np.ndarray] = {}
    speed_px_by_level: dict[str, np.ndarray] = {}
    frame_path_distance_mm_by_level: dict[str, np.ndarray] = {}
    frame_path_distance_px_by_level: dict[str, np.ndarray] = {}
    acceleration_mm_by_level: dict[str, np.ndarray] = {}
    acceleration_px_by_level: dict[str, np.ndarray] = {}
    smoothed_acceleration_mm_by_level: dict[str, np.ndarray] = {}
    smoothed_acceleration_px_by_level: dict[str, np.ndarray] = {}
    for level in TRACK_KINEMATICS_SPEED_LEVELS:
        prefix = f"movement/speed/{level}"
        speed_px = _copy_bound_surface(bound_track, f"{prefix}/px")
        speed_mm = _copy_optional_bound_surface(bound_track, f"{prefix}/mm")
        path_px = _copy_optional_bound_surface(
            bound_track, f"{prefix}/frame_path_distance_px"
        )
        path_mm = _copy_optional_bound_surface(
            bound_track, f"{prefix}/frame_path_distance_mm"
        )
        accel_px = _copy_bound_surface(bound_track, f"{prefix}/acceleration_px")
        accel_mm = _copy_optional_bound_surface(
            bound_track, f"{prefix}/acceleration_mm"
        )
        smooth_accel_px = _copy_bound_surface(
            bound_track, f"{prefix}/smoothed_acceleration_px"
        )
        smooth_accel_mm = _copy_optional_bound_surface(
            bound_track, f"{prefix}/smoothed_acceleration_mm"
        )
        if speed_mm is not None:
            speed_mm_by_level[level] = speed_mm
        if speed_px is not None:
            speed_px_by_level[level] = speed_px
        if path_mm is not None:
            frame_path_distance_mm_by_level[level] = path_mm
        if path_px is not None:
            frame_path_distance_px_by_level[level] = path_px
        if accel_mm is not None:
            acceleration_mm_by_level[level] = accel_mm
        if accel_px is not None:
            acceleration_px_by_level[level] = accel_px
        if smooth_accel_mm is not None:
            smoothed_acceleration_mm_by_level[level] = smooth_accel_mm
        if smooth_accel_px is not None:
            smoothed_acceleration_px_by_level[level] = smooth_accel_px

    position_binding = bound_track.position_binding
    positions_mm_binding = position_binding.positions_mm
    for required in requested_speed_levels:
        source_level = TRACK_KINEMATICS_SOURCE_SPEED_LEVELS[required]
        if required not in speed_px_by_level:
            raise ValueError(
                f"{track_path} is missing required speed level "
                f"'{source_level}_px'"
            )
        if positions_mm_binding is not None and required not in speed_mm_by_level:
            raise ValueError(
                f"{track_path} has physical position authority but is missing "
                f"required speed level '{source_level}_mm'"
            )
    run_attrs, track_attrs = _manifest_owned_attrs(bound_run, bound_track)
    result = TrackKinematicsTrackTables(
        run_name=resolved_name,
        scope=scope,
        run_path=run_path,
        track_id=int(track_id),
        track_path=track_path,
        run_attrs=run_attrs,
        track_attrs=track_attrs,
        authority_status="verified_canonical_track_motion_v1",
        motion_manifest_sha256=bound_run.manifest_sha256,
        positions_px_descriptor_sha256=(
            position_binding.positions_px.descriptor.digest()
        ),
        positions_mm_descriptor_sha256=(
            positions_mm_binding.descriptor.digest()
            if positions_mm_binding is not None
            else None
        ),
        positions_px_descriptor=position_binding.positions_px.descriptor,
        positions_mm_descriptor=(
            positions_mm_binding.descriptor
            if positions_mm_binding is not None
            else None
        ),
        track_sample_key=_copy_bound_surface(bound_track, "track_sample_key"),
        source_acquisition_frame_index=source_acquisition_frame_index,
        source_frame_interpolation=_copy_bound_surface(
            bound_track, "source_frame_interpolation"
        ),
        source_instance_key=_copy_bound_surface(bound_track, "source_instance_key"),
        source_row_index=_copy_bound_surface(bound_track, "source_row_index"),
        frame_indices=frame_indices,
        speed_mm_by_level=speed_mm_by_level,
        speed_px_by_level=speed_px_by_level,
        frame_path_distance_mm_by_level=frame_path_distance_mm_by_level,
        frame_path_distance_px_by_level=frame_path_distance_px_by_level,
        acceleration_mm_by_level=acceleration_mm_by_level,
        acceleration_px_by_level=acceleration_px_by_level,
        smoothed_acceleration_mm_by_level=smoothed_acceleration_mm_by_level,
        smoothed_acceleration_px_by_level=smoothed_acceleration_px_by_level,
        delta_seconds=_copy_bound_surface(bound_track, "delta_seconds"),
        transition_valid=_copy_bound_surface(bound_track, "transition_valid"),
        sample_valid=_copy_bound_surface(bound_track, "sample_valid"),
        time_seconds=_copy_bound_surface(bound_track, "time_seconds"),
        heading_degrees=_copy_bound_surface(bound_track, "heading_degrees"),
        heading_radians=_copy_bound_surface(bound_track, "heading_radians"),
        smoothed_heading_degrees=_copy_bound_surface(
            bound_track, "smoothed_heading_degrees"
        ),
        smoothed_heading_radians=_copy_bound_surface(
            bound_track, "smoothed_heading_radians"
        ),
        delta_heading_degrees=_copy_bound_surface(
            bound_track, "delta_heading_degrees"
        ),
        delta_heading_smoothed_degrees=_copy_bound_surface(
            bound_track, "delta_heading_smoothed_degrees"
        ),
        angular_velocity_deg_s=_copy_bound_surface(
            bound_track, "angular_velocity_deg_s"
        ),
        angular_velocity_smoothed_deg_s=_copy_bound_surface(
            bound_track, "angular_velocity_smoothed_deg_s"
        ),
        angular_speed_raw_deg_s=_copy_bound_surface(
            bound_track, "angular_speed_raw_deg_s"
        ),
        angular_speed_smoothed_deg_s=_copy_bound_surface(
            bound_track, "angular_speed_smoothed_deg_s"
        ),
        detection_source=_copy_bound_surface(bound_track, "detection_source"),
        sample_reason_code=_copy_bound_surface(
            bound_track, "sample_reason_code"
        ),
        transition_reason_code=_copy_bound_surface(
            bound_track, "transition_reason_code"
        ),
        positions_mm=_copy_optional_bound_surface(bound_track, "positions_mm"),
        positions_px=_copy_bound_surface(bound_track, "positions_px"),
        cumulative_path_distance_mm=_copy_optional_bound_surface(
            bound_track, "cumulative_path_distance_mm"
        ),
        cumulative_path_distance_px=_copy_bound_surface(
            bound_track, "cumulative_path_distance_px"
        ),
    )
    # Revalidate after every value and metadata copy.  A concurrent replacement
    # or mutation is therefore rejected instead of returning a mixed snapshot.
    bound_run.assert_verified()
    return result


def load_legacy_track_kinematics_track_for_inspection(
    root: zarr.Group,
    *,
    run_name: str = "latest",
    scope: str = "offline",
    track_id: int = 0,
    required_speed_levels: Iterable[str] = TRACK_KINEMATICS_SPEED_LEVELS,
) -> TrackKinematicsTrackTables:
    """Read an unverified historical physical layout for audit/migration only.

    This function intentionally does not mint canonical authority.  Normal
    scientific and presentation code must call :func:`load_track_kinematics_track`.
    """

    requested_speed_levels = _validate_requested_speed_levels(
        required_speed_levels
    )
    run_group, resolved_name, run_path = resolve_track_kinematics_run(
        root,
        run_name=run_name,
        scope=scope,
        historical_inspection=True,
    )
    if "tracks" not in run_group:
        raise ValueError(f"Track kinematics run {resolved_name!r} has no tracks group")
    track_key = f"id_{int(track_id)}"
    tracks = run_group["tracks"]
    if track_key not in tracks:
        available = list_track_ids(run_group)
        raise ValueError(
            f"Track {track_key} not found in track kinematics run {resolved_name!r}; "
            f"available track ids: {available}"
        )
    track_group = tracks[track_key]
    track_path = f"{run_path}/tracks/{track_key}"
    label = track_path
    frame_indices = _require_array(track_group, "frame_indices", label=label).astype(
        np.int64, copy=False
    )

    speed_mm_by_level: dict[str, np.ndarray] = {}
    speed_px_by_level: dict[str, np.ndarray] = {}
    frame_path_distance_mm_by_level: dict[str, np.ndarray] = {}
    frame_path_distance_px_by_level: dict[str, np.ndarray] = {}
    acceleration_mm_by_level: dict[str, np.ndarray] = {}
    acceleration_px_by_level: dict[str, np.ndarray] = {}
    smoothed_acceleration_mm_by_level: dict[str, np.ndarray] = {}
    smoothed_acceleration_px_by_level: dict[str, np.ndarray] = {}
    for level in TRACK_KINEMATICS_SPEED_LEVELS:
        (
            speed_mm,
            speed_px,
            path_mm,
            path_px,
            accel_mm,
            accel_px,
            smooth_accel_mm,
            smooth_accel_px,
        ) = _load_legacy_speed_level_for_inspection(track_group, level=level)
        if speed_mm is not None:
            speed_mm_by_level[level] = speed_mm
        if speed_px is not None:
            speed_px_by_level[level] = speed_px
        if path_mm is not None:
            frame_path_distance_mm_by_level[level] = path_mm
        if path_px is not None:
            frame_path_distance_px_by_level[level] = path_px
        if accel_mm is not None:
            acceleration_mm_by_level[level] = accel_mm
        if accel_px is not None:
            acceleration_px_by_level[level] = accel_px
        if smooth_accel_mm is not None:
            smoothed_acceleration_mm_by_level[level] = smooth_accel_mm
        if smooth_accel_px is not None:
            smoothed_acceleration_px_by_level[level] = smooth_accel_px

    for required in requested_speed_levels:
        if required not in speed_mm_by_level:
            source_level = TRACK_KINEMATICS_SOURCE_SPEED_LEVELS[required]
            raise ValueError(f"{label} is missing required speed level '{source_level}_mm'")

    return TrackKinematicsTrackTables(
        run_name=resolved_name,
        scope=scope,
        run_path=run_path,
        track_id=int(track_id),
        track_path=track_path,
        run_attrs=_group_attrs(run_group),
        track_attrs=_group_attrs(track_group),
        authority_status="unverified_legacy_inspection_only",
        motion_manifest_sha256=None,
        positions_px_descriptor_sha256=None,
        positions_mm_descriptor_sha256=None,
        positions_px_descriptor=None,
        positions_mm_descriptor=None,
        track_sample_key=_optional_array(track_group, "track_sample_key"),
        source_acquisition_frame_index=_optional_array(
            track_group, "source_acquisition_frame_index"
        ),
        source_frame_interpolation=_optional_array(
            track_group, "source_frame_interpolation"
        ),
        source_instance_key=_optional_array(track_group, "source_instance_key"),
        source_row_index=_optional_array(track_group, "source_row_index"),
        frame_indices=frame_indices,
        speed_mm_by_level=speed_mm_by_level,
        speed_px_by_level=speed_px_by_level,
        frame_path_distance_mm_by_level=frame_path_distance_mm_by_level,
        frame_path_distance_px_by_level=frame_path_distance_px_by_level,
        acceleration_mm_by_level=acceleration_mm_by_level,
        acceleration_px_by_level=acceleration_px_by_level,
        smoothed_acceleration_mm_by_level=smoothed_acceleration_mm_by_level,
        smoothed_acceleration_px_by_level=smoothed_acceleration_px_by_level,
        delta_seconds=_optional_array(track_group, "delta_seconds"),
        transition_valid=_optional_array(track_group, "transition_valid"),
        sample_valid=_optional_array(track_group, "sample_valid"),
        time_seconds=_optional_array(track_group, "time_seconds"),
        heading_degrees=_optional_array(track_group, "heading_degrees"),
        heading_radians=_optional_array(track_group, "heading_radians"),
        smoothed_heading_degrees=_optional_array(
            track_group, "smoothed_heading_degrees"
        ),
        smoothed_heading_radians=_optional_array(
            track_group, "smoothed_heading_radians"
        ),
        delta_heading_degrees=_optional_array(track_group, "delta_heading_degrees"),
        delta_heading_smoothed_degrees=_optional_array(
            track_group, "delta_heading_smoothed_degrees"
        ),
        angular_velocity_deg_s=_optional_array(
            track_group, "angular_velocity_deg_s"
        ),
        angular_velocity_smoothed_deg_s=_optional_array(
            track_group, "angular_velocity_smoothed_deg_s"
        ),
        angular_speed_raw_deg_s=_optional_array(
            track_group, "angular_speed_raw_deg_s"
        ),
        angular_speed_smoothed_deg_s=_optional_array(
            track_group, "angular_speed_smoothed_deg_s"
        ),
        detection_source=_optional_array(track_group, "detection_source"),
        sample_reason_code=_optional_array(track_group, "sample_reason_code"),
        transition_reason_code=_optional_array(
            track_group, "transition_reason_code"
        ),
        positions_mm=_optional_array(track_group, "positions_mm"),
        positions_px=_optional_array(track_group, "positions_px"),
        cumulative_path_distance_mm=_optional_array(
            track_group, "cumulative_path_distance_mm"
        ),
        cumulative_path_distance_px=_optional_array(
            track_group, "cumulative_path_distance_px"
        ),
    )
