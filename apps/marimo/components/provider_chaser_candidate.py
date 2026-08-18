"""Read-only Marimo projections for unpromoted provider-aware chaser candidates.

This module is deliberately a consumer boundary, not a producer.  It reads one
manifest-validated candidate and exact, selector-ineligible upstream sources;
it never writes Zarr, resolves ``latest``, changes a selector, interpolates a
row, or promotes a candidate into canonical analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.chaser_egocentric_bearing import (
    compute_egocentric_chaser_bearing,
)
from fisheye.analysis.provider_chaser_distance_candidates import (
    MANIFEST_DIGEST_ATTR,
    PARENT_PATH,
    validate_provider_chaser_distance_candidate,
)
from fisheye.analysis.swim_bout_frame_axis import canonical_frame_axis_sha256
from fisheye.analysis.swim_bout_io import (
    SwimBoutTables,
    load_exact_selector_ineligible_default_swim_bout_tables,
)
from fisheye.analysis.swim_bout_schema import (
    SWIM_BOUT_LAYOUT,
    SWIM_BOUT_RUN_SCHEMA_ID,
    SWIM_BOUT_RUN_SCHEMA_VERSION,
)
from fisheye.analysis_workflows.materializers.provider_track_motion import (
    PROVIDER_TRACK_MOTION_MANIFEST_ATTR,
    PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR,
    PROVIDER_TRACK_MOTION_PARENT_PATH,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    ProviderTrackMotionSourceHandle,
    load_provider_track_motion_source_handle,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.zarr.columnar import load_structured_dataset
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root

from .common import normalize_path
from .registry import (
    PROVIDER_CHASER_CANDIDATE_RENDERER,
    InteractiveSpecOption,
)


BASE_ANALYSIS_IDS = ("static_artifacts", "provenance")
GazeReadiness = "unavailable_missing_eye_angle_authority"


def _readonly_array(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _exact_candidate_run_path(option: InteractiveSpecOption) -> str:
    run_path = normalize_path(option.run_path)
    prefix = f"{PARENT_PATH}/"
    if not run_path.startswith(prefix):
        raise ValueError(
            "Provider chaser candidate must be selected by its exact candidate run path."
        )
    run_name = run_path.removeprefix(prefix)
    if not run_name or "/" in run_name or run_name in {".", "..", "latest"}:
        raise ValueError("Provider chaser candidate run name is invalid.")
    return run_path


def _require_candidate_option(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
) -> tuple[Path, str, Any, Mapping[str, Any]]:
    if option.renderer != PROVIDER_CHASER_CANDIDATE_RENDERER:
        raise ValueError("Selected option is not a provider chaser candidate.")
    run_path = _exact_candidate_run_path(option)
    manifest_sha256 = str(option.attrs.get(MANIFEST_DIGEST_ATTR) or "").strip()
    if not manifest_sha256:
        raise ValueError("Provider chaser candidate has no manifest digest authority.")
    archive = Path(zarr_path)
    validation = validate_provider_chaser_distance_candidate(
        archive / run_path,
        use_consolidated=True,
        archive_path=archive,
        archive_run_path=run_path,
        expected_manifest_sha256=manifest_sha256,
    )
    if not validation.get("valid"):
        raise ValueError(
            "Provider chaser candidate failed exact manifest validation: "
            f"{validation.get('errors', [])!r}."
        )
    if option.spec.get("candidate_status") != "unpromoted_selector_ineligible":
        raise ValueError(
            "Provider chaser candidate status is not explicitly unpromoted."
        )
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[run_path]
    attrs = dict(getattr(run, "attrs", {}))
    if attrs.get("stage_selector_eligible") is not False:
        raise ValueError("Provider chaser candidate is not selector-ineligible.")
    if attrs.get("row_axis") != "stimulus_samples":
        raise ValueError("Provider chaser candidate row axis is not stimulus_samples.")
    if attrs.get(MANIFEST_DIGEST_ATTR) != manifest_sha256:
        raise ValueError(
            "Provider chaser candidate option digest differs from the run."
        )
    return archive, run_path, run, attrs


def _read_candidate_array(run: Any, path: str) -> np.ndarray:
    try:
        return _readonly_array(run[path][:])
    except Exception as exc:
        raise ValueError(
            f"Provider chaser candidate array is missing: {path!r}."
        ) from exc


def _group_names(group: Any) -> tuple[str, ...]:
    keys = getattr(group, "group_keys", None)
    if not callable(keys):
        return ()
    try:
        return tuple(sorted(str(value) for value in keys()))
    except Exception:
        return ()


def _exact_child_path(value: Any, *, parent: str, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be one exact path.")
    path = value.strip().strip("/")
    prefix = f"{parent}/"
    name = path.removeprefix(prefix)
    if (
        value != path
        or not path.startswith(prefix)
        or not name
        or "/" in name
        or name in {".", "..", "latest", "default"}
    ):
        raise ValueError(f"{label} must name one exact child below {parent!r}.")
    return path


@dataclass(frozen=True)
class _ResolvedProviderSources:
    motion: ProviderTrackMotionSourceHandle
    motion_run_path: str
    motion_manifest_sha256: str
    swim_bout_tables: SwimBoutTables | None
    swim_bout_run_path: str | None
    swim_bout_array_manifest_sha256: str | None
    swim_bout_frame_axis_contract_sha256: str | None


def _provider_position_source_from_manifest(manifest: Any) -> Mapping[str, Any] | None:
    if not isinstance(manifest, Mapping):
        return None
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return None
    authority = payload.get("source_authority")
    if not isinstance(authority, Mapping):
        return None
    record = authority.get("record")
    if not isinstance(record, Mapping):
        return None
    position_source = record.get("position_source")
    return position_source if isinstance(position_source, Mapping) else None


def _resolve_exact_provider_motion(
    archive: Path,
    root: Any,
    *,
    source_position_run_path: str,
    source_position_manifest_sha256: str,
) -> tuple[ProviderTrackMotionSourceHandle, str, str]:
    parent = root[PROVIDER_TRACK_MOTION_PARENT_PATH]
    parent_attrs = set(getattr(parent, "attrs", {}))
    forbidden = {
        "latest",
        "latest_complete",
        "latest_pending",
        "latest_provider",
        "authoritative_run",
        "authoritative",
        "current",
        "default",
        "fallback",
        "selected",
    }
    if forbidden.intersection(parent_attrs):
        raise ValueError("Provider-motion namespace contains selector attributes.")

    matches: list[tuple[str, str]] = []
    for run_name in _group_names(parent):
        run_path = f"{PROVIDER_TRACK_MOTION_PARENT_PATH}/{run_name}"
        run = parent[run_name]
        attrs = getattr(run, "attrs", {})
        manifest = attrs.get(PROVIDER_TRACK_MOTION_MANIFEST_ATTR)
        position_source = _provider_position_source_from_manifest(manifest)
        if position_source is None:
            continue
        if (
            position_source.get("run_path") == source_position_run_path
            and position_source.get("manifest_sha256")
            == source_position_manifest_sha256
        ):
            digest = str(attrs.get(PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR) or "")
            if len(digest) != 64 or digest.lower() != digest:
                raise ValueError(
                    f"Provider-motion run {run_path!r} has no valid manifest digest."
                )
            matches.append((run_path, digest))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one provider-motion run matching the candidate position "
            f"source; found {len(matches)}."
        )
    motion_run_path, motion_manifest_sha256 = matches[0]
    motion = load_provider_track_motion_source_handle(
        archive,
        motion_run_path,
        use_consolidated=True,
        expected_manifest_sha256=motion_manifest_sha256,
        require_authoritative_timing=False,
    )
    if motion.selector_eligible is not False:
        raise ValueError("Provider-motion source is not selector-ineligible.")
    source_position = _mapping(
        motion.source_authority.get("position_source"), label="provider position source"
    )
    if (
        source_position.get("run_path") != source_position_run_path
        or source_position.get("manifest_sha256") != source_position_manifest_sha256
    ):
        raise ValueError(
            "Strict provider-motion source authority does not match the candidate."
        )
    if motion.provider_manifest_sha256 != motion_manifest_sha256:
        raise ValueError(
            "Provider-motion verification digest is not bound to its manifest."
        )
    return motion, motion_run_path, motion_manifest_sha256


def _resolve_exact_swim_bout(
    root: Any,
    *,
    motion_manifest_sha256: str,
) -> tuple[SwimBoutTables, str, str, str]:
    parent = root["analysis/swim_bout_runs"]
    matches: list[str] = []
    for run_name in _group_names(parent):
        run = parent[run_name]
        attrs = getattr(run, "attrs", {})
        if (
            attrs.get("schema_id") == SWIM_BOUT_RUN_SCHEMA_ID
            and attrs.get("schema_version") == SWIM_BOUT_RUN_SCHEMA_VERSION
            and attrs.get("layout") == SWIM_BOUT_LAYOUT
            and attrs.get("palette_run_completion_status") == "complete"
            and attrs.get("stage_selector_eligible") is False
            and attrs.get("source_track_motion_manifest_sha256")
            == motion_manifest_sha256
        ):
            matches.append(run_name)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one complete selector-ineligible v8 swim-bout run "
            f"for provider motion {motion_manifest_sha256}; found {len(matches)}."
        )
    run_name = matches[0]
    tables = load_exact_selector_ineligible_default_swim_bout_tables(
        root,
        run_name=run_name,
    )
    if (
        tables.candidate.default_signal_id != tables.signal.signal_id
        or not tables.signal.is_default
    ):
        raise ValueError(
            "Exact swim-bout loader did not return its declared default signal."
        )
    attrs = dict(tables.run_attrs)
    array_manifest = _mapping(
        attrs.get("array_schema_manifest"), label="swim-bout array_schema_manifest"
    )
    array_payload = _mapping(
        array_manifest.get("payload"), label="swim-bout array_schema_manifest.payload"
    )
    array_digest = str(array_manifest.get("payload_digest") or "")
    if canonical_json_sha256(dict(array_payload)) != array_digest:
        raise ValueError("Swim-bout array-schema manifest payload digest is stale.")
    frame_contract = _mapping(
        attrs.get("frame_axis_contract"), label="swim-bout frame_axis_contract"
    )
    if frame_contract.get("schema_id") is None:
        raise ValueError(
            "Swim-bout frame-axis contract is missing its schema identity."
        )
    if (
        frame_contract.get("source_track_motion_manifest_sha256")
        != motion_manifest_sha256
    ):
        raise ValueError(
            "Swim-bout frame-axis contract is bound to another motion run."
        )
    frame_digest = canonical_json_sha256(dict(frame_contract))
    if frame_contract.get("content_sha256"):
        axis = tables.series.get("frame_indices")
        if (
            axis is not None
            and canonical_frame_axis_sha256(np.asarray(axis))
            != frame_contract["content_sha256"]
        ):
            raise ValueError("Swim-bout frame-axis content differs from its contract.")
    return tables, f"analysis/swim_bout_runs/{run_name}", array_digest, frame_digest


def _load_source_bundle(
    archive: Path,
    root: Any,
    *,
    candidate_attrs: Mapping[str, Any],
    require_swim_bout: bool,
) -> _ResolvedProviderSources:
    position_path = _exact_child_path(
        candidate_attrs.get("source_position_run_path"),
        parent="analysis/subject_position_runs/observation",
        label="candidate source_position_run_path",
    )
    position_digest = str(candidate_attrs.get("source_position_manifest_sha256") or "")
    if len(position_digest) != 64 or position_digest.lower() != position_digest:
        raise ValueError("Candidate source_position_manifest_sha256 is invalid.")
    motion, motion_path, motion_digest = _resolve_exact_provider_motion(
        archive,
        root,
        source_position_run_path=position_path,
        source_position_manifest_sha256=position_digest,
    )
    try:
        swim, swim_path, array_digest, frame_digest = _resolve_exact_swim_bout(
            root,
            motion_manifest_sha256=motion_digest,
        )
    except (KeyError, ValueError):
        if require_swim_bout:
            raise
        swim = None
        swim_path = None
        array_digest = None
        frame_digest = None
    return _ResolvedProviderSources(
        motion=motion,
        motion_run_path=motion_path,
        motion_manifest_sha256=motion_digest,
        swim_bout_tables=swim,
        swim_bout_run_path=swim_path,
        swim_bout_array_manifest_sha256=array_digest,
        swim_bout_frame_axis_contract_sha256=frame_digest,
    )


def _epoch_labels(run: Any) -> dict[int, str]:
    try:
        ids = np.asarray(run["epoch_summary/window_id"][:], dtype=np.int64)
        labels = np.asarray(run["epoch_summary/label_bytes"][:])
    except Exception as exc:
        raise ValueError("Candidate epoch summary is incomplete.") from exc
    if ids.ndim != 1 or labels.shape[0] != ids.size:
        raise ValueError("Candidate epoch summary arrays are not aligned.")
    return {
        int(window_id): decode_null_terminated_text(label).strip()
        or f"epoch_{int(window_id)}"
        for window_id, label in zip(ids, labels, strict=True)
    }


def _resolve_semantic_chaser_labels(
    root: Any,
    *,
    source_stimulus_run_path: str,
    source_row_indices: np.ndarray,
    chaser_indices: np.ndarray,
) -> tuple[str, ...]:
    stimulus_path = _exact_child_path(
        source_stimulus_run_path,
        parent="analysis/stimulus_runs",
        label="candidate source_stimulus_run_path",
    )
    stimulus = root[stimulus_path]
    tracking = stimulus["tracking_data"]
    chaser_group = tracking["chaser_states"]
    chaser_attrs = getattr(chaser_group, "attrs", {})
    if (
        chaser_attrs.get("schema_id") != "citrus.tracking.chaser_states"
        or chaser_attrs.get("schema_version") != 5
        or chaser_attrs.get("coordinate_descriptor_status") != "canonical"
    ):
        raise ValueError(
            "Candidate source stimulus chaser_states is not the sealed canonical v5 surface."
        )
    records, _ = load_structured_dataset(tracking, "chaser_states")
    if (
        records.dtype.names is None
        or "chaser_behavior_class_id" not in records.dtype.names
    ):
        raise ValueError(
            "Source stimulus run lacks sealed chaser_behavior_class_id semantics."
        )
    if (
        source_row_indices.ndim != 2
        or source_row_indices.shape[1] != chaser_indices.size
    ):
        raise ValueError(
            "Candidate source stimulus row lineage does not match the chaser axis."
        )
    if np.any(source_row_indices < 0) or np.any(source_row_indices >= records.shape[0]):
        raise ValueError(
            "Candidate source stimulus row lineage is outside the exact source rowset."
        )

    try:
        enum_group = stimulus["enums"]["chaser_behavior_classes"]
    except KeyError:
        enum_group = None
    try:
        if enum_group is None:
            raise KeyError("chaser_behavior_classes")
        enum_ids = np.asarray(enum_group["id"][:])
        enum_name_values = np.asarray(enum_group["name"][:])
        if enum_ids.ndim != 1 or enum_name_values.shape[0] != enum_ids.size:
            raise ValueError(
                "Source stimulus chaser behavior enum arrays are not aligned."
            )
    except (KeyError, TypeError, ValueError):
        # In-memory readers and historical columnar fixtures may expose the
        # enum as a structured table instead of the maintained fixed-width
        # byte arrays.  The production v5 surface takes the direct branch.
        enums, _ = load_structured_dataset(stimulus["enums"], "chaser_behavior_classes")
        if enums.dtype.names is None or not {"id", "name"}.issubset(enums.dtype.names):
            raise ValueError("Source stimulus chaser behavior enum is not sealed.")
        enum_ids = np.asarray(enums["id"])
        enum_name_values = np.asarray(enums["name"])
    enum_names: dict[int, str] = {}
    for enum_id_value, enum_name_value in zip(enum_ids, enum_name_values, strict=True):
        enum_id = int(enum_id_value)
        enum_name = decode_null_terminated_text(enum_name_value).strip()
        if not enum_name or enum_id in enum_names:
            raise ValueError("Source stimulus chaser behavior enum is ambiguous.")
        enum_names[enum_id] = enum_name

    labels: list[str] = []
    for column, chaser_index in enumerate(chaser_indices.tolist()):
        rows = source_row_indices[:, column].astype(np.int64, copy=False)
        if "chaser_index" in records.dtype.names and not np.all(
            np.asarray(records["chaser_index"])[rows] == int(chaser_index)
        ):
            raise ValueError("Candidate stimulus row lineage changes chaser identity.")
        class_ids = np.unique(np.asarray(records["chaser_behavior_class_id"])[rows])
        names = {enum_names.get(int(class_id)) for class_id in class_ids}
        if None in names:
            raise ValueError(
                "Candidate stimulus references an unknown behavior enum value."
            )
        labels.append(
            "unknown"
            if not names
            else next(iter(names))
            if len(names) == 1
            else "mixed"
        )
    return tuple(labels)


def _motion_row_indices(
    motion: ProviderTrackMotionSourceHandle,
    *,
    candidate_position_rows: np.ndarray,
    candidate_acquisition_frames: np.ndarray,
) -> np.ndarray:
    position_rows = np.asarray(motion.source_position_row_index, dtype=np.int64)
    if np.unique(position_rows).size != position_rows.size:
        raise ValueError("Provider motion source_position_row_index is ambiguous.")
    by_position = {
        int(value): index for index, value in enumerate(position_rows.tolist())
    }
    mapped = np.full(candidate_position_rows.shape, -1, dtype=np.int64)
    for index, position_row in enumerate(candidate_position_rows.tolist()):
        if int(position_row) < 0:
            continue
        try:
            motion_row = by_position[int(position_row)]
        except KeyError as exc:
            raise ValueError(
                "Candidate source_position_run_row_index has no exact provider-motion row."
            ) from exc
        if int(motion.source_acquisition_frame_index[motion_row]) != int(
            candidate_acquisition_frames[index]
        ):
            raise ValueError(
                "Candidate and provider-motion acquisition-frame lineage disagree."
            )
        mapped[index] = motion_row
    if np.any(mapped >= 0):
        track_ids = np.asarray(
            motion.track_sample_key[mapped[mapped >= 0], 0], dtype=np.int64
        )
        if np.unique(track_ids).size != 1:
            raise ValueError(
                "Candidate position lineage maps to multiple provider tracks."
            )
    return mapped


def _circular_summary(values: np.ndarray) -> tuple[float, float]:
    angles = np.asarray(values, dtype=np.float64)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return float("nan"), float("nan")
    vector = np.mean(np.exp(1j * np.deg2rad(angles)))
    mean = float(((np.rad2deg(np.angle(vector)) + 180.0) % 360.0) - 180.0)
    return mean, float(np.abs(vector))


def _build_bout_rows(
    tables: SwimBoutTables,
    *,
    sample_acquisition_frames: np.ndarray,
    stimulus_frame_nums: np.ndarray,
    epoch_ids: np.ndarray,
    epoch_labels: Mapping[int, str],
    chaser_indices: np.ndarray,
    chaser_labels: tuple[str, ...],
    distance_mm: np.ndarray,
    bearing_deg: np.ndarray,
    bearing_valid: np.ndarray,
) -> tuple[Mapping[str, Any], ...]:
    required = {
        "bout_id",
        "start_frame",
        "end_frame",
        "duration_s",
        "path_length_mm",
        "peak_physical_speed_mm_s",
    }
    if tables.bouts.dtype.names is None or not required.issubset(
        tables.bouts.dtype.names
    ):
        raise ValueError("Exact swim-bout default table lacks required bout metrics.")
    rows: list[Mapping[str, Any]] = []
    for source_bout_row_index, bout in enumerate(tables.bouts):
        start_frame = int(bout["start_frame"])
        mapped_samples = np.flatnonzero(
            sample_acquisition_frames == start_frame
        ).astype(np.int64)
        mapped_epoch_ids = (
            np.unique(epoch_ids[mapped_samples])
            if mapped_samples.size
            else np.zeros(0, dtype=np.int64)
        )
        mapped_epoch_ids = mapped_epoch_ids[mapped_epoch_ids >= 0]
        epoch_id: int | None = (
            int(mapped_epoch_ids[0]) if mapped_epoch_ids.size == 1 else None
        )
        epoch_label = epoch_labels.get(epoch_id) if epoch_id is not None else None
        for column, chaser_index in enumerate(chaser_indices.tolist()):
            valid_samples = mapped_samples[
                bearing_valid[mapped_samples, column]
                & np.isfinite(distance_mm[mapped_samples, column])
                & np.isfinite(bearing_deg[mapped_samples, column])
            ]
            distances = distance_mm[valid_samples, column]
            bearings = bearing_deg[valid_samples, column]
            bearing_mean, bearing_resultant = _circular_summary(bearings)
            row = {
                "source_bout_row_index": int(source_bout_row_index),
                "bout_id": int(bout["bout_id"]),
                "chaser_index": int(chaser_index),
                "chaser_label": str(chaser_labels[column]),
                "source_bout_start_frame": start_frame,
                "source_bout_end_frame": int(bout["end_frame"]),
                "source_bout_peak_frame": int(bout["peak_frame"])
                if "peak_frame" in (tables.bouts.dtype.names or ())
                else None,
                "duration_s": float(bout["duration_s"]),
                "path_length_mm": float(bout["path_length_mm"]),
                "peak_speed_mm_s": float(bout["peak_physical_speed_mm_s"]),
                "epoch_id": epoch_id,
                "epoch_label": epoch_label,
                "mapped_stimulus_sample_count": int(mapped_samples.size),
                "support_count": int(valid_samples.size),
                "onset_distance_median_mm": float(np.median(distances))
                if distances.size
                else float("nan"),
                "onset_distance_min_mm": float(np.min(distances))
                if distances.size
                else float("nan"),
                "onset_distance_max_mm": float(np.max(distances))
                if distances.size
                else float("nan"),
                "onset_bearing_mean_deg": bearing_mean,
                "onset_bearing_resultant": bearing_resultant,
                "onset_stimulus_sample_indices": tuple(
                    int(value) for value in mapped_samples.tolist()
                ),
                "onset_stimulus_frame_nums": tuple(
                    int(value) for value in stimulus_frame_nums[mapped_samples].tolist()
                ),
            }
            rows.append(MappingProxyType(row))
    return tuple(rows)


@dataclass(frozen=True)
class ProviderChaserCandidateProjection:
    """Immutable in-memory, read-only projection of one candidate run."""

    zarr_path: Path
    candidate_run_path: str
    candidate_manifest_sha256: str
    row_axis: str
    stimulus_frame_num: np.ndarray
    timestamp_ns: np.ndarray
    source_acquisition_frame_index: np.ndarray
    stimulus_epoch_window_id: np.ndarray
    epoch_labels: Mapping[int, str]
    chaser_indices: np.ndarray
    chaser_labels: tuple[str, ...]
    distance_mm: np.ndarray
    bearing_deg: np.ndarray
    bearing_valid: np.ndarray
    source_position_run_row_index: np.ndarray
    source_motion_row_index: np.ndarray
    source_motion_run_path: str
    source_motion_manifest_sha256: str
    source_motion_verification_digest: str
    bout_rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    gaze_readiness: str = GazeReadiness

    def __post_init__(self) -> None:
        for name in (
            "stimulus_frame_num",
            "timestamp_ns",
            "source_acquisition_frame_index",
            "stimulus_epoch_window_id",
            "chaser_indices",
            "distance_mm",
            "bearing_deg",
            "bearing_valid",
            "source_position_run_row_index",
            "source_motion_row_index",
        ):
            value = _readonly_array(getattr(self, name))
            object.__setattr__(self, name, value)
        object.__setattr__(self, "epoch_labels", _freeze(dict(self.epoch_labels)))
        object.__setattr__(self, "provenance", _freeze(dict(self.provenance)))
        object.__setattr__(
            self, "bout_rows", tuple(_freeze(dict(row)) for row in self.bout_rows)
        )
        if self.row_axis != "stimulus_samples":
            raise ValueError(
                "Provider candidate projection row axis must be stimulus_samples."
            )
        if (
            self.bearing_deg.shape != self.distance_mm.shape
            or self.bearing_valid.shape != self.distance_mm.shape
        ):
            raise ValueError(
                "Candidate projection distance, bearing, and validity arrays disagree."
            )
        if self.distance_mm.shape[0] != self.stimulus_frame_num.size:
            raise ValueError(
                "Candidate projection arrays do not align to stimulus samples."
            )


def _load_projection_from_candidate(
    archive: Path,
    run_path: str,
    run: Any,
    attrs: Mapping[str, Any],
    *,
    sources: _ResolvedProviderSources,
) -> ProviderChaserCandidateProjection:
    stimulus_frame_num = _read_candidate_array(
        run, "samples/stimulus_frame_num"
    ).astype(np.int64, copy=False)
    timestamp_ns = _read_candidate_array(run, "samples/timestamp_ns").astype(
        np.int64, copy=False
    )
    source_acquisition = _read_candidate_array(
        run, "samples/source_acquisition_frame_index"
    ).astype(np.int64, copy=False)
    epoch_ids = _read_candidate_array(run, "samples/stimulus_epoch_window_id").astype(
        np.int64, copy=False
    )
    source_position_rows = _read_candidate_array(
        run, "positions/source_position_run_row_index"
    ).astype(np.int64, copy=False)
    source_stimulus_rows = _read_candidate_array(
        run, "samples/source_stimulus_run_row_index"
    ).astype(np.int64, copy=False)
    chaser_indices = _read_candidate_array(run, "chasers/chaser_index").astype(
        np.int64, copy=False
    )
    fish_arena = _read_candidate_array(run, "positions/fish_position_arena_xy").astype(
        np.float64, copy=False
    )
    fish_valid = _read_candidate_array(run, "positions/fish_valid").astype(
        bool, copy=False
    )
    chaser_arena = _read_candidate_array(
        run, "positions/chaser_position_arena_xy"
    ).astype(np.float64, copy=False)
    chaser_valid = _read_candidate_array(run, "positions/chaser_valid").astype(
        bool, copy=False
    )
    distance_mm = _read_candidate_array(run, "distances/distance_mm").astype(
        np.float64, copy=False
    )
    if (
        source_position_rows.shape != stimulus_frame_num.shape
        or timestamp_ns.shape != stimulus_frame_num.shape
        or source_acquisition.shape != stimulus_frame_num.shape
    ):
        raise ValueError(
            "Candidate position and acquisition lineage is not sample-aligned."
        )
    if source_stimulus_rows.shape != (stimulus_frame_num.size, chaser_indices.size):
        raise ValueError("Candidate stimulus row lineage is not chaser-aligned.")
    if chaser_arena.shape != (stimulus_frame_num.size, chaser_indices.size, 2):
        raise ValueError("Candidate chaser positions are not sample-aligned.")
    if distance_mm.shape != (stimulus_frame_num.size, chaser_indices.size):
        raise ValueError("Candidate distances are not sample-aligned.")
    motion_rows = _motion_row_indices(
        sources.motion,
        candidate_position_rows=source_position_rows,
        candidate_acquisition_frames=source_acquisition,
    )
    heading = np.full(stimulus_frame_num.size, np.nan, dtype=np.float64)
    heading_valid = np.zeros(stimulus_frame_num.size, dtype=bool)
    mapped = motion_rows >= 0
    heading[mapped] = np.asarray(sources.motion.arrays["smoothed_heading_degrees"])[
        motion_rows[mapped]
    ]
    heading_valid[mapped] = (
        np.asarray(sources.motion.angular_sample_valid)[motion_rows[mapped]]
        & np.asarray(sources.motion.body_frame_source_valid)[motion_rows[mapped]]
        & np.isfinite(heading[mapped])
    )
    _vector, bearing, _alignment, _lateral, bearing_valid = (
        compute_egocentric_chaser_bearing(
            fish_arena_xy=fish_arena,
            chaser_arena_xy=chaser_arena,
            fish_heading_deg=heading,
            fish_valid=fish_valid,
            chaser_valid=chaser_valid,
            fish_heading_valid=heading_valid,
            distance_mm=distance_mm,
        )
    )
    labels = _resolve_semantic_chaser_labels(
        open_zarr_root(archive, mode="r", use_consolidated=True),
        source_stimulus_run_path=str(attrs.get("source_stimulus_run_path") or ""),
        source_row_indices=source_stimulus_rows,
        chaser_indices=chaser_indices,
    )
    epoch_label_map = _epoch_labels(run)
    if sources.swim_bout_tables is None:
        bout_rows: tuple[Mapping[str, Any], ...] = ()
    else:
        bout_rows = _build_bout_rows(
            sources.swim_bout_tables,
            sample_acquisition_frames=source_acquisition,
            stimulus_frame_nums=stimulus_frame_num,
            epoch_ids=epoch_ids,
            epoch_labels=epoch_label_map,
            chaser_indices=chaser_indices,
            chaser_labels=labels,
            distance_mm=distance_mm,
            bearing_deg=bearing,
            bearing_valid=bearing_valid,
        )
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    try:
        eye_angle_parent = root["analysis/eye_angle_runs"]
    except KeyError:
        gaze_readiness = GazeReadiness
    else:
        gaze_readiness = (
            "present_not_projected_unvalidated"
            if _group_names(eye_angle_parent)
            else GazeReadiness
        )
    provenance = {
        "candidate": {
            "run_path": run_path,
            "manifest_sha256": str(attrs[MANIFEST_DIGEST_ATTR]),
            "selector_eligible": False,
            "row_axis": "stimulus_samples",
        },
        "position": {
            "run_path": attrs.get("source_position_run_path"),
            "manifest_sha256": attrs.get("source_position_manifest_sha256"),
            "estimator_id": attrs.get("source_position_estimator_id"),
        },
        "provider_motion": {
            "run_path": sources.motion_run_path,
            "manifest_sha256": sources.motion_manifest_sha256,
            "verification_digest": sources.motion.verification_digest,
            "heading_array": "smoothed_heading_degrees",
            "angular_validity_array": "angular_sample_valid",
            "body_frame_validity_array": "body_frame_source_valid",
            "source_position_row_array": "source_position_row_index",
            "timing_is_authoritative": bool(sources.motion.timing_is_authoritative),
            "temporal_claims": "descriptive_only_no_derivatives_or_lag_claims",
        },
        "stimulus": {
            "run_path": attrs.get("source_stimulus_run_path"),
            "row_axis": "stimulus_samples",
            "semantic_label_source": "sealed_chaser_behavior_class_enum",
            "source_row_index_array": "samples/source_stimulus_run_row_index",
        },
        "swim_bout": {
            "run_path": sources.swim_bout_run_path,
            "array_schema_manifest_sha256": sources.swim_bout_array_manifest_sha256,
            "frame_axis_contract_sha256": sources.swim_bout_frame_axis_contract_sha256,
            "selector_eligible": False
            if sources.swim_bout_tables is not None
            else None,
            "default_candidate_id": (
                sources.swim_bout_tables.candidate.candidate_id
                if sources.swim_bout_tables is not None
                else None
            ),
            "default_signal_id": (
                sources.swim_bout_tables.signal.signal_id
                if sources.swim_bout_tables is not None
                else None
            ),
        },
        "readiness": {
            "gaze": gaze_readiness,
            "selector_ineligible": True,
            "promoted": False,
        },
    }
    return ProviderChaserCandidateProjection(
        zarr_path=archive,
        candidate_run_path=run_path,
        candidate_manifest_sha256=str(attrs[MANIFEST_DIGEST_ATTR]),
        row_axis="stimulus_samples",
        stimulus_frame_num=stimulus_frame_num,
        timestamp_ns=timestamp_ns,
        source_acquisition_frame_index=source_acquisition,
        stimulus_epoch_window_id=epoch_ids,
        epoch_labels=epoch_label_map,
        chaser_indices=chaser_indices,
        chaser_labels=labels,
        distance_mm=distance_mm.astype(np.float32),
        bearing_deg=bearing,
        bearing_valid=bearing_valid,
        source_position_run_row_index=source_position_rows,
        source_motion_row_index=motion_rows,
        source_motion_run_path=sources.motion_run_path,
        source_motion_manifest_sha256=sources.motion_manifest_sha256,
        source_motion_verification_digest=sources.motion.verification_digest,
        bout_rows=bout_rows,
        provenance=provenance,
        gaze_readiness=gaze_readiness,
    )


def available_provider_chaser_candidate_analysis_ids(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
) -> tuple[str, ...]:
    """Return capabilities only after exact candidate/source preflight.

    Static artifacts and provenance belong to the candidate itself.  Bearing is
    exposed only when one exact provider-motion source matches its position
    lineage.  Bout response additionally requires one exact v8,
    selector-ineligible swim-bout source.  Gaze is intentionally never exposed
    by this provider because it has no eye-angle authority.
    """

    archive, _run_path, _run, attrs = _require_candidate_option(zarr_path, option)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    try:
        sources = _load_source_bundle(
            archive,
            root,
            candidate_attrs=attrs,
            require_swim_bout=False,
        )
    except (KeyError, TypeError, ValueError, OSError):
        return BASE_ANALYSIS_IDS
    ids = [*BASE_ANALYSIS_IDS, "egocentric_bearing"]
    if sources.swim_bout_tables is not None:
        ids.append("bout_response")
    return tuple(ids)


def load_provider_chaser_candidate_projection(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
    *,
    require_bout: bool = True,
) -> ProviderChaserCandidateProjection:
    """Load one exact candidate and its read-only downstream projections.

    Bearing-only explorer routing may set ``require_bout=False`` explicitly;
    the ordinary two-argument API remains fail-closed for the exact default
    swim-bout source required by the full projection.
    """

    if type(require_bout) is not bool:
        raise TypeError("require_bout must be an exact bool.")

    archive, run_path, run, attrs = _require_candidate_option(zarr_path, option)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    sources = _load_source_bundle(
        archive,
        root,
        candidate_attrs=attrs,
        require_swim_bout=require_bout,
    )
    return _load_projection_from_candidate(
        archive,
        run_path,
        run,
        attrs,
        sources=sources,
    )


def _coerce_output_args(
    go: Any,
    projection: ProviderChaserCandidateProjection | None,
) -> tuple[Any, ProviderChaserCandidateProjection]:
    # palette_explorer.py on the active branch already routes this component
    # with (mo, projection). Keep that call working while exposing the explicit
    # public (mo, go, projection) API for direct use and tests.
    if projection is None:
        projection = go
        import plotly.graph_objects as go_module

        go = go_module
    if not isinstance(projection, ProviderChaserCandidateProjection):
        raise TypeError("projection must be a ProviderChaserCandidateProjection.")
    return go, projection


def _unpromoted_header(
    mo: Any, projection: ProviderChaserCandidateProjection, title: str
) -> Any:
    return mo.md(
        f"## {title}\n\n"
        "**Exploratory, read-only, unpromoted candidate.** These views do not "
        "change canonical analysis or selector state.\n\n"
        f"Candidate: `{projection.candidate_run_path}`  \n"
        f"Row axis: `{projection.row_axis}`  \n"
        f"Gaze readiness: `{projection.gaze_readiness}`  \n"
        "Plot colors distinguish displayed series only; they do not encode the "
        "recorded stimulus color."
    )


def build_provider_chaser_candidate_bearing_output(
    mo: Any,
    go: Any,
    projection: ProviderChaserCandidateProjection | None = None,
) -> Any:
    """Build descriptive Plotly bearing distributions and polar views."""

    go, projection = _coerce_output_args(go, projection)
    figures: list[Any] = []
    for epoch_id, epoch_label in projection.epoch_labels.items():
        figure = go.Figure()
        figure.update_layout(
            title=f"Egocentric chaser bearing — {epoch_label} (candidate)",
            template="plotly_white",
            polar={
                "angularaxis": {"direction": "counterclockwise", "rotation": 90},
                "radialaxis": {"title": "valid-sample probability", "range": [0, 1]},
            },
            showlegend=True,
        )
        epoch_mask = projection.stimulus_epoch_window_id == int(epoch_id)
        for column, (chaser_index, label) in enumerate(
            zip(
                projection.chaser_indices.tolist(),
                projection.chaser_labels,
                strict=True,
            )
        ):
            mask = epoch_mask & projection.bearing_valid[:, column]
            values = projection.bearing_deg[mask, column].astype(np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            edges = np.linspace(-180.0, 180.0, 25)
            counts, _ = np.histogram(values, bins=edges)
            total = float(np.sum(counts))
            centers = (edges[:-1] + edges[1:]) * 0.5
            figure.add_trace(
                go.Barpolar(
                    theta=centers,
                    r=(counts / total).tolist(),
                    name=f"chaser {int(chaser_index)} · {label}",
                    hovertemplate=(
                        f"chaser {int(chaser_index)} · {label}<br>"
                        "bearing %{theta:.0f}°<br>probability %{r:.3f}<extra></extra>"
                    ),
                    opacity=0.68,
                )
            )
        figures.append(figure)
    if not figures:
        return mo.vstack(
            [
                _unpromoted_header(mo, projection, "Candidate egocentric bearing"),
                mo.md("No valid bearing samples."),
            ]
        )
    trace = go.Figure()
    time_s = (projection.timestamp_ns - projection.timestamp_ns[0]).astype(
        np.float64
    ) / 1e9
    for column, (chaser_index, label) in enumerate(
        zip(projection.chaser_indices.tolist(), projection.chaser_labels, strict=True)
    ):
        mask = projection.bearing_valid[:, column] & np.isfinite(
            projection.bearing_deg[:, column]
        )
        if np.any(mask):
            display_indices = np.flatnonzero(mask)
            display_step = max(1, int(np.ceil(display_indices.size / 20_000)))
            display_indices = display_indices[::display_step]
            trace.add_trace(
                go.Scattergl(
                    x=time_s[display_indices],
                    y=projection.bearing_deg[display_indices, column],
                    mode="lines",
                    name=f"chaser {int(chaser_index)} · {label}",
                )
            )
    trace.update_layout(
        title=(
            "Bearing over stimulus samples "
            "(display-decimated; descriptive; no lagged inference)"
        ),
        template="plotly_white",
        xaxis_title="Stimulus time (s)",
        yaxis_title="Bearing (deg)",
    )
    return mo.vstack(
        [
            _unpromoted_header(mo, projection, "Candidate egocentric bearing"),
            *figures,
            trace,
        ]
    )


def build_provider_chaser_candidate_bout_response_output(
    mo: Any,
    go: Any,
    projection: ProviderChaserCandidateProjection | None = None,
) -> Any:
    """Build descriptive bout metric versus onset-distance views."""

    go, projection = _coerce_output_args(go, projection)
    if not projection.bout_rows:
        return mo.vstack(
            [
                _unpromoted_header(mo, projection, "Candidate chaser bout response"),
                mo.md("No exact swim-bout projection is available."),
            ]
        )
    figures: list[Any] = []
    distance_distribution = go.Figure()
    for label in dict.fromkeys(projection.chaser_labels):
        values = np.asarray(
            [
                row["onset_distance_median_mm"]
                for row in projection.bout_rows
                if row["chaser_label"] == label
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size:
            distance_distribution.add_trace(
                go.Histogram(
                    x=values,
                    name=label,
                    histnorm="probability",
                    opacity=0.6,
                    hovertemplate=(
                        f"{label}<br>onset distance %{{x:.2f}} mm"
                        "<br>probability %{y:.3f}<extra></extra>"
                    ),
                )
            )
    distance_distribution.update_layout(
        title="Candidate bout-onset distance distribution",
        template="plotly_white",
        barmode="overlay",
        xaxis_title="median bout-onset distance (mm)",
        yaxis_title="probability per displayed bin",
        legend_title="semantic chaser",
    )
    figures.append(distance_distribution)
    for metric, title, y_label in (
        ("peak_speed_mm_s", "peak speed", "peak physical speed (mm/s)"),
        ("path_length_mm", "path length", "bout path length (mm)"),
        ("duration_s", "duration", "bout duration (s)"),
    ):
        figure = go.Figure()
        for label in dict.fromkeys(projection.chaser_labels):
            rows = [row for row in projection.bout_rows if row["chaser_label"] == label]
            x = np.asarray(
                [row["onset_distance_median_mm"] for row in rows], dtype=np.float64
            )
            y = np.asarray([row[metric] for row in rows], dtype=np.float64)
            support = np.asarray([row["support_count"] for row in rows], dtype=np.int64)
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                continue
            figure.add_trace(
                go.Scatter(
                    x=x[valid],
                    y=y[valid],
                    mode="markers",
                    name=label,
                    marker={"size": np.maximum(7, np.sqrt(support[valid] + 1.0) * 3)},
                    customdata=support[valid],
                    hovertemplate=(
                        f"{label}<br>onset distance %{{x:.2f}} mm<br>"
                        f"{y_label} %{{y:.3f}}<br>support %{{customdata}} samples<extra></extra>"
                    ),
                )
            )
        figure.update_layout(
            title=f"Candidate chaser bout response: {title}",
            template="plotly_white",
            xaxis_title="median onset distance (mm)",
            yaxis_title=y_label,
            legend_title="semantic chaser",
        )
        figures.append(figure)
    return mo.vstack(
        [_unpromoted_header(mo, projection, "Candidate chaser bout response"), *figures]
    )


__all__ = [
    "ProviderChaserCandidateProjection",
    "available_provider_chaser_candidate_analysis_ids",
    "build_provider_chaser_candidate_bearing_output",
    "build_provider_chaser_candidate_bout_response_output",
    "load_provider_chaser_candidate_projection",
]
