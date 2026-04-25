"""Check modern subject-mask eye-component coverage against keypoint-valid rows.

This diagnostic targets the current subject-mask surface:
`refined_subject_masks_runs/<run>` and `subject_mask_runs/<run>`.
Legacy `eye_masks_runs` and `refined_eye_masks_runs` are intentionally not
consulted here.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zarr

from fisheye.pose.schema import resolve_required_keypoint_indices_from_attrs
from fisheye.refinement.subject_eye_assignment import assign_eyes_union_to_lr
from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.environment import resolve_log_dir as resolve_shared_log_dir
from fisheye.shared.provenance_attrs import resolve_source_keypoints_run
from fisheye.shared.row_alignment import assert_row_alignment
from fisheye.shared.type_conversions import normalize_attr as _normalize_attr
from fisheye.utils.zarr_io import open_zarr_root


SUCCESS_DATASET_CANDIDATES = ("detection_success", "refined_success", "source_success")
SUBJECT_STAGE_CHOICES = ("auto", "refined_subject_masks_runs", "subject_mask_runs")
KEYPOINT_GROUP_CHOICES = ("refined_keypoints_runs", "keypoints_runs")
EYE_MODE_CHOICES = ("auto", "lr", "union")
EYE_PAIR_COMPONENTS = ("eye_left", "eye_right")
EYE_UNION_COMPONENT = "eyes_union"

_COMPONENT_ALIASES = {
    "eye": "eyes_union",
    "eyes": "eyes_union",
    "eyes_union": "eyes_union",
    "eye_union": "eyes_union",
    "eye-union": "eyes_union",
    "eye_left": "eye_left",
    "left_eye": "eye_left",
    "left-eye": "eye_left",
    "eye_right": "eye_right",
    "right_eye": "eye_right",
    "right-eye": "eye_right",
}

JsonLogger = SharedJsonLogger
_run_id = make_run_id


@dataclass(frozen=True)
class EyeComponentSelection:
    mode: str
    component_indices: dict[str, int]
    required_components: tuple[str, ...]


@dataclass
class CoverageReport:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    subject_stage: Optional[str] = None
    subject_run: Optional[str] = None
    label_schema_id: Optional[str] = None
    mask_labels: list[str] = field(default_factory=list)
    available_components: list[str] = field(default_factory=list)
    eye_component_mode: Optional[str] = None
    eye_component_indices: dict[str, int] = field(default_factory=dict)
    component_review_states: dict[str, str] = field(default_factory=dict)
    source_refined_eye_masks_run: Optional[str] = None
    source_eye_masks_run: Optional[str] = None
    latest_refined_eye_masks_run: Optional[str] = None
    keypoint_group: Optional[str] = None
    keypoint_run: Optional[str] = None
    success_dataset: Optional[str] = None
    keypoint_eye_indices: dict[str, int] = field(default_factory=dict)
    total_rois: int = 0
    keypoint_valid_rows: int = 0
    rows_with_eye_component_masks: int = 0
    rows_missing_eye_component_masks: int = 0
    component_present_rows: dict[str, int] = field(default_factory=dict)
    eyes_union_assignment_status: Optional[str] = None
    eyes_union_assignment_summary: dict[str, object] = field(default_factory=dict)
    failure_targets: list[dict[str, int]] = field(default_factory=list)
    sample_missing: list[dict[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return self.status in {"fail", "missing", "error"}


def _is_group(value: object) -> bool:
    return hasattr(value, "attrs") and callable(getattr(value, "get", None))


def _canonical_component(value: object) -> Optional[str]:
    text = _normalize_attr(value)
    if text is None:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    return _COMPONENT_ALIASES.get(normalized, normalized)


def _canonical_components(values: object) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    labels: list[str] = []
    for value in values:
        label = _canonical_component(value)
        if label is not None:
            labels.append(label)
    return labels


def _as_array(array_like: object, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hasattr(array_like, "shape") and hasattr(array_like, "__getitem__"):
        data = array_like[:]  # type: ignore[index]
    else:
        data = array_like
    if dtype is None:
        return np.asarray(data)
    return np.asarray(data, dtype=dtype)


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        value = _normalize_attr(root.attrs.get(key))
        if value:
            lowered = value.lower()
            if lowered in {"analysis", "training"}:
                return lowered
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _iter_zarr(paths: Sequence[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


def _collect_zarr_paths(paths: Sequence[Path], recursive: bool) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in _iter_zarr(paths, recursive):
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _load_paths_file(path: Path) -> list[Path]:
    out: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        out.append(Path(value))
    return out


def _group_names(parent: zarr.Group) -> list[str]:
    if hasattr(parent, "group_keys"):
        return sorted(str(name) for name in parent.group_keys())
    names: list[str] = []
    for name in parent.keys():
        try:
            obj = parent[name]
        except Exception:
            continue
        if _is_group(obj):
            names.append(str(name))
    return sorted(names)


def _latest_run_name(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_attr(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    names = _group_names(parent)
    return names[-1] if names else None


def _ordered_run_names(parent: zarr.Group) -> list[str]:
    names = _group_names(parent)
    latest = _normalize_attr(parent.attrs.get("latest"))
    ordered: list[str] = []
    if latest and latest in names:
        ordered.append(latest)
    for name in reversed(names):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _eye_mode_order(eye_mode: str) -> tuple[str, ...]:
    if eye_mode == "auto":
        return ("lr", "union")
    return (eye_mode,)


def _subject_run_supports_eye_mode(run_group: zarr.Group, eye_mode: str) -> bool:
    mask_labels = _canonical_components(run_group.attrs.get("mask_labels"))
    if not mask_labels:
        return False

    available_arr = run_group.get("available_channels")
    if available_arr is None:
        return False

    try:
        available_channels = _as_array(available_arr, dtype=bool)
        _resolve_eye_components(mask_labels, available_channels, eye_mode=eye_mode)
    except ValueError:
        return False
    return True


def _resolve_subject_run(
    root: zarr.Group,
    *,
    stage: str,
    explicit_run: Optional[str],
    eye_mode: str,
) -> tuple[str, str, zarr.Group]:
    if stage not in SUBJECT_STAGE_CHOICES:
        raise ValueError(f"Unsupported subject-mask stage '{stage}'.")

    stage_order = ("refined_subject_masks_runs", "subject_mask_runs") if stage == "auto" else (stage,)

    if explicit_run:
        for stage_name in stage_order:
            parent = root.get(stage_name)
            if _is_group(parent) and explicit_run in parent:
                return stage_name, str(explicit_run), parent[str(explicit_run)]
        raise ValueError(f"Subject-mask run '{explicit_run}' not found in selected stage(s): {stage_order}.")

    first_available: Optional[tuple[str, str, zarr.Group]] = None
    for stage_name in stage_order:
        parent = root.get(stage_name)
        if not _is_group(parent):
            continue
        run_names = _ordered_run_names(parent)
        if not run_names:
            continue

        if first_available is None:
            first_name = run_names[0]
            first_available = (stage_name, first_name, parent[first_name])

        for resolved_eye_mode in _eye_mode_order(eye_mode):
            for run_name in run_names:
                candidate = parent[run_name]
                if _subject_run_supports_eye_mode(candidate, resolved_eye_mode):
                    return stage_name, run_name, candidate

    if first_available is not None:
        return first_available
    raise ValueError(f"No subject-mask runs found in selected stage(s): {stage_order}.")


def _resolve_keypoint_run(
    root: zarr.Group,
    subject_group: zarr.Group,
    *,
    explicit_group: Optional[str],
    explicit_run: Optional[str],
    allow_latest_fallback: bool,
) -> tuple[str, str, zarr.Group, list[str]]:
    notes: list[str] = []

    def _resolve_within_group(group_name: str, run_name: Optional[str]) -> tuple[str, str, zarr.Group]:
        group = root.get(group_name)
        if not _is_group(group):
            raise ValueError(f"Keypoint group '{group_name}' not found.")
        if run_name:
            if run_name not in group:
                raise ValueError(f"Keypoint run '{run_name}' not found in {group_name}.")
            return group_name, str(run_name), group[str(run_name)]
        latest = _latest_run_name(group)
        if latest is None:
            raise ValueError(f"No runs found in keypoint group '{group_name}'.")
        notes.append(f"using_latest_override_group:{group_name}/{latest}")
        return group_name, latest, group[latest]

    if explicit_group:
        return (*_resolve_within_group(str(explicit_group), explicit_run), notes)

    refined = root.get("refined_keypoints_runs")
    raw = root.get("keypoints_runs")

    if explicit_run:
        in_refined = _is_group(refined) and explicit_run in refined
        in_raw = _is_group(raw) and explicit_run in raw
        if in_refined and in_raw:
            notes.append(
                f"keypoint_run '{explicit_run}' found in both groups; preferring refined_keypoints_runs"
            )
            return "refined_keypoints_runs", str(explicit_run), refined[str(explicit_run)], notes
        if in_refined:
            return "refined_keypoints_runs", str(explicit_run), refined[str(explicit_run)], notes
        if in_raw:
            return "keypoints_runs", str(explicit_run), raw[str(explicit_run)], notes
        raise ValueError(f"Keypoint run '{explicit_run}' not found in refined/raw keypoint groups.")

    source_group = _normalize_attr(subject_group.attrs.get("source_keypoint_group"))
    source_run = _normalize_attr(resolve_source_keypoints_run(subject_group.attrs))
    if source_group and source_run:
        source_parent = root.get(source_group)
        if _is_group(source_parent) and source_run in source_parent:
            notes.append(f"using_source_lineage:{source_group}/{source_run}")
            return source_group, source_run, source_parent[source_run], notes
        raise ValueError(f"Source lineage keypoint run missing: {source_group}/{source_run}")
    if source_run:
        matches: list[tuple[str, zarr.Group]] = []
        for group_name in KEYPOINT_GROUP_CHOICES:
            group = root.get(group_name)
            if _is_group(group) and source_run in group:
                matches.append((group_name, group[source_run]))
        if len(matches) == 1:
            group_name, group = matches[0]
            notes.append(f"using_source_run:{group_name}/{source_run}")
            return group_name, source_run, group, notes
        if len(matches) > 1:
            notes.append(
                f"source keypoint run '{source_run}' found in both groups; preferring refined_keypoints_runs"
            )
            refined_group = root["refined_keypoints_runs"]
            return "refined_keypoints_runs", source_run, refined_group[source_run], notes
        raise ValueError(f"Source keypoint run '{source_run}' not found in keypoint groups.")

    source_subject_lineage = _resolve_keypoint_lineage_from_source_subject_run(root, subject_group)
    if source_subject_lineage is not None:
        resolved_group, resolved_run, resolved_keypoint_group, note = source_subject_lineage
        notes.append(note)
        return resolved_group, resolved_run, resolved_keypoint_group, notes

    if not allow_latest_fallback:
        raise ValueError(
            "Subject-mask run is missing keypoint lineage. Pass --keypoint-run or "
            "--allow-latest-keypoint-fallback for explicit recovery behavior."
        )

    for group_name in KEYPOINT_GROUP_CHOICES:
        group = root.get(group_name)
        if not _is_group(group):
            continue
        latest = _latest_run_name(group)
        if latest:
            notes.append(f"fallback_latest:{group_name}/{latest}")
            return group_name, latest, group[latest], notes
    raise ValueError("No keypoint runs found in refined/raw keypoint groups.")


def _resolve_success_dataset_name(kp_group: zarr.Group) -> Optional[str]:
    for name in SUCCESS_DATASET_CANDIDATES:
        if name in kp_group:
            return name
    return None


def _run_from_stage_path(value: object, stage_name: str) -> Optional[str]:
    text = _normalize_attr(value)
    if not text:
        return None
    marker = f"{stage_name}/"
    if marker not in text:
        return None
    suffix = text.split(marker, 1)[1]
    run_name = suffix.split("/", 1)[0].strip()
    return run_name or None


def _resolve_source_refined_eye_masks_run(subject_group: zarr.Group) -> Optional[str]:
    provenance = _coerce_mapping(subject_group.attrs.get("provenance")) or {}
    provenance_inputs = _coerce_mapping(provenance.get("inputs")) or {}
    return (
        _normalize_attr(subject_group.attrs.get("source_refined_eye_masks_run"))
        or _normalize_attr(provenance_inputs.get("source_refined_eye_masks_run"))
        or _run_from_stage_path(subject_group.attrs.get("source_probability_path"), "refined_eye_masks_runs")
        or _run_from_stage_path(provenance_inputs.get("source_probability_path"), "refined_eye_masks_runs")
    )


def _resolve_source_eye_masks_run(subject_group: zarr.Group) -> Optional[str]:
    provenance = _coerce_mapping(subject_group.attrs.get("provenance")) or {}
    provenance_inputs = _coerce_mapping(provenance.get("inputs")) or {}
    return (
        _normalize_attr(subject_group.attrs.get("source_eye_masks_run"))
        or _normalize_attr(provenance_inputs.get("source_eye_masks_run"))
        or _run_from_stage_path(subject_group.attrs.get("source_probability_path"), "eye_masks_runs")
        or _run_from_stage_path(provenance_inputs.get("source_probability_path"), "eye_masks_runs")
    )


def _resolve_latest_refined_eye_masks_run(root: zarr.Group) -> Optional[str]:
    parent = root.get("refined_eye_masks_runs")
    if not _is_group(parent):
        return None
    return _latest_run_name(parent)


def _coerce_mapping(value: object) -> Optional[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return value
    return None


def _resolve_keypoint_lineage_from_source_subject_run(
    root: zarr.Group,
    subject_group: zarr.Group,
) -> tuple[str, str, zarr.Group, str] | None:
    provenance = _coerce_mapping(subject_group.attrs.get("provenance")) or {}
    provenance_inputs = _coerce_mapping(provenance.get("inputs")) or {}
    source_subject_run = (
        _normalize_attr(provenance_inputs.get("source_eye_subject_mask_run"))
        or _normalize_attr(subject_group.attrs.get("source_eye_subject_mask_run"))
        or _normalize_attr(provenance_inputs.get("source_subject_mask_run"))
        or _normalize_attr(subject_group.attrs.get("source_subject_mask_run"))
    )
    if not source_subject_run:
        return None

    subject_parent = root.get("subject_mask_runs")
    if not _is_group(subject_parent) or source_subject_run not in subject_parent:
        return None
    source_subject_group = subject_parent[source_subject_run]

    source_group = _normalize_attr(source_subject_group.attrs.get("source_keypoint_group"))
    source_run = _normalize_attr(resolve_source_keypoints_run(source_subject_group.attrs))
    if not source_run:
        return None

    if source_group:
        keypoint_parent = root.get(source_group)
        if _is_group(keypoint_parent) and source_run in keypoint_parent:
            return (
                source_group,
                source_run,
                keypoint_parent[source_run],
                f"using_source_eye_subject_lineage:subject_mask_runs/{source_subject_run}",
            )

    matches: list[tuple[str, zarr.Group]] = []
    for group_name in KEYPOINT_GROUP_CHOICES:
        keypoint_parent = root.get(group_name)
        if _is_group(keypoint_parent) and source_run in keypoint_parent:
            matches.append((group_name, keypoint_parent[source_run]))
    if len(matches) == 1:
        group_name, keypoint_group = matches[0]
        return (
            group_name,
            source_run,
            keypoint_group,
            f"using_source_eye_subject_lineage:subject_mask_runs/{source_subject_run}",
        )
    if len(matches) > 1:
        refined_parent = root["refined_keypoints_runs"]
        return (
            "refined_keypoints_runs",
            source_run,
            refined_parent[source_run],
            f"using_source_eye_subject_lineage:subject_mask_runs/{source_subject_run}",
        )
    return None


def _resolve_eye_components(
    mask_labels: Sequence[str],
    available_channels: np.ndarray,
    *,
    eye_mode: str,
) -> EyeComponentSelection:
    label_to_idx: dict[str, int] = {}
    for idx, label in enumerate(mask_labels):
        label_to_idx.setdefault(label, int(idx))

    def _available(component: str) -> bool:
        idx = label_to_idx.get(component)
        if idx is None or idx >= int(available_channels.shape[0]):
            return False
        return bool(available_channels[idx])

    lr_available = all(_available(component) for component in EYE_PAIR_COMPONENTS)
    union_available = _available(EYE_UNION_COMPONENT)

    if eye_mode == "auto":
        if lr_available:
            eye_mode = "lr"
        elif union_available:
            eye_mode = "union"
        else:
            raise ValueError(
                "No available modern eye components found. Expected available eye_left+eye_right "
                "or available eyes_union in mask_labels/available_channels."
            )

    if eye_mode == "lr":
        missing = [component for component in EYE_PAIR_COMPONENTS if component not in label_to_idx]
        unavailable = [
            component for component in EYE_PAIR_COMPONENTS if component in label_to_idx and not _available(component)
        ]
        if missing or unavailable:
            parts: list[str] = []
            if missing:
                parts.append(f"missing labels: {', '.join(missing)}")
            if unavailable:
                parts.append(f"unavailable channels: {', '.join(unavailable)}")
            raise ValueError("Requested LR eye components are not available (" + "; ".join(parts) + ").")
        return EyeComponentSelection(
            mode="lr",
            required_components=EYE_PAIR_COMPONENTS,
            component_indices={component: label_to_idx[component] for component in EYE_PAIR_COMPONENTS},
        )

    if eye_mode == "union":
        if EYE_UNION_COMPONENT not in label_to_idx:
            raise ValueError("Requested union eye component is missing label: eyes_union.")
        if not _available(EYE_UNION_COMPONENT):
            raise ValueError("Requested union eye component is unavailable: eyes_union.")
        return EyeComponentSelection(
            mode="union",
            required_components=(EYE_UNION_COMPONENT,),
            component_indices={EYE_UNION_COMPONENT: label_to_idx[EYE_UNION_COMPONENT]},
        )

    raise ValueError(f"Unsupported eye component mode '{eye_mode}'.")


def _component_review_states(run_group: zarr.Group, components: Sequence[str]) -> dict[str, str]:
    raw = run_group.attrs.get("component_review_statuses")
    if not isinstance(raw, Mapping):
        return {}
    states: dict[str, str] = {}
    for component in components:
        payload = raw.get(component)
        if isinstance(payload, Mapping):
            state = _normalize_attr(payload.get("state"))
            if state:
                states[str(component)] = state
    return states


def _compute_keypoint_eye_valid_mask(
    keypoints_roi: np.ndarray,
    success_flags: np.ndarray,
    *,
    eye_indices: Mapping[str, int],
) -> np.ndarray:
    if keypoints_roi.ndim != 3 or keypoints_roi.shape[2] < 2:
        raise ValueError("keypoints_roi must have shape (N, K, >=2).")
    if success_flags.ndim != 1:
        raise ValueError("Keypoint success array must have shape (N,).")
    if int(keypoints_roi.shape[0]) != int(success_flags.shape[0]):
        raise ValueError(
            f"Row mismatch between keypoints_roi ({keypoints_roi.shape[0]}) "
            f"and success flags ({success_flags.shape[0]})."
        )

    try:
        left_idx = int(eye_indices["eye_left"])
        right_idx = int(eye_indices["eye_right"])
    except KeyError as exc:
        raise ValueError("Resolved keypoint eye indices must include eye_left and eye_right.") from exc

    keypoint_count = int(keypoints_roi.shape[1])
    if left_idx >= keypoint_count or right_idx >= keypoint_count:
        raise ValueError(
            f"Resolved keypoint eye indices out of bounds for keypoints_roi shape {tuple(keypoints_roi.shape)}."
        )

    left = np.asarray(keypoints_roi[:, left_idx, :2], dtype=np.float32)
    right = np.asarray(keypoints_roi[:, right_idx, :2], dtype=np.float32)
    finite = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
    distinct = np.any(np.abs(left - right) > 1e-6, axis=1)
    return np.asarray(success_flags, dtype=bool) & finite & distinct


def _component_present_from_metrics(
    run_group: zarr.Group,
    *,
    total_rois: int,
    component_indices: Mapping[str, int],
) -> Optional[dict[str, np.ndarray]]:
    metrics = run_group.get("metrics")
    if not _is_group(metrics):
        return None
    mask_present_arr = metrics.get("mask_present")
    if mask_present_arr is None:
        return None
    mask_present = _as_array(mask_present_arr, dtype=bool)
    if mask_present.ndim != 2 or int(mask_present.shape[0]) != int(total_rois):
        return None
    max_idx = max(component_indices.values(), default=-1)
    if max_idx >= int(mask_present.shape[1]):
        return None
    return {
        component: np.asarray(mask_present[:, idx], dtype=bool)
        for component, idx in component_indices.items()
    }


def _component_present_from_masks(
    masks_arr: object,
    *,
    total_rois: int,
    component_indices: Mapping[str, int],
) -> dict[str, np.ndarray]:
    present: dict[str, np.ndarray] = {}
    for component, idx in component_indices.items():
        data = masks_arr[:, idx, :, :]  # type: ignore[index]
        arr = np.asarray(data)
        if arr.ndim != 3 or int(arr.shape[0]) != int(total_rois):
            raise ValueError(
                f"masks_roi component '{component}' slice has invalid shape {tuple(arr.shape)}."
            )
        present[component] = np.asarray(arr.reshape(arr.shape[0], -1).any(axis=1), dtype=bool)
    return present


def _resolve_component_present(
    run_group: zarr.Group,
    masks_arr: object,
    *,
    total_rois: int,
    component_indices: Mapping[str, int],
    notes: list[str],
) -> dict[str, np.ndarray]:
    from_metrics = _component_present_from_metrics(
        run_group,
        total_rois=total_rois,
        component_indices=component_indices,
    )
    if from_metrics is not None:
        notes.append("using_metrics:metrics/mask_present")
        return from_metrics
    notes.append("computed_component_presence_from:masks_roi")
    return _component_present_from_masks(
        masks_arr,
        total_rois=total_rois,
        component_indices=component_indices,
    )


def _resolve_frame_indices(root: zarr.Group, subject_group: zarr.Group, total_rois: int) -> Optional[np.ndarray]:
    direct = subject_group.get("frame_indices")
    if direct is not None:
        candidate = _as_array(direct, dtype=np.int64)
        if candidate.ndim == 1 and int(candidate.shape[0]) == int(total_rois):
            return candidate

    source_crop_run = _normalize_attr(subject_group.attrs.get("source_crop_run"))
    if not source_crop_run:
        return None
    crop_parent = root.get("crop_runs")
    if not _is_group(crop_parent) or source_crop_run not in crop_parent:
        return None
    crop_group = crop_parent[source_crop_run]
    crop_frame_indices = crop_group.get("frame_indices")
    if crop_frame_indices is None:
        return None
    candidate = _as_array(crop_frame_indices, dtype=np.int64)
    if candidate.ndim != 1 or int(candidate.shape[0]) != int(total_rois):
        return None
    return candidate


def _collect_failure_samples(
    fail_indices: np.ndarray,
    frame_indices: Optional[np.ndarray],
    limit: int,
) -> list[dict[str, int]]:
    samples: list[dict[str, int]] = []
    for roi_idx in fail_indices[: max(0, int(limit))]:
        payload = {"roi_idx": int(roi_idx)}
        if frame_indices is not None:
            payload["frame_idx"] = int(frame_indices[int(roi_idx)])
        samples.append(payload)
    return samples


def _evaluate_eyes_union_assignment(
    masks_arr: object,
    *,
    union_component_index: int,
    keypoints_roi: np.ndarray,
    success_flags: np.ndarray,
    keypoint_valid: np.ndarray,
    eye_indices: Mapping[str, int],
) -> tuple[str, dict[str, object]]:
    assignment = assign_eyes_union_to_lr(
        np.asarray(masks_arr[:, union_component_index, :, :], dtype=np.uint8),  # type: ignore[index]
        keypoints_roi=np.asarray(keypoints_roi, dtype=np.float32),
        keypoint_success=np.asarray(success_flags, dtype=bool),
        eye_keypoint_indices=(int(eye_indices["eye_left"]), int(eye_indices["eye_right"])),
    )
    summary = dict(assignment.summary)
    status = np.asarray(assignment.assignment_status, dtype=object)
    usable_status = np.isin(status, ["assigned", "assigned_needs_review"])
    valid_rows = np.asarray(keypoint_valid, dtype=bool)
    valid_usable_rows = int(np.count_nonzero(valid_rows & usable_status))
    valid_needs_review_rows = int(np.count_nonzero(valid_rows & (status == "assigned_needs_review")))
    valid_failed_rows = int(np.count_nonzero(valid_rows & ~usable_status))

    summary["keypoint_valid_rows"] = int(np.count_nonzero(valid_rows))
    summary["keypoint_valid_assigned_rows"] = valid_usable_rows
    summary["keypoint_valid_assigned_needs_review_rows"] = valid_needs_review_rows
    summary["keypoint_valid_failed_rows"] = valid_failed_rows

    if int(summary["keypoint_valid_rows"]) <= 0:
        readiness = "no_keypoint_valid_rows"
    elif valid_usable_rows <= 0:
        readiness = "not_ready"
    elif valid_failed_rows > 0 or valid_needs_review_rows > 0:
        readiness = "ready_partial"
    else:
        readiness = "ready"
    return readiness, summary


def _analyze_root(
    *,
    root: zarr.Group,
    zarr_path: Path,
    stage: str,
    subject_run: Optional[str],
    eye_mode: str,
    keypoint_group: Optional[str],
    keypoint_run: Optional[str],
    allow_latest_keypoint_fallback: bool,
    sample_limit: int,
) -> CoverageReport:
    try:
        resolved_stage, resolved_subject_run, subject_group = _resolve_subject_run(
            root,
            stage=stage,
            explicit_run=subject_run,
            eye_mode=eye_mode,
        )
    except ValueError as exc:
        return CoverageReport(zarr_path=zarr_path, status="missing", reason=str(exc))

    report = CoverageReport(
        zarr_path=zarr_path,
        status="pass",
        subject_stage=resolved_stage,
        subject_run=resolved_subject_run,
        label_schema_id=_normalize_attr(subject_group.attrs.get("label_schema_id")),
    )

    masks_arr = subject_group.get("masks_roi")
    if masks_arr is None:
        report.status = "missing"
        report.reason = f"{resolved_stage}/{resolved_subject_run} missing masks_roi."
        return report
    masks_shape = tuple(int(dim) for dim in getattr(masks_arr, "shape", ()))
    if len(masks_shape) != 4:
        report.status = "missing"
        report.reason = (
            f"{resolved_stage}/{resolved_subject_run}/masks_roi must have shape (N, C, H, W), "
            f"got {masks_shape}."
        )
        return report
    report.total_rois = int(masks_shape[0])
    channel_count = int(masks_shape[1])

    mask_labels = _canonical_components(subject_group.attrs.get("mask_labels"))
    if not mask_labels:
        report.status = "missing"
        report.reason = f"{resolved_stage}/{resolved_subject_run} missing mask_labels attr."
        return report
    if len(mask_labels) != channel_count:
        report.status = "missing"
        report.reason = (
            f"{resolved_stage}/{resolved_subject_run} mask_labels count {len(mask_labels)} "
            f"does not match masks_roi channel count {channel_count}."
        )
        report.mask_labels = mask_labels
        return report
    report.mask_labels = list(mask_labels)

    available_arr = subject_group.get("available_channels")
    if available_arr is None:
        report.status = "missing"
        report.reason = f"{resolved_stage}/{resolved_subject_run} missing available_channels."
        return report
    available_channels = _as_array(available_arr, dtype=bool)
    if available_channels.ndim != 1 or int(available_channels.shape[0]) != channel_count:
        report.status = "missing"
        report.reason = (
            f"{resolved_stage}/{resolved_subject_run}/available_channels must have shape ({channel_count},), "
            f"got {tuple(available_channels.shape)}."
        )
        return report
    report.available_components = [
        label for label, available in zip(mask_labels, available_channels) if bool(available)
    ]

    try:
        selection = _resolve_eye_components(mask_labels, available_channels, eye_mode=eye_mode)
    except ValueError as exc:
        report.status = "missing"
        report.reason = str(exc)
        return report
    report.eye_component_mode = selection.mode
    report.eye_component_indices = dict(selection.component_indices)
    report.component_review_states = _component_review_states(subject_group, selection.required_components)
    report.source_refined_eye_masks_run = _resolve_source_refined_eye_masks_run(subject_group)
    report.source_eye_masks_run = _resolve_source_eye_masks_run(subject_group)
    report.latest_refined_eye_masks_run = _resolve_latest_refined_eye_masks_run(root)

    try:
        kp_group_name, kp_run_name, kp_group, kp_notes = _resolve_keypoint_run(
            root,
            subject_group,
            explicit_group=keypoint_group,
            explicit_run=keypoint_run,
            allow_latest_fallback=allow_latest_keypoint_fallback,
        )
    except ValueError as exc:
        report.status = "missing"
        report.reason = str(exc)
        return report
    report.keypoint_group = kp_group_name
    report.keypoint_run = kp_run_name
    report.notes.extend(kp_notes)

    keypoints_arr = kp_group.get("keypoints_roi")
    if keypoints_arr is None:
        report.status = "missing"
        report.reason = f"{kp_group_name}/{kp_run_name} missing keypoints_roi."
        return report
    success_name = _resolve_success_dataset_name(kp_group)
    if success_name is None:
        report.status = "missing"
        report.reason = (
            f"{kp_group_name}/{kp_run_name} missing success dataset "
            f"({', '.join(SUCCESS_DATASET_CANDIDATES)})."
        )
        return report
    success_arr = kp_group.get(success_name)
    if success_arr is None:
        report.status = "missing"
        report.reason = f"{kp_group_name}/{kp_run_name} missing {success_name}."
        return report
    report.success_dataset = success_name

    keypoints_roi = _as_array(keypoints_arr, dtype=np.float32)
    success_flags = _as_array(success_arr, dtype=bool)
    try:
        eye_indices = resolve_required_keypoint_indices_from_attrs(
            kp_group.attrs,
            EYE_PAIR_COMPONENTS,
            keypoint_count=int(keypoints_roi.shape[1]) if keypoints_roi.ndim == 3 else None,
        )
        assert_row_alignment(
            report.total_rois,
            (
                (f"{kp_group_name}/{kp_run_name}/keypoints_roi", keypoints_roi),
                (f"{kp_group_name}/{kp_run_name}/{success_name}", success_flags),
            ),
            stage=f"{resolved_stage}/{resolved_subject_run} coverage inputs",
        )
        keypoint_valid = _compute_keypoint_eye_valid_mask(
            keypoints_roi,
            success_flags,
            eye_indices=eye_indices,
        )
        component_present = _resolve_component_present(
            subject_group,
            masks_arr,
            total_rois=report.total_rois,
            component_indices=selection.component_indices,
            notes=report.notes,
        )
    except ValueError as exc:
        report.status = "missing"
        report.reason = str(exc)
        return report

    report.keypoint_eye_indices = {key: int(value) for key, value in eye_indices.items()}
    report.component_present_rows = {
        component: int(np.asarray(values, dtype=bool).sum())
        for component, values in component_present.items()
    }

    if selection.mode == "lr":
        eye_component_present = np.ones((report.total_rois,), dtype=bool)
        for component in EYE_PAIR_COMPONENTS:
            eye_component_present &= np.asarray(component_present[component], dtype=bool)
    else:
        eye_component_present = np.asarray(component_present[EYE_UNION_COMPONENT], dtype=bool)
        try:
            assignment_status, assignment_summary = _evaluate_eyes_union_assignment(
                masks_arr,
                union_component_index=int(selection.component_indices[EYE_UNION_COMPONENT]),
                keypoints_roi=keypoints_roi,
                success_flags=success_flags,
                keypoint_valid=keypoint_valid,
                eye_indices=eye_indices,
            )
            report.eyes_union_assignment_status = assignment_status
            report.eyes_union_assignment_summary = assignment_summary
            report.notes.append("eyes_union_assignment_dry_run_checked")
        except ValueError as exc:
            report.eyes_union_assignment_status = "error"
            report.eyes_union_assignment_summary = {"error": str(exc)}
            report.notes.append("eyes_union_assignment_dry_run_error")

    fail_rows = np.flatnonzero(keypoint_valid & ~eye_component_present)
    frame_indices = _resolve_frame_indices(root, subject_group, report.total_rois)

    report.keypoint_valid_rows = int(keypoint_valid.sum())
    report.rows_with_eye_component_masks = int((keypoint_valid & eye_component_present).sum())
    report.rows_missing_eye_component_masks = int(fail_rows.size)
    report.failure_targets = _collect_failure_samples(fail_rows, frame_indices, int(fail_rows.size))
    report.sample_missing = _collect_failure_samples(fail_rows, frame_indices, sample_limit)

    if report.keypoint_valid_rows == 0:
        report.notes.append("no_keypoint_valid_rows")
    if fail_rows.size > 0:
        report.status = "fail"
        report.reason = "keypoint-valid rows missing required subject-mask eye component(s)."
    return report


def _report_to_log_payload(report: CoverageReport) -> dict[str, object]:
    return {
        "zarr": str(report.zarr_path),
        "status": report.status,
        "reason": report.reason,
        "subject_stage": report.subject_stage,
        "subject_run": report.subject_run,
        "label_schema_id": report.label_schema_id,
        "mask_labels": report.mask_labels,
        "available_components": report.available_components,
        "eye_component_mode": report.eye_component_mode,
        "eye_component_indices": report.eye_component_indices,
        "component_review_states": report.component_review_states,
        "source_refined_eye_masks_run": report.source_refined_eye_masks_run,
        "source_eye_masks_run": report.source_eye_masks_run,
        "latest_refined_eye_masks_run": report.latest_refined_eye_masks_run,
        "keypoint_group": report.keypoint_group,
        "keypoint_run": report.keypoint_run,
        "success_dataset": report.success_dataset,
        "keypoint_eye_indices": report.keypoint_eye_indices,
        "total_rois": report.total_rois,
        "keypoint_valid_rows": report.keypoint_valid_rows,
        "rows_with_eye_component_masks": report.rows_with_eye_component_masks,
        "rows_missing_eye_component_masks": report.rows_missing_eye_component_masks,
        "component_present_rows": report.component_present_rows,
        "eyes_union_assignment_status": report.eyes_union_assignment_status,
        "eyes_union_assignment_summary": report.eyes_union_assignment_summary,
        "failure_targets": report.failure_targets,
        "sample_missing": report.sample_missing,
        "notes": report.notes,
    }


def _print_report(report: CoverageReport, *, show_pass: bool) -> None:
    if report.status == "pass" and not show_pass:
        return

    print(f"\n=== {report.zarr_path} ===")
    print(f"status: {report.status.upper()}")
    if report.reason:
        print(f"reason: {report.reason}")
    if report.subject_stage and report.subject_run:
        print(f"subject_run: {report.subject_stage}/{report.subject_run}")
    if report.label_schema_id:
        print(f"label_schema_id: {report.label_schema_id}")
    if report.eye_component_mode:
        print(f"eye_component_mode: {report.eye_component_mode}")
    if report.eye_component_indices:
        print(f"eye_component_indices: {report.eye_component_indices}")
    if report.component_review_states:
        print(f"component_review_states: {report.component_review_states}")
    if report.source_refined_eye_masks_run:
        print(f"source_refined_eye_masks_run: {report.source_refined_eye_masks_run}")
    if report.latest_refined_eye_masks_run:
        print(f"latest_refined_eye_masks_run: {report.latest_refined_eye_masks_run}")
    if report.keypoint_group and report.keypoint_run:
        print(f"keypoint_run: {report.keypoint_group}/{report.keypoint_run}")
    if report.keypoint_eye_indices:
        print(f"keypoint_eye_indices: {report.keypoint_eye_indices}")
    if report.success_dataset:
        print(f"success_dataset: {report.success_dataset}")
    if report.total_rois:
        print(f"total_rois: {report.total_rois}")
        print(f"keypoint_valid_rows: {report.keypoint_valid_rows}")
        print(f"rows_with_eye_component_masks: {report.rows_with_eye_component_masks}")
        print(f"rows_missing_eye_component_masks: {report.rows_missing_eye_component_masks}")
        if report.component_present_rows:
            print(f"component_present_rows: {report.component_present_rows}")
    if report.eyes_union_assignment_status:
        print(f"eyes_union_assignment_status: {report.eyes_union_assignment_status}")
    if report.eyes_union_assignment_summary:
        summary = report.eyes_union_assignment_summary
        fields = [
            "assigned_rows",
            "assigned_needs_review_rows",
            "failed_rows",
            "keypoint_valid_assigned_rows",
            "keypoint_valid_assigned_needs_review_rows",
            "keypoint_valid_failed_rows",
        ]
        compact = {key: summary[key] for key in fields if key in summary}
        print(f"eyes_union_assignment_summary: {compact}")
    for note in report.notes:
        print(f"note: {note}")
    for sample in report.sample_missing:
        if "frame_idx" in sample:
            print(f"sample_missing: roi={sample['roi_idx']} frame={sample['frame_idx']}")
        else:
            print(f"sample_missing: roi={sample['roi_idx']}")


def _append_review_list(path: Path, zarr_paths: Sequence[Path]) -> int:
    raw_lines: list[str] = []
    existing: set[str] = set()
    if path.exists():
        raw_lines = path.read_text(encoding="utf-8").splitlines()
        for line in raw_lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            existing.add(value)

    added = 0
    for zarr_path in zarr_paths:
        value = str(zarr_path)
        if value in existing:
            continue
        existing.add(value)
        raw_lines.append(value)
        added += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    output = "\n".join(raw_lines)
    if output and not output.endswith("\n"):
        output += "\n"
    path.write_text(output, encoding="utf-8")
    return added


def _write_frame_flag_file(path: Path, reports: Sequence[CoverageReport]) -> tuple[int, int]:
    payload: dict[str, list[dict[str, int]]] = {}
    target_count = 0
    for report in reports:
        if report.status != "fail" or not report.failure_targets:
            continue
        entries = [dict(item) for item in report.failure_targets]
        payload[str(report.zarr_path)] = entries
        target_count += len(entries)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(payload), target_count


def _shell_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def _command_payload(argv: Sequence[str]) -> dict[str, object]:
    normalized = [str(item) for item in argv]
    return {
        "argv": normalized,
        "shell": _shell_command(normalized),
    }


def _preferred_refined_eye_masks_run(report: CoverageReport) -> Optional[str]:
    return report.source_refined_eye_masks_run or report.latest_refined_eye_masks_run


def _repair_plan_command_payloads(
    report: CoverageReport,
    *,
    frame_flag_file: Optional[Path],
) -> dict[str, object]:
    frame_flag_arg = str(frame_flag_file) if frame_flag_file is not None else "<frame_flag_file>"

    eye_args = [
        "scripts/py",
        "-m",
        "fisheye.tune.eye_mask_review",
        str(report.zarr_path),
        "--manual",
    ]
    refined_eye_run = _preferred_refined_eye_masks_run(report)
    if refined_eye_run:
        eye_args.extend(["--refined-run", refined_eye_run])
    eye_args.extend(
        [
            "--frame-flag-file",
            frame_flag_arg,
            "--review-state",
            "approved",
            "--review-method",
            "manual",
            "--review-intended-use",
            "training",
        ]
    )

    keypoint_args = [
        "scripts/py",
        "-m",
        "fisheye.tune.keypoint_review",
        str(report.zarr_path),
        "--manual",
        "--refined-run",
        report.keypoint_run or "<refined_keypoints_run>",
        "--frames",
        frame_flag_arg,
        "--review-state",
        "approved",
        "--review-method",
        "manual",
        "--review-intended-use",
        "training",
    ]

    keypoint_note = None
    if report.keypoint_group != "refined_keypoints_runs":
        keypoint_note = (
            "keypoint_review edits refined_keypoints_runs; resolve the refined "
            "run corresponding to this keypoint source before applying."
        )

    return {
        "eye_mask_review": {
            "purpose": "Repair missing subject/eye component masks when keypoints are valid.",
            "source_refined_eye_masks_run": report.source_refined_eye_masks_run,
            "latest_refined_eye_masks_run": report.latest_refined_eye_masks_run,
            **_command_payload(eye_args),
        },
        "keypoint_review": {
            "purpose": "Mark fish_present_no_keypoints or other keypoint-row failures.",
            "keypoint_group": report.keypoint_group,
            "keypoint_run": report.keypoint_run,
            "note": keypoint_note,
            **_command_payload(keypoint_args),
        },
    }


def _repair_plan_rows(
    reports: Sequence[CoverageReport],
    *,
    frame_flag_file: Optional[Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report in reports:
        if report.status != "fail" or not report.failure_targets:
            continue
        for target in report.failure_targets:
            row: dict[str, object] = {
                "zarr": str(report.zarr_path),
                "status": report.status,
                "reason": report.reason,
                "target": dict(target),
                "subject_stage": report.subject_stage,
                "subject_run": report.subject_run,
                "label_schema_id": report.label_schema_id,
                "eye_component_mode": report.eye_component_mode,
                "eye_component_indices": dict(report.eye_component_indices),
                "source_refined_eye_masks_run": report.source_refined_eye_masks_run,
                "source_eye_masks_run": report.source_eye_masks_run,
                "latest_refined_eye_masks_run": report.latest_refined_eye_masks_run,
                "keypoint_group": report.keypoint_group,
                "keypoint_run": report.keypoint_run,
                "success_dataset": report.success_dataset,
                "keypoint_eye_indices": dict(report.keypoint_eye_indices),
                "frame_flag_file": str(frame_flag_file) if frame_flag_file is not None else None,
                "classification_required": True,
                "classification_options": [
                    "missing_eye_component_mask",
                    "fish_present_no_keypoints",
                    "detection_or_crop_issue",
                ],
                "repair_options": _repair_plan_command_payloads(
                    report,
                    frame_flag_file=frame_flag_file,
                ),
            }
            rows.append(row)
    return rows


def _write_repair_plan_file(
    path: Path,
    reports: Sequence[CoverageReport],
    *,
    frame_flag_file: Optional[Path],
) -> int:
    rows = _repair_plan_rows(reports, frame_flag_file=frame_flag_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def _resolve_default_roots(paths: Optional[list[Path]]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: Sequence[Path]) -> Path:
    return resolve_shared_log_dir(
        arg_log_dir,
        roots,
        log_subdir="check_subject_mask_keypoint_coverage",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["all", "analysis", "training"],
        default="all",
        help="Restrict zarr scope by use (default: all).",
    )
    parser.add_argument(
        "--stage",
        choices=SUBJECT_STAGE_CHOICES,
        default="auto",
        help="Subject-mask stage to check (default: auto prefers refined_subject_masks_runs).",
    )
    parser.add_argument("--subject-run", help="Specific subject-mask run name to check.")
    parser.add_argument(
        "--eye-mode",
        choices=EYE_MODE_CHOICES,
        default="auto",
        help="Eye component mode (default: auto prefers available eye_left+eye_right over eyes_union).",
    )
    parser.add_argument(
        "--keypoint-group",
        choices=KEYPOINT_GROUP_CHOICES,
        help="Explicit keypoint group override.",
    )
    parser.add_argument("--keypoint-run", help="Explicit keypoint run override.")
    parser.add_argument(
        "--allow-latest-keypoint-fallback",
        action="store_true",
        help=(
            "Compatibility/recovery mode: use latest refined/raw keypoints when "
            "subject-mask keypoint lineage is absent."
        ),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Maximum sample failing rows to print per zarr (default: 10).",
    )
    parser.add_argument(
        "--append-review-list",
        type=Path,
        help="Append failing zarr paths to this file for downstream subject-mask review.",
    )
    parser.add_argument(
        "--write-frame-flag-file",
        type=Path,
        help=(
            "Write failing ROI/frame targets as JSON accepted by eye_mask_review "
            "--frame-flag-file. The file is overwritten with this run's failures."
        ),
    )
    parser.add_argument(
        "--write-repair-plan",
        type=Path,
        help=(
            "Write one JSONL row per failing ROI/frame with lineage metadata and "
            "candidate eye/keypoint repair commands."
        ),
    )
    parser.add_argument("--show-pass", action="store_true", help="Print PASS records too.")
    parser.add_argument("--strict", action="store_true", help="Exit with status 1 on any fail/missing/error.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help=(
            "Directory for JSONL logs (default: $PALETTE_LOG_ROOT/check_subject_mask_keypoint_coverage "
            "or <recordings_root>/logs/check_subject_mask_keypoint_coverage)."
        ),
    )
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logging.")
    return parser


def run(args: argparse.Namespace) -> int:
    file_list_paths: list[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    explicit_paths: list[Path] = []
    if args.paths:
        explicit_paths.extend(args.paths)
    if file_list_paths:
        explicit_paths.extend(file_list_paths)

    roots = _resolve_default_roots(explicit_paths if explicit_paths else None)
    zarr_paths = _collect_zarr_paths(roots, recursive=bool(args.recursive))
    if not zarr_paths:
        print("No zarr files found.")
        return 1

    logger: Optional[JsonLogger] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"check_subject_mask_keypoint_coverage_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            roots=[str(path) for path in roots],
            file_list=[str(path) for path in (args.file_list or [])],
            recursive=bool(args.recursive),
            zarr_use=str(args.zarr_use),
            stage=str(args.stage),
            subject_run=args.subject_run,
            eye_mode=str(args.eye_mode),
            keypoint_group=args.keypoint_group,
            keypoint_run=args.keypoint_run,
            allow_latest_keypoint_fallback=bool(args.allow_latest_keypoint_fallback),
            sample_limit=int(max(0, args.sample_limit)),
            append_review_list=str(args.append_review_list) if args.append_review_list else None,
            write_frame_flag_file=str(args.write_frame_flag_file) if args.write_frame_flag_file else None,
            write_repair_plan=str(args.write_repair_plan) if args.write_repair_plan else None,
        )

    zarr_scanned = 0
    zarr_checked = 0
    zarr_pass = 0
    zarr_fail = 0
    zarr_missing = 0
    zarr_error = 0
    filtered_zarr_use = 0
    union_assignment_status_counts: dict[str, int] = {}
    failing_paths: list[Path] = []
    failing_reports: list[CoverageReport] = []

    for zarr_path in zarr_paths:
        zarr_scanned += 1
        try:
            root = open_zarr_root(zarr_path, mode="r")
        except Exception as exc:
            zarr_error += 1
            report = CoverageReport(zarr_path=zarr_path, status="error", reason=str(exc))
            _print_report(report, show_pass=bool(args.show_pass))
            if logger is not None:
                logger.log("zarr_error", **_report_to_log_payload(report))
            continue

        observed_use = _infer_zarr_use(root, zarr_path)
        if args.zarr_use != "all" and observed_use != args.zarr_use:
            filtered_zarr_use += 1
            if logger is not None:
                logger.log(
                    "zarr_filtered",
                    zarr=str(zarr_path),
                    reason=f"zarr_use={observed_use or 'unknown'}",
                )
            continue

        report = _analyze_root(
            root=root,
            zarr_path=zarr_path,
            stage=str(args.stage),
            subject_run=args.subject_run,
            eye_mode=str(args.eye_mode),
            keypoint_group=args.keypoint_group,
            keypoint_run=args.keypoint_run,
            allow_latest_keypoint_fallback=bool(args.allow_latest_keypoint_fallback),
            sample_limit=max(0, int(args.sample_limit)),
        )
        _print_report(report, show_pass=bool(args.show_pass))

        zarr_checked += 1
        if report.status == "pass":
            zarr_pass += 1
        elif report.status == "fail":
            zarr_fail += 1
            failing_paths.append(zarr_path)
            failing_reports.append(report)
        elif report.status == "missing":
            zarr_missing += 1
        else:
            zarr_error += 1
        if report.eyes_union_assignment_status:
            status_key = str(report.eyes_union_assignment_status)
            union_assignment_status_counts[status_key] = (
                union_assignment_status_counts.get(status_key, 0) + 1
            )
        if logger is not None:
            logger.log("zarr_checked", **_report_to_log_payload(report))

    review_list_added = 0
    if args.append_review_list:
        review_list_path = args.append_review_list.expanduser().resolve()
        review_list_added = _append_review_list(review_list_path, failing_paths)
        print(
            f"\nReview list updated: {review_list_path} "
            f"(added={review_list_added}, failing={len(failing_paths)})"
        )
        if logger is not None:
            logger.log(
                "review_list_updated",
                path=str(review_list_path),
                added=review_list_added,
                failing=len(failing_paths),
            )

    frame_flag_zarrs_written = 0
    frame_flag_targets_written = 0
    if args.write_frame_flag_file:
        frame_flag_path = args.write_frame_flag_file.expanduser().resolve()
        frame_flag_zarrs_written, frame_flag_targets_written = _write_frame_flag_file(
            frame_flag_path,
            failing_reports,
        )
        print(
            f"\nFrame flag file written: {frame_flag_path} "
            f"(zarrs={frame_flag_zarrs_written}, targets={frame_flag_targets_written})"
        )
        if logger is not None:
            logger.log(
                "frame_flag_file_written",
                path=str(frame_flag_path),
                zarrs=frame_flag_zarrs_written,
                targets=frame_flag_targets_written,
            )

    repair_plan_rows_written = 0
    if args.write_repair_plan:
        repair_plan_path = args.write_repair_plan.expanduser().resolve()
        repair_plan_frame_flag_path = (
            args.write_frame_flag_file.expanduser().resolve()
            if args.write_frame_flag_file
            else None
        )
        repair_plan_rows_written = _write_repair_plan_file(
            repair_plan_path,
            failing_reports,
            frame_flag_file=repair_plan_frame_flag_path,
        )
        print(
            f"\nRepair plan written: {repair_plan_path} "
            f"(rows={repair_plan_rows_written})"
        )
        if logger is not None:
            logger.log(
                "repair_plan_written",
                path=str(repair_plan_path),
                rows=repair_plan_rows_written,
                frame_flag_file=(
                    str(repair_plan_frame_flag_path)
                    if repair_plan_frame_flag_path is not None
                    else None
                ),
            )

    issues = (zarr_fail > 0) or (zarr_missing > 0) or (zarr_error > 0)
    print(
        "\nSubject-mask keypoint coverage summary: "
        f"zarr_scanned={zarr_scanned} "
        f"zarr_checked={zarr_checked} "
        f"zarr_pass={zarr_pass} "
        f"zarr_fail={zarr_fail} "
        f"zarr_missing={zarr_missing} "
        f"filtered_zarr_use={filtered_zarr_use} "
        f"errors={zarr_error} "
        f"review_list_added={review_list_added} "
        f"frame_flag_targets_written={frame_flag_targets_written} "
        f"repair_plan_rows_written={repair_plan_rows_written} "
        f"union_assignment_statuses={union_assignment_status_counts} "
        f"issues={'yes' if issues else 'no'}"
    )

    if logger is not None:
        logger.log(
            "run_end",
            zarr_scanned=zarr_scanned,
            zarr_checked=zarr_checked,
            zarr_pass=zarr_pass,
            zarr_fail=zarr_fail,
            zarr_missing=zarr_missing,
            filtered_zarr_use=filtered_zarr_use,
            errors=zarr_error,
            review_list_added=review_list_added,
            frame_flag_zarrs_written=frame_flag_zarrs_written,
            frame_flag_targets_written=frame_flag_targets_written,
            repair_plan_rows_written=repair_plan_rows_written,
            union_assignment_statuses=dict(union_assignment_status_counts),
            issues=issues,
        )
        logger.close()

    if zarr_checked == 0:
        return 1
    if args.strict and issues:
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
