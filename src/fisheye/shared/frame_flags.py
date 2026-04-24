"""Shared helpers for ROI/frame flag JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


FLAG_IDENTITY_FIELDS = ("source_refined_row_id", "source_detect_row_index")


def _coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_json_scalar(value: object) -> bool:
    return isinstance(value, (str, bool, int, float)) or value is None


def normalize_flag_entry(
    item: object,
    *,
    preserve_extra_keys: Sequence[str] = (),
) -> Optional[dict[str, object]]:
    """Normalize a legacy frame flag entry into the dict shape used by tools."""

    if isinstance(item, Mapping):
        frame_idx = _coerce_int(item.get("frame_idx"))
        roi_idx = _coerce_int(item.get("roi_idx"))
        payload: dict[str, object] = {"frame_idx": frame_idx, "roi_idx": roi_idx}

        for key in FLAG_IDENTITY_FIELDS:
            if key not in item:
                continue
            value = _coerce_int(item.get(key))
            if value is not None:
                payload[key] = value

        for key in preserve_extra_keys:
            if key in FLAG_IDENTITY_FIELDS or key not in item:
                continue
            value = item.get(key)
            if _is_json_scalar(value):
                payload[key] = value

        if (
            frame_idx is None
            and roi_idx is None
            and not any(key in payload for key in FLAG_IDENTITY_FIELDS)
        ):
            return None
        return payload

    frame_idx = _coerce_int(item)
    if frame_idx is None:
        return None
    return {"frame_idx": frame_idx, "roi_idx": None}


def load_frame_flags(
    path: Path,
    *,
    preserve_extra_keys: Sequence[str] = (),
) -> dict[str, list[dict[str, object]]]:
    """Load a JSON frame-flag file while preserving known scalar metadata."""

    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")

    parsed: dict[str, list[dict[str, object]]] = {}
    for key, value in data.items():
        entries: list[dict[str, object]] = []
        if isinstance(value, list):
            for item in value:
                entry = normalize_flag_entry(item, preserve_extra_keys=preserve_extra_keys)
                if entry is not None:
                    entries.append(entry)
        parsed[str(key)] = entries
    return parsed


def _entry_identity_key(entry: Mapping[str, object]) -> tuple[str, int] | None:
    refined_row_id = _coerce_int(entry.get("source_refined_row_id"))
    if refined_row_id is not None and refined_row_id >= 0:
        return "source_refined_row_id", int(refined_row_id)
    source_detect_row = _coerce_int(entry.get("source_detect_row_index"))
    if source_detect_row is not None and source_detect_row >= 0:
        return "source_detect_row_index", int(source_detect_row)
    return None


def _entry_frame_roi_key(entry: Mapping[str, object]) -> tuple[int | None, int | None]:
    return _coerce_int(entry.get("frame_idx")), _coerce_int(entry.get("roi_idx"))


def _merge_flag_entry(target: dict[str, object], source: Mapping[str, object]) -> bool:
    changed = False
    for key, value in source.items():
        if key in {"frame_idx", "roi_idx"}:
            continue
        if not _is_json_scalar(value):
            continue
        if target.get(key) != value:
            target[key] = value
            changed = True
    return changed


def append_flagged_frame(
    flag_path: Path,
    zarr_path: str,
    frame_idx: int,
    roi_idx: Optional[int],
    *,
    extra_fields: Optional[Mapping[str, object]] = None,
    preserve_extra_keys: Sequence[str] = (),
) -> None:
    """Append or upgrade a flagged ROI entry without breaking legacy files."""

    flag_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_frame_flags(flag_path, preserve_extra_keys=preserve_extra_keys)
    entries = data.get(zarr_path, [])
    payload: dict[str, object] = {
        "frame_idx": int(frame_idx),
        "roi_idx": int(roi_idx) if roi_idx is not None else None,
    }
    if extra_fields:
        _merge_flag_entry(payload, extra_fields)

    identity_key = _entry_identity_key(payload)
    frame_roi_key = _entry_frame_roi_key(payload)
    for entry in entries:
        entry_identity_key = _entry_identity_key(entry)
        if identity_key is not None and entry_identity_key == identity_key:
            _merge_flag_entry(entry, payload)
            break
        if entry_identity_key is None and _entry_frame_roi_key(entry) == frame_roi_key:
            _merge_flag_entry(entry, payload)
            break
        if identity_key is None and _entry_frame_roi_key(entry) == frame_roi_key:
            _merge_flag_entry(entry, payload)
            break
    else:
        entries.append(payload)

    entries.sort(
        key=lambda item: (
            _coerce_int(item.get("frame_idx")) if _coerce_int(item.get("frame_idx")) is not None else -1,
            _coerce_int(item.get("roi_idx")) if _coerce_int(item.get("roi_idx")) is not None else -1,
            str(_entry_identity_key(item) or ""),
        )
    )
    data[zarr_path] = entries
    flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def candidate_flag_keys(zarr_path: str) -> list[str]:
    raw = str(zarr_path)
    expanded = Path(raw).expanduser()
    candidates = {raw, str(expanded)}
    try:
        candidates.add(str(expanded.resolve(strict=False)))
    except Exception:
        pass
    return list(candidates)


def entries_for_zarr_path(
    payload: Mapping[str, Sequence[Mapping[str, object]]],
    zarr_path: str,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for key in candidate_flag_keys(zarr_path):
        value = payload.get(key)
        if isinstance(value, list):
            entries.extend(dict(item) for item in value if isinstance(item, Mapping))
    return entries


def _optional_int_array(values: Optional[Sequence[object] | np.ndarray]) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1:
        return None
    return arr


def _identity_lookup(values: Optional[Sequence[object] | np.ndarray]) -> Optional[dict[int, list[int]]]:
    arr = _optional_int_array(values)
    if arr is None:
        return None
    lookup: dict[int, list[int]] = {}
    for idx, value in enumerate(arr.tolist()):
        int_value = int(value)
        if int_value < 0:
            continue
        lookup.setdefault(int_value, []).append(int(idx))
    return lookup


def resolve_flagged_roi_indices(
    entries: Sequence[Mapping[str, object]],
    *,
    total_rois: int,
    frame_indices: Optional[Sequence[object] | np.ndarray] = None,
    source_refined_row_ids: Optional[Sequence[object] | np.ndarray] = None,
    source_detect_row_index: Optional[Sequence[object] | np.ndarray] = None,
) -> np.ndarray:
    """Resolve flag entries to current ROI row indices.

    Stable IDs are authoritative when both the flag and current crop run expose
    them. Legacy frame/ROI fallback is used only when no usable identity lookup
    was available for the entry.
    """

    total = int(total_rois)
    if total <= 0:
        return np.zeros((0,), dtype=np.int32)

    frame_arr = _optional_int_array(frame_indices)
    refined_lookup = _identity_lookup(source_refined_row_ids)
    detect_lookup = _identity_lookup(source_detect_row_index)
    roi_set: set[int] = set()

    for entry in entries:
        attempted_identity = False
        refined_row_id = _coerce_int(entry.get("source_refined_row_id"))
        if refined_row_id is not None and refined_row_id >= 0 and refined_lookup is not None:
            attempted_identity = True
            matches = refined_lookup.get(int(refined_row_id), [])
            roi_set.update(idx for idx in matches if 0 <= idx < total)
            if matches:
                continue

        detect_row = _coerce_int(entry.get("source_detect_row_index"))
        if detect_row is not None and detect_row >= 0 and detect_lookup is not None:
            attempted_identity = True
            matches = detect_lookup.get(int(detect_row), [])
            roi_set.update(idx for idx in matches if 0 <= idx < total)
            if matches:
                continue

        if attempted_identity:
            continue

        roi_idx = _coerce_int(entry.get("roi_idx"))
        if roi_idx is not None and 0 <= roi_idx < total:
            roi_set.add(int(roi_idx))
            continue

        frame_idx = _coerce_int(entry.get("frame_idx"))
        if frame_idx is not None and frame_arr is not None:
            matches = np.where(frame_arr == int(frame_idx))[0]
            roi_set.update(int(idx) for idx in matches.tolist() if 0 <= int(idx) < total)

    if not roi_set:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray(sorted(roi_set), dtype=np.int32)


def row_identity_payload(
    row_idx: int,
    *,
    source_refined_row_ids: Optional[Sequence[object] | np.ndarray] = None,
    source_detect_row_index: Optional[Sequence[object] | np.ndarray] = None,
) -> dict[str, int]:
    payload: dict[str, int] = {}
    idx = int(row_idx)
    refined = _optional_int_array(source_refined_row_ids)
    if refined is not None and 0 <= idx < int(refined.shape[0]):
        value = int(refined[idx])
        if value >= 0:
            payload["source_refined_row_id"] = value
    detect = _optional_int_array(source_detect_row_index)
    if detect is not None and 0 <= idx < int(detect.shape[0]):
        value = int(detect[idx])
        if value >= 0:
            payload["source_detect_row_index"] = value
    return payload


def load_row_identity_arrays(
    group: object,
    *,
    total_rois: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    arrays: list[Optional[np.ndarray]] = []
    for name in ("source_refined_row_ids", "source_detect_row_index"):
        raw = None
        getter = getattr(group, "get", None)
        if callable(getter):
            raw = getter(name)
        if raw is None:
            arrays.append(None)
            continue
        try:
            arr = np.asarray(raw[:], dtype=np.int64)
        except Exception:
            arrays.append(None)
            continue
        if arr.ndim == 1 and int(arr.shape[0]) == int(total_rois):
            arrays.append(arr)
        else:
            arrays.append(None)
    return arrays[0], arrays[1]


def resolve_row_identity_arrays(
    primary_group: object,
    fallback_group: object | None = None,
    *,
    total_rois: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve stable ROI row lineage from a preferred group with fallback."""

    primary_refined, primary_detect = load_row_identity_arrays(primary_group, total_rois=total_rois)
    fallback_refined = None
    fallback_detect = None
    if fallback_group is not None:
        fallback_refined, fallback_detect = load_row_identity_arrays(fallback_group, total_rois=total_rois)

    refined = primary_refined if primary_refined is not None else fallback_refined
    detect = primary_detect if primary_detect is not None else fallback_detect
    if refined is None:
        refined = np.full((total_rois,), -1, dtype=np.int64)
    else:
        refined = refined.astype(np.int64, copy=False)
    if detect is None:
        detect = np.full((total_rois,), -1, dtype=np.int32)
    else:
        detect = detect.astype(np.int32, copy=False)
    return refined, detect
