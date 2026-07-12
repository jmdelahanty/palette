"""Pure extraction of normalized protocol and stimulus-step registry rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.type_conversions import normalize_attr


@dataclass(frozen=True)
class StimulusMetadataExtraction:
    protocols: tuple[dict[str, Any], ...]
    protocol_steps: tuple[dict[str, Any], ...]
    recording_runs: tuple[dict[str, Any], ...]
    recording_steps: tuple[dict[str, Any], ...]
    recording_modes: tuple[dict[str, Any], ...]


def _group_keys(group: Any | None) -> list[str]:
    if group is None:
        return []
    try:
        return sorted(str(name) for name in group.group_keys())
    except Exception:
        try:
            return sorted(str(name) for name in group.keys())
        except Exception:
            return []


def _mapping(value: Any) -> Mapping[str, Any] | None:
    value = normalize_attr(value)
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        json_attr_safe(normalize_attr(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _text(value: Any) -> str | None:
    value = normalize_attr(value)
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _integer(value: Any) -> int | None:
    value = normalize_attr(value)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    value = normalize_attr(value)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if mapping.get(name) is not None:
            return mapping[name]
    return None


def _mode(attrs: Mapping[str, Any], definition: Mapping[str, Any] | None) -> str:
    parameters = definition.get("parameters") if definition else None
    if not isinstance(parameters, Mapping):
        parameters = {}
    value = _first(attrs, "stimulus_mode", "stimulus_mode_str")
    if value is None and definition is not None:
        value = _first(definition, "stimulus_mode", "stimulus_mode_str", "mode")
    if value is None:
        value = _first(parameters, "stimulus_mode", "stimulus_mode_str", "mode")
    return (_text(value) or "UNKNOWN").upper()


def _materialized_steps(
    run_group: Any,
    protocol_steps: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    steps_group = run_group.get("steps")
    rows: list[dict[str, Any]] = []
    for ordinal, name in enumerate(_group_keys(steps_group)):
        step_group = steps_group[name]
        attrs = {str(key): normalize_attr(value) for key, value in step_group.attrs.items()}
        index = _integer(attrs.get("step_index"))
        if index is None:
            index = _integer(str(name).replace("step_", "", 1))
        if index is None:
            index = ordinal
        definition = protocol_steps[index] if 0 <= index < len(protocol_steps) else None
        rows.append(
            {
                "step_index": index,
                "step_name": _text(attrs.get("step_name")) or str(name),
                "stimulus_mode": _mode(attrs, definition),
                "start_camera_frame": _integer(
                    _first(attrs, "start_camera_frame", "start_frame")
                ),
                "end_camera_frame": _integer(
                    _first(attrs, "end_camera_frame", "end_frame")
                ),
                "duration_s": _number(attrs.get("duration_s")),
                "step_attrs_json": _canonical_json(attrs),
            }
        )
    return sorted(rows, key=lambda row: int(row["step_index"]))


def extract_stimulus_metadata(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: str | None,
) -> StimulusMetadataExtraction:
    """Extract protocols, steps, and recording-level mode counts from all runs."""

    analysis = root.get("analysis")
    parent = analysis.get("stimulus_runs") if analysis is not None else None
    if parent is None:
        return StimulusMetadataExtraction((), (), (), (), ())

    extracted_utc = utc_now()
    source_zarr_path = str(Path(zarr_path).expanduser().resolve(strict=False))
    latest = _text(parent.attrs.get("latest"))
    protocols: dict[str, dict[str, Any]] = {}
    protocol_step_rows: dict[tuple[str, int], dict[str, Any]] = {}
    recording_runs: list[dict[str, Any]] = []
    recording_steps: list[dict[str, Any]] = []
    recording_modes: list[dict[str, Any]] = []

    for run_name in _group_keys(parent):
        run_group = parent[run_name]
        payload = _mapping(run_group.attrs.get("protocol_json"))
        raw_protocol_steps = payload.get("steps", []) if payload is not None else []
        protocol_steps = [
            step for step in raw_protocol_steps if isinstance(step, Mapping)
        ] if isinstance(raw_protocol_steps, list) else []
        materialized = _materialized_steps(run_group, protocol_steps)
        definition = payload if payload is not None else {
            "protocol_name": _text(run_group.attrs.get("protocol_name")),
            "steps": materialized,
        }
        definition_json = _canonical_json(definition)
        protocol_hash = sha256(definition_json.encode("utf-8")).hexdigest()
        protocol_name = (
            _text(payload.get("protocol_name")) if payload is not None else None
        ) or _text(run_group.attrs.get("protocol_name"))
        protocols[protocol_hash] = {
            "protocol_hash": protocol_hash,
            "protocol_name": protocol_name,
            "step_count": len(protocol_steps) if protocol_steps else len(materialized),
            "protocol_json": definition_json,
            "definition_source": (
                "stimulus_protocol_json" if payload is not None else "materialized_steps"
            ),
            "extracted_utc": extracted_utc,
        }
        for index, step in enumerate(protocol_steps):
            parameters = step.get("parameters")
            protocol_step_rows[(protocol_hash, index)] = {
                "protocol_hash": protocol_hash,
                "step_index": index,
                "step_name": _text(step.get("step_name") or step.get("name")),
                "stimulus_mode": _mode({}, step),
                "duration_s": _number(step.get("duration_s")),
                "parameters_json": (
                    _canonical_json(parameters) if parameters is not None else None
                ),
                "step_definition_json": _canonical_json(step),
            }

        source_path = f"analysis/stimulus_runs/{run_name}"
        source_hash = sha256(
            _canonical_json(
                {
                    "protocol": definition,
                    "steps": materialized,
                    "run_attrs": dict(run_group.attrs),
                }
            ).encode("utf-8")
        ).hexdigest()
        recording_runs.append(
            {
                "recording_id": str(recording_id) if recording_id else None,
                "stimulus_run_id": run_name,
                "protocol_hash": protocol_hash,
                "protocol_name": protocol_name,
                "is_latest": int(run_name == latest),
                "step_count": len(materialized),
                "source_path": source_path,
                "source_metadata_sha256": source_hash,
                "source_zarr_path": source_zarr_path,
                "extracted_utc": extracted_utc,
            }
        )
        counts: Counter[str] = Counter()
        durations: defaultdict[str, float] = defaultdict(float)
        has_duration: set[str] = set()
        for row in materialized:
            mode = str(row["stimulus_mode"])
            counts[mode] += 1
            if row["duration_s"] is not None:
                durations[mode] += float(row["duration_s"])
                has_duration.add(mode)
            recording_steps.append(
                {
                    "stimulus_run_id": run_name,
                    **row,
                }
            )
        for mode in sorted(counts):
            recording_modes.append(
                {
                    "stimulus_run_id": run_name,
                    "stimulus_mode": mode,
                    "step_count": counts[mode],
                    "total_duration_s": durations[mode] if mode in has_duration else None,
                }
            )

    return StimulusMetadataExtraction(
        protocols=tuple(protocols[key] for key in sorted(protocols)),
        protocol_steps=tuple(protocol_step_rows[key] for key in sorted(protocol_step_rows)),
        recording_runs=tuple(recording_runs),
        recording_steps=tuple(recording_steps),
        recording_modes=tuple(recording_modes),
    )


__all__ = ["StimulusMetadataExtraction", "extract_stimulus_metadata"]
