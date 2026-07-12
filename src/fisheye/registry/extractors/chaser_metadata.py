"""Pure extraction of configured, variable-cardinality chaser metadata."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import zarr

from fisheye.analysis.chaser_behavior import resolve_configured_chaser_behaviors
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr


@dataclass(frozen=True)
class ChaserMetadataIssue:
    stimulus_run_id: str
    source_path: str
    reason: str
    detail: str


@dataclass(frozen=True)
class ChaserMetadataExtraction:
    rows: tuple[dict[str, Any], ...]
    stimulus_run_count: int
    issues: tuple[ChaserMetadataIssue, ...]


def _group_keys(group: zarr.Group) -> list[str]:
    try:
        return sorted(str(name) for name in group.group_keys())
    except Exception:
        return sorted(str(name) for name in group.keys())


def _protocol_payload(value: Any) -> Mapping[str, Any] | None:
    decoded = normalize_attr(value)
    if isinstance(decoded, Mapping):
        return decoded
    if not isinstance(decoded, str) or not decoded.strip():
        return None
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def extract_recording_chaser_metadata(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: str | None,
) -> ChaserMetadataExtraction:
    """Extract one scalar-valued row per configured chaser in every stimulus run."""

    analysis = root.get("analysis")
    parent = analysis.get("stimulus_runs") if analysis is not None else None
    if parent is None:
        return ChaserMetadataExtraction(rows=(), stimulus_run_count=0, issues=())

    extracted_utc = utc_now()
    rows: list[dict[str, Any]] = []
    issues: list[ChaserMetadataIssue] = []
    run_names = _group_keys(parent)
    for run_name in run_names:
        source_path = f"analysis/stimulus_runs/{run_name}"
        run_group = parent[run_name]
        payload = _protocol_payload(run_group.attrs.get("protocol_json"))
        if payload is None:
            issues.append(
                ChaserMetadataIssue(
                    stimulus_run_id=run_name,
                    source_path=source_path,
                    reason="protocol_json_missing_or_invalid",
                    detail="Configured chaser behavior could not be resolved.",
                )
            )
            continue
        try:
            behaviors = resolve_configured_chaser_behaviors(payload)
        except ValueError as exc:
            issues.append(
                ChaserMetadataIssue(
                    stimulus_run_id=run_name,
                    source_path=source_path,
                    reason="configured_chasers_unresolved",
                    detail=str(exc),
                )
            )
            continue

        indices = [behavior.chaser_index for behavior in behaviors]
        if len(indices) != len(set(indices)):
            issues.append(
                ChaserMetadataIssue(
                    stimulus_run_id=run_name,
                    source_path=source_path,
                    reason="duplicate_chaser_index",
                    detail=f"Configured chaser indices are not unique: {indices}",
                )
            )
            continue

        protocol_json = _canonical_json(payload)
        source_hash = sha256(protocol_json.encode("utf-8")).hexdigest()
        protocol_name = normalize_attr(payload.get("protocol_name"))
        for behavior in behaviors:
            rows.append(
                {
                    "recording_id": str(recording_id) if recording_id else None,
                    "stimulus_run_id": run_name,
                    "chaser_index": int(behavior.chaser_index),
                    "behavior_class_id": int(behavior.behavior_class_id),
                    "behavior_class": behavior.behavior_class,
                    "enable_chase": int(behavior.enable_chase),
                    "enable_random_movement": int(behavior.enable_random_movement),
                    "behavior_mode": behavior.behavior_mode,
                    "raw_color_rgba_json": _canonical_json(list(behavior.raw_color_rgba)),
                    "start_position_preset": behavior.start_position_preset or None,
                    "end_position_preset": behavior.end_position_preset or None,
                    "protocol_name": str(protocol_name) if protocol_name else None,
                    "source_path": source_path,
                    "source_kind": "stimulus_protocol_json",
                    "source_metadata_sha256": source_hash,
                    "source_zarr_path": str(Path(zarr_path).expanduser().resolve(strict=False)),
                    "extracted_utc": extracted_utc,
                }
            )

    return ChaserMetadataExtraction(
        rows=tuple(rows),
        stimulus_run_count=len(run_names),
        issues=tuple(issues),
    )


__all__ = [
    "ChaserMetadataExtraction",
    "ChaserMetadataIssue",
    "extract_recording_chaser_metadata",
]
