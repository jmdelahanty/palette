"""Pure extraction of normalized protocol and stimulus-step registry rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.protocol_execution_contract import (
    PALETTE_PROTOCOL_EXECUTION_SCHEMA_ID,
    PALETTE_PROTOCOL_EXECUTION_SCHEMA_VERSION,
    ProtocolExecutionIndex,
    build_protocol_frame_correspondence_proxy_payload,
    validate_protocol_execution_index,
)
from fisheye.shared.protocol_semantic_contract import (
    PALETTE_PROTOCOL_RECIPE_SCHEMA_ID,
    PALETTE_PROTOCOL_RECIPE_SCHEMA_VERSION,
    PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_ID,
    PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_VERSION,
    ProtocolSemanticContractError,
    ProtocolSemanticSnapshot,
    validate_protocol_semantic_snapshot,
)
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


def _json_array(values: Any) -> str:
    """Serialize one validated bounded JSON value without string coercion."""

    return json.dumps(
        json_attr_safe(values),
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


def _array_utf8(group: Any, name: str) -> str:
    try:
        values = np.asarray(group[name][:])
    except (KeyError, TypeError) as exc:
        raise ProtocolSemanticContractError(
            f"Materialized protocol semantic snapshot lacks {name!r}."
        ) from exc
    if values.dtype != np.dtype(np.uint8) or values.ndim != 1:
        raise ProtocolSemanticContractError(
            f"Materialized protocol semantic snapshot {name!r} must be one-dimensional uint8."
        )
    try:
        return values.tobytes().decode("utf-8")
    except (TypeError, UnicodeDecodeError, ValueError) as exc:
        raise ProtocolSemanticContractError(
            f"Materialized protocol semantic snapshot {name!r} is not exact UTF-8 bytes."
        ) from exc


def _materialized_semantic_state(
    run_group: Any,
) -> tuple[str | None, ProtocolSemanticSnapshot | None]:
    """Reload and validate the exact semantic snapshot and every step binding."""

    status = _text(run_group.attrs.get("protocol_semantic_status"))
    semantic_attr_names = {
        "protocol_semantic_hash",
        "protocol_semantic_snapshot_path",
        "protocol_recipe_schema_id",
        "protocol_recipe_schema_version",
        "protocol_recipe_step_count",
        "protocol_recipe_mode_sequence",
        "protocol_recipe_label",
    }
    present_attrs = {
        name for name in semantic_attr_names if name in run_group.attrs
    }
    has_snapshot = "protocol_semantic_snapshot" in run_group
    if status is None:
        if present_attrs or has_snapshot:
            raise ProtocolSemanticContractError(
                "Stimulus run contains partial semantic metadata without an explicit status."
            )
        return None, None
    if status == "legacy_missing":
        if present_attrs or has_snapshot:
            raise ProtocolSemanticContractError(
                "legacy_missing stimulus run contains contradictory semantic identity."
            )
        steps_group = run_group.get("steps")
        if steps_group is not None and _text(
            steps_group.attrs.get("protocol_semantic_status")
        ) not in (None, "legacy_missing"):
            raise ProtocolSemanticContractError(
                "legacy_missing stimulus run has contradictory step semantic status."
            )
        return status, None
    if status != "verified":
        raise ProtocolSemanticContractError(
            f"Unknown protocol_semantic_status {status!r}."
        )

    missing_attrs = sorted(semantic_attr_names - present_attrs)
    if missing_attrs or not has_snapshot:
        missing = missing_attrs + ([] if has_snapshot else ["protocol_semantic_snapshot"])
        raise ProtocolSemanticContractError(
            "Verified stimulus run has partial semantic storage; missing "
            + ", ".join(missing)
            + "."
        )
    if run_group.attrs.get("protocol_semantic_snapshot_path") != (
        "protocol_semantic_snapshot"
    ):
        raise ProtocolSemanticContractError(
            "Verified stimulus run points at an unexpected semantic snapshot path."
        )

    snapshot_group = run_group["protocol_semantic_snapshot"]
    semantic_hash = _text(run_group.attrs.get("protocol_semantic_hash"))
    if semantic_hash is None:
        raise ProtocolSemanticContractError(
            "Verified stimulus run lacks a semantic hash."
        )
    semantic_text = _array_utf8(
        snapshot_group,
        "protocol_semantic_json_utf8",
    )
    trial_text = _array_utf8(
        snapshot_group,
        "protocol_trial_index_json_utf8",
    )
    source_snapshot_schema_version = snapshot_group.attrs.get(
        "source_snapshot_schema_version",
        1,
    )
    source_snapshot_policy_id = snapshot_group.attrs.get(
        "source_snapshot_policy_id",
        "citrus.protocol.snapshot.legacy_v1",
    )
    snapshot = validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_text,
        trial_index_json=trial_text,
        trial_index_hash=(
            snapshot_group.attrs.get("protocol_trial_index_hash")
            if source_snapshot_schema_version == 2
            else None
        ),
        snapshot_schema_version=source_snapshot_schema_version,
        snapshot_policy_id=source_snapshot_policy_id,
    )
    recipe = snapshot.recipe_record()
    if (
        snapshot_group.attrs.get("schema_id")
        != PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_ID
        or snapshot_group.attrs.get("schema_version")
        != PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_VERSION
        or snapshot_group.attrs.get("protocol_semantic_hash") != semantic_hash
        or snapshot_group.attrs.get("protocol_trial_index_sha256")
        != snapshot.trial_index_sha256
        or snapshot_group.attrs.get("protocol_trial_index_integrity_status")
        != snapshot.trial_index_integrity_status
        or snapshot_group.attrs.get("source_snapshot_schema_version")
        != snapshot.snapshot_schema_version
        or snapshot_group.attrs.get("source_snapshot_policy_id")
        != snapshot.snapshot_policy_id
        or snapshot_group.attrs.get("source_trial_index_schema_version")
        != snapshot.trial_index_schema_version
        or snapshot_group.attrs.get("recipe") != recipe
        or run_group.attrs.get("protocol_recipe_schema_id")
        != PALETTE_PROTOCOL_RECIPE_SCHEMA_ID
        or run_group.attrs.get("protocol_recipe_schema_version")
        != PALETTE_PROTOCOL_RECIPE_SCHEMA_VERSION
        or run_group.attrs.get("protocol_recipe_step_count") != snapshot.step_count
        or run_group.attrs.get("protocol_recipe_mode_sequence")
        != list(snapshot.mode_sequence)
        or run_group.attrs.get("protocol_recipe_label") != snapshot.recipe_label
    ):
        raise ProtocolSemanticContractError(
            "Materialized protocol semantic attrs differ from the exact snapshot."
        )
    if snapshot.snapshot_schema_version == 2:
        if snapshot_group.attrs.get("protocol_trial_index_hash") != (
            snapshot.trial_index_sha256
        ):
            raise ProtocolSemanticContractError(
                "Materialized producer trial-index hash differs from the snapshot."
            )
    elif snapshot_group.attrs.get("palette_computed_trial_index_sha256") != (
        snapshot.trial_index_sha256
    ):
        raise ProtocolSemanticContractError(
            "Materialized Palette trial-index digest differs from the snapshot."
        )

    steps_group = run_group.get("steps")
    if steps_group is None:
        raise ProtocolSemanticContractError(
            "Verified protocol semantic snapshot has no materialized steps."
        )
    expected_names = {f"step_{step.step_index}" for step in snapshot.steps}
    if set(_group_keys(steps_group)) != expected_names:
        raise ProtocolSemanticContractError(
            "Materialized stimulus steps differ from the exact semantic recipe."
        )
    if (
        steps_group.attrs.get("protocol_semantic_status") != "verified"
        or steps_group.attrs.get("protocol_semantic_hash") != semantic_hash
    ):
        raise ProtocolSemanticContractError(
            "Materialized stimulus steps lack verified semantic parent binding."
        )
    for identity in snapshot.steps:
        attrs = steps_group[f"step_{identity.step_index}"].attrs
        expected_color = (
            list(identity.resolved_color_rgba8)
            if identity.resolved_color_rgba8 is not None
            else None
        )
        duration_s = _number(attrs.get("duration_s"))
        if (
            attrs.get("protocol_semantic_status") != "verified"
            or attrs.get("protocol_semantic_hash") != semantic_hash
            or attrs.get("protocol_semantic_step_index") != identity.step_index
            or attrs.get("protocol_semantic_step_ref")
            != f"protocol_semantic_snapshot@recipe.steps[{identity.step_index}]"
            or attrs.get("step_index") != identity.step_index
            or attrs.get("stimulus_mode_id") != identity.stimulus_mode_id
            or attrs.get("stimulus_mode") != identity.stimulus_mode
            or duration_s is None
            or abs(duration_s - identity.duration_s) > 1e-9
            or attrs.get("stimulus_family") != identity.stimulus_family
            or attrs.get("protocol_trial_index_status") != identity.index_status
            or attrs.get("display_context") != identity.display_context
            or attrs.get("resolved_color_rgba8") != expected_color
        ):
            raise ProtocolSemanticContractError(
                "Materialized stimulus step semantic binding differs from the exact "
                f"snapshot at step_index={identity.step_index}."
            )
    return status, snapshot


def _materialized_execution_state(
    run_group: Any,
    snapshot: ProtocolSemanticSnapshot | None,
) -> tuple[ProtocolExecutionIndex | None, Mapping[str, Any] | None]:
    """Validate exact v2 execution bytes and the sealed correspondence proxy."""

    execution_group = run_group.get("protocol_execution")
    if snapshot is None or snapshot.snapshot_schema_version == 1:
        if execution_group is not None:
            raise ProtocolSemanticContractError(
                "A non-v2 stimulus run unexpectedly contains protocol execution."
            )
        return None, None
    if execution_group is None:
        raise ProtocolSemanticContractError(
            "Verified snapshot-v2 stimulus run lacks protocol execution evidence."
        )
    execution_text = _array_utf8(
        execution_group,
        "execution_index_json_utf8",
    )
    execution_hash = _text(execution_group.attrs.get("execution_index_hash"))
    if execution_hash is None:
        raise ProtocolSemanticContractError(
            "Materialized protocol execution lacks its exact-byte hash."
        )
    execution = validate_protocol_execution_index(
        execution_json=execution_text,
        execution_hash=execution_hash,
        snapshot=snapshot,
    )
    if (
        execution_group.attrs.get("schema_id")
        != PALETTE_PROTOCOL_EXECUTION_SCHEMA_ID
        or execution_group.attrs.get("schema_version")
        != PALETTE_PROTOCOL_EXECUTION_SCHEMA_VERSION
        or execution_group.attrs.get("source")
        != "citrus_h5_protocol_execution"
        or execution_group.attrs.get("source_policy_id")
        != "citrus.protocol.execution_index.half_open_stimulus_frames.v1"
        or execution_group.attrs.get("status") != execution.status
        or execution_group.attrs.get("protocol_trial_index_hash")
        != snapshot.trial_index_sha256
        or execution_group.attrs.get("authoritative_interval_axis")
        != "stimulus_frame_num"
        or execution_group.attrs.get("camera_frame_role")
        != "correspondence_only"
        or execution_group.attrs.get("acquisition_containment_status")
        != "unavailable_without_sealed_stimulus_to_acquisition_mapping"
        or run_group.attrs.get("protocol_execution_status") != execution.status
        or run_group.attrs.get("protocol_execution_hash")
        != execution.execution_hash
        or run_group.attrs.get("protocol_interval_axis")
        != "stimulus_frame_num"
        or run_group.attrs.get("protocol_camera_frame_role")
        != "correspondence_only"
        or run_group.attrs.get("protocol_acquisition_containment_status")
        != "unavailable_without_sealed_stimulus_to_acquisition_mapping"
        or run_group.attrs.get("protocol_selector_eligibility")
        != "blocked_missing_sealed_stimulus_to_acquisition_mapping"
    ):
        raise ProtocolSemanticContractError(
            "Materialized protocol execution attrs differ from exact v2 evidence."
        )

    steps_group = run_group.get("steps")
    if steps_group is None or (
        steps_group.attrs.get("source") != "citrus_protocol_execution_index"
        or steps_group.attrs.get("authoritative_interval_axis")
        != "stimulus_frame_num"
        or steps_group.attrs.get("camera_frame_role") != "correspondence_only"
        or steps_group.attrs.get("protocol_execution_hash")
        != execution.execution_hash
        or steps_group.attrs.get("protocol_execution_status") != execution.status
    ):
        raise ProtocolSemanticContractError(
            "Materialized stimulus steps lack exact v2 execution binding."
        )
    for realized in execution.steps:
        step_group = steps_group.get(f"step_{realized.step_index}")
        if step_group is None:
            raise ProtocolSemanticContractError(
                "Materialized v2 execution is missing a realized step."
            )
        interval = realized.interval
        attrs = step_group.attrs
        if (
            attrs.get("authoritative_interval_axis") != "stimulus_frame_num"
            or attrs.get("start_stimulus_frame_inclusive")
            != interval.start_stimulus_frame_inclusive
            or attrs.get("end_stimulus_frame_exclusive")
            != interval.end_stimulus_frame_exclusive
            or attrs.get("first_camera_frame_id_correspondence")
            != interval.first_camera_frame_id
            or attrs.get("last_camera_frame_id_correspondence")
            != interval.last_camera_frame_id
            or attrs.get("camera_frame_role") != "correspondence_only"
            or attrs.get("execution_completion_status")
            != realized.completion_status
            or attrs.get("execution_end_reason") != realized.end_reason
        ):
            raise ProtocolSemanticContractError(
                "Materialized v2 step differs from exact execution evidence at "
                f"step_index={realized.step_index}."
            )
        phases_group = step_group.get("execution_phases")
        if realized.chaser_phases is None:
            if phases_group is not None:
                raise ProtocolSemanticContractError(
                    "Non-chaser execution step unexpectedly contains phases."
                )
            continue
        if phases_group is None or set(_group_keys(phases_group)) != set(
            realized.chaser_phases
        ):
            raise ProtocolSemanticContractError(
                "Materialized chaser phases differ from exact execution evidence."
            )
        for phase_name, phase in realized.chaser_phases.items():
            phase_attrs = phases_group[phase_name].attrs
            if any(
                phase_attrs.get(name) != expected
                for name, expected in {
                    **phase.to_record(),
                    "authoritative_interval_axis": "stimulus_frame_num",
                    "camera_frame_role": "correspondence_only",
                    "acquisition_containment_status": (
                        "unavailable_without_sealed_stimulus_to_acquisition_mapping"
                    ),
                }.items()
            ):
                raise ProtocolSemanticContractError(
                    "Materialized chaser phase differs from exact execution "
                    f"evidence at {phase_name!r}."
                )

    proxy = execution_group.get("frame_correspondence_proxy")
    if proxy is None:
        raise ProtocolSemanticContractError(
            "Materialized protocol execution lacks its correspondence proxy."
        )
    required_arrays = {
        "stimulus_frame_num",
        "camera_frame_id_correspondence",
        "protocol_step_index",
        "chaser_phase_id",
        "in_realized_protocol",
    }
    observed_names = set(str(name) for name in proxy.array_keys())
    if observed_names != required_arrays:
        raise ProtocolSemanticContractError(
            "Protocol correspondence proxy exposes an unexpected array set."
        )
    stimulus = np.asarray(proxy["stimulus_frame_num"][:])
    camera = np.asarray(proxy["camera_frame_id_correspondence"][:])
    expected_arrays, expected_manifest = (
        build_protocol_frame_correspondence_proxy_payload(
            execution,
            raw_stimulus=stimulus,
            raw_camera=camera,
        )
    )
    for name, expected in expected_arrays.items():
        observed = np.asarray(proxy[name][:])
        if (
            observed.dtype != expected.dtype
            or observed.shape != expected.shape
            or not np.array_equal(observed, expected)
        ):
            raise ProtocolSemanticContractError(
                f"Protocol correspondence proxy array {name!r} is not sealed "
                "to the exact execution intervals."
            )
    for name, expected in expected_manifest.items():
        if strict_json_dumps(proxy.attrs.get(name)) != strict_json_dumps(expected):
            raise ProtocolSemanticContractError(
                f"Protocol correspondence proxy manifest field {name!r} differs."
            )
    manifest_sha256 = sha256(
        strict_json_dumps(expected_manifest).encode("utf-8")
    ).hexdigest()
    coverage_status = (
        "complete"
        if expected_manifest["missing_realized_stimulus_frame_count"] == 0
        else "incomplete_proxy"
    )
    if (
        proxy.attrs.get("manifest_sha256") != manifest_sha256
        or proxy.attrs.get("coverage_status") != coverage_status
    ):
        raise ProtocolSemanticContractError(
            "Protocol correspondence proxy seal or coverage status differs."
        )
    return execution, {
        "coverage_status": coverage_status,
        "manifest_sha256": manifest_sha256,
        "missing_realized_stimulus_frame_count": expected_manifest[
            "missing_realized_stimulus_frame_count"
        ],
    }


def _materialized_steps(
    run_group: Any,
    protocol_steps: list[Mapping[str, Any]],
    *,
    semantic_status: str | None,
    semantic_snapshot: ProtocolSemanticSnapshot | None,
) -> list[dict[str, Any]]:
    steps_group = run_group.get("steps")
    rows: list[dict[str, Any]] = []
    for ordinal, name in enumerate(_group_keys(steps_group)):
        step_group = steps_group[name]
        attrs = {str(key): normalize_attr(value) for key, value in step_group.attrs.items()}
        phases_group = step_group.get("execution_phases")
        phase_records = (
            {
                phase_name: {
                    str(key): normalize_attr(value)
                    for key, value in phases_group[phase_name].attrs.items()
                }
                for phase_name in _group_keys(phases_group)
            }
            if phases_group is not None
            else None
        )
        index = _integer(attrs.get("step_index"))
        if index is None:
            index = _integer(str(name).replace("step_", "", 1))
        if index is None:
            index = ordinal
        definition = protocol_steps[index] if 0 <= index < len(protocol_steps) else None
        identity = (
            semantic_snapshot.steps[index]
            if semantic_snapshot is not None
            and 0 <= index < semantic_snapshot.step_count
            else None
        )
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
                "start_stimulus_frame_inclusive": _integer(
                    attrs.get("start_stimulus_frame_inclusive")
                ),
                "end_stimulus_frame_exclusive": _integer(
                    attrs.get("end_stimulus_frame_exclusive")
                ),
                "first_camera_frame_id_correspondence": _integer(
                    attrs.get("first_camera_frame_id_correspondence")
                ),
                "last_camera_frame_id_correspondence": _integer(
                    attrs.get("last_camera_frame_id_correspondence")
                ),
                "authoritative_interval_axis": _text(
                    attrs.get("authoritative_interval_axis")
                ),
                "execution_completion_status": _text(
                    attrs.get("execution_completion_status")
                ),
                "execution_end_reason": _text(attrs.get("execution_end_reason")),
                "protocol_execution_phases_json": (
                    _json_array(phase_records)
                    if phase_records is not None
                    else None
                ),
                "duration_s": _number(attrs.get("duration_s")),
                "protocol_semantic_status": semantic_status,
                "protocol_semantic_hash": (
                    semantic_snapshot.semantic_hash
                    if semantic_snapshot is not None
                    else None
                ),
                "protocol_semantic_step_index": (
                    identity.step_index if identity is not None else None
                ),
                "protocol_semantic_step_ref": (
                    f"protocol_semantic_snapshot@recipe.steps[{identity.step_index}]"
                    if identity is not None
                    else None
                ),
                "protocol_semantic_stimulus_mode_id": (
                    identity.stimulus_mode_id if identity is not None else None
                ),
                "protocol_semantic_duration_s": (
                    identity.duration_s if identity is not None else None
                ),
                "stimulus_family": (
                    identity.stimulus_family if identity is not None else None
                ),
                "display_context": (
                    identity.display_context if identity is not None else None
                ),
                "protocol_trial_index_status": (
                    identity.index_status if identity is not None else None
                ),
                "resolved_color_rgba8_json": (
                    _json_array(list(identity.resolved_color_rgba8))
                    if identity is not None
                    and identity.resolved_color_rgba8 is not None
                    else None
                ),
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
        semantic_status, semantic_snapshot = _materialized_semantic_state(run_group)
        execution_index, correspondence_proxy_record = (
            _materialized_execution_state(run_group, semantic_snapshot)
        )
        payload = _mapping(run_group.attrs.get("protocol_json"))
        raw_protocol_steps = payload.get("steps", []) if payload is not None else []
        protocol_steps = [
            step for step in raw_protocol_steps if isinstance(step, Mapping)
        ] if isinstance(raw_protocol_steps, list) else []
        materialized = _materialized_steps(
            run_group,
            protocol_steps,
            semantic_status=semantic_status,
            semantic_snapshot=semantic_snapshot,
        )
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
        semantic_fields = {
            "protocol_semantic_status": semantic_status,
            "protocol_semantic_hash": (
                semantic_snapshot.semantic_hash
                if semantic_snapshot is not None
                else None
            ),
            "palette_computed_trial_index_sha256": (
                semantic_snapshot.trial_index_sha256
                if semantic_snapshot is not None
                and semantic_snapshot.snapshot_schema_version == 1
                else None
            ),
            "protocol_trial_index_sha256": (
                semantic_snapshot.trial_index_sha256
                if semantic_snapshot is not None
                else None
            ),
            "producer_protocol_trial_index_hash": (
                semantic_snapshot.trial_index_sha256
                if semantic_snapshot is not None
                and semantic_snapshot.snapshot_schema_version == 2
                else None
            ),
            "protocol_trial_index_integrity_status": (
                semantic_snapshot.trial_index_integrity_status
                if semantic_snapshot is not None
                else None
            ),
            "protocol_snapshot_schema_version": (
                semantic_snapshot.snapshot_schema_version
                if semantic_snapshot is not None
                else None
            ),
            "protocol_snapshot_policy_id": (
                semantic_snapshot.snapshot_policy_id
                if semantic_snapshot is not None
                else None
            ),
            "protocol_trial_index_schema_version": (
                semantic_snapshot.trial_index_schema_version
                if semantic_snapshot is not None
                else None
            ),
            "protocol_execution_status": _text(
                execution_index.status if execution_index is not None else None
            ),
            "protocol_execution_hash": _text(
                execution_index.execution_hash
                if execution_index is not None
                else None
            ),
            "protocol_interval_axis": _text(
                run_group.attrs.get("protocol_interval_axis")
            ),
            "protocol_acquisition_containment_status": _text(
                run_group.attrs.get("protocol_acquisition_containment_status")
            ),
            "protocol_frame_correspondence_proxy_status": (
                _text(correspondence_proxy_record.get("coverage_status"))
                if correspondence_proxy_record is not None
                else None
            ),
            "protocol_frame_correspondence_proxy_manifest_sha256": (
                _text(correspondence_proxy_record.get("manifest_sha256"))
                if correspondence_proxy_record is not None
                else None
            ),
            "protocol_frame_correspondence_proxy_missing_count": (
                _integer(
                    correspondence_proxy_record.get(
                        "missing_realized_stimulus_frame_count"
                    )
                )
                if correspondence_proxy_record is not None
                else None
            ),
            "protocol_recipe_schema_id": (
                PALETTE_PROTOCOL_RECIPE_SCHEMA_ID
                if semantic_snapshot is not None
                else None
            ),
            "protocol_recipe_schema_version": (
                PALETTE_PROTOCOL_RECIPE_SCHEMA_VERSION
                if semantic_snapshot is not None
                else None
            ),
            "protocol_recipe_step_count": (
                semantic_snapshot.step_count
                if semantic_snapshot is not None
                else None
            ),
            "protocol_recipe_mode_sequence_json": (
                _json_array(list(semantic_snapshot.mode_sequence))
                if semantic_snapshot is not None
                else None
            ),
            "protocol_recipe_label": (
                semantic_snapshot.recipe_label
                if semantic_snapshot is not None
                else None
            ),
        }
        source_hash = sha256(
            _canonical_json(
                {
                    "protocol": definition,
                    "steps": materialized,
                    "run_attrs": dict(run_group.attrs),
                    "semantic": semantic_fields,
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
                **semantic_fields,
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
