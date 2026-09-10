"""Materialize one explicit selector-ineligible provider behavior chain.

The task document freezes every source and destination run identity.  Versions
3 and 4 consume a separately published, manifest-pinned subject-position
provider; versions 1 and 2 retain their keypoint-triad position adapter.
Version 4 additionally seals the task with a canonical digest and emits a
self-verifiable receipt.  This utility never resolves ``latest`` and never
changes a production selector.  It exists to make commit-pinned cohort canaries
retryable without combining scientific selection with publication.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.stimulus_epoch_runs import (
    build_stimulus_epoch_result,
    write_stimulus_epoch_run,
)
from fisheye.analysis.stimulus_epoch_schema import (
    stimulus_epoch_logical_content_sha256,
    validate_legacy_stimulus_epoch_source,
)
from fisheye.analysis.swim_bout_frame_axis import canonical_frame_axis_sha256
from fisheye.analysis.swim_bout_io import (
    SwimBoutTables,
    load_exact_selector_ineligible_default_swim_bout_tables,
)
from fisheye.analysis_workflows.body_frame_source_handle import (
    load_body_frame_source_handle,
)
from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
    materialize_provider_epoch_behavior_summary,
)
from fisheye.analysis_workflows.materializers.provider_track_motion import (
    plan_provider_track_motion_run,
    prepare_provider_track_motion,
    publish_provider_track_motion_run,
)
from fisheye.analysis_workflows.materializers.single_subject_tracking import (
    plan_single_subject_tracking_run,
    publish_single_subject_tracking_run,
)
from fisheye.analysis_workflows.materializers.stimulus_epochs import (
    materialize_stimulus_epoch_candidate,
)
from fisheye.analysis_workflows.materializers.subject_position import (
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.materializers.swim_bouts import materialize_swim_bouts
from fisheye.analysis_workflows.position_body_frame_motion import (
    bind_position_body_frame_to_tracking,
    compose_position_body_frame_motion_authority,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    load_protocol_semantic_chaser_selection_source_handle,
)
from fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle import (
    load_provider_epoch_behavior_summary_source_handle,
)
from fisheye.analysis_workflows.provider_recording_timing_authority import (
    load_provider_recording_timing_authority,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    load_provider_track_motion_source_handle,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelection,
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    load_subject_position_source_handle,
    require_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    load_tracking_source_handle,
)
from fisheye.shared.anatomy_profile import load_anatomy_profile
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.run_lineage_fingerprint import (
    canonical_lineage_json,
    compute_run_lineage_hash,
)
from fisheye.shared.subject_position_expression import (
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
)
from fisheye.shared.subject_position_keypoint_source import (
    KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY,
    KeypointPositionSourcePolicy,
    load_bound_keypoint_position_source,
)
from fisheye.shared.subject_position_preparation import prepare_subject_position_input
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

TASK_SCHEMA_ID = "palette.provider_behavior_chain_task"
LEGACY_TASK_SCHEMA_VERSION = 1
SEMANTIC_TASK_SCHEMA_VERSION = 2
EXPLICIT_POSITION_TASK_SCHEMA_VERSION = 3
RECEIPTED_TASK_SCHEMA_VERSION = 4
# Preserve the original constant for legacy task producers.
TASK_SCHEMA_VERSION = LEGACY_TASK_SCHEMA_VERSION
SUPPORTED_TASK_SCHEMA_VERSIONS = (
    LEGACY_TASK_SCHEMA_VERSION,
    SEMANTIC_TASK_SCHEMA_VERSION,
    EXPLICIT_POSITION_TASK_SCHEMA_VERSION,
    RECEIPTED_TASK_SCHEMA_VERSION,
)
RESULT_SCHEMA_ID = "palette.provider_behavior_chain_result"
RESULT_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_ID = "palette.provider_behavior_chain_receipt"
RECEIPT_SCHEMA_VERSION = 1
LINEAR_ONLY_DISPOSITION = "linear_motion_and_swim_bouts_only_heading_excluded_v1"
KEYPOINT_BINDING_ID = "zebrafish_larva_keypoint_traditional_v2_v1"
SUMMARY_SPEED_LEVEL = "filtered"
RECEIPT_STAGE_NAMES = (
    "position",
    "tracking",
    "motion",
    "swim_bouts",
    "stimulus_epochs_v1",
    "stimulus_epochs_v2",
    "epoch_summary",
)
RECEIPT_STAGE_DIGEST_FIELDS = {
    "position": ("manifest_sha256", "decoded_content_sha256"),
    "tracking": ("manifest_sha256", "verification_digest"),
    "motion": ("manifest_sha256", "verification_digest"),
    "swim_bouts": (
        "lineage_hash",
        "content_sha256",
        "source_motion_manifest_sha256",
    ),
    "stimulus_epochs_v1": (
        "logical_content_sha256",
        "lineage_hash",
        "lineage_payload_sha256",
    ),
    "stimulus_epochs_v2": ("manifest_sha256", "selection_sha256"),
    "epoch_summary": ("manifest_sha256",),
}

_EXPLICIT_POSITION_TASK_SCHEMA_VERSIONS = frozenset(
    {EXPLICIT_POSITION_TASK_SCHEMA_VERSION, RECEIPTED_TASK_SCHEMA_VERSION}
)
_RECEIPTED_TASK_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "analysis_zarr",
        "source_runs",
        "output_runs",
        "protocol_semantic_selection_run",
        "fps",
        "arena_id",
        "metric_disposition",
        "task_sha256",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "status",
        "completed_at_utc",
        "task_schema_id",
        "task_schema_version",
        "task_sha256",
        "recording_id",
        "analysis_zarr",
        "metric_disposition",
        "selector_updates",
        "source_payloads_rewritten",
        "stages",
        "stage_sha256s",
        "receipt_sha256",
    }
)


class ProviderBehaviorChainError(ValueError):
    """Raised when an explicit cohort task is incomplete or inconsistent."""


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderBehaviorChainError(f"{label} must be one JSON object.")
    return dict(value)


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderBehaviorChainError(f"{label} must be one nonempty exact string.")
    return value


def _run_name(value: Any, *, label: str) -> str:
    name = _text(value, label=label)
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ProviderBehaviorChainError(f"{label} must be one bare run name.")
    return name


def _sha256(value: Any, *, label: str) -> str:
    digest = _text(value, label=label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ProviderBehaviorChainError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return digest


def _subject_position_reference(value: Any) -> dict[str, str]:
    reference = _object(value, label="source_runs.position")
    if set(reference) != {"run_path", "manifest_sha256"}:
        raise ProviderBehaviorChainError(
            "source_runs.position must contain exactly run_path and manifest_sha256."
        )
    run_path = _text(
        reference.get("run_path"),
        label="source_runs.position.run_path",
    )
    prefix = "analysis/subject_position_runs/observation/"
    if not run_path.startswith(prefix):
        raise ProviderBehaviorChainError(
            "source_runs.position.run_path must name one observation position run."
        )
    run_name = _run_name(
        run_path[len(prefix) :],
        label="source_runs.position.run_path",
    )
    if run_name in {"latest", "latest_complete", "latest_pending", "authoritative_run"}:
        raise ProviderBehaviorChainError(
            "source_runs.position.run_path must name one concrete non-selector run."
        )
    canonical_path = f"{prefix}{run_name}"
    if run_path != canonical_path:
        raise ProviderBehaviorChainError(
            "source_runs.position.run_path is not canonical."
        )
    return {
        "run_path": canonical_path,
        "manifest_sha256": _sha256(
            reference.get("manifest_sha256"),
            label="source_runs.position.manifest_sha256",
        ),
    }


def _task_digest(task: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in task.items() if key != "task_sha256"}
    )


def _normalize_task(
    value: Mapping[str, Any],
    *,
    verify_receipted_digest: bool,
) -> dict[str, Any]:
    payload = json_attr_safe(_object(value, label="task"))
    schema_version = payload.get("schema_version")
    if (
        payload.get("schema_id") != TASK_SCHEMA_ID
        or type(schema_version) is not int
        or schema_version not in SUPPORTED_TASK_SCHEMA_VERSIONS
    ):
        raise ProviderBehaviorChainError("Task schema identity is unsupported.")
    receipted = schema_version == RECEIPTED_TASK_SCHEMA_VERSION
    persisted_task_sha256: str | None = None
    if receipted:
        if set(payload) != _RECEIPTED_TASK_FIELDS:
            raise ProviderBehaviorChainError(
                "Provider behavior chain task v4 has an unexpected field set."
            )
        persisted_task_sha256 = _sha256(
            payload.get("task_sha256"),
            label="task_sha256",
        )
        if verify_receipted_digest and persisted_task_sha256 != _task_digest(payload):
            raise ProviderBehaviorChainError(
                "Provider behavior chain task_sha256 is stale or the task was tampered."
            )
    payload["recording_id"] = _text(payload.get("recording_id"), label="recording_id")
    archive_text = _text(payload.get("analysis_zarr"), label="analysis_zarr")
    archive = Path(archive_text).expanduser().resolve()
    if receipted and archive_text != str(archive):
        raise ProviderBehaviorChainError(
            "Provider behavior chain task v4 analysis_zarr must be one canonical "
            "absolute path."
        )
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    if archive.name != f"{payload['recording_id']}_analysis.zarr":
        raise ProviderBehaviorChainError(
            "Analysis Zarr basename differs from the frozen recording identity."
        )
    payload["analysis_zarr"] = str(archive)
    sources = _object(payload.get("source_runs"), label="source_runs")
    outputs = _object(payload.get("output_runs"), label="output_runs")
    position_is_explicit = schema_version in _EXPLICIT_POSITION_TASK_SCHEMA_VERSIONS
    if receipted and set(sources) != {"position", "body_frame", "stimulus"}:
        raise ProviderBehaviorChainError(
            "Provider behavior chain task v4 source_runs has an unexpected field set."
        )
    if receipted and set(outputs) != {
        "tracking",
        "motion",
        "swim_bouts",
        "stimulus_epochs_v1",
        "stimulus_epochs_v2",
        "epoch_summary",
    }:
        raise ProviderBehaviorChainError(
            "Provider behavior chain task v4 output_runs has an unexpected field set."
        )
    if position_is_explicit and "keypoint" in sources:
        raise ProviderBehaviorChainError(
            f"Provider behavior chain task v{schema_version} must not carry "
            "source_runs.keypoint; "
            "position is one independently published input."
        )
    if position_is_explicit and "position" in outputs:
        raise ProviderBehaviorChainError(
            f"Provider behavior chain task v{schema_version} must not carry "
            "output_runs.position; "
            "the exact position run is an input."
        )
    source_run_names = (
        ("body_frame", "stimulus")
        if position_is_explicit
        else ("keypoint", "body_frame", "stimulus")
    )
    for key in source_run_names:
        sources[key] = _run_name(sources.get(key), label=f"source_runs.{key}")
    if position_is_explicit:
        sources["position"] = _subject_position_reference(sources.get("position"))
    output_run_names = (() if position_is_explicit else ("position",)) + (
        "tracking",
        "motion",
        "swim_bouts",
        "stimulus_epochs_v1",
        "stimulus_epochs_v2",
        "epoch_summary",
    )
    for key in output_run_names:
        outputs[key] = _run_name(outputs.get(key), label=f"output_runs.{key}")
    payload["source_runs"] = sources
    payload["output_runs"] = outputs
    semantic_run = payload.get("protocol_semantic_selection_run")
    if schema_version == LEGACY_TASK_SCHEMA_VERSION and semantic_run is not None:
        raise ProviderBehaviorChainError(
            "protocol_semantic_selection_run requires provider behavior chain task v2."
        )
    if (
        schema_version
        in {
            SEMANTIC_TASK_SCHEMA_VERSION,
            EXPLICIT_POSITION_TASK_SCHEMA_VERSION,
            RECEIPTED_TASK_SCHEMA_VERSION,
        }
        and semantic_run is None
    ):
        raise ProviderBehaviorChainError(
            f"Provider behavior chain task v{schema_version} requires "
            "protocol_semantic_selection_run."
        )
    if semantic_run is not None:
        payload["protocol_semantic_selection_run"] = _run_name(
            semantic_run,
            label="protocol_semantic_selection_run",
        )
    fps = payload.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise ProviderBehaviorChainError("fps must be one positive number.")
    if receipted and type(fps) is not float:
        raise ProviderBehaviorChainError(
            "Provider behavior chain task v4 fps must use the canonical JSON "
            "floating-point representation."
        )
    payload["fps"] = float(fps)
    if receipted:
        arena_id = payload.get("arena_id")
        if type(arena_id) is not int or arena_id < 0:
            raise ProviderBehaviorChainError(
                "Provider behavior chain task v4 arena_id must be an integer >= 0."
            )
    if payload.get("metric_disposition") != LINEAR_ONLY_DISPOSITION:
        raise ProviderBehaviorChainError(
            "The task must explicitly bind the linear-only talk metric disposition."
        )
    if (
        receipted
        and verify_receipted_digest
        and persisted_task_sha256 != _task_digest(payload)
    ):
        raise ProviderBehaviorChainError(
            "Provider behavior chain task v4 is not in its canonical sealed form."
        )
    return payload


def build_receipted_task(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize and seal one explicit-position v4 task document."""

    payload = json_attr_safe(_object(value, label="task"))
    payload["schema_id"] = TASK_SCHEMA_ID
    payload["schema_version"] = RECEIPTED_TASK_SCHEMA_VERSION
    payload.pop("task_sha256", None)
    archive_text = _text(payload.get("analysis_zarr"), label="analysis_zarr")
    payload["analysis_zarr"] = str(Path(archive_text).expanduser().resolve())
    fps = payload.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise ProviderBehaviorChainError("fps must be one positive number.")
    payload["fps"] = float(fps)
    payload.setdefault("arena_id", 0)
    payload["task_sha256"] = _task_digest(payload)
    return _normalize_task(payload, verify_receipted_digest=True)


def load_task(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        payload = _object(source, label="task")
    else:
        path = Path(source).expanduser().resolve()
        payload = _object(
            json.loads(path.read_text(encoding="utf-8")),
            label="task",
        )
    return _normalize_task(payload, verify_receipted_digest=True)


def build_chain_receipt(
    task: Mapping[str, Any],
    stages: Mapping[str, Any],
    *,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a task-bound receipt with exact stage and envelope digests."""

    normalized_task = _normalize_task(task, verify_receipted_digest=True)
    if normalized_task["schema_version"] != RECEIPTED_TASK_SCHEMA_VERSION:
        raise ProviderBehaviorChainError(
            "A provider behavior chain receipt requires one sealed task v4."
        )
    safe_stages = json_attr_safe(_object(stages, label="receipt stages"))
    if set(safe_stages) != set(RECEIPT_STAGE_NAMES) or any(
        not isinstance(safe_stages[name], Mapping) for name in RECEIPT_STAGE_NAMES
    ):
        raise ProviderBehaviorChainError(
            "Provider behavior chain receipt stages are incomplete or unexpected."
        )
    for name in RECEIPT_STAGE_NAMES:
        stage = _object(safe_stages[name], label=f"receipt stage {name}")
        if stage.get("status") not in {"published", "reused"}:
            raise ProviderBehaviorChainError(
                f"Provider behavior chain receipt stage {name} status is invalid."
            )
        _text(stage.get("run_path"), label=f"receipt stage {name} run_path")
        for field_name in RECEIPT_STAGE_DIGEST_FIELDS[name]:
            try:
                _sha256(
                    stage.get(field_name),
                    label=f"receipt stage {name} {field_name}",
                )
            except ProviderBehaviorChainError as exc:
                raise ProviderBehaviorChainError(
                    f"Provider behavior chain receipt stage {name} artifact binding "
                    "is incomplete or invalid."
                ) from exc
    completed = (
        datetime.now(timezone.utc).isoformat()
        if completed_at_utc is None
        else _text(completed_at_utc, label="completed_at_utc")
    )
    body = json_attr_safe(
        {
            "schema_id": RECEIPT_SCHEMA_ID,
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": completed,
            "task_schema_id": TASK_SCHEMA_ID,
            "task_schema_version": RECEIPTED_TASK_SCHEMA_VERSION,
            "task_sha256": normalized_task["task_sha256"],
            "recording_id": normalized_task["recording_id"],
            "analysis_zarr": normalized_task["analysis_zarr"],
            "metric_disposition": LINEAR_ONLY_DISPOSITION,
            "selector_updates": False,
            "source_payloads_rewritten": False,
            "stages": safe_stages,
            "stage_sha256s": {
                name: canonical_json_sha256(safe_stages[name])
                for name in RECEIPT_STAGE_NAMES
            },
        }
    )
    return {**body, "receipt_sha256": canonical_json_sha256(body)}


def validate_chain_receipt(
    source: str | Path | Mapping[str, Any],
    *,
    expected_task_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a persisted v1 chain receipt without consulting mutable selectors."""

    if isinstance(source, Mapping):
        receipt = json_attr_safe(_object(source, label="receipt"))
    else:
        path = Path(source).expanduser().resolve()
        receipt = _object(
            json.loads(path.read_text(encoding="utf-8")),
            label="receipt",
        )
    if set(receipt) != _RECEIPT_FIELDS:
        raise ProviderBehaviorChainError(
            "Provider behavior chain receipt has an unexpected field set."
        )
    if (
        receipt.get("schema_id") != RECEIPT_SCHEMA_ID
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("task_schema_id") != TASK_SCHEMA_ID
        or receipt.get("task_schema_version") != RECEIPTED_TASK_SCHEMA_VERSION
        or receipt.get("metric_disposition") != LINEAR_ONLY_DISPOSITION
        or receipt.get("selector_updates") is not False
        or receipt.get("source_payloads_rewritten") is not False
    ):
        raise ProviderBehaviorChainError(
            "Provider behavior chain receipt identity or safety state is invalid."
        )
    _text(receipt.get("completed_at_utc"), label="receipt completed_at_utc")
    _text(receipt.get("recording_id"), label="receipt recording_id")
    _text(receipt.get("analysis_zarr"), label="receipt analysis_zarr")
    task_sha256 = _sha256(receipt.get("task_sha256"), label="receipt task_sha256")
    if expected_task_sha256 is not None and task_sha256 != _sha256(
        expected_task_sha256,
        label="expected task_sha256",
    ):
        raise ProviderBehaviorChainError(
            "Provider behavior chain receipt belongs to another task."
        )
    stages = _object(receipt.get("stages"), label="receipt stages")
    stage_sha256s = _object(
        receipt.get("stage_sha256s"),
        label="receipt stage_sha256s",
    )
    if set(stages) != set(RECEIPT_STAGE_NAMES) or set(stage_sha256s) != set(
        RECEIPT_STAGE_NAMES
    ):
        raise ProviderBehaviorChainError(
            "Provider behavior chain receipt stage set is incomplete or unexpected."
        )
    for name in RECEIPT_STAGE_NAMES:
        stage = _object(stages.get(name), label=f"receipt stage {name}")
        if stage.get("status") not in {"published", "reused"}:
            raise ProviderBehaviorChainError(
                f"Provider behavior chain receipt stage {name} status is invalid."
            )
        _text(stage.get("run_path"), label=f"receipt stage {name} run_path")
        for field_name in RECEIPT_STAGE_DIGEST_FIELDS[name]:
            try:
                _sha256(
                    stage.get(field_name),
                    label=f"receipt stage {name} {field_name}",
                )
            except ProviderBehaviorChainError as exc:
                raise ProviderBehaviorChainError(
                    f"Provider behavior chain receipt stage {name} artifact binding "
                    "is incomplete or invalid."
                ) from exc
        persisted = _sha256(
            stage_sha256s.get(name),
            label=f"receipt stage {name} digest",
        )
        if persisted != canonical_json_sha256(stage):
            raise ProviderBehaviorChainError(
                f"Provider behavior chain receipt stage {name} digest is stale."
            )
    persisted_receipt_sha256 = _sha256(
        receipt.get("receipt_sha256"),
        label="receipt_sha256",
    )
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if persisted_receipt_sha256 != canonical_json_sha256(body):
        raise ProviderBehaviorChainError(
            "Provider behavior chain receipt_sha256 is stale or the receipt was tampered."
        )
    return receipt


def _path(archive: Path, parent: str, run_name: str) -> Path:
    return archive.joinpath(*parent.split("/"), run_name)


def _position(task: Mapping[str, Any], scratch: Path) -> tuple[Any, dict[str, Any]]:
    archive = Path(str(task["analysis_zarr"]))
    sources = task["source_runs"]
    if task["schema_version"] in _EXPLICIT_POSITION_TASK_SCHEMA_VERSIONS:
        reference = sources["position"]
        validate_direct_consolidated_subtree(
            archive,
            subtree_path=reference["run_path"],
        )
        handle = load_subject_position_source_handle(
            archive,
            reference["run_path"],
            expected_selector_eligible=False,
            expected_manifest_sha256=reference["manifest_sha256"],
            use_consolidated=True,
        )
        direct = load_subject_position_source_handle(
            archive,
            reference["run_path"],
            expected_selector_eligible=False,
            expected_manifest_sha256=reference["manifest_sha256"],
            use_consolidated=False,
        )
        if (
            direct.manifest_sha256 != handle.manifest_sha256
            or direct.decoded_content_sha256 != handle.decoded_content_sha256
        ):
            raise ProviderBehaviorChainError(
                "Subject-position direct content digest differs from its published "
                "consolidated generation."
            )
        return handle, {
            "status": "reused",
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "decoded_content_sha256": handle.decoded_content_sha256,
        }
    outputs = task["output_runs"]
    run_path = f"analysis/subject_position_runs/observation/{outputs['position']}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        handle = load_subject_position_source_handle(
            archive,
            run_path,
            expected_selector_eligible=False,
            use_consolidated=True,
        )
        return handle, {
            "status": "reused",
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "decoded_content_sha256": handle.decoded_content_sha256,
        }
    profile_path = Path(_text(task.get("anatomy_profile"), label="anatomy_profile"))
    profile = load_anatomy_profile(profile_path)
    keypoint_path = f"keypoints_runs/{sources['keypoint']}"
    keypoints = load_bound_keypoint_position_source(
        archive,
        run_path=keypoint_path,
        policy=KeypointPositionSourcePolicy(
            anatomy_profile=profile,
            binding_id=KEYPOINT_BINDING_ID,
            authority_mode=KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY,
        ),
    )
    prepared = prepare_subject_position_input(
        keypoints,
        estimator_id=KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
        software_record=_object(task.get("software"), label="software"),
    )
    plan = plan_subject_position_run(
        archive,
        prepared,
        run_name=outputs["position"],
        scratch_root=scratch / "position",
    )
    result = publish_subject_position_run(plan, keep_scratch=False)
    handle = load_subject_position_source_handle(
        archive,
        run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    return handle, {
        "status": "published",
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "decoded_content_sha256": handle.decoded_content_sha256,
        "publication": result["publication"],
    }


def _tracking(
    task: Mapping[str, Any], position: Any, scratch: Path
) -> tuple[Any, dict[str, Any]]:
    position = require_subject_position_source_handle(position)
    archive = Path(str(task["analysis_zarr"]))
    run_name = task["output_runs"]["tracking"]
    run_path = f"tracking_runs/{run_name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        handle = load_tracking_source_handle(
            archive,
            run_path,
            expected_selector_eligible=False,
            use_consolidated=True,
        )
        source = handle.manifest["payload"]["source"]
        expected = {
            "source_authority_kind": "subject_position_run",
            "source_subject_position_run": position.run_path,
            "source_subject_position_manifest_sha256": position.manifest_sha256,
            "source_subject_position_decoded_content_sha256": (
                position.decoded_content_sha256
            ),
        }
        if any(source.get(name) != value for name, value in expected.items()):
            raise ProviderBehaviorChainError(
                "Existing tracking output binds another subject-position authority."
            )
        return handle, {
            "status": "reused",
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "verification_digest": handle.verification_digest,
        }
    plan = plan_single_subject_tracking_run(
        position,
        arena_id=int(task.get("arena_id", 0)),
        run_name=run_name,
        scratch_root=scratch / "tracking",
    )
    result = publish_single_subject_tracking_run(plan, keep_scratch=False)
    handle = load_tracking_source_handle(
        archive,
        run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    return handle, {
        "status": "published",
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "verification_digest": handle.verification_digest,
        "publication": result["publication"],
    }


def _motion(
    task: Mapping[str, Any], position: Any, tracking: Any, scratch: Path
) -> tuple[Any, dict[str, Any]]:
    archive = Path(str(task["analysis_zarr"]))
    run_name = task["output_runs"]["motion"]
    run_path = f"analysis/track_kinematics_runs/provider/{run_name}"
    body_path = f"analysis/body_frame_runs/{task['source_runs']['body_frame']}"
    body_frame = load_body_frame_source_handle(
        archive,
        run_path=body_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    authority = compose_position_body_frame_motion_authority(position, body_frame)
    tracked = bind_position_body_frame_to_tracking(authority, tracking)
    timing = load_provider_recording_timing_authority(
        archive,
        required=True,
        use_consolidated=True,
    )
    assert timing is not None
    if timing.nominal_fps != float(task["fps"]):
        raise ProviderBehaviorChainError(
            "Task FPS differs from the recording timing authority's nominal FPS."
        )
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        handle = load_provider_track_motion_source_handle(
            archive,
            run_path,
            use_consolidated=True,
            require_authoritative_timing=True,
        )
        if (
            handle.source_authority_sha256 != authority.authority_sha256
            or handle.tracked_input_sha256 != tracked.authority_sha256
            or handle.temporal_authority_sha256 != timing.sha256
        ):
            raise ProviderBehaviorChainError(
                "Existing motion output binds another composed authority."
            )
        return handle, {
            "status": "reused",
            "run_path": handle.run_path,
            "manifest_sha256": handle.provider_manifest_sha256,
            "verification_digest": handle.verification_digest,
        }
    prepared = prepare_provider_track_motion(
        tracked,
        fps=float(task["fps"]),
        smooth_seconds=0.05,
        hysteresis_high_px=4.0,
        hysteresis_low_px=2.0,
        hysteresis_min_frames=3,
        temporal_authority=timing,
    )
    plan = plan_provider_track_motion_run(
        archive,
        prepared,
        run_name=run_name,
        scratch_root=scratch / "motion",
    )
    result = publish_provider_track_motion_run(plan, keep_scratch=False)
    handle = load_provider_track_motion_source_handle(
        archive,
        run_path,
        use_consolidated=True,
        require_authoritative_timing=True,
    )
    return handle, {
        "status": "published",
        "run_path": handle.run_path,
        "manifest_sha256": handle.provider_manifest_sha256,
        "verification_digest": handle.verification_digest,
        "publication": result["publication"],
    }


def _swim_bout_content_sha256(tables: SwimBoutTables) -> str:
    arrays = {
        "bouts": tables.bouts,
        "peak_events": tables.peak_events,
        "inter_bout_intervals": tables.inter_bout_intervals,
        "inter_bout_interval_histogram": tables.inter_bout_interval_histogram,
        "global_metrics": tables.global_metrics,
        "trials": tables.trials,
        "bout_points": tables.bout_points,
        **{f"series/{name}": value for name, value in tables.series.items()},
    }
    return canonical_json_sha256(
        {
            "run_path": tables.run_path,
            "candidate_id": int(tables.candidate.candidate_id),
            "signal_id": int(tables.signal.signal_id),
            "lineage_hash": tables.run_attrs.get("lineage_hash"),
            "arrays": {
                name: array_values_sha256(np.asarray(value))
                for name, value in sorted(arrays.items())
            },
        }
    )


def _load_matching_swim_bout(
    archive: Path,
    *,
    run_name: str,
    motion: Any,
) -> SwimBoutTables:
    run_path = f"analysis/swim_bout_runs/{run_name}"
    validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    consolidated = load_exact_selector_ineligible_default_swim_bout_tables(
        open_zarr_root(archive, mode="r", use_consolidated=True),
        run_name=run_name,
    )
    direct = load_exact_selector_ineligible_default_swim_bout_tables(
        open_zarr_root(archive, mode="r", use_consolidated=False),
        run_name=run_name,
    )
    attrs = consolidated.run_attrs
    authority = attrs.get("source_track_motion_authority")
    frame_axis = attrs.get("frame_axis_contract")
    expected_frame_sha256 = canonical_frame_axis_sha256(
        np.asarray(motion.source_acquisition_frame_index, dtype=np.int64)
    )
    if (
        _swim_bout_content_sha256(direct) != _swim_bout_content_sha256(consolidated)
        or attrs.get("source_track_kinematics_scope") != "provider"
        or attrs.get("source_track_kinematics_run") != motion.run_name
        or attrs.get("source_track_motion_manifest_sha256")
        != motion.provider_manifest_sha256
        or int(attrs.get("track_id", -1)) != 0
        or not isinstance(authority, Mapping)
        or authority.get("motion_manifest_sha256") != motion.provider_manifest_sha256
        or authority.get("provider_verification_digest") != motion.verification_digest
        or authority.get("track_id") != 0
        or authority.get("track_row_start") != 0
        or authority.get("track_row_stop") != motion.row_count
        or not isinstance(frame_axis, Mapping)
        or frame_axis.get("source_track_motion_manifest_sha256")
        != motion.provider_manifest_sha256
        or frame_axis.get("content_sha256") != expected_frame_sha256
    ):
        raise ProviderBehaviorChainError(
            "Existing swim-bout output binds another provider-motion authority."
        )
    return consolidated


def _swim_bouts(
    task: Mapping[str, Any], motion: Any, scratch: Path
) -> tuple[SwimBoutTables, dict[str, Any]]:
    archive = Path(str(task["analysis_zarr"]))
    outputs = task["output_runs"]
    run_name = outputs["swim_bouts"]
    target = _path(archive, "analysis/swim_bout_runs", run_name)
    if target.exists():
        tables = _load_matching_swim_bout(
            archive,
            run_name=run_name,
            motion=motion,
        )
        return tables, {
            "status": "reused",
            "run_path": tables.run_path,
            "lineage_hash": tables.run_attrs["lineage_hash"],
            "content_sha256": _swim_bout_content_sha256(tables),
            "source_motion_manifest_sha256": motion.provider_manifest_sha256,
        }
    writer_arguments = (
        "--track-kinematics-scope",
        "provider",
        "--track-kinematics-run",
        outputs["motion"],
        "--track-id",
        "0",
        "--method",
        "peak_event",
        "--default-level",
        "exponential",
        "--exponential-tau-s",
        "0.25",
        "--min-peak-prominence-mm-s",
        "4.0",
        "--min-peak-distance-s",
        "0.10",
        "--peak-width-rel-height",
        "0.98",
        "--layout",
        "compact_v2",
    )
    result = materialize_swim_bouts(
        archive,
        scratch_root=scratch / "swim_bouts",
        run_name=run_name,
        writer_arguments=writer_arguments,
        promote=False,
        copy_backend="python",
        apply=True,
        keep_scratch=False,
    )
    tables = _load_matching_swim_bout(
        archive,
        run_name=run_name,
        motion=motion,
    )
    return tables, {
        "status": "published",
        "run_path": tables.run_path,
        "lineage_hash": tables.run_attrs["lineage_hash"],
        "content_sha256": _swim_bout_content_sha256(tables),
        "source_motion_manifest_sha256": motion.provider_manifest_sha256,
        "result": result,
    }


def _validate_legacy_epoch_reuse(
    archive: Path,
    *,
    run_name: str,
    source_stimulus_run: str,
) -> dict[str, Any]:
    run_path = f"analysis/stimulus_epoch_runs/{run_name}"
    validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    records: list[dict[str, Any]] = []
    for use_consolidated in (True, False):
        run = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=use_consolidated,
        )[run_path]
        errors = validate_legacy_stimulus_epoch_source(run)
        if errors:
            raise ProviderBehaviorChainError(
                "Existing legacy stimulus-epoch output is invalid: " + "; ".join(errors)
            )
        attrs = run.attrs
        lineage_payload_json = attrs.get("lineage_payload_json")
        try:
            lineage_payload = json.loads(lineage_payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProviderBehaviorChainError(
                "Existing legacy stimulus-epoch lineage payload is invalid."
            ) from exc
        if (
            not isinstance(lineage_payload, Mapping)
            or canonical_lineage_json(lineage_payload) != lineage_payload_json
            or compute_run_lineage_hash(lineage_payload) != attrs.get("lineage_hash")
        ):
            raise ProviderBehaviorChainError(
                "Existing legacy stimulus-epoch lineage payload is stale."
            )
        if (
            attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
            or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or attrs.get("stage_selector_eligible") is not False
            or attrs.get("source_stimulus_run") != source_stimulus_run
            or attrs.get("source_stimulus_path")
            != f"analysis/stimulus_runs/{source_stimulus_run}"
        ):
            raise ProviderBehaviorChainError(
                "Existing legacy stimulus-epoch output binds another stimulus "
                "authority or lifecycle."
            )
        records.append(
            {
                "logical_content_sha256": stimulus_epoch_logical_content_sha256(run),
                "lineage_hash": attrs.get("lineage_hash"),
                "lineage_payload_sha256": hashlib.sha256(
                    lineage_payload_json.encode("utf-8")
                ).hexdigest(),
            }
        )
    if records[0] != records[1]:
        raise ProviderBehaviorChainError(
            "Legacy stimulus-epoch direct metadata differs from its published "
            "consolidated generation."
        )
    return records[0]


def _validate_epoch_candidate_reuse(
    archive: Path,
    *,
    run_name: str,
    source_run_name: str,
    source_record: Mapping[str, Any],
    source_stimulus_run: str,
) -> ResolvedEpochSelection:
    selection = resolve_exact_stimulus_epoch_selection(archive, run_name=run_name)
    run = open_zarr_root(
        archive,
        mode="r",
        use_consolidated=True,
    )[f"analysis/stimulus_epoch_runs/{run_name}"]
    attrs = run.attrs
    if (
        attrs.get("source_stimulus_epoch_run") != source_run_name
        or attrs.get("source_stimulus_epoch_path")
        != f"analysis/stimulus_epoch_runs/{source_run_name}"
        or attrs.get("source_stimulus_run") != source_stimulus_run
        or attrs.get("source_stimulus_epoch_logical_content_sha256")
        != source_record["logical_content_sha256"]
        or attrs.get("source_stimulus_epoch_lineage_hash")
        != source_record["lineage_hash"]
        or attrs.get("source_stimulus_epoch_lineage_payload_sha256")
        != source_record["lineage_payload_sha256"]
    ):
        raise ProviderBehaviorChainError(
            "Existing stimulus-epoch candidate binds another legacy epoch or "
            "stimulus authority."
        )
    return selection


def _epochs(
    task: Mapping[str, Any], scratch: Path
) -> tuple[dict[str, Any], dict[str, Any], ResolvedEpochSelection]:
    archive = Path(str(task["analysis_zarr"]))
    sources = task["source_runs"]
    outputs = task["output_runs"]
    v1_name = outputs["stimulus_epochs_v1"]
    v1_path = _path(archive, "analysis/stimulus_epoch_runs", v1_name)
    if v1_path.exists():
        v1_record = _validate_legacy_epoch_reuse(
            archive,
            run_name=v1_name,
            source_stimulus_run=sources["stimulus"],
        )
        v1 = {
            "status": "reused",
            "run_path": f"analysis/stimulus_epoch_runs/{v1_name}",
            **v1_record,
        }
    else:
        epoch_result = build_stimulus_epoch_result(
            archive,
            run_name=v1_name,
            stimulus_run=sources["stimulus"],
        )
        written = write_stimulus_epoch_run(
            archive,
            epoch_result,
            overwrite=False,
            selector_ineligible=True,
        )
        v1 = {"status": "published", "run_path": str(written)}
        direct = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=False,
        )[f"analysis/stimulus_epoch_runs/{v1_name}"]
        v1_record = {
            "logical_content_sha256": stimulus_epoch_logical_content_sha256(direct),
            "lineage_hash": direct.attrs["lineage_hash"],
            "lineage_payload_sha256": direct.attrs["lineage_payload_sha256"],
        }
        v1.update(v1_record)
    v2_name = outputs["stimulus_epochs_v2"]
    v2_path = _path(archive, "analysis/stimulus_epoch_runs", v2_name)
    if v2_path.exists():
        selection = _validate_epoch_candidate_reuse(
            archive,
            run_name=v2_name,
            source_run_name=v1_name,
            source_record=v1_record,
            source_stimulus_run=sources["stimulus"],
        )
        v2 = {
            "status": "reused",
            "run_path": selection.run_path,
            "manifest_sha256": selection.run_manifest_digest,
            "selection_sha256": selection.selection_digest,
        }
    else:
        result = materialize_stimulus_epoch_candidate(
            archive,
            source_run=v1_name,
            run_name=v2_name,
            scratch_root=scratch / "stimulus_epochs_v2",
            allow_selector_ineligible_source=True,
            copy_backend="python",
            apply=True,
            keep_scratch=False,
        )
        v2 = {
            "status": "published",
            "run_path": f"analysis/stimulus_epoch_runs/{v2_name}",
            "result": result,
        }
        selection = _validate_epoch_candidate_reuse(
            archive,
            run_name=v2_name,
            source_run_name=v1_name,
            source_record=v1_record,
            source_stimulus_run=sources["stimulus"],
        )
        v2.update(
            manifest_sha256=selection.run_manifest_digest,
            selection_sha256=selection.selection_digest,
        )
    return v1, v2, selection


def _load_matching_summary(task: Mapping[str, Any]) -> Any:
    archive = Path(str(task["analysis_zarr"]))
    outputs = task["output_runs"]
    run_name = outputs["epoch_summary"]
    motion = load_provider_track_motion_source_handle(
        archive,
        f"analysis/track_kinematics_runs/provider/{outputs['motion']}",
        use_consolidated=True,
        require_authoritative_timing=True,
    )
    bouts = _load_matching_swim_bout(
        archive,
        run_name=outputs["swim_bouts"],
        motion=motion,
    )
    selection = resolve_exact_stimulus_epoch_selection(
        archive,
        run_name=outputs["stimulus_epochs_v2"],
    )
    semantic = load_protocol_semantic_chaser_selection_source_handle(
        archive,
        run_name=task["protocol_semantic_selection_run"],
        expected_recording_id=task["recording_id"],
        use_consolidated=True,
        deep_audit=True,
    )
    handle = load_provider_epoch_behavior_summary_source_handle(
        archive,
        run_name=run_name,
        expected_recording_id=task["recording_id"],
        expected_semantic_selection=semantic.source_binding(),
        use_consolidated=True,
        deep_audit=True,
    )
    sources = handle.manifest["sources"]
    motion_binding = sources.get("provider_motion")
    bout_binding = sources.get("swim_bouts")
    epoch_binding = sources.get("epoch_selection")
    if (
        not isinstance(motion_binding, Mapping)
        or motion_binding.get("run_path") != motion.run_path
        or motion_binding.get("manifest_sha256") != motion.provider_manifest_sha256
        or motion_binding.get("verification_digest") != motion.verification_digest
        or not isinstance(bout_binding, Mapping)
        or bout_binding.get("run_path") != bouts.run_path
        or bout_binding.get("lineage_hash") != bouts.run_attrs.get("lineage_hash")
        or bout_binding.get("source_track_motion_manifest_sha256")
        != motion.provider_manifest_sha256
        or not isinstance(epoch_binding, Mapping)
        or epoch_binding.get("sha256") != selection.selection_digest
    ):
        raise ProviderBehaviorChainError(
            "Existing epoch summary binds another motion, bout, or epoch authority."
        )
    return handle


def _summary(task: Mapping[str, Any], scratch: Path) -> dict[str, Any]:
    archive = Path(str(task["analysis_zarr"]))
    outputs = task["output_runs"]
    run_name = outputs["epoch_summary"]
    run_path = f"analysis/stimulus_epoch_behavior_summary_runs/{run_name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        handle = _load_matching_summary(task)
        return {
            "status": "reused",
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
        }
    result = materialize_provider_epoch_behavior_summary(
        archive,
        scratch_root=scratch / "epoch_summary",
        run_name=run_name,
        epoch_run_name=outputs["stimulus_epochs_v2"],
        protocol_semantic_selection_run_name=task.get(
            "protocol_semantic_selection_run"
        ),
        motion_run=outputs["motion"],
        swim_bout_run_name=outputs["swim_bouts"],
        track_id=0,
        speed_level=SUMMARY_SPEED_LEVEL,
        copy_backend="python",
        apply=True,
        keep_scratch=False,
    )
    handle = _load_matching_summary(task)
    return {
        "status": "published",
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "result": result,
    }


def materialize_chain(task: Mapping[str, Any], *, scratch_root: Path) -> dict[str, Any]:
    task = _normalize_task(task, verify_receipted_digest=True)
    scratch = scratch_root.expanduser().resolve()
    archive = Path(str(task["analysis_zarr"]))
    if scratch == archive or scratch.is_relative_to(archive):
        raise ProviderBehaviorChainError("Scratch root must be outside the archive.")
    stages: dict[str, Any] = {}
    position, stages["position"] = _position(task, scratch)
    tracking, stages["tracking"] = _tracking(task, position, scratch)
    motion, stages["motion"] = _motion(task, position, tracking, scratch)
    if not np.isclose(
        float(motion.computation_record["parameters"]["fps"]),
        float(task["fps"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ProviderBehaviorChainError("Published motion FPS differs from the task.")
    _bouts, stages["swim_bouts"] = _swim_bouts(task, motion, scratch)
    (
        stages["stimulus_epochs_v1"],
        stages["stimulus_epochs_v2"],
        _epoch_selection,
    ) = _epochs(task, scratch)
    stages["epoch_summary"] = _summary(task, scratch)
    if task["schema_version"] == RECEIPTED_TASK_SCHEMA_VERSION:
        return build_chain_receipt(task, stages)
    return json_attr_safe(
        {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "recording_id": task["recording_id"],
            "analysis_zarr": str(archive),
            "metric_disposition": LINEAR_ONLY_DISPOSITION,
            "selector_updates": False,
            "source_payloads_rewritten": False,
            "stages": stages,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--result-json",
        dest="result_json",
        type=Path,
        help="legacy result path, also accepted for v4 receipts",
    )
    destination.add_argument(
        "--receipt-json",
        dest="result_json",
        type=Path,
        help="v4 receipt path",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    task = load_task(args.task_json)
    if not args.apply:
        result: Mapping[str, Any] = {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "planned",
            "task": task,
            "selector_updates": False,
        }
    else:
        result = materialize_chain(task, scratch_root=args.scratch_root)
        write_json_atomic(args.result_json, result)
        if task["schema_version"] == RECEIPTED_TASK_SCHEMA_VERSION:
            result = validate_chain_receipt(
                args.result_json,
                expected_task_sha256=task["task_sha256"],
            )
    print(
        json.dumps(
            json_attr_safe(result), indent=None if args.json else 2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPLICIT_POSITION_TASK_SCHEMA_VERSION",
    "LINEAR_ONLY_DISPOSITION",
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "RECEIPT_STAGE_DIGEST_FIELDS",
    "RECEIPT_STAGE_NAMES",
    "RECEIPTED_TASK_SCHEMA_VERSION",
    "SUMMARY_SPEED_LEVEL",
    "ProviderBehaviorChainError",
    "build_chain_receipt",
    "build_receipted_task",
    "load_task",
    "materialize_chain",
    "validate_chain_receipt",
]
