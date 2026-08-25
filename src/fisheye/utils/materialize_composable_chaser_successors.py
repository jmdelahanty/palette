"""Plan or publish the composable chaser successor analytics for one recording.

The command loads only explicitly named immutable inputs.  It prepares the
controller-trial, generalized bout-response, escape/freeze, and gaze
successors in dependency order and performs no writes by default.  ``--apply``
publishes each ready product below its selector-ineligible successor parent;
it never activates a selector or updates the Palette registry.

Unavailable optional inputs are reported per module.  A blocked gaze source,
for example, does not prevent controller/bout/escape preparation.  A failed
dependency does prevent publication of its consumers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    load_chaser_relative_frame_source_handle,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.controller_trial_successor import (
    prepare_controller_trial_successor_from_handles,
)
from fisheye.analysis_workflows.escape_freeze_successor import (
    prepare_escape_freeze_successor_from_handles,
)
from fisheye.analysis_workflows.eye_gaze_source_handle import (
    load_eye_gaze_source_handle,
)
from fisheye.analysis_workflows.gaze_tracking_successor import (
    prepare_gaze_tracking_successor_from_handles,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    prepare_generalized_bout_response_successor_from_handles,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    load_protocol_semantic_chaser_selection_source_handle,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    load_provider_track_motion_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic


REPORT_SCHEMA_ID = "palette.analysis.composable_chaser_successor.operator_report"
REPORT_SCHEMA_VERSION = 1
DISPOSITION = "selector_ineligible_non_authoritative_trial_v1"

CONTROLLER = "controller_chase_trials"
BOUT_RESPONSE = "generalized_chaser_bout_response"
ESCAPE_FREEZE = "chaser_escape_freeze_v2"
GAZE_TRACKING = "chaser_gaze_tracking_v2"
MODULE_ORDER = (CONTROLLER, BOUT_RESPONSE, ESCAPE_FREEZE, GAZE_TRACKING)
MODULE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    CONTROLLER: (),
    BOUT_RESPONSE: (CONTROLLER,),
    ESCAPE_FREEZE: (CONTROLLER, BOUT_RESPONSE),
    GAZE_TRACKING: (),
}
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_FORBIDDEN_RUN_NAMES = frozenset(
    {"latest", "current", "selected", "authoritative", "default"}
)


class ComposableChaserSuccessorOperatorError(ValueError):
    """Raised when the operator request itself is ambiguous or unsafe."""


def _run_name(value: object) -> str:
    if (
        type(value) is not str
        or _RUN_NAME_RE.fullmatch(value) is None
        or value.lower() in _FORBIDDEN_RUN_NAMES
    ):
        raise ComposableChaserSuccessorOperatorError(
            "run_name must be one exact non-selector child name."
        )
    return value


def _module_ids(values: Sequence[str] | None) -> tuple[str, ...]:
    requested = tuple(values) if values else MODULE_ORDER
    unknown = sorted(set(requested) - set(MODULE_ORDER))
    if unknown:
        raise ComposableChaserSuccessorOperatorError(
            f"Unknown successor modules: {unknown!r}."
        )
    if len(set(requested)) != len(requested):
        raise ComposableChaserSuccessorOperatorError(
            "Requested successor modules must be unique."
        )
    needed = set(requested)
    while True:
        prior = set(needed)
        for module_id in tuple(needed):
            needed.update(MODULE_DEPENDENCIES[module_id])
        if needed == prior:
            break
    return tuple(module_id for module_id in MODULE_ORDER if module_id in needed)


def _error_record(exc: Exception) -> dict[str, str]:
    return {
        "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
        "message": str(exc),
    }


def _attempt_source(
    *,
    source_id: str,
    loader: Callable[[], Any],
) -> tuple[Any | None, dict[str, Any]]:
    try:
        handle = loader()
    except Exception as exc:  # eligibility report must retain exact loader failure
        return None, {
            "source_id": source_id,
            "status": "blocked",
            "reason_code": "source_handle_unavailable",
            "error": _error_record(exc),
        }
    record: dict[str, Any] = {
        "source_id": source_id,
        "status": "ready",
    }
    for field in (
        "recording_id",
        "run_name",
        "run_path",
        "manifest_sha256",
        "provider_manifest_sha256",
        "logical_manifest_sha256",
        "convention_receipt_sha256",
    ):
        value = getattr(handle, field, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                record[field] = value
    return handle, record


def _missing_operator_source(
    source_id: str,
    *,
    reason_code: str,
    message: str,
) -> tuple[None, dict[str, Any]]:
    return None, {
        "source_id": source_id,
        "status": "blocked",
        "reason_code": reason_code,
        "error": {
            "exception_type": (
                "fisheye.utils.materialize_composable_chaser_successors."
                "ComposableChaserSuccessorOperatorError"
            ),
            "message": message,
        },
    }


def _prepared_record(module_id: str, prepared: Any) -> dict[str, Any]:
    manifest = getattr(prepared, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise ComposableChaserSuccessorOperatorError(
            f"Prepared module {module_id!r} lacks its scientific manifest."
        )
    dimensions = manifest.get("dimensions", {})
    return {
        "module_id": module_id,
        "status": "prepared",
        "reason_code": "scientific_payload_prepared",
        "dependencies": list(MODULE_DEPENDENCIES[module_id]),
        "scientific_payload_sha256": str(manifest["payload_digest"]),
        "dimensions": dict(dimensions) if isinstance(dimensions, Mapping) else {},
        "selector_eligible": False,
        "production_authority": False,
    }


def _blocked_module(
    module_id: str,
    *,
    reason_code: str,
    blocking_sources: Sequence[str] = (),
    blocking_modules: Sequence[str] = (),
    error: Exception | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "module_id": module_id,
        "status": "blocked",
        "reason_code": reason_code,
        "dependencies": list(MODULE_DEPENDENCIES[module_id]),
        "blocking_sources": list(blocking_sources),
        "blocking_modules": list(blocking_modules),
        "selector_eligible": False,
        "production_authority": False,
    }
    if error is not None:
        result["error"] = _error_record(error)
    return result


def run_composable_chaser_successors(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    relative_frame_run: str | None,
    semantic_selection_run: str | None,
    provider_motion_run_path: str | None,
    swim_bout_run_name: str | None,
    track_id: int = 0,
    expected_recording_id: str | None = None,
    eye_run_name: str | None = None,
    eye_convention_receipt: Mapping[str, Any] | None = None,
    eye_channel_variant: str = "smoothed",
    include_body_extension: bool = True,
    speed_level: str = "filtered",
    modules: Sequence[str] | None = None,
    apply: bool = False,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Prepare or publish every eligible requested successor."""

    archive = Path(analysis_zarr).expanduser().resolve()
    name = _run_name(run_name)
    needed = _module_ids(modules)
    requested = set(modules or MODULE_ORDER)
    if type(track_id) is not int or track_id < 0:
        raise ComposableChaserSuccessorOperatorError(
            "track_id must be one nonnegative exact integer."
        )
    if type(apply) is not bool or type(include_body_extension) is not bool:
        raise ComposableChaserSuccessorOperatorError(
            "apply and include_body_extension must be exact booleans."
        )

    sources: dict[str, dict[str, Any]] = {}
    handles: dict[str, Any | None] = {}

    if relative_frame_run is None:
        handles["relative_frame"], sources["relative_frame"] = (
            _missing_operator_source(
                "relative_frame",
                reason_code="missing_operator_argument",
                message="--relative-frame-run is required by every successor.",
            )
        )
    else:
        handles["relative_frame"], sources["relative_frame"] = _attempt_source(
            source_id="relative_frame",
            loader=lambda: load_chaser_relative_frame_source_handle(
                archive,
                run_name=relative_frame_run,
                expected_recording_id=expected_recording_id,
                use_consolidated=True,
            ),
        )

    if semantic_selection_run is None:
        handles["semantic_selection"], sources["semantic_selection"] = (
            _missing_operator_source(
                "semantic_selection",
                reason_code="missing_operator_argument",
                message="--semantic-selection-run is required by the requested successors.",
            )
        )
    else:
        handles["semantic_selection"], sources["semantic_selection"] = (
            _attempt_source(
                source_id="semantic_selection",
                loader=lambda: load_protocol_semantic_chaser_selection_source_handle(
                    archive,
                    run_name=semantic_selection_run,
                    expected_recording_id=expected_recording_id,
                    use_consolidated=True,
                    deep_audit=True,
                ),
            )
        )

    if BOUT_RESPONSE in needed or ESCAPE_FREEZE in needed:
        if provider_motion_run_path is None:
            handles["provider_motion"], sources["provider_motion"] = (
                _missing_operator_source(
                    "provider_motion",
                    reason_code="missing_operator_argument",
                    message=(
                        "--provider-motion-run-path is required by bout and "
                        "escape/freeze successors."
                    ),
                )
            )
        else:
            handles["provider_motion"], sources["provider_motion"] = (
                _attempt_source(
                    source_id="provider_motion",
                    loader=lambda: load_provider_track_motion_source_handle(
                        archive,
                        provider_motion_run_path,
                        use_consolidated=True,
                    ),
                )
            )
        if swim_bout_run_name is None:
            sources["swim_bouts"] = {
                "source_id": "swim_bouts",
                "status": "blocked",
                "reason_code": "missing_operator_argument",
            }
        else:
            sources["swim_bouts"] = {
                "source_id": "swim_bouts",
                "status": "deferred_to_exact_adapter",
                "run_name": swim_bout_run_name,
                "track_id": track_id,
            }

    if GAZE_TRACKING in needed:
        if eye_run_name is None or eye_convention_receipt is None:
            handles["eye_gaze"], sources["eye_gaze"] = _missing_operator_source(
                "eye_gaze",
                reason_code="reviewed_eye_source_not_supplied",
                message=(
                    "Gaze requires both --eye-run-name and a convention-review receipt."
                ),
            )
        else:
            handles["eye_gaze"], sources["eye_gaze"] = _attempt_source(
                source_id="eye_gaze",
                loader=lambda: load_eye_gaze_source_handle(
                    archive,
                    run_name=eye_run_name,
                    convention_receipt=eye_convention_receipt,
                    channel_variant=eye_channel_variant,
                ),
            )

    prepared: dict[str, Any] = {}
    module_records: dict[str, dict[str, Any]] = {}

    def source_block(module_id: str, required: Sequence[str]) -> bool:
        missing = [source_id for source_id in required if handles.get(source_id) is None]
        if missing:
            module_records[module_id] = _blocked_module(
                module_id,
                reason_code="source_handle_unavailable",
                blocking_sources=missing,
            )
            return True
        return False

    for module_id in needed:
        blocked_dependencies = [
            dependency
            for dependency in MODULE_DEPENDENCIES[module_id]
            if dependency not in prepared
        ]
        if blocked_dependencies:
            module_records[module_id] = _blocked_module(
                module_id,
                reason_code="dependency_unavailable",
                blocking_modules=blocked_dependencies,
            )
            continue
        try:
            if module_id == CONTROLLER:
                if source_block(module_id, ("relative_frame", "semantic_selection")):
                    continue
                result = prepare_controller_trial_successor_from_handles(
                    handles["relative_frame"],
                    handles["semantic_selection"],
                )
            elif module_id == BOUT_RESPONSE:
                if source_block(
                    module_id,
                    ("relative_frame", "semantic_selection", "provider_motion"),
                ):
                    continue
                if swim_bout_run_name is None:
                    module_records[module_id] = _blocked_module(
                        module_id,
                        reason_code="missing_operator_argument",
                        blocking_sources=("swim_bouts",),
                    )
                    continue
                result = prepare_generalized_bout_response_successor_from_handles(
                    handles["relative_frame"],
                    handles["semantic_selection"],
                    prepared[CONTROLLER],
                    handles["provider_motion"],
                    swim_bout_run_name=swim_bout_run_name,
                    track_id=track_id,
                    include_body_extension=include_body_extension,
                )
            elif module_id == ESCAPE_FREEZE:
                if source_block(module_id, ("relative_frame", "provider_motion")):
                    continue
                result = prepare_escape_freeze_successor_from_handles(
                    handles["relative_frame"],
                    handles["provider_motion"],
                    prepared[CONTROLLER],
                    prepared[BOUT_RESPONSE],
                    track_id=track_id,
                    speed_level=speed_level,
                )
            elif module_id == GAZE_TRACKING:
                if source_block(
                    module_id,
                    ("relative_frame", "semantic_selection", "eye_gaze"),
                ):
                    continue
                result = prepare_gaze_tracking_successor_from_handles(
                    handles["relative_frame"],
                    handles["semantic_selection"],
                    handles["eye_gaze"],
                )
            else:  # pragma: no cover - MODULE_ORDER is closed above
                raise AssertionError(module_id)
        except Exception as exc:
            module_records[module_id] = _blocked_module(
                module_id,
                reason_code="scientific_preparation_failed",
                error=exc,
            )
            continue
        prepared[module_id] = result
        module_records[module_id] = _prepared_record(module_id, result)

    plans: dict[str, Any] = {}
    for module_id in needed:
        if module_id not in prepared:
            continue
        unavailable_dependencies = [
            dependency
            for dependency in MODULE_DEPENDENCIES[module_id]
            if dependency not in plans
        ]
        if unavailable_dependencies:
            module_records[module_id] = _blocked_module(
                module_id,
                reason_code="dependency_publication_preflight_unavailable",
                blocking_modules=unavailable_dependencies,
            )
            prepared.pop(module_id, None)
            continue
        try:
            plan = build_composable_chaser_successor_publication_plan(
                archive,
                run_name=name,
                prepared=prepared[module_id],
            )
        except Exception as exc:
            module_records[module_id] = _blocked_module(
                module_id,
                reason_code="publication_preflight_failed",
                error=exc,
            )
            prepared.pop(module_id, None)
            continue
        plans[module_id] = plan
        module_records[module_id].update(
            {
                "status": "planned",
                "reason_code": "immutable_publication_preflight_passed",
                "successor_kind": plan.successor_kind,
                "run_path": plan.run_path,
            }
        )

    if apply:
        published: set[str] = set()
        for module_id in needed:
            if module_id not in plans:
                continue
            unavailable_dependencies = [
                dependency
                for dependency in MODULE_DEPENDENCIES[module_id]
                if dependency not in published
            ]
            if unavailable_dependencies:
                module_records[module_id] = _blocked_module(
                    module_id,
                    reason_code="dependency_publication_unavailable",
                    blocking_modules=unavailable_dependencies,
                )
                continue
            try:
                publication = publish_composable_chaser_successor_run(
                    plans[module_id],
                    scratch_root=scratch_root,
                    copy_backend=copy_backend,
                )
            except Exception as exc:
                module_records[module_id] = _blocked_module(
                    module_id,
                    reason_code="publication_failed",
                    error=exc,
                )
                continue
            published.add(module_id)
            module_records[module_id].update(
                {
                    "status": "published",
                    "reason_code": "published_selector_ineligible",
                    "publication": publication,
                }
            )

    ordered_records = []
    for module_id in needed:
        record = dict(module_records[module_id])
        record["explicitly_requested"] = module_id in requested
        ordered_records.append(record)
    ready_status = "published" if apply else "planned"
    ready_count = sum(row["status"] == ready_status for row in ordered_records)
    blocked_count = sum(row["status"] == "blocked" for row in ordered_records)
    if ready_count and blocked_count:
        overall = "published_partial" if apply else "planned_partial"
    elif ready_count:
        overall = "published_selector_ineligible" if apply else "planned_no_writes"
    else:
        overall = "blocked_no_products"

    return {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": overall,
        "disposition": DISPOSITION,
        "analysis_zarr": str(archive),
        "run_name": name,
        "apply": apply,
        "requested_modules": [
            module_id for module_id in MODULE_ORDER if module_id in requested
        ],
        "execution_modules": list(needed),
        "sources": sources,
        "modules": ordered_records,
        "summary": {
            "ready_or_published_count": ready_count,
            "blocked_count": blocked_count,
            "module_count": len(ordered_records),
        },
        "selector_eligible": False,
        "production_authority": False,
        "production_selector_activation": False,
        "registry_update": False,
    }


def _receipt(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ComposableChaserSuccessorOperatorError(
            "Eye convention receipt must decode to one JSON object."
        )
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--relative-frame-run")
    parser.add_argument("--semantic-selection-run")
    parser.add_argument("--provider-motion-run-path")
    parser.add_argument("--swim-bout-run-name")
    parser.add_argument("--track-id", type=int, default=0)
    parser.add_argument("--eye-run-name")
    parser.add_argument("--eye-convention-receipt", type=Path)
    parser.add_argument(
        "--eye-channel-variant",
        choices=("raw", "smoothed"),
        default="smoothed",
    )
    parser.add_argument(
        "--module",
        action="append",
        choices=MODULE_ORDER,
        help="Requested module; repeat as needed. Default: all four.",
    )
    parser.add_argument(
        "--no-body-extension",
        action="store_true",
        help="Run generalized bout response without optional body-frame fields.",
    )
    parser.add_argument(
        "--speed-level",
        choices=("raw", "smoothed", "filtered", "averaged"),
        default="filtered",
    )
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish ready immutable selector-ineligible products; default is dry-run.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_composable_chaser_successors(
        args.analysis_zarr,
        run_name=args.run_name,
        relative_frame_run=args.relative_frame_run,
        semantic_selection_run=args.semantic_selection_run,
        provider_motion_run_path=args.provider_motion_run_path,
        swim_bout_run_name=args.swim_bout_run_name,
        track_id=args.track_id,
        expected_recording_id=args.expected_recording_id,
        eye_run_name=args.eye_run_name,
        eye_convention_receipt=_receipt(args.eye_convention_receipt),
        eye_channel_variant=args.eye_channel_variant,
        include_body_extension=not args.no_body_extension,
        speed_level=args.speed_level,
        modules=args.module,
        apply=args.apply,
        scratch_root=args.scratch_root,
        copy_backend=args.copy_backend,
    )
    if args.output_json is not None:
        write_json_atomic(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 2 if result["status"] == "blocked_no_products" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BOUT_RESPONSE",
    "CONTROLLER",
    "DISPOSITION",
    "ESCAPE_FREEZE",
    "GAZE_TRACKING",
    "MODULE_ORDER",
    "REPORT_SCHEMA_ID",
    "REPORT_SCHEMA_VERSION",
    "ComposableChaserSuccessorOperatorError",
    "build_arg_parser",
    "main",
    "run_composable_chaser_successors",
]
