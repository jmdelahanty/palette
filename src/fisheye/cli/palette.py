from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import zarr

from fisheye.registry.stage_catalog import STAGE_SPECS, StageSpec
from fisheye.registry.stage_complete import _STEP_RUN_PARENTS
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    describe_run_parent,
)
from fisheye.status_page.query import open_readonly_connection, resolve_registry_path


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2
EXIT_USAGE = 3

SCHEMA = "palette.cli.workflow_oracle.v1"


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str | None
    recording_id: str | None
    zarr_path: Path
    zarr_use: str | None = None
    status: str | None = None
    registry_path: Path | None = None


@dataclass
class StageState:
    stage: str
    state: str
    run: str | None = None
    artifact: str | None = None
    blocked_by: list[str] | None = None
    completion: str | None = None
    completed_at_utc: str | None = None
    catalog_artifacts: list[str] | None = None
    resolved_artifacts: list[str] | None = None
    mismatches: list[str] | None = None

    @property
    def complete(self) -> bool:
        return self.state == "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "state": self.state,
            "run": self.run,
            "artifact": self.artifact,
            "blocked_by": self.blocked_by or [],
            "completion": self.completion,
            "completed_at_utc": self.completed_at_utc,
            "catalog_artifacts": self.catalog_artifacts or [],
            "resolved_artifacts": self.resolved_artifacts or [],
            "mismatches": self.mismatches or [],
        }


class PaletteArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - argparse integration
        self.print_usage(sys.stderr)
        raise PaletteUsageError(message)


class PaletteUsageError(Exception):
    pass


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _path_exists(group: Any, path: str) -> bool:
    current = group
    for part in [piece for piece in str(path).strip("/").split("/") if piece]:
        try:
            if part not in current:
                return False
            current = current[part]
        except Exception:
            return False
    return True


def _get_group(group: Any, path: str) -> Any | None:
    current = group
    for part in [piece for piece in str(path).strip("/").split("/") if piece]:
        try:
            if part not in current:
                return None
            current = current[part]
        except Exception:
            return None
    return current if hasattr(current, "attrs") else None


def _group_keys(group: Any) -> list[str]:
    keys = getattr(group, "group_keys", None)
    if callable(keys):
        try:
            return sorted(str(item) for item in keys())
        except Exception:
            return []
    try:
        return sorted(str(key) for key, value in group.items() if hasattr(value, "attrs"))
    except Exception:
        return []


def _is_zarr_path(path: Path) -> bool:
    return path.is_dir() and (
        path.suffix == ".zarr"
        or (path / "zarr.json").is_file()
        or (path / ".zgroup").is_file()
    )


def _resolve_explicit_zarr_path(raw: str) -> tuple[Path | None, list[str]]:
    path = Path(raw).expanduser()
    if not path.exists():
        return None, []
    if _is_zarr_path(path):
        return path.resolve(), []
    if not path.is_dir():
        return None, [f"path is not a zarr directory: {path}"]
    candidates = sorted(candidate for candidate in path.iterdir() if _is_zarr_path(candidate))
    if not candidates:
        return None, [f"directory contains no zarr stores: {path}"]
    training = [candidate for candidate in candidates if candidate.name.endswith("_training.zarr")]
    if len(training) == 1:
        return training[0].resolve(), []
    if len(candidates) == 1:
        return candidates[0].resolve(), []
    names = ", ".join(candidate.name for candidate in candidates[:8])
    return None, [f"directory contains multiple zarr stores; pass one explicitly: {names}"]


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return open_readonly_connection(path)


def _resolve_from_registry(recording: str, registry_path: Path) -> DatasetRef | None:
    with _connect_readonly(registry_path) as conn:
        rows = conn.execute(
            """
            SELECT dataset_id, recording_id, zarr_path, zarr_use, status
            FROM datasets
            WHERE dataset_id = ?
               OR recording_id = ?
               OR zarr_path = ?
            ORDER BY
                CASE WHEN status = 'active' THEN 0 ELSE 1 END,
                CASE WHEN zarr_use = 'training' THEN 0 ELSE 1 END,
                dataset_id
            LIMIT 2;
            """,
            (recording, recording, recording),
        ).fetchall()
    if not rows:
        return None
    if len(rows) > 1:
        active_training = [
            row
            for row in rows
            if str(row["status"] or "") == "active" and str(row["zarr_use"] or "") == "training"
        ]
        row = active_training[0] if len(active_training) == 1 else rows[0]
    else:
        row = rows[0]
    return DatasetRef(
        dataset_id=_normalize_text(row["dataset_id"]),
        recording_id=_normalize_text(row["recording_id"]),
        zarr_path=Path(str(row["zarr_path"])).expanduser().resolve(),
        zarr_use=_normalize_text(row["zarr_use"]),
        status=_normalize_text(row["status"]),
        registry_path=registry_path,
    )


def _resolve_dataset(recording: str, registry: Path | None) -> tuple[DatasetRef | None, list[str]]:
    explicit, explicit_notes = _resolve_explicit_zarr_path(recording)
    if explicit is not None:
        registry_path = resolve_registry_path(registry, cwd=Path.cwd())
        if registry_path.exists():
            resolved = _resolve_from_registry(str(explicit), registry_path)
            if resolved is not None:
                return resolved, explicit_notes
        return DatasetRef(dataset_id=None, recording_id=None, zarr_path=explicit), explicit_notes
    if Path(recording).expanduser().exists():
        return None, explicit_notes
    registry_path = resolve_registry_path(registry, cwd=Path.cwd())
    if not registry_path.exists():
        return None, [
            f"registry not found: {registry_path}",
            "pass an explicit zarr path or --registry /path/to/palette_registry.sqlite",
        ]
    resolved = _resolve_from_registry(recording, registry_path)
    if resolved is None:
        return None, [
            f"recording not found in registry: {recording}",
            "register the zarr with scripts/py -m fisheye.registry.scan --registry <registry> <zarr-path>",
            "or pass an explicit zarr path",
        ]
    return resolved, explicit_notes


def _stage_parent_paths(spec: StageSpec) -> list[str]:
    mapped = list(_STEP_RUN_PARENTS.get(spec.id, ()))
    if mapped:
        return mapped
    return list(spec.artifact_families)


def _is_run_parent(group: Any) -> bool:
    if group is None:
        return False
    if _group_keys(group):
        return True
    attrs = getattr(group, "attrs", {})
    return any(str(key) in attrs for key in ("latest", "latest_complete", "palette_completion_epoch"))


def _completed_at_for_run(parent: Any, run_name: str | None) -> str | None:
    if not run_name:
        return None
    try:
        run = parent[run_name]
    except Exception:
        return None
    return _normalize_text(getattr(run, "attrs", {}).get(RUN_COMPLETED_AT_ATTR))


def _kind_from_run_summary(run_summary: dict[str, Any] | None) -> str:
    if not run_summary:
        return "unknown"
    status = str(run_summary.get("completion_status") or "")
    if status == "legacy_complete":
        return "legacy_assumed"
    if bool(run_summary.get("has_completion_contract")):
        return "instrumented"
    if bool(run_summary.get("complete")):
        return "legacy_assumed"
    return status or "unknown"


def _completion_from_parent(parent: Any, parent_path: str) -> tuple[str | None, str | None, str | None]:
    summary = describe_run_parent(parent, parent_path=parent_path)
    run_name = _normalize_text(summary.get("resolved_latest_complete"))
    if run_name is None:
        return None, None, None
    runs = summary.get("runs") or []
    run_summary = None
    for candidate in runs:
        if str(candidate.get("name") or "") == run_name:
            run_summary = dict(candidate)
            break
    return run_name, _kind_from_run_summary(run_summary), _completed_at_for_run(parent, run_name)


def _completion_from_quality_reports(root: Any) -> tuple[str | None, str | None, str | None, str | None]:
    detect_parent = _get_group(root, "detect_runs")
    if detect_parent is None:
        return None, None, None, None
    candidates: list[tuple[str, str, str | None, str | None]] = []
    for detect_run in _group_keys(detect_parent):
        quality_parent = _get_group(detect_parent, f"{detect_run}/quality_reports")
        if quality_parent is None:
            continue
        run_name, kind, completed_at = _completion_from_parent(
            quality_parent,
            f"detect_runs/{detect_run}/quality_reports",
        )
        if run_name is not None:
            candidates.append((detect_run, run_name, kind, completed_at))
    if not candidates:
        return None, None, None, None
    candidates.sort(key=lambda item: item[3] or "")
    detect_run, run_name, kind, completed_at = candidates[-1]
    return run_name, kind, completed_at, f"detect_runs/{detect_run}/quality_reports"


def _completion_for_artifact(root: Any, spec: StageSpec, path: str) -> tuple[bool, str | None, str | None, str | None]:
    if spec.id == "detect_quality":
        run_name, kind, completed_at, quality_path = _completion_from_quality_reports(root)
        if run_name is not None:
            return True, run_name, quality_path, kind or "instrumented"
        return False, None, None, None
    group = _get_group(root, path)
    if group is None:
        if _path_exists(root, path):
            return True, None, path, "artifact_present"
        return False, None, None, None
    if _is_run_parent(group):
        run_name, kind, completed_at = _completion_from_parent(group, path)
        if run_name is not None:
            return True, run_name, path, kind or "instrumented"
        if _group_keys(group):
            return False, None, path, None
    return True, None, path, "artifact_present"


def _raw_complete(root: Any) -> bool:
    return any(
        _path_exists(root, candidate)
        for candidate in (
            "raw_video/images_full",
            "raw_video/images_ds",
            "raw_video/original_frame_indices",
            "analysis_metadata",
        )
    )


def _stage_base_completion(root: Any, spec: StageSpec) -> StageState:
    mismatches: list[str] = []
    catalog_artifacts = list(spec.artifact_families)
    resolved_artifacts = _stage_parent_paths(spec)
    if catalog_artifacts != resolved_artifacts:
        mismatches.append(
            "catalog_artifacts_differ_from_completion_mapping:"
            f"catalog={catalog_artifacts};resolved={resolved_artifacts}"
        )
    if spec.id == "raw":
        if _raw_complete(root):
            return StageState(
                stage=spec.id,
                state="complete",
                artifact="raw_video",
                completion="artifact_present",
                catalog_artifacts=catalog_artifacts,
                resolved_artifacts=resolved_artifacts,
                mismatches=mismatches,
            )
        if not catalog_artifacts:
            mismatches.append("catalog_stage_has_no_artifact_family")
        return StageState(
            stage=spec.id,
            state="missing",
            catalog_artifacts=catalog_artifacts,
            resolved_artifacts=resolved_artifacts,
            mismatches=mismatches,
        )
    if not resolved_artifacts:
        mismatches.append("catalog_stage_has_no_artifact_family")
        return StageState(
            stage=spec.id,
            state="missing",
            catalog_artifacts=catalog_artifacts,
            resolved_artifacts=resolved_artifacts,
            mismatches=mismatches,
        )
    for artifact in resolved_artifacts:
        complete, run_name, resolved_artifact, completion_kind = _completion_for_artifact(root, spec, artifact)
        if complete:
            completed_at = None
            group = _get_group(root, resolved_artifact or artifact)
            if group is not None and run_name is not None:
                completed_at = _completed_at_for_run(group, run_name)
            if spec.id == "detect_quality" and completed_at is None:
                _, _, completed_at, _ = _completion_from_quality_reports(root)
            return StageState(
                stage=spec.id,
                state="complete",
                run=run_name,
                artifact=resolved_artifact or artifact,
                completion=completion_kind,
                completed_at_utc=completed_at,
                catalog_artifacts=catalog_artifacts,
                resolved_artifacts=resolved_artifacts,
                mismatches=mismatches,
            )
    return StageState(
        stage=spec.id,
        state="missing",
        catalog_artifacts=catalog_artifacts,
        resolved_artifacts=resolved_artifacts,
        mismatches=mismatches,
    )


def inspect_stages(root: Any) -> list[StageState]:
    by_stage: dict[str, StageState] = {}
    out: list[StageState] = []
    for spec in STAGE_SPECS:
        state = _stage_base_completion(root, spec)
        if not state.complete:
            blocked_by = [dep for dep in spec.depends_on if not by_stage.get(dep, StageState(dep, "missing")).complete]
            if blocked_by:
                state.state = "blocked"
                state.blocked_by = blocked_by
        by_stage[spec.id] = state
        out.append(state)
    return out


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _find_stale(stages: Sequence[StageState]) -> list[dict[str, Any]]:
    by_stage = {stage.stage: stage for stage in stages}
    stale: list[dict[str, Any]] = []
    for spec in STAGE_SPECS:
        upstream = by_stage.get(spec.id)
        if upstream is None or not upstream.complete:
            continue
        upstream_time = _parse_time(upstream.completed_at_utc)
        if upstream_time is None:
            continue
        for downstream_id in spec.invalidates:
            downstream = by_stage.get(downstream_id)
            if downstream is None or not downstream.complete:
                continue
            downstream_time = _parse_time(downstream.completed_at_utc)
            if downstream_time is not None and upstream_time > downstream_time:
                stale.append(
                    {
                        "stage": downstream.stage,
                        "invalidated_by": upstream.stage,
                        "upstream_completed_at_utc": upstream.completed_at_utc,
                        "downstream_completed_at_utc": downstream.completed_at_utc,
                        "basis": "catalog_invalidates_timestamp_comparison",
                    }
                )
    return stale


def _command_context(dataset: DatasetRef) -> dict[str, str]:
    zarr_path = str(dataset.zarr_path)
    recording_dir = str(dataset.zarr_path.parent.parent) if dataset.zarr_path.parent.name == "zarr" else str(dataset.zarr_path.parent)
    registry = str(dataset.registry_path) if dataset.registry_path is not None else "<registry.sqlite>"
    recording = dataset.recording_id or dataset.dataset_id or dataset.zarr_path.stem
    return {
        "zarr_path": zarr_path,
        "recording_dir": recording_dir,
        "registry": registry,
        "recording": recording,
    }


def _action_for_stage(stage_id: str, dataset: DatasetRef) -> dict[str, Any]:
    ctx = _command_context(dataset)
    actions: dict[str, tuple[str, str]] = {
        "raw": (
            "import",
            "scripts/py -m fisheye.utils.import_organized_recordings_analysis --registry {registry} {recording_dir}",
        ),
        "detect": (
            "detect",
            "scripts/py -m fisheye.utils.run_detect_with_registry_model --recording-dir {recording_dir} --registry {registry}",
        ),
        "detect_quality": (
            "detect-quality",
            "scripts/submit_detect_quality_refine_bsub.sh --registry {registry} --path-contains {recording}",
        ),
        "refined_detect": (
            "refine-detect",
            "scripts/submit_detect_quality_refine_bsub.sh --registry {registry} --path-contains {recording}",
        ),
        "crop": (
            "crop",
            "scripts/submit_crop_batches_bsub.sh --source registry --registry {registry} --path-contains {recording}",
        ),
        "keypoints": (
            "keypoints",
            "scripts/py -m fisheye.utils.run_keypoints_with_registry_model --recording-dir {recording_dir} --registry {registry}",
        ),
        "subject_masks": (
            "subject-masks",
            "scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline {zarr_path}",
        ),
        "refined_subject_masks": (
            "refine-subject-masks",
            "scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline {zarr_path} --workflow-stage finalization",
        ),
        "tracks": (
            "track",
            "scripts/py -m fisheye.utils.run_recording_analysis_pipeline --registry {registry} --recording-dir {recording_dir}",
        ),
    }
    item = actions.get(stage_id)
    if item is None:
        return {
            "palette_verb": stage_id.replace("_", "-"),
            "action": "",
            "action_status": "unmapped",
            "mismatch": "no_current_command_mapping_for_catalog_stage",
        }
    verb, template = item
    return {
        "palette_verb": verb,
        "action": template.format(**ctx),
        "action_status": "mapped",
    }


def _mismatch_report(stages: Sequence[StageState], actions: Sequence[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for stage in stages:
        for mismatch in stage.mismatches or []:
            report.append({"stage": stage.stage, "mismatch": mismatch})
    for action in actions:
        mismatch = action.get("mismatch")
        if mismatch:
            report.append({"stage": action.get("stage"), "mismatch": mismatch})
    return report


def _base_envelope(command: str, dataset: DatasetRef | None, *, status: str, reason_code: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "command": command,
        "status": status,
        "reason_code": reason_code,
        "recording": dataset.recording_id if dataset else None,
        "dataset_id": dataset.dataset_id if dataset else None,
        "zarr_path": str(dataset.zarr_path) if dataset else None,
        "run": None,
        "artifacts": [str(dataset.zarr_path)] if dataset else [],
        "metrics": {},
        "next_hints": [],
        "provenance": {
            "generated_at_utc": _utc_now(),
            "read_only": True,
            "completion_source": "fisheye.shared.zarr_run_completion",
            "catalog_source": "fisheye.registry.stage_catalog",
        },
    }


def build_status_payload(dataset: DatasetRef, stages: Sequence[StageState]) -> dict[str, Any]:
    payload = _base_envelope("palette status", dataset, status="ok", reason_code="OK")
    counts = {
        "complete": sum(1 for stage in stages if stage.state == "complete"),
        "missing": sum(1 for stage in stages if stage.state == "missing"),
        "blocked": sum(1 for stage in stages if stage.state == "blocked"),
    }
    payload["metrics"] = counts
    payload["stages"] = [stage.to_dict() for stage in stages]
    payload["mismatches"] = _mismatch_report(stages)
    return payload


def build_plan_payload(dataset: DatasetRef, stages: Sequence[StageState]) -> dict[str, Any]:
    next_items: list[dict[str, Any]] = []
    for stage in stages:
        if stage.state != "missing":
            continue
        action = _action_for_stage(stage.stage, dataset)
        action["stage"] = stage.stage
        action["reason"] = "dependencies_complete"
        next_items.append(action)
    stale = _find_stale(stages)
    payload = _base_envelope("palette plan", dataset, status="ok", reason_code="OK")
    payload["next"] = next_items
    payload["stale"] = stale
    payload["metrics"] = {
        "next_count": len(next_items),
        "stale_count": len(stale),
        "complete": sum(1 for stage in stages if stage.state == "complete"),
        "missing": sum(1 for stage in stages if stage.state == "missing"),
        "blocked": sum(1 for stage in stages if stage.state == "blocked"),
    }
    payload["stages"] = [stage.to_dict() for stage in stages]
    payload["mismatches"] = _mismatch_report(stages, next_items)
    payload["provenance"]["stale_basis"] = "catalog_invalidates_timestamp_comparison"
    return payload


def _print_status_table(dataset: DatasetRef, stages: Sequence[StageState]) -> None:
    print(f"recording: {dataset.recording_id or '(unknown)'}")
    print(f"dataset_id: {dataset.dataset_id or '(explicit zarr path)'}")
    print(f"zarr_path: {dataset.zarr_path}")
    print("")
    print(f"{'stage':28} {'state':30} {'run':40} artifact")
    print("-" * 112)
    for stage in stages:
        label = stage.state
        if stage.state == "blocked":
            label = "blocked_by: " + ",".join(stage.blocked_by or [])
        elif stage.state == "complete" and stage.completion == "legacy_assumed":
            label = "complete (legacy-assumed)"
        print(f"{stage.stage:28} {label:30} {(stage.run or ''):40} {stage.artifact or ''}")


def _print_plan_table(dataset: DatasetRef, payload: dict[str, Any]) -> None:
    print(f"recording: {dataset.recording_id or '(unknown)'}")
    print(f"dataset_id: {dataset.dataset_id or '(explicit zarr path)'}")
    print(f"zarr_path: {dataset.zarr_path}")
    print("")
    if not payload["next"] and not payload["stale"]:
        print("plan: no runnable missing or stale stages found")
        return
    if payload["next"]:
        print("next:")
        for item in payload["next"]:
            action = item.get("action") or "(no current command mapping)"
            print(f"  - {item['stage']} [{item['palette_verb']}]: {action}")
    if payload["stale"]:
        print("stale:")
        for item in payload["stale"]:
            print(
                f"  - {item['stage']} invalidated_by={item['invalidated_by']} "
                f"basis={item['basis']}"
            )


def _blocked_payload(command: str, recording: str, hints: Sequence[str]) -> dict[str, Any]:
    payload = _base_envelope(command, None, status="blocked", reason_code="RECORDING_NOT_FOUND")
    payload["recording"] = recording
    payload["next_hints"] = list(hints)
    return payload


def _run_readonly(command: str, recording: str, registry: Path | None, json_output: bool) -> int:
    dataset, hints = _resolve_dataset(recording, registry)
    if dataset is None:
        payload = _blocked_payload(f"palette {command}", recording, hints)
        if json_output:
            _print_json(payload)
        else:
            for hint in hints:
                print(hint, file=sys.stderr)
        return EXIT_BLOCKED
    root = open_zarr_group_direct(dataset.zarr_path, mode="r")
    stages = inspect_stages(root)
    if command == "status":
        payload = build_status_payload(dataset, stages)
        if json_output:
            _print_json(payload)
        else:
            _print_status_table(dataset, stages)
    elif command == "plan":
        payload = build_plan_payload(dataset, stages)
        if json_output:
            _print_json(payload)
        else:
            _print_plan_table(dataset, payload)
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"unknown command: {command}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = PaletteArgumentParser(
        prog="palette",
        description="Palette workflow oracle. Read-only status and plan commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "plan"):
        sub = subparsers.add_parser(name, help=f"Read-only workflow {name}.")
        sub.add_argument("recording", help="Recording id, dataset id, zarr path, or directory containing one zarr.")
        sub.add_argument("--registry", type=Path, help="Path to palette_registry.sqlite.")
        sub.add_argument("--json", action="store_true", help="Print a JSON envelope.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except PaletteUsageError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    try:
        return _run_readonly(
            args.command,
            str(args.recording),
            args.registry,
            bool(args.json),
        )
    except BrokenPipeError:  # pragma: no cover - shell behavior
        return EXIT_FAILED
    except Exception as exc:
        payload = {
            "schema": SCHEMA,
            "command": f"palette {getattr(args, 'command', '')}".strip(),
            "status": "failed",
            "reason_code": "UNHANDLED_ERROR",
            "recording": getattr(args, "recording", None),
            "run": None,
            "artifacts": [],
            "metrics": {},
            "next_hints": [str(exc)],
            "provenance": {"generated_at_utc": _utc_now(), "read_only": True},
        }
        if bool(getattr(args, "json", False)):
            _print_json(payload)
        else:
            print(f"palette {getattr(args, 'command', '')} failed: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
