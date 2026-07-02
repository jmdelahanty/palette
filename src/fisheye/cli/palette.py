from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
import json
import shlex
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import zarr

from fisheye.cli.envelope import (
    EXIT_BLOCKED,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    SCHEMA,
    build_envelope,
    build_run_provenance,
    exit_code_for_status,
    json_ready,
)
from fisheye.registry.stage_catalog import STAGE_SPECS, StageSpec
from fisheye.registry.stage_complete import _STEP_RUN_PARENTS
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    describe_run_parent,
)
from fisheye.status_page.query import open_readonly_connection, resolve_registry_path

_COMPLETION_PARENT_OVERRIDES: dict[str, tuple[str, ...]] = {
    # Catalog-vs-reality bridge: stage_catalog still names this artifact
    # "background", while the live background runner writes background_runs/.
    "background": ("background_runs", "background"),
}


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
    deprecated: bool = False
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
            "deprecated": self.deprecated,
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
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True, default=_json_default))


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
    override = _COMPLETION_PARENT_OVERRIDES.get(spec.id)
    if override:
        return list(override)
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
        state.deprecated = bool(spec.deprecated)
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
        "background": (
            "background",
            "scripts/py -m fisheye.preprocessing.background {zarr_path}",
        ),
        "detect": (
            "detect",
            "palette detect {zarr_path} --registry {registry} --apply",
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
            "palette crop {zarr_path} --registry {registry} --apply",
        ),
        "keypoints": (
            "keypoints",
            "palette keypoints {zarr_path} --registry {registry} --apply",
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
        if stage.deprecated:
            continue
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


def _stage_spec(stage_id: str) -> StageSpec:
    for spec in STAGE_SPECS:
        if spec.id == stage_id:
            return spec
    raise KeyError(stage_id)


def _stage_map(stages: Sequence[StageState]) -> dict[str, StageState]:
    return {stage.stage: stage for stage in stages}


def _recording_dir_for_dataset(dataset: DatasetRef) -> Path:
    if dataset.zarr_path.parent.name == "zarr":
        return dataset.zarr_path.parent.parent
    return dataset.zarr_path.parent


def _registry_for_dataset(dataset: DatasetRef, requested: Path | None) -> Path | None:
    if requested is not None:
        return requested.expanduser().resolve()
    return dataset.registry_path


def _result_to_dict(result: Any) -> dict[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
            if isinstance(converted, Mapping):
                return dict(converted)
        except Exception:
            pass
    if isinstance(result, Mapping):
        return dict(result)
    return {
        key: getattr(result, key)
        for key in dir(result)
        if not key.startswith("_") and not callable(getattr(result, key, None))
    }


def _join_command(parts: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if part is not None and str(part) != "")


def _append_option(parts: list[Any], flag: str, value: Any, *, multi: bool = False) -> None:
    if value is None or value is False:
        return
    if value is True:
        parts.append(flag)
        return
    parts.append(flag)
    if multi and isinstance(value, (list, tuple)):
        parts.extend(value)
    else:
        parts.append(value)


def _resolved_palette_command(verb: str, dataset: DatasetRef, args: argparse.Namespace, *, apply: bool) -> str:
    parts: list[Any] = ["palette", verb, dataset.zarr_path]
    registry = _registry_for_dataset(dataset, getattr(args, "registry", None))
    if registry is not None:
        parts.extend(["--registry", registry])
    if apply:
        parts.append("--apply")
    else:
        parts.append("--dry-run")
    if bool(getattr(args, "force", False)):
        parts.append("--force")
    option_names = {
        "detect": (
            ("--video", "video", False),
            ("--output", "output", False),
            ("--set-id", "set_id", False),
            ("--require-unique", "require_unique", False),
            ("--top-k", "top_k", False),
            ("--include-non-success", "include_non_success", False),
            ("--config", "config", False),
            ("--conf", "conf", False),
            ("--iou", "iou", False),
            ("--max-det", "max_det", False),
            ("--batch-size", "batch_size", False),
            ("--resize-dims", "resize_dims", True),
            ("--imgsz", "imgsz", True),
            ("--decode-backend", "decode_backend", False),
            ("--cpu", "cpu", False),
            ("--write-raw-video-metadata", "write_raw_video_metadata", False),
            ("--overwrite-raw-video-metadata", "overwrite_raw_video_metadata", False),
        ),
        "crop": (
            ("--config", "config", False),
            ("--force-new", "force_new", False),
            ("--crop-storage-mode", "crop_storage_mode", False),
            ("--source-type", "source_type", False),
            ("--source-path", "source_path", False),
            ("--selection-policy", "selection_policy", False),
            ("--scheduler", "scheduler", False),
            ("--num-workers", "num_workers", False),
            ("--acceleration", "acceleration", False),
            ("--external-write-backend", "external_write_backend", False),
            ("--external-roi-storage", "external_roi_storage", False),
            ("--external-use-sharding", "external_use_sharding", False),
            ("--no-external-use-sharding", "no_external_use_sharding", False),
            ("--external-roi-chunk-size", "external_roi_chunk_size", False),
            ("--external-roi-shard-size", "external_roi_shard_size", False),
            ("--external-gpu-chunk-frames", "external_gpu_chunk_frames", False),
            ("--require-kvikio", "require_kvikio", False),
            ("--no-gpu", "no_gpu", False),
            ("--force-cpu", "force_cpu", False),
            ("--verbose", "verbose", False),
        ),
        "keypoints": (
            ("--output", "output", False),
            ("--set-id", "set_id", False),
            ("--require-unique", "require_unique", False),
            ("--top-k", "top_k", False),
            ("--include-non-success", "include_non_success", False),
            ("--run-name", "run_name", False),
            ("--crop-run", "crop_run", False),
            ("--pose-schema", "pose_schema", False),
            ("--batch-size", "batch_size", False),
            ("--device", "device", False),
            ("--imgsz", "imgsz", False),
            ("--conf", "conf", False),
            ("--iou", "iou", False),
            ("--max-det", "max_det", False),
            ("--mask-threshold", "mask_threshold", False),
            ("--roi-cache-policy", "roi_cache_policy", False),
            ("--roi-cache-dir", "roi_cache_dir", False),
            ("--roi-cache-manifest", "roi_cache_manifest", False),
            ("--stage-roi-cache-to-scratch", "stage_roi_cache_to_scratch", False),
            ("--roi-cache-staging-dir", "roi_cache_staging_dir", False),
            ("--profile-timings", "profile_timings", False),
            ("--progress-jsonl", "progress_jsonl", False),
            ("--progress-every-batches", "progress_every_batches", False),
            ("--input-mode", "input_mode", False),
            ("--cpu", "cpu", False),
            ("--verbose", "verbose", False),
        ),
    }
    for flag, name, multi in option_names.get(verb, ()):
        if hasattr(args, name):
            _append_option(parts, flag, getattr(args, name), multi=multi)
    return _join_command(parts)


def _arg_params(args: argparse.Namespace, exclude: Iterable[str] = ()) -> dict[str, Any]:
    excluded = {"command", "recording", "json", "apply", "dry_run", *set(exclude)}
    return {key: value for key, value in vars(args).items() if key not in excluded}


def _input_run_ids(stages: Sequence[StageState], deps: Sequence[str]) -> dict[str, Any]:
    by_stage = _stage_map(stages)
    return {
        dep: by_stage[dep].run
        for dep in deps
        if dep in by_stage and by_stage[dep].run is not None
    }


def _precondition_payload(
    *,
    verb: str,
    stage_id: str,
    dataset: DatasetRef,
    stages: Sequence[StageState],
    provenance: dict[str, Any],
    force: bool = False,
) -> dict[str, Any] | None:
    spec = _stage_spec(stage_id)
    by_stage = _stage_map(stages)
    missing = [dep for dep in spec.depends_on if not by_stage.get(dep, StageState(dep, "missing")).complete]
    if not missing:
        return None
    hints = []
    for dep in missing:
        action = _action_for_stage(dep, dataset).get("action")
        if action:
            hints.append(str(action))
    reason_code = "BLOCKED_BY_" + "_AND_".join(dep.upper() for dep in missing)
    if force:
        override = {
            "stage": stage_id,
            "blocked_by": missing,
            "reason_code": reason_code,
            "warning": "Catalog dependency gate explicitly overridden by --force.",
        }
        overrides = provenance.setdefault("forced_dependency_overrides", [])
        if isinstance(overrides, list):
            overrides.append(override)
        else:
            provenance["forced_dependency_overrides"] = [override]
        return None
    return build_envelope(
        command=f"palette {verb}",
        status="blocked",
        reason_code=reason_code,
        recording=dataset.recording_id,
        dataset_id=dataset.dataset_id,
        zarr_path=dataset.zarr_path,
        artifacts=[dataset.zarr_path],
        metrics={"missing_dependency_count": len(missing)},
        next_hints=hints,
        provenance=provenance,
        blocked_by=missing,
    )


def _forced_envelope_fields(provenance: Mapping[str, Any]) -> dict[str, Any]:
    overrides = provenance.get("forced_dependency_overrides")
    if not isinstance(overrides, list) or not overrides:
        return {}
    blocked_by: list[str] = []
    for override in overrides:
        if not isinstance(override, Mapping):
            continue
        values = override.get("blocked_by")
        if isinstance(values, list):
            blocked_by.extend(str(value) for value in values)
    return {
        "forced": True,
        "blocked_by": sorted(set(blocked_by)),
    }


def _next_hints_from_plan(dataset: DatasetRef) -> list[str]:
    try:
        root = open_zarr_group_direct(dataset.zarr_path, mode="r")
        payload = build_plan_payload(dataset, inspect_stages(root))
    except Exception:
        return []
    hints: list[str] = []
    for item in payload.get("next", []):
        if isinstance(item, Mapping) and item.get("action"):
            hints.append(str(item["action"]))
    return hints


def _run_artifacts(zarr_path: Path, parent: str, run: str | None) -> list[str]:
    artifacts = [str(zarr_path)]
    if run:
        artifacts.append(str(zarr_path / parent / run))
    return artifacts


def _print_run_table(payload: Mapping[str, Any]) -> None:
    print(f"command: {payload.get('command')}")
    print(f"status: {payload.get('status')}")
    print(f"reason_code: {payload.get('reason_code')}")
    print(f"recording: {payload.get('recording') or '(unknown)'}")
    print(f"zarr_path: {payload.get('zarr_path')}")
    if payload.get("run"):
        print(f"run: {payload.get('run')}")
    if payload.get("resolved_command"):
        print(f"resolved_command: {payload.get('resolved_command')}")
    hints = payload.get("next_hints") or []
    if hints:
        print("next_hints:")
        for hint in hints:
            print(f"  - {hint}")


def _emit_run_payload(payload: dict[str, Any], json_output: bool) -> int:
    if json_output:
        _print_json(payload)
    else:
        _print_run_table(payload)
    return exit_code_for_status(str(payload.get("status") or "failed"))


def _runner_stdout_context(json_output: bool):
    return redirect_stdout(sys.stderr) if json_output else nullcontext()


def _run_detect(args: argparse.Namespace, dataset: DatasetRef, stages: Sequence[StageState]) -> dict[str, Any]:
    with _runner_stdout_context(bool(args.json)):
        from fisheye.utils.run_detect_with_registry_model import run_detect_with_registry_model

    apply = bool(args.apply)
    resolved_command = _resolved_palette_command("detect", dataset, args, apply=apply)
    spec = _stage_spec("detect")
    params = {
        **_arg_params(args),
        "recording_dir": _recording_dir_for_dataset(dataset),
        "output": args.output or dataset.zarr_path,
        "dry_run": not apply,
    }
    provenance = build_run_provenance(
        command="palette detect",
        params=params,
        input_run_ids=_input_run_ids(stages, spec.depends_on),
        cwd=Path.cwd(),
    )
    blocked = _precondition_payload(
        verb="detect",
        stage_id="detect",
        dataset=dataset,
        stages=stages,
        provenance=provenance,
        force=bool(args.force),
    )
    if blocked is not None:
        blocked["resolved_command"] = resolved_command
        return blocked

    with _runner_stdout_context(bool(args.json)):
        result = run_detect_with_registry_model(
            recording_dir=_recording_dir_for_dataset(dataset),
            video=args.video,
            output=args.output or dataset.zarr_path,
            registry=_registry_for_dataset(dataset, args.registry),
            set_id=args.set_id,
            require_unique=bool(args.require_unique),
            top_k=int(args.top_k),
            include_non_success=bool(args.include_non_success),
            dry_run=not apply,
            config=args.config,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            resize_dims=args.resize_dims,
            imgsz=args.imgsz,
            decode_backend=args.decode_backend,
            cpu=bool(args.cpu),
            write_raw_video_metadata=bool(args.write_raw_video_metadata),
            overwrite_raw_video_metadata=bool(args.overwrite_raw_video_metadata),
            argv=[],
            cli_provenance=provenance if apply else None,
        )
    payload = _result_to_dict(result)
    ok = bool(payload.get("ok"))
    status = "dry_run" if ok and not apply else ("ok" if ok else "failed")
    reason = "DRY_RUN" if status == "dry_run" else ("OK" if ok else str(payload.get("reason") or "RUNNER_FAILED").upper())
    run = payload.get("detect_run") if isinstance(payload.get("detect_run"), str) else None
    hints = [_resolved_palette_command("detect", dataset, args, apply=True)] if status == "dry_run" else _next_hints_from_plan(dataset)
    return build_envelope(
        command="palette detect",
        status=status,
        reason_code=reason,
        recording=dataset.recording_id,
        dataset_id=dataset.dataset_id,
        zarr_path=dataset.zarr_path,
        run=run,
        artifacts=_run_artifacts(dataset.zarr_path, "detect_runs", run),
        metrics={
            "selected_model_path": payload.get("selected_model_path"),
            "selected_run_id": payload.get("selected_run_id"),
            "selected_set_id": payload.get("selected_set_id"),
        },
        next_hints=hints,
        provenance=provenance,
        resolved_command=resolved_command,
        runner_result=payload,
        **_forced_envelope_fields(provenance),
    )


def _run_keypoints(args: argparse.Namespace, dataset: DatasetRef, stages: Sequence[StageState]) -> dict[str, Any]:
    with _runner_stdout_context(bool(args.json)):
        from fisheye.utils.run_keypoints_with_registry_model import run_keypoints_with_registry_model

    apply = bool(args.apply)
    resolved_command = _resolved_palette_command("keypoints", dataset, args, apply=apply)
    spec = _stage_spec("keypoints")
    params = {
        **_arg_params(args),
        "recording_dir": _recording_dir_for_dataset(dataset),
        "output": args.output or dataset.zarr_path,
        "dry_run": not apply,
    }
    provenance = build_run_provenance(
        command="palette keypoints",
        params=params,
        input_run_ids=_input_run_ids(stages, spec.depends_on),
        cwd=Path.cwd(),
    )
    blocked = _precondition_payload(
        verb="keypoints",
        stage_id="keypoints",
        dataset=dataset,
        stages=stages,
        provenance=provenance,
        force=bool(args.force),
    )
    if blocked is not None:
        blocked["resolved_command"] = resolved_command
        return blocked

    with _runner_stdout_context(bool(args.json)):
        result = run_keypoints_with_registry_model(
            recording_dir=_recording_dir_for_dataset(dataset),
            output=args.output or dataset.zarr_path,
            registry=_registry_for_dataset(dataset, args.registry),
            set_id=args.set_id,
            require_unique=bool(args.require_unique),
            top_k=int(args.top_k),
            include_non_success=bool(args.include_non_success),
            dry_run=not apply,
            run_name=args.run_name,
            crop_run=args.crop_run,
            pose_schema=args.pose_schema,
            batch_size=int(args.batch_size),
            device=args.device,
            imgsz=args.imgsz,
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            mask_threshold=float(args.mask_threshold),
            roi_cache_policy=args.roi_cache_policy,
            roi_cache_dir=args.roi_cache_dir,
            roi_cache_manifest=args.roi_cache_manifest,
            stage_roi_cache_to_scratch=bool(args.stage_roi_cache_to_scratch),
            roi_cache_staging_dir=args.roi_cache_staging_dir,
            profile_timings=bool(args.profile_timings),
            progress_jsonl=args.progress_jsonl,
            progress_every_batches=int(args.progress_every_batches),
            input_mode=args.input_mode,
            cpu=bool(args.cpu),
            verbose=bool(args.verbose),
            argv=[],
            cli_provenance=provenance if apply else None,
        )
    payload = _result_to_dict(result)
    ok = bool(payload.get("ok"))
    status = "dry_run" if ok and not apply else ("ok" if ok else "failed")
    reason = "DRY_RUN" if status == "dry_run" else ("OK" if ok else str(payload.get("reason") or "RUNNER_FAILED").upper())
    run = payload.get("keypoint_run") if isinstance(payload.get("keypoint_run"), str) else None
    hints = [_resolved_palette_command("keypoints", dataset, args, apply=True)] if status == "dry_run" else _next_hints_from_plan(dataset)
    return build_envelope(
        command="palette keypoints",
        status=status,
        reason_code=reason,
        recording=dataset.recording_id,
        dataset_id=dataset.dataset_id,
        zarr_path=dataset.zarr_path,
        run=run,
        artifacts=_run_artifacts(dataset.zarr_path, "keypoints_runs", run),
        metrics={
            "selected_model_path": payload.get("selected_model_path"),
            "selected_run_id": payload.get("selected_run_id"),
            "selected_set_id": payload.get("selected_set_id"),
        },
        next_hints=hints,
        provenance=provenance,
        resolved_command=resolved_command,
        runner_result=payload,
        **_forced_envelope_fields(provenance),
    )


def _run_crop(args: argparse.Namespace, dataset: DatasetRef, stages: Sequence[StageState]) -> dict[str, Any]:
    with _runner_stdout_context(bool(args.json)):
        from fisheye.tracking.crop import crop_detections
        from fisheye.utils.crop_batch import _build_plan, _load_config, _normalize_path

    apply = bool(args.apply)
    resolved_command = _resolved_palette_command("crop", dataset, args, apply=apply)
    spec = _stage_spec("crop")
    config = _load_config(args.config)
    source_type = args.source_type or "auto"
    source_path = _normalize_path(args.source_path)
    params = {
        **_arg_params(args),
        "zarr_path": dataset.zarr_path,
        "source_type": source_type,
        "source_path": source_path,
        "dry_run": not apply,
    }
    provenance = build_run_provenance(
        command="palette crop",
        params=params,
        input_run_ids=_input_run_ids(stages, spec.depends_on),
        cwd=Path.cwd(),
    )
    blocked = _precondition_payload(
        verb="crop",
        stage_id="crop",
        dataset=dataset,
        stages=stages,
        provenance=provenance,
        force=bool(args.force),
    )
    if blocked is not None:
        blocked["resolved_command"] = resolved_command
        return blocked

    plan = _build_plan(
        dataset.zarr_path,
        config,
        source_type,
        source_path,
        args.selection_policy,
        bool(args.force_new),
        args.crop_storage_mode,
    )
    if plan.status in {"missing", "invalid"}:
        return build_envelope(
            command="palette crop",
            status="blocked",
            reason_code=f"CROP_PLAN_{plan.status.upper()}",
            recording=dataset.recording_id,
            dataset_id=dataset.dataset_id,
            zarr_path=dataset.zarr_path,
            artifacts=[dataset.zarr_path],
            metrics={"crop_plan_status": plan.status, "crop_plan_reason": plan.reason},
            next_hints=[_action_for_stage("refined_detect", dataset).get("action") or ""],
            provenance=provenance,
            resolved_command=resolved_command,
            crop_plan=json_ready(plan),
            **_forced_envelope_fields(provenance),
        )
    if not apply:
        return build_envelope(
            command="palette crop",
            status="dry_run",
            reason_code="DRY_RUN",
            recording=dataset.recording_id,
            dataset_id=dataset.dataset_id,
            zarr_path=dataset.zarr_path,
            artifacts=[dataset.zarr_path],
            metrics={
                "crop_plan_status": plan.status,
                "crop_plan_reason": plan.reason,
                "source_type": plan.source_type,
                "source_path": plan.source_path,
                "roi_size": plan.roi_size,
                "crop_storage_mode": plan.crop_storage_mode,
            },
            next_hints=[_resolved_palette_command("crop", dataset, args, apply=True)],
            provenance=provenance,
            resolved_command=resolved_command,
            crop_plan=json_ready(plan),
            **_forced_envelope_fields(provenance),
        )

    external_use_sharding = None
    if bool(args.external_use_sharding):
        external_use_sharding = True
    if bool(args.no_external_use_sharding):
        external_use_sharding = False
    with _runner_stdout_context(bool(args.json)):
        result = crop_detections(
            zarr_path=str(dataset.zarr_path),
            config=config,
            source_type=plan.source_type or source_type,
            source_path=plan.source_path,
            selection_policy=plan.selection_policy,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            console=None,
            acceleration=args.acceleration,
            external_write_backend=args.external_write_backend,
            external_roi_storage=args.external_roi_storage,
            external_use_sharding=external_use_sharding,
            external_roi_chunk_size=args.external_roi_chunk_size,
            external_roi_shard_size=args.external_roi_shard_size,
            external_gpu_chunk_frames=args.external_gpu_chunk_frames,
            external_require_kvikio=args.require_kvikio,
            crop_storage_mode=plan.crop_storage_mode,
            use_gpu_allowed=not bool(args.no_gpu),
            force_cpu=bool(args.force_cpu),
            verbose=bool(args.verbose),
            cli_provenance=provenance,
        )
    payload = _result_to_dict(result)
    run = payload.get("run_name") if isinstance(payload.get("run_name"), str) else None
    return build_envelope(
        command="palette crop",
        status="ok",
        reason_code="OK",
        recording=dataset.recording_id,
        dataset_id=dataset.dataset_id,
        zarr_path=dataset.zarr_path,
        run=run,
        artifacts=_run_artifacts(dataset.zarr_path, "crop_runs", run),
        metrics={
            "total_crops": payload.get("total_crops"),
            "frames_with_crops": payload.get("frames_with_crops"),
            "percent_cropped": payload.get("percent_cropped"),
            "duration_seconds": payload.get("duration_seconds"),
            "detection_source_type": payload.get("detection_source_type"),
            "detection_source_path": payload.get("detection_source_path"),
            "crop_storage_mode": payload.get("crop_storage_mode"),
        },
        next_hints=_next_hints_from_plan(dataset),
        provenance=provenance,
        resolved_command=resolved_command,
        runner_result=payload,
        **_forced_envelope_fields(provenance),
    )


def _run_mutating_verb(args: argparse.Namespace) -> int:
    dataset, hints = _resolve_dataset(str(args.recording), args.registry)
    if dataset is None:
        payload = _blocked_payload(f"palette {args.command}", str(args.recording), hints)
        payload["status"] = "blocked"
        if bool(getattr(args, "json", False)):
            _print_json(payload)
        else:
            for hint in hints:
                print(hint, file=sys.stderr)
        return EXIT_BLOCKED
    root = open_zarr_group_direct(dataset.zarr_path, mode="r")
    stages = inspect_stages(root)
    if args.command == "detect":
        payload = _run_detect(args, dataset, stages)
    elif args.command == "crop":
        payload = _run_crop(args, dataset, stages)
    elif args.command == "keypoints":
        payload = _run_keypoints(args, dataset, stages)
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"unknown run verb: {args.command}")
    return _emit_run_payload(payload, bool(args.json))


def _add_common_run_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("recording", help="Recording id, dataset id, zarr path, or directory containing one zarr.")
    sub.add_argument("--registry", type=Path, help="Path to palette_registry.sqlite.")
    sub.add_argument("--json", action="store_true", help="Print a JSON envelope.")
    mode = sub.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Execute the live runner.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve and report without writing. This is the default.")
    sub.add_argument(
        "--force",
        action="store_true",
        help="Override unmet catalog dependencies with loud provenance in the run envelope.",
    )


def _add_detect_args(sub: argparse.ArgumentParser) -> None:
    _add_common_run_args(sub)
    sub.add_argument("--video", type=Path, help="Optional explicit camera video path.")
    sub.add_argument("--output", type=Path, help="Optional explicit output zarr path.")
    sub.add_argument("--set-id", type=str, help="Optional model set filter.")
    sub.add_argument("--require-unique", action="store_true", help="Fail if top model scores tie.")
    sub.add_argument("--top-k", type=int, default=5, help="Number of model candidates to record.")
    sub.add_argument("--include-non-success", action="store_true", help="Include non-success training runs.")
    sub.add_argument("--config", type=str, default=None, help="Optional detect_yolo config path.")
    sub.add_argument("--conf", type=float, default=None, help="Optional confidence threshold override.")
    sub.add_argument("--iou", type=float, default=None, help="Optional IoU threshold override.")
    sub.add_argument("--max-det", type=int, default=None, help="Optional max detections override.")
    sub.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    sub.add_argument("--resize-dims", nargs="+", type=int, default=None, help="Inference size override.")
    sub.add_argument("--imgsz", nargs="+", type=int, default=None, help="YOLO inference size alias.")
    sub.add_argument("--decode-backend", type=str, default=None, help="Video decode backend passed through to detect.")
    sub.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    sub.add_argument("--write-raw-video-metadata", action="store_true", help="Write metadata-only raw_video attrs.")
    sub.add_argument("--overwrite-raw-video-metadata", action="store_true", help="Overwrite existing raw_video metadata attrs.")


def _add_crop_args(sub: argparse.ArgumentParser) -> None:
    _add_common_run_args(sub)
    sub.add_argument("--config", type=Path, default=None, help="Optional crop config YAML.")
    sub.add_argument("--force-new", action="store_true", help="Always create a new crop run on apply.")
    sub.add_argument("--crop-storage-mode", choices=["materialized", "geometry_only"], default=None)
    sub.add_argument("--source-type", choices=["auto", "refined", "detect", "manual", "filtered", "interpolated"], default=None)
    sub.add_argument("--source-path", type=str, default=None)
    sub.add_argument("--selection-policy", choices=["training", "full_recording"], default=None)
    sub.add_argument("--scheduler", choices=["processes", "threads", "distributed"], default=None)
    sub.add_argument("--num-workers", type=int, default=None)
    sub.add_argument("--acceleration", choices=["auto", "gpu", "cpu"], default=None)
    sub.add_argument("--external-write-backend", choices=["standard", "kvikio"], default=None)
    sub.add_argument("--external-roi-storage", choices=["compressed", "uncompressed"], default=None)
    sub.add_argument("--external-use-sharding", action="store_true")
    sub.add_argument("--no-external-use-sharding", action="store_true")
    sub.add_argument("--external-roi-chunk-size", type=int, default=None)
    sub.add_argument("--external-roi-shard-size", type=int, default=None)
    sub.add_argument("--external-gpu-chunk-frames", type=int, default=None)
    sub.add_argument("--require-kvikio", action="store_true")
    sub.add_argument("--no-gpu", action="store_true")
    sub.add_argument("--force-cpu", action="store_true")
    sub.add_argument("--verbose", action="store_true")


def _add_keypoints_args(sub: argparse.ArgumentParser) -> None:
    _add_common_run_args(sub)
    sub.add_argument("--output", type=Path, help="Optional explicit output zarr path.")
    sub.add_argument("--set-id", type=str, help="Optional model set filter.")
    sub.add_argument("--require-unique", action="store_true", help="Fail if top model scores tie.")
    sub.add_argument("--top-k", type=int, default=5, help="Number of model candidates to record.")
    sub.add_argument("--include-non-success", action="store_true", help="Include non-success training runs.")
    sub.add_argument("--run-name", type=str, default=None, help="Optional explicit keypoints run name.")
    sub.add_argument("--crop-run", type=str, default=None, help="Optional explicit crop run name.")
    sub.add_argument("--pose-schema", type=str, default="traditional_v1")
    sub.add_argument("--batch-size", type=int, default=256)
    sub.add_argument("--device", type=str, default=None)
    sub.add_argument("--imgsz", type=int, default=None)
    sub.add_argument("--conf", type=float, default=0.25)
    sub.add_argument("--iou", type=float, default=0.5)
    sub.add_argument("--max-det", type=int, default=1)
    sub.add_argument("--mask-threshold", type=float, default=0.5)
    sub.add_argument("--roi-cache-policy", choices=["auto", "prefer", "require", "off"], default="auto")
    sub.add_argument("--roi-cache-dir", type=Path, default=None)
    sub.add_argument("--roi-cache-manifest", type=Path, default=None)
    sub.add_argument("--stage-roi-cache-to-scratch", action="store_true")
    sub.add_argument("--roi-cache-staging-dir", type=Path, default=None)
    sub.add_argument("--profile-timings", action="store_true")
    sub.add_argument("--progress-jsonl", type=Path, default=None)
    sub.add_argument("--progress-every-batches", type=int, default=1)
    sub.add_argument("--input-mode", choices=["numpy-list", "torch-batch"], default="numpy-list")
    sub.add_argument("--cpu", action="store_true")
    sub.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = PaletteArgumentParser(
        prog="palette",
        description="Palette workflow oracle and narrow-waist run command.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "plan"):
        sub = subparsers.add_parser(name, help=f"Read-only workflow {name}.")
        sub.add_argument("recording", help="Recording id, dataset id, zarr path, or directory containing one zarr.")
        sub.add_argument("--registry", type=Path, help="Path to palette_registry.sqlite.")
        sub.add_argument("--json", action="store_true", help="Print a JSON envelope.")
    _add_detect_args(subparsers.add_parser("detect", help="Resolve and run detection through the registry model shim."))
    _add_crop_args(subparsers.add_parser("crop", help="Plan or run crop extraction through the live crop runner."))
    _add_keypoints_args(subparsers.add_parser("keypoints", help="Resolve and run keypoints through the registry model shim."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except PaletteUsageError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    try:
        if args.command in {"detect", "crop", "keypoints"}:
            return _run_mutating_verb(args)
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
