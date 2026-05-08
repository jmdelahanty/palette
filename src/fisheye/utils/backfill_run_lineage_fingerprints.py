"""Backfill run-level lineage fingerprints on existing analysis runs.

Default mode is dry-run. Use ``--apply`` to write attrs. Backfilled
fingerprints are marked ``best_effort`` because current v1 runs do not all
expose complete refined authoring revisions or source content hashes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fisheye.shared.run_lineage_fingerprint import (
    LINEAGE_ATTR_NAMES,
    build_run_lineage_payload,
    compute_run_lineage_hash,
    write_run_lineage_attrs,
)
from fisheye.utils.zarr_io import open_zarr_root


@dataclass(frozen=True)
class RunParentSpec:
    family: str
    parent_path: tuple[str, ...]


@dataclass(frozen=True)
class RunLineageBackfillResult:
    zarr_path: Path
    run_family: str
    run_id: str
    run_path: str
    action: str
    fingerprint: str | None = None
    message: str = ""


RUN_PARENT_SPECS = (
    RunParentSpec("track_kinematics_run", ("analysis", "track_kinematics_runs", "offline")),
    RunParentSpec("swim_bout_run", ("analysis", "swim_bout_runs")),
    RunParentSpec("bout_kinematics_run", ("analysis", "bout_kinematics_runs")),
    RunParentSpec("stimulus_response_run", ("analysis", "stimulus_response_runs")),
    RunParentSpec("eye_angle_run", ("analysis", "eye_angle_runs")),
    RunParentSpec("subject_shape_run", ("analysis", "subject_shape_runs")),
    RunParentSpec("tail_kinematics_run", ("analysis", "tail_kinematics_runs")),
    RunParentSpec("tail_posture_view_run", ("analysis", "tail_posture_view_runs")),
    RunParentSpec("bout_classification_run", ("analysis", "bout_classification_runs")),
)

TRANSIENT_LINEAGE_KEYS = {
    "artifacts",
    "command",
    "created_at_utc",
    "created_utc",
    "environment",
    "git_branch",
    "host",
    "hostname",
    "latest",
    "operator",
    "output",
    "output_path",
    "platform",
    "run_name",
    "run_path",
    "scheduler",
    "timestamp_utc",
    "wall_time_s",
    "wall_time_seconds",
    "zarr_path",
}

TOP_LEVEL_PARAMETER_KEYS = {
    "boundary_frame_rounding_policy",
    "boundary_mode",
    "boundary_window_s",
    "default_heading_level",
    "default_level",
    "detection_method",
    "duration_frame_rounding_policy",
    "effective_min_bout_duration_s",
    "effective_min_gap_duration_s",
    "exponential_source_level",
    "exponential_tau_s",
    "fps",
    "gap_merge_min_gap_duration_s",
    "gap_merge_min_gap_source",
    "gap_merge_policy",
    "gap_merge_policy_active",
    "method",
    "min_bout_duration_s",
    "min_gap_duration_s",
    "min_gap_frames",
    "moving_threshold_mm_s",
    "physical_active_boundary_constraint",
    "physical_active_boundary_margin_s",
    "physical_active_boundary_policy",
    "physical_active_signal_array",
    "physical_active_signal_level",
    "physical_active_threshold_mm_s",
    "pixel_to_mm",
    "projection_deadzone",
    "projection_speed_deadzone_mm_s",
    "radial_singularity_epsilon_mm",
    "resolved_boundary_window_frames",
    "resolved_min_bout_frames",
    "resolved_min_gap_frames",
    "shape_split_policy",
    "source_track_id",
    "stimulus_radial_polarity",
    "stimulus_radial_polarity_source",
    "stimulus_radial_sign",
    "threshold_crossing_interpolation",
    "track_id",
    "window_lengths_s",
}

TOP_LEVEL_PARAMETER_PREFIXES = (
    "early_",
    "hysteresis_",
    "min_peak_",
    "peak_",
    "smooth",
    "smoothing_",
    "threshold_",
)


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _coerce_mapping(value: Any) -> dict[str, Any]:
    value = _parse_jsonish(value)
    return dict(value) if isinstance(value, Mapping) else {}


def _attrs_dict(group: Any) -> dict[str, Any]:
    attrs = getattr(group, "attrs", {})
    try:
        return {str(key): _parse_jsonish(value) for key, value in attrs.items()}
    except AttributeError:
        return {}


def _strip_transient(value: Any) -> Any:
    value = _parse_jsonish(value)
    if isinstance(value, Mapping):
        return {
            str(key): _strip_transient(item)
            for key, item in value.items()
            if str(key) not in TRANSIENT_LINEAGE_KEYS
        }
    if isinstance(value, list):
        return [_strip_transient(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_transient(item) for item in value]
    return value


def _get_nested_group(root: Any, path: Sequence[str]) -> Any | None:
    group = root
    for part in path:
        try:
            if part not in group:
                return None
            group = group[part]
        except Exception:
            return None
    return group


def _group_names(group: Any) -> list[str]:
    try:
        return sorted(str(name) for name in group.group_keys())
    except Exception:
        try:
            return sorted(str(name) for name in group.keys())
        except Exception:
            return []


def _is_parameter_attr(key: str) -> bool:
    return key in TOP_LEVEL_PARAMETER_KEYS or key.startswith(TOP_LEVEL_PARAMETER_PREFIXES)


def _source_ref_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    refs = _coerce_mapping(attrs.get("source_refs"))
    for key, value in attrs.items():
        if not key.startswith("source_"):
            continue
        if key in {"source_fingerprint", "source_lineage_hash"}:
            continue
        refs[key] = value
    return _strip_transient(refs)


def _source_fingerprint_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if "fingerprint" not in key:
            continue
        if key in LINEAGE_ATTR_NAMES:
            continue
        out[key] = value
    refs = _coerce_mapping(attrs.get("source_refs"))
    for key, value in refs.items():
        if "fingerprint" in str(key):
            out[str(key)] = value
    return _strip_transient(out)


def _parameter_attrs(attrs: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _coerce_mapping(provenance.get("parameters"))
    parameters.update(_coerce_mapping(attrs.get("parameters")))
    for key, value in attrs.items():
        if _is_parameter_attr(key):
            parameters[key] = value
    return _strip_transient(parameters)


def _code_attrs(attrs: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    git = _coerce_mapping(provenance.get("git"))
    commit = git.get("commit") or git.get("commit_hash") or attrs.get("git_commit")
    dirty = git.get("is_dirty")
    if dirty is None:
        dirty = attrs.get("git_dirty") or attrs.get("git_is_dirty")
    code = {
        "git_commit": commit,
        "git_dirty": dirty,
    }
    return {key: value for key, value in code.items() if value is not None}


def build_payload_for_run(run_family: str, run_group: Any) -> dict[str, Any]:
    """Build a best-effort lineage payload from existing run attrs."""

    attrs = _attrs_dict(run_group)
    provenance = _coerce_mapping(attrs.get("provenance"))
    analysis_schema = {
        "schema_id": attrs.get("schema_id"),
        "schema_version": attrs.get("schema_version") or attrs.get("output_schema_version"),
        "row_axis": attrs.get("row_axis"),
    }
    method = attrs.get("method") or attrs.get("detection_method") or provenance.get("stage")
    method_version = attrs.get("method_version") or provenance.get("version")
    return build_run_lineage_payload(
        run_family=run_family,
        analysis_schema=analysis_schema,
        method=None if method is None else str(method),
        method_version=None if method_version is None else str(method_version),
        source_refs=_source_ref_attrs(attrs),
        source_fingerprints=_source_fingerprint_attrs(attrs),
        parameters=_parameter_attrs(attrs, provenance),
        code=_code_attrs(attrs, provenance),
    )


def backfill_zarr_run_lineage_fingerprints(
    zarr_path: Path,
    *,
    apply: bool = False,
    overwrite: bool = False,
    run_families: set[str] | None = None,
) -> list[RunLineageBackfillResult]:
    """Backfill lineage attrs for known derived-analysis run families."""

    root = open_zarr_root(zarr_path, mode="r+" if apply else "r")
    results: list[RunLineageBackfillResult] = []
    for spec in RUN_PARENT_SPECS:
        if run_families is not None and spec.family not in run_families:
            continue
        parent = _get_nested_group(root, spec.parent_path)
        if parent is None:
            continue
        for run_id in _group_names(parent):
            try:
                run_group = parent[run_id]
            except Exception as exc:
                results.append(
                    RunLineageBackfillResult(
                        zarr_path=zarr_path,
                        run_family=spec.family,
                        run_id=run_id,
                        run_path="/".join((*spec.parent_path, run_id)),
                        action="error",
                        message=str(exc),
                    )
                )
                continue
            attrs = getattr(run_group, "attrs", {})
            existing = attrs.get("source_fingerprint") or attrs.get("source_lineage_hash")
            run_path = "/".join((*spec.parent_path, run_id))
            if existing and not overwrite:
                results.append(
                    RunLineageBackfillResult(
                        zarr_path=zarr_path,
                        run_family=spec.family,
                        run_id=run_id,
                        run_path=run_path,
                        action="skip_existing",
                        fingerprint=str(existing),
                    )
                )
                continue
            payload = build_payload_for_run(spec.family, run_group)
            fingerprint = compute_run_lineage_hash(payload)
            if apply:
                write_run_lineage_attrs(
                    run_group,
                    payload,
                    fingerprint_status="best_effort",
                    overwrite=overwrite,
                )
                action = "wrote" if not existing else "overwrote"
            else:
                action = "would_write" if not existing else "would_overwrite"
            results.append(
                RunLineageBackfillResult(
                    zarr_path=zarr_path,
                    run_family=spec.family,
                    run_id=run_id,
                    run_path=run_path,
                    action=action,
                    fingerprint=fingerprint,
                )
            )
    return results


def _iter_zarr_paths(paths: Sequence[Path], *, recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for raw in paths:
        path = raw.expanduser()
        if path.suffix == ".zarr" and path.is_dir():
            candidates = [path]
        elif recursive and path.is_dir():
            candidates = sorted(candidate for candidate in path.rglob("*.zarr") if candidate.is_dir())
        elif path.is_dir():
            candidates = sorted(path.glob("*.zarr")) + sorted(path.glob("*/zarr/*.zarr"))
        else:
            candidates = []
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill best-effort run-level lineage fingerprints on Palette analysis Zarrs.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr paths or directories containing Zarrs")
    parser.add_argument("--recursive", action="store_true", help="recursively discover *.zarr under directories")
    parser.add_argument("--apply", action="store_true", help="write attrs instead of dry-running")
    parser.add_argument("--overwrite", action="store_true", help="replace existing lineage fingerprint attrs")
    parser.add_argument(
        "--run-family",
        action="append",
        choices=sorted(spec.family for spec in RUN_PARENT_SPECS),
        help="limit to one run family; may be repeated",
    )
    parser.add_argument("--json", action="store_true", help="print JSONL results")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_families = set(args.run_family) if args.run_family else None
    all_results: list[RunLineageBackfillResult] = []
    for zarr_path in _iter_zarr_paths(args.paths, recursive=bool(args.recursive)):
        try:
            all_results.extend(
                backfill_zarr_run_lineage_fingerprints(
                    zarr_path,
                    apply=bool(args.apply),
                    overwrite=bool(args.overwrite),
                    run_families=run_families,
                )
            )
        except Exception as exc:
            all_results.append(
                RunLineageBackfillResult(
                    zarr_path=zarr_path,
                    run_family="",
                    run_id="",
                    run_path="",
                    action="error",
                    message=str(exc),
                )
            )
    if args.json:
        for result in all_results:
            print(json.dumps(result.__dict__, default=str, sort_keys=True))
    else:
        for result in all_results:
            prefix = f"{result.zarr_path} {result.run_path}".strip()
            message = f" ({result.message})" if result.message else ""
            print(f"{result.action}: {prefix}{message}")
        counts: dict[str, int] = {}
        for result in all_results:
            counts[result.action] = counts.get(result.action, 0) + 1
        print(f"summary: {json.dumps(counts, sort_keys=True)}")
    return 1 if any(result.action == "error" for result in all_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
