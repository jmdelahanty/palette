"""Materialize track kinematics on node-local storage and publish atomically.

The authoritative recording is opened read-only during computation.  The
legacy track writer produces a completed local run, that run is copied into
Zarr v3 indexed shards, and only then is it copied to a hidden sibling on the
shared filesystem and atomically renamed into the canonical run family.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import zarr

from ...analysis import track_kinematics as track_writer
from ...shared.json_safety import json_attr_safe
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import require_runs_parent
from ...shared.zarr_sharded_copy import copy_completed_run_to_sharded
from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group


MATERIALIZATION_SCHEMA_ID = "palette.track_kinematics_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.track_kinematics_run_publish.v1"
DEFAULT_OUTPUT_SHARD_ROWS = 262_144
MANAGED_WRITER_ARGUMENTS = {
    "--keypoint-run",
    "--no-write",
    "--offline-only",
    "--offline-run-name",
    "--online-only",
    "--output-zarr-path",
}


@dataclass(frozen=True)
class TrackKinematicsMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    sharded_run: Path
    keypoint_run: str
    run_name: str
    output_shard_rows: int
    shard_workers: int
    writer_arguments: tuple[str, ...]

    @property
    def local_run_path(self) -> Path:
        return (
            self.local_zarr
            / "analysis"
            / "track_kinematics_runs"
            / "offline"
            / self.run_name
        )

    @property
    def target_run_path(self) -> Path:
        return (
            self.source_zarr
            / "analysis"
            / "track_kinematics_runs"
            / "offline"
            / self.run_name
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "sharded_run": str(self.sharded_run),
            "target_run_path": str(self.target_run_path),
            "keypoint_run": self.keypoint_run,
            "run_name": self.run_name,
            "output_shard_rows": int(self.output_shard_rows),
            "shard_workers": int(self.shard_workers),
            "writer_arguments": list(self.writer_arguments),
        }


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe track-kinematics run name: {run_name!r}.")
    return value


def build_track_kinematics_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    keypoint_run: str,
    run_name: str,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    shard_workers: int = 1,
    writer_arguments: Sequence[str] = (),
) -> TrackKinematicsMaterializationPlan:
    """Build a read-only plan; no scratch or archive paths are created."""

    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Scratch root must not be inside the authoritative source Zarr.")
    name = _validate_run_name(run_name)
    keypoints = str(keypoint_run).strip()
    if not keypoints:
        raise ValueError("keypoint_run is required.")
    if int(output_shard_rows) <= 0 or int(shard_workers) <= 0:
        raise ValueError("output_shard_rows and shard_workers must be positive.")
    forwarded = tuple(str(value) for value in writer_arguments)
    forbidden = sorted(
        argument.split("=", 1)[0]
        for argument in forwarded
        if argument.split("=", 1)[0] in MANAGED_WRITER_ARGUMENTS
    )
    if forbidden:
        raise ValueError(
            "Track materializer owns these writer arguments: "
            + ", ".join(forbidden)
        )
    target = source / "analysis" / "track_kinematics_runs" / "offline" / name
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing authoritative run: {target}")
    return TrackKinematicsMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=scratch / "track-output.zarr",
        sharded_run=scratch / "track-run-sharded",
        keypoint_run=keypoints,
        run_name=name,
        output_shard_rows=int(output_shard_rows),
        shard_workers=int(shard_workers),
        writer_arguments=forwarded,
    )


def _iter_arrays(group: zarr.Group, prefix: str = ""):
    for name, array in group.arrays():
        yield f"{prefix}/{name}" if prefix else str(name), array
    for name, child in group.groups():
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_arrays(child, child_prefix)


def _validate_track_run(path: Path, *, require_sharded: bool) -> dict[str, Any]:
    errors: list[str] = []
    group = open_zarr_root(path, mode="r")
    if str(group.attrs.get("palette_run_completion_status")) != "complete":
        errors.append("run is not complete")
    if str(group.attrs.get("schema_id")) != "analysis.track_kinematics_runs":
        errors.append("missing or invalid track-kinematics schema_id")
    if int(group.attrs.get("schema_version", -1)) != 1:
        errors.append("missing or invalid track-kinematics schema_version")
    if str(group.attrs.get("method_version")) != "track_kinematics.v1":
        errors.append("missing or invalid track-kinematics method_version")
    if str(group.attrs.get("row_axis")) != "track_samples":
        errors.append("missing or invalid track-kinematics row_axis")
    if not isinstance(group.attrs.get("source_refs"), dict):
        errors.append("missing track-kinematics source_refs")
    if not isinstance(group.attrs.get("parameters"), dict):
        errors.append("missing track-kinematics parameters")
    tracks = group.get("tracks")
    if not isinstance(tracks, zarr.Group):
        errors.append("missing tracks group")
        track_names: list[str] = []
    else:
        track_names = sorted(str(name) for name in tracks.group_keys())
    if not track_names:
        errors.append("no track groups")
    required = (
        "frame_indices",
        "positions_px",
        "speed_raw_px",
        "speed_filtered_px",
        "speed_smoothed_px",
        "acceleration_px",
        "heading_degrees",
        "sample_valid",
        "delta_seconds",
    )
    track_rows: dict[str, int] = {}
    for name in track_names:
        track = tracks[name]
        frame_indices = track.get("frame_indices")
        if not isinstance(frame_indices, zarr.Array):
            errors.append(f"{name}: missing frame_indices")
            continue
        row_count = int(frame_indices.shape[0])
        track_rows[name] = row_count
        if int(track.attrs.get("num_samples", -1)) != row_count:
            errors.append(f"{name}: num_samples mismatch")
        for array_name in required:
            item = track.get(array_name)
            if not isinstance(item, zarr.Array):
                errors.append(f"{name}: missing {array_name}")
            elif int(item.shape[0]) != row_count:
                errors.append(f"{name}: row mismatch for {array_name}")

    array_count = 0
    sharded_count = 0
    for array_path, array in _iter_arrays(group):
        array_count += 1
        shards = getattr(array, "shards", None)
        if shards is not None:
            sharded_count += 1
            chunks = tuple(int(value) for value in array.chunks)
            outer = tuple(int(value) for value in shards)
            if any(outer[i] % chunks[i] for i in range(len(chunks))):
                errors.append(f"{array_path}: shard grid is not chunk aligned")
        elif require_sharded and int(array.ndim) >= 1:
            errors.append(f"{array_path}: expected indexed sharding")
    layout = group.attrs.get("physical_storage_layout")
    if require_sharded and not isinstance(layout, dict):
        errors.append("missing physical_storage_layout")
    return {
        "valid": not errors,
        "errors": errors,
        "track_rows": track_rows,
        "array_count": array_count,
        "sharded_array_count": sharded_count,
        "require_sharded": bool(require_sharded),
    }


def publish_track_kinematics_run(
    plan: TrackKinematicsMaterializationPlan,
    *,
    materialization_payload: dict[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    """Copy to a hidden sibling, validate, rename, then update pointers."""

    def validate(path: Path) -> dict[str, Any]:
        return _validate_track_run(path, require_sharded=True)

    def prepare(root: zarr.Group) -> tuple[zarr.Group, zarr.Group]:
        track_parent = require_runs_parent(
            root.require_group("analysis"),
            "track_kinematics_runs",
        )
        return track_parent, track_parent.require_group("offline")

    def complete(
        root: zarr.Group,
        _parent: zarr.Group,
        run: zarr.Group,
    ) -> None:
        track_writer.mark_track_kinematics_run_complete(
            root,
            run,
            run_name=plan.run_name,
            run_type="offline",
        )

    def verify(root: zarr.Group) -> None:
        pointer_parent = root["analysis/track_kinematics_runs"]
        pointer_offline = pointer_parent["offline"]
        if (
            str(pointer_parent.attrs.get("latest")) != f"offline/{plan.run_name}"
            or str(pointer_parent.attrs.get("latest_complete")) != f"offline/{plan.run_name}"
            or str(pointer_parent.attrs.get("latest_offline")) != plan.run_name
            or str(pointer_offline.attrs.get("latest")) != plan.run_name
        ):
            raise RuntimeError("Track-kinematics parent pointers were not updated consistently.")

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.sharded_run,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="track-kinematics-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy="node_local_output_sharded_atomic_run_group_publish",
            rollback_policy="remove_new_target_and_restore_both_parent_attr_sets",
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        payload_metadata={
            "local_run_path": str(plan.local_run_path),
            "sharded_run_path": str(plan.sharded_run),
            "copy_backend": copy_backend,
            "materialization": json_attr_safe(materialization_payload),
        },
    )


def materialize_track_kinematics(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    keypoint_run: str,
    run_name: str,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    shard_workers: int = 1,
    writer_arguments: Sequence[str] = (),
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = build_track_kinematics_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        keypoint_run=keypoint_run,
        run_name=run_name,
        output_shard_rows=output_shard_rows,
        shard_workers=shard_workers,
        writer_arguments=writer_arguments,
    )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.to_json(),
    }
    if not apply:
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        writer_argv = [
            str(plan.source_zarr),
            "--output-zarr-path",
            str(plan.local_zarr),
            "--offline-only",
            "--keypoint-run",
            plan.keypoint_run,
            "--offline-run-name",
            plan.run_name,
            *plan.writer_arguments,
        ]
        compute_started = time.perf_counter()
        track_writer.main(writer_argv)
        compute_seconds = float(time.perf_counter() - compute_started)
        regular_validation = _validate_track_run(plan.local_run_path, require_sharded=False)
        if not regular_validation["valid"]:
            raise RuntimeError(f"Local regular track run is invalid: {regular_validation}")
        sharded_copy = copy_completed_run_to_sharded(
            plan.local_run_path,
            plan.sharded_run,
            row_count_array=None,
            shard_rows=plan.output_shard_rows,
            workers=plan.shard_workers,
        )
        local_payload = {
            "source_access": "authoritative_zarr_read_only",
            "compute_output": "node_local_zarr",
            "compute_duration_seconds": compute_seconds,
            "writer_arguments": writer_argv,
            "regular_validation": regular_validation,
            "sharded_copy": sharded_copy,
        }
        sharded = open_zarr_root(plan.sharded_run, mode="a")
        sharded.attrs["node_local_materialization"] = json_attr_safe(local_payload)
        publish = publish_track_kinematics_run(
            plan,
            materialization_payload=local_payload,
            copy_backend=copy_backend,
        )
        result.update(
            {
                "status": "complete",
                "local_materialization": local_payload,
                "publish": publish,
            }
        )
        succeeded = True
        return result
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_track_kinematics_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_track_kinematics_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute track kinematics locally, shard it, and atomically publish."
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--keypoint-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output-shard-rows", type=int, default=DEFAULT_OUTPUT_SHARD_ROWS)
    parser.add_argument("--shard-workers", type=int, default=1)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] != "--":
        parser.error(
            "unrecognized materializer arguments; place track-writer arguments after --"
        )
    writer_arguments = tuple(remaining)
    if writer_arguments[:1] == ("--",):
        writer_arguments = writer_arguments[1:]
    result = materialize_track_kinematics(
        args.zarr_path,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        keypoint_run=args.keypoint_run,
        run_name=args.run_name,
        output_shard_rows=args.output_shard_rows,
        shard_workers=args.shard_workers,
        writer_arguments=writer_arguments,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
