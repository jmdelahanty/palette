"""Publish an immutable clipped view of one canonical refined-detection run.

The collection contains no copied detections.  Each member is an exact
half-open acquisition-frame slice of the same complete recording-level refined
run, preserving that run as the sole downstream detection authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.cluster.native_detection_authority import (
    recording_frame_work_unit_intervals,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
)


COLLECTION_SCHEMA = "palette.registered_geometry_clipped_refined_collection.v1"
SLICE_MODE = "canonical_recording_refined_slice_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_from_plan(plan_path: Path, target_id: str) -> Mapping[str, Any]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Clipped plan has no target_plans array.")
    matches = [
        value
        for value in targets
        if isinstance(value, Mapping) and value.get("target_id") == target_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one clipped plan target {target_id!r}.")
    return matches[0]


def _instances(group: Any) -> Any:
    return group["instances"] if "instances" in group else group


def _gate_consumption(
    refined: Any,
    *,
    requirement: str,
    expected_gate_run: str | None,
) -> dict[str, Any]:
    evidence = refined.attrs.get("registered_detection_gate")
    if not isinstance(evidence, Mapping):
        raise ValueError("Canonical refined run lacks registered gate evidence.")
    result = dict(evidence)
    if result.get("requirement") != requirement:
        raise ValueError("Refined gate requirement differs from the collection plan.")
    observed_run = str(result.get("gate_run") or "").strip() or None
    if observed_run != expected_gate_run:
        raise ValueError("Refined gate run differs from the exact planned gate run.")
    if requirement == "required" and (
        result.get("status") != "applied" or result.get("applied") is not True
    ):
        raise ValueError("Required registered gate was not validly consumed.")
    if requirement == "off" and result.get("status") != "off":
        raise ValueError("Off-mode refined run does not record an off gate state.")
    return result


def finalize_registered_clipped_refined_collection(
    *,
    analysis_zarr: Path,
    target: Mapping[str, Any],
    collection_id: str,
    refined_run: str,
    recording_frame_index: Path,
    gate_requirement: str,
    gate_run: str | None,
) -> dict[str, Any]:
    archive = analysis_zarr.expanduser().resolve()
    frame_index = recording_frame_index.expanduser().resolve()
    collection_name = str(collection_id).strip()
    refined_name = str(refined_run).strip()
    if not collection_name or "/" in collection_name:
        raise ValueError("collection_id must be one nonempty group name.")
    if not refined_name or "/" in refined_name:
        raise ValueError("refined_run must be one nonempty group name.")
    requirement = str(gate_requirement).strip()
    if requirement not in {"off", "if_available", "required"}:
        raise ValueError("Unsupported registered gate requirement.")
    expected_gate = str(gate_run or "").strip() or None
    if requirement == "required" and expected_gate is None:
        raise ValueError("Required geometry needs an exact gate run.")

    root = open_zarr_group_direct(archive, mode="a")
    refined = root[f"refined_detect_runs/{refined_name}"]
    if refined.attrs.get("status") not in {"complete", "completed"}:
        raise ValueError("Canonical refined run is not complete.")
    gate = _gate_consumption(
        refined,
        requirement=requirement,
        expected_gate_run=expected_gate,
    )
    table = _instances(refined)
    frame_indices = np.asarray(table["frame_indices"][:], dtype=np.int64).reshape(-1)
    keys = np.asarray(table["instance_key"][:], dtype=np.uint64).reshape(-1)
    if frame_indices.shape != keys.shape:
        raise ValueError("Canonical refined frame/key row counts differ.")
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise ValueError("Canonical refined instance_key is not unique.")

    authority = target.get("native_detection_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("Target lacks native recording authority.")
    n_frames = int(authority.get("n_frames") or 0)
    if n_frames <= 0 or np.any(frame_indices < 0) or np.any(frame_indices >= n_frames):
        raise ValueError("Canonical refined frames exceed the recording authority.")
    intervals = recording_frame_work_unit_intervals(frame_index, n_frames=n_frames)
    clips = target.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("Target has no clipped work units.")

    selected: list[dict[str, Any]] = []
    covered_rows = np.zeros(keys.shape[0], dtype=bool)
    for clip in sorted(clips, key=lambda value: int(value["clip_index"])):
        clip_index = int(clip["clip_index"])
        clip_id = str(clip["clip_id"])
        start, stop = intervals[(clip_index, clip_id)]
        row_mask = (frame_indices >= start) & (frame_indices < stop)
        if np.any(covered_rows & row_mask):
            raise ValueError("Canonical refined clip slices overlap detection rows.")
        covered_rows |= row_mask
        raw_source = clip.get("source")
        source = dict(raw_source) if isinstance(raw_source, Mapping) else {}
        source.setdefault("video_path", str(clip.get("video_path") or ""))
        source.setdefault("metadata_path", str(clip.get("metadata_path") or ""))
        source.setdefault("keyframe_path", str(clip.get("keyframe_path") or ""))
        if not str(source.get("video_path") or ""):
            raise ValueError(f"Clip {clip_id!r} lacks its exact source video.")
        selected.append(
            {
                "clip_index": clip_index,
                "clip_id": clip_id,
                "work_unit_id": str(clip["work_unit_id"]),
                "camera_serial": str(clip["camera_serial"]),
                "detect_run": str(refined.attrs.get("source_detect_run") or ""),
                "detect_group_path": str(refined.attrs.get("source_detect_path") or ""),
                "refined_detect_run": refined_name,
                "refined_group_path": f"refined_detect_runs/{refined_name}",
                "source_mode": SLICE_MODE,
                "canonical_parent_frame_start": int(start),
                "canonical_parent_frame_stop": int(stop),
                "row_count": int(np.count_nonzero(row_mask)),
                "source": dict(source),
            }
        )
    if not np.all(covered_rows):
        raise ValueError("Canonical refined rows are not covered by clip slices exactly once.")

    payload = {
        "schema": COLLECTION_SCHEMA,
        "collection_id": collection_name,
        "recording_identity": str(authority.get("recording_identity") or ""),
        "canonical_refined_run": refined_name,
        "canonical_refined_group_path": f"refined_detect_runs/{refined_name}",
        "canonical_row_count": int(keys.shape[0]),
        "canonical_instance_key_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
        "recording_frame_index": str(frame_index),
        "recording_frame_index_sha256": _sha256_file(frame_index),
        "selected_run_count": len(selected),
        "selected_runs": selected,
        "registered_detection_gate": gate,
        "registered_detection_gate_requirement": requirement,
        "selection_policy_id": str(
            target.get("registered_dish_geometry", {}).get("selection_policy_id")
            if isinstance(target.get("registered_dish_geometry"), Mapping)
            else ""
        ),
        "selector_eligible": False,
        "raw_detections_preserved": True,
    }
    payload_digest = canonical_json_sha256(payload)
    finalized = root.require_group("experiment_index").require_group("finalized_runs")
    if collection_name in finalized:
        raise FileExistsError(f"Immutable clipped collection already exists: {collection_name}")
    collection = finalized.create_group(collection_name)
    mark_run_started(collection, run_name=collection_name, stage="refined_detect_collection")
    collection.attrs["status"] = "running"
    collection.attrs.update({**payload, "payload_digest": payload_digest})
    mark_run_complete(collection, run_name=collection_name)
    collection.attrs["status"] = "complete"
    consolidate_metadata_capture_expected_warnings(archive)
    return {
        "status": "complete",
        "schema": COLLECTION_SCHEMA,
        "analysis_zarr": str(archive),
        "collection_id": collection_name,
        "collection_path": f"experiment_index/finalized_runs/{collection_name}",
        "payload_digest": payload_digest,
        "selected_run_count": len(selected),
        "canonical_row_count": int(keys.shape[0]),
        "registered_detection_gate": gate,
        "selector_eligible": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--recording-frame-index", type=Path, required=True)
    parser.add_argument(
        "--registered-gate-requirement",
        choices=("off", "if_available", "required"),
        required=True,
    )
    parser.add_argument("--registered-gate-run")
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        target = _target_from_plan(args.plan.expanduser().resolve(), args.target_id)
        result = finalize_registered_clipped_refined_collection(
            analysis_zarr=args.analysis_zarr,
            target=target,
            collection_id=args.collection_id,
            refined_run=args.refined_run,
            recording_frame_index=args.recording_frame_index,
            gate_requirement=args.registered_gate_requirement,
            gate_run=args.registered_gate_run,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "schema": COLLECTION_SCHEMA,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECTION_SCHEMA",
    "SLICE_MODE",
    "finalize_registered_clipped_refined_collection",
]
