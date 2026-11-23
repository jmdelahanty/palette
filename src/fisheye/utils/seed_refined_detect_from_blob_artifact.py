"""Turn a blob-detection artifact into a reviewable refined-detections run.

``detect_traditional --artifact-only`` writes background-subtraction blobs to
``detection_artifact_runs/<run>``, which no review or training tool reads.
This tool copies those boxes, unchanged, into a new ``refined_detect_runs``
entry in the sparse-instances format, one row per box with a minted
``instance_key``. The labeling web app's ``detect_training`` task can then
show every box in a frame for a reviewer to fix, and the merged training
export reads the reviewed run.

The boxes are unreviewed candidates: the run is written without a review
status, so training export still requires review and approval.

Default is a dry run. ``--apply`` writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from fisheye.shared.refined_detect_curation import write_curated_refined_detect_surfaces
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)

TOOL_NAME = "seed_refined_detect_from_blob_artifact"
ARTIFACT_FAMILY = "detection_artifact_runs"
REFINED_FAMILY = "refined_detect_runs"
RUN_PREFIX = "refined_blob_seed_"
SEED_REASON = "blob_seed_unreviewed"


def plan_seed(root: Any, *, artifact_run: str | None = None) -> dict[str, Any]:
    """Read the artifact and check it against the archive's frames. Writes nothing."""

    raw = root.get("raw_video")
    if raw is None or "images_ds" not in raw:
        raise ValueError("archive has no raw_video/images_ds; build a sampled training archive first")
    frames, height, width = (int(n) for n in raw["images_ds"].shape[:3])
    artifacts = root.get(ARTIFACT_FAMILY)
    names = sorted(artifacts.group_keys()) if artifacts is not None else []
    if artifact_run is None:
        if len(names) != 1:
            raise ValueError(f"name the artifact run with --artifact-run; found {names}")
        artifact_run = names[0]
    if artifact_run not in names:
        raise ValueError(f"{ARTIFACT_FAMILY}/{artifact_run} not found; found {names}")
    artifact = artifacts[artifact_run]
    if artifact.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ValueError(f"{ARTIFACT_FAMILY}/{artifact_run} is not complete")
    frame_indices = np.asarray(artifact["frame_indices"][:], dtype=np.int32).reshape(-1)
    boxes = np.asarray(artifact["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    counts = np.asarray(artifact["frame_counts"][:], dtype=np.int64).reshape(-1)
    if counts.shape[0] != frames:
        raise ValueError(
            f"artifact covers {counts.shape[0]} frames but raw_video/images_ds holds {frames}"
        )
    if frame_indices.shape[0] != boxes.shape[0] or int(counts.sum()) != boxes.shape[0]:
        raise ValueError("artifact arrays disagree on the number of boxes")
    if boxes.shape[0] and (frame_indices.min() < 0 or frame_indices.max() >= frames):
        raise ValueError("artifact holds a frame index outside the archive")
    if np.any(np.diff(frame_indices) < 0):
        raise ValueError("artifact rows are not ordered by frame")
    if boxes.shape[0] and (not np.isfinite(boxes).all() or boxes.min() < 0.0 or boxes.max() > 1.0):
        raise ValueError("artifact boxes are not normalised to [0, 1]")
    source_hw = raw.attrs.get("original_resolution")
    if source_hw is not None:
        source_height, source_width = (int(n) for n in list(source_hw)[:2])
        # Blob boxes are normalised on the downsampled frame. They mean the same thing
        # on the source frame only when the downsample kept the aspect ratio.
        if source_height * width != source_width * height:
            raise ValueError(
                "raw_video/images_ds does not share the source frame's aspect ratio; "
                "normalised blob boxes would not map onto the source frame"
            )
    return {
        "artifact_run": artifact_run,
        "frames": frames,
        "boxes": int(boxes.shape[0]),
        "frames_with_boxes": int(np.count_nonzero(counts)),
        "max_boxes_per_frame": int(counts.max()) if frames else 0,
        "boxes_per_frame_histogram": {
            str(n): int(c) for n, c in zip(*np.unique(counts, return_counts=True))
        },
        "_frame_indices": frame_indices,
        "_boxes": boxes,
    }


def seed_refined_detect(
    zarr_path: Path,
    *,
    artifact_run: str | None = None,
    run_name: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    zarr_path = Path(zarr_path)
    root = open_zarr_group_direct(zarr_path, mode="a" if apply else "r")
    plan = plan_seed(root, artifact_run=artifact_run)
    frame_indices, boxes = plan.pop("_frame_indices"), plan.pop("_boxes")
    run_name = run_name or f"{RUN_PREFIX}{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    report = {"zarr_path": str(zarr_path), "refined_run": run_name, **plan}
    existing = root.get(REFINED_FAMILY)
    if existing is not None and run_name in existing:
        raise ValueError(f"{REFINED_FAMILY}/{run_name} already exists")
    if not apply:
        return {**report, "status": "would_write"}

    parent = require_runs_parent(root, REFINED_FAMILY)
    group = parent.create_group(run_name)
    mark_run_started(group, run_name=run_name, stage="refined_detect")
    note_pending_latest(parent, run_name)
    parameters = {
        "method": "blob_artifact_seed",
        "source_detection_artifact_run": plan["artifact_run"],
        "boxes_changed": False,
    }
    group.attrs.update(
        {
            "source_detection_artifact_run": plan["artifact_run"],
            "source_detection_artifact_path": f"{ARTIFACT_FAMILY}/{plan['artifact_run']}",
            "refined_family_path": REFINED_FAMILY,
            "refinement_timestamp": datetime.now(timezone.utc).isoformat(),
            "operations": ["blob_artifact_seed"],
            "parameters": parameters,
            "coverage_frames_total": plan["frames"],
        }
    )
    rows = int(boxes.shape[0])
    write_curated_refined_detect_surfaces(
        root,
        zarr_path=zarr_path,
        refined_run_name=run_name,
        instance_frame_indices=frame_indices,
        instance_bbox_norm_coords=boxes,
        instance_source_kind_labels=["raw_detect"] * rows,
        instance_reason_labels=[SEED_REASON] * rows,
        # No bound detect run: -1 makes the writer mint a stable key per box.
        instance_source_detect_row_index=np.full(rows, -1, dtype=np.int32),
        instance_manual_edit_flags=np.zeros(rows, dtype=bool),
        instance_confidence_scores=np.ones(rows, dtype=np.float32),
        instance_class_ids=np.zeros(rows, dtype=np.int32),
        command=" ".join(sys.argv),
        source_context={
            "selection_policy": "blob_artifact_seed_all_boxes",
            "source_detection_artifact_run": plan["artifact_run"],
        },
    )
    instances = group["instances"]
    if "instance_key" not in instances or int(instances["frame_indices"].shape[0]) != rows:
        raise ValueError("the refined run was not written with one keyed row per box")
    mark_run_complete(
        group,
        parent_group=parent,
        run_name=run_name,
        run_provenance=build_writer_run_provenance(
            command=TOOL_NAME,
            params=parameters,
            input_run_ids={"detection_artifact_run": plan["artifact_run"]},
        ),
    )
    return {**report, "status": "written"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--artifact-run", default=None, help="detection_artifact_runs entry to copy")
    parser.add_argument("--run-name", default=None, help="name of the refined run to create")
    parser.add_argument("--apply", action="store_true", help="write the run; without it nothing changes")
    args = parser.parse_args(argv)
    result = seed_refined_detect(
        args.zarr_path, artifact_run=args.artifact_run, run_name=args.run_name, apply=args.apply
    )
    json.dump({"tool": TOOL_NAME, **result}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
