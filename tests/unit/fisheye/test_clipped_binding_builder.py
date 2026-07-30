from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.shared.zarr.clipped_binding_builder import (
    build_clipped_refined_detection_binding,
)
from fisheye.utils.plan_clipped_detect_refine_workflow import PLAN_SCHEMA


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_builds_binding_from_collection_frame_map_and_strict_receipts(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    collection = (
        root.require_group("experiment_index")
        .require_group("finalized_runs")
        .create_group("collection")
    )
    selected = []
    receipts = []
    work_units = []
    frame_rows = []
    parent = 0
    for clip_index in range(2):
        clip_id = f"clip_{clip_index:06d}"
        detect_path = f"clips/{clip_id}/detect_runs/raw_{clip_index}"
        refined_path = f"clips/{clip_id}/refined_detect_runs/refined_{clip_index}"
        selected.append(
            {
                "clip_index": clip_index,
                "clip_id": clip_id,
                "camera_serial": "2010095",
                "detect_group_path": detect_path,
                "refined_group_path": refined_path,
            }
        )
        work_units.append(
            {
                "clip_index": clip_index,
                "clip_id": clip_id,
                "camera_serial": "2010095",
                "frame_count": 2,
                "source": {"video_path": f"clips/{clip_id}/Cam2010095.mp4"},
            }
        )
        refined_archive = tmp_path / f"refined_{clip_index}.zarr"
        refined_root = zarr.open_group(
            str(refined_archive),
            mode="w",
            zarr_format=3,
        )
        run = refined_root.require_group("refined_detect_runs").create_group(
            f"strict_{clip_index}"
        )
        digest = f"{clip_index + 1:x}" * 64
        run.attrs["run_manifest"] = {
            "payload_digest": digest,
            "payload": {"run_id": f"strict_{clip_index}"},
        }
        receipt_path = tmp_path / f"receipt_{clip_index}.json"
        _write_json(
            receipt_path,
            {
                "status": "complete",
                "clip": {
                    "clip_index": clip_index,
                    "clip_id": clip_id,
                    "parent_frame_start": parent,
                    "parent_frame_stop": parent + 2,
                },
                "sources": {
                    "detect_group_path": detect_path,
                    "refined_group_path": refined_path,
                },
                "refined": {
                    "archive": str(refined_archive),
                    "run_id": f"strict_{clip_index}",
                    "manifest_digest": digest,
                },
            },
        )
        receipts.append(receipt_path)
        for local in range(2):
            frame_rows.append(
                {
                    "camera_serial": "2010095",
                    "clip_id": clip_id,
                    "clip_local_frame_index": local,
                    "parent_frame_index": parent,
                    "recording_frame_id": parent + 1,
                }
            )
            parent += 1
    collection.attrs.update(
        {
            "collection_id": "collection",
            "selected_runs": selected,
            "status": "complete",
        }
    )
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "schema_version": PLAN_SCHEMA,
            "recording_id": "recording",
            "work_units": work_units,
        },
    )
    frame_index = tmp_path / "recording_frame_index.parquet"
    pq.write_table(pa.Table.from_pylist(frame_rows), frame_index)
    clip_index = tmp_path / "recording_clip_index.json"
    _write_json(
        clip_index,
        {"recording_id": "recording", "rows": work_units},
    )
    output = tmp_path / "binding.json"

    binding, receipt = build_clipped_refined_detection_binding(
        analysis_zarr=analysis,
        detection_plan_path=plan_path,
        collection_id="collection",
        recording_frame_index=frame_index,
        recording_clip_index=clip_index,
        strict_evidence_receipts=receipts,
        output_path=output,
    )

    assert binding.n_frames == 4
    assert binding.camera_serial == "2010095"
    assert [clip.parent_frame_start for clip in binding.clips] == [0, 2]
    assert [clip.source_refined_run_id for clip in binding.clips] == [
        "strict_0",
        "strict_1",
    ]
    assert output.is_file()
    assert output.with_suffix(".receipt.json").is_file()
    assert receipt["digest_algorithms"]["frame_maps"] == ("sha256_canonical_rows_v1")
