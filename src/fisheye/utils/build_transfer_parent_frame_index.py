"""Build one bounded-memory convenience index from an immutable transfer-v2.

This adapter uses the existing frame-index table schema and transfer CSV parser.
It writes only a fresh external output directory; it never organizes/moves source
payloads, admits media/clocks, mints import receipts or updates a registry. The
manifest is written last. Interrupted directories are not automatically reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.recording_transfer_snapshot import (
    file_ref,
    frame_map,
    plan_parent_recordings,
    require,
    verify_transfer_snapshot,
)
from fisheye.utils.build_recording_frame_index import (
    ARTIFACT_ROLE,
    MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TABLE_COLUMNS,
    TABLE_SCHEMA,
)

MODULE_NAME = "fisheye.utils.build_transfer_parent_frame_index"
BINDING_SCHEMA = "palette.recording_frame_index.transfer_v2_source.v1"


def build_transfer_parent_frame_index(
    recording_dir: Path,
    *,
    camera_id: str,
    output_dir: Path,
    batch_rows: int = 65536,
    dry_run: bool = False,
    organization_plan: dict | None = None,
) -> dict:
    """Index a single exact parent without changing legacy index/digest grammar.

    New transfer bindings are additive only on this opt-in path. Buffer size is
    an execution/storage parameter; changing it may change Parquet bytes, never
    source identities, row values, or original transport snapshot digests.
    """
    started = time.perf_counter()
    require(
        type(batch_rows) is int and 1 <= batch_rows <= 1048576,
        "batch_rows must be an integer from 1 to 1048576",
    )
    require(type(camera_id) is str and bool(camera_id), "camera_id must be exact text")
    transfer = verify_transfer_snapshot(recording_dir)
    plans = plan_parent_recordings(transfer)
    matches = [plan for plan in plans if plan.camera_id == camera_id]
    require(len(matches) == 1, "camera_id does not select one exact parent")
    plan = matches[0]
    root = transfer.root
    index_recording_root = root
    organized_paths = None
    if organization_plan is not None:
        from fisheye.utils.organize_transfer_recordings import (
            resolve_materialized_parent_sources,
        )

        require(
            organization_plan.get("source_dir") == str(root),
            "organization plan source differs",
        )
        require(
            plan.recording_layout == "rolling_clips",
            "organized collection indexing requires rolling_clips",
        )
        index_recording_root, organized_paths = resolve_materialized_parent_sources(
            organization_plan,
            camera_id=camera_id,
        )
    output = Path(output_dir).absolute()
    require(
        not any(path.is_symlink() for path in (output, *output.parents)),
        "output destination contains a symlink",
    )
    output = output.resolve()
    if organized_paths is not None:
        require(
            output == index_recording_root / "derived/recording_frame_index",
            "organized index must use its exact parent index directory",
        )
    require(
        output != root and root not in output.parents and output not in root.parents,
        "output must be separate from the immutable source tree",
    )
    if output.exists():
        raise FileExistsError(
            f"Refusing existing output (including incomplete prior attempts): {output}"
        )
    parquet_path = output / "recording_frame_index.parquet"
    clip_index_path = output / "recording_clip_index.json"
    manifest_path = output / "recording_frame_index_manifest.json"
    source_parent = next(
        parent
        for parent in transfer.snapshot["parents"]
        if parent["parent_key"]["camera_serial"] == camera_id
    )
    binding = {
        "schema": BINDING_SCHEMA,
        "snapshot_id": transfer.snapshot_id,
        "recording_id": plan.recording_id,
        "session_uuid": plan.session_uuid,
        "camera_id": camera_id,
        "source_root": str(root),
        "recording_layout": plan.recording_layout,
        "recording_payload_kind": plan.recording_payload_kind,
    }
    if organized_paths is not None:
        binding.update(
            {
                "schema": "palette.recording_frame_index.organized_transfer_v2_source.v1",
                "organized_recording_root": str(index_recording_root),
                "organization_plan_sha256": organization_plan["plan_sha256"],
                "source_to_organized_paths": organized_paths,
            }
        )
    writer = None
    rows: list[dict] = []
    row_count = max_buffered = 0
    inputs = []
    clip_rows = []
    source_files = []
    projected_manifests = []
    output_identity = None

    def require_output_ownership() -> None:
        if output_identity is not None:
            current = output.lstat()
            require(
                not output.is_symlink()
                and (current.st_dev, current.st_ino) == output_identity,
                "output directory ownership lost during indexing",
            )

    def flush() -> None:
        if rows:
            if writer is not None:
                writer.write_table(pa.Table.from_pylist(rows, schema=TABLE_SCHEMA))
            rows.clear()

    try:
        if not dry_run:
            # Exclusive reservation: never adopt another worker's directory.
            output.mkdir(parents=True, exist_ok=False)
            reserved = output.stat()
            output_identity = (reserved.st_dev, reserved.st_ino)
            writer = pq.ParquetWriter(parquet_path, TABLE_SCHEMA, compression="zstd")
        for clip in source_parent["clips"]:
            require(clip["clip_index"] <= 2**31 - 1, "clip_index exceeds signed-int32")
            full = next(
                item for item in clip["outputs"] if item["output_kind"] == "full"
            )
            metadata_path = root / full["metadata"]["path"]
            video_path = root / full["video"]["path"]
            keyframe = next(
                (
                    str(root / sidecar["artifact"]["path"])
                    for sidecar in full["sidecars"]
                    if sidecar["role"] == "keyframes"
                ),
                None,
            )
            clip_manifest = root / clip["directory"] / "clip_manifest.json"
            clip_manifest_text = str(clip_manifest) if clip_manifest.is_file() else None
            recorded_metadata_path = metadata_path
            if organized_paths is not None:
                video_path = (
                    index_recording_root / organized_paths[full["video"]["path"]]
                )
                recorded_metadata_path = (
                    index_recording_root / organized_paths[full["metadata"]["path"]]
                )
                if keyframe is not None:
                    keyframe = str(
                        index_recording_root
                        / organized_paths[Path(keyframe).relative_to(root).as_posix()]
                    )
                projection_path = (
                    output / f"clip_{clip['clip_index']:06d}_projection.json"
                )
                clip_manifest_text = str(projection_path)
                projected_outputs = {}
                for item in clip["outputs"]:
                    projected_outputs[item["output_kind"]] = {
                        "output_kind": item["output_kind"],
                        "video": organized_paths[item["video"]["path"]],
                        "metadata": organized_paths[item["metadata"]["path"]],
                        **{
                            field: item["frame_map"][field]
                            for field in (
                                "frame_count",
                                "first_recording_frame_id",
                                "last_recording_frame_id",
                            )
                        },
                    }
                if not dry_run:
                    require_output_ownership()
                    write_json_atomic(
                        projection_path,
                        {
                            "schema_id": "palette.transfer_organized_clip_projection.v1",
                            "generated_by": MODULE_NAME,
                            "source_transfer": binding,
                            "original_clip_manifest": (
                                file_ref(
                                    root, clip_manifest.relative_to(root).as_posix()
                                )
                                if clip_manifest.is_file()
                                else None
                            ),
                            "original_recording_session": file_ref(
                                root, "recording_session.json"
                            ),
                            "clip_index": clip["clip_index"],
                            "clip_id": clip["clip_id"],
                            "recording_outputs": {camera_id: projected_outputs},
                        },
                        overwrite=False,
                    )
                    projected_manifests.append(file_ref(output, projection_path.name))
            constants = {
                "session_id": plan.session_uuid,
                "recording_id": plan.recording_id,
                "producer": MODULE_NAME,
                "recording_folder": str(index_recording_root),
                "source_layout": plan.recording_layout,
                "recording_backend_mode": plan.recording_layout,
                "camera_serial": camera_id,
                "clip_index": clip["clip_index"],
                "clip_id": clip["clip_id"],
                "video_path": str(video_path),
                "metadata_path": str(recorded_metadata_path),
                "keyframe_path": keyframe,
                "clip_manifest_path": clip_manifest_text,
                "clip_directory": clip["directory"],
                "clip_recording_folder": (
                    str(video_path.parent)
                    if organized_paths is not None
                    else str(root / clip["directory"])
                ),
            }

            def append(
                local: int, frame: int, timestamp: int, timestamp_sys: int
            ) -> None:
                nonlocal row_count, max_buffered
                require(
                    all(
                        value <= 2**63 - 1
                        for value in (frame, timestamp, timestamp_sys)
                    ),
                    "source uint64 value exceeds existing signed-int64 frame-index schema",
                )
                require(frame == row_count + 1, "dense parent frame continuity changed")
                rows.append(
                    {
                        **constants,
                        "recording_frame_id": frame,
                        "parent_frame_index": frame - 1,
                        "clip_local_frame_index": local,
                        "timestamp": timestamp,
                        "timestamp_sys": timestamp_sys,
                    }
                )
                row_count += 1
                max_buffered = max(max_buffered, len(rows))
                if len(rows) == batch_rows:
                    flush()

            previous = row_count
            mapping = frame_map(metadata_path, "full", previous, 0, on_row=append)
            require(
                mapping == full["frame_map"], "source frame map changed during indexing"
            )
            inputs.append(
                {"clip_id": clip["clip_id"], "camera_serial": camera_id, **mapping}
            )
            clip_rows.append(
                {
                    **constants,
                    "frame_count": mapping["frame_count"],
                    "first_recording_frame_id": mapping["first_recording_frame_id"],
                    "last_recording_frame_id": mapping["last_recording_frame_id"],
                }
            )
            source_files.extend([full["video"], full["metadata"]])
        flush()
    finally:
        if writer is not None:
            writer.close()
    require(row_count == plan.total_frames, "parent total changed during indexing")
    current = verify_transfer_snapshot(root)
    require(
        current.snapshot_id == transfer.snapshot_id,
        "source generation changed during indexing",
    )
    require_output_ownership()
    if organized_paths is not None:
        # Recheck the exact organized byte map after all rows were written.
        resolved_root, resolved_paths = resolve_materialized_parent_sources(
            organization_plan, camera_id=camera_id
        )
        require(
            (resolved_root, resolved_paths) == (index_recording_root, organized_paths),
            "organized source generation changed during indexing",
        )
    # Index creation does not validate encoded media or clock suitability. These
    # remain independent gates in the actual source importer.
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "frame_index_schema_version": SCHEMA_VERSION,
        "generated_by": MODULE_NAME,
        "generated_at_utc": utc_now(),
        "host": socket.gethostname(),
        "artifact_role": ARTIFACT_ROLE,
        "status": "ok",
        "dry_run": dry_run,
        "source_authority": "recording_clip_index + per_clip_metadata_csv",
        "source_transfer": binding,
        "source_layout": plan.recording_layout,
        "recording_folder": str(index_recording_root),
        "recording_id": plan.recording_id,
        "session_id": plan.session_uuid,
        "camera_serials": [camera_id],
        "recording_clip_index_json": str(clip_index_path),
        "recording_frame_index_path": str(parquet_path),
        "recording_frame_index_manifest_path": str(manifest_path),
        "row_count": row_count,
        "columns": TABLE_COLUMNS,
        "recording_frame_id_min": 1,
        "recording_frame_id_max": row_count,
        "frame_id_base": "one",
        "inputs": inputs,
        "source_files": source_files,
        "checks": [{"status": "ok", "code": "exact_dense_parent_source_rows"}],
        "failure_count": 0,
        "clock_validity": "not_evaluated_by_index",
        "import_complete": False,
        "registry_admitted": False,
        "source_cleanup_authorized": False,
        "execution": {"batch_rows": batch_rows, "max_buffered_rows": max_buffered},
        "duration_seconds": time.perf_counter() - started,
    }
    if not dry_run:
        manifest["parquet_sha256"] = file_ref(output, parquet_path.name)["sha256"]
        mapped_fields = {}
        if organized_paths is not None:
            manifest["projected_clip_manifests"] = projected_manifests
            mapped_fields = {
                "schema_id": "palette.orange_external_ipc_recording_clip_index.v1",
                "mode": "rolling_clips",
                "rows": clip_rows,
                "camera_ranges": {
                    camera_id: {
                        "clip_count": len(clip_rows),
                        "total_frame_count": row_count,
                        "first_recording_frame_id": 1,
                        "last_recording_frame_id": row_count,
                    }
                },
            }
        write_json_atomic(
            clip_index_path,
            {
                "recording_id": plan.recording_id,
                "session_id": plan.session_uuid,
                "producer": MODULE_NAME,
                "recording_backend_mode": plan.recording_layout,
                "source_transfer": binding,
                "clips": clip_rows,
                **mapped_fields,
            },
            overwrite=False,
        )
        manifest["recording_clip_index_sha256"] = file_ref(
            output, clip_index_path.name
        )["sha256"]
        require_output_ownership()
        write_json_atomic(manifest_path, manifest, overwrite=False)
    return {
        **manifest,
        "wrote_parquet": not dry_run,
        "manifest_path": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-rows", type=int, default=65536)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_transfer_parent_frame_index(**vars(args))
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
