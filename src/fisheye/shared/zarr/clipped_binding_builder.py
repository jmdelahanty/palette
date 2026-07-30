"""Build the exact clipped refined-detection binding from persisted evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_schema import (
    RefinedDetectionClipBinding,
    RefinedDetectionClippedBinding,
)
from fisheye.utils.plan_clipped_detect_refine_workflow import PLAN_SCHEMA


CLIPPED_BINDING_BUILD_SCHEMA_ID = "palette.refined_detection.clipped_binding_build"
CLIPPED_BINDING_BUILD_SCHEMA_VERSION = 1
_FRAME_COLUMNS = (
    "camera_serial",
    "clip_id",
    "clip_local_frame_index",
    "parent_frame_index",
    "recording_frame_id",
)


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _group_at(root: Any, path: str) -> Any:
    group = root
    for component in str(path).strip("/").split("/"):
        group = group[component]
    return group


def _start_row_digest(*, scope: str) -> Any:
    digest = hashlib.sha256()
    digest.update(b"palette.sha256_canonical_rows.v1\x00")
    digest.update(scope.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(canonical_json_bytes(list(_FRAME_COLUMNS)))
    digest.update(b"\x00")
    return digest


def _update_row_digest(digest: Any, row: Sequence[object]) -> None:
    encoded = canonical_json_bytes(list(row))
    digest.update(len(encoded).to_bytes(8, "little", signed=False))
    digest.update(encoded)


def _frame_map_digests(
    path: Path,
    *,
    camera_serial: str,
    clips: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[int, str]]:
    parquet = pq.ParquetFile(path.expanduser().resolve())
    missing = sorted(set(_FRAME_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Recording frame index is missing columns {missing!r}.")
    by_id = {str(item["clip_id"]): item for item in clips}
    if len(by_id) != len(clips):
        raise ValueError("Strict clip receipts contain duplicate clip ids.")
    full = _start_row_digest(scope=f"camera:{camera_serial}")
    per_clip = {
        int(item["clip_index"]): _start_row_digest(
            scope=f"camera:{camera_serial}:clip:{item['clip_id']}"
        )
        for item in clips
    }
    counts = {int(item["clip_index"]): 0 for item in clips}
    expected_parent = 0
    for batch in parquet.iter_batches(columns=list(_FRAME_COLUMNS), batch_size=65_536):
        columns = [
            batch.column(index).to_pylist() for index in range(len(_FRAME_COLUMNS))
        ]
        for row in zip(*columns, strict=True):
            camera, clip_id, local, parent, recording_frame_id = row
            if str(camera) != camera_serial:
                continue
            clip = by_id.get(str(clip_id))
            if clip is None:
                raise ValueError(
                    f"Recording frame index contains unbound clip {clip_id!r}."
                )
            clip_index = int(clip["clip_index"])
            local_int = int(local)
            parent_int = int(parent)
            recording_int = int(recording_frame_id)
            if parent_int != expected_parent:
                raise ValueError(
                    "Selected camera parent_frame_index must be dense and ordered."
                )
            if local_int != counts[clip_index]:
                raise ValueError(
                    f"Clip {clip_id!r} local frame mapping is not dense and ordered."
                )
            expected_clip_parent = int(clip["parent_frame_start"]) + local_int
            if parent_int != expected_clip_parent:
                raise ValueError(
                    f"Clip {clip_id!r} frame mapping differs from its bound interval."
                )
            if recording_int != parent_int + 1:
                raise ValueError(
                    "recording_frame_id must equal one-based parent_frame_index."
                )
            canonical_row = (
                camera_serial,
                str(clip_id),
                local_int,
                parent_int,
                recording_int,
            )
            _update_row_digest(full, canonical_row)
            _update_row_digest(per_clip[clip_index], canonical_row)
            counts[clip_index] += 1
            expected_parent += 1
    for item in clips:
        clip_index = int(item["clip_index"])
        expected = int(item["parent_frame_stop"]) - int(item["parent_frame_start"])
        if counts[clip_index] != expected:
            raise ValueError(
                f"Clip {item['clip_id']!r} frame-map count differs from strict evidence."
            )
        if int(item["parent_frame_start"]) + counts[clip_index] != int(
            item["parent_frame_stop"]
        ):
            raise ValueError(f"Clip {item['clip_id']!r} interval is inconsistent.")
    return full.hexdigest(), {
        index: digest.hexdigest() for index, digest in per_clip.items()
    }


def build_clipped_refined_detection_binding(
    *,
    analysis_zarr: Path,
    detection_plan_path: Path,
    collection_id: str,
    recording_frame_index: Path,
    recording_clip_index: Path,
    strict_evidence_receipts: Sequence[Path],
    output_path: Path | None = None,
) -> tuple[RefinedDetectionClippedBinding, Mapping[str, object]]:
    """Reopen every declaration and build one non-hand-authored binding."""

    if not strict_evidence_receipts:
        raise ValueError("At least one strict clip evidence receipt is required.")
    plan = _read_json(detection_plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("Detection plan schema is not supported.")
    work_units = plan.get("work_units")
    if not isinstance(work_units, list) or not work_units:
        raise ValueError("Detection plan has no work units.")
    units_by_clip = {
        (int(item["clip_index"]), str(item["clip_id"])): item
        for item in work_units
        if isinstance(item, Mapping)
    }
    receipts = [_read_json(path) for path in strict_evidence_receipts]
    receipts.sort(key=lambda item: int(item["clip"]["clip_index"]))
    if [int(item["clip"]["clip_index"]) for item in receipts] != list(
        range(len(receipts))
    ):
        raise ValueError("Strict evidence receipts must cover [0, clip_count).")
    if any(item.get("status") != "complete" for item in receipts):
        raise ValueError("Every strict clip evidence receipt must be complete.")

    archive = analysis_zarr.expanduser().resolve()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    collection_path = f"experiment_index/finalized_runs/{collection_id}"
    collection = _group_at(root, collection_path)
    collection_attrs = dict(collection.attrs)
    if collection_attrs.get("collection_id") != collection_id:
        raise ValueError("Finalized collection id differs from the requested binding.")
    selected = collection_attrs.get("selected_runs")
    if not isinstance(selected, list) or len(selected) != len(receipts):
        raise ValueError("Finalized collection does not match strict evidence count.")
    selected_by_clip = {
        (int(item["clip_index"]), str(item["clip_id"])): item
        for item in selected
        if isinstance(item, Mapping)
    }

    cameras = {str(item["camera_serial"]) for item in selected_by_clip.values()}
    if len(cameras) != 1:
        raise ValueError("Clipped refined binding supports exactly one camera.")
    camera_serial = next(iter(cameras))
    recording_clip_index_document = _read_json(recording_clip_index)
    recording_clip_index_digest = canonical_json_sha256(recording_clip_index_document)
    clip_rows: list[dict[str, Any]] = []
    for receipt in receipts:
        clip = receipt["clip"]
        key = (int(clip["clip_index"]), str(clip["clip_id"]))
        unit = units_by_clip.get(key)
        selected_run = selected_by_clip.get(key)
        if unit is None or selected_run is None:
            raise ValueError(f"Clip {key!r} is absent from plan or collection.")
        if receipt["sources"]["detect_group_path"] != selected_run.get(
            "detect_group_path"
        ) or receipt["sources"]["refined_group_path"] != selected_run.get(
            "refined_group_path"
        ):
            raise ValueError(f"Clip {key!r} source paths differ from the collection.")
        refined_archive = Path(str(receipt["refined"]["archive"])).resolve()
        refined_run_id = str(receipt["refined"]["run_id"])
        refined_root = zarr.open_group(
            str(refined_archive),
            mode="r",
            use_consolidated=False,
        )
        manifest = refined_root[f"refined_detect_runs/{refined_run_id}"].attrs.get(
            "run_manifest"
        )
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("payload_digest") != (receipt["refined"]["manifest_digest"])
        ):
            raise ValueError(f"Clip {key!r} refined manifest receipt is stale.")
        source = unit.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"Clip {key!r} has no source descriptor.")
        media_descriptor = {
            "recording_clip_index_digest": recording_clip_index_digest,
            "clip_id": key[1],
            "clip_index": key[0],
            "camera_serial": camera_serial,
            "frame_count": unit.get("frame_count"),
            "source": dict(source),
        }
        canonical_json_bytes(media_descriptor)
        clip_rows.append(
            {
                "clip_index": key[0],
                "clip_id": key[1],
                "camera_serial": camera_serial,
                "media_identity": str(source.get("video_path") or ""),
                "media_digest": canonical_json_sha256(media_descriptor),
                "parent_frame_start": int(clip["parent_frame_start"]),
                "parent_frame_stop": int(clip["parent_frame_stop"]),
                "source_refined_run_id": refined_run_id,
                "source_refined_manifest_digest": manifest["payload_digest"],
            }
        )

    frame_index_digest, clip_frame_digests = _frame_map_digests(
        recording_frame_index,
        camera_serial=camera_serial,
        clips=clip_rows,
    )
    clips = tuple(
        RefinedDetectionClipBinding(
            clip_index=int(item["clip_index"]),
            clip_id=str(item["clip_id"]),
            media_identity=str(item["media_identity"]),
            media_digest=str(item["media_digest"]),
            parent_frame_start=int(item["parent_frame_start"]),
            parent_frame_stop=int(item["parent_frame_stop"]),
            frame_map_digest=clip_frame_digests[int(item["clip_index"])],
            source_refined_run_id=str(item["source_refined_run_id"]),
            source_refined_manifest_digest=str(item["source_refined_manifest_digest"]),
        )
        for item in clip_rows
    )
    binding = RefinedDetectionClippedBinding(
        collection_id=collection_id,
        collection_manifest_digest=canonical_json_sha256(collection_attrs),
        camera_serial=camera_serial,
        video_identity=str(plan.get("recording_id") or ""),
        video_manifest_digest=recording_clip_index_digest,
        recording_frame_index_digest=frame_index_digest,
        clips=clips,
    )
    receipt: dict[str, object] = {
        "schema_id": CLIPPED_BINDING_BUILD_SCHEMA_ID,
        "schema_version": CLIPPED_BINDING_BUILD_SCHEMA_VERSION,
        "status": "complete",
        "selector_eligible": False,
        "registry_updated": False,
        "analysis_zarr": str(archive),
        "detection_plan": str(detection_plan_path.expanduser().resolve()),
        "collection_path": collection_path,
        "recording_frame_index": str(recording_frame_index.expanduser().resolve()),
        "recording_clip_index": str(recording_clip_index.expanduser().resolve()),
        "strict_evidence_receipts": [
            str(path.expanduser().resolve()) for path in strict_evidence_receipts
        ],
        "binding": binding.as_manifest(),
        "digest_algorithms": {
            "documents": "sha256_canonical_json_v1",
            "frame_maps": "sha256_canonical_rows_v1",
            "media": "sha256_canonical_media_descriptor_v1",
        },
    }
    if output_path is not None:
        write_json_atomic(output_path.expanduser().resolve(), binding.as_manifest())
        write_json_atomic(
            output_path.expanduser().resolve().with_suffix(".receipt.json"),
            receipt,
        )
    return binding, receipt


__all__ = [
    "CLIPPED_BINDING_BUILD_SCHEMA_ID",
    "CLIPPED_BINDING_BUILD_SCHEMA_VERSION",
    "build_clipped_refined_detection_binding",
]
