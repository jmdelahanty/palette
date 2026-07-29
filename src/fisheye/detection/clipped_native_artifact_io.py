"""Load imported clipped detection artifacts for native canonical binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr

from fisheye.detection.clipped_native_binding import (
    ARTIFACT_COORDINATE_CONTRACT,
    ARTIFACT_ROW_ID_CONTRACT,
    ClippedDetectionArtifactMember,
)
from fisheye.utils.validate_imported_run_group import validate_imported_run_group


CLIPPED_DETECTION_WORK_UNIT_REPORT_SCHEMA = (
    "palette.clipped_detection_work_unit_report.v1"
)


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent_frame_mapping(
    path: Path,
    *,
    camera_serial: str,
    clip_id: str,
) -> np.ndarray:
    frame_index = path.expanduser().resolve()
    if not frame_index.is_file():
        raise FileNotFoundError(f"recording_frame_index.parquet not found: {frame_index}")
    parquet = pq.ParquetFile(frame_index)
    required = {
        "camera_serial",
        "clip_id",
        "clip_local_frame_index",
        "parent_frame_index",
    }
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"recording frame index is missing columns: {missing}")
    table = pq.read_table(frame_index, columns=sorted(required)).combine_chunks()
    cameras = np.asarray(
        pc.cast(table["camera_serial"], "string").to_pylist(),
        dtype=object,
    )
    clips = np.asarray(pc.cast(table["clip_id"], "string").to_pylist(), dtype=object)
    selected = (cameras == str(camera_serial)) & (clips == str(clip_id))
    if not np.any(selected):
        raise ValueError(
            "recording frame index has no rows for "
            f"camera={camera_serial!r}, clip={clip_id!r}."
        )
    local = np.asarray(
        table["clip_local_frame_index"].to_numpy(),
        dtype=np.int64,
    )[selected]
    parent = np.asarray(table["parent_frame_index"].to_numpy(), dtype=np.int64)[
        selected
    ]
    order = np.argsort(local, kind="stable")
    local = local[order]
    parent = parent[order]
    if not np.array_equal(local, np.arange(local.shape[0], dtype=np.int64)):
        raise ValueError("clip-local frame mapping must be dense from zero.")
    if np.any(parent < 0) or np.unique(parent).shape[0] != parent.shape[0]:
        raise ValueError("parent frame mapping must be nonnegative and one-to-one.")
    return parent


def _artifact_group_path(report: Mapping[str, Any]) -> str:
    value = str(report.get("target_group_path") or "").strip().strip("/")
    parts = Path(value).parts
    if len(parts) < 2 or parts[-2] != "detection_artifact_runs":
        raise ValueError(
            "Work-unit report target must be a detection_artifact_runs package, "
            f"not a canonical detect_runs path: {value!r}."
        )
    return value


def load_clipped_detection_artifact_member(
    report_path: Path,
    *,
    analysis_zarr: Path,
    recording_frame_index: Path,
    recording_identity: str,
    source_width: int,
    source_height: int,
) -> tuple[ClippedDetectionArtifactMember, dict[str, object]]:
    """Revalidate one imported package and return its exact logical arrays."""

    report_file = report_path.expanduser().resolve()
    report = _read_strict_json(report_file)
    if report.get("schema") != CLIPPED_DETECTION_WORK_UNIT_REPORT_SCHEMA:
        raise ValueError(f"Unsupported work-unit report schema: {report_file}")
    if report.get("status") != "ok":
        raise ValueError(f"Work-unit report is not complete: {report_file}")
    if report.get("recording_id") != recording_identity:
        raise ValueError("Work-unit recording identity differs from the assembler.")
    target_archive = Path(str(report.get("target_zarr") or "")).expanduser().resolve()
    archive = analysis_zarr.expanduser().resolve()
    if target_archive != archive:
        raise ValueError("Work-unit report targets a different analysis archive.")
    group_path = _artifact_group_path(report)
    validation = validate_imported_run_group(
        zarr_path=archive,
        target_group_path=group_path,
        validate_source_tarball=False,
    )
    if validation.get("status") not in {"ok", "pass"}:
        raise ValueError(
            "Imported detection artifact failed fresh validation: "
            + json.dumps(validation, sort_keys=True, default=str)
        )

    imported = report.get("import")
    if not isinstance(imported, Mapping):
        raise ValueError("Work-unit report has no import receipt binding.")
    validation_receipt = (
        ((report.get("validation") or {}).get("validations") or {}).get("receipt")
        if isinstance(report.get("validation"), Mapping)
        else None
    )
    receipt_value = imported.get("receipt_path")
    if not receipt_value and isinstance(validation_receipt, Mapping):
        receipt_value = validation_receipt.get("path")
    if not receipt_value:
        raise ValueError("Work-unit report has no durable import-receipt path.")
    receipt_path = Path(str(receipt_value)).expanduser().resolve()
    receipt = _read_strict_json(receipt_path)
    if receipt.get("target_archive_path") != str(archive):
        raise ValueError("Artifact import receipt targets a different archive.")
    if receipt.get("target_group_path") != group_path:
        raise ValueError("Artifact import receipt group differs from the report.")
    manifest = receipt.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Artifact import receipt has no manifest.")
    manifest_sha256 = _sha256_file(receipt_path)
    persisted_manifest_sha256 = str(receipt.get("manifest_sha256") or "")
    report_manifest_sha256 = str(imported.get("manifest_sha256") or "")
    if report_manifest_sha256 and persisted_manifest_sha256 != report_manifest_sha256:
        raise ValueError("Work-unit and import-receipt manifest digests disagree.")
    clip_context = manifest.get("clip_context")
    if not isinstance(clip_context, Mapping):
        raise ValueError("Artifact manifest has no clip context.")
    for name in ("recording_id", "clip_id", "clip_index", "camera_serial"):
        if clip_context.get(name) != report.get(name):
            raise ValueError(f"Artifact manifest/report {name} identity mismatch.")
    if clip_context.get("recording_id") != recording_identity:
        raise ValueError("Artifact manifest recording identity mismatch.")

    run = zarr.open_group(
        str(archive / group_path),
        mode="r",
        use_consolidated=False,
    )
    if run.attrs.get("coordinate_contract") != ARTIFACT_COORDINATE_CONTRACT:
        raise ValueError("Imported detection package has the wrong coordinate contract.")
    if run.attrs.get("artifact_row_id_contract") != ARTIFACT_ROW_ID_CONTRACT:
        raise ValueError("Imported detection package has the wrong row-id contract.")
    if run.attrs.get("stage_selector_eligible") is not False:
        raise ValueError("Imported detection package became selector eligible.")
    observed_dimensions = (
        int(run.attrs.get("source_video_width") or 0),
        int(run.attrs.get("source_video_height") or 0),
    )
    if observed_dimensions != (int(source_width), int(source_height)):
        raise ValueError(
            "Artifact source dimensions differ from the recording authority: "
            f"observed={observed_dimensions}, expected={(source_width, source_height)}."
        )
    clip_id = str(report["clip_id"])
    camera_serial = str(report["camera_serial"])
    parent_frames = _load_parent_frame_mapping(
        recording_frame_index,
        camera_serial=camera_serial,
        clip_id=clip_id,
    )
    run_tree_hash = str(
        (manifest.get("checksums") or {}).get("run_group_tree_hash")
        if isinstance(manifest.get("checksums"), Mapping)
        else ""
    )
    member = ClippedDetectionArtifactMember(
        work_unit_id=f"{clip_id}:camera_{camera_serial}",
        artifact_run_id=str(receipt["run_name"]),
        clip_id=clip_id,
        clip_index=int(report["clip_index"]),
        camera_serial=camera_serial,
        source_width=int(source_width),
        source_height=int(source_height),
        artifact_manifest_sha256=persisted_manifest_sha256,
        run_group_tree_sha256=run_tree_hash,
        parent_frame_indices=parent_frames,
        frame_indices=np.asarray(run["frame_indices"][:]),
        bbox_norm_coords=np.asarray(run["bbox_norm_coords"][:]),
        scores=np.asarray(run["scores"][:]),
        class_ids=np.asarray(run["class_ids"][:]),
        artifact_row_id=np.asarray(run["artifact_row_id"][:]),
        frame_counts=np.asarray(run["frame_counts"][:]),
        n_detections=np.asarray(run["n_detections"][:]),
    )
    evidence: dict[str, object] = {
        "report_path": str(report_file),
        "report_sha256": _sha256_file(report_file),
        "receipt_path": str(receipt_path),
        "receipt_sha256": manifest_sha256,
        "artifact_group_path": group_path,
        "artifact_manifest_sha256": persisted_manifest_sha256,
        "run_group_tree_sha256": run_tree_hash,
        "run_provenance": dict(run.attrs.get("run_provenance") or {}),
    }
    return member, evidence


def load_clipped_detection_artifact_members(
    report_paths: Sequence[Path],
    **kwargs: Any,
) -> tuple[tuple[ClippedDetectionArtifactMember, ...], tuple[dict[str, object], ...]]:
    loaded = [
        load_clipped_detection_artifact_member(path, **kwargs)
        for path in report_paths
    ]
    return (
        tuple(item[0] for item in loaded),
        tuple(item[1] for item in loaded),
    )


__all__ = [
    "CLIPPED_DETECTION_WORK_UNIT_REPORT_SCHEMA",
    "load_clipped_detection_artifact_member",
    "load_clipped_detection_artifact_members",
]
