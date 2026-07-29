"""Publish a selector-ineligible coordinate-catalog canary package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import (
    sha256_file,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    CanonicalDetectionShadowPublication,
    publish_legacy_canonical_detection_shadow,
    validate_canonical_detection_shadow_publication,
)
from fisheye.shared.zarr.crop_manifest import CropPixelAuthority
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_shadow import (
    CropGeometryShadowPublication,
    publish_refined_crop_geometry_shadow,
    validate_crop_geometry_shadow_publication,
)
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
)
from fisheye.shared.zarr.refined_detection_shadow import (
    publish_refined_detection_shadow,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_refined_detection_transition,
)


CANARY_HANDOFF_SCHEMA_ID = "palette.coordinate_catalog.crimson_canary_handoff"
CANARY_HANDOFF_SCHEMA_VERSION = 1
CANONICAL_ARTIFACT_NAME = "canonical_source.zarr"
REFINED_ARTIFACT_NAME = "refined.zarr"
CROP_ARTIFACT_NAME = "crops.zarr"
HANDOFF_NAME = "handoff_manifest.json"
_UUID_NAMESPACE = uuid.UUID("ed840d8a-e505-4f83-891e-e1752b989e84")


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _palette_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    return {
        "repository": str(repository),
        "commit": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "worktree_clean": _git(repository, "status", "--short") == "",
        "driver": str(Path(__file__).resolve().relative_to(repository)),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
    }


def _require_safe_destination(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if ".palette_benchmarks" not in resolved.parts:
        raise ValueError("Canary destination must be in .palette_benchmarks.")
    if resolved.suffix:
        raise ValueError("Canary destination must be a package directory, not a file.")
    if resolved.exists():
        raise FileExistsError(f"Canary destination already exists: {resolved}")
    return resolved


def _require_work_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Canary work root does not exist: {resolved}")
    if not (
        resolved.is_relative_to(Path("/tmp").resolve())
        or ".palette_scratch" in resolved.parts
    ):
        raise ValueError("Canary work root must be local /tmp or .palette_scratch.")
    return resolved


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {token}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_nonfinite)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _artifact_evidence(
    path: Path,
    *,
    final_path: Path,
    run_id: str,
    manifest: Mapping[str, Any],
    logical_content_digest: str,
    dimensions: Mapping[str, Any],
    receipt_name: str,
) -> dict[str, object]:
    trees = _tree_hashes(path)
    coordinate = manifest["payload"]["coordinate_contract"]
    return {
        "run_id": run_id,
        "server_path": str(final_path),
        "macos_path": str(final_path).replace(
            "/groups/johnson/johnsonlab",
            "/Volumes/johnsonlab",
            1,
        ),
        "manifest_schema_version": manifest["schema_version"],
        "manifest_digest": manifest["payload_digest"],
        "coordinate_catalog_digest": coordinate["digest"],
        "logical_content_digest": logical_content_digest,
        "dimensions": dict(dimensions),
        "storage_profile_id": manifest["payload"]["storage_plan"]["storage_profile"][
            "profile_id"
        ],
        "receipt_sha256": sha256_file(path / receipt_name),
        "tree_digest": canonical_json_sha256(trees),
        "storage_stats": storage_stats(path),
    }


def _frame_cardinality(offsets: Any) -> dict[str, int]:
    values = np.asarray(offsets[:], dtype=np.int64)
    counts = np.diff(values)
    return {
        "empty_frame_count": int(np.count_nonzero(counts == 0)),
        "multi_row_frame_count": int(np.count_nonzero(counts > 1)),
        "maximum_rows_per_frame": int(counts.max(initial=0)),
    }


def _conversion_samples(
    canonical: CanonicalDetectionShadowPublication,
    crop: CropGeometryShadowPublication,
) -> dict[str, object]:
    if canonical.dimensions.n_instances <= 0 or crop.dimensions.n_instances <= 0:
        raise ValueError("Coordinate canary requires at least one presented row.")

    normalized = np.asarray(
        canonical.arrays["instances/bbox_norm_coords"][0],
        dtype=np.float32,
    )
    expected_bbox, expected_center = derive_canonical_detection_geometry(
        normalized.reshape(1, 4),
        source_width=canonical.dimensions.source_width,
        source_height=canonical.dimensions.source_height,
    )
    stored_bbox = np.asarray(
        canonical.arrays["instances/bbox_img_xyxy"][0], dtype=np.float32
    )
    stored_center = np.asarray(
        canonical.arrays["instances/centers_img_xy"][0], dtype=np.float32
    )
    np.testing.assert_array_equal(stored_bbox, expected_bbox[0])
    np.testing.assert_array_equal(stored_center, expected_center[0])

    origin = np.asarray(crop.arrays["roi_coordinates_full"][0], dtype=np.int32)
    extent = np.asarray(crop.arrays["roi_sizes_full"][0], dtype=np.int32)
    roi_bbox = np.asarray(crop.arrays["bbox_roi_xyxy"][0], dtype=np.float32)
    source_bbox = np.asarray(crop.arrays["bbox_img_xyxy"][0], dtype=np.float32)
    translated = roi_bbox + np.asarray(
        [origin[0], origin[1], origin[0], origin[1]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(translated, source_bbox)

    return {
        "normalized_detection": {
            "row_index": 0,
            "source_width": canonical.dimensions.source_width,
            "source_height": canonical.dimensions.source_height,
            "bbox_norm_cxcywh": normalized.tolist(),
            "expected_bbox_source_camera_xyxy": stored_bbox.tolist(),
            "expected_center_source_camera_xy": stored_center.tolist(),
            "maximum_absolute_error": 0.0,
        },
        "rowwise_roi_to_source": {
            "row_index": 0,
            "instance_key": int(crop.arrays["instance_key"][0]),
            "roi_origin_source_camera_xy": origin.tolist(),
            "roi_extent_wh": extent.tolist(),
            "bbox_roi_xyxy": roi_bbox.tolist(),
            "expected_bbox_source_camera_xyxy": source_bbox.tolist(),
            "maximum_absolute_error": 0.0,
        },
    }


def _pixel_authority_document(
    *,
    recording_identity: str,
    camera_identity: str,
    video_reference: Path,
    video_sha256: str,
    recording_manifest: Path,
    recording_manifest_sha256: str,
    n_frames: int,
    source_width: int,
    source_height: int,
) -> dict[str, object]:
    stat = video_reference.stat()
    return {
        "schema_id": "palette.coordinate_catalog_canary.pixel_authority",
        "schema_version": 1,
        "recording_identity": recording_identity,
        "camera_identity": camera_identity,
        "recording_manifest": {
            "path": str(recording_manifest.resolve()),
            "sha256": recording_manifest_sha256,
        },
        "source_video": {
            "path": str(video_reference.resolve()),
            "sha256": video_sha256,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "frame_axis": {
            "n_frames": n_frames,
            "index_domain": "zero_based_acquisition_camera_frame",
        },
        "decoded_pixel_contract": {
            "source_width": source_width,
            "source_height": source_height,
            "dtype": "uint8",
            "channels": "grayscale",
            "axis_order": "yx",
        },
    }


def publish_coordinate_catalog_canary(
    *,
    source_analysis_zarr: Path,
    recording_identity: str,
    camera_identity: str,
    video_reference: Path,
    recording_manifest: Path,
    legacy_detect_run: str,
    legacy_refined_run: str,
    destination: Path,
    work_root: Path,
    canonical_run_id: str,
    refined_run_id: str,
    crop_run_id: str,
    crop_size: int,
    crimson_commit: str,
    crimson_review_path: str,
    crimson_review_sha256: str,
    allow_initialize_missing_source_keys: bool = False,
) -> dict[str, object]:
    """Create locally, validate, copy, and atomically expose one canary package."""

    final_path = _require_safe_destination(destination)
    local_root = _require_work_root(work_root)
    source_archive = source_analysis_zarr.expanduser().resolve()
    video = video_reference.expanduser().resolve()
    recording_record = recording_manifest.expanduser().resolve()
    if not source_archive.is_dir() or source_archive.suffix != ".zarr":
        raise FileNotFoundError(f"Source analysis Zarr is missing: {source_archive}")
    if not video.is_file() or not recording_record.is_file():
        raise FileNotFoundError("Video reference or recording manifest is missing.")
    if type(crop_size) is not int or crop_size <= 0:
        raise ValueError("crop_size must be a positive exact integer.")
    if type(allow_initialize_missing_source_keys) is not bool:
        raise TypeError("allow_initialize_missing_source_keys must be an exact bool.")
    recording_document = _strict_json(recording_record)
    manifest_recording_identity = recording_document.get(
        "session_uuid", recording_document.get("recording_id")
    )
    if manifest_recording_identity != recording_identity:
        raise ValueError("Recording manifest identity differs from the canary source.")
    if str(recording_document.get("camera_id")) != camera_identity:
        raise ValueError("Recording manifest camera differs from the canary source.")
    manifest_files = recording_document.get("files")
    camera_files = (
        manifest_files.get("cams") if isinstance(manifest_files, Mapping) else None
    )
    if not isinstance(camera_files, list) or video.name not in {
        Path(str(value)).name for value in camera_files
    }:
        raise ValueError("Recording manifest does not bind the source video.")

    detect_source = source_archive / "detect_runs" / legacy_detect_run
    refined_source = source_archive / "refined_detect_runs" / legacy_refined_run
    source_metadata_paths = (
        source_archive / "zarr.json",
        detect_source / "zarr.json",
        refined_source / "zarr.json",
        recording_record,
    )
    if not all(path.is_file() for path in source_metadata_paths):
        raise FileNotFoundError("One or more source metadata declarations are missing.")
    source_metadata_before = {
        str(path): sha256_file(path) for path in source_metadata_paths
    }
    video_stat_before = (int(video.stat().st_size), int(video.stat().st_mtime_ns))

    session = Path(
        tempfile.mkdtemp(prefix="coordinate-catalog-canary-", dir=local_root)
    )
    package = session / "package"
    package.mkdir()
    partial = final_path.with_name(f".{final_path.name}.partial-{uuid.uuid4().hex}")
    palette = _palette_provenance()
    renamed_to_final = False
    publication_complete = False
    try:
        canonical = publish_legacy_canonical_detection_shadow(
            source_group_path=detect_source,
            recording_identity=recording_identity,
            source_run_id=legacy_detect_run,
            destination=package / CANONICAL_ARTIFACT_NAME,
            run_id=canonical_run_id,
            shadow_root=package,
            coordinate_catalog=True,
        )
        refined_group = zarr.open_group(
            str(refined_source), mode="r", use_consolidated=False
        )
        detect_group = zarr.open_group(
            str(detect_source), mode="r", use_consolidated=False
        )
        transition = build_refined_detection_transition(
            refined_group,
            n_frames=canonical.dimensions.n_frames,
            source_width=canonical.dimensions.source_width,
            source_height=canonical.dimensions.source_height,
            recording_identity=recording_identity,
            source_detect_group=detect_group,
            allow_initialize_missing_source_keys=(allow_initialize_missing_source_keys),
        )
        refined_ids = np.asarray(
            transition.arrays["instances/refined_row_ids"], dtype=np.int64
        )
        next_refined_row_id = int(refined_ids.max()) + 1 if refined_ids.size else 0
        refined = publish_refined_detection_shadow(
            transition,
            destination=package / REFINED_ARTIFACT_NAME,
            run_id=refined_run_id,
            lineage=RefinedDetectionSnapshotLineage(
                lineage_id=str(
                    uuid.uuid5(
                        _UUID_NAMESPACE,
                        f"coordinate-catalog-lineage:{recording_identity}",
                    )
                ),
                snapshot_id=str(
                    uuid.uuid5(
                        _UUID_NAMESPACE,
                        f"coordinate-catalog-snapshot:{recording_identity}:{refined_run_id}",
                    )
                ),
                recording_identity=recording_identity,
                next_refined_row_id=next_refined_row_id,
            ),
            canonical_source=canonical,
            shadow_root=package,
            coordinate_catalog=True,
        )

        video_sha256 = sha256_file(video)
        recording_manifest_sha256 = sha256_file(recording_record)
        pixel_document = _pixel_authority_document(
            recording_identity=recording_identity,
            camera_identity=camera_identity,
            video_reference=video,
            video_sha256=video_sha256,
            recording_manifest=recording_record,
            recording_manifest_sha256=recording_manifest_sha256,
            n_frames=canonical.dimensions.n_frames,
            source_width=canonical.dimensions.source_width,
            source_height=canonical.dimensions.source_height,
        )
        pixel_document_digest = canonical_json_sha256(pixel_document)
        bound_refined = bind_refined_detection_crop_source(
            refined.output_path,
            run_id=refined_run_id,
            allow_selector_ineligible_benchmark=True,
        )
        crop_policy = CropGeometryPolicy(
            purpose="coordinate_catalog_crimson_canary",
            size_mode=CropSizeMode.FIXED_PER_RUN,
            fixed_size_wh=(crop_size, crop_size),
            padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
        )
        crop = publish_refined_crop_geometry_shadow(
            bound_refined,
            policy=crop_policy,
            pixel_authority=CropPixelAuthority(
                authority_id=f"source_video_sha256:{video_sha256}",
                authority_manifest_digest=pixel_document_digest,
                recording_identity=recording_identity,
                camera_identity=camera_identity,
                n_frames=canonical.dimensions.n_frames,
                source_width=canonical.dimensions.source_width,
                source_height=canonical.dimensions.source_height,
            ),
            destination=package / CROP_ARTIFACT_NAME,
            run_id=crop_run_id,
            shadow_root=package,
            coordinate_catalog=True,
        )
        local_errors = {
            "canonical": list(
                validate_canonical_detection_shadow_publication(canonical)
            ),
            "crop": list(validate_crop_geometry_shadow_publication(crop)),
        }
        if any(local_errors.values()):
            raise RuntimeError(f"Local canary validation failed: {local_errors}")

        samples = _conversion_samples(canonical, crop)
        final_artifact_paths = {
            "canonical": final_path / CANONICAL_ARTIFACT_NAME,
            "refined": final_path / REFINED_ARTIFACT_NAME,
            "crop": final_path / CROP_ARTIFACT_NAME,
        }
        artifact_evidence = {
            "canonical": _artifact_evidence(
                canonical.output_path,
                final_path=final_artifact_paths["canonical"],
                run_id=canonical.run_id,
                manifest=canonical.manifest,
                logical_content_digest=canonical.manifest["payload"]["logical_content"][
                    "digest"
                ],
                dimensions=canonical.dimensions.as_manifest(),
                receipt_name="shadow_publication_receipt.json",
            ),
            "refined": _artifact_evidence(
                refined.output_path,
                final_path=final_artifact_paths["refined"],
                run_id=refined.run_id,
                manifest=refined.manifest,
                logical_content_digest=refined.receipt["logical_content_digest"],
                dimensions=transition.dimensions.as_manifest(),
                receipt_name="shadow_publication_receipt.json",
            ),
            "crop": _artifact_evidence(
                crop.output_path,
                final_path=final_artifact_paths["crop"],
                run_id=crop.run_id,
                manifest=crop.manifest,
                logical_content_digest=crop.manifest["payload"]["logical_content"][
                    "digest"
                ],
                dimensions=crop.dimensions.as_manifest(),
                receipt_name="shadow_publication_receipt.json",
            ),
        }
        local_tree_hashes = {
            name: _tree_hashes(path)
            for name, path in {
                "canonical": canonical.output_path,
                "refined": refined.output_path,
                "crop": crop.output_path,
            }.items()
        }

        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package, partial)
        copied_tree_hashes = {
            name: _tree_hashes(partial / artifact_name)
            for name, artifact_name in {
                "canonical": CANONICAL_ARTIFACT_NAME,
                "refined": REFINED_ARTIFACT_NAME,
                "crop": CROP_ARTIFACT_NAME,
            }.items()
        }
        if copied_tree_hashes != local_tree_hashes:
            raise RuntimeError("Copied artifact trees differ from local publications.")

        staged_canonical_root = zarr.open_group(
            str(partial / CANONICAL_ARTIFACT_NAME),
            mode="r",
            use_consolidated=True,
        )
        staged_canonical_run = staged_canonical_root[f"detect_runs/{canonical.run_id}"]
        staged_canonical = CanonicalDetectionShadowPublication(
            output_path=partial / CANONICAL_ARTIFACT_NAME,
            run_id=canonical.run_id,
            dimensions=canonical.dimensions,
            plans=canonical.plans,
            manifest=dict(staged_canonical_run.attrs["run_manifest"]),
            arrays={
                path: staged_canonical_run[path]
                for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
            },
            receipt=_strict_json(
                partial / CANONICAL_ARTIFACT_NAME / "shadow_publication_receipt.json"
            ),
        )
        staged_canonical_errors = validate_canonical_detection_shadow_publication(
            staged_canonical
        )
        staged_refined = bind_refined_detection_crop_source(
            partial / REFINED_ARTIFACT_NAME,
            run_id=refined.run_id,
            allow_selector_ineligible_benchmark=True,
        )
        staged_crop_root = zarr.open_group(
            str(partial / CROP_ARTIFACT_NAME), mode="r", use_consolidated=True
        )
        staged_crop_run = staged_crop_root[f"crop_runs/{crop.run_id}"]
        staged_crop = CropGeometryShadowPublication(
            output_path=partial / CROP_ARTIFACT_NAME,
            run_id=crop.run_id,
            dimensions=crop.dimensions,
            plans=crop.plans,
            manifest=dict(staged_crop_run.attrs["run_manifest"]),
            arrays={
                path: staged_crop_run[path]
                for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths
            },
            source_manifest=staged_refined.manifest,
            source_arrays=staged_refined.arrays,
            receipt=_strict_json(
                partial / CROP_ARTIFACT_NAME / "shadow_publication_receipt.json"
            ),
        )
        staged_crop_errors = validate_crop_geometry_shadow_publication(staged_crop)
        if staged_canonical_errors or staged_crop_errors:
            raise RuntimeError(
                "Staged canary validation failed: "
                f"canonical={staged_canonical_errors}; crop={staged_crop_errors}"
            )

        source_metadata_after = {
            str(path): sha256_file(path) for path in source_metadata_paths
        }
        if source_metadata_after != source_metadata_before:
            raise RuntimeError("Source metadata changed during canary publication.")
        video_stat_after = (int(video.stat().st_size), int(video.stat().st_mtime_ns))
        if video_stat_after != video_stat_before:
            raise RuntimeError("Source video changed during canary publication.")

        payload: dict[str, object] = {
            "status": "complete",
            "created_at_utc": utc_now(),
            "purpose": "crimson_coordinate_catalog_archive_gate",
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "production_state_changes": [],
            "palette": palette,
            "crimson_review": {
                "commit": crimson_commit,
                "path": crimson_review_path,
                "sha256": crimson_review_sha256,
                "verdict": "accepted_for_selector_ineligible_canary",
            },
            "source": {
                "analysis_zarr": str(source_archive),
                "recording_identity": recording_identity,
                "camera_identity": camera_identity,
                "legacy_detect_run": legacy_detect_run,
                "legacy_refined_run": legacy_refined_run,
                "historical_identity_migration": {
                    "allow_initialize_missing_source_keys": (
                        allow_initialize_missing_source_keys
                    ),
                    "transition_identity_initializations": transition.report[
                        "identity_initializations"
                    ],
                },
                "open_mode": "read_only_direct_metadata",
                "pixel_authority_document": pixel_document,
                "pixel_authority_document_digest": pixel_document_digest,
                "source_metadata_sha256_before_and_after": source_metadata_before,
            },
            "artifacts": artifact_evidence,
            "crop_policy": crop_policy.as_manifest(),
            "lineage_bindings": {
                "refined_source_canonical_manifest_digest": refined.receipt[
                    "source_manifest_digest"
                ],
                "crop_source_refined_manifest_digest": crop.manifest["payload"][
                    "source_refined_snapshot"
                ]["run_manifest_digest"],
                "crop_pixel_authority_manifest_digest": crop.manifest["payload"][
                    "source_pixel_authority"
                ]["authority_manifest_digest"],
                "crop_policy_digest": crop_policy.payload_digest,
            },
            "coordinate_samples": samples,
            "frame_cardinality": {
                "canonical": _frame_cardinality(
                    canonical.arrays["instances/frame_row_offsets"]
                ),
                "refined_instances": _frame_cardinality(
                    bound_refined.arrays["instances/frame_row_offsets"]
                ),
                "refined_source_detections": _frame_cardinality(
                    bound_refined.arrays["source_detections/frame_row_offsets"]
                ),
            },
            "validation": {
                "local_publication_errors": local_errors,
                "copied_artifact_tree_equality": True,
                "staged_canonical_errors": [],
                "staged_refined_full_publication_valid": True,
                "staged_crop_errors": [],
                "direct_consolidated_metadata_equivalence": True,
                "exact_coordinate_catalog_digests": True,
                "source_metadata_unchanged": True,
                "source_video_stat_unchanged": True,
                "strict_json": True,
            },
            "publication": {
                "local_scratch_materialization": True,
                "copy_back_then_atomic_directory_rename": True,
                "final_server_path": str(final_path),
                "final_macos_path": str(final_path).replace(
                    "/groups/johnson/johnsonlab",
                    "/Volumes/johnsonlab",
                    1,
                ),
            },
        }
        handoff = {
            "schema_id": CANARY_HANDOFF_SCHEMA_ID,
            "schema_version": CANARY_HANDOFF_SCHEMA_VERSION,
            "payload_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "payload_digest": canonical_json_sha256(payload),
            "payload": payload,
        }
        write_json_atomic(partial / HANDOFF_NAME, handoff)
        handoff_sha256 = sha256_file(partial / HANDOFF_NAME)
        partial.rename(final_path)
        renamed_to_final = True

        final_tree_hashes = {
            name: _tree_hashes(final_path / artifact_name)
            for name, artifact_name in {
                "canonical": CANONICAL_ARTIFACT_NAME,
                "refined": REFINED_ARTIFACT_NAME,
                "crop": CROP_ARTIFACT_NAME,
            }.items()
        }
        if final_tree_hashes != local_tree_hashes:
            raise RuntimeError("Final artifact trees differ after atomic rename.")
        if sha256_file(final_path / HANDOFF_NAME) != handoff_sha256:
            raise RuntimeError("Final handoff digest differs after atomic rename.")
        publication_complete = True
        return {
            "status": "complete",
            "destination": str(final_path),
            "macos_path": str(final_path).replace(
                "/groups/johnson/johnsonlab",
                "/Volumes/johnsonlab",
                1,
            ),
            "handoff_manifest": str(final_path / HANDOFF_NAME),
            "handoff_sha256": handoff_sha256,
            "payload_digest": handoff["payload_digest"],
            "artifacts": artifact_evidence,
        }
    finally:
        if partial.exists():
            shutil.rmtree(partial)
        if renamed_to_final and not publication_complete and final_path.exists():
            shutil.rmtree(final_path)
        if session.exists():
            shutil.rmtree(session)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-analysis-zarr", type=Path, required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--camera-identity", required=True)
    parser.add_argument("--video-reference", type=Path, required=True)
    parser.add_argument("--recording-manifest", type=Path, required=True)
    parser.add_argument("--legacy-detect-run", required=True)
    parser.add_argument("--legacy-refined-run", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--canonical-run-id", required=True)
    parser.add_argument("--refined-run-id", required=True)
    parser.add_argument("--crop-run-id", required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument(
        "--allow-initialize-missing-source-keys",
        action="store_true",
        help=(
            "Explicit historical migration for source-audit tables that predate "
            "durable instance keys. Modern sources must not need this flag."
        ),
    )
    parser.add_argument("--crimson-commit", required=True)
    parser.add_argument("--crimson-review-path", required=True)
    parser.add_argument("--crimson-review-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = publish_coordinate_catalog_canary(
        source_analysis_zarr=args.source_analysis_zarr,
        recording_identity=args.recording_identity,
        camera_identity=args.camera_identity,
        video_reference=args.video_reference,
        recording_manifest=args.recording_manifest,
        legacy_detect_run=args.legacy_detect_run,
        legacy_refined_run=args.legacy_refined_run,
        destination=args.destination,
        work_root=args.work_root,
        canonical_run_id=args.canonical_run_id,
        refined_run_id=args.refined_run_id,
        crop_run_id=args.crop_run_id,
        crop_size=args.crop_size,
        allow_initialize_missing_source_keys=(
            args.allow_initialize_missing_source_keys
        ),
        crimson_commit=args.crimson_commit,
        crimson_review_path=args.crimson_review_path,
        crimson_review_sha256=args.crimson_review_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
