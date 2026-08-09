"""Publish a benchmark-only crop-v2 archive for Crimson read measurements."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence
import uuid

from fisheye.shared.atomic_run_publisher import (
    TreeInventory,
    tree_inventory,
)
from fisheye.shared.import_source_fingerprint import (
    source_stat_fingerprint_attrs,
)
from fisheye.shared.import_video_metadata import (
    publish_external_video_acquisition_authority,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.source_video_metadata import build_source_video_metadata_v2
from fisheye.shared.zarr.benchmark_runtime import storage_stats, utc_now
from fisheye.shared.zarr.crop_manifest import validate_crop_publication
from fisheye.shared.zarr.crop_pixel_authority import (
    bind_refined_crop_source_pixel_authority,
)
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_shadow import crop_metadata_declaration_maps
from fisheye.shared.zarr.crop_snapshot_publication import (
    publish_crop_geometry_production_candidate,
)
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_detection_crop_source import (
    BoundRefinedDetectionCropSource,
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE,
    REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE,
    REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE,
    build_refined_detection_activation_candidate_manifest,
    build_refined_detection_authority_provenance,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)


CROP_READ_FIXTURE_SCHEMA_ID = "palette.crop_geometry.crimson_read_fixture"
CROP_READ_FIXTURE_SCHEMA_VERSION = 1
HANDOFF_NAME = "handoff_manifest.json"
ARCHIVE_NAME = "analysis.zarr"
_SELECTOR_ATTRIBUTES = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


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
    }


def _require_safe_destination(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if ".palette_benchmarks" not in resolved.parts:
        raise ValueError("Crop read fixture must be below .palette_benchmarks.")
    if resolved.suffix:
        raise ValueError("Crop read fixture destination must be a package directory.")
    if resolved.exists():
        raise FileExistsError(f"Crop read fixture already exists: {resolved}")
    return resolved


def _require_work_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Fixture work root does not exist: {resolved}")
    if not (
        resolved.is_relative_to(Path("/tmp").resolve())
        or ".palette_scratch" in resolved.parts
    ):
        raise ValueError("Fixture work root must be local /tmp or .palette_scratch.")
    return resolved


def _macos_path(path: Path) -> str | None:
    text = str(path)
    prefix = "/groups/johnson/johnsonlab"
    if text == prefix or text.startswith(f"{prefix}/"):
        return text.replace(prefix, "/Volumes/johnsonlab", 1)
    return None


def _inventory_equal(left: TreeInventory, right: TreeInventory) -> bool:
    return (
        left.files == right.files
        and left.inventory_sha256 == right.inventory_sha256
        and left.content_sha256 == right.content_sha256
    )


def _copy_tree_verified(source: Path, destination: Path) -> dict[str, object]:
    total_started = perf_counter()
    inventory_started = perf_counter()
    before = tree_inventory(source, hash_content=True)
    source_inventory_seconds = perf_counter() - inventory_started
    copy_started = perf_counter()
    shutil.copytree(source, destination)
    copy_seconds = perf_counter() - copy_started
    inventory_started = perf_counter()
    after = tree_inventory(destination, hash_content=True)
    destination_inventory_seconds = perf_counter() - inventory_started
    if not _inventory_equal(before, after):
        raise RuntimeError("Fixture seed copy differs from its source tree.")
    return {
        "source": before.to_json(),
        "destination": after.to_json(),
        "exact_tree_equality": True,
        "timing_seconds": {
            "source_inventory": source_inventory_seconds,
            "copy": copy_seconds,
            "destination_inventory": destination_inventory_seconds,
            "total": perf_counter() - total_started,
        },
    }


def _refined_recording_identity(source: BoundRefinedDetectionCropSource) -> str:
    return source.manifest["payload"]["snapshot_lineage"][
        "manual_instance_key_allocator"
    ]["recording_identity"]


def _source_video_metadata(
    *,
    source_video: Path,
    recording_path: Path,
    camera_identity: str,
    source: BoundRefinedDetectionCropSource,
    fps: float,
    codec: str,
    pixel_format: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dimensions = source.dimensions
    raw = {
        "source_path": str(source_video),
        "camera_id": camera_identity,
        "width": dimensions.source_width,
        "height": dimensions.source_height,
        "total_frames": dimensions.n_frames,
        "fps": float(fps),
        "duration_seconds": float(dimensions.n_frames / fps),
        "codec": codec,
        "pix_fmt": pixel_format,
    }
    fingerprint = source_stat_fingerprint_attrs(
        source_video,
        attr_prefix="source_video",
        extra={
            "codec": codec,
            "pix_fmt": pixel_format,
            "width": dimensions.source_width,
            "height": dimensions.source_height,
            "fps": float(fps),
            "frame_count": dimensions.n_frames,
        },
    )
    return (
        build_source_video_metadata_v2(
            raw,
            recording_path=recording_path,
            fingerprint_attrs=fingerprint,
        ),
        fingerprint,
    )


def _activate_benchmark_refined_authority(
    archive: Path,
    *,
    run_id: str,
    palette_commit: str,
) -> dict[str, object]:
    root = open_zarr_root(archive, mode="a")
    parent = root["refined_detect_runs"]
    run = parent[run_id]
    manifest = run.attrs[REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE]
    candidate = build_refined_detection_activation_candidate_manifest(manifest)
    run.attrs[REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE] = candidate
    authority = build_refined_detection_authority_provenance(
        run_id=run_id,
        run_manifest_digest=candidate["payload_digest"],
        approved_by="palette_benchmark_fixture",
        approved_at_utc=utc_now(),
        review_method="selector_ineligible_coordinate_canary_accepted_by_crimson",
        intended_use="analysis",
        git_sha=palette_commit,
        note=(
            "Benchmark-archive-local authority used only to exercise the strict "
            "crop publication binder."
        ),
    )
    parent.attrs[REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE] = run_id
    parent.attrs[REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE] = authority
    run.attrs["stage_selector_eligible"] = True
    consolidate_metadata_capture_expected_warnings(archive)
    return {
        "run_id": run_id,
        "activated_manifest_digest": candidate["payload_digest"],
        "authority_provenance": authority,
        "scope": "benchmark_archive_only",
    }


def _prepare_seed_archive(
    *,
    source_refined_zarr: Path,
    source_refined_run_id: str,
    local_archive: Path,
    source_video: Path,
    recording_path: Path,
    camera_identity: str,
    fps: float,
    codec: str,
    pixel_format: str,
    palette_commit: str,
) -> dict[str, object]:
    source = bind_refined_detection_crop_source(
        source_refined_zarr,
        run_id=source_refined_run_id,
        allow_selector_ineligible_benchmark=True,
    )
    copy_receipt = _copy_tree_verified(source_refined_zarr, local_archive)
    metadata, fingerprint = _source_video_metadata(
        source_video=source_video,
        recording_path=recording_path,
        camera_identity=camera_identity,
        source=source,
        fps=fps,
        codec=codec,
        pixel_format=pixel_format,
    )
    root = open_zarr_root(local_archive, mode="a")
    attrs = dict(root.attrs)
    attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "fixture_schema_id": CROP_READ_FIXTURE_SCHEMA_ID,
            "fixture_schema_version": CROP_READ_FIXTURE_SCHEMA_VERSION,
            "recording_id": _refined_recording_identity(source),
            "camera_id": camera_identity,
            "recording_path": str(recording_path),
            "source_video_path": str(source_video),
            "source_path": str(source_video),
            "source_video_metadata": metadata,
        }
    )
    root.attrs.put(attrs)
    raw = root.require_group("raw_video")
    if tuple(raw.array_keys()):
        raise RuntimeError("Benchmark seed unexpectedly contains raw-video arrays.")
    raw.attrs["source_path"] = str(source_video)
    acquisition = publish_external_video_acquisition_authority(root)
    activation = _activate_benchmark_refined_authority(
        local_archive,
        run_id=source_refined_run_id,
        palette_commit=palette_commit,
    )
    rebound = bind_refined_detection_crop_source(local_archive)
    pixels = bind_refined_crop_source_pixel_authority(
        rebound,
        expected_camera_identity=camera_identity,
    )
    pixels.assert_verified()
    return {
        "source_copy": copy_receipt,
        "source_refined_manifest_digest": source.manifest["payload_digest"],
        "source_refined_logical_content_digest": source.logical_content_digest,
        "source_video_metadata": metadata,
        "source_video_fingerprint": fingerprint,
        "acquisition_authority": acquisition,
        "refined_authority": activation,
        "pixel_authority_digest": pixels.binding_document_digest,
    }


def _validate_final_archive(
    archive: Path,
    *,
    crop_run_id: str,
    camera_identity: str,
) -> dict[str, object]:
    source = bind_refined_detection_crop_source(archive)
    pixels = bind_refined_crop_source_pixel_authority(
        source,
        expected_camera_identity=camera_identity,
    )
    pixels.assert_verified()
    root = open_zarr_root(archive, mode="r")
    if root.attrs.get("benchmark_only") is not True:
        raise RuntimeError("Crop read fixture lost benchmark-only root state.")
    if root.attrs.get("selector_eligible") is not False:
        raise RuntimeError("Crop read fixture root became selector-eligible.")
    family = root["crop_runs"]
    selected_by = [
        name for name in _SELECTOR_ATTRIBUTES if family.attrs.get(name) == crop_run_id
    ]
    if selected_by:
        raise RuntimeError(
            f"Crop fixture run is unexpectedly selected by {selected_by}."
        )
    run = family[crop_run_id]
    if run.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError("Crop fixture run became selector-eligible.")
    expected_completion = {
        RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
        RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
        RUN_NAME_ATTR: crop_run_id,
    }
    observed_completion = {name: run.attrs.get(name) for name in expected_completion}
    if observed_completion != expected_completion:
        raise RuntimeError(
            "Crop fixture completion envelope differs from the standard contract: "
            f"expected {expected_completion}, observed {observed_completion}."
        )
    if "roi_images" in run:
        raise RuntimeError(
            "Geometry-only crop fixture unexpectedly contains roi_images."
        )
    manifest = run.attrs["run_manifest"]
    logical_dimensions = manifest["payload"]["logical_schema"]["dimensions"]
    dimensions = CropDimensions(
        n_frames=logical_dimensions["n_frames"],
        n_instances=logical_dimensions["n_instances"],
        source_width=logical_dimensions["source_width"],
        source_height=logical_dimensions["source_height"],
    )
    profile = storage_profile_from_manifest(
        manifest["payload"]["storage_plan"]["storage_profile"]
    )
    plans = plan_crop_geometry_storage(dimensions, profile=profile)
    direct, consolidated = crop_metadata_declaration_maps(
        archive,
        run_id=crop_run_id,
        plans=plans,
    )
    arrays = {path: run[path] for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths}
    errors = validate_crop_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
        source_manifest=source.manifest,
        source_arrays=source.arrays,
    )
    if errors:
        raise RuntimeError("Final crop fixture validation failed: " + "; ".join(errors))
    if manifest["payload"]["source_pixel_authority"] != (
        pixels.pixel_authority.as_manifest()
    ):
        raise RuntimeError("Crop manifest differs from the live bound pixel authority.")
    return {
        "crop_run_manifest_digest": manifest["payload_digest"],
        "crop_logical_content_digest": manifest["payload"]["logical_content"]["digest"],
        "storage_profile_id": profile.profile_id,
        "array_count": len(arrays),
        "direct_consolidated_metadata_equal": True,
        "source_refined_authority_valid": True,
        "source_pixel_authority_valid": True,
        "frame_count": dimensions.n_frames,
        "row_count": dimensions.n_instances,
        "geometry_only": True,
        "selector_eligible": False,
        "completion_contract_valid": True,
    }


def _handoff_envelope(payload: Mapping[str, Any]) -> dict[str, object]:
    document = dict(payload)
    return {
        "schema_id": CROP_READ_FIXTURE_SCHEMA_ID,
        "schema_version": CROP_READ_FIXTURE_SCHEMA_VERSION,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(document),
        "payload": document,
    }


def publish_crop_geometry_read_fixture(
    *,
    source_refined_zarr: Path,
    source_refined_run_id: str,
    source_video: Path,
    recording_path: Path,
    camera_identity: str,
    fps: float,
    codec: str,
    pixel_format: str,
    destination: Path,
    work_root: Path,
    crop_run_id: str,
    policy: CropGeometryPolicy,
    copy_backend: str = "python",
) -> dict[str, object]:
    """Publish one complete benchmark package without production-state changes."""

    source_path = source_refined_zarr.expanduser().resolve()
    video_path = source_video.expanduser().resolve()
    recording_root = recording_path.expanduser().resolve()
    output = _require_safe_destination(destination)
    scratch = _require_work_root(work_root)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Refined seed archive not found: {source_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {video_path}")
    if not recording_root.is_dir():
        raise FileNotFoundError(f"Recording root not found: {recording_root}")
    if not video_path.is_relative_to(recording_root):
        raise ValueError("Source video must be contained by recording_path.")
    if type(fps) is not float or fps <= 0:
        raise ValueError("fps must be an exact positive float.")
    for value, label in (
        (source_refined_run_id, "source_refined_run_id"),
        (camera_identity, "camera_identity"),
        (codec, "codec"),
        (pixel_format, "pixel_format"),
        (crop_run_id, "crop_run_id"),
    ):
        if type(value) is not str or not value or value != value.strip():
            raise ValueError(f"{label} must be an exact nonempty string.")

    provenance = _palette_provenance()
    session = scratch / f"palette_crop_read_fixture_{uuid.uuid4().hex}"
    local_archive = session / ARCHIVE_NAME
    partial = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    partial_archive = partial / ARCHIVE_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    session.mkdir(parents=True, exist_ok=False)
    partial.mkdir(parents=False, exist_ok=False)
    success = False
    try:
        source_before = tree_inventory(source_path, hash_content=True)
        seed = _prepare_seed_archive(
            source_refined_zarr=source_path,
            source_refined_run_id=source_refined_run_id,
            local_archive=local_archive,
            source_video=video_path,
            recording_path=recording_root,
            camera_identity=camera_identity,
            fps=fps,
            codec=codec,
            pixel_format=pixel_format,
            palette_commit=str(provenance["commit"]),
        )
        seed_copy = _copy_tree_verified(local_archive, partial_archive)
        publication = publish_crop_geometry_production_candidate(
            analysis_zarr=partial_archive,
            run_id=crop_run_id,
            policy=policy,
            expected_camera_identity=camera_identity,
            scratch_root=scratch,
            copy_backend=copy_backend,
        )
        pre_rename_validation = _validate_final_archive(
            partial_archive,
            crop_run_id=crop_run_id,
            camera_identity=camera_identity,
        )
        source_after = tree_inventory(source_path, hash_content=True)
        if not _inventory_equal(source_before, source_after):
            raise RuntimeError(
                "Source refined seed changed during fixture publication."
            )
        analysis_inventory = tree_inventory(partial_archive, hash_content=True)
        final_archive = output / ARCHIVE_NAME
        payload = {
            "status": "complete",
            "created_at_utc": utc_now(),
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_updated": False,
            "production_state_changes": [],
            "palette": provenance,
            "source": {
                "refined_zarr": str(source_path),
                "refined_run_id": source_refined_run_id,
                "refined_tree": source_before.to_json(),
                "refined_source_unchanged": True,
                "video_path": str(video_path),
                "recording_path": str(recording_root),
                "camera_identity": camera_identity,
                "fps": fps,
                "codec": codec,
                "pixel_format": pixel_format,
            },
            "seed": seed,
            "seed_copy_to_shared_staging": seed_copy,
            "artifact": {
                "server_package_path": str(output),
                "server_archive_path": str(final_archive),
                "macos_package_path": _macos_path(output),
                "macos_archive_path": _macos_path(final_archive),
                "crop_run_id": crop_run_id,
                "crop_group_path": f"crop_runs/{crop_run_id}",
                "analysis_tree": analysis_inventory.to_json(),
                "storage_stats": storage_stats(partial_archive),
            },
            "publication": publication,
            "validation": pre_rename_validation,
            "handoff_policy": {
                "read_workload": "backend_neutral_crop_v2_13_array_v1",
                "profile_promotion_evidence": False,
                "reason": "representative_integration_fixture_not_full_duration",
            },
        }
        envelope = _handoff_envelope(payload)
        write_json_atomic(partial / HANDOFF_NAME, envelope)
        os.replace(partial, output)
        final_validation = _validate_final_archive(
            final_archive,
            crop_run_id=crop_run_id,
            camera_identity=camera_identity,
        )
        final_inventory = tree_inventory(final_archive, hash_content=True)
        if not _inventory_equal(analysis_inventory, final_inventory):
            raise RuntimeError("Final crop fixture tree changed during atomic rename.")
        persisted = json.loads((output / HANDOFF_NAME).read_text(encoding="utf-8"))
        if (
            persisted != envelope
            or persisted["payload_digest"]
            != canonical_json_sha256(persisted["payload"])
            or final_validation != pre_rename_validation
        ):
            raise RuntimeError("Final crop fixture handoff or validation changed.")
        success = True
        return envelope
    except BaseException as exc:
        failure_root = partial if partial.exists() else output
        if failure_root.exists():
            try:
                write_json_atomic(
                    failure_root / "publication_failure.json",
                    {
                        "schema_id": CROP_READ_FIXTURE_SCHEMA_ID,
                        "status": "failed",
                        "failed_at_utc": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            except Exception:
                pass
        raise
    finally:
        if success and session.exists():
            shutil.rmtree(session)


def _policy(crop_size: int) -> CropGeometryPolicy:
    return CropGeometryPolicy(
        purpose="subject_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(int(crop_size), int(crop_size)),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-refined-zarr", type=Path, required=True)
    parser.add_argument("--source-refined-run", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--recording-path", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--pixel-format", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--crop-run-id", required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = publish_crop_geometry_read_fixture(
        source_refined_zarr=args.source_refined_zarr,
        source_refined_run_id=args.source_refined_run,
        source_video=args.source_video,
        recording_path=args.recording_path,
        camera_identity=args.camera,
        fps=float(args.fps),
        codec=args.codec,
        pixel_format=args.pixel_format,
        destination=args.destination,
        work_root=args.work_root,
        crop_run_id=args.crop_run_id,
        policy=_policy(args.crop_size),
        copy_backend=args.copy_backend,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_NAME",
    "CROP_READ_FIXTURE_SCHEMA_ID",
    "CROP_READ_FIXTURE_SCHEMA_VERSION",
    "HANDOFF_NAME",
    "publish_crop_geometry_read_fixture",
]
