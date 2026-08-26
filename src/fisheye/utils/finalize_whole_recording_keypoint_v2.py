#!/usr/bin/env python3
"""Finalize one terminal whole-recording result into production candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.shared.hybrid_crop_provider import (
    validate_hybrid_provider_strict_crop_geometry,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.keypoint_terminal_pixel_evidence import (
    DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE,
    validate_direct_hybrid_terminal_pixel_evidence,
)
from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.clipped_keypoint_finalization import (
    clip_terminal_result_from_yolo_arrays,
    publish_selector_ineligible_clipped_keypoint_chain,
)
from fisheye.shared.zarr.crop_shadow import (
    CropGeometryShadowPublication,
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_SIGNED_PROFILE,
    build_crop_run_reference,
    validate_crop_run_reference,
)
from fisheye.shared.zarr.keypoint_bundle_production_publication import (
    publish_keypoint_v2_production_candidate_chain,
)
from fisheye.shared.zarr.keypoint_manifest import (
    keypoint_preprocessing_from_manifest,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
    KeypointChainPublicationDispositions,
    KeypointPublicationDisposition,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    initial_refined_keypoint_snapshot_identity,
)
from fisheye.shared.zarr_run_completion import is_run_complete
from fisheye.utils.run_whole_recording_keypoint_terminal import (
    TERMINAL_RECEIPT_NAME,
    WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID,
    WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_VERSION,
)


WHOLE_RECORDING_KEYPOINT_FINALIZATION_SCHEMA_ID = (
    "palette.keypoint.whole_recording_production_finalization"
)
WHOLE_RECORDING_KEYPOINT_FINALIZATION_SCHEMA_VERSION = 1
DIRECT_HYBRID_TERMINAL_RECEIPT_SCHEMA_VERSION = 2
_SOURCE_ARRAY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "source_acquisition_frame_index",
    "frame_indices",
    "keypoints_roi",
    "keypoints_img",
    "keypoint_confidences",
    "confidence",
    "pose_bbox_xyxy_roi",
    "pose_bbox_xyxy_img",
    "detection_success",
    "pose_failure_codes",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _require_node_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not (str(resolved).startswith("/scratch/") or str(resolved).startswith("/tmp/")):
        raise ValueError("scratch_root must be under /scratch or /tmp.")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_terminal(
    terminal_artifact: Path,
    *,
    expected_analysis_zarr: Path,
    expected_crop_run: str,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    artifact = terminal_artifact.expanduser().resolve()
    receipt = _read_json(artifact / TERMINAL_RECEIPT_NAME)
    receipt_version = receipt.get("schema_version")
    if (
        receipt.get("schema_id") != WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID
        or receipt_version
        not in (
            WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_VERSION,
            DIRECT_HYBRID_TERMINAL_RECEIPT_SCHEMA_VERSION,
        )
    ):
        raise ValueError("Terminal receipt schema mismatch.")
    payload = receipt.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Terminal receipt lacks its payload.")
    if receipt.get("payload_digest") != canonical_json_sha256(payload):
        raise ValueError("Terminal receipt payload digest mismatch.")
    if payload.get("status") != "complete":
        raise ValueError("Terminal receipt is not complete.")
    if Path(payload.get("analysis_zarr", "")).expanduser().resolve() != (
        expected_analysis_zarr
    ):
        raise ValueError("Terminal receipt binds a different analysis archive.")
    if payload.get("crop_run") != expected_crop_run:
        raise ValueError("Terminal receipt binds a different crop run.")
    if receipt_version == WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_VERSION:
        cache = payload.get("cache")
        if not isinstance(cache, dict):
            raise ValueError("Terminal receipt lacks cache evidence.")
        cache_manifest = Path(cache.get("manifest_path", "")).expanduser().resolve()
        if not cache_manifest.is_file() or _sha256_file(cache_manifest) != cache.get(
            "manifest_sha256"
        ):
            raise ValueError("Terminal cache manifest changed after inference.")
    else:
        evidence = validate_direct_hybrid_terminal_pixel_evidence(
            payload.get("pixel_evidence")
        )
        if evidence.get("provider_run") != expected_crop_run:
            raise ValueError("Terminal pixel evidence binds a different crop provider.")
    root = zarr.open_group(
        str(artifact), mode="r", zarr_format=3, use_consolidated=True
    )
    run_id = str(payload.get("terminal_run_id") or "")
    run = root[f"keypoint_terminal_runs/{run_id}"]
    if (
        not is_run_complete(run)
        or run.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Terminal run is not complete and selector-ineligible.")
    expected_hashes = payload.get("source_array_hashes")
    observed_hashes = {
        path: sha256_array(run[path][...]) for path in _SOURCE_ARRAY_PATHS
    }
    if expected_hashes != observed_hashes:
        raise ValueError("Terminal arrays changed after their receipt was sealed.")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("Terminal receipt lacks model evidence.")
    return receipt, run, model


def _terminal_input_evidence_digest(receipt: Mapping[str, Any]) -> str:
    """Return the sealed input identity used by clipped finalization.

    The downstream clipped-finalization field retains its historical
    ``input_package_manifest_digest`` name. Cache-backed terminals bind the
    cache manifest bytes, while direct-hybrid terminals bind their complete
    typed pixel/geometry/shard evidence object instead of inventing a cache.
    """

    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Terminal receipt lacks its payload.")
    if receipt.get("schema_version") == DIRECT_HYBRID_TERMINAL_RECEIPT_SCHEMA_VERSION:
        evidence = validate_direct_hybrid_terminal_pixel_evidence(
            payload.get("pixel_evidence")
        )
        return canonical_json_sha256(evidence)
    cache = payload.get("cache")
    if not isinstance(cache, Mapping):
        raise ValueError("Terminal receipt lacks cache evidence.")
    digest = str(cache.get("manifest_sha256") or "").strip()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("Terminal cache manifest digest is invalid.")
    return digest


def _dispositions(
    *,
    analysis_zarr: Path,
    terminal_receipt_digest: str,
    run_ids: Mapping[str, str],
) -> KeypointChainPublicationDispositions:
    def one(stage: str) -> KeypointPublicationDisposition:
        provenance = build_run_provenance(
            command="fisheye.utils.finalize_whole_recording_keypoint_v2",
            params={
                "stage": stage,
                "analysis_zarr": str(analysis_zarr),
                "terminal_receipt_digest": terminal_receipt_digest,
                "run_id": run_ids[stage],
                "selector_activation": "deferred",
            },
            input_run_ids={"terminal_receipt": terminal_receipt_digest},
            cwd=Path.cwd(),
        )
        return KeypointPublicationDisposition(
            mode=KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
            publication_owner_uuid=uuid.uuid4().hex,
            run_provenance=provenance,
        )

    return KeypointChainPublicationDispositions(
        raw=one("raw"),
        quality=one("quality"),
        refined=one("refined"),
        body_frame=one("body_frame"),
    )


def _require_terminal_crop_provider_compatible(
    *,
    archive: Path,
    crop: CropGeometryShadowPublication,
    terminal_crop_run: str,
    terminal_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind a terminal pixel provider to the strict crop-v2 geometry authority.

    Whole-recording inference may read pixels from a signed hybrid acquisition /
    offline provider while publication remains bound to the independently sealed
    crop-v2 geometry run.  The two are interchangeable only for this terminal
    result when their complete ordered geometry rowsets agree exactly and the
    terminal cache receipt binds the live signed provider identity.
    """

    provider_run = str(terminal_crop_run).strip()
    if not provider_run or "/" in provider_run:
        raise ValueError("terminal_crop_run must be one nonempty archive group name.")
    if provider_run == crop.run_id:
        return {
            "mode": "strict_crop_v2_pixels",
            "geometry_crop_run": crop.run_id,
            "terminal_crop_run": provider_run,
            "ordered_geometry_coverage_exact": True,
        }

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    try:
        provider = root[f"crop_runs/{provider_run}"]
    except KeyError as exc:
        raise FileNotFoundError(
            f"Terminal crop provider not found: crop_runs/{provider_run}"
        ) from exc
    reference = validate_crop_run_reference(
        build_crop_run_reference(provider, run_id=provider_run)
    )
    if reference.get("profile") != CROP_RUN_REFERENCE_SIGNED_PROFILE:
        raise ValueError(
            "A separate terminal crop provider must use the signed-current profile."
        )

    payload = terminal_receipt.get("payload")
    receipt_version = terminal_receipt.get("schema_version")
    pixel_evidence: Mapping[str, Any] | None = None
    if receipt_version == DIRECT_HYBRID_TERMINAL_RECEIPT_SCHEMA_VERSION:
        if not isinstance(payload, Mapping):
            raise ValueError("Terminal receipt lacks its payload.")
        pixel_evidence = validate_direct_hybrid_terminal_pixel_evidence(
            payload.get("pixel_evidence")
        )
        if (
            pixel_evidence.get("profile")
            != DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE
            or pixel_evidence.get("provider_run") != provider_run
            or pixel_evidence.get("geometry_crop_run") != crop.run_id
            or pixel_evidence.get("geometry_crop_manifest_digest")
            != canonical_json_sha256(crop.manifest)
        ):
            raise ValueError(
                "Direct-hybrid terminal evidence binds different provider or geometry."
            )
        observed_reference = pixel_evidence.get("provider_reference")
    else:
        cache = payload.get("cache") if isinstance(payload, Mapping) else None
        if not isinstance(cache, Mapping):
            raise ValueError("Terminal receipt lacks cache evidence.")
        cache_manifest_path = Path(
            str(cache.get("manifest_path") or "")
        ).expanduser().resolve()
        cache_manifest = _read_json(cache_manifest_path)
        cache_source = cache_manifest.get("source")
        observed_reference = (
            cache_source.get("crop_run_reference")
            if isinstance(cache_source, Mapping)
            else None
        )
    if observed_reference != reference:
        raise ValueError(
            "Terminal cache binds a different signed crop-provider reference."
        )

    provider_record_sha256 = str(provider.attrs.get("provider_record_sha256") or "")
    resolved = validate_hybrid_provider_strict_crop_geometry(
        provider,
        crop,
        expected_provider_record_sha256=provider_record_sha256,
    )
    if pixel_evidence is not None:
        observed_binding = pixel_evidence.get("provider_binding")
        expected_binding = {
            name: resolved[name]
            for name in (
                "provider_record_sha256",
                "source_pixel_fingerprint",
                "source_rowset_fingerprint",
                "source_row_signature_spec_digest",
            )
        }
        if observed_binding != expected_binding:
            raise ValueError(
                "Direct-hybrid terminal evidence differs from the live signed provider."
            )
    preprocessing = payload.get("preprocessing")
    document = (
        preprocessing.get("document")
        if isinstance(preprocessing, Mapping)
        else None
    )
    roi_provider = document.get("roi_provider") if isinstance(document, Mapping) else None
    expected_terminal_provider = {
        "crop_run": provider_run,
        "record_sha256": resolved["provider_record_sha256"],
        "source_pixel_fingerprint": resolved["source_pixel_fingerprint"],
        "source_rowset_fingerprint": resolved["source_rowset_fingerprint"],
        "source_row_signature_spec_digest": resolved[
            "source_row_signature_spec_digest"
        ],
    }
    if not isinstance(roi_provider, Mapping) or any(
        roi_provider.get(name) != value
        for name, value in expected_terminal_provider.items()
    ):
        raise ValueError(
            "Terminal preprocessing evidence differs from the live signed provider."
        )

    return {
        "mode": "signed_hybrid_pixels_with_strict_crop_v2_geometry",
        "geometry_crop_run": crop.run_id,
        "geometry_crop_manifest_digest": crop.manifest["payload_digest"],
        "terminal_crop_run": provider_run,
        "terminal_crop_reference": reference,
        "terminal_pixel_evidence_profile": (
            pixel_evidence.get("profile") if pixel_evidence is not None else None
        ),
        "provider_record_sha256": resolved["provider_record_sha256"],
        "source_pixel_fingerprint": resolved["source_pixel_fingerprint"],
        "source_rowset_fingerprint": resolved["source_rowset_fingerprint"],
        "ordered_geometry_coverage_exact": True,
        "exact_geometry_paths": list(resolved["exact_geometry_paths"]),
    }


def finalize_whole_recording_keypoint_v2(
    *,
    analysis_zarr: Path,
    crop_run: str,
    terminal_artifact: Path,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    recording_identity: str,
    refined_lineage_id: str,
    refined_snapshot_id: str,
    scratch_root: Path,
    result_json: Path,
    copy_backend: str = "python",
    terminal_crop_run: str | None = None,
) -> Mapping[str, Any]:
    archive = analysis_zarr.expanduser().resolve()
    resolved_terminal_crop_run = str(terminal_crop_run or crop_run).strip()
    receipt, terminal, model = _load_terminal(
        terminal_artifact,
        expected_analysis_zarr=archive,
        expected_crop_run=resolved_terminal_crop_run,
    )
    payload = receipt["payload"]
    input_evidence_digest = _terminal_input_evidence_digest(receipt)
    binding = model.get("pose_model_schema_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("Terminal receipt lacks an exact pose-model binding.")
    if model.get("pose_model_schema_binding_digest") != canonical_json_sha256(binding):
        raise ValueError("Terminal pose-model binding digest mismatch.")
    preprocessing_value = payload.get("preprocessing")
    if not isinstance(preprocessing_value, Mapping):
        raise ValueError("Terminal receipt lacks preprocessing evidence.")
    preprocessing = keypoint_preprocessing_from_manifest(preprocessing_value)
    crop = open_persisted_crop_geometry_publication(archive, run_id=crop_run)
    crop_provider_binding = _require_terminal_crop_provider_compatible(
        archive=archive,
        crop=crop,
        terminal_crop_run=resolved_terminal_crop_run,
        terminal_receipt=receipt,
    )
    crop_payload = crop.manifest.get("payload")
    crop_source = (
        crop_payload.get("source_refined_snapshot")
        if isinstance(crop_payload, Mapping)
        else None
    )
    if (
        not isinstance(crop_source, Mapping)
        or crop_source.get("recording_identity") != recording_identity
    ):
        raise ValueError(
            "Requested recording_identity differs from the crop-v2 authority."
        )
    source_crop_arrays = {
        "instance_key": crop.arrays["instance_key"],
        "source_crop_row_ids": np.arange(
            int(crop.arrays["instance_key"].shape[0]), dtype=np.int64
        ),
        "frame_indices": crop.arrays["frame_indices"],
        "roi_coordinates_full": crop.arrays["roi_coordinates_full"],
        "roi_sizes_full": crop.arrays["roi_sizes_full"],
    }
    yolo_arrays = {path: terminal[path] for path in _SOURCE_ARRAY_PATHS}
    result = clip_terminal_result_from_yolo_arrays(
        crop,
        yolo_arrays,
        source_crop_arrays=source_crop_arrays,
        clip_id="whole_recording",
        clip_index=0,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing,
        input_package_manifest_digest=input_evidence_digest,
    )
    identity = initial_refined_keypoint_snapshot_identity(
        recording_identity=recording_identity,
        lineage_id=refined_lineage_id,
        snapshot_id=refined_snapshot_id,
    )
    run_ids = {
        "raw": raw_run_id,
        "quality": quality_run_id,
        "refined": refined_run_id,
        "body_frame": body_frame_run_id,
    }
    dispositions = _dispositions(
        analysis_zarr=archive,
        terminal_receipt_digest=receipt["payload_digest"],
        run_ids=run_ids,
    )
    scratch = _require_node_scratch(scratch_root) / f"finalize_{uuid.uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        chain = publish_selector_ineligible_clipped_keypoint_chain(
            crop,
            (result,),
            pose_model_schema_binding=binding,
            preprocessing=preprocessing,
            bundle_root=scratch / "bundle",
            raw_run_id=raw_run_id,
            quality_run_id=quality_run_id,
            refined_run_id=refined_run_id,
            body_frame_run_id=body_frame_run_id,
            refined_identity=identity,
            created_by="whole_recording_keypoint_v2_producer",
            dispositions=dispositions,
        )
        publication = publish_keypoint_v2_production_candidate_chain(
            analysis_zarr=archive,
            chain=chain,
            copy_backend=copy_backend,
        )
        final = {
            "schema_id": WHOLE_RECORDING_KEYPOINT_FINALIZATION_SCHEMA_ID,
            "schema_version": WHOLE_RECORDING_KEYPOINT_FINALIZATION_SCHEMA_VERSION,
            "status": "complete",
            "analysis_zarr": str(archive),
            "terminal_artifact": str(terminal_artifact.expanduser().resolve()),
            "terminal_receipt_digest": receipt["payload_digest"],
            "crop_provider_binding": crop_provider_binding,
            "runs": publication["runs"],
            "publication": publication,
            "selector_eligible": False,
            "selector_activation": "deferred_separate_reviewed_change",
            "registry_updated": False,
        }
        write_json_atomic(result_json.expanduser().resolve(), final)
        return final
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument(
        "--terminal-crop-run",
        help=(
            "Signed pixel-provider crop run bound by the terminal receipt. "
            "Defaults to the strict --crop-run authority."
        ),
    )
    parser.add_argument("--terminal-artifact", type=Path, required=True)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--body-frame-run", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--refined-lineage-id", required=True)
    parser.add_argument("--refined-snapshot-id", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = finalize_whole_recording_keypoint_v2(
        analysis_zarr=args.analysis_zarr,
        crop_run=args.crop_run,
        terminal_artifact=args.terminal_artifact,
        raw_run_id=args.raw_run,
        quality_run_id=args.quality_run,
        refined_run_id=args.refined_run,
        body_frame_run_id=args.body_frame_run,
        recording_identity=args.recording_identity,
        refined_lineage_id=args.refined_lineage_id,
        refined_snapshot_id=args.refined_snapshot_id,
        scratch_root=args.scratch_root,
        result_json=args.result_json,
        copy_backend=args.copy_backend,
        terminal_crop_run=args.terminal_crop_run,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
