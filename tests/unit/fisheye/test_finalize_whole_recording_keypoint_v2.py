from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.keypoint_terminal_pixel_evidence import (
    build_direct_hybrid_terminal_pixel_evidence,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.keypoint_manifest import KeypointPreprocessingReference
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.utils import finalize_whole_recording_keypoint_v2 as mod
from fisheye.utils.run_whole_recording_keypoint_terminal import (
    TERMINAL_RECEIPT_NAME,
    WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID,
    WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_VERSION,
)
from tests.unit.fisheye.test_clipped_keypoint_finalization import _crop
from tests.unit.fisheye.test_keypoint_publication import _pose_binding


class _FakeCropProvider:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.attrs = {
            "provider_record_sha256": "a" * 64,
            "source_refined_run_id": "refined_v1",
        }
        self._arrays = arrays

    def __contains__(self, name: str) -> bool:
        return name in self._arrays

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]


def test_terminal_hybrid_provider_must_exactly_match_crop_v2_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays = {
        "instance_key": np.asarray([11, 12], dtype=np.uint64),
        "source_refined_row_ids": np.asarray([3, 4], dtype=np.int64),
        "frame_indices": np.asarray([5, 6], dtype=np.int64),
        "source_acquisition_frame_index": np.asarray([5, 6], dtype=np.int64),
        "roi_coordinates_full": np.asarray([[10, 20], [30, 40]], dtype=np.int32),
        "roi_sizes_full": np.asarray([[384, 384], [384, 384]], dtype=np.int32),
    }
    provider = _FakeCropProvider(arrays)
    monkeypatch.setattr(
        mod.zarr,
        "open_group",
        lambda *_a, **_k: {"crop_runs/provider_v1": provider},
    )
    reference = {
        "schema_id": "palette.crop_geometry.run_reference",
        "schema_version": 1,
        "profile": "signed_current_source_v1",
        "run_id": "provider_v1",
        "crop_signature": {"provider": "test"},
        "crop_revision": 1,
    }
    monkeypatch.setattr(mod, "build_crop_run_reference", lambda *_a, **_k: reference)
    monkeypatch.setattr(mod, "validate_crop_run_reference", lambda value: value)
    signed = {
        "provider_record_sha256": "a" * 64,
        "source_pixel_fingerprint": "b" * 64,
        "source_rowset_fingerprint": "c" * 64,
        "source_row_signature_spec_digest": "d" * 64,
    }
    monkeypatch.setattr(
        mod,
        "validate_hybrid_provider_strict_crop_geometry",
        lambda *_a, **_k: {
            **signed,
            "exact_geometry_paths": list(arrays),
        },
    )
    cache_manifest = tmp_path / "cache.json"
    write_json_atomic(
        cache_manifest,
        {"source": {"crop_run_reference": reference}},
    )
    terminal_receipt = {
        "payload": {
            "cache": {"manifest_path": str(cache_manifest)},
            "preprocessing": {
                "document": {
                    "roi_provider": {
                        "crop_run": "provider_v1",
                        "record_sha256": "a" * 64,
                        "source_pixel_fingerprint": "b" * 64,
                        "source_rowset_fingerprint": "c" * 64,
                        "source_row_signature_spec_digest": "d" * 64,
                    }
                }
            },
        }
    }
    crop = SimpleNamespace(
        run_id="crop_v2",
        arrays={name: np.array(values, copy=True) for name, values in arrays.items()},
        manifest={
            "payload_digest": "e" * 64,
            "payload": {"source_refined_snapshot": {"run_id": "refined_v1"}},
        },
    )

    binding = mod._require_terminal_crop_provider_compatible(
        archive=tmp_path,
        crop=crop,
        terminal_crop_run="provider_v1",
        terminal_receipt=terminal_receipt,
    )
    assert binding["mode"] == (
        "signed_hybrid_pixels_with_strict_crop_v2_geometry"
    )
    assert binding["ordered_geometry_coverage_exact"] is True

    monkeypatch.setattr(
        mod,
        "validate_hybrid_provider_strict_crop_geometry",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ValueError(
                "Hybrid pixel provider differs from crop-v2 geometry at: "
                "roi_coordinates_full"
            )
        ),
    )
    with pytest.raises(ValueError, match="roi_coordinates_full"):
        mod._require_terminal_crop_provider_compatible(
            archive=tmp_path,
            crop=crop,
            terminal_crop_run="provider_v1",
            terminal_receipt=terminal_receipt,
        )


def test_terminal_direct_hybrid_evidence_needs_no_cache_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays = {
        "instance_key": np.asarray([11, 12], dtype=np.uint64),
        "source_refined_row_ids": np.asarray([3, 4], dtype=np.int64),
        "frame_indices": np.asarray([5, 6], dtype=np.int64),
        "source_acquisition_frame_index": np.asarray([5, 6], dtype=np.int64),
        "roi_coordinates_full": np.asarray([[10, 20], [30, 40]], dtype=np.int32),
        "roi_sizes_full": np.asarray([[384, 384], [384, 384]], dtype=np.int32),
    }
    provider = _FakeCropProvider(arrays)
    monkeypatch.setattr(
        mod.zarr,
        "open_group",
        lambda *_a, **_k: {"crop_runs/provider_v1": provider},
    )
    reference = {
        "schema_id": "palette.crop_geometry.run_reference",
        "schema_version": 1,
        "profile": "signed_current_source_v1",
        "run_id": "provider_v1",
        "crop_signature": {"provider": "test"},
        "crop_revision": 1,
    }
    monkeypatch.setattr(mod, "build_crop_run_reference", lambda *_a, **_k: reference)
    monkeypatch.setattr(mod, "validate_crop_run_reference", lambda value: value)
    signed = {
        "provider_record_sha256": "a" * 64,
        "source_pixel_fingerprint": "b" * 64,
        "source_rowset_fingerprint": "c" * 64,
        "source_row_signature_spec_digest": "d" * 64,
    }
    monkeypatch.setattr(
        mod,
        "validate_hybrid_provider_strict_crop_geometry",
        lambda *_a, **_k: {
            **signed,
            "exact_geometry_paths": list(arrays),
        },
    )
    crop = SimpleNamespace(
        run_id="crop_v2",
        arrays={name: np.array(values, copy=True) for name, values in arrays.items()},
        manifest={
            "payload_digest": "e" * 64,
            "payload": {"source_refined_snapshot": {"run_id": "refined_v1"}},
        },
    )
    evidence = build_direct_hybrid_terminal_pixel_evidence(
        provider_run="provider_v1",
        provider_reference=reference,
        provider_binding=signed,
        geometry_crop_run="crop_v2",
        geometry_crop_manifest_digest=canonical_json_sha256(crop.manifest),
        source_shard_runs=("shard_0",),
        source_shard_evidence_digest="f" * 64,
    )
    terminal_receipt = {
        "schema_version": mod.DIRECT_HYBRID_TERMINAL_RECEIPT_SCHEMA_VERSION,
        "payload": {
            "pixel_evidence": evidence,
            "preprocessing": {
                "document": {
                    "roi_provider": {
                        "crop_run": "provider_v1",
                        "record_sha256": "a" * 64,
                        "source_pixel_fingerprint": "b" * 64,
                        "source_rowset_fingerprint": "c" * 64,
                        "source_row_signature_spec_digest": "d" * 64,
                    }
                }
            },
        },
    }

    binding = mod._require_terminal_crop_provider_compatible(
        archive=tmp_path,
        crop=crop,
        terminal_crop_run="provider_v1",
        terminal_receipt=terminal_receipt,
    )

    assert binding["terminal_pixel_evidence_profile"] == evidence["profile"]
    assert binding["ordered_geometry_coverage_exact"] is True


def test_finalizer_publishes_four_crop_bound_candidates_without_activation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop = _crop(tmp_path)
    archive = tmp_path / "recording_analysis.zarr"
    archive_root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    crop_parent = archive_root.create_group("crop_runs")
    crop_parent.attrs["palette_completion_epoch"] = COMPLETION_EPOCH_STRICT
    persisted_crop = crop_parent.create_group(crop.run_id)
    persisted_crop.attrs.update(
        {
            "status": "complete",
            "stage_selector_eligible": False,
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "run_manifest": crop.manifest,
        }
    )
    consolidate_metadata_capture_expected_warnings(archive)
    monkeypatch.setattr(
        mod, "open_persisted_crop_geometry_publication", lambda *_a, **_k: crop
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.clipped_keypoint_finalization."
        "validate_crop_geometry_shadow_publication",
        lambda publication: (),
    )

    keys = np.asarray(crop.arrays["instance_key"], dtype=np.uint64)
    frames = np.asarray(crop.arrays["frame_indices"], dtype=np.int64)
    origins = np.asarray(crop.arrays["roi_coordinates_full"], dtype=np.float64)
    points = np.asarray(
        [
            [[5, 10], [15, 5], [15, 15]],
            [[6, 11], [16, 6], [16, 16]],
            [[7, 12], [17, 7], [17, 17]],
            [[8, 13], [18, 8], [18, 18]],
        ],
        dtype=np.float64,
    )
    bbox = np.asarray([[1, 1, 19, 19]] * 4, dtype=np.float64)
    source_arrays = {
        "instance_key": keys,
        "source_crop_row_ids": np.arange(keys.size, dtype=np.int64),
        "source_acquisition_frame_index": np.asarray(
            crop.arrays["source_acquisition_frame_index"], dtype=np.int64
        ),
        "frame_indices": frames,
        "keypoints_roi": points,
        "keypoints_img": points + origins[:, None, :],
        "keypoint_confidences": np.full((keys.size, 3), 0.9, dtype=np.float64),
        "confidence": np.full(keys.size, 0.95, dtype=np.float64),
        "pose_bbox_xyxy_roi": bbox,
        "pose_bbox_xyxy_img": bbox + np.column_stack((origins, origins)),
        "detection_success": np.ones(keys.size, dtype=bool),
        "pose_failure_codes": np.zeros(keys.size, dtype=np.uint8),
    }
    terminal = tmp_path / "terminal.zarr"
    terminal_root = zarr.open_group(str(terminal), mode="w", zarr_format=3)
    run = terminal_root.create_group("keypoint_terminal_runs").create_group("terminal")
    run.attrs.update(
        {
            "status": "complete",
            "stage_selector_eligible": False,
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
        }
    )
    for name, values in source_arrays.items():
        run.create_array(name, data=values, chunks=values.shape)
    consolidate_metadata_capture_expected_warnings(terminal)

    cache_manifest = tmp_path / "cache.json"
    cache_manifest.write_text("{}\n", encoding="utf-8")
    binding = _pose_binding()
    preprocessing = KeypointPreprocessingReference(
        profile_id="yolo_pose_flat_cache_v1",
        profile_version=1,
        input_mode="flat_bin_node_scratch",
        document={"cache": "test"},
    ).as_manifest()
    payload = {
        "status": "complete",
        "analysis_zarr": str(archive.resolve()),
        "crop_run": crop.run_id,
        "terminal_run_id": "terminal",
        "cache": {
            "manifest_path": str(cache_manifest),
            "manifest_sha256": hashlib.sha256(cache_manifest.read_bytes()).hexdigest(),
        },
        "model": {
            "pose_model_schema_binding": binding,
            "pose_model_schema_binding_digest": canonical_json_sha256(binding),
        },
        "preprocessing": preprocessing,
        "source_array_hashes": {
            name: sha256_array(values) for name, values in source_arrays.items()
        },
    }
    receipt = {
        "schema_id": WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_ID,
        "schema_version": WHOLE_RECORDING_KEYPOINT_TERMINAL_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    write_json_atomic(terminal / TERMINAL_RECEIPT_NAME, receipt)
    result_json = tmp_path / "result.json"

    result = mod.finalize_whole_recording_keypoint_v2(
        analysis_zarr=archive,
        crop_run=crop.run_id,
        terminal_artifact=terminal,
        raw_run_id="raw_v2",
        quality_run_id="quality_v1",
        refined_run_id="refined_v2",
        body_frame_run_id="body_v1",
        recording_identity="keypoint_v2_canary",
        refined_lineage_id="33333333-3333-4333-8333-333333333333",
        refined_snapshot_id="44444444-4444-4444-8444-444444444444",
        scratch_root=tmp_path / "scratch",
        result_json=result_json,
    )

    assert result["status"] == "complete"
    assert result["selector_eligible"] is False
    assert result["registry_updated"] is False
    assert result_json.is_file()
    published = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    for path in (
        "keypoints_runs/raw_v2",
        "keypoint_quality_runs/quality_v1",
        "refined_keypoints_runs/refined_v2",
        "analysis/body_frame_runs/body_v1",
    ):
        assert published[path].attrs["production_candidate"] is True
        assert published[path].attrs["stage_selector_eligible"] is False
