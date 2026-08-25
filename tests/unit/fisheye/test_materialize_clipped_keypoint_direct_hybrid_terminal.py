from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.keypoint_terminal_pixel_evidence import (
    build_direct_hybrid_terminal_pixel_evidence,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.keypoint_manifest import KeypointPreprocessingReference
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.utils import finalize_whole_recording_keypoint_v2 as finalizer
from fisheye.utils import materialize_clipped_keypoint_direct_hybrid_terminal as mod
from fisheye.utils.audit_clipped_keypoint_direct_hybrid import (
    REPORT_SCHEMA_ID,
    REPORT_SCHEMA_VERSION,
)
from tests.unit.fisheye.test_keypoint_publication import _pose_binding


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_direct_hybrid_writer_round_trips_through_unpatched_terminal_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    provider = root.create_group("crop_runs").create_group("provider_v1")
    provider.attrs["provider_record_sha256"] = "a" * 64
    shard = root.create_group("keypoint_shard_runs").create_group("shard_0")
    shard.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE
    arrays = {
        "instance_key": np.asarray([11, 12], dtype=np.uint64),
        "source_crop_row_ids": np.asarray([0, 1], dtype=np.int64),
        "source_acquisition_frame_index": np.asarray([5, 6], dtype=np.int64),
        "frame_indices": np.asarray([5, 6], dtype=np.int64),
        "keypoints_roi": np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            dtype=np.float64,
        ),
        "keypoints_img": np.asarray(
            [
                [[11.0, 22.0], [13.0, 24.0]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            dtype=np.float64,
        ),
        "keypoint_confidences": np.asarray(
            [[0.9, 0.8], [np.nan, np.nan]], dtype=np.float64
        ),
        "confidence": np.asarray([0.95, np.nan], dtype=np.float64),
        "pose_bbox_xyxy_roi": np.asarray(
            [[0.0, 0.0, 5.0, 6.0], [np.nan] * 4], dtype=np.float32
        ),
        "pose_bbox_xyxy_img": np.asarray(
            [[10.0, 20.0, 15.0, 26.0], [np.nan] * 4], dtype=np.float32
        ),
        "detection_success": np.asarray([True, False], dtype=bool),
        "pose_failure_codes": np.asarray([0, 1], dtype=np.uint8),
    }
    for name, values in arrays.items():
        shard.create_array(name, data=values, chunks=values.shape)

    reference = {
        "schema_id": "palette.crop_geometry.run_reference",
        "schema_version": 1,
        "profile": "signed_current_source_v1",
        "run_id": "provider_v1",
        "crop_signature": {"provider": "test"},
        "crop_revision": 1,
    }
    provider_binding = {
        "provider_record_sha256": "a" * 64,
        "source_pixel_fingerprint": "b" * 64,
        "source_rowset_fingerprint": "c" * 64,
        "source_row_signature_spec_digest": "d" * 64,
    }
    crop_manifest = {"payload_digest": "e" * 64, "payload": {}}
    crop = SimpleNamespace(
        run_id="crop_v2",
        manifest=crop_manifest,
        arrays={"instance_key": arrays["instance_key"]},
    )
    pixel_evidence = build_direct_hybrid_terminal_pixel_evidence(
        provider_run="provider_v1",
        provider_reference=reference,
        provider_binding=provider_binding,
        geometry_crop_run="crop_v2",
        geometry_crop_manifest_digest=canonical_json_sha256(crop_manifest),
        source_shard_runs=("shard_0",),
        source_shard_evidence_digest="f" * 64,
    )
    preprocessing = KeypointPreprocessingReference(
        profile_id="signed_hybrid_pixels_with_strict_crop_v2_geometry_v1",
        profile_version=1,
        input_mode="numpy_list",
        document={"observed_runtime": "fixture"},
    ).as_manifest()
    binding = _pose_binding()

    recovery_manifest = tmp_path / "recovery.json"
    write_json_atomic(
        recovery_manifest,
        {
            "targets": [
                {
                    "target_id": "camera",
                    "analysis_zarr": str(archive.resolve()),
                    "target_geometry_crop_run": "crop_v2",
                    "source_keypoint_shards": ["shard_0"],
                }
            ]
        },
    )
    target_proof = {
        "target_id": "camera",
        "status": "migratable",
        "terminal_pixel_evidence": pixel_evidence,
        "source_shard_evidence_digest": "f" * 64,
        "model": {
            "set_id": "set",
            "run_id": "run",
            "path": "/models/model.pt",
            "sha256": binding["model"]["sha256"],
            "pose_model_schema_binding": binding,
            "pose_model_schema_binding_digest": canonical_json_sha256(binding),
        },
        "preprocessing": preprocessing,
        "shards": [
            {
                "shard_run": "shard_0",
                "status": "migratable",
                "scientific_array_hashes": {
                    name: sha256_array(values) for name, values in arrays.items()
                },
            }
        ],
    }
    proof_payload = {
        "status": "migratable",
        "production_state_changes": [],
        "recovery_manifest": str(recovery_manifest.resolve()),
        "recovery_manifest_sha256": _sha256_file(recovery_manifest),
        "targets": [target_proof],
    }
    proof_report = tmp_path / "proof.json"
    write_json_atomic(
        proof_report,
        {
            "schema_id": REPORT_SCHEMA_ID,
            "schema_version": REPORT_SCHEMA_VERSION,
            "payload_digest": canonical_json_sha256(proof_payload),
            "payload": proof_payload,
        },
    )

    monkeypatch.setattr(
        mod,
        "open_persisted_crop_geometry_publication",
        lambda *_a, **_k: crop,
    )
    monkeypatch.setattr(mod, "build_crop_run_reference", lambda *_a, **_k: reference)
    monkeypatch.setattr(mod, "validate_crop_run_reference", lambda value: value)
    monkeypatch.setattr(
        mod,
        "validate_hybrid_provider_strict_crop_geometry",
        lambda *_a, **_k: {
            **provider_binding,
            "exact_geometry_paths": [],
        },
    )
    output = tmp_path / "terminal.zarr"
    result = mod.materialize_direct_hybrid_terminal(
        proof_report=proof_report,
        target_id="camera",
        terminal_run_id="terminal_v1",
        terminal_output=output,
    )

    assert result["status"] == "complete"
    receipt, terminal, model = finalizer._load_terminal(
        output,
        expected_analysis_zarr=archive.resolve(),
        expected_crop_run="provider_v1",
    )
    assert receipt["schema_version"] == 2
    assert terminal.attrs["stage_selector_eligible"] is False
    assert model["pose_model_schema_binding"] == binding
    np.testing.assert_array_equal(terminal["instance_key"][:], arrays["instance_key"])
