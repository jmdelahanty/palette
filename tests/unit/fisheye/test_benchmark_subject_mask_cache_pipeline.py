from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.diagnostics.benchmark_subject_mask_cache_pipeline import (
    _bound_run_manifest,
    _read_benchmark,
    _refined_arrays,
    _require_destination,
    _require_existing_node_local,
    _require_node_local,
    _resume_cache,
    _stage_cache,
    _validate_published_metadata_equivalence,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def test_pipeline_path_guards(tmp_path: Path) -> None:
    scratch = _require_node_local(tmp_path / "scratch")
    assert scratch.is_dir()
    with pytest.raises(FileExistsError, match="already exists"):
        _require_node_local(scratch)
    with pytest.raises(ValueError, match="node-local"):
        _require_node_local(Path("/groups/example/scratch"))
    assert _require_existing_node_local(scratch) == scratch
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _require_existing_node_local(tmp_path / "absent")

    root = tmp_path / "benchmarks"
    destination = _require_destination(root / "candidate", benchmark_root=root)
    assert destination == (root / "candidate").resolve()
    with pytest.raises(ValueError, match="below the benchmark root"):
        _require_destination(tmp_path / "outside", benchmark_root=root)


def test_stage_cache_rewrites_and_verifies_local_payload(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    payload_path = source / "cache.bin"
    payload_path.write_bytes(payload.tobytes(order="C"))
    payload_sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path = source / "cache.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "palette_roi_cache_flat_bin_v1",
                "layout": "flat_bin_v1",
                "cache_complete": True,
                "array": {
                    "bin_path": payload_path.name,
                    "dtype": "uint8",
                    "order": "C",
                    "shape": [2, 3, 4],
                    "sha256": payload_sha,
                },
            }
        ),
        encoding="utf-8",
    )

    local_manifest, receipt = _stage_cache(manifest_path, tmp_path / "staged")

    staged = json.loads(local_manifest.read_text(encoding="utf-8"))
    assert staged["array"]["bin_path"] == payload_path.name
    assert (
        local_manifest.parent / staged["array"]["bin_path"]
    ).read_bytes() == payload_path.read_bytes()
    assert receipt["payload_sha256"] == payload_sha
    resumed_manifest, resumed = _resume_cache(local_manifest)
    assert resumed_manifest == local_manifest
    assert resumed["resumed"] is True
    assert resumed["payload_sha256"] == payload_sha


def test_refined_adapter_binds_authoritative_crop_identity() -> None:
    frames = np.asarray([0, 0, 2], dtype=np.int64)
    crop = {
        "instance_key": np.asarray([11, 12, 21], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
        "source_crop_xywh": np.asarray(
            [[0, 0, 8, 8], [1, 0, 8, 8], [2, 0, 8, 8]],
            dtype=np.float32,
        ),
    }
    draft = {
        "source_crop_row_ids": np.arange(3, dtype=np.int64),
        "masks_roi": np.zeros((3, 4, 8, 8), dtype=np.uint8),
        "available_channels": np.ones((4,), dtype=bool),
        "metrics/mask_present": np.zeros((3, 4), dtype=bool),
        "metrics/area_px": np.zeros((3, 4), dtype=np.int32),
        "metrics/centroid_xy": np.zeros((3, 4, 2), dtype=np.float32),
        "metrics/centroid_valid": np.zeros((3, 4), dtype=bool),
        "metrics/bbox_xyxy": np.zeros((3, 4, 4), dtype=np.int32),
        "metrics/bbox_valid": np.zeros((3, 4), dtype=bool),
    }

    arrays = _refined_arrays(draft, crop, n_frames=3)

    np.testing.assert_array_equal(arrays["instance_key"], crop["instance_key"])
    np.testing.assert_array_equal(arrays["source_acquisition_frame_index"], frames)
    np.testing.assert_array_equal(
        arrays["frame_row_offsets"], np.asarray([0, 2, 2, 3], dtype=np.int64)
    )
    assert arrays["source_crop_xywh"] is crop["source_crop_xywh"]


def test_refined_adapter_rejects_incomplete_or_reordered_rows() -> None:
    crop = {
        "instance_key": np.asarray([11, 12], dtype=np.uint64),
        "source_acquisition_frame_index": np.asarray([0, 1], dtype=np.int64),
        "source_crop_xywh": np.zeros((2, 4), dtype=np.float32),
    }
    draft = {
        "source_crop_row_ids": np.asarray([1, 0], dtype=np.int64),
        "masks_roi": np.zeros((2, 4, 8, 8), dtype=np.uint8),
        "available_channels": np.ones((4,), dtype=bool),
        "metrics/mask_present": np.zeros((2, 4), dtype=bool),
        "metrics/area_px": np.zeros((2, 4), dtype=np.int32),
        "metrics/centroid_xy": np.zeros((2, 4, 2), dtype=np.float32),
        "metrics/centroid_valid": np.zeros((2, 4), dtype=bool),
        "metrics/bbox_xyxy": np.zeros((2, 4, 4), dtype=np.int32),
        "metrics/bbox_valid": np.zeros((2, 4), dtype=bool),
    }

    with pytest.raises(ValueError, match="complete crop rowset in order"):
        _refined_arrays(draft, crop, n_frames=2)


def test_bound_run_manifest_recomputes_payload_digest() -> None:
    payload = {"run_id": "crop_001", "stage": "crop"}

    class _Group:
        attrs = {
            "run_manifest": {
                "schema_id": "fixture",
                "schema_version": 1,
                "payload_digest": canonical_json_sha256(payload),
                "payload": payload,
            }
        }

    assert _bound_run_manifest(_Group(), label="fixture")["payload"] == payload
    _Group.attrs["run_manifest"]["payload"]["stage"] = "tampered"
    with pytest.raises(ValueError, match="payload digest is invalid"):
        _bound_run_manifest(_Group(), label="fixture")


def test_read_benchmark_opens_exact_local_runs_without_consolidated_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offsets = np.asarray([0, 2, 2, 3], dtype=np.int64)
    payloads = {
        "mask_probs_roi": np.zeros((3, 4, 2, 2), dtype=np.float32),
        "masks_roi": np.zeros((3, 4, 2, 2), dtype=np.uint8),
        "observation_quality_flags": np.zeros((3,), dtype=np.uint16),
    }
    opened: list[tuple[Path, str, bool]] = []

    def fake_open_group(
        path: str, *, mode: str, use_consolidated: bool
    ) -> dict[str, np.ndarray]:
        resolved = Path(path)
        opened.append((resolved, mode, use_consolidated))
        if "subject_mask_quality_runs" in resolved.parts:
            payload_name = "observation_quality_flags"
        elif "refined_subject_masks_runs" in resolved.parts:
            payload_name = "masks_roi"
        else:
            payload_name = "mask_probs_roi"
        return {
            "frame_row_offsets": offsets,
            payload_name: payloads[payload_name],
        }

    monkeypatch.setattr(
        "fisheye.diagnostics.benchmark_subject_mask_cache_pipeline.zarr.open_group",
        fake_open_group,
    )

    result = _read_benchmark(
        raw_store=tmp_path / "raw.zarr",
        raw_run="raw",
        refined_store=tmp_path / "refined.zarr",
        refined_run="refined",
        quality_store=tmp_path / "quality.zarr",
        quality_run="quality",
    )

    assert set(result) == {"raw", "refined", "quality"}
    assert opened == [
        (
            tmp_path / "raw.zarr" / "subject_mask_runs" / "raw",
            "r",
            False,
        ),
        (
            tmp_path / "refined.zarr" / "refined_subject_masks_runs" / "refined",
            "r",
            False,
        ),
        (
            tmp_path / "quality.zarr" / "subject_mask_quality_runs" / "quality",
            "r",
            False,
        ),
    ]


def test_read_benchmark_uses_exact_persisted_payload_names(tmp_path: Path) -> None:
    offsets = np.asarray([0, 2, 2, 3], dtype=np.int64)
    stores = (
        (
            tmp_path / "raw.zarr",
            "subject_mask_runs",
            "raw",
            "mask_probs_roi",
            np.zeros((3, 4, 2, 2), dtype=np.uint8),
        ),
        (
            tmp_path / "refined.zarr",
            "refined_subject_masks_runs",
            "refined",
            "masks_roi",
            np.zeros((3, 4, 2, 2), dtype=np.uint8),
        ),
        (
            tmp_path / "quality.zarr",
            "subject_mask_quality_runs",
            "quality",
            "observation_quality_flags",
            np.zeros((3,), dtype=np.uint16),
        ),
    )
    for store, family_name, run_name, payload_name, payload in stores:
        root = zarr.open_group(str(store), mode="w", zarr_format=3)
        run = root.create_group(family_name).create_group(run_name)
        run.create_array("frame_row_offsets", data=offsets)
        run.create_array(payload_name, data=payload)
        zarr.consolidate_metadata(str(store))

    result = _read_benchmark(
        raw_store=tmp_path / "raw.zarr",
        raw_run="raw",
        refined_store=tmp_path / "refined.zarr",
        refined_run="refined",
        quality_store=tmp_path / "quality.zarr",
        quality_run="quality",
    )

    assert set(result) == {"raw", "refined", "quality"}
    assert all(item["offset_reads"] == 1 for item in result.values())
    for store, family_name, run_name, _payload_name, _payload in stores:
        receipt = _validate_published_metadata_equivalence(
            store,
            run_path=f"{family_name}/{run_name}",
        )
        assert receipt["declaration_count"] == 3
