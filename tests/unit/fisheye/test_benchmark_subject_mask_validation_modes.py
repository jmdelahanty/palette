from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.diagnostics.benchmark_subject_mask_validation_modes import (
    CANARY_SCHEMA_ID,
    _io_delta,
    _require_new_node_local_scratch,
    _validate_handoff,
    run_canary,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_core_publication import (
    publish_selector_ineligible_subject_mask_core_snapshot,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    derive_subject_mask_frame_row_offsets,
    derive_subject_mask_metrics,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _fixture() -> (
    tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]
):
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    crop = {
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
        "source_crop_xywh": np.asarray(
            [[0, 0, 8, 8], [1, 0, 8, 8], [0, 1, 8, 8], [1, 1, 8, 8]],
            dtype=np.float32,
        ),
    }
    common = {
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        **crop,
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(frames, n_frames=4),
    }
    raw_masks = np.zeros((4, 3, 8, 8), dtype=np.uint8)
    raw_masks[:, 0, 1:7, 1:7] = 1
    raw_masks[:, 1, 2, 2] = 1
    raw_masks[:, 2, 5, 3] = 1
    raw_metrics = derive_subject_mask_metrics(raw_masks)
    probabilities = raw_masks * np.uint8(255)
    raw = {
        **common,
        "mask_probs_roi": probabilities,
        "available_channels": np.ones((3,), dtype=bool),
        "metrics/prob_max": np.max(
            probabilities.astype(np.float32) / np.float32(255.0), axis=(2, 3)
        ).astype(np.float32),
        **{f"metrics/{name}": values for name, values in raw_metrics.items()},
    }
    refined_masks = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    refined_masks[:, 0, 1:7, 1:7] = 1
    refined_masks[:, 1, 2, 2] = 1
    refined_masks[:, 2, 2, 5] = 1
    refined_masks[:, 3, 5, 3] = 1
    refined_metrics = derive_subject_mask_metrics(refined_masks)
    refined = {
        **common,
        "masks_roi": refined_masks,
        "available_channels": np.ones((4,), dtype=bool),
        **{f"metrics/{name}": values for name, values in refined_metrics.items()},
    }
    return raw, refined, crop


def _create_crop_store(path: Path, crop: dict[str, np.ndarray]) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    run = root.create_group("crop_runs/crop_fixture")
    for name, values in crop.items():
        run.create_array(name, data=values, chunks=values.shape)


def _create_reference_handoff(path: Path) -> None:
    path.mkdir()
    raw, refined, crop = _fixture()
    crop_store = path.parent / "crop.zarr"
    _create_crop_store(crop_store, crop)
    source_manifest = {
        "schema_id": "palette.subject_mask.validation_mode_test_source",
        "schema_version": 1,
        "payload_digest": "a" * 64,
    }
    raw_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        raw,
        source_crop_arrays=crop,
        source_manifest=source_manifest,
        n_frames=4,
        components=SubjectMaskComponentRegistry(
            ("subject_body", "eyes_union", "swim_bladder")
        ),
        destination=path / "raw.zarr",
        run_id="raw_fixture",
        kind="raw_probability_uint8",
        source_run_path="subject_mask_shard_runs/raw_fixture",
        created_by="pytest",
    )
    refined_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        refined,
        source_crop_arrays=crop,
        source_manifest=source_manifest,
        n_frames=4,
        components=SubjectMaskComponentRegistry(
            ("subject_body", "eye_left", "eye_right", "swim_bladder")
        ),
        destination=path / "refined.zarr",
        run_id="refined_fixture",
        kind="refined_dense_core",
        source_run_path="refined_subject_masks_runs/refined_fixture_draft",
        created_by="pytest",
    )
    payload = {
        "schema_id": "palette.subject_mask_cache_pipeline_benchmark",
        "schema_version": 1,
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
        "inputs": {
            "source_crop_zarr": str(crop_store),
            "crop_run": "crop_fixture",
            "resume_source_palette_commit": "0" * 40,
        },
        "outputs": {
            "raw": {
                "run_id": raw_publication.run_id,
                "logical_content_digest": raw_publication.manifest["payload"][
                    "logical_content"
                ]["digest"],
            },
            "refined": {
                "run_id": refined_publication.run_id,
                "logical_content_digest": refined_publication.manifest["payload"][
                    "logical_content"
                ]["digest"],
            },
            "quality": {
                "run_id": "quality_fixture",
                "logical_content_digest": "b" * 64,
            },
        },
    }
    handoff = {"payload": payload, "payload_digest": canonical_json_sha256(payload)}
    _write_json(path / "handoff_manifest.json", handoff)


def test_node_local_guard_and_io_delta(tmp_path: Path) -> None:
    scratch = _require_new_node_local_scratch(tmp_path / "scratch")
    assert scratch.is_dir()
    with pytest.raises(FileExistsError, match="already exists"):
        _require_new_node_local_scratch(scratch)
    with pytest.raises(ValueError, match="node-local"):
        _require_new_node_local_scratch(Path("/groups/example"))
    assert _io_delta({"read_bytes": 4}, {"read_bytes": 9, "write_bytes": 2}) == {
        "read_bytes": 5,
        "write_bytes": 2,
    }


def test_handoff_digest_and_isolation_are_required(tmp_path: Path) -> None:
    payload = {
        "schema_id": "palette.subject_mask_cache_pipeline_benchmark",
        "schema_version": 1,
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
        "inputs": {},
        "outputs": {name: {"run_id": name} for name in ("raw", "refined", "quality")},
    }
    path = tmp_path / "handoff.json"
    _write_json(
        path, {"payload": payload, "payload_digest": canonical_json_sha256(payload)}
    )
    assert _validate_handoff(path)["payload"] == payload
    payload["selector_eligible"] = True
    _write_json(
        path, {"payload": payload, "payload_digest": canonical_json_sha256(payload)}
    )
    with pytest.raises(ValueError, match="isolated fixture"):
        _validate_handoff(path)


def test_small_real_zarr_canary_proves_validation_mode_equivalence(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    _create_reference_handoff(reference)

    result = run_canary(
        reference_root=reference,
        scratch_root=tmp_path / "canary",
        physical_unit_workers=2,
    )

    assert result["payload_digest"] == canonical_json_sha256(result["payload"])
    payload = result["payload"]
    assert payload["schema_id"] == CANARY_SCHEMA_ID
    assert payload["status"] == "complete"
    assert payload["result"] == "pass"
    assert payload["execution"] == {
        "physical_unit_workers_requested": 2,
        "parallel_write_policy": (
            "bounded_threaded_disjoint_whole_physical_row_bands_v1"
        ),
    }
    assert payload["production_state_changes"] == []
    for name in ("raw", "refined"):
        case = payload["cases"][name]
        assert case["equality"]["exact_logical_content"] is True
        assert case["equality"]["exact_storage_plan"] is True
        assert (
            case["reference_full"]["selector_isolation"]["selector_eligible"] is False
        )
        assert (
            case["production_streaming"]["selector_isolation"]["registry_registered"]
            is False
        )
    progress = json.loads(
        (tmp_path / "canary" / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"
