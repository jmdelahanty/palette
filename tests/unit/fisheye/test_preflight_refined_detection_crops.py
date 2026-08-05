from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.utils import preflight_refined_detection_crops as preflight


class _VerifiedPixels(SimpleNamespace):
    def assert_verified(self) -> None:
        return None


class _Manifest(SimpleNamespace):
    def as_manifest(self):  # noqa: ANN201
        return dict(self.value)


def test_preflight_reports_exact_zero_padding_without_zarr_writes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    source = SimpleNamespace(
        run_id="refined_v2",
        manifest={"payload_digest": "a" * 64},
        logical_content_digest="b" * 64,
        selection_mode="explicit_selector_ineligible_benchmark",
    )
    pixels = _VerifiedPixels(
        binding_document_digest="c" * 64,
        source_video_path=tmp_path / "source.mp4",
        pixel_authority=_Manifest(value={"schema_id": "pixel"}),
    )
    dimensions = _Manifest(
        n_instances=3,
        source_width=100,
        source_height=80,
        value={
            "n_frames": 3,
            "n_instances": 3,
            "source_width": 100,
            "source_height": 80,
        },
    )
    prepared = SimpleNamespace(
        dimensions=dimensions,
        arrays={
            "instance_key": np.asarray([10, 20, 30], dtype=np.uint64),
            "frame_indices": np.asarray([0, 1, 2], dtype=np.int64),
            "source_crop_xywh": np.asarray(
                [[-2, 10, 20, 20], [20, 20, 20, 20], [90, 75, 20, 20]],
                dtype=np.int32,
            ),
        },
    )
    bind_calls = []

    def bind(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        bind_calls.append((args, kwargs))
        return source

    monkeypatch.setattr(preflight, "bind_refined_detection_crop_source", bind)
    monkeypatch.setattr(
        preflight,
        "bind_refined_crop_source_pixel_authority",
        lambda *_args, **_kwargs: pixels,
    )
    monkeypatch.setattr(
        preflight,
        "prepare_crop_geometry_from_refined_source",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        preflight,
        "plan_crop_geometry_storage",
        lambda _dimensions: _Manifest(value={"profile": "test"}),
    )

    result = preflight.inspect_refined_detection_crop_preflight(
        analysis_zarr=archive,
        refined_run_id="refined_v2",
        policy=CropGeometryPolicy(
            purpose="zebrafish_pose_subject_mask_input",
            size_mode=CropSizeMode.FIXED_PER_RUN,
            fixed_size_wh=(20, 20),
            padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
        ),
        max_examples=2,
    )

    assert result["status"] == "ready"
    assert result["padding"]["padded_row_count"] == 2
    assert result["padding"]["fully_contained_row_count"] == 1
    assert result["padding"]["max_padding_ltrb"] == [2, 0, 10, 15]
    assert result["padding"]["examples"][0]["padding_ltrb"] == [2, 0, 0, 0]
    assert result["padding"]["examples"][1]["padding_ltrb"] == [0, 0, 10, 15]
    assert result["crop_zarr_writes"] is False
    assert result["selector_activation"] == "none"
    assert result["registry_updated"] is False
    assert bind_calls[0][1] == {
        "run_id": "refined_v2",
        "allow_selector_ineligible_benchmark": True,
    }

    source.selection_mode = "approved_authoritative_refined_v1"
    production = preflight.inspect_refined_detection_crop_preflight(
        analysis_zarr=archive,
        policy=CropGeometryPolicy(
            purpose="zebrafish_pose_subject_mask_input",
            size_mode=CropSizeMode.FIXED_PER_RUN,
            fixed_size_wh=(20, 20),
            padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
        ),
        max_examples=0,
    )
    assert bind_calls[1] == ((archive.resolve(),), {})
    assert production["selection_mode"] == "approved_authoritative_refined_v1"


def test_cohort_preflight_aggregates_frozen_plan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    payload = {
        "schema_id": "palette.accept_all_refined_detection.batch_plan",
        "schema_version": 1,
        "created_at_utc": "2026-08-05T12:00:00Z",
        "canonical_successor_plan_digest": "a" * 64,
        "canonical_run_id": "canonical_v3",
        "refined_run_id": "refined_v2",
        "candidate_count": 1,
        "candidates": [
            {
                "analysis_zarr": str(archive),
                "inspection": {
                    "analysis_zarr": str(archive),
                    "recording_identity": "recording",
                    "source": {"run_id": "canonical_v3"},
                    "target": {"run_id": "refined_v2"},
                },
            }
        ],
    }
    plan = {
        **payload,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "plan_digest": canonical_json_sha256(payload),
    }
    monkeypatch.setattr(
        preflight,
        "inspect_refined_detection_crop_preflight",
        lambda **_kwargs: {
            "analysis_zarr": str(archive),
            "refined_run_id": "refined_v2",
            "refined_manifest_digest": "b" * 64,
            "refined_logical_content_digest": "c" * 64,
            "dimensions": {"n_instances": 10},
            "pixel_authority": {"binding_document_digest": "d" * 64},
            "padding": {
                "padded_row_count": 3,
                "fully_contained_row_count": 7,
                "max_padding_ltrb": [1, 2, 3, 4],
                "examples": [],
            },
            "array_content_sha256": {"instance_key": "e" * 64},
        },
    )
    policy = CropGeometryPolicy(
        purpose="zebrafish_pose_subject_mask_input",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(348, 348),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )

    result = preflight.inspect_refined_detection_crop_cohort(plan, policy=policy)

    assert result["archive_count"] == 1
    assert result["total_instance_count"] == 10
    assert result["total_padded_row_count"] == 3
    assert result["affected_archive_count"] == 1
    assert result["max_padding_ltrb"] == [1, 2, 3, 4]
    assert result["crop_zarr_writes"] is False
