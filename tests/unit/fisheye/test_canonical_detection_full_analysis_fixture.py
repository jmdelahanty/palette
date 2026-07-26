from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.diagnostics.build_canonical_detection_full_analysis_fixtures import main
from fisheye.shared.zarr import canonical_detection_full_analysis_fixture as fixture
from fisheye.shared.zarr.canonical_detection_benchmark import (
    build_canonical_detection_benchmark_input,
    write_detection_benchmark_candidate,
)
from fisheye.shared.zarr.detection_storage import plan_canonical_detection_storage
from fisheye.shared.zarr.storage_profiles import make_benchmark_storage_profile


class _Array:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __getitem__(self, selection):
        return self.values[selection]


def _benchmark_input():
    frames = np.asarray([0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    centers = np.linspace(0.25, 0.75, frames.size, dtype=np.float32)
    return build_canonical_detection_benchmark_input(
        {
            "frame_indices": _Array(frames),
            "bbox_norm_coords": _Array(
                np.column_stack(
                    (
                        centers,
                        centers,
                        np.full(frames.size, 0.1, dtype=np.float32),
                        np.full(frames.size, 0.1, dtype=np.float32),
                    )
                ).astype(np.float32)
            ),
            "scores": _Array(np.linspace(0.5, 0.95, frames.size, dtype=np.float32)),
            "class_ids": _Array(np.zeros(frames.size, dtype=np.int32)),
        },
        recording_identity="recording-a",
        frame_count=10,
        source_width=640,
        source_height=480,
    )


def _make_group(root: zarr.Group, path: str, *, value: int) -> None:
    group = root.create_group(path)
    group.create_array("value", data=np.asarray([value], dtype=np.int32), chunks=(1,))


def _fixture_inputs(
    tmp_path: Path,
    *,
    integration: bool = False,
    full_duration: bool = False,
) -> tuple[Path, Path, Path, Path]:
    if integration and full_duration:
        raise ValueError("A fixture cannot be both bounded and full-duration.")
    benchmark_root = tmp_path / "benchmarks"
    source_recording = tmp_path / "recording-a"
    source_video = source_recording / "cams" / "camera.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"video-association-only")
    source_archive = source_recording / "zarr" / "recording-a_analysis.zarr"
    root = zarr.open_group(str(source_archive), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "recording-a",
            "n_frames": 10,
            "source_video_width": 640,
            "source_video_height": 480,
            "source_video_total_frames": 10,
            "fps": 30.0,
        }
    )
    root.create_group("detect_runs")
    _make_group(root, "raw_video", value=1)
    _make_group(root, "analysis/calibration", value=2)
    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs.update({"latest": "selected", "latest_complete": "selected"})
    keypoints = root.create_group("refined_keypoints_runs/selected")
    keypoints.create_array(
        "value",
        data=np.asarray([3], dtype=np.int32),
        chunks=(1,),
    )
    row_frames = np.asarray(
        [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    frame_counts = np.bincount(row_frames, minlength=10).astype(np.int32)
    keypoints.create_array("frame_counts", data=frame_counts, chunks=(4,))
    keypoints.create_array("n_rois", data=frame_counts, chunks=(4,))
    keypoints.create_array("frame_indices", data=row_frames, chunks=(4,))
    keypoints.create_array(
        "keypoints_img",
        data=np.arange(24, dtype=np.float32).reshape(12, 2),
        chunks=(4, 2),
    )

    masks_parent = root.create_group("refined_subject_masks_runs")
    masks_parent.attrs.update({"latest": "selected", "latest_complete": "selected"})
    masks = root.create_group("refined_subject_masks_runs/selected")
    masks.create_array("frame_counts", data=frame_counts, chunks=(4,))
    masks.create_array("frame_indices", data=row_frames, chunks=(4,))
    masks.create_array(
        "masks_roi",
        data=np.arange(12 * 2 * 4 * 4, dtype=np.uint8).reshape(12, 2, 4, 4),
        chunks=(4, 1, 4, 4),
    )
    contours = masks.create_group("components/subject_body/contours")
    contour_lengths = np.asarray([2, 0, 1, 3, 1, 2, 0, 1, 2, 1, 1, 2], dtype=np.int32)
    contour_ptr = np.full(12, -1, dtype=np.int64)
    next_point = 0
    for row_index, row_length in enumerate(contour_lengths):
        if row_length:
            contour_ptr[row_index] = next_point
            next_point += int(row_length)
    contour_points = np.arange(
        int(contour_lengths.sum()) * 2,
        dtype=np.float32,
    ).reshape(-1, 2)
    contours.create_array("ptr", data=contour_ptr, chunks=(4,))
    contours.create_array("len", data=contour_lengths, chunks=(4,))
    contours.create_array("points_xy", data=contour_points, chunks=(4, 2))

    timeline = root.create_group("analysis/timeline")
    timeline.create_array(
        "second_indices",
        data=np.arange(5, dtype=np.int64),
        chunks=(2,),
    )
    timeline.create_array(
        "speed_per_second",
        data=np.linspace(0.0, 1.0, 5, dtype=np.float32),
        chunks=(2,),
    )
    zarr.consolidate_metadata(str(source_archive))
    root_metadata_path = source_archive / "zarr.json"
    root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
    root_metadata["attributes"]["imageio_metadata"] = {"nframes": float("inf")}
    root_metadata_path.write_text(
        json.dumps(root_metadata, allow_nan=True),
        encoding="utf-8",
    )

    benchmark_input = _benchmark_input()
    candidate_root = benchmark_root / "canonical_detection_storage" / "candidates"
    regular_path = candidate_root / "regular.zarr"
    hybrid_path = candidate_root / "hybrid.zarr"
    regular_profile = make_benchmark_storage_profile(
        target_chunk_bytes=32,
        target_shard_bytes=64,
        shard_immutable=False,
    )
    hybrid_profile = replace(
        make_benchmark_storage_profile(
            target_chunk_bytes=16,
            target_shard_bytes=64,
            shard_immutable=True,
            target_chunk_bytes_by_access={"eager": 16},
        ),
        eager_max_bytes=16,
    )
    write_detection_benchmark_candidate(
        benchmark_input,
        destination=regular_path,
        plans=plan_canonical_detection_storage(
            benchmark_input.dimensions,
            profile=regular_profile,
        ),
        benchmark_root=benchmark_root,
    )
    write_detection_benchmark_candidate(
        benchmark_input,
        destination=hybrid_path,
        plans=plan_canonical_detection_storage(
            benchmark_input.dimensions,
            profile=hybrid_profile,
        ),
        benchmark_root=benchmark_root,
    )

    spec_path = tmp_path / "fixture-spec.json"
    selected_products = [
        {"product": "raw_video", "path": "raw_video"},
        {"product": "calibration", "path": "analysis/calibration"},
        {
            "product": "refined_keypoints",
            "path": "refined_keypoints_runs/selected",
        },
    ]
    selector_overrides = {
        "refined_keypoints_runs": {
            "latest": "selected",
            "latest_complete": "selected",
        }
    }
    integration_window = None
    if integration or full_duration:
        selected_products.extend(
            [
                {
                    "product": "refined_subject_masks",
                    "path": "refined_subject_masks_runs/selected",
                },
                {"product": "timeline", "path": "analysis/timeline"},
            ]
        )
        selector_overrides["refined_subject_masks_runs"] = {
            "latest": "selected",
            "latest_complete": "selected",
        }
        integration_window = {
            "classification": (
                "integration_fixture"
                if integration
                else "full_duration_promotion_fixture"
            ),
            "camera_frame_start": 0,
            "camera_frame_stop": 4 if integration else 10,
            "source_observation_rows": 12,
            "frame_counts_path": "refined_subject_masks_runs/selected/frame_counts",
            "frame_indices_path": "refined_subject_masks_runs/selected/frame_indices",
            "additional_prefix_axes": [
                {
                    "name": "seconds",
                    "source_length": 5,
                    "selected_length": 2 if integration else 5,
                    "index_path": "analysis/timeline/second_indices",
                    "index_validation": (
                        "identity_prefix" if integration else "monotonic_unique"
                    ),
                }
            ],
            "csr_group_paths": [
                "refined_subject_masks_runs/selected/components/subject_body/contours"
            ],
        }
    spec_path.write_text(
        json.dumps(
            {
                "schema_id": fixture.FIXTURE_SPEC_SCHEMA_ID,
                "schema_version": fixture.FIXTURE_SPEC_SCHEMA_VERSION,
                "fixture_id": "tiny",
                "recording_id": "recording-a",
                "source_recording": str(source_recording),
                "source_archive": str(source_archive),
                "source_video": str(source_video),
                "source_video_relative_path": "cams/camera.mp4",
                "detection_run_name": fixture.FIXTURE_RUN_NAME,
                "source_expectations": {
                    "recording_id": "recording-a",
                    "n_frames": 10,
                    "source_video_width": 640,
                    "source_video_height": 480,
                    "source_video_total_frames": 10,
                    "fps": 30.0,
                },
                "selected_products": selected_products,
                "selector_overrides": selector_overrides,
                "candidates": {
                    "regular": {
                        "path": str(regular_path),
                        "expected_profile_id": regular_profile.profile_id,
                    },
                    "hybrid": {
                        "path": str(hybrid_path),
                        "expected_profile_id": hybrid_profile.profile_id,
                    },
                },
                "crimson_contract": {
                    "commit": fixture.CRIMSON_CONTRACT_COMMIT,
                    "document_sha256": fixture.CRIMSON_CONTRACT_SHA256,
                },
                "integration_window": integration_window,
            }
        ),
        encoding="utf-8",
    )
    return spec_path, benchmark_root, source_archive, source_video


def _thaw(root: Path) -> None:
    os.chmod(root, 0o755)
    for path in root.rglob("*"):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def test_plan_is_read_only_and_cli_defaults_to_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path, benchmark_root, _source, _video = _fixture_inputs(tmp_path)
    destination = (
        benchmark_root / "canonical_detection_storage" / "full_analysis" / "tiny"
    )

    assert (
        main(
            [
                "--spec",
                str(spec_path),
                "--destination",
                str(destination),
                "--benchmark-root",
                str(benchmark_root),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "planned"
    assert report["payload_io_performed"] is False
    assert report["detection_run"] == fixture.FIXTURE_RUN_NAME
    assert report["source"]["video"]["copied"] is False
    assert not destination.exists()


def test_publish_builds_exact_immutable_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, benchmark_root, source_archive, source_video = _fixture_inputs(
        tmp_path,
        full_duration=True,
    )
    spec = fixture.load_full_analysis_fixture_spec(spec_path)
    destination = (
        benchmark_root / "canonical_detection_storage" / "full_analysis" / "tiny"
    )
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    source_metadata_before = (source_archive / "zarr.json").read_bytes()
    video_before = source_video.read_bytes()
    monkeypatch.setattr(
        fixture,
        "_palette_code_identity",
        lambda: {
            "repository": "/test/palette",
            "commit": "test-commit",
            "clean": True,
            "dirty_path_count": 0,
        },
    )

    result = fixture.publish_full_analysis_fixture_pair(
        spec,
        destination=destination,
        benchmark_root=benchmark_root,
        pair_copy_mode="copy",
        expected_palette_commit="test-commit",
        scratch_root=scratch_root,
    )

    try:
        assert result["status"] == "published_immutable"
        assert result["payload_io_performed"] is True
        assert result["pair_copy_mode_resolved"] == "copy"
        assert result["nondetection_pair_exact"] is True
        assert result["decoded_detection_pair_exact"] is True
        assert result["publication_receipt"]["complete_tree_payload_hashing"] is False
        assert result["publication_receipt"]["direct_and_consolidated_open"] is True
        assert result["row_relationship_validation"]["all_relationships_valid"]
        assert result["summary"]["camera_frames"] == 10
        assert result["summary"]["full_duration_fixture_published"] is True
        assert result["summary"]["full_duration_gate_satisfied"] is False
        pair_manifest = json.loads(
            (destination / "pair_manifest.json").read_text(encoding="utf-8")
        )
        assert (
            pair_manifest["publication_receipt_relative_path"]
            == "publication_receipt.json"
        )
        assert source_metadata_before == (source_archive / "zarr.json").read_bytes()
        assert video_before == source_video.read_bytes()
        assert not any(
            source_video.name == path.name for path in destination.rglob("*")
        )

        arrays_by_layout: dict[str, dict[str, np.ndarray]] = {}
        for layout in ("regular", "hybrid"):
            archive_path = destination / f"{layout}.zarr"
            root = zarr.open_group(
                str(archive_path),
                mode="r",
                use_consolidated=True,
            )
            assert root.attrs["benchmark_only"] is True
            assert root.attrs["selector_eligible"] is False
            assert root["detect_runs"].attrs["latest"] == fixture.FIXTURE_RUN_NAME
            run = root[f"detect_runs/{fixture.FIXTURE_RUN_NAME}"]
            arrays_by_layout[layout] = {
                path: np.asarray(run[path][:])
                for path in fixture.CANONICAL_DETECTION_SCHEMA_V1.binding_paths
            }
            assert np.asarray(root["raw_video/value"][:]).tolist() == [1]
            assert np.asarray(root["analysis/calibration/value"][:]).tolist() == [2]
            assert (
                json.loads(
                    (destination / f"{layout}_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )["source_unchanged"]
                is True
            )
        assert all(
            np.array_equal(
                arrays_by_layout["regular"][path],
                arrays_by_layout["hybrid"][path],
            )
            for path in arrays_by_layout["regular"]
        )
        assert (destination.stat().st_mode & 0o777) == 0o555
        assert all(
            (path.stat().st_mode & 0o777) == (0o555 if path.is_dir() else 0o444)
            for path in destination.rglob("*")
        )
        assert not list(
            scratch_root.glob("palette_canonical_detection_full_analysis_*")
        )
    finally:
        _thaw(destination)


def test_publish_builds_bounded_integration_pair_with_exact_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, benchmark_root, source_archive, _source_video = _fixture_inputs(
        tmp_path,
        integration=True,
    )
    spec = fixture.load_full_analysis_fixture_spec(spec_path)
    destination = (
        benchmark_root / "canonical_detection_storage" / "full_analysis" / "tiny"
    )
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    source_metadata_before = (source_archive / "zarr.json").read_bytes()
    monkeypatch.setattr(
        fixture,
        "_palette_code_identity",
        lambda: {
            "repository": "/test/palette",
            "commit": "test-commit",
            "clean": True,
            "dirty_path_count": 0,
        },
    )

    result = fixture.publish_full_analysis_fixture_pair(
        spec,
        destination=destination,
        benchmark_root=benchmark_root,
        pair_copy_mode="copy",
        expected_palette_commit="test-commit",
        scratch_root=scratch_root,
    )

    try:
        assert result["evidence_scope"]["classification"] == "integration_fixture"
        assert result["summary"]["selected_camera_frames"] == 4
        assert result["summary"]["selected_observation_rows"] == 6
        assert result["summary"]["full_duration_gate_satisfied"] is False
        assert result["nondetection_pair_exact"] is True
        assert result["decoded_detection_pair_exact"] is True
        assert source_metadata_before == (source_archive / "zarr.json").read_bytes()
        source_root = json.loads(source_metadata_before)
        assert math.isinf(source_root["attributes"]["imageio_metadata"]["nframes"])

        pair_manifest = json.loads(
            (destination / "pair_manifest.json").read_text(encoding="utf-8")
        )
        relationships = pair_manifest["row_relationship_validation"]
        for path_key in (
            "observation_frame_identity_paths",
            "frame_count_paths",
            "csr_group_paths",
        ):
            assert relationships[path_key] == sorted(relationships[path_key])
        normalizations = pair_manifest["logical_slice_manifest"][
            "nonfinite_normalizations"
        ]
        assert any(
            item["source_metadata_path"] == "zarr.json"
            and item["json_pointer"] == "/attributes/imageio_metadata/nframes"
            and item["original_value"] == "positive_infinity"
            and item["fixture_value"] is None
            for item in normalizations
        )

        arrays_by_layout: dict[str, dict[str, np.ndarray]] = {}
        for layout in ("regular", "hybrid"):
            archive_path = destination / f"{layout}.zarr"
            root = zarr.open_group(
                str(archive_path),
                mode="r",
                use_consolidated=True,
            )
            assert (
                root.attrs["fixture_evidence_classification"] == "integration_fixture"
            )
            assert root.attrs["imageio_metadata"]["nframes"] is None
            assert root["refined_keypoints_runs/selected/frame_counts"].shape == (4,)
            assert root["refined_keypoints_runs/selected/frame_indices"].shape == (6,)
            assert root["refined_subject_masks_runs/selected/masks_roi"].shape == (
                6,
                2,
                4,
                4,
            )
            assert root["analysis/timeline/second_indices"].shape == (2,)
            contour_group = root[
                "refined_subject_masks_runs/selected/components/subject_body/contours"
            ]
            assert contour_group["ptr"].shape == (6,)
            assert contour_group["len"].shape == (6,)
            assert contour_group["points_xy"].shape == (9, 2)
            run = root[f"detect_runs/{fixture.FIXTURE_RUN_NAME}"]
            offsets = np.asarray(run["instances/frame_row_offsets"][:])
            frame_indices = np.asarray(run["instances/frame_indices"][:])
            assert offsets.shape == (5,)
            assert offsets.tolist() == [0, 2, 3, 5, 6]
            assert offsets[-1] == frame_indices.shape[0]
            arrays_by_layout[layout] = {
                path: np.asarray(run[path][:])
                for path in fixture.CANONICAL_DETECTION_SCHEMA_V1.binding_paths
            }
        assert all(
            np.array_equal(
                arrays_by_layout["regular"][path],
                arrays_by_layout["hybrid"][path],
            )
            for path in arrays_by_layout["regular"]
        )
        assert result["publication_receipt"]["payload_verification"] == (
            "exact_logical_slice_hashes"
        )
        assert not list(scratch_root.glob("palette_canonical_detection_integration_*"))
    finally:
        _thaw(destination)


def test_failure_preserves_incomplete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, benchmark_root, _source, _video = _fixture_inputs(tmp_path)
    spec = fixture.load_full_analysis_fixture_spec(spec_path)
    destination = (
        benchmark_root / "canonical_detection_storage" / "full_analysis" / "tiny"
    )
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    monkeypatch.setattr(
        fixture,
        "_consolidate_and_validate",
        lambda _root: (_ for _ in ()).throw(RuntimeError("forced validation failure")),
    )
    monkeypatch.setattr(
        fixture,
        "_palette_code_identity",
        lambda: {
            "repository": "/test/palette",
            "commit": "test-commit",
            "clean": True,
            "dirty_path_count": 0,
        },
    )

    with pytest.raises(RuntimeError, match="incomplete evidence remains"):
        fixture.publish_full_analysis_fixture_pair(
            spec,
            destination=destination,
            benchmark_root=benchmark_root,
            pair_copy_mode="copy",
            expected_palette_commit="test-commit",
            scratch_root=scratch_root,
        )

    incomplete = list(destination.parent.glob(".tiny.incomplete.*"))
    scratch_incomplete = list(
        scratch_root.glob("palette_canonical_detection_full_analysis_*")
    )
    assert not destination.exists()
    assert len(incomplete) == 1
    assert len(scratch_incomplete) == 1
    failure = json.loads((incomplete[0] / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "incomplete_failed"
    assert failure["cleanup_policy"] == "explicit_manual_cleanup_only"


def test_destination_and_source_detection_selection_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be exactly"):
        fixture.require_safe_full_analysis_destination(
            tmp_path / "outside" / "tiny",
            benchmark_root=tmp_path / "benchmarks",
            fixture_id="tiny",
        )
    payload = {
        "schema_id": fixture.FIXTURE_SPEC_SCHEMA_ID,
        "schema_version": fixture.FIXTURE_SPEC_SCHEMA_VERSION,
        "fixture_id": "tiny",
        "recording_id": "recording",
        "source_recording": str(tmp_path),
        "source_archive": str(tmp_path / "source.zarr"),
        "source_video": str(tmp_path / "video.mp4"),
        "source_video_relative_path": "video.mp4",
        "detection_run_name": fixture.FIXTURE_RUN_NAME,
        "source_expectations": {"n_frames": 1},
        "selected_products": [{"product": "detect", "path": "detect_runs/run"}],
        "selector_overrides": {},
        "candidates": {
            layout: {"path": str(tmp_path / layout), "expected_profile_id": layout}
            for layout in ("regular", "hybrid")
        },
        "crimson_contract": {
            "commit": fixture.CRIMSON_CONTRACT_COMMIT,
            "document_sha256": fixture.CRIMSON_CONTRACT_SHA256,
        },
    }
    with pytest.raises(ValueError, match="Source detect_runs cannot be copied"):
        fixture.FullAnalysisFixtureSpec.from_payload(payload)


def test_integration_shape_planning_rejects_undeclared_large_axis() -> None:
    with pytest.raises(ValueError, match="undeclared large leading axis"):
        fixture._target_array_shape(
            "selected/unknown",
            (2049, 2),
            cardinalities={"leading_axes": {}, "point_axes": {}},
        )
