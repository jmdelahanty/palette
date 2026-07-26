from __future__ import annotations

from dataclasses import replace
import json
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
    frames = np.arange(10, dtype=np.int32)
    return build_canonical_detection_benchmark_input(
        {
            "frame_indices": _Array(frames),
            "bbox_norm_coords": _Array(
                np.tile(
                    np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
                    (10, 1),
                )
            ),
            "scores": _Array(np.linspace(0.5, 0.95, 10, dtype=np.float32)),
            "class_ids": _Array(np.zeros(10, dtype=np.int32)),
        },
        recording_identity="recording-a",
        frame_count=10,
        source_width=640,
        source_height=480,
    )


def _make_group(root: zarr.Group, path: str, *, value: int) -> None:
    group = root.create_group(path)
    group.create_array("value", data=np.asarray([value], dtype=np.int32), chunks=(1,))


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
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
    _make_group(root, "refined_keypoints_runs/selected", value=3)
    zarr.consolidate_metadata(str(source_archive))

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
                "selected_products": [
                    {"product": "raw_video", "path": "raw_video"},
                    {"product": "calibration", "path": "analysis/calibration"},
                    {
                        "product": "refined_keypoints",
                        "path": "refined_keypoints_runs/selected",
                    },
                ],
                "selector_overrides": {
                    "refined_keypoints_runs": {
                        "latest": "selected",
                        "latest_complete": "selected",
                    }
                },
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
    spec_path, benchmark_root, source_archive, source_video = _fixture_inputs(tmp_path)
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
        assert result["publication_receipt"][
            "exact_relative_path_size_content_match"
        ] is True
        pair_manifest = json.loads(
            (destination / "pair_manifest.json").read_text(encoding="utf-8")
        )
        assert (
            pair_manifest["publication_receipt_relative_path"]
            == "publication_receipt.json"
        )
        assert source_metadata_before == (source_archive / "zarr.json").read_bytes()
        assert video_before == source_video.read_bytes()
        assert not any(source_video.name == path.name for path in destination.rglob("*"))

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
            assert json.loads(
                (destination / f"{layout}_manifest.json").read_text(encoding="utf-8")
            )["source_unchanged"] is True
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
        assert not list(scratch_root.glob("palette_canonical_detection_full_analysis_*"))
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
