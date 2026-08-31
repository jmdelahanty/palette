from __future__ import annotations

import json
from pathlib import Path
import tarfile
import threading
import time

import zarr

from fisheye.cluster.subject_masks import publish_receipt_composed_bundle as mod


def _package(
    tmp_path: Path, *, index: int = 0, work_unit_index: int | None = None
) -> Path:
    unit_index = int(index if work_unit_index is None else work_unit_index)
    payload = tmp_path / f"payload_{index}"
    run_name = f"refined_clip_{index}"
    run_path = payload / "refined_subject_masks_runs" / run_name
    zarr.open_group(str(run_path), mode="w", zarr_format=3)
    evidence = payload / "publication_evidence"
    (evidence / "raw_final_layout_unit").mkdir(parents=True)
    (evidence / "refined_final_layout_unit").mkdir()
    (evidence / "quality_partition").mkdir()
    (evidence / "sampled_contour_receipt.json").write_text("{}\n", encoding="utf-8")
    publication = {
        "schema_id": "palette.subject_mask.clip_publication_evidence",
        "schema_version": 1,
        "producer_commit": "c" * 40,
        "work_unit_id": f"unit_{unit_index}",
        "work_unit_index": unit_index,
        "source_clip_id": f"clip_{unit_index}",
        "source_clip_index": unit_index,
        "global_frame_interval": {
            "start_frame": unit_index * 10,
            "stop_frame": (unit_index + 1) * 10,
        },
        "global_row_interval": {
            "start_row": unit_index * 3,
            "stop_row": (unit_index + 1) * 3,
        },
    }
    (payload / "package.json").write_text(
        json.dumps(
            {
                "schema_id": mod.PACKAGE_SCHEMA_ID,
                "package_completion_status": "complete",
                "run_group_path": f"refined_subject_masks_runs/{run_name}",
                "publication_evidence": publication,
            }
        ),
        encoding="utf-8",
    )
    package = tmp_path / f"clip_{index}.tar.gz"
    with tarfile.open(package, "w:gz") as handle:
        for child in payload.iterdir():
            handle.add(child, arcname=child.name)
    return package


def test_receipt_composed_wrapper_assembles_workers_and_forces_strict_profile(
    tmp_path: Path, monkeypatch
) -> None:
    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_0"
    root.require_group("crop_runs").create_group("crop_0")
    root.require_group("subject_mask_shard_runs").create_group("raw_clip_0")
    package = _package(tmp_path)
    captured = {}

    def fake_publish(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        assert (
            Path(kwargs["draft_zarr"])
            .joinpath("refined_subject_masks_runs", "refined_clip_0", "zarr.json")
            .is_file()
        )
        return {"status": "complete"}

    monkeypatch.setattr(mod, "publish_recording_subject_mask_bundle", fake_publish)
    result = mod.publish_receipt_composed_bundle(
        analysis_zarr=analysis,
        crop_run="crop_0",
        raw_draft_runs=("raw_clip_0",),
        refined_package_paths=(package,),
        raw_run="raw_published",
        refined_run="refined_published",
        quality_run="quality_published",
        cache_run="cache_published",
        bundle_id="bundle_published",
        producer_commit="c" * 40,
        local_output_root=tmp_path / "output",
        quality_scratch_root=tmp_path / "quality",
        allow_signed_hybrid_crop_rebase=True,
    )

    assert result["publication_profile"] == "receipt_composed_clip_workers_v1"
    assert captured["require_complete_final_layout_units"] is True
    assert captured["require_worker_quality"] is True
    assert captured["require_worker_sampled_contours"] is True
    assert captured["activate"] is False
    assert captured["allow_signed_hybrid_crop_rebase"] is True
    assert captured["refined_draft_runs"] == ("refined_clip_0",)
    assert result["package_extraction"]["workers_effective"] == 1
    assert result["package_extraction"]["package_count"] == 1
    assert result["package_extraction"]["compressed_bytes"] == package.stat().st_size
    telemetry = result["receipt_composed_runtime_telemetry"]
    assert telemetry["materializer"] == ("publish_receipt_composed_subject_mask_bundle")
    assert [phase["name"] for phase in telemetry["phases"]] == [
        "output_and_staging_setup",
        "assembly_root_setup",
        "assembly_metadata_and_links",
        "package_extraction_and_validation",
        "recording_bundle_publication",
        "staging_cleanup",
    ]
    assert captured["expected_work_units"] == [
        {
            "work_unit_id": "unit_0",
            "work_unit_index": 0,
            "source_clip_id": "clip_0",
            "source_clip_index": 0,
            "frame_start": 0,
            "frame_stop": 10,
            "row_start": 0,
            "row_stop": 3,
        }
    ]


def test_receipt_composed_wrapper_rejects_package_without_evidence(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.require_group("crop_runs").create_group("crop_0")
    root.require_group("subject_mask_shard_runs").create_group("raw_clip_0")
    package = _package(tmp_path)
    replacement = tmp_path / "bad_payload"
    replacement.mkdir()
    zarr.open_group(
        str(replacement / "refined_subject_masks_runs" / "refined_clip_0"),
        mode="w",
        zarr_format=3,
    )
    (replacement / "package.json").write_text(
        json.dumps(
            {
                "schema_id": mod.PACKAGE_SCHEMA_ID,
                "package_completion_status": "complete",
                "run_group_path": "refined_subject_masks_runs/refined_clip_0",
            }
        ),
        encoding="utf-8",
    )
    with tarfile.open(package, "w:gz") as handle:
        for child in replacement.iterdir():
            handle.add(child, arcname=child.name)

    try:
        mod.publish_receipt_composed_bundle(
            analysis_zarr=analysis,
            crop_run="crop_0",
            raw_draft_runs=("raw_clip_0",),
            refined_package_paths=(package,),
            raw_run="raw_published",
            refined_run="refined_published",
            quality_run="quality_published",
            cache_run="cache_published",
            bundle_id="bundle_published",
            producer_commit="c" * 40,
            local_output_root=tmp_path / "output_bad",
            quality_scratch_root=tmp_path / "quality_bad",
        )
    except ValueError as exc:
        assert "publication evidence" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Package without publication evidence was accepted.")


def test_receipt_composed_wrapper_extracts_independent_packages_concurrently_but_assembles_deterministically(
    tmp_path: Path, monkeypatch
) -> None:
    analysis = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(analysis), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_0"
    root.require_group("crop_runs").create_group("crop_0")
    raw_parent = root.require_group("subject_mask_shard_runs")
    raw_parent.create_group("raw_clip_1")
    raw_parent.create_group("raw_clip_0")
    package_1 = _package(tmp_path, index=1, work_unit_index=1)
    package_0 = _package(tmp_path, index=0, work_unit_index=0)

    original_extract = mod._extract_one_clip_package
    lock = threading.Lock()
    active = 0
    peak = 0

    def observed_extract(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            return original_extract(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    captured = {}

    def fake_publish(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        for run_name in ("refined_clip_1", "refined_clip_0"):
            assert (
                Path(kwargs["draft_zarr"])
                .joinpath("refined_subject_masks_runs", run_name, "zarr.json")
                .is_file()
            )
        return {"status": "complete"}

    monkeypatch.setattr(mod, "_extract_one_clip_package", observed_extract)
    monkeypatch.setattr(mod, "publish_recording_subject_mask_bundle", fake_publish)
    result = mod.publish_receipt_composed_bundle(
        analysis_zarr=analysis,
        crop_run="crop_0",
        raw_draft_runs=("raw_clip_1", "raw_clip_0"),
        refined_package_paths=(package_1, package_0),
        raw_run="raw_published",
        refined_run="refined_published",
        quality_run="quality_published",
        cache_run="cache_published",
        bundle_id="bundle_published",
        producer_commit="c" * 40,
        local_output_root=tmp_path / "parallel_output",
        quality_scratch_root=tmp_path / "quality",
        package_extract_workers=2,
    )

    assert peak == 2
    assert captured["refined_draft_runs"] == (
        "refined_clip_1",
        "refined_clip_0",
    )
    assert [item["work_unit_index"] for item in captured["expected_work_units"]] == [
        0,
        1,
    ]
    assert result["package_extraction"] == {
        "transport": "gzip_tar_independent_packages_v1",
        "package_count": 2,
        "compressed_bytes": package_1.stat().st_size + package_0.stat().st_size,
        "workers_requested": 2,
        "workers_effective": 2,
        "ordering": "input_index_results_with_work_unit_index_plan_sort_v1",
        "shared_tree_mutation": "serialized_after_parallel_validation_v1",
    }
