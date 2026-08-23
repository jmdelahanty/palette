from __future__ import annotations

import json
from pathlib import Path
import tarfile

import zarr

from fisheye.cluster.subject_masks import publish_receipt_composed_bundle as mod


def _package(tmp_path: Path) -> Path:
    payload = tmp_path / "payload"
    run_path = payload / "refined_subject_masks_runs" / "refined_clip_0"
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
        "work_unit_id": "unit_0",
        "work_unit_index": 0,
        "source_clip_id": "clip_0",
        "source_clip_index": 0,
        "global_frame_interval": {"start_frame": 0, "stop_frame": 10},
        "global_row_interval": {"start_row": 0, "stop_row": 3},
    }
    (payload / "package.json").write_text(
        json.dumps(
            {
                "schema_id": mod.PACKAGE_SCHEMA_ID,
                "package_completion_status": "complete",
                "run_group_path": "refined_subject_masks_runs/refined_clip_0",
                "publication_evidence": publication,
            }
        ),
        encoding="utf-8",
    )
    package = tmp_path / "clip_0.tar.gz"
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
