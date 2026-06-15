from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.roi_pixel_contract import orange_mono_pynvvc_luma_pixel_contract
from fisheye.utils import audit_zarr_pixel_contracts
from fisheye.utils.audit_zarr_pixel_contracts import ZarrCandidate
from fisheye.utils.audit_zarr_pixel_contracts import audit_zarr_path
from fisheye.utils.audit_zarr_pixel_contracts import main


def _write_node(path: Path, *, node_type: str = "group", attrs: dict | None = None, shape: list[int] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "zarr_format": 3,
        "node_type": node_type,
        "attributes": attrs or {},
    }
    if shape is not None:
        payload["shape"] = shape
        payload["data_type"] = "uint8"
    (path / "zarr.json").write_text(json.dumps(payload), encoding="utf-8")


def test_audit_clipped_training_raw_images_reports_legacy_opencv_backfill(tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_clipped_training.zarr"
    _write_node(zarr_path)
    _write_node(
        zarr_path / "raw_video",
        attrs={
            "import_method": "fisheye.utils.create_clipped_training_zarr",
            "source_layout": "rolling_clips",
        },
    )
    _write_node(zarr_path / "raw_video" / "images_full", node_type="array", attrs={}, shape=[3, 4512, 4512])

    rows = audit_zarr_path(ZarrCandidate(path=zarr_path, source="test"))
    image_row = next(row for row in rows if row["surface_path"] == "raw_video/images_full")

    assert image_row["zarr_kind"] == "clipped_training"
    assert image_row["missing_fields"] == ["decode_backend", "pixel_contract"]
    assert image_row["backfill"]["status"] == "infer_legacy_opencv_gray_from_writer"
    assert image_row["backfill"]["suggested_pixel_contract_name"] == "opencv_bgr2gray_uint8"


def test_audit_crop_run_reads_contract_name_from_structured_contract(tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_training.zarr"
    contract = orange_mono_pynvvc_luma_pixel_contract()
    _write_node(zarr_path)
    _write_node(zarr_path / "crop_runs")
    _write_node(
        zarr_path / "crop_runs" / "crop_x_pynvvc_luma_v1",
        attrs={
            "decode_backend": "pynvvc_luma",
            "roi_pixel_contract": contract,
        },
    )
    _write_node(
        zarr_path / "crop_runs" / "crop_x_pynvvc_luma_v1" / "roi_images",
        node_type="array",
        shape=[5, 512, 512],
    )

    rows = audit_zarr_path(ZarrCandidate(path=zarr_path, source="test"))
    crop_row = next(row for row in rows if row["surface_path"] == "crop_runs/crop_x_pynvvc_luma_v1")
    roi_row = next(row for row in rows if row["surface_path"] == "crop_runs/crop_x_pynvvc_luma_v1/roi_images")

    assert crop_row["pixel_contract_name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert crop_row["backfill"]["status"] == "safe_scalar_name_backfill"
    assert roi_row["pixel_contract_name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert roi_row["backfill"]["status"] == "present"


def test_crop_contract_report_focuses_current_crop_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_training.zarr"
    report_path = tmp_path / "crop_report.json"
    output_path = tmp_path / "audit.jsonl"
    contract = orange_mono_pynvvc_luma_pixel_contract()
    _write_node(zarr_path)
    _write_node(zarr_path / "crop_runs", attrs={"latest_complete": "crop_current"})
    _write_node(zarr_path / "crop_runs" / "crop_old", attrs={})
    _write_node(
        zarr_path / "crop_runs" / "crop_current",
        attrs={
            "crop_storage_mode": "materialized",
            "roi_pixel_contract": contract,
            "roi_pixel_contract_name": contract["name"],
        },
    )

    assert (
        main(
            [
                str(zarr_path),
                "--output-jsonl",
                str(output_path),
                "--crop-contract-report-json",
                str(report_path),
            ]
        )
        == 0
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    current_row = next(row for row in rows if row["surface_path"] == "crop_runs/crop_current")
    old_row = next(row for row in rows if row["surface_path"] == "crop_runs/crop_old")

    assert current_row["is_current_crop_run"] is True
    assert current_row["current_crop_selector"] == "crop_runs.attrs.latest_complete"
    assert old_row["is_current_crop_run"] is False
    assert report["current_crop_run_rows"] == 1
    assert report["current_crop_runs_with_contract"] == 1
    assert report["current_crop_runs_missing_contract"] == 0
    assert report["contract_counts"] == {"orange_mono_pynvvc_luma_uint8_v1": 1}
    assert report["crop_storage_mode_counts"] == {"materialized": 1}


def test_audit_main_writes_jsonl(tmp_path: Path) -> None:
    zarr_path = tmp_path / "detect_merged.zarr"
    output = tmp_path / "audit.jsonl"
    _write_node(
        zarr_path,
        attrs={
            "training_export": {
                "task": "detect",
                "input_format": "gray",
            }
        },
    )
    _write_node(zarr_path / "raw_video", attrs={"downsample_formats": ["gray"]})
    _write_node(zarr_path / "raw_video" / "images_ds", node_type="array", attrs={"format": "gray"}, shape=[2, 640, 640])

    assert main([str(zarr_path), "--output-jsonl", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert {row["surface_path"] for row in rows} == {".", "raw_video", "raw_video/images_ds"}
    root_row = next(row for row in rows if row["surface_path"] == ".")
    assert root_row["backfill"]["status"] == "missing_export_contract"


def test_audit_source_video_metadata_from_raw_video_attrs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_analysis.zarr"
    _write_node(zarr_path)
    _write_node(
        zarr_path / "raw_video",
        attrs={
            "source_path": "cams/Cam2010093_fish.mp4",
            "video_codec": "hevc",
            "video_pix_fmt": "yuv420p",
            "video_width": 4512,
            "video_height": 4512,
            "fps": 60,
            "source_video_total_frames": 1200,
            "color_range": "limited",
            "source_video_sha256": "abc123",
        },
    )

    rows = audit_zarr_path(ZarrCandidate(path=zarr_path, source="test"), include_source_video_metadata=True)
    source_row = next(row for row in rows if row["record_type"] == "source_video_metadata")

    assert source_row["source_scope"] == "single_video"
    assert source_row["codec"] == "hevc"
    assert source_row["pix_fmt"] == "yuv420p"
    assert source_row["width"] == 4512
    assert source_row["height"] == 4512
    assert source_row["fps"] == 60.0
    assert source_row["frame_count"] == 1200
    assert source_row["colorimetry"] == {"color_range": "limited"}
    assert source_row["fingerprint"] == "abc123"
    assert source_row["missing_fields"] == []
    assert source_row["backfill"]["status"] == "present"


def test_audit_source_video_metadata_from_clipped_sidecar(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "fish_clipped_training.zarr"
    _write_node(zarr_path)
    _write_node(zarr_path / "raw_video", attrs={"source_layout": "rolling_clips"})
    (recording_dir / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_id": "clip_000000",
                        "video_path": "clips/clip_000000/Cam2010093_fish.mp4",
                        "codec": "hevc",
                        "pix_fmt": "yuv420p",
                        "width": 4512,
                        "height": 4512,
                        "fps": 30,
                        "frame_count": 54000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = audit_zarr_path(ZarrCandidate(path=zarr_path, source="test"), include_source_video_metadata=True)
    source_rows = [row for row in rows if row["record_type"] == "source_video_metadata"]

    assert len(source_rows) == 1
    source_row = source_rows[0]
    assert source_row["source_scope"] == "clipped_sidecar"
    assert source_row["source_id"] == "clip_000000"
    assert source_row["source_video_path"] == "clips/clip_000000/Cam2010093_fish.mp4"
    assert source_row["codec"] == "hevc"
    assert source_row["pix_fmt"] == "yuv420p"
    assert source_row["missing_fields"] == ["colorimetry", "fingerprint"]
    assert source_row["backfill"]["status"] == "clipped_sidecar_partial"


def test_apply_safe_scalar_name_backfill_updates_crop_run_metadata(tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_training.zarr"
    output = tmp_path / "audit.jsonl"
    summary = tmp_path / "summary.json"
    contract = orange_mono_pynvvc_luma_pixel_contract()
    crop_path = zarr_path / "crop_runs" / "crop_x_pynvvc_luma_v1"
    _write_node(zarr_path)
    _write_node(zarr_path / "crop_runs")
    _write_node(crop_path, attrs={"roi_pixel_contract": contract})

    assert (
        main(
            [
                str(zarr_path),
                "--apply-safe-scalar-name-backfill",
                "--output-jsonl",
                str(output),
                "--summary-json",
                str(summary),
            ]
        )
        == 0
    )

    crop_payload = json.loads((crop_path / "zarr.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    action_row = next(row for row in rows if row["record_type"] == "safe_scalar_name_backfill_action")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))

    assert crop_payload["attributes"]["roi_pixel_contract_name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert action_row["status"] == "updated"
    assert summary_payload["safe_scalar_name_backfill_action_counts"] == {"updated": 1}


def test_apply_inferred_legacy_crop_contract_updates_crop_run_metadata(tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_analysis.zarr"
    output = tmp_path / "audit.jsonl"
    summary = tmp_path / "summary.json"
    crop_path = zarr_path / "crop_runs" / "crop_legacy"
    _write_node(zarr_path)
    _write_node(zarr_path / "crop_runs", attrs={"latest": "crop_legacy"})
    _write_node(
        crop_path,
        attrs={
            "crop_storage_mode": "materialized",
            "video_source_type": "external",
            "roi_live_acceleration_effective": "gpu",
        },
    )

    assert (
        main(
            [
                str(zarr_path),
                "--apply-inferred-legacy-crop-contracts",
                "--output-jsonl",
                str(output),
                "--summary-json",
                str(summary),
            ]
        )
        == 0
    )

    crop_payload = json.loads((crop_path / "zarr.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    action_row = next(row for row in rows if row["record_type"] == "inferred_legacy_crop_contract_action")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))

    assert crop_payload["attributes"]["roi_pixel_contract_name"] == "decord_rgb_channel_mean_uint8"
    assert crop_payload["attributes"]["roi_pixel_contract"]["name"] == "decord_rgb_channel_mean_uint8"
    assert action_row["status"] == "updated"
    assert action_row["roi_pixel_contract_name"] == "decord_rgb_channel_mean_uint8"
    assert summary_payload["inferred_legacy_crop_contract_action_counts"] == {"updated": 1}


def test_apply_inferred_legacy_crop_contract_can_limit_to_current_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_analysis.zarr"
    output = tmp_path / "audit.jsonl"
    current_path = zarr_path / "crop_runs" / "crop_current"
    old_path = zarr_path / "crop_runs" / "crop_old"
    _write_node(zarr_path)
    _write_node(zarr_path / "crop_runs", attrs={"latest_complete": "crop_current"})
    for crop_path in (current_path, old_path):
        _write_node(
            crop_path,
            attrs={
                "crop_storage_mode": "materialized",
                "video_source_type": "external",
                "roi_live_acceleration_effective": "gpu",
            },
        )

    assert (
        main(
            [
                str(zarr_path),
                "--apply-inferred-legacy-crop-contracts",
                "--apply-current-crop-runs-only",
                "--output-jsonl",
                str(output),
            ]
        )
        == 0
    )

    current_payload = json.loads((current_path / "zarr.json").read_text(encoding="utf-8"))
    old_payload = json.loads((old_path / "zarr.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    action_rows = [row for row in rows if row["record_type"] == "inferred_legacy_crop_contract_action"]

    assert len(action_rows) == 1
    assert action_rows[0]["surface_path"] == "crop_runs/crop_current"
    assert current_payload["attributes"]["roi_pixel_contract_name"] == "decord_rgb_channel_mean_uint8"
    assert "roi_pixel_contract_name" not in old_payload["attributes"]


def test_apply_inferred_legacy_crop_contract_refuses_canonical_contract(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_analysis.zarr"
    output = tmp_path / "audit.jsonl"
    crop_path = zarr_path / "crop_runs" / "crop_bad"
    contract = orange_mono_pynvvc_luma_pixel_contract()
    _write_node(zarr_path)
    _write_node(zarr_path / "crop_runs", attrs={"latest": "crop_bad"})
    _write_node(crop_path, attrs={"crop_storage_mode": "materialized"})

    def fake_guidance(**kwargs):
        return {
            "status": "infer_from_crop_run_attrs",
            "confidence": "medium",
            "action": "test only",
            "suggested_roi_pixel_contract_name": contract["name"],
            "suggested_roi_pixel_contract": contract,
        }

    monkeypatch.setattr(audit_zarr_pixel_contracts, "_backfill_guidance", fake_guidance)

    assert (
        main(
            [
                str(zarr_path),
                "--apply-inferred-legacy-crop-contracts",
                "--output-jsonl",
                str(output),
            ]
        )
        == 0
    )

    crop_payload = json.loads((crop_path / "zarr.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    action_row = next(row for row in rows if row["record_type"] == "inferred_legacy_crop_contract_action")

    assert "roi_pixel_contract_name" not in crop_payload["attributes"]
    assert action_row["status"] == "skipped"
    assert action_row["reason"] == "refusing_to_infer_current_canonical_contract"


def test_source_video_backfill_plan_jsonl_is_report_only(tmp_path: Path) -> None:
    zarr_path = tmp_path / "fish_analysis.zarr"
    plan_path = tmp_path / "source_plan.jsonl"
    _write_node(zarr_path)
    _write_node(
        zarr_path / "raw_video",
        attrs={
            "source_path": "cams/Cam2010093_fish.mp4",
            "video_codec": "hevc",
            "video_pix_fmt": "yuv420p",
            "video_width": 4512,
            "video_height": 4512,
            "fps": 60,
            "source_video_total_frames": 1200,
        },
    )

    assert main([str(zarr_path), "--source-video-backfill-plan-jsonl", str(plan_path), "--output-jsonl", str(tmp_path / "audit.jsonl")]) == 0
    plan_rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]

    assert len(plan_rows) == 1
    plan_row = plan_rows[0]
    assert plan_row["record_type"] == "source_video_backfill_plan"
    assert plan_row["missing_fields"] == ["colorimetry", "fingerprint"]
    assert plan_row["ffprobe_needed"] is True
    assert plan_row["ffprobe_fields"] == ["colorimetry"]
    assert plan_row["fingerprint_needed"] is True
    assert plan_row["can_probe_without_path_repair"] is True


def test_apply_source_video_stat_fingerprint_updates_raw_video_attrs(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "fish_analysis.zarr"
    video_path = recording_dir / "cams" / "Cam2010093_fish.mp4"
    output = tmp_path / "audit.jsonl"
    summary = tmp_path / "summary.json"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake video bytes")
    _write_node(zarr_path)
    _write_node(
        zarr_path / "raw_video",
        attrs={
            "source_path": "cams/Cam2010093_fish.mp4",
            "video_codec": "hevc",
            "video_pix_fmt": "yuv420p",
            "video_width": 4512,
            "video_height": 4512,
            "fps": 60,
            "source_video_total_frames": 1200,
            "color_range": "limited",
        },
    )

    assert (
        main(
            [
                str(zarr_path),
                "--apply-source-video-stat-fingerprint",
                "--output-jsonl",
                str(output),
                "--summary-json",
                str(summary),
            ]
        )
        == 0
    )

    raw_payload = json.loads((zarr_path / "raw_video" / "zarr.json").read_text(encoding="utf-8"))
    raw_attrs = raw_payload["attributes"]
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    action_row = next(row for row in rows if row["record_type"] == "source_video_stat_fingerprint_action")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))

    assert raw_attrs["source_video_fingerprint_strategy"] == "stat_v1"
    assert raw_attrs["source_video_fingerprint"]
    assert raw_attrs["source_video_size_bytes"] == len(b"fake video bytes")
    assert isinstance(raw_attrs["source_video_mtime_ns"], int)
    assert raw_attrs["source_video_fingerprint_payload"]["frame_count"] == 1200
    assert action_row["status"] == "updated"
    assert action_row["source_video_fingerprint"] == raw_attrs["source_video_fingerprint"]
    assert summary_payload["source_video_stat_fingerprint_action_counts"] == {"updated": 1}


def test_apply_source_video_ffprobe_colorimetry_updates_reported_attrs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_dir = tmp_path / "recording"
    zarr_path = recording_dir / "zarr" / "fish_analysis.zarr"
    video_path = recording_dir / "cams" / "Cam2010093_fish.mp4"
    output = tmp_path / "audit.jsonl"
    summary = tmp_path / "summary.json"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake video bytes")
    _write_node(zarr_path)
    _write_node(
        zarr_path / "raw_video",
        attrs={
            "source_path": "cams/Cam2010093_fish.mp4",
            "video_codec": "hevc",
            "video_pix_fmt": "yuv420p",
            "video_width": 4512,
            "video_height": 4512,
            "fps": 60,
            "source_video_total_frames": 1200,
            "source_video_fingerprint": "stat-digest",
        },
    )

    def fake_check_output(command: list[str], text: bool) -> str:
        assert command[0] == "fake-ffprobe"
        assert text is True
        return json.dumps({"streams": [{"color_range": "tv"}]})

    monkeypatch.setattr(audit_zarr_pixel_contracts.subprocess, "check_output", fake_check_output)

    assert (
        main(
            [
                str(zarr_path),
                "--apply-source-video-ffprobe-colorimetry",
                "--ffprobe-bin",
                "fake-ffprobe",
                "--output-jsonl",
                str(output),
                "--summary-json",
                str(summary),
            ]
        )
        == 0
    )

    raw_payload = json.loads((zarr_path / "raw_video" / "zarr.json").read_text(encoding="utf-8"))
    raw_attrs = raw_payload["attributes"]
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    action_row = next(row for row in rows if row["record_type"] == "source_video_ffprobe_colorimetry_action")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))

    assert raw_attrs["color_range"] == "tv"
    assert raw_attrs["source_video_colorimetry_source"] == "ffprobe_stream"
    assert "color_space" not in raw_attrs
    assert action_row["status"] == "updated"
    assert action_row["colorimetry"] == {"color_range": "tv"}
    assert summary_payload["source_video_ffprobe_colorimetry_action_counts"] == {"updated": 1}
