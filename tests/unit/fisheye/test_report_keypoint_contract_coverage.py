from __future__ import annotations

import json
from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.utils.report_keypoint_contract_coverage import (
    MISSING,
    build_keypoint_contract_coverage_report,
    main,
)


def _add_dataset(registry: Registry, tmp_path: Path, dataset_id: str, recording_id: str) -> None:
    registry.upsert_dataset(
        dataset_id,
        session_uuid=dataset_id,
        zarr_path=tmp_path / f"{dataset_id}.zarr",
        recording_id=recording_id,
        artifact_kind="analysis",
        zarr_use="source_analysis",
    )


def _insert_keypoint_run(
    registry: Registry,
    *,
    dataset_id: str,
    recording_id: str,
    keypoint_run: str,
    contract: str | None,
    read_mode: str | None = None,
    cache_backend: str | None = None,
    input_mode: str | None = None,
    created_utc: str = "2026-06-15T00:00:00Z",
) -> None:
    with registry.conn:
        registry.conn.execute(
            """
            INSERT INTO keypoint_performance (
                dataset_id,
                recording_id,
                keypoint_run,
                keypoint_created_utc,
                keypoint_method,
                source_roi_pixel_contract_name,
                source_roi_read_mode,
                source_roi_cache_backend,
                input_mode_effective,
                total_rois,
                successful_detections,
                updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                dataset_id,
                recording_id,
                keypoint_run,
                created_utc,
                "yolo_pose",
                contract,
                read_mode,
                cache_backend,
                input_mode,
                10,
                9,
                created_utc,
            ),
        )


def test_keypoint_contract_report_flags_mixed_explicit_and_unknown_groups(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        _add_dataset(registry, tmp_path, "dataset_orange", "rec_mixed")
        _add_dataset(registry, tmp_path, "dataset_nv12", "rec_mixed")
        _add_dataset(registry, tmp_path, "dataset_unknown", "rec_unknown")
        _add_dataset(registry, tmp_path, "dataset_partial", "rec_partial")
        _add_dataset(registry, tmp_path, "dataset_partial_unknown", "rec_partial")

        _insert_keypoint_run(
            registry,
            dataset_id="dataset_orange",
            recording_id="rec_mixed",
            keypoint_run="keypoints_orange",
            contract="orange_mono_pynvvc_luma_uint8_v1",
            read_mode="materialized_crop_run",
            input_mode="tensor",
        )
        _insert_keypoint_run(
            registry,
            dataset_id="dataset_nv12",
            recording_id="rec_mixed",
            keypoint_run="keypoints_nv12",
            contract="nv12_luma_plane_uint8",
            read_mode="flat_bin_roi_cache",
            cache_backend="flat_bin_v1",
            input_mode="tensor",
        )
        _insert_keypoint_run(
            registry,
            dataset_id="dataset_unknown",
            recording_id="rec_unknown",
            keypoint_run="keypoints_unknown",
            contract=None,
        )
        _insert_keypoint_run(
            registry,
            dataset_id="dataset_partial",
            recording_id="rec_partial",
            keypoint_run="keypoints_partial",
            contract="orange_mono_pynvvc_luma_uint8_v1",
        )
        _insert_keypoint_run(
            registry,
            dataset_id="dataset_partial_unknown",
            recording_id="rec_partial",
            keypoint_run="keypoints_partial_unknown",
            contract=None,
        )

        report = build_keypoint_contract_coverage_report(
            registry_path,
            source_relation="all",
            group_by="recording",
        )

        groups = {group["group_id"]: group for group in report["groups"]}
        assert report["contract_counts"] == {
            MISSING: 2,
            "nv12_luma_plane_uint8": 1,
            "orange_mono_pynvvc_luma_uint8_v1": 2,
        }
        assert report["read_mode_counts"] == {
            MISSING: 3,
            "flat_bin_roi_cache": 1,
            "materialized_crop_run": 1,
        }
        assert report["cache_backend_counts"] == {MISSING: 4, "flat_bin_v1": 1}
        assert report["input_mode_counts"] == {MISSING: 3, "tensor": 2}
        assert groups["rec_mixed"]["contract_group_status"] == "mixed_explicit"
        assert groups["rec_mixed"]["compatibility_status"] == "candidate_compatible"
        assert groups["rec_unknown"]["contract_group_status"] == "unknown_only"
        assert groups["rec_partial"]["contract_group_status"] == "explicit_with_unknown"
    finally:
        registry.close()


def test_keypoint_contract_report_marks_mixed_explicit_unreviewed_when_not_candidate(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        _add_dataset(registry, tmp_path, "dataset_orange", "rec")
        _add_dataset(registry, tmp_path, "dataset_other", "rec")
        _insert_keypoint_run(
            registry,
            dataset_id="dataset_orange",
            recording_id="rec",
            keypoint_run="keypoints_orange",
            contract="orange_mono_pynvvc_luma_uint8_v1",
        )
        _insert_keypoint_run(
            registry,
            dataset_id="dataset_other",
            recording_id="rec",
            keypoint_run="keypoints_other",
            contract="other_contract_v1",
        )

        report = build_keypoint_contract_coverage_report(
            registry_path,
            source_relation="all",
            group_by="recording",
        )

        assert report["groups"][0]["contract_group_status"] == "mixed_explicit"
        assert report["groups"][0]["compatibility_status"] == "needs_review"
    finally:
        registry.close()


def test_keypoint_contract_report_cli_writes_jsonl(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    output_jsonl = tmp_path / "report.jsonl"
    registry = Registry(registry_path)
    try:
        _add_dataset(registry, tmp_path, "dataset", "rec")
        _insert_keypoint_run(
            registry,
            dataset_id="dataset",
            recording_id="rec",
            keypoint_run="keypoints",
            contract="nv12_luma_plane_uint8",
            read_mode="flat_bin_roi_cache",
            cache_backend="flat_bin_v1",
            input_mode="tensor",
        )

        assert (
            main(
                [
                    "--registry",
                    str(registry_path),
                    "--source-relation",
                    "all",
                    "--group-by",
                    "dataset",
                    "--output-jsonl",
                    str(output_jsonl),
                ]
            )
            == 0
        )

        stdout = capsys.readouterr().out
        assert "keypoint_contract_coverage_report" in stdout
        rows = [
            json.loads(line)
            for line in output_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["record_type"] for row in rows] == ["contract_group", "keypoint_run"]
        assert rows[0]["contract_group_status"] == "explicit_single"
        assert rows[1]["source_roi_pixel_contract_name"] == "nv12_luma_plane_uint8"
    finally:
        registry.close()
