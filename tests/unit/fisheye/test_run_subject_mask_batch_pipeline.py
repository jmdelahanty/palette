from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.shared.run_provenance import validate_run_provenance
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from fisheye.utils import run_subject_mask_batch_pipeline as mod


def test_parser_defaults_to_sampled_contours_and_sharded_postcompute() -> None:
    args = mod._build_parser().parse_args(["/recordings"])

    assert args.finalize_postcompute_backend == "process_shards"
    assert args.finalize_dense_mask_row_chunk == mod.DEFAULT_FINALIZE_DENSE_MASK_ROW_CHUNK
    assert args.mask_storage == "dense_uint8"
    assert args.mask_rle_validation_mode == "invariants"
    assert args.subject_output_parent == mod.SUBJECT_MASK_CANONICAL_OUTPUT_PARENT
    assert args.write_eye_geometry is True
    assert args.write_sampled_component_contours is True
    assert args.write_component_contours is False
    assert args.mask_probs_shard_rois == mod.DEFAULT_MASK_PROBS_SHARD_ROIS


def test_parser_accepts_full_ragged_contour_opt_in() -> None:
    args = mod._build_parser().parse_args(["/recordings", "--write-component-contours"])

    assert args.write_component_contours is True
    assert args.write_sampled_component_contours is True


def test_parser_accepts_regular_probability_chunk_override() -> None:
    args = mod._build_parser().parse_args(["/recordings", "--no-mask-probs-sharding"])

    assert args.mask_probs_shard_rois is None


def test_parser_accepts_subject_mask_shard_output_parent() -> None:
    args = mod._build_parser().parse_args(["/recordings", "--subject-output-parent", "subject_mask_shard_runs"])

    assert args.subject_output_parent == mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT


def test_parser_accepts_dense_plus_compact_mask_storage_mode() -> None:
    args = mod._build_parser().parse_args(["/recordings", "--mask-storage", "dense_and_rle"])

    assert args.mask_storage == "dense_and_rle"


def test_parser_rejects_compact_only_mask_storage_mode() -> None:
    with pytest.raises(SystemExit):
        mod._build_parser().parse_args(["/recordings", "--mask-storage", "bitpacked_v1"])

    with pytest.raises(SystemExit):
        mod._build_parser().parse_args(["/recordings", "--mask-storage", "rle_v1"])


def test_default_output_staging_root_falls_back_when_user_scratch_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER", "palette_test_user_without_scratch")
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LSB_JOBINDEX", "7")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    assert mod._default_output_staging_root() == (  # noqa: SLF001
        tmp_path
        / "palette"
        / "palette_test_user_without_scratch"
        / "12345"
        / "array_7"
        / "subject_mask_output_staging"
    )


def test_safe_artifact_filename_hashes_long_names() -> None:
    filename = mod._safe_artifact_filename(
        ("recording_analysis", "subject_masks_" + ("very_long_" * 40)),
        ".workflow.profile.jsonl",
    )

    assert len(filename) <= mod.MAX_ARTIFACT_FILENAME_CHARS
    assert filename.endswith(".workflow.profile.jsonl")
    assert "__" in filename


def test_subject_mask_publish_provenance_is_valid_for_subject_and_refined_runs(tmp_path: Path) -> None:
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "recording_analysis.zarr"),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="keypoints_runs",
        assignment_keypoint_run="keypoints_run",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=True,
    )
    ctx = mod.OutputStagingContext(
        source_zarr_path=tmp_path / "recording_analysis.zarr",
        staged_zarr_path=tmp_path / "scratch" / "recording_analysis.zarr",
        staging_root=tmp_path / "scratch",
    )
    publish_payload = {"schema": "test"}

    subject = mod._subject_mask_publish_provenance(  # noqa: SLF001
        ctx=ctx,
        plan=plan,
        publish_payload=publish_payload,
        refined=False,
    )
    refined = mod._subject_mask_publish_provenance(  # noqa: SLF001
        ctx=ctx,
        plan=plan,
        publish_payload=publish_payload,
        refined=True,
    )

    assert validate_run_provenance(subject).valid is True
    assert validate_run_provenance(refined).valid is True
    assert subject["input_run_ids"] == {
        "crop": "crop_run",
        "assignment_keypoints": "keypoints_run",
    }
    assert refined["input_run_ids"] == {
        "crop": "crop_run",
        "assignment_keypoints": "keypoints_run",
        "subject_mask": "subject_run",
    }


def test_subject_mask_publish_preserves_staged_input_artifacts(tmp_path: Path) -> None:
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "recording_analysis.zarr"),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="keypoints_runs",
        assignment_keypoint_run="keypoints_run",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=False,
    )
    ctx = mod.OutputStagingContext(
        source_zarr_path=tmp_path / "recording_analysis.zarr",
        staged_zarr_path=tmp_path / "scratch" / "recording_analysis.zarr",
        staging_root=tmp_path / "scratch",
    )
    run_group = SimpleNamespace(
        attrs={
            "run_provenance": {
                "input_artifacts": [
                    {
                        "role": "sam3_checkpoint",
                        "path": "/tmp/sam3.pt",
                        "fingerprint_scheme": "content_v1",
                        "sha256": "a" * 64,
                    }
                ]
            }
        }
    )

    provenance = mod._subject_mask_publish_provenance(  # noqa: SLF001
        ctx=ctx,
        plan=plan,
        publish_payload={"schema": "test"},
        refined=False,
    )
    provenance = mod.append_input_artifacts(  # noqa: SLF001
        provenance,
        mod._existing_run_input_artifacts(run_group),  # noqa: SLF001
    )

    assert provenance["input_artifacts"] == [
        {
            "role": "sam3_checkpoint",
            "path": "/tmp/sam3.pt",
            "fingerprint_scheme": "content_v1",
            "sha256": "a" * 64,
        }
    ]


def test_run_group_storage_stats_reports_layout_and_top_level_file_pressure(tmp_path: Path) -> None:
    run_path = tmp_path / "refined_run"
    run = zarr.open_group(str(run_path), mode="w")
    run.create_array(
        "metric",
        data=np.arange(8, dtype=np.float32).reshape(4, 2),
        chunks=(2, 2),
        overwrite=True,
    )

    stats = mod._run_group_storage_stats(run_path)  # noqa: SLF001

    assert stats["schema"] == "palette_run_group_storage_stats_v1"
    assert stats["file_count"] == 4
    assert stats["metadata_file_count"] == 2
    assert stats["payload_file_count"] == 2
    assert stats["array_count"] == 1
    assert stats["stat_error_count"] == 0
    assert stats["metadata_error_count"] == 0
    assert stats["top_level"]["__root__"]["file_count"] == 1
    assert stats["top_level"]["metric"]["file_count"] == 3
    assert stats["arrays"][0]["path"] == "metric"
    assert stats["arrays"][0]["shape"] == [4, 2]
    assert stats["arrays"][0]["data_type"] == "float32"
    assert stats["arrays"][0]["chunk_shape"] == [2, 2]
    assert stats["arrays"][0]["file_count"] == 3
    assert stats["arrays"][0]["metadata_file_count"] == 1
    assert stats["arrays"][0]["payload_file_count"] == 2
    assert stats["arrays"][0]["apparent_bytes"] > 0
    assert stats["scan_duration_seconds"] >= 0.0


def test_consolidate_metadata_quietly_suppresses_expected_sidecar_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_consolidate(store: str, *, path: str | None = None) -> None:
        assert store == "/tmp/archive.zarr"
        assert path is None
        warnings.warn("Object at logs is not recognized as a component of a Zarr hierarchy.", UserWarning)
        warnings.warn("Object at .failed is not recognized as a component of a Zarr hierarchy.", UserWarning)
        warnings.warn("Object at .imports is not recognized as a component of a Zarr hierarchy.", UserWarning)
        warnings.warn("Object at .incoming is not recognized as a component of a Zarr hierarchy.", UserWarning)

    monkeypatch.setattr(mod.zarr, "consolidate_metadata", _fake_consolidate)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod._consolidate_metadata_quietly("/tmp/archive.zarr")

    assert caught == []


def test_consolidate_metadata_quietly_does_not_hide_unexpected_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_consolidate(store: str, *, path: str | None = None) -> None:
        assert store == "/tmp/archive.zarr"
        assert path is None
        warnings.warn("unexpected zarr consolidation issue", UserWarning)

    monkeypatch.setattr(mod.zarr, "consolidate_metadata", _fake_consolidate)

    with pytest.warns(UserWarning, match="unexpected zarr consolidation issue"):
        mod._consolidate_metadata_quietly("/tmp/archive.zarr")


def test_refresh_subject_mask_registry_views_updates_summary_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "recording_analysis.zarr"
    calls: list[tuple[str, str, Path, str, str]] = []
    closed: list[bool] = []

    class _FakeConn:
        def execute(self, _sql: str, _params: tuple[str, str]) -> "_FakeConn":
            return self

        def fetchone(self) -> dict[str, str]:
            return {"dataset_id": "dataset-1", "recording_id": "recording-1", "zarr_use": "analysis"}

    class _FakeRegistry:
        def __init__(self, path: Path) -> None:
            assert path == registry_path
            self.conn = _FakeConn()

        def refresh_subject_mask_performance_for_dataset(
            self,
            dataset_id: str,
            *,
            zarr_path: Path,
            recording_id: str,
            zarr_use: str,
        ) -> int:
            calls.append(("performance", dataset_id, zarr_path, recording_id, zarr_use))
            return 2

        def refresh_subject_mask_component_quality_for_dataset(
            self,
            dataset_id: str,
            *,
            zarr_path: Path,
            recording_id: str,
            zarr_use: str,
        ) -> int:
            calls.append(("component_quality", dataset_id, zarr_path, recording_id, zarr_use))
            return 8

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)

    result = mod._refresh_subject_mask_registry_views(registry_path=registry_path, zarr_path=zarr_path)

    assert result["registry_refresh_status"] == "ok"
    assert result["dataset_id"] == "dataset-1"
    assert result["subject_mask_performance_rows"] == 2
    assert result["subject_mask_component_quality_rows"] == 8
    assert calls == [
        ("performance", "dataset-1", zarr_path.resolve(), "recording-1", "analysis"),
        ("component_quality", "dataset-1", zarr_path.resolve(), "recording-1", "analysis"),
    ]
    assert closed == [True]


def test_refresh_subject_mask_registry_views_skips_without_dataset_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeConn:
        def execute(self, _sql: str, _params: tuple[str, str]) -> "_FakeConn":
            return self

        def fetchone(self) -> None:
            return None

    class _FakeRegistry:
        def __init__(self, _path: Path) -> None:
            self.conn = _FakeConn()

        def close(self) -> None:
            pass

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)

    result = mod._refresh_subject_mask_registry_views(
        registry_path=tmp_path / "registry.sqlite",
        zarr_path=tmp_path / "missing_analysis.zarr",
    )

    assert result["registry_refresh_status"] == "skipped"
    assert result["reason"] == "dataset_not_in_registry"


def test_zarr_paths_from_report_reads_unique_result_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    {"zarr_path": "/data/a_analysis.zarr"},
                    {"zarr_path": "/data/b_analysis.zarr"},
                    {"zarr_path": "/data/a_analysis.zarr"},
                    {"not_zarr_path": "/data/ignored.zarr"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert mod._zarr_paths_from_report(report) == [
        Path("/data/a_analysis.zarr"),
        Path("/data/b_analysis.zarr"),
    ]


def test_zarr_paths_from_report_falls_back_to_plan_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"plans": [{"zarr_path": "/data/planned_analysis.zarr"}]}), encoding="utf-8")

    assert mod._zarr_paths_from_report(report) == [Path("/data/planned_analysis.zarr")]


def test_zarr_paths_from_report_requires_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"results": [{"error": "missing"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="did not contain any zarr_path"):
        mod._zarr_paths_from_report(report)


def test_resolve_crop_run_prefers_latest_any_for_geometry_only(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.create_group("crop_materialized").attrs["crop_storage_mode"] = "materialized"
    crop_parent.create_group("crop_geometry").attrs["crop_storage_mode"] = "geometry_only"

    assert mod._resolve_crop_run(zarr_path) == "crop_geometry"


def test_emit_paths_prints_selected_archives(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.create_group("crop_geometry")
    keypoints_parent = root.require_group("keypoints_runs")
    keypoints_parent.attrs["latest"] = "keypoints_latest"
    keypoints_parent.create_group("keypoints_latest")

    assert mod.main([str(tmp_path), "--emit-paths"]) == 0

    assert capsys.readouterr().out.strip() == str(zarr_path)


def test_validate_outputs_accepts_compact_refined_mask_store(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    subject_parent = root.require_group("subject_mask_runs")
    subject = subject_parent.create_group("subject_run")
    subject.attrs["mask_labels"] = list(mod.RAW_COMPONENTS)
    subject.create_array("mask_probs_roi", data=np.zeros((2, 3, 4, 4), dtype=np.uint8))

    refined_parent = root.require_group("refined_subject_masks_runs")
    refined = refined_parent.create_group("refined_run")
    refined.attrs["mask_labels"] = list(mod.REFINED_COMPONENTS)
    refined.attrs["label_schema_id"] = "subject_v1_lr"
    masks = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    masks[:, 0, 1:3, 1:3] = 1
    write_component_rle_mask_store_from_dense(
        refined,
        masks,
        component_names=mod.REFINED_COMPONENTS,
        encode_row_chunk_size=1,
    )
    components = refined.require_group("components")
    for component in mod.REFINED_COMPONENTS:
        components.require_group(component)

    status, details = mod.validate_outputs(
        zarr_path,
        subject_run="subject_run",
        refined_run="refined_run",
    )

    assert status == "ok"
    assert "refined_mask_store=component_rle_v1" in details


def _seed_subject_mask_batch_prereqs(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_run"
    crop_parent.create_group("crop_run")
    keypoint_parent = root.require_group("refined_keypoints_runs")
    keypoint_parent.attrs["latest"] = "refined_keypoints"
    keypoint_parent.create_group("refined_keypoints")


def test_build_archive_plan_inference_mode_ignores_unrelated_subject_runs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root.require_group("subject_mask_runs").create_group("old_subject_run")

    plan = mod.build_archive_plan(
        zarr_path,
        subject_run_name="target_subject_run",
        refined_run_name="target_refined_run",
        force_inference=False,
        force_finalization=False,
        workflow_stage="inference",
    )

    assert plan.run_inference is True
    assert plan.run_finalization is False


def test_build_archive_plan_can_select_explicit_crop_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root["crop_runs"].create_group("clip_proxy")

    plan = mod.build_archive_plan(
        zarr_path,
        subject_run_name="target_subject_run",
        refined_run_name="target_refined_run",
        force_inference=False,
        force_finalization=False,
        workflow_stage="inference",
        crop_run_name="clip_proxy",
    )

    assert plan.crop_run == "clip_proxy"
    assert plan.run_inference is True


def test_build_archive_plan_rejects_missing_explicit_crop_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)

    plan = mod.build_archive_plan(
        zarr_path,
        subject_run_name="target_subject_run",
        refined_run_name="target_refined_run",
        force_inference=False,
        force_finalization=False,
        workflow_stage="inference",
        crop_run_name="missing_proxy",
    )

    assert plan.crop_run is None
    assert plan.run_inference is False
    assert plan.skip_reason == "missing_crop_run"


def test_inference_only_plan_can_omit_assignment_keypoints(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)

    plan = mod.build_archive_plan(
        zarr_path,
        subject_run_name="target_subject_run",
        refined_run_name="target_refined_run",
        force_inference=False,
        force_finalization=False,
        workflow_stage="inference",
        resolve_assignment_keypoints=False,
    )

    assert plan.assignment_keypoint_group is None
    assert plan.assignment_keypoint_run is None
    assert plan.run_inference is True


def test_finalization_plan_binds_explicit_assignment_keypoint_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    exact = root["refined_keypoints_runs"].create_group("refined_keypoints_exact")
    exact.attrs["palette_run_completion_status"] = "complete"
    root.require_group("subject_mask_runs").create_group("target_subject_run")

    plan = mod.build_archive_plan(
        zarr_path,
        subject_run_name="target_subject_run",
        refined_run_name="target_refined_run",
        force_inference=False,
        force_finalization=False,
        workflow_stage="finalization",
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoint_run="refined_keypoints_exact",
    )

    assert plan.assignment_keypoint_group == "refined_keypoints_runs"
    assert plan.assignment_keypoint_run == "refined_keypoints_exact"
    assert plan.run_finalization is True


def test_finalization_plan_fails_closed_when_explicit_keypoints_are_missing(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root.require_group("subject_mask_runs").create_group("target_subject_run")

    with pytest.raises(FileNotFoundError, match="missing_exact_run"):
        mod.build_archive_plan(
            zarr_path,
            subject_run_name="target_subject_run",
            refined_run_name="target_refined_run",
            force_inference=False,
            force_finalization=False,
            workflow_stage="finalization",
            assignment_keypoint_group="refined_keypoints_runs",
            assignment_keypoint_run="missing_exact_run",
        )


def test_finalization_plan_fails_closed_when_explicit_keypoints_are_incomplete(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root["refined_keypoints_runs"].create_group("refined_keypoints_incomplete")
    root.require_group("subject_mask_runs").create_group("target_subject_run")

    with pytest.raises(RuntimeError, match="not complete"):
        mod.build_archive_plan(
            zarr_path,
            subject_run_name="target_subject_run",
            refined_run_name="target_refined_run",
            force_inference=False,
            force_finalization=False,
            workflow_stage="finalization",
            assignment_keypoint_group="refined_keypoints_runs",
            assignment_keypoint_run="refined_keypoints_incomplete",
        )


def test_build_archive_plan_finalization_mode_uses_latest_existing_subject_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    subject_parent = root.require_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "old_subject_run"
    subject_parent.create_group("old_subject_run")

    plan = mod.build_archive_plan(
        zarr_path,
        subject_run_name="target_subject_run",
        refined_run_name="target_refined_run",
        force_inference=False,
        force_finalization=False,
        workflow_stage="finalization",
    )

    assert plan.run_inference is False
    assert plan.run_finalization is True
    assert plan.subject_run == "old_subject_run"
    assert mod._selected_subject_run_for_finalization(plan) == "old_subject_run"


def test_build_archive_plan_finalization_mode_targets_matching_subject_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root.require_group("subject_mask_runs").create_group("target_subject_run")

    plan = mod.build_archive_plan(
        zarr_path,
        subject_run_name="target_subject_run",
        refined_run_name="target_refined_run",
        force_inference=False,
        force_finalization=False,
        workflow_stage="finalization",
    )

    assert plan.run_inference is False
    assert plan.run_finalization is True
    assert mod._selected_subject_run_for_finalization(plan) == "target_subject_run"


def test_inference_command_passes_cache_manifest_and_model_resolution_flags(tmp_path: Path) -> None:
    manifest = tmp_path / "sample.flat_roi_cache.json"
    args = SimpleNamespace(
        registry=tmp_path / "registry.sqlite",
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=7,
        model_require_unique=True,
        model_include_non_success=True,
        device="0",
        batch_size=192,
        mask_probs_dtype="uint8",
        mask_probs_chunk_rois=32,
        mask_probs_shard_rois=2048,
        output_queue_size=2,
        profile_timings=False,
        roi_cache_policy="never",
        roi_live_acceleration="auto",
        roi_live_gpu_chunk_frames=32,
        roi_cache_dir=None,
        roi_cache_manifest=manifest,
        overwrite=False,
    )
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "recording_analysis.zarr"),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoint_run="refined_run",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=True,
    )

    cmd = mod._inference_command(args, plan)

    assert "--roi-cache-manifest" in cmd
    assert cmd[cmd.index("--roi-cache-manifest") + 1] == str(manifest)
    assert "--roi-cache-expected-archive-path" not in cmd
    assert "--profile-timings" not in cmd
    assert "--model-require-unique" in cmd
    assert "--model-include-non-success" in cmd
    assert cmd[cmd.index("--model-top-k") + 1] == "7"
    assert cmd[cmd.index("--mask-probs-shard-rois") + 1] == "2048"


def test_inference_command_forwards_regular_probability_chunk_override(tmp_path: Path) -> None:
    args = SimpleNamespace(
        registry=tmp_path / "registry.sqlite",
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=5,
        model_require_unique=False,
        model_include_non_success=False,
        device="0",
        batch_size=128,
        mask_probs_dtype="uint8",
        mask_probs_chunk_rois=32,
        mask_probs_shard_rois=None,
        output_queue_size=2,
        profile_timings=False,
        roi_cache_policy="never",
        roi_live_acceleration="auto",
        roi_live_gpu_chunk_frames=32,
        roi_cache_dir=None,
        roi_cache_manifest=None,
        overwrite=False,
    )
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "recording_analysis.zarr"),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group=None,
        assignment_keypoint_run=None,
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=False,
    )

    cmd = mod._inference_command(args, plan)

    assert "--no-mask-probs-sharding" in cmd
    assert "--mask-probs-shard-rois" not in cmd


def test_inference_command_omits_missing_assignment_keypoints(tmp_path: Path) -> None:
    args = SimpleNamespace(
        registry=tmp_path / "registry.sqlite",
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=1,
        model_require_unique=False,
        model_include_non_success=False,
        device="0",
        batch_size=128,
        mask_probs_dtype="uint8",
        mask_probs_chunk_rois=32,
        mask_probs_shard_rois=2048,
        output_queue_size=2,
        roi_cache_policy="always",
        roi_live_acceleration="auto",
        roi_live_gpu_chunk_frames=32,
        roi_cache_dir=None,
        roi_cache_manifest=None,
        source_roi_cache_alias_manifest=None,
        source_roi_cache_row_index_path=None,
        source_collection_id=None,
        source_collection_path=None,
        source_clip_id=None,
        source_clip_index=None,
        source_work_unit_id=None,
        source_shard_id=None,
        profile_timings=False,
        overwrite=False,
        subject_output_parent="subject_mask_shard_runs",
    )
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "recording_analysis.zarr"),
        subject_run="subject_run",
        refined_run="refined_run",
        subject_output_parent="subject_mask_shard_runs",
        crop_run="clip_proxy",
        assignment_keypoint_group=None,
        assignment_keypoint_run=None,
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=False,
    )

    cmd = mod._inference_command(args, plan)

    assert "--assignment-keypoint-group" not in cmd
    assert "--assignment-keypoint-run" not in cmd


def test_inference_command_can_validate_cache_against_canonical_archive(tmp_path: Path) -> None:
    manifest = tmp_path / "sample.flat_roi_cache.json"
    canonical_zarr = tmp_path / "recording_analysis.zarr"
    staged_zarr = tmp_path / "scratch" / "recording_analysis__subject_run.zarr"
    args = SimpleNamespace(
        registry=tmp_path / "registry.sqlite",
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=5,
        model_require_unique=False,
        model_include_non_success=False,
        device="0",
        batch_size=128,
        mask_probs_dtype="uint8",
        mask_probs_chunk_rois=32,
        output_queue_size=2,
        profile_timings=True,
        roi_cache_policy="never",
        roi_live_acceleration="auto",
        roi_live_gpu_chunk_frames=32,
        roi_cache_dir=None,
        roi_cache_manifest=manifest,
        overwrite=False,
    )
    plan = mod.ArchivePlan(
        zarr_path=str(staged_zarr),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoint_run="refined_run",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=False,
    )

    cmd = mod._inference_command(
        args,
        plan,
        defer_registry_status=True,
        roi_cache_expected_archive_path=canonical_zarr,
    )

    assert str(staged_zarr) in cmd
    assert cmd[cmd.index("--roi-cache-manifest") + 1] == str(manifest)
    assert cmd[cmd.index("--roi-cache-expected-archive-path") + 1] == str(canonical_zarr)
    assert "--profile-timings" in cmd
    assert "--defer-registry-status" in cmd


def test_inference_command_passes_shard_output_parent_and_lineage(tmp_path: Path) -> None:
    manifest = tmp_path / "clip_000004.alias.json"
    row_index = tmp_path / "clip_000004.rows.json"
    args = SimpleNamespace(
        registry=tmp_path / "registry.sqlite",
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=5,
        model_require_unique=False,
        model_include_non_success=False,
        device="0",
        batch_size=128,
        subject_output_parent=mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT,
        mask_probs_dtype="uint8",
        mask_probs_chunk_rois=32,
        output_queue_size=2,
        profile_timings=False,
        roi_cache_policy="never",
        roi_live_acceleration="auto",
        roi_live_gpu_chunk_frames=32,
        roi_cache_dir=None,
        roi_cache_manifest=manifest,
        source_roi_cache_alias_manifest=None,
        source_roi_cache_row_index_path=row_index,
        source_collection_id="sleepyfish_collection",
        source_collection_path="/groups/example_collection.zarr",
        source_clip_id="clip_000004",
        source_clip_index=4,
        source_work_unit_id="clip_000004_subject_masks",
        source_shard_id="clip_000004",
        overwrite=False,
    )
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "collection.zarr"),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_proxy",
        assignment_keypoint_group="keypoints_runs",
        assignment_keypoint_run="keypoints_clip_000004",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=False,
        subject_output_parent=mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT,
    )

    cmd = mod._inference_command(args, plan)

    assert cmd[cmd.index("--output-parent") + 1] == "subject_mask_shard_runs"
    assert cmd[cmd.index("--source-roi-cache-alias-manifest") + 1] == str(manifest)
    assert cmd[cmd.index("--source-roi-cache-row-index-path") + 1] == str(row_index)
    assert cmd[cmd.index("--source-collection-id") + 1] == "sleepyfish_collection"
    assert cmd[cmd.index("--source-collection-path") + 1] == "/groups/example_collection.zarr"
    assert cmd[cmd.index("--source-clip-id") + 1] == "clip_000004"
    assert cmd[cmd.index("--source-clip-index") + 1] == "4"
    assert cmd[cmd.index("--source-work-unit-id") + 1] == "clip_000004_subject_masks"
    assert cmd[cmd.index("--source-shard-id") + 1] == "clip_000004"


def test_validate_outputs_can_target_subject_mask_shard_parent(tmp_path: Path) -> None:
    zarr_path = tmp_path / "collection.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    run = root.require_group("subject_mask_shard_runs").create_group("subject_run")
    run.attrs["mask_labels"] = list(mod.RAW_COMPONENTS)
    run.create_array("mask_probs_roi", shape=(2, 3, 4, 4), dtype="u1")

    status, detail = mod.validate_outputs(
        zarr_path,
        subject_run="subject_run",
        refined_run="refined_run",
        subject_output_parent=mod.SUBJECT_MASK_SHARD_OUTPUT_PARENT,
        require_subject=True,
        require_refined=False,
    )

    assert status == "ok"
    assert "subject_mask_labels" in detail


def test_main_rejects_shard_output_parent_with_finalization() -> None:
    rc = mod.main(["/recordings", "--subject-output-parent", "subject_mask_shard_runs", "--workflow-stage", "all"])

    assert rc == 2


def test_finalization_command_passes_postcompute_options(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.require_group("subject_mask_runs").create_group("subject_run")
    args = SimpleNamespace(
        finalize_chunk_size=256,
        metric_level="cheap",
        finalize_execution_backend="process_shards",
        finalize_num_workers=8,
        finalize_dense_mask_row_chunk=512,
        finalize_postcompute_backend="process_shards",
        finalize_postcompute_chunk_size=512,
        finalize_postcompute_num_workers=4,
        mask_storage="dense_and_rle",
        mask_rle_validation_mode="invariants",
        write_eye_geometry=True,
        write_component_contours=True,
        write_sampled_component_contours=True,
        sampled_contour_row_chunk=1024,
        sampled_contour_k=["subject_body=128"],
        retain_source_seeds=True,
        progress_dir=None,
        overwrite=False,
    )
    plan = mod.ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoint_run="refined_keypoints",
        has_subject_runs=True,
        has_refined_subject_runs=False,
        run_inference=False,
        run_finalization=True,
    )

    cmd = mod._finalization_command(args, plan)

    assert cmd[cmd.index("--postcompute-backend") + 1] == "process_shards"
    assert cmd[cmd.index("--postcompute-chunk-size") + 1] == "512"
    assert cmd[cmd.index("--postcompute-num-workers") + 1] == "4"
    assert cmd[cmd.index("--dense-mask-row-chunk") + 1] == "512"
    assert cmd[cmd.index("--mask-storage") + 1] == "dense_and_rle"
    assert cmd[cmd.index("--mask-rle-validation-mode") + 1] == "invariants"
    assert "--write-eye-geometry" in cmd
    assert "--write-component-contours" in cmd
    assert "--write-sampled-component-contours" in cmd
    assert cmd[cmd.index("--sampled-contour-row-chunk") + 1] == "1024"
    assert cmd[cmd.index("--sampled-contour-k") + 1] == "subject_body=128"
    assert "--retain-source-seeds" in cmd


def test_finalization_command_uses_length_safe_progress_filename(tmp_path: Path) -> None:
    args = SimpleNamespace(
        finalize_chunk_size=256,
        metric_level="cheap",
        finalize_execution_backend="process_shards",
        finalize_num_workers=8,
        finalize_dense_mask_row_chunk=None,
        finalize_postcompute_backend="process_shards",
        finalize_postcompute_chunk_size=None,
        finalize_postcompute_num_workers=None,
        mask_storage="dense_and_rle",
        mask_rle_validation_mode="invariants",
        write_eye_geometry=True,
        write_component_contours=True,
        write_sampled_component_contours=True,
        sampled_contour_row_chunk=1024,
        sampled_contour_k=[],
        retain_source_seeds=False,
        progress_dir=tmp_path / "progress",
        overwrite=False,
    )
    long_run = "refined_subject_masks_" + ("invariant_validation_" * 20)
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "recording_analysis.zarr"),
        subject_run="subject_run",
        refined_run=long_run,
        crop_run="crop_run",
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoint_run="refined_keypoints",
        has_subject_runs=True,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=True,
    )

    cmd = mod._finalization_command(args, plan)
    progress_path = Path(cmd[cmd.index("--progress-jsonl") + 1])

    assert len(progress_path.name) <= mod.MAX_ARTIFACT_FILENAME_CHARS
    assert progress_path.name.endswith(".finalization.progress.jsonl")


def test_workflow_profile_path_uses_length_safe_filename(tmp_path: Path) -> None:
    args = SimpleNamespace(
        workflow_profile_dir=tmp_path / "profiles",
        progress_dir=None,
        workflow_stage="finalization",
    )
    long_run = "refined_subject_masks_" + ("invariant_validation_" * 20)
    plan = mod.ArchivePlan(
        zarr_path=str(tmp_path / "recording_analysis.zarr"),
        subject_run="subject_run",
        refined_run=long_run,
        crop_run="crop_run",
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoint_run="refined_keypoints",
        has_subject_runs=True,
        has_refined_subject_runs=False,
        run_inference=False,
        run_finalization=True,
    )

    profile_path = mod._workflow_profile_path(args, plan)

    assert profile_path is not None
    assert len(profile_path.name) <= mod.MAX_ARTIFACT_FILENAME_CHARS
    assert profile_path.name.endswith(".workflow.profile.jsonl")


def test_staged_output_overlay_keeps_inputs_symlinked_and_outputs_local(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["recording_id"] = "recording_1"
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_run"
    crop_parent.create_group("crop_run")
    root.require_group("keypoints_runs").create_group("keypoints_run")
    root.require_group("subject_mask_runs").attrs["latest"] = "old_subject"
    root.require_group("refined_subject_masks_runs")
    plan = mod.ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="keypoints_runs",
        assignment_keypoint_run="keypoints_run",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=True,
    )

    ctx = mod._prepare_output_staging_zarr(
        zarr_path,
        plan=plan,
        staging_root=tmp_path / "scratch",
        overwrite=False,
    )

    assert (ctx.staged_zarr_path / "crop_runs").is_symlink()
    assert (ctx.staged_zarr_path / "keypoints_runs").is_symlink()
    assert not (ctx.staged_zarr_path / "subject_mask_runs").is_symlink()
    assert not (ctx.staged_zarr_path / "refined_subject_masks_runs").is_symlink()
    staged_root = zarr.open_group(str(ctx.staged_zarr_path), mode="r")
    assert staged_root.attrs["recording_id"] == "recording_1"
    assert staged_root["subject_mask_runs"].attrs["latest"] == "old_subject"


def test_staged_output_overlay_is_cleaned_after_inference_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _seed_subject_mask_batch_prereqs(zarr_path)
    scratch = tmp_path / "scratch"

    monkeypatch.setattr(mod, "_run_command", lambda *_args, **_kwargs: "failed_exit_99")

    exit_code = mod.main(
        [
            str(tmp_path),
            "--apply",
            "--workflow-stage",
            "inference",
            "--run-label",
            "cleanup_test",
            "--stage-output-to-scratch",
            "--output-staging-dir",
            str(scratch),
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--roi-cache-policy",
            "never",
        ]
    )

    assert exit_code == 1
    assert not list(scratch.glob("*.zarr"))


def test_staged_output_overlay_can_copy_finalization_input_subject_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    subject_parent = root.require_group("subject_mask_runs")
    source_subject = subject_parent.create_group("subject_run")
    source_subject.attrs["method"] = "unet_subject_mask_segmenter"
    source_subject.create_array("mask_probs_roi", data=np.zeros((1, 3, 2, 2), dtype=np.uint8))
    root.require_group("refined_subject_masks_runs")
    plan = mod.ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="keypoints_runs",
        assignment_keypoint_run="keypoints_run",
        has_subject_runs=True,
        has_refined_subject_runs=False,
        run_inference=False,
        run_finalization=True,
    )

    ctx = mod._prepare_output_staging_zarr(
        zarr_path,
        plan=plan,
        staging_root=tmp_path / "scratch",
        overwrite=False,
        stage_finalization_input=True,
    )

    staged_subject = ctx.staged_zarr_path / "subject_mask_runs" / "subject_run"
    assert staged_subject.is_dir()
    assert not staged_subject.is_symlink()
    staged_root = zarr.open_group(str(ctx.staged_zarr_path), mode="r")
    assert staged_root["subject_mask_runs/subject_run"].attrs["method"] == "unet_subject_mask_segmenter"
    assert np.asarray(staged_root["subject_mask_runs/subject_run/mask_probs_roi"]).shape == (1, 3, 2, 2)


def test_subject_mask_handoff_package_streams_finalization_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.require_group("subject_mask_runs")
    root.require_group("refined_subject_masks_runs")
    inference_plan = mod.ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="keypoints_runs",
        assignment_keypoint_run="keypoints_run",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=False,
    )
    ctx = mod._prepare_output_staging_zarr(
        zarr_path,
        plan=inference_plan,
        staging_root=tmp_path / "scratch_infer",
        overwrite=False,
    )
    staged = zarr.open_group(str(ctx.staged_zarr_path), mode="a")
    subject = staged["subject_mask_runs"].create_group("subject_run")
    subject.attrs["method"] = "unet_subject_mask_segmenter"
    subject.attrs["mask_labels"] = list(mod.RAW_COMPONENTS)
    subject.attrs["summary_statistics"] = {"rows_total": 1}
    subject.create_array("mask_probs_roi", data=np.ones((1, 3, 2, 2), dtype=np.uint8))

    monkeypatch.setattr(mod, "emit_subject_mask_stage_completion", lambda *_args, **_kwargs: True)

    handoff_dir = tmp_path / "nrs_handoff"
    mod._publish_staged_outputs(ctx, plan=inference_plan, overwrite=False, handoff_package_dir=handoff_dir)

    published = zarr.open_group(str(zarr_path), mode="r")
    package = published["subject_mask_runs/subject_run"].attrs["cluster_run_package"]
    package_path = Path(package["artifact_path"])
    assert package_path.is_file()
    assert package_path.parent == handoff_dir.resolve()

    finalization_plan = mod.ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="keypoints_runs",
        assignment_keypoint_run="keypoints_run",
        has_subject_runs=True,
        has_refined_subject_runs=False,
        run_inference=False,
        run_finalization=True,
    )
    finalization_ctx = mod._prepare_output_staging_zarr(
        zarr_path,
        plan=finalization_plan,
        staging_root=tmp_path / "scratch_finalize",
        overwrite=False,
        stage_finalization_input=True,
    )

    staged_subject = finalization_ctx.staged_zarr_path / "subject_mask_runs" / "subject_run"
    assert staged_subject.is_dir()
    assert not staged_subject.is_symlink()
    staged_root = zarr.open_group(str(finalization_ctx.staged_zarr_path), mode="r")
    assert int(np.asarray(staged_root["subject_mask_runs/subject_run/mask_probs_roi"][:]).sum()) == 12


def test_publish_staged_outputs_copies_groups_and_emits_real_path_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.require_group("subject_mask_runs")
    root.require_group("refined_subject_masks_runs")
    plan = mod.ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run="subject_run",
        refined_run="refined_run",
        crop_run="crop_run",
        assignment_keypoint_group="keypoints_runs",
        assignment_keypoint_run="keypoints_run",
        has_subject_runs=False,
        has_refined_subject_runs=False,
        run_inference=True,
        run_finalization=True,
    )
    ctx = mod._prepare_output_staging_zarr(
        zarr_path,
        plan=plan,
        staging_root=tmp_path / "scratch",
        overwrite=False,
    )
    staged = zarr.open_group(str(ctx.staged_zarr_path), mode="a")
    subject = staged["subject_mask_runs"].create_group("subject_run")
    subject.attrs["mask_labels"] = list(mod.RAW_COMPONENTS)
    subject.attrs["method"] = "unet_subject_mask_segmenter"
    subject.attrs["summary_statistics"] = {"rows_total": 1}
    subject.create_array("mask_probs_roi", data=np.zeros((1, 3, 2, 2), dtype=np.uint8))
    refined = staged["refined_subject_masks_runs"].create_group("refined_run")
    refined.attrs["mask_labels"] = list(mod.REFINED_COMPONENTS)
    refined.attrs["method"] = "smart_finalize_subject_masks_v1"
    refined.attrs["summary_statistics"] = {"rows_total": 1}
    refined.create_array("masks_roi", data=np.zeros((1, 4, 2, 2), dtype=np.uint8))
    components = refined.require_group("components")
    for component in mod.REFINED_COMPONENTS:
        components.require_group(component)

    emitted: list[tuple[str, str, Path]] = []

    def _emit_subject(_root, zarr_path_arg, *, run_name, **_kwargs):
        emitted.append(("subject", run_name, Path(zarr_path_arg)))
        return True

    def _emit_refined(_root, zarr_path_arg, *, run_name, **_kwargs):
        emitted.append(("refined", run_name, Path(zarr_path_arg)))
        return True

    monkeypatch.setattr(mod, "emit_subject_mask_stage_completion", _emit_subject)
    monkeypatch.setattr(mod, "emit_refined_subject_mask_stage_completion", _emit_refined)

    publish_summary = mod._publish_staged_outputs(ctx, plan=plan, overwrite=False)

    published = zarr.open_group(str(zarr_path), mode="r")
    assert "subject_run" in published["subject_mask_runs"]
    assert "refined_run" in published["refined_subject_masks_runs"]
    assert published["subject_mask_runs"].attrs["latest"] == "subject_run"
    assert published["refined_subject_masks_runs"].attrs["latest"] == "refined_run"
    assert published["subject_mask_runs"]["subject_run"].attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert published["refined_subject_masks_runs"]["refined_run"].attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert ("subject", "subject_run", zarr_path) in emitted
    assert ("refined", "refined_run", zarr_path) in emitted
    assert publish_summary["publish_backend"] == "multiple"
    assert publish_summary["published_run_group_count"] == 2
    assert publish_summary["publish_file_count"] > 0
    assert publish_summary["publish_apparent_bytes"] > 0
    assert publish_summary["publish_allocated_bytes"] >= 0
    assert publish_summary["publish_storage_scan_duration_seconds"] >= 0.0
    assert publish_summary["publish_copy_duration_seconds"] >= 0.0
    assert publish_summary["publish_commit_duration_seconds"] >= 0.0
    published_by_parent = {
        item["parent"]: item
        for item in publish_summary["published_run_groups"]
    }
    refined_publish = published_by_parent["refined_subject_masks_runs"]
    assert refined_publish["publish_backend"] == "shutil.copytree"
    assert refined_publish["storage_stats"]["schema"] == "palette_run_group_storage_stats_v1"
    assert refined_publish["storage_stats"]["top_level"]["masks_roi"]["file_count"] >= 1
    assert refined_publish["storage_stats"]["arrays"][0]["path"] == "masks_roi"
