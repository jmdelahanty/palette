from __future__ import annotations

import json
import warnings
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from fisheye.utils import run_subject_mask_batch_pipeline as mod


def test_parser_defaults_to_sharded_postcompute_for_batch_workflow() -> None:
    args = mod._build_parser().parse_args(["/recordings"])

    assert args.finalize_postcompute_backend == "process_shards"
    assert args.finalize_dense_mask_row_chunk == mod.DEFAULT_FINALIZE_DENSE_MASK_ROW_CHUNK
    assert args.mask_storage == "dense_uint8"
    assert args.mask_rle_validation_mode == "invariants"


def test_parser_accepts_compact_mask_storage_mode() -> None:
    args = mod._build_parser().parse_args(["/recordings", "--mask-storage", "rle_v1"])

    assert args.mask_storage == "rle_v1"


def test_safe_artifact_filename_hashes_long_names() -> None:
    filename = mod._safe_artifact_filename(
        ("recording_analysis", "subject_masks_" + ("very_long_" * 40)),
        ".workflow.profile.jsonl",
    )

    assert len(filename) <= mod.MAX_ARTIFACT_FILENAME_CHARS
    assert filename.endswith(".workflow.profile.jsonl")
    assert "__" in filename


def test_consolidate_metadata_quietly_suppresses_expected_zarr_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_consolidate(path: str) -> None:
        assert path == "/tmp/archive.zarr"
        warnings.warn("Object at logs is not recognized as a component of a Zarr hierarchy.", UserWarning)
        warnings.warn("Object at .failed is not recognized as a component of a Zarr hierarchy.", UserWarning)
        warnings.warn("Object at .imports is not recognized as a component of a Zarr hierarchy.", UserWarning)
        warnings.warn("Object at .incoming is not recognized as a component of a Zarr hierarchy.", UserWarning)
        warnings.warn(
            "Consolidated metadata is currently not part in the Zarr format 3 specification. "
            "It may not be supported by other zarr implementations and may change in the future.",
            UserWarning,
        )

    monkeypatch.setattr(mod.zarr, "consolidate_metadata", _fake_consolidate)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod._consolidate_metadata_quietly("/tmp/archive.zarr")

    assert caught == []


def test_consolidate_metadata_quietly_does_not_hide_unexpected_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_consolidate(path: str) -> None:
        warnings.warn("unexpected zarr consolidation issue", UserWarning)

    monkeypatch.setattr(mod.zarr, "consolidate_metadata", _fake_consolidate)

    with pytest.warns(UserWarning, match="unexpected zarr consolidation issue"):
        mod._consolidate_metadata_quietly("/tmp/archive.zarr")


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


def test_build_archive_plan_finalization_mode_requires_matching_subject_run(tmp_path: Path) -> None:
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
        workflow_stage="finalization",
    )

    assert plan.run_inference is False
    assert plan.run_finalization is False
    assert "target_subject_mask_run_missing" in plan.skip_reason


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


def test_finalization_command_passes_postcompute_options(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.require_group("subject_mask_runs").create_group("subject_run")
    args = SimpleNamespace(
        finalize_chunk_size=256,
        metric_level="cheap",
        finalize_execution_backend="process_shards",
        finalize_scheduler="processes",
        finalize_num_workers=8,
        finalize_dense_mask_row_chunk=512,
        finalize_postcompute_backend="process_shards",
        finalize_postcompute_chunk_size=512,
        finalize_postcompute_num_workers=4,
        mask_storage="dense_and_rle",
        mask_rle_validation_mode="invariants",
        write_eye_geometry=True,
        write_component_contours=True,
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
    assert "--retain-source-seeds" in cmd


def test_finalization_command_uses_length_safe_progress_filename(tmp_path: Path) -> None:
    args = SimpleNamespace(
        finalize_chunk_size=256,
        metric_level="cheap",
        finalize_execution_backend="process_shards",
        finalize_scheduler="processes",
        finalize_num_workers=8,
        finalize_dense_mask_row_chunk=None,
        finalize_postcompute_backend="process_shards",
        finalize_postcompute_chunk_size=None,
        finalize_postcompute_num_workers=None,
        mask_storage="dense_and_rle",
        mask_rle_validation_mode="invariants",
        write_eye_geometry=True,
        write_component_contours=True,
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

    mod._publish_staged_outputs(ctx, plan=plan, overwrite=False)

    published = zarr.open_group(str(zarr_path), mode="r")
    assert "subject_run" in published["subject_mask_runs"]
    assert "refined_run" in published["refined_subject_masks_runs"]
    assert published["subject_mask_runs"].attrs["latest"] == "subject_run"
    assert published["refined_subject_masks_runs"].attrs["latest"] == "refined_run"
    assert published["subject_mask_runs"]["subject_run"].attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert published["refined_subject_masks_runs"]["refined_run"].attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert ("subject", "subject_run", zarr_path) in emitted
    assert ("refined", "refined_run", zarr_path) in emitted
