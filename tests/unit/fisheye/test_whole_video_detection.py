from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.cluster.lsf import LsfExecutionMode, build_bsub_command
from fisheye.cluster.whole_video_detection import (
    AUTHORITATIVE_FULL_FRAME_ROLE,
    build_plan,
    discover_registry_whole_video_targets,
    materialize_plan_bundle,
)
from fisheye.cluster.clipped_detection import DetectionModelSpec
from fisheye.registry.db import Registry


def _seed_recording(
    registry: Registry,
    root: Path,
    *,
    recording_id: str,
    camera_id: str,
    dataset_suffix: str = "",
    stream_role: str = AUTHORITATIVE_FULL_FRAME_ROLE,
) -> tuple[Path, Path]:
    recording_dir = root / recording_id
    zarr_path = recording_dir / "zarr" / f"{recording_id}{dataset_suffix}_analysis.zarr"
    video_path = recording_dir / "cams" / f"Cam{camera_id}_{recording_id}.mp4"
    zarr_path.mkdir(parents=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    (zarr_path / "zarr.json").write_text(
        '{"zarr_format":3,"node_type":"group","attributes":{}}\n',
        encoding="utf-8",
    )
    video_path.write_bytes(b"video")
    dataset_id = f"dataset_{recording_id}{dataset_suffix}"
    registry.upsert_dataset(
        dataset_id,
        session_uuid=f"session_{recording_id}",
        zarr_path=zarr_path,
        recording_id=recording_id,
        zarr_use="analysis",
        zarr_origin="recording_analysis",
    )
    registry.upsert_recording(
        recording_id=recording_id,
        session_uuid=f"session_{recording_id}",
        recording_name=recording_id,
        recording_path=str(recording_dir),
        recording_type="behavior",
        camera_id=camera_id,
    )
    registry.conn.execute(
        """
        INSERT INTO acquisition_video_streams (
            dataset_id, stream_key, recording_id, zarr_use, role, output_kind,
            camera_id, video_path, availability_status, inventory_status,
            video_exists, updated_utc
        ) VALUES (?, 'full', ?, 'analysis', ?, 'full', ?, ?, 'ok', 'complete', 1,
                  '2026-07-27T00:00:00Z')
        """,
        (dataset_id, recording_id, stream_role, camera_id, str(video_path)),
    )
    registry.conn.commit()
    return zarr_path, video_path


def test_registry_discovery_resolves_exact_authoritative_full_streams(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        first_zarr, first_video = _seed_recording(
            registry,
            tmp_path / "recordings",
            recording_id="2026-07-21T19-38-32Z_arena_1_Batman",
            camera_id="2010093",
        )
        _seed_recording(
            registry,
            tmp_path / "recordings",
            recording_id="2026-07-21T19-38-32Z_arena_2_Batman",
            camera_id="2010094",
        )
        targets = discover_registry_whole_video_targets(
            registry,
            path_contains="Batman",
        )
    finally:
        registry.close()

    assert len(targets) == 2
    assert targets[0].target.analysis_zarr == first_zarr.resolve()
    assert targets[0].target.work_units[0].video_path == first_video.resolve()
    assert targets[0].target.work_units[0].frame_mapping.mode.value == "identity"
    assert targets[0].stream_role == AUTHORITATIVE_FULL_FRAME_ROLE


def test_registry_discovery_fails_closed_on_ambiguous_analysis_dataset(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        recording_id = "2026-07-21T19-38-32Z_arena_1_Batman"
        _seed_recording(
            registry,
            tmp_path / "recordings",
            recording_id=recording_id,
            camera_id="2010093",
        )
        _seed_recording(
            registry,
            tmp_path / "recordings",
            recording_id=recording_id,
            camera_id="2010093",
            dataset_suffix="_duplicate",
        )
        with pytest.raises(ValueError, match="one active analysis dataset"):
            discover_registry_whole_video_targets(
                registry,
                recording_ids=(recording_id,),
            )
    finally:
        registry.close()


def test_whole_video_cohort_is_one_bounded_atomic_detection_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        for index in range(3):
            _seed_recording(
                registry,
                tmp_path / "recordings",
                recording_id=f"2026-07-21T19-38-32Z_arena_{index + 1}_Batman",
                camera_id=str(2010093 + index),
            )
    finally:
        registry.close()

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").write_text("#!/bin/sh\n", encoding="utf-8")
    model_path = tmp_path / "models" / "detect.pt"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")
    model = DetectionModelSpec(
        set_id="detect_set",
        run_id="detect_model_run",
        path=model_path,
        sha256="d" * 64,
    )
    monkeypatch.setattr(
        "fisheye.cluster.whole_video_detection.resolve_detection_model_for_targets",
        lambda *_args, **_kwargs: model,
    )

    run_root = tmp_path / "run"
    plan = build_plan(
        registry_path=registry_path,
        repo=repo,
        run_root=run_root,
        run_label="batman_detection_canary",
        detection_set_id=model.set_id,
        detection_run_id=model.run_id,
        path_contains="Batman",
        max_concurrent=2,
    )

    assert len(plan.targets) == 3
    assert len(plan.lsf_workflow.jobs) == 1
    job = plan.lsf_workflow.jobs[0]
    assert job.execution_group is not None
    assert job.execution_group.mode is LsfExecutionMode.ARRAY
    assert job.execution_group.max_concurrent == 2
    assert len(job.execution_group.tasks) == 3
    assert job.resources.gpus == 1
    assert job.resources.queue == "gpu_l4"
    assert build_bsub_command(job)[2].endswith("[1-3]%2")
    commands = [task.command for task in job.execution_group.tasks]
    assert all(
        "fisheye.utils.run_detection_local_publish" in command for command in commands
    )
    assert all("--model-sha256" in command for command in commands)
    assert len({command[command.index("--zarr") + 1] for command in commands}) == 3
    assert len({command[command.index("--video") + 1] for command in commands}) == 3
    assert plan.lsf_workflow.metadata is not None
    assert plan.lsf_workflow.metadata["target_count"] == 3

    first = materialize_plan_bundle(plan)
    second = materialize_plan_bundle(plan)
    assert first == second
    assert (run_root / "plan.json").is_file()
    assert (run_root / "lsf_plan.json").is_file()


def test_registry_video_query_filters_current_authority(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        recording_id = "2026-07-21T19-38-32Z_arena_1_Batman"
        _seed_recording(
            registry,
            tmp_path / "recordings",
            recording_id=recording_id,
            camera_id="2010093",
        )
        rows = registry.query_acquisition_video_streams_current(
            recording_id=recording_id,
            output_kind="full",
            role=AUTHORITATIVE_FULL_FRAME_ROLE,
            availability_status="ok",
            require_video=True,
        )
        missing = registry.query_acquisition_video_streams_current(
            recording_id=recording_id,
            output_kind="crop",
            require_video=True,
        )
    finally:
        registry.close()

    assert len(rows) == 1
    assert rows[0]["camera_id"] == "2010093"
    assert missing == []
