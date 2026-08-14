from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fisheye.cluster.clipped_inference import CampaignTarget, ModelBinding
from fisheye.cluster import native_detection_campaign as campaign
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _write_zarr_node(path: Path, attrs: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs,
            }
        ),
        encoding="utf-8",
    )


def _target(tmp_path: Path) -> CampaignTarget:
    recording = tmp_path / "recording"
    archive = recording / "zarr" / "recording_analysis.zarr"
    authority_path = "analysis/acquisition_camera_frames/2010093"
    record = {
        "schema_id": "palette.acquisition_camera_frame",
        "schema_version": 2,
        "recording_id": "recording_a",
        "camera_id": "2010093",
        "width_px": 640,
        "height_px": 480,
        "source_total_frames": 3,
        "frame_count": 3,
    }
    digest = canonical_json_sha256(record)
    _write_zarr_node(
        archive,
        {
            "recording_id": "recording_a",
            "unrelated_legacy_metric": float("inf"),
        },
    )
    _write_zarr_node(
        archive / "raw_video",
        {
            "acquisition_authority_publication_status": {
                "status": "published_canonical_v1",
                "authority_path": authority_path,
            }
        },
    )
    _write_zarr_node(
        archive / authority_path,
        {
            "acquisition_camera_frame": record,
            "acquisition_camera_frame_sha256": digest,
        },
    )
    clips = recording / "clips"
    clips.mkdir(parents=True)
    (clips / "clip_000000.mp4").write_bytes(b"video")
    (recording / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "recording_id": "recording_a",
                "clips": [
                    {
                        "clip_id": "clip_000000",
                        "clip_index": 0,
                        "camera_serial": "2010093",
                        "video_path": "clips/clip_000000.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "camera_serial": ["2010093"] * 3,
                "clip_id": ["clip_000000"] * 3,
                "clip_local_frame_index": [0, 1, 2],
                "parent_frame_index": [0, 1, 2],
            }
        ),
        recording / "recording_frame_index.parquet",
    )
    return CampaignTarget(
        target_id="recording_a",
        recording_id="recording_a",
        recording_dir=recording,
        analysis_zarr=archive,
    )


def test_archive_authority_uses_one_verified_acquisition_record(tmp_path: Path) -> None:
    authority = campaign.load_native_archive_authority(_target(tmp_path))

    assert authority.n_frames == 3
    assert (authority.source_width, authority.source_height) == (640, 480)
    assert authority.frame == authority.pixel
    assert authority.frame.record_ref == (
        "/analysis/acquisition_camera_frames/2010093@acquisition_camera_frame"
    )


def test_campaign_materializes_canonical_v3_publish_and_registry_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = _target(tmp_path)
    repo = tmp_path / "repo"
    (repo / "configs" / "fisheye").mkdir(parents=True)
    (repo / "configs" / "fisheye" / "yolo_detect_config.yaml").write_text(
        "model: fixture\n", encoding="utf-8"
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    monkeypatch.setattr(campaign, "_repo_commit", lambda _repo: "4" * 40)
    monkeypatch.setattr(
        campaign,
        "validate_registered_analysis_zarr",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        campaign,
        "resolve_detection_model_binding",
        lambda **_kwargs: ModelBinding(
            task="detect",
            set_id="detect_set",
            run_id="detect_run",
            path=model,
            sha256="3" * 64,
        ),
    )
    run_root = tmp_path / "run"
    plan = campaign.build_plan(
        targets=(target,),
        run_label="canary",
        repo=repo,
        registry_path=tmp_path / "registry.sqlite",
        run_root=run_root,
        detection_set_id="detect_set",
        detection_run_id="detect_run",
        detect_array_concurrency=1,
    )

    payload = campaign.materialize_plan_bundle(plan)

    assert payload["selector_activation"] == (
        "atomic_after_canonical_v3_validation"
    )
    assert payload["registry_update"] == "serial_safe_shadow_reconciliation"
    assert len(plan.workflow.jobs) == 3
    array_job, publish_job, registry_job = plan.workflow.topological_jobs()
    assert array_job.metadata["execution_mode"] == "array"
    assert publish_job.dependency.upstream_job_keys == (array_job.job_key,)
    assert registry_job.dependency.upstream_job_keys == (publish_job.job_key,)
    assert "fisheye.utils.registry_rescan" in registry_job.command
    assert (run_root / "plan.json").is_file()
    assert (run_root / "lsf_plan.json").is_file()
    assert (run_root / "targets" / "recording_a" / "detection_plan.json").is_file()
    assert plan.target_plans[0]["authority"]["source_frame_authority"] == (
        plan.target_plans[0]["authority"]["source_pixel_authority"]
    )
