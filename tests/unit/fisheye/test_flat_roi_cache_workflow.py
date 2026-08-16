from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fisheye.cluster import flat_roi_cache as mod
from fisheye.cluster.keypoints.common import FlatRoiCacheBinding
from fisheye.cluster.lsf import LsfExecutionMode, LsfResources


class _FakeGroup(dict):
    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})


def test_default_shared_cache_root_uses_johnson_nrs() -> None:
    assert mod.DEFAULT_SHARED_CACHE_ROOT == Path(
        "/nrs/johnson/palette_staging/flat_roi_cache"
    )


def test_publish_flat_roi_cache_builds_locally_and_publishes_canonical_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    analysis_zarr = tmp_path / "recording_analysis.zarr"
    analysis_zarr.mkdir()
    final_manifest = tmp_path / "nrs" / "recording.flat_roi_cache.json"
    scratch = tmp_path / "scratch"

    def fake_build_flat_roi_cache(**kwargs):
        local_manifest = Path(kwargs["manifest_path"])
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        local_payload = local_manifest.with_suffix(".bin")
        local_payload.write_bytes(b"\x01" * 12)
        payload = {
            "schema": "palette_roi_cache_flat_bin_v1",
            "layout": "flat_bin_v1",
            "cache_complete": True,
            "cache_key": "cache-key",
            "manifest_path": str(local_manifest),
            "source": {
                "archive_path": str(analysis_zarr.resolve()),
                "crop_run_name": "crop_001",
                "crop_signature": "signature-001",
                "crop_revision": "revision-001",
            },
            "array": {
                "bin_path": local_payload.name,
                "dtype": "uint8",
                "shape": [3, 2, 2],
                "order": "C",
                "total_bytes": 12,
                "sha256": None,
            },
        }
        local_manifest.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return payload

    monkeypatch.setattr(mod, "build_flat_roi_cache", fake_build_flat_roi_cache)

    report = mod.publish_flat_roi_cache(
        analysis_zarr=analysis_zarr,
        crop_run="crop_001",
        manifest_path=final_manifest,
        scratch_dir=scratch,
        batch_size=1024,
        decode_backend="pynvvc_luma",
        roi_live_acceleration="cpu",
        roi_live_gpu_chunk_frames=32,
        expected_contract={
            "crop_signature": "signature-001",
            "crop_revision": "revision-001",
            "shape": [3, 2, 2],
            "total_bytes": 12,
        },
    )

    final_payload = final_manifest.with_suffix(".bin")
    published = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["shape"] == [3, 2, 2]
    assert final_payload.read_bytes() == b"\x01" * 12
    assert published["manifest_path"] == str(final_manifest.resolve())
    assert published["publisher"]["publish_policy"] == (
        "payload_first_manifest_last"
    )


def test_cache_job_exposes_planned_artifacts_as_a_reusable_fragment_input(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").touch()
    manifest = tmp_path / "cache" / "recording.flat_roi_cache.json"
    cache = FlatRoiCacheBinding(
        manifest_path=manifest,
        manifest_sha256=None,
        payload_path=manifest.with_suffix(".bin"),
        crop_run="crop_001",
        cache_key=None,
        crop_signature="signature-001",
        crop_revision="revision-001",
        shape=(3, 348, 348),
        total_bytes=3 * 348 * 348,
        payload_sha256=None,
        availability="planned",
        producer_job_key="cache:recording_a",
    )

    job = mod.build_flat_roi_cache_job(
        workflow_id="workflow_a",
        target_id="recording_a",
        analysis_zarr=tmp_path / "recording_analysis.zarr",
        cache=cache,
        repo=repo,
        run_root=tmp_path / "run",
        resources=LsfResources(
            queue="gpu_l4", ncores=4, mem_gb=64, gpus=1, walltime="2:00"
        ),
    )

    assert job.job_key == "cache:recording_a"
    assert job.dependency is None
    assert "fisheye.cluster.flat_roi_cache" in job.command
    assert str(manifest) in job.command
    assert "--expected-contract-json" not in job.command
    contract_path = (
        tmp_path / "run" / "cache_contracts" / "recording_a.json"
    )
    assert "--expected-contract-json-file" in job.command
    assert str(contract_path) in job.command
    assert job.metadata["expected_contract_path"] == str(contract_path)
    assert job.metadata["publish_policy"] == "payload_first_manifest_last"


def test_cache_cli_loads_expected_contract_from_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract = {
        "crop_signature": "signature-001",
        "crop_revision": "revision-001",
        "shape": [3, 2, 2],
        "total_bytes": 12,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    captured = {}

    def fake_publish(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(mod, "publish_flat_roi_cache", fake_publish)

    result = mod.main(
        [
            "--analysis-zarr",
            str(tmp_path / "recording.zarr"),
            "--crop-run",
            "crop_001",
            "--manifest-path",
            str(tmp_path / "cache.json"),
            "--scratch-dir",
            str(tmp_path / "scratch"),
            "--expected-contract-json-file",
            str(contract_path),
        ]
    )

    assert result == 0
    assert captured["expected_contract"] == contract


def test_plan_flat_roi_cache_binding_reads_only_live_crop_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop = _FakeGroup(
        {
            "roi_coordinates_full": SimpleNamespace(shape=(7, 4)),
            "frame_indices": SimpleNamespace(shape=(7,)),
        },
        attrs={
            "roi_size": [384, 384],
            "crop_signature": {"source": "video-a"},
            "crop_revision": "revision-001",
        },
    )
    crop_parent = _FakeGroup({"crop_001": crop})
    root = _FakeGroup({"crop_runs": crop_parent})
    monkeypatch.setattr(
        mod,
        "open_zarr_group_direct",
        lambda *_args, **_kwargs: root,
    )
    monkeypatch.setattr(
        mod, "is_run_complete_in_parent", lambda *_args, **_kwargs: True
    )
    manifest = tmp_path / "cache" / "recording.flat_roi_cache.json"

    binding = mod.plan_flat_roi_cache_binding(
        analysis_zarr=tmp_path / "recording_analysis.zarr",
        crop_run="crop_001",
        manifest_path=manifest,
        producer_job_key="cache:recording_a",
    )

    assert binding.availability == "planned"
    assert binding.manifest_sha256 is None
    assert binding.payload_path == manifest.resolve().with_suffix(".bin")
    assert binding.shape == (7, 384, 384)
    assert binding.total_bytes == 7 * 384 * 384
    assert binding.source_kind == "unknown"
    assert not binding.nvdec_bundle_eligible


def test_plan_binding_marks_geometry_only_external_video_as_nvdec_bundle_eligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop = _FakeGroup(
        {
            "roi_coordinates_full": SimpleNamespace(shape=(11, 4)),
            "frame_indices": SimpleNamespace(shape=(11,)),
        },
        attrs={
            "roi_size": [512, 512],
            "crop_signature": {"source": "video-a"},
            "crop_revision": "revision-001",
            "crop_storage_mode": "geometry_only",
            "source_video_path": "/groups/recording/cam.mp4",
        },
    )
    crop_parent = _FakeGroup({"crop_001": crop})
    root = _FakeGroup(
        {"crop_runs": crop_parent},
        attrs={"video_width": 4512, "video_height": 4512},
    )
    monkeypatch.setattr(mod, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        mod, "is_run_complete_in_parent", lambda *_args, **_kwargs: True
    )

    binding = mod.plan_flat_roi_cache_binding(
        analysis_zarr=tmp_path / "recording_analysis.zarr",
        crop_run="crop_001",
        manifest_path=tmp_path / "cache" / "recording.flat_roi_cache.json",
        producer_job_key="cache:recording_a",
    )

    assert binding.source_kind == "source_video_path"
    assert binding.nvdec_bundle_eligible
    assert binding.nvdec_bundle_reason == (
        "geometry_only_external_video_with_known_dimensions"
    )


def test_plan_binding_resolves_v2_root_video_locator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recording = tmp_path / "recording"
    video = recording / "cams" / "source.mp4"
    analysis_zarr = recording / "zarr" / "analysis.zarr"
    crop = _FakeGroup(
        {
            "roi_coordinates_full": SimpleNamespace(shape=(11, 4)),
            "frame_indices": SimpleNamespace(shape=(11,)),
        },
        attrs={
            "roi_size": [512, 512],
            "crop_signature": {"source": "video-a"},
            "crop_revision": "revision-001",
            "crop_storage_mode": "geometry_only",
        },
    )
    root = _FakeGroup(
        {"crop_runs": _FakeGroup({"crop_001": crop})},
        attrs={
            "recording_path": str(recording),
            "source_video_path": str(video),
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "cams/source.mp4",
                },
            },
            "video_width": 4512,
            "video_height": 4512,
        },
    )
    monkeypatch.setattr(mod, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        mod, "is_run_complete_in_parent", lambda *_args, **_kwargs: True
    )

    binding = mod.plan_flat_roi_cache_binding(
        analysis_zarr=analysis_zarr,
        crop_run="crop_001",
        manifest_path=tmp_path / "cache" / "recording.flat_roi_cache.json",
        producer_job_key="cache:recording_a",
    )

    assert binding.source_kind == "source_video_path"
    assert binding.nvdec_bundle_eligible


def _planned_target(
    tmp_path: Path,
    target_id: str,
    *,
    total_bytes: int,
    nvdec_eligible: bool = True,
) -> mod.PlannedFlatRoiCacheTarget:
    manifest = tmp_path / "cache" / f"{target_id}.flat_roi_cache.json"
    cache = FlatRoiCacheBinding(
        manifest_path=manifest,
        manifest_sha256=None,
        payload_path=manifest.with_suffix(".bin"),
        crop_run=f"crop_{target_id}",
        cache_key=None,
        crop_signature=f"signature-{target_id}",
        crop_revision="revision-001",
        shape=(total_bytes, 1, 1),
        total_bytes=total_bytes,
        payload_sha256=None,
        availability="planned",
        producer_job_key=f"cache:{target_id}",
        source_kind=("source_video_path" if nvdec_eligible else "roi_images"),
        nvdec_bundle_eligible=nvdec_eligible,
        nvdec_bundle_reason=(
            "geometry_only_external_video_with_known_dimensions"
            if nvdec_eligible
            else "materialized_roi_images_do_not_use_nvdec"
        ),
    )
    return mod.PlannedFlatRoiCacheTarget(
        target_id=target_id,
        analysis_zarr=tmp_path / f"{target_id}.zarr",
        cache=cache,
    )


def test_cache_bundle_planner_packs_external_videos_by_sessions_and_payload(
    tmp_path: Path,
) -> None:
    targets = [
        _planned_target(tmp_path, f"recording_{index:02d}", total_bytes=10)
        for index in range(10)
    ]
    targets.append(
        _planned_target(
            tmp_path,
            "materialized",
            total_bytes=5,
            nvdec_eligible=False,
        )
    )

    bundles = mod.plan_flat_roi_cache_bundles(
        targets,
        max_workers=8,
        max_payload_bytes=80,
    )

    assert [len(bundle.targets) for bundle in bundles] == [8, 2, 1]
    assert [bundle.total_payload_bytes for bundle in bundles] == [80, 20, 5]
    assert [bundle.nvdec_bundle_eligible for bundle in bundles] == [True, True, False]
    assert bundles[0].targets[0].cache.producer_job_key == bundles[0].job_key
    assert bundles[2].targets[0].target_id == "materialized"


def test_cache_bundle_job_uses_bounded_in_allocation_tasks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").touch()
    bundle = mod.plan_flat_roi_cache_bundles(
        [
            _planned_target(tmp_path, f"recording_{index}", total_bytes=10)
            for index in range(3)
        ],
        max_workers=8,
        max_payload_bytes=100,
    )[0]

    job = mod.build_flat_roi_cache_bundle_job(
        workflow_id="daily_recordings",
        bundle=bundle,
        repo=repo,
        run_root=tmp_path / "run",
        resources=LsfResources(
            queue="gpu_l4",
            ncores=8,
            mem_gb=64,
            gpus=0,
            extra_lsf_args=("-gpu", "num=1:mode=shared:j_exclusive=no"),
        ),
    )

    assert job.execution_group is not None
    assert job.execution_group.mode is LsfExecutionMode.BUNDLE
    assert job.execution_group.max_concurrent == 3
    assert len(job.execution_group.tasks) == 3
    assert job.metadata["cache_bundle"]["total_payload_bytes"] == 30
    assert all(
        "fisheye.cluster.flat_roi_cache" in task.command
        for task in job.execution_group.tasks
    )
