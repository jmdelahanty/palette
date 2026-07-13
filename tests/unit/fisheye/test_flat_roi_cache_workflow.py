from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fisheye.cluster import flat_roi_cache as mod
from fisheye.cluster.keypoints.common import FlatRoiCacheBinding
from fisheye.cluster.lsf import LsfResources


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
            "roi_size": [348, 348],
            "crop_signature": {"source": "video-a"},
            "crop_revision": "revision-001",
        },
    )
    crop_parent = _FakeGroup({"crop_001": crop})
    root = _FakeGroup({"crop_runs": crop_parent})
    monkeypatch.setattr(mod, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
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
    assert binding.shape == (7, 348, 348)
    assert binding.total_bytes == 7 * 348 * 348
