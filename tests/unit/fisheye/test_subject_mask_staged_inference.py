from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fisheye.cluster.subject_masks.staged_inference as staged


def _cache_manifest(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    payload = bytes(range(32))
    binary = source / "clip.bin"
    binary.write_bytes(payload)
    manifest = source / "clip.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "palette_roi_cache_flat_bin_v1",
                "layout": "flat_bin_v1",
                "cache_complete": True,
                "source": {
                    "archive_path": str(tmp_path / "recording_analysis.zarr"),
                    "crop_run_name": "crop_clip_001",
                },
                "array": {
                    "bin_path": binary.name,
                    "dtype": "uint8",
                    "shape": [2, 4, 4],
                    "order": "C",
                    "total_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_staged_inference_forwards_authenticated_local_manifest_and_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = _cache_manifest(tmp_path)
    staging_dir = tmp_path / "scratch"
    receipt_path = tmp_path / "receipts" / "worker.json"
    calls: list[list[str]] = []

    def fake_inference(arguments: list[str]) -> None:
        calls.append(arguments)
        manifest_index = arguments.index("--roi-cache-manifest") + 1
        local_manifest = Path(arguments[manifest_index])
        assert local_manifest.parent == staging_dir
        local = json.loads(local_manifest.read_text(encoding="utf-8"))
        local_payload = local_manifest.parent / local["array"]["bin_path"]
        assert local_payload.read_bytes() == bytes(range(32))

    monkeypatch.setattr(staged.infer_unet_subject_masks, "main", fake_inference)

    def fake_evidence(arguments: list[str]) -> dict[str, object]:
        attempt_id = arguments[arguments.index("--attempt-id") + 1]
        return {
            "attempt_id": attempt_id,
            "source_roi_pixels_sha256": hashlib.sha256(bytes(range(32))).hexdigest(),
        }

    monkeypatch.setattr(staged, "_completed_run_evidence", fake_evidence)

    staged.main(
        [
            str(tmp_path / "recording_analysis.zarr"),
            "--run-name",
            "clip_001",
            "--output-parent",
            "subject_mask_shard_runs",
            "--roi-cache-manifest",
            str(source_manifest),
            "--roi-cache-staging-dir",
            str(staging_dir),
            "--worker-receipt-json",
            str(receipt_path),
        ]
    )

    assert len(calls) == 1
    assert not staging_dir.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["roi_cache_staging"]["verification"] == (
        "single_pass_copy_stream_sha256_v1"
    )
    assert receipt["roi_cache_staging"]["copy"]["staged_sha256"] == (
        hashlib.sha256(bytes(range(32))).hexdigest()
    )


def test_staged_inference_writes_failed_receipt_and_cleans_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = _cache_manifest(tmp_path)
    staging_dir = tmp_path / "scratch"
    receipt_path = tmp_path / "receipts" / "worker.json"

    def fail(_arguments: list[str]) -> None:
        raise RuntimeError("inference failed")

    monkeypatch.setattr(staged.infer_unet_subject_masks, "main", fail)

    with pytest.raises(RuntimeError, match="inference failed"):
        staged.main(
            [
                str(tmp_path / "recording_analysis.zarr"),
                "--run-name",
                "clip_001",
                "--output-parent",
                "subject_mask_shard_runs",
                "--roi-cache-manifest",
                str(source_manifest),
                "--roi-cache-staging-dir",
                str(staging_dir),
                "--worker-receipt-json",
                str(receipt_path),
            ]
        )

    assert not staging_dir.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "RuntimeError"


def test_direct_crop_provider_forwards_row_partition_without_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "receipts" / "worker.json"
    calls: list[list[str]] = []

    monkeypatch.setattr(
        staged.infer_unet_subject_masks,
        "main",
        lambda arguments: calls.append(arguments),
    )

    def fake_evidence(arguments: list[str]) -> dict[str, object]:
        return {
            "attempt_id": arguments[arguments.index("--attempt-id") + 1],
            "source_crop_run": "crop_hybrid_recording",
        }

    monkeypatch.setattr(staged, "_completed_run_evidence", fake_evidence)

    staged.main(
        [
            "--direct-crop-provider",
            "--worker-receipt-json",
            str(receipt_path),
            str(tmp_path / "recording_analysis.zarr"),
            "--run-name",
            "clip_001",
            "--output-parent",
            "subject_mask_shard_runs",
            "--crop-run",
            "crop_hybrid_recording",
            "--source-crop-row-start",
            "100",
            "--source-crop-row-stop",
            "200",
        ]
    )

    assert len(calls) == 1
    assert "--roi-cache-manifest" not in calls[0]
    assert calls[0][calls[0].index("--source-crop-row-start") + 1] == "100"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["pixel_source_mode"] == "direct_crop_provider"
    assert receipt["staging_dir"] is None
