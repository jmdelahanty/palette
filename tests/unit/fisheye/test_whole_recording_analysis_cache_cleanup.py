from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fisheye.cluster import whole_recording_analysis_cache_cleanup as mod


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_cache(root: Path, name: str) -> tuple[Path, Path, str]:
    cache_dir = root / "wave_01" / "roi_cache"
    payload_path = cache_dir / f"{name}.flat_roi_cache.bin"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes((name.encode("utf-8") + b"\x00") * 8)
    manifest_path = cache_dir / f"{name}.flat_roi_cache.json"
    _write_json(
        manifest_path,
        {
            "schema": "palette_roi_cache_flat_bin_v1",
            "layout": "flat_bin_v1",
            "cache_complete": True,
            "array": {
                "bin_path": payload_path.name,
                "dtype": "uint8",
                "shape": [1, 1, payload_path.stat().st_size],
                "order": "C",
                "total_bytes": payload_path.stat().st_size,
            },
        },
    )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_path, payload_path, digest


def _make_plan(path: Path, targets: list[dict[str, object]]) -> Path:
    _write_json(
        path,
        {
            "schema": mod.ANALYSIS_PLAN_SCHEMA,
            "target_count": len(targets),
            "targets": targets,
        },
    )
    return path


def test_cleanup_is_planned_by_default_and_deletes_only_exact_cache_files(
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "nrs" / "flat_roi_cache"
    manifest_a, payload_a, digest_a = _make_cache(allowed_root, "recording_a")
    manifest_b, payload_b, digest_b = _make_cache(allowed_root, "recording_b")
    unrelated = manifest_a.parent / "keep.txt"
    unrelated.write_text("keep\n", encoding="utf-8")
    plan = _make_plan(
        tmp_path / "run" / "plan.json",
        [
            {
                "target_id": "recording_a",
                "roi_cache_manifest": str(manifest_a),
                "roi_cache_manifest_sha256": digest_a,
                "roi_cache_payload": str(payload_a),
            },
            {
                "target_id": "recording_b",
                "roi_cache_manifest": str(manifest_b),
                "roi_cache_manifest_sha256": digest_b,
                "roi_cache_payload": str(payload_b),
            },
        ],
    )

    planned = mod.cleanup_roi_caches(plan, allowed_root=allowed_root)
    assert planned["status"] == "planned"
    assert planned["cache_count"] == 2
    assert manifest_a.exists() and payload_a.exists()

    deleted = mod.cleanup_roi_caches(plan, allowed_root=allowed_root, apply=True)
    assert deleted["status"] == "deleted"
    assert len(deleted["deleted_paths"]) == 4
    assert not manifest_a.exists() and not payload_a.exists()
    assert not manifest_b.exists() and not payload_b.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_cleanup_rejects_artifact_outside_allowed_root_before_deleting(
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "allowed"
    manifest_a, payload_a, digest_a = _make_cache(allowed_root, "recording_a")
    manifest_b, payload_b, digest_b = _make_cache(tmp_path / "outside", "recording_b")
    plan = _make_plan(
        tmp_path / "plan.json",
        [
            {
                "target_id": "recording_a",
                "roi_cache_manifest": str(manifest_a),
                "roi_cache_manifest_sha256": digest_a,
                "roi_cache_payload": str(payload_a),
            },
            {
                "target_id": "recording_b",
                "roi_cache_manifest": str(manifest_b),
                "roi_cache_manifest_sha256": digest_b,
                "roi_cache_payload": str(payload_b),
            },
        ],
    )

    with pytest.raises(ValueError, match="outside allowed root"):
        mod.cleanup_roi_caches(plan, allowed_root=allowed_root, apply=True)

    assert manifest_a.exists() and payload_a.exists()
    assert manifest_b.exists() and payload_b.exists()


def test_cleanup_rejects_changed_manifest_before_deleting(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    manifest, payload, digest = _make_cache(allowed_root, "recording_a")
    plan = _make_plan(
        tmp_path / "plan.json",
        [
            {
                "target_id": "recording_a",
                "roi_cache_manifest": str(manifest),
                "roi_cache_manifest_sha256": digest,
                "roi_cache_payload": str(payload),
            }
        ],
    )
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest changed"):
        mod.cleanup_roi_caches(plan, allowed_root=allowed_root, apply=True)

    assert manifest.exists() and payload.exists()


def test_cleanup_accepts_late_bound_manifest_after_validating_source_identity(
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "allowed"
    manifest, payload, _digest = _make_cache(allowed_root, "recording_a")
    analysis_zarr = tmp_path / "recording_a_analysis.zarr"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["source"] = {
        "archive_path": str(analysis_zarr),
        "crop_run_name": "crop_001",
        "crop_signature": "signature-001",
        "crop_revision": "revision-001",
    }
    _write_json(manifest, manifest_payload)
    plan = _make_plan(
        tmp_path / "plan.json",
        [
            {
                "target_id": "recording_a",
                "analysis_zarr": str(analysis_zarr),
                "crop_run": "crop_001",
                "roi_cache_manifest": str(manifest),
                "roi_cache_manifest_sha256": None,
                "roi_cache_payload": str(payload),
                "roi_cache_contract": {
                    "crop_signature": "signature-001",
                    "crop_revision": "revision-001",
                    "shape": manifest_payload["array"]["shape"],
                    "total_bytes": manifest_payload["array"]["total_bytes"],
                },
            }
        ],
    )

    artifacts = mod.load_cleanup_artifacts(plan, allowed_root=allowed_root)

    assert len(artifacts) == 1
    assert artifacts[0].manifest_sha256 == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()


def test_repair_cleanup_plan_backfills_exact_binding_without_mutating_source(
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "allowed"
    manifest, payload, digest = _make_cache(allowed_root, "recording_a")
    analysis_zarr = tmp_path / "recording_a_analysis.zarr"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["source"] = {
        "archive_path": str(analysis_zarr),
        "crop_run_name": "crop_001",
        "crop_signature": "signature-001",
        "crop_revision": "revision-001",
    }
    _write_json(manifest, manifest_payload)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    keypoint_plan = tmp_path / "run" / "keypoints" / "plan.json"
    _write_json(
        keypoint_plan,
        {
            "schema": mod.KEYPOINT_PLAN_SCHEMA,
            "targets": [
                {
                    "target": {
                        "target_id": "recording_a",
                        "analysis_zarr": str(analysis_zarr),
                        "roi_cache_manifest": str(manifest),
                    },
                    "cache": {
                        "manifest_path": str(manifest),
                        "manifest_sha256": digest,
                        "payload_path": str(payload),
                        "crop_run": "crop_001",
                        "crop_signature": "signature-001",
                        "crop_revision": "revision-001",
                        "shape": manifest_payload["array"]["shape"],
                        "total_bytes": manifest_payload["array"]["total_bytes"],
                        "availability": "existing",
                        "producer_job_key": None,
                    },
                }
            ],
        },
    )
    source_plan = _make_plan(
        tmp_path / "run" / "plan.json",
        [
            {
                "target_id": "recording_a",
                "analysis_zarr": str(analysis_zarr),
                "crop_run": "crop_001",
                "roi_cache_manifest": str(manifest),
            }
        ],
    )
    source_payload = json.loads(source_plan.read_text(encoding="utf-8"))
    source_payload["keypoint_plan_path"] = str(keypoint_plan)
    _write_json(source_plan, source_payload)
    repaired_plan = tmp_path / "run" / "cleanup" / "plan.repaired.json"

    repaired = mod.repair_cleanup_plan_from_keypoints(
        source_plan,
        output_path=repaired_plan,
    )

    assert "roi_cache_payload" not in json.loads(
        source_plan.read_text(encoding="utf-8")
    )["targets"][0]
    assert repaired["targets"][0]["roi_cache_payload"] == str(payload)
    assert repaired["targets"][0]["roi_cache_manifest_sha256"] == digest
    assert repaired["cleanup_plan_repair"]["source_analysis_plan"] == str(
        source_plan.resolve()
    )
    artifacts = mod.load_cleanup_artifacts(
        repaired_plan,
        allowed_root=allowed_root,
    )
    assert len(artifacts) == 1
