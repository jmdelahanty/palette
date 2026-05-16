from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.utils import batch_migrate_training_crop_pixel_contract as mod
from fisheye.utils.batch_migrate_training_crop_pixel_contract import (
    MigrationCandidate,
    batch_migrate_training_crop_pixel_contract,
    discover_candidates,
)


def _make_archive(tmp_path: Path, name: str = "training.zarr") -> Path:
    zarr_path = tmp_path / name
    root = zarr.open_group(str(zarr_path), mode="w")
    crops = root.create_group("crop_runs")
    crops.attrs["latest"] = "crop_old"
    crops.attrs["latest_materialized"] = "crop_old"
    crop = crops.create_group("crop_old")
    crop.attrs["roi_size"] = [2, 2]
    return zarr_path


def test_batch_migrator_dry_run_defers_label_plan_until_target_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_regenerate(**kwargs):
        calls.append(kwargs)
        return {
            "status": "dry_run",
            "source_crop_run": kwargs["source_crop_run"],
            "target_crop_run": kwargs["target_crop_run"],
        }

    monkeypatch.setattr(mod, "regenerate_training_crops_pynvvc", fake_regenerate)
    jsonl_report = tmp_path / "report.jsonl"

    records, summary = batch_migrate_training_crop_pixel_contract(
        candidates=[MigrationCandidate(zarr_path=zarr_path)],
        apply=False,
        jsonl_report=jsonl_report,
    )

    assert summary["status"] == "ok"
    assert records[0]["status"] == "planned"
    assert records[0]["source_crop_run"] == "crop_old"
    assert records[0]["target_crop_run"] == "crop_old_pynvvc_luma_v1"
    assert records[0]["label_report"]["status"] == "deferred_until_target_crop_exists"
    assert calls[0]["dry_run"] is True
    assert json.loads(jsonl_report.read_text(encoding="utf-8").splitlines()[0])["status"] == "planned"


def test_batch_migrator_apply_calls_crop_then_label_then_parity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path)
    calls: list[str] = []

    def fake_regenerate(**kwargs):
        calls.append("crop")
        return {
            "status": "ok",
            "source_crop_run": kwargs["source_crop_run"],
            "target_crop_run": kwargs["target_crop_run"],
        }

    def fake_migrate_labels(**kwargs):
        calls.append("labels")
        return {
            "status": "ok",
            "source_crop_run": kwargs["source_crop_run"],
            "target_crop_run": kwargs["target_crop_run"],
            "migrations": [{"family": "keypoints_runs"}],
            "skipped": [],
        }

    def fake_parity(**kwargs):
        calls.append("parity")
        return {"status": "ok", "crop_run": kwargs["crop_run"]}

    monkeypatch.setattr(mod, "regenerate_training_crops_pynvvc", fake_regenerate)
    monkeypatch.setattr(mod, "migrate_training_label_runs_identity", fake_migrate_labels)
    monkeypatch.setattr(mod, "check_training_crop_pynvvc_pixel_parity", fake_parity)

    records, summary = batch_migrate_training_crop_pixel_contract(
        candidates=[MigrationCandidate(zarr_path=zarr_path)],
        apply=True,
        parity_sample_count=3,
    )

    assert summary["status"] == "ok"
    assert records[0]["status"] == "ok"
    assert calls == ["crop", "labels", "parity"]


def test_discover_candidates_from_explicit_paths_does_not_need_registry(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)

    candidates, discovery = discover_candidates(
        zarr_paths=[zarr_path],
        registry_path=None,
        zarr_use="training",
        dataset_status=None,
        path_contains=None,
        limit=None,
        include_training_artifacts=False,
        approval_family="keypoints",
        required_review_state="approved",
        required_review_intended_use="training",
    )

    assert [candidate.zarr_path for candidate in candidates] == [zarr_path]
    assert discovery["selected_unique"] == 1
    assert discovery["registry"] is None
