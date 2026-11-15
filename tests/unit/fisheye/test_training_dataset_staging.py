from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.training.training_dataset_staging import stage_training_datasets


def _write_merged_training_export(
    path: Path, *, explicit_immutable: bool = True
) -> None:
    path.mkdir(parents=True)
    attrs: dict[str, object] = {
        "zarr_purpose": "training",
        "training_export": {
            "task": "pose",
            "set_id": path.stem,
            "logical_dataset_hash": {
                "schema_id": "palette.logical_materialized_pose_training_dataset",
                "schema_version": 2,
                "algorithm": "sha256",
                "digest": "a" * 64,
            },
        },
    }
    if explicit_immutable:
        attrs.update(
            {
                "training_artifact_status": "complete",
                "training_artifact_mutability": "immutable",
                "immutable_training_publication": {
                    "schema_id": (
                        "palette.immutable_merged_keypoint_training_publication"
                    ),
                    "schema_version": 2,
                },
            }
        )
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
    for group_name in ("crop_runs", "keypoints_runs", "splits"):
        group = path / group_name
        group.mkdir()
        (group / "zarr.json").write_text(
            json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}),
            encoding="utf-8",
        )
    payload = path / "crop_runs" / "pixels.bin"
    payload.write_bytes(bytes(range(128)))


def test_required_staging_copies_and_verifies_complete_tree(tmp_path: Path) -> None:
    source = tmp_path / "source_merged.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _write_merged_training_export(source)

    staged = stage_training_datasets(
        {"pose": source},
        mode="required",
        scratch_root=scratch,
        max_total_bytes=1024 * 1024,
        min_free_bytes_after_stage=0,
    )
    try:
        effective = staged.effective_paths["pose"]
        assert effective != source.resolve()
        assert (effective / "crop_runs" / "pixels.bin").read_bytes() == bytes(
            range(128)
        )
        record = staged.receipt["datasets"]["pose"]
        assert record["action"] == "staged"
        assert record["eligibility"] == "explicit_immutable_merged_training_export"
        assert record["verification"]["status"] == "exact_physical_tree_match"
        assert len(record["verification"]["content_sha256"]) == 64
        assert staged.stage_root is not None and staged.stage_root.exists()
    finally:
        stage_root = staged.stage_root
        staged.cleanup()
    assert stage_root is not None and not stage_root.exists()


def test_auto_stages_legacy_merged_export_on_shared_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "legacy_merged.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _write_merged_training_export(source, explicit_immutable=False)
    monkeypatch.setattr(
        "fisheye.training.training_dataset_staging._is_shared_storage",
        lambda candidate: Path(candidate).resolve() == source.resolve(),
    )

    staged = stage_training_datasets(
        {"legacy": source},
        mode="auto",
        scratch_root=scratch,
        max_total_bytes=1024 * 1024,
        min_free_bytes_after_stage=0,
    )
    try:
        record = staged.receipt["datasets"]["legacy"]
        assert record["action"] == "staged"
        assert record["eligibility"] == "legacy_closed_merged_training_export"
    finally:
        staged.cleanup()


def test_auto_leaves_live_recording_zarr_on_shared_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "recording_training.zarr"
    source.mkdir()
    (source / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"zarr_purpose": "training"},
            }
        ),
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(
        "fisheye.training.training_dataset_staging._is_shared_storage",
        lambda _path: True,
    )

    staged = stage_training_datasets(
        {"recording": source},
        mode="auto",
        scratch_root=scratch,
        max_total_bytes=1024 * 1024,
        min_free_bytes_after_stage=0,
    )

    assert staged.effective_paths["recording"] == source.resolve()
    assert staged.receipt["datasets"]["recording"]["action"] == "not_eligible"
    assert staged.stage_root is None


def test_auto_size_gate_is_visible_and_required_mode_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source_merged.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _write_merged_training_export(source)
    monkeypatch.setattr(
        "fisheye.training.training_dataset_staging._is_shared_storage",
        lambda candidate: Path(candidate).resolve() == source.resolve(),
    )

    staged = stage_training_datasets(
        {"pose": source},
        mode="auto",
        scratch_root=scratch,
        max_total_bytes=1,
        min_free_bytes_after_stage=0,
    )
    assert staged.effective_paths["pose"] == source.resolve()
    assert staged.receipt["datasets"]["pose"]["action"] == "size_gate_skipped"

    with pytest.raises(ValueError, match="max_total_bytes"):
        stage_training_datasets(
            {"pose": source},
            mode="required",
            scratch_root=scratch,
            max_total_bytes=1,
            min_free_bytes_after_stage=0,
        )
