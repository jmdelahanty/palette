"""Registry lifecycle logging tests for the subject-mask U-Net trainer."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.segmentation import train_unet_subject_masks as mod
from fisheye.training.zarr_subject_mask_dataset import SubjectMaskDatasetBundle


class _TinySubjectDataset:
    def __init__(self, rows: int = 2) -> None:
        base_img = np.zeros((1, 32, 32), dtype=np.float32)
        base_mask = np.zeros((3, 32, 32), dtype=np.float32)
        base_mask[0, 8:14, 8:14] = 1.0
        base_mask[1, 12:18, 12:18] = 1.0
        valid = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        self.samples = [
            {"img": base_img.copy(), "masks": base_mask.copy(), "valid_channels": valid.copy()}
            for _ in range(rows)
        ]
        self.groups = [[idx] for idx in range(rows)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _write_minimal_config(path: Path, zarr_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "datasets:",
                "  ds1:",
                f"    zarr_path: {zarr_path}",
                "training_params:",
                "  model: unet_small",
                "  epochs: 1",
                "  batch: 2",
                "  batch_size: 2",
                "  imgsz: 32",
                "  lr0: 0.001",
                "  momentum: 0.9",
                "  weight_decay: 0.0",
                "  patience: 3",
                "  device: cpu",
                "  label_schema_id: subject_v1_union",
                "random_seed: 0",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )


def test_train_unet_subject_masks_logs_registry_in_progress_then_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_dir = tmp_path / "source.zarr"
    zarr_dir.mkdir(parents=True)
    (zarr_dir / "zarr.json").write_text("{}", encoding="utf-8")
    cfg_path = tmp_path / "subject_mask.yaml"
    _write_minimal_config(cfg_path, zarr_dir)

    bundle = SubjectMaskDatasetBundle(
        train_dataset=_TinySubjectDataset(rows=2),
        val_dataset=_TinySubjectDataset(rows=2),
        meta_list=[{"dataset_name": "tiny", "length": 2}],
        label_schema_id="subject_v1_union",
        mask_labels=("subject_body", "eyes_union", "swim_bladder"),
    )
    monkeypatch.setattr(mod, "build_subject_mask_training_datasets", lambda config, console: bundle)
    monkeypatch.setattr(mod, "build_invocation_record", lambda **_: {"tool": "test"})
    monkeypatch.setattr(mod, "get_git_info", lambda: {"commit_hash": "abc", "branch": "main"})
    monkeypatch.setattr(mod, "get_environment_info", lambda: {"platform": {"hostname": "test-host"}})

    calls: list[dict[str, object]] = []

    def _fake_record(**kwargs):
        calls.append(
            {
                "run_id": kwargs["run_id"],
                "status": kwargs["status"],
                "final_metrics": kwargs["final_metrics"],
                "model_path": kwargs["model_path"],
                "metrics_path": kwargs["metrics_path"],
            }
        )

    monkeypatch.setattr(mod, "_record_registry_training_run", _fake_record)

    mod.main(
        [
            str(cfg_path),
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--set-id",
            "subject_mask_set_v001",
            "--run-name",
            "subject_mask_unet_registry_test",
            "--output-dir",
            str(tmp_path / "runs"),
            "--no-progress",
            "--no-compile",
        ]
    )

    assert [entry["status"] for entry in calls] == ["in_progress", "success"]
    assert calls[0]["run_id"] == "subject_mask_unet_registry_test"
    assert calls[1]["run_id"] == "subject_mask_unet_registry_test"
    assert calls[1]["final_metrics"]["stage"] == "completed"
    assert calls[1]["final_metrics"]["status_detail"] == "training_complete"
    assert calls[1]["final_metrics"]["label_schema_id"] == "subject_v1_union"
    assert Path(calls[1]["model_path"]).name == "best_model.pt"
    assert Path(calls[1]["metrics_path"]).name == "training_history.json"
