"""Registry lifecycle logging tests for the subject-mask U-Net trainer."""

from __future__ import annotations

import json
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


def _channel_summary(
    *,
    labels: list[str],
    supervised: dict[str, int],
    positive: dict[str, int] | None = None,
) -> dict[str, object]:
    positive = positive or {label: 0 for label in labels}
    return {
        "label_schema_id": "subject_v1_union",
        "mask_labels": labels,
        "coverage_class": "partial_subject_masks",
        "contains_only_eye_masks": False,
        "available_labels": [label for label in labels if supervised.get(label, 0) > 0],
        "missing_labels": [label for label in labels if supervised.get(label, 0) <= 0],
        "supervised_row_counts": {label: supervised.get(label, 0) for label in labels},
        "positive_row_counts": {label: positive.get(label, 0) for label in labels},
        "negative_row_counts": {
            label: max(0, supervised.get(label, 0) - positive.get(label, 0)) for label in labels
        },
        "unsupervised_row_counts": {label: 0 for label in labels},
    }


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
        meta_list=[
            {
                "dataset_name": "tiny",
                "length": 2,
                "channel_supervision_summary": _channel_summary(
                    labels=["subject_body", "eyes_union", "swim_bladder"],
                    supervised={"subject_body": 2, "eyes_union": 2, "swim_bladder": 0},
                    positive={"subject_body": 2, "eyes_union": 2, "swim_bladder": 0},
                ),
            }
        ],
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
    assert calls[1]["final_metrics"]["best_epoch"] == 1
    assert calls[1]["final_metrics"]["epochs"] == 1
    assert calls[1]["final_metrics"]["train_samples"] == 2
    assert calls[1]["final_metrics"]["val_samples"] == 2
    assert calls[1]["final_metrics"]["label_schema_id"] == "subject_v1_union"
    assert calls[1]["final_metrics"]["mask_labels"] == [
        "subject_body",
        "eyes_union",
        "swim_bladder",
    ]
    assert calls[1]["final_metrics"]["coverage_class"] == "partial_subject_masks"
    assert calls[1]["final_metrics"]["component_groups"] == ["body", "eyes"]
    assert calls[1]["final_metrics"]["component_coverage_key"] == "body+eyes"
    assert calls[1]["final_metrics"]["available_labels"] == ["subject_body", "eyes_union"]
    assert calls[1]["final_metrics"]["missing_labels"] == ["swim_bladder"]
    assert calls[1]["final_metrics"]["supervised_row_counts"] == {
        "subject_body": 2,
        "eyes_union": 2,
        "swim_bladder": 0,
    }
    model_summary = calls[1]["final_metrics"]["subject_mask_model_summary"]
    assert model_summary["summarized_artifact_count"] == 1
    assert model_summary["component_coverage_key"] == "body+eyes"
    assert Path(calls[1]["model_path"]).name == "best_model.pt"
    assert Path(calls[1]["metrics_path"]).name == "training_history.json"
    live_history_path = (
        tmp_path
        / "runs"
        / "subject_mask_unet_registry_test"
        / "training_history_live.jsonl"
    )
    assert live_history_path.exists()
    live_rows = [
        json.loads(line)
        for line in live_history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert live_rows[0]["event"] == "epoch_metrics"
    assert live_rows[0]["epoch"] == 1
    assert "val_dice_subject_body" in live_rows[0]


def test_subject_mask_model_summary_tracks_component_combinations() -> None:
    summary = mod._build_subject_mask_model_summary(
        meta_list=[
            {
                "channel_supervision_summary": {
                    "supervised_row_counts": {
                        "subject_body": 0,
                        "eye_left": 5,
                        "eye_right": 5,
                        "swim_bladder": 5,
                    },
                    "positive_row_counts": {
                        "subject_body": 0,
                        "eye_left": 4,
                        "eye_right": 4,
                        "swim_bladder": 3,
                    },
                    "negative_row_counts": {
                        "subject_body": 0,
                        "eye_left": 1,
                        "eye_right": 1,
                        "swim_bladder": 2,
                    },
                    "unsupervised_row_counts": {
                        "subject_body": 5,
                        "eye_left": 0,
                        "eye_right": 0,
                        "swim_bladder": 0,
                    },
                }
            }
        ],
        label_schema_id="subject_v1_lr",
        mask_labels=("subject_body", "eye_left", "eye_right", "swim_bladder"),
    )

    assert summary["label_schema_id"] == "subject_v1_lr"
    assert summary["coverage_class"] == "partial_subject_masks"
    assert summary["component_groups"] == ["eyes", "swim_bladder"]
    assert summary["component_coverage_key"] == "eyes+swim_bladder"
    assert summary["available_labels"] == ["eye_left", "eye_right", "swim_bladder"]
    assert summary["missing_labels"] == ["subject_body"]


def test_validation_preview_writer_outputs_composite_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image = np.zeros((1, 24, 24), dtype=np.float32)
    image[0, 4:20, 4:20] = 0.35
    target = np.zeros((3, 24, 24), dtype=np.float32)
    target[0, 5:19, 5:19] = 1.0
    target[1, 8:12, 8:16] = 1.0
    target[2, 14:17, 10:14] = 1.0
    pred = target * 0.8

    written = mod._write_validation_previews(
        output_dir=tmp_path / "validation_previews",
        images=[image],
        targets=[target],
        pred_probs=[pred],
        valid_channels=[np.array([1.0, 1.0, 1.0], dtype=np.float32)],
        mask_labels=("subject_body", "eyes_union", "swim_bladder"),
        epoch=1,
        thresholds=[0.10, 0.25, 0.50],
    )

    assert written == [tmp_path / "validation_previews" / "epoch_001" / "sample_000.png"]
    assert written[0].exists()
    assert written[0].stat().st_size > 0
