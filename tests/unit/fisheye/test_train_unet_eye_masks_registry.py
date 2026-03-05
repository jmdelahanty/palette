"""Registry lifecycle logging tests for the eye-mask U-Net trainer."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.segmentation import train_unet_eye_masks as mod


class _TinyChunkedDataset:
    def __init__(self, *, label_mode: str, rows: int = 2) -> None:
        channels = 1 if label_mode == "union" else 2
        base_img = np.zeros((1, 32, 32), dtype=np.float32)
        base_mask = np.zeros((channels, 32, 32), dtype=np.float32)
        base_mask[0, 8:14, 8:14] = 1.0
        if channels > 1:
            base_mask[1, 18:24, 18:24] = 1.0
        self.samples = [{"img": base_img.copy(), "masks": base_mask.copy()} for _ in range(rows)]
        self.groups = [[idx] for idx in range(rows)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _write_minimal_config(path: Path, zarr_path: Path, *, label_mode: str = "lr") -> None:
    path.write_text(
        "\n".join(
            [
                "datasets:",
                "  ds1:",
                f"    zarr_path: {zarr_path}",
                "    crop_run: merged_eye_masks",
                "    mask_run: merged_eye_masks",
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
                f"  label_mode: {label_mode}",
                "  label_source: yolo",
                "random_seed: 0",
                "num_workers: 0",
                "nc: 1",
                "names: [eye]",
            ]
        ),
        encoding="utf-8",
    )


def test_train_unet_logs_registry_in_progress_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zarr_dir = tmp_path / "source.zarr"
    zarr_dir.mkdir(parents=True)
    (zarr_dir / "zarr.json").write_text("{}", encoding="utf-8")
    cfg_path = tmp_path / "eye_mask.yaml"
    _write_minimal_config(cfg_path, zarr_dir, label_mode="lr")

    train_ds = _TinyChunkedDataset(label_mode="lr", rows=2)
    val_ds = _TinyChunkedDataset(label_mode="lr", rows=2)
    monkeypatch.setattr(
        mod,
        "_assemble_training_datasets",
        lambda config, console: (train_ds, val_ds, [{"dataset_name": "tiny", "length": 2}]),
    )
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
            "eye_mask_set_v001",
            "--run-name",
            "eye_mask_unet_registry_test",
            "--output-dir",
            str(tmp_path / "runs"),
            "--no-progress",
            "--no-compile",
        ]
    )

    assert [entry["status"] for entry in calls] == ["in_progress", "success"]
    assert calls[0]["run_id"] == "eye_mask_unet_registry_test"
    assert calls[1]["run_id"] == "eye_mask_unet_registry_test"
    assert calls[1]["final_metrics"]["stage"] == "completed"
    assert calls[1]["final_metrics"]["status_detail"] == "training_complete"
    assert Path(calls[1]["model_path"]).name == "best_model.pt"
    assert Path(calls[1]["metrics_path"]).name == "training_history.json"


def test_train_unet_logs_registry_failed_on_dataset_load_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zarr_dir = tmp_path / "source.zarr"
    zarr_dir.mkdir(parents=True)
    (zarr_dir / "zarr.json").write_text("{}", encoding="utf-8")
    cfg_path = tmp_path / "eye_mask.yaml"
    _write_minimal_config(cfg_path, zarr_dir, label_mode="lr")

    monkeypatch.setattr(
        mod,
        "_assemble_training_datasets",
        lambda config, console: (_ for _ in ()).throw(RuntimeError("dataset exploded")),
    )
    monkeypatch.setattr(mod, "build_invocation_record", lambda **_: {"tool": "test"})

    calls: list[dict[str, object]] = []

    def _fake_record(**kwargs):
        calls.append(
            {
                "run_id": kwargs["run_id"],
                "status": kwargs["status"],
                "final_metrics": kwargs["final_metrics"],
            }
        )

    monkeypatch.setattr(mod, "_record_registry_training_run", _fake_record)

    with pytest.raises(RuntimeError, match="dataset exploded"):
        mod.main(
            [
                str(cfg_path),
                "--registry",
                str(tmp_path / "registry.sqlite"),
                "--set-id",
                "eye_mask_set_v001",
                "--run-name",
                "eye_mask_unet_registry_fail",
                "--no-progress",
                "--no-compile",
            ]
        )

    assert len(calls) == 1
    assert calls[0]["status"] == "failed"
    assert calls[0]["run_id"] == "eye_mask_unet_registry_fail"
    assert calls[0]["final_metrics"]["stage"] == "dataset_load"
    assert calls[0]["final_metrics"]["error_type"] == "RuntimeError"


def test_resolve_output_base_dir_prefers_cli_output_dir(tmp_path: Path) -> None:
    console = Console()
    explicit = tmp_path / "explicit_out"
    resolved = mod._resolve_output_base_dir(
        output_dir=str(explicit),
        configured_project="runs/eye_masks",
        set_id="eye_mask_set_v001",
        config_path=tmp_path / "cfg.yaml",
        console=console,
        nvme_root=tmp_path / "nvme",
    )
    assert resolved == explicit.resolve()


def test_resolve_output_base_dir_preserves_absolute_configured_project(tmp_path: Path) -> None:
    console = Console()
    configured = tmp_path / "custom_models"
    resolved = mod._resolve_output_base_dir(
        output_dir=None,
        configured_project=str(configured),
        set_id="eye_mask_set_v001",
        config_path=tmp_path / "cfg.yaml",
        console=console,
        nvme_root=tmp_path / "nvme",
    )
    assert resolved == configured.resolve()


def test_resolve_output_base_dir_promotes_legacy_runs_project_to_nvme_models(tmp_path: Path) -> None:
    console = Console()
    nvme_root = tmp_path / "nvme1"
    nvme_root.mkdir(parents=True)
    resolved = mod._resolve_output_base_dir(
        output_dir=None,
        configured_project="runs/eye_masks",
        set_id="eye_mask_cedar_shadow_v001",
        config_path=tmp_path / "cfg.yaml",
        console=console,
        nvme_root=nvme_root,
    )
    assert resolved == (nvme_root / "models" / "eye_masks" / "eye_mask_cedar_shadow_v001").resolve()
