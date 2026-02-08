"""Path/config/registry helper tests for keypoint training."""

import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.training.config import PoseConfig
from fisheye.training.train_keypoints import (
    _build_default_run_name,
    _infer_set_slug,
    _load_manifest_set_id,
    _record_registry_training_run,
    _snapshot_training_inputs,
    _strip_manifest_suffixes,
)


def test_strip_manifest_suffixes_handles_repeated_suffix() -> None:
    assert _strip_manifest_suffixes("pose_cedar_v001.manifest") == "pose_cedar_v001"
    assert _strip_manifest_suffixes("pose_cedar_v001.manifest.manifest") == "pose_cedar_v001"


def test_infer_set_slug_prefers_set_id_over_config_stem() -> None:
    cfg = Path("/tmp/pose_cedar_v001.manifest.yaml")
    assert _infer_set_slug("pose_cedar_v002", cfg) == "pose_cedar_v002"
    assert _infer_set_slug(None, cfg) == "pose_cedar_v001"


def test_load_manifest_set_id_reads_manifest_json(tmp_path: Path) -> None:
    manifest = tmp_path / "pose.manifest.json"
    manifest.write_text(json.dumps({"set_id": "pose_cedar_v003"}), encoding="utf-8")
    assert _load_manifest_set_id(str(manifest)) == "pose_cedar_v003"
    assert _load_manifest_set_id(None) is None


def test_pose_config_accepts_keypoint_run(tmp_path: Path) -> None:
    zarr_dir = tmp_path / "sample.zarr"
    zarr_dir.mkdir(parents=True)
    (zarr_dir / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}', encoding="utf-8")

    cfg_path = tmp_path / "pose.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "train: ./dummy_train.txt",
                "val: ./dummy_val.txt",
                "nc: 1",
                "names: [fish]",
                "task: pose",
                "kpt_shape: [3, 3]",
                "datasets:",
                "  arena1:",
                f"    zarr_path: {zarr_dir}",
                "    source_type: filtered",
                "    input_format: gray",
                "    keypoint_run: latest_traditional",
                "training_params:",
                "  model: yolov8n-pose.pt",
                "  epochs: 1",
                "  batch: 1",
                "  imgsz: 256",
                "  lr0: 0.001",
                "  momentum: 0.9",
                "  weight_decay: 0.0005",
                "  patience: 1",
                "  device: '0'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = PoseConfig.from_yaml(cfg_path)
    assert cfg.datasets["arena1"].keypoint_run == "latest_traditional"


def test_record_registry_training_run_upserts_status(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    config_path = tmp_path / "pose.yaml"
    manifest_path = tmp_path / "pose.manifest.json"
    model_path = tmp_path / "best.pt"
    metrics_path = tmp_path / "results.csv"
    config_path.write_text("task: pose\n", encoding="utf-8")
    manifest_path.write_text('{"set_id":"pose_cedar_v001"}\n', encoding="utf-8")
    model_path.write_bytes(b"model")
    metrics_path.write_text("epoch,metrics/mAP50(P)\n0,0.5\n", encoding="utf-8")

    args = SimpleNamespace(registry=registry_path)
    console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
    invocation = {"tool": "test", "argv": []}

    _record_registry_training_run(
        args=args,
        console=console,
        invocation_payload=invocation,
        run_id="pose_run_001",
        set_id="pose_cedar_v001",
        config_path=config_path,
        manifest_path=manifest_path,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "preflight_and_training", "status_detail": "training_started"},
    )
    _record_registry_training_run(
        args=args,
        console=console,
        invocation_payload=invocation,
        run_id="pose_run_001",
        set_id="pose_cedar_v001",
        config_path=config_path,
        manifest_path=manifest_path,
        model_path=model_path,
        metrics_path=metrics_path,
        status="success",
        final_metrics={"stage": "completed", "status_detail": "training_complete"},
    )

    registry = Registry(registry_path)
    row = registry.conn.execute(
        "SELECT status, final_metrics_json, model_path, metrics_path FROM training_runs WHERE run_id = ?",
        ("pose_run_001",),
    ).fetchone()
    registry.close()

    assert row is not None
    assert row["status"] == "success"
    payload = json.loads(row["final_metrics_json"])
    assert payload["status_detail"] == "training_complete"
    assert row["model_path"] == str(model_path)
    assert row["metrics_path"] == str(metrics_path)


def test_snapshot_training_inputs_copies_config_manifest_and_invocation(tmp_path: Path) -> None:
    config = tmp_path / "pose.yaml"
    manifest = tmp_path / "pose.manifest.json"
    run_dir = tmp_path / "pose_run"
    config.write_text("task: pose\n", encoding="utf-8")
    manifest.write_text('{"set_id":"pose_cedar_v001"}\n', encoding="utf-8")

    written = _snapshot_training_inputs(
        run_dir=run_dir,
        config_path=config,
        manifest_path=manifest,
        invocation_payload={"tool": "fisheye.training.train_keypoints", "argv": ["pose.yaml"]},
    )

    assert (run_dir / "inputs" / "pose.yaml").exists()
    assert (run_dir / "inputs" / "pose.manifest.json").exists()
    invocation_path = run_dir / "inputs" / "train_invocation.json"
    assert invocation_path.exists()
    payload = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert payload["tool"] == "fisheye.training.train_keypoints"
    assert len(written) == 3


def test_build_default_run_name_uses_manifest_hints() -> None:
    run_name = _build_default_run_name(
        manifest_hints={
            "rig_name": "omnifin0",
            "dish_design": "cedar dish",
            "canvas_name": "Feeding",
            "task": "pose",
            "set_id": "pose_cedar_v001",
            "manifest_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        },
        task_fallback="pose",
        timestamp="20260206-200000",
        pid=999,
    )
    assert run_name == "omnifin0_cedar_dish_feeding_v001_pose_20260206-200000_01234567"
