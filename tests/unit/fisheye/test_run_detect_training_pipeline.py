"""Tests for detect pipeline orchestration wrapper."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import run_detect_training_pipeline as pipeline


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    db.upsert_dataset(
        "dataset_1",
        session_uuid="dataset_1",
        zarr_path=zarr_path,
    )
    db.upsert_provenance(
        "dataset_1",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={
            "has_images_ds": True,
            "has_images_ds_rgb": False,
            "downsample_formats_json": '["gray"]',
        },
        zarr_purpose=None,
    )
    db.close()


def test_export_merged_invokes_exporter(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "out.manifest.json"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        manifest_path.write_text("{}", encoding="utf-8")

    def fake_export(cli: list[str]) -> int:
        calls["export"] = list(cli)
        return 0

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)
    monkeypatch.setattr(pipeline.export_zarr, "main", fake_export)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-manifest",
            str(manifest_path),
            "--export-merged",
            "--merge-out-zarr",
            str(tmp_path / "merged.zarr"),
            "--merge-split",
            "0.7/0.3",
            "--merge-seed",
            "123",
            "--merge-overwrite",
        ]
    )
    assert rc == 0

    prepare_cli = calls["prepare"]
    export_cli = calls["export"]
    assert str(tmp_path / "dataset_1.zarr") in prepare_cli
    assert "--out-manifest" in prepare_cli
    assert "--merge" in export_cli
    assert "--manifest" in export_cli
    assert str(manifest_path) in export_cli
    assert "--registry" in export_cli
    assert str(registry_path) in export_cli
    assert "--out-zarr" in export_cli
    assert str(tmp_path / "merged.zarr") in export_cli
    assert "--split" in export_cli
    assert "0.7/0.3" in export_cli
    assert "--seed" in export_cli
    assert "123" in export_cli
    assert "--overwrite" in export_cli


def test_export_merged_auto_sets_out_manifest_when_missing(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: dict[str, list[str]] = {}
    monkeypatch.chdir(tmp_path)
    dataset_root = tmp_path / "datasets_root"
    monkeypatch.setenv("PALETTE_TRAINING_DATASETS_ROOT", str(dataset_root))

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        assert "--out-manifest" in cli
        manifest_path = Path(cli[cli.index("--out-manifest") + 1])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")

    def fake_export(cli: list[str]) -> int:
        calls["export"] = list(cli)
        return 0

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)
    monkeypatch.setattr(pipeline.export_zarr, "main", fake_export)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--set-name",
            "detect_smoke",
            "--set-version",
            "1",
            "--export-merged",
        ]
    )
    assert rc == 0
    prepare_cli = calls["prepare"]
    export_cli = calls["export"]
    manifest_path = Path(prepare_cli[prepare_cli.index("--out-manifest") + 1])
    expected_manifest_path = dataset_root / "detect_detect_smoke_v001" / "detect_detect_smoke_v001.manifest.json"
    assert manifest_path.resolve() == expected_manifest_path.resolve()
    assert "--manifest" in export_cli
    assert str(manifest_path) in export_cli


def test_train_runs_detection_after_preflight(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_config = tmp_path / "prep" / "detect.yaml"
    out_manifest = tmp_path / "prep" / "detect.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text("task: detect\n", encoding="utf-8")
        out_manifest.write_text('{"set_id":"detect_smoke_v001"}', encoding="utf-8")

    def fake_run(cmd: list[str], check: bool = False):
        calls["train"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
            "--train",
        ]
    )
    assert rc == 0
    train_cmd = calls["train"]
    assert str(out_config) in train_cmd
    assert "--manifest" in train_cmd
    assert str(out_manifest) in train_cmd
    assert "--set-id" in train_cmd
    assert "detect_smoke_v001" in train_cmd
    assert "--registry" in train_cmd
    assert str(registry_path) in train_cmd


def test_train_after_merged_export_uses_merged_outputs(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    preflight_manifest = tmp_path / "prep" / "detect_v001.manifest.json"
    merged_out_dir = tmp_path / "datasets" / "detect_set_v001"
    merged_config = merged_out_dir / "detect_set_v001.yaml"
    merged_manifest = merged_out_dir / "detect_set_v001.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        preflight_manifest.parent.mkdir(parents=True, exist_ok=True)
        preflight_manifest.write_text(json.dumps({"set_id": "detect_set_v001"}), encoding="utf-8")

    def fake_export(cli: list[str]) -> int:
        calls["export"] = list(cli)
        merged_out_dir.mkdir(parents=True, exist_ok=True)
        merged_config.write_text("task: detect\n", encoding="utf-8")
        merged_manifest.write_text('{"set_id":"detect_set_v001"}', encoding="utf-8")
        return 0

    def fake_run(cmd: list[str], check: bool = False):
        calls["train"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)
    monkeypatch.setattr(pipeline.export_zarr, "main", fake_export)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--set-name",
            "detect_set",
            "--set-version",
            "1",
            "--out-manifest",
            str(preflight_manifest),
            "--export-merged",
            "--merge-out-dir",
            str(merged_out_dir),
            "--train",
        ]
    )
    assert rc == 0
    train_cmd = calls["train"]
    assert str(merged_config) in train_cmd
    assert "--manifest" in train_cmd
    assert str(merged_manifest) in train_cmd
    assert "--set-id" in train_cmd
    assert "detect_set_v001" in train_cmd
    assert "--registry" in train_cmd
    assert str(registry_path) in train_cmd


def test_train_forwards_export_flags_to_train_detection(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_config = tmp_path / "prep" / "detect.yaml"
    out_manifest = tmp_path / "prep" / "detect.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text("task: detect\n", encoding="utf-8")
        out_manifest.write_text('{"set_id":"detect_smoke_v001"}', encoding="utf-8")

    def fake_run(cmd: list[str], check: bool = False):
        calls["train"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
            "--train",
            "--export-onnx",
            "--export-trt",
            "--onnx-opset",
            "13",
            "--onnx-simplify",
            "--onnx-path",
            str(tmp_path / "existing.onnx"),
            "--nms-conf",
            "0.75",
            "--nms-iou",
            "0.6",
            "--nms-topk",
            "2",
            "--trt-precision",
            "int8",
            "--trtexec",
            "/usr/bin/trtexec",
            "--trt-cuda-graph",
            "--trt-profiling",
            "--trt-verbose",
            "--profile",
        ]
    )
    assert rc == 0
    train_cmd = calls["train"]
    assert "--export-onnx" in train_cmd
    assert "--export-trt" in train_cmd
    assert "--onnx-opset" in train_cmd and "13" in train_cmd
    assert "--onnx-simplify" in train_cmd
    assert "--onnx-path" in train_cmd and str(tmp_path / "existing.onnx") in train_cmd
    assert "--nms-conf" in train_cmd and "0.75" in train_cmd
    assert "--nms-iou" in train_cmd and "0.6" in train_cmd
    assert "--nms-topk" in train_cmd and "2" in train_cmd
    assert "--trt-precision" in train_cmd and "int8" in train_cmd
    assert "--trtexec" in train_cmd and "/usr/bin/trtexec" in train_cmd
    assert "--trt-cuda-graph" in train_cmd
    assert "--trt-profiling" in train_cmd
    assert "--trt-verbose" in train_cmd
    assert "--profile" in train_cmd


def test_train_defaults_trt_precision_to_fp16_when_exporting_trt(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_config = tmp_path / "prep" / "detect.yaml"
    out_manifest = tmp_path / "prep" / "detect.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text("task: detect\n", encoding="utf-8")
        out_manifest.write_text('{"set_id":"detect_smoke_v001"}', encoding="utf-8")

    def fake_run(cmd: list[str], check: bool = False):
        calls["train"] = list(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-config",
            str(out_config),
            "--out-manifest",
            str(out_manifest),
            "--train",
            "--export-trt",
        ]
    )
    assert rc == 0
    train_cmd = calls["train"]
    assert "--export-trt" in train_cmd
    assert "--trt-precision" in train_cmd
    assert "fp16" in train_cmd


def test_train_cannot_be_combined_with_dry_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    db = Registry(registry_path)
    db.close()

    try:
        pipeline.main(["--registry", str(registry_path), "--train", "--dry-run"])
    except SystemExit as exc:
        assert "--train cannot be combined with --dry-run" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit when --train and --dry-run are combined.")


def test_train_requires_manifest_set_id(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_config = tmp_path / "prep" / "detect.yaml"
    out_manifest = tmp_path / "prep" / "detect.manifest.json"

    def fake_prepare(cli: list[str]) -> None:
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text("task: detect\n", encoding="utf-8")
        out_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)

    try:
        pipeline.main(
            [
                "--registry",
                str(registry_path),
                "--out-config",
                str(out_config),
                "--out-manifest",
                str(out_manifest),
                "--train",
            ]
        )
    except SystemExit as exc:
        assert "requires manifest set_id" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit when --train manifest lacks set_id.")
