"""Tests for detect pipeline orchestration wrapper."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import run_detect_training_pipeline as pipeline


def _seed_dataset(db: Registry, dataset_id: str, zarr_path: Path) -> None:
    db.upsert_dataset(
        dataset_id,
        session_uuid=dataset_id,
        zarr_path=zarr_path,
    )
    db.upsert_provenance(
        dataset_id,
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


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    _seed_dataset(db, "dataset_1", zarr_path)
    db.close()


def _upsert_detect_quality(
    db: Registry,
    *,
    dataset_id: str,
    review_state: str = "approved",
    review_intended_use: str = "training",
    interpolated_detections_rate: float | None = 0.1,
    zarr_mtime_ns: int | None = 101,
) -> None:
    db.upsert_detect_quality(
        dataset_id=dataset_id,
        refined_run="refined_run_1",
        refined_created_utc=None,
        source_detect_run="detect_run_1",
        detect_method="detect",
        review_state=review_state,
        review_intended_use=review_intended_use,
        review_reviewer=None,
        review_timestamp_utc=None,
        review_resolved_group="filtered",
        total_detections=10,
        real_detections=9,
        interpolated_detections=1,
        interpolated_detections_rate=interpolated_detections_rate,
        zarr_mtime_ns=zarr_mtime_ns,
    )


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


def test_aggregate_training_data_card_invokes_aggregator(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_manifest = tmp_path / "prep" / "detect.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text('{"set_id":"detect_smoke_v001","datasets":[{"dataset_id":"dataset_1","zarr_path":"x"}]}', encoding="utf-8")

    def fake_aggregate(cli: list[str]) -> int:
        calls["aggregate"] = list(cli)
        return 0

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)
    monkeypatch.setattr(pipeline.aggregate_data_card, "main", fake_aggregate)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-manifest",
            str(out_manifest),
            "--aggregate-training-data-card",
            "--data-card-output",
            str(tmp_path / "card.json"),
            "--data-card-split",
            "train",
            "--data-card-no-plots",
            "--data-card-plot-dir",
            str(tmp_path / "card_plots"),
            "--data-card-plot-prefix",
            "manual_train",
            "--data-card-plot-heatmap-bin-factor",
            "3",
        ]
    )
    assert rc == 0
    aggregate_cli = calls["aggregate"]
    assert "--manifest" in aggregate_cli and str(out_manifest) in aggregate_cli
    assert "--registry" in aggregate_cli and str(registry_path) in aggregate_cli
    assert "--output" in aggregate_cli and str(tmp_path / "card.json") in aggregate_cli
    assert "--split" in aggregate_cli and "train" in aggregate_cli
    assert "--no-plots" in aggregate_cli
    assert "--plot-dir" in aggregate_cli and str(tmp_path / "card_plots") in aggregate_cli
    assert "--plot-prefix" in aggregate_cli and "manual_train" in aggregate_cli
    assert "--plot-heatmap-bin-factor" in aggregate_cli and "3" in aggregate_cli


def test_build_dataset_invokes_export_and_aggregator(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    out_manifest = tmp_path / "prep" / "detect.manifest.json"
    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text('{"set_id":"detect_smoke_v001","datasets":[{"dataset_id":"dataset_1","zarr_path":"x"}]}', encoding="utf-8")

    def fake_aggregate(cli: list[str]) -> int:
        calls["aggregate"] = list(cli)
        return 0

    def fake_export(cli: list[str]) -> int:
        calls["export"] = list(cli)
        return 0

    monkeypatch.setattr(pipeline.pdt, "main", fake_prepare)
    monkeypatch.setattr(pipeline.aggregate_data_card, "main", fake_aggregate)
    monkeypatch.setattr(pipeline.export_zarr, "main", fake_export)

    rc = pipeline.main(
        [
            "--registry",
            str(registry_path),
            "--out-manifest",
            str(out_manifest),
            "--build-dataset",
        ]
    )
    assert rc == 0
    assert "--manifest" in calls["aggregate"] and str(out_manifest) in calls["aggregate"]
    assert "--registry" in calls["aggregate"] and str(registry_path) in calls["aggregate"]
    assert "--manifest" in calls["export"] and str(out_manifest) in calls["export"]
    assert "--registry" in calls["export"] and str(registry_path) in calls["export"]
    assert "--merge" in calls["export"]


def test_build_dataset_cannot_be_combined_with_dry_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    db = Registry(registry_path)
    db.close()
    with pytest.raises(SystemExit, match="--build-dataset cannot be combined with --dry-run"):
        pipeline.main(
            [
                "--registry",
                str(registry_path),
                "--build-dataset",
                "--dry-run",
            ]
        )


def test_aggregate_training_data_card_cannot_be_combined_with_dry_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    db = Registry(registry_path)
    db.close()
    with pytest.raises(SystemExit, match="--aggregate-training-data-card cannot be combined with --dry-run"):
        pipeline.main(
            [
                "--registry",
                str(registry_path),
                "--aggregate-training-data-card",
                "--dry-run",
            ]
        )


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


@pytest.mark.parametrize(
    ("observed_patch", "expected_error"),
    [
        ({"source_detect_run": "wrong_detect_run"}, "source_detect_run divergence"),
        ({"zarr_mtime_ns": 202}, "detect_quality row is stale"),
    ],
)
def test_detect_quality_gate_rejects_stale_or_divergent_rows(
    tmp_path: Path,
    monkeypatch,
    observed_patch: dict[str, object],
    expected_error: str,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    dataset_path = tmp_path / "dataset_1.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, "dataset_1", dataset_path)
    _upsert_detect_quality(db, dataset_id="dataset_1")
    db.close()

    observed = {
        "source_detect_run": "detect_run_1",
        "review_state": "approved",
        "review_intended_use": "training",
        "review_resolved_group": "filtered",
        "total_detections": 10,
        "real_detections": 9,
        "interpolated_detections": 1,
        "interpolated_detections_rate": 0.1,
        "zarr_mtime_ns": 101,
    }
    observed.update(observed_patch)

    def fake_resolve(zarr_path: Path, *, expected_refined_run: str) -> dict[str, object]:
        assert zarr_path == dataset_path
        assert expected_refined_run == "refined_run_1"
        return dict(observed)

    def fail_prepare(_cli: list[str]) -> None:  # pragma: no cover - defensive branch
        raise AssertionError("prepare_detect_training should not run when quality validation fails.")

    monkeypatch.setattr(pipeline.prepare_from_registry, "_resolve_detect_quality_from_zarr", fake_resolve)
    monkeypatch.setattr(pipeline.pdt, "main", fail_prepare)

    with pytest.raises(ValueError, match=expected_error):
        pipeline.main(
            [
                "--registry",
                str(registry_path),
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--max-interpolated-detections-rate",
                "0.2",
                "--dry-run",
            ]
        )


def test_detect_quality_exclusion_reasons_are_concrete(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    db = Registry(registry_path)
    dataset_ids = [
        "missing_quality",
        "review_state_mismatch",
        "review_use_mismatch",
        "missing_interp_rate",
        "interp_rate_high",
    ]
    for dataset_id in dataset_ids:
        _seed_dataset(db, dataset_id, tmp_path / f"{dataset_id}.zarr")
    _upsert_detect_quality(
        db,
        dataset_id="review_state_mismatch",
        review_state="pending",
        review_intended_use="training",
        interpolated_detections_rate=0.1,
    )
    _upsert_detect_quality(
        db,
        dataset_id="review_use_mismatch",
        review_state="approved",
        review_intended_use="full_recording",
        interpolated_detections_rate=0.1,
    )
    _upsert_detect_quality(
        db,
        dataset_id="missing_interp_rate",
        review_state="approved",
        review_intended_use="training",
        interpolated_detections_rate=None,
    )
    _upsert_detect_quality(
        db,
        dataset_id="interp_rate_high",
        review_state="approved",
        review_intended_use="training",
        interpolated_detections_rate=0.8,
    )
    db.close()

    with pytest.raises(SystemExit, match="No datasets remain after detect quality filtering."):
        pipeline.main(
            [
                "--registry",
                str(registry_path),
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--max-interpolated-detections-rate",
                "0.3",
            ]
        )

    output = capsys.readouterr().out
    assert "Detect quality SQL filter excluded 5 dataset(s):" in output
    assert "Detect quality filter summary: passed=0 excluded=5 reasons=" in output
    assert "missing_quality_row=1" in output
    assert "missing_quality_row" in output
    assert "review_state_mismatch:pending!=approved=1" in output
    assert "review_state_mismatch:pending!=approved" in output
    assert "review_use_mismatch:full_recording!=training=1" in output
    assert "review_use_mismatch:full_recording!=training" in output
    assert "missing_interpolated_detections_rate=1" in output
    assert "missing_interpolated_detections_rate" in output
    assert "interpolated_rate_above_threshold:0.800000>0.300000=1" in output
    assert "interpolated_rate_above_threshold:0.800000>0.300000" in output
