"""Path/helper tests for export_onnx argument resolution."""

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.export_onnx import (
    _build_metadata_props,
    _resolve_input_shape,
    _resolve_weights_path,
)


def test_resolve_weights_path_prefers_explicit_weights(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    explicit = tmp_path / "manual.pt"
    explicit.write_bytes(b"pt")
    ((run_dir / "weights") / "best.pt").write_bytes(b"best")

    resolved = _resolve_weights_path(str(explicit), run_dir)
    assert resolved == explicit


def test_resolve_weights_path_uses_best_then_last(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)

    last = weights_dir / "last.pt"
    last.write_bytes(b"last")
    assert _resolve_weights_path(None, run_dir) == last

    best = weights_dir / "best.pt"
    best.write_bytes(b"best")
    assert _resolve_weights_path(None, run_dir) == best


def test_resolve_input_shape_prefers_explicit_input_shape(tmp_path: Path) -> None:
    shape, source = _resolve_input_shape([1, 3, 512, 768], None, tmp_path)
    assert shape == [1, 3, 512, 768]
    assert source == "cli:input_shape"


def test_resolve_input_shape_imgsz_override_wins(tmp_path: Path) -> None:
    shape, source = _resolve_input_shape(None, [1024], tmp_path)
    assert shape == [1, 3, 1024, 1024]
    assert source == "cli:imgsz"


def test_resolve_input_shape_uses_training_report_effective_imgsz(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "20260207-010101_detection_training_report.yaml"
    report = {
        "training_history": {
            "effective_imgsz": [640, 640],
            "effective_training_params": {"imgsz": 1024},
        },
        "training_params": {"imgsz": 320},
    }
    report_path.write_text(yaml.safe_dump(report), encoding="utf-8")

    shape, source = _resolve_input_shape(None, None, run_dir)
    assert shape == [1, 3, 640, 640]
    assert source == "training_report:effective_imgsz"


def test_resolve_input_shape_falls_back_to_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shape, source = _resolve_input_shape(None, None, run_dir)
    assert shape == [1, 3, 640, 640]
    assert source == "default:640"


def test_resolve_input_shape_reads_legacy_tagged_report_via_base_loader(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "20260207-010101_detection_training_report.yaml"
    report_path.write_text(
        "\n".join(
            [
                "train: !!python/object/apply:pathlib.PosixPath",
                "- dummy_train.txt",
                "training_history:",
                "  effective_imgsz:",
                "  - 640",
                "  - 768",
            ]
        ),
        encoding="utf-8",
    )

    shape, source = _resolve_input_shape(None, None, run_dir)
    assert shape == [1, 3, 640, 768]
    assert source == "training_report:effective_imgsz"


def test_build_metadata_props_includes_core_fields() -> None:
    metadata = _build_metadata_props(
        run_id="run_123",
        set_id="detect_set_v001",
        manifest_sha256="ABC123",
        system_hostname="host1",
        torch_version="2.6.0+cu124",
        cuda_version="12.4",
        exported_at_utc="2026-02-07T21:00:00Z",
    )
    assert metadata["run_id"] == "run_123"
    assert metadata["set_id"] == "detect_set_v001"
    assert metadata["manifest_sha256"] == "abc123"
    assert metadata["system_hostname"] == "host1"
    assert metadata["torch_version"] == "2.6.0+cu124"
    assert metadata["cuda_version"] == "12.4"
    assert metadata["exported_at_utc"] == "2026-02-07T21:00:00Z"


def test_build_metadata_props_omits_empty_optional_fields() -> None:
    metadata = _build_metadata_props(
        run_id=None,
        set_id="",
        manifest_sha256=None,
        exported_at_utc="2026-02-07T21:00:00Z",
    )
    assert metadata == {"exported_at_utc": "2026-02-07T21:00:00Z"}
