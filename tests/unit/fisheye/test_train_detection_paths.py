"""Path helper tests for detection training output defaults."""

from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.train_detection import (
    _build_default_run_name,
    _infer_set_slug,
    _snapshot_training_inputs,
    _strip_manifest_suffixes,
)


def test_strip_manifest_suffixes_handles_repeated_suffix() -> None:
    assert _strip_manifest_suffixes("detect_cedar_v001.manifest") == "detect_cedar_v001"
    assert _strip_manifest_suffixes("detect_cedar_v001.manifest.manifest") == "detect_cedar_v001"


def test_infer_set_slug_prefers_set_id_over_config_stem() -> None:
    cfg = Path("/tmp/detect_cedar_v001.manifest.yaml")
    assert _infer_set_slug("detect_cedar_v002", cfg) == "detect_cedar_v002"
    assert _infer_set_slug(None, cfg) == "detect_cedar_v001"


def test_snapshot_training_inputs_copies_config_manifest_and_invocation(tmp_path: Path) -> None:
    config = tmp_path / "detect.yaml"
    manifest = tmp_path / "detect.manifest.json"
    run_dir = tmp_path / "run"
    config.write_text("task: detect\n", encoding="utf-8")
    manifest.write_text('{"set_id":"detect_cedar_v001"}\n', encoding="utf-8")

    written = _snapshot_training_inputs(
        run_dir=run_dir,
        config_path=config,
        manifest_path=manifest,
        invocation_payload={"tool": "fisheye.training.train_detection", "argv": ["detect.yaml"]},
    )

    assert (run_dir / "inputs" / "detect.yaml").exists()
    assert (run_dir / "inputs" / "detect.manifest.json").exists()
    invocation_path = run_dir / "inputs" / "train_invocation.json"
    assert invocation_path.exists()
    payload = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert payload["tool"] == "fisheye.training.train_detection"
    assert len(written) == 3


def test_build_default_run_name_uses_manifest_hints() -> None:
    run_name = _build_default_run_name(
        manifest_summary={
            "manifest_dish_design": "cedar dish",
            "manifest_canvas_name": "DefaultScreen",
            "manifest_task": "detect",
        },
        task_fallback="detect",
        timestamp="20260206-200000",
        pid=1234,
    )
    assert run_name == "cedar_dish_defaultscreen_detect_20260206-200000_1234"
