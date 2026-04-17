from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.refinement import refine_eye_masks as mod


def test_get_zarr_array_opens_without_consolidated_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    mod._ZARR_GROUP_CACHE.clear()
    mod._ZARR_ARRAY_CACHE.clear()

    seen: dict[str, Any] = {}
    fake_group = {"demo/path": np.array([1, 2, 3], dtype=np.int32)}

    def _fake_open(path: str, mode: str = "r", **kwargs: Any) -> dict[str, Any]:
        seen["path"] = path
        seen["mode"] = mode
        seen["kwargs"] = kwargs
        return fake_group

    monkeypatch.setattr(mod.zarr, "open", _fake_open)
    arr = mod._get_zarr_array("demo.zarr", "demo/path")
    assert np.array_equal(arr, np.array([1, 2, 3], dtype=np.int32))
    assert seen["mode"] == "r"
    assert seen["kwargs"].get("use_consolidated") is False


def test_refine_eye_masks_root_open_uses_live_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _FakeRoot(dict):
        def get(self, key: str, default: Any = None) -> Any:
            return super().get(key, default)

    def _fake_open(path: str, mode: str = "a", **kwargs: Any) -> _FakeRoot:
        seen["path"] = path
        seen["mode"] = mode
        seen["kwargs"] = kwargs
        return _FakeRoot()

    monkeypatch.setattr(mod.zarr, "open", _fake_open)

    with pytest.raises(ValueError, match="missing eye_masks_runs"):
        mod.refine_eye_masks("demo.zarr")

    assert seen["mode"] == "a"
    assert seen["kwargs"].get("use_consolidated") is False


def test_process_and_write_chunk_open_uses_live_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roi_output = mod.ROIOutput(
        masks=np.ones((2, 3, 3), dtype=np.uint8),
        ellipse_params=np.zeros((2, 5), dtype=np.float32),
        ellipse_success=np.array([True, True], dtype=bool),
        centroids=np.zeros((2, 2), dtype=np.float32),
        contours=(None, None),
        eye_separation=5.0,
        reason="refined",
        smoothing_changed=np.array([False, False], dtype=bool),
        reassigned_pixels=0,
    )

    run_group = {
        "masks_roi": np.zeros((1, 2, 3, 3), dtype=np.uint8),
        "ellipse_params": np.zeros((1, 2, 5), dtype=np.float32),
        "ellipse_success": np.zeros((1, 2), dtype=bool),
        "eye_separation": np.full((1,), np.nan, dtype=np.float32),
    }
    root = {"refined_eye_masks_runs/refined_001": run_group}

    seen: dict[str, Any] = {}

    def _fake_open(path: str, mode: str = "a", **kwargs: Any) -> dict[str, Any]:
        seen["path"] = path
        seen["mode"] = mode
        seen["kwargs"] = kwargs
        return root

    monkeypatch.setattr(mod.zarr, "open", _fake_open)
    monkeypatch.setattr(
        mod,
        "_process_refine_chunk",
        lambda *args, **kwargs: [(0, roi_output)],
    )
    monkeypatch.setattr(
        mod,
        "_write_chunk_metrics",
        lambda *args, **kwargs: None,
    )

    result = mod._process_and_write_chunk(
        zarr_path="demo.zarr",
        source_masks_path="unused/masks",
        source_probs_path=None,
        run_group_path="refined_eye_masks_runs/refined_001",
        keypoints_path="unused/keypoints",
        heading_path="unused/heading",
        success_path="unused/success",
        start=0,
        stop=1,
        eye_keypoint_indices=(1, 2),
        write_probabilities=False,
    )

    assert result is None
    assert seen["mode"] == "a"
    assert seen["kwargs"].get("use_consolidated") is False
    assert int(run_group["masks_roi"][0].sum()) > 0


def test_refresh_refined_eye_masks_registry_rows_targeted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    class _FakeRegistry:
        def __init__(self, _path: Path):
            pass

        def upsert_dataset(self, dataset_id: str, **kwargs: Any) -> None:
            calls["dataset_id"] = dataset_id
            calls["upsert_kwargs"] = kwargs

        def refresh_eye_mask_performance_for_dataset(self, dataset_id: str, **kwargs: Any) -> int:
            calls["perf"] = (dataset_id, kwargs)
            return 5

        def refresh_eye_mask_quality_for_dataset(self, dataset_id: str, **kwargs: Any) -> int:
            calls["quality"] = (dataset_id, kwargs)
            return 3

        def close(self) -> None:
            calls["closed"] = True

    class _Root:
        attrs = {"recording_id": "rec_001", "zarr_use": "analysis", "zarr_purpose": "analysis"}

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)
    monkeypatch.setattr(mod, "resolve_dataset_id", lambda _root, _path: ("dataset_001", "session_001"))
    monkeypatch.setattr(
        mod.RegistryPaths,
        "from_env",
        staticmethod(lambda _cwd: mod.RegistryPaths(path=tmp_path / "registry.sqlite")),
    )

    zarr_path = tmp_path / "recording_analysis.zarr"
    mod._refresh_refined_eye_masks_registry_rows(
        root=_Root(),  # type: ignore[arg-type]
        zarr_path=zarr_path,
        console=None,
    )

    assert calls["dataset_id"] == "dataset_001"
    assert calls["perf"][0] == "dataset_001"
    assert calls["quality"][0] == "dataset_001"
    assert calls["perf"][1]["zarr_path"] == zarr_path
    assert calls["quality"][1]["zarr_path"] == zarr_path
    assert calls["perf"][1]["recording_id"] == "rec_001"
    assert calls["quality"][1]["recording_id"] == "rec_001"
    assert calls["perf"][1]["zarr_use"] == "analysis"
    assert calls["quality"][1]["zarr_use"] == "analysis"
    assert calls["closed"] is True


def test_main_emits_error_status_before_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    emitted: list[dict[str, Any]] = []

    class _Root:
        attrs = {"recording_id": "rec_001"}

    monkeypatch.setattr(mod, "refine_eye_masks", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: _Root())

    def _capture_emit(**kwargs: Any) -> None:
        emitted.append(kwargs)

    monkeypatch.setattr(mod, "_emit_refined_eye_masks_status", _capture_emit)

    with pytest.raises(SystemExit) as excinfo:
        mod.main(
            [
                str(tmp_path / "recording_analysis.zarr"),
                "--run-name",
                "refined_eye_masks_test",
            ]
        )

    assert excinfo.value.code == 1
    assert emitted
    payload = emitted[0]
    assert payload["status"] == "error"
    assert payload["run_name"] == "refined_eye_masks_test"
    assert payload["method"] == "refine_eye_masks"
    details = payload["details"]
    assert details["reason"] == "runtime_error"
    assert "boom" in details["error"]


def test_build_arg_parser_defaults_to_binary_refined_output() -> None:
    parser = mod._build_arg_parser()
    args = parser.parse_args(["demo.zarr"])

    assert args.probability_threshold is None
    assert args.write_refined_probabilities is False


def test_build_arg_parser_accepts_refined_probability_flags() -> None:
    parser = mod._build_arg_parser()
    args = parser.parse_args(
        [
            "demo.zarr",
            "--probability-threshold",
            "0.5",
            "--write-refined-probabilities",
        ]
    )

    assert args.probability_threshold == pytest.approx(0.5)
    assert args.write_refined_probabilities is True


def test_build_arg_parser_accepts_distributed_scheduler() -> None:
    parser = mod._build_arg_parser()
    args = parser.parse_args(["demo.zarr", "--scheduler", "distributed"])

    assert args.scheduler == "distributed"


def test_resolve_probability_threshold_prefers_cli_arg() -> None:
    class _Run:
        attrs = {mod._RECOMMENDED_PROBABILITY_THRESHOLD_ATTR: 0.61}

    threshold, source = mod._resolve_probability_threshold(source_run=_Run(), explicit_threshold=0.33)

    assert threshold == pytest.approx(0.33)
    assert source == "cli_arg"


def test_resolve_probability_threshold_uses_recommended_metadata_when_cli_missing() -> None:
    class _Run:
        attrs = {mod._RECOMMENDED_PROBABILITY_THRESHOLD_ATTR: 0.58}

    threshold, source = mod._resolve_probability_threshold(source_run=_Run(), explicit_threshold=None)

    assert threshold == pytest.approx(0.58)
    assert source == f"source_run_attr:{mod._RECOMMENDED_PROBABILITY_THRESHOLD_ATTR}"


def test_resolve_probability_threshold_falls_back_to_default_for_invalid_metadata() -> None:
    class _Run:
        attrs = {mod._RECOMMENDED_PROBABILITY_THRESHOLD_ATTR: 9.9}

    threshold, source = mod._resolve_probability_threshold(source_run=_Run(), explicit_threshold=None)

    assert threshold == pytest.approx(mod._DEFAULT_PROBABILITY_THRESHOLD)
    assert source == "default"
