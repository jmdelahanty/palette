from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fisheye.segmentation import eye_segmentation_yolo as yolo_mod
from fisheye.segmentation import infer_unet_eye_masks as unet_mod


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:  # noqa: A003
        return super().get(key, default)


class _FakeRegistry:
    def __init__(self) -> None:
        self.upsert_dataset_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def upsert_dataset(self, *args: Any, **kwargs: Any) -> None:
        self.upsert_dataset_calls.append((args, kwargs))


def test_yolo_emit_eye_masks_status_writes_ok_and_invalidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)

    run = _FakeGroup(
        attrs={
            "method": "yolo_eye_segmentation",
            "source_crop_run": "crop_001",
            "source_keypoints_run": "kp_001",
            "successful_eyes": 20,
            "successful_roi_pairs": 9,
            "total_rois": 12,
            "successful_roi_pair_rate": 0.75,
            "probability_stats": {"pmax_mean": 0.8},
            "eye_mask_review_status": {"state": "pending"},
        }
    )
    parent = _FakeGroup({"eye_masks_run_001": run})
    root = _FakeGroup(
        {"eye_masks_runs": parent},
        attrs={"recording_id": "rec_001", "zarr_use": "analysis", "zarr_purpose": "analysis"},
    )
    registry = _FakeRegistry()

    status_calls: list[dict[str, Any]] = []
    cascade_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(yolo_mod, "resolve_dataset_id", lambda *_args, **_kwargs: ("ds_001", "sess_001"))
    monkeypatch.setattr(
        yolo_mod,
        "upsert_recording_step_status",
        lambda *_args, **kwargs: status_calls.append(kwargs),
    )
    monkeypatch.setattr(
        yolo_mod,
        "invalidate_downstream_steps",
        lambda *_args, **kwargs: cascade_calls.append(kwargs),
    )

    yolo_mod._emit_eye_masks_status(  # noqa: SLF001
        registry=registry,  # type: ignore[arg-type]
        root=root,  # type: ignore[arg-type]
        zarr_path=zarr_path,
        status="ok",
        reason="present",
        run_name="eye_masks_run_001",
        requested_crop_run="crop_001",
        method_hint="yolo_eye_segmentation",
        status_details={"selected_model_path": "/tmp/eye_model.pt"},
        error_text=None,
        console=None,
    )

    assert registry.upsert_dataset_calls
    assert len(status_calls) == 1
    assert status_calls[0]["step_name"] == "eye_masks"
    assert status_calls[0]["status"] == "ok"
    assert status_calls[0]["run_name"] == "eye_masks_run_001"
    assert status_calls[0]["method"] == "yolo_eye_segmentation"
    assert status_calls[0]["coverage_pct"] == pytest.approx(75.0)
    details = status_calls[0]["details_json"]
    assert isinstance(details, dict)
    assert details.get("reason") == "present"
    assert details.get("selected_model_path") == "/tmp/eye_model.pt"
    assert len(cascade_calls) == 1
    assert cascade_calls[0]["step_name"] == "eye_masks"
    assert cascade_calls[0]["trigger_run_name"] == "eye_masks_run_001"


def test_yolo_emit_eye_masks_status_error_does_not_invalidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)

    root = _FakeGroup(attrs={"recording_id": "rec_001"})
    registry = _FakeRegistry()

    status_calls: list[dict[str, Any]] = []
    cascade_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(yolo_mod, "resolve_dataset_id", lambda *_args, **_kwargs: ("ds_001", "sess_001"))
    monkeypatch.setattr(
        yolo_mod,
        "upsert_recording_step_status",
        lambda *_args, **kwargs: status_calls.append(kwargs),
    )
    monkeypatch.setattr(
        yolo_mod,
        "invalidate_downstream_steps",
        lambda *_args, **kwargs: cascade_calls.append(kwargs),
    )

    yolo_mod._emit_eye_masks_status(  # noqa: SLF001
        registry=registry,  # type: ignore[arg-type]
        root=root,  # type: ignore[arg-type]
        zarr_path=zarr_path,
        status="error",
        reason="runtime_error",
        run_name=None,
        requested_crop_run=None,
        method_hint="yolo_eye_segmentation",
        status_details=None,
        error_text="gpu oom",
        console=None,
    )

    assert len(status_calls) == 1
    assert status_calls[0]["status"] == "error"
    assert cascade_calls == []


def test_unet_emit_eye_masks_status_writes_ok_and_invalidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)

    run = _FakeGroup(
        attrs={
            "method": "unet_eye_mask_segmenter",
            "source_crop_run": "crop_002",
            "source_eye_masks_run": "eye_source_001",
            "source_keypoints_run": "kp_002",
            "probabilities_channels": 1,
            "total_rois": 10,
            "inference_duration_seconds": 2.5,
            "masks_from": "threshold(mask_probs_roi, thr=0.5)",
        }
    )
    parent = _FakeGroup({"eye_masks_unet_001": run})
    root = _FakeGroup({"eye_masks_runs": parent}, attrs={"recording_id": "rec_002"})
    registry = _FakeRegistry()

    status_calls: list[dict[str, Any]] = []
    cascade_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(unet_mod, "resolve_dataset_id", lambda *_args, **_kwargs: ("ds_002", "sess_002"))
    monkeypatch.setattr(
        unet_mod,
        "upsert_recording_step_status",
        lambda *_args, **kwargs: status_calls.append(kwargs),
    )
    monkeypatch.setattr(
        unet_mod,
        "invalidate_downstream_steps",
        lambda *_args, **kwargs: cascade_calls.append(kwargs),
    )

    unet_mod._emit_eye_masks_status(  # noqa: SLF001
        registry=registry,  # type: ignore[arg-type]
        root=root,  # type: ignore[arg-type]
        zarr_path=zarr_path,
        status="ok",
        reason="present",
        run_name="eye_masks_unet_001",
        requested_crop_run="crop_002",
        method_hint="unet_eye_mask_segmenter",
        status_details={"selected_set_id": "eye_masks_set_002"},
        error_text=None,
        console=None,
    )

    assert registry.upsert_dataset_calls
    assert len(status_calls) == 1
    assert status_calls[0]["method"] == "unet_eye_mask_segmenter"
    assert status_calls[0]["coverage_pct"] == pytest.approx(100.0)
    details = status_calls[0]["details_json"]
    assert isinstance(details, dict)
    assert details.get("selected_set_id") == "eye_masks_set_002"
    assert details.get("reason") == "present"
    assert len(cascade_calls) == 1
