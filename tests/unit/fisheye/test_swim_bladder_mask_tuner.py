from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.tune import dispatcher
from fisheye.tune import subject_mask_tuner as subject_mod
from fisheye.tune import swim_bladder_mask_tuner as mod


class _FakeGroup(dict):
    def __init__(self, *args: object, attrs: dict[str, object] | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: object = None) -> object:
        return super().get(key, default)

    def require_group(self, key: str) -> "_FakeGroup":
        value = self.get(key)
        if isinstance(value, _FakeGroup):
            return value
        value = _FakeGroup()
        self[key] = value
        return value


def test_build_swim_bladder_params_merges_saved_values() -> None:
    params = mod._build_swim_bladder_params(
        {
            "roi_padding": 24,
            "pre_threshold": 77,
            "sobel_strength": 0.25,
            "min_area": 12,
            "max_area": 88,
            "min_circularity": 0.45,
            "closing_radius": 2,
            "opening_radius": 1,
        }
    )

    assert params == {
        "roi_padding": 24,
        "pre_threshold": 77,
        "sobel_strength": 0.25,
        "min_area": 12,
        "max_area": 88,
        "min_circularity": 0.45,
        "closing_radius": 2,
        "opening_radius": 1,
    }


def test_compute_swim_bladder_patch_preview_selects_region_nearest_center() -> None:
    patch = np.full((24, 24), 180, dtype=np.uint8)
    patch[3:7, 3:7] = 10
    patch[14:20, 14:20] = 10

    preview = mod._compute_swim_bladder_patch_preview(
        patch,
        center_xy=(4.0, 4.0),
        params={
            "roi_padding": 18,
            "pre_threshold": 50,
            "sobel_strength": 0.0,
            "min_area": 4,
            "max_area": None,
            "min_circularity": None,
            "closing_radius": 0,
            "opening_radius": 0,
        },
    )

    assert int(preview["proposal_mask"].sum()) == 16
    assert preview["stats"]["selected_area"] == 16
    assert preview["stats"]["bbox_xyxy"] == [3, 3, 7, 7]


def test_save_subject_mask_params_supports_pre_normalized_swim_bladder_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _FakeGroup({"analysis_metadata": _FakeGroup()})
    monkeypatch.setattr(subject_mod, "open_zarr_root", lambda *args, **kwargs: root)

    ok, message = subject_mod.save_subject_mask_params(
        "recording_training.zarr",
        {
            "roi_padding": 20,
            "pre_threshold": 71,
            "sobel_strength": 0.3,
            "min_area": 10,
            "max_area": 99,
            "min_circularity": 0.5,
            "closing_radius": 2,
            "opening_radius": 1,
        },
        context={"storage_component_name": "swim_bladder"},
        component_name="swim_bladder",
        method="global_threshold_otsu",
        extra_entry_fields={
            "subject_method_family": "swim_bladder_patch_threshold_v1",
            "output_labels": ["swim_bladder"],
            "storage_component": "swim_bladder",
        },
        normalize_params=False,
    )

    assert ok is True
    assert message == "swim_bladder tuning saved"
    tuning = root["analysis_metadata"].attrs[subject_mod.TUNING_KEY]
    entry = tuning["components"]["swim_bladder"]
    assert entry["method"] == "global_threshold_otsu"
    assert entry["subject_method_family"] == "swim_bladder_patch_threshold_v1"
    assert entry["storage_component"] == "swim_bladder"
    assert entry["tuned_parameters"]["roi_padding"] == 20
    assert entry["tuned_parameters"]["min_circularity"] == 0.5


def test_dispatcher_swim_bladder_patch_alias_invokes_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir()
    captured: dict[str, object] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(mod, "main", _fake_main)

    result = dispatcher.run_tuner("swimbladder-patch-mask", str(zarr_path), frame_idx=9)

    assert result == 0
    assert captured["argv"] == [str(zarr_path), "--roi-index", "9"]
