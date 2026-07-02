from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from fisheye.registry import maintenance
from fisheye.status_page import query as status_query
from fisheye.tune import dispatcher
from fisheye.tune import subject_mask_tuner as mod
from fisheye.utils import apply_tuning_by_camera
from fisheye.utils import check_recording_steps


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


def test_save_subject_mask_params_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _FakeGroup({"analysis_metadata": _FakeGroup()})
    monkeypatch.setattr(mod, "open_zarr_root", lambda *args, **kwargs: root)

    ok, message = mod.save_subject_mask_params(
        "recording_analysis.zarr",
        {
            "diff_threshold": 63,
            "gaussian_blur_kernel": 4,
            "closing_radius": 5,
            "opening_radius": 2,
            "min_area": 144,
            "keep_largest_component": True,
        },
        context={
            "crop_run": "crop_001",
            "background_run": "background_001",
            "background_array": "background_full",
            "roi_index": 17,
            "frame_index": 99,
            "detection_index": 5,
        },
    )

    assert ok is True
    assert message == "subject_body tuning saved"

    analysis = root["analysis_metadata"]
    tuning = analysis.attrs[mod.TUNING_KEY]
    assert tuning["latest_component"] == "subject_body"
    assert "subject_body" in tuning["components"]

    loaded = mod._load_tuned_params_from_root(root, "subject_body")
    assert loaded == {
        "diff_threshold": 63,
        "gaussian_blur_kernel": 5,
        "closing_radius": 5,
        "opening_radius": 2,
        "min_area": 144,
        "keep_largest_component": True,
    }


def test_save_subject_mask_params_preserves_other_component_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = _FakeGroup(
        attrs={
            mod.TUNING_KEY: {
                "version": "2.0",
                "components": {
                    "subject_body": {
                        "method": "traditional_subject_mask_seed",
                        "version": "1.0",
                        "tuned_parameters": {"diff_threshold": 51},
                        "context": {"storage_component_name": "subject_body"},
                    }
                },
            }
        }
    )
    root = _FakeGroup({"analysis_metadata": analysis})
    monkeypatch.setattr(mod, "open_zarr_root", lambda *args, **kwargs: root)

    ok, _message = mod.save_subject_mask_params(
        "recording_analysis.zarr",
        {
            "roi_padding": 14,
            "pre_threshold": 33,
            "sobel_strength": 0.25,
            "min_area": 18,
            "max_area": 120,
            "min_circularity": 0.4,
            "closing_radius": 3,
            "opening_radius": 1,
            "min_eye_separation": 5.0,
            "max_eye_separation": 40.0,
        },
        component_name="eyes_union",
        method="global_threshold_otsu",
        mirror_eye_tuning_compat=True,
        extra_entry_fields={"migrated_from_eye_mask_tuning": False},
        context={"storage_component_name": "eyes_union"},
    )

    assert ok is True
    tuning = analysis.attrs[mod.TUNING_KEY]
    assert set(tuning["components"]) == {"subject_body", "eyes_union"}
    assert tuning["components"]["subject_body"]["tuned_parameters"] == {"diff_threshold": 51}
    assert tuning["components"]["eyes_union"]["method"] == "global_threshold_otsu"
    assert tuning["components"]["eyes_union"]["subject_method_family"] == "eye_mask_threshold_lr_v1"
    assert tuning["components"]["eyes_union"]["output_labels"] == ["eye_left", "eye_right"]
    assert tuning["components"]["eyes_union"]["storage_component"] == "eyes_union"
    assert tuning["components"]["eyes_union"]["context"]["storage_component_name"] == "eyes_union"
    assert analysis.attrs[mod.EYE_TUNING_COMPAT_KEY]["tuned_parameters"]["roi_padding"] == 14
    assert "rotate_roi" not in analysis.attrs[mod.EYE_TUNING_COMPAT_KEY]["tuned_parameters"]


def test_postprocess_subject_mask_keeps_largest_component_only() -> None:
    binary = np.zeros((12, 14), dtype=np.uint8)
    binary[1:5, 1:5] = 1
    binary[6:11, 7:13] = 1

    processed, stats = mod._postprocess_subject_mask(
        binary,
        closing_radius_value=0,
        opening_radius_value=0,
        min_area_value=4,
        keep_largest=True,
    )

    assert processed.dtype == np.uint8
    assert int(processed.sum()) == 30
    assert stats["candidate_components"] == 2
    assert stats["kept_components"] == 1
    assert stats["largest_area"] == 30
    assert stats["kept_area"] == 30
    assert stats["bbox_xyxy"] == [7, 6, 13, 11]


def test_compute_subject_mask_seed_respects_allowed_mask() -> None:
    roi = np.full((4, 4), 180, dtype=np.uint8)
    roi[2:, 2:] = 20
    background = np.full((4, 4), 180, dtype=np.uint8)
    allowed_mask = np.zeros((4, 4), dtype=np.uint8)
    allowed_mask[:2, :2] = 1

    preview = mod._compute_subject_mask_seed(
        roi,
        background,
        {
            "diff_threshold": 30,
            "gaussian_blur_kernel": 0,
            "closing_radius": 0,
            "opening_radius": 0,
            "min_area": 1,
            "keep_largest_component": True,
        },
        allowed_mask=allowed_mask,
    )

    assert int(preview["processed_mask"].sum()) == 0
    assert int(preview["diff_raw"][2:, 2:].sum()) == 0


def test_extract_dish_mask_roi_projects_saved_rectangle() -> None:
    root = _FakeGroup(
        {
            "analysis_metadata": _FakeGroup(
                attrs={
                    "dish_mask": {
                        "shape": "rectangle",
                        "tuned_on_array": "images_full",
                        "rectangle": {"roi": [0, 0, 3, 3]},
                        "metrics": {"image_shape": [8, 8]},
                    }
                }
            ),
            "raw_video": _FakeGroup({"images_full": np.zeros((1, 8, 8), dtype=np.uint8)}),
        },
        attrs={"width": 8, "height": 8},
    )

    dish_mask_image, tuned_on_array, _shape = mod._resolve_dish_mask_projection(root)

    assert dish_mask_image is not None
    assert tuned_on_array == "images_full"

    inside = mod._extract_dish_mask_roi(
        dish_mask_image,
        tuned_on_array=tuned_on_array,
        top_left_xy_full=np.asarray([0, 0], dtype=np.float32),
        roi_shape=(4, 4),
        full_shape=(8, 8),
    )
    outside = mod._extract_dish_mask_roi(
        dish_mask_image,
        tuned_on_array=tuned_on_array,
        top_left_xy_full=np.asarray([5, 5], dtype=np.float32),
        roi_shape=(4, 4),
        full_shape=(8, 8),
    )

    assert int(np.asarray(inside).sum()) > 0
    assert int(np.asarray(outside).sum()) == 0


def test_resolve_subject_component_defaults_to_only_available_channel() -> None:
    subject_run = _FakeGroup(
        {
            "masks_roi": np.zeros((2, 3, 4, 4), dtype=np.uint8),
            "mask_probs_roi": np.zeros((2, 3, 4, 4), dtype=np.uint8),
            "available_channels": np.asarray([False, True, False], dtype=bool),
        },
        attrs={
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "probabilities_encoding": "linear_uint8_0_255",
        },
    )
    parent = _FakeGroup({"subject_masks_canary_001": subject_run}, attrs={"latest": "subject_masks_canary_001"})
    root = _FakeGroup({"subject_mask_runs": parent})

    source = mod._load_subject_mask_source(root, None)
    component_name, component_index = mod._resolve_subject_component(source, None)

    assert source is not None
    assert source.run_name == "subject_masks_canary_001"
    assert component_name == "eyes_union"
    assert component_index == 1


def test_resolve_subject_component_defaults_to_derived_eye_union_for_lr_subject_run() -> None:
    subject_run = _FakeGroup(
        {
            "masks_roi": np.zeros((2, 4, 4, 4), dtype=np.uint8),
            "mask_probs_roi": np.zeros((2, 4, 4, 4), dtype=np.uint8),
            "available_channels": np.asarray([False, True, True, False], dtype=bool),
        },
        attrs={
            "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "probabilities_encoding": "linear_uint8_0_255",
        },
    )
    parent = _FakeGroup({"subject_masks_canary_001": subject_run}, attrs={"latest": "subject_masks_canary_001"})
    root = _FakeGroup({"subject_mask_runs": parent})

    source = mod._load_subject_mask_source(root, None)
    component_name, component_index = mod._resolve_subject_component(source, None)

    assert source is not None
    assert component_name == "eyes_union"
    assert component_index is None


def test_subject_component_preview_derives_union_from_lr_channels() -> None:
    masks = np.zeros((1, 4, 4, 4), dtype=np.uint8)
    masks[0, 1, 0, 0] = 1
    masks[0, 2, 1, 1] = 1
    probs = np.zeros((1, 4, 4, 4), dtype=np.uint8)
    probs[0, 1, 0, 0] = 255
    probs[0, 2, 1, 1] = 128
    subject_run = _FakeGroup(
        {
            "masks_roi": masks,
            "mask_probs_roi": probs,
            "available_channels": np.asarray([False, True, True, False], dtype=bool),
        },
        attrs={
            "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "probabilities_encoding": "linear_uint8_0_255",
        },
    )
    source = mod.SubjectMaskSource(
        run_name="subject_masks_canary_001",
        group=subject_run,
        mask_labels=("subject_body", "eye_left", "eye_right", "swim_bladder"),
        available_channels=np.asarray([False, True, True, False], dtype=bool),
        masks_roi=masks,
        mask_probs_roi=probs,
        probability_encoding="linear_uint8_0_255",
    )

    preview = mod._subject_component_preview(
        source,
        roi_index=0,
        component_name="eyes_union",
        component_index=None,
    )

    np.testing.assert_array_equal(
        preview["mask"],
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_allclose(
        preview["prob"],
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 128.0 / 255.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert preview["derived"] is True


def test_resolve_subject_component_rejects_unavailable_requested_component() -> None:
    subject_run = _FakeGroup(
        {
            "masks_roi": np.zeros((2, 3, 4, 4), dtype=np.uint8),
            "mask_probs_roi": np.zeros((2, 3, 4, 4), dtype=np.uint8),
            "available_channels": np.asarray([False, True, False], dtype=bool),
        },
        attrs={
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "probabilities_encoding": "linear_uint8_0_255",
        },
    )
    parent = _FakeGroup({"subject_masks_canary_001": subject_run}, attrs={"latest": "subject_masks_canary_001"})
    root = _FakeGroup({"subject_mask_runs": parent})
    source = mod._load_subject_mask_source(root, None)

    with pytest.raises(ValueError, match="unavailable"):
        mod._resolve_subject_component(source, "subject_body")


def test_require_gui_display_reports_missing_tmux_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("TMUX", "/tmp/tmux-test/default,1,0")

    with pytest.raises(RuntimeError, match="tmux"):
        mod._require_gui_display()


def test_subject_mask_tuner_is_registered_in_cli_surfaces() -> None:
    assert "subject_mask_tuning" in apply_tuning_by_camera.DEFAULT_KEYS
    assert "subject_mask_tuning" in check_recording_steps.DEFAULT_TUNING_KEYS
    assert check_recording_steps._STEP_NAME_ALIASES["subject_mask_tuning"] == "subject_mask_tuning"  # noqa: SLF001
    assert check_recording_steps._OVERVIEW_STEP_PREFIX["subject_mask_tuning"] == "subject_mask_tuning"  # noqa: SLF001
    assert "subject_mask_tuning" in maintenance.RECORDING_TUNING_STEP_NAMES
    assert "subject_mask_tuning" in status_query.STEP_SORT_ORDER


@pytest.mark.parametrize(
    ("tuner_name", "expected_component"),
    [
        ("subject-body-mask", "subject_body"),
        ("eye-union-mask", "eyes_union"),
        ("eye-left-mask", "eye_left"),
        ("eye-right-mask", "eye_right"),
        ("swimbladder-mask", "swim_bladder"),
    ],
)
def test_dispatcher_subject_mask_aliases_pass_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tuner_name: str,
    expected_component: str,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir()
    captured: dict[str, object] = {}

    def _fake_main(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(mod, "main", _fake_main)

    result = dispatcher.run_tuner(tuner_name, str(zarr_path), frame_idx=7)

    assert result == 0
    assert captured["zarr_path"] == str(zarr_path)
    assert captured["roi_index"] == 7
    assert captured["component"] == expected_component


def test_list_tuners_mentions_subject_mask() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    dispatcher.list_tuners(console)

    output = buffer.getvalue()
    assert "subject-mask" in output
    assert "raw subject-mask components" in output
    assert "eye-union-mask" in output
    assert "eye-left-mask" in output
    assert "eye-right-mask" in output
