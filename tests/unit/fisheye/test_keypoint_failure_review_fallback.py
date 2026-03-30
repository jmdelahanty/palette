from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.api.synchronous as zarr_sync_api
import zarr.core.sync as zarr_sync
from zarr.storage import MemoryStore

from fisheye.pose.metric_schema import DerivedMetricStorage, metric_schema_from_package
from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.tune import keypoint_failure_review as mod
from fisheye.tune.keypoint_failure_review import (
    _build_manual_reason,
    _build_no_reviewable_failures_auto_review,
    _active_index_from_key,
    _apply_review_status,
    _display_colors,
    _empty_review_auto_state,
    _load_raw_failure_indices,
    _mark_edit_applied,
    _roi_diagonal_from_roi_images,
    _set_review_derived_metric_row,
    launch_review,
    _resolve_full_frame_dimensions,
    _resolve_review_intended_use,
)


@pytest.fixture(autouse=True)
def _patch_zarr_sync(monkeypatch):
    def _sync_via_asyncio_run(coro, loop=None, timeout=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}
        error = {}

        def _runner():
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover - defensive
                error["exc"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "exc" in error:
            raise error["exc"]
        return result.get("value")

    monkeypatch.setattr(zarr_sync, "sync", _sync_via_asyncio_run)
    monkeypatch.setattr(zarr_sync_api, "sync", _sync_via_asyncio_run)


def test_resolve_full_frame_dimensions_from_root_attrs_when_images_full_missing(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.create_group("raw_video")

    full_h, full_w = _resolve_full_frame_dimensions(root)
    assert full_h == 4512
    assert full_w == 4512


def test_resolve_full_frame_dimensions_from_images_ds_when_attrs_missing(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis_ds_only.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", shape=(3, 720, 1280), dtype="u1")

    full_h, full_w = _resolve_full_frame_dimensions(root)
    assert full_h == 720
    assert full_w == 1280


def test_build_manual_reason_is_canonical_and_idempotent() -> None:
    first = _build_manual_reason("manual_correction|geometry_issue", geom_ok=False)
    second = _build_manual_reason(first, geom_ok=False)
    assert first == "manual_correction|geometry_issue"
    assert second == first


def test_display_colors_extends_beyond_default_triplet() -> None:
    colors = _display_colors(5)
    assert len(colors) == 5
    assert colors[:3] == ["#22c55e", "#1a66f3", "#f85151"]


def test_active_index_from_key_supports_dynamic_digits_and_cycle_keys() -> None:
    assert _active_index_from_key("1", label_count=5, current_idx=0) == 0
    assert _active_index_from_key("5", label_count=5, current_idx=0) == 4
    assert _active_index_from_key("6", label_count=5, current_idx=0) is None
    assert _active_index_from_key("]", label_count=5, current_idx=4) == 0
    assert _active_index_from_key("[", label_count=5, current_idx=0) == 4


def test_roi_diagonal_from_roi_images_uses_crop_shape() -> None:
    roi_images = np.zeros((7, 3, 4), dtype=np.uint8)
    assert _roi_diagonal_from_roi_images(roi_images) == 5.0


def test_review_helpers_update_and_clear_derived_metric_rows(tmp_path: Path) -> None:
    schema = metric_schema_from_package("traditional_v2")
    storage = DerivedMetricStorage(
        schema=schema,
        values=np.full((2, len(schema.metrics)), np.nan, dtype=np.float32),
        values_norm=np.full((2, len(schema.metrics)), np.nan, dtype=np.float32),
        valid=np.zeros((2, len(schema.metrics)), dtype=bool),
    )

    labels = ["swim_bladder", "eye_left", "eye_right", "snout_tip", "tail_tip"]
    points = np.array(
        [
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    changed = _set_review_derived_metric_row(
        storage,
        roi_idx=0,
        keypoints_roi=points,
        keypoint_labels=labels,
        roi_diagonal=5.0,
    )
    assert changed is True
    np.testing.assert_allclose(np.asarray(storage.values[0], dtype=np.float32), np.array([3.0, 1.0, 2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(storage.valid[0], dtype=bool), np.array([True, True, True, True], dtype=bool))

    changed = _set_review_derived_metric_row(
        storage,
        roi_idx=0,
        keypoints_roi=None,
        keypoint_labels=labels,
        roi_diagonal=5.0,
    )
    assert changed is True
    np.testing.assert_array_equal(np.asarray(storage.valid[0], dtype=bool), np.array([False, False, False, False], dtype=bool))
    assert np.isnan(np.asarray(storage.values[0], dtype=np.float32)).all()


def test_mark_edit_applied_sets_flag_once(tmp_path: Path) -> None:
    root = zarr.open_group(store=tmp_path / "edit_applied.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_1")
    edit_applied = refined.create_array("edit_applied", data=np.array([False, True], dtype=bool))

    assert _mark_edit_applied(edit_applied, 0) is True
    assert _mark_edit_applied(edit_applied, 0) is False
    assert _mark_edit_applied(edit_applied, 1) is False
    np.testing.assert_array_equal(np.asarray(edit_applied[:], dtype=bool), np.asarray([True, True], dtype=bool))


def test_resolve_review_intended_use_prefers_existing_status(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis_with_existing_review.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["zarr_use"] = "analysis"
    refined = root.create_group("refined_keypoints_runs").create_group("refined_1")
    refined.attrs["keypoint_review_status"] = {
        "state": "approved",
        "method": "algorithmic",
        "intended_use": "full_recording",
    }

    resolved = _resolve_review_intended_use(
        requested=None,
        refined=refined,
        root=root,
        zarr_path=str(zarr_path),
    )
    assert resolved == "full_recording"


def test_resolve_review_intended_use_uses_training_for_training_zarr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["zarr_use"] = "training"
    refined = root.create_group("refined_keypoints_runs").create_group("refined_1")

    resolved = _resolve_review_intended_use(
        requested=None,
        refined=refined,
        root=root,
        zarr_path=str(zarr_path),
    )
    assert resolved == "training"


def test_apply_review_status_triggers_registry_sync_hook(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    parent = root.create_group("refined_keypoints_runs")
    run = parent.create_group("refined_1")

    seen: dict[str, object] = {}

    def _fake_sync(**kwargs):  # noqa: ANN003
        seen.update(kwargs)
        return {"synced": True}

    monkeypatch.setattr(mod, "_sync_registry_after_review_status", _fake_sync)

    payload, sync = _apply_review_status(
        parent,
        "refined_1",
        run,
        zarr_path=str(zarr_path),
        state="approved",
        method="manual",
        intended_use="full_recording",
        reviewer="tester",
        notes=None,
    )

    assert payload["state"] == "approved"
    assert payload["method"] == "manual"
    assert payload["intended_use"] == "full_recording"
    assert parent.attrs["keypoint_review_status_latest"] == "refined_1"
    assert sync["synced"] is True
    assert seen["zarr_path"] == str(zarr_path)
    assert seen["refined_run"] == "refined_1"


def test_empty_review_auto_state_allows_fish_present_no_keypoints_only(tmp_path: Path) -> None:
    root = zarr.open_group(store=tmp_path / "fish_only.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_1")
    refined.create_array("refined_success", data=np.array([False, False], dtype=bool))
    write_reason_columns(
        refined,
        np.array(["fish_present_no_keypoints", "fish_present_no_keypoints"], dtype=object),
        chunk_size=2,
        include_reason_text=True,
        overwrite=True,
    )
    raw_failures = _load_raw_failure_indices(refined)
    allowed, reason = _empty_review_auto_state(refined, raw_failures)
    assert allowed is True
    assert reason is None


def test_empty_review_auto_state_blocks_detection_issue(tmp_path: Path) -> None:
    root = zarr.open_group(store=tmp_path / "detection_issue.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_1")
    refined.create_array("refined_success", data=np.array([False], dtype=bool))
    write_reason_columns(
        refined,
        np.array(["detection_issue"], dtype=object),
        chunk_size=1,
        include_reason_text=True,
        overwrite=True,
    )
    raw_failures = _load_raw_failure_indices(refined)
    allowed, reason = _empty_review_auto_state(refined, raw_failures)
    assert allowed is False
    assert reason == "detection_issue_present"


def test_build_no_reviewable_failures_auto_review_contains_policy_metadata(tmp_path: Path) -> None:
    root = zarr.open_group(store=tmp_path / "policy_meta.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_1")
    refined.create_array("refined_success", data=np.array([False, False], dtype=bool))
    write_reason_columns(
        refined,
        np.array(["fish_present_no_keypoints", "fish_present_no_keypoints"], dtype=object),
        chunk_size=2,
        include_reason_text=True,
        overwrite=True,
    )
    raw_failures = _load_raw_failure_indices(refined)
    payload = _build_no_reviewable_failures_auto_review(
        refined=refined,
        raw_failures=raw_failures,
        state="approved",
    )
    assert payload["policy_id"] == "keypoint_no_reviewable_failures_v1"
    assert payload["policy_version"] == 1
    assert payload["result"] == "approved"
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["raw_failure_count"] == 2
    assert evidence["fish_present_no_keypoints_count"] == 2
    assert evidence["detection_issue_count"] == 0


def test_launch_review_auto_applies_algorithmic_status_for_no_reviewable_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "analysis_auto_approve.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "refined_1"
    refined = refined_parent.create_group("refined_1")
    refined.create_array("refined_success", data=np.array([False], dtype=bool))
    write_reason_columns(
        refined,
        np.array(["fish_present_no_keypoints"], dtype=object),
        chunk_size=1,
        include_reason_text=True,
        overwrite=True,
    )
    refined.attrs["source_crop_run"] = "crop_1"
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_1")
    crop.create_array("roi_images", data=np.zeros((1, 8, 8), dtype=np.uint8))
    crop.create_array("roi_coordinates_full", data=np.zeros((1, 2), dtype=np.float64))
    crop.create_array("frame_indices", data=np.array([0], dtype=np.int64))

    seen: dict[str, object] = {}

    def _fake_apply(
        refined_parent_arg,
        refined_run_arg,
        refined_arg,
        *,
        zarr_path: str,
        state: str,
        method: str,
        intended_use: str,
        reviewer: str | None,
        notes: str | None,
        registry_path=None,
        auto_review=None,
    ):
        seen["method"] = method
        seen["reviewer"] = reviewer
        seen["auto_review"] = auto_review
        return (
            {"state": state, "method": method, "intended_use": intended_use},
            {"synced": True},
        )

    monkeypatch.setattr(mod, "_apply_review_status", _fake_apply)

    launch_review(str(zarr_path), review_state="approved")
    assert seen["method"] == "algorithmic"
    assert seen["reviewer"] == "auto:keypoint_no_reviewable_failures_v1"
    auto_review = seen["auto_review"]
    assert isinstance(auto_review, dict)
    assert auto_review.get("policy_id") == "keypoint_no_reviewable_failures_v1"


def test_launch_review_uses_direct_refined_run_lookup_when_parent_membership_is_stale(
    monkeypatch,
) -> None:
    class _FakeAttrs(dict):
        pass

    class _FakeRefined:
        attrs = {"keypoint_review_status": {"intended_use": "training"}}

    class _FakeParent:
        def __init__(self) -> None:
            self.attrs = _FakeAttrs(latest="refined_keypoints_traditional_v2_seed_001")

        def __contains__(self, key: object) -> bool:
            return False

    class _FakeCropGroup(dict):
        pass

    class _FakeRoot:
        def __init__(self) -> None:
            self.attrs = {"zarr_use": "training", "width": 8, "height": 8}
            self._refined = _FakeRefined()
            self._parent = _FakeParent()
            crop = _FakeCropGroup()
            crop["roi_images"] = np.zeros((1, 8, 8), dtype=np.uint8)
            crop["roi_coordinates_full"] = np.zeros((1, 2), dtype=np.float64)
            crop["frame_indices"] = np.array([0], dtype=np.int64)
            self._crop = crop

        def get(self, key: str):
            if key == "refined_keypoints_runs":
                return self._parent
            return None

        def __getitem__(self, key: str):
            if key == "refined_keypoints_runs/refined_keypoints_traditional_v2_seed_001":
                return self._refined
            if key == "crop_runs/crop_1":
                return self._crop
            raise KeyError(key)

    fake_root = _FakeRoot()

    monkeypatch.setattr(mod, "open_zarr_root", lambda *args, **kwargs: fake_root)
    monkeypatch.setattr(mod, "_get_latest_run", lambda *_args, **_kwargs: "crop_1")
    monkeypatch.setattr(mod, "_resolve_full_frame_dimensions", lambda root: (8, 8))
    monkeypatch.setattr(mod, "_load_failure_indices", lambda refined, include_all=False: np.array([], dtype="i4"))
    monkeypatch.setattr(mod, "_load_raw_failure_indices", lambda refined: np.array([], dtype="i4"))
    monkeypatch.setattr(mod, "_empty_review_auto_state", lambda refined, raw_failures: (True, None))
    monkeypatch.setattr(
        mod,
        "_build_no_reviewable_failures_auto_review",
        lambda **kwargs: {"policy_id": "keypoint_no_reviewable_failures_v1"},
    )

    seen: dict[str, object] = {}

    def _fake_apply(
        refined_parent_arg,
        refined_run_arg,
        refined_arg,
        *,
        zarr_path: str,
        state: str,
        method: str,
        intended_use: str,
        reviewer: str | None,
        notes: str | None,
        registry_path=None,
        auto_review=None,
    ):
        seen["refined_parent"] = refined_parent_arg
        seen["refined_run"] = refined_run_arg
        seen["refined"] = refined_arg
        seen["auto_review"] = auto_review
        return (
            {"state": state, "method": method, "intended_use": intended_use},
            {"synced": True},
        )

    monkeypatch.setattr(mod, "_apply_review_status", _fake_apply)

    launch_review("/tmp/fake_training.zarr", review_state="approved")

    assert seen["refined_parent"] is fake_root.get("refined_keypoints_runs")
    assert seen["refined_run"] == "refined_keypoints_traditional_v2_seed_001"
    assert seen["refined"] is fake_root["refined_keypoints_runs/refined_keypoints_traditional_v2_seed_001"]
