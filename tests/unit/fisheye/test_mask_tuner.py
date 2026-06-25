from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fisheye.tune import mask_tuner as mod


class _FakeAttrs(dict):
    pass


class _FakeGroup(dict):
    def __init__(self, attrs: dict | None = None, children: dict | None = None) -> None:
        super().__init__(children or {})
        self.attrs = _FakeAttrs(attrs or {})


class _FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def test_resolve_video_array_prefers_existing_images_ds() -> None:
    images_ds = _FakeArray((10, 64, 64))
    root = _FakeGroup(children={"raw_video": _FakeGroup(children={"images_ds": images_ds})})

    array, array_name, source = mod._resolve_video_array(root, use_full_res=False)

    assert array is images_ds
    assert array_name == "images_ds"
    assert source == "raw_video/images_ds"


def test_resolve_video_array_uses_metadata_only_source_video_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    captured: dict[str, object] = {}

    class FakePreviewArray:
        def __init__(self, video_path_arg, *, frame_count, source_shape, preview_shape):
            captured["video_path"] = video_path_arg
            captured["frame_count"] = frame_count
            captured["source_shape"] = source_shape
            captured["preview_shape"] = preview_shape
            self.shape = (frame_count, *preview_shape)

    monkeypatch.setattr(mod, "_SourceVideoPreviewArray", FakePreviewArray)
    root = _FakeGroup(
        attrs={
            "source_video_path": str(video_path),
            "inference_resolution": [640, 640],
            "source_video_resolution": [4512, 4512],
            "total_frames": 19235,
        },
        children={"raw_video": _FakeGroup(attrs={"total_frames": 19235})},
    )

    array, array_name, source = mod._resolve_video_array(root, use_full_res=False)

    assert array.shape == (19235, 640, 640)
    assert array_name == "images_ds"
    assert str(video_path) in source
    assert captured == {
        "video_path": video_path,
        "frame_count": 19235,
        "source_shape": (4512, 4512),
        "preview_shape": (640, 640),
    }


def test_resolve_video_array_reports_incomplete_metadata_only_source() -> None:
    root = _FakeGroup(children={"raw_video": _FakeGroup()})

    with pytest.raises(ValueError, match="source-video fallback metadata is incomplete"):
        mod._resolve_video_array(root, use_full_res=False)


def test_save_headless_circle_mask_refuses_existing_mask(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = {
        "shape": "circle",
        "detected_circle": {"center": [10, 11], "radius": 12},
    }
    root = _FakeGroup(
        children={
            "analysis_metadata": _FakeGroup(attrs={"dish_mask": existing}),
            "raw_video": _FakeGroup(children={"images_ds": _FakeArray((10, 64, 64))}),
        }
    )

    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    def fail_save(*_args, **_kwargs):
        raise AssertionError("save_mask_to_zarr should not be called")

    monkeypatch.setattr(mod, "save_mask_to_zarr", fail_save)

    result = mod.save_headless_circle_mask(
        tmp_path / "example.zarr",
        center=[20, 21],
        radius=22,
    )

    assert result["status"] == "exists"
    assert result["saved"] is False
    assert result["existing_mask"]["detected_circle"]["center"] == [10, 11]


def test_save_headless_circle_mask_writes_reviewed_circle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    images_ds = _FakeArray((11, 80, 96))
    root = _FakeGroup(children={"raw_video": _FakeGroup(children={"images_ds": images_ds})})
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    captured: dict[str, object] = {}

    def fake_save(zarr_path, mask_definition, array_name, frame_index, params=None, image_shape=None):
        captured["zarr_path"] = zarr_path
        captured["mask_definition"] = mask_definition
        captured["array_name"] = array_name
        captured["frame_index"] = frame_index
        captured["params"] = params
        captured["image_shape"] = image_shape
        return True

    monkeypatch.setattr(mod, "save_mask_to_zarr", fake_save)

    result = mod.save_headless_circle_mask(
        tmp_path / "example.zarr",
        center=[30, 31],
        radius=32,
        frame_idx=99,
    )

    assert result["status"] == "saved"
    assert result["array_name"] == "images_ds"
    assert result["frame_index"] == 10
    assert captured["array_name"] == "images_ds"
    assert captured["frame_index"] == 10
    assert captured["image_shape"] == (80, 96)
    assert captured["mask_definition"] == {
        "shape": "circle",
        "method": "headless_circle",
        "detected_circle": {"center": [30, 31], "radius": 32},
    }


def test_sync_registry_after_save_records_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class Result:
        synced = True
        dataset_id = "dataset_1"
        message = "ok"

        def to_dict(self) -> dict[str, object]:
            return {"synced": True, "dataset_id": self.dataset_id}

    def fake_sync(zarr_path, registry_path, *, method, source, details):
        captured["zarr_path"] = zarr_path
        captured["registry_path"] = registry_path
        captured["method"] = method
        captured["source"] = source
        captured["details"] = details
        return Result()

    monkeypatch.setattr(mod, "sync_dish_mask_registry_status", fake_sync)

    result = mod._sync_registry_after_save(
        tmp_path / "example.zarr",
        tmp_path / "registry.sqlite",
        method="headless_circle",
        shape="circle",
        array_name="images_ds",
        frame_index=5,
    )

    assert result == {"synced": True, "dataset_id": "dataset_1"}
    assert captured["method"] == "headless_circle"
    assert captured["source"] == "mask_tuner"
    assert captured["details"] == {
        "shape": "circle",
        "array_name": "images_ds",
        "frame_index": 5,
    }


def test_query_registry_dish_mask_candidates_filters_missing_analysis_paths(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    conn = sqlite3.connect(registry_path)
    try:
        conn.executescript(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT,
                zarr_use TEXT,
                status TEXT
            );
            CREATE TABLE recording_step_status (
                dataset_id TEXT,
                step_name TEXT,
                status TEXT
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO datasets (dataset_id, recording_id, zarr_path, zarr_use, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    "missing_goodcop",
                    "rec1",
                    "/groups/recordings/arena_1_GoodCopBadCop/zarr/arena_1_analysis.zarr",
                    "analysis",
                    "ok",
                ),
                (
                    "error_goodcop",
                    "rec2",
                    "/groups/recordings/arena_2_GoodCopBadCop/zarr/arena_2_analysis.zarr",
                    "analysis",
                    "ok",
                ),
                (
                    "ok_goodcop",
                    "rec3",
                    "/groups/recordings/arena_3_GoodCopBadCop/zarr/arena_3_analysis.zarr",
                    "analysis",
                    "ok",
                ),
                (
                    "other_name",
                    "rec4",
                    "/groups/recordings/other/zarr/other_analysis.zarr",
                    "analysis",
                    "ok",
                ),
                (
                    "training_goodcop",
                    "rec5",
                    "/groups/recordings/arena_5_GoodCopBadCop/zarr/training.zarr",
                    "training",
                    "ok",
                ),
                (
                    "deleted_goodcop",
                    "rec6",
                    "/groups/recordings/arena_6_GoodCopBadCop/zarr/deleted_analysis.zarr",
                    "analysis",
                    "deleted",
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO recording_step_status (dataset_id, step_name, status)
            VALUES (?, ?, ?)
            """,
            [
                ("error_goodcop", "dish_mask", "error"),
                ("ok_goodcop", "dish_mask", "ok"),
                ("missing_goodcop", "detect", "ok"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    candidates = mod._query_registry_dish_mask_candidates(
        registry_path,
        path_contains=("GoodCopBadCop",),
        missing_only=True,
    )

    assert [candidate.dataset_id for candidate in candidates] == [
        "missing_goodcop",
        "error_goodcop",
    ]
    assert [candidate.dish_mask_status for candidate in candidates] == [None, "error"]
