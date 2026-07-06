from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fisheye.shared.import_profile_contract import (
    PROFILE_LEGACY_DECORD_TRAINING_OR_FULL,
    PROFILE_METADATA_ONLY_ANALYSIS,
    PROFILE_MISSING_RAW_VIDEO,
    PROFILE_SAMPLED_TRAINING_PYNVVC_LUMA,
    classify_import_profile,
)
from fisheye.utils import check_import_profile


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.attrs: dict[str, Any] = {}


class FakeGroup:
    def __init__(
        self,
        *,
        attrs: dict[str, Any] | None = None,
        children: dict[str, Any] | None = None,
    ) -> None:
        self.attrs = attrs or {}
        self._children = children or {}

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def __getitem__(self, name: str) -> Any:
        return self._children[name]

    def keys(self) -> list[str]:
        return list(self._children)

    def group_keys(self) -> list[str]:
        return [name for name, child in self._children.items() if isinstance(child, FakeGroup)]


def _root_with_raw(raw_attrs: dict[str, Any], *, root_attrs: dict[str, Any] | None = None, arrays: dict[str, Any] | None = None) -> FakeGroup:
    raw = FakeGroup(attrs=raw_attrs, children=arrays or {})
    return FakeGroup(attrs=root_attrs or {}, children={"raw_video": raw})


def test_classifies_metadata_only_analysis_profile_ok() -> None:
    root = _root_with_raw(
        {
            "import_method": "metadata_only",
            "import_stage": "metadata_only",
            "total_frames": 1000,
            "fps": 100.0,
            "source_path": "/data/source.mp4",
            "source_video_fingerprint": "stat_v1:abc",
            "color_range": "tv",
            "video_codec": "hevc",
            "video_pix_fmt": "yuv420p",
        },
        root_attrs={"zarr_purpose": "analysis"},
    )

    report = classify_import_profile(root)

    assert report.profile == PROFILE_METADATA_ONLY_ANALYSIS
    assert report.status == "ok"
    assert report.required_missing == ()


def test_classifies_video_prefixed_colorimetry_as_present() -> None:
    root = _root_with_raw(
        {
            "import_method": "metadata_only",
            "import_stage": "metadata_only",
            "total_frames": 1000,
            "fps": 100.0,
            "source_path": "/data/source.mp4",
            "source_video_fingerprint": "stat_v1:abc",
            "video_color_space": "bt709",
            "video_codec": "hevc",
            "video_pix_fmt": "yuv420p",
        },
        root_attrs={"zarr_purpose": "analysis"},
    )

    report = classify_import_profile(root)

    assert report.profile == PROFILE_METADATA_ONLY_ANALYSIS
    assert report.status == "ok"
    assert "MISSING_COLORIMETRY" not in report.reason_codes


def test_classifies_sampled_training_pynvvc_profile_ok() -> None:
    root = _root_with_raw(
        {
            "import_method": "pynvvc_luma_sampled_training",
            "decode_backend": "pynvvc_luma",
            "frame_step": 700,
            "source_frame_count": 140_000,
            "pixel_contract_name": "orange_mono_pynvvc_luma_uint8_v1",
            "source_video_path": "/data/source.mp4",
            "source_video_fingerprint": "stat_v1:def",
            "color_matrix": "source_encoded_nv12_y_plane",
        },
        root_attrs={"zarr_purpose": "training"},
        arrays={
            "images_full": FakeArray((200, 4512, 4512)),
            "images_ds": FakeArray((200, 640, 640)),
            "original_frame_indices": FakeArray((200,)),
        },
    )

    report = classify_import_profile(root)

    assert report.profile == PROFILE_SAMPLED_TRAINING_PYNVVC_LUMA
    assert report.status == "ok"
    assert "raw_video/images_full" in report.arrays_present
    assert "raw_video/original_frame_indices" in report.arrays_present


def test_sampled_training_profile_missing_required_fields_is_incomplete() -> None:
    root = _root_with_raw(
        {
            "import_method": "pynvvc_luma_sampled_training",
            "decode_backend": "pynvvc_luma",
            "frame_step": 700,
            "source_video_path": "/data/source.mp4",
        },
        root_attrs={"zarr_purpose": "training"},
        arrays={"images_full": FakeArray((200, 4512, 4512))},
    )

    report = classify_import_profile(root)

    assert report.profile == PROFILE_SAMPLED_TRAINING_PYNVVC_LUMA
    assert report.status == "incomplete"
    assert "MISSING_REQUIRED_IMPORT_PROFILE_FIELDS" in report.reason_codes
    assert "raw_video/original_frame_indices" in report.required_missing
    assert "raw_video.attrs.source_frame_count" in report.required_missing
    assert "raw_video.attrs.pixel_contract_name" in report.required_missing


def test_historical_training_arrays_do_not_imply_active_pynvvc_profile() -> None:
    root = _root_with_raw(
        {
            "import_method": "kvikio_zarr",
            "import_stage": "complete",
        },
        root_attrs={"zarr_purpose": "training"},
        arrays={
            "images_full": FakeArray((200, 4512, 4512)),
            "images_ds": FakeArray((200, 640, 640)),
            "original_frame_indices": FakeArray((200,)),
            "timestamps": FakeArray((200,)),
        },
    )

    report = classify_import_profile(root)

    assert report.profile == PROFILE_LEGACY_DECORD_TRAINING_OR_FULL
    assert report.status == "warning"
    assert report.profile != PROFILE_SAMPLED_TRAINING_PYNVVC_LUMA
    assert "MISSING_REQUIRED_IMPORT_PROFILE_FIELDS" not in report.reason_codes


def test_classifies_historical_decord_profile_but_keeps_it_separate() -> None:
    root = _root_with_raw(
        {
            "import_method": "fisheye.capture.import_video",
            "import_stage": "complete",
            "source_path": "/data/source.mp4",
        },
        arrays={"images_ds": FakeArray((100, 640, 640))},
    )

    report = classify_import_profile(root)

    assert report.profile == PROFILE_LEGACY_DECORD_TRAINING_OR_FULL
    assert report.status == "warning"
    assert "MISSING_SOURCE_VIDEO_FINGERPRINT" in report.reason_codes


def test_missing_raw_video_is_incomplete() -> None:
    root = FakeGroup(attrs={"zarr_purpose": "analysis"})

    report = classify_import_profile(root)

    assert report.profile == PROFILE_MISSING_RAW_VIDEO
    assert report.status == "incomplete"
    assert report.required_missing == ("raw_video",)


def test_check_import_profile_jsonl_cli_uses_classifier_without_real_zarr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _root_with_raw(
        {
            "import_method": "metadata_only",
            "import_stage": "metadata_only",
            "total_frames": 10,
            "fps": 100,
            "source_path": "/data/source.mp4",
            "source_video_fingerprint": "stat_v1:abc",
            "color_range": "tv",
        },
        root_attrs={"zarr_purpose": "analysis"},
    )

    monkeypatch.setattr(check_import_profile, "open_zarr_root", lambda path, mode="r": root)

    rc = check_import_profile.main(["--jsonl", "/tmp/example.zarr"])

    assert rc == 0
    row = json.loads(capsys.readouterr().out)
    assert row["zarr_path"] == "/tmp/example.zarr"
    assert row["profile"] == PROFILE_METADATA_ONLY_ANALYSIS
    assert row["status"] == "ok"


def test_check_import_profile_compact_summary_cli_uses_classifier_without_real_zarr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _root_with_raw(
        {
            "import_method": "metadata_only",
            "import_stage": "metadata_only",
            "total_frames": 10,
            "fps": 100,
            "source_path": "/data/source.mp4",
            "source_video_fingerprint": "stat_v1:abc",
            "video_color_range": "tv",
        },
        root_attrs={"zarr_purpose": "analysis"},
    )
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(check_import_profile, "open_zarr_root", lambda path, mode="r": root)

    rc = check_import_profile.main(["--jsonl", "--compact", "--summary", str(summary_path), "/tmp/example.zarr"])

    assert rc == 0
    row = json.loads(capsys.readouterr().out)
    assert "attrs_observed" not in row
    summary = json.loads(summary_path.read_text())
    assert summary["total"] == 1
    assert summary["status_counts"] == {"ok": 1}
    assert summary["profile_counts"] == {PROFILE_METADATA_ONLY_ANALYSIS: 1}


def test_check_import_profile_fail_on_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(check_import_profile, "open_zarr_root", lambda path, mode="r": FakeGroup())

    rc = check_import_profile.main(["--fail-on-incomplete", str(Path("/tmp/missing_raw.zarr"))])

    assert rc == 1
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["profile"] == PROFILE_MISSING_RAW_VIDEO
