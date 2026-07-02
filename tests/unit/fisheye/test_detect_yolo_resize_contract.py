from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fisheye.detection import detect_yolo as mod


def test_normalize_legacy_video_resize_maps_width_height_to_height_width() -> None:
    assert mod._normalize_legacy_video_resize([1280, 768]) == [768, 1280]  # noqa: SLF001


def test_resize_dims_to_imgsz_returns_scalar_for_square_and_list_for_rectangular() -> None:
    assert mod._resize_dims_to_imgsz([640, 640]) == 640  # noqa: SLF001
    assert mod._resize_dims_to_imgsz([768, 1280]) == [768, 1280]  # noqa: SLF001


def test_normalize_decode_backend_defaults_to_auto() -> None:
    assert mod._normalize_decode_backend(None) == "auto"  # noqa: SLF001
    assert mod._normalize_decode_backend("") == "auto"  # noqa: SLF001
    assert mod._normalize_decode_backend("pynvvc_nv12_rgb") == "pynvvc_nv12_rgb"  # noqa: SLF001


def test_normalize_decode_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported decode backend"):
        mod._normalize_decode_backend("not_a_backend")  # noqa: SLF001


def test_decord_gpu_init_refuses_cpu_fallback_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_decord_available", lambda: True)  # noqa: SLF001
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="GPU decode unavailable; refusing CPU fallback"):
        mod._init_decord_reader(  # noqa: SLF001
            Path("video.mp4"),
            prefer_gpu=True,
            console=mod.Console(file=None),
        )


def test_decord_cpu_init_refuses_opencv_fallback(monkeypatch) -> None:
    class _FailingVideoReader:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("decord unavailable")

    monkeypatch.setattr(mod, "_decord_available", lambda: True)  # noqa: SLF001
    monkeypatch.setattr(mod, "VideoReader", _FailingVideoReader)
    monkeypatch.setattr(mod, "cpu", lambda: object())

    with pytest.raises(RuntimeError, match="Requested Decord CPU decoder failed"):
        mod._init_decord_reader(  # noqa: SLF001
            Path("video.mp4"),
            prefer_gpu=False,
            console=mod.Console(file=None),
        )


def test_record_timing_accumulates_perf_counter_elapsed(monkeypatch) -> None:
    timings = {"read_decode_seconds_total": 1.0}
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 12.5)

    elapsed = mod._record_timing(timings, "read_decode_seconds_total", 10.0)  # noqa: SLF001

    assert elapsed == pytest.approx(2.5)
    assert timings["read_decode_seconds_total"] == pytest.approx(3.5)


def test_pynvvc_streamed_batch_materializes_frames_before_surface_reuse() -> None:
    reusable_surface = torch.full((1, 1), 1, dtype=torch.uint8)
    second_surface = torch.full((1, 1), 2, dtype=torch.uint8)

    def frame_iter():
        yield reusable_surface
        reusable_surface.fill_(99)
        yield second_surface

    processed, count, read_seconds, preprocess_seconds = mod._read_and_preprocess_pynvvc_batch(  # noqa: SLF001
        frame_iter=frame_iter(),
        max_batch_frames=2,
        decode_backend_effective=mod.BACKEND_PYNVVC_LUMA_RGB,
        source_height=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        resize_hw=(1, 1),
    )

    assert count == 2
    assert read_seconds >= 0.0
    assert preprocess_seconds >= 0.0
    assert processed is not None
    values = torch.round(processed[:, 0, 0, 0] * 255.0).to(torch.int64).tolist()
    assert values == [1, 2]


def test_pynvvc_streamed_batch_returns_partial_owned_batch() -> None:
    def frame_iter():
        yield torch.full((1, 1), 7, dtype=torch.uint8)

    processed, count, _read_seconds, _preprocess_seconds = mod._read_and_preprocess_pynvvc_batch(  # noqa: SLF001
        frame_iter=frame_iter(),
        max_batch_frames=4,
        decode_backend_effective=mod.BACKEND_PYNVVC_LUMA_RGB,
        source_height=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        resize_hw=(1, 1),
    )

    assert count == 1
    assert processed is not None
    assert tuple(processed.shape) == (1, 3, 1, 1)
    assert torch.round(processed[0, :, 0, 0] * 255.0).to(torch.int64).tolist() == [7, 7, 7]


def test_opencv_batch_iterator_flushes_partial_batch_when_frame_count_is_overreported() -> None:
    class _FakeCapture:
        def __init__(self, frames: list[np.ndarray], reported_frame_count: int) -> None:
            self.frames = frames
            self.reported_frame_count = reported_frame_count
            self.position = 0

        def read(self):
            if self.position >= len(self.frames):
                return False, None
            frame = self.frames[self.position]
            self.position += 1
            return True, frame.copy()

    class _FakeCv2:
        COLOR_BGR2RGB = object()

        @staticmethod
        def cvtColor(frame: np.ndarray, _code) -> np.ndarray:
            return frame[..., ::-1].copy()

        @staticmethod
        def resize(_frame: np.ndarray, _size: tuple[int, int]) -> np.ndarray:
            raise AssertionError("resize should not be called")

    frames = [
        np.full((2, 2, 3), [index, index + 10, index + 20], dtype=np.uint8)
        for index in range(3)
    ]
    cap = _FakeCapture(frames, reported_frame_count=5)
    assert cap.reported_frame_count > len(frames)

    batches = list(
        mod._iter_opencv_rgb_batches(  # noqa: SLF001
            cap=cap,
            batch_size=2,
            pre_resize_dims=None,
            cv2_module=_FakeCv2,
        )
    )

    assert [indices for indices, _batch, _read, _preprocess, _frame_idx in batches] == [
        [0, 1],
        [2],
    ]

    detection_frame_indices: list[int] = []
    for indices, batch_frames, _read, _preprocess, _frame_idx in batches:
        fake_results = [object() for _frame in batch_frames]
        for batch_i, _result in enumerate(fake_results):
            detection_frame_indices.append(indices[batch_i])

    assert detection_frame_indices == [0, 1, 2]
    assert batches[-1][4] == 3
    np.testing.assert_array_equal(batches[-1][1][0][0, 0], np.array([22, 12, 2], dtype=np.uint8))


def test_detect_yolo_rejects_conflicting_resize_dims_and_imgsz() -> None:
    with pytest.raises(ValueError, match="Conflicting CLI overrides"):
        mod.detect_yolo(
            video_path="missing_video.mp4",
            model_path="missing_model.pt",
            output_zarr="missing_output.zarr",
            resize_dims=[768, 1280],
            imgsz=[640, 640],
        )
