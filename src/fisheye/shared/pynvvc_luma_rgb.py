"""PyNvVideoCodec luma-to-RGB tensor helpers for high-resolution detection."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


BACKEND_PYNVVC_LUMA_RGB = "pynvvc_luma_rgb"


class PynvvcLumaRgbReader:
    """Sequential PyNvVideoCodec reader that emits raw NV12 CUDA tensors."""

    def __init__(self, video_path: Path, *, start_frame: int = 0, gpu_id: int = 0) -> None:
        if start_frame != 0:
            raise ValueError(
                f"{BACKEND_PYNVVC_LUMA_RGB} is currently sequential-only; "
                "start_frame must be 0."
            )
        try:
            import PyNvVideoCodec as nvc  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"PyNvVideoCodec import failed; cannot use {BACKEND_PYNVVC_LUMA_RGB}: {exc}"
            ) from exc

        self.nvc = nvc
        self.demuxer = nvc.CreateDemuxer(filename=str(video_path))
        self.decoder = nvc.CreateDecoder(
            gpuid=int(gpu_id),
            codec=self.demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        self.packet_iter = iter(self.demuxer)
        self.source_height = int(self.demuxer.Height())
        self.source_width = int(self.demuxer.Width())
        self.codec = str(self.demuxer.GetNvCodecId())
        self.frame_rate = float(self.demuxer.FrameRate())
        self._eof = False
        self._pending_frames: list[torch.Tensor] = []

    def decode_next(self, count: int) -> list[torch.Tensor]:
        frames: list[torch.Tensor] = []
        if self._pending_frames:
            take = min(count, len(self._pending_frames))
            frames.extend(self._pending_frames[:take])
            self._pending_frames = self._pending_frames[take:]
        while len(frames) < count and not self._eof:
            try:
                packet = next(self.packet_iter)
            except StopIteration:
                self._eof = True
                break
            for frame in self.decoder.Decode(packet):
                tensor = torch.from_dlpack(frame)
                if len(frames) < count:
                    frames.append(tensor)
                else:
                    self._pending_frames.append(tensor)
        return frames

    def close(self) -> None:
        self._pending_frames = []
        self.packet_iter = iter(())
        if hasattr(self, "decoder"):
            del self.decoder
        if hasattr(self, "demuxer"):
            del self.demuxer


def preprocess_luma_rgb(
    raw_frames: Sequence[torch.Tensor],
    *,
    source_height: int,
    device: torch.device,
    dtype: torch.dtype,
    resize_hw: tuple[int, int] | list[int] | None,
) -> torch.Tensor:
    """Convert raw NV12 luma planes into normalized RGB tensors.

    `resize_hw` is canonical Palette order: `(height, width)`.
    """

    if resize_hw is None:
        raise ValueError(f"{BACKEND_PYNVVC_LUMA_RGB} requires a resolved resize.")

    height, width = int(resize_hw[0]), int(resize_hw[1])
    y_planes = [frame[:source_height, :].contiguous() for frame in raw_frames]
    luma = torch.stack(y_planes, dim=0).unsqueeze(1).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
    resized = F.interpolate(
        luma,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    rgb = resized.expand(-1, 3, -1, -1).mul(1.0 / 255.0)
    return rgb.contiguous(memory_format=torch.channels_last)
