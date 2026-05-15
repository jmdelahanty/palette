"""PyNvVideoCodec luma-to-RGB tensor helpers for high-resolution detection."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


BACKEND_PYNVVC_LUMA_RGB = "pynvvc_luma_rgb"
BACKEND_PYNVVC_NV12_RGB = "pynvvc_nv12_rgb"
PYNVVC_BACKENDS = (BACKEND_PYNVVC_NV12_RGB, BACKEND_PYNVVC_LUMA_RGB)


class PynvvcLumaRgbReader:
    """Sequential PyNvVideoCodec reader that emits raw NV12 CUDA tensors."""

    def __init__(self, video_path: Path, *, start_frame: int = 0, gpu_id: int = 0) -> None:
        if start_frame != 0:
            raise ValueError(
                "PyNvVideoCodec sequential reader is currently start-at-zero only; "
                "start_frame must be 0."
            )
        try:
            import PyNvVideoCodec as nvc  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"PyNvVideoCodec import failed; cannot use PyNvVideoCodec backends: {exc}"
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


def preprocess_nv12_rgb(
    raw_frames: Sequence[torch.Tensor],
    *,
    source_height: int,
    device: torch.device,
    dtype: torch.dtype,
    resize_hw: tuple[int, int] | list[int] | None,
) -> torch.Tensor:
    """Convert raw NV12 frames into normalized RGB tensors.

    The conversion uses the common BT.601 limited-range YUV-to-RGB matrix after
    resizing the Y, U, and V planes to the model input size. Resizing before
    color conversion avoids materializing full-resolution RGB frames for
    high-resolution recordings.
    """

    if resize_hw is None:
        raise ValueError(f"{BACKEND_PYNVVC_NV12_RGB} requires a resolved resize.")

    height, width = int(resize_hw[0]), int(resize_hw[1])
    y_planes = []
    u_planes = []
    v_planes = []
    for frame in raw_frames:
        y_planes.append(frame[:source_height, :].contiguous())
        uv = frame[source_height:, :].contiguous()
        u_planes.append(uv[:, 0::2])
        v_planes.append(uv[:, 1::2])

    y = torch.stack(y_planes, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
    u = torch.stack(u_planes, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)
    v = torch.stack(v_planes, dim=0).unsqueeze(1).to(device=device, dtype=torch.float32)

    y = F.interpolate(y, size=(height, width), mode="bilinear", align_corners=False)
    u = F.interpolate(u, size=(height, width), mode="bilinear", align_corners=False)
    v = F.interpolate(v, size=(height, width), mode="bilinear", align_corners=False)

    c = (y - 16.0).clamp_min(0.0)
    d = u - 128.0
    e = v - 128.0
    r = (1.16438356 * c + 1.59602678 * e).clamp(0.0, 255.0)
    g = (1.16438356 * c - 0.39176229 * d - 0.81296765 * e).clamp(0.0, 255.0)
    b = (1.16438356 * c + 2.01723214 * d).clamp(0.0, 255.0)
    rgb = torch.cat((r, g, b), dim=1).to(dtype=dtype).mul(1.0 / 255.0)
    return rgb.contiguous(memory_format=torch.channels_last)
