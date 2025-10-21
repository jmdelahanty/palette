"""Lightweight U-Net architecture for eye-mask refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two-layer conv block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    """MaxPool followed by a double conv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    """Upsample (transpose conv) then concatenate skip connection and convolve."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = _pad_to_match(x, skip.shape[-2:])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


def _pad_to_match(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Symmetrically pad spatial dims of ``x`` to match ``target_hw``."""
    diff_y = target_hw[0] - x.size(-2)
    diff_x = target_hw[1] - x.size(-1)
    if diff_y == 0 and diff_x == 0:
        return x
    pad = (
        diff_x // 2,
        diff_x - diff_x // 2,
        diff_y // 2,
        diff_y - diff_y // 2,
    )
    return F.pad(x, pad)


@dataclass
class UNetConfig:
    """Configuration for :class:`UNetSmall`."""

    in_channels: int
    out_channels: int
    base_channels: int = 32


class UNetSmall(nn.Module):
    """Compact U-Net backbone"""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive integers")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.config = UNetConfig(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            base_channels=int(base_channels),
        )

        b = self.config.base_channels
        self.inc = _DoubleConv(self.config.in_channels, b)
        self.down1 = _Down(b, b * 2)
        self.down2 = _Down(b * 2, b * 4)
        self.down3 = _Down(b * 4, b * 8)
        self.down4 = _Down(b * 8, b * 16)

        self.bottleneck = _DoubleConv(b * 16, b * 16)

        self.up1 = _Up(b * 16, b * 8)
        self.up2 = _Up(b * 8, b * 4)
        self.up3 = _Up(b * 4, b * 2)
        self.up4 = _Up(b * 2, b)

        self.out = nn.Conv2d(b, self.config.out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


__all__ = ["UNetSmall", "UNetConfig"]
