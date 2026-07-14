"""Shared UI helpers for Palette marimo explorer components."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np


def normalize_path(path: str) -> str:
    return "/".join(part for part in str(path).strip("/").split("/") if part)


def join_path(*parts: str) -> str:
    return "/".join(normalize_path(part) for part in parts if normalize_path(part))


def png_bytes_to_markdown_image(mo: Any, png_bytes: bytes, *, alt_text: str) -> Any:
    if not png_bytes:
        return mo.md(f"*No PNG bytes available for `{alt_text}`.*")
    encoded = base64.b64encode(bytes(png_bytes)).decode("ascii")
    return mo.md(
        f'<img alt="{alt_text}" src="data:image/png;base64,{encoded}" '
        'style="max-width:100%; height:auto; border: 1px solid #ddd;" />'
    )


def add_epoch_overlays(fig: Any, windows_df: Any) -> None:
    if not len(windows_df):
        return
    if hasattr(windows_df, "iter_rows"):
        rows = windows_df.iter_rows(named=True)
    elif hasattr(windows_df, "to_dict"):
        rows = windows_df.to_dict("records")
    else:
        rows = windows_df
    for row in rows:
        fig.add_vrect(
            x0=float(row["start_time_s"]),
            x1=float(row["end_time_s"]),
            fillcolor="rgba(80, 120, 180, 0.08)",
            line_width=0,
            layer="below",
        )


def apply_full_width_timeseries_layout(
    fig: Any,
    *,
    title: str,
    yaxis_title: str,
    height: int = 430,
) -> None:
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=int(height),
        margin=dict(l=56, r=20, t=56, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
    )


def finite_time_extent(values: Any) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return 0.0, 0.0
    finite = array[np.isfinite(array)]
    if not finite.size:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())
