"""Helpers for parsing encoder metadata embedded in MP4 container tags."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


_RES_RE = re.compile(r"^(\\d+)x(\\d+)$", re.IGNORECASE)


def _coerce_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_encoder_comment(comment: Optional[str]) -> Dict[str, Any]:
    """
    Parse encoder settings stored in an MP4 comment tag.

    Expected format:
      "nvenc codec=hevc; preset=p1; tuning=ll; res=4512x4512; fps=60; ..."
    """
    if not comment:
        return {}

    parts = [part.strip() for part in comment.split(";") if part.strip()]
    raw: Dict[str, Any] = {}
    encoder_name = None
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            raw[key.strip().lower()] = value.strip()
        else:
            if encoder_name is None:
                encoder_name = part.strip()
            else:
                raw[part.strip().lower()] = True

    fields: Dict[str, Any] = {}
    if encoder_name:
        fields["encoder_name"] = encoder_name

    mapping = {
        "codec": "encoder_codec",
        "preset": "encoder_preset",
        "tuning": "encoder_tuning",
        "rc": "encoder_rc",
        "res": "encoder_res",
        "fps": "encoder_fps",
        "color": "encoder_color",
        "bpp": "encoder_bpp",
        "target_bps": "encoder_target_bps",
    }

    for key, out in mapping.items():
        if key not in raw:
            continue
        value = raw.get(key)
        if out in {"encoder_fps", "encoder_bpp"}:
            fields[out] = _coerce_float(str(value))
        elif out in {"encoder_color", "encoder_target_bps"}:
            fields[out] = _coerce_int(str(value))
        else:
            fields[out] = str(value)

    if "encoder_res" in fields:
        match = _RES_RE.match(fields["encoder_res"])
        if match:
            fields["encoder_res_width"] = _coerce_int(match.group(1))
            fields["encoder_res_height"] = _coerce_int(match.group(2))

    if raw:
        fields["encoder_params"] = raw

    return fields
