"""Orange frame-identity names across the frame_identity_v1 adoption.

agent-contracts citrus-orange/frame_identity_v1_2026-10-12.md (PR #87): one
name per counter. ``recording_frame_id``, ``stream_frame_id``,
``camera_frame_id`` and ``stimulus_frame_num`` are the only frame-counter
suffixes; positions keep ``*_index`` names. Orange adopts with no aliases
(Orange 0c821bb). Recordings made before adoption keep their old names and are
never rewritten (section 8), so every reader here accepts exactly one of the
old or the new name for a field and refuses a file that carries both.

Renames Palette reads (section 6):

- crop metadata ``local_frame_id`` -> ``stream_frame_id`` (same counter);
- recorder metadata ``frame_index_within_gop`` -> ``position_in_gop``;
- recorder keyframes ``keyframe_frames`` -> ``keyframe_video_frame_indices``.

Persisted Palette array names (for example ``source_crop_local_frame_ids``)
keep their names; the values are the same counter either way.
"""

from __future__ import annotations

import re
from typing import Any, Collection, Mapping

FRAME_COUNTER_SUFFIXES = (
    "recording_frame_id",
    "stream_frame_id",
    "camera_frame_id",
    "stimulus_frame_num",
)
FRAME_IDENTITY_CONTRACT = "agent-contracts citrus-orange/frame_identity_v1"
# A name that refers to a frame: ends in frame_id / frame_num / frame_ids, or
# is exactly "frame". Positions (``*_index``) are not frame references.
_FRAME_REFERENCE = re.compile(r"(^|_)frame_(id|ids|num)$|^frame$")


class FrameIdentityError(ValueError):
    """A file mixes old and new frame-identity names or breaks the suffix rule."""


def _one_of(fields: Collection[str], new: str, old: str, *, label: str) -> str | None:
    has_new, has_old = new in fields, old in fields
    if has_new and has_old:
        raise FrameIdentityError(
            f"{label} carries both {new!r} and its retired name {old!r}; "
            f"{FRAME_IDENTITY_CONTRACT} allows no aliases."
        )
    if has_new:
        return new
    if has_old:
        return old
    return None


def require_frame_identity_names(fields: Collection[str], *, label: str) -> None:
    """Refuse a frame reference without one of the four counter suffixes.

    Applied only to a file that has adopted the contract (it already uses a
    new name); older files keep their names.
    """

    bad = sorted(
        name
        for name in fields
        if _FRAME_REFERENCE.search(name)
        and not any(name.endswith(suffix) for suffix in FRAME_COUNTER_SUFFIXES)
    )
    if bad:
        raise FrameIdentityError(
            f"{label} adopts {FRAME_IDENTITY_CONTRACT} but has frame references "
            f"without a counter suffix: {', '.join(bad)}."
        )


def crop_meta_stream_frame_field(fields: Collection[str], *, label: str) -> str | None:
    """The crop metadata column holding Orange's stream counter, if any.

    ``stream_frame_id`` (adopted; the file must then follow the suffix rule)
    or ``local_frame_id`` (before adoption). None when neither is present.
    """

    field = _one_of(fields, "stream_frame_id", "local_frame_id", label=label)
    if field == "stream_frame_id":
        require_frame_identity_names(fields, label=label)
    return field


def gop_position_field(fields: Collection[str], *, label: str) -> str | None:
    """``position_in_gop`` (adopted) or ``frame_index_within_gop`` (before)."""

    field = _one_of(fields, "position_in_gop", "frame_index_within_gop", label=label)
    if field == "position_in_gop":
        require_frame_identity_names(fields, label=label)
    return field


def keyframe_video_frame_indices(payload: Mapping[str, Any], *, label: str) -> Any:
    """The keyframe list: ``keyframe_video_frame_indices`` (adopted) or
    ``keyframe_frames`` (before). Raises when neither or both are present."""

    field = _one_of(
        payload, "keyframe_video_frame_indices", "keyframe_frames", label=label
    )
    if field is None:
        raise FrameIdentityError(
            f"{label} has neither keyframe_video_frame_indices nor keyframe_frames."
        )
    return payload[field]


def keyframe_field_name(payload: Mapping[str, Any], *, label: str) -> str:
    """The name the payload uses for its keyframe list (for writers that must
    keep the source's own name when they rewrite it)."""

    keyframe_video_frame_indices(payload, label=label)
    return (
        "keyframe_video_frame_indices"
        if "keyframe_video_frame_indices" in payload
        else "keyframe_frames"
    )


__all__ = [
    "FRAME_COUNTER_SUFFIXES",
    "FRAME_IDENTITY_CONTRACT",
    "FrameIdentityError",
    "crop_meta_stream_frame_field",
    "gop_position_field",
    "keyframe_field_name",
    "keyframe_video_frame_indices",
    "require_frame_identity_names",
]
