"""Exact PyNvVideoCodec frame recovery from a preceding GOP keyframe."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np


def decode_one_frame_from_preceding_keyframe(
    *,
    demuxer: Any,
    decoder: Any,
    target_frame_index: int,
    materialize_frame: Callable[[Any], np.ndarray],
    max_packets_per_seek: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Seek to a GOP keyframe and prove the exact requested display frame.

    The proof relies on the demuxer's target relation, strictly increasing
    packet presentation timestamps, and an ordered pending-packet/display-frame
    queue.  This accounts for decoder startup latency without treating the first
    returned surface as the requested frame.
    """

    target = int(target_frame_index)
    if target < 0:
        raise ValueError("target_frame_index must be nonnegative")
    if int(max_packets_per_seek) <= 0:
        raise ValueError("max_packets_per_seek must be positive")

    seek_timestamp = int(demuxer.TimestampFromFrame(target))
    demuxer.Seek(seek_timestamp)
    keyframe_packet_pts: int | None = None
    previous_packet_pts: int | None = None
    target_packet_number: int | None = None
    pending: deque[tuple[int, int]] = deque()
    for packet_count in range(1, int(max_packets_per_seek) + 1):
        packet = demuxer.Demux()
        if int(packet.bsl) <= 0:
            raise RuntimeError("PyNvVideoCodec seek reached an empty packet")
        packet_pts = int(packet.pts)
        if previous_packet_pts is not None and packet_pts <= previous_packet_pts:
            raise RuntimeError(
                "Exact-frame seek does not support reordered/nonmonotonic packet PTS"
            )
        previous_packet_pts = packet_pts
        if packet_count == 1:
            if int(packet.key) != 1:
                raise RuntimeError("PyNvVideoCodec seek did not land on a keyframe")
            keyframe_packet_pts = packet_pts
        relation = int(demuxer.isSeekDone(packet_pts, target))
        if relation not in {-1, 0, 1}:
            raise RuntimeError(
                f"PyNvVideoCodec returned invalid seek relation {relation}"
            )
        if relation == 0:
            if target_packet_number is not None:
                raise RuntimeError("PyNvVideoCodec reported the target packet twice")
            target_packet_number = packet_count
        elif relation > 0 and target_packet_number is None:
            raise RuntimeError(f"PyNvVideoCodec seek passed target frame {target}")
        pending.append((relation, packet_pts))
        decoded = list(decoder.Decode(packet))
        if len(decoded) > len(pending):
            raise RuntimeError(
                "PyNvVideoCodec produced more display frames than submitted packets"
            )
        for frame in decoded:
            output_relation, output_pts = pending.popleft()
            if output_relation == 0:
                if target_packet_number is None:
                    raise RuntimeError("target display frame preceded its packet")
                return materialize_frame(frame), {
                    "target_frame_index": target,
                    "seek_timestamp": seek_timestamp,
                    "keyframe_packet_pts": keyframe_packet_pts,
                    "target_packet_pts": output_pts,
                    "target_packet_number": target_packet_number,
                    "packets_submitted_through_target_output": packet_count,
                    "packets_after_target_for_decoder_latency": (
                        packet_count - target_packet_number
                    ),
                    "exact_frame_proof": (
                        "demuxer_isSeekDone_exact_monotonic_pts_ordered_display_queue"
                    ),
                }
    raise RuntimeError(
        f"PyNvVideoCodec seek exceeded {max_packets_per_seek} packets for frame {target}"
    )


__all__ = ["decode_one_frame_from_preceding_keyframe"]
