from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.audit_arena_geometry_detection_gates import (
    Circle,
    _decode_one_frame_from_preceding_keyframe,
    classify_gate_results,
    select_boundary_sentinel_rows,
    select_review_rows,
    signed_circle_distance,
)


class _Packet:
    def __init__(self, frame: int, *, key: bool = False) -> None:
        self.bsl = 10
        self.key = int(key)
        self.pts = frame


class _Demuxer:
    def __init__(self, packets: list[_Packet]) -> None:
        self.packets = iter(packets)
        self.seek_target: int | None = None

    def TimestampFromFrame(self, frame: int) -> int:  # noqa: N802
        return frame

    def Seek(self, timestamp: int) -> None:  # noqa: N802
        self.seek_target = timestamp

    def Demux(self) -> _Packet:  # noqa: N802
        return next(self.packets)

    def isSeekDone(self, pts: int, target: int) -> int:  # noqa: N802
        return -1 if pts < target else 0 if pts == target else 1


class _Decoder:
    def Decode(self, packet: _Packet) -> list[np.ndarray]:  # noqa: N802
        return [np.full((2, 2), packet.pts, dtype=np.int64)]


def test_signed_circle_distance_is_positive_inside_and_inclusive_at_boundary() -> None:
    circle = Circle(center_x_px=10.0, center_y_px=20.0, radius_px=5.0)
    centers = np.asarray(
        [
            [10.0, 20.0],
            [13.0, 24.0],
            [16.0, 20.0],
        ]
    )

    distance = signed_circle_distance(centers, circle)

    np.testing.assert_allclose(distance, [5.0, 0.0, -1.0], atol=0.0, rtol=0.0)
    categories = classify_gate_results(distance, np.asarray([1.0, -1.0, 1.0]))
    assert categories.tolist() == [
        "both_inside",
        "palette_only",
        "acquisition_only",
    ]


def test_classify_gate_results_covers_all_four_categories() -> None:
    categories = classify_gate_results(
        np.asarray([1.0, -1.0, 0.0, -0.1]),
        np.asarray([1.0, -1.0, -0.1, 0.0]),
    )

    assert categories.tolist() == [
        "both_inside",
        "both_outside",
        "palette_only",
        "acquisition_only",
    ]


def test_review_selection_uses_temporal_quantiles_per_disagreement_class() -> None:
    categories = np.asarray(
        [
            "palette_only",
            "both_inside",
            "palette_only",
            "acquisition_only",
            "palette_only",
            "acquisition_only",
            "palette_only",
            "acquisition_only",
        ]
    )
    frames = np.asarray([40, 10, 10, 70, 30, 20, 20, 50])

    selected = select_review_rows(
        categories=categories,
        frame_indices=frames,
        max_per_category=2,
    )

    assert selected.tolist() == [2, 5, 0, 3]


def test_gate_helpers_fail_closed_on_invalid_shapes_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        signed_circle_distance(np.asarray([1.0, 2.0]), Circle(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        classify_gate_results(np.asarray([np.nan]), np.asarray([0.0]))
    with pytest.raises(ValueError, match="positive"):
        select_review_rows(
            categories=np.asarray(["palette_only"]),
            frame_indices=np.asarray([0]),
            max_per_category=0,
        )


def test_boundary_sentinels_choose_nearest_row_in_each_temporal_partition() -> None:
    selected = select_boundary_sentinel_rows(
        frame_indices=np.asarray([0, 10, 20, 30, 40, 50]),
        palette_signed_distance_px=np.asarray([9.0, 2.0, 8.0, 4.0, 1.0, 3.0]),
        acquisition_signed_distance_px=np.asarray([8.0, 3.0, 7.0, 5.0, 2.0, 4.0]),
        max_rows=3,
    )

    assert selected.tolist() == [1, 3, 4]


def test_keyframe_seek_decodes_only_through_exact_requested_frame() -> None:
    demuxer = _Demuxer(
        [_Packet(100, key=True), _Packet(101), _Packet(102), _Packet(103)]
    )

    frame, proof = _decode_one_frame_from_preceding_keyframe(
        demuxer=demuxer,
        decoder=_Decoder(),
        target_frame_index=102,
        materialize_frame=np.asarray,
    )

    assert demuxer.seek_target == 102
    np.testing.assert_array_equal(frame, np.full((2, 2), 102))
    assert proof["packets_submitted_through_target_output"] == 3
    assert proof["keyframe_packet_pts"] == 100
    assert proof["target_packet_pts"] == 102


def test_keyframe_seek_maps_decoder_latency_through_ordered_packet_queue() -> None:
    class OnePacketLatencyDecoder:
        def __init__(self) -> None:
            self.previous: _Packet | None = None

        def Decode(self, packet: _Packet) -> list[np.ndarray]:  # noqa: N802
            previous = self.previous
            self.previous = packet
            if previous is None:
                return []
            return [np.full((2, 2), previous.pts, dtype=np.int64)]

    frame, proof = _decode_one_frame_from_preceding_keyframe(
        demuxer=_Demuxer(
            [
                _Packet(100, key=True),
                _Packet(101),
                _Packet(102),
                _Packet(103),
            ]
        ),
        decoder=OnePacketLatencyDecoder(),
        target_frame_index=102,
        materialize_frame=np.asarray,
    )

    np.testing.assert_array_equal(frame, np.full((2, 2), 102))
    assert proof["target_packet_number"] == 3
    assert proof["packets_submitted_through_target_output"] == 4
    assert proof["packets_after_target_for_decoder_latency"] == 1


def test_keyframe_seek_rejects_non_keyframe_or_reordered_packet_pts() -> None:
    with pytest.raises(RuntimeError, match="did not land on a keyframe"):
        _decode_one_frame_from_preceding_keyframe(
            demuxer=_Demuxer([_Packet(100, key=False)]),
            decoder=_Decoder(),
            target_frame_index=100,
            materialize_frame=np.asarray,
        )

    with pytest.raises(RuntimeError, match="nonmonotonic packet PTS"):
        _decode_one_frame_from_preceding_keyframe(
            demuxer=_Demuxer([_Packet(100, key=True), _Packet(102), _Packet(101)]),
            decoder=_Decoder(),
            target_frame_index=103,
            materialize_frame=np.asarray,
        )
