from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis.chaser_epoch_behavior_summary import _make_per_epoch_chaser
from fisheye.analysis.chaser_escape_freeze_summary import _chaser_column
from fisheye.analysis.chaser_near_field_occupancy import (
    ChaserNearFieldIdentity,
    ChaserNearFieldPhase,
    _build_summary as build_near_field_summary,
)
from fisheye.analysis.chaser_quadrant_occupancy import (
    ChaserQuadrantPhase,
    _build_summary as build_quadrant_summary,
    _read_chaser_roles_from_distance_run,
    resolve_chaser_roles_from_protocol_payload,
)


class _Array:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    def __getitem__(self, key: object) -> np.ndarray:
        return self.values[key]


class _Group(dict[str, object]):
    def __init__(
        self, *args: object, attrs: dict[str, object] | None = None, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}


def _payload(chaser_count: int) -> dict[str, object]:
    return {
        "steps": [
            {
                "parameters": {
                    "chasers": [
                        {
                            "chaser_index": index,
                            "enable_chase": index == 0,
                            "enable_random_movement": index == 1,
                            "color_r": float(index == 0),
                            "color_g": float(index == 1),
                            "color_b": float(index >= 2),
                        }
                        for index in range(chaser_count)
                    ]
                }
            }
        ]
    }


def _windows() -> tuple[ChaserDistanceWindow, ...]:
    return (
        ChaserDistanceWindow(0, "pre_event", 0, 4, 0.0, 0.5, 0.5),
        ChaserDistanceWindow(1, "post_event", 5, 9, 0.5, 1.0, 0.5),
    )


@pytest.mark.parametrize("chaser_count", [1, 2, 3])
def test_quadrant_summary_preserves_every_chaser(chaser_count: int) -> None:
    chasers = resolve_chaser_roles_from_protocol_payload(_payload(chaser_count))
    phases = (
        ChaserQuadrantPhase(0, "pre_static", "pre_event", 0, 4, 0, 4, 0),
        ChaserQuadrantPhase(1, "post_static", "post_event", 5, 9, 5, 9, 0),
    )
    shape = (len(phases), chaser_count)
    summary = build_quadrant_summary(
        recording_id="recording",
        chasers=chasers,
        phases=phases,
        arrays={
            "median_distance_mm": np.arange(np.prod(shape), dtype=np.float32).reshape(
                shape
            ),
            "occupancy_fraction": np.full(shape, 0.25, dtype=np.float32),
            "tracking_dropout_fraction": np.zeros(shape, dtype=np.float32),
            "valid_frame_count": np.full(shape, 5, dtype=np.int64),
            "chaser_quadrant_code": np.zeros(shape, dtype=np.int8),
        },
    )

    assert summary["chaser_count"] == chaser_count
    assert len(summary["per_chaser"]) == chaser_count
    assert sum(row["chaser_count"] for row in summary["per_role"]) == chaser_count


@pytest.mark.parametrize("chaser_count", [1, 2, 3])
def test_near_field_summary_preserves_every_chaser(chaser_count: int) -> None:
    roles = resolve_chaser_roles_from_protocol_payload(_payload(chaser_count))
    chasers = tuple(
        ChaserNearFieldIdentity(
            chaser_index=role.chaser_index,
            behavior_class_id=role.behavior_class_id,
            behavior_class=role.behavior_class,
            raw_color_rgba=role.raw_color_rgba,
            raw_color_hex=role.raw_color_hex,
        )
        for role in roles
    )
    phases = (
        ChaserNearFieldPhase(0, "pre_static", 0, 4, 5),
        ChaserNearFieldPhase(1, "post_static", 5, 9, 5),
    )
    shape = (len(phases), chaser_count)
    percentiles = np.asarray([5.0, 10.0], dtype=np.float32)
    summary = build_near_field_summary(
        recording_id="recording",
        chasers=chasers,
        phases=phases,
        percentile_values=percentiles,
        approach_percentile_mm=np.ones((*shape, percentiles.size), dtype=np.float32),
        approach_percentile_cdf_fraction=np.broadcast_to(
            percentiles.reshape(1, 1, -1) / 100.0,
            (*shape, percentiles.size),
        ).copy(),
        near_zone_occupancy_fraction=np.full(shape, 0.25, dtype=np.float32),
        near_zone_entry_rate_per_min=np.ones(shape, dtype=np.float32),
        tracking_dropout_fraction=np.zeros(shape, dtype=np.float32),
        thigmotaxis_fraction=np.zeros(len(phases), dtype=np.float32),
        mean_speed_mm_s=np.ones(len(phases), dtype=np.float32),
        immobile_fraction=np.zeros(len(phases), dtype=np.float32),
        valid_distance_count=np.full(shape, 5, dtype=np.int64),
    )

    assert summary["chaser_count"] == chaser_count
    assert len(summary["per_chaser"]) == chaser_count
    assert sum(row["chaser_count"] for row in summary["per_role"]) == chaser_count


@pytest.mark.parametrize("chaser_count", [1, 2, 3])
def test_epoch_summary_emits_window_by_chaser_rows(chaser_count: int) -> None:
    shape = (len(_windows()), chaser_count)
    chasers = _Group(
        {
            "chaser_index": _Array(np.arange(chaser_count, dtype=np.int16)),
            "behavior_class_id": _Array(np.arange(chaser_count, dtype=np.int8) + 1),
            "behavior_class_label_bytes": _Array(
                np.asarray(
                    [b"aggressive", b"random_non_chasing", b"inert"][:chaser_count]
                )
            ),
        }
    )
    epoch_summary = _Group(
        {
            "valid_frame_count": _Array(np.full(shape, 5, dtype=np.int64)),
            "mean_distance_mm": _Array(np.ones(shape, dtype=np.float32)),
            "min_distance_mm": _Array(np.ones(shape, dtype=np.float32)),
            "p05_distance_mm": _Array(np.ones(shape, dtype=np.float32)),
            "p50_distance_mm": _Array(np.ones(shape, dtype=np.float32)),
            "p95_distance_mm": _Array(np.ones(shape, dtype=np.float32)),
            "fraction_within_threshold": _Array(np.ones(shape, dtype=np.float32)),
        },
        attrs={"threshold_mm": 5.0},
    )
    records = _make_per_epoch_chaser(
        windows=_windows(),
        run_group=_Group({"chasers": chasers, "epoch_summary": epoch_summary}),  # type: ignore[arg-type]
    )

    assert records.shape == (len(_windows()) * chaser_count,)
    assert sorted(set(records["chaser_index"].tolist())) == list(range(chaser_count))


@pytest.mark.parametrize("chaser_count", [1, 2, 3])
def test_escape_summary_can_select_any_persisted_chaser(chaser_count: int) -> None:
    indices = _Array(np.arange(chaser_count, dtype=np.int16))
    group = _Group(
        {
            "chasers": _Group({"chaser_index": indices}),
            "chasers/chaser_index": indices,
        }
    )

    assert _chaser_column(group, chaser_count - 1) == chaser_count - 1  # type: ignore[arg-type]


def test_quadrant_v1_refuses_time_varying_role_intervals() -> None:
    run = _Group(
        {
            "chasers": _Group(
                {
                    "chaser_index": _Array([0]),
                    "behavior_class_id": _Array([1]),
                    "behavior_class_label_bytes": _Array([b"aggressive"]),
                    "raw_color_rgba": _Array([[1.0, 0.0, 0.0, 1.0]]),
                }
            ),
            "chaser_role_intervals": _Group(
                {
                    "chaser_index": _Array([0, 0]),
                    "behavior_class_id": _Array([3, 1]),
                    "start_frame": _Array([0, 50]),
                    "end_frame": _Array([49, 99]),
                }
            ),
        },
        attrs={"total_frames": 100},
    )

    with pytest.raises(ValueError, match="stable whole-recording role interval"):
        _read_chaser_roles_from_distance_run(  # type: ignore[arg-type]
            run,
            protocol_payload=_payload(1),
        )
