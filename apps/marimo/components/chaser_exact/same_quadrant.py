"""Persisted same-quadrant occupancy for exact chaser successors."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .array_requirements import SAME_QUADRANT_ARRAYS
from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import plain

SAME_QUADRANT_DISPLAY_RECIPE = "paired_provider_persisted_same_quadrant_denominators_v1"
_COLORS = ("#1f77b4", "#d95f02")
_Stratum = tuple[int, int, int, int]


def _array(handle: Any, name: str, *, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(handle.array(name), dtype=dtype)
    except KeyError as exc:
        raise ValueError(
            f"Same-quadrant occupancy lacks persisted array {name!r}."
        ) from exc


def _same_quadrant_rows(
    handle: Any,
) -> Mapping[_Stratum, Mapping[str, float | int]]:
    handle.require_verified_arrays(SAME_QUADRANT_ARRAYS)
    columns = {
        "epoch_role_code": _array(handle, "metric_epoch_role_code", dtype=np.int64),
        "epoch_window_id": _array(handle, "metric_epoch_window_id", dtype=np.int64),
        "behavior_role_code": _array(
            handle, "metric_behavior_role_code", dtype=np.int64
        ),
        "chaser_identity_code": _array(
            handle, "metric_chaser_identity_code", dtype=np.int64
        ),
        "candidate_frame_count": _array(
            handle, "metric_candidate_frame_count", dtype=np.int64
        ),
        "valid_distance_frame_count": _array(
            handle, "metric_valid_distance_frame_count", dtype=np.int64
        ),
        "same_quadrant_valid_frame_count": _array(
            handle, "metric_same_quadrant_valid_frame_count", dtype=np.int64
        ),
        "same_quadrant_fraction_valid": _array(
            handle, "metric_same_quadrant_fraction_valid", dtype=np.float64
        ),
        "same_quadrant_fraction_candidate": _array(
            handle, "metric_same_quadrant_fraction_candidate", dtype=np.float64
        ),
    }
    size = columns["epoch_role_code"].size
    if any(values.ndim != 1 or values.size != size for values in columns.values()):
        raise ValueError("Same-quadrant columns have inconsistent row axes.")
    rows: dict[_Stratum, Mapping[str, float | int]] = {}
    for index in range(size):
        key = (
            int(columns["epoch_role_code"][index]),
            int(columns["epoch_window_id"][index]),
            int(columns["behavior_role_code"][index]),
            int(columns["chaser_identity_code"][index]),
        )
        if key in rows:
            raise ValueError("Same-quadrant strata are duplicated.")
        candidate = int(columns["candidate_frame_count"][index])
        valid = int(columns["valid_distance_frame_count"][index])
        same = int(columns["same_quadrant_valid_frame_count"][index])
        valid_fraction = float(columns["same_quadrant_fraction_valid"][index])
        candidate_fraction = float(columns["same_quadrant_fraction_candidate"][index])
        if candidate < 0 or valid < 0 or same < 0 or same > valid or valid > candidate:
            raise ValueError("Same-quadrant support is not nested.")
        expected_valid = same / valid if valid else np.nan
        expected_candidate = same / candidate if candidate else np.nan
        for observed, expected, label in (
            (valid_fraction, expected_valid, "valid"),
            (candidate_fraction, expected_candidate, "candidate"),
        ):
            if np.isnan(expected):
                if np.isfinite(observed):
                    raise ValueError(
                        f"Same-quadrant {label} fraction has no denominator."
                    )
            elif not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Same-quadrant {label} fraction disagrees with persisted counts."
                )
        rows[key] = {
            "candidate_frame_count": candidate,
            "valid_distance_frame_count": valid,
            "same_quadrant_valid_frame_count": same,
            "same_quadrant_fraction_valid": valid_fraction,
            "same_quadrant_fraction_candidate": candidate_fraction,
        }
    if not rows:
        raise ValueError("Same-quadrant occupancy exposes no strata.")
    return rows


def _stratum_label(handle: Any, key: _Stratum) -> str:
    epochs = identity_registry(handle.scientific_manifest, "epoch_role")
    behaviors = identity_registry(handle.scientific_manifest, "behavior_role")
    chasers = identity_registry(handle.scientific_manifest, "chaser")
    return (
        f"{epochs.get(str(key[0]), f'epoch {key[0]}')}"
        f" · window {key[1]}"
        f" · {behaviors.get(str(key[2]), f'role {key[2]}')}"
        f" · {chasers.get(str(key[3]), f'chaser {key[3]}')}"
    )


def build_exact_same_quadrant_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    """Render separately denominated persisted same-quadrant fractions."""

    rows = tuple(_same_quadrant_rows(handle) for handle in projection.radials)
    if set(rows[0]) != set(rows[1]):
        raise ValueError("Paired same-quadrant products expose different strata.")
    keys = sorted(rows[0])
    labels = [_stratum_label(projection.radials[0], key) for key in keys]
    valid_figure = go.Figure()
    candidate_figure = go.Figure()
    for provider_index, (provider_id, provider_rows) in enumerate(
        zip(projection.provider_ids, rows, strict=True)
    ):
        color = _COLORS[provider_index]
        customdata = np.asarray(
            [
                [
                    provider_rows[key]["same_quadrant_valid_frame_count"],
                    provider_rows[key]["valid_distance_frame_count"],
                    provider_rows[key]["candidate_frame_count"],
                ]
                for key in keys
            ],
            dtype=np.int64,
        )
        valid_figure.add_trace(
            go.Bar(
                x=labels,
                y=[
                    100.0 * float(provider_rows[key]["same_quadrant_fraction_valid"])
                    for key in keys
                ],
                name=provider_id,
                marker_color=color,
                customdata=customdata,
                hovertemplate=(
                    "same quadrant=%{y:.4f}% of valid distance rows<br>"
                    "same rows=%{customdata[0]:,.0f}<br>"
                    "valid distance rows=%{customdata[1]:,.0f}<br>"
                    "candidate rows=%{customdata[2]:,.0f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
        candidate_figure.add_trace(
            go.Bar(
                x=labels,
                y=[
                    100.0
                    * float(provider_rows[key]["same_quadrant_fraction_candidate"])
                    for key in keys
                ],
                name=provider_id,
                marker_color=color,
                customdata=customdata,
                hovertemplate=(
                    "same quadrant=%{y:.4f}% of candidate rows<br>"
                    "same rows=%{customdata[0]:,.0f}<br>"
                    "valid distance rows=%{customdata[1]:,.0f}<br>"
                    "candidate rows=%{customdata[2]:,.0f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    display = {
        "recipe_id": SAME_QUADRANT_DISPLAY_RECIPE,
        "source_arrays": list(SAME_QUADRANT_ARRAYS),
        "valid_denominator": "metric_valid_distance_frame_count",
        "candidate_denominator": "metric_candidate_frame_count",
        "numerator": "metric_same_quadrant_valid_frame_count",
        "quadrant_geometry": "persisted_reviewed_arena_center_axes",
        "provider_strata": [
            {
                "provider_id": provider_id,
                "strata": [
                    {
                        "epoch_role_code": key[0],
                        "epoch_window_id": key[1],
                        "behavior_role_code": key[2],
                        "chaser_identity_code": key[3],
                        **{
                            name: (
                                None
                                if isinstance(value, float) and not np.isfinite(value)
                                else value
                            )
                            for name, value in provider_rows[key].items()
                        },
                    }
                    for key in keys
                ],
            }
            for provider_id, provider_rows in zip(
                projection.provider_ids, rows, strict=True
            )
        ],
        "scientific_recomputation": False,
        "inference": "none_recording_local_descriptive",
    }
    meta = plain(projection.provenance)
    meta["same_quadrant_display"] = display
    valid_figure.update_layout(
        title="Fish and chaser in the same arena quadrant · valid-distance denominator",
        barmode="group",
        yaxis_title="same-quadrant fraction of valid rows (%)",
        meta=meta,
    )
    candidate_figure.update_layout(
        title="Fish and chaser in the same arena quadrant · candidate denominator",
        barmode="group",
        yaxis_title="same-quadrant fraction of candidate rows (%)",
        meta=meta,
    )
    return mo.vstack(
        [
            mo.callout(
                "These are persisted scalar same-quadrant summaries. Valid-row "
                "and all-candidate denominators remain separate; this view does "
                "not reconstruct the full 4×4 joint quadrant matrix.",
                kind="info",
            ),
            valid_figure,
            candidate_figure,
        ]
    )


__all__ = [
    "SAME_QUADRANT_DISPLAY_RECIPE",
    "_same_quadrant_rows",
    "build_exact_same_quadrant_output",
]
