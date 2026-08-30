"""Persisted radial and near-field view for exact chaser successors."""

from __future__ import annotations

from typing import Any

import numpy as np

from .array_requirements import RADIAL_NEAR_FIELD_ARRAYS
from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import plain


def _metric_rows(handle: Any) -> dict[tuple[int, int, int], dict[str, float]]:
    epoch = np.asarray(handle.array("metric_epoch_role_code"), dtype=np.int64)
    behavior = np.asarray(handle.array("metric_behavior_role_code"), dtype=np.int64)
    chaser = np.asarray(handle.array("metric_chaser_identity_code"), dtype=np.int64)
    names = (
        "distance_p25_mm",
        "distance_p50_mm",
        "distance_p75_mm",
        "near_zone_fraction_valid",
        "near_zone_dwell_s",
        "near_zone_entry_rate_per_min_valid_time",
    )
    columns = {
        name: np.asarray(handle.array(f"metric_{name}"), dtype=np.float64)
        for name in names
    }
    if any(
        values.size != epoch.size for values in (*columns.values(), behavior, chaser)
    ):
        raise ValueError("Radial metric columns have inconsistent lengths.")
    return {
        (int(epoch[index]), int(behavior[index]), int(chaser[index])): {
            name: float(values[index]) for name, values in columns.items()
        }
        for index in range(epoch.size)
    }


def _stratum_label(handle: Any, key: tuple[int, int, int]) -> str:
    epochs = identity_registry(handle.scientific_manifest, "epoch_role")
    behaviors = identity_registry(handle.scientific_manifest, "behavior_role")
    return (
        f"{epochs.get(str(key[0]), f'epoch {key[0]}')} · "
        f"{behaviors.get(str(key[1]), f'role {key[1]}')} · chaser {key[2]}"
    )


def build_exact_radial_near_field_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    """Render sealed paired-provider radial and exact-time near-field summaries."""

    handles = projection.radials
    for handle in handles:
        handle.require_verified_arrays(RADIAL_NEAR_FIELD_ARRAYS)
    rows = tuple(_metric_rows(handle) for handle in handles)
    if set(rows[0]) != set(rows[1]) or not rows[0]:
        raise ValueError("Paired radial products expose different or empty strata.")
    keys = sorted(rows[0])
    labels = [_stratum_label(handles[0], key) for key in keys]
    colors = ("#1f77b4", "#d95f02")

    distance = go.Figure()
    near = go.Figure()
    visits = go.Figure()
    radial = go.Figure()
    for provider_index, (provider_id, provider_rows, handle) in enumerate(
        zip(projection.provider_ids, rows, handles, strict=True)
    ):
        median = np.asarray([provider_rows[key]["distance_p50_mm"] for key in keys])
        low = np.asarray([provider_rows[key]["distance_p25_mm"] for key in keys])
        high = np.asarray([provider_rows[key]["distance_p75_mm"] for key in keys])
        distance.add_trace(
            go.Scatter(
                x=labels,
                y=median,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": high - median,
                    "arrayminus": median - low,
                },
                mode="markers",
                name=provider_id,
                marker={"color": colors[provider_index], "size": 9},
            )
        )
        near.add_trace(
            go.Bar(
                x=labels,
                y=[provider_rows[key]["near_zone_fraction_valid"] for key in keys],
                name=f"{provider_id} · fraction",
                marker_color=colors[provider_index],
            )
        )
        visits.add_trace(
            go.Bar(
                x=labels,
                y=[provider_rows[key]["near_zone_dwell_s"] for key in keys],
                name=f"{provider_id} · dwell",
                marker_color=colors[provider_index],
            )
        )
        visits.add_trace(
            go.Scatter(
                x=labels,
                y=[
                    provider_rows[key]["near_zone_entry_rate_per_min_valid_time"]
                    for key in keys
                ],
                name=f"{provider_id} · entries/min",
                mode="lines+markers",
                line={
                    "color": colors[provider_index],
                    "dash": ("solid", "dash")[provider_index],
                },
                yaxis="y2",
            )
        )
        epoch = np.asarray(handle.array("radial_epoch_role_code"), dtype=np.int64)
        behavior = np.asarray(handle.array("radial_behavior_role_code"), dtype=np.int64)
        chaser = np.asarray(handle.array("radial_chaser_identity_code"), dtype=np.int64)
        start = np.asarray(handle.array("radial_bin_start_mm"), dtype=np.float64)
        end = np.asarray(handle.array("radial_bin_end_mm"), dtype=np.float64)
        selection = np.asarray(
            handle.array("radial_selection_index_geometric"), dtype=np.float64
        )
        for key, label in zip(keys, labels, strict=True):
            mask = (epoch == key[0]) & (behavior == key[1]) & (chaser == key[2])
            order = np.argsort(start[mask])
            radial.add_trace(
                go.Scatter(
                    x=((start[mask][order] + end[mask][order]) / 2.0),
                    y=selection[mask][order],
                    mode="lines",
                    name=f"{label} · {provider_id}",
                    line={
                        "color": colors[provider_index],
                        "dash": ("solid", "dash")[provider_index],
                    },
                )
            )
    figure_meta = plain(projection.provenance)
    distance.update_layout(
        title="Fish–chaser distance median and IQR",
        yaxis_title="distance (mm)",
        meta=figure_meta,
    )
    radial.update_layout(
        title="Area-corrected moving-chaser radial selection",
        xaxis_title="distance (mm)",
        yaxis_title="geometric selection index",
        meta=figure_meta,
    )
    near.update_layout(
        title="Near-field occupancy",
        barmode="group",
        yaxis={"title": "fraction of valid rows"},
        meta=figure_meta,
    )
    visits.update_layout(
        title="Exact-session-time near-field dwell and entry rate",
        barmode="group",
        yaxis={"title": "dwell (s)"},
        yaxis2={
            "title": "entries/min valid time",
            "overlaying": "y",
            "side": "right",
        },
        meta=figure_meta,
    )
    return mo.vstack(
        [
            mo.callout(
                "Persisted paired-provider summaries; exact session time, reviewed moving-reference geometry, no interpolation.",
                kind="info",
            ),
            distance,
            radial,
            near,
            visits,
        ]
    )


__all__ = ["build_exact_radial_near_field_output"]
