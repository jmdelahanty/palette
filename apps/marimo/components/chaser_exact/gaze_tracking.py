"""Persisted exact body-frame gaze tracking view."""

from __future__ import annotations

from typing import Any

import numpy as np

from .array_requirements import GAZE_TRACKING_ARRAYS
from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import plain

GAZE_DISPLAY_MAX_POINTS = 6_000
GAZE_ERROR_BIN_WIDTH_DEG = 5.0


def _uniform_indices(indices: np.ndarray, *, maximum: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size <= maximum:
        return values
    selected = np.linspace(0, values.size - 1, maximum, dtype=np.int64)
    return values[np.unique(selected)]


def _histogram_probability(
    values: np.ndarray, *, bin_width_deg: float = GAZE_ERROR_BIN_WIDTH_DEG
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(values, dtype=np.float64).reshape(-1)
    observed = observed[np.isfinite(observed)]
    edges = np.arange(-180.0, 180.0 + bin_width_deg, bin_width_deg)
    counts = np.histogram(observed, bins=edges)[0].astype(np.int64)
    probability = (
        counts.astype(np.float64) / int(np.sum(counts))
        if np.sum(counts)
        else np.zeros(counts.size, dtype=np.float64)
    )
    return (edges[:-1] + edges[1:]) / 2.0, probability, counts


def build_exact_gaze_tracking_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    """Render verified persisted gaze rows, summaries, and lock events."""

    handle = projection.gaze_tracking
    if handle is None:
        raise ValueError("Exact gaze-tracking projection is unavailable.")
    handle.require_verified_arrays(GAZE_TRACKING_ARRAYS)

    arrays = {name: np.asarray(handle.array(name)) for name in GAZE_TRACKING_ARRAYS}
    row_count = arrays["gaze_row_id"].size
    row_names = (
        "acquisition_frame_id",
        "semantic_role_code",
        "eye_code",
        "chaser_identity_code",
        "distance_mm",
        "bearing_deg",
        "gaze_signed_deg",
        "vergence_deg",
        "valid",
        "accessible",
        "gaze_error_deg",
        "lock_on",
    )
    if any(arrays[name].size != row_count for name in row_names):
        raise ValueError("Persisted gaze row arrays have inconsistent lengths.")
    valid = arrays["valid"].astype(bool)
    accessible = arrays["accessible"].astype(bool)
    lock = arrays["lock_on"].astype(bool)
    if np.any(lock & (~valid | ~accessible)):
        raise ValueError("Persisted lock rows are not valid accessible gaze rows.")
    if np.any(
        valid
        & (
            ~np.isfinite(arrays["bearing_deg"])
            | ~np.isfinite(arrays["gaze_signed_deg"])
            | ~np.isfinite(arrays["gaze_error_deg"])
        )
    ):
        raise ValueError("A persisted valid gaze row contains a non-finite angle.")

    roles = identity_registry(handle.scientific_manifest, "semantic_role")
    eyes = identity_registry(handle.scientific_manifest, "eye")
    colors = {1: "#2c7fb8", 2: "#d95f0e"}
    scatter = go.Figure()
    error = go.Figure()
    usable = valid & accessible
    for eye_code in (1, 2):
        eye_rows = usable & (arrays["eye_code"] == eye_code)
        indices = _uniform_indices(
            np.flatnonzero(eye_rows), maximum=GAZE_DISPLAY_MAX_POINTS
        )
        scatter.add_trace(
            go.Scattergl(
                x=arrays["bearing_deg"][indices],
                y=arrays["gaze_signed_deg"][indices],
                mode="markers",
                name=str(eyes.get(str(eye_code), f"eye {eye_code}")),
                marker={
                    "size": 4,
                    "opacity": 0.35,
                    "color": colors[eye_code],
                },
                customdata=np.column_stack(
                    (
                        arrays["acquisition_frame_id"][indices],
                        arrays["chaser_identity_code"][indices],
                        arrays["distance_mm"][indices],
                        arrays["gaze_error_deg"][indices],
                    )
                ),
                hovertemplate=(
                    "target %{x:.1f}°<br>gaze %{y:.1f}°"
                    "<br>frame %{customdata[0]:.0f}<br>chaser %{customdata[1]:.0f}"
                    "<br>distance %{customdata[2]:.2f} mm"
                    "<br>error %{customdata[3]:.1f}°<extra></extra>"
                ),
            )
        )
        centers, probability, counts = _histogram_probability(
            arrays["gaze_error_deg"][eye_rows]
        )
        error.add_trace(
            go.Bar(
                x=centers,
                y=probability,
                width=GAZE_ERROR_BIN_WIDTH_DEG,
                name=str(eyes.get(str(eye_code), f"eye {eye_code}")),
                marker_color=colors[eye_code],
                opacity=0.65,
                customdata=counts,
                hovertemplate=(
                    "error bin %{x:.1f}°<br>probability %{y:.4f}"
                    "<br>accessible rows %{customdata}<extra></extra>"
                ),
            )
        )
    scatter.add_trace(
        go.Scatter(
            x=[-180, 180],
            y=[-180, 180],
            mode="lines",
            name="identity",
            line={"color": "#444", "dash": "dash", "width": 1},
        )
    )
    scatter.update_layout(
        title="Eye gaze versus exact chaser bearing in the accepted body frame",
        xaxis_title="chaser bearing (deg; anatomical left positive)",
        yaxis_title="gaze direction (deg; anatomical left positive)",
        xaxis={"range": [-180, 180]},
        yaxis={"range": [-180, 180]},
    )
    error.update_layout(
        title="Accessible-row gaze error distributions",
        barmode="overlay",
        xaxis_title="wrapped gaze minus bearing (deg)",
        yaxis_title="probability within eye",
    )

    summary_count = arrays["summary_row_id"].size
    summary_names = tuple(
        name for name in GAZE_TRACKING_ARRAYS if name.startswith("summary_")
    )
    if any(arrays[name].size != summary_count for name in summary_names):
        raise ValueError("Persisted gaze summary arrays have inconsistent lengths.")
    labels = [
        (
            f"{roles.get(str(int(role)), f'role {int(role)}')} · "
            f"{eyes.get(str(int(eye)), f'eye {int(eye)}')} · chaser {int(chaser)}"
        )
        for role, eye, chaser in zip(
            arrays["summary_role_code"],
            arrays["summary_eye_code"],
            arrays["summary_chaser_identity_code"],
            strict=True,
        )
    ]
    summary = go.Figure()
    summary.add_trace(
        go.Bar(
            x=labels,
            y=arrays["summary_lock_fraction"],
            name="lock fraction",
            customdata=np.column_stack(
                (
                    arrays["summary_valid_sample_count"],
                    arrays["summary_accessible_sample_count"],
                    arrays["summary_lock_sample_count"],
                    arrays["summary_median_abs_error_deg"],
                )
            ),
            hovertemplate=(
                "%{x}<br>lock fraction %{y:.4f}<br>valid %{customdata[0]:.0f}"
                "<br>accessible %{customdata[1]:.0f}<br>locked %{customdata[2]:.0f}"
                "<br>median |error| %{customdata[3]:.2f}°<extra></extra>"
            ),
        )
    )
    summary.add_trace(
        go.Scatter(
            x=labels,
            y=arrays["summary_tracking_gain"],
            name="static tracking gain",
            mode="markers",
            marker={"color": "#7a5195", "size": 8},
            yaxis="y2",
            customdata=np.column_stack(
                (
                    arrays["summary_regression_sample_count"],
                    arrays["summary_tracking_correlation"],
                    arrays["summary_tracking_intercept_deg"],
                )
            ),
            hovertemplate=(
                "%{x}<br>static gain %{y:.4f}"
                "<br>samples %{customdata[0]:.0f}"
                "<br>correlation %{customdata[1]:.4f}"
                "<br>intercept %{customdata[2]:.2f}°<extra></extra>"
            ),
        )
    )
    summary.update_layout(
        title="Persisted lock fraction and static gaze-tracking gain",
        yaxis={"title": "lock fraction", "range": [0, 1]},
        yaxis2={
            "title": "static tracking gain",
            "overlaying": "y",
            "side": "right",
        },
    )
    dynamic = go.Figure()
    dynamic.add_trace(
        go.Scatter(
            x=labels,
            y=arrays["summary_dynamic_zero_lag_gain"],
            name="zero-lag dynamic gain",
            mode="markers",
            marker={"color": "#2b8cbe", "size": 8},
            customdata=np.column_stack(
                (
                    arrays["summary_dynamic_zero_lag_correlation"],
                    arrays["summary_dynamic_zero_lag_sample_count"],
                )
            ),
            hovertemplate=(
                "%{x}<br>zero-lag gain %{y:.4f}"
                "<br>correlation %{customdata[0]:.4f}"
                "<br>samples %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    dynamic.add_trace(
        go.Scatter(
            x=labels,
            y=arrays["summary_dynamic_best_lag_gain"],
            name="causal best-lag dynamic gain",
            mode="markers",
            marker={"color": "#e34a33", "size": 8},
            customdata=np.column_stack(
                (
                    arrays["summary_dynamic_best_lag_seconds"],
                    arrays["summary_dynamic_best_lag_frames"],
                    arrays["summary_dynamic_best_lag_correlation"],
                    arrays["summary_dynamic_best_lag_sample_count"],
                )
            ),
            hovertemplate=(
                "%{x}<br>gain %{y:.4f}<br>lag %{customdata[0]:.4f} s"
                " (%{customdata[1]:.0f} frames)<br>correlation %{customdata[2]:.4f}"
                "<br>samples %{customdata[3]:.0f}<extra></extra>"
            ),
        )
    )
    dynamic.update_layout(
        title="Persisted dynamic gaze compensation",
        yaxis_title="wrapped Δgaze / Δbearing gain",
    )

    control_count = arrays["control_summary_row_id"].size
    control_names = tuple(
        name for name in GAZE_TRACKING_ARRAYS if name.startswith("control_")
    )
    if control_count != summary_count or any(
        arrays[name].size != control_count for name in control_names
    ):
        raise ValueError(
            "Persisted real-versus-rotated control arrays are inconsistent."
        )
    if not (
        np.array_equal(arrays["control_role_code"], arrays["summary_role_code"])
        and np.array_equal(arrays["control_eye_code"], arrays["summary_eye_code"])
        and np.array_equal(
            arrays["control_chaser_identity_code"],
            arrays["summary_chaser_identity_code"],
        )
    ):
        raise ValueError("Persisted control rows do not align with real summary rows.")
    controls = go.Figure()
    for name, count_name, label, color in (
        (
            "control_tracking_gain_excess_vs_virtual",
            "control_tracking_gain_virtual_valid_count",
            "static gain: real − rotated mean",
            "#756bb1",
        ),
        (
            "control_dynamic_zero_lag_gain_excess_vs_virtual",
            "control_dynamic_zero_lag_gain_virtual_valid_count",
            "zero-lag gain: real − rotated mean",
            "#2b8cbe",
        ),
        (
            "control_dynamic_best_lag_gain_excess_vs_virtual",
            "control_dynamic_best_lag_gain_virtual_valid_count",
            "best-lag gain: real − rotated mean",
            "#e34a33",
        ),
        (
            "control_lock_fraction_excess_vs_virtual",
            "control_lock_fraction_virtual_valid_count",
            "lock fraction: real − rotated mean",
            "#31a354",
        ),
    ):
        controls.add_trace(
            go.Bar(
                x=labels,
                y=arrays[name],
                name=label,
                marker_color=color,
                customdata=np.column_stack(
                    (
                        arrays["control_virtual_reference_count"],
                        arrays[count_name],
                    )
                ),
                hovertemplate=(
                    "%{x}<br>contrast %{y:.4f}"
                    "<br>accepted references %{customdata[0]:.0f}"
                    "<br>finite null metrics %{customdata[1]:.0f}<extra></extra>"
                ),
            )
        )
    controls.add_trace(
        go.Scatter(
            x=labels,
            y=arrays["control_median_abs_error_improvement_vs_virtual_deg"],
            name="error improvement: rotated mean − real (deg)",
            mode="markers",
            marker={"color": "#636363", "size": 8},
            yaxis="y2",
            customdata=np.column_stack(
                (
                    arrays["control_virtual_reference_count"],
                    arrays["control_median_abs_error_virtual_valid_count"],
                )
            ),
            hovertemplate=(
                "%{x}<br>error improvement %{y:.3f}°"
                "<br>accepted references %{customdata[0]:.0f}"
                "<br>finite null metrics %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    controls.update_layout(
        title="Persisted real-versus-rotated spatial controls",
        barmode="group",
        yaxis={"title": "real minus rotated-control mean"},
        yaxis2={
            "title": "median |error| improvement (deg)",
            "overlaying": "y",
            "side": "right",
        },
    )

    event_count = arrays["lock_event_row_id"].size
    event_names = tuple(
        name for name in GAZE_TRACKING_ARRAYS if name.startswith("lock_event_")
    )
    if any(arrays[name].size != event_count for name in event_names):
        raise ValueError("Persisted lock-event arrays have inconsistent lengths.")
    events = go.Figure()
    for eye_code in (1, 2):
        mask = arrays["lock_event_eye_code"] == eye_code
        events.add_trace(
            go.Scatter(
                x=arrays["lock_event_start_acquisition_frame_id"][mask],
                y=arrays["lock_event_duration_s"][mask],
                mode="markers",
                name=str(eyes.get(str(eye_code), f"eye {eye_code}")),
                marker={"color": colors[eye_code], "size": 8},
                customdata=np.column_stack(
                    (
                        arrays["lock_event_chaser_identity_code"][mask],
                        arrays["lock_event_sample_count"][mask],
                        arrays["lock_event_median_abs_error_deg"][mask],
                    )
                ),
                hovertemplate=(
                    "start frame %{x:.0f}<br>duration %{y:.3f} s"
                    "<br>chaser %{customdata[0]:.0f}<br>samples %{customdata[1]:.0f}"
                    "<br>median |error| %{customdata[2]:.2f}°<extra></extra>"
                ),
            )
        )
    events.update_layout(
        title="Persisted sustained lock-on events",
        xaxis_title="start acquisition frame",
        yaxis_title="duration (s)",
    )

    figure_meta = {
        **plain(projection.provenance),
        "display_recipe": {
            "recipe_id": "persisted_exact_body_frame_gaze_tracking_v1",
            "scatter_max_points_per_eye": GAZE_DISPLAY_MAX_POINTS,
            "scatter_projection": "source_order_uniform_endpoint_preserving_v1",
            "error_bin_width_deg": GAZE_ERROR_BIN_WIDTH_DEG,
            "error_normalization": "probability_within_eye_accessible_rows",
            "scientific_recomputation": False,
        },
    }
    for figure in (scatter, error, summary, dynamic, controls, events):
        figure.update_layout(meta=figure_meta)
    return mo.vstack(
        [
            mo.callout(
                "This schema-v3 view uses persisted exact rows, dynamic summaries, "
                "sustained lock events, and reviewed-arena rotated controls. The "
                "viewer does not reconstruct scientific metrics.",
                kind="info",
            ),
            scatter,
            error,
            summary,
            dynamic,
            controls,
            events,
        ]
    )


__all__ = [
    "GAZE_DISPLAY_MAX_POINTS",
    "GAZE_ERROR_BIN_WIDTH_DEG",
    "_histogram_probability",
    "_uniform_indices",
    "build_exact_gaze_tracking_output",
]
