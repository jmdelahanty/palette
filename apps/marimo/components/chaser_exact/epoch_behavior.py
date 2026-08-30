"""Persisted protocol-semantic speed, path, bout, and IBI summaries."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)

from .array_requirements import EPOCH_BEHAVIOR_ARRAYS
from .projection import ExactChaserSuccessorProjection
from .provenance import freeze, plain


EPOCH_BEHAVIOR_DISPLAY_RECIPE = "persisted_semantic_v2_epoch_motion_bouts_v1"
_BOUT_METRICS = (
    ("bout_duration_s", "Bout duration", "s"),
    ("bout_path_length_mm", "Bout path length", "mm"),
    ("bout_net_heading_change_deg", "Net heading change", "deg"),
    ("abs_bout_net_heading_change_deg", "Absolute heading change", "deg"),
    ("bout_heading_path_deg", "Heading path", "deg"),
)
_ROLE_COLORS = {
    "chaser_pre": "#4c78a8",
    "chaser_training": "#e45756",
    "chaser_post": "#54a24b",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Epoch behavior lacks {label}.")
    return value


def _decode(values: Any) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    return np.asarray(
        [
            (
                bytes(value).rstrip(b"\x00").decode("utf-8")
                if isinstance(value, (bytes, np.bytes_))
                else str(value)
            )
            for value in array
        ],
        dtype=object,
    )


def _array(handle: Any, path: str, *, size: int | None = None) -> np.ndarray:
    try:
        values = np.asarray(handle.array(path)).reshape(-1)
    except KeyError as exc:
        raise ValueError(
            f"Epoch behavior lacks required persisted array {path!r}."
        ) from exc
    if size is not None and values.shape != (size,):
        raise ValueError(f"Epoch behavior array {path!r} has the wrong row count.")
    return values


def _finite_rate(count: np.ndarray, duration_s: np.ndarray) -> np.ndarray:
    return np.divide(
        count.astype(np.float64),
        duration_s.astype(np.float64) / 60.0,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=duration_s > 0,
    )


def _histogram_table(
    handle: Any,
    *,
    table: str,
    expected_metrics: tuple[str, ...],
    epoch: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    metric = _decode(_array(handle, f"{table}/metric_name"))
    n_rows = metric.size
    arrays = {
        name: _array(handle, f"{table}/{name}", size=n_rows)
        for name in (
            "window_id",
            "window_index",
            "start_frame",
            "end_frame",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "bin_index",
            "bin_left",
            "bin_right",
            "bin_center",
            "bin_width",
            "hist_count",
            "hist_fraction",
            "source_sample_count",
            "finite_sample_count",
            "protocol_semantic_step_index",
        )
    }
    text = {
        name: _decode(_array(handle, f"{table}/{name}", size=n_rows))
        for name in (
            "units",
            "window_label",
            "bin_policy",
            "analysis_role",
            "source_interval_sha256",
            "protocol_semantic_hash",
            "protocol_semantic_step_ref",
        )
    }
    if set(metric.tolist()) != set(expected_metrics):
        raise ValueError(f"Epoch behavior {table} metric roster is incompatible.")
    role_indices = {role: index for index, role in enumerate(CHASER_WINDOW_ROLES)}
    strata: dict[str, dict[str, np.ndarray]] = {
        metric_name: {} for metric_name in expected_metrics
    }
    edges_by_metric: dict[str, np.ndarray] = {}
    for metric_name in expected_metrics:
        for role in CHASER_WINDOW_ROLES:
            indices = np.flatnonzero(
                (metric == metric_name) & (text["analysis_role"] == role)
            )
            if not indices.size:
                raise ValueError(
                    f"Epoch behavior {table} lacks {metric_name!r} for {role!r}."
                )
            order = np.argsort(arrays["bin_index"][indices], kind="stable")
            indices = indices[order]
            expected_bins = np.arange(indices.size, dtype=np.int64)
            left = arrays["bin_left"][indices].astype(np.float64)
            right = arrays["bin_right"][indices].astype(np.float64)
            center = arrays["bin_center"][indices].astype(np.float64)
            width = arrays["bin_width"][indices].astype(np.float64)
            count = arrays["hist_count"][indices].astype(np.int64)
            fraction = arrays["hist_fraction"][indices].astype(np.float64)
            finite_count = arrays["finite_sample_count"][indices].astype(np.int64)
            source_count = arrays["source_sample_count"][indices].astype(np.int64)
            epoch_index = role_indices[role]
            identity_checks = (
                np.all(arrays["window_index"][indices] == epoch_index),
                np.all(arrays["window_id"][indices] == epoch["window_id"][epoch_index]),
                np.all(
                    arrays["start_frame"][indices] == epoch["start_frame"][epoch_index]
                ),
                np.all(arrays["end_frame"][indices] == epoch["end_frame"][epoch_index]),
                np.allclose(
                    arrays["start_time_s"][indices],
                    epoch["start_time_s"][epoch_index],
                ),
                np.allclose(
                    arrays["end_time_s"][indices],
                    epoch["end_time_s"][epoch_index],
                ),
                np.allclose(
                    arrays["duration_s"][indices],
                    epoch["duration_s"][epoch_index],
                ),
                np.all(text["window_label"][indices] == role),
                np.all(
                    text["source_interval_sha256"][indices]
                    == epoch["source_interval_sha256"][epoch_index]
                ),
                np.all(
                    text["protocol_semantic_hash"][indices]
                    == epoch["protocol_semantic_hash"][epoch_index]
                ),
                np.all(
                    arrays["protocol_semantic_step_index"][indices]
                    == epoch["protocol_semantic_step_index"][epoch_index]
                ),
                np.all(
                    text["protocol_semantic_step_ref"][indices]
                    == epoch["protocol_semantic_step_ref"][epoch_index]
                ),
            )
            total = int(np.sum(count))
            expected_fraction = (
                count.astype(np.float64) / total
                if total
                else np.full(count.shape, np.nan, dtype=np.float64)
            )
            if (
                not np.array_equal(arrays["bin_index"][indices], expected_bins)
                or np.any(~np.isfinite(left))
                or np.any(~np.isfinite(right))
                or np.any(~np.isfinite(center))
                or np.any(~np.isfinite(width))
                or np.any(width <= 0)
                or np.any(right <= left)
                or not np.allclose(center, (left + right) / 2.0)
                or not np.allclose(width, right - left)
                or (indices.size > 1 and not np.allclose(left[1:], right[:-1]))
                or np.any(count < 0)
                or np.unique(finite_count).size != 1
                or np.unique(source_count).size != 1
                or int(finite_count[0]) != total
                or int(source_count[0]) < total
                or not np.allclose(fraction, expected_fraction, equal_nan=True)
                or not all(identity_checks)
            ):
                raise ValueError(
                    f"Epoch behavior {table} persisted bins or support are inconsistent."
                )
            edges = np.concatenate((left[:1], right))
            if metric_name in edges_by_metric:
                if not np.array_equal(edges_by_metric[metric_name], edges):
                    raise ValueError(
                        f"Epoch behavior {metric_name!r} bins differ across epochs."
                    )
            else:
                edges_by_metric[metric_name] = edges
            strata[metric_name][role] = indices
    return freeze(
        {
            "metric": metric,
            **arrays,
            **text,
            "strata": strata,
            "edges_by_metric": edges_by_metric,
        }
    )


def _epoch_behavior_values(
    projection: ExactChaserSuccessorProjection,
) -> Mapping[str, Any]:
    """Validate the three persisted semantic rows and their persisted histograms."""

    handle = projection.epoch_behavior
    if handle is None:
        raise ValueError(
            "Epoch behavior requires one exact persisted semantic-v2 child."
        )
    handle.require_verified_arrays(EPOCH_BEHAVIOR_ARRAYS)
    n = len(CHASER_WINDOW_ROLES)
    numeric_names = (
        "track_id",
        "window_id",
        "window_index",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "total_span_frames",
        "provider_sample_count",
        "valid_tracked_frame_count",
        "missing_frame_count",
        "tracking_dropout_fraction",
        "valid_tracked_duration_s",
        "motion_valid_sample_count",
        "speed_sample_count",
        "mean_speed_mm_s",
        "median_speed_mm_s",
        "p05_speed_mm_s",
        "p95_speed_mm_s",
        "max_speed_mm_s",
        "total_path_mm",
        "bout_count",
        "bout_rate_per_min",
        "median_bout_duration_s",
        "mean_bout_duration_s",
        "median_bout_path_length_mm",
        "mean_bout_path_length_mm",
        "bout_heading_sample_count",
        "mean_bout_net_heading_change_deg",
        "median_bout_net_heading_change_deg",
        "mean_abs_bout_net_heading_change_deg",
        "median_abs_bout_net_heading_change_deg",
        "mean_bout_heading_path_deg",
        "median_bout_heading_path_deg",
        "inter_bout_interval_count",
        "mean_inter_bout_interval_s",
        "median_inter_bout_interval_s",
        "p05_inter_bout_interval_s",
        "p95_inter_bout_interval_s",
        "inter_bout_interval_rate_per_min",
        "protocol_semantic_step_index",
    )
    values = {
        name: _array(handle, f"per_epoch_fish/{name}", size=n) for name in numeric_names
    }
    text = {
        name: _decode(_array(handle, f"per_epoch_fish/{name}", size=n))
        for name in (
            "window_label",
            "rate_denominator",
            "motion_validity_rule",
            "analysis_role",
            "source_interval_sha256",
            "protocol_semantic_hash",
            "protocol_semantic_step_ref",
        )
    }
    values.update(text)
    roles = np.asarray(CHASER_WINDOW_ROLES, dtype=object)
    semantic = _mapping(handle.semantic_selection, label="semantic selection")
    role_records = semantic.get("semantic_role_bindings")
    if not isinstance(role_records, (list, tuple)) or len(role_records) != n:
        raise ValueError("Epoch behavior semantic role records are incomplete.")
    records = {record.get("analysis_role"): record for record in role_records}
    expected_hash = semantic.get("protocol_semantic_hash")
    if (
        not np.array_equal(values["analysis_role"], roles)
        or not np.array_equal(values["window_label"], roles)
        or not np.array_equal(values["window_index"], np.arange(n))
        or np.unique(values["window_id"]).size != n
        or np.unique(values["track_id"]).size != 1
        or int(values["track_id"][0]) != int(handle.manifest["parameters"]["track_id"])
        or np.any(values["end_frame"] < values["start_frame"])
        or np.any(~np.isfinite(values["start_time_s"]))
        or np.any(~np.isfinite(values["end_time_s"]))
        or np.any(~np.isfinite(values["duration_s"]))
        or np.any(values["end_time_s"] <= values["start_time_s"])
        or not np.allclose(
            values["duration_s"], values["end_time_s"] - values["start_time_s"]
        )
        or np.any(
            values["total_span_frames"]
            != values["end_frame"] - values["start_frame"] + 1
        )
        or np.any(values["provider_sample_count"] < values["valid_tracked_frame_count"])
        or np.any(values["provider_sample_count"] > values["total_span_frames"])
        or np.any(
            values["missing_frame_count"]
            != values["total_span_frames"] - values["valid_tracked_frame_count"]
        )
        or np.any(
            values["motion_valid_sample_count"] > values["valid_tracked_frame_count"]
        )
        or np.any(values["speed_sample_count"] > values["motion_valid_sample_count"])
        or np.any(values["valid_tracked_duration_s"] < 0)
        or np.any(values["valid_tracked_duration_s"] > values["duration_s"] + 1e-9)
        or np.any(values["bout_count"] < 0)
        or np.any(values["inter_bout_interval_count"] < 0)
        or np.any(values["total_path_mm"] < 0)
        or not np.all(values["rate_denominator"] == "valid_tracked_duration_s")
        or not np.all(
            values["motion_validity_rule"] == "linear_sample_valid_and_transition_valid"
        )
        or not np.all(values["protocol_semantic_hash"] == expected_hash)
    ):
        raise ValueError(
            "Epoch behavior semantic rows or denominators are inconsistent."
        )
    expected_dropout = np.divide(
        values["missing_frame_count"].astype(np.float64),
        values["total_span_frames"].astype(np.float64),
        out=np.full(n, np.nan),
        where=values["total_span_frames"] > 0,
    )
    if (
        not np.allclose(
            values["tracking_dropout_fraction"], expected_dropout, equal_nan=True
        )
        or not np.allclose(
            values["bout_rate_per_min"],
            _finite_rate(values["bout_count"], values["valid_tracked_duration_s"]),
            equal_nan=True,
        )
        or not np.allclose(
            values["inter_bout_interval_rate_per_min"],
            _finite_rate(
                values["inter_bout_interval_count"],
                values["valid_tracked_duration_s"],
            ),
            equal_nan=True,
        )
    ):
        raise ValueError("Epoch behavior persisted rates or coverage are inconsistent.")
    for index, role in enumerate(CHASER_WINDOW_ROLES):
        record = records.get(role)
        if not isinstance(record, Mapping) or any(
            (
                int(values["window_id"][index]) != record.get("source_window_id"),
                int(values["start_frame"][index]) != record.get("selected_start_frame"),
                int(values["end_frame"][index]) + 1
                != record.get("selected_end_frame_exclusive"),
                values["source_interval_sha256"][index]
                != record.get("source_interval_sha256"),
                int(values["protocol_semantic_step_index"][index])
                != record.get("protocol_semantic_step_index"),
                values["protocol_semantic_step_ref"][index]
                != record.get("protocol_semantic_step_ref"),
            )
        ):
            raise ValueError(
                "Epoch behavior row differs from semantic source identity."
            )
    bout_hist = _histogram_table(
        handle,
        table="per_epoch_bout_histograms",
        expected_metrics=tuple(item[0] for item in _BOUT_METRICS),
        epoch=values,
    )
    ibi_hist = _histogram_table(
        handle,
        table="per_epoch_inter_bout_interval_histograms",
        expected_metrics=("inter_bout_interval_s",),
        epoch=values,
    )
    return freeze({**values, "bout_hist": bout_hist, "ibi_hist": ibi_hist})


def _add_role_histograms(
    figure: Any,
    go: Any,
    *,
    table: Mapping[str, Any],
    metric_name: str,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    for role in CHASER_WINDOW_ROLES:
        indices = table["strata"][metric_name][role]
        figure.add_trace(
            go.Scatter(
                x=np.asarray(table["bin_center"])[indices],
                y=np.asarray(table["hist_fraction"])[indices] * 100.0,
                mode="lines+markers",
                name=role,
                legendgroup=role,
                showlegend=show_legend,
                line={"color": _ROLE_COLORS[role]},
                customdata=np.column_stack(
                    (
                        np.asarray(table["hist_count"])[indices],
                        np.asarray(table["finite_sample_count"])[indices],
                        np.asarray(table["bin_left"])[indices],
                        np.asarray(table["bin_right"])[indices],
                    )
                ),
                hovertemplate=(
                    "bin [%{customdata[2]:.3g}, %{customdata[3]:.3g})"
                    "<br>fraction=%{y:.3f}%<br>count=%{customdata[0]:.0f}"
                    "<br>finite support=%{customdata[1]:.0f}<extra>%{fullData.name}</extra>"
                ),
            ),
            row=row,
            col=col,
        )


def build_exact_epoch_behavior_output(
    mo: Any,
    go: Any,
    projection: ExactChaserSuccessorProjection,
) -> Any:
    """Render exact persisted epoch summaries without recomputing epoch math."""

    from plotly.subplots import make_subplots

    values = _epoch_behavior_values(projection)
    roles = list(CHASER_WINDOW_ROLES)
    colors = [_ROLE_COLORS[role] for role in roles]

    summary = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Persisted physical speed",
            "Persisted path length",
            "Tracking and motion coverage",
            "Valid tracked duration",
        ),
    )
    for name, label in (("mean_speed_mm_s", "mean"), ("median_speed_mm_s", "median")):
        summary.add_trace(
            go.Bar(x=roles, y=values[name], name=f"{label} speed"), row=1, col=1
        )
    summary.add_trace(
        go.Bar(x=roles, y=values["total_path_mm"], marker_color=colors, name="path"),
        row=1,
        col=2,
    )
    summary.add_trace(
        go.Bar(
            x=roles,
            y=(1.0 - values["tracking_dropout_fraction"]) * 100.0,
            name="tracked / epoch span",
        ),
        row=2,
        col=1,
    )
    summary.add_trace(
        go.Bar(
            x=roles,
            y=np.divide(
                values["motion_valid_sample_count"].astype(np.float64),
                values["total_span_frames"].astype(np.float64),
                out=np.zeros(3),
                where=values["total_span_frames"] > 0,
            )
            * 100.0,
            name="motion-valid / epoch span",
        ),
        row=2,
        col=1,
    )
    summary.add_trace(
        go.Bar(
            x=roles,
            y=values["valid_tracked_duration_s"],
            marker_color=colors,
            name="valid duration",
        ),
        row=2,
        col=2,
    )
    summary.update_yaxes(title_text="mm/s", row=1, col=1)
    summary.update_yaxes(title_text="mm", row=1, col=2)
    summary.update_yaxes(title_text="percent", row=2, col=1)
    summary.update_yaxes(title_text="s", row=2, col=2)
    summary.update_layout(title="Protocol-semantic epoch locomotion", barmode="group")

    bout = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Bout count", "Bout rate", "Median duration", "Median path"),
    )
    for row, col, name, title in (
        (1, 1, "bout_count", "count"),
        (1, 2, "bout_rate_per_min", "bouts/min valid time"),
        (2, 1, "median_bout_duration_s", "s"),
        (2, 2, "median_bout_path_length_mm", "mm"),
    ):
        bout.add_trace(
            go.Bar(x=roles, y=values[name], marker_color=colors, showlegend=False),
            row=row,
            col=col,
        )
        bout.update_yaxes(title_text=title, row=row, col=col)
    bout.update_layout(title="Persisted swim-bout summaries")

    hist = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=tuple(item[1] for item in _BOUT_METRICS) + ("",),
    )
    for index, (metric_name, _label, units) in enumerate(_BOUT_METRICS):
        row = index // 2 + 1
        col = index % 2 + 1
        _add_role_histograms(
            hist,
            go,
            table=values["bout_hist"],
            metric_name=metric_name,
            row=row,
            col=col,
            show_legend=index == 0,
        )
        hist.update_xaxes(title_text=units, row=row, col=col)
        hist.update_yaxes(title_text="percent", row=row, col=col)
    hist.update_layout(title="Persisted bout-kinematics distributions")

    ibi = make_subplots(rows=1, cols=1)
    _add_role_histograms(
        ibi,
        go,
        table=values["ibi_hist"],
        metric_name="inter_bout_interval_s",
        row=1,
        col=1,
        show_legend=True,
    )
    ibi.update_xaxes(title_text="inter-bout interval (s)")
    ibi.update_yaxes(title_text="percent")
    ibi.update_layout(title="Persisted inter-bout interval distributions")

    handle = projection.epoch_behavior
    display = {
        "recipe_id": EPOCH_BEHAVIOR_DISPLAY_RECIPE,
        "source_arrays": list(EPOCH_BEHAVIOR_ARRAYS),
        "source_speed_level": handle.manifest["parameters"]["physical_speed_level"],
        "rate_denominator": "valid_tracked_duration_s",
        "motion_validity_rule": "linear_sample_valid_and_transition_valid",
        "epoch_binding": plain(handle.semantic_selection),
        "source_provider_motion": plain(handle.manifest["sources"]["provider_motion"]),
        "source_swim_bouts": plain(handle.manifest["sources"]["swim_bouts"]),
        "histogram_bins": {
            metric: np.asarray(edges).tolist()
            for metric, edges in {
                **values["bout_hist"]["edges_by_metric"],
                **values["ibi_hist"]["edges_by_metric"],
            }.items()
        },
        "spatial_metrics": "omitted_requires_separately_selected_position_provider",
        "protocol_to_acquisition_alignment": handle.manifest["parameters"][
            "protocol_to_acquisition_alignment"
        ],
        "physical_presentation_timing": "not_claimed",
        "viewer_rebinning": "prohibited",
        "viewer_epoch_recomputation": "prohibited",
        "scientific_recomputation": False,
    }
    meta = plain(projection.provenance)
    meta["epoch_behavior_display"] = display
    for figure in (summary, bout, hist, ibi):
        figure.update_layout(meta=meta)
    return mo.vstack(
        [
            mo.callout(
                "This semantic-v2 view reads persisted provider-motion, swim-bout, "
                "and IBI summaries. Rates use valid tracked duration. Spatial metrics "
                "are intentionally separate, and the sealed epoch alignment is a "
                "protocol-to-acquisition proxy rather than physical presentation timing.",
                kind="info",
            ),
            summary,
            bout,
            hist,
            ibi,
        ]
    )


__all__ = [
    "EPOCH_BEHAVIOR_DISPLAY_RECIPE",
    "_epoch_behavior_values",
    "build_exact_epoch_behavior_output",
]
