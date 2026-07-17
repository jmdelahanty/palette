"""Bounded renderer for persisted bout-kinematics visualization contracts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from fisheye.shared.zarr_io import open_zarr_root
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
from fisheye.visualization.bout_kinematics_interactive import (
    BOUT_SCHEMA_TO_ANALYSIS_ID,
)

from .common import join_path, normalize_path, png_bytes_to_markdown_image
from .provenance import build_spec_provenance_panel
from .registry import InteractiveSpecOption, discover_recording_explorer_spec_options


MAX_BOUT_SNAPSHOT_BYTES = 25_000_000
_ANALYSIS_ORDER = ("heading", "movement", "eye_gaze")

_METRIC_LABELS = {
    "net_delta_heading_deg": "Net heading change",
    "abs_net_delta_heading_deg": "Absolute heading change",
    "within_heading_range_deg": "Within-bout heading range",
    "within_heading_peak_to_peak_deg": "Within-bout heading peak-to-peak",
    "within_heading_path_deg": "Within-bout heading path",
    "within_heading_std_deg": "Within-bout heading standard deviation",
    "within_angular_velocity_mean_deg_s": "Mean angular velocity",
    "within_angular_speed_mean_deg_s": "Mean angular speed",
    "within_angular_speed_max_deg_s": "Peak angular speed",
    "within_angular_velocity_std_deg_s": "Angular velocity standard deviation",
    "detector_duration_s": "Detector bout duration",
    "physical_active_duration_s": "Physical active duration",
    "physical_active_duration_s_interpolated": "Interpolated physical active duration",
    "physical_active_path_length_mm": "Physical active path length",
    "physical_active_mean_speed_mm_s": "Physical active mean speed",
    "physical_active_peak_speed_mm_s": "Physical active peak speed",
    "pre_vergence_gaze_mean_deg": "Pre-bout vergence",
    "post_vergence_gaze_mean_deg": "Post-bout vergence",
    "within_bout_vergence_gaze_mean_deg": "Within-bout mean vergence",
    "within_bout_vergence_gaze_max_deg": "Within-bout maximum vergence",
    "within_bout_vergence_gaze_range_deg": "Within-bout vergence range",
    "within_bout_converged_fraction": "Within-bout converged fraction",
}
_METRIC_UNITS = {
    field: "degrees/s" if field.endswith("_deg_s") else "degrees"
    for field in _METRIC_LABELS
    if "heading" in field or "angular" in field or "gaze" in field
}
_METRIC_UNITS.update(
    {
        "detector_duration_s": "s",
        "physical_active_duration_s": "s",
        "physical_active_duration_s_interpolated": "s",
        "physical_active_path_length_mm": "mm",
        "physical_active_mean_speed_mm_s": "mm/s",
        "physical_active_peak_speed_mm_s": "mm/s",
        "within_bout_converged_fraction": "fraction",
    }
)
_METRIC_RANGES = {
    "net_delta_heading_deg": (-180.0, 180.0),
    "abs_net_delta_heading_deg": (0.0, 180.0),
    "within_bout_converged_fraction": (0.0, 1.0),
}


@dataclass(frozen=True)
class BoutSnapshot:
    analysis_id: str
    option: InteractiveSpecOption
    artifact_path: str
    png_bytes: bytes


@dataclass(frozen=True)
class BoutKinematicsControls:
    option: InteractiveSpecOption
    metric_by_label: Mapping[str, str]
    metric_picker: Any
    heading_level_picker: Any
    bins_picker: Any
    valid_only: Any
    show_snapshot: Any
    view: Any


@dataclass(frozen=True)
class BoutMetricProjection:
    analysis_id: str
    metric: str
    metric_label: str
    unit: str
    heading_level: Optional[str]
    bin_left: np.ndarray
    bin_right: np.ndarray
    counts: np.ndarray
    probabilities: np.ndarray
    cumulative_percent: np.ndarray
    source_row_count: int
    selected_level_row_count: int
    finite_row_count: int
    plotted_row_count: int
    validity_excluded_count: int
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float
    source_paths_read: tuple[str, ...]
    load_duration_ms: float


def bout_options_for_run(
    zarr_path: Path | str,
    selected_option: InteractiveSpecOption,
) -> tuple[InteractiveSpecOption, ...]:
    options = discover_recording_explorer_spec_options(
        zarr_path,
        run_path_filter=selected_option.run_path,
    )
    bout_options = [
        option for option in options if option.schema_id in BOUT_SCHEMA_TO_ANALYSIS_ID
    ]
    if (
        selected_option.schema_id in BOUT_SCHEMA_TO_ANALYSIS_ID
        and selected_option.artifact_path not in {item.artifact_path for item in bout_options}
    ):
        bout_options.append(selected_option)
    return tuple(sorted(bout_options, key=lambda item: item.artifact_name))


def available_bout_analysis_ids(
    zarr_path: Path | str,
    selected_option: InteractiveSpecOption,
) -> tuple[str, ...]:
    available = {
        BOUT_SCHEMA_TO_ANALYSIS_ID[option.schema_id]
        for option in bout_options_for_run(zarr_path, selected_option)
        if option.schema_id in BOUT_SCHEMA_TO_ANALYSIS_ID
    }
    ordered = [analysis_id for analysis_id in _ANALYSIS_ORDER if analysis_id in available]
    if available:
        ordered.append("provenance")
    return tuple(ordered)


def _option_for_analysis(
    options: Iterable[InteractiveSpecOption],
    analysis_id: str,
) -> Optional[InteractiveSpecOption]:
    for option in options:
        if BOUT_SCHEMA_TO_ANALYSIS_ID.get(str(option.schema_id or "")) == analysis_id:
            return option
    return None


def available_bout_metric_fields(option: InteractiveSpecOption) -> tuple[str, ...]:
    """Return histogram-capable fields declared by the persisted plot spec."""

    metrics: list[str] = []
    panels = option.spec.get("panels", [])
    if not isinstance(panels, list):
        return ()
    for panel in panels:
        if not isinstance(panel, Mapping) or "histogram" not in str(panel.get("kind") or ""):
            continue
        raw_metrics = panel.get("metrics", [])
        if not isinstance(raw_metrics, list):
            continue
        for metric in raw_metrics:
            field = str(metric or "").strip()
            if field and field not in metrics:
                metrics.append(field)
    return tuple(metrics)


def build_bout_controls(
    mo: Any,
    *,
    zarr_path: Path | str,
    selected_option: InteractiveSpecOption,
    analysis_id: str,
) -> Optional[BoutKinematicsControls]:
    if analysis_id not in _ANALYSIS_ORDER:
        return None
    option = _option_for_analysis(
        bout_options_for_run(zarr_path, selected_option),
        analysis_id,
    )
    if option is None:
        return None
    metrics = available_bout_metric_fields(option)
    if not metrics:
        return None
    metric_by_label = {
        _METRIC_LABELS.get(metric, metric.replace("_", " ").title()): metric
        for metric in metrics
    }
    metric_picker = mo.ui.dropdown(
        options=list(metric_by_label),
        value=next(iter(metric_by_label)),
        label="Bout metric",
    )
    heading_level_picker = None
    if analysis_id == "heading":
        heading_levels = [
            str(level) for level in option.spec.get("heading_levels", []) if str(level).strip()
        ]
        if not heading_levels:
            heading_levels = [str(option.spec.get("default_heading_level") or "heading_smoothed")]
        default_level = str(option.spec.get("default_heading_level") or heading_levels[0])
        if default_level not in heading_levels:
            default_level = heading_levels[0]
        heading_level_picker = mo.ui.dropdown(
            options=heading_levels,
            value=default_level,
            label="Heading representation",
        )
    declared_bins = 40
    for panel in option.spec.get("panels", []):
        if isinstance(panel, Mapping) and "histogram" in str(panel.get("kind") or ""):
            declared_bins = int(panel.get("bins") or declared_bins)
            break
    default_bins = min(100, max(10, int(round(declared_bins / 5.0) * 5)))
    bins_picker = mo.ui.slider(
        start=10,
        stop=100,
        value=default_bins,
        step=5,
        label="Histogram bins",
    )
    valid_only = mo.ui.checkbox(value=True, label="Require valid bout window")
    show_snapshot = mo.ui.checkbox(value=False, label="Show persisted reference snapshot")
    controls = [metric_picker]
    if heading_level_picker is not None:
        controls.append(heading_level_picker)
    controls.extend([bins_picker, valid_only, show_snapshot])
    return BoutKinematicsControls(
        option=option,
        metric_by_label=metric_by_label,
        metric_picker=metric_picker,
        heading_level_picker=heading_level_picker,
        bins_picker=bins_picker,
        valid_only=valid_only,
        show_snapshot=show_snapshot,
        view=mo.hstack(controls, widths="equal"),
    )


def _owned_source_path(option: InteractiveSpecOption, raw_path: object) -> str:
    source_path = normalize_path(str(raw_path or ""))
    marker = "analysis/bout_kinematics_runs/"
    if not source_path.startswith(marker):
        raise ValueError(f"Bout metric source is outside the bout run family: {source_path!r}")
    remainder = source_path[len(marker) :]
    _persisted_run_name, separator, relative_path = remainder.partition("/")
    if not separator or not relative_path:
        raise ValueError(f"Bout metric source has no run-relative path: {source_path!r}")
    return join_path(option.run_path, relative_path)


def _decode_text_rows(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 2 and raw.dtype == np.uint8:
        return np.asarray(
            [row.tobytes().split(b"\x00", 1)[0].decode("utf-8") for row in raw],
            dtype=object,
        )
    return np.asarray(
        [
            (value.tobytes() if isinstance(value, np.bytes_) else bytes(value))
            .split(b"\x00", 1)[0]
            .decode("utf-8")
            if isinstance(value, (bytes, bytearray, np.bytes_))
            else str(value)
            for value in raw
        ],
        dtype=object,
    )


def _validity_fields_for_metric(analysis_id: str, metric: str) -> tuple[str, ...]:
    if analysis_id == "movement":
        return ("physical_active_valid",) if metric.startswith("physical_active_") else ()
    if analysis_id == "eye_gaze":
        if metric.startswith("pre_"):
            return ("pre_eye_window_valid",)
        if metric.startswith("post_"):
            return ("post_eye_window_valid",)
        return ("within_eye_window_valid",)
    if metric.startswith("within_angular_"):
        return ("within_angular_velocity_valid",)
    if metric.startswith("within_heading_"):
        return ("within_window_valid",)
    if metric in {"net_delta_heading_deg", "abs_net_delta_heading_deg"}:
        return ("pre_window_valid", "post_window_valid")
    return ()


def _finite_summary(values: np.ndarray) -> tuple[float, float, float, float, float]:
    if not values.size:
        return (float("nan"),) * 5
    return tuple(float(value) for value in np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]))  # type: ignore[return-value]


def load_bout_metric_projection(
    zarr_path: Path | str,
    selected_option: InteractiveSpecOption,
    *,
    analysis_id: str,
    metric: str,
    heading_level: Optional[str] = None,
    bins: int = 40,
    valid_only: bool = True,
) -> BoutMetricProjection:
    started = time.perf_counter()
    options = bout_options_for_run(zarr_path, selected_option)
    option = _option_for_analysis(options, analysis_id)
    if option is None:
        raise ValueError(f"No persisted {analysis_id!r} bout spec is available in this run.")
    if metric not in available_bout_metric_fields(option):
        raise ValueError(f"Metric {metric!r} is not declared by {option.schema_id}.")

    if analysis_id == "heading":
        levels = tuple(str(level) for level in option.spec.get("heading_levels", []))
        level = str(heading_level or option.spec.get("default_heading_level") or "").strip()
        if levels and level not in levels:
            raise ValueError(f"Heading level {level!r} is not declared; expected one of {levels}.")
    elif analysis_id == "movement":
        level = "movement"
    elif analysis_id == "eye_gaze":
        level = "eye_gaze"
    else:
        raise ValueError(f"Unsupported interactive bout analysis: {analysis_id!r}")

    source_paths = option.spec.get("source_paths", {})
    if not isinstance(source_paths, Mapping):
        raise ValueError("Bout plot spec source_paths is not a mapping.")
    source_key = f"{level}.{metric}"
    if source_key not in source_paths:
        raise ValueError(f"Bout plot spec does not declare source path {source_key!r}.")
    metric_path = _owned_source_path(option, source_paths[source_key])
    table_path, _, field_name = metric_path.rpartition("/")
    if field_name != metric:
        raise ValueError(
            f"Bout metric source field {field_name!r} does not match requested metric {metric!r}."
        )

    root = open_zarr_root(Path(zarr_path), mode="r")
    try:
        table = root[table_path]
        metric_values = np.asarray(table[metric][:], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Could not read bout metric column {metric_path!r}.") from exc
    if metric_values.ndim != 1:
        raise ValueError(f"Bout metric column must be one-dimensional, got {metric_values.shape}.")

    read_paths = [metric_path]
    source_row_count = int(metric_values.size)
    selected_mask = np.ones(source_row_count, dtype=bool)
    if analysis_id == "heading" and "heading_level_bytes" in table:
        level_path = join_path(table_path, "heading_level_bytes")
        labels = _decode_text_rows(np.asarray(table["heading_level_bytes"][:]))
        if labels.shape[0] != source_row_count:
            raise ValueError("heading_level_bytes row count does not match the metric column.")
        selected_mask &= labels == level
        read_paths.append(level_path)
    selected_level_row_count = int(np.count_nonzero(selected_mask))

    validity_mask = np.ones(source_row_count, dtype=bool)
    validity_fields = _validity_fields_for_metric(analysis_id, metric) if valid_only else ()
    for validity_field in validity_fields:
        if validity_field not in table:
            raise ValueError(
                f"Valid-only filtering requires missing column {table_path}/{validity_field}."
            )
        validity = np.asarray(table[validity_field][:], dtype=bool)
        if validity.shape != (source_row_count,):
            raise ValueError(
                f"Validity column {validity_field!r} shape {validity.shape} does not match metric rows."
            )
        validity_mask &= validity
        read_paths.append(join_path(table_path, validity_field))

    finite_mask = np.isfinite(metric_values)
    finite_row_count = int(np.count_nonzero(selected_mask & finite_mask))
    validity_excluded_count = int(
        np.count_nonzero(selected_mask & finite_mask & ~validity_mask)
    )
    plotted_values = metric_values[selected_mask & finite_mask & validity_mask]
    requested_bins = min(200, max(5, int(bins)))
    histogram_range = _METRIC_RANGES.get(metric)
    if plotted_values.size:
        counts, edges = np.histogram(
            plotted_values,
            bins=requested_bins,
            range=histogram_range,
        )
    else:
        empty_range = histogram_range or (0.0, 1.0)
        edges = np.linspace(empty_range[0], empty_range[1], requested_bins + 1)
        counts = np.zeros(requested_bins, dtype=np.int64)
    probabilities = (
        counts.astype(np.float64) / float(counts.sum())
        if counts.sum()
        else np.zeros_like(counts, dtype=np.float64)
    )
    summary = _finite_summary(plotted_values)
    return BoutMetricProjection(
        analysis_id=analysis_id,
        metric=metric,
        metric_label=_METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
        unit=_METRIC_UNITS.get(metric, "value"),
        heading_level=level if analysis_id == "heading" else None,
        bin_left=np.asarray(edges[:-1], dtype=np.float64),
        bin_right=np.asarray(edges[1:], dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
        probabilities=probabilities,
        cumulative_percent=np.cumsum(probabilities) * 100.0,
        source_row_count=source_row_count,
        selected_level_row_count=selected_level_row_count,
        finite_row_count=finite_row_count,
        plotted_row_count=int(plotted_values.size),
        validity_excluded_count=validity_excluded_count,
        minimum=summary[0],
        q25=summary[1],
        median=summary[2],
        q75=summary[3],
        maximum=summary[4],
        source_paths_read=tuple(read_paths),
        load_duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def resolve_bout_snapshot_path(option: InteractiveSpecOption) -> str:
    raw_snapshot = str(option.attrs.get("snapshot_artifact") or "").strip()
    if not raw_snapshot:
        raise ValueError(f"Bout spec has no snapshot_artifact: {option.artifact_path}")
    snapshot = normalize_path(raw_snapshot)
    if not snapshot:
        raise ValueError(f"Bout spec has an empty snapshot_artifact: {option.artifact_path}")
    if "/" not in snapshot:
        return join_path(option.run_path, "visualizations", snapshot)
    if snapshot.startswith("visualizations/"):
        return join_path(option.run_path, snapshot)
    return snapshot


def load_bout_snapshot(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
    *,
    analysis_id: str,
    max_bytes: int = MAX_BOUT_SNAPSHOT_BYTES,
) -> BoutSnapshot:
    artifact_path = resolve_bout_snapshot_path(option)
    root = open_zarr_root(Path(zarr_path), mode="r")
    try:
        artifact = root[artifact_path]
    except Exception as exc:
        raise ValueError(f"Bout snapshot is missing: {artifact_path}") from exc
    byte_length = int(getattr(artifact, "attrs", {}).get("byte_length") or 0)
    if byte_length > int(max_bytes):
        raise ValueError(
            f"Bout snapshot is {byte_length:,} bytes, above the {int(max_bytes):,}-byte display limit."
        )
    resolved_path, png_bytes = load_png_artifact_bytes(root, artifact_path)
    if len(png_bytes) > int(max_bytes):
        raise ValueError(
            f"Bout snapshot is {len(png_bytes):,} bytes, above the {int(max_bytes):,}-byte display limit."
        )
    return BoutSnapshot(
        analysis_id=analysis_id,
        option=option,
        artifact_path=resolved_path,
        png_bytes=png_bytes,
    )


def build_bout_metric_figure(go: Any, projection: BoutMetricProjection) -> Any:
    centers = (projection.bin_left + projection.bin_right) / 2.0
    widths = projection.bin_right - projection.bin_left
    customdata = np.column_stack(
        [
            projection.bin_left,
            projection.bin_right,
            projection.probabilities * 100.0,
            projection.cumulative_percent,
        ]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=projection.counts,
            width=widths,
            customdata=customdata,
            name="Bouts",
            marker_color="#3B82F6",
            hovertemplate=(
                "Bin: %{customdata[0]:.3g}–%{customdata[1]:.3g}<br>"
                "Bouts: %{y:,}<br>Percent: %{customdata[2]:.2f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection.bin_right,
            y=projection.cumulative_percent,
            mode="lines",
            name="Cumulative",
            yaxis="y2",
            line={"color": "#DC2626", "width": 2},
            hovertemplate="≤ %{x:.3g}: %{y:.2f}%<extra></extra>",
        )
    )
    level_suffix = f" · {projection.heading_level}" if projection.heading_level else ""
    fig.update_layout(
        title=f"{projection.metric_label}{level_suffix}",
        xaxis_title=(
            projection.metric_label
            if projection.unit == "value"
            else f"{projection.metric_label} ({projection.unit})"
        ),
        yaxis={"title": "Bout count", "rangemode": "tozero"},
        yaxis2={
            "title": "Cumulative (%)",
            "overlaying": "y",
            "side": "right",
            "range": [0.0, 100.0],
        },
        barmode="overlay",
        hovermode="x unified",
        height=500,
        margin={"l": 65, "r": 70, "t": 60, "b": 70},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return fig


def _format_summary_value(value: float, unit: str) -> str:
    if not np.isfinite(value):
        return "n/a"
    suffix = "" if unit in {"value", "fraction"} else f" {unit}"
    return f"{value:.3g}{suffix}"


def build_bout_kinematics_output(
    mo: Any,
    *,
    zarr_path: Path | str,
    selected_option: InteractiveSpecOption,
    analysis_id: str,
    go: Any = None,
    projection: Optional[BoutMetricProjection] = None,
    show_snapshot: bool = False,
) -> Any:
    options = bout_options_for_run(zarr_path, selected_option)
    if analysis_id == "provenance":
        sections: dict[str, Any] = {}
        for option in options:
            option_analysis = BOUT_SCHEMA_TO_ANALYSIS_ID.get(str(option.schema_id or ""), "bout")
            sections[option_analysis.replace("_", " ").title()] = build_spec_provenance_panel(
                mo,
                spec=option.spec,
                artifact_attrs=option.attrs,
                option=option,
            )
        return mo.accordion(sections) if sections else mo.md("No bout provenance is available.")

    option = _option_for_analysis(options, analysis_id)
    if option is None:
        return mo.callout(
            f"No persisted `{analysis_id}` bout visualization is present in this run.",
            kind="warn",
        )
    if projection is not None and go is not None:
        items: list[Any] = [
            mo.md(f"### {option.title}"),
            build_bout_metric_figure(go, projection),
            mo.hstack(
                [
                    mo.stat(label="Plotted bouts", value=f"{projection.plotted_row_count:,}"),
                    mo.stat(
                        label="Median",
                        value=_format_summary_value(projection.median, projection.unit),
                    ),
                    mo.stat(
                        label="IQR",
                        value=(
                            f"{_format_summary_value(projection.q25, projection.unit)} – "
                            f"{_format_summary_value(projection.q75, projection.unit)}"
                        ),
                    ),
                    mo.stat(label="Column load", value=f"{projection.load_duration_ms:.1f} ms"),
                ],
                widths="equal",
            ),
            mo.callout(
                (
                    f"Read {len(projection.source_paths_read)} selected column(s); "
                    f"{projection.validity_excluded_count:,} finite row(s) were excluded by "
                    "the validity filter. Only aggregated histogram bins are sent to Plotly."
                ),
                kind="info",
            ),
        ]
        if show_snapshot:
            try:
                snapshot = load_bout_snapshot(
                    zarr_path,
                    option,
                    analysis_id=analysis_id,
                )
                items.append(
                    mo.accordion(
                        {
                            "Persisted reference snapshot": mo.vstack(
                                [
                                    mo.md(f"`{snapshot.artifact_path}`"),
                                    png_bytes_to_markdown_image(
                                        mo,
                                        snapshot.png_bytes,
                                        alt_text=option.title or analysis_id,
                                    ),
                                ]
                            )
                        }
                    )
                )
            except Exception as exc:
                items.append(
                    mo.callout(
                        f"Reference snapshot could not be loaded: `{type(exc).__name__}: {exc}`",
                        kind="warn",
                    )
                )
        items.append(
            mo.accordion(
                {
                    "Columns and contract": mo.tree(
                        {
                            "metric": projection.metric,
                            "heading_level": projection.heading_level,
                            "source_paths_read": list(projection.source_paths_read),
                            "source_row_count": projection.source_row_count,
                            "selected_level_row_count": projection.selected_level_row_count,
                            "finite_row_count": projection.finite_row_count,
                            "plotted_row_count": projection.plotted_row_count,
                            "schema_id": option.schema_id,
                            "renderer": option.renderer,
                        }
                    )
                }
            )
        )
        return mo.vstack(items)
    try:
        snapshot = load_bout_snapshot(
            zarr_path,
            option,
            analysis_id=analysis_id,
        )
    except Exception as exc:
        return mo.callout(
            f"Bout visualization could not be loaded: `{type(exc).__name__}: {exc}`",
            kind="danger",
        )

    details: Mapping[str, Any] = {
        "run": option.run_name,
        "schema_id": option.schema_id,
        "renderer": option.renderer,
        "snapshot_artifact": snapshot.artifact_path,
        "source_paths": option.spec.get("source_paths", {}),
        "parameters": option.spec.get("parameters", {}),
    }
    return mo.vstack(
        [
            mo.md(f"### {option.title}"),
            png_bytes_to_markdown_image(
                mo,
                snapshot.png_bytes,
                alt_text=option.title or analysis_id,
            ),
            mo.accordion({"Contract and sources": mo.tree(dict(details))}),
        ]
    )
