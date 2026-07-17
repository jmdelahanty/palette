"""Interactive, bounded tail-kinematics views for the recording explorer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
from plotly.subplots import make_subplots

from .common import png_bytes_to_markdown_image


TYPICAL_LARVAL_TAIL_BEAT_HZ = (20.0, 40.0)


def _numeric_matrix(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((frame.height, 0), dtype=np.float32)
    return np.column_stack(
        [frame.get_column(name).cast(pl.Float32, strict=False).to_numpy() for name in columns]
    )


def _symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanpercentile(finite, 99.0)), np.finfo(np.float32).eps)


def _heatmap_figure(
    go: Any,
    *,
    time_s: np.ndarray,
    sample_s: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_title: str,
) -> Any:
    limit = _symmetric_limit(values)
    figure = go.Figure(
        go.Heatmap(
            x=time_s,
            y=sample_s,
            z=values.T,
            colorscale="RdBu",
            reversescale=True,
            zmid=0.0,
            zmin=-limit,
            zmax=limit,
            connectgaps=False,
            colorbar=dict(title=colorbar_title),
            hovertemplate=(
                "time=%{x:.3f}s<br>normalized tail position=%{y:.3f}"
                "<br>value=%{z:.4g}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Normalized tail position (base → tip)",
        height=430,
        margin=dict(l=70, r=30, t=65, b=55),
    )
    return figure


def _bout_columns(frame: pl.DataFrame) -> tuple[str | None, str | None]:
    schema = set(frame.columns)
    start = next((name for name in ("start_s", "start_time_s") if name in schema), None)
    stop = next((name for name in ("end_s", "end_time_s") if name in schema), None)
    return start, stop


def _synchronized_trace_figure(
    go: Any,
    *,
    frame: pl.DataFrame,
    scalar_columns: Sequence[str],
    bout_frame: pl.DataFrame,
    position_frame: pl.DataFrame,
) -> Any:
    has_position = position_frame.height > 0 and {"time_s", "x", "y"}.issubset(
        position_frame.columns
    )
    figure = make_subplots(
        rows=2 if has_position else 1,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}]] + ([[{}]] if has_position else []),
        row_heights=[0.62, 0.38] if has_position else [1.0],
        vertical_spacing=0.08,
        subplot_titles=(
            ("Tail scalar traces and persisted bout intervals", "Fish x/y position")
            if has_position
            else ("Tail scalar traces and persisted bout intervals",)
        ),
    )
    for name in scalar_columns:
        if name not in frame.columns:
            continue
        figure.add_trace(
            go.Scattergl(
                x=frame["time_s"],
                y=frame[name],
                mode="lines",
                name=name,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    start_name, stop_name = _bout_columns(bout_frame)
    if start_name is not None and stop_name is not None and bout_frame.height:
        starts = bout_frame[start_name].cast(pl.Float64, strict=False).to_numpy()
        stops = bout_frame[stop_name].cast(pl.Float64, strict=False).to_numpy()
        widths = stops - starts
        valid = np.isfinite(starts) & np.isfinite(stops) & (widths > 0)
        bout_id_name = next(
            (name for name in ("bout_id", "source_bout_id") if name in bout_frame.columns),
            None,
        )
        bout_ids = (
            bout_frame[bout_id_name].to_numpy()
            if bout_id_name is not None
            else np.arange(bout_frame.height, dtype=np.int64)
        )
        if valid.any():
            figure.add_trace(
                go.Bar(
                    x=(starts[valid] + stops[valid]) / 2.0,
                    y=np.ones(int(np.count_nonzero(valid)), dtype=np.float32),
                    width=widths[valid],
                    base=np.zeros(int(np.count_nonzero(valid)), dtype=np.float32),
                    name="Persisted swim bouts",
                    marker=dict(color="#f59e0b", line=dict(width=0)),
                    opacity=0.22,
                    customdata=np.column_stack([bout_ids[valid], starts[valid], stops[valid]]),
                    hovertemplate=(
                        "bout=%{customdata[0]}<br>start=%{customdata[1]:.3f}s"
                        "<br>end=%{customdata[2]:.3f}s<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
    figure.update_yaxes(
        range=[0, 1],
        showgrid=False,
        showticklabels=False,
        title_text="Bout intervals",
        row=1,
        col=1,
        secondary_y=True,
    )

    if has_position:
        for name, color in (("x", "#2563eb"), ("y", "#dc2626")):
            figure.add_trace(
                go.Scattergl(
                    x=position_frame["time_s"],
                    y=position_frame[name],
                    mode="lines",
                    name=f"position_{name}",
                    line=dict(color=color, width=1.2),
                ),
                row=2,
                col=1,
            )
        unit = (
            str(position_frame["unit"][0])
            if "unit" in position_frame.columns and position_frame.height
            else "coordinate units"
        )
        figure.update_yaxes(title_text=f"Position ({unit})", row=2, col=1)

    figure.update_xaxes(title_text="Time (s)", row=2 if has_position else 1, col=1)
    figure.update_yaxes(title_text="Tail value", row=1, col=1, secondary_y=False)
    figure.update_layout(
        barmode="overlay",
        height=650 if has_position else 470,
        margin=dict(l=65, r=60, t=70, b=55),
        legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="left", x=0.0),
    )
    return figure


def build_tail_kinematics_figures(
    go: Any,
    *,
    projection: Any,
) -> Mapping[str, Any]:
    """Build figures without a pandas/Arrow bridge or hidden array reads."""

    frame = projection.frame.collect()
    metadata = projection.metadata
    time_s = frame["time_s"].to_numpy() if "time_s" in frame.columns else np.asarray([])
    figures: dict[str, Any] = {}

    angle_columns = tuple(metadata.get("angle_columns", ()))
    angle_sample_s = np.asarray(metadata.get("angle_sample_s", ()), dtype=np.float32)
    if angle_columns and angle_sample_s.size == len(angle_columns):
        figures["angle_kymograph"] = _heatmap_figure(
            go,
            time_s=time_s,
            sample_s=angle_sample_s,
            values=_numeric_matrix(frame, angle_columns),
            title="Body-frame local tail tangent angles",
            colorbar_title="Angle (deg)",
        )

    curvature_columns = tuple(metadata.get("curvature_columns", ()))
    curvature_sample_s = np.asarray(metadata.get("curvature_sample_s", ()), dtype=np.float32)
    if curvature_columns and curvature_sample_s.size == len(curvature_columns):
        figures["curvature_kymograph"] = _heatmap_figure(
            go,
            time_s=time_s,
            sample_s=curvature_sample_s,
            values=_numeric_matrix(frame, curvature_columns),
            title="Dense subject-shape spline curvature",
            colorbar_title="Curvature (px⁻¹)",
        )

    bout_lazy = projection.related_frames.get("bout_intervals")
    position_lazy = projection.related_frames.get("position_trace")
    figures["synchronized_traces"] = _synchronized_trace_figure(
        go,
        frame=frame,
        scalar_columns=tuple(metadata.get("scalar_columns", ())),
        bout_frame=bout_lazy.collect() if bout_lazy is not None else pl.DataFrame(),
        position_frame=position_lazy.collect() if position_lazy is not None else pl.DataFrame(),
    )
    return figures


def _sampling_message(fps: float | None) -> tuple[str, str]:
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return (
            "warn",
            "Recording FPS is unavailable. Tail-beat frequency and phase interpretation "
            "are disabled.",
        )
    nyquist = fps / 2.0
    lower, upper = TYPICAL_LARVAL_TAIL_BEAT_HZ
    if nyquist < lower:
        return (
            "danger",
            f"This recording is {fps:g} Hz (Nyquist {nyquist:g} Hz), below the typical "
            f"{lower:g}–{upper:g} Hz larval tail-beat band. The posture traces are valid "
            "at acquired frames, but oscillation frequency, phase, and wave-speed estimates "
            "would be aliased and are intentionally not computed.",
        )
    if nyquist < upper:
        return (
            "warn",
            f"This recording is {fps:g} Hz (Nyquist {nyquist:g} Hz), so only part of the "
            f"typical {lower:g}–{upper:g} Hz larval tail-beat band is resolvable.",
        )
    return (
        "info",
        f"This recording is {fps:g} Hz (Nyquist {nyquist:g} Hz). This viewer exposes "
        "posture and curvature only; it does not infer tail-beat frequency, phase, or wave speed.",
    )


def build_tail_kinematics_output(
    mo: Any,
    go: Any,
    *,
    projection: Any,
) -> Any:
    figures = build_tail_kinematics_figures(go, projection=projection)
    frame = projection.frame.collect()
    valid_fraction = (
        float(frame["valid"].cast(pl.Boolean, strict=False).mean() or 0.0)
        if "valid" in frame.columns and frame.height
        else 0.0
    )
    fps = projection.metadata.get("fps")
    stats = mo.hstack(
        [
            mo.stat(label="Rows projected", value=f"{projection.row_count:,}"),
            mo.stat(label="Valid tail rows", value=f"{100.0 * valid_fraction:.1f}%"),
            mo.stat(label="FPS", value=f"{float(fps):g}" if fps is not None else "unknown"),
            mo.stat(label="Zarr read ms", value=f"{projection.load_duration_ms:.1f}"),
        ]
    )
    warning_kind, warning_text = _sampling_message(
        float(fps) if fps is not None else None
    )
    warning = (
        mo.callout(mo.md(warning_text), kind=warning_kind)
        if hasattr(mo, "callout")
        else mo.md(warning_text)
    )
    pieces: list[Any] = [
        stats,
        warning,
        mo.md(
            f"{projection.note} Source arrays: "
            f"`{', '.join(projection.source_paths) or 'none'}`"
        ),
        mo.md(
            "The angle surface is the canonical 10-position body-frame tangent representation. "
            "The curvature surface is the exact 32-position subject-shape source named by the "
            "tail run. Missing or invalid rows remain gaps; the viewer does not interpolate them."
        ),
    ]
    companion_notes = projection.metadata.get("companion_notes", ())
    if companion_notes:
        pieces.append(
            mo.md(
                "Companion availability: "
                + "; ".join(str(note) for note in companion_notes)
            )
        )
    for name in ("angle_kymograph", "curvature_kymograph", "synchronized_traces"):
        figure = figures.get(name)
        if figure is not None:
            pieces.append(figure)

    provenance = projection.metadata.get("provenance", {})
    if isinstance(provenance, Mapping) and provenance:
        pieces.extend(
            [
                mo.md("### Persisted computation contract"),
                (
                    mo.tree(dict(provenance), label="Tail computation and source provenance")
                    if hasattr(mo, "tree")
                    else mo.md(f"Provenance: `{dict(provenance)}`")
                ),
            ]
        )

    pieces.append(mo.md("### Persisted snapshots"))
    pngs = projection.metadata.get("persisted_pngs", ())
    if pngs:
        for index, artifact in enumerate(pngs):
            if artifact.get("bytes"):
                pieces.extend(
                    [
                        mo.md(f"`{artifact.get('path')}`"),
                        png_bytes_to_markdown_image(
                            mo,
                            artifact["bytes"],
                            alt_text=f"tail-kinematics persisted snapshot {index + 1}",
                        ),
                    ]
                )
            else:
                pieces.append(
                    mo.md(
                        f"Persisted snapshot could not be loaded: `{artifact.get('path')}` — "
                        f"`{artifact.get('error')}`"
                    )
                )
    else:
        pieces.append(
            mo.md(
                "No analysis-owned tail-kinematics PNG is persisted for this run. The "
                "interactive views above are rendered read-only from canonical arrays."
            )
        )
    return mo.vstack(pieces)


__all__ = [
    "TYPICAL_LARVAL_TAIL_BEAT_HZ",
    "build_tail_kinematics_figures",
    "build_tail_kinematics_output",
]
