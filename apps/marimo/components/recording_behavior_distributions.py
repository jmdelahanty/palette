"""Read-only Marimo component for sealed recording-level distributions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fisheye.analysis_workflows.recording_behavior_distribution_storage import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    PARENT_PATH,
    SCHEMA_ID,
    SCHEMA_VERSION,
    RecordingBehaviorDistributionSourceHandle,
    load_recording_behavior_distribution_source_handle,
)
from fisheye.group_statistics.recording_behavior_distribution_views import (
    RecordingBehaviorDistributionView,
    available_recording_distribution_metrics,
    available_recording_distribution_scopes,
    build_recording_behavior_distribution_view,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    distribution_metric_display_text,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


PROVIDER_ID = "recording_behavior_distributions"
ANALYSIS_ID = "distributions"
RENDERER_ID = "palette-recording-behavior-distributions-v1"
_FORBIDDEN_SELECTORS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "selected_run",
    }
)
_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
)


@dataclass(frozen=True, slots=True)
class RecordingBehaviorDistributionOption:
    zarr_path: Path
    run_path: str
    run_name: str
    label: str
    recording_id: str
    manifest_sha256: str
    result_record_sha256: str


def _group_names(group: Any) -> tuple[str, ...]:
    try:
        return tuple(sorted(str(name) for name in group.group_keys()))
    except AttributeError:
        return tuple(
            sorted(
                str(name)
                for name in group.keys()
                if hasattr(group[name], "attrs")
            )
        )


def discover_recording_behavior_distribution_options(
    zarr_path: str | Path,
    *,
    run_path_filter: str | None = None,
) -> tuple[RecordingBehaviorDistributionOption, ...]:
    """Discover immutable candidates from consolidated metadata only."""

    archive = Path(zarr_path).expanduser().resolve()
    try:
        root = open_zarr_root(archive, mode="r", use_consolidated=True)
        parent = root[PARENT_PATH]
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return ()
    if _FORBIDDEN_SELECTORS.intersection(parent.attrs):
        return ()
    options = []
    for run_name in _group_names(parent):
        if run_name.casefold() in _FORBIDDEN_SELECTORS:
            continue
        run = parent[run_name]
        attrs = run.attrs
        manifest = attrs.get(MANIFEST_ATTR)
        if not isinstance(manifest, Mapping):
            continue
        manifest = dict(manifest)
        run_path = f"{PARENT_PATH}/{run_name}"
        if run_path_filter is not None and str(run_path_filter).strip("/") != run_path:
            continue
        if (
            attrs.get("schema_id") != SCHEMA_ID
            or attrs.get("schema_version") != SCHEMA_VERSION
            or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or attrs.get("stage_selector_eligible") is not False
            or attrs.get("production_authority") is not False
            or attrs.get("selection") != "none"
            or manifest.get("run_path") != run_path
            or manifest.get("run_name") != run_name
            or manifest.get("selector_eligible") is not False
            or attrs.get(MANIFEST_DIGEST_ATTR) != canonical_json_sha256(manifest)
        ):
            continue
        recording_id = str(manifest.get("recording_id") or "")
        result_digest = str(manifest.get("result_record_sha256") or "")
        if not recording_id or len(result_digest) != 64:
            continue
        digest = canonical_json_sha256(manifest)
        options.append(
            RecordingBehaviorDistributionOption(
                zarr_path=archive,
                run_path=run_path,
                run_name=run_name,
                label=f"{run_name} · {digest[:12]}",
                recording_id=recording_id,
                manifest_sha256=digest,
                result_record_sha256=result_digest,
            )
        )
    return tuple(options)


def load_recording_behavior_distribution_handle(
    zarr_path: str | Path,
    option: RecordingBehaviorDistributionOption,
) -> RecordingBehaviorDistributionSourceHandle:
    """Deep-validate the small exact table carrier selected by the user."""

    archive = Path(zarr_path).expanduser().resolve()
    if (
        type(option) is not RecordingBehaviorDistributionOption
        or option.zarr_path != archive
        or option.run_path != f"{PARENT_PATH}/{option.run_name}"
    ):
        raise ValueError("Recording-distribution option differs from the selected Zarr.")
    handle = load_recording_behavior_distribution_source_handle(
        archive,
        run_name=option.run_name,
        expected_recording_id=option.recording_id,
    )
    if (
        canonical_json_sha256(dict(handle.manifest)) != option.manifest_sha256
        or handle.result_record["record_sha256"] != option.result_record_sha256
    ):
        raise ValueError("Recording-distribution run changed after discovery.")
    return handle


def recording_distribution_metric_options(
    handle: RecordingBehaviorDistributionSourceHandle,
) -> Mapping[str, str]:
    """Return human-readable labels mapped to exact metric IDs."""

    result: dict[str, str] = {}
    for row in available_recording_distribution_metrics(handle):
        metric_id = str(row["metric_id"])
        label, _definition = distribution_metric_display_text(row)
        if label in result:
            label = f"{label} [{metric_id}]"
        result[label] = metric_id
    return result


def recording_distribution_weighting_options(
    handle: RecordingBehaviorDistributionSourceHandle,
    metric_id: str,
) -> tuple[str, ...]:
    matches = [
        row
        for row in available_recording_distribution_metrics(handle)
        if row["metric_id"] == metric_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown recording distribution metric: {metric_id!r}")
    return tuple(str(value) for value in matches[0]["weighting_ids"])


def recording_distribution_scope_options(
    handle: RecordingBehaviorDistributionSourceHandle,
) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for row in available_recording_distribution_scopes(handle):
        scope_id = str(row["scope_id"])
        label = str(row["scope_label"])
        if label in result:
            label = f"{label} [{scope_id}]"
        result[label] = scope_id
    return result


def load_recording_behavior_distribution_projection(
    handle: RecordingBehaviorDistributionSourceHandle,
    *,
    metric_id: str,
    weighting_id: str,
    scope_ids: Sequence[str] = (),
) -> RecordingBehaviorDistributionView:
    return build_recording_behavior_distribution_view(
        handle,
        metric_id=metric_id,
        weighting_id=weighting_id,
        scope_ids=scope_ids,
    )


def recording_behavior_distribution_figure(
    view: RecordingBehaviorDistributionView,
    *,
    maximum_columns: int = 4,
) -> go.Figure:
    """Render the exact shared projection with arbitrary persisted scopes."""

    if type(view) is not RecordingBehaviorDistributionView:
        raise TypeError("view must be one RecordingBehaviorDistributionView")
    metric_label, metric_definition = distribution_metric_display_text(view.metric)
    scopes = tuple(sorted(view.scopes, key=lambda row: int(row["order"])))
    if not scopes or maximum_columns < 1:
        raise ValueError("A positive panel width and at least one scope are required.")
    columns = min(maximum_columns, len(scopes))
    rows = math.ceil(len(scopes) / columns)
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[str(scope["scope_label"]) for scope in scopes],
    )
    identities = sorted(
        {
            (series.group_key_sha256, series.source_identity_key_sha256)
            for series in view.series
        }
    )
    colors = {
        identity: _COLORS[index % len(_COLORS)]
        for index, identity in enumerate(identities)
    }
    bar_opacity = 0.78 if len(identities) == 1 else 0.42
    shown: set[tuple[str, str]] = set()
    for panel_index, scope in enumerate(scopes):
        row_index = panel_index // columns + 1
        column_index = panel_index % columns + 1
        finite_evidence = False
        for series in view.series:
            if series.scope_id != scope["scope_id"]:
                continue
            identity = (series.group_key_sha256, series.source_identity_key_sha256)
            color = colors[identity]
            values = series.fraction * 100.0
            if series.bin_center.size and np.any(np.isfinite(values)):
                finite_evidence = True
                figure.add_trace(
                    go.Bar(
                        x=series.bin_center,
                        y=values,
                        width=series.bin_right - series.bin_left,
                        name=series.label,
                        legendgroup="|".join(identity),
                        showlegend=identity not in shown,
                        opacity=bar_opacity,
                        marker={
                            "color": color,
                            "line": {"color": color, "width": 1.0},
                        },
                        customdata=np.column_stack(
                            (
                                series.bin_left,
                                series.bin_right,
                                series.bin_count,
                                series.bin_weight,
                            )
                        ),
                        hovertemplate=(
                            f"{series.label}"
                            "<br>bin left=%{customdata[0]:.4g}"
                            "<br>bin right=%{customdata[1]:.4g}"
                            "<br>count=%{customdata[2]:,.0f}"
                            "<br>weight=%{customdata[3]:.6g}"
                            "<br>probability=%{y:.4g}%<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=column_index,
                )
                shown.add(identity)
        if not finite_evidence:
            figure.add_annotation(
                text="No valid evidence",
                x=0.5,
                y=0.5,
                xref=("x domain" if panel_index == 0 else f"x{panel_index + 1} domain"),
                yref=("y domain" if panel_index == 0 else f"y{panel_index + 1} domain"),
                showarrow=False,
            )
        figure.update_xaxes(
            title_text=f"{view.metric['unit']}",
            type="log" if view.metric.get("axis_scale") == "log10" else "linear",
            row=row_index,
            col=column_index,
        )
        figure.update_yaxes(
            title_text="Probability per bin (%)",
            row=row_index,
            col=column_index,
        )
    figure.update_layout(
        title=(
            f"{metric_label} · {view.weighting_id} weighted · persisted bins"
            + (
                ""
                if metric_definition is None
                else f"<br><sup>{metric_definition}</sup>"
            )
        ),
        height=max(440, 360 * rows),
        barmode="overlay",
        bargap=0,
        bargroupgap=0,
        hovermode="closest",
        template="plotly_white",
        meta={
            "recording_id": view.recording_id,
            "distribution_run_id": view.distribution_run_id,
            "distribution_manifest_sha256": view.manifest_sha256,
            "result_record_sha256": view.result_record_sha256,
            "view_sha256": view.view_sha256,
            "viewer_rebinning": "prohibited",
            "viewer_clipping": "none",
            "histogram_rendering": "exact_bin_width_bars_v1",
        },
    )
    return figure


def build_recording_behavior_distribution_output(
    mo: Any,
    view: RecordingBehaviorDistributionView,
) -> Any:
    figure = recording_behavior_distribution_figure(view)
    support_rows = [
        {
            "scope": series.scope_label,
            "group": series.label,
            "valid": int(series.support["valid_count"]),
            "excluded": int(series.support["excluded_count"]),
            "uncovered": int(series.support["uncovered_count"]),
            "denominator": float(series.support["denominator_weight"]),
            "unit": series.weight_unit,
            "event_rate_per_valid_min": series.support.get(
                "event_rate_per_valid_min"
            ),
        }
        for series in view.series
    ]
    provenance_rows = [
        {"field": "distribution_run_id", "value": view.distribution_run_id},
        {"field": "manifest_sha256", "value": view.manifest_sha256},
        {"field": "result_record_sha256", "value": view.result_record_sha256},
        {"field": "verification_digest", "value": view.verification_digest},
        {"field": "view_sha256", "value": view.view_sha256},
        {
            "field": "histogram_recipe_sha256",
            "value": view.metric["histogram_recipe_sha256"],
        },
    ]
    return mo.vstack(
        [
            mo.callout(
                "This view uses the exact persisted bins, memberships, and "
                "denominators. Viewer-side rebinning and clipping are disabled.",
                kind="info",
            ),
            mo.ui.plotly(figure),
            mo.md("### Support and exclusions"),
            mo.ui.table(support_rows, selection=None),
            mo.md("### Exact provenance"),
            mo.ui.table(provenance_rows, selection=None),
        ]
    )


__all__ = [
    "ANALYSIS_ID",
    "PROVIDER_ID",
    "RENDERER_ID",
    "RecordingBehaviorDistributionOption",
    "build_recording_behavior_distribution_output",
    "discover_recording_behavior_distribution_options",
    "load_recording_behavior_distribution_handle",
    "load_recording_behavior_distribution_projection",
    "recording_behavior_distribution_figure",
    "recording_distribution_metric_options",
    "recording_distribution_scope_options",
    "recording_distribution_weighting_options",
]
