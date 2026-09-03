"""Render exact persisted near-field visits and seal an external plot receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Collection, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    load_composable_chaser_successor_source_handle,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.chaser_near_field_visits import (
    VISIT_TRAJECTORY_ARRAY_NAMES,
    ChaserNearFieldVisitViewError,
    NearFieldVisitPanel,
    NearFieldVisitTrajectory,
    NearFieldVisitTrajectoryView,
    validated_near_field_visit_trajectory_view,
)


RECEIPT_SCHEMA_ID = "palette.analysis.chaser_near_field_visits.plot_receipt"
RECEIPT_SCHEMA_VERSION = 1
PLOT_RECIPE_ID = "persisted_exact_near_field_visit_trajectories_v1"
PLOT_DPI = 180
PLOT_COLUMNS = 3
PANEL_SIZE_INCHES = (5.2, 4.8)
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_VISIT_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)


class ChaserNearFieldVisitPlotError(ValueError):
    """Raised when an exact persisted visit view cannot be rendered safely."""


def _fail(message: str) -> None:
    raise ChaserNearFieldVisitPlotError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _panel_key(panel: NearFieldVisitPanel) -> tuple[int, int, int]:
    return (
        panel.epoch_role_code,
        panel.epoch_window_id,
        panel.chaser_identity_code,
    )


def _visit_key(visit: NearFieldVisitTrajectory) -> tuple[int, int, int]:
    return (
        visit.epoch_role_code,
        visit.epoch_window_id,
        visit.chaser_identity_code,
    )


def _normalized_filter(values: Collection[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    result = tuple(sorted({int(value) for value in values}))
    if not result:
        _fail("An explicit visit display filter must not be empty.")
    return result


def _select(
    view: NearFieldVisitTrajectoryView,
    *,
    epoch_role_codes: Collection[int] | None,
    chaser_identity_codes: Collection[int] | None,
    include_censored: bool,
    include_short: bool,
) -> tuple[tuple[NearFieldVisitPanel, ...], tuple[NearFieldVisitTrajectory, ...]]:
    epochs = _normalized_filter(epoch_role_codes)
    chasers = _normalized_filter(chaser_identity_codes)
    panels = tuple(
        panel
        for panel in view.panels
        if (epochs is None or panel.epoch_role_code in epochs)
        and (chasers is None or panel.chaser_identity_code in chasers)
    )
    if not panels:
        _fail("Visit display filters match no persisted epoch/chaser summary rows.")
    panel_keys = {_panel_key(panel) for panel in panels}
    visits = tuple(
        visit
        for visit in view.visits
        if _visit_key(visit) in panel_keys
        and (include_censored or visit.complete)
        and (include_short or visit.quality != "short_visit_retained")
    )
    return panels, visits


def near_field_visit_plot_parameters(
    view: NearFieldVisitTrajectoryView,
    *,
    panels: Sequence[NearFieldVisitPanel],
    visits: Sequence[NearFieldVisitTrajectory],
    epoch_role_codes: Collection[int] | None,
    chaser_identity_codes: Collection[int] | None,
    include_censored: bool,
    include_short: bool,
    limit_mm: float,
    columns: int,
) -> dict[str, Any]:
    """Return the complete scientific-selection and display recipe."""

    return json_attr_safe(
        {
            "scientific_source": {
                "array_names": list(VISIT_TRAJECTORY_ARRAY_NAMES),
                "visit_membership": "persisted_ragged_samples_only",
                "segmentation": "prohibited",
                "interpolation": "prohibited",
                "coordinate_convention": view.coordinate_convention,
                "near_zone_radius_mm": view.near_zone_radius_mm,
                "near_entry_radius_mm": view.near_entry_radius_mm,
                "near_exit_radius_mm": view.near_exit_radius_mm,
            },
            "selection": {
                "epoch_role_codes": (list(_normalized_filter(epoch_role_codes) or [])),
                "chaser_identity_codes": (
                    list(_normalized_filter(chaser_identity_codes) or [])
                ),
                "empty_filter_means_all": True,
                "include_censored": bool(include_censored),
                "include_short": bool(include_short),
                "panel_keys": [
                    {
                        "epoch_role_code": panel.epoch_role_code,
                        "epoch_window_id": panel.epoch_window_id,
                        "chaser_identity_code": panel.chaser_identity_code,
                        "behavior_role_code": panel.behavior_role_code,
                    }
                    for panel in panels
                ],
                "visit_row_ids": [visit.visit_row_id for visit in visits],
                "visit_key_sha256": [visit.visit_key_sha256 for visit in visits],
            },
            "rendering": {
                "plot_recipe_id": PLOT_RECIPE_ID,
                "columns": int(columns),
                "panel_size_inches": list(PANEL_SIZE_INCHES),
                "png_dpi": PLOT_DPI,
                "canonical_axis_limit_mm": float(limit_mm),
                "canonical_y_display": "positive_down",
                "arena_wall": ("not_drawn_because_reference_may_move_within_a_visit"),
                "chaser_marker": {
                    "shape": "circle",
                    "fill": "neutral_gray",
                    "appearance_authority_claimed": False,
                },
                "visit_color": "ordinal_palette_display_only",
                "complete_line_style": "solid",
                "censored_line_style": "dashed",
                "short_visit_alpha": 0.45,
                "ordinary_visit_alpha": 0.9,
                "start_marker": "circle",
                "last_retained_marker": "triangle",
                "cpa_marker": "x",
                "distance_time_origin": "first_retained_visit_sample",
            },
            "viewer_policy": {
                "scientific_recomputation": False,
                "visit_reconstruction": "prohibited",
                "gap_bridging": "prohibited",
                "short_visit_dropping_by_default": False,
                "censored_visit_dropping_by_default": False,
                "chaser_role_inference": "prohibited",
                "stimulus_color_inference": "prohibited",
            },
        }
    )


def _figure_grid(panel_count: int, columns: int) -> tuple[Any, np.ndarray]:
    n_columns = min(columns, max(1, panel_count))
    n_rows = max(1, math.ceil(panel_count / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(PANEL_SIZE_INCHES[0] * n_columns, PANEL_SIZE_INCHES[1] * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    return figure, axes.reshape(-1)


def _panel_title(panel: NearFieldVisitPanel, displayed: int) -> str:
    return (
        f"{panel.epoch_role} · window {panel.epoch_window_id}\n"
        f"{panel.chaser_identity} · {panel.behavior_role}\n"
        f"persisted {panel.total_visit_count} · displayed {displayed} · "
        f"complete {panel.complete_visit_count} · censored {panel.censored_visit_count} · "
        f"short {panel.short_visit_count}"
    )


def _style(visit: NearFieldVisitTrajectory, index: int) -> dict[str, Any]:
    return {
        "color": _VISIT_COLORS[index % len(_VISIT_COLORS)],
        "linestyle": "-" if visit.complete else "--",
        "alpha": 0.45 if visit.quality == "short_visit_retained" else 0.9,
        "linewidth": 1.6,
    }


def _visits_for_panel(
    panel: NearFieldVisitPanel,
    visits: Sequence[NearFieldVisitTrajectory],
) -> list[NearFieldVisitTrajectory]:
    return sorted(
        (visit for visit in visits if _visit_key(visit) == _panel_key(panel)),
        key=lambda visit: (visit.visit_ordinal, visit.visit_row_id),
    )


def _trajectory_figure(
    view: NearFieldVisitTrajectoryView,
    panels: Sequence[NearFieldVisitPanel],
    visits: Sequence[NearFieldVisitTrajectory],
    *,
    limit_mm: float,
    columns: int,
) -> Any:
    figure, axes = _figure_grid(len(panels), columns)
    for axis, panel in zip(axes, panels, strict=False):
        selected = _visits_for_panel(panel, visits)
        for index, visit in enumerate(selected):
            style = _style(visit, index)
            x = np.where(visit.canonical_valid, visit.canonical_x_mm, np.nan)
            y = np.where(visit.canonical_valid, visit.canonical_y_mm, np.nan)
            axis.plot(x, y, **style)
            valid_rows = np.flatnonzero(visit.canonical_valid)
            if valid_rows.size:
                first = int(valid_rows[0])
                last = int(valid_rows[-1])
                axis.scatter(
                    [x[first]],
                    [y[first]],
                    marker="o",
                    s=20,
                    color=style["color"],
                    alpha=style["alpha"],
                    zorder=4,
                )
                axis.scatter(
                    [x[last]],
                    [y[last]],
                    marker="^",
                    s=24,
                    color=style["color"],
                    alpha=style["alpha"],
                    zorder=4,
                )
            cpa = visit.cpa_sample_ordinal
            if visit.canonical_valid[cpa]:
                axis.scatter(
                    [x[cpa]],
                    [y[cpa]],
                    marker="x",
                    s=28,
                    color=style["color"],
                    alpha=style["alpha"],
                    zorder=5,
                )
        if math.isclose(
            view.near_zone_radius_mm,
            view.near_entry_radius_mm,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            axis.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    view.near_entry_radius_mm,
                    fill=False,
                    color="#111111",
                    linewidth=1.0,
                )
            )
        else:
            axis.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    view.near_zone_radius_mm,
                    fill=False,
                    color="#777777",
                    linestyle=":",
                    linewidth=1.0,
                )
            )
            axis.add_patch(
                plt.Circle(
                    (0.0, 0.0),
                    view.near_entry_radius_mm,
                    fill=False,
                    color="#111111",
                    linewidth=1.0,
                )
            )
        axis.add_patch(
            plt.Circle(
                (0.0, 0.0),
                view.near_exit_radius_mm,
                fill=False,
                color="#555555",
                linestyle="--",
                linewidth=1.0,
            )
        )
        axis.scatter(
            [0.0],
            [0.0],
            marker="o",
            s=42,
            facecolor="#777777",
            edgecolor="#111111",
            zorder=6,
        )
        axis.annotate(
            "arena centre direction",
            xy=(0.92 * limit_mm, 0.0),
            xytext=(0.42 * limit_mm, -0.10 * limit_mm),
            arrowprops={"arrowstyle": "->", "color": "#555555"},
            fontsize=7,
            color="#555555",
        )
        axis.set_title(_panel_title(panel, len(selected)), fontsize=9)
        axis.set_xlim(-limit_mm, limit_mm)
        axis.set_ylim(-limit_mm, limit_mm)
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("canonical x (mm; arena centre → +x)")
        axis.set_ylabel("canonical y (mm; source +y down)")
        axis.grid(alpha=0.15)
    for axis in axes[len(panels) :]:
        axis.set_visible(False)
    legend = [
        Line2D([], [], color="#333333", linestyle="-", label="complete visit"),
        Line2D([], [], color="#333333", linestyle="--", label="censored visit"),
        Line2D([], [], color="#333333", alpha=0.45, label="short visit retained"),
        Line2D([], [], color="#333333", marker="o", linestyle="", label="first sample"),
        Line2D(
            [],
            [],
            color="#333333",
            marker="^",
            linestyle="",
            label="last retained sample",
        ),
        Line2D(
            [], [], color="#333333", marker="x", linestyle="", label="closest approach"
        ),
        Line2D(
            [],
            [],
            color="#777777",
            marker="o",
            linestyle="",
            label="chaser origin; neutral fill",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncols=4,
        fontsize=8,
    )
    figure.suptitle(
        f"Exact near-field visit trajectories · {view.recording_id}\n"
        "persisted membership and canonical samples · no interpolation or role inference",
        fontsize=13,
    )
    return figure


def _distance_figure(
    view: NearFieldVisitTrajectoryView,
    panels: Sequence[NearFieldVisitPanel],
    visits: Sequence[NearFieldVisitTrajectory],
    *,
    columns: int,
) -> Any:
    figure, axes = _figure_grid(len(panels), columns)
    for axis, panel in zip(axes, panels, strict=False):
        selected = _visits_for_panel(panel, visits)
        for index, visit in enumerate(selected):
            style = _style(visit, index)
            axis.plot(visit.time_from_first_sample_s, visit.distance_mm, **style)
            cpa = visit.cpa_sample_ordinal
            axis.scatter(
                [visit.time_from_first_sample_s[cpa]],
                [visit.distance_mm[cpa]],
                marker="x",
                s=28,
                color=style["color"],
                alpha=style["alpha"],
                zorder=4,
            )
        axis.axhline(
            view.near_entry_radius_mm,
            color="#111111",
            linewidth=1.0,
            label="entry threshold",
        )
        axis.axhline(
            view.near_exit_radius_mm,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            label="exit threshold",
        )
        if not math.isclose(
            view.near_zone_radius_mm,
            view.near_entry_radius_mm,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            axis.axhline(
                view.near_zone_radius_mm,
                color="#777777",
                linestyle=":",
                linewidth=1.0,
                label="near-zone threshold",
            )
        axis.set_title(_panel_title(panel, len(selected)), fontsize=9)
        axis.set_xlabel("time from first retained visit sample (s)")
        axis.set_ylabel("fish–chaser distance (mm)")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.15)
    for axis in axes[len(panels) :]:
        axis.set_visible(False)
    legend = [
        Line2D([], [], color="#333333", linestyle="-", label="complete visit"),
        Line2D([], [], color="#333333", linestyle="--", label="censored visit"),
        Line2D([], [], color="#333333", alpha=0.45, label="short visit retained"),
        Line2D([], [], color="#111111", linestyle="-", label="entry threshold"),
        Line2D([], [], color="#555555", linestyle="--", label="exit threshold"),
        Line2D(
            [], [], color="#333333", marker="x", linestyle="", label="closest approach"
        ),
    ]
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncols=3,
        fontsize=8,
    )
    figure.suptitle(
        f"Exact near-field visit distance traces · {view.recording_id}\n"
        "time origin is each visit's first retained sample; censored boundaries remain visible",
        fontsize=13,
    )
    return figure


def _save_figure(figure: Any, stem: Path) -> tuple[Path, Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    temporary_png = png.with_name(f".{png.name}.tmp")
    temporary_pdf = pdf.with_name(f".{pdf.name}.tmp")
    try:
        figure.savefig(
            temporary_png,
            dpi=PLOT_DPI,
            format="png",
            bbox_inches="tight",
            pad_inches=0.12,
        )
        figure.savefig(
            temporary_pdf,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.12,
        )
        os.replace(temporary_png, png)
        os.replace(temporary_pdf, pdf)
    finally:
        plt.close(figure)
        temporary_png.unlink(missing_ok=True)
        temporary_pdf.unlink(missing_ok=True)
    return png, pdf


def render_near_field_visit_trajectories(
    handle: Any,
    *,
    output_stem: str | Path,
    epoch_role_codes: Collection[int] | None = None,
    chaser_identity_codes: Collection[int] | None = None,
    include_censored: bool = True,
    include_short: bool = True,
    limit_mm: float | None = None,
    columns: int = PLOT_COLUMNS,
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    """Render canonical XY and distance-time views from exact visit samples."""

    if columns <= 0:
        _fail("Plot columns must be positive.")
    try:
        view = validated_near_field_visit_trajectory_view(handle)
    except ChaserNearFieldVisitViewError as exc:
        raise ChaserNearFieldVisitPlotError(str(exc)) from exc
    panels, visits = _select(
        view,
        epoch_role_codes=epoch_role_codes,
        chaser_identity_codes=chaser_identity_codes,
        include_censored=include_censored,
        include_short=include_short,
    )
    resolved_limit = (
        max(8.0, 1.25 * view.near_exit_radius_mm)
        if limit_mm is None
        else float(limit_mm)
    )
    if not math.isfinite(resolved_limit) or resolved_limit <= view.near_exit_radius_mm:
        _fail("Canonical axis limit must be finite and greater than the exit radius.")
    parameters = near_field_visit_plot_parameters(
        view,
        panels=panels,
        visits=visits,
        epoch_role_codes=epoch_role_codes,
        chaser_identity_codes=chaser_identity_codes,
        include_censored=include_censored,
        include_short=include_short,
        limit_mm=resolved_limit,
        columns=columns,
    )
    stem = Path(output_stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    trajectory_files = _save_figure(
        _trajectory_figure(
            view,
            panels,
            visits,
            limit_mm=resolved_limit,
            columns=columns,
        ),
        stem.with_name(f"{stem.name}_trajectories"),
    )
    distance_files = _save_figure(
        _distance_figure(view, panels, visits, columns=columns),
        stem.with_name(f"{stem.name}_distance_traces"),
    )
    return (*trajectory_files, *distance_files), parameters


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bundle-name")
    parser.add_argument("--expected-recording-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-validation-receipt")
    parser.add_argument("--epoch-role-code", type=int, action="append")
    parser.add_argument("--chaser-identity-code", type=int, action="append")
    parser.add_argument("--exclude-censored", action="store_true")
    parser.add_argument("--exclude-short", action="store_true")
    parser.add_argument("--limit-mm", type=float)
    parser.add_argument("--columns", type=int, default=PLOT_COLUMNS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if _RUN_NAME_RE.fullmatch(args.run_name) is None:
        _fail("run_name must be one exact non-selector child name.")
    bundle_name = args.bundle_name or args.run_name
    if _RUN_NAME_RE.fullmatch(bundle_name) is None:
        _fail("bundle_name must be one exact safe output basename.")
    archive = args.analysis_zarr.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    stem = output_dir / f"{bundle_name}_near_field_visits"
    receipt_path = output_dir / f"{bundle_name}_near_field_visit_plot_receipt.json"
    expected = (
        stem.with_name(f"{stem.name}_trajectories").with_suffix(".png"),
        stem.with_name(f"{stem.name}_trajectories").with_suffix(".pdf"),
        stem.with_name(f"{stem.name}_distance_traces").with_suffix(".png"),
        stem.with_name(f"{stem.name}_distance_traces").with_suffix(".pdf"),
        receipt_path,
    )
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("Near-field visit plot output already exists.")

    receipt_bound = args.source_validation_receipt is not None
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_near_field_visits",
        run_name=args.run_name,
        expected_recording_id=args.expected_recording_id,
        use_consolidated=True,
        deep_audit=not receipt_bound,
        direct_validation_receipt=args.source_validation_receipt,
        required_array_names=(VISIT_TRAJECTORY_ARRAY_NAMES if receipt_bound else None),
    )
    files, parameters = render_near_field_visit_trajectories(
        handle,
        output_stem=stem,
        epoch_role_codes=args.epoch_role_code,
        chaser_identity_codes=args.chaser_identity_code,
        include_censored=not args.exclude_censored,
        include_short=not args.exclude_short,
        limit_mm=args.limit_mm,
        columns=args.columns,
    )
    output_records = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in files
    ]
    source_binding = {
        "successor_kind": handle.successor_kind,
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "scientific_payload_sha256": handle.scientific_payload_sha256,
        "verification_mode": handle.verification_mode,
        "verified_array_names": list(handle.verified_array_names),
        "validation_receipt_sha256": handle.receipt_digest,
    }
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "plot_recipe_id": PLOT_RECIPE_ID,
        "recording_id": handle.recording_id,
        "run_name": args.run_name,
        "bundle_name": bundle_name,
        "source_binding": source_binding,
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "outputs": output_records,
        "plot_parameters": parameters,
        "plot_parameters_sha256": canonical_json_sha256(parameters),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
    }
    receipt = {**body, "payload_sha256": canonical_json_sha256(body)}
    write_json_atomic(receipt_path, receipt)
    print(
        json.dumps(
            {**receipt, "receipt_path": str(receipt_path)}, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PLOT_RECIPE_ID",
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "ChaserNearFieldVisitPlotError",
    "near_field_visit_plot_parameters",
    "render_near_field_visit_trajectories",
]
