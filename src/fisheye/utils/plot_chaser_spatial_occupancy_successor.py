"""Render exact pre/training/post paired-provider spatial occupancy heatmaps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    ComposableChaserSuccessorSourceHandle,
    load_composable_chaser_successor_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.chaser_spatial_occupancy_display import (
    DEFAULT_DISPLAY_MODE_ID,
    DISPLAY_RECIPE_ID,
    STATIC_EXPORT_MODE_IDS,
    SpatialOccupancyDisplayMode,
    SpatialOccupancyDisplayPlan,
    build_spatial_occupancy_display_plan,
)


RECEIPT_SCHEMA_ID = "palette.analysis.chaser_spatial_occupancy.plot_receipt"
RECEIPT_SCHEMA_VERSION = 4
PLOT_RECIPE_ID = DISPLAY_RECIPE_ID
PLOT_DPI = 180
PLOT_FIGURE_SIZE_INCHES = (15.0, 15.0)
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SPATIAL_OCCUPANCY_PLOT_ARRAY_NAMES = (
    "candidate_frame_count",
    "in_arena_coverage_fraction_candidate",
    "in_arena_position_frame_count",
    "arena_bin_center_mask",
    "occupancy_count",
    "occupancy_density_valid_in_arena",
    "occupancy_fraction_candidate_epoch",
    "x_bin_edges_mm",
    "y_bin_edges_mm",
)


class ChaserSpatialOccupancyPlotError(ValueError):
    """Raised when spatial occupancy cannot be plotted exactly."""


def _fail(message: str) -> None:
    raise ChaserSpatialOccupancyPlotError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(handle: ComposableChaserSuccessorSourceHandle, name: str) -> np.ndarray:
    try:
        return np.asarray(handle.array(name))
    except KeyError:
        _fail(f"Spatial occupancy lacks required array {name!r}.")


def _registry(manifest: Mapping[str, Any], name: str) -> Mapping[str, str]:
    registries = manifest.get("identity_registries")
    if not isinstance(registries, Mapping) or not isinstance(
        registries.get(name), Mapping
    ):
        _fail(f"Spatial occupancy lacks {name!r} identity registry.")
    return {str(key): str(value) for key, value in registries[name].items()}


def _validate_plot_arrays(
    handle: ComposableChaserSuccessorSourceHandle,
) -> dict[str, Any]:
    if handle.successor_kind != "chaser_spatial_occupancy":
        _fail("Plot source is not a chaser spatial-occupancy successor.")
    try:
        handle.require_verified_authority()
        handle.require_verified_arrays(SPATIAL_OCCUPANCY_PLOT_ARRAY_NAMES)
    except (TypeError, ValueError) as exc:
        _fail(f"Spatial occupancy lacks verified plot authority: {exc}")
    scientific = handle.scientific_manifest
    dimensions = scientific.get("dimensions")
    if not isinstance(dimensions, Mapping):
        _fail("Spatial occupancy dimensions are absent.")
    expected = (
        int(dimensions["n_providers"]),
        int(dimensions["n_epochs"]),
        int(dimensions["grid_rows"]),
        int(dimensions["grid_columns"]),
    )
    if expected[:2] != (2, 3) or expected[2] <= 0 or expected[3] <= 0:
        _fail("Heatmap recipe requires two providers and three semantic epochs.")
    counts = _array(handle, "occupancy_count").astype(np.int64)
    density = _array(handle, "occupancy_density_valid_in_arena").astype(np.float64)
    fraction = _array(handle, "occupancy_fraction_candidate_epoch").astype(np.float64)
    if (
        counts.shape != expected
        or density.shape != expected
        or fraction.shape != expected
    ):
        _fail("Spatial occupancy grid arrays have inconsistent dimensions.")
    if (
        np.any(counts < 0)
        or not np.all(np.isfinite(density))
        or not np.all(np.isfinite(fraction))
        or np.any(density < 0)
        or np.any(fraction < 0)
    ):
        _fail("Spatial occupancy arrays contain invalid counts or fractions.")
    x_edges = _array(handle, "x_bin_edges_mm").astype(np.float64)
    y_edges = _array(handle, "y_bin_edges_mm").astype(np.float64)
    arena_mask = _array(handle, "arena_bin_center_mask").astype(bool)
    if (
        x_edges.shape != (expected[3] + 1,)
        or y_edges.shape != (expected[2] + 1,)
        or arena_mask.shape != expected[2:]
        or np.any(~np.isfinite(x_edges))
        or np.any(~np.isfinite(y_edges))
        or np.any(np.diff(x_edges) <= 0)
        or np.any(np.diff(y_edges) <= 0)
    ):
        _fail("Spatial occupancy bin edges or arena mask are invalid.")
    candidate = _array(handle, "candidate_frame_count").astype(np.int64)
    in_arena = _array(handle, "in_arena_position_frame_count").astype(np.int64)
    coverage = _array(handle, "in_arena_coverage_fraction_candidate").astype(np.float64)
    if (
        candidate.shape != (2, 3)
        or in_arena.shape != (2, 3)
        or coverage.shape != (2, 3)
    ):
        _fail("Spatial occupancy denominator arrays have invalid dimensions.")
    if np.any(candidate <= 0) or np.any(in_arena < 0) or np.any(in_arena > candidate):
        _fail("Spatial occupancy denominator counts are invalid.")
    if not np.allclose(counts.sum(axis=(2, 3)), in_arena, rtol=0, atol=0):
        _fail("Spatial occupancy counts do not conserve in-arena rows.")
    expected_density_sum = (in_arena > 0).astype(np.float64)
    if not np.allclose(
        density.sum(axis=(2, 3)), expected_density_sum, rtol=1e-10, atol=1e-12
    ):
        _fail("Conditional spatial occupancy densities are not normalized.")
    if not np.allclose(fraction.sum(axis=(2, 3)), coverage, rtol=1e-10, atol=1e-12):
        _fail("Candidate-normalized spatial occupancy does not equal coverage.")
    provider_registry = _registry(scientific, "provider_role")
    epoch_registry = _registry(scientific, "epoch_role")
    if provider_registry != {"0": "keypoint", "1": "detection"}:
        _fail("Spatial occupancy provider-role order is not keypoint/detection.")
    if epoch_registry != {
        "0": "chaser_pre",
        "1": "chaser_training",
        "2": "chaser_post",
    }:
        _fail("Spatial occupancy epoch-role order is not pre/training/post.")
    return {
        "counts": counts,
        "density": density,
        "fraction": fraction,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "arena_mask": arena_mask,
        "candidate": candidate,
        "in_arena": in_arena,
        "coverage": coverage,
        "provider_registry": provider_registry,
        "epoch_registry": epoch_registry,
    }


def _display_plan(
    handle: ComposableChaserSuccessorSourceHandle,
    values: Mapping[str, Any],
) -> SpatialOccupancyDisplayPlan:
    arena = handle.scientific_manifest.get("arena")
    grid = handle.scientific_manifest.get("grid")
    if not isinstance(arena, Mapping) or not isinstance(grid, Mapping):
        _fail("Spatial occupancy plot authorities are incomplete.")
    try:
        return build_spatial_occupancy_display_plan(
            counts=np.asarray(values["counts"]),
            density_valid_in_arena=np.asarray(values["density"]),
            fraction_candidate_epoch=np.asarray(values["fraction"]),
            x_edges_mm=np.asarray(values["x_edges"]),
            y_edges_mm=np.asarray(values["y_edges"]),
            arena_bin_center_mask=np.asarray(values["arena_mask"]),
            in_arena_denominator=np.asarray(values["in_arena"]),
            candidate_denominator=np.asarray(values["candidate"]),
            arena_radius_mm=float(arena["radius_mm"]),
            declared_bin_width_mm=float(grid["bin_width_mm"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"Spatial occupancy display recipe cannot be resolved: {exc}")


def _plot_parameters(
    handle: ComposableChaserSuccessorSourceHandle,
    values: Mapping[str, Any],
    plan: SpatialOccupancyDisplayPlan,
) -> dict[str, Any]:
    arena = handle.scientific_manifest.get("arena")
    grid = handle.scientific_manifest.get("grid")
    sources = handle.scientific_manifest.get("sources")
    if (
        not isinstance(arena, Mapping)
        or not isinstance(grid, Mapping)
        or not isinstance(sources, Mapping)
        or not isinstance(sources.get("position_providers"), Sequence)
    ):
        _fail("Spatial occupancy plot authorities are incomplete.")
    x_edges = np.asarray(values["x_edges"])
    y_edges = np.asarray(values["y_edges"])
    return {
        "scientific_coordinates": {
            "x_bin_edges_mm": [float(value) for value in x_edges],
            "y_bin_edges_mm": [float(value) for value in y_edges],
            "x_bin_widths_mm": [float(value) for value in np.diff(x_edges)],
            "y_bin_widths_mm": [float(value) for value in np.diff(y_edges)],
            "declared_grid": {str(key): value for key, value in grid.items()},
            "arena_radius_mm": float(arena["radius_mm"]),
            "coordinate_orientation": "+x_right_+y_down",
            "epoch_order": ["chaser_pre", "chaser_training", "chaser_post"],
            "provider_order": [
                str(record["provider_id"]) for record in sources["position_providers"]
            ],
        },
        "display_recipe": plan.provenance_record(),
        "provider_epoch_denominators": {
            "candidate_frame_count": np.asarray(values["candidate"]).tolist(),
            "in_arena_position_frame_count": np.asarray(values["in_arena"]).tolist(),
            "in_arena_coverage_fraction_candidate": np.asarray(
                values["coverage"]
            ).tolist(),
        },
        "static_rendering": {
            "export_mode_ids": list(STATIC_EXPORT_MODE_IDS),
            "recommended_default_mode_id": DEFAULT_DISPLAY_MODE_ID,
            "figure_size_inches": list(PLOT_FIGURE_SIZE_INCHES),
            "subplot_grid": [3, 3],
            "png_dpi": PLOT_DPI,
            "density_colormap": "viridis",
            "difference_colormap": "RdBu_r",
            "pcolormesh_shading": "auto",
            "axis_aspect": "equal",
            "y_axis_display": "reversed_to_preserve_+y_down_image_coordinates",
            "arena_outline_linewidth_points": 1.0,
            "constrained_layout": True,
        },
    }


def spatial_occupancy_plot_parameters(
    handle: ComposableChaserSuccessorSourceHandle,
) -> dict[str, Any]:
    """Return the shared numerical recipe plus static adapter parameters."""

    values = _validate_plot_arrays(handle)
    return _plot_parameters(handle, values, _display_plan(handle, values))


def _render_spatial_occupancy_mode(
    handle: ComposableChaserSuccessorSourceHandle,
    *,
    values: Mapping[str, Any],
    mode: SpatialOccupancyDisplayMode,
    output_stem: Path,
) -> tuple[Path, Path]:
    payload = mode.payload
    surface = payload.surface
    density = np.asarray(payload.values_percent)
    difference = np.asarray(payload.difference_percentage_points)
    x_edges = np.asarray(surface.x_edges_mm)
    y_edges = np.asarray(surface.y_edges_mm)
    arena = handle.scientific_manifest.get("arena")
    if not isinstance(arena, Mapping):
        _fail("Spatial occupancy arena record is absent.")
    radius_mm = float(arena["radius_mm"])
    providers = handle.scientific_manifest["sources"]["position_providers"]
    provider_ids = [str(record["provider_id"]) for record in providers]
    epoch_labels = ("pre", "training", "post")

    figure, axes = plt.subplots(
        3,
        3,
        figsize=PLOT_FIGURE_SIZE_INCHES,
        constrained_layout=True,
    )
    density_images = []
    difference_images = []
    for epoch_index, epoch_label in enumerate(epoch_labels):
        for provider_index in range(2):
            ax = axes[provider_index, epoch_index]
            image = ax.pcolormesh(
                x_edges,
                y_edges,
                density[provider_index, epoch_index],
                shading="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=mode.value_limit,
            )
            density_images.append(image)
            ax.add_patch(
                Circle((0.0, 0.0), radius_mm, fill=False, color="white", linewidth=1.0)
            )
            coverage = 100.0 * values["coverage"][provider_index, epoch_index]
            ax.set_title(
                f"{epoch_label} · coverage {coverage:.2f}%\n"
                f"n={values['in_arena'][provider_index, epoch_index]:,}/"
                f"{values['candidate'][provider_index, epoch_index]:,}"
            )
        ax = axes[2, epoch_index]
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            difference[epoch_index],
            shading="auto",
            cmap="RdBu_r",
            vmin=-mode.difference_limit,
            vmax=mode.difference_limit,
        )
        difference_images.append(image)
        ax.add_patch(
            Circle((0.0, 0.0), radius_mm, fill=False, color="black", linewidth=1.0)
        )
        ax.set_title(f"{epoch_label} · detection − keypoint")

    row_labels = (
        f"keypoint\n{provider_ids[0]}",
        f"detection\n{provider_ids[1]}",
        "provider difference\npercentage points/bin",
    )
    for row_index, row_label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(f"{row_label}\ny from arena center (mm; +down)")
    for ax in axes.reshape(-1):
        ax.set_aspect("equal")
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
        ax.set_ylim(float(y_edges[-1]), float(y_edges[0]))
        ax.set_xlabel("x from arena center (mm)")
    figure.colorbar(
        density_images[0],
        ax=axes[:2, :].ravel().tolist(),
        shrink=0.8,
        label=payload.normalization.colorbar,
    )
    figure.colorbar(
        difference_images[0],
        ax=axes[2, :].ravel().tolist(),
        shrink=0.8,
        label=payload.normalization.difference_colorbar,
    )
    figure.suptitle(
        f"Exact protocol-semantic spatial occupancy · {handle.recording_id}\n"
        f"{mode.label} · canonical 2 mm counts retained · "
        "selector-ineligible",
        fontsize=13,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    temporary_png = png.with_name(f".{png.name}.tmp")
    temporary_pdf = pdf.with_name(f".{pdf.name}.tmp")
    try:
        figure.savefig(temporary_png, dpi=PLOT_DPI, format="png")
        figure.savefig(temporary_pdf, format="pdf")
        os.replace(temporary_png, png)
        os.replace(temporary_pdf, pdf)
    finally:
        plt.close(figure)
        temporary_png.unlink(missing_ok=True)
        temporary_pdf.unlink(missing_ok=True)
    return png, pdf


def render_spatial_occupancy_heatmaps(
    handle: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render the shared recommended occupancy mode for direct callers."""

    values = _validate_plot_arrays(handle)
    plan = _display_plan(handle, values)
    return _render_spatial_occupancy_mode(
        handle,
        values=values,
        mode=plan.default_mode,
        output_stem=output_stem,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--bundle-name",
        help="Output bundle basename; defaults to the exact source run name.",
    )
    parser.add_argument("--expected-recording-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-validation-receipt")
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
    receipt_path = output_dir / f"{bundle_name}_spatial_occupancy_plot_receipt.json"
    mode_stems = {
        mode_id: output_dir / f"{bundle_name}_{mode_id}_heatmaps"
        for mode_id in STATIC_EXPORT_MODE_IDS
    }
    expected = tuple(
        path
        for stem in mode_stems.values()
        for path in (stem.with_suffix(".png"), stem.with_suffix(".pdf"))
    ) + (receipt_path,)
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("Spatial occupancy plot output already exists.")
    receipt_bound = args.source_validation_receipt is not None
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_spatial_occupancy",
        run_name=args.run_name,
        expected_recording_id=args.expected_recording_id,
        use_consolidated=True,
        deep_audit=not receipt_bound,
        direct_validation_receipt=args.source_validation_receipt,
        required_array_names=(
            SPATIAL_OCCUPANCY_PLOT_ARRAY_NAMES if receipt_bound else None
        ),
    )
    values = _validate_plot_arrays(handle)
    display_plan = _display_plan(handle, values)
    rendered: list[tuple[SpatialOccupancyDisplayMode, Path, Path]] = []
    for mode_id in STATIC_EXPORT_MODE_IDS:
        mode = display_plan.mode(mode_id)
        png, pdf = _render_spatial_occupancy_mode(
            handle,
            values=values,
            mode=mode,
            output_stem=mode_stems[mode_id],
        )
        rendered.append((mode, png, pdf))
    plot_parameters = _plot_parameters(handle, values, display_plan)
    source_binding = {
        "successor_kind": handle.successor_kind,
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "scientific_payload_sha256": handle.scientific_payload_sha256,
        "deep_content_audit": handle.deep_audited,
        "verification_mode": handle.verification_mode,
        "verified_array_names": list(handle.verified_array_names),
        "validation_receipt_sha256": handle.receipt_digest,
    }
    outputs = []
    for mode, png, pdf in rendered:
        for path in (png, pdf):
            outputs.append(
                {
                    "path": str(path),
                    "media_type": (
                        "image/png" if path.suffix == ".png" else "application/pdf"
                    ),
                    "display_mode_id": mode.mode_id,
                    "artifact_role": (
                        "recommended_default"
                        if mode.mode_id == DEFAULT_DISPLAY_MODE_ID
                        else "full_range_reference"
                    ),
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "plot_recipe_id": PLOT_RECIPE_ID,
        "recording_id": handle.recording_id,
        "run_name": args.run_name,
        "bundle_name": bundle_name,
        "source_binding": source_binding,
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "outputs": outputs,
        "plot_parameters": plot_parameters,
        "plot_parameters_sha256": canonical_json_sha256(plot_parameters),
        "plot_policy": {
            "source_selection": "explicit_exact_run_name_no_selector_discovery",
            "source_validation": handle.verification_mode,
            "display_recipe_source": (
                "fisheye.visualization.chaser_spatial_occupancy_display"
            ),
            "display_recipe_id": DISPLAY_RECIPE_ID,
            "provider_comparison": "shared_physical_bins_and_shared_density_scale",
            "epoch_membership": "exact_protocol_semantic_half_open_intervals",
            "normalization": (
                "shared_recipe_conditional_valid_in_arena_with_coverage_annotation"
            ),
            "default_display": DEFAULT_DISPLAY_MODE_ID,
            "full_range_reference": "required_and_published",
            "missing_positions": "remain_missing_no_interpolation",
            "coordinate_orientation": "+x_right_+y_down",
            "scientific_authority": False,
            "zarr_visualization_storage": (
                "not_written_external_receipt_bound_static_artifacts"
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
    }
    receipt = {**body, "payload_sha256": canonical_json_sha256(body)}
    write_json_atomic(receipt_path, receipt)
    print(
        json.dumps(
            {**receipt, "receipt_path": str(receipt_path)},
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PLOT_RECIPE_ID",
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "ChaserSpatialOccupancyPlotError",
    "main",
    "render_spatial_occupancy_heatmaps",
    "spatial_occupancy_plot_parameters",
]
