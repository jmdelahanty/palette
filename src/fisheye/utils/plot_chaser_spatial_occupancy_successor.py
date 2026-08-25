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


RECEIPT_SCHEMA_ID = "palette.analysis.chaser_spatial_occupancy.plot_receipt"
RECEIPT_SCHEMA_VERSION = 1
PLOT_RECIPE_ID = "paired_provider_exact_epoch_spatial_occupancy_heatmap_v1"
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


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


def _array(
    handle: ComposableChaserSuccessorSourceHandle, name: str
) -> np.ndarray:
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
    if handle.deep_audited is not True:
        _fail("Spatial occupancy plotting requires a deep content audit.")
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
    density = _array(handle, "occupancy_density_valid_in_arena").astype(
        np.float64
    )
    fraction = _array(handle, "occupancy_fraction_candidate_epoch").astype(
        np.float64
    )
    if counts.shape != expected or density.shape != expected or fraction.shape != expected:
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
    if (
        x_edges.shape != (expected[3] + 1,)
        or y_edges.shape != (expected[2] + 1,)
        or np.any(np.diff(x_edges) <= 0)
        or np.any(np.diff(y_edges) <= 0)
    ):
        _fail("Spatial occupancy bin edges are invalid.")
    candidate = _array(handle, "candidate_frame_count").astype(np.int64)
    in_arena = _array(handle, "in_arena_position_frame_count").astype(np.int64)
    coverage = _array(handle, "in_arena_coverage_fraction_candidate").astype(
        np.float64
    )
    if candidate.shape != (2, 3) or in_arena.shape != (2, 3) or coverage.shape != (2, 3):
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
    if not np.allclose(
        fraction.sum(axis=(2, 3)), coverage, rtol=1e-10, atol=1e-12
    ):
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
        "candidate": candidate,
        "in_arena": in_arena,
        "coverage": coverage,
        "provider_registry": provider_registry,
        "epoch_registry": epoch_registry,
    }


def render_spatial_occupancy_heatmaps(
    handle: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render keypoint, detection, and matched-provider difference heatmaps."""

    values = _validate_plot_arrays(handle)
    density = values["density"] * 100.0
    difference = density[1] - density[0]
    finite_density = density[np.isfinite(density)]
    if not finite_density.size or float(np.max(finite_density)) <= 0:
        _fail("Spatial occupancy has no positive density to plot.")
    vmax = float(np.max(finite_density))
    diff_limit = float(np.max(np.abs(difference)))
    if diff_limit <= 0:
        diff_limit = max(vmax * 1e-6, np.finfo(np.float64).eps)
    x_edges = values["x_edges"]
    y_edges = values["y_edges"]
    arena = handle.scientific_manifest.get("arena")
    if not isinstance(arena, Mapping):
        _fail("Spatial occupancy arena record is absent.")
    radius_mm = float(arena["radius_mm"])
    providers = handle.scientific_manifest["sources"]["position_providers"]
    provider_ids = [str(record["provider_id"]) for record in providers]
    epoch_labels = ("pre", "training", "post")

    figure, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
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
                vmax=vmax,
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
            cmap="coolwarm",
            vmin=-diff_limit,
            vmax=diff_limit,
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
        label="occupancy (% of valid in-arena provider frames per bin)",
    )
    figure.colorbar(
        difference_images[0],
        ax=axes[2, :].ravel().tolist(),
        shrink=0.8,
        label="detection − keypoint occupancy (percentage points/bin)",
    )
    figure.suptitle(
        f"Exact protocol-semantic spatial occupancy · {handle.recording_id}\n"
        "shared reviewed physical grid · missing positions retained in coverage · "
        "selector-ineligible",
        fontsize=13,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    temporary_png = png.with_name(f".{png.name}.tmp")
    temporary_pdf = pdf.with_name(f".{pdf.name}.tmp")
    try:
        figure.savefig(temporary_png, dpi=180, format="png")
        figure.savefig(temporary_pdf, format="pdf")
        os.replace(temporary_png, png)
        os.replace(temporary_pdf, pdf)
    finally:
        plt.close(figure)
        temporary_png.unlink(missing_ok=True)
        temporary_pdf.unlink(missing_ok=True)
    return png, pdf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-recording-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if _RUN_NAME_RE.fullmatch(args.run_name) is None:
        _fail("run_name must be one exact non-selector child name.")
    archive = args.analysis_zarr.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    stem = output_dir / f"{args.run_name}_heatmaps"
    receipt_path = output_dir / f"{args.run_name}_spatial_occupancy_plot_receipt.json"
    expected = (stem.with_suffix(".png"), stem.with_suffix(".pdf"), receipt_path)
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("Spatial occupancy plot output already exists.")
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_spatial_occupancy",
        run_name=args.run_name,
        expected_recording_id=args.expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
    )
    png, pdf = render_spatial_occupancy_heatmaps(handle, output_stem=stem)
    source_binding = {
        "successor_kind": handle.successor_kind,
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "scientific_payload_sha256": handle.scientific_payload_sha256,
        "deep_content_audit": True,
    }
    outputs = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in (png, pdf)
    ]
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "plot_recipe_id": PLOT_RECIPE_ID,
        "recording_id": handle.recording_id,
        "run_name": args.run_name,
        "source_binding": source_binding,
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "outputs": outputs,
        "plot_policy": {
            "source_selection": "explicit_exact_run_name_no_selector_discovery",
            "source_validation": "deep_array_content_audit",
            "provider_comparison": "shared_physical_bins_and_shared_density_scale",
            "epoch_membership": "exact_protocol_semantic_half_open_intervals",
            "normalization": "conditional_valid_in_arena_density_with_coverage_annotation",
            "missing_positions": "remain_missing_no_interpolation",
            "coordinate_orientation": "+x_right_+y_down",
            "scientific_authority": False,
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
]
