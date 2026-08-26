"""Plot one exact radial/near-field successor and seal an external receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    load_composable_chaser_successor_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


RECEIPT_SCHEMA_ID = "palette.analysis.chaser_radial_near_field.plot_receipt"
RECEIPT_SCHEMA_VERSION = 2
PLOT_RECIPE_ID = "chaser_radial_near_field_summary_v2"
PLOT_DPI = 180
PLOT_FIGURE_SIZE_INCHES = (16.0, 10.0)


class ChaserRadialNearFieldPlotError(ValueError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels(manifest: Mapping[str, Any], name: str) -> Mapping[str, str]:
    value = manifest.get("identity_registries", {}).get(name, {})
    if not isinstance(value, Mapping):
        raise ChaserRadialNearFieldPlotError(f"Missing {name!r} registry.")
    return {str(key): str(item) for key, item in value.items()}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def radial_near_field_plot_parameters(handle: Any) -> dict[str, Any]:
    """Return the complete numerical and rendering recipe for one plot."""

    scientific = handle.scientific_manifest

    def array(name: str) -> np.ndarray:
        return np.asarray(handle.array(name))

    starts = array("radial_bin_start_mm").astype(np.float64)
    ends = array("radial_bin_end_mm").astype(np.float64)
    if starts.size != ends.size or np.any(~np.isfinite(starts)) or np.any(
        ~np.isfinite(ends)
    ):
        raise ChaserRadialNearFieldPlotError(
            "Radial plot bin boundaries are absent or non-finite."
        )
    bins = sorted({(float(start), float(end)) for start, end in zip(starts, ends)})
    config = scientific.get("config")
    if not isinstance(config, Mapping):
        raise ChaserRadialNearFieldPlotError("Radial scientific config is absent.")
    return {
        "scientific_coordinates": {
            "radial_bins_mm": [
                {
                    "start_inclusive_mm": start,
                    "end_exclusive_mm": end,
                    "center_for_plot_mm": (start + end) / 2.0,
                }
                for start, end in bins
            ],
            "near_zone_radius_mm": float(config["near_zone_radius_mm"]),
            "config": _plain(config),
            "epoch_registry": dict(_labels(scientific, "epoch_role")),
            "behavior_registry": dict(_labels(scientific, "behavior_role")),
            "chaser_registry": dict(_labels(scientific, "chaser")),
        },
        "normalization_and_scale": {
            "distance_summary": "median_with_25th_and_75th_percentiles_mm",
            "radial_selection": "persisted_area_corrected_geometric_selection_index",
            "near_fraction_denominator": "valid_distance_rows",
            "entry_rate_denominator": "valid_session_time_minutes",
            "near_fraction_y_min": 0.0,
            "radial_reference_line": 0.0,
            "missing_value_policy": "remain_missing_no_interpolation",
        },
        "rendering": {
            "figure_size_inches": list(PLOT_FIGURE_SIZE_INCHES),
            "subplot_grid": [2, 2],
            "png_dpi": PLOT_DPI,
            "constrained_layout": True,
            "distance_errorbar_format": "o",
            "distance_errorbar_capsize_points": 4.0,
            "radial_line_marker": "o",
            "radial_line_width_points": 1.2,
            "near_fraction_color": "#4c78a8",
            "paired_bar_width": 0.38,
            "label_rotation_degrees": 30.0,
        },
    }


def render(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    output_stem: str | Path,
    source_validation_receipt: str | Path | None = None,
) -> dict[str, Any]:
    handle = load_composable_chaser_successor_source_handle(
        analysis_zarr,
        successor_kind="chaser_radial_near_field",
        run_name=run_name,
        deep_audit=True,
        direct_validation_receipt=source_validation_receipt,
    )
    scientific = handle.scientific_manifest
    epoch_registry = _labels(scientific, "epoch_role")
    behavior_registry = _labels(scientific, "behavior_role")
    chaser_registry = _labels(scientific, "chaser")

    def array(name: str) -> np.ndarray:
        return np.asarray(handle.array(name))

    epoch = array("metric_epoch_role_code").astype(np.int64)
    behavior = array("metric_behavior_role_code").astype(np.int64)
    chaser = array("metric_chaser_identity_code").astype(np.int64)
    median = array("metric_distance_p50_mm").astype(np.float64)
    p25 = array("metric_distance_p25_mm").astype(np.float64)
    p75 = array("metric_distance_p75_mm").astype(np.float64)
    near_fraction = array("metric_near_zone_fraction_valid").astype(np.float64)
    near_dwell = array("metric_near_zone_dwell_s").astype(np.float64)
    entry_rate = array("metric_near_zone_entry_rate_per_min_valid_time").astype(np.float64)
    count = array("metric_valid_distance_frame_count").astype(np.int64)
    labels = [
        f"{epoch_registry[str(int(e))]}\n{behavior_registry[str(int(b))]} · {chaser_registry[str(int(c))]}"
        for e, b, c in zip(epoch, behavior, chaser, strict=True)
    ]
    x = np.arange(len(labels))
    plot_parameters = radial_near_field_plot_parameters(handle)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=PLOT_FIGURE_SIZE_INCHES,
        constrained_layout=True,
    )

    valid = np.isfinite(median) & np.isfinite(p25) & np.isfinite(p75)
    axes[0, 0].errorbar(
        x[valid],
        median[valid],
        yerr=np.vstack((median[valid] - p25[valid], p75[valid] - median[valid])),
        fmt="o",
        capsize=4,
    )
    for index in np.flatnonzero(valid):
        axes[0, 0].annotate(
            f"n={count[index]}", (x[index], median[index]), xytext=(0, 7),
            textcoords="offset points", ha="center", fontsize=7,
        )
    axes[0, 0].set_ylabel("fish–chaser distance (mm)")
    axes[0, 0].set_title("Simple distance: median and interquartile range")

    r_epoch = array("radial_epoch_role_code").astype(np.int64)
    r_behavior = array("radial_behavior_role_code").astype(np.int64)
    r_chaser = array("radial_chaser_identity_code").astype(np.int64)
    r_start = array("radial_bin_start_mm").astype(np.float64)
    r_end = array("radial_bin_end_mm").astype(np.float64)
    selection = array("radial_selection_index_geometric").astype(np.float64)
    for key in sorted(set(zip(r_epoch.tolist(), r_behavior.tolist(), r_chaser.tolist()))):
        mask = (r_epoch == key[0]) & (r_behavior == key[1]) & (r_chaser == key[2])
        order = np.argsort(r_start[mask])
        center = (r_start[mask][order] + r_end[mask][order]) / 2.0
        values = selection[mask][order]
        finite = np.isfinite(values)
        axes[0, 1].plot(
            center[finite], values[finite], marker="o", linewidth=1.2,
            label=(
                f"{epoch_registry[str(key[0])]} · "
                f"{behavior_registry[str(key[1])]}"
            ),
        )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0, 1].set(xlabel="fish–chaser distance (mm)", ylabel="geometric selection index")
    axes[0, 1].set_title("Area-corrected moving-chaser radial selection")
    axes[0, 1].legend(fontsize=7, ncols=2)

    axes[1, 0].bar(x, near_fraction, color="#4c78a8")
    axes[1, 0].set_ylabel("fraction of valid distance rows")
    axes[1, 0].set_title(
        f"Near-field occupancy (≤{scientific['config']['near_zone_radius_mm']:g} mm)"
    )
    axes[1, 0].set_ylim(bottom=0)

    width = 0.38
    axes[1, 1].bar(x - width / 2, near_dwell, width, label="exact dwell (s)")
    axes[1, 1].bar(x + width / 2, entry_rate, width, label="entries/min valid time")
    axes[1, 1].set_title("Exact session-time near-field visits")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.reshape(-1):
        ax.grid(axis="y", alpha=0.2)
        if ax is not axes[0, 1]:
            ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    provider = scientific["position_provider"]
    figure.suptitle(
        f"Chaser distance, radial rings, and near field · {handle.recording_id}\n"
        f"position provider: {provider['provider_id']} · exact session time · "
        "selector-ineligible",
        fontsize=13,
    )

    stem = Path(output_stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
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
    files = {
        "png": {"path": str(png), "sha256": _file_sha256(png)},
        "pdf": {"path": str(pdf), "sha256": _file_sha256(pdf)},
    }
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "plot_recipe_id": PLOT_RECIPE_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recording_id": handle.recording_id,
        "source": {
            "successor_kind": handle.successor_kind,
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "scientific_payload_sha256": handle.scientific_payload_sha256,
            "deep_content_audit": True,
            "verification_mode": handle.metadata_equivalence.get(
                "verification_mode", "direct_consolidated_equivalence"
            ),
            "validation_receipt_sha256": handle.metadata_equivalence.get(
                "receipt_sha256"
            ),
            "relative_frame": dict(scientific["sources"]["relative_frame"]),
            "protocol_semantic_selection": dict(
                scientific["sources"]["protocol_semantic_selection"]
            ),
            "position_provider": dict(provider),
        },
        "files": files,
        "plot_parameters": plot_parameters,
        "plot_parameters_sha256": canonical_json_sha256(plot_parameters),
        "selector_eligible": False,
        "production_authority": False,
    }
    receipt = {**body, "payload_sha256": canonical_json_sha256(body)}
    receipt_path = stem.with_suffix(".receipt.json")
    write_json_atomic(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--source-validation-receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = render(
        args.analysis_zarr,
        run_name=args.run_name,
        output_stem=args.output_stem,
        source_validation_receipt=args.source_validation_receipt,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PLOT_RECIPE_ID",
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "ChaserRadialNearFieldPlotError",
    "radial_near_field_plot_parameters",
    "render",
]
