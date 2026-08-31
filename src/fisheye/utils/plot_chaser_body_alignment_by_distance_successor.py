"""Render one exact anatomical alignment-by-distance successor publication."""

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
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np

from fisheye.analysis_workflows.chaser_body_alignment_by_distance_successor import (
    SUMMARY_VIEW_ARRAY_NAMES,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    ComposableChaserSuccessorSourceHandle,
    load_composable_chaser_successor_source_handle,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.chaser_body_alignment_by_distance import (
    validate_persisted_body_alignment_summary,
)


RECEIPT_SCHEMA_ID = "palette.analysis.chaser_body_alignment_by_distance.plot_receipt"
RECEIPT_SCHEMA_VERSION = 1
PLOT_RECIPE_ID = "persisted_anatomical_alignment_distance_bins_static_v2"
PLOT_DPI = 180
PLOT_FIGURE_SIZE_INCHES = (16.0, 15.0)
PLOT_SAVEFIG_BBOX_INCHES = "tight"
PLOT_SAVEFIG_PAD_INCHES = 0.12
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_EPOCH_COLORS = {
    1: "#4c78a8",
    2: "#e45756",
    3: "#54a24b",
}
_LINESTYLES = ("-", "--", ":", "-.")
_EPOCH_FALLBACK = {
    "1": "chaser_pre",
    "2": "chaser_training",
    "3": "chaser_post",
}


class ChaserBodyAlignmentPlotError(ValueError):
    """Raised when exact persisted alignment evidence cannot be plotted."""


def _fail(message: str) -> None:
    raise ChaserBodyAlignmentPlotError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry(manifest: Mapping[str, Any], name: str) -> dict[str, str]:
    registries = manifest.get("identity_registries")
    if not isinstance(registries, Mapping):
        return {}
    registry = registries.get(name)
    if not isinstance(registry, Mapping):
        return {}
    return {str(key): str(value) for key, value in registry.items()}


def _factorized_legend_entries(
    values: Mapping[str, Any],
    scientific: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return compact visible labels while retaining canonical identities."""

    epoch_registry = {
        **_EPOCH_FALLBACK,
        **_registry(scientific, "epoch_role"),
    }
    chaser_registry = _registry(scientific, "chaser")
    behavior_registry = _registry(scientific, "behavior_role")
    identities = tuple(int(value) for value in values["identities"])
    summary_identity = np.asarray(
        values["summary_chaser_identity_code"], dtype=np.int64
    )
    summary_role = np.asarray(
        values["summary_chaser_behavior_role_code"], dtype=np.int64
    )

    epoch_entries = [
        {
            "epoch_role_code": epoch,
            "visible_label": epoch_registry.get(str(epoch), f"epoch {epoch}"),
            "color": _EPOCH_COLORS[epoch],
        }
        for epoch in (1, 2, 3)
    ]
    chaser_entries: list[dict[str, Any]] = []
    for position, identity in enumerate(identities):
        role_codes = sorted(
            int(value) for value in np.unique(summary_role[summary_identity == identity])
        )
        if not role_codes:
            _fail(f"Chaser identity code {identity} has no persisted behavior role.")
        role_labels = [
            behavior_registry.get(str(code), f"role {code}") for code in role_codes
        ]
        chaser_entries.append(
            {
                "chaser_identity_code": identity,
                "canonical_identity": chaser_registry.get(
                    str(identity), f"chaser {identity}"
                ),
                "behavior_role_codes": role_codes,
                "behavior_role_labels": role_labels,
                "visible_label": f"chaser {identity} · {'/'.join(role_labels)}",
                "linestyle": _LINESTYLES[position % len(_LINESTYLES)],
            }
        )
    return epoch_entries, chaser_entries


def body_alignment_plot_parameters(
    handle: ComposableChaserSuccessorSourceHandle,
) -> dict[str, Any]:
    """Return the exact persisted coordinates and complete rendering recipe."""

    if handle.successor_kind != "chaser_body_alignment_by_distance":
        _fail("Plot source is not a body-alignment-by-distance successor.")
    values = validate_persisted_body_alignment_summary(handle)
    scientific = handle.scientific_manifest
    distance_recipe = scientific.get("distance_bin_recipe")
    angle_convention = scientific.get("coordinate_and_angle_convention")
    denominators = scientific.get("denominators")
    if (
        not isinstance(distance_recipe, Mapping)
        or not isinstance(angle_convention, Mapping)
        or not isinstance(denominators, Mapping)
    ):
        _fail("Body-alignment plot authorities are incomplete.")
    identities = tuple(int(value) for value in values["identities"])
    epoch_registry = {
        **_EPOCH_FALLBACK,
        **_registry(scientific, "epoch_role"),
    }
    chaser_registry = _registry(scientific, "chaser")
    behavior_registry = _registry(scientific, "behavior_role")
    epoch_legend_entries, chaser_legend_entries = _factorized_legend_entries(
        values, scientific
    )
    legend_entry_count = len(epoch_legend_entries) + len(chaser_legend_entries)
    return json_attr_safe(
        {
            "scientific_coordinates": {
                "distance_bin_edges_mm": values["distance_bin_edges_mm"],
                "distance_bin_centers_mm": np.unique(
                    np.asarray(values["summary_distance_bin_center_mm"])
                ),
                "distance_bin_recipe": distance_recipe,
                "epoch_order": [
                    "chaser_pre",
                    "chaser_training",
                    "chaser_post",
                ],
                "chaser_identity_codes": identities,
                "identity_labels": {
                    "epoch_role": epoch_registry,
                    "chaser": chaser_registry,
                    "behavior_role": behavior_registry,
                },
            },
            "scientific_surfaces": {
                "source_arrays": list(SUMMARY_VIEW_ARRAY_NAMES),
                "alignment": "summary_mean_alignment_cos",
                "alignment_interquartile_range": [
                    "summary_alignment_cos_p25",
                    "summary_alignment_cos_p75",
                ],
                "absolute_bearing": "summary_mean_abs_bearing_deg",
                "absolute_bearing_interquartile_range": [
                    "summary_abs_bearing_p25_deg",
                    "summary_abs_bearing_p75_deg",
                ],
                "circular_bearing": [
                    "summary_circular_mean_bearing_deg",
                    "summary_circular_resultant_length",
                ],
                "support": [
                    "summary_candidate_row_count",
                    "summary_joint_valid_row_count",
                ],
                "joint_valid_fraction": (
                    "summary_joint_valid_row_count / summary_candidate_row_count"
                ),
                "alignment_definition": "cos(body_bearing_deg)",
                "distance_surface": "base/relative_distance_physical",
                "denominator_policy": denominators,
                "angle_convention": angle_convention,
            },
            "rendering": {
                "figure_size_inches": list(PLOT_FIGURE_SIZE_INCHES),
                "subplot_grid": [3, 2],
                "png_dpi": PLOT_DPI,
                "epoch_color_map": {
                    str(key): value for key, value in _EPOCH_COLORS.items()
                },
                "chaser_line_styles": list(_LINESTYLES),
                "chaser_line_style_by_identity_code": {
                    str(identity): _LINESTYLES[index % len(_LINESTYLES)]
                    for index, identity in enumerate(identities)
                },
                "role_style_policy": (
                    "epoch controls curve color; chaser identity controls line "
                    "style; behavior role is a separate persisted label; no "
                    "style is inferred from stimulus pixel color"
                ),
                "interquartile_band_alpha": 0.12,
                "constrained_layout": True,
                "legend": {
                    "factorization_policy": (
                        "epoch_color_plus_chaser_identity_line_style_and_"
                        "behavior_role_text_v1"
                    ),
                    "title": "epoch color · chaser style and behavior role",
                    "epoch_entries": epoch_legend_entries,
                    "chaser_entries": chaser_legend_entries,
                    "location": "lower_center_outside_axes",
                    "bbox_to_anchor": [0.5, -0.055],
                    "ncols": min(5, max(1, legend_entry_count)),
                    "fontsize_points": 8,
                },
                "savefig": {
                    "bbox_inches": PLOT_SAVEFIG_BBOX_INCHES,
                    "pad_inches": PLOT_SAVEFIG_PAD_INCHES,
                },
            },
            "viewer_policy": {
                "rebinning": "prohibited",
                "scientific_groupby": "prohibited",
                "interpolation": "prohibited",
                "body_origin_distance_substitution": "prohibited",
                "motion_heading_fallback": "prohibited",
                "scientific_recomputation": False,
            },
        }
    )


def render_body_alignment_by_distance(
    handle: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render six panels directly from the persisted summary-bin rows."""

    values = validate_persisted_body_alignment_summary(handle)
    parameters = body_alignment_plot_parameters(handle)
    role_code = np.asarray(values["summary_epoch_role_code"], dtype=np.int64)
    identity = np.asarray(values["summary_chaser_identity_code"], dtype=np.int64)
    bin_index = np.asarray(values["summary_distance_bin_index"], dtype=np.int64)
    centers = np.asarray(values["summary_distance_bin_center_mm"], dtype=np.float64)
    candidate = np.asarray(values["summary_candidate_row_count"], dtype=np.int64)
    joint = np.asarray(values["summary_joint_valid_row_count"], dtype=np.int64)

    figure, axes = plt.subplots(
        3,
        2,
        figsize=PLOT_FIGURE_SIZE_INCHES,
        constrained_layout=True,
    )
    for epoch in (1, 2, 3):
        for chaser_position, chaser in enumerate(values["identities"]):
            member = (role_code == epoch) & (identity == int(chaser))
            indices = np.flatnonzero(member)[
                np.argsort(bin_index[member], kind="stable")
            ]
            color = _EPOCH_COLORS[epoch]
            linestyle = _LINESTYLES[chaser_position % len(_LINESTYLES)]
            line_kwargs = {
                "color": color,
                "linestyle": linestyle,
                "linewidth": 1.7,
                "marker": "o",
                "markersize": 3.5,
            }
            x = centers[indices]
            alignment = np.asarray(
                values["summary_mean_alignment_cos"], dtype=np.float64
            )[indices]
            axes[0, 0].plot(x, alignment, **line_kwargs)
            axes[0, 0].fill_between(
                x,
                np.asarray(values["summary_alignment_cos_p25"], dtype=np.float64)[
                    indices
                ],
                np.asarray(values["summary_alignment_cos_p75"], dtype=np.float64)[
                    indices
                ],
                color=color,
                alpha=0.12,
            )

            axes[0, 1].plot(
                x,
                np.asarray(values["summary_mean_abs_bearing_deg"], dtype=np.float64)[
                    indices
                ],
                **line_kwargs,
            )
            axes[0, 1].fill_between(
                x,
                np.asarray(values["summary_abs_bearing_p25_deg"], dtype=np.float64)[
                    indices
                ],
                np.asarray(values["summary_abs_bearing_p75_deg"], dtype=np.float64)[
                    indices
                ],
                color=color,
                alpha=0.12,
            )
            axes[1, 0].plot(
                x,
                np.asarray(
                    values["summary_circular_mean_bearing_deg"],
                    dtype=np.float64,
                )[indices],
                **line_kwargs,
            )
            axes[1, 1].plot(
                x,
                np.asarray(
                    values["summary_circular_resultant_length"],
                    dtype=np.float64,
                )[indices],
                **line_kwargs,
            )
            fraction = np.divide(
                joint[indices].astype(np.float64),
                candidate[indices].astype(np.float64),
                out=np.full(indices.size, np.nan, dtype=np.float64),
                where=candidate[indices] > 0,
            )
            axes[2, 0].plot(x, 100.0 * fraction, **line_kwargs)
            axes[2, 1].plot(x, candidate[indices], **line_kwargs)
            axes[2, 1].plot(
                x,
                joint[indices],
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
                alpha=0.55,
            )

    axes[0, 0].axhline(0.0, color="#777777", linestyle=":", linewidth=1.0)
    axes[0, 0].set_title("Mean anatomical alignment (band: P25–P75)")
    axes[0, 0].set_ylabel("cos(body bearing): +1 front, −1 behind")
    axes[0, 0].set_ylim(-1.05, 1.05)
    axes[0, 1].set_title("Mean absolute anatomical bearing (band: P25–P75)")
    axes[0, 1].set_ylabel("absolute bearing (degrees)")
    axes[0, 1].set_ylim(0.0, 180.0)
    axes[1, 0].set_title("Circular mean anatomical bearing")
    axes[1, 0].set_ylabel("bearing (degrees)")
    axes[1, 0].set_ylim(-180.0, 180.0)
    axes[1, 1].set_title("Circular resultant length")
    axes[1, 1].set_ylabel("resultant")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[2, 0].set_title("Jointly valid anatomical fraction")
    axes[2, 0].set_ylabel("joint valid / distance valid (%)")
    axes[2, 0].set_ylim(0.0, 100.0)
    axes[2, 1].set_title("Persisted support (solid candidate; faint joint valid)")
    axes[2, 1].set_ylabel("rows")
    for ax in axes.reshape(-1):
        ax.set_xlabel("fish–chaser distance (mm; persisted bins)")
        ax.grid(alpha=0.2)
    legend_parameters = parameters["rendering"]["legend"]
    epoch_entries = legend_parameters["epoch_entries"]
    chaser_entries = legend_parameters["chaser_entries"]
    legend_handles = [
        Line2D([], [], color=entry["color"], linewidth=2.0)
        for entry in epoch_entries
    ] + [
        Line2D(
            [],
            [],
            color="#444444",
            linestyle=entry["linestyle"],
            linewidth=1.7,
            marker="o",
            markersize=3.5,
        )
        for entry in chaser_entries
    ]
    legend_labels = [entry["visible_label"] for entry in epoch_entries] + [
        entry["visible_label"] for entry in chaser_entries
    ]
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=tuple(legend_parameters["bbox_to_anchor"]),
        ncols=int(legend_parameters["ncols"]),
        fontsize=float(legend_parameters["fontsize_points"]),
        title=legend_parameters["title"],
    )
    figure.suptitle(
        f"Exact anatomical body alignment by chaser distance · {handle.recording_id}\n"
        "persisted semantic-epoch bins · no rebinning · selector-ineligible",
        fontsize=13,
    )

    # Calling the recipe builder above is intentional: rendering and receipt
    # creation fail on the same exact persisted contract.
    if parameters["viewer_policy"]["rebinning"] != "prohibited":
        _fail("Body-alignment rendering requires the persisted no-rebin policy.")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    temporary_png = png.with_name(f".{png.name}.tmp")
    temporary_pdf = pdf.with_name(f".{pdf.name}.tmp")
    try:
        savefig_parameters = parameters["rendering"]["savefig"]
        figure.savefig(
            temporary_png,
            dpi=PLOT_DPI,
            format="png",
            bbox_inches=savefig_parameters["bbox_inches"],
            pad_inches=float(savefig_parameters["pad_inches"]),
        )
        figure.savefig(
            temporary_pdf,
            format="pdf",
            bbox_inches=savefig_parameters["bbox_inches"],
            pad_inches=float(savefig_parameters["pad_inches"]),
        )
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
    stem = output_dir / f"{bundle_name}_body_alignment_by_distance"
    receipt_path = output_dir / f"{bundle_name}_body_alignment_plot_receipt.json"
    expected = (stem.with_suffix(".png"), stem.with_suffix(".pdf"), receipt_path)
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("Body-alignment plot output already exists.")

    receipt_bound = args.source_validation_receipt is not None
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_body_alignment_by_distance",
        run_name=args.run_name,
        expected_recording_id=args.expected_recording_id,
        use_consolidated=True,
        deep_audit=not receipt_bound,
        direct_validation_receipt=args.source_validation_receipt,
        required_array_names=(SUMMARY_VIEW_ARRAY_NAMES if receipt_bound else None),
    )
    png, pdf = render_body_alignment_by_distance(handle, output_stem=stem)
    plot_parameters = body_alignment_plot_parameters(handle)
    source_binding = {
        "successor_kind": handle.successor_kind,
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "scientific_payload_sha256": handle.scientific_payload_sha256,
        "verification_mode": handle.verification_mode,
        "verified_array_names": list(handle.verified_array_names),
        "validation_receipt_sha256": handle.receipt_digest,
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
        "bundle_name": bundle_name,
        "source_binding": source_binding,
        "source_binding_sha256": canonical_json_sha256(source_binding),
        "outputs": outputs,
        "plot_parameters": plot_parameters,
        "plot_parameters_sha256": canonical_json_sha256(plot_parameters),
        "plot_policy": {
            "source_selection": "explicit_exact_run_name_no_selector_discovery",
            "source_validation": handle.verification_mode,
            "summary_source": "persisted_epoch_chaser_distance_bins",
            "viewer_rebinning": "prohibited",
            "viewer_scientific_groupby": "prohibited",
            "interpolation": "prohibited",
            "motion_heading_fallback": "prohibited",
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
    "ChaserBodyAlignmentPlotError",
    "body_alignment_plot_parameters",
    "main",
    "render_body_alignment_by_distance",
]
