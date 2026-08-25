"""Plot one exact radial/near-field successor and seal an external receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
RECEIPT_SCHEMA_VERSION = 1


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


def render(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    output_stem: str | Path,
) -> dict[str, Any]:
    handle = load_composable_chaser_successor_source_handle(
        analysis_zarr,
        successor_kind="chaser_radial_near_field",
        run_name=run_name,
        deep_audit=True,
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
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

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
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    plt.close(figure)
    files = {
        "png": {"path": str(png), "sha256": _file_sha256(png)},
        "pdf": {"path": str(pdf), "sha256": _file_sha256(pdf)},
    }
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recording_id": handle.recording_id,
        "source": {
            "successor_kind": handle.successor_kind,
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "scientific_payload_sha256": handle.scientific_payload_sha256,
            "deep_content_audit": True,
            "relative_frame": dict(scientific["sources"]["relative_frame"]),
            "protocol_semantic_selection": dict(
                scientific["sources"]["protocol_semantic_selection"]
            ),
            "position_provider": dict(provider),
        },
        "files": files,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = render(
        args.analysis_zarr, run_name=args.run_name, output_stem=args.output_stem
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
