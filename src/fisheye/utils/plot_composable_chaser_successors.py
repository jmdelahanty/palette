"""Plot exact selector-ineligible composable chaser successors.

The command never resolves a selector and never writes to the analysis Zarr.
It deep-audits three explicitly named immutable products, verifies their
dependency payload digests, and writes one dashboard in PNG and PDF form plus
an external content-hash receipt.
"""

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
import numpy as np

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    ComposableChaserSuccessorSourceHandle,
    load_composable_chaser_successor_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


RECEIPT_SCHEMA_ID = "palette.analysis.composable_chaser_successor.plot_receipt"
RECEIPT_SCHEMA_VERSION = 1
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_KINDS = (
    "controller_chase_trials",
    "generalized_chaser_bout_response",
    "chaser_escape_freeze",
)


class ComposableChaserPlotError(ValueError):
    """Raised when exact plotted products are missing, mixed, or malformed."""


def _fail(message: str) -> None:
    raise ComposableChaserPlotError(message)


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
        _fail(f"{handle.successor_kind} lacks plot array {name!r}.")


def _verify_chain(
    controller: ComposableChaserSuccessorSourceHandle,
    bout: ComposableChaserSuccessorSourceHandle,
    escape: ComposableChaserSuccessorSourceHandle,
) -> None:
    handles = (controller, bout, escape)
    if tuple(handle.successor_kind for handle in handles) != _KINDS:
        _fail("Plot handles are missing, reordered, or have the wrong product kinds.")
    if len({handle.recording_id for handle in handles}) != 1:
        _fail("Plot products belong to different recordings.")
    if not all(handle.deep_audited is True for handle in handles):
        _fail("Plot products must be loaded through deep content audits.")
    bout_sources = bout.scientific_manifest.get("sources")
    escape_sources = escape.scientific_manifest.get("sources")
    if not isinstance(bout_sources, Mapping) or not isinstance(
        escape_sources, Mapping
    ):
        _fail("Plot product source bindings are absent.")
    if (
        bout_sources.get("controller_trial_payload_sha256")
        != controller.scientific_payload_sha256
        or escape_sources.get("controller_trial_payload_sha256")
        != controller.scientific_payload_sha256
        or escape_sources.get("bout_response_payload_sha256")
        != bout.scientific_payload_sha256
    ):
        _fail("Plot product dependency payload digests are stale or mixed.")


def _trial_panel(ax: Any, controller: ComposableChaserSuccessorSourceHandle) -> None:
    start = _array(controller, "start_acquisition_frame_id").astype(np.int64)
    end = _array(controller, "end_acquisition_frame_id_inclusive").astype(np.int64)
    ordinal = _array(controller, "trial_ordinal").astype(np.int64)
    logged = _array(controller, "logged_trial_id").astype(np.int64)
    chaser = _array(controller, "chaser_identity_code").astype(np.int64)
    gap = _array(controller, "gap_fraction").astype(np.float64)
    if not (start.size == end.size == ordinal.size == logged.size == chaser.size == gap.size):
        _fail("Controller trial table columns have different lengths.")
    colors = plt.get_cmap("tab10")
    for row in range(start.size):
        width = max(1, int(end[row] - start[row] + 1))
        ax.barh(
            row,
            width,
            left=int(start[row]),
            height=0.62,
            color=colors(int(chaser[row]) % 10),
            alpha=0.82,
        )
        ax.text(
            int(start[row]) + width / 2,
            row,
            f"ID {logged[row]} · gap {100 * gap[row]:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(
        np.arange(start.size),
        [f"chaser {code}, trial {order}" for code, order in zip(chaser, ordinal)],
    )
    ax.set_xlabel("acquisition frame ID")
    ax.set_title("Exact producer-authored controller trials")
    ax.grid(axis="x", alpha=0.2)


def _distance_label(low: float, high: float) -> str:
    return f"{low:g}–∞" if not np.isfinite(high) else f"{low:g}–{high:g}"


def _bout_panel(ax: Any, bout: ComposableChaserSuccessorSourceHandle) -> None:
    role = _array(bout, "summary_role_code").astype(np.int64)
    chaser = _array(bout, "summary_chaser_identity_code").astype(np.int64)
    band = _array(bout, "summary_distance_bin_index").astype(np.int64)
    low = _array(bout, "summary_distance_bin_start_mm").astype(np.float64)
    high = _array(bout, "summary_distance_bin_end_mm").astype(np.float64)
    rate = _array(bout, "summary_bout_rate_per_min").astype(np.float64)
    count = _array(bout, "summary_bout_count").astype(np.int64)
    registry = bout.scientific_manifest.get("identity_registries", {})
    role_registry = registry.get("semantic_role", {}) if isinstance(registry, Mapping) else {}
    series = sorted(set(zip(role.tolist(), chaser.tolist())))
    labels_by_band: dict[int, str] = {}
    for role_code, chaser_code in series:
        mask = (role == role_code) & (chaser == chaser_code)
        order = np.argsort(band[mask])
        x = band[mask][order]
        y = rate[mask][order]
        valid = np.isfinite(y)
        name = role_registry.get(str(role_code), f"role {role_code}")
        ax.plot(
            x[valid],
            y[valid],
            marker="o",
            linewidth=1.4,
            label=f"{name}, chaser {chaser_code}",
        )
        for band_index, lo, hi in zip(band[mask], low[mask], high[mask]):
            labels_by_band[int(band_index)] = _distance_label(float(lo), float(hi))
        for x_value, y_value, n_value in zip(x[valid], y[valid], count[mask][order][valid]):
            ax.annotate(str(int(n_value)), (x_value, y_value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    ticks = sorted(labels_by_band)
    ax.set_xticks(ticks, [labels_by_band[value] for value in ticks])
    ax.set_xlabel("distance band (mm); labels show bout count")
    ax.set_ylabel("bout rate per minute of valid exposure")
    ax.set_title("Generalized bout response by semantic role")
    ax.grid(alpha=0.2)
    if series:
        ax.legend(fontsize=7, ncols=2)


def _response_panel(ax: Any, escape: ComposableChaserSuccessorSourceHandle) -> None:
    response = _array(escape, "trial_response_class_code").astype(np.int64)
    labels = ("insufficient", "escape", "freeze", "other")
    counts = [int(np.count_nonzero(response == code)) for code in range(4)]
    bars = ax.bar(labels, counts, color=("#999999", "#d95f02", "#1b9e77", "#7570b3"))
    ax.bar_label(bars)
    ax.set_ylabel("exact trial count")
    ax.set_title("Trial response classes")
    ax.grid(axis="y", alpha=0.2)


def _sweep_panel(ax: Any, escape: ComposableChaserSuccessorSourceHandle) -> None:
    trial = _array(escape, "sweep_trial_row_id").astype(np.int64)
    threshold = _array(escape, "sweep_speed_threshold_mm_s").astype(np.float64)
    rate = _array(escape, "sweep_escape_event_rate_per_min").astype(np.float64)
    for trial_row in sorted(set(trial.tolist())):
        mask = trial == trial_row
        valid = mask & np.isfinite(rate)
        ax.plot(
            threshold[valid],
            rate[valid],
            marker="o",
            linewidth=1.3,
            label=f"trial row {trial_row}",
        )
    ax.set_xlabel("escape speed threshold (mm/s)")
    ax.set_ylabel("escape events/min valid trial time")
    ax.set_title("Threshold sensitivity")
    ax.grid(alpha=0.2)
    if trial.size:
        ax.legend(fontsize=7)


def render_dashboard(
    controller: ComposableChaserSuccessorSourceHandle,
    bout: ComposableChaserSuccessorSourceHandle,
    escape: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render one exact dashboard to PNG and PDF."""

    _verify_chain(controller, bout, escape)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    _trial_panel(axes[0, 0], controller)
    _bout_panel(axes[0, 1], bout)
    _response_panel(axes[1, 0], escape)
    _sweep_panel(axes[1, 1], escape)
    projection = bout.scientific_manifest["sources"]["motion"].get(
        "relative_frame_projection", {}
    )
    missing = projection.get("missing_relative_frame_count", "unknown")
    figure.suptitle(
        f"Composable chaser analytics · {controller.recording_id}\n"
        "exact logged trials · selector-ineligible · no interpolation · "
        f"provider-missing relative frames: {missing}",
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
    stem = output_dir / f"{args.run_name}_dashboard"
    receipt_path = output_dir / f"{args.run_name}_plot_receipt.json"
    expected_outputs = (stem.with_suffix(".png"), stem.with_suffix(".pdf"), receipt_path)
    if not args.overwrite and any(path.exists() for path in expected_outputs):
        raise FileExistsError("Plot output already exists; pass --overwrite explicitly.")
    handles = tuple(
        load_composable_chaser_successor_source_handle(
            archive,
            successor_kind=kind,
            run_name=args.run_name,
            expected_recording_id=args.expected_recording_id,
            use_consolidated=True,
            deep_audit=True,
        )
        for kind in _KINDS
    )
    controller, bout, escape = handles
    png, pdf = render_dashboard(controller, bout, escape, output_stem=stem)
    source_bindings = {
        handle.successor_kind: {
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "scientific_payload_sha256": handle.scientific_payload_sha256,
        }
        for handle in handles
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
        "recording_id": controller.recording_id,
        "run_name": args.run_name,
        "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_json_sha256(source_bindings),
        "outputs": outputs,
        "plot_policy": {
            "source_selection": "explicit_run_names_only_no_selector_discovery",
            "source_validation": "strict_loader_deep_array_content_audit",
            "dependency_validation": "exact_scientific_payload_digests",
            "missing_provider_frames": "invalid_no_interpolation",
            "scientific_authority": False,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
    }
    receipt = {**body, "payload_sha256": canonical_json_sha256(body)}
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({**receipt, "receipt_path": str(receipt_path)}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "ComposableChaserPlotError",
    "main",
    "render_dashboard",
]
