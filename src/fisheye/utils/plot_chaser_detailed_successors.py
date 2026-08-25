"""Render detailed plots from exact immutable chaser successor products.

The command resolves no selector and writes nothing to the analysis Zarr.  It
deep-audits an explicitly named controller/bout/escape chain, its exact
relative-frame source, and explicit keypoint and detection radial successors.
All panels are direct views or plotting-coordinate transforms of persisted
arrays; no missing samples are interpolated and no scientific table is
recomputed inside the plotter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    load_chaser_relative_frame_source_handle,
)
from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    DETAILED_PLOT_BASE_ARRAY_NAMES,
    load_chaser_relative_frame_targeted_source_handle,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    ComposableChaserSuccessorSourceHandle,
    load_composable_chaser_successor_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


RECEIPT_SCHEMA_ID = "palette.analysis.chaser_detailed_plot_bundle.receipt"
RECEIPT_SCHEMA_VERSION = 2
PLOT_RECIPE_ID = "sealed_chaser_detailed_plot_bundle_v2"
PLOT_DPI = 180
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_CHAIN_KINDS = (
    "controller_chase_trials",
    "generalized_chaser_bout_response",
    "chaser_escape_freeze",
)


class ChaserDetailedPlotError(ValueError):
    """Raised when detailed plot inputs are missing, stale, or mixed."""


def _fail(message: str) -> None:
    raise ChaserDetailedPlotError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(handle: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(handle.array(name))
    except KeyError:
        _fail(f"{getattr(handle, 'successor_kind', 'input')} lacks array {name!r}.")


def _registry(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    registries = manifest.get("identity_registries")
    if not isinstance(registries, Mapping):
        return {}
    value = registries.get(name)
    return value if isinstance(value, Mapping) else {}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_figure(figure: Any, output_stem: Path) -> tuple[Path, Path]:
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


def verify_detailed_plot_inputs(
    controller: ComposableChaserSuccessorSourceHandle,
    bout: ComposableChaserSuccessorSourceHandle,
    escape: ComposableChaserSuccessorSourceHandle,
    relative_keypoint: Any,
    relative_detection: Any,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    radial_detection: ComposableChaserSuccessorSourceHandle,
) -> None:
    """Reject mixed recordings, dependencies, providers, or shallow handles."""

    chain = (controller, bout, escape)
    if tuple(handle.successor_kind for handle in chain) != _CHAIN_KINDS:
        _fail("Detailed plot chain has missing, reordered, or incorrect products.")
    radial = (radial_keypoint, radial_detection)
    if any(handle.successor_kind != "chaser_radial_near_field" for handle in radial):
        _fail("Detailed provider comparison requires two radial/near-field products.")
    if len(
        {handle.recording_id for handle in (*chain, *radial)}
        | {relative_keypoint.recording_id, relative_detection.recording_id}
    ) != 1:
        _fail("Detailed plot inputs belong to different recordings.")
    if not all(handle.deep_audited is True for handle in (*chain, *radial)):
        _fail("Composable detailed plot inputs require deep array-content audits.")

    bout_sources = bout.scientific_manifest.get("sources")
    escape_sources = escape.scientific_manifest.get("sources")
    controller_source = controller.scientific_manifest.get("source_relative_frame")
    if not all(isinstance(value, Mapping) for value in (bout_sources, escape_sources, controller_source)):
        _fail("Detailed plot dependency bindings are absent.")
    if (
        bout_sources.get("controller_trial_payload_sha256")
        != controller.scientific_payload_sha256
        or escape_sources.get("controller_trial_payload_sha256")
        != controller.scientific_payload_sha256
        or escape_sources.get("bout_response_payload_sha256")
        != bout.scientific_payload_sha256
    ):
        _fail("Detailed plot chain contains stale or mixed payload dependencies.")
    if (
        controller_source.get("manifest_sha256") != relative_keypoint.manifest_sha256
        or controller_source.get("run_path") != relative_keypoint.run_path
    ):
        _fail("Controller trials are not bound to the supplied relative-frame run.")

    providers = []
    for handle, relative in zip(radial, (relative_keypoint, relative_detection)):
        provider = handle.scientific_manifest.get("position_provider")
        sources = handle.scientific_manifest.get("sources")
        if not isinstance(provider, Mapping) or not isinstance(sources, Mapping):
            _fail("A radial comparison product lacks provider/source bindings.")
        if provider.get("status") != "first_class_explicit_authority":
            _fail("A radial position provider is not a first-class explicit authority.")
        relative_source = sources.get("relative_frame")
        if not isinstance(relative_source, Mapping) or (
            relative_source.get("run_path") != relative.run_path
            or relative_source.get("manifest_sha256") != relative.manifest_sha256
        ):
            _fail("A radial product is not bound to its supplied relative-frame run.")
        providers.append(str(provider.get("provider_id")))
    if len(set(providers)) != 2:
        _fail("Provider comparison requires two distinct explicit position providers.")

    radial_sources = [handle.scientific_manifest["sources"] for handle in radial]
    for source_name in ("protocol_semantic_selection", "arena_geometry_and_scale"):
        values = [source.get(source_name) for source in radial_sources]
        if not all(isinstance(value, Mapping) for value in values) or (
            canonical_json_sha256(_plain(values[0]))
            != canonical_json_sha256(_plain(values[1]))
        ):
            _fail(f"Provider comparison has mismatched {source_name} bindings.")
    for policy_name in ("coordinate_policy", "scale_policy"):
        values = [relative.manifest.get(policy_name) for relative in (relative_keypoint, relative_detection)]
        if not all(isinstance(value, Mapping) for value in values) or (
            canonical_json_sha256(_plain(values[0]))
            != canonical_json_sha256(_plain(values[1]))
        ):
            _fail(f"Provider relative frames have mismatched {policy_name} bindings.")
    for array_name in (
        "acquisition_frame_id",
        "timestamp_ns",
        "timestamp_valid",
        "selection_member",
        "chaser_identity_code",
        "chaser_occurrence_member",
        "chaser_behavior_role_code",
        "chaser_position_xy_px",
        "chaser_position_valid",
    ):
        if not np.array_equal(
            relative_keypoint.base_array(array_name),
            relative_detection.base_array(array_name),
        ):
            _fail(
                "Provider comparison does not preserve identical chaser/timing "
                f"evidence array {array_name!r}."
            )


def _radial_cdf_rows(handle: Any) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    epoch = _array(handle, "cdf_epoch_role_code").astype(np.int64)
    behavior = _array(handle, "cdf_behavior_role_code").astype(np.int64)
    chaser = _array(handle, "cdf_chaser_identity_code").astype(np.int64)
    threshold = _array(handle, "cdf_threshold_mm").astype(np.float64)
    fraction = _array(handle, "cdf_fraction_at_or_below").astype(np.float64)
    if not (epoch.size == behavior.size == chaser.size == threshold.size == fraction.size):
        _fail("Radial CDF table columns have different lengths.")
    result = {}
    for key in sorted(set(zip(epoch.tolist(), behavior.tolist(), chaser.tolist()))):
        mask = (epoch == key[0]) & (behavior == key[1]) & (chaser == key[2])
        order = np.argsort(threshold[mask])
        result[key] = (threshold[mask][order], fraction[mask][order])
    return result


def render_provider_distance_cdf(
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    radial_detection: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Plot exact persisted distance CDFs for two first-class providers."""

    handles = (radial_keypoint, radial_detection)
    rows = tuple(_radial_cdf_rows(handle) for handle in handles)
    if set(rows[0]) != set(rows[1]) or not rows[0]:
        _fail("Provider CDF products do not expose the same non-empty strata.")
    keys = sorted(rows[0])
    columns = min(3, len(keys))
    row_count = int(math.ceil(len(keys) / columns))
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(5.2 * columns, 3.7 * row_count),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    epoch_registry = _registry(radial_keypoint.scientific_manifest, "epoch_role")
    behavior_registry = _registry(radial_keypoint.scientific_manifest, "behavior_role")
    provider_names = [
        str(handle.scientific_manifest["position_provider"]["provider_id"])
        for handle in handles
    ]
    for index, key in enumerate(keys):
        ax = axes.flat[index]
        for provider_index, (name, provider_rows) in enumerate(zip(provider_names, rows)):
            x, y = provider_rows[key]
            valid = np.isfinite(x) & np.isfinite(y)
            ax.plot(
                x[valid],
                y[valid],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=name,
                color=("#1f77b4", "#d95f02")[provider_index],
            )
        epoch_name = epoch_registry.get(str(key[0]), f"epoch {key[0]}")
        behavior_name = behavior_registry.get(str(key[1]), f"role {key[1]}")
        ax.set_title(f"{epoch_name} · {behavior_name} · chaser {key[2]}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        if index == 0:
            ax.legend(fontsize=8)
    for index in range(len(keys), axes.size):
        axes.flat[index].set_visible(False)
    figure.supxlabel("fish–chaser distance threshold (mm)")
    figure.supylabel("fraction of valid rows at or below threshold")
    figure.suptitle(
        f"First-class position-provider distance CDF comparison · {radial_keypoint.recording_id}\n"
        "matched chaser/timestamp arrays and semantic/geometry authorities · no interpolation",
        fontsize=13,
    )
    return _save_figure(figure, output_stem)


def _distance_band_labels(bout: Any) -> tuple[np.ndarray, list[str]]:
    band = _array(bout, "summary_distance_bin_index").astype(np.int64)
    low = _array(bout, "summary_distance_bin_start_mm").astype(np.float64)
    high = _array(bout, "summary_distance_bin_end_mm").astype(np.float64)
    labels = {}
    for code, lo, hi in zip(band, low, high):
        labels[int(code)] = f"{lo:g}–∞" if not np.isfinite(hi) else f"{lo:g}–{hi:g}"
    codes = np.asarray(sorted(labels), dtype=np.int64)
    return codes, [labels[int(code)] for code in codes]


def render_bout_response_details(
    bout: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render persisted generalized bout-response summary tables as heatmaps."""

    role = _array(bout, "summary_role_code").astype(np.int64)
    chaser = _array(bout, "summary_chaser_identity_code").astype(np.int64)
    band = _array(bout, "summary_distance_bin_index").astype(np.int64)
    if not (role.size == chaser.size == band.size):
        _fail("Bout summary identity columns have different lengths.")
    series = sorted(set(zip(role.tolist(), chaser.tolist())))
    band_codes, band_labels = _distance_band_labels(bout)
    if not series or not band_codes.size:
        _fail("Bout summary has no plot strata.")
    role_registry = _registry(bout.scientific_manifest, "semantic_role")
    series_labels = [
        f"{role_registry.get(str(code), f'role {code}')} · chaser {identity}"
        for code, identity in series
    ]
    metrics = (
        ("summary_bout_rate_per_min", "Bout rate per valid minute", ".1f"),
        ("summary_median_peak_speed_mm_s", "Median peak speed (mm/s)", ".1f"),
        ("summary_median_net_displacement_mm", "Median net displacement (mm)", ".2f"),
        ("summary_median_duration_s", "Median bout duration (s)", ".3f"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (array_name, title, format_spec) in zip(axes.flat, metrics):
        values = _array(bout, array_name).astype(np.float64)
        if values.size != role.size:
            _fail(f"Bout summary metric {array_name!r} has the wrong length.")
        matrix = np.full((len(series), band_codes.size), np.nan, dtype=np.float64)
        for row_index, key in enumerate(series):
            for column_index, band_code in enumerate(band_codes):
                matches = np.flatnonzero(
                    (role == key[0]) & (chaser == key[1]) & (band == band_code)
                )
                if matches.size > 1:
                    _fail("Bout summary contains a duplicated semantic/chaser/band row.")
                if matches.size == 1:
                    matrix[row_index, column_index] = values[matches[0]]
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(band_codes.size), band_labels)
        ax.set_yticks(np.arange(len(series)), series_labels)
        ax.set_xlabel("distance at bout onset (mm)")
        ax.set_title(title)
        for row_index, column_index in np.argwhere(np.isfinite(matrix)):
            ax.text(
                column_index,
                row_index,
                format(matrix[row_index, column_index], format_spec),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row_index, column_index] > np.nanmedian(matrix) else "black",
            )
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"Generalized bout-response details · {bout.recording_id}\n"
        "persisted summary rows · valid-time denominators · no pooled-frame inference",
        fontsize=13,
    )
    return _save_figure(figure, output_stem)


def render_trial_escape_details(
    escape: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render persisted per-trial and per-event escape/freeze evidence."""

    ordinal = _array(escape, "trial_ordinal").astype(np.int64)
    logged = _array(escape, "trial_logged_id").astype(np.int64)
    response = _array(escape, "trial_response_class_code").astype(np.int64)
    count = _array(escape, "trial_escape_event_count").astype(np.int64)
    rate = _array(escape, "trial_escape_event_rate_per_min").astype(np.float64)
    latency = _array(escape, "trial_first_escape_latency_s").astype(np.float64)
    trigger_distance = _array(escape, "trial_trigger_distance_mm").astype(np.float64)
    freeze_fraction = _array(escape, "trial_freeze_low_speed_fraction").astype(np.float64)
    freeze_coverage = _array(escape, "trial_freeze_valid_fraction").astype(np.float64)
    recapture_fraction = _array(escape, "trial_recapture_fraction").astype(np.float64)
    separation_gain = _array(escape, "trial_mean_separation_gain_mm").astype(np.float64)
    columns = (
        logged,
        response,
        count,
        rate,
        latency,
        trigger_distance,
        freeze_fraction,
        freeze_coverage,
        recapture_fraction,
        separation_gain,
    )
    if not ordinal.size or any(value.size != ordinal.size for value in columns):
        _fail("Escape/freeze per-trial columns have different or empty lengths.")
    order = np.argsort(ordinal)
    ordinal = ordinal[order]
    response_registry = _registry(escape.scientific_manifest, "response_class")

    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    ax = axes[0, 0]
    response_codes = sorted(set(response.tolist()))
    response_y = {code: index for index, code in enumerate(response_codes)}
    ax.scatter(ordinal, [response_y[int(value)] for value in response[order]], s=55)
    for x, y, trial_id in zip(ordinal, response[order], logged[order]):
        ax.annotate(f"ID {trial_id}", (x, response_y[int(y)]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_yticks(
        range(len(response_codes)),
        [response_registry.get(str(code), f"class {code}") for code in response_codes],
    )
    ax.set_xlabel("trial ordinal")
    ax.set_title("Exact trial response classification")
    ax.grid(alpha=0.2)

    event_trial = _array(escape, "event_controller_trial_row_id").astype(np.int64)
    event_latency = _array(escape, "event_latency_from_trigger_s").astype(np.float64)
    event_peak = _array(escape, "event_peak_speed_mm_s").astype(np.float64)
    event_distance = _array(escape, "event_distance_at_onset_mm").astype(np.float64)
    event_recaptured = _array(escape, "event_recaptured").astype(bool)
    if not (
        event_trial.size
        == event_latency.size
        == event_peak.size
        == event_distance.size
        == event_recaptured.size
    ):
        _fail("Escape event table columns have different lengths.")
    if np.any((event_trial < 0) | (event_trial >= ordinal.size)):
        _fail("Escape event references an unavailable controller-trial row.")
    event_ordinal = _array(escape, "trial_ordinal").astype(np.int64)[event_trial]
    ax = axes[0, 1]
    valid_event = np.isfinite(event_latency) & np.isfinite(event_peak)
    if np.any(valid_event):
        scatter = ax.scatter(
            event_latency[valid_event],
            event_ordinal[valid_event],
            c=event_peak[valid_event],
            cmap="plasma",
            s=55,
            edgecolors=np.where(event_recaptured[valid_event], "black", "none"),
        )
        figure.colorbar(scatter, ax=ax, label="peak speed (mm/s)")
    else:
        ax.text(0.5, 0.5, "no valid escape events", transform=ax.transAxes, ha="center")
    ax.set_xlabel("event latency from trigger (s)")
    ax.set_ylabel("trial ordinal")
    ax.set_title("Escape-event raster; black edge = recaptured")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for recaptured, marker, label in ((False, "x", "not recaptured"), (True, "o", "recaptured")):
        valid = event_recaptured == recaptured
        valid &= np.isfinite(event_distance) & np.isfinite(event_peak)
        if np.any(valid):
            ax.scatter(event_distance[valid], event_peak[valid], marker=marker, s=55, label=label)
    ax.set_xlabel("distance at escape onset (mm)")
    ax.set_ylabel("peak speed (mm/s)")
    ax.set_title("Per-event onset state")
    ax.grid(alpha=0.2)
    if event_trial.size:
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    bars = ax.bar(ordinal, count[order], color="#d95f02", alpha=0.75, label="event count")
    ax.bar_label(bars, fontsize=8)
    ax.set_xlabel("trial ordinal")
    ax.set_ylabel("escape event count")
    rate_ax = ax.twinx()
    rate_ax.plot(ordinal, rate[order], color="#1f77b4", marker="o", label="events/min")
    rate_ax.set_ylabel("escape events/min valid time")
    ax.set_title("Escape count and exact-time rate")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2, 0]
    valid_latency = np.isfinite(latency[order])
    ax.plot(ordinal[valid_latency], latency[order][valid_latency], marker="o", color="#d62728")
    ax.set_xlabel("trial ordinal")
    ax.set_ylabel("first escape latency (s)", color="#d62728")
    trigger_ax = ax.twinx()
    valid_trigger = np.isfinite(trigger_distance[order])
    trigger_ax.plot(ordinal[valid_trigger], trigger_distance[order][valid_trigger], marker="s", color="#2ca02c")
    trigger_ax.set_ylabel("trigger distance (mm)", color="#2ca02c")
    ax.set_title("First-event latency and trigger distance")
    ax.grid(alpha=0.2)

    ax = axes[2, 1]
    ax.plot(ordinal, freeze_fraction[order], marker="o", label="freeze low-speed fraction")
    ax.plot(ordinal, freeze_coverage[order], marker="s", label="freeze-window coverage")
    ax.plot(ordinal, recapture_fraction[order], marker="^", label="recapture fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("trial ordinal")
    ax.set_ylabel("fraction")
    gain_ax = ax.twinx()
    valid_gain = np.isfinite(separation_gain[order])
    gain_ax.plot(ordinal[valid_gain], separation_gain[order][valid_gain], color="black", linestyle="--", marker="d", label="mean separation gain")
    gain_ax.set_ylabel("mean separation gain (mm)")
    ax.set_title("Freeze, recapture, and separation evidence")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="lower left")

    figure.suptitle(
        f"Trial and escape/freeze details · {escape.recording_id}\n"
        "exact logged trials · persisted event rows · selector-ineligible",
        fontsize=13,
    )
    return _save_figure(figure, output_stem)


def render_trial_distance_traces(
    controller: ComposableChaserSuccessorSourceHandle,
    relative: Any,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Plot direct distance rows in exact session time aligned to trial triggers."""

    start = _array(controller, "start_source_frame_row").astype(np.int64)
    end = _array(controller, "end_source_frame_row_exclusive").astype(np.int64)
    ordinal = _array(controller, "trial_ordinal").astype(np.int64)
    logged = _array(controller, "logged_trial_id").astype(np.int64)
    chaser_code = _array(controller, "chaser_identity_code").astype(np.int64)
    trigger = _array(controller, "trigger_timestamp_ns").astype(np.int64)
    trigger_valid = _array(controller, "trigger_timestamp_valid").astype(bool)
    if not start.size or any(
        value.size != start.size
        for value in (end, ordinal, logged, chaser_code, trigger, trigger_valid)
    ):
        _fail("Controller trace index columns have different or empty lengths.")
    if np.any((start < 0) | (end <= start) | (end > relative.n_frames)):
        _fail("Controller trial trace bounds leave the relative-frame row axis.")

    distance = np.asarray(relative.base_frame_chaser("relative_distance_physical"), dtype=np.float64)
    distance_valid = np.asarray(relative.base_frame_chaser("relative_physical_valid"), dtype=bool)
    timestamp = np.asarray(relative.base_frame_chaser("timestamp_ns"), dtype=np.int64)
    timestamp_valid = np.asarray(relative.base_frame_chaser("timestamp_valid"), dtype=bool)
    identity = np.asarray(relative.base_frame_chaser("chaser_identity_code"), dtype=np.int64)
    active = _array(controller, "logged_active_trial_member").astype(bool)
    if active.size != relative.n_rows:
        _fail("Controller active membership does not match the relative-frame row axis.")
    active = active.reshape(relative.n_frames, relative.n_chasers)

    columns = min(2, start.size)
    row_count = int(math.ceil(start.size / columns))
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(7.2 * columns, 3.8 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    for trial_row in range(start.size):
        ax = axes.flat[trial_row]
        frame_slice = slice(int(start[trial_row]), int(end[trial_row]))
        identities = np.unique(identity[frame_slice])
        matching_columns = np.flatnonzero(
            np.all(identity[frame_slice] == chaser_code[trial_row], axis=0)
        )
        if matching_columns.size != 1 or chaser_code[trial_row] not in identities:
            _fail("Trial chaser identity does not map to one relative-frame column.")
        column = int(matching_columns[0])
        if not trigger_valid[trial_row]:
            _fail("A plotted controller trial lacks an exact trigger timestamp.")
        times = (timestamp[frame_slice, column] - trigger[trial_row]) / 1e9
        valid = timestamp_valid[frame_slice, column] & distance_valid[frame_slice, column]
        values = distance[frame_slice, column].copy()
        values[~valid] = np.nan
        ax.plot(times, values, linewidth=1.2, color="#1f77b4", label="valid distance")
        active_valid = active[frame_slice, column] & valid
        ax.scatter(times[active_valid], values[active_valid], s=4, color="#d95f02", alpha=0.55, label="exact active member")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("session time from trigger (s)")
        ax.set_ylabel("fish–chaser distance (mm)")
        ax.set_title(
            f"trial {ordinal[trial_row]} · logged ID {logged[trial_row]} · chaser {chaser_code[trial_row]}"
        )
        ax.grid(alpha=0.2)
        if trial_row == 0:
            ax.legend(fontsize=8)
    for trial_row in range(start.size, axes.size):
        axes.flat[trial_row].set_visible(False)
    provider = relative.source_authorities["fish_position"]["provider_id"]
    figure.suptitle(
        f"Trigger-aligned direct distance traces · {controller.recording_id}\n"
        f"position provider: {provider} · exact session time · no interpolation",
        fontsize=13,
    )
    return _save_figure(figure, output_stem)


def detailed_plot_parameters(
    controller: ComposableChaserSuccessorSourceHandle,
    bout: ComposableChaserSuccessorSourceHandle,
    escape: ComposableChaserSuccessorSourceHandle,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    radial_detection: ComposableChaserSuccessorSourceHandle,
) -> dict[str, Any]:
    """Return every numerical coordinate and rendering choice in the bundle."""

    band = _array(bout, "summary_distance_bin_index").astype(np.int64)
    low = _array(bout, "summary_distance_bin_start_mm").astype(np.float64)
    high = _array(bout, "summary_distance_bin_end_mm").astype(np.float64)
    if not (band.size == low.size == high.size):
        _fail("Bout plot distance-bin columns have different lengths.")
    distance_bins: dict[int, dict[str, Any]] = {}
    for code, start, end in zip(band, low, high, strict=True):
        record = {
            "bin_index": int(code),
            "start_mm_inclusive": float(start),
            "end_mm_exclusive": float(end) if np.isfinite(end) else None,
        }
        previous = distance_bins.setdefault(int(code), record)
        if previous != record:
            _fail("Bout plot distance-bin evidence is inconsistent.")

    radial_handles = (radial_keypoint, radial_detection)
    cdf_parameters = []
    cdf_rows = []
    for handle in radial_handles:
        thresholds = _array(handle, "cdf_threshold_mm").astype(np.float64)
        if np.any(~np.isfinite(thresholds)):
            _fail("Radial CDF plot contains non-finite thresholds.")
        rows = _radial_cdf_rows(handle)
        cdf_rows.append(rows)
        config = handle.scientific_manifest.get("config")
        provider = handle.scientific_manifest.get("position_provider")
        if not isinstance(config, Mapping) or not isinstance(provider, Mapping):
            _fail("Radial plot config or provider authority is absent.")
        cdf_parameters.append(
            {
                "provider_id": str(provider["provider_id"]),
                "cdf_thresholds_mm": [
                    float(value) for value in np.unique(thresholds)
                ],
                "radial_config": _plain(config),
            }
        )
    if set(cdf_rows[0]) != set(cdf_rows[1]) or not cdf_rows[0]:
        _fail("Provider CDF plot strata are empty or mismatched.")
    cdf_panel_count = len(cdf_rows[0])
    cdf_columns = min(3, cdf_panel_count)
    cdf_rows_count = int(math.ceil(cdf_panel_count / cdf_columns))

    sweep_thresholds = _array(
        escape, "sweep_speed_threshold_mm_s"
    ).astype(np.float64)
    if np.any(~np.isfinite(sweep_thresholds)):
        _fail("Escape plot threshold sweep contains non-finite values.")
    trial_count = int(_array(controller, "trial_ordinal").size)
    if trial_count <= 0:
        _fail("Detailed trace plot has no exact trials.")
    trace_columns = min(2, trial_count)
    trace_rows = int(math.ceil(trial_count / trace_columns))
    return {
        "scientific_coordinates": {
            "provider_distance_cdf": cdf_parameters,
            "bout_distance_bins": [distance_bins[key] for key in sorted(distance_bins)],
            "escape_speed_thresholds_mm_s": [
                float(value) for value in np.unique(sweep_thresholds)
            ],
            "escape_response_class_registry": _plain(
                _registry(escape.scientific_manifest, "response_class")
            ),
            "trial_trace_time_reference": "timestamp_ns_minus_exact_trigger_timestamp_ns",
            "trial_trace_distance_array": "relative_distance_physical",
            "trial_trace_membership_overlay": "logged_active_trial_member",
            "missing_value_policy": "remain_missing_no_interpolation",
        },
        "rendering": {
            "png_dpi": PLOT_DPI,
            "pdf_mode": "vector",
            "constrained_layout": True,
            "provider_distance_cdf": {
                "subplot_grid": [cdf_rows_count, cdf_columns],
                "figure_size_inches": [
                    5.2 * cdf_columns,
                    3.7 * cdf_rows_count,
                ],
                "shared_x_axis": True,
                "shared_y_axis": True,
                "y_limits": [-0.02, 1.02],
                "line_marker": "o",
                "marker_size_points": 3.0,
                "line_width_points": 1.4,
                "provider_colors": ["#1f77b4", "#d95f02"],
            },
            "bout_response_details": {
                "subplot_grid": [2, 2],
                "figure_size_inches": [15.0, 10.0],
                "colormap": "viridis",
                "matrix_aspect": "auto",
                "annotation_formats": [".1f", ".1f", ".2f", ".3f"],
            },
            "trial_escape_details": {
                "subplot_grid": [3, 2],
                "figure_size_inches": [15.0, 13.0],
                "event_colormap": "plasma",
                "scatter_size_points_squared": 55.0,
                "fraction_y_limits": [-0.05, 1.05],
            },
            "trial_distance_traces": {
                "subplot_grid": [trace_rows, trace_columns],
                "figure_size_inches": [
                    7.2 * trace_columns,
                    3.8 * trace_rows,
                ],
                "distance_line_width_points": 1.2,
                "distance_color": "#1f77b4",
                "active_member_marker_size_points_squared": 4.0,
                "active_member_color": "#d95f02",
                "active_member_alpha": 0.55,
                "trigger_line_x_seconds": 0.0,
            },
        },
        "output_families": [
            "provider_distance_cdf",
            "bout_response_details",
            "trial_escape_details",
            "trial_distance_traces",
        ],
    }


def render_detailed_bundle(
    controller: ComposableChaserSuccessorSourceHandle,
    bout: ComposableChaserSuccessorSourceHandle,
    escape: ComposableChaserSuccessorSourceHandle,
    relative_keypoint: Any,
    relative_detection: Any,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    radial_detection: ComposableChaserSuccessorSourceHandle,
    *,
    output_dir: Path,
    bundle_name: str,
) -> tuple[Path, ...]:
    """Validate all sources and render the four detailed figure families."""

    verify_detailed_plot_inputs(
        controller,
        bout,
        escape,
        relative_keypoint,
        relative_detection,
        radial_keypoint,
        radial_detection,
    )
    outputs = []
    for renderer, suffix, args in (
        (render_provider_distance_cdf, "provider_distance_cdf", (radial_keypoint, radial_detection)),
        (render_bout_response_details, "bout_response_details", (bout,)),
        (render_trial_escape_details, "trial_escape_details", (escape,)),
        (
            render_trial_distance_traces,
            "trial_distance_traces",
            (controller, relative_keypoint),
        ),
    ):
        outputs.extend(
            renderer(*args, output_stem=output_dir / f"{bundle_name}_{suffix}")
        )
    return tuple(outputs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--relative-frame-run", required=True)
    parser.add_argument("--detection-relative-frame-run", required=True)
    parser.add_argument("--keypoint-relative-frame-receipt")
    parser.add_argument("--detection-relative-frame-receipt")
    parser.add_argument("--keypoint-radial-run", required=True)
    parser.add_argument("--detection-radial-run", required=True)
    parser.add_argument("--expected-recording-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _name(value: str, *, label: str) -> str:
    if _RUN_NAME_RE.fullmatch(value) is None:
        _fail(f"{label} must be one exact non-selector name.")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_name = _name(args.run_name, label="run_name")
    relative_run = _name(args.relative_frame_run, label="relative_frame_run")
    detection_relative_run = _name(
        args.detection_relative_frame_run,
        label="detection_relative_frame_run",
    )
    keypoint_run = _name(args.keypoint_radial_run, label="keypoint_radial_run")
    detection_run = _name(args.detection_radial_run, label="detection_radial_run")
    bundle_name = _name(args.bundle_name, label="bundle_name")
    archive = args.analysis_zarr.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    receipt_path = output_dir / f"{bundle_name}_receipt.json"
    suffixes = (
        "provider_distance_cdf",
        "bout_response_details",
        "trial_escape_details",
        "trial_distance_traces",
    )
    expected = tuple(
        output_dir / f"{bundle_name}_{suffix}.{extension}"
        for suffix in suffixes
        for extension in ("png", "pdf")
    ) + (receipt_path,)
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("Detailed plot output already exists; pass --overwrite explicitly.")

    controller, bout, escape = tuple(
        load_composable_chaser_successor_source_handle(
            archive,
            successor_kind=kind,
            run_name=run_name,
            expected_recording_id=args.expected_recording_id,
            use_consolidated=True,
            deep_audit=True,
        )
        for kind in _CHAIN_KINDS
    )
    receipt_values = (
        args.keypoint_relative_frame_receipt,
        args.detection_relative_frame_receipt,
    )
    if any(value is not None for value in receipt_values) and not all(
        value is not None for value in receipt_values
    ):
        _fail("Keypoint and detection relative-frame receipts must be supplied together.")
    if all(value is not None for value in receipt_values):
        relative_keypoint = load_chaser_relative_frame_targeted_source_handle(
            args.keypoint_relative_frame_receipt,
            required_base_arrays=DETAILED_PLOT_BASE_ARRAY_NAMES,
            collapsed_frame_arrays=(),
            expected_analysis_zarr=archive,
            expected_recording_id=args.expected_recording_id,
            expected_run_name=relative_run,
        )
        relative_detection = load_chaser_relative_frame_targeted_source_handle(
            args.detection_relative_frame_receipt,
            required_base_arrays=DETAILED_PLOT_BASE_ARRAY_NAMES,
            collapsed_frame_arrays=(),
            expected_analysis_zarr=archive,
            expected_recording_id=args.expected_recording_id,
            expected_run_name=detection_relative_run,
        )
    else:
        relative_keypoint = load_chaser_relative_frame_source_handle(
            archive,
            run_name=relative_run,
            expected_recording_id=args.expected_recording_id,
            use_consolidated=True,
        )
        relative_detection = load_chaser_relative_frame_source_handle(
            archive,
            run_name=detection_relative_run,
            expected_recording_id=args.expected_recording_id,
            use_consolidated=True,
        )
    radial_keypoint = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_radial_near_field",
        run_name=keypoint_run,
        expected_recording_id=args.expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
    )
    radial_detection = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_radial_near_field",
        run_name=detection_run,
        expected_recording_id=args.expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
    )
    outputs = render_detailed_bundle(
        controller,
        bout,
        escape,
        relative_keypoint,
        relative_detection,
        radial_keypoint,
        radial_detection,
        output_dir=output_dir,
        bundle_name=bundle_name,
    )
    plot_parameters = detailed_plot_parameters(
        controller,
        bout,
        escape,
        radial_keypoint,
        radial_detection,
    )
    source_bindings = {
        "controller_chase_trials": {
            "run_path": controller.run_path,
            "manifest_sha256": controller.manifest_sha256,
            "scientific_payload_sha256": controller.scientific_payload_sha256,
        },
        "generalized_chaser_bout_response": {
            "run_path": bout.run_path,
            "manifest_sha256": bout.manifest_sha256,
            "scientific_payload_sha256": bout.scientific_payload_sha256,
        },
        "chaser_escape_freeze": {
            "run_path": escape.run_path,
            "manifest_sha256": escape.manifest_sha256,
            "scientific_payload_sha256": escape.scientific_payload_sha256,
        },
        "relative_frame_keypoint": {
            "run_path": relative_keypoint.run_path,
            "manifest_sha256": relative_keypoint.manifest_sha256,
            "payload_sha256": relative_keypoint.payload_digest,
            "verification_mode": relative_keypoint.verification_mode,
            "validation_receipt_sha256": getattr(
                relative_keypoint, "receipt_digest", None
            ),
        },
        "relative_frame_detection": {
            "run_path": relative_detection.run_path,
            "manifest_sha256": relative_detection.manifest_sha256,
            "payload_sha256": relative_detection.payload_digest,
            "verification_mode": relative_detection.verification_mode,
            "validation_receipt_sha256": getattr(
                relative_detection, "receipt_digest", None
            ),
        },
        "radial_keypoint": {
            "run_path": radial_keypoint.run_path,
            "manifest_sha256": radial_keypoint.manifest_sha256,
            "scientific_payload_sha256": radial_keypoint.scientific_payload_sha256,
        },
        "radial_detection": {
            "run_path": radial_detection.run_path,
            "manifest_sha256": radial_detection.manifest_sha256,
            "scientific_payload_sha256": radial_detection.scientific_payload_sha256,
        },
    }
    output_records = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in outputs
    ]
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "plot_recipe_id": PLOT_RECIPE_ID,
        "recording_id": controller.recording_id,
        "bundle_name": bundle_name,
        "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_json_sha256(source_bindings),
        "outputs": output_records,
        "plot_parameters": plot_parameters,
        "plot_parameters_sha256": canonical_json_sha256(plot_parameters),
        "plot_policy": {
            "source_selection": "explicit_exact_run_names_only_no_selector_discovery",
            "source_validation": (
                "deep_successor_audits_plus_receipt_bound_targeted_relative_array_rehash"
                if getattr(relative_keypoint, "receipt_digest", None) is not None
                else "deep_array_content_audit"
            ),
            "scientific_arrays": "persisted_arrays_only",
            "plot_transforms": "sorting_masking_and_exact_trigger_time_subtraction_only",
            "interpolation": "prohibited",
            "missing_rows": "remain_missing",
            "scientific_authority": False,
        },
        "unavailable_recipe_evidence": {
            "individual_near_visits": "not_persisted_in_radial_successor_v1",
            "escape_aligned_distance_traces": "not_persisted_in_escape_successor_v2",
            "ring_entry_video": "deferred_sealed_video_successor",
            "gaze_and_bearing": "reviewed_body_frame_and_eye_orientation_unavailable",
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
    "PLOT_RECIPE_ID",
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "ChaserDetailedPlotError",
    "main",
    "detailed_plot_parameters",
    "render_bout_response_details",
    "render_detailed_bundle",
    "render_provider_distance_cdf",
    "render_trial_distance_traces",
    "render_trial_escape_details",
    "verify_detailed_plot_inputs",
]
