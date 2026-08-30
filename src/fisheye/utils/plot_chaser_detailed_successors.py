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
from dataclasses import dataclass
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
from matplotlib.patches import Circle  # noqa: E402
import numpy as np

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    load_chaser_relative_frame_source_handle,
)
from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    DETAILED_PLOT_BASE_ARRAY_NAMES,
    DETAILED_PLOT_BODY_ARRAY_NAMES,
    load_chaser_relative_frame_targeted_source_handle,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    ComposableChaserSuccessorSourceHandle,
    load_composable_chaser_successor_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.chaser_appearance import (
    ChaserAppearanceProjection,
    ChaserAppearanceProjectionError,
    load_chaser_appearance_projection,
)
from fisheye.visualization.chaser_body_bearing_distance import (
    BEARING_BIN_WIDTH_DEG,
    DENSITY_COLOR_CMAX_QUANTILE,
    DISPLAY_RECIPE_ID as BODY_BEARING_DISTANCE_DISPLAY_RECIPE_ID,
    DISTANCE_BIN_WIDTH_MM,
    STATIC_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER,
    BodyBearingDistanceHistogram,
    bearing_bin_edges_deg,
    body_bearing_distance_histogram,
    body_bearing_distance_valid_mask,
    distance_bin_edges_mm,
    positive_probability_color_max,
    uniformly_sample_indices,
)


RECEIPT_SCHEMA_ID = "palette.analysis.chaser_detailed_plot_bundle.receipt"
RECEIPT_SCHEMA_VERSION = 5
PLOT_RECIPE_ID = "sealed_chaser_detailed_plot_bundle_v5"
PLOT_DPI = 180
DENSE_DISPLAY_ALGORITHM = "all_exact_source_rows_rasterized_no_interpolation_v1"
STATIC_TRAJECTORY_ROLE_MARKER_MAX_PER_PANEL_CHASER = 64
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


def _load_exact_chaser_appearance(
    relative_keypoint: Any,
) -> ChaserAppearanceProjection:
    """Bind static display semantics to the same exact protocol as Marimo."""

    try:
        archive = relative_keypoint.analysis_zarr_path
        manifest = relative_keypoint.run_manifest
        recording_id = relative_keypoint.recording_id
    except AttributeError:
        _fail(
            "Static chaser appearance requires an exact relative-frame handle "
            "with archive and run-manifest identity."
        )
    identity = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_identity_code"),
        dtype=np.int64,
    )
    role = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_behavior_role_code"),
        dtype=np.int64,
    )
    expected_shape = (relative_keypoint.n_frames, relative_keypoint.n_chasers)
    if (
        identity.shape != expected_shape
        or role.shape != expected_shape
        or not np.all(identity == identity[:1])
        or not np.all(role == role[:1])
    ):
        _fail("Static chaser appearance has unstable identity or behavior roles.")
    try:
        projection = load_chaser_appearance_projection(
            archive,
            relative_manifest=manifest,
            identity_code_by_column=identity[0],
            behavior_role_code_by_column=role[0],
            expected_recording_id=recording_id,
        )
    except ChaserAppearanceProjectionError as exc:
        _fail(f"Static chaser appearance binding failed: {exc}")
    return _validated_exact_chaser_appearance(relative_keypoint, projection)


def _validated_exact_chaser_appearance(
    relative_keypoint: Any,
    projection: ChaserAppearanceProjection,
) -> ChaserAppearanceProjection:
    """Require a supplied appearance projection to match the plotted columns."""

    if projection.recording_id != relative_keypoint.recording_id:
        _fail("Static chaser appearance belongs to another recording.")
    identity = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_identity_code"),
        dtype=np.int64,
    )
    role = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_behavior_role_code"),
        dtype=np.int64,
    )
    expected_shape = (relative_keypoint.n_frames, relative_keypoint.n_chasers)
    if (
        identity.shape != expected_shape
        or role.shape != expected_shape
        or not np.all(identity == identity[:1])
        or not np.all(role == role[:1])
    ):
        _fail("Static chaser appearance has unstable identity or behavior roles.")
    by_identity = projection.by_identity_code()
    if (
        len(projection.appearances) != relative_keypoint.n_chasers
        or len(by_identity) != relative_keypoint.n_chasers
    ):
        _fail("Static chaser appearance cardinality differs from the plotted axis.")
    for column in range(relative_keypoint.n_chasers):
        appearance = by_identity.get(int(identity[0, column]))
        if appearance is None or appearance.behavior_role_code != int(role[0, column]):
            _fail(
                "Static chaser appearance identity/role differs from the plotted axis."
            )
    return projection


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
    for manifest_name in ("epoch_records", "arena"):
        values = [handle.scientific_manifest.get(manifest_name) for handle in radial]
        if any(value is None for value in values) or (
            canonical_json_sha256({manifest_name: _plain(values[0])})
            != canonical_json_sha256({manifest_name: _plain(values[1])})
        ):
            _fail(f"Provider comparison has mismatched {manifest_name} evidence.")
    _epoch_records(radial_keypoint)
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
    for relative in (relative_keypoint, relative_detection):
        _collapsed_frame_scalar(relative, "acquisition_frame_id")
        _collapsed_frame_scalar(relative, "timestamp_ns")
        _collapsed_frame_scalar(relative, "timestamp_valid")
        _collapsed_frame_scalar(relative, "selection_member")
        _collapsed_fish_frame(relative)
    body_authority = relative_keypoint.source_authorities.get("body_frame")
    if not isinstance(body_authority, Mapping):
        _fail("Keypoint relative frame lacks an explicit body-frame authority.")
    try:
        body_bearing = np.asarray(
            relative_keypoint.body_frame_chaser("body_bearing_deg"),
            dtype=np.float64,
        )
        body_valid = np.asarray(
            relative_keypoint.body_frame_chaser("body_bearing_valid"), dtype=bool
        )
    except (KeyError, ValueError) as exc:
        _fail(f"Keypoint relative frame lacks sealed body-bearing evidence: {exc}")
    expected_body_shape = (
        relative_keypoint.n_frames,
        relative_keypoint.n_chasers,
    )
    if (
        body_bearing.shape != expected_body_shape
        or body_valid.shape != expected_body_shape
    ):
        _fail("Keypoint body-bearing arrays do not preserve frame/chaser shape.")
    _body_bearing_distance_plot_data(relative_keypoint, radial_keypoint)


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


def _epoch_records(handle: Any) -> tuple[dict[str, Any], ...]:
    raw = handle.scientific_manifest.get("epoch_records")
    if not isinstance(raw, (list, tuple)) or not raw:
        _fail("Radial successor lacks exact epoch records for detailed plotting.")
    records = []
    for item in raw:
        if not isinstance(item, Mapping):
            _fail("A radial epoch record is not a mapping.")
        start = item.get("start_frame")
        end = item.get("end_frame_exclusive")
        window_id = item.get("window_id")
        role = item.get("analysis_role")
        if (
            type(start) is not int
            or type(end) is not int
            or end <= start
            or type(window_id) is not int
            or type(role) is not str
            or not role
        ):
            _fail("A radial epoch record has an invalid exact half-open boundary.")
        records.append(
            {
                "window_id": window_id,
                "analysis_role": role,
                "start_frame_inclusive": start,
                "end_frame_exclusive": end,
            }
        )
    records.sort(key=lambda item: (item["start_frame_inclusive"], item["window_id"]))
    if len({item["window_id"] for item in records}) != len(records):
        _fail("Radial epoch records contain duplicate window identities.")
    if any(
        left["end_frame_exclusive"] > right["start_frame_inclusive"]
        for left, right in zip(records, records[1:], strict=False)
    ):
        _fail("Radial epoch records overlap.")
    return tuple(records)


def _collapsed_frame_scalar(relative: Any, name: str) -> np.ndarray:
    values = np.asarray(relative.base_frame_chaser(name))
    if values.ndim != 2 or values.shape != (relative.n_frames, relative.n_chasers):
        _fail(f"Relative-frame scalar {name!r} has an invalid frame/chaser shape.")
    if not np.all(values == values[:, :1]):
        _fail(f"Relative-frame scalar {name!r} is not repeated identically per chaser.")
    return values[:, 0]


def _collapsed_fish_frame(relative: Any) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(
        relative.base_frame_chaser("fish_position_xy_px"), dtype=np.float64
    )
    valid = np.asarray(relative.base_frame_chaser("fish_position_valid"), dtype=bool)
    expected_position_shape = (relative.n_frames, relative.n_chasers, 2)
    expected_valid_shape = (relative.n_frames, relative.n_chasers)
    if positions.shape != expected_position_shape or valid.shape != expected_valid_shape:
        _fail("Fish-position arrays do not preserve frame/chaser/source-xy shape.")
    if not np.all(valid == valid[:, :1]):
        _fail("Fish-position validity is not repeated identically per chaser.")
    repeated = positions[:, :1, :]
    if not np.all((positions == repeated) | (np.isnan(positions) & np.isnan(repeated))):
        _fail("Fish-position values are not repeated identically per chaser.")
    collapsed = positions[:, 0, :]
    collapsed_valid = valid[:, 0] & np.all(np.isfinite(collapsed), axis=1)
    return collapsed, collapsed_valid


@dataclass(frozen=True)
class _BodyBearingDistancePlotData:
    distance_mm: np.ndarray
    bearing_deg: np.ndarray
    frame_id: np.ndarray
    identity: np.ndarray
    role: np.ndarray
    panels: tuple[tuple[str, np.ndarray], ...]
    panel_valid: tuple[np.ndarray, ...]
    distance_edges_mm: np.ndarray
    bearing_edges_deg: np.ndarray
    histograms: tuple[BodyBearingDistanceHistogram, ...]
    role_registry: Mapping[str, Any]


def _body_bearing_distance_plot_data(
    relative_keypoint: Any,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
) -> _BodyBearingDistancePlotData:
    """Build the shared exact-row display projection for static polar figures."""

    distance = np.asarray(
        relative_keypoint.base_frame_chaser("relative_distance_physical"),
        dtype=np.float64,
    )
    distance_valid = np.asarray(
        relative_keypoint.base_frame_chaser("relative_physical_valid"), dtype=bool
    )
    bearing = np.asarray(
        relative_keypoint.body_frame_chaser("body_bearing_deg"), dtype=np.float64
    )
    bearing_valid = np.asarray(
        relative_keypoint.body_frame_chaser("body_bearing_valid"), dtype=bool
    )
    occurrence = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_occurrence_member"), dtype=bool
    )
    identity = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_identity_code"), dtype=np.int64
    )
    role = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_behavior_role_code"),
        dtype=np.int64,
    )
    expected_shape = (
        relative_keypoint.n_frames,
        relative_keypoint.n_chasers,
    )
    if any(
        values.shape != expected_shape
        for values in (
            distance,
            distance_valid,
            bearing,
            bearing_valid,
            occurrence,
            identity,
            role,
        )
    ):
        _fail("Body-bearing distance arrays do not preserve frame/chaser shape.")
    if not np.all(identity == identity[:1]) or not np.all(role == role[:1]):
        _fail("Body-bearing distance has unstable chaser identity or behavior role.")

    frame_id = _collapsed_frame_scalar(
        relative_keypoint, "acquisition_frame_id"
    ).astype(np.int64)
    selected = _collapsed_frame_scalar(relative_keypoint, "selection_member").astype(
        bool
    )
    full_member = np.ones(expected_shape, dtype=bool)
    try:
        full_valid = body_bearing_distance_valid_mask(
            distance,
            bearing,
            distance_valid,
            bearing_valid,
            occurrence,
            full_member,
        )
    except ValueError as exc:
        _fail(str(exc))
    if not np.any(full_valid):
        _fail(
            "Keypoint body-frame evidence has no jointly valid bearing/distance rows."
        )
    distance_edges = distance_bin_edges_mm(distance, full_valid)
    bearing_edges = bearing_bin_edges_deg()
    panels: list[tuple[str, np.ndarray]] = [
        ("full recording", np.ones(relative_keypoint.n_frames, dtype=bool))
    ]
    for record in _epoch_records(radial_keypoint):
        panels.append(
            (
                str(record["analysis_role"]),
                selected
                & (frame_id >= int(record["start_frame_inclusive"]))
                & (frame_id < int(record["end_frame_exclusive"])),
            )
        )
    if any(not np.any(member) for _, member in panels):
        _fail("A body-bearing distance panel has no exact source rows.")

    panel_valid: list[np.ndarray] = []
    histograms: list[BodyBearingDistanceHistogram] = []
    for _label, member in panels:
        try:
            valid = body_bearing_distance_valid_mask(
                distance,
                bearing,
                distance_valid,
                bearing_valid,
                occurrence,
                np.broadcast_to(member[:, None], expected_shape),
            )
        except ValueError as exc:
            _fail(str(exc))
        panel_valid.append(valid)
        for column in range(relative_keypoint.n_chasers):
            try:
                histograms.append(
                    body_bearing_distance_histogram(
                        distance[:, column],
                        bearing[:, column],
                        valid[:, column],
                        distance_edges_mm=distance_edges,
                        bearing_edges_deg=bearing_edges,
                    )
                )
            except ValueError as exc:
                _fail(str(exc))
    return _BodyBearingDistancePlotData(
        distance_mm=distance,
        bearing_deg=bearing,
        frame_id=frame_id,
        identity=identity,
        role=role,
        panels=tuple(panels),
        panel_valid=tuple(panel_valid),
        distance_edges_mm=distance_edges,
        bearing_edges_deg=bearing_edges,
        histograms=tuple(histograms),
        role_registry=_registry(radial_keypoint.scientific_manifest, "behavior_role"),
    )


def _body_bearing_distance_panel_records(
    data: _BodyBearingDistancePlotData,
) -> list[dict[str, Any]]:
    records = []
    histogram_index = 0
    for label, _member in data.panels:
        for column in range(data.identity.shape[1]):
            histogram = data.histograms[histogram_index]
            histogram_index += 1
            records.append(
                {
                    "panel": label,
                    "chaser_identity_code": int(data.identity[0, column]),
                    "valid_row_count": histogram.denominator,
                }
            )
    return records


def _radial_metric_rows(handle: Any) -> dict[tuple[int, int, int], dict[str, float]]:
    columns = {
        "epoch": _array(handle, "metric_epoch_role_code").astype(np.int64),
        "behavior": _array(handle, "metric_behavior_role_code").astype(np.int64),
        "chaser": _array(handle, "metric_chaser_identity_code").astype(np.int64),
        "distance_p25_mm": _array(handle, "metric_distance_p25_mm").astype(np.float64),
        "distance_p50_mm": _array(handle, "metric_distance_p50_mm").astype(np.float64),
        "distance_p75_mm": _array(handle, "metric_distance_p75_mm").astype(np.float64),
        "near_fraction": _array(handle, "metric_near_zone_fraction_valid").astype(np.float64),
        "near_dwell_s": _array(handle, "metric_near_zone_dwell_s").astype(np.float64),
        "entry_rate_per_min": _array(
            handle, "metric_near_zone_entry_rate_per_min_valid_time"
        ).astype(np.float64),
        "valid_distance_count": _array(
            handle, "metric_valid_distance_frame_count"
        ).astype(np.int64),
    }
    lengths = {value.size for value in columns.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        _fail("Radial metric summary columns have different or empty lengths.")
    rows: dict[tuple[int, int, int], dict[str, float]] = {}
    for index in range(columns["epoch"].size):
        key = (
            int(columns["epoch"][index]),
            int(columns["behavior"][index]),
            int(columns["chaser"][index]),
        )
        if key in rows:
            _fail("Radial metric summary contains a duplicated stratum.")
        rows[key] = {
            name: float(values[index])
            for name, values in columns.items()
            if name not in {"epoch", "behavior", "chaser"}
        }
    return rows


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


def _stratum_labels(handle: Any, keys: Sequence[tuple[int, int, int]]) -> list[str]:
    epoch_registry = _registry(handle.scientific_manifest, "epoch_role")
    behavior_registry = _registry(handle.scientific_manifest, "behavior_role")
    return [
        (
            f"{epoch_registry.get(str(epoch), f'epoch {epoch}')}\n"
            f"{behavior_registry.get(str(behavior), f'role {behavior}')} · "
            f"chaser {chaser}"
        )
        for epoch, behavior, chaser in keys
    ]


def render_provider_radial_near_field_summary(
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    radial_detection: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render persisted paired-provider distance, ring, and near-field summaries."""

    handles = (radial_keypoint, radial_detection)
    provider_ids = [
        str(handle.scientific_manifest["position_provider"]["provider_id"])
        for handle in handles
    ]
    rows = tuple(_radial_metric_rows(handle) for handle in handles)
    if set(rows[0]) != set(rows[1]) or not rows[0]:
        _fail("Paired radial metric products expose different or empty strata.")
    keys = sorted(rows[0])
    labels = _stratum_labels(radial_keypoint, keys)
    x = np.arange(len(keys), dtype=np.float64)
    offsets = (-0.18, 0.18)
    colors = ("#1f77b4", "#d95f02")

    figure, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    for provider_index, (provider_id, provider_rows) in enumerate(
        zip(provider_ids, rows, strict=True)
    ):
        median = np.asarray(
            [provider_rows[key]["distance_p50_mm"] for key in keys], dtype=float
        )
        p25 = np.asarray(
            [provider_rows[key]["distance_p25_mm"] for key in keys], dtype=float
        )
        p75 = np.asarray(
            [provider_rows[key]["distance_p75_mm"] for key in keys], dtype=float
        )
        valid = np.isfinite(median) & np.isfinite(p25) & np.isfinite(p75)
        axes[0, 0].errorbar(
            x[valid] + offsets[provider_index],
            median[valid],
            yerr=np.vstack((median[valid] - p25[valid], p75[valid] - median[valid])),
            fmt="o",
            capsize=3,
            color=colors[provider_index],
            label=provider_id,
        )
        near_fraction = np.asarray(
            [provider_rows[key]["near_fraction"] for key in keys], dtype=float
        )
        axes[1, 0].bar(
            x + offsets[provider_index],
            near_fraction,
            width=0.34,
            color=colors[provider_index],
            alpha=0.78,
            label=provider_id,
        )
        dwell = np.asarray(
            [provider_rows[key]["near_dwell_s"] for key in keys], dtype=float
        )
        axes[1, 1].bar(
            x + offsets[provider_index],
            dwell,
            width=0.34,
            color=colors[provider_index],
            alpha=0.62,
            label=f"{provider_id} · dwell",
        )

    axes[0, 0].set_title("Simple distance: median and interquartile range")
    axes[0, 0].set_ylabel("fish–chaser distance (mm)")
    axes[0, 0].legend(fontsize=7)

    radial_keys = []
    radial_rows = []
    for handle in handles:
        epoch = _array(handle, "radial_epoch_role_code").astype(np.int64)
        behavior = _array(handle, "radial_behavior_role_code").astype(np.int64)
        chaser = _array(handle, "radial_chaser_identity_code").astype(np.int64)
        start = _array(handle, "radial_bin_start_mm").astype(np.float64)
        end = _array(handle, "radial_bin_end_mm").astype(np.float64)
        selection = _array(handle, "radial_selection_index_geometric").astype(
            np.float64
        )
        if not (
            epoch.size
            == behavior.size
            == chaser.size
            == start.size
            == end.size
            == selection.size
        ):
            _fail("Paired radial selection columns have different lengths.")
        keys_for_provider = sorted(
            set(zip(epoch.tolist(), behavior.tolist(), chaser.tolist()))
        )
        radial_keys.append(keys_for_provider)
        radial_rows.append((epoch, behavior, chaser, start, end, selection))
    if radial_keys[0] != radial_keys[1] or radial_keys[0] != keys or not radial_keys[0]:
        _fail("Paired radial selection products expose different or empty strata.")
    stratum_colors = plt.get_cmap("tab10")
    for provider_index, (provider_id, columns) in enumerate(
        zip(provider_ids, radial_rows, strict=True)
    ):
        epoch, behavior, chaser, start, end, selection = columns
        for stratum_index, key in enumerate(radial_keys[0]):
            mask = (epoch == key[0]) & (behavior == key[1]) & (chaser == key[2])
            order = np.argsort(start[mask])
            center = (start[mask][order] + end[mask][order]) / 2.0
            values = selection[mask][order]
            finite = np.isfinite(center) & np.isfinite(values)
            axes[0, 1].plot(
                center[finite],
                values[finite],
                color=stratum_colors(stratum_index % 10),
                linestyle=("-", "--")[provider_index],
                linewidth=1.15,
                label=(
                    f"{labels[stratum_index].replace(chr(10), ' · ')} · {provider_id}"
                ),
            )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0, 1].set_title("Area-corrected moving-chaser radial selection")
    axes[0, 1].set_xlabel("fish–chaser distance (mm)")
    axes[0, 1].set_ylabel("geometric selection index")
    axes[0, 1].legend(fontsize=5.8, ncols=2)

    near_radius = float(
        radial_keypoint.scientific_manifest["config"]["near_zone_radius_mm"]
    )
    axes[1, 0].set_title(f"Near-field occupancy (≤{near_radius:g} mm)")
    axes[1, 0].set_ylabel("fraction of valid distance rows")
    axes[1, 0].set_ylim(bottom=0.0)

    entry_axis = axes[1, 1].twinx()
    for provider_index, (provider_id, provider_rows) in enumerate(
        zip(provider_ids, rows, strict=True)
    ):
        rates = np.asarray(
            [provider_rows[key]["entry_rate_per_min"] for key in keys], dtype=float
        )
        entry_axis.plot(
            x + offsets[provider_index],
            rates,
            marker=("o", "s")[provider_index],
            color=colors[provider_index],
            linewidth=1.1,
            label=f"{provider_id} · entries/min",
        )
    axes[1, 1].set_title("Exact-session-time near-field visits")
    axes[1, 1].set_ylabel("dwell (s)")
    entry_axis.set_ylabel("entries/min valid time")
    left_handles, left_labels = axes[1, 1].get_legend_handles_labels()
    right_handles, right_labels = entry_axis.get_legend_handles_labels()
    axes[1, 1].legend(
        left_handles + right_handles,
        left_labels + right_labels,
        fontsize=6,
        ncols=2,
    )

    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=7)
    for ax in axes.reshape(-1):
        ax.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Paired-provider chaser distance, radial rings, and near field · "
        f"{radial_keypoint.recording_id}\n"
        "persisted summaries · exact session time · no interpolation · selector-ineligible",
        fontsize=13,
    )
    return _save_figure(figure, output_stem)


def render_provider_epoch_distance_traces(
    relative_keypoint: Any,
    relative_detection: Any,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render full-recording and exact-epoch distance traces for both providers."""

    relatives = (relative_keypoint, relative_detection)
    provider_ids = [
        str(relative.source_authorities["fish_position"]["provider_id"])
        for relative in relatives
    ]
    epochs = _epoch_records(radial_keypoint)
    frame_id = _collapsed_frame_scalar(relative_keypoint, "acquisition_frame_id").astype(
        np.int64
    )
    timestamp = _collapsed_frame_scalar(relative_keypoint, "timestamp_ns").astype(
        np.int64
    )
    timestamp_valid = _collapsed_frame_scalar(
        relative_keypoint, "timestamp_valid"
    ).astype(bool)
    selection_member = _collapsed_frame_scalar(
        relative_keypoint, "selection_member"
    ).astype(bool)
    if not np.any(timestamp_valid):
        _fail("Distance traces have no exact valid session timestamp.")
    reference_ns = int(timestamp[np.flatnonzero(timestamp_valid)[0]])
    time_s = (timestamp.astype(np.float64) - float(reference_ns)) / 1e9

    identities = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_identity_code"), dtype=np.int64
    )
    roles = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_behavior_role_code"), dtype=np.int64
    )
    if not (
        np.all(identities == identities[:1]) and np.all(roles == roles[:1])
    ):
        _fail("Distance traces have unstable chaser identity or behavior-role columns.")
    behavior_registry = _registry(radial_keypoint.scientific_manifest, "behavior_role")
    row_specs: list[tuple[str, np.ndarray]] = [
        ("full recording", np.ones(frame_id.size, dtype=bool))
    ]
    for record in epochs:
        row_specs.append(
            (
                str(record["analysis_role"]),
                selection_member
                & (frame_id >= int(record["start_frame_inclusive"]))
                & (frame_id < int(record["end_frame_exclusive"])),
            )
        )
    if any(not np.any(mask) for _, mask in row_specs):
        _fail("A full-recording or exact-epoch distance panel has no source rows.")

    figure, axes = plt.subplots(
        len(row_specs),
        relative_keypoint.n_chasers,
        figsize=(15.5, 3.15 * len(row_specs)),
        sharex=False,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    colors = ("#1f77b4", "#d95f02")
    for row_index, (row_label, row_mask) in enumerate(row_specs):
        row_indices = np.flatnonzero(row_mask)
        for chaser_column in range(relative_keypoint.n_chasers):
            ax = axes[row_index, chaser_column]
            for provider_index, (provider_id, relative) in enumerate(
                zip(provider_ids, relatives, strict=True)
            ):
                distance = np.asarray(
                    relative.base_frame_chaser("relative_distance_physical"),
                    dtype=np.float64,
                )[:, chaser_column]
                valid = np.asarray(
                    relative.base_frame_chaser("relative_physical_valid"), dtype=bool
                )[:, chaser_column]
                occurrence = np.asarray(
                    relative.base_frame_chaser("chaser_occurrence_member"), dtype=bool
                )[:, chaser_column]
                local_valid = (
                    valid[row_indices]
                    & occurrence[row_indices]
                    & timestamp_valid[row_indices]
                )
                local_distance = distance[row_indices]
                displayed_values = local_distance.copy()
                displayed_values[~local_valid] = np.nan
                ax.plot(
                    time_s[row_indices],
                    displayed_values,
                    color=colors[provider_index],
                    linewidth=0.9,
                    alpha=0.82,
                    rasterized=True,
                    label=provider_id,
                )
            identity = int(identities[0, chaser_column])
            role = behavior_registry.get(
                str(int(roles[0, chaser_column])),
                f"role {int(roles[0, chaser_column])}",
            )
            ax.set_title(f"{row_label} · {role} · chaser {identity}")
            ax.set_xlabel("session time from first valid timestamp (s)")
            ax.set_ylabel("distance (mm)")
            ax.grid(alpha=0.2)
            if row_index == 0 and chaser_column == 0:
                ax.legend(fontsize=7)
    figure.suptitle(
        f"Full-recording and exact-epoch fish–chaser distance · "
        f"{radial_keypoint.recording_id}\n"
        "paired first-class providers · all exact rows rasterized · "
        "missing rows break traces",
        fontsize=13,
    )
    return _save_figure(figure, output_stem)


def render_provider_epoch_trajectory_overlays(
    relative_keypoint: Any,
    relative_detection: Any,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    chaser_appearance: ChaserAppearanceProjection,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render exact-epoch fish position samples with logged chaser overlays."""

    relatives = (relative_keypoint, relative_detection)
    provider_ids = [
        str(relative.source_authorities["fish_position"]["provider_id"])
        for relative in relatives
    ]
    epochs = _epoch_records(radial_keypoint)
    arena = radial_keypoint.scientific_manifest.get("arena")
    if not isinstance(arena, Mapping):
        _fail("Trajectory overlay lacks its reviewed arena record.")
    center_x = float(arena["center_x_px"])
    center_y = float(arena["center_y_px"])
    radius = float(arena["radius_px"])
    if not all(np.isfinite(value) for value in (center_x, center_y, radius)) or radius <= 0:
        _fail("Trajectory overlay arena circle is non-finite or nonpositive.")
    frame_id = _collapsed_frame_scalar(relative_keypoint, "acquisition_frame_id").astype(
        np.int64
    )
    selection_member = _collapsed_frame_scalar(
        relative_keypoint, "selection_member"
    ).astype(bool)
    chaser_xy = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_position_xy_px"), dtype=np.float64
    )
    chaser_valid = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_position_valid"), dtype=bool
    )
    occurrence = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_occurrence_member"), dtype=bool
    )
    roles = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_behavior_role_code"), dtype=np.int64
    )
    identities = np.asarray(
        relative_keypoint.base_frame_chaser("chaser_identity_code"), dtype=np.int64
    )
    if chaser_xy.shape != (relative_keypoint.n_frames, relative_keypoint.n_chasers, 2):
        _fail("Trajectory chaser positions do not preserve frame/chaser/source-xy shape.")
    if not (np.all(roles == roles[:1]) and np.all(identities == identities[:1])):
        _fail("Trajectory overlay has unstable chaser identity or behavior roles.")
    appearance_projection = _validated_exact_chaser_appearance(
        relative_keypoint, chaser_appearance
    )
    appearance_by_identity = appearance_projection.by_identity_code()

    figure, axes = plt.subplots(
        len(relatives),
        len(epochs),
        figsize=(16.5, 10.5),
        constrained_layout=True,
        squeeze=False,
    )
    for provider_index, (provider_id, relative) in enumerate(
        zip(provider_ids, relatives, strict=True)
    ):
        fish_xy, fish_valid = _collapsed_fish_frame(relative)
        for epoch_index, record in enumerate(epochs):
            ax = axes[provider_index, epoch_index]
            epoch_mask = (
                selection_member
                & (frame_id >= int(record["start_frame_inclusive"]))
                & (frame_id < int(record["end_frame_exclusive"]))
            )
            epoch_rows = np.flatnonzero(epoch_mask)
            if not epoch_rows.size:
                _fail("An exact-epoch trajectory panel has no source rows.")
            local_fish = fish_xy[epoch_rows]
            local_fish_valid = fish_valid[epoch_rows]
            ax.scatter(
                local_fish[local_fish_valid, 0],
                local_fish[local_fish_valid, 1],
                color="#222222",
                s=0.35,
                alpha=0.18,
                edgecolors="none",
                rasterized=True,
                label=f"fish · {provider_id}",
            )
            for chaser_column in range(relative_keypoint.n_chasers):
                local_chaser = chaser_xy[epoch_rows, chaser_column]
                local_valid = (
                    chaser_valid[epoch_rows, chaser_column]
                    & occurrence[epoch_rows, chaser_column]
                )
                identity = int(identities[0, chaser_column])
                role_code = int(roles[0, chaser_column])
                appearance = appearance_by_identity.get(identity)
                if appearance is None or appearance.behavior_role_code != role_code:
                    _fail(
                        "Trajectory identity/role columns differ from the sealed "
                        "appearance projection."
                    )
                ax.scatter(
                    local_chaser[local_valid, 0],
                    local_chaser[local_valid, 1],
                    color=appearance.experimental_color_hex,
                    s=1.2,
                    alpha=0.55,
                    edgecolors="none",
                    rasterized=True,
                    label="_nolegend_",
                )
                role_marker_rows = uniformly_sample_indices(
                    np.flatnonzero(local_valid),
                    maximum=STATIC_TRAJECTORY_ROLE_MARKER_MAX_PER_PANEL_CHASER,
                )
                ax.scatter(
                    local_chaser[role_marker_rows, 0],
                    local_chaser[role_marker_rows, 1],
                    color=appearance.experimental_color_hex,
                    marker=appearance.matplotlib_role_marker,
                    s=26.0,
                    alpha=0.85,
                    edgecolors=appearance.contrast_outline_hex,
                    linewidths=0.45,
                    rasterized=True,
                    label=(
                        f"{appearance.behavior_role} · protocol chaser "
                        f"{appearance.chaser_index}"
                    ),
                )
            ax.add_patch(
                Circle(
                    (center_x, center_y),
                    radius,
                    fill=False,
                    color="#666666",
                    linewidth=0.9,
                )
            )
            margin = radius * 1.03
            ax.set_xlim(center_x - margin, center_x + margin)
            ax.set_ylim(center_y + margin, center_y - margin)
            ax.set_aspect("equal")
            ax.set_title(f"{record['analysis_role']} · {provider_id}", fontsize=9)
            ax.set_xlabel("source-camera x (px)")
            ax.set_ylabel("source-camera y (px; +down)")
            ax.grid(alpha=0.15)
            if provider_index == 0 and epoch_index == 0:
                ax.legend(fontsize=6)
    figure.suptitle(
        f"Exact-epoch fish position samples with logged chaser overlays · "
        f"{radial_keypoint.recording_id}\n"
        "reviewed circular-arena context · protocol color + independent role "
        "glyphs · all valid exact rows retained",
        fontsize=13,
    )
    return _save_figure(figure, output_stem)


def _configure_body_bearing_distance_polar_axis(
    ax: Any, *, radial_max_mm: float
) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        [0.0, 90.0, 180.0, 270.0],
        labels=["front", "left", "behind", "right"],
    )
    ax.set_ylim(0.0, float(radial_max_mm))
    ax.grid(alpha=0.22)


def render_keypoint_body_bearing_distance_point_cloud(
    relative_keypoint: Any,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render display-bounded exact anatomical-bearing/distance point rows."""

    data = _body_bearing_distance_plot_data(relative_keypoint, radial_keypoint)
    figure, axes = plt.subplots(
        len(data.panels),
        relative_keypoint.n_chasers,
        figsize=(5.2 * relative_keypoint.n_chasers, 4.4 * len(data.panels)),
        constrained_layout=True,
        squeeze=False,
        subplot_kw={"projection": "polar"},
    )
    for row_index, (panel_label, _member) in enumerate(data.panels):
        valid = data.panel_valid[row_index]
        for column in range(relative_keypoint.n_chasers):
            ax = axes[row_index, column]
            valid_indices = np.flatnonzero(valid[:, column])
            display_indices = uniformly_sample_indices(
                valid_indices,
                maximum=STATIC_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER,
            )
            if display_indices.size:
                ax.scatter(
                    np.deg2rad(data.bearing_deg[display_indices, column]),
                    data.distance_mm[display_indices, column],
                    s=2.0,
                    alpha=0.24,
                    edgecolors="none",
                    color=("#3b5b92", "#d95f02")[column % 2],
                    rasterized=True,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "no jointly valid rows",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            role_code = int(data.role[0, column])
            role = data.role_registry.get(str(role_code), f"role {role_code}")
            ax.set_title(
                f"{panel_label} · {role} · chaser {int(data.identity[0, column])}\n"
                f"valid n={valid_indices.size:,}; displayed n={display_indices.size:,}",
                fontsize=8.5,
            )
            _configure_body_bearing_distance_polar_axis(
                ax, radial_max_mm=float(data.distance_edges_mm[-1])
            )
    figure.suptitle(
        f"Anatomical body-frame chaser bearing × distance point rows · "
        f"{relative_keypoint.recording_id}\n"
        "accepted keypoint body axis · exact acquisition rows · no interpolation",
        fontsize=13,
    )
    return _save_figure(figure, output_stem)


def render_keypoint_body_bearing_distance_density(
    relative_keypoint: Any,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    *,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Render all-row joint anatomical-bearing/distance probability density."""

    data = _body_bearing_distance_plot_data(relative_keypoint, radial_keypoint)
    color_max = positive_probability_color_max(data.histograms)
    figure, axes = plt.subplots(
        len(data.panels),
        relative_keypoint.n_chasers,
        figsize=(5.2 * relative_keypoint.n_chasers, 4.4 * len(data.panels)),
        constrained_layout=True,
        squeeze=False,
        subplot_kw={"projection": "polar"},
    )
    theta_edges, radial_edges = np.meshgrid(
        np.deg2rad(data.bearing_edges_deg),
        data.distance_edges_mm,
    )
    mesh = None
    histogram_index = 0
    for row_index, (panel_label, _member) in enumerate(data.panels):
        for column in range(relative_keypoint.n_chasers):
            ax = axes[row_index, column]
            histogram = data.histograms[histogram_index]
            histogram_index += 1
            ax.grid(False)
            mesh = ax.pcolormesh(
                theta_edges,
                radial_edges,
                np.ma.masked_where(histogram.counts == 0, histogram.probability),
                shading="flat",
                cmap="viridis",
                vmin=0.0,
                vmax=float(color_max),
                rasterized=True,
            )
            if histogram.denominator == 0:
                ax.text(
                    0.5,
                    0.5,
                    "no jointly valid rows",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            role_code = int(data.role[0, column])
            role = data.role_registry.get(str(role_code), f"role {role_code}")
            ax.set_title(
                f"{panel_label} · {role} · chaser {int(data.identity[0, column])}\n"
                f"joint denominator n={histogram.denominator:,}",
                fontsize=8.5,
            )
            _configure_body_bearing_distance_polar_axis(
                ax, radial_max_mm=float(data.distance_edges_mm[-1])
            )
    if mesh is not None:
        figure.colorbar(
            mesh,
            ax=axes.reshape(-1).tolist(),
            label="fraction of jointly valid rows per bin",
            shrink=0.72,
        )
    figure.suptitle(
        f"Anatomical body-frame chaser bearing × distance density · "
        f"{relative_keypoint.recording_id}\n"
        f"{DISTANCE_BIN_WIDTH_MM:g} mm × {BEARING_BIN_WIDTH_DEG:g}° bins · "
        "probability within panel/chaser · every valid exact row",
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
    relative_keypoint: Any,
    radial_keypoint: ComposableChaserSuccessorSourceHandle,
    radial_detection: ComposableChaserSuccessorSourceHandle,
    *,
    chaser_appearance: ChaserAppearanceProjection | None = None,
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
    epoch_records = _epoch_records(radial_keypoint)
    radial_metric_rows = tuple(_radial_metric_rows(handle) for handle in radial_handles)
    if set(radial_metric_rows[0]) != set(radial_metric_rows[1]) or not radial_metric_rows[0]:
        _fail("Provider radial summary plot strata are empty or mismatched.")
    arena = radial_keypoint.scientific_manifest.get("arena")
    if not isinstance(arena, Mapping):
        _fail("Provider trajectory plot arena evidence is absent.")
    bearing_distance = _body_bearing_distance_plot_data(
        relative_keypoint, radial_keypoint
    )
    appearance_projection = _validated_exact_chaser_appearance(
        relative_keypoint,
        chaser_appearance or _load_exact_chaser_appearance(relative_keypoint),
    )
    return {
        "scientific_coordinates": {
            "provider_distance_cdf": cdf_parameters,
            "provider_radial_near_field_summary": {
                "metric_arrays": {
                    "distance_p25_mm": "metric_distance_p25_mm",
                    "distance_p50_mm": "metric_distance_p50_mm",
                    "distance_p75_mm": "metric_distance_p75_mm",
                    "near_fraction": "metric_near_zone_fraction_valid",
                    "near_dwell_s": "metric_near_zone_dwell_s",
                    "entry_rate_per_min": (
                        "metric_near_zone_entry_rate_per_min_valid_time"
                    ),
                    "radial_selection": "radial_selection_index_geometric",
                },
                "strata": [list(key) for key in sorted(radial_metric_rows[0])],
            },
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
            "full_and_epoch_trace_time_reference": (
                "timestamp_ns_minus_first_valid_recording_timestamp_ns"
            ),
            "full_and_epoch_trace_distance_array": "relative_distance_physical",
            "exact_epoch_records": list(epoch_records),
            "trajectory_position_arrays": {
                "fish": "fish_position_xy_px",
                "chaser": "chaser_position_xy_px",
                "coordinate_space": "source_camera_continuous_pixel_xy_top_left_y_down",
                "arena": _plain(arena),
            },
            "keypoint_body_bearing_distance": {
                "recipe_id": BODY_BEARING_DISTANCE_DISPLAY_RECIPE_ID,
                "position_provider_id": str(
                    relative_keypoint.source_authorities["fish_position"]["provider_id"]
                ),
                "body_frame_authority": _plain(
                    relative_keypoint.source_authorities["body_frame"]
                ),
                "source_arrays": [
                    "base/relative_distance_physical",
                    "base/relative_physical_valid",
                    "body/body_bearing_deg",
                    "body/body_bearing_valid",
                    "base/chaser_occurrence_member",
                ],
                "joint_validity": (
                    "panel_member_and_chaser_occurrence_and_"
                    "relative_physical_valid_and_body_bearing_valid"
                ),
                "distance_bin_edges_mm": (bearing_distance.distance_edges_mm.tolist()),
                "bearing_bin_edges_deg": (bearing_distance.bearing_edges_deg.tolist()),
                "density_normalization": "probability_within_panel_chaser",
                "panel_denominators": _body_bearing_distance_panel_records(
                    bearing_distance
                ),
                "interpolation": "prohibited",
                "body_axis_fallback": "prohibited",
                "detection_position_substitution": "prohibited",
            },
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
            "provider_radial_near_field_summary": {
                "subplot_grid": [2, 2],
                "figure_size_inches": [17.0, 11.0],
                "paired_provider_offsets": [-0.18, 0.18],
                "provider_colors": ["#1f77b4", "#d95f02"],
                "density_or_summary_recomputation": False,
            },
            "provider_epoch_distance_traces": {
                "subplot_grid": [1 + len(epoch_records), 2],
                "figure_size_inches": [15.5, 3.15 * (1 + len(epoch_records))],
                "display_algorithm": DENSE_DISPLAY_ALGORITHM,
                "all_exact_source_rows_retained": True,
                "pdf_dense_lines_rasterized": True,
            },
            "provider_epoch_trajectory_overlays": {
                "subplot_grid": [2, len(epoch_records)],
                "figure_size_inches": [16.5, 10.5],
                "display_algorithm": DENSE_DISPLAY_ALGORITHM,
                "all_valid_source_rows_retained": True,
                "fish_marker_size_points_squared": 0.35,
                "chaser_marker_size_points_squared": 1.2,
                "chaser_color_source": "sealed_protocol_rgba",
                "chaser_role_encoding": (
                    "independent_bounded_exact_row_marker_shape_and_legend_text"
                ),
                "chaser_role_marker_max_per_panel_chaser": (
                    STATIC_TRAJECTORY_ROLE_MARKER_MAX_PER_PANEL_CHASER
                ),
                "chaser_role_marker_sampling": (
                    "source_order_uniform_including_endpoints"
                ),
                "chaser_role_marker_size_points_squared": 26.0,
                "appearance_projection": appearance_projection.provenance_record(),
                "index_or_role_color_fallback": "prohibited",
                "pdf_dense_points_rasterized": True,
                "position_samples_connected_by_lines": False,
                "quantitative_occupancy_inference": False,
            },
            "keypoint_body_bearing_distance_point_cloud": {
                "subplot_grid": [
                    len(bearing_distance.panels),
                    relative_keypoint.n_chasers,
                ],
                "figure_size_inches": [
                    5.2 * relative_keypoint.n_chasers,
                    4.4 * len(bearing_distance.panels),
                ],
                "display_sampling": ("source_order_uniform_including_endpoints"),
                "max_rows_per_panel_chaser": (
                    STATIC_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER
                ),
                "pdf_dense_points_rasterized": True,
                "scientific_inference": False,
            },
            "keypoint_body_bearing_distance_density": {
                "subplot_grid": [
                    len(bearing_distance.panels),
                    relative_keypoint.n_chasers,
                ],
                "figure_size_inches": [
                    5.2 * relative_keypoint.n_chasers,
                    4.4 * len(bearing_distance.panels),
                ],
                "distance_bin_width_mm": DISTANCE_BIN_WIDTH_MM,
                "bearing_bin_width_deg": BEARING_BIN_WIDTH_DEG,
                "normalization": "probability_within_panel_chaser",
                "color_cmax_quantile": DENSITY_COLOR_CMAX_QUANTILE,
                "all_jointly_valid_exact_rows_retained": True,
                "pdf_density_mesh_rasterized": True,
                "scientific_recomputation": False,
            },
        },
        "output_families": [
            "provider_distance_cdf",
            "bout_response_details",
            "trial_escape_details",
            "trial_distance_traces",
            "provider_radial_near_field_summary",
            "provider_epoch_distance_traces",
            "provider_epoch_trajectory_overlays",
            "keypoint_body_bearing_distance_point_cloud",
            "keypoint_body_bearing_distance_density",
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
    chaser_appearance: ChaserAppearanceProjection | None = None,
) -> tuple[Path, ...]:
    """Validate all sources and render the nine detailed figure families."""

    verify_detailed_plot_inputs(
        controller,
        bout,
        escape,
        relative_keypoint,
        relative_detection,
        radial_keypoint,
        radial_detection,
    )
    appearance_projection = _validated_exact_chaser_appearance(
        relative_keypoint,
        chaser_appearance or _load_exact_chaser_appearance(relative_keypoint),
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
        (
            render_provider_radial_near_field_summary,
            "provider_radial_near_field_summary",
            (radial_keypoint, radial_detection),
        ),
        (
            render_provider_epoch_distance_traces,
            "provider_epoch_distance_traces",
            (relative_keypoint, relative_detection, radial_keypoint),
        ),
        (
            render_provider_epoch_trajectory_overlays,
            "provider_epoch_trajectory_overlays",
            (
                relative_keypoint,
                relative_detection,
                radial_keypoint,
                appearance_projection,
            ),
        ),
        (
            render_keypoint_body_bearing_distance_point_cloud,
            "keypoint_body_bearing_distance_point_cloud",
            (relative_keypoint, radial_keypoint),
        ),
        (
            render_keypoint_body_bearing_distance_density,
            "keypoint_body_bearing_distance_density",
            (relative_keypoint, radial_keypoint),
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
    parser.add_argument("--controller-validation-receipt")
    parser.add_argument("--bout-validation-receipt")
    parser.add_argument("--escape-validation-receipt")
    parser.add_argument("--keypoint-radial-run", required=True)
    parser.add_argument("--detection-radial-run", required=True)
    parser.add_argument("--keypoint-radial-validation-receipt")
    parser.add_argument("--detection-radial-validation-receipt")
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
        "provider_radial_near_field_summary",
        "provider_epoch_distance_traces",
        "provider_epoch_trajectory_overlays",
        "keypoint_body_bearing_distance_point_cloud",
        "keypoint_body_bearing_distance_density",
    )
    expected = tuple(
        output_dir / f"{bundle_name}_{suffix}.{extension}"
        for suffix in suffixes
        for extension in ("png", "pdf")
    ) + (receipt_path,)
    if not args.overwrite and any(path.exists() for path in expected):
        raise FileExistsError("Detailed plot output already exists; pass --overwrite explicitly.")

    exact_child_receipts = (
        args.controller_validation_receipt,
        args.bout_validation_receipt,
        args.escape_validation_receipt,
        args.keypoint_radial_validation_receipt,
        args.detection_radial_validation_receipt,
    )
    if any(value is not None for value in exact_child_receipts) and not all(
        value is not None for value in exact_child_receipts
    ):
        _fail("All five exact-child source receipts must be supplied together.")
    controller, bout, escape = tuple(
        load_composable_chaser_successor_source_handle(
            archive,
            successor_kind=kind,
            run_name=run_name,
            expected_recording_id=args.expected_recording_id,
            use_consolidated=True,
            deep_audit=True,
            direct_validation_receipt=source_receipt,
        )
        for kind, source_receipt in zip(
            _CHAIN_KINDS, exact_child_receipts[:3], strict=True
        )
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
            required_body_arrays=DETAILED_PLOT_BODY_ARRAY_NAMES,
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
        direct_validation_receipt=args.keypoint_radial_validation_receipt,
    )
    radial_detection = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_radial_near_field",
        run_name=detection_run,
        expected_recording_id=args.expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
        direct_validation_receipt=args.detection_radial_validation_receipt,
    )
    chaser_appearance = _load_exact_chaser_appearance(relative_keypoint)
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
        chaser_appearance=chaser_appearance,
    )
    plot_parameters = detailed_plot_parameters(
        controller,
        bout,
        escape,
        relative_keypoint,
        radial_keypoint,
        radial_detection,
        chaser_appearance=chaser_appearance,
    )
    source_bindings = {
        "controller_chase_trials": {
            "run_path": controller.run_path,
            "manifest_sha256": controller.manifest_sha256,
            "scientific_payload_sha256": controller.scientific_payload_sha256,
            "verification_mode": controller.metadata_equivalence.get(
                "verification_mode", "direct_consolidated_equivalence"
            ),
            "validation_receipt_sha256": controller.metadata_equivalence.get(
                "receipt_sha256"
            ),
        },
        "generalized_chaser_bout_response": {
            "run_path": bout.run_path,
            "manifest_sha256": bout.manifest_sha256,
            "scientific_payload_sha256": bout.scientific_payload_sha256,
            "verification_mode": bout.metadata_equivalence.get(
                "verification_mode", "direct_consolidated_equivalence"
            ),
            "validation_receipt_sha256": bout.metadata_equivalence.get(
                "receipt_sha256"
            ),
        },
        "chaser_escape_freeze": {
            "run_path": escape.run_path,
            "manifest_sha256": escape.manifest_sha256,
            "scientific_payload_sha256": escape.scientific_payload_sha256,
            "verification_mode": escape.metadata_equivalence.get(
                "verification_mode", "direct_consolidated_equivalence"
            ),
            "validation_receipt_sha256": escape.metadata_equivalence.get(
                "receipt_sha256"
            ),
        },
        "relative_frame_keypoint": {
            "run_path": relative_keypoint.run_path,
            "manifest_sha256": relative_keypoint.manifest_sha256,
            "payload_sha256": relative_keypoint.payload_digest,
            "verification_mode": relative_keypoint.verification_mode,
            "validation_receipt_sha256": getattr(
                relative_keypoint, "receipt_digest", None
            ),
            "body_array_paths": [
                f"body/{name}" for name in DETAILED_PLOT_BODY_ARRAY_NAMES
            ],
            "body_frame_authority": _plain(
                relative_keypoint.source_authorities["body_frame"]
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
            "verification_mode": radial_keypoint.metadata_equivalence.get(
                "verification_mode", "direct_consolidated_equivalence"
            ),
            "validation_receipt_sha256": radial_keypoint.metadata_equivalence.get(
                "receipt_sha256"
            ),
        },
        "radial_detection": {
            "run_path": radial_detection.run_path,
            "manifest_sha256": radial_detection.manifest_sha256,
            "scientific_payload_sha256": radial_detection.scientific_payload_sha256,
            "verification_mode": radial_detection.metadata_equivalence.get(
                "verification_mode", "direct_consolidated_equivalence"
            ),
            "validation_receipt_sha256": radial_detection.metadata_equivalence.get(
                "receipt_sha256"
            ),
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
            "plot_transforms": (
                "sorting_masking_exact_time_origin_subtraction_and_"
                "display_only_body_bearing_distance_histogramming_and_"
                "dense_artist_rasterization_only"
            ),
            "interpolation": "prohibited",
            "missing_rows": "remain_missing",
            "scientific_authority": False,
        },
        "unavailable_recipe_evidence": {
            "individual_near_visits": "not_persisted_in_radial_successor_v1",
            "escape_aligned_distance_traces": "not_persisted_in_escape_successor_v2",
            "ring_entry_video": "deferred_sealed_video_successor",
            "gaze": (
                "reviewed_eye_orientation_successor_not_supplied_to_"
                "detailed_plot_bundle_v4"
            ),
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
    "render_keypoint_body_bearing_distance_density",
    "render_keypoint_body_bearing_distance_point_cloud",
    "render_provider_distance_cdf",
    "render_provider_epoch_distance_traces",
    "render_provider_epoch_trajectory_overlays",
    "render_provider_radial_near_field_summary",
    "render_trial_distance_traces",
    "render_trial_escape_details",
    "verify_detailed_plot_inputs",
]
