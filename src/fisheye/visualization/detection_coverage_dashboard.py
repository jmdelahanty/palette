"""Shared detection coverage dashboard for raw and refined detection runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import zarr

from fisheye.shared.frame_domains import FrameDomain, FrameDomainError, FrameDomains
from fisheye.shared.plot_artifacts import PlotArtifactResult, write_png_visualization_artifact
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import resolve_zarr_run


DEFAULT_ROLLING_WINDOW_FRAMES = 100
DEFAULT_FRAMES_PER_ROW = 500
DETECTION_COVERAGE_DASHBOARD_ARTIFACT = "detection_coverage_dashboard_png"
DETECTION_COVERAGE_DASHBOARD_SCHEMA_ID = "palette.visualization.detection_coverage_dashboard.v1"


@dataclass(frozen=True)
class DetectionCoverageSeries:
    """Per-frame detection presence data for one detection surface."""

    name: str
    frame_counts: np.ndarray
    quality_flags: Optional[np.ndarray] = None
    frame_offset: int = 0
    attrs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        counts = np.asarray(self.frame_counts)
        if counts.ndim != 1:
            raise ValueError("frame_counts must be one-dimensional")
        if self.quality_flags is not None:
            flags = np.asarray(self.quality_flags)
            if flags.ndim != 1:
                raise ValueError("quality_flags must be one-dimensional")
            if flags.shape[0] != counts.shape[0]:
                raise ValueError("quality_flags length must match frame_counts")
        if int(self.frame_offset) < 0:
            raise ValueError("frame_offset must be non-negative")

    @property
    def total_frames(self) -> int:
        return int(np.asarray(self.frame_counts).shape[0])

    @property
    def presence_mask(self) -> np.ndarray:
        return np.asarray(self.frame_counts) > 0


@dataclass(frozen=True)
class DetectionCoverageSummary:
    """Compact coverage/gap summary for dashboard text and downstream checks."""

    name: str
    total_frames: int
    frames_with_detections: int
    missing_frames: int
    coverage_percent: float
    missing_segment_count: int
    mean_missing_segment_frames: float
    median_missing_segment_frames: float
    max_missing_segment_frames: int
    longest_present_run_frames: int
    coverage_status: str
    gap_burden: str
    temporal_continuity: str
    review_priority: str

    def as_text_lines(self) -> list[str]:
        return [
            f"{self.name}",
            f"Coverage: {self.coverage_percent:.2f}%",
            f"Frames: {self.frames_with_detections:,}/{self.total_frames:,}",
            f"Missing frames: {self.missing_frames:,}",
            f"Missing segments: {self.missing_segment_count:,}",
            f"Max gap: {self.max_missing_segment_frames:,} frames",
            f"Median gap: {self.median_missing_segment_frames:.1f} frames",
            f"Coverage status: {self.coverage_status}",
            f"Gap burden: {self.gap_burden}",
            f"Temporal continuity: {self.temporal_continuity}",
            f"Review priority: {self.review_priority}",
        ]


def _open_zarr_root(zarr_path: str | Path, *, mode: str = "r") -> zarr.Group:
    path = Path(zarr_path).expanduser()
    try:
        root = zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        root = zarr.open_group(str(path), mode=mode, consolidated=False)
    try:
        setattr(root, "_palette_fs_path", path)
        setattr(root, "_palette_open_mode", mode)
    except Exception:
        pass
    return root


def _safe_int(value: object) -> Optional[int]:
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return integer if integer >= 0 else None


def _stored_zarr_frame_count_from_domains(root: zarr.Group) -> Optional[int]:
    try:
        return int(FrameDomains(root=root).count(FrameDomain.STORED_ZARR))
    except FrameDomainError:
        return None


def _resolve_total_frames(
    root: zarr.Group,
    groups: Sequence[zarr.Group],
    *,
    minimum: int = 0,
) -> int:
    attr_names = (
        "coverage_frames_total",
        "total_frames",
        "num_frames",
        "n_frames",
        "frame_count",
        "total_frame_count",
        "video_frame_count",
        "camera_frame_count",
    )
    candidates: list[int] = [int(minimum)]
    for group in groups:
        for name in attr_names:
            value = _safe_int(group.attrs.get(name))
            if value is not None:
                candidates.append(value)
    for name in attr_names + ("frames",):
        value = _safe_int(root.attrs.get(name))
        if value is not None:
            candidates.append(value)
    raw_video = root.get("raw_video")
    if raw_video is not None:
        for name in ("images_full", "images_ds"):
            if name in raw_video:
                try:
                    candidates.append(int(raw_video[name].shape[0]))
                except Exception:
                    pass
        stored_count = _stored_zarr_frame_count_from_domains(root)
        if stored_count is not None:
            candidates.append(stored_count)
    return max(candidates)


def _counts_from_group(
    root: zarr.Group,
    group: zarr.Group,
    *,
    fallback_groups: Sequence[zarr.Group] = (),
    quality_flags: Optional[np.ndarray] = None,
) -> np.ndarray:
    if "frame_counts" in group:
        counts = np.asarray(group["frame_counts"][:], dtype=np.int64).reshape(-1)
        minimum = int(counts.shape[0])
    elif "n_detections" in group:
        counts = np.asarray(group["n_detections"][:], dtype=np.int64).reshape(-1)
        minimum = int(counts.shape[0])
    elif "frame_indices" in group:
        frame_indices = np.asarray(group["frame_indices"][:], dtype=np.int64).reshape(-1)
        minimum = int(frame_indices.max() + 1) if frame_indices.size else 0
        counts = np.bincount(frame_indices, minlength=minimum).astype(np.int64, copy=False)
    else:
        raise ValueError(f"{group.path or '<group>'} does not contain frame_counts or frame_indices")

    if quality_flags is not None:
        minimum = max(minimum, int(quality_flags.shape[0]))
    total = _resolve_total_frames(root, (group, *fallback_groups), minimum=minimum)
    if total > counts.shape[0]:
        counts = np.pad(counts, (0, total - counts.shape[0]), mode="constant")
    elif total < counts.shape[0]:
        counts = counts[:total]
    return counts.astype(np.int32, copy=False)


def _normalize_flags(flags: Optional[np.ndarray], total_frames: int) -> Optional[np.ndarray]:
    if flags is None:
        return None
    out = np.asarray(flags, dtype=np.int32).reshape(-1)
    if out.shape[0] < total_frames:
        out = np.pad(out, (0, total_frames - out.shape[0]), mode="constant", constant_values=-1)
    elif out.shape[0] > total_frames:
        out = out[:total_frames]
    return out


def _resolve_quality_group(
    detect_group: zarr.Group,
    quality_run: Optional[str],
) -> tuple[Optional[zarr.Group], Optional[str]]:
    reports = detect_group.get("quality_reports")
    if reports is None:
        return None, None
    requested = normalize_attr(quality_run)
    if requested:
        if requested not in reports:
            raise ValueError(f"Quality run '{requested}' not found under {detect_group.path}/quality_reports")
        return reports[requested], requested
    latest = normalize_attr(reports.attrs.get("latest"))
    if latest and latest in reports:
        return reports[latest], latest
    names = sorted(str(name) for name in reports.group_keys())
    if not names:
        return None, None
    return reports[names[-1]], names[-1]


def load_raw_detect_coverage_series(
    zarr_path: str | Path,
    *,
    detect_run: Optional[str] = None,
    quality_run: Optional[str] = None,
) -> DetectionCoverageSeries:
    """Load per-frame coverage from ``detect_runs/<run>``."""
    root = _open_zarr_root(zarr_path)
    detect_group, resolved_detect_run = resolve_zarr_run(
        root,
        "detect_runs",
        detect_run,
        fallback_to_latest=True,
        fallback_to_sorted="last",
        latest_aliases=("latest",),
        run_label="Detection run",
    )
    quality_group, resolved_quality_run = _resolve_quality_group(detect_group, quality_run)
    quality_flags = None
    if quality_group is not None and "quality_flags" in quality_group:
        quality_flags = np.asarray(quality_group["quality_flags"][:], dtype=np.int32).reshape(-1)
    counts = _counts_from_group(root, detect_group, quality_flags=quality_flags)
    return DetectionCoverageSeries(
        name=f"raw/{resolved_detect_run}",
        frame_counts=counts,
        quality_flags=_normalize_flags(quality_flags, int(counts.shape[0])),
        attrs={
            "source_kind": "raw_detection",
            "detect_run": resolved_detect_run,
            "quality_run": resolved_quality_run,
            "source_path": f"detect_runs/{resolved_detect_run}",
        },
    )


def _resolve_refined_parent(root: zarr.Group) -> tuple[str, zarr.Group]:
    if "refined_detect_runs" in root:
        return "refined_detect_runs", root["refined_detect_runs"]
    raise ValueError("refined_detect_runs not found in store")


def _refined_coverage_payload_group(refined_group: zarr.Group) -> zarr.Group:
    if "instances" in refined_group:
        instances = refined_group["instances"]
        if "frame_counts" in instances or "frame_indices" in instances:
            return instances
    if "frame_counts" in refined_group or "frame_indices" in refined_group:
        return refined_group
    raise ValueError(f"{refined_group.path or '<refined run>'} has no coverage arrays")


def load_refined_detect_coverage_series(
    zarr_path: str | Path,
    *,
    refined_run: Optional[str] = None,
) -> DetectionCoverageSeries:
    """Load per-frame coverage from ``refined_detect_runs/<run>``."""
    root = _open_zarr_root(zarr_path)
    parent_path, _ = _resolve_refined_parent(root)
    refined_group, resolved_refined_run = resolve_zarr_run(
        root,
        parent_path,
        refined_run,
        fallback_to_latest=True,
        fallback_to_sorted="last",
        latest_aliases=("latest",),
        run_label="Refined detection run",
    )
    payload = _refined_coverage_payload_group(refined_group)
    counts = _counts_from_group(root, payload, fallback_groups=(refined_group,))
    source_detect_run = normalize_attr(refined_group.attrs.get("source_detect_run"))
    return DetectionCoverageSeries(
        name=f"refined/{resolved_refined_run}",
        frame_counts=counts,
        attrs={
            "source_kind": "refined_detection",
            "refined_run": resolved_refined_run,
            "source_detect_run": source_detect_run,
            "source_path": f"{parent_path}/{resolved_refined_run}",
            "payload_path": str(payload.path or ""),
        },
    )


def compute_missing_segments(presence_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return missing-frame segments as ``(start, end)`` pairs, end exclusive."""
    present = np.asarray(presence_mask, dtype=bool).reshape(-1)
    missing = ~present
    if missing.size == 0 or not bool(np.any(missing)):
        return []
    padded = np.pad(missing.astype(np.int8), (1, 1), mode="constant", constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _longest_true_run(mask: np.ndarray) -> int:
    present = np.asarray(mask, dtype=bool).reshape(-1)
    if present.size == 0 or not bool(np.any(present)):
        return 0
    padded = np.pad(present.astype(np.int8), (1, 1), mode="constant", constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return int(np.max(ends - starts)) if starts.size else 0


def _status_from_coverage(coverage_percent: float) -> str:
    if coverage_percent >= 99.0:
        return "high"
    if coverage_percent >= 95.0:
        return "moderate"
    return "low"


def _status_from_gaps(max_gap: int, median_gap: float, missing_segment_count: int) -> str:
    if max_gap <= 3 and median_gap <= 2.0:
        return "low"
    if max_gap <= 30 and missing_segment_count <= 100:
        return "moderate"
    return "high"


def _status_from_continuity(longest_present_run: int, total_frames: int) -> str:
    if total_frames <= 0:
        return "low"
    fraction = longest_present_run / float(total_frames)
    if fraction >= 0.8:
        return "high"
    if fraction >= 0.5:
        return "moderate"
    return "low"


def _review_priority(coverage_status: str, gap_burden: str, temporal_continuity: str) -> str:
    if coverage_status == "high" and gap_burden == "low" and temporal_continuity in {"high", "moderate"}:
        return "low"
    if coverage_status == "low" or gap_burden == "high" or temporal_continuity == "low":
        return "high"
    return "medium"


def summarize_detection_coverage(series: DetectionCoverageSeries) -> DetectionCoverageSummary:
    """Compute coverage and missing-segment statistics for one series."""
    mask = series.presence_mask
    total = int(mask.shape[0])
    present = int(np.count_nonzero(mask))
    missing_segments = compute_missing_segments(mask)
    gap_lengths = np.asarray([end - start for start, end in missing_segments], dtype=np.int64)
    max_gap = int(gap_lengths.max()) if gap_lengths.size else 0
    mean_gap = float(gap_lengths.mean()) if gap_lengths.size else 0.0
    median_gap = float(np.median(gap_lengths)) if gap_lengths.size else 0.0
    coverage = (present / total * 100.0) if total else 0.0
    longest_run = _longest_true_run(mask)
    coverage_status = _status_from_coverage(coverage)
    gap_burden = _status_from_gaps(max_gap, median_gap, len(missing_segments))
    continuity = _status_from_continuity(longest_run, total)
    return DetectionCoverageSummary(
        name=series.name,
        total_frames=total,
        frames_with_detections=present,
        missing_frames=total - present,
        coverage_percent=float(coverage),
        missing_segment_count=len(missing_segments),
        mean_missing_segment_frames=mean_gap,
        median_missing_segment_frames=median_gap,
        max_missing_segment_frames=max_gap,
        longest_present_run_frames=longest_run,
        coverage_status=coverage_status,
        gap_burden=gap_burden,
        temporal_continuity=continuity,
        review_priority=_review_priority(coverage_status, gap_burden, continuity),
    )


def _rolling_coverage_percent(frame_counts: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(frame_counts) > 0
    if mask.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    clipped_window = max(1, min(int(window), int(mask.size)))
    kernel = np.ones(clipped_window, dtype=np.float64)
    values = np.convolve(mask.astype(np.float64), kernel, mode="valid") / clipped_window * 100.0
    x = np.arange(values.size, dtype=np.float64) + (clipped_window - 1) / 2.0
    return x, values


def _presence_grid(frame_counts: np.ndarray, frames_per_row: int) -> np.ndarray:
    mask = (np.asarray(frame_counts) > 0).astype(np.float32)
    width = max(1, int(frames_per_row))
    rows = int(np.ceil(mask.size / width)) if mask.size else 1
    grid = np.full((rows, width), np.nan, dtype=np.float32)
    if mask.size:
        grid.reshape(-1)[: mask.size] = mask
    return grid


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_detection_coverage_dashboard(
    series: Sequence[DetectionCoverageSeries],
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW_FRAMES,
    frames_per_row: int = DEFAULT_FRAMES_PER_ROW,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a dashboard figure for one or more detection coverage series."""
    if not series:
        raise ValueError("at least one coverage series is required")

    summaries = [summarize_detection_coverage(item) for item in series]
    fig_height = max(9.5, 5.0 + 2.2 * len(series))
    fig = plt.figure(figsize=(18, fig_height), constrained_layout=False)
    fig.suptitle(title or "Detection Coverage Dashboard", fontsize=18, fontweight="bold")
    gs = fig.add_gridspec(
        3 + len(series),
        2,
        height_ratios=[*(1.0 for _ in series), 1.5, 1.2, 1.4],
        hspace=0.55,
        wspace=0.3,
    )

    presence_cmap = ListedColormap(["#c73e3a", "#0f7a3a"])
    for idx, item in enumerate(series):
        ax = fig.add_subplot(gs[idx, :])
        grid = _presence_grid(item.frame_counts, frames_per_row)
        masked = np.ma.masked_invalid(grid)
        ax.imshow(masked, aspect="auto", cmap=presence_cmap, vmin=0, vmax=1, resample=False)
        ax.set_title(f"{item.name}: presence map ({summaries[idx].coverage_percent:.2f}% coverage)")
        ax.set_ylabel(f"Frame block\nx{frames_per_row}")
        ax.set_xlabel("Frame within block")
        ax.set_xlim(0, frames_per_row - 1)

    ax_roll = fig.add_subplot(gs[len(series), :])
    for item in series:
        x, values = _rolling_coverage_percent(item.frame_counts, rolling_window)
        if values.size:
            ax_roll.plot(x + item.frame_offset, values, linewidth=1.5, label=item.name)
    ax_roll.axhline(95.0, color="#c73e3a", linestyle="--", linewidth=1.2, label="95% reference")
    ax_roll.set_title(f"Rolling Detection Coverage ({rolling_window}-frame window)")
    ax_roll.set_xlabel("Frame index")
    ax_roll.set_ylabel("Coverage (%)")
    ax_roll.set_ylim(-2, 102)
    ax_roll.legend(loc="best", fontsize=9)
    _style_axis(ax_roll)

    ax_hist = fig.add_subplot(gs[len(series) + 1, 0])
    any_gaps = False
    for item in series:
        gap_lengths = [end - start for start, end in compute_missing_segments(item.presence_mask)]
        if gap_lengths:
            any_gaps = True
            bins = np.arange(1, max(gap_lengths) + 2) - 0.5
            ax_hist.hist(gap_lengths, bins=bins, alpha=0.55, label=item.name)
    ax_hist.set_title("Missing Segment Lengths")
    ax_hist.set_xlabel("Segment length (frames)")
    ax_hist.set_ylabel("Count")
    if any_gaps:
        ax_hist.set_xscale("log")
        ax_hist.legend(loc="best", fontsize=9)
    else:
        ax_hist.text(0.5, 0.5, "No missing segments", ha="center", va="center", transform=ax_hist.transAxes)
    _style_axis(ax_hist)

    ax_counts = fig.add_subplot(gs[len(series) + 1, 1])
    max_len = max(item.total_frames for item in series)
    for item in series:
        ax_counts.plot(
            np.arange(item.total_frames) + item.frame_offset,
            np.asarray(item.frame_counts),
            linewidth=0.8,
            alpha=0.75,
            label=item.name,
        )
    ax_counts.set_title("Detections Per Frame")
    ax_counts.set_xlabel("Frame index")
    ax_counts.set_ylabel("Count")
    ax_counts.set_xlim(0, max(1, max_len))
    ax_counts.legend(loc="best", fontsize=9)
    _style_axis(ax_counts)

    ax_summary = fig.add_subplot(gs[len(series) + 2, :])
    ax_summary.axis("off")
    columns = [
        "\n".join(summary.as_text_lines())
        for summary in summaries
    ]
    x_positions = np.linspace(0.02, 0.98, num=len(columns) + 1)[:-1]
    for x, text in zip(x_positions, columns):
        ax_summary.text(
            x,
            0.95,
            text,
            transform=ax_summary.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f0f0f0", "edgecolor": "#666666", "alpha": 0.95},
        )

    fig.subplots_adjust(top=0.92, hspace=0.55, wspace=0.3)
    return fig


def render_detection_coverage_png(
    series: Sequence[DetectionCoverageSeries],
    *,
    rolling_window: int = DEFAULT_ROLLING_WINDOW_FRAMES,
    frames_per_row: int = DEFAULT_FRAMES_PER_ROW,
    title: Optional[str] = None,
    dpi: int = 150,
) -> bytes:
    """Render the dashboard as PNG bytes."""
    fig = create_detection_coverage_dashboard(
        series,
        rolling_window=rolling_window,
        frames_per_row=frames_per_row,
        title=title,
    )
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _series_attrs_for_kind(
    series: Sequence[DetectionCoverageSeries],
    source_kind: str,
) -> Optional[Mapping[str, Any]]:
    for item in series:
        attrs = item.attrs or {}
        if normalize_attr(attrs.get("source_kind")) == source_kind:
            return attrs
    return None


def _source_metadata(series: Sequence[DetectionCoverageSeries]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_runs: dict[str, Any] = {}
    source_paths: dict[str, Any] = {}
    raw_attrs = _series_attrs_for_kind(series, "raw_detection")
    if raw_attrs:
        source_runs["detect_run"] = raw_attrs.get("detect_run")
        source_runs["quality_run"] = raw_attrs.get("quality_run")
        source_paths["raw_detection"] = raw_attrs.get("source_path")
    refined_attrs = _series_attrs_for_kind(series, "refined_detection")
    if refined_attrs:
        source_runs["refined_run"] = refined_attrs.get("refined_run")
        source_runs["source_detect_run"] = refined_attrs.get("source_detect_run")
        source_paths["refined_detection"] = refined_attrs.get("source_path")
        source_paths["refined_payload"] = refined_attrs.get("payload_path")
    return source_runs, source_paths


def write_detection_coverage_dashboard_artifact(
    zarr_path: str | Path,
    series: Sequence[DetectionCoverageSeries],
    png_bytes: bytes,
    *,
    mode: str,
    artifact_name: str = DETECTION_COVERAGE_DASHBOARD_ARTIFACT,
    rolling_window: int = DEFAULT_ROLLING_WINDOW_FRAMES,
    frames_per_row: int = DEFAULT_FRAMES_PER_ROW,
    title: Optional[str] = None,
    dpi: int = 150,
    overwrite: bool = True,
) -> PlotArtifactResult:
    """Persist a rendered dashboard PNG under the run that owns the view.

    Raw-only dashboards are stored under ``detect_runs/<run>/visualizations``.
    Refined and comparison dashboards are stored under
    ``refined_detect_runs/<run>/visualizations`` because that run carries the
    refined-to-raw lineage.
    """
    if mode not in {"raw", "refined", "compare"}:
        raise ValueError("mode must be raw, refined, or compare")
    if not series:
        raise ValueError("at least one coverage series is required")
    if not png_bytes:
        raise ValueError("png_bytes must not be empty")

    root = _open_zarr_root(zarr_path, mode="a")
    if mode == "raw":
        raw_attrs = _series_attrs_for_kind(series, "raw_detection")
        if not raw_attrs or not raw_attrs.get("detect_run"):
            raise ValueError("raw dashboard persistence requires a raw detection series")
        target_group, target_run = resolve_zarr_run(
            root,
            "detect_runs",
            str(raw_attrs["detect_run"]),
            fallback_to_latest=False,
            run_label="Detection run",
        )
        target_kind = "raw_detection"
        target_path = f"detect_runs/{target_run}"
    else:
        refined_attrs = _series_attrs_for_kind(series, "refined_detection")
        if not refined_attrs or not refined_attrs.get("refined_run"):
            raise ValueError("refined/compare dashboard persistence requires a refined detection series")
        parent_path, _ = _resolve_refined_parent(root)
        target_group, target_run = resolve_zarr_run(
            root,
            parent_path,
            str(refined_attrs["refined_run"]),
            fallback_to_latest=False,
            run_label="Refined detection run",
        )
        target_kind = "refined_detection"
        target_path = f"{parent_path}/{target_run}"

    source_runs, source_paths = _source_metadata(series)
    summaries = [asdict(summarize_detection_coverage(item)) for item in series]
    params = {
        "mode": mode,
        "rolling_window_frames": int(rolling_window),
        "frames_per_row": int(frames_per_row),
        "title": title,
        "dpi": int(dpi),
    }
    signature_parts = [
        DETECTION_COVERAGE_DASHBOARD_SCHEMA_ID,
        mode,
        target_path,
        str(int(rolling_window)),
        str(int(frames_per_row)),
        ",".join(item.name for item in series),
    ]
    return write_png_visualization_artifact(
        target_group,
        artifact_name,
        png_bytes,
        description="Detection coverage dashboard for raw/refined detection presence and gap statistics.",
        created_by="fisheye.visualization.detection_coverage_dashboard",
        artifact_signature="|".join(signature_parts),
        role="diagnostic_dashboard",
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=params,
        extra_attrs={
            "dashboard_schema_id": DETECTION_COVERAGE_DASHBOARD_SCHEMA_ID,
            "dashboard_mode": mode,
            "target_kind": target_kind,
            "target_run": target_run,
            "target_path": target_path,
            "coverage_summaries": summaries,
        },
        overwrite=overwrite,
    )


def _load_series_for_args(args: argparse.Namespace) -> list[DetectionCoverageSeries]:
    if args.mode == "raw":
        return [
            load_raw_detect_coverage_series(
                args.zarr_path,
                detect_run=args.detect_run,
                quality_run=args.quality_run,
            )
        ]
    if args.mode == "refined":
        return [
            load_refined_detect_coverage_series(
                args.zarr_path,
                refined_run=args.refined_run,
            )
        ]
    if args.mode == "compare":
        refined = load_refined_detect_coverage_series(args.zarr_path, refined_run=args.refined_run)
        source_detect = normalize_attr((refined.attrs or {}).get("source_detect_run"))
        raw = load_raw_detect_coverage_series(
            args.zarr_path,
            detect_run=args.detect_run or source_detect,
            quality_run=args.quality_run,
        )
        return [raw, refined]
    raise ValueError(f"Unsupported mode: {args.mode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create raw/refined detection coverage dashboards.")
    parser.add_argument("zarr_path", help="Analysis Zarr archive path.")
    parser.add_argument("--mode", choices=("raw", "refined", "compare"), default="compare")
    parser.add_argument("--detect-run", help="Raw detection run name (default: latest or refined source).")
    parser.add_argument("--quality-run", help="Raw detection quality run name (default: latest).")
    parser.add_argument("--refined-run", help="Refined detection run name (default: latest).")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW_FRAMES)
    parser.add_argument("--frames-per-row", type=int, default=DEFAULT_FRAMES_PER_ROW)
    parser.add_argument("--title", help="Optional figure title.")
    parser.add_argument("--output", "-o", help="PNG output path.")
    parser.add_argument(
        "--write-zarr-artifact",
        action="store_true",
        help="Persist the PNG under the selected detection/refined detection run's visualizations group.",
    )
    parser.add_argument(
        "--artifact-name",
        default=DETECTION_COVERAGE_DASHBOARD_ARTIFACT,
        help=f"Zarr visualization artifact name (default: {DETECTION_COVERAGE_DASHBOARD_ARTIFACT}).",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="Do not replace an existing Zarr artifact.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.output and not args.write_zarr_artifact:
        parser.error("provide --output, --write-zarr-artifact, or both")
    loaded = _load_series_for_args(args)
    png = render_detection_coverage_png(
        loaded,
        rolling_window=args.rolling_window,
        frames_per_row=args.frames_per_row,
        title=args.title,
        dpi=args.dpi,
    )
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(png)
        print(f"Wrote {output}")
    if args.write_zarr_artifact:
        result = write_detection_coverage_dashboard_artifact(
            args.zarr_path,
            loaded,
            png,
            mode=args.mode,
            artifact_name=args.artifact_name,
            rolling_window=args.rolling_window,
            frames_per_row=args.frames_per_row,
            title=args.title,
            dpi=args.dpi,
            overwrite=not args.no_overwrite,
        )
        print(f"Persisted {result.path} ({result.byte_length} bytes)")
    for item in loaded:
        summary = summarize_detection_coverage(item)
        print(
            f"{summary.name}: coverage={summary.coverage_percent:.2f}% "
            f"missing_segments={summary.missing_segment_count} "
            f"max_gap={summary.max_missing_segment_frames} "
            f"review_priority={summary.review_priority}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
