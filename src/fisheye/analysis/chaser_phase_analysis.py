#!/usr/bin/env python3
"""
Phase-based visualization of chaser-fish interactions using Palette Zarr archives.

Loads fish detections and stimulus metadata directly from the archive – including
tracking data imported via ``analysis/stimulus_runs`` – to generate phase-specific
heatmaps, distance distributions, and summary metrics without requiring the
original stimulus H5 file.
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from fisheye.analysis.chaser_metrics_loader import load_chaser_metrics

try:
    from PIL import Image, PngImagePlugin
except ImportError:  # pragma: no cover
    Image = None
    PngImagePlugin = None

from fisheye.analysis.chaser_metrics_loader import (
    ChaserMetricsBundle,
    load_chaser_metrics,
)
from fisheye.utils.system import get_git_info


@dataclass
class ChaserAlignedData:
    """Aligned chaser/fish position data with derived distances."""

    frame_numbers: np.ndarray
    timestamps: np.ndarray
    fish_x: np.ndarray
    fish_y: np.ndarray
    chaser_x: np.ndarray
    chaser_y: np.ndarray
    distances: np.ndarray
    fish_interpolated: np.ndarray
    chase_events: List[Dict]
    metadata: Dict


PLOT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def _normalize_column(values: np.ndarray) -> np.ndarray:
    """Return numpy array with bytes decoded to UTF-8 strings when necessary."""
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    if values.dtype.kind == "O":
        decoded = [
            item.decode("utf-8", errors="ignore") if isinstance(item, (bytes, bytearray)) else item
            for item in values
        ]
        return np.array(decoded, dtype=object)
    return values


def _safe_len_dataset(group: Any, dataset: str) -> Optional[int]:
    """Return length of dataset within group, handling missing data gracefully."""
    if group is None:
        return None
    try:
        arr = group[dataset]
    except (KeyError, AttributeError, TypeError):
        return None
    try:
        return int(arr.shape[0])
    except Exception:
        try:
            return len(arr)  # type: ignore[arg-type]
        except Exception:
            return None


def _resolve_latest_run(group: Optional[Any]) -> Tuple[Optional[str], Optional[Any]]:
    """Return latest run name and group from a parent, if available."""
    if group is None:
        return None, None
    latest = group.attrs.get("latest")
    if isinstance(latest, str) and latest in group:
        return latest, group[latest]
    return None, None


def _collect_pipeline_provenance(root: Any) -> Dict[str, Any]:
    """Collect detect → crop → keypoint → ID provenance details for embedding."""
    issues: List[str] = []

    detect_parent = root.get("detect_runs")
    detect_run, detect_group = _resolve_latest_run(detect_parent)
    detect_rows = _safe_len_dataset(detect_group, "bbox_norm_coords")

    refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
    refined_run, refined_group = _resolve_latest_run(refined_parent)
    refined_rows = None
    refined_source_detect = None
    if refined_group is not None:
        interp_group = refined_group.get("interpolated")
        refined_rows = _safe_len_dataset(interp_group, "bbox_norm_coords")
        refined_source_detect = refined_group.attrs.get("source_detect_run")
        if detect_run and refined_source_detect and detect_run != refined_source_detect:
            issues.append(
                f"Refined detect run '{refined_run}' references detect '{refined_source_detect}' while latest detect is '{detect_run}'."
            )

    crop_parent = root.get("crop_runs")
    crop_run, crop_group = _resolve_latest_run(crop_parent)
    crop_rois = _safe_len_dataset(crop_group, "roi_images")
    crop_source = crop_group.attrs.get("detection_source_path") if crop_group is not None else None
    if crop_group is not None:
        if refined_run and refined_group is not None:
            manual_label = refined_group.attrs.get("manual_review_latest")
            if not manual_label and "manual" in refined_group:
                manual_label = "manual"
            expected = refined_group.path + (f"/{manual_label}" if manual_label else "/interpolated")
            if crop_source and crop_source != expected:
                issues.append(
                    f"Crop run '{crop_run}' sourced from '{crop_source}' but refined detection path is '{expected}'."
                )
        elif detect_run:
            expected = f"detect_runs/{detect_run}"
            if crop_source and crop_source != expected:
                issues.append(
                    f"Crop run '{crop_run}' sourced from '{crop_source}' but latest detect run is '{expected}'."
                )

    key_parent = root.get("refined_keypoints_runs") or root.get("keypoints_runs")
    key_run, key_group = _resolve_latest_run(key_parent)
    key_rows = _safe_len_dataset(key_group, "heading")
    key_source_crop = key_group.attrs.get("source_crop_run") if key_group is not None else None
    if key_source_crop and crop_run and key_source_crop != crop_run:
        issues.append(
            f"Keypoint run '{key_run}' references crop '{key_source_crop}' but latest crop is '{crop_run}'."
        )

    arena_parent = root.get("arena_assignment_runs")
    arena_run, arena_group = _resolve_latest_run(arena_parent)
    arena_rows = _safe_len_dataset(arena_group, "arena_ids")
    arena_source_detect = arena_group.attrs.get("source_detect_run") if arena_group is not None else None
    arena_source_refined = arena_group.attrs.get("source_refined_run") if arena_group is not None else None
    if arena_source_detect and detect_run and arena_source_detect != detect_run:
        issues.append(
            f"Arena assignment run '{arena_run}' references detect '{arena_source_detect}' but latest detect is '{detect_run}'."
        )
    if arena_source_refined and refined_run and arena_source_refined != refined_run:
        issues.append(
            f"Arena assignment run '{arena_run}' references refined detect '{arena_source_refined}' but latest refined detect is '{refined_run}'."
        )

    if refined_rows is not None and key_rows is not None and refined_rows != key_rows:
        issues.append(
            f"Refined detection count ({refined_rows}) != keypoint heading count ({key_rows})."
        )
    if refined_rows is not None and arena_rows is not None and refined_rows != arena_rows:
        issues.append(
            f"Refined detection count ({refined_rows}) != arena assignment count ({arena_rows})."
        )
    if crop_rois is not None and refined_rows is not None and crop_rois != refined_rows:
        issues.append(
            f"Crop ROI count ({crop_rois}) != refined detection count ({refined_rows})."
        )

    return {
        "runs": {
            "detect": detect_run,
            "refined_detect": refined_run,
            "crop": crop_run,
            "keypoints": key_run,
            "arena_assignment": arena_run,
        },
        "row_counts": {
            "detect": detect_rows,
            "refined_detect": refined_rows,
            "crop": crop_rois,
            "keypoints": key_rows,
            "arena_assignment": arena_rows,
        },
        "sources": {
            "refined_detect_source_detect": refined_source_detect,
            "crop_source": crop_source,
            "keypoints_source_crop": key_source_crop,
            "arena_source_detect": arena_source_detect,
            "arena_source_refined": arena_source_refined,
        },
        "issues": issues,
    }


def _load_table_from_group(obj: Any) -> pd.DataFrame:
    """
    Restore a structured dataset that was written via ``import_stimulus_to_zarr``.

    The importer stores structured arrays column-wise under a subgroup with
    ``field_names`` metadata; here we reconstruct a pandas DataFrame for ease of
    downstream processing.
    """
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(obj, "array_keys"):
        data = obj[:]
        return pd.DataFrame(data)

    field_names = obj.attrs.get("field_names")
    if not field_names:
        data: Dict[str, np.ndarray] = {}
        for name in obj.array_keys():
            data[name] = obj[name][:]
        return pd.DataFrame(data)

    table: Dict[str, np.ndarray] = {}
    for field in field_names:
        if field not in obj:
            continue
        table[field] = _normalize_column(obj[field][:])
    return pd.DataFrame(table)


class ChaserPhaseAnalyzer:
    """Align detections from Palette Zarr archives with embedded stimulus metadata."""

    def __init__(
        self,
        zarr_path: Path,
        *,
        detect_run: Optional[str] = None,
        stimulus_run: Optional[str] = None,
        roi_id: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.zarr_path = Path(zarr_path)
        self.detect_run = detect_run
        self.stimulus_run = stimulus_run
        self.roi_id = roi_id
        self.verbose = verbose

        self.root = zarr.open_group(str(self.zarr_path), mode="r")
        self._load_zarr_detection_data()
        self._load_stimulus_data()
        self.aligned_data = self._align_and_calculate()

        if self.verbose:
            self._print_summary()

    def _load_zarr_detection_data(self) -> None:
        """Load per-frame detections from the requested run within the Zarr archive."""
        self.fps = float(
            self.root.attrs.get(
                "fps",
                self.root.attrs.get("source_video_metadata", {}).get("fps", 60.0),
            )
        )
        self.camera_width = int(
            self.root.attrs.get(
                "width",
                self.root.attrs.get("source_video_metadata", {}).get("width", 4512),
            )
        )
        self.camera_height = int(
            self.root.attrs.get(
                "height",
                self.root.attrs.get("source_video_metadata", {}).get("height", 4512),
            )
        )
        self.pixel_to_mm = None
        if "calibration" in self.root:
            self.pixel_to_mm = self.root["calibration"].attrs.get("pixel_to_mm")

        detect_group, detect_path, source_detect = self._resolve_detection_group()
        self.detect_path = detect_path
        self.source_detect_run = source_detect

        counts_key = "frame_counts" if "frame_counts" in detect_group else "n_detections"
        if counts_key not in detect_group:
            raise ValueError(
                f"Detection group '{detect_path}' is missing per-frame counts "
                "(expected 'frame_counts' or 'n_detections')."
            )

        self.per_frame_counts = detect_group[counts_key][:]
        self.bbox_norm_coords = detect_group["bbox_norm_coords"][:]
        if self.per_frame_counts.ndim != 1:
            raise ValueError("Per-frame detection counts array must be one-dimensional.")
        self.total_frames = int(self.per_frame_counts.shape[0])

        self.detection_mask = self._resolve_detection_mask()
        self.fish_positions: List[Dict] = []

        cumulative = np.cumsum(np.insert(self.per_frame_counts, 0, 0))
        for frame_idx in range(self.total_frames):
            start = cumulative[frame_idx]
            end = cumulative[frame_idx + 1]
            if end <= start:
                continue

            selection_index = start
            if self.detection_mask is not None:
                frame_mask = self.detection_mask[start:end]
                if not np.any(frame_mask):
                    continue
                selection_index = start + int(np.argmax(frame_mask))

            bbox = self.bbox_norm_coords[selection_index]
            center_x = bbox[0] * self.camera_width
            center_y = bbox[1] * self.camera_height

            self.fish_positions.append(
                {
                    "frame": frame_idx,
                    "x": center_x,
                    "y": center_y,
                    "interpolated": False,
                }
            )

    def _resolve_detection_group(
        self,
    ) -> Tuple[Any, str, Optional[str]]:
        """
        Resolve the detection group to analyze.

        Returns the group object, its Zarr path, and the underlying source detect run
        (used for ROI filtering provenance).
        """
        if self.detect_run and "/" in self.detect_run:
            detect_path = self.detect_run.strip("/")
            try:
                detect_group = self.root[detect_path]
            except KeyError as exc:
                raise ValueError(f"Detection path '{detect_path}' not found in archive.") from exc
        else:
            detect_parent = self.root.get("detect_runs")
            if detect_parent is None:
                raise ValueError("Archive does not contain a 'detect_runs' group.")
            run_name = self.detect_run or detect_parent.attrs.get("latest")
            if run_name is None:
                raise ValueError("No detection run specified and 'detect_runs' has no latest attribute.")
            if run_name not in detect_parent:
                raise ValueError(f"Detection run '{run_name}' not found in detect_runs.")
            detect_path = f"detect_runs/{run_name}"
            detect_group = detect_parent[run_name]

        parts = detect_path.split("/")
        source_detect_run: Optional[str] = None
        if parts[0] == "detect_runs" and len(parts) >= 2:
            source_detect_run = parts[1]
        elif parts[0] == "refined_detect_runs" and len(parts) >= 2:
            parent_path = "/".join(parts[:2])
            refined_group = self.root[parent_path]
            source_detect_run = refined_group.attrs.get("source_detect_run")

        if not hasattr(detect_group, "array_keys") or not hasattr(detect_group, "attrs"):
            raise ValueError(f"Detection path '{detect_path}' is not a Zarr group.")

        return detect_group, detect_path, source_detect_run

    def _resolve_detection_mask(self) -> Optional[np.ndarray]:
        """Return detection mask for the requested ROI ID, using arena assignment runs if available."""
        if self.roi_id is None:
            return None

        candidates = ("arena_assignment_runs",)
        for parent_name in candidates:
            if parent_name not in self.root:
                continue

            parent_group = self.root[parent_name]
            keys_fn = getattr(parent_group, "group_keys", None)
            try:
                run_names = list(keys_fn()) if callable(keys_fn) else []
            except Exception:
                run_names = []

            for run_name in run_names:
                assign_group = parent_group[run_name]
                source_detect = assign_group.attrs.get("source_detect_run")
                if self.source_detect_run and source_detect != self.source_detect_run:
                    continue
                if "arena_ids" not in assign_group:
                    continue

                arena_ids = assign_group["arena_ids"][:]
                if arena_ids.shape[0] != self.bbox_norm_coords.shape[0]:
                    if self.verbose:
                        print(
                            "Warning: arena_ids length does not match detections for "
                            f"arena assignment run '{run_name}'. ROI filter disabled."
                        )
                    return None

                if self.verbose:
                    print(f"Filtering detections using ROI {self.roi_id} from {parent_name}/{run_name}")
                return arena_ids == self.roi_id

        if self.verbose:
            print(
                f"Warning: ROI {self.roi_id} requested but no matching arena assignment run "
                "was found. Using the first detection per frame instead."
            )
        return None

    def _load_stimulus_data(self) -> None:
        """Load frame metadata, chaser states, and events from analysis/stimulus_runs."""
        analysis_group = self.root.get("analysis")
        if analysis_group is None or "stimulus_runs" not in analysis_group:
            raise ValueError(
                "Archive is missing 'analysis/stimulus_runs'. "
                "Run 'python -m fisheye.analysis.import_stimulus_to_zarr' first."
            )

        stimulus_parent = analysis_group["stimulus_runs"]
        run_name = self.stimulus_run or stimulus_parent.attrs.get("latest")
        if run_name is None:
            raise ValueError(
                "No stimulus run specified and analysis/stimulus_runs has no latest attribute."
            )
        if run_name not in stimulus_parent:
            raise ValueError(f"Stimulus run '{run_name}' not found in analysis/stimulus_runs.")

        self.stimulus_run = run_name
        run_group = stimulus_parent[run_name]

        video_meta_group = run_group.get("video_metadata")
        if video_meta_group is None or "frame_metadata" not in video_meta_group:
            raise ValueError(f"Stimulus run '{run_name}' lacks video_metadata/frame_metadata.")
        self.frame_metadata = _load_table_from_group(video_meta_group["frame_metadata"])
        # Ensure numeric columns are numeric for alignment calculations.
        for column in ("stimulus_frame_num", "triggering_camera_frame_id"):
            if column in self.frame_metadata.columns:
                self.frame_metadata[column] = pd.to_numeric(self.frame_metadata[column], errors="coerce")

        tracking_group = run_group.get("tracking_data")
        if tracking_group is None or "chaser_states" not in tracking_group:
            raise ValueError(f"Stimulus run '{run_name}' lacks tracking_data/chaser_states.")
        self.chaser_states = _load_table_from_group(tracking_group["chaser_states"])
        if "stimulus_frame_num" not in self.chaser_states.columns:
            raise ValueError("Chaser states do not contain 'stimulus_frame_num'.")
        for column in ("stimulus_frame_num", "chaser_pos_x", "chaser_pos_y"):
            if column in self.chaser_states.columns:
                self.chaser_states[column] = pd.to_numeric(self.chaser_states[column], errors="coerce")

        self.chaser_states = self.chaser_states.dropna(subset=["stimulus_frame_num"])
        self.chaser_states["stimulus_frame_num"] = self.chaser_states["stimulus_frame_num"].astype(int)

        self.chase_events: List[Dict] = []
        if "events" in run_group:
            events_df = _load_table_from_group(run_group["events"])
            event_type_col = None
            for candidate in ("event_type_id", "event_id"):
                if candidate in events_df.columns:
                    event_type_col = candidate
                    break
            if event_type_col:
                timestamp_col = None
                for candidate in ("timestamp_ns_epoch", "timestamp_ns_session", "timestamp_ns"):
                    if candidate in events_df.columns:
                        timestamp_col = candidate
                        events_df[candidate] = pd.to_numeric(events_df[candidate], errors="coerce")
                        break
                camera_col = None
                for candidate in ("camera_frame_id", "camera_frame_num"):
                    if candidate in events_df.columns:
                        camera_col = candidate
                        events_df[candidate] = pd.to_numeric(events_df[candidate], errors="coerce")
                        break

                for _, row in events_df.iterrows():
                    evt = int(row[event_type_col]) if not pd.isna(row[event_type_col]) else None
                    if evt not in (27, 28):
                        continue
                    timestamp_s = None
                    if timestamp_col and not pd.isna(row[timestamp_col]):
                        timestamp_s = float(row[timestamp_col]) / 1e9
                    camera_frame = -1
                    if camera_col and not pd.isna(row[camera_col]):
                        camera_frame = int(row[camera_col])
                    self.chase_events.append(
                        {
                            "frame": camera_frame,
                            "type": "start" if evt == 27 else "end",
                            "timestamp": timestamp_s,
                        }
                    )

        coord_info = run_group.attrs.get("coordinate_transform")
        self.texture_width = 358
        self.texture_height = 358
        self.texture_to_camera_scale = self.camera_width / self.texture_width
        if coord_info:
            try:
                parsed = json.loads(coord_info) if isinstance(coord_info, str) else coord_info
                dims = parsed.get("camera_dimensions")
                if dims and len(dims) == 2:
                    self.camera_width = int(dims[0])
                    self.camera_height = int(dims[1])
                scale = parsed.get("texture_to_camera_scale")
                if scale:
                    self.texture_to_camera_scale = float(scale)
            except (TypeError, ValueError):
                if self.verbose:
                    print("Warning: Unable to parse coordinate_transform metadata; using defaults.")

        # Pre-compute lookup for chaser states by stimulus frame for faster alignment.
        self._chaser_by_stimulus = (
            self.chaser_states.sort_values("stimulus_frame_num")
            .drop_duplicates(subset="stimulus_frame_num", keep="first")
            .set_index("stimulus_frame_num")
        )

    def _align_and_calculate(self) -> ChaserAlignedData:
        """Align chaser and fish data, calculating distances frame-by-frame."""
        frame_numbers = np.arange(self.total_frames, dtype=np.int32)
        timestamps = frame_numbers / self.fps
        fish_x = np.full(self.total_frames, np.nan, dtype=float)
        fish_y = np.full(self.total_frames, np.nan, dtype=float)
        chaser_x = np.full(self.total_frames, np.nan, dtype=float)
        chaser_y = np.full(self.total_frames, np.nan, dtype=float)
        fish_interpolated = np.zeros(self.total_frames, dtype=bool)

        for pos in self.fish_positions:
            idx = pos["frame"]
            fish_x[idx] = pos["x"]
            fish_y[idx] = pos["y"]
            fish_interpolated[idx] = pos["interpolated"]

        if {"stimulus_frame_num", "triggering_camera_frame_id"}.issubset(self.frame_metadata.columns):
            stim_frames = self.frame_metadata["stimulus_frame_num"].to_numpy()
            cam_frames = self.frame_metadata["triggering_camera_frame_id"].to_numpy()
            valid_mask = ~np.isnan(stim_frames) & ~np.isnan(cam_frames)
            stim_frames = stim_frames[valid_mask].astype(int)
            cam_frames = cam_frames[valid_mask].astype(int)

            if self.verbose:
                print(
                    f"Aligning {len(stim_frames)} stimulus records with camera frames "
                    f"using stimulus run '{self.stimulus_run}'."
                )

            for stim_frame, cam_frame in zip(stim_frames, cam_frames):
                if cam_frame < 0 or cam_frame >= self.total_frames:
                    continue
                if stim_frame not in self._chaser_by_stimulus.index:
                    continue
                chaser = self._chaser_by_stimulus.loc[stim_frame]
                chaser_x[cam_frame] = float(chaser["chaser_pos_x"]) * self.texture_to_camera_scale
                chaser_y[cam_frame] = float(chaser["chaser_pos_y"]) * self.texture_to_camera_scale
        else:
            if self.verbose:
                print("Warning: frame metadata missing alignment columns; chaser alignment skipped.")

        distances = np.sqrt((fish_x - chaser_x) ** 2 + (fish_y - chaser_y) ** 2)

        metadata = {
            "zarr_source": str(self.zarr_path),
            "detect_path": self.detect_path,
            "stimulus_run": self.stimulus_run,
            "fps": self.fps,
            "pixel_to_mm": self.pixel_to_mm,
            "texture_to_camera_scale": self.texture_to_camera_scale,
            "total_frames": self.total_frames,
            "valid_frames": int(np.sum(~np.isnan(distances))),
        }

        return ChaserAlignedData(
            frame_numbers=frame_numbers,
            timestamps=timestamps,
            fish_x=fish_x,
            fish_y=fish_y,
            chaser_x=chaser_x,
            chaser_y=chaser_y,
            distances=distances,
            fish_interpolated=fish_interpolated,
            chase_events=self.chase_events,
            metadata=metadata,
        )

    def _print_summary(self) -> None:
        """Print analysis summary to the console."""
        print("\n" + "=" * 60)
        print("CHASER-FISH DISTANCE ANALYSIS SUMMARY")
        print("=" * 60)

        valid = ~np.isnan(self.aligned_data.distances)
        if not np.any(valid):
            print("No overlapping detections between fish and chaser positions.")
            return

        distances = self.aligned_data.distances[valid]
        print(f"\nDetection source: {self.detect_path}")
        print(f"Stimulus run: analysis/stimulus_runs/{self.stimulus_run}")
        print(f"\nDistance statistics (pixels):")
        print(f"  Mean:   {np.mean(distances):.1f}")
        print(f"  Median: {np.median(distances):.1f}")
        print(f"  Min:    {np.min(distances):.1f}")
        print(f"  Max:    {np.max(distances):.1f}")

        coverage = int(np.sum(valid))
        print(f"\nCoverage:")
        print(f"  Valid frames: {coverage} / {self.total_frames}")
        print(f"  Coverage: {coverage / self.total_frames * 100:.1f}%")
        if np.any(self.aligned_data.fish_interpolated):
            interp = int(np.sum(self.aligned_data.fish_interpolated))
            print(
                f"  Interpolated detections: {interp} "
                f"({interp / coverage * 100:.1f}%)"
            )


def _build_offline_aligned_data(
    analyzer: ChaserPhaseAnalyzer,
    bundle: ChaserMetricsBundle,
) -> ChaserAlignedData:
    total_frames = analyzer.total_frames
    fps = analyzer.fps if analyzer.fps else 60.0

    frame_numbers = np.arange(total_frames, dtype=np.int32)
    fish_x = np.full(total_frames, np.nan, dtype=float)
    fish_y = np.full(total_frames, np.nan, dtype=float)
    chaser_x = np.full(total_frames, np.nan, dtype=float)
    chaser_y = np.full(total_frames, np.nan, dtype=float)
    distances = np.full(total_frames, np.nan, dtype=float)
    fish_interpolated = np.ones(total_frames, dtype=bool)
    timestamps = np.full(total_frames, np.nan, dtype=float)

    camera_frames = np.asarray(bundle.camera_frame_ids, dtype=np.int64)
    timestamp_ns = np.asarray(bundle.timestamp_ns, dtype=np.int64)
    has_offline = np.asarray(bundle.offline.get("has_offline"), dtype=bool)
    fish_positions = np.asarray(bundle.offline.get("fish_centroid_px"), dtype=np.float64)
    chaser_positions = np.asarray(bundle.offline.get("chaser_position_px"), dtype=np.float64)
    distances_offline = np.asarray(bundle.offline.get("distance_px"), dtype=np.float64)

    for idx, cam_frame in enumerate(camera_frames):
        if cam_frame < 0 or cam_frame >= total_frames:
            continue
        if idx < timestamp_ns.shape[0] and timestamp_ns[idx] >= 0:
            timestamps[cam_frame] = timestamp_ns[idx] / 1e9
        if idx < has_offline.shape[0] and has_offline[idx]:
            fish_interpolated[cam_frame] = False
        if idx < fish_positions.shape[0]:
            fish_pt = fish_positions[idx]
            if np.all(np.isfinite(fish_pt)):
                fish_x[cam_frame] = float(fish_pt[0])
                fish_y[cam_frame] = float(fish_pt[1])
        if idx < chaser_positions.shape[0]:
            chaser_pt = chaser_positions[idx]
            if np.all(np.isfinite(chaser_pt)):
                chaser_x[cam_frame] = float(chaser_pt[0])
                chaser_y[cam_frame] = float(chaser_pt[1])
        if idx < distances_offline.shape[0]:
            distances[cam_frame] = float(distances_offline[idx])

    missing_timestamps = np.isnan(timestamps)
    if np.any(missing_timestamps):
        timestamps[missing_timestamps] = frame_numbers[missing_timestamps] / fps

    metadata = {
        "alignment": "offline_metrics",
        "stimulus_run": bundle.provenance.get("stimulus_run"),
        "metrics_run": bundle.provenance.get("metrics_run"),
        "source_keypoints_run": bundle.provenance.get("source_keypoints_run"),
        "chaser_index": bundle.provenance.get("chaser_index"),
        "fps": analyzer.fps,
        "total_frames": total_frames,
    }

    return ChaserAlignedData(
        frame_numbers=frame_numbers,
        timestamps=timestamps,
        fish_x=fish_x,
        fish_y=fish_y,
        chaser_x=chaser_x,
        chaser_y=chaser_y,
        distances=distances,
        fish_interpolated=fish_interpolated,
        chase_events=analyzer.chase_events,
        metadata=metadata,
    )


def identify_experimental_phases(analyzer: ChaserPhaseAnalyzer) -> Dict[str, Dict]:
    """Return frame ranges for pre-training, training, and post-training phases."""
    fps = analyzer.aligned_data.metadata["fps"]
    total_frames = analyzer.aligned_data.metadata["total_frames"]

    pre_end = min(int(300 * fps), total_frames)
    train_end = min(int(450 * fps), total_frames)
    post_end = min(int(750 * fps), total_frames)

    phases = {
        "pre_training": {
            "start": 0,
            "end": pre_end,
            "duration_s": pre_end / fps,
        },
        "training": {
            "start": pre_end,
            "end": max(pre_end, train_end),
            "duration_s": max(train_end - pre_end, 0) / fps,
        },
        "post_training": {
            "start": max(train_end, pre_end),
            "end": max(post_end, max(train_end, pre_end)),
            "duration_s": max(post_end - max(train_end, pre_end), 0) / fps,
        },
    }
    phases["post_training"]["end"] = min(phases["post_training"]["end"], total_frames)
    return phases


def calculate_phase_metrics(
    analyzer: ChaserPhaseAnalyzer, phase_frames: Tuple[int, int]
) -> Dict[str, float]:
    """Calculate summary metrics for a single phase."""
    start_frame, end_frame = phase_frames
    data = analyzer.aligned_data
    start_frame = max(int(start_frame), 0)
    end_frame = min(int(end_frame), len(data.distances))

    phase_distances = data.distances[start_frame:end_frame]
    phase_fish_x = data.fish_x[start_frame:end_frame]
    phase_fish_y = data.fish_y[start_frame:end_frame]
    phase_chaser_x = data.chaser_x[start_frame:end_frame]
    phase_chaser_y = data.chaser_y[start_frame:end_frame]

    valid_mask = ~np.isnan(phase_distances)
    valid_distances = phase_distances[valid_mask]

    metrics: Dict[str, float] = {}
    if valid_distances.size > 0:
        metrics["mean_distance"] = float(np.mean(valid_distances))
        metrics["median_distance"] = float(np.median(valid_distances))
        metrics["min_distance"] = float(np.min(valid_distances))
        metrics["max_distance"] = float(np.max(valid_distances))
        metrics["std_distance"] = float(np.std(valid_distances))
        metrics["q25_distance"] = float(np.percentile(valid_distances, 25))
        metrics["q75_distance"] = float(np.percentile(valid_distances, 75))

        close_threshold = 500
        medium_threshold = 1500
        metrics["time_close_pct"] = (
            float(np.sum(valid_distances < close_threshold)) / valid_distances.size * 100
        )
        metrics["time_medium_pct"] = (
            float(
                np.sum(
                    (valid_distances >= close_threshold)
                    & (valid_distances < medium_threshold)
                )
            )
            / valid_distances.size
            * 100
        )
        metrics["time_far_pct"] = (
            float(np.sum(valid_distances >= medium_threshold))
            / valid_distances.size
            * 100
        )

        if valid_distances.size > 1:
            velocity = np.diff(phase_distances) * analyzer.fps
            valid_vel = velocity[~np.isnan(velocity)]
            if valid_vel.size > 0:
                metrics["mean_relative_velocity"] = float(np.mean(valid_vel))
                metrics["approach_events"] = float(np.sum(valid_vel < -50))
                metrics["escape_events"] = float(np.sum(valid_vel > 50))
    else:
        null_keys = [
            "mean_distance",
            "median_distance",
            "min_distance",
            "max_distance",
            "std_distance",
            "q25_distance",
            "q75_distance",
            "time_close_pct",
            "time_medium_pct",
            "time_far_pct",
            "mean_relative_velocity",
            "approach_events",
            "escape_events",
        ]
        for key in null_keys:
            metrics[key] = np.nan

    metrics["coverage_pct"] = float(np.sum(valid_mask)) / max(
        len(phase_distances), 1
    ) * 100
    metrics["valid_frames"] = float(np.sum(valid_mask))
    metrics["total_frames"] = float(len(phase_distances))

    return metrics


def create_phase_heatmap(
    fish_x: np.ndarray,
    fish_y: np.ndarray,
    chaser_x: np.ndarray,
    chaser_y: np.ndarray,
    arena_size: Tuple[int, int],
    bins: int = 50,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Create occupancy heatmaps for fish and chaser positions."""
    valid_fish = ~(np.isnan(fish_x) | np.isnan(fish_y))
    valid_chaser = ~(np.isnan(chaser_x) | np.isnan(chaser_y))

    x_range = [0, arena_size[0]]
    y_range = [0, arena_size[1]]

    fish_heatmap = None
    chaser_heatmap = None
    xedges = np.linspace(x_range[0], x_range[1], bins + 1)
    yedges = np.linspace(y_range[0], y_range[1], bins + 1)

    if np.any(valid_fish):
        fish_hist, xedges, yedges = np.histogram2d(
            fish_x[valid_fish],
            fish_y[valid_fish],
            bins=bins,
            range=[x_range, y_range],
        )
        fish_heatmap = gaussian_filter(fish_hist.T, sigma=1.5)

    if np.any(valid_chaser):
        chaser_hist, _, _ = np.histogram2d(
            chaser_x[valid_chaser],
            chaser_y[valid_chaser],
            bins=bins,
            range=[x_range, y_range],
        )
        chaser_heatmap = gaussian_filter(chaser_hist.T, sigma=1.5)

    return fish_heatmap, chaser_heatmap, (xedges, yedges)


def format_metric(value: float, suffix: str = "", precision: int = 1) -> str:
    """Format metric values with graceful handling of NaNs."""
    if value is None or np.isnan(value):
        return "—"
    return f"{value:.{precision}f}{suffix}"


def _to_serializable(value: Any) -> Any:
    """Recursively convert numpy types to JSON-friendly primitives."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_serializable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _compose_provenance(
    analyzer: ChaserPhaseAnalyzer,
    phases: Dict[str, Dict],
    metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Build provenance payload for embedding in plot metadata."""
    payload = {
        "zarr_path": str(analyzer.zarr_path),
        "detect_path": getattr(analyzer, "detect_path", None),
        "stimulus_run": analyzer.stimulus_run,
        "roi_id": analyzer.roi_id,
        "git": get_git_info(),
        "phases": _to_serializable(phases),
        "metrics": _to_serializable(metrics),
        "metadata": _to_serializable(analyzer.aligned_data.metadata),
        "pipeline": _to_serializable(_collect_pipeline_provenance(analyzer.root)),
    }
    return payload


def _serialize_xmp_packet(payload: Dict[str, Any]) -> str:
    """Serialize provenance payload into an XMP packet."""
    json_payload = json.dumps(_to_serializable(payload), separators=(",", ":"), ensure_ascii=True)
    return (
        '<?xpacket begin="\\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:palette="https://palette.hhmi.org/ns/analysis/">\n'
        f'   <palette:provenance>{json_payload}</palette:provenance>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )


def _save_plot_with_metadata(
    fig: plt.Figure,
    target_path: Path,
    base_payload: Dict[str, Any],
) -> None:
    """Write figure to disk with embedded XMP provenance."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(base_payload)
    payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    xmp_packet = _serialize_xmp_packet(payload)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    if Image is not None and PngImagePlugin is not None:
        image = Image.open(buf)
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("XML:com.adobe.xmp", xmp_packet)
        image.save(target_path, pnginfo=pnginfo)
    else:  # pragma: no cover
        with open(target_path, "wb") as handle:
            handle.write(buf.getvalue())

    print(f"Phase analysis plot saved to: {target_path}")


def _default_plot_filename(analyzer: ChaserPhaseAnalyzer) -> str:
    """Generate default filename for interactive saves."""
    detect_slug = getattr(analyzer, "detect_path", None) or "detect"
    detect_slug = detect_slug.replace("/", "_")
    roi_part = f"_roi{analyzer.roi_id}" if analyzer.roi_id is not None else ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"phase_analysis_{detect_slug}{roi_part}_{timestamp}.png"


def _attach_save_handlers(
    fig: plt.Figure,
    analyzer: ChaserPhaseAnalyzer,
    base_payload: Dict[str, Any],
) -> None:
    """Attach key and toolbar handlers that save plots with provenance."""
    canvas = getattr(fig, "canvas", None)
    if canvas is None:
        return

    def _save_to_default() -> None:
        filename = _default_plot_filename(analyzer)
        target = PLOT_OUTPUT_DIR / filename
        _save_plot_with_metadata(fig, target, base_payload)

    manager = getattr(canvas, "manager", None)
    toolbar = getattr(manager, "toolbar", None)
    if toolbar is not None:
        def _toolbar_save(*args, **kwargs) -> None:
            _save_to_default()

        toolbar.save_figure = _toolbar_save
    else:
        def _on_key(event) -> None:
            if getattr(event, "key", "").lower() == "s":
                _save_to_default()

        canvas.mpl_connect("key_press_event", _on_key)


def plot_phase_analysis(
    analyzer: ChaserPhaseAnalyzer,
    phases: Dict[str, Dict],
    bins: int = 50,
    save_path: Optional[Path] = None,
    show: bool = True,
    title_suffix: str = "",
) -> Dict[str, Dict[str, float]]:
    """Create the phase comparison visualization."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.3,
        wspace=0.25,
        height_ratios=[1, 1, 1, 0.3],
    )

    phase_names = ["pre_training", "training", "post_training"]
    phase_titles = [
        "Pre-Training (0-5 min)",
        "Training (5-7.5 min)",
        "Post-Training (7.5-12.5 min)",
    ]
    phase_colors = ["blue", "red", "green"]

    all_metrics: Dict[str, Dict[str, float]] = {}
    arena_size = (analyzer.camera_width, analyzer.camera_height)

    for col, (phase_name, phase_title, color) in enumerate(
        zip(phase_names, phase_titles, phase_colors)
    ):
        phase_info = phases[phase_name]
        start_frame = phase_info["start"]
        end_frame = phase_info["end"]

        metrics = calculate_phase_metrics(analyzer, (start_frame, end_frame))
        all_metrics[phase_name] = metrics

        data = analyzer.aligned_data
        phase_fish_x = data.fish_x[start_frame:end_frame]
        phase_fish_y = data.fish_y[start_frame:end_frame]
        phase_chaser_x = data.chaser_x[start_frame:end_frame]
        phase_chaser_y = data.chaser_y[start_frame:end_frame]
        phase_distances = data.distances[start_frame:end_frame]

        ax1 = fig.add_subplot(gs[0, col])
        fish_heat, chaser_heat, edges = create_phase_heatmap(
            phase_fish_x, phase_fish_y, phase_chaser_x, phase_chaser_y, arena_size, bins
        )

        if fish_heat is not None:
            im1 = ax1.imshow(
                fish_heat,
                origin="lower",
                aspect="equal",
                cmap="hot",
                extent=[
                    edges[0][0],
                    edges[0][-1],
                    edges[1][0],
                    edges[1][-1],
                ],
            )
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title(f"{phase_title}\nFish Occupancy", fontweight="bold")
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")
        circle = Circle(
            (arena_size[0] / 2, arena_size[1] / 2),
            min(arena_size) / 2 - 256,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
            alpha=0.5,
        )
        ax1.add_patch(circle)

        ax2 = fig.add_subplot(gs[1, col])
        if chaser_heat is not None:
            im2 = ax2.imshow(
                chaser_heat,
                origin="lower",
                aspect="equal",
                cmap="cool",
                extent=[
                    edges[0][0],
                    edges[0][-1],
                    edges[1][0],
                    edges[1][-1],
                ],
            )
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title("Chaser Occupancy", fontweight="bold")
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        circle = Circle(
            (arena_size[0] / 2, arena_size[1] / 2),
            min(arena_size) / 2 - 256,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
            alpha=0.5,
        )
        ax2.add_patch(circle)

        ax3 = fig.add_subplot(gs[2, col])
        valid_distances = phase_distances[~np.isnan(phase_distances)]
        if valid_distances.size > 0:
            ax3.hist(
                valid_distances,
                bins=50,
                alpha=0.7,
                color=color,
                edgecolor="black",
                density=True,
            )
            if valid_distances.size > 1:
                kde = gaussian_kde(valid_distances)
                x_range = np.linspace(
                    valid_distances.min(), valid_distances.max(), 200
                )
                ax3.plot(x_range, kde(x_range), color="black", linewidth=2)
            ax3.axvline(
                metrics.get("mean_distance", np.nan),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {format_metric(metrics.get('mean_distance', np.nan))}",
            )
            ax3.axvline(
                metrics.get("median_distance", np.nan),
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {format_metric(metrics.get('median_distance', np.nan))}",
            )
            ax3.set_xlabel("Distance (pixels)")
            ax3.set_ylabel("Probability Density")
            ax3.set_title("Distance Distribution")
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")
    headers = ["Metric", "Pre-Training", "Training", "Post-Training"]
    table_data = [
        [
            "Mean Distance (px)",
            format_metric(all_metrics["pre_training"].get("mean_distance")),
            format_metric(all_metrics["training"].get("mean_distance")),
            format_metric(all_metrics["post_training"].get("mean_distance")),
        ],
        [
            "Median Distance (px)",
            format_metric(all_metrics["pre_training"].get("median_distance")),
            format_metric(all_metrics["training"].get("median_distance")),
            format_metric(all_metrics["post_training"].get("median_distance")),
        ],
        [
            "Min Distance (px)",
            format_metric(all_metrics["pre_training"].get("min_distance")),
            format_metric(all_metrics["training"].get("min_distance")),
            format_metric(all_metrics["post_training"].get("min_distance")),
        ],
        [
            "Time Close (<500px)",
            format_metric(all_metrics["pre_training"].get("time_close_pct"), "%"),
            format_metric(all_metrics["training"].get("time_close_pct"), "%"),
            format_metric(all_metrics["post_training"].get("time_close_pct"), "%"),
        ],
        [
            "Time Medium (500-1500px)",
            format_metric(all_metrics["pre_training"].get("time_medium_pct"), "%"),
            format_metric(all_metrics["training"].get("time_medium_pct"), "%"),
            format_metric(all_metrics["post_training"].get("time_medium_pct"), "%"),
        ],
        [
            "Time Far (>1500px)",
            format_metric(all_metrics["pre_training"].get("time_far_pct"), "%"),
            format_metric(all_metrics["training"].get("time_far_pct"), "%"),
            format_metric(all_metrics["post_training"].get("time_far_pct"), "%"),
        ],
        [
            "Approach Events",
            format_metric(all_metrics["pre_training"].get("approach_events"), "", 0),
            format_metric(all_metrics["training"].get("approach_events"), "", 0),
            format_metric(all_metrics["post_training"].get("approach_events"), "", 0),
        ],
        [
            "Escape Events",
            format_metric(all_metrics["pre_training"].get("escape_events"), "", 0),
            format_metric(all_metrics["training"].get("escape_events"), "", 0),
            format_metric(all_metrics["post_training"].get("escape_events"), "", 0),
        ],
        [
            "Data Coverage",
            format_metric(all_metrics["pre_training"].get("coverage_pct"), "%"),
            format_metric(all_metrics["training"].get("coverage_pct"), "%"),
            format_metric(all_metrics["post_training"].get("coverage_pct"), "%"),
        ],
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.23, 0.23, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white")

    phase_colors = ["#e8f4ff", "#ffe8e8", "#e8ffe8"]
    for row_idx in range(1, len(table_data) + 1):
        for col_idx in range(1, 4):
            table[(row_idx, col_idx)].set_facecolor(phase_colors[col_idx - 1])

    suffix = f" {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Phase-Based Chaser-Fish Analysis{suffix}: Pre-Training vs Training vs Post-Training",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    base_payload = _compose_provenance(analyzer, phases, all_metrics)

    if save_path:
        _save_plot_with_metadata(
            fig,
            Path(save_path),
            base_payload,
        )

    _attach_save_handlers(fig, analyzer, base_payload)

    if show:
        plt.show()

    return all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-based analysis of chaser-fish interactions using Palette Zarr data.",
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--detect-run",
        type=str,
        help=(
            "Detection run to analyze (e.g., 'detect_runs/<run>' or the "
            "preferred current refined override "
            "'refined_detect_runs/<run>/instances')."
        ),
    )
    parser.add_argument(
        "--stimulus-run",
        type=str,
        help="Stimulus import run under analysis/stimulus_runs (default: latest).",
    )
    parser.add_argument(
        "--roi-id",
        type=int,
        help="Filter to a specific ROI identity if assignments are available.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins for heatmap histograms (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate plots without displaying them interactively.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Render only the offline metrics visualization (skip online analysis).",
    )
    parser.add_argument(
        "--online-only",
        action="store_true",
        help="Render only the online analysis (skip offline metrics).",
    )
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to use when loading offline metrics (default: 0).",
    )
    parser.add_argument(
        "--metrics-run",
        type=str,
        help="Specific legacy analysis/chaser_fish_metrics/<run> to use for offline metrics (default: latest).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyzer = ChaserPhaseAnalyzer(
        args.zarr_path,
        detect_run=args.detect_run,
        stimulus_run=args.stimulus_run,
        roi_id=args.roi_id,
        verbose=not args.quiet,
    )

    phases = identify_experimental_phases(analyzer)
    print("\nExperimental Phases:")
    for phase_name, phase_info in phases.items():
        duration_frames = phase_info["end"] - phase_info["start"]
        duration_s = duration_frames / analyzer.fps if analyzer.fps else 0.0
        print(
            f"  {phase_name}: frames {phase_info['start']}-{phase_info['end']} "
            f"({duration_s:.1f} seconds)"
        )

    render_online = not args.offline_only
    render_offline = True
    if args.online_only:
        render_online = True
        render_offline = False
    if args.offline_only and args.online_only:
        render_offline = True

    output_path = args.output
    metrics_online: Optional[Dict[str, Dict[str, float]]] = None

    if render_online:
        metrics_online = plot_phase_analysis(
            analyzer,
            phases,
            bins=args.bins,
            save_path=output_path,
            show=not args.no_show,
            title_suffix="(Online Chaser States)",
        )

    def _print_summary(label: str, summary_metrics: Dict[str, Dict[str, float]]) -> None:
        print("\n" + "=" * 60)
        print(f"PHASE COMPARISON SUMMARY {label}")
        print("=" * 60)
        print("\nMean Distance (pixels):")
        for phase in ["pre_training", "training", "post_training"]:
            value = format_metric(summary_metrics[phase].get("mean_distance"))
            print(f"  {phase}: {value}")

        print("\nTime Spent Close (<500 pixels):")
        for phase in ["pre_training", "training", "post_training"]:
            value = format_metric(summary_metrics[phase].get("time_close_pct"), "%")
            print(f"  {phase}: {value}")

    if metrics_online is not None:
        _print_summary("(Online)", metrics_online)

    if render_offline:
        try:
            bundle = load_chaser_metrics(
                args.zarr_path,
                stimulus_run=analyzer.stimulus_run,
                metrics_run=args.metrics_run,
                chaser_index=args.chaser_index,
            )
        except Exception as exc:  # pragma: no cover - CLI feedback
            print(f"\nWarning: unable to load offline metrics ({exc}). Skipping offline visualization.")
            return 0

        offline_aligned = _build_offline_aligned_data(analyzer, bundle)
        if np.all(np.isnan(offline_aligned.distances)):
            print("\nWarning: offline metrics contain no valid distances; skipping offline visualization.")
            return 0

        offline_output = None
        if output_path:
            offline_output = output_path.with_name(
                f"{output_path.stem}_offline{output_path.suffix}"
            )

        original_data = analyzer.aligned_data
        original_detect_path = getattr(analyzer, "detect_path", None)

        analyzer.aligned_data = offline_aligned
        analyzer.detect_path = f"offline_metrics/{bundle.provenance.get('metrics_run', 'latest')}"

        offline_metrics = plot_phase_analysis(
            analyzer,
            phases,
            bins=args.bins,
            save_path=offline_output,
            show=not args.no_show,
            title_suffix="(Offline Metrics)",
        )

        _print_summary("(Offline)", offline_metrics)

        analyzer.aligned_data = original_data
        analyzer.detect_path = original_detect_path
        _print_summary("(Offline)", offline_metrics)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
