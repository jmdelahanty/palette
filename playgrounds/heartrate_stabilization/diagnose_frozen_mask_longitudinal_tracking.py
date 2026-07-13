from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_frozen_heart_masks_longitudinal import _read_mask
from extract_reliable_local_rostral_heartrate import load_dataset


def _mask_at_pixels(mask: np.ndarray, pixel_xy: np.ndarray) -> np.ndarray:
    xy = np.rint(np.asarray(pixel_xy, dtype=np.float64)).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < mask.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < mask.shape[0])
    )
    selected = np.zeros(xy.shape[0], dtype=bool)
    selected[inside] = mask[xy[inside, 1], xy[inside, 0]]
    return selected


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relate frozen-mask frequency disagreement to tracking diagnostics."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--lower-edge-max-hz", type=float, default=2.2)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_npz)
    original = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    masks = {
        "original_38": original,
        "consensus_9": consensus,
        "intersection_8": original & consensus,
        "union_39": original | consensus,
    }
    selected_pixels = {
        name: _mask_at_pixels(mask, dataset.pixel_xy) for name, mask in masks.items()
    }
    with Path(args.longitudinal_csv).open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
    output_rows: list[dict[str, Any]] = []
    for source in source_rows:
        frame_start = int(source["window_frame_start"])
        frame_stop = int(source["window_frame_stop_inclusive"])
        start = int(np.searchsorted(frame_indices, frame_start, side="left"))
        stop = int(np.searchsorted(frame_indices, frame_stop, side="right"))
        name = str(source["mask"])
        selected = selected_pixels[name]
        valid = np.asarray(dataset.pixel_valid[start:stop, selected], dtype=bool)
        per_frame_valid = np.mean(valid, axis=1)
        source_xy = np.asarray(dataset.source_xy[start:stop, selected], dtype=np.float64)
        consecutive_valid = valid[1:] & valid[:-1]
        source_step = np.linalg.norm(np.diff(source_xy, axis=0), axis=2)
        source_step[~consecutive_valid] = np.nan
        motion = np.asarray(dataset.motion_prediction[start:stop, selected], dtype=np.float64)
        motion[~valid] = np.nan
        gradient = np.asarray(dataset.gradient_magnitude[start:stop, selected], dtype=np.float64)
        gradient[~valid] = np.nan
        frequency = float(source["candidate_frequency_hz"] or "nan")
        lower_edge = bool(
            np.isfinite(frequency)
            and frequency <= float(args.lower_edge_max_hz) + 1e-9
        )
        row: dict[str, Any] = {
            **source,
            "frequency_in_lower_search_edge": lower_edge,
            "mask_sample_valid_fraction": float(np.mean(valid)),
            "mask_all_pixels_valid_frame_fraction": float(np.mean(np.all(valid, axis=1))),
            "median_valid_pixel_fraction_per_frame": float(np.median(per_frame_valid)),
            "median_source_step_px": _finite_median(source_step),
            "median_abs_motion_prediction": _finite_median(np.abs(motion)),
            "median_gradient_magnitude": _finite_median(gradient),
            "median_transform_uncertainty": _finite_median(
                np.asarray(dataset.transform_uncertainty[start:stop], dtype=np.float64)
            ),
        }
        for nuisance_name in (
            "local_translation_px",
            "local_rotation_deg",
            "detection_confidence",
        ):
            if nuisance_name in dataset.nuisance_names:
                column = dataset.nuisance_names.index(nuisance_name)
                row[f"median_{nuisance_name}"] = _finite_median(
                    np.asarray(dataset.nuisance_values[start:stop, column], dtype=np.float64)
                )
        output_rows.append(row)

    fields = list(output_rows[0])
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".tracking_diagnostics.csv")
    summary_path = output_prefix.with_suffix(".tracking_diagnostics.summary.json")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    broad = [
        row
        for row in output_rows
        if row["mask"] in {"original_38", "union_39"} and row["status"] == "ok"
    ]
    lower_edge_rows = [row for row in broad if row["frequency_in_lower_search_edge"]]
    ordinary_rows = [row for row in broad if not row["frequency_in_lower_search_edge"]]
    metric_names = (
        "mask_sample_valid_fraction",
        "mask_all_pixels_valid_frame_fraction",
        "median_valid_pixel_fraction_per_frame",
        "median_source_step_px",
        "median_abs_motion_prediction",
        "median_gradient_magnitude",
        "median_transform_uncertainty",
        "median_local_translation_px",
        "median_local_rotation_deg",
        "median_detection_confidence",
    )
    summary = {
        "interpretation": "tracking_association_diagnostic_not_causal_test",
        "lower_edge_max_hz": float(args.lower_edge_max_hz),
        "broad_mask_lower_edge_row_count": len(lower_edge_rows),
        "broad_mask_ordinary_row_count": len(ordinary_rows),
        "lower_edge_window_indices": sorted(
            {int(row["window_index"]) for row in lower_edge_rows}
        ),
        "metric_medians": {
            metric: {
                "lower_edge": _finite_median(
                    np.asarray([float(row.get(metric, math.nan)) for row in lower_edge_rows])
                ),
                "ordinary": _finite_median(
                    np.asarray([float(row.get(metric, math.nan)) for row in ordinary_rows])
                ),
            }
            for metric in metric_names
        },
        "outputs": {"csv": str(csv_path), "summary_json": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
