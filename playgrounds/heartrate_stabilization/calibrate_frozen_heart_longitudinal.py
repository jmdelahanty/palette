from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analyze_frozen_heart_masks_longitudinal import _read_mask, _window_dataset
from compare_frozen_heart_masks import _load_null_batch, _write_null_batch
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.dynamic_heart_support import (
    compute_dynamic_support_null_batch,
    prepare_dynamic_support_null_context,
)
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig, analyze_heartrate


_INFERENTIAL_MASKS = ("consensus_9", "intersection_8")


def _window_identity(
    dataset_path: Path,
    masks: Mapping[str, np.ndarray],
    config: HeartrateConfig,
    *,
    frame_start: int,
    frame_stop_inclusive: int,
    seed: int,
) -> str:
    stat = dataset_path.stat()
    payload = {
        "schema": "longitudinal_frozen_mask_null_v1",
        "dataset": str(dataset_path.resolve()),
        "dataset_size": int(stat.st_size),
        "dataset_mtime_ns": int(stat.st_mtime_ns),
        "frame_start": int(frame_start),
        "frame_stop_inclusive": int(frame_stop_inclusive),
        "config": asdict(config),
        "seed": int(seed),
        "masks": {
            name: hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()
            for name, mask in masks.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or ())


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Familywise latent-null calibration for frozen-mask longitudinal windows."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-max-hz", type=float, default=4.0)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--surrogate-count", type=int, default=199)
    parser.add_argument("--surrogate-batch-size", type=int, default=25)
    parser.add_argument("--surrogate-workers", type=int, default=4)
    parser.add_argument("--surrogate-batch-dir", type=Path)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    count = int(args.surrogate_count)
    batch_size = int(args.surrogate_batch_size)
    workers = int(args.surrogate_workers)
    if count < 1 or batch_size < 1 or workers < 1:
        raise ValueError("surrogate count, batch size, and workers must be positive")
    dataset = load_dataset(args.dataset_npz)
    original = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    masks = {
        "consensus_9": consensus,
        "intersection_8": original & consensus,
    }
    rows, original_fields = _read_rows(args.longitudinal_csv)
    rows_by_window: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        rows_by_window.setdefault(int(row["window_index"]), {})[str(row["mask"])] = row
    eligible_indices = [
        window_index
        for window_index, by_mask in sorted(rows_by_window.items())
        if all(by_mask.get(name, {}).get("status") == "ok" for name in _INFERENTIAL_MASKS)
    ]
    if not eligible_indices:
        raise ValueError("no windows have both compact frozen masks available")
    config = HeartrateConfig(
        band_min_hz=float(args.frequency_min_hz),
        band_max_hz=float(args.frequency_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
        surrogate_count=0,
        random_seed=int(args.seed),
    ).validated()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    batch_dir = (
        Path(args.surrogate_batch_dir)
        if args.surrogate_batch_dir is not None
        else output_prefix.parent / f"{output_prefix.name}.surrogate_batches"
    )
    null_latent = np.full(
        (len(eligible_indices), len(_INFERENTIAL_MASKS), count),
        np.nan,
        dtype=np.float64,
    )
    observed = np.full(
        (len(eligible_indices), len(_INFERENTIAL_MASKS)),
        np.nan,
        dtype=np.float64,
    )
    frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
    for eligible_position, window_index in enumerate(eligible_indices):
        by_mask = rows_by_window[window_index]
        frame_start = int(by_mask[_INFERENTIAL_MASKS[0]]["window_frame_start"])
        frame_stop = int(by_mask[_INFERENTIAL_MASKS[0]]["window_frame_stop_inclusive"])
        start = int(np.searchsorted(frame_indices, frame_start, side="left"))
        stop = int(np.searchsorted(frame_indices, frame_stop, side="right"))
        window = _window_dataset(dataset, start, stop)
        base = analyze_heartrate(window, config)
        context = prepare_dynamic_support_null_context(
            window,
            config,
            base,
            heart_masks=masks,
            frequency_min_hz=float(args.frequency_min_hz),
            frequency_max_hz=float(args.frequency_max_hz),
        )
        window_seed = int(args.seed) + 100_003 * int(window_index + 1)
        identity = _window_identity(
            Path(args.dataset_npz),
            masks,
            config,
            frame_start=frame_start,
            frame_stop_inclusive=frame_stop,
            seed=window_seed,
        )
        for mask_position, name in enumerate(_INFERENTIAL_MASKS):
            observed[eligible_position, mask_position] = float(by_mask[name]["latent_score"])
        for batch_start in range(0, count, batch_size):
            batch_stop = min(batch_start + batch_size, count)
            indices = np.arange(batch_start, batch_stop, dtype=np.int64)
            batch_path = batch_dir / (
                f"window_{window_index:03d}.{identity[:16]}."
                f"surrogates_{batch_start:06d}_{batch_stop:06d}.npz"
            )
            scores = _load_null_batch(
                batch_path,
                identity=identity,
                names=_INFERENTIAL_MASKS,
                expected_indices=indices,
            )
            if scores is None:
                batch = compute_dynamic_support_null_batch(
                    window,
                    config,
                    base,
                    heart_masks=masks,
                    surrogate_indices=indices,
                    seed=window_seed,
                    frequency_min_hz=float(args.frequency_min_hz),
                    frequency_max_hz=float(args.frequency_max_hz),
                    workers=workers,
                    context=context,
                )
                scores = {
                    "support": batch.support_scores,
                    "shared_phase": batch.shared_phase_scores,
                    "latent": batch.latent_scores,
                }
                _write_null_batch(
                    batch_path,
                    identity=identity,
                    indices=indices,
                    scores=scores,
                )
            for mask_position, name in enumerate(_INFERENTIAL_MASKS):
                null_latent[
                    eligible_position,
                    mask_position,
                    batch_start:batch_stop,
                ] = scores["latent"][name]
        print(
            f"window {eligible_position + 1}/{len(eligible_indices)} "
            f"index={window_index} frames={frame_start}:{frame_stop} nulls={count}",
            flush=True,
        )

    maximum_null = np.nanmax(null_latent, axis=(0, 1))
    threshold = float(np.quantile(maximum_null, 0.95, method="higher"))
    p_values = (1.0 + np.sum(maximum_null[None, None, :] >= observed[:, :, None], axis=2)) / (
        count + 1.0
    )
    exceeds = (p_values <= 0.05) & (observed > threshold)
    observed_median = np.nanmedian(observed, axis=0)
    null_median_by_mask = np.nanmedian(null_latent, axis=0)
    null_max_median_across_masks = np.nanmax(null_median_by_mask, axis=0)
    median_p_values = (
        1.0
        + np.sum(
            null_max_median_across_masks[None, :] >= observed_median[:, None],
            axis=1,
        )
    ) / (count + 1.0)
    fields = original_fields + [
        "latent_global_familywise_p_value",
        "latent_global_familywise_exceeds_null",
    ]
    position_by_window = {value: index for index, value in enumerate(eligible_indices)}
    mask_position = {name: index for index, name in enumerate(_INFERENTIAL_MASKS)}
    for row in rows:
        window_index = int(row["window_index"])
        name = str(row["mask"])
        if window_index in position_by_window and name in mask_position:
            wi = position_by_window[window_index]
            mi = mask_position[name]
            row["latent_global_familywise_p_value"] = float(p_values[wi, mi])
            row["latent_global_familywise_exceeds_null"] = bool(exceeds[wi, mi])
        else:
            row["latent_global_familywise_p_value"] = ""
            row["latent_global_familywise_exceeds_null"] = ""
    corrected_csv = output_prefix.with_suffix(".global_null.csv")
    arrays_path = output_prefix.with_suffix(".global_null.arrays.npz")
    summary_path = output_prefix.with_suffix(".global_null.summary.json")
    _write_rows(corrected_csv, rows, fields)
    np.savez_compressed(
        arrays_path,
        eligible_window_indices=np.asarray(eligible_indices, dtype=np.int32),
        mask_names=np.asarray(_INFERENTIAL_MASKS),
        observed_latent_scores=observed.astype(np.float32),
        null_max_latent_by_window_mask=null_latent.astype(np.float32),
        null_max_latent_across_windows_masks=maximum_null.astype(np.float32),
        familywise_p_values=p_values.astype(np.float32),
        familywise_exceeds_null=exceeds,
        observed_median_latent_by_mask=observed_median.astype(np.float32),
        null_median_latent_by_mask=null_median_by_mask.astype(np.float32),
        null_max_median_latent_across_masks=null_max_median_across_masks.astype(
            np.float32
        ),
        sustained_median_familywise_p_values=median_p_values.astype(np.float32),
    )
    summary = {
        "interpretation": "global_frozen_mask_latent_null_not_cardiac_identity_validation",
        "dataset_npz": str(args.dataset_npz),
        "longitudinal_csv": str(args.longitudinal_csv),
        "surrogate_count": count,
        "eligible_window_count": len(eligible_indices),
        "mask_names": list(_INFERENTIAL_MASKS),
        "frequency_bounds_hz": [
            float(args.frequency_min_hz),
            float(args.frequency_max_hz),
        ],
        "frequency_step_hz": float(args.frequency_step_hz),
        "familywise_scope": "maximum_across_eligible_windows_two_compact_masks_and_frequency_search",
        "latent_familywise_threshold": threshold,
        "significant_window_count_by_mask": {
            name: int(np.count_nonzero(exceeds[:, mask_position[name]]))
            for name in _INFERENTIAL_MASKS
        },
        "sustained_median_test": {
            "selection_status": "posthoc_after_reviewing_maximum_window_test",
            "scope": "exploratory_median_across_all_eligible_windows_maximum_across_two_compact_masks",
            "observed_median_latent_by_mask": {
                name: float(observed_median[mask_position[name]])
                for name in _INFERENTIAL_MASKS
            },
            "familywise_p_value_by_mask": {
                name: float(median_p_values[mask_position[name]])
                for name in _INFERENTIAL_MASKS
            },
            "null_maximum": float(np.max(null_max_median_across_masks)),
        },
        "outputs": {
            "corrected_csv": str(corrected_csv),
            "arrays_npz": str(arrays_path),
            "summary_json": str(summary_path),
            "surrogate_batch_dir": str(batch_dir),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary_json: {summary_path}")
    print(f"corrected_csv: {corrected_csv}")
    print(f"arrays_npz: {arrays_path}")
    print(f"threshold: {threshold:.6f}")
    print(f"significant_windows: {summary['significant_window_count_by_mask']}")


if __name__ == "__main__":
    main()
