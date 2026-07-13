from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.consensus_heart_mask import familywise_max_p_values
from fisheye.analysis.dynamic_heart_support import (
    analyze_dynamic_heart_support,
    attach_dynamic_support_nulls,
    compute_dynamic_support_null_batch,
    prepare_dynamic_support_null_context,
)
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig, analyze_heartrate


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _read_mask(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain {key!r}")
        return np.asarray(data[key], dtype=bool)


def _analysis_identity(
    dataset_path: Path,
    masks: Mapping[str, np.ndarray],
    config: HeartrateConfig,
    *,
    frequency_min_hz: float,
    frequency_max_hz: float,
    seed: int,
) -> str:
    stat = dataset_path.stat()
    payload = {
        "schema": "shared_dynamic_support_surrogates_v1",
        "dataset": str(dataset_path.resolve()),
        "dataset_size": int(stat.st_size),
        "dataset_mtime_ns": int(stat.st_mtime_ns),
        "config": asdict(config),
        "frequency_bounds_hz": [float(frequency_min_hz), float(frequency_max_hz)],
        "seed": int(seed),
        "masks": {
            name: hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()
            for name, mask in masks.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_null_batch(
    path: Path,
    *,
    identity: str,
    names: tuple[str, ...],
    expected_indices: np.ndarray,
) -> dict[str, dict[str, np.ndarray]] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["analysis_identity"].item()) != identity:
                return None
            indices = np.asarray(data["surrogate_indices"], dtype=np.int64)
            if not np.array_equal(indices, expected_indices):
                return None
            return {
                statistic: {
                    name: np.asarray(data[f"{statistic}__{name}"], dtype=np.float64)
                    for name in names
                }
                for statistic in ("support", "shared_phase", "latent")
            }
    except (KeyError, OSError, ValueError):
        return None


def _write_null_batch(
    path: Path,
    *,
    identity: str,
    indices: np.ndarray,
    scores: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.npz")
    arrays: dict[str, Any] = {
        "analysis_identity": np.asarray(identity),
        "surrogate_indices": np.asarray(indices, dtype=np.int64),
    }
    for statistic, by_name in scores.items():
        for name, values in by_name.items():
            arrays[f"{statistic}__{name}"] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_figure(
    path: Path,
    masks: Mapping[str, np.ndarray],
    summaries: Mapping[str, Any],
    *,
    frequency_bounds_hz: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    union = np.logical_or.reduce(list(masks.values()))
    yy, xx = np.nonzero(union)
    x0, x1 = max(0, int(np.min(xx)) - 2), min(union.shape[1], int(np.max(xx)) + 3)
    y0, y1 = max(0, int(np.min(yy)) - 2), min(union.shape[0], int(np.max(yy)) + 3)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for axis, (name, mask) in zip(axes[0], masks.items()):
        axis.imshow(mask[y0:y1, x0:x1], cmap="gray_r", vmin=0, vmax=1)
        axis.set_title(f"{name} ({np.count_nonzero(mask)} px)")
        axis.set_xticks([])
        axis.set_yticks([])
    names = list(masks)
    x = np.arange(len(names))
    width = 0.24
    for offset, statistic in enumerate(("support", "shared_phase", "latent")):
        values = [summaries[name][statistic]["observed"] for name in names]
        axes[1, 0].bar(x + (offset - 1) * width, values, width, label=statistic)
        corrected = [summaries[name][statistic]["familywise_p_value"] for name in names]
        axes[1, 1].bar(x + (offset - 1) * width, corrected, width, label=statistic)
    axes[1, 0].set_title("Observed scores")
    axes[1, 1].set_title("Familywise p-values")
    axes[1, 1].axhline(0.05, color="black", ls="--", lw=1)
    for axis in axes[1, :2]:
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.legend(fontsize=8)
    axes[1, 2].bar(x, [summaries[name]["control_ratio"] for name in names])
    axes[1, 2].axhline(1.1, color="black", ls="--", lw=1)
    axes[1, 2].set_title("Latent/control ratio")
    axes[1, 2].set_xticks(x, names, rotation=25, ha="right")
    axes[1, 3].bar(x, [summaries[name]["frequency_hz"] for name in names])
    axes[1, 3].set_ylim(*frequency_bounds_hz)
    axes[1, 3].set_title("Selected frequency")
    axes[1, 3].set_xticks(x, names, rotation=25, ha="right")
    fig.suptitle("Untouched-interval frozen-mask comparison")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prespecified frozen heart masks with maximum-statistic correction."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument("--frequency-min-hz", type=float, default=3.0)
    parser.add_argument("--frequency-max-hz", type=float, default=3.5)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--base-surrogate-count", type=int, default=39)
    parser.add_argument("--dynamic-surrogate-count", type=int, default=199)
    parser.add_argument("--surrogate-batch-size", type=int, default=25)
    parser.add_argument("--surrogate-workers", type=int, default=1)
    parser.add_argument("--surrogate-batch-dir", type=Path)
    parser.add_argument("--recompute-surrogate-batches", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset_npz)
    original = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    if original.shape != dataset.image_shape_hw or consensus.shape != dataset.image_shape_hw:
        raise ValueError("frozen masks must match the dataset canonical image shape")
    masks = {
        "original_38": original,
        "consensus_9": consensus,
        "intersection_8": original & consensus,
        "union_39": original | consensus,
    }
    config = HeartrateConfig(
        band_min_hz=float(args.frequency_min_hz),
        band_max_hz=float(args.frequency_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
        surrogate_count=int(args.base_surrogate_count),
        alpha=float(args.alpha),
        random_seed=int(args.seed),
    ).validated()
    base = analyze_heartrate(dataset, config)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    surrogate_seed = int(args.seed) + 1000
    batch_size = int(args.surrogate_batch_size)
    surrogate_count = int(args.dynamic_surrogate_count)
    if batch_size < 1 or surrogate_count < 0 or int(args.surrogate_workers) < 1:
        raise ValueError("surrogate counts, batch size, and worker count must be valid")
    batch_dir = (
        Path(args.surrogate_batch_dir)
        if args.surrogate_batch_dir is not None
        else output_prefix.parent / f"{output_prefix.name}.surrogate_batches"
    )
    identity = _analysis_identity(
        Path(args.dataset_npz),
        masks,
        config,
        frequency_min_hz=float(args.frequency_min_hz),
        frequency_max_hz=float(args.frequency_max_hz),
        seed=surrogate_seed,
    )
    dynamic_observed = {
        name: analyze_dynamic_heart_support(
            dataset,
            config,
            base,
            heart_mask=mask,
            mask_is_independent=True,
            frequency_min_hz=float(args.frequency_min_hz),
            frequency_max_hz=float(args.frequency_max_hz),
            surrogate_count=0,
        )
        for name, mask in masks.items()
    }
    merged: dict[str, dict[str, list[np.ndarray]]] = {
        statistic: {name: [] for name in masks}
        for statistic in ("support", "shared_phase", "latent")
    }
    null_context = prepare_dynamic_support_null_context(
        dataset,
        config,
        base,
        heart_masks=masks,
        frequency_min_hz=float(args.frequency_min_hz),
        frequency_max_hz=float(args.frequency_max_hz),
    )
    for start in range(0, surrogate_count, batch_size):
        stop = min(start + batch_size, surrogate_count)
        indices = np.arange(start, stop, dtype=np.int64)
        batch_path = batch_dir / f"{identity[:16]}.surrogates_{start:06d}_{stop:06d}.npz"
        scores = None if args.recompute_surrogate_batches else _load_null_batch(
            batch_path,
            identity=identity,
            names=tuple(masks),
            expected_indices=indices,
        )
        if scores is None:
            batch = compute_dynamic_support_null_batch(
                dataset,
                config,
                base,
                heart_masks=masks,
                surrogate_indices=indices,
                seed=surrogate_seed,
                frequency_min_hz=float(args.frequency_min_hz),
                frequency_max_hz=float(args.frequency_max_hz),
                workers=int(args.surrogate_workers),
                context=null_context,
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
            print(f"computed surrogate batch {start}:{stop}: {batch_path}")
        else:
            print(f"reused surrogate batch {start}:{stop}: {batch_path}")
        for statistic in merged:
            for name in masks:
                merged[statistic][name].append(scores[statistic][name])
    dynamic = {
        name: attach_dynamic_support_nulls(
            dynamic_observed[name],
            config,
            support_scores=np.concatenate(merged["support"][name])
            if surrogate_count
            else np.empty(0),
            shared_phase_scores=np.concatenate(merged["shared_phase"][name])
            if surrogate_count
            else np.empty(0),
            latent_scores=np.concatenate(merged["latent"][name])
            if surrogate_count
            else np.empty(0),
        )
        for name in masks
    }
    statistics = {
        "support": ("support_score", "null_max_support_scores"),
        "shared_phase": ("shared_phase_score", "null_max_shared_phase_scores"),
        "latent": ("latent_score", "null_max_latent_scores"),
    }
    corrected: dict[str, dict[str, Any]] = {name: {} for name in masks}
    maximum_nulls: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for statistic, (observed_attr, null_attr) in statistics.items():
        observed = {name: float(getattr(result, observed_attr)) for name, result in dynamic.items()}
        nulls = {name: np.asarray(getattr(result, null_attr)) for name, result in dynamic.items()}
        p_values, threshold, exceeds, maximum_null = familywise_max_p_values(
            observed,
            nulls,
            alpha=float(args.alpha),
        )
        thresholds[statistic] = threshold
        maximum_nulls[statistic] = maximum_null
        for name in masks:
            corrected[name][statistic] = {
                "observed": observed[name],
                "familywise_p_value": p_values[name],
                "familywise_exceeds_null": exceeds[name],
            }
    summaries: dict[str, Any] = {}
    for name, result in dynamic.items():
        summaries[name] = {
            **corrected[name],
            "pixel_count": int(np.count_nonzero(result.pixel_groups["heart_support"])),
            "frequency_hz": float(result.frequency_hz),
            "control_ratio": float(result.control_ratio),
            "strongest_control": result.strongest_control,
            "confirmatory_eligible": bool(result.confirmatory_eligible),
            "interpretation": str(result.interpretation),
        }
    summary_path = output_prefix.with_suffix(".mask_comparison.summary.json")
    arrays_path = output_prefix.with_suffix(".mask_comparison.arrays.npz")
    figure_path = output_prefix.with_suffix(".mask_comparison.diagnostic.png")
    summary = {
        "dataset_npz": str(args.dataset_npz),
        "base_detected": bool(base.detected),
        "base_reason": str(base.reason),
        "frequency_bounds_hz": [float(args.frequency_min_hz), float(args.frequency_max_hz)],
        "dynamic_surrogate_count": int(args.dynamic_surrogate_count),
        "surrogate_batch_size": int(args.surrogate_batch_size),
        "surrogate_workers": int(args.surrogate_workers),
        "surrogate_rng_contract": "seed_and_global_surrogate_index_v1",
        "surrogate_execution": "shared_nuisance_and_frequency_coefficients_across_masks",
        "surrogate_analysis_identity": identity,
        "surrogate_batch_dir": str(batch_dir),
        "familywise_scope": "maximum_across_four_masks_and_each_masks_frequency_search",
        "thresholds": thresholds,
        "masks": summaries,
        "interpretation": "consensus_9_is_exploratory_challenger; original_38_is_prior_frozen_benchmark",
    }
    summary_path.write_text(json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        arrays_path,
        original_38_mask=original.astype(np.uint8),
        consensus_9_mask=consensus.astype(np.uint8),
        intersection_8_mask=(original & consensus).astype(np.uint8),
        union_39_mask=(original | consensus).astype(np.uint8),
        null_max_support_across_masks=maximum_nulls["support"].astype(np.float32),
        null_max_shared_phase_across_masks=maximum_nulls["shared_phase"].astype(np.float32),
        null_max_latent_across_masks=maximum_nulls["latent"].astype(np.float32),
        **{
            f"{name}_{suffix}": np.asarray(values, dtype=np.float32)
            for name, result in dynamic.items()
            for suffix, values in (
                ("frequency_grid_hz", result.frequency_grid_hz),
                ("frequency_support_scores", result.frequency_support_scores),
                (
                    "frequency_shared_phase_scores",
                    result.frequency_shared_phase_scores,
                ),
                ("frequency_latent_scores", result.frequency_latent_scores),
            )
        },
    )
    _write_figure(
        figure_path,
        masks,
        summaries,
        frequency_bounds_hz=(
            float(args.frequency_min_hz),
            float(args.frequency_max_hz),
        ),
    )
    for name, values in summaries.items():
        print(
            f"{name}: pixels={values['pixel_count']} frequency={values['frequency_hz']:.3f} "
            f"latent_p_fwer={values['latent']['familywise_p_value']:.6f} "
            f"control_ratio={values['control_ratio']:.3f}"
        )
    print(f"summary_json: {summary_path}")
    print(f"arrays_npz: {arrays_path}")
    print(f"diagnostic_png: {figure_path}")


if __name__ == "__main__":
    main()
