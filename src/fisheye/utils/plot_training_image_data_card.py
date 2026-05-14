#!/usr/bin/env python3
"""Render plot PNGs from a training-image data-card JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Sequence

import numpy as np


METRIC_NAMES = (
    "mean_intensity_p50",
    "contrast_p50",
    "sharpness_p50",
    "clip_dark_rate_mean",
    "clip_bright_rate_mean",
    "illumination_center_edge_p50",
    "fish_bg_contrast_p50",
)

try:  # pragma: no cover - import fallback is environment-specific
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _default_plot_dir(card_path: Path) -> Path:
    return card_path.parent / f"{card_path.stem}.plots"


def _default_prefix(card_payload: Mapping[str, Any], card_path: Path) -> str:
    return _normalize_text(card_payload.get("set_id")) or card_path.stem


def _parse_histogram(hist: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    edges_raw = hist.get("bin_edges")
    counts_raw = hist.get("counts")
    if not isinstance(edges_raw, Sequence) or not isinstance(counts_raw, Sequence):
        raise ValueError("Histogram payload missing bin_edges/counts sequence.")
    edges = np.asarray([float(v) for v in edges_raw], dtype=np.float64)
    counts = np.asarray([float(v) for v in counts_raw], dtype=np.float64)
    if edges.size != counts.size + 1:
        raise ValueError("Histogram bin_edges length must be counts length + 1.")
    return edges, counts


def _plot_histogram(
    *,
    hist: Mapping[str, Any],
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    assert plt is not None
    edges, counts = _parse_histogram(hist)
    widths = np.diff(edges)
    centers = edges[:-1] + (widths / 2.0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, counts, width=widths, color="#2E6F95", edgecolor="#123146", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_values(
    *,
    values: Sequence[Any],
    title: str,
    xlabel: str,
    output_path: Path,
) -> bool:
    assert plt is not None
    arr = np.asarray([float(v) for v in values if _as_float(v) is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    bins = min(40, max(8, int(np.ceil(np.sqrt(arr.size)))))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(arr, bins=bins, color="#4A7C59", edgecolor="#264A38", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Dataset Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _expected_plot_paths(*, card_payload: Mapping[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    paths: list[Path] = []
    if isinstance(card_payload.get("intensity_histogram_aggregate"), Mapping):
        paths.append(output_dir / f"{prefix}.intensity_histogram.png")
    metric_values = card_payload.get("metric_values")
    if isinstance(metric_values, Mapping):
        for metric in METRIC_NAMES:
            values = metric_values.get(metric)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                paths.append(output_dir / f"{prefix}.{metric}.png")
    return paths


def _open_paths(paths: Sequence[Path]) -> int:
    open_errors = 0
    for path in paths:
        try:
            completed = subprocess.run(["xdg-open", str(path)], check=False)
            if int(completed.returncode) != 0:
                open_errors += 1
        except Exception:
            open_errors += 1
    return open_errors


def generate_training_image_data_card_plots(
    *,
    card_payload: Mapping[str, Any],
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    if plt is None:
        raise RuntimeError(f"matplotlib is required to generate plots: {_MATPLOTLIB_IMPORT_ERROR}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    hist = card_payload.get("intensity_histogram_aggregate")
    if isinstance(hist, Mapping):
        output_path = output_dir / f"{prefix}.intensity_histogram.png"
        _plot_histogram(
            hist=hist,
            title="Training Image Intensity Histogram",
            xlabel="Intensity",
            output_path=output_path,
        )
        generated.append(output_path)

    metric_values = card_payload.get("metric_values")
    if isinstance(metric_values, Mapping):
        for metric in METRIC_NAMES:
            values = metric_values.get(metric)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                continue
            output_path = output_dir / f"{prefix}.{metric}.png"
            if _plot_values(values=values, title=metric, xlabel=metric, output_path=output_path):
                generated.append(output_path)
    return generated


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True, help="Training-image data-card JSON path.")
    parser.add_argument("--output-dir", type=Path, help="Output plot directory.")
    parser.add_argument("--prefix", type=str, help="Output filename prefix.")
    parser.add_argument("--view", action="store_true", help="Open plot files with xdg-open.")
    parser.add_argument("--force", action="store_true", help="Regenerate plots even if present.")
    parser.add_argument("--dry-run", action="store_true", help="Plan plots but do not write PNG files.")
    args = parser.parse_args(argv)
    if args.view and args.dry_run:
        parser.error("--view cannot be combined with --dry-run.")

    card_path = Path(args.card)
    payload = json.loads(card_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        print(f"Training image data-card plotting failed: card is not a JSON object: {card_path}")
        return 1
    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_plot_dir(card_path)
    prefix = str(args.prefix).strip() if args.prefix is not None else _default_prefix(payload, card_path)
    if not prefix:
        prefix = card_path.stem

    expected = _expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix=prefix)
    existing = [path for path in expected if path.exists()]
    if args.dry_run:
        print(f"Training image data-card plots: mode=dry-run output_dir={output_dir} prefix={prefix}")
        return 0
    if args.view and not args.force and expected and len(existing) == len(expected):
        errors = _open_paths(existing)
        print(f"Training image data-card plots: mode=view-existing opened={len(existing)} output_dir={output_dir}")
        return 1 if errors else 0

    try:
        generated = generate_training_image_data_card_plots(
            card_payload=payload,
            output_dir=output_dir,
            prefix=prefix,
        )
    except Exception as exc:
        print(f"Training image data-card plotting failed: {exc}")
        return 1
    if args.view:
        errors = _open_paths(generated)
        print(f"Training image data-card plots: mode=apply+view generated={len(generated)} output_dir={output_dir}")
        return 1 if errors else 0
    print(f"Training image data-card plots: mode=apply generated={len(generated)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
