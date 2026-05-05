"""Plot cross-recording bout-kinematics metrics from Palette Parquet exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl


DEFAULT_MEASUREMENT_LEVEL = "heading_smoothed"
DEFAULT_EXPORT_ROOT = Path("/nvme1/exports/palette_analytics")

HEADING_FIELDS = (
    "net_delta_heading_deg",
    "abs_net_delta_heading_deg",
    "within_heading_path_deg",
    "within_heading_peak_to_peak_deg",
    "within_angular_speed_mean_deg_s",
    "within_angular_speed_max_deg_s",
)

FIELD_TITLES = {
    "net_delta_heading_deg": "Net heading change per bout (deg)",
    "abs_net_delta_heading_deg": "Absolute net heading change per bout (deg)",
    "within_heading_path_deg": "Within-bout heading path (deg)",
    "within_heading_peak_to_peak_deg": "Within-bout heading peak-to-peak (deg)",
    "within_angular_speed_mean_deg_s": "Mean angular speed within bout (deg/s)",
    "within_angular_speed_max_deg_s": "Max angular speed within bout (deg/s)",
}


def resolve_export_run_id(export_root: Path, export_run_id: str) -> str:
    """Resolve ``latest`` against export manifests."""

    if export_run_id != "latest":
        return export_run_id
    manifest_dir = export_root / "v1" / "manifests"
    manifests = sorted(manifest_dir.glob("export_run_id=*.json"))
    if not manifests:
        raise FileNotFoundError(f"No export manifests found under {manifest_dir}")
    return manifests[-1].name.removeprefix("export_run_id=").removesuffix(".json")


def load_bout_kinematics_metrics(
    export_root: Path,
    export_run_id: str,
    *,
    measurement_level: str = DEFAULT_MEASUREMENT_LEVEL,
) -> pl.DataFrame:
    """Load one measurement level from ``bout_kinematics_metrics`` Parquet parts."""

    table_dir = export_root / "v1" / "bout_kinematics_metrics" / f"export_run_id={export_run_id}"
    files = sorted(table_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No bout_kinematics_metrics Parquet files found under {table_dir}")
    frame = (
        pl.scan_parquet(str(table_dir / "*.parquet"), hive_partitioning=True)
        .filter(pl.col("measurement_level") == measurement_level)
        .collect()
    )
    if frame.is_empty():
        raise ValueError(
            f"No bout_kinematics_metrics rows for measurement_level={measurement_level!r} "
            f"in export_run_id={export_run_id!r}."
        )
    return frame


def _finite_values(frame: pl.DataFrame, field: str) -> np.ndarray:
    if field not in frame.columns:
        return np.asarray([], dtype=np.float64)
    values = frame[field].drop_nulls().to_numpy()
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _metric_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q05": None,
            "q25": None,
            "q75": None,
            "q95": None,
        }
    q05, q25, q75, q95 = np.nanquantile(values, [0.05, 0.25, 0.75, 0.95])
    return {
        "n": int(values.size),
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
    }


def summarize_heading_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable summary stats for heading fields."""

    metrics = {
        field: _metric_summary(_finite_values(frame, field))
        for field in HEADING_FIELDS
    }
    by_mode: dict[str, Any] = {}
    if "stimulus_mode" in frame.columns and "net_delta_heading_deg" in frame.columns:
        mode_rows = frame.select(["stimulus_mode", "net_delta_heading_deg"]).drop_nulls()
        for row in mode_rows.group_by("stimulus_mode").agg(
            pl.len().alias("n"),
            pl.col("net_delta_heading_deg").mean().alias("mean"),
            pl.col("net_delta_heading_deg").median().alias("median"),
        ).iter_rows(named=True):
            by_mode[str(row["stimulus_mode"])] = {
                "n": int(row["n"]),
                "mean": None if row["mean"] is None else float(row["mean"]),
                "median": None if row["median"] is None else float(row["median"]),
            }
    return {
        "row_count": int(frame.height),
        "metrics": metrics,
        "net_delta_by_mode": by_mode,
    }


def write_summary_files(
    summary: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "plot_summary.json"
    tsv_path = output_dir / "swim_bout_heading_summary.tsv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with tsv_path.open("w") as handle:
        handle.write("metric\tn\tmean\tmedian\tq05\tq25\tq75\tq95\n")
        for metric, stats in summary.get("metrics", {}).items():
            handle.write(
                f"{metric}\t{stats['n']}\t{_tsv_value(stats['mean'])}\t"
                f"{_tsv_value(stats['median'])}\t{_tsv_value(stats['q05'])}\t"
                f"{_tsv_value(stats['q25'])}\t{_tsv_value(stats['q75'])}\t"
                f"{_tsv_value(stats['q95'])}\n"
            )
    return json_path, tsv_path


def _tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_plots(
    frame: pl.DataFrame,
    *,
    output_dir: Path,
    export_run_id: str,
    measurement_level: str,
    dpi: int = 180,
) -> list[Path]:
    """Write lab-meeting style PNG plots for one measurement level."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        f"Parquet swim-bout heading metrics ({export_run_id}; {measurement_level}; {frame.height:,} rows)",
        fontsize=16,
        fontweight="bold",
    )
    for ax, field in zip(axes.ravel(), HEADING_FIELDS[:4]):
        values = _finite_values(frame, field)
        if values.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            continue
        bins = (
            np.linspace(-180, 180, 121)
            if field == "net_delta_heading_deg"
            else np.linspace(0, max(10.0, float(np.nanquantile(values, 0.99))), 80)
        )
        ax.hist(values, bins=bins, color="#2a9d8f", alpha=0.85, edgecolor="white", linewidth=0.35)
        if field == "net_delta_heading_deg":
            ax.axvline(0, color="black", lw=1, alpha=0.6)
        median = float(np.nanmedian(values))
        mean = float(np.nanmean(values))
        ax.axvline(median, color="#e76f51", lw=2, label=f"median {median:.1f}")
        ax.set_title(FIELD_TITLES[field])
        ax.set_xlabel("Degrees")
        ax.set_ylabel("Bout count")
        ax.text(
            0.98,
            0.95,
            f"n={values.size:,}\nmean={mean:.1f}\nmedian={median:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9),
        )
        ax.legend(loc="upper left", frameon=False)
        ax.grid(alpha=0.2)
    path = output_dir / "swim_bout_heading_histograms_overall.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    if "stimulus_mode" in frame.columns and "net_delta_heading_deg" in frame.columns:
        written.append(_write_net_heading_by_mode_plot(frame, output_dir, dpi=dpi))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Parquet swim-bout angular speed metrics", fontsize=16, fontweight="bold")
    for ax, field in zip(axes, HEADING_FIELDS[4:]):
        values = _finite_values(frame, field)
        if values.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            continue
        upper = max(10.0, float(np.nanquantile(values, 0.99)))
        ax.hist(values, bins=np.linspace(0, upper, 80), color="#264653", alpha=0.85, edgecolor="white", linewidth=0.35)
        median = float(np.nanmedian(values))
        ax.axvline(median, color="#f4a261", lw=2, label=f"median {median:.1f}")
        ax.set_title(FIELD_TITLES[field])
        ax.set_xlabel("deg/s")
        ax.set_ylabel("Bout count")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.2)
    path = output_dir / "swim_bout_angular_speed_histograms.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)
    return written


def _write_net_heading_by_mode_plot(frame: pl.DataFrame, output_dir: Path, *, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    mode_frame = frame.select(["stimulus_mode", "net_delta_heading_deg"]).drop_nulls()
    mode_counts = (
        mode_frame.group_by("stimulus_mode")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= 20)
        .sort("n", descending=True)
    )
    mode_order = [str(item) for item in mode_counts["stimulus_mode"].to_list()]
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(mode_order))))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Parquet net heading change per swim bout by stimulus mode", fontsize=16, fontweight="bold")
    box_data: list[np.ndarray] = []
    for color, mode in zip(colors, mode_order):
        values = mode_frame.filter(pl.col("stimulus_mode") == mode)["net_delta_heading_deg"].to_numpy()
        box_data.append(values)
        axes[0].hist(
            values,
            bins=np.linspace(-180, 180, 121),
            histtype="step",
            linewidth=1.8,
            density=True,
            label=f"{mode} (n={values.size:,})",
            color=color,
        )
    axes[0].axvline(0, color="black", lw=1, alpha=0.6)
    axes[0].set_xlabel("Net delta heading (deg)")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.2)
    if mode_order:
        axes[0].legend(fontsize=8, frameon=False)
        box = axes[1].boxplot(
            box_data,
            tick_labels=mode_order,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black"),
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
    axes[1].axhline(0, color="black", lw=1, alpha=0.6)
    axes[1].set_ylabel("Net delta heading (deg)")
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].grid(axis="y", alpha=0.2)
    path = output_dir / "swim_bout_net_heading_by_stimulus.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_export(
    *,
    export_root: Path,
    export_run_id: str,
    output_dir: Path,
    measurement_level: str = DEFAULT_MEASUREMENT_LEVEL,
    dpi: int = 180,
) -> dict[str, Any]:
    resolved_run_id = resolve_export_run_id(export_root, export_run_id)
    frame = load_bout_kinematics_metrics(
        export_root,
        resolved_run_id,
        measurement_level=measurement_level,
    )
    summary = summarize_heading_metrics(frame)
    summary.update({
        "export_run_id": resolved_run_id,
        "source": "parquet:bout_kinematics_metrics",
        "measurement_level": measurement_level,
    })
    plots = write_plots(
        frame,
        output_dir=output_dir,
        export_run_id=resolved_run_id,
        measurement_level=measurement_level,
        dpi=dpi,
    )
    json_path, tsv_path = write_summary_files(summary, output_dir)
    summary["output_files"] = [str(path) for path in [*plots, json_path, tsv_path]]
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot cross-recording bout-kinematics metrics from Palette Parquet exports.",
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--export-run-id", default="latest", help="Export run id or 'latest'.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measurement-level", default=DEFAULT_MEASUREMENT_LEVEL)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = plot_export(
        export_root=args.export_root.expanduser().resolve(),
        export_run_id=args.export_run_id,
        output_dir=args.output_dir.expanduser().resolve(),
        measurement_level=args.measurement_level,
        dpi=int(args.dpi),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
