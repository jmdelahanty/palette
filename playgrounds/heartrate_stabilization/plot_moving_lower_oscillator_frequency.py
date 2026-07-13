from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _window_rows(
    path: Path,
    *,
    method: str,
    window_seconds: float,
) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["method"] == method
            and np.isclose(float(row["window_seconds"]), float(window_seconds))
            and row["status"] == "ok"
        ]


def _contiguous_runs(
    rows: list[dict[str, str]],
    *,
    tolerance_s: float = 1e-6,
) -> list[list[dict[str, str]]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row["window_start_s"]))
    runs: list[list[dict[str, str]]] = [[ordered[0]]]
    for row in ordered[1:]:
        previous = runs[-1][-1]
        if abs(float(row["window_start_s"]) - float(previous["window_stop_s"])) <= tolerance_s:
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot moving-fish lower-mask candidate oscillator frequency over time."
    )
    parser.add_argument("--window-csv", type=Path, required=True)
    parser.add_argument("--method", default="lower_raw_mean")
    parser.add_argument("--aggregate-frequency-hz", type=float, default=3.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_4s = _window_rows(args.window_csv, method=args.method, window_seconds=4.0)
    rows_8s = _window_rows(args.window_csv, method=args.method, window_seconds=8.0)
    if not rows_8s:
        raise ValueError("no scorable 8-second windows found")

    figure, axis = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    if rows_4s:
        axis.scatter(
            [float(row["window_mid_s"]) / 60.0 for row in rows_4s],
            [float(row["candidate_frequency_hz"]) for row in rows_4s],
            s=18,
            color="#92999c",
            alpha=0.42,
            linewidths=0,
            label="4 s estimate",
            zorder=2,
        )
    for run_index, run in enumerate(_contiguous_runs(rows_8s)):
        time_min = [float(row["window_mid_s"]) / 60.0 for row in run]
        frequency_hz = [float(row["candidate_frequency_hz"]) for row in run]
        axis.plot(
            time_min,
            frequency_hz,
            color="#087f8c",
            marker="o",
            markersize=4.2,
            linewidth=1.6,
            label="8 s estimate" if run_index == 0 else None,
            zorder=3,
        )
    axis.axhline(
        float(args.aggregate_frequency_hz),
        color="#25282a",
        linestyle="--",
        linewidth=1.2,
        label=f"full-recording peak ({args.aggregate_frequency_hz:.2f} Hz)",
        zorder=1,
    )
    axis.set(
        title="Freely moving fish: frozen lower-mask candidate oscillator",
        xlabel="Recording time (minutes)",
        ylabel="Candidate oscillator frequency (Hz)",
        ylim=(1.95, 4.05),
    )
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.text(
        0.01,
        0.98,
        "Unsmoothed; lines do not cross unscorable gaps",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#55595b",
    )
    secondary = axis.secondary_yaxis(
        "right",
        functions=(lambda hz: hz * 60.0, lambda cycles_per_min: cycles_per_min / 60.0),
    )
    secondary.set_ylabel("Candidate cycles/min")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
