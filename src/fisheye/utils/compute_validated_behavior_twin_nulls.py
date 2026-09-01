"""Compute rotated virtual-twin null summaries from a validated-behavior export.

Reads one exact validated-behavior export generation (manifest-selected, no
globbing) and derives, per recording x provider_role x epoch x chaser x
rotation in {0, 60, 120, 180, 240, 300} degrees, the near-field distance
summaries against a chaser rotated about the arena center.  Rotation 0 is the
observed chaser and must reproduce ``radial_near_field_summary`` within
tolerance; the run refuses to write outputs otherwise unless
``--allow-parity-failure`` is passed.

The export publishes no explicit arena-center coordinates, so the center is
recovered per recording x provider from the summary's own
``fish_arena_radius_mean_mm`` constraints (see
``fisheye.analytics_exports.validated_behavior_twin_nulls``); the recovery
residual is persisted per row and gated.

Outputs under ``--output-dir``: ``twin_null_summaries.parquet``,
``twin_excess.parquet``, ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.analytics_exports.validated_behavior_twin_nulls import (
    POLICY_PARITY,
    ROTATION_DEGREES,
    TwinChaserTrack,
    TwinEpochWindow,
    TwinNullError,
    compute_twin_excess,
    compute_twin_rows_for_provider,
    fit_arena_center_mm,
)

RELATIVE_SAMPLE_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "provider_role",
    "acquisition_frame_id",
    "timestamp_ns_session",
    "timestamp_valid",
    "fish_position_x_px",
    "fish_position_y_px",
    "chaser_position_x_px",
    "chaser_position_y_px",
    "chaser_identity_code",
    "chaser_identity",
    "behavior_role",
    "chaser_behavior_role_valid",
    "selection_member",
    "chaser_occurrence_member",
    "relative_distance_px",
    "relative_distance_mm",
    "relative_physical_valid",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "provider_role",
    "epoch_role",
    "epoch_window_id",
    "chaser_identity_code",
    "chaser_identity",
    "behavior_role",
    "valid_distance_frame_count",
    "near_zone_frame_count",
    "near_zone_fraction_valid",
    "near_zone_entry_count",
    "distance_mean_mm",
    "distance_p05_mm",
    "distance_p25_mm",
    "distance_p50_mm",
    "distance_p75_mm",
    "distance_p95_mm",
    "fish_arena_radius_mean_mm",
    "fish_arena_radius_p50_mm",
    "fish_wall_distance_mean_mm",
    "near_zone_radius_mm",
    "near_entry_radius_mm",
    "near_exit_radius_mm",
    "radial_policy_sha256",
    "arena_authority_sha256",
)

PARITY_METRICS: tuple[str, ...] = (
    "valid_distance_frame_count",
    "near_zone_frame_count",
    "near_zone_fraction_valid",
    "near_zone_entry_count",
    "distance_mean_mm",
    "distance_p05_mm",
    "distance_p25_mm",
    "distance_p50_mm",
    "distance_p75_mm",
    "distance_p95_mm",
)

CENTER_FIT_TOLERANCE_MM = 0.01
SCALE_SPREAD_TOLERANCE = 1e-5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recording_scale_mm_per_px(frame_df: pl.DataFrame) -> float:
    """Recover the constant mm/px scale from paired relative distances."""

    paired = frame_df.filter(
        pl.col("relative_physical_valid") & (pl.col("relative_distance_px") > 1.0)
    )
    if paired.height == 0:
        raise TwinNullError("No paired px/mm relative distances to derive scale.")
    ratio = (
        paired["relative_distance_mm"].cast(pl.Float64)
        / paired["relative_distance_px"].cast(pl.Float64)
    ).to_numpy()
    spread = float(ratio.max() - ratio.min())
    if spread > SCALE_SPREAD_TOLERANCE:
        raise TwinNullError(
            f"mm/px ratio is not constant (spread {spread:.3e}); refusing to "
            "treat the physical scale as one scalar."
        )
    return float(np.median(ratio))


def _epoch_windows(epochs_df: pl.DataFrame) -> list[TwinEpochWindow]:
    windows = [
        TwinEpochWindow(
            epoch_window_id=int(row["epoch_window_id"]),
            epoch_role=str(row["analysis_role"]),
            start_frame=int(row["start_frame"]),
            end_frame_exclusive=int(row["end_frame_exclusive"]),
        )
        for row in epochs_df.iter_rows(named=True)
    ]
    if not windows:
        raise TwinNullError("Recording has no semantic epoch windows.")
    return windows


def _provider_frame_axis(
    provider_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[TwinChaserTrack], float]:
    """Pivot frame-major chaser-minor rows onto one shared frame axis."""

    scale = _recording_scale_mm_per_px(provider_df)
    chaser_codes = sorted(provider_df["chaser_identity_code"].unique().to_list())
    per_chaser: dict[int, pl.DataFrame] = {}
    frame_axis: np.ndarray | None = None
    for code in chaser_codes:
        part = provider_df.filter(pl.col("chaser_identity_code") == code).sort(
            "acquisition_frame_id"
        )
        frames = part["acquisition_frame_id"].to_numpy()
        if frames.size > 1 and np.any(np.diff(frames) <= 0):
            raise TwinNullError(
                "Duplicate or unsorted acquisition frames for one chaser."
            )
        if frame_axis is None:
            frame_axis = frames
        elif frames.shape != frame_axis.shape or np.any(frames != frame_axis):
            raise TwinNullError("Chasers disagree on the acquisition frame axis.")
        per_chaser[int(code)] = part
    if frame_axis is None:
        raise TwinNullError("Provider slice has no chaser rows.")
    first = per_chaser[int(chaser_codes[0])]
    timestamp_ns = first["timestamp_ns_session"].to_numpy().astype(np.int64)
    timestamp_valid = first["timestamp_valid"].to_numpy().astype(bool)
    fish_xy_mm = (
        np.column_stack(
            [
                first["fish_position_x_px"].to_numpy().astype(np.float64),
                first["fish_position_y_px"].to_numpy().astype(np.float64),
            ]
        )
        * scale
    )
    tracks: list[TwinChaserTrack] = []
    for code in chaser_codes:
        part = per_chaser[int(code)]
        chaser_xy_mm = (
            np.column_stack(
                [
                    part["chaser_position_x_px"].to_numpy().astype(np.float64),
                    part["chaser_position_y_px"].to_numpy().astype(np.float64),
                ]
            )
            * scale
        )
        valid = (
            part["selection_member"].to_numpy().astype(bool)
            & part["chaser_occurrence_member"].to_numpy().astype(bool)
            & part["chaser_behavior_role_valid"].to_numpy().astype(bool)
            & part["relative_physical_valid"].to_numpy().astype(bool)
        )
        tracks.append(
            TwinChaserTrack(
                chaser_identity_code=int(code),
                chaser_identity=str(part["chaser_identity"][0]),
                behavior_role=str(part["behavior_role"][0]),
                chaser_xy_mm=chaser_xy_mm,
                valid=valid,
            )
        )
    return frame_axis, timestamp_ns, timestamp_valid, fish_xy_mm, tracks, scale


def _reconstruction_deviation_mm(
    fish_xy_mm: np.ndarray, tracks: Sequence[TwinChaserTrack], provider_df: pl.DataFrame
) -> float:
    """Max |reconstructed distance - published relative_distance_mm|."""

    worst = 0.0
    for track in tracks:
        part = provider_df.filter(
            pl.col("chaser_identity_code") == track.chaser_identity_code
        ).sort("acquisition_frame_id")
        published = part["relative_distance_mm"].to_numpy().astype(np.float64)
        delta = fish_xy_mm - track.chaser_xy_mm
        recon = np.hypot(delta[:, 0], delta[:, 1])
        mask = track.valid & np.isfinite(published)
        if np.any(mask):
            worst = max(worst, float(np.max(np.abs(recon[mask] - published[mask]))))
    return worst


def _process_recording(
    dataset: ValidatedBehaviorExportDataset,
    recording_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return (summary rows, parity rows, per-recording diagnostics)."""

    frame_df = (
        dataset.table("chaser_relative_samples")
        .scan(
            columns=list(RELATIVE_SAMPLE_COLUMNS),
            predicate=pl.col("recording_id") == recording_id,
        )
        .collect()
    )
    if frame_df.height == 0:
        raise TwinNullError(f"{recording_id}: no chaser_relative_samples rows.")
    epochs_df = (
        dataset.table("semantic_epochs")
        .scan(
            columns=[
                "recording_id",
                "epoch_window_id",
                "analysis_role",
                "start_frame",
                "end_frame_exclusive",
            ],
            predicate=pl.col("recording_id") == recording_id,
        )
        .collect()
    )
    summary_df = (
        dataset.table("radial_near_field_summary")
        .scan(
            columns=list(SUMMARY_COLUMNS),
            predicate=pl.col("recording_id") == recording_id,
        )
        .collect()
    )
    if summary_df.height == 0:
        raise TwinNullError(f"{recording_id}: no radial_near_field_summary rows.")
    policy = summary_df.select(
        [
            "near_zone_radius_mm",
            "near_entry_radius_mm",
            "near_exit_radius_mm",
            "radial_policy_sha256",
            "arena_authority_sha256",
        ]
    ).unique()
    if policy.height != 1:
        raise TwinNullError(
            f"{recording_id}: near-field policy constants are not unique."
        )
    policy_row = policy.row(0, named=True)
    epochs = _epoch_windows(epochs_df)
    epoch_by_id = {window.epoch_window_id: window for window in epochs}

    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"recording_id": recording_id, "providers": {}}
    for provider_role in sorted(frame_df["provider_role"].unique().to_list()):
        provider_df = frame_df.filter(pl.col("provider_role") == provider_role)
        (
            frame_axis,
            timestamp_ns,
            timestamp_valid,
            fish_xy_mm,
            tracks,
            scale,
        ) = _provider_frame_axis(provider_df)
        provider_summary = summary_df.filter(pl.col("provider_role") == provider_role)
        if provider_summary.height == 0:
            raise TwinNullError(
                f"{recording_id}/{provider_role}: no published summary rows."
            )

        constraints: list[tuple[np.ndarray, float]] = []
        arena_radius_terms: list[float] = []
        track_by_code = {track.chaser_identity_code: track for track in tracks}
        for summary_row in provider_summary.iter_rows(named=True):
            window = epoch_by_id.get(int(summary_row["epoch_window_id"]))
            track = track_by_code.get(int(summary_row["chaser_identity_code"]))
            if window is None or track is None:
                raise TwinNullError(
                    f"{recording_id}/{provider_role}: summary row references an "
                    "unknown epoch window or chaser."
                )
            in_epoch = (frame_axis >= window.start_frame) & (
                frame_axis < window.end_frame_exclusive
            )
            mask = track.valid & in_epoch
            target = summary_row["fish_arena_radius_mean_mm"]
            if target is not None and math.isfinite(float(target)):
                constraints.append((mask, float(target)))
            radius_mean = summary_row["fish_arena_radius_mean_mm"]
            wall_mean = summary_row["fish_wall_distance_mean_mm"]
            if (
                radius_mean is not None
                and wall_mean is not None
                and math.isfinite(float(radius_mean))
                and math.isfinite(float(wall_mean))
            ):
                arena_radius_terms.append(float(radius_mean) + float(wall_mean))
        if not arena_radius_terms:
            raise TwinNullError(
                f"{recording_id}/{provider_role}: cannot recover the arena radius."
            )
        arena_radius_mm = float(np.median(arena_radius_terms))
        if float(np.max(arena_radius_terms) - np.min(arena_radius_terms)) > 1e-3:
            raise TwinNullError(
                f"{recording_id}/{provider_role}: arena-radius reconstruction "
                "disagrees across summary rows."
            )
        fit = fit_arena_center_mm(fish_xy_mm, constraints)
        if fit.max_abs_residual_mm > CENTER_FIT_TOLERANCE_MM:
            raise TwinNullError(
                f"{recording_id}/{provider_role}: arena-center recovery residual "
                f"{fit.max_abs_residual_mm:.6f} mm exceeds "
                f"{CENTER_FIT_TOLERANCE_MM} mm."
            )
        recon_dev = _reconstruction_deviation_mm(fish_xy_mm, tracks, provider_df)
        center = np.asarray(fit.center_mm, dtype=np.float64)
        provider_rows = compute_twin_rows_for_provider(
            frame_id=frame_axis,
            timestamp_ns=timestamp_ns,
            timestamp_valid=timestamp_valid,
            fish_xy_mm=fish_xy_mm,
            chasers=tracks,
            epochs=epochs,
            center_mm=center,
            near_zone_radius_mm=float(policy_row["near_zone_radius_mm"]),
            near_entry_radius_mm=float(policy_row["near_entry_radius_mm"]),
            near_exit_radius_mm=float(policy_row["near_exit_radius_mm"]),
        )
        for row in provider_rows:
            row["export_run_id"] = dataset.export_run_id
            row["recording_id"] = recording_id
            row["provider_role"] = provider_role
            row["near_zone_radius_mm"] = float(policy_row["near_zone_radius_mm"])
            row["near_entry_radius_mm"] = float(policy_row["near_entry_radius_mm"])
            row["near_exit_radius_mm"] = float(policy_row["near_exit_radius_mm"])
            row["radial_policy_sha256"] = str(policy_row["radial_policy_sha256"])
            row["arena_authority_sha256"] = str(policy_row["arena_authority_sha256"])
            row["arena_center_x_mm"] = fit.center_mm[0]
            row["arena_center_y_mm"] = fit.center_mm[1]
            row["arena_radius_mm"] = arena_radius_mm
            row["center_fit_max_abs_residual_mm"] = fit.max_abs_residual_mm
            row["mm_per_px"] = scale
            row["reconstruction_max_abs_deviation_mm"] = recon_dev
            row["policy_parity"] = POLICY_PARITY
        rows.extend(provider_rows)

        recomputed_by_key = {
            (int(row["epoch_window_id"]), int(row["chaser_identity_code"])): row
            for row in provider_rows
            if int(row["rotation_deg"]) == 0
        }
        for summary_row in provider_summary.iter_rows(named=True):
            key = (
                int(summary_row["epoch_window_id"]),
                int(summary_row["chaser_identity_code"]),
            )
            recomputed = recomputed_by_key[key]
            parity: dict[str, Any] = {
                "recording_id": recording_id,
                "provider_role": provider_role,
                "epoch_window_id": key[0],
                "chaser_identity_code": key[1],
                "behavior_role": summary_row["behavior_role"],
            }
            for metric in PARITY_METRICS:
                published = summary_row[metric]
                ours = recomputed.get(metric)
                if published is None or ours is None:
                    deviation = (
                        0.0 if published is None and ours is None else math.inf
                    )
                else:
                    deviation = abs(float(ours) - float(published))
                parity[metric] = deviation
            parity_rows.append(parity)
        diagnostics["providers"][provider_role] = {
            "mm_per_px": scale,
            "arena_center_mm": list(fit.center_mm),
            "arena_radius_mm": arena_radius_mm,
            "center_fit_max_abs_residual_mm": fit.max_abs_residual_mm,
            "center_fit_constraints": fit.constraint_count,
            "reconstruction_max_abs_deviation_mm": recon_dev,
        }
    return rows, parity_rows, diagnostics


def _parity_report(parity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for metric in PARITY_METRICS:
        values = [float(row[metric]) for row in parity_rows]
        finite = [value for value in values if math.isfinite(value)]
        report[metric] = {
            "max_abs_deviation": max(finite) if finite else None,
            "rows": len(values),
            "non_finite_rows": len(values) - len(finite),
        }
    return report


def _parity_acceptable(report: dict[str, Any]) -> bool:
    fraction_tolerance = 0.01
    for metric, entry in report.items():
        if entry["non_finite_rows"]:
            return False
        worst = entry["max_abs_deviation"]
        if worst is None:
            continue
        if metric in {
            "valid_distance_frame_count",
            "near_zone_frame_count",
            "near_zone_entry_count",
        }:
            if worst > 0:
                return False
        elif metric == "near_zone_fraction_valid":
            if worst > fraction_tolerance:
                return False
        elif worst > 0.5:  # mm-scale metrics: 0.5 mm ~ well under 1% of range
            return False
    return True


def run(
    *,
    export_root: str,
    run_id: str,
    output_dir: str,
    overwrite: bool = False,
    max_recordings: int | None = None,
    recordings: Sequence[str] | None = None,
    allow_parity_failure: bool = False,
) -> Path:
    out_dir = Path(output_dir).expanduser()
    summaries_path = out_dir / "twin_null_summaries.parquet"
    excess_path = out_dir / "twin_excess.parquet"
    manifest_path = out_dir / "manifest.json"
    existing = [
        path for path in (summaries_path, excess_path, manifest_path) if path.exists()
    ]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite existing outputs without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ValidatedBehaviorExportDataset.open(
        export_root, run_id, validate=True, full_part_hashes=False
    )
    roster = (
        dataset.table("radial_near_field_summary")
        .scan(columns=["recording_id"])
        .unique()
        .collect()["recording_id"]
        .sort()
        .to_list()
    )
    if recordings is not None:
        requested = list(dict.fromkeys(recordings))
        missing = sorted(set(requested) - set(roster))
        if missing:
            raise SystemExit(f"Unknown recordings requested: {missing}")
        roster = requested
    if max_recordings is not None:
        roster = roster[: int(max_recordings)]

    all_rows: list[dict[str, Any]] = []
    all_parity: list[dict[str, Any]] = []
    per_recording: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, recording_id in enumerate(roster, start=1):
        rec_started = time.monotonic()
        rows, parity_rows, diagnostics = _process_recording(dataset, recording_id)
        diagnostics["elapsed_s"] = round(time.monotonic() - rec_started, 3)
        all_rows.extend(rows)
        all_parity.extend(parity_rows)
        per_recording.append(diagnostics)
        print(
            f"[{index}/{len(roster)}] {recording_id}: {len(rows)} rows "
            f"in {diagnostics['elapsed_s']}s",
            file=sys.stderr,
            flush=True,
        )

    parity_report = _parity_report(all_parity)
    parity_ok = _parity_acceptable(parity_report)
    if not parity_ok and not allow_parity_failure:
        print(json.dumps(parity_report, indent=2), file=sys.stderr)
        raise SystemExit(
            "Rotation-0 parity against radial_near_field_summary FAILED; "
            "refusing to publish twins (--allow-parity-failure to override)."
        )

    summaries = pl.DataFrame(all_rows, infer_schema_length=None)
    summaries.write_parquet(summaries_path)
    excess = pl.DataFrame(
        compute_twin_excess(all_rows), infer_schema_length=None
    )
    excess.write_parquet(excess_path)

    manifest = {
        "artifact": "validated_behavior_twin_null_summaries",
        "analysis_status": "exploratory",
        "policy_parity": POLICY_PARITY,
        "policy_parity_note": (
            "Distance quantiles, valid counts, and near_zone_fraction_valid "
            "follow the published summary definitions and are parity-checked "
            "at rotation 0.  The hysteresis entry count reimplements the "
            "entry/gap-censoring idea of "
            "exact_session_time_hysteresis_5mm_6mm_gap_censored_v1 but does "
            "not reproduce its dwell/censor bookkeeping, so exact policy "
            "parity is not claimed."
        ),
        "arena_center_recovery": (
            "No export table carries explicit arena-center coordinates; the "
            "center was recovered per recording x provider by solving for the "
            "point whose per-epoch mean fish distance reproduces the "
            "published fish_arena_radius_mean_mm constraints (gated at "
            f"{CENTER_FIT_TOLERANCE_MM} mm max residual)."
        ),
        "source_run_id": dataset.export_run_id,
        "dataset_cache_identity_sha256": dataset.cache_identity,
        "rotation_degrees": list(ROTATION_DEGREES),
        "recording_count": len(roster),
        "row_count": int(summaries.height),
        "rotation_zero_parity": {
            "acceptable": parity_ok,
            "per_metric": parity_report,
        },
        "per_recording": per_recording,
        "elapsed_s": round(time.monotonic() - started, 3),
        "files": {
            summaries_path.name: _sha256_file(summaries_path),
            excess_path.name: _sha256_file(excess_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {summaries_path} ({summaries.height} rows)", file=sys.stderr)
    print(f"Wrote {excess_path} ({excess.height} rows)", file=sys.stderr)
    print(f"Wrote {manifest_path}", file=sys.stderr)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument(
        "--recordings",
        nargs="*",
        default=None,
        help="Explicit recording_id subset (default: every summary recording).",
    )
    parser.add_argument(
        "--allow-parity-failure",
        action="store_true",
        help="Publish even when rotation-0 parity fails (diagnostics only).",
    )
    args = parser.parse_args(argv)
    run(
        export_root=args.export_root,
        run_id=args.run_id,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_recordings=args.max_recordings,
        recordings=args.recordings,
        allow_parity_failure=args.allow_parity_failure,
    )


if __name__ == "__main__":
    main()
