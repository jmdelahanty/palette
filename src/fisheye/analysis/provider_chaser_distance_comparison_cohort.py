"""Run a bounded multi-recording provider chaser-distance comparison.

The cohort is selected deterministically from an immutable cohort input
document: the earliest and latest eligible recording for every arena.  Each
recording is evaluated with the exact dual-provider canary, and aggregation is
performed over recording-level metrics so camera frames are not mistaken for
independent animals.

This command never writes an analysis Zarr, registry, selector, or provider
promotion.  ``--apply`` only publishes selector-ineligible operational
evidence below an explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fisheye.analysis.provider_chaser_distance_comparison import (  # noqa: E402
    build_comparison,
    publish_operational_canary,
)
from fisheye.shared.json_safety import json_attr_safe  # noqa: E402
from fisheye.shared.system_metadata import get_git_info  # noqa: E402
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256  # noqa: E402


SCHEMA_ID = "palette.provider_chaser_distance_comparison_cohort_canary"
SCHEMA_VERSION = 1
SELECTION_POLICY_ID = "earliest_latest_eligible_recording_per_arena_v1"
AGGREGATION_POLICY_ID = "recording_is_analysis_unit_no_frame_pooling_v1"
_ARENA_RE = re.compile(r"(?:^|_)arena_(?P<arena>[0-9]+)(?:_|$)")
_CAMERA_RE = re.compile(r"/source_camera/(?P<camera>[^/]+)/")
COHORT_OUTPUT_FILES = (
    "cohort_report.json",
    "recording_summary.csv",
    "epoch_role_summary.csv",
    "provider_coverage.png",
    "provider_disagreement.png",
)


class ProviderChaserDistanceCohortError(ValueError):
    """Raised when a bounded provider-comparison cohort is invalid."""


def _fail(message: str) -> None:
    raise ProviderChaserDistanceCohortError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be one object.")
    return _plain(value)


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{label} must be one nonempty canonical string.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arena(recording_id: str) -> str:
    match = _ARENA_RE.search(recording_id)
    if match is None:
        _fail(f"Recording identity has no canonical arena token: {recording_id!r}.")
    return f"arena_{int(match.group('arena'))}"


def _eligible_entries(
    cohort_inputs: Path,
    *,
    first_provider_key: str,
    second_provider_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cohort_inputs = Path(cohort_inputs).expanduser().resolve()
    if not cohort_inputs.is_file():
        raise FileNotFoundError(
            f"Cohort input document does not exist: {cohort_inputs}"
        )
    raw_bytes = cohort_inputs.read_bytes()
    document = _mapping(json.loads(raw_bytes), label="cohort input document")
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        _fail("Cohort input document has no entries.")
    first_provider_key = _text(first_provider_key, label="first provider key")
    second_provider_key = _text(second_provider_key, label="second provider key")
    if first_provider_key == second_provider_key:
        _fail("Provider keys must be distinct.")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        entry = _mapping(raw_entry, label="cohort entry")
        recording_id = _text(entry.get("recording_id"), label="recording identity")
        if recording_id in seen:
            _fail(f"Duplicate recording identity in cohort input: {recording_id!r}.")
        seen.add(recording_id)
        analysis_zarr = Path(
            _text(entry.get("analysis_zarr"), label="analysis Zarr path")
        ).expanduser()
        providers = _mapping(entry.get("providers"), label="provider declarations")
        for provider_key in (first_provider_key, second_provider_key):
            provider = _mapping(
                providers.get(provider_key),
                label=f"provider declaration {provider_key!r}",
            )
            _text(
                provider.get("sealed_run_name"),
                label=f"sealed run for {provider_key!r}",
            )
        entries.append(
            {
                **entry,
                "recording_id": recording_id,
                "analysis_zarr": str(analysis_zarr),
                "arena": _arena(recording_id),
            }
        )
    return entries, {
        "path": str(cohort_inputs),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "schema_id": document.get("schema_id"),
        "schema_version": document.get("schema_version"),
        "declared_recording_count": document.get("recording_count"),
        "observed_recording_count": len(entries),
        "plan_digest": document.get("plan_digest"),
    }


def select_arena_extremes(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the earliest and latest eligible recording in every arena."""

    by_arena: dict[str, list[dict[str, Any]]] = {}
    for raw_entry in entries:
        entry = _mapping(raw_entry, label="eligible cohort entry")
        recording_id = _text(entry.get("recording_id"), label="recording identity")
        arena = _text(entry.get("arena"), label="arena identity")
        if arena != _arena(recording_id):
            _fail(f"Arena identity disagrees with recording {recording_id!r}.")
        by_arena.setdefault(arena, []).append(entry)
    if not by_arena:
        _fail("No eligible arenas are available.")

    selected: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for arena in sorted(by_arena, key=lambda value: int(value.split("_")[-1])):
        candidates = sorted(by_arena[arena], key=lambda item: item["recording_id"])
        if len(candidates) < 2:
            _fail(f"Arena {arena!r} has fewer than two eligible recordings.")
        first = candidates[0]
        last = candidates[-1]
        if first["recording_id"] == last["recording_id"]:
            _fail(f"Arena {arena!r} has no distinct temporal extremes.")
        selected.extend((first, last))
        strata.append(
            {
                "arena": arena,
                "eligible_recording_count": len(candidates),
                "earliest_recording_id": first["recording_id"],
                "latest_recording_id": last["recording_id"],
            }
        )
    selected.sort(key=lambda item: item["recording_id"])
    selected_ids = [entry["recording_id"] for entry in selected]
    if len(selected_ids) != len(set(selected_ids)):
        _fail("Arena-extreme selection produced duplicate recordings.")
    record = {
        "schema_id": f"{SCHEMA_ID}.selection",
        "schema_version": 1,
        "policy_id": SELECTION_POLICY_ID,
        "eligible_recording_count": len(entries),
        "selected_recording_count": len(selected),
        "strata": strata,
        "selected_recording_ids": selected_ids,
        "selection_rationale": (
            "Bound all arena/camera strata and both temporal extremes without "
            "selecting on provider agreement or scientific outcome."
        ),
    }
    record["selection_sha256"] = canonical_json_sha256(record)
    return selected, record


def _camera_serial(report: Mapping[str, Any], labels: Sequence[str]) -> str:
    observed: set[str] = set()
    for label in labels:
        provider = _mapping(report["providers"][label], label=f"provider {label!r}")
        source = _mapping(
            provider.get("source_position_provider"),
            label=f"source position provider {label!r}",
        )
        coordinate = _text(
            source.get("coordinate_authority_id"),
            label=f"coordinate authority {label!r}",
        )
        match = _CAMERA_RE.search(coordinate)
        if match is None:
            _fail(
                f"Coordinate authority has no source-camera identity: {coordinate!r}."
            )
        observed.add(match.group("camera"))
    if len(observed) != 1:
        _fail("Compared providers bind different source-camera identities.")
    return next(iter(observed))


def _finite_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _aggregate_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    first_label: str,
    second_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    recording_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for report in reports:
        recording_id = _text(
            report.get("recording_id"), label="report recording identity"
        )
        providers = _mapping(report.get("providers"), label="report providers")
        first = _mapping(providers.get(first_label), label=f"provider {first_label!r}")
        second = _mapping(
            providers.get(second_label), label=f"provider {second_label!r}"
        )
        frame_count = int(first.get("frame_count", 0))
        if frame_count <= 0 or int(second.get("frame_count", 0)) != frame_count:
            _fail(f"Provider frame counts disagree for {recording_id!r}.")
        first_fraction = float(first["valid_source_position_fraction"])
        second_fraction = float(second["valid_source_position_fraction"])
        overall = _mapping(
            report.get("overall_provider_comparison"),
            label="overall provider comparison",
        )
        comparison_rows = [
            _mapping(row, label="provider comparison row")
            for row in report.get("provider_comparison", [])
        ]
        distance_rows = [
            row
            for row in comparison_rows
            if row.get("comparison_kind") == "epoch_chaser"
        ]
        if not distance_rows:
            _fail(f"No epoch/chaser comparison rows exist for {recording_id!r}.")
        distance_p50 = [
            float(row["absolute_distance_delta_p50_mm"])
            for row in distance_rows
            if row.get("absolute_distance_delta_p50_mm") is not None
        ]
        distance_p95 = [
            float(row["absolute_distance_delta_p95_mm"])
            for row in distance_rows
            if row.get("absolute_distance_delta_p95_mm") is not None
        ]
        recording_rows.append(
            {
                "recording_id": recording_id,
                "arena": _arena(recording_id),
                "camera_serial": _camera_serial(report, (first_label, second_label)),
                "frame_count": frame_count,
                f"{first_label}_valid_position_fraction": first_fraction,
                f"{second_label}_valid_position_fraction": second_fraction,
                "second_minus_first_coverage_percentage_points": 100.0
                * (second_fraction - first_fraction),
                "common_position_frame_count": int(
                    overall["common_position_frame_count"]
                ),
                "common_position_fraction": float(
                    overall["common_position_frame_count"] / frame_count
                ),
                "position_delta_p50_px": float(overall["position_delta_p50_px"]),
                "position_delta_p95_px": float(overall["position_delta_p95_px"]),
                "position_delta_p99_px": float(overall["position_delta_p99_px"]),
                "nearest_chaser_agreement_fraction": float(
                    overall["nearest_chaser_agreement_fraction"]
                ),
                "maximum_epoch_role_distance_delta_p50_mm": max(distance_p50),
                "maximum_epoch_role_distance_delta_p95_mm": max(distance_p95),
                f"{first_label}_manifest_sha256": first["manifest_sha256"],
                f"{second_label}_manifest_sha256": second["manifest_sha256"],
            }
        )

        metrics = [
            _mapping(row, label="per-epoch metric")
            for row in report.get("per_epoch_metrics", [])
        ]
        metric_index = {
            (
                str(row["provider_label"]),
                int(row["epoch_window_id"]),
                str(row["behavior_role"]),
            ): row
            for row in metrics
        }
        for comparison in distance_rows:
            window_id = int(comparison["epoch_window_id"])
            role = str(comparison["behavior_role"])
            first_metric = metric_index.get((first_label, window_id, role))
            second_metric = metric_index.get((second_label, window_id, role))
            if first_metric is None or second_metric is None:
                _fail(
                    f"Epoch/provider metric binding is incomplete for {recording_id!r}."
                )
            epoch_rows.append(
                {
                    "recording_id": recording_id,
                    "arena": _arena(recording_id),
                    "camera_serial": recording_rows[-1]["camera_serial"],
                    "epoch_window_id": window_id,
                    "epoch_label": str(comparison["epoch_label"]),
                    "behavior_role": role,
                    f"{first_label}_valid_distance_fraction": first_metric[
                        "valid_distance_fraction"
                    ],
                    f"{second_label}_valid_distance_fraction": second_metric[
                        "valid_distance_fraction"
                    ],
                    f"{first_label}_distance_p50_mm": first_metric["distance_p50_mm"],
                    f"{second_label}_distance_p50_mm": second_metric["distance_p50_mm"],
                    "common_distance_frame_count": comparison[
                        "common_distance_frame_count"
                    ],
                    "signed_distance_delta_mean_mm": comparison[
                        "signed_distance_delta_mean_mm"
                    ],
                    "absolute_distance_delta_p50_mm": comparison[
                        "absolute_distance_delta_p50_mm"
                    ],
                    "absolute_distance_delta_p95_mm": comparison[
                        "absolute_distance_delta_p95_mm"
                    ],
                }
            )

    summary_fields = (
        f"{first_label}_valid_position_fraction",
        f"{second_label}_valid_position_fraction",
        "second_minus_first_coverage_percentage_points",
        "common_position_fraction",
        "position_delta_p50_px",
        "position_delta_p95_px",
        "position_delta_p99_px",
        "nearest_chaser_agreement_fraction",
        "maximum_epoch_role_distance_delta_p50_mm",
        "maximum_epoch_role_distance_delta_p95_mm",
    )
    summary = {
        field: _finite_summary([float(row[field]) for row in recording_rows])
        for field in summary_fields
    }
    return recording_rows, epoch_rows, summary


def build_cohort_comparison(
    cohort_inputs: Path,
    *,
    first_provider_key: str,
    second_provider_key: str,
    first_label: str,
    second_label: str,
    cdf_thresholds_mm: Sequence[float],
    workers: int,
) -> dict[str, Any]:
    """Build a read-only arena-stratified cohort comparison."""

    if workers < 1 or workers > 8:
        _fail("workers must be between 1 and 8.")
    entries, source_document = _eligible_entries(
        cohort_inputs,
        first_provider_key=first_provider_key,
        second_provider_key=second_provider_key,
    )
    selected, selection = select_arena_extremes(entries)

    def build(entry: Mapping[str, Any]) -> dict[str, Any]:
        providers = _mapping(entry["providers"], label="selected provider declarations")
        first_provider = _mapping(
            providers[first_provider_key],
            label=f"provider {first_provider_key!r}",
        )
        second_provider = _mapping(
            providers[second_provider_key],
            label=f"provider {second_provider_key!r}",
        )
        return build_comparison(
            Path(entry["analysis_zarr"]),
            first_run=_text(
                first_provider["sealed_run_name"],
                label="first sealed provider run",
            ),
            second_run=_text(
                second_provider["sealed_run_name"],
                label="second sealed provider run",
            ),
            first_label=first_label,
            second_label=second_label,
            cdf_thresholds_mm=cdf_thresholds_mm,
        )

    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
        reports = list(executor.map(build, selected))
    observed_ids = [str(report["recording_id"]) for report in reports]
    if observed_ids != selection["selected_recording_ids"]:
        _fail("Comparison results do not preserve the deterministic selected order.")
    recording_rows, epoch_rows, summary = _aggregate_reports(
        reports,
        first_label=first_label,
        second_label=second_label,
    )
    camera_by_arena: dict[str, set[str]] = {}
    for row in recording_rows:
        camera_by_arena.setdefault(str(row["arena"]), set()).add(
            str(row["camera_serial"])
        )
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "computed_read_only",
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "provider_default_promoted": False,
        "source_cohort_document": source_document,
        "selection": selection,
        "provider_comparison": {
            "first_provider_key": first_provider_key,
            "second_provider_key": second_provider_key,
            "first_label": first_label,
            "second_label": second_label,
        },
        "aggregation_policy": {
            "policy_id": AGGREGATION_POLICY_ID,
            "analysis_unit": "recording",
            "frame_pooling_across_recordings": False,
            "animal_pooling": False,
            "inferential_statistics": False,
        },
        "camera_authority_by_arena": {
            arena: sorted(cameras) for arena, cameras in sorted(camera_by_arena.items())
        },
        "recording_summary": recording_rows,
        "epoch_role_summary": epoch_rows,
        "recording_level_distribution_summary": summary,
        "temporal_caveat": (
            "Controller-input-provenance proxy only; state presentation time and "
            "camera exposure alignment are unavailable."
        ),
        "decision_status": "evidence_only_no_provider_promotion",
        "_per_recording_reports": reports,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        _fail(f"Refusing to write empty table {path.name!r}.")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        _fail(f"Table {path.name!r} has inconsistent columns.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _short_label(recording_id: str) -> str:
    prefix, arena = recording_id.split("_arena_", 1)
    return f"{prefix[5:16]}\nA{arena.split('_', 1)[0]}"


def _plot_coverage(report: Mapping[str, Any], path: Path) -> None:
    rows = report["recording_summary"]
    provider = report["provider_comparison"]
    first = str(provider["first_label"])
    second = str(provider["second_label"])
    labels = [_short_label(str(row["recording_id"])) for row in rows]
    x = np.arange(len(rows))
    first_coverage = [
        100.0 * float(row[f"{first}_valid_position_fraction"]) for row in rows
    ]
    second_coverage = [
        100.0 * float(row[f"{second}_valid_position_fraction"]) for row in rows
    ]
    coverage_floor = max(
        0.0,
        5.0 * np.floor((min(first_coverage + second_coverage) - 2.0) / 5.0),
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].bar(
        x - 0.2,
        first_coverage,
        width=0.4,
        label=first,
    )
    axes[0].bar(
        x + 0.2,
        second_coverage,
        width=0.4,
        label=second,
    )
    axes[0].set_ylabel("valid position frames (%)")
    axes[0].set_ylim(coverage_floor, 100.2)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        x,
        [100.0 * float(row["nearest_chaser_agreement_fraction"]) for row in rows],
    )
    axes[1].set_ylabel("nearest-chaser agreement (%)")
    axes[1].set_ylim(98, 100.05)
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Arena-stratified provider coverage canary")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_disagreement(report: Mapping[str, Any], path: Path) -> None:
    rows = report["recording_summary"]
    labels = [_short_label(str(row["recording_id"])) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].bar(
        x - 0.2,
        [float(row["position_delta_p50_px"]) for row in rows],
        width=0.4,
        label="p50",
    )
    axes[0].bar(
        x + 0.2,
        [float(row["position_delta_p95_px"]) for row in rows],
        width=0.4,
        label="p95",
    )
    axes[0].set_ylabel("provider position separation (px)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        x - 0.2,
        [float(row["maximum_epoch_role_distance_delta_p50_mm"]) for row in rows],
        width=0.4,
        label="maximum epoch/role p50",
    )
    axes[1].bar(
        x + 0.2,
        [float(row["maximum_epoch_role_distance_delta_p95_mm"]) for row in rows],
        width=0.4,
        label="maximum epoch/role p95",
    )
    axes[1].set_ylabel("absolute chaser-distance difference (mm)")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Detection versus keypoint provider disagreement")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def publish_cohort_canary(
    report: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Atomically publish cohort evidence outside all canonical Zarrs."""

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace existing cohort output: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        completed = _plain(report)
        reports = completed.pop("_per_recording_reports")
        references: list[dict[str, Any]] = []
        for recording_report in reports:
            recording_id = str(recording_report["recording_id"])
            result = publish_operational_canary(
                recording_report,
                output_dir=temporary / "recordings" / recording_id,
            )
            references.append(
                {
                    "recording_id": recording_id,
                    "relative_path": f"recordings/{recording_id}",
                    "artifact_manifest_sha256": result["artifact_manifest"][
                        "manifest_sha256"
                    ],
                }
            )
        completed.update(
            {
                "status": "complete_selector_ineligible_operational_canary",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "software": get_git_info(),
                "per_recording_artifacts": references,
            }
        )
        (temporary / "cohort_report.json").write_text(
            json.dumps(json_attr_safe(completed), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(
            temporary / "recording_summary.csv",
            completed["recording_summary"],
        )
        _write_csv(
            temporary / "epoch_role_summary.csv",
            completed["epoch_role_summary"],
        )
        _plot_coverage(completed, temporary / "provider_coverage.png")
        _plot_disagreement(completed, temporary / "provider_disagreement.png")
        for name in COHORT_OUTPUT_FILES:
            path = temporary / name
            if not path.is_file() or path.stat().st_size <= 0:
                _fail(f"Expected cohort artifact {name!r} is absent or empty.")
        artifacts = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            relative = path.relative_to(temporary).as_posix()
            artifacts.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        manifest = {
            "schema_id": f"{SCHEMA_ID}.artifact_manifest",
            "schema_version": 1,
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return {
            "schema_id": f"{SCHEMA_ID}.publication_result",
            "schema_version": 1,
            "status": "published_selector_ineligible_operational_canary",
            "output_dir": str(output_dir),
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "artifact_count": manifest["artifact_count"],
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated numbers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_inputs", type=Path)
    parser.add_argument(
        "--first-provider-key",
        default="detection_bbox_centroid",
    )
    parser.add_argument("--second-provider-key", default="keypoint_triad")
    parser.add_argument("--first-label", default="detection")
    parser.add_argument("--second-label", default="keypoint")
    parser.add_argument(
        "--cdf-thresholds-mm",
        type=_parse_thresholds,
        default=tuple(float(value) for value in range(0, 101, 2)),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_cohort_comparison(
        args.cohort_inputs,
        first_provider_key=args.first_provider_key,
        second_provider_key=args.second_provider_key,
        first_label=args.first_label,
        second_label=args.second_label,
        cdf_thresholds_mm=args.cdf_thresholds_mm,
        workers=args.workers,
    )
    if args.apply:
        if args.output_dir is None:
            raise SystemExit("--apply requires --output-dir")
        payload = publish_cohort_canary(report, output_dir=args.output_dir)
    else:
        payload = {
            "schema_id": f"{SCHEMA_ID}.plan",
            "schema_version": 1,
            "status": "planned_no_writes",
            "output_dir": None if args.output_dir is None else str(args.output_dir),
            "selection": report["selection"],
            "camera_authority_by_arena": report["camera_authority_by_arena"],
            "recording_level_distribution_summary": report[
                "recording_level_distribution_summary"
            ],
            "planned_top_level_artifacts": [
                *COHORT_OUTPUT_FILES,
                "artifact_manifest.json",
            ],
            "per_recording_artifact_sets": len(report["recording_summary"]),
            "writes": False,
        }
    print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AGGREGATION_POLICY_ID",
    "ProviderChaserDistanceCohortError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SELECTION_POLICY_ID",
    "build_cohort_comparison",
    "publish_cohort_canary",
    "select_arena_extremes",
]
