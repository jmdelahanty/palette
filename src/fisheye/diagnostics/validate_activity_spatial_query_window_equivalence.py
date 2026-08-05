"""Prove one bounded activity export equals its source-backed frame projection."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import zarr

from fisheye.analytics_exports import activity_spatial_time_bins as activity
from fisheye.analytics_exports.contracts import ACTIVITY_SPATIAL_TIME_BINS_TABLE
from fisheye.analytics_exports.publication import (
    export_manifest_path,
    manifest_selected_part_files_from_payload,
    sha256_file,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

EVIDENCE_SCHEMA_ID = "palette.activity_spatial_query.frame_window_equivalence"
EVIDENCE_SCHEMA_VERSION = 1


def _read_manifest(export_root: Path, export_run_id: str) -> tuple[Path, dict[str, Any]]:
    path = export_manifest_path(export_root, export_run_id).resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Export manifest is missing or unsafe: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {raw}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"Export manifest is not one JSON object: {path}")
    return path, value


def _require_benchmark_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if not any("benchmark" in component.lower() for component in output.parts):
        raise ValueError("Equivalence evidence must be benchmark-namespaced.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace immutable evidence: {output}")
    return output


def _write_strict_json(path: Path, value: Mapping[str, Any]) -> None:
    json.dumps(value, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary evidence path exists: {temporary}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decoded_receipt(
    *,
    export_root: Path,
    manifest: Mapping[str, Any],
    frame_start: int | None = None,
    frame_stop_exclusive: int | None = None,
) -> dict[str, Any]:
    parts = manifest_selected_part_files_from_payload(
        export_root,
        manifest,
        ACTIVITY_SPATIAL_TIME_BINS_TABLE,
        allow_legacy_layout=False,
    )
    dataset = ds.dataset([str(path) for path in parts], format="parquet")
    expression = None
    if frame_start is not None or frame_stop_exclusive is not None:
        if frame_start is None or frame_stop_exclusive is None:
            raise ValueError("Interior frame filter is incomplete.")
        start = ds.field("start_acquisition_frame_index")
        stop = ds.field("end_acquisition_frame_index_exclusive")
        expression = (start >= frame_start) & (stop <= frame_stop_exclusive)
    hasher = activity._DecodedPayloadHasher()
    for batch in dataset.scanner(filter=expression, batch_size=65_536).to_batches():
        hasher.update(batch.to_pydict())
    return hasher.finish()


def _source_array(track_group: Any, path: str, start: int, stop: int) -> np.ndarray:
    node = track_group
    for component in path.split("/"):
        node = node[component]
    values = np.asarray(node[start:stop])
    if values.shape[0] != stop - start:
        raise ValueError(f"Source array {path!r} changed while recomputing rows.")
    return values


def _source_recomputed_receipt(
    *,
    source_binding: Mapping[str, Any],
    binning: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    zarr_path = Path(str(source_binding["zarr_path"])).expanduser().resolve()
    track_binding = source_binding["track_source_binding"]
    bout_bindings = source_binding["swim_bout_runs_by_track"]
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)
    bound = activity.bind_activity_spatial_sources(
        root,
        zarr_path=zarr_path,
        recording_id=str(source_binding["recording_id"]),
        track_kinematics_run=str(track_binding["run_name"]),
        track_scope=str(track_binding["scope"]),
        swim_bout_runs_by_track={
            int(track_id): str(value["run_name"])
            for track_id, value in bout_bindings.items()
        },
    )
    # NumPy dtype descriptors contain tuples in memory and strict-JSON arrays
    # after persistence. Compare their canonical JSON forms, not Python
    # container identity.
    if canonical_json_sha256(bound.binding) != canonical_json_sha256(source_binding):
        raise ValueError("Persisted source binding differs from the live authority.")

    selected_start = int(binning["source_frame_start"])
    selected_stop = int(binning["source_frame_stop_exclusive"])
    rows: list[dict[str, Any]] = []
    edge_keys: list[dict[str, int]] = []
    for track_record in track_binding["tracks"]:
        track_id = int(track_record["track_id"])
        bout_source = bound.bout_sources[track_id]
        frame_axis = np.asarray(bout_source.frame_axis, dtype=np.int64)
        if frame_axis.size == 0:
            continue
        effective_start = max(int(frame_axis[0]), selected_start)
        effective_stop = min(int(frame_axis[-1]) + 1, selected_stop)
        if effective_start >= effective_stop:
            continue
        first = int(np.searchsorted(frame_axis, effective_start, side="left"))
        stop = int(np.searchsorted(frame_axis, effective_stop, side="left"))
        track_group = bound.track_source.run_group["tracks"][f"id_{track_id}"]
        source_frames = _source_array(
            track_group,
            "source_acquisition_frame_index",
            first,
            stop,
        ).astype(np.int64, copy=False)
        if not np.array_equal(source_frames, frame_axis[first:stop]):
            raise ValueError(f"Track {track_id} frame authority changed.")
        metrics = activity.summarize_activity_spatial_track(
            track_id=track_id,
            source_acquisition_frame_index=source_frames,
            source_observed=_source_array(
                track_group, "source_observed", first, stop
            ),
            sample_valid=_source_array(track_group, "sample_valid", first, stop),
            position_finite=_source_array(
                track_group, "position_finite", first, stop
            ),
            transition_valid=_source_array(
                track_group, "transition_valid", first, stop
            ),
            positions_mm=_source_array(track_group, "positions_mm", first, stop),
            filtered_speed_mm_s=_source_array(
                track_group, "movement/speed/filtered/mm", first, stop
            ),
            filtered_path_distance_mm=_source_array(
                track_group,
                "movement/speed/filtered/frame_path_distance_mm",
                first,
                stop,
            ),
            bouts=bout_source.events.bouts,
            source_sample_rate_hz=float(binning["source_sample_rate_hz"]),
            requested_bin_size_s=float(binning["requested_bin_size_s"]),
            track_frame_span=(effective_start, effective_stop - 1),
        )
        static = activity._static_row_values(
            source_binding=source_binding,
            bout_binding=bout_source.binding,
            binning_contract=binning,
        )
        for metric in metrics:
            row = {**static, **metric}
            rows.append(row)
            if (
                int(metric["start_acquisition_frame_index"]) < selected_start
                or int(metric["end_acquisition_frame_index_exclusive"])
                > selected_stop
            ):
                edge_keys.append(
                    {
                        "track_id": track_id,
                        "time_bin_index": int(metric["time_bin_index"]),
                        "expected_track_frame_count": int(
                            metric["expected_track_frame_count"]
                        ),
                    }
                )
    table = activity._rows_to_arrow_table(rows)
    hasher = activity._DecodedPayloadHasher()
    hasher.update(table.to_pydict())
    return hasher.finish(), edge_keys


def validate_activity_spatial_query_window_equivalence(
    *,
    full_export_root: Path,
    full_export_run_id: str,
    bounded_export_root: Path,
    bounded_export_run_id: str,
    output: Path,
) -> dict[str, Any]:
    """Validate and persist one immutable source-backed equality receipt."""

    destination = _require_benchmark_output(output)
    full_root = full_export_root.expanduser().resolve()
    bounded_root = bounded_export_root.expanduser().resolve()
    if full_root == bounded_root:
        raise ValueError("Full and bounded exports must have distinct roots.")
    full_path, full = _read_manifest(full_root, full_export_run_id)
    bounded_path, bounded = _read_manifest(bounded_root, bounded_export_run_id)
    full_file_sha = sha256_file(full_path)
    bounded_file_sha = sha256_file(bounded_path)
    full_validation = activity.validate_activity_spatial_time_bins_export_payload(
        full_root, full
    )
    bounded_validation = activity.validate_activity_spatial_time_bins_export_payload(
        bounded_root, bounded
    )
    full_envelope = full["activity_spatial_time_bins_export"]
    bounded_envelope = bounded["activity_spatial_time_bins_export"]
    full_binning = full_envelope["binning_contract"]
    bounded_binning = bounded_envelope["binning_contract"]
    if (
        full_envelope.get("schema_version")
        != activity.ACTIVITY_SPATIAL_EXPORT_SCHEMA_VERSION
        or bounded_envelope.get("schema_version")
        != activity.ACTIVITY_SPATIAL_EXPORT_SCHEMA_VERSION_V4
        or full_binning.get("schema_version")
        != activity.ACTIVITY_SPATIAL_BINNING_SCHEMA_VERSION
        or bounded_binning.get("schema_version")
        != activity.ACTIVITY_SPATIAL_BINNING_SCHEMA_VERSION_V3
    ):
        raise ValueError("Expected one unbounded v2/v3 and one bounded v3/v4 export.")
    if full_envelope["source_binding"] != bounded_envelope["source_binding"]:
        raise ValueError("Full and bounded exports bind different source authority.")
    expected_bounded = activity.activity_spatial_binning_contract(
        source_sample_rate_hz=float(full_binning["source_sample_rate_hz"]),
        requested_bin_size_s=float(full_binning["requested_bin_size_s"]),
        source_frame_start=bounded_binning.get("source_frame_start"),
        source_frame_stop_exclusive=bounded_binning.get(
            "source_frame_stop_exclusive"
        ),
    )
    if dict(bounded_binning) != expected_bounded:
        raise ValueError("Bounded binning is not the exact v3 form of v2.")
    frame_start = int(bounded_binning["source_frame_start"])
    frame_stop = int(bounded_binning["source_frame_stop_exclusive"])

    full_interior = _decoded_receipt(
        export_root=full_root,
        manifest=full,
        frame_start=frame_start,
        frame_stop_exclusive=frame_stop,
    )
    bounded_interior = _decoded_receipt(
        export_root=bounded_root,
        manifest=bounded,
        frame_start=frame_start,
        frame_stop_exclusive=frame_stop,
    )
    full_columns = dict(full_interior["column_sha256"])
    bounded_columns = dict(bounded_interior["column_sha256"])
    full_columns.pop("source_lineage_hash")
    bounded_columns.pop("source_lineage_hash")
    if (
        full_interior["row_count"] != bounded_interior["row_count"]
        or full_columns != bounded_columns
    ):
        raise ValueError("Bounded interior bins differ from the unbounded export.")

    recomputed, edge_keys = _source_recomputed_receipt(
        source_binding=bounded_envelope["source_binding"],
        binning=bounded_binning,
    )
    if recomputed != bounded_envelope["decoded_payload"]:
        raise ValueError("Bounded export differs from source-backed recomputation.")
    if (
        sha256_file(full_path) != full_file_sha
        or sha256_file(bounded_path) != bounded_file_sha
    ):
        raise RuntimeError("An export manifest changed during equality validation.")

    git = get_git_info(Path(__file__).resolve().parents[3])
    body = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "palette_git": git,
        "evidence_eligible": git.get("is_dirty") is False,
        "full_export": {
            "root": str(full_root),
            "run_id": full_export_run_id,
            "manifest_path": str(full_path),
            "manifest_file_sha256": full_file_sha,
            "validation": full_validation,
        },
        "bounded_export": {
            "root": str(bounded_root),
            "run_id": bounded_export_run_id,
            "manifest_path": str(bounded_path),
            "manifest_file_sha256": bounded_file_sha,
            "validation": bounded_validation,
        },
        "frame_interval": {
            "start": frame_start,
            "stop_exclusive": frame_stop,
            "frame_count": frame_stop - frame_start,
        },
        "interior_equality": {
            "equal": True,
            "row_count": int(bounded_interior["row_count"]),
            "excluded_contract_column": "source_lineage_hash",
            "column_sha256": bounded_columns,
        },
        "source_backed_recomputation": {
            "equal": True,
            "receipt": recomputed,
            "selection_edge_rows": edge_keys,
        },
        "manifest_nonmutation": True,
        "production_state_changes": [],
        "promotion_authorized": False,
    }
    document = {
        "schema_id": EVIDENCE_SCHEMA_ID,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "payload": body,
        "payload_digest": canonical_json_sha256(body),
    }
    _write_strict_json(destination, document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export-root", type=Path, required=True)
    parser.add_argument("--full-export-run-id", required=True)
    parser.add_argument("--bounded-export-root", type=Path, required=True)
    parser.add_argument("--bounded-export-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_activity_spatial_query_window_equivalence(
        full_export_root=args.full_export_root,
        full_export_run_id=args.full_export_run_id,
        bounded_export_root=args.bounded_export_root,
        bounded_export_run_id=args.bounded_export_run_id,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["payload"]["status"],
                "output": str(args.output.expanduser().resolve()),
                "payload_digest": result["payload_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EVIDENCE_SCHEMA_ID",
    "EVIDENCE_SCHEMA_VERSION",
    "validate_activity_spatial_query_window_equivalence",
]
