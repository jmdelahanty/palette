"""Rank latest completed Zarr runs for storage-layout follow-up.

The audit is intentionally filesystem-only. It reads ``zarr.json`` metadata
and stats existing payload files; it never opens an array or writes into the
source archive. Parent ``latest_complete`` pointers define the completed run
surface so abandoned, hidden, and historical runs do not distort the ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.utils.audit_zarr_array_sizes import (
    ZarrArrayAuditRow,
    scan_zarr_array_sizes,
)


_INTERMEDIATE_FAMILY_NAMES = {
    "crop_runs",
    "detect_runs",
    "keypoint_shard_runs",
    "subject_mask_shard_runs",
}


@dataclass(frozen=True)
class LatestRunPointer:
    family_path: str
    run_name: str
    run_path: str
    pointer_key: str
    completion_status: str
    schema_id: str | None
    rank_eligible: bool


@dataclass(frozen=True)
class RunStorageAudit:
    family_path: str
    run_name: str
    run_path: str
    pointer_key: str
    completion_status: str
    schema_id: str | None
    rank_eligible: bool
    array_count: int
    sharded_array_count: int
    logical_bytes: int
    physical_bytes: int
    physical_file_count: int
    expected_payload_file_count: int
    compression_ratio_logical_to_physical: float | None
    average_physical_file_bytes: float | None
    wide_full_width_inner_chunk_array_count: int
    wide_full_width_inner_chunk_bytes: int
    wide_full_width_inner_chunk_array_paths: tuple[str, ...]
    unsharded_many_chunk_array_count: int
    unsharded_expected_chunk_count: int
    unsharded_many_chunk_array_paths: tuple[str, ...]
    small_payload_file_array_count: int
    small_payload_file_count: int
    small_payload_file_array_paths: tuple[str, ...]
    publication_provenance_status: str
    publication_evidence: tuple[str, ...]
    candidate_reasons: tuple[str, ...]
    top_arrays_by_file_count: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _iter_run_family_metadata(root: Path) -> Iterable[tuple[Path, Mapping[str, Any]]]:
    """Yield conventional run-family groups without walking historical runs.

    Palette run families live either directly below the archive root or one
    level below ``analysis``. Recursing through the complete archive merely to
    find these pointer groups is especially costly on PRFS and defeats the
    purpose of a lightweight audit.
    """

    bases = [root, root / "analysis"]
    for base in bases:
        if not base.is_dir():
            continue
        for directory in sorted(base.iterdir()):
            if not directory.is_dir() or not directory.name.endswith("_runs"):
                continue
            payload = _read_json(directory / "zarr.json")
            if payload is not None and payload.get("node_type") == "group":
                yield directory, payload


def _run_completion_status(
    run_metadata: Mapping[str, Any],
    *,
    pointer_key: str,
) -> str:
    attrs = run_metadata.get("attributes")
    if isinstance(attrs, Mapping):
        for key in (
            "palette_run_completion_status",
            "completion_status",
            "run_status",
            "status",
        ):
            value = attrs.get(key)
            if value not in (None, ""):
                return str(value)
    if pointer_key == "latest_complete":
        return "complete_via_parent_pointer"
    return "unknown"


def discover_latest_completed_runs(
    zarr_path: Path,
    *,
    include_family: re.Pattern[str] | None = None,
    exclude_family: re.Pattern[str] | None = None,
    allow_legacy_latest: bool = False,
) -> list[LatestRunPointer]:
    """Resolve one authoritative completed run from every ``*_runs`` family."""

    zarr_path = zarr_path.expanduser().resolve()
    pointers: list[LatestRunPointer] = []
    for family_dir, metadata in _iter_run_family_metadata(zarr_path):
        family_path = family_dir.relative_to(zarr_path).as_posix()
        if include_family is not None and include_family.search(family_path) is None:
            continue
        if exclude_family is not None and exclude_family.search(family_path) is not None:
            continue
        attrs = metadata.get("attributes")
        if not isinstance(attrs, Mapping):
            continue
        pointer_key = "latest_complete"
        run_name = attrs.get(pointer_key)
        if not isinstance(run_name, str) or not run_name:
            if not allow_legacy_latest:
                continue
            pointer_key = "latest"
            run_name = attrs.get(pointer_key)
        if not isinstance(run_name, str) or not run_name:
            continue

        run_dir = family_dir / run_name
        run_metadata = _read_json(run_dir / "zarr.json")
        if run_metadata is None or run_metadata.get("node_type") != "group":
            continue
        run_attrs = run_metadata.get("attributes")
        schema_id = None
        if isinstance(run_attrs, Mapping) and run_attrs.get("schema_id") is not None:
            schema_id = str(run_attrs["schema_id"])
        pointers.append(
            LatestRunPointer(
                family_path=family_path,
                run_name=run_name,
                run_path=run_dir.relative_to(zarr_path).as_posix(),
                pointer_key=pointer_key,
                completion_status=_run_completion_status(
                    run_metadata,
                    pointer_key=pointer_key,
                ),
                schema_id=schema_id,
                rank_eligible=family_dir.name not in _INTERMEDIATE_FAMILY_NAMES,
            )
        )
    return sorted(pointers, key=lambda item: item.family_path)


def _publication_evidence(run_metadata: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    attrs = run_metadata.get("attributes")
    if not isinstance(attrs, Mapping):
        return "unrecorded", ()
    evidence: list[str] = []
    for key in (
        "cluster_output_staging",
        "node_local_materialization",
        "node_local_source_staging",
        "materialization_mode",
        "physical_storage_layout",
    ):
        value = attrs.get(key)
        if value not in (None, "", False, {}, []):
            evidence.append(key)
    if "cluster_output_staging" in evidence:
        status = "staged_publication_recorded"
    elif "node_local_materialization" in evidence or "materialization_mode" in evidence:
        status = "materialization_strategy_recorded"
    else:
        status = "unrecorded"
    return status, tuple(evidence)


def _expected_payload_files(row: ZarrArrayAuditRow) -> int:
    if row.shard_shape is not None:
        return int(row.shard_count or 0)
    return int(row.chunk_count or 0)


def _array_summary(row: ZarrArrayAuditRow) -> dict[str, Any]:
    return {
        "array_path": row.array_path,
        "shape": list(row.shape),
        "chunk_shape": list(row.chunk_shape) if row.chunk_shape is not None else None,
        "shard_shape": list(row.shard_shape) if row.shard_shape is not None else None,
        "logical_bytes": row.logical_bytes,
        "physical_bytes": row.physical_bytes,
        "physical_file_count": row.physical_file_count,
        "expected_payload_file_count": _expected_payload_files(row),
    }


def audit_run_storage(
    zarr_path: Path,
    pointer: LatestRunPointer,
    *,
    many_chunk_threshold: int = 1000,
    small_payload_threshold_bytes: int = 1024 * 1024,
    small_payload_min_files: int = 32,
    wide_column_threshold: int = 32,
) -> RunStorageAudit:
    run_dir = zarr_path / pointer.run_path
    rows = scan_zarr_array_sizes(run_dir, collect_physical=True)
    logical_bytes = sum(int(row.logical_bytes or 0) for row in rows)
    physical_bytes = sum(int(row.physical_bytes or 0) for row in rows)
    physical_file_count = sum(int(row.physical_file_count or 0) for row in rows)
    expected_payload_file_count = sum(_expected_payload_files(row) for row in rows)

    wide_full_width = [
        row
        for row in rows
        if len(row.shape) == 2
        and row.shape[0] >= many_chunk_threshold
        and row.shape[-1] >= wide_column_threshold
        and row.chunk_shape is not None
        and row.chunk_shape[-1] >= row.shape[-1]
        and row.logical_bytes is not None
        and row.logical_bytes >= small_payload_threshold_bytes
        and row.dtype.lower()
        in {"float16", "float32", "float64", "int16", "int32", "int64"}
    ]
    unsharded_many = [
        row
        for row in rows
        if row.shard_shape is None and int(row.chunk_count or 0) >= many_chunk_threshold
    ]
    small_payload = []
    for row in rows:
        count = int(row.physical_file_count or 0)
        if count < small_payload_min_files:
            continue
        average = float(row.physical_bytes or 0) / count if count else 0.0
        if average < small_payload_threshold_bytes:
            small_payload.append(row)

    run_metadata = _read_json(run_dir / "zarr.json") or {}
    publication_status, publication_evidence = _publication_evidence(run_metadata)
    reasons: list[str] = []
    if physical_file_count >= many_chunk_threshold:
        reasons.append("run fans out across many payload files")
    if unsharded_many:
        reasons.append("unsharded arrays have many logical chunks")
    if small_payload:
        reasons.append("many small payload files can amplify PRFS metadata cost")
    if wide_full_width:
        reasons.append("wide arrays decode every column in an inner chunk")
    if publication_status == "unrecorded":
        reasons.append("publication/materialization strategy is not recorded")

    compression_ratio = None
    if physical_bytes > 0:
        compression_ratio = float(logical_bytes) / float(physical_bytes)
    average_physical_file_bytes = None
    if physical_file_count > 0:
        average_physical_file_bytes = float(physical_bytes) / float(physical_file_count)

    top_arrays = sorted(
        rows,
        key=lambda row: (
            -int(row.physical_file_count or 0),
            -int(row.physical_bytes or 0),
            row.array_path,
        ),
    )[:5]
    return RunStorageAudit(
        family_path=pointer.family_path,
        run_name=pointer.run_name,
        run_path=pointer.run_path,
        pointer_key=pointer.pointer_key,
        completion_status=pointer.completion_status,
        schema_id=pointer.schema_id,
        rank_eligible=pointer.rank_eligible,
        array_count=len(rows),
        sharded_array_count=sum(row.shard_shape is not None for row in rows),
        logical_bytes=logical_bytes,
        physical_bytes=physical_bytes,
        physical_file_count=physical_file_count,
        expected_payload_file_count=expected_payload_file_count,
        compression_ratio_logical_to_physical=compression_ratio,
        average_physical_file_bytes=average_physical_file_bytes,
        wide_full_width_inner_chunk_array_count=len(wide_full_width),
        wide_full_width_inner_chunk_bytes=sum(
            int(row.chunk_logical_bytes or 0) for row in wide_full_width
        ),
        wide_full_width_inner_chunk_array_paths=tuple(
            row.array_path for row in wide_full_width
        ),
        unsharded_many_chunk_array_count=len(unsharded_many),
        unsharded_expected_chunk_count=sum(
            int(row.chunk_count or 0) for row in unsharded_many
        ),
        unsharded_many_chunk_array_paths=tuple(
            row.array_path for row in unsharded_many
        ),
        small_payload_file_array_count=len(small_payload),
        small_payload_file_count=sum(
            int(row.physical_file_count or 0) for row in small_payload
        ),
        small_payload_file_array_paths=tuple(row.array_path for row in small_payload),
        publication_provenance_status=publication_status,
        publication_evidence=publication_evidence,
        candidate_reasons=tuple(reasons),
        top_arrays_by_file_count=tuple(_array_summary(row) for row in top_arrays),
    )


def _sha256_files(paths: Iterable[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _guard_paths(zarr_path: Path, pointers: Sequence[LatestRunPointer]) -> list[Path]:
    paths = [zarr_path / "zarr.json"]
    for pointer in pointers:
        paths.append(zarr_path / pointer.family_path / "zarr.json")
        paths.append(zarr_path / pointer.run_path / "zarr.json")
    return [path for path in paths if path.is_file()]


def _rank(
    rows: Sequence[RunStorageAudit],
    key: str,
    *,
    include_intermediate: bool,
) -> list[str]:
    eligible = [row for row in rows if include_intermediate or row.rank_eligible]
    return [
        row.run_path
        for row in sorted(
            eligible,
            key=lambda row: (-int(getattr(row, key)), row.run_path),
        )
    ]


def build_storage_audit_report(
    zarr_path: Path,
    *,
    include_family: re.Pattern[str] | None = None,
    exclude_family: re.Pattern[str] | None = None,
    allow_legacy_latest: bool = False,
    include_intermediate_rankings: bool = False,
) -> dict[str, Any]:
    """Build a read-only report for authoritative completed run pointers."""

    zarr_path = zarr_path.expanduser().resolve()
    if not (zarr_path / "zarr.json").is_file():
        raise FileNotFoundError(f"Zarr v3 root metadata not found: {zarr_path}")
    pointers = discover_latest_completed_runs(
        zarr_path,
        include_family=include_family,
        exclude_family=exclude_family,
        allow_legacy_latest=allow_legacy_latest,
    )
    guard_paths = _guard_paths(zarr_path, pointers)
    digest_before = _sha256_files(guard_paths, root=zarr_path)
    runs = [audit_run_storage(zarr_path, pointer) for pointer in pointers]
    digest_after = _sha256_files(guard_paths, root=zarr_path)
    if digest_before != digest_after:
        raise RuntimeError("Source metadata changed while the read-only audit was running")

    return {
        "schema_id": "palette.analysis_storage_candidate_audit.v1",
        "zarr_path": str(zarr_path),
        "selection": {
            "source": "parent latest_complete pointers",
            "allow_legacy_latest": bool(allow_legacy_latest),
            "include_intermediate_rankings": bool(include_intermediate_rankings),
        },
        "read_only_guard": {
            "metadata_file_count": len(guard_paths),
            "metadata_sha256_before": digest_before,
            "metadata_sha256_after": digest_after,
            "unchanged": True,
            "array_values_read": False,
            "source_writes_performed": False,
        },
        "summary": {
            "run_count": len(runs),
            "rank_eligible_run_count": sum(row.rank_eligible for row in runs),
            "array_count": sum(row.array_count for row in runs),
            "logical_bytes": sum(row.logical_bytes for row in runs),
            "physical_bytes": sum(row.physical_bytes for row in runs),
            "physical_file_count": sum(row.physical_file_count for row in runs),
        },
        "rankings": {
            "by_physical_file_count": _rank(
                runs,
                "physical_file_count",
                include_intermediate=include_intermediate_rankings,
            ),
            "by_physical_bytes": _rank(
                runs,
                "physical_bytes",
                include_intermediate=include_intermediate_rankings,
            ),
            "by_logical_bytes": _rank(
                runs,
                "logical_bytes",
                include_intermediate=include_intermediate_rankings,
            ),
            "by_unsharded_expected_chunk_count": _rank(
                runs,
                "unsharded_expected_chunk_count",
                include_intermediate=include_intermediate_rankings,
            ),
            "by_wide_full_width_inner_chunk_bytes": _rank(
                runs,
                "wide_full_width_inner_chunk_bytes",
                include_intermediate=include_intermediate_rankings,
            ),
        },
        "runs": [asdict(row) for row in runs],
    }


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"


def render_markdown(report: Mapping[str, Any]) -> str:
    runs = [row for row in report.get("runs", []) if isinstance(row, Mapping)]
    ranked_paths = report.get("rankings", {}).get("by_physical_file_count", [])
    by_path = {str(row.get("run_path")): row for row in runs}
    lines = [
        "# Analysis storage candidate audit",
        "",
        f"Source: `{report.get('zarr_path')}`",
        "",
        "The source was inspected read-only from metadata and payload file stats; no array values were decoded.",
        "",
        "| Rank | Run | Payload files | Physical | Logical | Sharded arrays | Unsharded chunks | Wide full-column chunks | Publication provenance |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, path in enumerate(ranked_paths, start=1):
        row = by_path[str(path)]
        lines.append(
            "| {rank} | `{path}` | {files:,} | {physical} | {logical} | {sharded:,}/{arrays:,} | {chunks:,} | {wide:,} | {publication} |".format(
                rank=rank,
                path=path,
                files=int(row.get("physical_file_count", 0)),
                physical=_format_bytes(row.get("physical_bytes")),
                logical=_format_bytes(row.get("logical_bytes")),
                sharded=int(row.get("sharded_array_count", 0)),
                arrays=int(row.get("array_count", 0)),
                chunks=int(row.get("unsharded_expected_chunk_count", 0)),
                wide=int(row.get("wide_full_width_inner_chunk_array_count", 0)),
                publication=row.get("publication_provenance_status", "unrecorded"),
            )
        )
    lines.extend(
        [
            "",
            "Rankings omit known shard/crop/detection intermediate families unless explicitly requested. Separate JSON rankings preserve physical-file, physical-byte, logical-byte, unsharded-chunk, and column-over-read orderings rather than collapsing them into one opaque score.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--include-family", help="Only include family paths matching this regex.")
    parser.add_argument("--exclude-family", help="Exclude family paths matching this regex.")
    parser.add_argument(
        "--allow-legacy-latest",
        action="store_true",
        help="Use latest only when a family has no latest_complete pointer.",
    )
    parser.add_argument(
        "--include-intermediate-rankings",
        action="store_true",
        help="Include known shard/crop/detection intermediate families in rankings.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_storage_audit_report(
        args.zarr_path,
        include_family=re.compile(args.include_family) if args.include_family else None,
        exclude_family=re.compile(args.exclude_family) if args.exclude_family else None,
        allow_legacy_latest=bool(args.allow_legacy_latest),
        include_intermediate_rankings=bool(args.include_intermediate_rankings),
    )
    json_text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text + "\n", encoding="utf-8")
    else:
        print(json_text)
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
