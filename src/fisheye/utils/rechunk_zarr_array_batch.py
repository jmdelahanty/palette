"""Batch rechunk low-risk Zarr arrays selected by explicit presets.

Default mode is dry-run. Use ``--apply`` to rewrite matching arrays one at a
time through :mod:`fisheye.utils.rechunk_zarr_array`.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from fisheye.utils.audit_zarr_array_sizes import ZarrArrayAuditRow, scan_zarr_array_sizes
from fisheye.utils.rechunk_zarr_array import (
    DEFAULT_STORAGE_PROFILE_ID,
    RechunkArraySummary,
    rechunk_zarr_array,
)


PRESET_PATTERNS: dict[str, tuple[str, ...]] = {
    "crimson-lineage-v1": (
        r"^crop_runs/[^/]+/(detection_indices|frame_indices|source_detect_row_index|source_refined_row_ids)$",
        r"^keypoints_runs/[^/]+/(detection_indices|frame_indices|source_detect_row_index|source_refined_row_ids)$",
        r"^refined_keypoints_runs/[^/]+/(detection_indices|frame_indices|source_detect_row_index|source_refined_row_ids)$",
        r"^detect_runs/[^/]+/(frame_counts|n_detections)$",
    ),
    "track-kinematics-v1": (
        r"^analysis/track_kinematics_runs/[^/]+/[^/]+/tracks/[^/]+/.+",
    ),
}


@dataclass(frozen=True)
class RechunkBatchSummary:
    zarr_path: str
    preset: str
    row_chunk: int
    storage_profile_id: str
    apply: bool
    matched_count: int
    planned_count: int
    updated_count: int
    already_matching_count: int
    old_chunk_count_total: int
    new_chunk_count_total: int
    rows: tuple[RechunkArraySummary, ...]


def _compile_preset(preset: str) -> tuple[re.Pattern[str], ...]:
    try:
        patterns = PRESET_PATTERNS[preset]
    except KeyError as exc:
        raise ValueError(f"Unsupported preset {preset!r}; choices: {sorted(PRESET_PATTERNS)}") from exc
    return tuple(re.compile(pattern) for pattern in patterns)


def _matches_any(path: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(path) for pattern in patterns)


def _is_candidate(
    row: ZarrArrayAuditRow,
    *,
    patterns: Sequence[re.Pattern[str]],
    row_chunk: int,
    min_chunk_count: int,
    only_if_row_chunk_smaller: bool,
) -> bool:
    if not _matches_any(row.array_path, patterns):
        return False
    if row.chunk_shape is None:
        return False
    if row.chunk_count is not None and int(row.chunk_count) < int(min_chunk_count):
        return False
    if only_if_row_chunk_smaller and int(row.chunk_shape[0]) >= int(row_chunk):
        return False
    return True


def find_batch_candidates(
    zarr_path: Path | str,
    *,
    preset: str,
    row_chunk: int,
    min_chunk_count: int = 2,
    only_if_row_chunk_smaller: bool = True,
) -> list[ZarrArrayAuditRow]:
    zarr_path = Path(zarr_path).expanduser().resolve()
    patterns = _compile_preset(preset)
    rows = scan_zarr_array_sizes(zarr_path, collect_physical=False)
    candidates = [
        row
        for row in rows
        if _is_candidate(
            row,
            patterns=patterns,
            row_chunk=row_chunk,
            min_chunk_count=min_chunk_count,
            only_if_row_chunk_smaller=only_if_row_chunk_smaller,
        )
    ]
    return sorted(candidates, key=lambda row: row.array_path)


def rechunk_zarr_array_batch(
    zarr_path: Path | str,
    *,
    preset: str = "crimson-lineage-v1",
    row_chunk: int = 16384,
    storage_profile_id: str = DEFAULT_STORAGE_PROFILE_ID,
    reason: str | None = None,
    min_chunk_count: int = 2,
    only_if_row_chunk_smaller: bool = True,
    apply: bool = False,
    limit: int | None = None,
) -> RechunkBatchSummary:
    zarr_path = Path(zarr_path).expanduser().resolve()
    candidates = find_batch_candidates(
        zarr_path,
        preset=preset,
        row_chunk=row_chunk,
        min_chunk_count=min_chunk_count,
        only_if_row_chunk_smaller=only_if_row_chunk_smaller,
    )
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]
    rows: list[RechunkArraySummary] = []
    for candidate in candidates:
        rows.append(
            rechunk_zarr_array(
                zarr_path,
                candidate.array_path,
                row_chunk=row_chunk,
                storage_profile_id=storage_profile_id,
                reason=reason or f"batch preset {preset}",
                apply=apply,
            )
        )

    return RechunkBatchSummary(
        zarr_path=str(zarr_path),
        preset=preset,
        row_chunk=int(row_chunk),
        storage_profile_id=storage_profile_id,
        apply=bool(apply),
        matched_count=len(candidates),
        planned_count=sum(1 for row in rows if row.status == "planned"),
        updated_count=sum(1 for row in rows if row.status == "updated"),
        already_matching_count=sum(1 for row in rows if row.status == "already_matching"),
        old_chunk_count_total=sum(row.old_chunk_count or 0 for row in rows),
        new_chunk_count_total=sum(row.new_chunk_count or 0 for row in rows),
        rows=tuple(rows),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Zarr root to scan and optionally rewrite.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_PATTERNS),
        default="crimson-lineage-v1",
        help="Explicit allowlist preset.",
    )
    parser.add_argument("--row-chunk", type=int, default=16384, help="New first-axis chunk size.")
    parser.add_argument(
        "--storage-profile-id",
        default=DEFAULT_STORAGE_PROFILE_ID,
        help="Storage/chunking profile stamp to write when applying.",
    )
    parser.add_argument("--reason", help="Reason stored in each array's rechunk provenance.")
    parser.add_argument(
        "--min-chunk-count",
        type=int,
        default=2,
        help="Skip arrays with fewer than this many current chunks.",
    )
    parser.add_argument(
        "--include-row-chunk-at-least-target",
        action="store_true",
        help="Do not skip arrays whose first-axis chunk is already >= --row-chunk.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of matched arrays, useful for canaries.")
    parser.add_argument("--apply", action="store_true", help="Rewrite matching arrays. Default is dry-run.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser


def _jsonable(summary: RechunkBatchSummary) -> dict[str, object]:
    payload = asdict(summary)
    return payload


def _print_summary(summary: RechunkBatchSummary) -> None:
    print(
        "{mode}\tpreset={preset}\tmatched={matched}\tplanned={planned}\tupdated={updated}\talready_matching={matching}\told_chunks={old}\tnew_chunks={new}".format(
            mode="applied" if summary.apply else "dry-run",
            preset=summary.preset,
            matched=summary.matched_count,
            planned=summary.planned_count,
            updated=summary.updated_count,
            matching=summary.already_matching_count,
            old=summary.old_chunk_count_total,
            new=summary.new_chunk_count_total,
        )
    )
    for row in summary.rows:
        print(
            "{status}\t{array}\t{old}->{new}\t{old_n}->{new_n}".format(
                status=row.status,
                array=row.array_path,
                old=row.old_chunks,
                new=row.new_chunks,
                old_n=row.old_chunk_count,
                new_n=row.new_chunk_count,
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = rechunk_zarr_array_batch(
        args.zarr_path,
        preset=args.preset,
        row_chunk=int(args.row_chunk),
        storage_profile_id=args.storage_profile_id,
        reason=args.reason,
        min_chunk_count=int(args.min_chunk_count),
        only_if_row_chunk_smaller=not bool(args.include_row_chunk_at_least_target),
        apply=bool(args.apply),
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(_jsonable(summary), sort_keys=True))
    else:
        _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
