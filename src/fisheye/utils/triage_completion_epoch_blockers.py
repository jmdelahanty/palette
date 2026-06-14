"""Summarize blocked strict-run-completion backfill parents for manual triage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Optional

REPORT_SCHEMA_ID = "palette.completion_epoch_blocker_triage.v1"
DEFERRED_NO_SPEC_PARENT_PATHS = frozenset(
    {
        "analysis/stimulus_response_runs",
        "analysis/swim_bout_runs",
    }
)
DEPRECATED_STAGE_KEYS = frozenset({"eye_masks", "refined_eye_masks"})
DEPRECATED_PARENT_PATHS = frozenset({"eye_masks_runs", "refined_eye_masks_runs"})


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object row")
            rows.append(value)
    return rows


def _counter_from_mapping(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        try:
            counter[str(key)] += int(count)
        except (TypeError, ValueError):
            continue
    return counter


def _ranked(counter: Counter[str], *, limit: int | None = None) -> list[dict[str, Any]]:
    return [{"key": str(key), "count": int(count)} for key, count in counter.most_common(limit)]


def _group_key(row: Mapping[str, Any]) -> str:
    stage = row.get("stage")
    if isinstance(stage, str) and stage:
        return stage
    parent_path = row.get("parent_path")
    return str(parent_path or "unknown")


def _recommendation(
    *,
    group_key: str,
    stage_spec_available_count: int,
    row_count: int,
    parent_paths: Counter[str],
    reason_counts: Counter[str],
    first_error_counts: Counter[str],
) -> str:
    if group_key in DEPRECATED_STAGE_KEYS or set(parent_paths).issubset(DEPRECATED_PARENT_PATHS):
        return "defer_deprecated_scope"
    if (
        row_count > 0
        and stage_spec_available_count == 0
        and set(parent_paths).issubset(DEFERRED_NO_SPEC_PARENT_PATHS)
    ):
        return "defer_scope_until_layout_specific_validator"
    if row_count > 0 and stage_spec_available_count == 0:
        return "add_stage_array_spec_or_defer_scope"
    if reason_counts and set(reason_counts) == {"no_stage_array_spec"}:
        return "add_stage_array_spec_or_defer_scope"

    error_text = "\n".join(first_error_counts)
    if "missing required array" in error_text or "missing required subgroup" in error_text:
        return "review_stage_spec_compatibility_or_backfill_missing_surface"
    if "leading dimension mismatch" in error_text:
        return "inspect_or_regenerate_corrupt_run_surface"
    if reason_counts.get("invalid", 0):
        return "inspect_invalid_runs"
    return "manual_review"


def build_triage_report(rows: Sequence[Mapping[str, Any]], *, examples_per_group: int = 5) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[_group_key(row)].append(row)

    groups: list[dict[str, Any]] = []
    global_reason_counts: Counter[str] = Counter()
    global_first_error_counts: Counter[str] = Counter()
    global_parent_paths: Counter[str] = Counter()

    for key, group_rows in by_group.items():
        reason_counts: Counter[str] = Counter()
        first_error_counts: Counter[str] = Counter()
        parent_paths: Counter[str] = Counter()
        stage_spec_available_count = 0
        blocked_child_count = 0
        examples: list[dict[str, Any]] = []

        for row in group_rows:
            parent_path = str(row.get("parent_path") or "unknown")
            parent_paths[parent_path] += 1
            global_parent_paths[parent_path] += 1
            if bool(row.get("stage_spec_available")):
                stage_spec_available_count += 1

            row_reasons = _counter_from_mapping(row.get("blocked_child_reason_counts"))
            row_errors = _counter_from_mapping(row.get("blocked_child_first_error_counts_top10"))
            reason_counts.update(row_reasons)
            first_error_counts.update(row_errors)
            global_reason_counts.update(row_reasons)
            global_first_error_counts.update(row_errors)
            blocked_child_count += sum(row_reasons.values())

            if len(examples) < examples_per_group:
                examples.append(
                    {
                        "zarr_path": row.get("zarr_path"),
                        "parent_path": row.get("parent_path"),
                        "stage": row.get("stage"),
                        "latest": row.get("latest"),
                        "latest_complete": row.get("latest_complete"),
                        "blocked_child_examples": row.get("blocked_child_examples", []),
                    }
                )

        groups.append(
            {
                "key": key,
                "recommendation": _recommendation(
                    group_key=key,
                    stage_spec_available_count=stage_spec_available_count,
                    row_count=len(group_rows),
                    parent_paths=parent_paths,
                    reason_counts=reason_counts,
                    first_error_counts=first_error_counts,
                ),
                "blocked_parent_count": len(group_rows),
                "blocked_child_count": int(blocked_child_count),
                "stage_spec_available_parent_count": stage_spec_available_count,
                "parent_path_counts": _ranked(parent_paths),
                "blocked_child_reason_counts": _ranked(reason_counts),
                "blocked_child_first_error_counts_top20": _ranked(first_error_counts, limit=20),
                "examples": examples,
            }
        )

    groups.sort(
        key=lambda item: (
            str(item["recommendation"]),
            -int(item["blocked_parent_count"]),
            str(item["key"]),
        )
    )
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "blocked_parent_count": len(rows),
        "group_count": len(groups),
        "blocked_parent_counts_by_parent_path": _ranked(global_parent_paths),
        "blocked_child_reason_counts": _ranked(global_reason_counts),
        "blocked_child_first_error_counts_top50": _ranked(global_first_error_counts, limit=50),
        "groups": groups,
    }


def _write_report(report: Mapping[str, Any], output_json: str | Path | None, *, emit_stdout: bool) -> None:
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if output_json:
        Path(output_json).expanduser().write_text(text + "\n", encoding="utf-8")
    if emit_stdout:
        print(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read backfill_completion_epoch --blocked-jsonl output and group blocked "
            "parents into manual triage buckets."
        )
    )
    parser.add_argument("blocked_jsonl", help="JSONL sidecar from backfill_completion_epoch --blocked-jsonl.")
    parser.add_argument("--output-json", help="Optional path to write the triage JSON report.")
    parser.add_argument(
        "--examples-per-group",
        type=int,
        default=5,
        help="Maximum example blocked parents to include for each group.",
    )
    parser.add_argument("--no-stdout", action="store_true", help="Write --output-json without printing JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = _read_jsonl(args.blocked_jsonl)
    report = build_triage_report(rows, examples_per_group=max(0, int(args.examples_per_group)))
    _write_report(report, args.output_json, emit_stdout=not bool(args.no_stdout))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
