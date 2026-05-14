#!/usr/bin/env python3
"""Resumable batch wrapper for interactive refined-detect review."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from fisheye.shared.batch_logging import make_run_id, utc_now
from fisheye.utils import list_unapproved_analysis_zarrs as unapproved


SCHEMA_NAME = "detect_review_batch_state"
SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class QueueItem:
    zarr_path: Path
    latest_refined_run: Optional[str] = None
    review_state: Optional[str] = None
    review_intended_use: Optional[str] = None
    review_resolved_group: Optional[str] = None
    reason: Optional[str] = None


def _read_queue_file(path: Path) -> list[QueueItem]:
    items: list[QueueItem] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        items.append(QueueItem(zarr_path=Path(text)))
    return items


def _write_queue(items: Sequence[QueueItem], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(str(item.zarr_path) for item in items)
    if payload:
        payload += "\n"
    output_path.write_text(payload, encoding="utf-8")


def _write_details(items: Sequence[QueueItem], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "zarr_path\tlatest_refined_run\treview_state\treview_intended_use\treview_resolved_group\treason"
    ]
    for item in items:
        lines.append(
            "\t".join(
                [
                    str(item.zarr_path),
                    item.latest_refined_run or "<none>",
                    item.review_state or "<none>",
                    item.review_intended_use or "<none>",
                    item.review_resolved_group or "<none>",
                    item.reason or "<none>",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_zarr_attrs(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _latest_refined_run_from_metadata(zarr_path: Path) -> Optional[str]:
    parent_attrs = _read_zarr_attrs(zarr_path / "refined_detect_runs" / "zarr.json")
    latest = parent_attrs.get("latest")
    text = str(latest).strip() if latest is not None else ""
    return text or None


def _read_detect_review_status(
    zarr_path: Path,
    *,
    refined_run: Optional[str],
) -> dict[str, Optional[str]]:
    run_name = refined_run or _latest_refined_run_from_metadata(zarr_path)
    result: dict[str, Optional[str]] = {
        "refined_run": run_name,
        "review_state": None,
        "review_intended_use": None,
        "review_resolved_group": None,
    }
    if not run_name:
        return result
    attrs = _read_zarr_attrs(zarr_path / "refined_detect_runs" / run_name / "zarr.json")
    status = attrs.get("detect_review_status")
    if not isinstance(status, dict):
        return result
    for key, out_key in (
        ("state", "review_state"),
        ("intended_use", "review_intended_use"),
        ("resolved_group", "review_resolved_group"),
    ):
        value = status.get(key)
        text = str(value).strip() if value is not None else ""
        result[out_key] = text or None
    return result


def _default_state_path(run_id: str) -> Path:
    return Path("/tmp") / f"detect_review_batch_{run_id}.state.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "entries": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        payload["entries"] = {}
    payload.setdefault("schema_name", SCHEMA_NAME)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("created_at_utc", utc_now())
    payload["updated_at_utc"] = utc_now()
    return payload


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at_utc"] = utc_now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _state_entry_for(payload: dict[str, Any], zarr_path: Path) -> dict[str, Any]:
    entries = payload.setdefault("entries", {})
    assert isinstance(entries, dict)
    return dict(entries.get(str(zarr_path)) or {})


def _update_state_entry(path: Path, payload: dict[str, Any], zarr_path: Path, **fields: Any) -> None:
    entries = payload.setdefault("entries", {})
    assert isinstance(entries, dict)
    existing = dict(entries.get(str(zarr_path)) or {})
    existing.update(fields)
    existing["zarr_path"] = str(zarr_path)
    entries[str(zarr_path)] = existing
    _write_state(path, payload)


def _parse_state_list(raw: str) -> set[str]:
    return {item.strip().lower() for item in str(raw).split(",") if item.strip()}


def _should_skip_for_resume(
    item: QueueItem,
    *,
    state_payload: dict[str, Any],
    skip_states: set[str],
) -> bool:
    entry = _state_entry_for(state_payload, item.zarr_path)
    state = entry.get("last_review_state")
    text = str(state).strip().lower() if state is not None else ""
    return bool(text and text in skip_states)


def _dedupe_items(items: Iterable[QueueItem]) -> list[QueueItem]:
    out: list[QueueItem] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.zarr_path.expanduser().resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _discover_queue(args: argparse.Namespace) -> list[QueueItem]:
    if args.queue_file is not None:
        return _dedupe_items(_read_queue_file(Path(args.queue_file)))

    if args.registry is not None:
        rows = unapproved._collect_unapproved_rows_from_registry(
            Path(args.registry),
            zarr_use_filter=str(args.zarr_use),
            approved_state=str(args.approved_state),
            path_contains=args.path_contains,
        )
    else:
        rows = unapproved._collect_unapproved_rows(
            unapproved._resolve_roots(args.paths),
            recursive=bool(args.recursive),
            zarr_use_filter=str(args.zarr_use),
            approved_state=str(args.approved_state),
        )
    return _dedupe_items(
        QueueItem(
            zarr_path=Path(row.zarr_path),
            latest_refined_run=row.latest_refined_run,
            review_state=row.review_state,
            review_intended_use=row.review_intended_use,
            review_resolved_group=row.review_resolved_group,
            reason=row.reason,
        )
        for row in rows
    )


def _build_detect_review_command(args: argparse.Namespace, item: QueueItem) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.tune.detect_review",
        str(item.zarr_path),
    ]
    refined_run = args.refined_run or item.latest_refined_run
    if refined_run:
        cmd.extend(["--refined-run", str(refined_run)])
    if args.variant:
        cmd.extend(["--variant", str(args.variant)])
    if args.all:
        cmd.append("--all")
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.frames:
        cmd.extend(["--frames", str(args.frames)])
    if args.use_full_res:
        cmd.append("--use-full-res")
    cmd.extend(["--review-state", str(args.review_state)])
    cmd.extend(["--review-method", str(args.review_method)])
    cmd.extend(["--review-intended-use", str(args.review_intended_use)])
    if args.reviewer:
        cmd.extend(["--reviewer", str(args.reviewer)])
    if args.review_notes:
        cmd.extend(["--review-notes", str(args.review_notes)])
    if args.registry:
        cmd.extend(["--registry", str(args.registry)])
    if args.no_registry_sync:
        cmd.append("--no-registry-sync")
    if args.skip_detection_profile:
        cmd.append("--skip-detection-profile")
    if args.overwrite_profile:
        cmd.append("--overwrite-profile")
    if args.no_update_curated:
        cmd.append("--no-update-curated")
    return cmd


def _command_for_display(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths for filesystem discovery (default follows list_unapproved_analysis_zarrs).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively discover Zarrs in filesystem mode.")
    parser.add_argument("--registry", type=Path, help="Registry SQLite path for queue discovery and detect_review sync.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="training",
        help="Queue discovery zarr-use filter (default: training).",
    )
    parser.add_argument(
        "--path-contains",
        type=str,
        default="_training.zarr",
        help="Registry queue substring filter (default: _training.zarr).",
    )
    parser.add_argument(
        "--approved-state",
        default="approved",
        help="Review state considered already approved during discovery (default: approved).",
    )
    parser.add_argument("--queue-file", type=Path, help="Existing text file with one Zarr path per line.")
    parser.add_argument("--queue-output", type=Path, help="Write the discovered queue to this text file.")
    parser.add_argument("--details-output", type=Path, help="Write discovered queue details to this TSV file.")
    parser.add_argument("--state-file", type=Path, help="Resumable JSON state file.")
    parser.add_argument("--resume", action="store_true", help="Skip queue entries already completed in --state-file.")
    parser.add_argument(
        "--resume-skip-states",
        default="approved,rejected,needs_review",
        help="Comma-separated review states to skip when --resume is set.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of queue entries processed.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without launching detect_review.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first detect_review non-zero exit.")

    parser.add_argument("--refined-run", help="Override refined detect run for all entries.")
    parser.add_argument("--variant", choices=["filtered", "interpolated", "refined"], help="Forwarded to detect_review.")
    parser.add_argument("--all", action="store_true", help="Forward --all to detect_review.")
    parser.add_argument("--max-frames", type=int, help="Forward --max-frames to detect_review.")
    parser.add_argument("--frames", help="Forward --frames to detect_review.")
    parser.add_argument("--use-full-res", action="store_true", help="Forward --use-full-res to detect_review.")
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state detect_review should set on approval (default: approved).",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method label forwarded to detect_review.",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Intended-use label forwarded to detect_review.",
    )
    parser.add_argument("--reviewer", help="Reviewer label forwarded to detect_review.")
    parser.add_argument("--review-notes", help="Review notes forwarded to detect_review.")
    parser.add_argument("--no-registry-sync", action="store_true", help="Forward --no-registry-sync to detect_review.")
    parser.add_argument(
        "--skip-detection-profile",
        action="store_true",
        help="Forward --skip-detection-profile to detect_review.",
    )
    parser.add_argument("--overwrite-profile", action="store_true", help="Forward --overwrite-profile to detect_review.")
    parser.add_argument("--no-update-curated", action="store_true", help="Forward --no-update-curated to detect_review.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_id = make_run_id()
    state_path = Path(args.state_file) if args.state_file is not None else _default_state_path(run_id)
    state_payload = _load_state(state_path)
    state_payload["run_id"] = state_payload.get("run_id") or run_id

    queue = _discover_queue(args)
    if args.queue_output is not None:
        _write_queue(queue, Path(args.queue_output))
    if args.details_output is not None:
        _write_details(queue, Path(args.details_output))

    if args.resume:
        skip_states = _parse_state_list(args.resume_skip_states)
        before = len(queue)
        queue = [
            item
            for item in queue
            if not _should_skip_for_resume(item, state_payload=state_payload, skip_states=skip_states)
        ]
        skipped = before - len(queue)
        if skipped:
            print(f"resume: skipped {skipped} queue entr{'y' if skipped == 1 else 'ies'}")

    if args.limit is not None:
        queue = queue[: max(0, int(args.limit))]

    print(f"queue_count: {len(queue)}")
    print(f"state_file: {state_path}")
    if args.queue_output is not None:
        print(f"queue_output: {args.queue_output}")
    if args.details_output is not None:
        print(f"details_output: {args.details_output}")

    failures = 0
    reviewed = 0
    for index, item in enumerate(queue, start=1):
        cmd = _build_detect_review_command(args, item)
        before_status = _read_detect_review_status(
            item.zarr_path,
            refined_run=args.refined_run or item.latest_refined_run,
        )
        _update_state_entry(
            state_path,
            state_payload,
            item.zarr_path,
            queue_index=index,
            queue_count=len(queue),
            command=cmd,
            command_text=_command_for_display(cmd),
            started_at_utc=utc_now(),
            before_status=before_status,
            last_outcome="dry_run" if args.dry_run else "started",
        )
        print(f"\n[{index}/{len(queue)}] {item.zarr_path}")
        print("+ " + _command_for_display(cmd))

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, check=False)
        after_status = _read_detect_review_status(
            item.zarr_path,
            refined_run=args.refined_run or item.latest_refined_run,
        )
        returncode = int(completed.returncode)
        if returncode != 0:
            outcome = "failed"
            failures += 1
        else:
            reviewed += 1
            outcome = after_status.get("review_state") or "no_review_state"
        _update_state_entry(
            state_path,
            state_payload,
            item.zarr_path,
            finished_at_utc=utc_now(),
            returncode=returncode,
            after_status=after_status,
            last_review_state=after_status.get("review_state"),
            last_review_intended_use=after_status.get("review_intended_use"),
            last_review_resolved_group=after_status.get("review_resolved_group"),
            last_outcome=outcome,
        )
        print(
            "result: "
            f"returncode={returncode} "
            f"review_state={after_status.get('review_state') or '-'} "
            f"intended_use={after_status.get('review_intended_use') or '-'}"
        )
        if returncode != 0 and args.stop_on_failure:
            break

    summary = {
        "queue_count": len(queue),
        "reviewed_processes": reviewed,
        "failures": failures,
        "state_file": str(state_path),
        "dry_run": bool(args.dry_run),
    }
    state_payload["summary"] = summary
    _write_state(state_path, state_payload)
    print("\nsummary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
