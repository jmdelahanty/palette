"""Summarize a clipped detect/refine LSF submission manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from fisheye.utils.submit_clipped_detect_refine_plan_bsub import SUBMISSION_SCHEMA


CHECK_SCHEMA = "palette.clipped_detect_refine_submission_check.v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_has_placeholder(path: str) -> bool:
    return "<" in path or ">" in path


def _job_is_placeholder(job_id: Any) -> bool:
    return _path_has_placeholder(str(job_id or ""))


def _derive_detect_paths(tarball_path: Path, job_id: str) -> dict[str, str]:
    name = tarball_path.name
    if name.endswith(".tar.gz"):
        stem = name[: -len(".tar.gz")]
    else:
        stem = tarball_path.stem
    return {
        "summary_json": str(tarball_path.with_name(f"{stem}.summary.json")),
        "transfer_json": str(tarball_path.with_name(f"{stem}.transfer.json")),
        "stdout": str(tarball_path.parent / f"{job_id}.out") if job_id else "",
        "stderr": str(tarball_path.parent / f"{job_id}.err") if job_id else "",
    }


def _file_record(path_value: Any) -> dict[str, Any]:
    path_text = str(path_value or "")
    record: dict[str, Any] = {"path": path_text, "exists": False}
    if not path_text or _path_has_placeholder(path_text):
        return record
    path = Path(path_text)
    record["exists"] = path.exists()
    if path.exists():
        try:
            record["size_bytes"] = path.stat().st_size
        except OSError:
            pass
    return record


def _load_status_payload(path_value: Any) -> tuple[dict[str, Any] | None, str | None]:
    path_text = str(path_value or "")
    if not path_text:
        return None, "missing status path"
    if _path_has_placeholder(path_text):
        return None, "placeholder status path"
    path = Path(path_text)
    if not path.exists():
        return None, "missing status JSON"
    try:
        payload = _read_json(path)
    except Exception as exc:
        return None, f"unreadable status JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "status JSON is not an object"
    return payload, None


def _stage_report(stage: Mapping[str, Any], *, work_unit_id: str | None = None) -> dict[str, Any]:
    stage_name = str(stage.get("stage") or "unknown")
    job_id = str(stage.get("job_id") or "")
    status_path = str(stage.get("status_json") or "")
    report: dict[str, Any] = {
        "kind": "stage",
        "stage": stage_name,
        "work_unit_id": work_unit_id,
        "job_id": job_id,
        "dependency": stage.get("dependency"),
        "script": str(stage.get("script") or ""),
        "status_json": status_path,
        "stdout": _file_record(stage.get("stdout")),
        "stderr": _file_record(stage.get("stderr")),
    }
    if _job_is_placeholder(job_id):
        report.update({"status": "planned", "reason": "placeholder job id"})
        return report

    payload, error = _load_status_payload(status_path)
    if error is not None:
        report.update({"status": "missing", "reason": error})
        return report
    assert payload is not None
    status = str(payload.get("status") or "")
    report.update(
        {
            "status": status if status else "invalid",
            "exit_code": payload.get("exit_code"),
            "host": payload.get("host"),
            "queue": payload.get("queue"),
            "stage_seconds": payload.get("stage_seconds"),
            "started_at_utc": payload.get("started_at_utc"),
            "finished_at_utc": payload.get("finished_at_utc"),
        }
    )
    if status not in {"ok", "failed"}:
        report["reason"] = f"unexpected status {status!r}"
    return report


def _detect_artifact_report(item: Mapping[str, Any]) -> dict[str, Any]:
    job_id = str(item.get("detect_job_id") or "")
    tarball_path = str(item.get("detect_artifact_tarball") or "")
    report: dict[str, Any] = {
        "kind": "detect_artifact",
        "stage": "detect_artifact",
        "work_unit_id": item.get("work_unit_id"),
        "clip_id": item.get("clip_id"),
        "camera_serial": item.get("camera_serial"),
        "job_id": job_id,
        "tarball": _file_record(tarball_path),
    }
    if _job_is_placeholder(job_id):
        report.update({"status": "planned", "reason": "placeholder job id"})
        return report

    derived_paths = _derive_detect_paths(Path(tarball_path), job_id) if tarball_path else {}
    report["summary_json"] = _file_record(derived_paths.get("summary_json"))
    report["transfer_json"] = _file_record(derived_paths.get("transfer_json"))
    report["stdout"] = _file_record(derived_paths.get("stdout"))
    report["stderr"] = _file_record(derived_paths.get("stderr"))

    summary_path = str(derived_paths.get("summary_json") or "")
    if not summary_path or not Path(summary_path).exists():
        report.update({"status": "missing", "reason": "missing detect artifact summary JSON"})
        return report
    try:
        summary = _read_json(Path(summary_path))
    except Exception as exc:
        report.update({"status": "invalid", "reason": f"unreadable detect artifact summary JSON: {exc}"})
        return report
    if not isinstance(summary, dict):
        report.update({"status": "invalid", "reason": "detect artifact summary JSON is not an object"})
        return report
    status = str(summary.get("status") or "")
    report.update(
        {
            "status": status if status else "invalid",
            "run_name": summary.get("run_name"),
            "target_group_path": summary.get("target_group_path"),
            "intended_target_group_path": summary.get("intended_target_group_path"),
            "artifact_scope": summary.get("artifact_scope"),
            "artifact_timing": summary.get("artifact_timing"),
        }
    )
    if status != "ok":
        report["reason"] = f"detect artifact status {status!r}"
    elif not bool(report["tarball"].get("exists")):
        report.update({"status": "missing", "reason": "missing detect artifact tarball"})
    return report


def _finalizer_report(finalizer: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not finalizer:
        return None
    return _stage_report(finalizer, work_unit_id="recording_collection")


def check_clipped_detect_refine_submission(
    submission_manifest: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(submission_manifest).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Submission manifest must be a JSON object: {manifest_path}")
    if manifest.get("schema_version") != SUBMISSION_SCHEMA:
        raise ValueError(
            f"Expected {SUBMISSION_SCHEMA}, got {manifest.get('schema_version')!r}: {manifest_path}"
        )

    reports: list[dict[str, Any]] = []
    work_item_reports: list[dict[str, Any]] = []
    for item in manifest.get("work_items", []):
        if not isinstance(item, Mapping):
            continue
        work_unit_id = str(item.get("work_unit_id") or "")
        detect_report = _detect_artifact_report(item)
        stage_reports = [
            _stage_report(stage, work_unit_id=work_unit_id)
            for stage in item.get("stages", [])
            if isinstance(stage, Mapping)
        ]
        reports.append(detect_report)
        reports.extend(stage_reports)
        work_item_reports.append(
            {
                "work_unit_id": work_unit_id,
                "clip_id": item.get("clip_id"),
                "camera_serial": item.get("camera_serial"),
                "detect_artifact": detect_report,
                "stages": stage_reports,
            }
        )

    finalizer = _finalizer_report(manifest.get("finalizer") if isinstance(manifest.get("finalizer"), Mapping) else None)
    if finalizer is not None:
        reports.append(finalizer)

    counts = Counter(str(report.get("status") or "unknown") for report in reports)
    failed_like = {"failed", "invalid", "unreadable"}
    incomplete_like = {"missing", "planned", "pending", "unknown"}
    if any(counts.get(status, 0) for status in failed_like):
        status = "failed"
    elif any(counts.get(status, 0) for status in incomplete_like):
        status = "incomplete"
    else:
        status = "ok"

    return {
        "status": status,
        "schema_version": CHECK_SCHEMA,
        "submission_manifest": str(manifest_path),
        "workflow_id": manifest.get("workflow_id"),
        "manifest_status": manifest.get("status"),
        "run_dir": manifest.get("run_dir"),
        "work_unit_count": manifest.get("work_unit_count"),
        "stage_count": len(reports),
        "status_counts": dict(sorted(counts.items())),
        "work_items": work_item_reports,
        "finalizer": finalizer,
    }


def _iter_flat_reports(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for item in payload.get("work_items", []):
        if not isinstance(item, Mapping):
            continue
        detect = item.get("detect_artifact")
        if isinstance(detect, Mapping):
            yield detect
        for stage in item.get("stages", []):
            if isinstance(stage, Mapping):
                yield stage
    finalizer = payload.get("finalizer")
    if isinstance(finalizer, Mapping):
        yield finalizer


def _format_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}s"
    return "-"


def _format_file_record(label: str, value: Any) -> str:
    if not isinstance(value, Mapping):
        return f"{label}=missing"
    path = str(value.get("path") or "")
    if not path:
        return f"{label}=missing"
    exists = bool(value.get("exists"))
    size = value.get("size_bytes")
    if exists and isinstance(size, int):
        return f"{label}={size}B"
    if exists:
        return f"{label}=exists"
    return f"{label}=missing"


def _print_human(payload: Mapping[str, Any]) -> None:
    print(f"submission_manifest: {payload.get('submission_manifest')}")
    print(f"workflow_id: {payload.get('workflow_id')}")
    print(f"status: {payload.get('status')}")
    print(f"manifest_status: {payload.get('manifest_status')}")
    print(f"run_dir: {payload.get('run_dir')}")
    print(f"status_counts: {payload.get('status_counts')}")
    print("")
    for report in _iter_flat_reports(payload):
        work_unit_id = report.get("work_unit_id") or "-"
        stage = report.get("stage") or report.get("kind")
        status = report.get("status")
        job_id = report.get("job_id") or "-"
        seconds = _format_seconds(report.get("stage_seconds"))
        reason = report.get("reason")
        line = f"{status:>10}  {stage:<28} job={job_id:<12} t={seconds:<8} {work_unit_id}"
        if reason:
            line += f"  reason={reason}"
        if status in {"missing", "failed", "invalid"}:
            log_bits = [
                _format_file_record("stdout", report.get("stdout")),
                _format_file_record("stderr", report.get("stderr")),
            ]
            if report.get("kind") == "detect_artifact":
                log_bits.extend(
                    [
                        _format_file_record("summary", report.get("summary_json")),
                        _format_file_record("tarball", report.get("tarball")),
                    ]
                )
            line += "  " + " ".join(log_bits)
        print(line)
    if payload.get("status") == "ok":
        print("\nsubmission_check=ok")
    elif payload.get("status") == "incomplete":
        print("\nsubmission_check=incomplete")
    else:
        print("\nsubmission_check=failed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a clipped detect/refine LSF submission manifest."
    )
    parser.add_argument("submission_manifest", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every stage is complete and ok.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = check_clipped_detect_refine_submission(args.submission_manifest)
    except Exception as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    if payload["status"] == "failed":
        return 1
    if args.require_complete and payload["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
