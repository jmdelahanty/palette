"""Freeze one browser geometry decision and optionally submit its LSF DAG."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.geometry_review_approval import (
    build_geometry_review_approval_workflow,
)
from fisheye.cluster.lsf import build_ssh_bsub_runner, submit_lsf_workflow
from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.registry.geometry_review_approval import (
    build_geometry_review_approval_request,
    persist_geometry_review_approval_request,
    verify_geometry_review_registry_precondition,
)


def _git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        raise ValueError(f"Cannot resolve Palette commit: {commit.stderr.strip()}")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise ValueError(f"Cannot inspect Palette worktree: {status.stderr.strip()}")
    return commit.stdout.strip().lower(), not bool(status.stdout.strip())


def _existing_submission(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Existing submission receipt is unreadable: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Existing submission receipt is malformed: {path}")
    return payload


def _durable_approval_root(
    value: str | Path,
    *,
    analysis_zarr: str | Path,
) -> Path:
    root = Path(value).expanduser().resolve()
    archive = Path(analysis_zarr).expanduser().resolve()
    if any(part.lower() == "staging" for part in root.parts):
        raise ValueError(
            "Geometry approval state must use a durable operations root outside staging."
        )
    if root == archive or archive in root.parents or root in archive.parents:
        raise ValueError(
            "Geometry approval state must remain outside the canonical analysis Zarr."
        )
    return root


def prepare_geometry_review_approval_submission(
    *,
    registry_path: str | Path,
    dataset_id: str,
    recording_id: str,
    analysis_zarr: str | Path,
    fit_review_run: str,
    acquisition_candidate_run: str,
    source_detection_group_path: str,
    selected_candidate_kind: str,
    semantic_compatibility: str,
    reviewer: str,
    reviewed_at_utc: str,
    decision_reason: str,
    palette_repo: str | Path,
    approval_root: str | Path,
    submit: bool = False,
    submit_host: str | None = None,
    required_ci_success: bool = False,
) -> dict[str, Any]:
    """Persist an immutable request and optionally submit its complete DAG."""

    repo = Path(palette_repo).expanduser().resolve()
    commit, repo_clean = _git_state(repo)
    if submit and not repo_clean:
        raise RuntimeError(
            "LSF approval submission requires a clean Palette deployment."
        )
    if submit and not required_ci_success:
        raise RuntimeError(
            "LSF approval submission requires explicit successful required-CI evidence."
        )
    request = build_geometry_review_approval_request(
        registry_path=registry_path,
        dataset_id=dataset_id,
        recording_id=recording_id,
        analysis_zarr=analysis_zarr,
        fit_review_run=fit_review_run,
        acquisition_candidate_run=acquisition_candidate_run,
        source_detection_group_path=source_detection_group_path,
        selected_candidate_kind=selected_candidate_kind,
        semantic_compatibility=semantic_compatibility,
        reviewer=reviewer,
        reviewed_at_utc=reviewed_at_utc,
        decision_reason=decision_reason,
        palette_commit=commit,
    )
    verify_geometry_review_registry_precondition(request)
    root = _durable_approval_root(
        approval_root,
        analysis_zarr=analysis_zarr,
    )
    request_path = persist_geometry_review_approval_request(
        request, request_root=root / "requests"
    )
    run_root = root / "runs" / request.request_id
    plan = build_geometry_review_approval_workflow(
        request,
        request_path=request_path,
        palette_repo=repo,
        run_root=run_root,
    )
    approval_plan_path = run_root / "approval_plan.json"
    lsf_plan_path = run_root / "lsf_plan.json"
    submission_path = run_root / "lsf_submission.json"
    write_json_snapshot(approval_plan_path, plan.to_json())
    result: dict[str, Any] = {
        "schema_id": "palette.geometry_review_approval_submission",
        "schema_version": 1,
        "status": "planned",
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "request_path": str(request_path),
        "approval_plan_path": str(approval_plan_path),
        "lsf_plan_path": str(lsf_plan_path),
        "submission_path": str(submission_path),
        "palette_repo": str(repo),
        "palette_commit": commit,
        "palette_repo_clean": repo_clean,
        "required_ci_success": bool(required_ci_success),
        "jobs_submitted": False,
        "workflow": plan.workflow.to_json(),
    }
    if not submit:
        return result
    existing = _existing_submission(submission_path)
    if existing is not None:
        if (
            existing.get("status") == "submitted"
            and existing.get("workflow_id") == request.request_id
        ):
            result.update(
                {
                    "status": "already_submitted",
                    "jobs_submitted": False,
                    "submission": dict(existing),
                }
            )
            return result
        raise RuntimeError(
            "Approval has an incomplete or failed prior LSF submission; refusing "
            "automatic duplicate submission."
        )
    runner = (
        build_ssh_bsub_runner(submit_host)
        if str(submit_host or "").strip()
        else subprocess.run
    )
    submission = submit_lsf_workflow(
        plan.workflow,
        cwd=repo,
        plan_path=lsf_plan_path,
        submission_path=submission_path,
        runner=runner,
    )
    result.update(
        {
            "status": "submitted",
            "jobs_submitted": True,
            "submission": submission,
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--fit-review-run", required=True)
    parser.add_argument("--acquisition-candidate-run", required=True)
    parser.add_argument("--source-detection-group", required=True)
    parser.add_argument("--select", choices=("palette", "acquisition"), required=True)
    parser.add_argument(
        "--semantic-compatibility",
        choices=(
            "same_feature_confirmed",
            "different_feature_confirmed",
            "projected_edges_unresolved",
        ),
        required=True,
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at-utc", required=True)
    parser.add_argument("--decision-reason", required=True)
    parser.add_argument("--palette-repo", type=Path, required=True)
    parser.add_argument("--approval-root", type=Path, required=True)
    parser.add_argument("--submit-host")
    parser.add_argument("--required-ci-success", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare_geometry_review_approval_submission(
        registry_path=args.registry,
        dataset_id=args.dataset_id,
        recording_id=args.recording_id,
        analysis_zarr=args.analysis_zarr,
        fit_review_run=args.fit_review_run,
        acquisition_candidate_run=args.acquisition_candidate_run,
        source_detection_group_path=args.source_detection_group,
        selected_candidate_kind=args.select,
        semantic_compatibility=args.semantic_compatibility,
        reviewer=args.reviewer,
        reviewed_at_utc=args.reviewed_at_utc,
        decision_reason=args.decision_reason,
        palette_repo=args.palette_repo,
        approval_root=args.approval_root,
        submit=bool(args.submit),
        submit_host=args.submit_host,
        required_ci_success=bool(args.required_ci_success),
    )
    if args.result_json is not None:
        write_json_snapshot(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
