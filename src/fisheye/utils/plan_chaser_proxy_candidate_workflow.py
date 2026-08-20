"""Plan or submit the selector-ineligible chaser proxy candidate LSF DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Sequence

from fisheye.cluster.chaser_proxy_candidate import (
    build_chaser_proxy_candidate_workflow,
)
from fisheye.cluster.lsf import (
    LsfResources,
    build_ssh_bsub_runner,
    submit_lsf_workflow,
)
from fisheye.shared.json_safety import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--palette-repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run-name", required=True)
    parser.add_argument("--proxy-run-name", required=True)
    parser.add_argument("--relative-frame-run-name", required=True)
    parser.add_argument("--analysis-profile", type=Path, required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--queue", default="short")
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument("--mem-gb", type=int, default=8)
    parser.add_argument("--walltime", default="1:00")
    parser.add_argument("--submit-host")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip().lower()
    if len(commit) != 40:
        raise ValueError("Palette checkout did not resolve one full commit SHA.")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return commit, not bool(status.strip())


def run(args: argparse.Namespace) -> dict[str, object]:
    repo = args.palette_repo.expanduser().resolve()
    palette_commit, palette_repo_clean = _git_state(repo)
    resources = LsfResources(
        queue=args.queue,
        ncores=args.ncores,
        mem_gb=args.mem_gb,
        gpus=0,
        walltime=args.walltime,
        span_hosts=1,
    )
    plan = build_chaser_proxy_candidate_workflow(
        workflow_id=args.workflow_id,
        repo=repo,
        run_root=args.run_root,
        analysis_zarr=args.analysis_zarr,
        source_run_name=args.source_run_name,
        proxy_run_name=args.proxy_run_name,
        relative_frame_run_name=args.relative_frame_run_name,
        analysis_profile_path=args.analysis_profile,
        palette_commit=palette_commit,
        expected_recording_id=args.expected_recording_id,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        resources=resources,
    )
    payload = plan.to_json()
    if not args.submit:
        return {
            **payload,
            "palette_repo_clean": palette_repo_clean,
            "status": "planned_no_submission",
        }
    if not palette_repo_clean:
        raise RuntimeError(
            "Candidate LSF submission requires a clean commit-pinned deployment."
        )
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    for child in ("logs", "status"):
        (run_root / child).mkdir()
    plan_path = run_root / "lsf_plan.json"
    write_json_atomic(plan_path, plan.workflow.to_json())
    submission = submit_lsf_workflow(
        plan.workflow,
        cwd=repo,
        plan_path=plan_path,
        submission_path=run_root / "lsf_submission.json",
        runner=(
            build_ssh_bsub_runner(args.submit_host)
            if str(args.submit_host or "").strip()
            else subprocess.run
        ),
    )
    return {
        **payload,
        "status": "submitted_selector_ineligible",
        "plan_path": str(plan_path),
        "palette_repo_clean": True,
        "submission": submission,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, sort_keys=True if args.json else False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
