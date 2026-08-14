"""Apply one frozen geometry-review decision through the exact keyed gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    publish_arena_geometry_candidate,
)
from fisheye.analysis_workflows.materializers.arena_geometry_comparison import (
    build_arena_geometry_comparison_plan,
    publish_arena_geometry_comparison,
)
from fisheye.analysis_workflows.materializers.arena_geometry_selection import (
    build_arena_geometry_selection_plan,
    publish_arena_geometry_selection,
)
from fisheye.analysis_workflows.materializers.registered_detection_gate import (
    build_registered_detection_gate_plan,
    publish_registered_detection_gate,
)
from fisheye.registry.geometry_review_approval import (
    GeometryReviewApprovalError,
    GeometryReviewApprovalRequest,
    detection_source_binding,
    load_geometry_review_approval_request,
    revalidate_geometry_review_approval_sources,
    verify_geometry_review_registry_precondition,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr_io import open_zarr_root

RESULT_SCHEMA_ID = "palette.geometry_review_approval_result"
RESULT_SCHEMA_VERSION = 1


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise GeometryReviewApprovalError(
            f"Cannot resolve Palette repository commit: {result.stderr.strip()}"
        )
    return result.stdout.strip().lower()


def _conflicting_selection(
    request: GeometryReviewApprovalRequest,
    *,
    expected_selection_run: str,
) -> str | None:
    root = open_zarr_root(request.analysis_zarr, mode="r", use_consolidated=False)
    try:
        parent = root["analysis/arena_geometry_selection"]
    except KeyError:
        return None
    for attr in ("latest", "latest_complete"):
        current = str(parent.attrs.get(attr) or "").strip()
        if current and current != expected_selection_run:
            return current
    return None


def apply_geometry_review_approval(
    request: GeometryReviewApprovalRequest,
    *,
    palette_repo: str | Path,
    scratch_root: str | Path,
    apply: bool,
) -> dict[str, Any]:
    """Revalidate and idempotently publish candidate, comparison, selection, gate."""

    repo = Path(palette_repo).expanduser().resolve()
    expected_commit = request.payload["identity"]["execution"]["palette_commit"]
    observed_commit = _git_commit(repo)
    if observed_commit != expected_commit:
        raise GeometryReviewApprovalError(
            "Approval job Palette commit differs from the frozen request: "
            f"{observed_commit} != {expected_commit}."
        )
    verify_geometry_review_registry_precondition(request)
    palette_plan = revalidate_geometry_review_approval_sources(request)
    identity = request.payload["identity"]
    evidence = identity["evidence"]
    decision = identity["decision"]
    pipeline = request.payload["pipeline"]
    scratch = Path(scratch_root).expanduser().resolve()
    result: dict[str, Any] = {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "validated",
        "mode": "apply" if apply else "dry_run",
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "palette_commit": observed_commit,
        "analysis_zarr": str(request.analysis_zarr),
        "fit_review_run": evidence["fit_review"]["run_name"],
        "acquisition_candidate_run": evidence["acquisition_candidate"]["run_name"],
        "palette_candidate_run": palette_plan.candidate_id,
        "palette_candidate_record_sha256": palette_plan.candidate_record_sha256,
        "selected_candidate_kind": decision["selected_candidate_kind"],
        "gate_run": pipeline["gate_run"],
        "raw_detections_mutated": False,
    }
    if not apply:
        result["status"] = "dry_run_validated_prepublication"
        return result

    scratch.mkdir(parents=True, exist_ok=True)
    candidate_result = publish_arena_geometry_candidate(
        palette_plan,
        scratch_root=scratch / "palette_candidate",
        copy_backend="python",
    )
    comparison_plan = build_arena_geometry_comparison_plan(
        request.analysis_zarr,
        acquisition_candidate_run=evidence["acquisition_candidate"]["run_name"],
        palette_candidate_run=palette_plan.candidate_id,
        semantic_compatibility=decision["semantic_compatibility"],
        policy_id=decision["comparison_policy_id"],
        semantic_review={
            "reviewer": decision["reviewer"],
            "reviewed_at_utc": decision["reviewed_at_utc"],
            "evidence_reason": decision["decision_reason"],
        },
        detect_source_group_path=identity["detection_source"]["group_path"],
    )
    comparison_result = publish_arena_geometry_comparison(
        comparison_plan,
        scratch_root=scratch / "comparison",
        copy_backend="python",
    )
    selected_run = (
        palette_plan.candidate_id
        if decision["selected_candidate_kind"] == "palette"
        else evidence["acquisition_candidate"]["run_name"]
    )
    selection_plan = build_arena_geometry_selection_plan(
        request.analysis_zarr,
        candidate_run=selected_run,
        selected_by=decision["reviewer"],
        decision_reason=decision["decision_reason"],
        decision_source="manual_review",
        comparison_run=comparison_plan.comparison_id,
    )
    conflict = _conflicting_selection(
        request, expected_selection_run=selection_plan.selection_id
    )
    if conflict is not None:
        raise GeometryReviewApprovalError(
            "A different geometry selection is already active; refusing browser "
            f"override: {conflict}."
        )
    selection_result = publish_arena_geometry_selection(
        selection_plan,
        scratch_root=scratch / "selection",
        copy_backend="python",
    )
    gate_plan = build_registered_detection_gate_plan(
        request.analysis_zarr,
        source_group_path=identity["detection_source"]["group_path"],
        selection_run=selection_plan.selection_id,
        output_run=pipeline["gate_run"],
    )
    gate_result = publish_registered_detection_gate(
        gate_plan,
        scratch_root=scratch / "gate",
        copy_backend="python",
    )
    root = open_zarr_root(request.analysis_zarr, mode="r", use_consolidated=False)
    current_detection = detection_source_binding(
        root, identity["detection_source"]["group_path"]
    )
    if current_detection != identity["detection_source"]:
        raise RuntimeError("Detection source changed during approval publication.")
    result.update(
        {
            "status": "complete",
            "candidate_publication": candidate_result,
            "comparison_publication": comparison_result,
            "selection_publication": selection_result,
            "gate_publication": gate_result,
            "comparison_run": comparison_plan.comparison_id,
            "comparison_record_sha256": comparison_plan.comparison_record_sha256,
            "selection_run": selection_plan.selection_id,
            "selection_record_sha256": selection_plan.selection_record_sha256,
            "selected_candidate_run": selected_run,
            "selected_candidate_record_sha256": (
                palette_plan.candidate_record_sha256
                if decision["selected_candidate_kind"] == "palette"
                else evidence["acquisition_candidate"]["candidate_record_sha256"]
            ),
            "gate_run": gate_plan.output_run,
            "gate_group_path": (f"analysis/detection_gate_runs/{gate_plan.output_run}"),
            "source_detection_binding_unchanged": True,
            "raw_detections_mutated": False,
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--palette-repo", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = load_geometry_review_approval_request(args.request_json)
    result = apply_geometry_review_approval(
        request,
        palette_repo=args.palette_repo,
        scratch_root=args.scratch_root,
        apply=bool(args.apply),
    )
    write_json_atomic(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
