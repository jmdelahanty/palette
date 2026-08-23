"""Plan or publish one exact detection-centroid subject-position run.

The source must be an explicitly named selector-ineligible canonical-v3
detection publication.  Planning performs the complete source binding and
freezes the publication-attempt UUID, so a cohort task can bind the expected
subject-position manifest before it submits any write.  Existing output is
reused only when that exact manifest still validates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fisheye.analysis_workflows.materializers.subject_position import (
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    load_subject_position_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.subject_position_detection_source import (
    load_persisted_selector_ineligible_detection_position_source,
)
from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
)
from fisheye.shared.subject_position_preparation import prepare_subject_position_input

RESULT_SCHEMA_ID = "palette.detection_subject_position_materialization"
RESULT_SCHEMA_VERSION = 1


def _software_record(*, palette_commit: str, workflow_id: str) -> dict[str, str]:
    commit = str(palette_commit or "").strip().lower()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise ValueError("palette_commit must be one full lowercase Git SHA.")
    workflow = str(workflow_id or "").strip()
    if not workflow or workflow != workflow_id:
        raise ValueError("workflow_id must be one nonempty exact string.")
    return {"package": "palette", "commit": commit, "workflow": workflow}


def build_plan(
    analysis_zarr: str | Path,
    *,
    source_run_path: str,
    output_run_name: str,
    scratch_root: str | Path,
    publication_attempt_uuid: str,
    palette_commit: str,
    workflow_id: str,
) -> Any:
    """Build the exact read-only subject-position publication plan."""

    archive = Path(analysis_zarr).expanduser().resolve()
    source = load_persisted_selector_ineligible_detection_position_source(
        archive,
        source_run_path,
    )
    prepared = prepare_subject_position_input(
        source,
        estimator_id=DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
        software_record=_software_record(
            palette_commit=palette_commit,
            workflow_id=workflow_id,
        ),
    )
    return plan_subject_position_run(
        archive,
        prepared,
        run_name=output_run_name,
        scratch_root=scratch_root,
        publication_attempt_uuid=publication_attempt_uuid,
    )


def execute(
    analysis_zarr: str | Path,
    *,
    source_run_path: str,
    output_run_name: str,
    scratch_root: str | Path,
    publication_attempt_uuid: str,
    palette_commit: str,
    workflow_id: str,
    expected_manifest_sha256: str | None,
    apply: bool,
) -> dict[str, Any]:
    archive = Path(analysis_zarr).expanduser().resolve()
    run_path = f"analysis/subject_position_runs/observation/{output_run_name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        if expected_manifest_sha256 is None:
            raise ValueError(
                "Existing output can be reused only with --expected-manifest-sha256."
            )
        handle = load_subject_position_source_handle(
            archive,
            run_path,
            expected_selector_eligible=False,
            expected_manifest_sha256=expected_manifest_sha256,
            use_consolidated=True,
        )
        if (
            handle.estimator_record.get("estimator_id")
            != DETECTION_BBOX_CENTROID_ESTIMATOR_ID
        ):
            raise ValueError("Existing output uses another position estimator.")
        return {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "reused_exact" if apply else "planned_reuse_no_writes",
            "analysis_zarr": str(archive),
            "source_run_path": source_run_path,
            "run_path": run_path,
            "manifest_sha256": handle.manifest_sha256,
            "selector_eligible": False,
            "writes": False,
        }

    plan = build_plan(
        archive,
        source_run_path=source_run_path,
        output_run_name=output_run_name,
        scratch_root=scratch_root,
        publication_attempt_uuid=publication_attempt_uuid,
        palette_commit=palette_commit,
        workflow_id=workflow_id,
    )
    manifest_sha256 = plan.final_manifest_sha256
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("Planned subject-position manifest differs from expectation.")
    if not apply:
        return {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "planned_no_writes",
            "analysis_zarr": str(archive),
            "source_run_path": source_run_path,
            "run_path": plan.run_path,
            "manifest_sha256": manifest_sha256,
            "publication_attempt_uuid": plan.publication_attempt_uuid,
            "selector_eligible": False,
            "writes": False,
            "plan": plan.as_dict(),
        }
    publication = publish_subject_position_run(plan, keep_scratch=False)
    handle = load_subject_position_source_handle(
        archive,
        plan.run_path,
        expected_selector_eligible=False,
        expected_manifest_sha256=manifest_sha256,
        use_consolidated=True,
    )
    return {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "published_selector_ineligible",
        "analysis_zarr": str(archive),
        "source_run_path": source_run_path,
        "run_path": plan.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "publication_attempt_uuid": plan.publication_attempt_uuid,
        "selector_eligible": False,
        "writes": True,
        "publication": publication,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--source-run-path", required=True)
    parser.add_argument("--output-run-name", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--publication-attempt-uuid", required=True)
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = execute(
        args.analysis_zarr,
        source_run_path=args.source_run_path,
        output_run_name=args.output_run_name,
        scratch_root=args.scratch_root,
        publication_attempt_uuid=args.publication_attempt_uuid,
        palette_commit=args.palette_commit,
        workflow_id=args.workflow_id,
        expected_manifest_sha256=args.expected_manifest_sha256,
        apply=args.apply,
    )
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    print(
        json.dumps(
            result,
            sort_keys=True if args.json else False,
            indent=None if args.json else 2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_plan", "execute", "main"]
