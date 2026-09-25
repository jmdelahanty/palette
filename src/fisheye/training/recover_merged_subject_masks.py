"""Recover matched masks and seed an extended pose in one training archive.

Each child is atomically imported without changing selectors. A report and
explicit web tasks are released only after every child passes validation.
Existing versions are never overwritten, including partially edited drafts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.recovered_training_review_contract import (
    REVIEW_SCHEMA,
    NATIVE_REVIEW_SCHEMA,
    initial_contract_digest,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import (
    is_run_complete,
    mark_run_complete,
    require_runs_parent,
)
from fisheye.training.materialize_recovered_pose_head_crops import (
    SELECTOR_NAMES,
    _digest_index_sha256,
)
from fisheye.training.recover_merged_training_recording import (
    validate_recovered_recording,
)
from fisheye.training.recovered_mask_review_payload import (
    array_hashes,
    build_review_payload,
    payload_crop_run,
    run_paths,
)
from fisheye.training.recovered_subject_mask_source import (
    read_recovered_mask_source,
    use_refined_mask_snapshot,
)
from fisheye.training.mask_tail_keypoints import (
    LEGACY_SCHEMA_NAME,
    SCHEMA_NAME,
    recipe_for_schema,
)


def validate_initial_payload(path):
    try:
        run = zarr.open_group(str(path), mode="r", use_consolidated=False)
        valid = (
            run.attrs.get("schema_id") in (REVIEW_SCHEMA, NATIVE_REVIEW_SCHEMA)
            and run.attrs.get("stage_selector_eligible") is False
            and is_run_complete(run, legacy_default=False)
            and bool(run.attrs.get("initial_array_sha256"))
            and array_hashes(run) == run.attrs["initial_array_sha256"]
            and initial_contract_digest(run) == run.attrs.get("initial_contract_sha256")
        )
        return {"valid": bool(valid)}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def review_tasks(archive, recording_id, result, version):
    paths = result["paths"]
    count = result.get("keypoint_count", 18)
    snout_note = (
        " Snout tip derived from the body contour; review its placement."
        if count == 19
        else ""
    )
    common = {
        "recording_id": recording_id,
        "zarr_use": "training",
        "state": "pending",
        "dataset_id": f"{recording_id}:recovered_mask_tail:{version}",
    }
    scope = {
        "zarr_path": str(archive),
        "crop_run": payload_crop_run(result),
        "selector_eligible": False,
        "registry_activation": "deferred",
        "review_method": "manual",
        "review_intended_use": "training",
        "auto_advance_on_save": True,
    }
    tasks = [
        {
            **common,
            "task_id": f"{recording_id}-tail{count}-{version}",
            "workflow_kind": "keypoints",
            "stage_group": "refined_keypoints_runs",
            "run_name": paths["pose_edit"].split("/")[1],
            "title": (
                "Review snout and tail; add pectoral fin landmarks (19 points)"
                if count == 19
                else "Add pectoral fin landmarks and review derived tail points"
            ),
            "priority": 80,
            "notes": "Head3 recovered; tail11 derived; fin4 missing. Keep all points inside the crop. Failed rows need mask review or manual landmarks."
            + snout_note,
            "scope": {
                **scope,
                "refined_run": paths["pose_edit"].split("/")[1],
                "include_all": True,
            },
        }
    ]
    failed = [row["roi_idx"] for row in result["failures"]]
    if failed:
        tasks.append(
            {
                **common,
                "task_id": f"{recording_id}-tail-mask-failures-{version}",
                "workflow_kind": "subject_mask_component",
                "component_name": "subject_body",
                "stage_group": "refined_subject_masks_runs",
                "run_name": paths["mask_edit"].split("/")[1],
                "title": "Inspect masks on rows where landmark derivation failed",
                "priority": 90,
                "notes": "Targeted diagnostic queue; completion is not evidence that every mask row was re-reviewed. Original pixels and failure details are preserved. Regenerate a new pose version after mask corrections.",
                "scope": {
                    **scope,
                    "subject_run": paths["mask"].split("/")[1],
                    "refined_run": paths["mask_edit"].split("/")[1],
                    "component_name": "subject_body",
                    "target_roi_indices": failed,
                },
            }
        )
    return tasks


def publish_review_payload(
    archive, local, paths, binding, source_check, *, resume=False
):
    """Publish validated unselected children; source owners supply identity checks."""
    root = zarr.open_group(str(local), mode="r", use_consolidated=False)
    publications = []
    for path in paths.values():
        family, name = path.split("/")
        expected = dict(root[path].attrs)
        target = archive / path
        if target.exists():
            existing = zarr.open_group(str(target), mode="r", use_consolidated=False)
            if (
                not resume
                or not validate_initial_payload(target)["valid"]
                or existing.attrs.get("source_bindings") != binding
                or existing.attrs.get("initial_array_sha256")
                != expected["initial_array_sha256"]
                or existing.attrs.get("initial_contract_sha256")
                != expected["initial_contract_sha256"]
            ):
                raise ValueError(
                    f"Existing version is changed, edited, or conflicting: {path}"
                )
            continue
        snapshot = {}

        def prepare(current):
            source_check(current)
            parent = require_runs_parent(current, family)
            snapshot.update({key: parent.attrs.get(key) for key in SELECTOR_NAMES})
            return current, parent

        def complete(current, parent, run):
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=name,
                run_provenance=run.attrs.get("run_provenance"),
            )

        def verify(current):
            if any(
                current[family].attrs.get(key) != value
                for key, value in snapshot.items()
            ):
                raise RuntimeError("Publication changed an existing stage selector")

        publications.append(
            atomic_publish_run_group(
                AtomicRunPublishSpec(
                    source_zarr=archive,
                    local_run_path=local / path,
                    target_run_path=target,
                    run_name=name,
                    lock_suffix="recovered_subject_masks",
                    publish_schema_id=expected["schema_id"],
                    policy="atomic_unselected_recovery_child_v1",
                    rollback_policy="retain_failed_selector_ineligible_child_v1",
                ),
                copy_backend="python",
                validate_run=validate_initial_payload,
                prepare_parents=prepare,
                complete_run=complete,
                verify_pointers=verify,
                payload_metadata={"source_bindings": binding},
            )
        )
    return publications


def recover_subject_masks(
    *,
    archive: Path,
    merged: Path,
    version: str,
    mask_run="merged_subject_masks",
    crop_run="merged_subject_masks",
    scratch_root=Path("/tmp"),
    apply=False,
    resume=False,
    legacy_unconsolidated_source=False,
    refined_mask_run=None,
    pose_schema=SCHEMA_NAME,
):
    recipe_for_schema(pose_schema)
    archive, merged = archive.resolve(), merged.resolve()
    paths = run_paths(version)
    if not resume and any((archive / path).exists() for path in paths.values()):
        raise FileExistsError(
            "Recovery version already exists; choose a new version or validate with --resume"
        )
    arrays, labels, binding = read_recovered_mask_source(
        archive,
        merged,
        mask_run=mask_run,
        crop_run=crop_run,
        legacy_unconsolidated_source=legacy_unconsolidated_source,
    )
    if refined_mask_run is not None:
        arrays, binding = use_refined_mask_snapshot(
            archive, refined_mask_run, arrays, labels, binding
        )
    if not apply:
        return {
            "status": "planned",
            "recording_id": binding["recording_id"],
            "row_count": len(arrays["roi_images"]),
            "paths": paths,
            "source_bindings": binding,
            "pose_schema": pose_schema,
        }
    with tempfile.TemporaryDirectory(
        prefix="palette-mask-recovery-", dir=scratch_root
    ) as temp:
        local = Path(temp) / "review.zarr"
        root = zarr.open_group(
            str(local), mode="w", zarr_format=3, use_consolidated=False
        )
        root.attrs.update(
            {"zarr_purpose": "training", "recording_id": binding["recording_id"]}
        )
        result = build_review_payload(
            root, arrays, labels, binding, version=version, pose_schema=pose_schema
        )

        def check_source(current):
            if (
                current.attrs.get("array_sha256") is None
                or _digest_index_sha256(current.attrs["array_sha256"])
                != binding["recovery_source_digest_index_sha256"]
            ):
                raise ValueError("Recovered source identity changed during publication")

        publications = publish_review_payload(
            archive, local, paths, binding, check_source, resume=resume
        )
    # Original recovery content is validated again; its digest grammar and all
    # existing head-crop/source arrays remain unchanged.
    validate_recovered_recording(archive, require_source_only=True)
    with archive_metadata_publication_lock(archive):
        consolidate_metadata_capture_expected_warnings(archive)
    published = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    for path in paths.values():
        if (
            path not in published
            or not validate_initial_payload(archive / path)["valid"]
        ):
            raise ValueError(f"Incomplete recovery publication: {path}")
    return {
        "status": "recovered",
        "archive": str(archive),
        "version": version,
        "recording_id": binding["recording_id"],
        "source_bindings": binding,
        **result,
        "publications": publications,
        "tasks": review_tasks(archive, binding["recording_id"], result, version),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--mask-run", default="merged_subject_masks")
    parser.add_argument("--crop-run", default="merged_subject_masks")
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pose-schema",
        choices=(SCHEMA_NAME, LEGACY_SCHEMA_NAME),
        default=SCHEMA_NAME,
        help="Pose schema; the default includes snout_tip. v1 is legacy compatibility only.",
    )
    parser.add_argument(
        "--refined-mask-run",
        help="Derive a new version from previously corrected dense masks",
    )
    parser.add_argument(
        "--legacy-unconsolidated-source",
        action="store_true",
        help="Explicit archaeology mode for historical exports without consolidated metadata",
    )
    args = vars(parser.parse_args(argv))
    report = args.pop("report")
    if report.exists():
        raise FileExistsError(report)
    result = recover_subject_masks(**args)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("status", "recording_id", "row_count")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
