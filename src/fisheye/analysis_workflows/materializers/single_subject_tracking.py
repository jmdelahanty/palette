"""Atomic selector-ineligible tracking from one exact subject-position rowset.

This materializer is intentionally narrow: one recording contains one subject
in one arena, every observation row retains its exact ``instance_key`` and
acquisition frame identity, and the resulting run is never made selectable.
It exists to let provider-motion canaries consume the same explicit tracking
authority as production workflows without mutating historical tracking
selectors or pretending that a subject-position run is a detection run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Mapping
import uuid

import numpy as np
import zarr

from fisheye.analysis_workflows.subject_position_source_handle import (
    SubjectPositionSourceHandle,
    load_subject_position_source_handle,
    require_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    load_tracking_source_handle,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.rowset_fingerprint import build_rowset_fingerprint
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    require_runs_parent,
)
from fisheye.tracking.run_manifest import (
    TRACKING_RUN_MANIFEST_ATTR,
    TRACKING_RUN_MANIFEST_DIGEST_ATTR,
    tracking_array_records,
    validate_tracking_run_manifest,
)
from fisheye.tracking.single_subject_per_arena import (
    TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,
    build_single_subject_per_arena_tracking,
    write_single_subject_per_arena_tracking_run,
)


SINGLE_SUBJECT_TRACKING_PUBLISH_SCHEMA_ID = (
    "palette.single_subject_tracking_run_publish"
)
SINGLE_SUBJECT_TRACKING_PUBLISH_SCHEMA_VERSION = 1
SINGLE_SUBJECT_TRACKING_PUBLISH_POLICY = (
    "subject_position_bound_atomic_nonpromoting_v1"
)
SINGLE_SUBJECT_TRACKING_RETRY_POLICY = "new_immutable_run_name_required"
TRACKING_PARENT_PATH = "tracking_runs"

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "authoritative_run_provenance",
)


def _safe_run_name(value: str) -> str:
    if (
        type(value) is not str
        or _RUN_NAME_RE.fullmatch(value) is None
        or value in {"latest", "latest_complete", "latest_pending", "authoritative_run"}
    ):
        raise ValueError(f"Invalid tracking run name: {value!r}.")
    return value


def _selector_attrs(parent: Any | None) -> dict[str, Any]:
    if parent is None:
        return {}
    return {
        name: json_attr_safe(parent.attrs[name])
        for name in _SELECTOR_ATTRS
        if name in parent.attrs
    }


def _readonly(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C").reshape(-1)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SingleSubjectTrackingRunPlan:
    """Exact immutable publication inputs for one non-promoting run."""

    source_zarr: Path
    source_position_run: str
    source_position_manifest_sha256: str
    source_position_decoded_content_sha256: str
    run_name: str
    arena_id: int
    instance_key: np.ndarray
    frame_indices: np.ndarray
    scratch_root: Path
    local_zarr: Path
    parent_selector_attrs: Mapping[str, Any]
    publication_attempt_uuid: str

    @property
    def run_path(self) -> str:
        return f"{TRACKING_PARENT_PATH}/{self.run_name}"

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr.joinpath(*self.run_path.split("/"))

    @property
    def row_count(self) -> int:
        return int(self.instance_key.shape[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": "palette.single_subject_tracking_run_plan",
            "schema_version": 1,
            "source_zarr": str(self.source_zarr),
            "source_position_run": self.source_position_run,
            "source_position_manifest_sha256": (
                self.source_position_manifest_sha256
            ),
            "source_position_decoded_content_sha256": (
                self.source_position_decoded_content_sha256
            ),
            "run_name": self.run_name,
            "run_path": self.run_path,
            "arena_id": self.arena_id,
            "row_count": self.row_count,
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "target_run_path": str(self.target_run_path),
            "parent_selector_attrs": dict(self.parent_selector_attrs),
            "publication_attempt_uuid": self.publication_attempt_uuid,
            "publication_policy": SINGLE_SUBJECT_TRACKING_PUBLISH_POLICY,
            "retry_policy": SINGLE_SUBJECT_TRACKING_RETRY_POLICY,
            "selector_eligible": False,
        }


def plan_single_subject_tracking_run(
    source_position: SubjectPositionSourceHandle,
    *,
    arena_id: int,
    run_name: str | None = None,
    scratch_root: str | Path,
    publication_attempt_uuid: str | None = None,
) -> SingleSubjectTrackingRunPlan:
    """Freeze one exact position rowset and reject ambiguous single-fish input."""

    source = require_subject_position_source_handle(source_position)
    if source.selector_eligible is not False:
        raise ValueError(
            "This non-promoting tracking materializer requires a selector-ineligible position source."
        )
    if type(arena_id) is not int or arena_id < 0:
        raise ValueError("arena_id must be one non-negative integer.")
    attempt = (
        str(uuid.UUID(publication_attempt_uuid))
        if publication_attempt_uuid is not None
        else str(uuid.uuid4())
    )
    chosen_name = _safe_run_name(
        run_name or f"tracking_position_{uuid.UUID(attempt).hex}"
    )
    archive = source.analysis_zarr_path.expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if scratch == archive or scratch.is_relative_to(archive):
        raise ValueError("Scratch root must be outside the authoritative archive.")
    local_zarr = scratch / f"{chosen_name}.zarr"
    target = archive.joinpath(*f"{TRACKING_PARENT_PATH}/{chosen_name}".split("/"))
    if target.exists():
        raise FileExistsError(f"Refusing existing tracking run: {target}")
    if local_zarr.exists():
        raise FileExistsError(f"Refusing existing local tracking attempt: {local_zarr}")

    instance_key = _readonly(source.instance_key[:], dtype=np.dtype("uint64"))
    frames = _readonly(
        source.source_acquisition_frame_index[:], dtype=np.dtype("int64")
    )
    source_rows = _readonly(source.source_row_index[:], dtype=np.dtype("int64"))
    if not np.array_equal(source_rows, np.arange(source.row_count, dtype=np.int64)):
        raise ValueError("Subject-position source rows are not dense ordered identity.")
    if instance_key.shape != frames.shape or instance_key.shape != source_rows.shape:
        raise ValueError("Subject-position identity arrays are not row aligned.")
    if np.unique(instance_key).shape[0] != instance_key.shape[0]:
        raise ValueError("Subject-position instance_key contains duplicates.")
    if np.any(frames < 0):
        raise ValueError("Subject-position acquisition frame indices must be non-negative.")
    arenas = np.full(frames.shape, arena_id, dtype=np.int32)
    # This is both a planning preflight and the scientific single-subject gate:
    # duplicate rows in one acquisition frame fail before any local write.
    build_single_subject_per_arena_tracking(arenas, frames, conflict_policy="fail")

    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    parent = root.get(TRACKING_PARENT_PATH)
    return SingleSubjectTrackingRunPlan(
        source_zarr=archive,
        source_position_run=source.run_path,
        source_position_manifest_sha256=source.manifest_sha256,
        source_position_decoded_content_sha256=source.decoded_content_sha256,
        run_name=chosen_name,
        arena_id=arena_id,
        instance_key=instance_key,
        frame_indices=frames,
        scratch_root=scratch,
        local_zarr=local_zarr,
        parent_selector_attrs=MappingProxyType(_selector_attrs(parent)),
        publication_attempt_uuid=attempt,
    )


def _validate_run_group(
    run_group: Any,
    *,
    plan: SingleSubjectTrackingRunPlan,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = run_group.attrs.get(TRACKING_RUN_MANIFEST_ATTR)
    validated = validate_tracking_run_manifest(
        manifest,
        expected_run_name=plan.run_name,
        expected_status=RUN_STATUS_COMPLETE,
        expected_selector_eligible=False,
    )
    digest = validated["manifest_sha256"]
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("Tracking manifest digest differs from the local authority.")
    if run_group.attrs.get(TRACKING_RUN_MANIFEST_DIGEST_ATTR) != digest:
        raise ValueError("Tracking manifest digest attr is stale.")
    if (
        run_group.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or run_group.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Tracking run is not complete and selector-ineligible.")
    if tracking_array_records(run_group) != validated["payload"]["arrays"]:
        raise ValueError("Tracking arrays differ from the immutable manifest.")
    keys = np.asarray(run_group["instance_key"][:])
    frames = np.asarray(run_group["frame_indices"][:])
    arenas = np.asarray(run_group["arena_ids"][:])
    tracks = np.asarray(run_group["track_ids"][:])
    source_rows = np.asarray(run_group["source_row_indices"][:])
    if keys.dtype != np.dtype("uint64") or not np.array_equal(keys, plan.instance_key):
        raise ValueError("Tracking instance keys differ from the planned rowset.")
    if not np.array_equal(frames.astype(np.int64), plan.frame_indices):
        raise ValueError("Tracking frame identities differ from the planned rowset.")
    if not np.array_equal(source_rows, np.arange(plan.row_count, dtype=source_rows.dtype)):
        raise ValueError("Tracking source row identity is not dense and ordered.")
    if np.any(arenas != plan.arena_id) or np.any(tracks != 0):
        raise ValueError("Tracking rows do not map to the one planned arena and track.")
    source = validated["payload"]["source"]
    if (
        source["source_authority_kind"] != "subject_position_run"
        or source["source_subject_position_run"] != plan.source_position_run
        or source["source_subject_position_manifest_sha256"]
        != plan.source_position_manifest_sha256
        or source["source_subject_position_decoded_content_sha256"]
        != plan.source_position_decoded_content_sha256
    ):
        raise ValueError("Tracking subject-position lineage differs from the plan.")
    expected_fingerprint = build_rowset_fingerprint(
        source_rowset_path=plan.source_position_run,
        row_count=plan.row_count,
        instance_keys=plan.instance_key,
    )
    if source["source_rowset_fingerprint"] != expected_fingerprint.fingerprint:
        raise ValueError("Tracking source rowset fingerprint is stale.")
    return {
        "valid": True,
        "run_path": plan.run_path,
        "row_count": plan.row_count,
        "track_count": 1 if plan.row_count else 0,
        "manifest_sha256": digest,
    }


def _materialize_local(plan: SingleSubjectTrackingRunPlan) -> dict[str, Any]:
    refreshed = load_subject_position_source_handle(
        plan.source_zarr,
        plan.source_position_run,
        expected_selector_eligible=False,
        use_consolidated=True,
        expected_manifest_sha256=plan.source_position_manifest_sha256,
    )
    if refreshed.decoded_content_sha256 != plan.source_position_decoded_content_sha256:
        raise ValueError("Subject-position decoded content changed after planning.")
    plan.local_zarr.parent.mkdir(parents=True, exist_ok=True)
    local_root = zarr.open_group(
        str(plan.local_zarr),
        mode="w-",
        zarr_format=3,
        use_consolidated=False,
    )
    arenas = np.full(plan.frame_indices.shape, plan.arena_id, dtype=np.int32)
    run_name, run_group, summary = write_single_subject_per_arena_tracking_run(
        root=local_root,
        arena_ids=arenas,
        frame_indices=plan.frame_indices,
        source_detect_run=None,
        source_arena_assignment_run=None,
        source_rowset_path=plan.source_position_run,
        conflict_policy="fail",
        instance_key=plan.instance_key,
        source_subject_position_run=plan.source_position_run,
        source_subject_position_manifest_sha256=(
            plan.source_position_manifest_sha256
        ),
        source_subject_position_decoded_content_sha256=(
            plan.source_position_decoded_content_sha256
        ),
        exact_run_name=plan.run_name,
        stage_selector_eligible=False,
    )
    if run_name != plan.run_name:
        raise RuntimeError("Tracking writer changed the exact planned run name.")
    validation = _validate_run_group(run_group, plan=plan)
    return {"summary": summary, "validation": validation}


def publish_single_subject_tracking_run(
    plan: SingleSubjectTrackingRunPlan,
    *,
    copy_backend: str = "python",
    keep_scratch: bool = True,
) -> dict[str, Any]:
    """Publish one exact candidate while preserving every tracking selector."""

    local = _materialize_local(plan)
    expected_manifest = str(local["validation"]["manifest_sha256"])
    acceptance: dict[str, Any] = {}

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_run_group(
            zarr.open_group(
                str(path), mode="r", zarr_format=3, use_consolidated=False
            ),
            plan=plan,
            expected_manifest_sha256=expected_manifest,
        )

    def prepare(root: Any) -> tuple[Any]:
        return (
            require_runs_parent(
                root,
                TRACKING_PARENT_PATH,
                completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
            ),
        )

    def complete(_root: Any, _parent: Any, run_group: Any) -> None:
        run_group.attrs["stage_selector_eligible"] = False
        mark_run_complete(
            run_group,
            parent_group=None,
            run_name=plan.run_name,
            run_provenance=run_group.attrs.get("run_provenance"),
        )

    def verify(root: Any) -> None:
        parent = root[TRACKING_PARENT_PATH]
        if _selector_attrs(parent) != dict(plan.parent_selector_attrs):
            raise RuntimeError("Tracking publication changed parent selectors.")
        _validate_run_group(
            parent[plan.run_name],
            plan=plan,
            expected_manifest_sha256=expected_manifest,
        )

    def finalize(_root: Any, _parent: Any, _run_group: Any) -> None:
        direct = load_tracking_source_handle(
            plan.source_zarr,
            plan.run_path,
            expected_selector_eligible=False,
            use_consolidated=False,
            expected_manifest_sha256=expected_manifest,
        )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        equivalence = validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=plan.run_path,
        )
        consolidated = load_tracking_source_handle(
            plan.source_zarr,
            plan.run_path,
            expected_selector_eligible=False,
            use_consolidated=True,
            expected_manifest_sha256=expected_manifest,
        )
        if direct.verification_digest != consolidated.verification_digest:
            raise RuntimeError("Tracking direct and consolidated authorities differ.")
        acceptance.update(
            {
                "direct_consolidated": equivalence.to_json(),
                "verification_digest": consolidated.verification_digest,
            }
        )

    def repair_failed(_target: Path) -> None:
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="single-subject-tracking",
            publish_schema_id=SINGLE_SUBJECT_TRACKING_PUBLISH_SCHEMA_ID,
            policy=SINGLE_SUBJECT_TRACKING_PUBLISH_POLICY,
            rollback_policy=(
                "retain_failed_tombstone_leave_parent_selectors_untouched"
            ),
            content_checksum=True,
            publication_attempt_uuid=plan.publication_attempt_uuid,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=finalize,
        repair_failed_publication_visibility=repair_failed,
        accept_persisted_activation_on_callback_error=False,
        payload_metadata={
            "source_subject_position_run": plan.source_position_run,
            "source_subject_position_manifest_sha256": (
                plan.source_position_manifest_sha256
            ),
            "tracking_manifest_sha256": expected_manifest,
            "selector_ineligible": True,
        },
    )
    result = {
        "schema_id": SINGLE_SUBJECT_TRACKING_PUBLISH_SCHEMA_ID,
        "schema_version": SINGLE_SUBJECT_TRACKING_PUBLISH_SCHEMA_VERSION,
        "plan": plan.as_dict(),
        "local": local,
        "publication": publication,
        "acceptance": acceptance,
    }
    if not keep_scratch and plan.local_zarr.exists():
        shutil.rmtree(plan.local_zarr)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--source-position-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arena-id", type=int, default=0)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--discard-scratch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = load_subject_position_source_handle(
        args.analysis_zarr,
        args.source_position_run,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    plan = plan_single_subject_tracking_run(
        source,
        arena_id=args.arena_id,
        run_name=args.run_name,
        scratch_root=args.scratch_root,
    )
    result: Mapping[str, Any]
    if args.apply:
        result = publish_single_subject_tracking_run(
            plan,
            copy_backend=args.copy_backend,
            keep_scratch=not args.discard_scratch,
        )
    else:
        result = plan.as_dict()
    encoded = json.dumps(json_attr_safe(result), indent=2, sort_keys=True)
    print(encoded if args.json else encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SINGLE_SUBJECT_TRACKING_PUBLISH_SCHEMA_ID",
    "SINGLE_SUBJECT_TRACKING_PUBLISH_SCHEMA_VERSION",
    "SINGLE_SUBJECT_TRACKING_PUBLISH_POLICY",
    "SingleSubjectTrackingRunPlan",
    "plan_single_subject_tracking_run",
    "publish_single_subject_tracking_run",
]
