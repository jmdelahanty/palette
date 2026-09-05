"""Node-local materialization and atomic nonpromoting distribution publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
import uuid

import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import mark_run_complete, require_runs_parent

from .recording_behavior_distribution_storage import (
    PARENT_PATH,
    load_recording_behavior_distribution_source_handle,
    selector_snapshot,
    validate_recording_behavior_distribution_run,
    write_recording_behavior_distribution_run,
)
from .recording_behavior_distribution_workflow import (
    PreparedRecordingBehaviorDistribution,
)


PUBLICATION_SCHEMA_ID = "palette.analysis.recording_behavior_distribution_publish"
PUBLICATION_POLICY_ID = "node_local_atomic_nonpromoting_recording_distribution_v1"
LOCK_SUFFIX = "recording-behavior-distribution-publish"
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class RecordingBehaviorDistributionPublicationError(ValueError):
    """A distribution materialization or publication plan is unsafe."""


def _fail(message: str) -> None:
    raise RecordingBehaviorDistributionPublicationError(message)


def _safe_run_name(value: object) -> str:
    if type(value) is not str or _RUN_NAME_RE.fullmatch(value) is None:
        _fail("run_name must be one safe exact child name.")
    if value.startswith(".") or value in {
        "latest",
        "latest_complete",
        "authoritative_run",
        "selected_run",
    }:
        _fail("run_name cannot be hidden or selector-like.")
    return value


@dataclass(frozen=True, slots=True)
class RecordingBehaviorDistributionPublicationPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    local_run_path: Path
    target_run_path: Path
    run_name: str
    recording_id: str
    result_record_sha256: str
    selector_attrs_before: Mapping[str, Any]

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{self.run_name}"


def build_recording_behavior_distribution_publication_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    prepared: PreparedRecordingBehaviorDistribution,
) -> RecordingBehaviorDistributionPublicationPlan:
    """Plan an immutable candidate without creating local or authoritative paths."""

    if type(prepared) is not PreparedRecordingBehaviorDistribution:
        _fail("prepared must be one PreparedRecordingBehaviorDistribution.")
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {source}")
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        _fail("Scratch root must be outside the authoritative analysis Zarr.")
    config = prepared.result.config
    if Path(source) != Path(str(config.source_record.get("analysis_zarr", source))):
        configured = config.source_record.get("analysis_zarr")
        if configured is not None and Path(str(configured)).resolve() != source:
            _fail("Prepared distribution names another analysis Zarr.")
    run_name = _safe_run_name(config.distribution_run_id)
    local_zarr = scratch / (
        f"recording_behavior_distribution_{run_name}_{uuid.uuid4().hex}.zarr"
    )
    local_run_path = local_zarr / PARENT_PATH / run_name
    target_run_path = source / PARENT_PATH / run_name
    if local_zarr.exists() or target_run_path.exists():
        raise FileExistsError("Planned local or target distribution path already exists.")
    root = open_zarr_root(source, mode="r", use_consolidated=False)
    parent = root.get(PARENT_PATH)
    if parent is not None and not isinstance(parent, zarr.Group):
        _fail("Recording-distribution parent path is not a Zarr group.")
    before = selector_snapshot(parent)
    return RecordingBehaviorDistributionPublicationPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=local_zarr,
        local_run_path=local_run_path,
        target_run_path=target_run_path,
        run_name=run_name,
        recording_id=config.recording_id,
        result_record_sha256=str(prepared.result.record["record_sha256"]),
        selector_attrs_before=before,
    )


def _validate_path(
    path: Path,
    *,
    plan: RecordingBehaviorDistributionPublicationPlan,
) -> Mapping[str, Any]:
    run = open_zarr_root(path, mode="r", use_consolidated=False)
    validated = validate_recording_behavior_distribution_run(
        run,
        expected_run_name=plan.run_name,
        expected_recording_id=plan.recording_id,
        expected_result_record_sha256=plan.result_record_sha256,
    )
    return MappingProxyType(
        {
            "valid": True,
            "manifest_sha256": validated["manifest_sha256"],
            "verification_digest": validated["verification_digest"],
            "result_record_sha256": plan.result_record_sha256,
        }
    )


def materialize_recording_behavior_distribution_locally(
    plan: RecordingBehaviorDistributionPublicationPlan,
    *,
    prepared: PreparedRecordingBehaviorDistribution,
) -> Mapping[str, Any]:
    """Write, consolidate, and validate the exact node-local candidate."""

    if prepared.result.record["record_sha256"] != plan.result_record_sha256:
        _fail("Prepared result changed after the publication plan was built.")
    if plan.local_zarr.exists():
        raise FileExistsError(f"Local candidate already exists: {plan.local_zarr}")
    provenance = build_writer_run_provenance(
        command="recording_behavior_distribution_publication",
        params={
            "run_name": plan.run_name,
            "run_path": plan.run_path,
            "result_record_sha256": plan.result_record_sha256,
            "publication_policy": PUBLICATION_POLICY_ID,
        },
        input_run_ids={
            "recording_distribution_source": prepared.result.config.source_record.get(
                "bundle_record_sha256",
                prepared.result.config.source_record.get(
                    "bundle_sha256", plan.result_record_sha256
                ),
            )
        },
        cwd=Path(__file__).resolve().parents[3],
    )
    write_recording_behavior_distribution_run(
        plan.local_zarr,
        run_name=plan.run_name,
        result=prepared.result,
        run_provenance=provenance,
    )
    validate_direct_consolidated_subtree(
        plan.local_zarr, subtree_path=plan.run_path
    )
    validated = _validate_path(plan.local_run_path, plan=plan)
    return MappingProxyType(
        {
            **dict(validated),
            "local_zarr": str(plan.local_zarr),
            "local_run_path": str(plan.local_run_path),
            "run_path": plan.run_path,
        }
    )


def publish_recording_behavior_distribution_candidate(
    plan: RecordingBehaviorDistributionPublicationPlan,
    *,
    copy_backend: str = "python",
) -> Mapping[str, Any]:
    """Atomically publish one complete candidate without changing selectors."""

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_path(path, plan=plan)

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(
                root.require_group("analysis"),
                "recording_behavior_distribution_runs",
            ),
        )

    def complete(
        _root: zarr.Group,
        _parent: zarr.Group,
        run: zarr.Group,
    ) -> None:
        mark_run_complete(
            run,
            parent_group=None,
            run_name=plan.run_name,
            run_provenance=run.attrs.get("run_provenance"),
        )
        run.attrs["stage_selector_eligible"] = False

    def verify(root: zarr.Group) -> None:
        parent = root[PARENT_PATH]
        if dict(selector_snapshot(parent)) != dict(plan.selector_attrs_before):
            raise RuntimeError(
                "Nonpromoting distribution publication changed selector attrs."
            )
        run = parent.get(plan.run_name)
        if not isinstance(run, zarr.Group) or run.attrs.get(
            "stage_selector_eligible"
        ) is not False:
            raise RuntimeError("Published distribution candidate is absent or eligible.")

    def activate(
        root: zarr.Group,
        _parent: zarr.Group,
        _run: zarr.Group,
    ) -> None:
        verify(root)
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        validate_direct_consolidated_subtree(
            plan.source_zarr, subtree_path=plan.run_path
        )
        handle = load_recording_behavior_distribution_source_handle(
            plan.source_zarr,
            run_name=plan.run_name,
            expected_recording_id=plan.recording_id,
        )
        if handle.result_record["record_sha256"] != plan.result_record_sha256:
            raise RuntimeError("Published consolidated result digest changed.")

    def repair(_target: Path) -> None:
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)

    result = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix=LOCK_SUFFIX,
            publish_schema_id=PUBLICATION_SCHEMA_ID,
            policy=PUBLICATION_POLICY_ID,
            rollback_policy=(
                "retain_failed_public_tombstone_leave_parent_selectors_untouched"
            ),
            persist_run_receipt=False,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=activate,
        repair_failed_publication_visibility=repair,
        payload_metadata={
            "run_path": plan.run_path,
            "result_record_sha256": plan.result_record_sha256,
            "selector_attrs_before": dict(plan.selector_attrs_before),
            "promotion_policy": "named_selector_ineligible_candidate_only",
        },
        accept_persisted_activation_on_callback_error=False,
    )
    return MappingProxyType(result)


__all__ = [
    "PUBLICATION_POLICY_ID",
    "PUBLICATION_SCHEMA_ID",
    "RecordingBehaviorDistributionPublicationError",
    "RecordingBehaviorDistributionPublicationPlan",
    "build_recording_behavior_distribution_publication_plan",
    "materialize_recording_behavior_distribution_locally",
    "publish_recording_behavior_distribution_candidate",
]
