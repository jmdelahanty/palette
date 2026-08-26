"""Immutable selector-ineligible semantic successor for historical stimulus runs.

This is a deliberately narrow migration boundary.  It accepts only a legacy
Citrus protocol snapshot (schema v1), copies one exact completed stimulus run,
adds the exact semantic snapshot bytes and per-step bindings from the raw H5,
and publishes a new named run without changing stimulus selectors.  Modern v2
recordings must use their sealed frame-bound acquisition contract instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np
import zarr

from fisheye.analysis.import_stimulus_to_zarr import (
    _bind_protocol_semantic_steps,
    _materialize_protocol_semantic_snapshot,
    _protocol_semantic_storage_state,
    _validate_protocol_semantic_steps,
)
from fisheye.analysis.stimulus_epoch_schema import (
    stimulus_group_logical_fingerprint,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.protocol_semantic_contract import (
    ProtocolSemanticSnapshot,
    read_materialized_protocol_semantic_snapshot,
    read_protocol_semantic_snapshot,
)
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
)


SCHEMA_ID = "palette.historical_protocol_semantic_stimulus_successor"
SCHEMA_VERSION = 1
MANIFEST_ATTR = "historical_protocol_semantic_successor_manifest"
MANIFEST_SHA256_ATTR = "historical_protocol_semantic_successor_manifest_sha256"
PUBLICATION_SCHEMA_ID = "palette.historical_protocol_semantic_stimulus_publication"
PUBLICATION_POLICY = "named_selector_ineligible_copy_on_write_v1"
PARENT_PATH = "analysis/stimulus_runs"

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SELECTOR_NAMES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative",
        "authoritative_run",
        "active",
        "active_run",
        "current",
        "current_run",
        "default",
        "default_run",
        "selected",
        "selected_run",
        "newest",
        "fallback",
    }
)
_LIFECYCLE_ATTRS = (
    "palette_run_completed_at_utc",
    "palette_run_failed_at_utc",
    "palette_run_error",
    "atomic_publication_receipt",
    "atomic_publication_tombstone",
    "atomic_publication_owner_uuid",
)


class HistoricalProtocolSemanticStimulusSuccessorError(RuntimeError):
    """Raised when a historical semantic successor cannot be proven exactly."""


def _fail(message: str) -> None:
    raise HistoricalProtocolSemanticStimulusSuccessorError(message)


def _name(value: object, *, field_name: str) -> str:
    if type(value) is not str or _NAME_RE.fullmatch(value) is None:
        _fail(f"{field_name} must be one exact bare run name.")
    if value.lower() in _SELECTOR_NAMES:
        _fail(f"{field_name} must not be a selector or fallback name.")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_tree_document(group: Any, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    arrays: dict[str, Any] = {}

    def walk(node: Any, prefix: str = "") -> None:
        for name, array in sorted(node.arrays(), key=lambda item: str(item[0])):
            path = f"{prefix}/{name}" if prefix else str(name)
            if any(path == item or path.startswith(f"{item}/") for item in exclude):
                continue
            values = np.ascontiguousarray(array[...])
            arrays[path] = {
                "dtype": str(np.dtype(array.dtype)),
                "shape": [int(value) for value in array.shape],
                "attrs": _plain(dict(array.attrs)),
                "sha256": sha256(values.tobytes(order="C")).hexdigest(),
            }
        for name, child in sorted(node.groups(), key=lambda item: str(item[0])):
            path = f"{prefix}/{name}" if prefix else str(name)
            if any(path == item or path.startswith(f"{item}/") for item in exclude):
                continue
            walk(child, path)

    walk(group)
    return {
        "schema_id": "palette.stimulus_array_tree_content",
        "schema_version": 1,
        "arrays": arrays,
    }


def _array_tree_sha256(group: Any, *, exclude: frozenset[str] = frozenset()) -> str:
    return canonical_json_sha256(_array_tree_document(group, exclude=exclude))


@dataclass(frozen=True, slots=True)
class HistoricalProtocolSemanticStimulusPlan:
    analysis_zarr: Path
    source_run_name: str
    run_name: str
    raw_h5: Path
    recording_id: str
    source_stimulus_fingerprint: str
    source_array_tree_sha256: str
    raw_h5_size_bytes: int
    raw_h5_sha256: str
    snapshot: ProtocolSemanticSnapshot = field(repr=False, compare=False)
    manifest: Mapping[str, Any] = field(repr=False)
    run_provenance: Mapping[str, Any] = field(repr=False)
    parent_attrs: Mapping[str, Any] = field(repr=False)

    @property
    def source_run_path(self) -> str:
        return f"{PARENT_PATH}/{self.source_run_name}"

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{self.run_name}"

    def receipt(self) -> dict[str, Any]:
        return {
            "status": "eligible_for_selector_ineligible_publication",
            "analysis_zarr": str(self.analysis_zarr),
            "recording_id": self.recording_id,
            "source_run_path": self.source_run_path,
            "run_path": self.run_path,
            "raw_h5": str(self.raw_h5),
            "raw_h5_size_bytes": self.raw_h5_size_bytes,
            "raw_h5_sha256": self.raw_h5_sha256,
            "protocol_semantic_hash": self.snapshot.semantic_hash,
            "protocol_trial_index_sha256": self.snapshot.trial_index_sha256,
            "protocol_trial_index_integrity_status": (
                self.snapshot.trial_index_integrity_status
            ),
            "source_stimulus_fingerprint": self.source_stimulus_fingerprint,
            "source_array_tree_sha256": self.source_array_tree_sha256,
            "manifest_sha256": canonical_json_sha256(dict(self.manifest)),
            "selector_eligible": False,
            "selector_activation": "none",
            "registry_update": False,
        }


def plan_historical_protocol_semantic_stimulus_successor(
    analysis_zarr: str | Path,
    *,
    source_run_name: str,
    run_name: str,
    raw_h5: str | Path,
) -> HistoricalProtocolSemanticStimulusPlan:
    """Verify all source evidence without mutating the archive."""

    archive = Path(analysis_zarr).expanduser().resolve()
    source_name = _name(source_run_name, field_name="source_run_name")
    target_name = _name(run_name, field_name="run_name")
    if source_name == target_name:
        _fail("The semantic successor must use a new immutable run name.")
    raw_path = Path(raw_h5).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw Citrus H5 does not exist: {raw_path}")

    source_path = f"{PARENT_PATH}/{source_name}"
    target_path = f"{PARENT_PATH}/{target_name}"
    try:
        validate_direct_consolidated_subtree(archive, subtree_path=source_path)
        direct = open_zarr_root(archive, mode="r", use_consolidated=False)
        consolidated = open_zarr_root(archive, mode="r", use_consolidated=True)
        source = consolidated[source_path]
        parent = consolidated[PARENT_PATH]
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        _fail(f"Unable to seal the exact source stimulus run: {exc}")
    if target_path in direct or target_path in consolidated:
        _fail("Target semantic successor run already exists.")
    if (
        source.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or source.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
    ):
        _fail("Source stimulus run is not complete under the run contract.")
    if source.attrs.get("palette_run_name") != source_name:
        _fail("Source stimulus run name does not match its exact path.")
    if source.attrs.get("protocol_semantic_status") is not None:
        _fail("Source stimulus run already has semantic status; use its exact authority.")

    try:
        with h5py.File(raw_path, "r") as raw:
            snapshot = read_protocol_semantic_snapshot(raw)
    except (OSError, TypeError, ValueError) as exc:
        _fail(f"Unable to read exact raw protocol semantic evidence: {exc}")
    if snapshot is None:
        _fail("Raw H5 has no producer protocol semantic snapshot.")
    if snapshot.snapshot_schema_version != 1:
        _fail(
            "Historical successor accepts only legacy snapshot schema v1; modern "
            "v2 evidence must use the sealed frame-bound acquisition contract."
        )
    try:
        _validate_protocol_semantic_steps(source, snapshot)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"Existing materialized step rows do not match the raw recipe: {exc}")

    recording_id = consolidated.attrs.get("recording_id")
    if type(recording_id) is not str or not recording_id:
        _fail("Analysis archive lacks one exact recording_id.")
    source_fingerprint = stimulus_group_logical_fingerprint(source)
    array_digest = _array_tree_sha256(source)
    raw_size = int(raw_path.stat().st_size)
    raw_digest = _file_sha256(raw_path)
    manifest = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "recording_id": recording_id,
        "run_name": target_name,
        "run_path": target_path,
        "source_stimulus_run": {
            "run_name": source_name,
            "run_path": source_path,
            "logical_fingerprint": source_fingerprint,
            "array_tree_sha256": array_digest,
        },
        "raw_protocol_semantic_source": {
            "path": str(raw_path),
            "size_bytes": raw_size,
            "sha256": raw_digest,
            "snapshot_schema_version": snapshot.snapshot_schema_version,
            "snapshot_policy_id": snapshot.snapshot_policy_id,
            "protocol_semantic_hash": snapshot.semantic_hash,
            "protocol_trial_index_sha256": snapshot.trial_index_sha256,
            "protocol_trial_index_integrity_status": (
                snapshot.trial_index_integrity_status
            ),
            "recipe": snapshot.recipe_record(),
        },
        "policy": {
            "copy_on_write": True,
            "source_arrays": "byte_decoded_content_preserved",
            "semantic_bytes": "exact_raw_h5_utf8",
            "step_binding": "exact_step_index_mode_duration_match",
            "legacy_trial_reconstruction": "prohibited",
            "selector_activation": "none",
        },
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    canonical_json_sha256(manifest)
    provenance = build_writer_run_provenance(
        command=(
            "fisheye.utils.materialize_historical_protocol_semantic_stimulus_successor"
        ),
        params={
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "run_name": target_name,
            "selector_eligible": False,
            "legacy_trial_reconstruction": "prohibited",
        },
        input_run_ids={
            "recording_id": recording_id,
            "source_stimulus_run_path": source_path,
            "source_stimulus_fingerprint": source_fingerprint,
            "raw_h5_sha256": raw_digest,
            "protocol_semantic_hash": snapshot.semantic_hash,
        },
    )
    return HistoricalProtocolSemanticStimulusPlan(
        analysis_zarr=archive,
        source_run_name=source_name,
        run_name=target_name,
        raw_h5=raw_path,
        recording_id=recording_id,
        source_stimulus_fingerprint=source_fingerprint,
        source_array_tree_sha256=array_digest,
        raw_h5_size_bytes=raw_size,
        raw_h5_sha256=raw_digest,
        snapshot=snapshot,
        manifest=MappingProxyType(_plain(manifest)),
        run_provenance=MappingProxyType(_plain(provenance)),
        parent_attrs=MappingProxyType(_plain(dict(parent.attrs))),
    )


def _write_local_run(
    plan: HistoricalProtocolSemanticStimulusPlan,
    local_run_path: Path,
) -> None:
    shutil.copytree(plan.analysis_zarr / plan.source_run_path, local_run_path)
    run = zarr.open_group(str(local_run_path), mode="a", use_consolidated=False)
    for name in _LIFECYCLE_ATTRS:
        if name in run.attrs:
            del run.attrs[name]
    mark_run_started(run, run_name=plan.run_name, stage="stimulus")
    run.attrs.update(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "stage_selector_eligible": False,
            "selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "registry_update": False,
            "historical_semantic_successor": True,
            "protocol_semantic_source_h5": str(plan.raw_h5),
            "protocol_semantic_source_h5_size_bytes": plan.raw_h5_size_bytes,
            "protocol_semantic_source_h5_sha256": plan.raw_h5_sha256,
            MANIFEST_ATTR: _plain(plan.manifest),
            MANIFEST_SHA256_ATTR: canonical_json_sha256(dict(plan.manifest)),
        }
    )
    _materialize_protocol_semantic_snapshot(run, plan.snapshot)
    _bind_protocol_semantic_steps(run, plan.snapshot)
    mark_run_complete(
        run,
        run_name=plan.run_name,
        run_provenance=dict(plan.run_provenance),
    )
    _validate_run(local_run_path, plan=plan)


def _validate_run(
    run_path: Path,
    *,
    plan: HistoricalProtocolSemanticStimulusPlan,
) -> dict[str, Any]:
    run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
    attrs = dict(run.attrs)
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        _fail("Semantic successor run is not complete.")
    if attrs.get("schema_id") != SCHEMA_ID or attrs.get("schema_version") != (
        SCHEMA_VERSION
    ):
        _fail("Semantic successor lacks its declared schema identity.")
    if attrs.get("palette_run_name") != plan.run_name:
        _fail("Semantic successor run name differs from its plan.")
    if (
        attrs.get("stage_selector_eligible") is not False
        or attrs.get("selector_eligible") is not False
        or attrs.get("selection") != "none"
        or attrs.get("production_authority") is not False
        or attrs.get("registry_update") is not False
    ):
        _fail("Semantic successor is not explicitly selector-ineligible.")
    if attrs.get(MANIFEST_ATTR) != _plain(plan.manifest) or attrs.get(
        MANIFEST_SHA256_ATTR
    ) != canonical_json_sha256(dict(plan.manifest)):
        _fail("Semantic successor manifest is missing or stale.")
    provenance = validate_run_provenance(attrs.get("run_provenance"))
    if not provenance.valid:
        _fail(f"Semantic successor run provenance is invalid: {provenance.errors}")
    try:
        state = _protocol_semantic_storage_state(run, plan.snapshot)
        reloaded = read_materialized_protocol_semantic_snapshot(run)
        _validate_protocol_semantic_steps(run, plan.snapshot)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        _fail(f"Semantic successor evidence does not reload exactly: {exc}")
    if state != "verified" or (
        reloaded.semantic_json != plan.snapshot.semantic_json
        or reloaded.trial_index_json != plan.snapshot.trial_index_json
        or reloaded.recipe_record() != plan.snapshot.recipe_record()
    ):
        _fail("Semantic successor bytes or recipe differ from the exact raw snapshot.")
    array_digest = _array_tree_sha256(
        run,
        exclude=frozenset({"protocol_semantic_snapshot"}),
    )
    if array_digest != plan.source_array_tree_sha256:
        _fail("Semantic successor changed a pre-existing stimulus array.")
    return {
        "valid": True,
        "run_name": plan.run_name,
        "manifest_sha256": canonical_json_sha256(dict(plan.manifest)),
        "source_array_tree_sha256": array_digest,
        "protocol_semantic_hash": plan.snapshot.semantic_hash,
        "selector_eligible": False,
    }


def publish_historical_protocol_semantic_stimulus_successor(
    plan: HistoricalProtocolSemanticStimulusPlan,
    *,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Atomically publish the planned run while leaving all selectors unchanged."""

    if type(plan) is not HistoricalProtocolSemanticStimulusPlan:
        raise TypeError("plan must be a historical semantic stimulus plan.")
    if int(plan.raw_h5.stat().st_size) != plan.raw_h5_size_bytes or _file_sha256(
        plan.raw_h5
    ) != plan.raw_h5_sha256:
        _fail("Raw H5 changed after the semantic successor plan was sealed.")
    root = open_zarr_root(plan.analysis_zarr, mode="r", use_consolidated=True)
    source = root[plan.source_run_path]
    if stimulus_group_logical_fingerprint(source) != plan.source_stimulus_fingerprint:
        _fail("Source stimulus run changed after the successor plan was sealed.")

    scratch = Path(scratch_root).expanduser().resolve() if scratch_root else None
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{plan.run_name}.historical-semantic.",
        dir=str(scratch) if scratch is not None else None,
    ) as temporary:
        local = Path(temporary) / "run.zarr"
        _write_local_run(plan, local)
        parent_snapshot: dict[str, Any] | None = None

        def validate(path: Path) -> Mapping[str, Any]:
            return _validate_run(path, plan=plan)

        def prepare(authoritative_root: Any) -> tuple[Any]:
            nonlocal parent_snapshot
            parent = authoritative_root[PARENT_PATH]
            observed = _plain(dict(parent.attrs))
            if parent_snapshot is None:
                parent_snapshot = observed
                if observed != _plain(plan.parent_attrs):
                    _fail("Stimulus parent metadata changed after planning.")
            elif observed != parent_snapshot:
                _fail("Stimulus parent metadata changed during publication.")
            return (parent,)

        def complete(_root: Any, _parent: Any, run: Any) -> None:
            run.attrs["stage_selector_eligible"] = False
            run.attrs["selector_eligible"] = False
            run.attrs["selection"] = "none"
            mark_run_complete(
                run,
                run_name=plan.run_name,
                run_provenance=dict(plan.run_provenance),
            )

        def verify(authoritative_root: Any) -> None:
            parent = authoritative_root[PARENT_PATH]
            if parent_snapshot is None or _plain(dict(parent.attrs)) != parent_snapshot:
                _fail("Semantic successor publication changed stimulus selectors.")
            _validate_run(plan.analysis_zarr / plan.run_path, plan=plan)

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.analysis_zarr,
                local_run_path=local,
                target_run_path=plan.analysis_zarr / plan.run_path,
                run_name=plan.run_name,
                lock_suffix="historical-protocol-semantic-stimulus-publication",
                publish_schema_id=PUBLICATION_SCHEMA_ID,
                policy=PUBLICATION_POLICY,
                rollback_policy="retain_failed_selector_ineligible_tombstone_v1",
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            payload_metadata={
                "recording_id": plan.recording_id,
                "run_path": plan.run_path,
                "manifest_sha256": canonical_json_sha256(dict(plan.manifest)),
                "selector_activation": "none",
                "registry_update": False,
            },
        )
    consolidation = consolidate_metadata_capture_expected_warnings(plan.analysis_zarr)
    metadata = validate_direct_consolidated_subtree(
        plan.analysis_zarr,
        subtree_path=plan.run_path,
    ).to_json()
    direct = open_zarr_root(plan.analysis_zarr, mode="r", use_consolidated=False)
    consolidated = open_zarr_root(plan.analysis_zarr, mode="r", use_consolidated=True)
    if (
        _plain(dict(direct[PARENT_PATH].attrs)) != _plain(plan.parent_attrs)
        or _plain(dict(consolidated[PARENT_PATH].attrs)) != _plain(plan.parent_attrs)
    ):
        _fail("Final metadata generation changed stimulus parent selectors.")
    validation = _validate_run(plan.analysis_zarr / plan.run_path, plan=plan)
    return {
        **plan.receipt(),
        "status": "published_selector_ineligible",
        "validation": validation,
        "metadata_equivalence": metadata,
        "consolidation": consolidation,
        "atomic_publication": publication,
    }


__all__ = [
    "HistoricalProtocolSemanticStimulusPlan",
    "HistoricalProtocolSemanticStimulusSuccessorError",
    "plan_historical_protocol_semantic_stimulus_successor",
    "publish_historical_protocol_semantic_stimulus_successor",
]
