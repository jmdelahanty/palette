"""Atomically publish one immutable chaser input-provenance proxy candidate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis_workflows.chaser_input_provenance_proxy_storage import (
    PreparedChaserInputProvenanceProxy,
    validate_prepared_chaser_input_provenance_proxy,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.zarr.chaser_input_provenance_proxy_schema import (
    CHASER_INPUT_PROVENANCE_PROXY_LAYOUT,
    CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH,
    CHASER_INPUT_PROVENANCE_PROXY_PUBLISH_SCHEMA_ID,
    CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID,
    CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION,
    validate_publication_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


MANIFEST_ATTR = "chaser_input_provenance_proxy_manifest"
MANIFEST_DIGEST_ATTR = "chaser_input_provenance_proxy_manifest_sha256"
RUN_PATH_ATTR = "run_path"
MATERIALIZATION_SCHEMA_ID = "palette.chaser_input_provenance_proxy_materialization"
MATERIALIZATION_SCHEMA_VERSION = 1

_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "active",
    "active_run",
    "current",
    "current_run",
    "default",
    "default_run",
    "selected",
    "selected_run",
    "publication_generation",
    "publication_policy",
)


class ChaserInputProvenanceProxyMaterializationError(RuntimeError):
    """Raised when a prepared proxy cannot be published exactly."""


def _safe_run_name(value: object) -> str:
    if type(value) is not str:
        raise TypeError("run_name must be one exact string.")
    name = value.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Unsafe proxy run name: {value!r}.")
    return name


def _plain_json(value: object, *, field: str) -> Any:
    def plain(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    try:
        return json.loads(
            json.dumps(
                plain(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ChaserInputProvenanceProxyMaterializationError(
            f"{field} is not strict JSON: {exc}"
        ) from exc


def _selector_snapshot(parent: Any | None) -> dict[str, Any]:
    if parent is None:
        return {}
    return {
        name: _plain_json(parent.attrs[name], field=f"parent selector {name}")
        for name in _SELECTOR_ATTRS
        if name in parent.attrs
    }


def _array_paths(group: Any) -> set[str]:
    arrays = group.get("arrays")
    if not isinstance(arrays, zarr.Group):
        raise ChaserInputProvenanceProxyMaterializationError(
            "Proxy run lacks its arrays group."
        )
    if set(str(name) for name in arrays.group_keys()):
        raise ChaserInputProvenanceProxyMaterializationError(
            "Proxy arrays group contains unexpected nested groups."
        )
    return set(str(name) for name in arrays.array_keys())


def _read_arrays(group: Any) -> dict[str, np.ndarray]:
    names = _array_paths(group)
    values: dict[str, np.ndarray] = {}
    for name in names:
        node = group[f"arrays/{name}"]
        if not isinstance(node, zarr.Array):
            raise ChaserInputProvenanceProxyMaterializationError(
                f"Proxy array path is not an array: {name!r}."
            )
        value = np.asarray(node[...])
        if value.dtype.hasobject or value.dtype.kind in {"U", "S"}:
            raise ChaserInputProvenanceProxyMaterializationError(
                f"Published proxy array {name!r} is object/string typed."
            )
        values[name] = value
    return values


def _validate_group(
    group: Any,
    *,
    expected_manifest: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
    require_complete: bool,
    label: str,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = _plain_json(expected_manifest, field="expected manifest")
    attrs = dict(group.attrs)
    if attrs.get(MANIFEST_ATTR) != manifest:
        errors.append(f"{label}: manifest differs from prepared evidence")
    if attrs.get(MANIFEST_DIGEST_ATTR) != canonical_json_sha256(manifest):
        errors.append(f"{label}: manifest digest is stale")
    if attrs.get("schema_id") != CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID:
        errors.append(f"{label}: schema_id is invalid")
    if attrs.get("schema_version") != CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION:
        errors.append(f"{label}: schema_version is invalid")
    if attrs.get("layout") != CHASER_INPUT_PROVENANCE_PROXY_LAYOUT:
        errors.append(f"{label}: layout is invalid")
    if attrs.get("stage_selector_eligible") is not False:
        errors.append(f"{label}: stage_selector_eligible is not false")
    if attrs.get("selector_eligible") is not False or attrs.get("selection") != "none":
        errors.append(f"{label}: run is not explicitly unselected")
    if require_complete and attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append(f"{label}: completion status is not complete")
    provenance = attrs.get("run_provenance")
    validation = validate_run_provenance(provenance)
    if not validation.valid:
        errors.append(f"{label}: run provenance is invalid: {validation.errors}")
    if provenance != _plain_json(expected_provenance, field="expected provenance"):
        errors.append(f"{label}: run provenance differs from the plan")
    try:
        arrays = _read_arrays(group)
        publication_manifest = {
            key: value for key, value in manifest.items() if key != "prepared_candidate"
        }
        validate_publication_manifest(publication_manifest, arrays)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"{label}: array/manifest validation failed: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_sha256": canonical_json_sha256(manifest),
    }


@dataclass(frozen=True, slots=True)
class ChaserInputProvenanceProxyMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    run_name: str
    prepared: PreparedChaserInputProvenanceProxy
    parent_selector_attrs: Mapping[str, Any]
    run_provenance: Mapping[str, Any]

    @property
    def run_path(self) -> str:
        return f"{CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH}/{self.run_name}"

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr / self.run_path

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr / self.run_path

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "run_name": self.run_name,
            "run_path": self.run_path,
            "target_run_path": str(self.target_run_path),
            "prepared_manifest_sha256": self.prepared.payload_digest,
            "selector_eligible": False,
            "selection": "none",
            "parent_selector_attrs": dict(self.parent_selector_attrs),
        }


def build_chaser_input_provenance_proxy_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    prepared: PreparedChaserInputProvenanceProxy,
) -> ChaserInputProvenanceProxyMaterializationPlan:
    validate_prepared_chaser_input_provenance_proxy(prepared)
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Authoritative source Zarr not found: {source}")
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Scratch root must not be inside the source Zarr.")
    name = _safe_run_name(run_name)
    target = source / CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH / name
    if target.exists():
        raise FileExistsError(f"Refusing existing target proxy run: {target}")
    root = open_zarr_root(source, mode="r", use_consolidated=False)
    record = prepared.manifest["acquisition_projection_record"]
    if root.attrs.get("recording_id") != record["recording_id"]:
        raise ChaserInputProvenanceProxyMaterializationError(
            "Source archive recording_id differs from the proxy projection."
        )
    parent = root.get(CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH)
    if parent is not None and not isinstance(parent, zarr.Group):
        raise ChaserInputProvenanceProxyMaterializationError(
            "Proxy parent path is not a Zarr group."
        )
    provenance = build_writer_run_provenance(
        command="chaser_input_provenance_proxy_materializer",
        params={
            "run_name": name,
            "run_path": f"{CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH}/{name}",
            "schema_id": CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID,
            "schema_version": CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION,
            "prepared_manifest_sha256": prepared.payload_digest,
            "selector_eligible": False,
        },
        input_run_ids={
            "recording_id": str(record["recording_id"]),
            "source_run_path": str(record["source_run_path"]),
        },
        input_artifacts=[
            {
                "kind": "verified_native_chaser_source",
                "recording_id": str(record["recording_id"]),
                "run_path": str(record["source_run_path"]),
                "manifest_sha256": str(record["source_manifest_sha256"]),
                "verification_digest": str(record["source_verification_digest"]),
            },
            {
                "kind": "acquisition_projection_record",
                "record": _plain_json(record, field="projection record"),
                "sha256": str(
                    prepared.manifest["acquisition_projection_record_sha256"]
                ),
            },
        ],
        cwd=Path(__file__).resolve().parents[4],
    )
    return ChaserInputProvenanceProxyMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=scratch / "chaser-input-provenance-proxy-candidate.zarr",
        run_name=name,
        prepared=prepared,
        parent_selector_attrs=MappingProxyType(_selector_snapshot(parent)),
        run_provenance=MappingProxyType(provenance),
    )


def _chunks_for(values: np.ndarray) -> tuple[int, ...]:
    return tuple(max(1, min(int(size), 16_384)) for size in values.shape)


def _write_local_candidate(
    plan: ChaserInputProvenanceProxyMaterializationPlan,
) -> None:
    if plan.local_zarr.exists():
        raise FileExistsError(f"Refusing existing local candidate: {plan.local_zarr}")
    plan.local_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = open_zarr_root(plan.local_zarr, mode="w-")
    root.attrs["recording_id"] = plan.prepared.manifest[
        "acquisition_projection_record"
    ]["recording_id"]
    parent = require_runs_parent(
        root.require_group("analysis"),
        "chaser_input_provenance_proxy_runs",
    )
    run = parent.create_group(plan.run_name)
    manifest = _plain_json(plan.prepared.manifest, field="prepared manifest")
    provenance = _plain_json(plan.run_provenance, field="run provenance")
    run.attrs.update(
        {
            "schema_id": CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID,
            "schema_version": CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION,
            "layout": CHASER_INPUT_PROVENANCE_PROXY_LAYOUT,
            "stage_selector_eligible": False,
            "selector_eligible": False,
            "selection": "none",
            RUN_PATH_ATTR: plan.run_path,
            "materialization_schema_id": MATERIALIZATION_SCHEMA_ID,
            "materialization_schema_version": MATERIALIZATION_SCHEMA_VERSION,
            MANIFEST_ATTR: json_attr_safe(manifest),
            MANIFEST_DIGEST_ATTR: canonical_json_sha256(manifest),
            "run_provenance": json_attr_safe(provenance),
        }
    )
    mark_run_started(
        run,
        run_name=plan.run_name,
        stage="chaser_input_provenance_proxy_materialization",
    )
    group = run.create_group("arrays")
    for name in sorted(plan.prepared.arrays):
        values = np.asarray(plan.prepared.arrays[name])
        group.create_array(
            name,
            data=values,
            chunks=_chunks_for(values),
            overwrite=False,
        )
    local = _validate_group(
        run,
        expected_manifest=manifest,
        expected_provenance=provenance,
        require_complete=False,
        label="local",
    )
    if not local["valid"]:
        raise ChaserInputProvenanceProxyMaterializationError(
            f"Local proxy validation failed: {local}"
        )
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=plan.run_name,
        run_provenance=provenance,
    )
    complete = _validate_group(
        run,
        expected_manifest=manifest,
        expected_provenance=provenance,
        require_complete=True,
        label="local-complete",
    )
    if not complete["valid"]:
        raise ChaserInputProvenanceProxyMaterializationError(
            f"Completed local proxy validation failed: {complete}"
        )


def materialize_chaser_input_provenance_proxy(
    source_zarr: str | Path,
    *,
    prepared: PreparedChaserInputProvenanceProxy,
    scratch_root: str | Path,
    run_name: str,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Publish one named immutable candidate without changing selectors."""

    plan = build_chaser_input_provenance_proxy_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        run_name=run_name,
        prepared=prepared,
    )
    _write_local_candidate(plan)
    manifest = _plain_json(prepared.manifest, field="prepared manifest")
    provenance = _plain_json(plan.run_provenance, field="run provenance")

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_group(
            open_zarr_root(path, mode="r", use_consolidated=False),
            expected_manifest=manifest,
            expected_provenance=provenance,
            require_complete=True,
            label="publication",
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(
                root.require_group("analysis"),
                "chaser_input_provenance_proxy_runs",
            ),
        )

    def complete(_root: zarr.Group, parent: zarr.Group, run: zarr.Group) -> None:
        run.attrs["stage_selector_eligible"] = False
        run.attrs["selector_eligible"] = False
        run.attrs["selection"] = "none"
        run.attrs["run_provenance"] = json_attr_safe(provenance)
        mark_run_complete(
            run,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=provenance,
        )

    def verify(root: zarr.Group) -> None:
        parent = root[CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH]
        if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
            raise ChaserInputProvenanceProxyMaterializationError(
                "Proxy publication changed a parent selector."
            )
        result = validate(plan.target_run_path)
        if not result["valid"]:
            raise ChaserInputProvenanceProxyMaterializationError(
                f"Published proxy run is invalid: {result}"
            )

    def finalize(_root: zarr.Group, _parent: zarr.Group, _run: zarr.Group) -> None:
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=plan.run_path,
        )
        for mode, label in ((False, "direct"), (True, "consolidated")):
            root = open_zarr_root(
                plan.source_zarr,
                mode="r",
                use_consolidated=mode,
            )
            parent = root[CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH]
            if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
                raise ChaserInputProvenanceProxyMaterializationError(
                    f"{label} proxy parent selectors changed."
                )
            result = _validate_group(
                root[plan.run_path],
                expected_manifest=manifest,
                expected_provenance=provenance,
                require_complete=True,
                label=label,
            )
            if not result["valid"]:
                raise ChaserInputProvenanceProxyMaterializationError(
                    f"{label} proxy validation failed: {result}"
                )

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="chaser-input-provenance-proxy-publish",
            publish_schema_id=CHASER_INPUT_PROVENANCE_PROXY_PUBLISH_SCHEMA_ID,
            policy="named_selector_ineligible_input_provenance_proxy_v1",
            rollback_policy="retain_failed_tombstone_leave_parent_selectors_untouched",
            content_checksum=True,
            persist_run_receipt=False,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=finalize,
        repair_failed_publication_visibility=(
            lambda _target: consolidate_metadata_capture_expected_warnings(
                plan.source_zarr
            )
        ),
        accept_persisted_activation_on_callback_error=False,
        payload_metadata={
            "selector_eligible": False,
            "selection": "none",
            "prepared_manifest_sha256": prepared.payload_digest,
            "run_path": plan.run_path,
        },
    )
    return {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "published_selector_ineligible",
        "run_name": plan.run_name,
        "run_path": plan.run_path,
        "target_run_path": str(plan.target_run_path),
        "prepared_manifest_sha256": prepared.payload_digest,
        "selector_eligible": False,
        "selection": "none",
        "publication": _plain_json(publication, field="publication receipt"),
    }


__all__ = [
    "MANIFEST_ATTR",
    "MANIFEST_DIGEST_ATTR",
    "ChaserInputProvenanceProxyMaterializationError",
    "ChaserInputProvenanceProxyMaterializationPlan",
    "build_chaser_input_provenance_proxy_materialization_plan",
    "materialize_chaser_input_provenance_proxy",
]
