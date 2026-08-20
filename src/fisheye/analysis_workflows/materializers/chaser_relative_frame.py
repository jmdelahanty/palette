"""Materialize one prepared chaser-relative frame candidate atomically.

The scientific computation and typed flattening live in
``chaser_relative_frame_storage``.  This module is intentionally only the
storage boundary: it writes a complete node-local candidate, validates every
declared array and digest, and publishes one named selector-ineligible child
under ``analysis/chaser_relative_frame_runs``.

No parent selector is owned by this materializer.  A successful publication is
therefore a durable immutable candidate, never an implicit default.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from ..chaser_relative_frame_storage import (
    PreparedChaserRelativeFrame,
    validate_prepared_chaser_relative_frame,
)
from ...shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from ...shared.coordinate_frame_record import array_values_sha256
from ...shared.json_safety import json_attr_safe
from ...shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from ...shared.zarr.chaser_relative_frame_schema import (
    CHASER_RELATIVE_FRAME_LAYOUT,
    CHASER_RELATIVE_FRAME_SCHEMA_ID,
    CHASER_RELATIVE_FRAME_SCHEMA_V1,
    ChaserRelativeFrameDimensions,
)
from ...shared.zarr.manifest_digest import canonical_json_sha256
from ...shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from ...shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


PARENT_PATH = "analysis/chaser_relative_frame_runs"
MATERIALIZATION_SCHEMA_ID = "palette.chaser_relative_frame_materialization"
MATERIALIZATION_SCHEMA_VERSION = 1
PUBLISH_SCHEMA_ID = "palette.chaser_relative_frame_run_publish.v1"
MANIFEST_ATTR = "chaser_relative_frame_manifest"
MANIFEST_DIGEST_ATTR = "chaser_relative_frame_manifest_sha256"
RUN_PATH_ATTR = "run_path"

_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_provider",
    "latest_any",
    "latest_materialized",
    "latest_composite",
    "latest_pending",
    "authoritative_run",
    "authoritative_run_provenance",
    "active_run",
    "active",
    "current_run",
    "current",
    "default_run",
    "default",
    "selected_run",
    "selected",
    "publication_generation",
    "publication_policy",
)
_BASE_AND_BODY = ("base", "body")


class ChaserRelativeFrameMaterializationError(RuntimeError):
    """Raised when a prepared relative-frame candidate cannot be published."""


def _safe_run_name(value: str) -> str:
    if type(value) is not str:
        raise TypeError("run_name must be one exact string.")
    name = value.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Unsafe chaser-relative frame run name: {value!r}.")
    return name


def _selector_snapshot(parent: Any | None) -> dict[str, Any]:
    if parent is None:
        return {}
    return {
        name: json_attr_safe(parent.attrs[name])
        for name in _SELECTOR_ATTRS
        if name in parent.attrs
    }


def _normal_json(value: object, *, field: str) -> Any:
    def plain(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    try:
        encoded = json.dumps(
            plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ChaserRelativeFrameMaterializationError(
            f"{field} is not strict JSON: {exc}"
        ) from exc


def _array_node(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    if not isinstance(node, zarr.Array):
        raise ChaserRelativeFrameMaterializationError(
            f"Declared array path is not a Zarr array: {path!r}."
        )
    return node


def _array_paths(group: Any, prefix: str) -> set[str]:
    parent = group.get(prefix)
    if not isinstance(parent, zarr.Group):
        raise ChaserRelativeFrameMaterializationError(
            f"Required array group {prefix!r} is missing."
        )
    names = set(str(name) for name in parent.array_keys())
    if set(str(name) for name in parent.group_keys()):
        raise ChaserRelativeFrameMaterializationError(
            f"Array group {prefix!r} contains nested groups."
        )
    return {f"{prefix}/{name}" for name in names}


def _read_array_values(group: Any, path: str) -> np.ndarray:
    node = _array_node(group, path)
    values = np.asarray(node[...])
    if values.dtype.hasobject or values.dtype.kind in {"U", "S"}:
        raise ChaserRelativeFrameMaterializationError(
            f"Published array {path!r} has non-numeric/string dtype {values.dtype}."
        )
    return values


def _expected_declarations(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list) or not declarations:
        raise ChaserRelativeFrameMaterializationError(
            "Manifest array_declarations must be a non-empty list."
        )
    result: dict[str, Mapping[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise ChaserRelativeFrameMaterializationError(
                "Manifest array declaration is not an object."
            )
        path = declaration.get("path")
        if type(path) is not str or path in result:
            raise ChaserRelativeFrameMaterializationError(
                "Manifest array paths must be unique exact strings."
            )
        if path.split("/", 1)[0] not in _BASE_AND_BODY or "/" not in path:
            raise ChaserRelativeFrameMaterializationError(
                f"Manifest array path is outside base/body: {path!r}."
            )
        result[path] = declaration
    return result


def _validate_group(
    group: Any,
    *,
    expected_manifest: Mapping[str, Any],
    label: str,
    require_complete: bool = True,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one direct or consolidated run view, including payload bytes."""

    errors: list[str] = []
    attrs = dict(group.attrs)
    expected = _normal_json(expected_manifest, field="expected_manifest")
    actual_manifest = attrs.get(MANIFEST_ATTR)
    if actual_manifest != expected:
        errors.append(f"{label}: run manifest differs from prepared manifest")
    expected_digest = canonical_json_sha256(expected)
    if attrs.get(MANIFEST_DIGEST_ATTR) != expected_digest:
        errors.append(f"{label}: manifest digest is stale or mismatched")
    if attrs.get("schema_id") != CHASER_RELATIVE_FRAME_SCHEMA_ID:
        errors.append(f"{label}: schema_id is invalid")
    if attrs.get("schema_version") != 1:
        errors.append(f"{label}: schema_version is invalid")
    if attrs.get("layout") != CHASER_RELATIVE_FRAME_LAYOUT:
        errors.append(f"{label}: layout is invalid")
    if attrs.get("stage_selector_eligible") is not False:
        errors.append(f"{label}: stage_selector_eligible is not false")
    if attrs.get("selector_eligible") is not False:
        errors.append(f"{label}: selector_eligible is not false")
    if attrs.get("selection") != "none":
        errors.append(f"{label}: selection is not none")
    if require_complete and attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append(f"{label}: completion status is not complete")
    provenance = attrs.get("run_provenance")
    provenance_result = validate_run_provenance(provenance)
    if not provenance_result.valid:
        errors.append(
            f"{label}: run provenance is invalid: "
            + "; ".join(provenance_result.errors)
        )
    if expected_provenance is not None and provenance != dict(expected_provenance):
        errors.append(f"{label}: run provenance differs from prepared provenance")

    try:
        declarations = _expected_declarations(expected)
        actual_paths = _array_paths(group, "base")
        body_present = bool(
            expected.get("schema_binding", {}).get("body_extension_present", False)
        )
        if body_present:
            actual_paths |= _array_paths(group, "body")
        elif group.get("body") is not None:
            errors.append(f"{label}: unexpected body extension group")
        if actual_paths != set(declarations):
            errors.append(
                f"{label}: array paths differ; expected={sorted(declarations)}, "
                f"actual={sorted(actual_paths)}"
            )
        dimensions_payload = expected.get("dimensions")
        if not isinstance(dimensions_payload, Mapping):
            raise ChaserRelativeFrameMaterializationError(
                "Manifest dimensions are missing."
            )
        dimensions = ChaserRelativeFrameDimensions(
            n_rows=int(dimensions_payload["n_rows"])
        )
        base_arrays: dict[str, np.ndarray] = {}
        body_arrays: dict[str, np.ndarray] = {}
        for path, declaration in declarations.items():
            values = _read_array_values(group, path)
            expected_dtype = declaration.get("dtype")
            expected_shape = declaration.get("shape")
            if values.dtype.str != expected_dtype:
                errors.append(
                    f"{label}: {path} dtype {values.dtype.str!r} differs from "
                    f"{expected_dtype!r}"
                )
            if list(values.shape) != expected_shape:
                errors.append(
                    f"{label}: {path} shape {list(values.shape)!r} differs from "
                    f"{expected_shape!r}"
                )
            if array_values_sha256(values) != declaration.get("content_sha256"):
                errors.append(f"{label}: {path} content digest mismatch")
            (body_arrays if path.startswith("body/") else base_arrays)[
                path.split("/", 1)[1]
            ] = values
        if not errors:
            CHASER_RELATIVE_FRAME_SCHEMA_V1.require(
                base_arrays,
                dimensions=dimensions,
                body_arrays=body_arrays or None,
            )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"{label}: schema/content validation failed: {exc}")

    return {
        "valid": not errors,
        "errors": errors,
        "manifest_sha256": expected_digest,
        "array_count": len(expected_manifest.get("array_declarations", [])),
    }


def validate_chaser_relative_frame_run(
    path: str | Path,
    *,
    expected_manifest: Mapping[str, Any],
    use_consolidated: bool = False,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a candidate run without changing its archive."""

    candidate = Path(path).expanduser().resolve()
    if use_consolidated:
        root = None
        relative_path = ""
        for ancestor in (candidate, *candidate.parents):
            metadata_path = ancestor / "zarr.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, Mapping) or not isinstance(
                metadata.get("consolidated_metadata"), Mapping
            ):
                continue
            root = open_zarr_root(ancestor, mode="r", use_consolidated=True)
            relative_path = candidate.relative_to(ancestor).as_posix()
            break
        if root is None:
            raise ChaserRelativeFrameMaterializationError(
                "Could not resolve an archive-root consolidated metadata view."
            )
        run = root if not relative_path else root[relative_path]
        label = "consolidated"
    else:
        run = open_zarr_root(candidate, mode="r", use_consolidated=False)
        label = "direct"
    return _validate_group(
        run,
        expected_manifest=expected_manifest,
        label=label,
        expected_provenance=expected_provenance,
    )


@dataclass(frozen=True, slots=True)
class ChaserRelativeFrameMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    run_name: str
    prepared: PreparedChaserRelativeFrame
    parent_selector_attrs: Mapping[str, Any]
    run_provenance: Mapping[str, Any]

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{self.run_name}"

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
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
            "run_path": self.run_path,
            "run_name": self.run_name,
            "prepared_manifest_sha256": self.prepared.payload_digest,
            "selector_eligible": False,
            "selection": "none",
            "parent_selector_attrs": dict(self.parent_selector_attrs),
            "run_provenance_sha256": canonical_json_sha256(
                _normal_json(self.run_provenance, field="run_provenance")
            ),
        }


def build_chaser_relative_frame_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    prepared: PreparedChaserRelativeFrame,
) -> ChaserRelativeFrameMaterializationPlan:
    if not isinstance(prepared, PreparedChaserRelativeFrame):
        raise TypeError("prepared must be one PreparedChaserRelativeFrame.")
    validate_prepared_chaser_relative_frame(prepared)
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
    target = source / PARENT_PATH / name
    if target.exists():
        raise FileExistsError(f"Refusing existing target run: {target}")
    root = open_zarr_root(source, mode="r", use_consolidated=False)
    source_recording_id = root.attrs.get("recording_id")
    prepared_recording_id = prepared.manifest.get("recording_id")
    if source_recording_id != prepared_recording_id:
        raise ChaserRelativeFrameMaterializationError(
            "Source root recording_id does not exactly match the prepared "
            f"recording_id: source={source_recording_id!r}, "
            f"prepared={prepared_recording_id!r}."
        )
    parent = root.get(PARENT_PATH)
    if parent is not None and not isinstance(parent, zarr.Group):
        raise ChaserRelativeFrameMaterializationError(
            f"Target parent is not a Zarr group: {PARENT_PATH}"
        )
    run_provenance = build_writer_run_provenance(
        command="chaser_relative_frame_materializer",
        params={
            "run_name": name,
            "run_path": f"{PARENT_PATH}/{name}",
            "prepared_manifest_sha256": prepared.payload_digest,
            "schema_id": CHASER_RELATIVE_FRAME_SCHEMA_ID,
            "schema_version": 1,
            "source_authorities": prepared.manifest.get("source_authorities"),
        },
        input_run_ids={
            "recording_id": str(prepared_recording_id),
            "prepared_chaser_relative_frame": prepared.payload_digest,
        },
        input_artifacts=[
            {
                "kind": "prepared_chaser_relative_frame_manifest",
                "recording_id": str(prepared_recording_id),
                "sha256": prepared.payload_digest,
            },
            {
                "kind": "source_provider_authorities",
                "record": prepared.manifest.get("source_authorities"),
                "sha256": canonical_json_sha256(
                    prepared.manifest.get("source_authorities")
                ),
            },
        ],
        cwd=Path(__file__).resolve().parents[4],
    )
    return ChaserRelativeFrameMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=scratch / "chaser-relative-frame-candidate.zarr",
        run_name=name,
        prepared=prepared,
        parent_selector_attrs=MappingProxyType(_selector_snapshot(parent)),
        run_provenance=MappingProxyType(run_provenance),
    )


def _chunks_for(values: np.ndarray) -> tuple[int, ...]:
    # Writes are serial and each array is an independent ownership unit.  Keep
    # the row dimension bounded for readers while never creating a zero chunk.
    return tuple(max(1, min(int(size), 16_384)) for size in values.shape)


def _write_local_candidate(plan: ChaserRelativeFrameMaterializationPlan) -> None:
    if plan.local_zarr.exists():
        raise FileExistsError(f"Refusing existing local candidate: {plan.local_zarr}")
    plan.local_zarr.parent.mkdir(parents=True, exist_ok=True)
    local_root = open_zarr_root(plan.local_zarr, mode="w-")
    parent = require_runs_parent(
        local_root.require_group("analysis"),
        "chaser_relative_frame_runs",
    )
    run = parent.create_group(plan.run_name)
    prepared = plan.prepared
    manifest = _normal_json(prepared.manifest, field="prepared.manifest")
    run.attrs.update(
        {
            "schema_id": CHASER_RELATIVE_FRAME_SCHEMA_ID,
            "schema_version": 1,
            "layout": CHASER_RELATIVE_FRAME_LAYOUT,
            "stage_selector_eligible": False,
            "selector_eligible": False,
            "selection": "none",
            "run_path": plan.run_path,
            "materialization_schema_id": MATERIALIZATION_SCHEMA_ID,
            "materialization_schema_version": MATERIALIZATION_SCHEMA_VERSION,
            MANIFEST_ATTR: json_attr_safe(manifest),
            MANIFEST_DIGEST_ATTR: canonical_json_sha256(manifest),
            "run_provenance": json_attr_safe(dict(plan.run_provenance)),
        }
    )
    mark_run_started(
        run,
        run_name=plan.run_name,
        stage="chaser_relative_frame_materialization",
    )
    for prefix, arrays in (("base", prepared.base_arrays), ("body", prepared.body_arrays)):
        if arrays is None:
            continue
        group = run.create_group(prefix)
        for name in sorted(arrays):
            values = np.asarray(arrays[name])
            group.create_array(
                name,
                data=values,
                chunks=_chunks_for(values),
                overwrite=False,
            )
    local_validation = _validate_group(
        run,
        expected_manifest=manifest,
        label="local",
        require_complete=False,
        expected_provenance=plan.run_provenance,
    )
    if not local_validation["valid"]:
        raise ChaserRelativeFrameMaterializationError(
            f"Local candidate validation failed: {local_validation}"
        )
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=plan.run_name,
        run_provenance=plan.run_provenance,
    )
    final_validation = _validate_group(
        run,
        expected_manifest=manifest,
        label="local-complete",
        expected_provenance=plan.run_provenance,
    )
    if not final_validation["valid"]:
        raise ChaserRelativeFrameMaterializationError(
            f"Local candidate completion validation failed: {final_validation}"
        )


def _publish_prepared_run(
    plan: ChaserRelativeFrameMaterializationPlan,
    *,
    copy_backend: str,
) -> dict[str, Any]:
    expected_manifest = _normal_json(plan.prepared.manifest, field="prepared.manifest")

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_group(
            open_zarr_root(path, mode="r", use_consolidated=False),
            expected_manifest=expected_manifest,
            label="publication",
            expected_provenance=plan.run_provenance,
        )

    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (
            require_runs_parent(
                root.require_group("analysis"),
                "chaser_relative_frame_runs",
            ),
        )

    def complete(_root: zarr.Group, _parent: zarr.Group, run: zarr.Group) -> None:
        parent = _parent
        if parent.attrs.get("latest_pending") == plan.run_name:
            raise ChaserRelativeFrameMaterializationError(
                "Refusing to clear an existing latest_pending selector for this run."
            )
        run.attrs["stage_selector_eligible"] = False
        run.attrs["selector_eligible"] = False
        run.attrs["selection"] = "none"
        run.attrs["run_provenance"] = json_attr_safe(dict(plan.run_provenance))
        mark_run_complete(
            run,
            parent_group=parent,
            run_name=plan.run_name,
            run_provenance=plan.run_provenance,
        )

    def verify(root: zarr.Group) -> None:
        parent = root[PARENT_PATH]
        if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
            raise ChaserRelativeFrameMaterializationError(
                "Selector-ineligible publication changed parent selectors."
            )
        result = validate(
            plan.source_zarr / plan.run_path,
        )
        if not result["valid"]:
            raise ChaserRelativeFrameMaterializationError(
                f"Published relative-frame run is invalid: {result}"
            )

    def finalize(_root: zarr.Group, _parent: zarr.Group, _run: zarr.Group) -> None:
        before = _selector_snapshot(
            open_zarr_root(plan.source_zarr, mode="r", use_consolidated=False).get(
                PARENT_PATH
            )
        )
        if before != dict(plan.parent_selector_attrs):
            raise ChaserRelativeFrameMaterializationError(
                "Parent selectors changed before archive reconsolidation."
            )
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        receipt = validate_direct_consolidated_subtree(
            plan.source_zarr,
            subtree_path=plan.run_path,
        )
        direct_root = open_zarr_root(
            plan.source_zarr,
            mode="r",
            use_consolidated=False,
        )
        consolidated_root = open_zarr_root(
            plan.source_zarr,
            mode="r",
            use_consolidated=True,
        )
        direct_parent = direct_root[PARENT_PATH]
        consolidated_parent = consolidated_root[PARENT_PATH]
        if _selector_snapshot(direct_parent) != dict(plan.parent_selector_attrs):
            raise ChaserRelativeFrameMaterializationError(
                "Direct parent selectors changed during publication."
            )
        if _selector_snapshot(consolidated_parent) != dict(plan.parent_selector_attrs):
            raise ChaserRelativeFrameMaterializationError(
                "Consolidated parent selectors changed during publication."
            )
        direct_run = direct_root[plan.run_path]
        consolidated_run = consolidated_root[plan.run_path]
        direct_validation = _validate_group(
            direct_run,
            expected_manifest=expected_manifest,
            label="direct",
            expected_provenance=plan.run_provenance,
        )
        consolidated_validation = _validate_group(
            consolidated_run,
            expected_manifest=expected_manifest,
            label="consolidated",
            expected_provenance=plan.run_provenance,
        )
        if not direct_validation["valid"] or not consolidated_validation["valid"]:
            raise ChaserRelativeFrameMaterializationError(
                "Direct/consolidated relative-frame validation failed: "
                f"direct={direct_validation}, consolidated={consolidated_validation}"
            )
        if direct_run.attrs.get(MANIFEST_ATTR) != consolidated_run.attrs.get(
            MANIFEST_ATTR
        ) or direct_run.attrs.get(MANIFEST_DIGEST_ATTR) != consolidated_run.attrs.get(
            MANIFEST_DIGEST_ATTR
        ):
            raise ChaserRelativeFrameMaterializationError(
                "Direct and consolidated run manifests do not agree."
            )
        if receipt.array_count != len(expected_manifest["array_declarations"]):
            raise ChaserRelativeFrameMaterializationError(
                "Direct/consolidated metadata receipt omitted declared arrays."
            )

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="chaser-relative-frame-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy="named_selector_ineligible_chaser_relative_frame_atomic_publish_v1",
            rollback_policy=(
                "retain_failed_tombstone_leave_parent_selectors_untouched"
            ),
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
            "prepared_manifest_sha256": plan.prepared.payload_digest,
            "run_path": plan.run_path,
        },
    )
    return publication


def materialize_chaser_relative_frame(
    source_zarr: str | Path,
    *,
    prepared: PreparedChaserRelativeFrame,
    scratch_root: str | Path,
    run_name: str,
    copy_backend: str = "python",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Prepare or publish one immutable selector-ineligible relative-frame run."""

    plan = build_chaser_relative_frame_materialization_plan(
        source_zarr,
        scratch_root=scratch_root,
        run_name=run_name,
        prepared=prepared,
    )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "planned" if not apply else "running",
        "selector_eligible": False,
        "selection": "none",
        "plan": plan.to_json(),
    }
    if not apply:
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    prepared_validation = validate_prepared_chaser_relative_frame(prepared)
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    started = time.perf_counter()
    try:
        _write_local_candidate(plan)
        local_validation = validate_chaser_relative_frame_run(
            plan.local_run_path,
            expected_manifest=prepared.manifest,
        )
        if not local_validation["valid"]:
            raise ChaserRelativeFrameMaterializationError(
                f"Local candidate is invalid: {local_validation}"
            )
        publication = _publish_prepared_run(plan, copy_backend=copy_backend)
        result.update(
            status="complete",
            prepared_validation=prepared_validation,
            local_validation=local_validation,
            compute_duration_seconds=float(time.perf_counter() - started),
            publication=publication,
        )
        succeeded = True
        return json_attr_safe(result)
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


publish_chaser_relative_frame_run = _publish_prepared_run
materialize_chaser_relative_frame_run = materialize_chaser_relative_frame


__all__ = [
    "MANIFEST_ATTR",
    "MANIFEST_DIGEST_ATTR",
    "MATERIALIZATION_SCHEMA_ID",
    "PARENT_PATH",
    "PUBLISH_SCHEMA_ID",
    "ChaserRelativeFrameMaterializationError",
    "ChaserRelativeFrameMaterializationPlan",
    "build_chaser_relative_frame_materialization_plan",
    "materialize_chaser_relative_frame",
    "materialize_chaser_relative_frame_run",
    "publish_chaser_relative_frame_run",
    "validate_chaser_relative_frame_run",
]
