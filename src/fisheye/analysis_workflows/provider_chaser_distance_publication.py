"""Receipt-backed publication and reading for provider chaser distances.

The pure successor in :mod:`provider_chaser_distance_successor` is useful for
tests and comparisons, but consumers need an immutable Zarr run with the same
lineage and completion conventions as the rest of Palette.  This module adds
that deliberately selector-ineligible publication boundary.

The ordinary path trusts a complete schema-v2 candidate-chain receipt.  It
therefore reads the receipt-bound source arrays once and validates their typed
declarations, but does not recompute upstream dense-array content digests.  A
separately named deep-audit helper remains available for maintenance and
investigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    ChaserRelativeFrameSourceHandle,
    load_chaser_relative_frame_source_handle_from_receipt,
)
from fisheye.analysis_workflows.chaser_proxy_candidate_receipt import (
    validate_chaser_proxy_candidate_receipt_for_source_load,
)
from fisheye.analysis_workflows.provider_chaser_distance_successor import (
    PREPARED_PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
    PreparedProviderChaserDistance,
    prepare_provider_chaser_distance_successor,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.provider_chaser_distance_schema import (
    PROVIDER_CHASER_DISTANCE_LAYOUT,
    PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
    PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
    PROVIDER_CHASER_DISTANCE_SCHEMA_V1,
    ProviderChaserDistanceDimensions,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


PROVIDER_CHASER_DISTANCE_RUNS_PARENT_PATH = (
    "analysis/provider_chaser_distance_runs"
)
PROVIDER_CHASER_DISTANCE_RUNS_PREFIX = (
    f"{PROVIDER_CHASER_DISTANCE_RUNS_PARENT_PATH}/"
)
PROVIDER_CHASER_DISTANCE_PUBLICATION_SCHEMA_ID = (
    "palette.analysis.provider_chaser_distance.publication"
)
PROVIDER_CHASER_DISTANCE_PUBLICATION_SCHEMA_VERSION = 1
PROVIDER_CHASER_DISTANCE_MANIFEST_ATTR = "provider_chaser_distance_manifest"
PROVIDER_CHASER_DISTANCE_MANIFEST_DIGEST_ATTR = (
    "provider_chaser_distance_manifest_sha256"
)
PROVIDER_CHASER_DISTANCE_PUBLICATION_POLICY = (
    "immutable_receipt_backed_selector_ineligible_v1"
)
PROVIDER_CHASER_DISTANCE_DEEP_AUDIT_POLICY = (
    "explicit_recompute_declared_array_hashes_v1"
)

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SELECTOR_ALIASES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "current_run",
        "default",
        "selected",
        "selected_run",
        "authoritative",
        "authoritative_run",
    }
)
_FORBIDDEN_SELECTOR_ATTRS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "selected",
        "selected_run",
        "current",
        "current_run",
        "default",
        "default_run",
    }
)


class ProviderChaserDistancePublicationError(ValueError):
    """Raised when a provider chaser-distance publication is invalid."""


def _fail(message: str) -> None:
    raise ProviderChaserDistancePublicationError(message)


def _run_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or value in _SELECTOR_ALIASES
        or _RUN_NAME_RE.fullmatch(value) is None
    ):
        _fail("run_name must be one concrete bare run name, not a selector or path.")
    return value


def _digest(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field_name} must be one lowercase SHA-256 digest.")
    return value


def _strict_json_object(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be one JSON object.")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"{field_name} is not strict JSON: {exc}")
    if not isinstance(decoded, dict):  # pragma: no cover
        _fail(f"{field_name} must decode to an object.")
    return decoded


def _copy_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_archive(value: str | Path) -> Path:
    archive = Path(value).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}")
    return archive


def _canonical_run_path(run_name: str) -> str:
    return f"{PROVIDER_CHASER_DISTANCE_RUNS_PREFIX}{_run_name(run_name)}"


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    value = receipt.get("record_sha256")
    return _digest(value, field_name="receipt.record_sha256")


def _array_declarations(
    run: Any,
    *,
    expected: list[Mapping[str, Any]],
    verify_content_hashes: bool,
) -> dict[str, np.ndarray]:
    expected_paths = [str(item.get("path")) for item in expected]
    if len(set(expected_paths)) != len(expected_paths):
        _fail("Persistent manifest has duplicate array declarations.")
    try:
        actual_paths = sorted(str(path) for path in run.array_keys())
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(f"Unable to enumerate persistent arrays: {exc}")
    if actual_paths != sorted(expected_paths):
        _fail("Persistent array paths differ from the immutable manifest.")
    arrays: dict[str, np.ndarray] = {}
    for declaration in expected:
        path = declaration.get("path")
        if type(path) is not str or not path:
            _fail("Persistent array declaration path is invalid.")
        try:
            node = run[path]
            if not isinstance(node, zarr.Array):
                raise TypeError("not an array")
            value = np.asarray(node[...])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            _fail(f"Unable to read persistent array {path!r}: {exc}")
        if value.dtype.str != declaration.get("dtype"):
            _fail(f"Persistent array {path!r} dtype differs from its declaration.")
        if list(value.shape) != declaration.get("shape"):
            _fail(f"Persistent array {path!r} shape differs from its declaration.")
        if verify_content_hashes:
            observed = array_values_sha256(value)
            if observed != declaration.get("content_sha256"):
                _fail(f"Persistent array {path!r} content digest differs.")
        value.setflags(write=False)
        arrays[path] = value
    return arrays


def _manifest_from_run(run: Any) -> tuple[dict[str, Any], str]:
    raw = run.attrs.get(PROVIDER_CHASER_DISTANCE_MANIFEST_ATTR)
    manifest = _strict_json_object(
        raw, field_name=PROVIDER_CHASER_DISTANCE_MANIFEST_ATTR
    )
    stored = run.attrs.get(PROVIDER_CHASER_DISTANCE_MANIFEST_DIGEST_ATTR)
    _digest(stored, field_name=PROVIDER_CHASER_DISTANCE_MANIFEST_DIGEST_ATTR)
    observed = canonical_json_sha256(manifest)
    if stored != observed:
        _fail("Persistent provider chaser-distance manifest digest is stale.")
    return manifest, observed


def _validate_embedded_source_receipt(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the durable receipt copy without reopening its source arrays."""

    raw_receipt = manifest.get("source_receipt")
    receipt = _strict_json_object(raw_receipt, field_name="manifest.source_receipt")
    embedded_digest = manifest.get("source_receipt_sha256")
    receipt_digest = _receipt_digest(receipt)
    if embedded_digest != receipt_digest:
        _fail("Embedded source receipt digest differs from its record_sha256.")
    try:
        validated = validate_chaser_proxy_candidate_receipt_for_source_load(
            receipt,
            expected_analysis_zarr=manifest.get("analysis_zarr"),
            expected_recording_id=manifest.get("recording_id"),
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Embedded source receipt is not a valid bounded authority: {exc}")
    return validated


def _validate_persistent_run(
    path: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_run_path: str | None = None,
    verify_content_hashes: bool = False,
    run: Any | None = None,
) -> dict[str, Any]:
    if run is None:
        try:
            run = open_zarr_root(path, mode="r", use_consolidated=False)
        except (OSError, TypeError, ValueError) as exc:
            _fail(f"Unable to open provider chaser-distance run: {exc}")
    manifest, manifest_digest = _manifest_from_run(run)
    if expected_manifest is not None and manifest != dict(expected_manifest):
        _fail("Persistent run manifest differs from the prepared publication.")
    if expected_run_path is not None and manifest.get("run_path") != expected_run_path:
        _fail("Persistent run path differs from its exact publication path.")
    if manifest.get("schema_id") != PROVIDER_CHASER_DISTANCE_SCHEMA_ID:
        _fail("Persistent run has the wrong logical schema identity.")
    if manifest.get("schema_version") != PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION:
        _fail("Persistent run has the wrong logical schema version.")
    if manifest.get("storage_schema") != {
        "schema_id": PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
        "schema_version": PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
        "layout": PROVIDER_CHASER_DISTANCE_LAYOUT,
    }:
        _fail("Persistent run storage schema binding is invalid.")
    _validate_embedded_source_receipt(manifest)
    if (
        manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
        or manifest.get("production_authority") is not False
        or manifest.get("registry_update") is not False
        or manifest.get("production_selector_activation") is not False
    ):
        _fail("Persistent run is not explicitly selector-ineligible.")
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        _fail("Persistent provider chaser-distance run is not complete.")
    if run.attrs.get("stage_selector_eligible") is not False:
        _fail("Persistent provider chaser-distance run is selector eligible.")
    if set(run.attrs).intersection(_FORBIDDEN_SELECTOR_ATTRS):
        _fail("Persistent run contains forbidden selector pointer attributes.")
    if not isinstance(run.attrs.get("run_provenance"), Mapping):
        _fail("Persistent provider chaser-distance run provenance is missing.")
    dimensions_raw = manifest.get("dimensions")
    if not isinstance(dimensions_raw, Mapping):
        _fail("Persistent run dimensions are missing.")
    try:
        dimensions = ProviderChaserDistanceDimensions(
            n_frames=int(dimensions_raw["n_frames"]),
            n_chasers=int(dimensions_raw["n_chasers"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"Persistent run dimensions are invalid: {exc}")
    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list):
        _fail("Persistent run array declarations are missing.")
    arrays = _array_declarations(
        run,
        expected=declarations,
        verify_content_hashes=verify_content_hashes,
    )
    try:
        PROVIDER_CHASER_DISTANCE_SCHEMA_V1.require(arrays, dimensions=dimensions)
    except ValueError as exc:
        _fail(f"Persistent provider chaser-distance arrays violate the schema: {exc}")
    return {
        "valid": True,
        "manifest_sha256": manifest_digest,
        "run_path": manifest["run_path"],
        "array_count": len(arrays),
        "row_count": dimensions.n_rows,
        "arrays": arrays,
    }


def _compact_persistent_validation(validation: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded evidence suitable for an atomic publication receipt.

    ``_validate_persistent_run`` returns the loaded arrays because readers need
    them after validation.  Atomic publication receipts are metadata, however,
    and must never embed those row-level payloads.  The manifest digest binds
    the full array declarations and their one-time content hashes; readable
    array paths, counts, and row cardinality are sufficient here.
    """

    arrays = validation.get("arrays")
    if not isinstance(arrays, Mapping):
        _fail("Persistent validation did not return its exact array mapping.")
    return {
        "valid": validation.get("valid") is True,
        "manifest_sha256": validation.get("manifest_sha256"),
        "run_path": validation.get("run_path"),
        "array_count": validation.get("array_count"),
        "row_count": validation.get("row_count"),
        "array_paths": sorted(str(path) for path in arrays),
        "row_evidence_storage": "zarr_arrays_not_publication_metadata",
    }


@dataclass(frozen=True, slots=True)
class ProviderChaserDistancePublicationPlan:
    """All immutable inputs needed for one selector-ineligible publication."""

    analysis_zarr: Path
    run_name: str
    run_path: str
    receipt: Mapping[str, Any] = field(repr=False)
    receipt_digest: str
    prepared: PreparedProviderChaserDistance = field(repr=False)
    manifest: Mapping[str, Any] = field(repr=False)
    run_provenance: Mapping[str, Any] = field(repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": PROVIDER_CHASER_DISTANCE_PUBLICATION_SCHEMA_ID,
            "schema_version": PROVIDER_CHASER_DISTANCE_PUBLICATION_SCHEMA_VERSION,
            "status": "dry_run_plan",
            "analysis_zarr": str(self.analysis_zarr),
            "run_name": self.run_name,
            "run_path": self.run_path,
            "receipt_digest": self.receipt_digest,
            "manifest_sha256": canonical_json_sha256(dict(self.manifest)),
            "dimensions": dict(self.prepared.manifest["dimensions"]),
            "array_count": len(self.prepared.arrays),
            "array_paths": sorted(self.prepared.arrays),
            "selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "registry_update": False,
            "target_exists": (self.analysis_zarr / self.run_path).exists(),
            "upstream_verification_mode": "receipt_backed",
            "upstream_dense_hash_recomputation": False,
        }


def _publication_manifest(
    prepared: PreparedProviderChaserDistance,
    *,
    receipt: Mapping[str, Any],
    receipt_digest: str,
    run_name: str,
    run_path: str,
    analysis_zarr: Path,
) -> dict[str, Any]:
    prepared_manifest = prepared.to_json()
    source_receipt = _copy_json(receipt)
    payload: dict[str, Any] = {
        "schema_id": PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
        "schema_version": PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
        "storage_schema": {
            "schema_id": PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
            "schema_version": PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
            "layout": PROVIDER_CHASER_DISTANCE_LAYOUT,
        },
        "run_path": run_path,
        "run_name": run_name,
        "recording_id": prepared_manifest["recording_id"],
        "status": "complete_selector_ineligible",
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "production_selector_activation": False,
        "registry_update": False,
        "publication_policy": PROVIDER_CHASER_DISTANCE_PUBLICATION_POLICY,
        "analysis_zarr": str(analysis_zarr),
        "dimensions": _copy_json(prepared_manifest["dimensions"]),
        "array_declarations": _copy_json(prepared_manifest["array_declarations"]),
        "source_receipt": source_receipt,
        "source_receipt_sha256": receipt_digest,
        "source_relative_frame": _copy_json(prepared_manifest["source"]),
        "source_provider_authorities": _copy_json(
            prepared_manifest["provider_authorities"]
        ),
        "source_handle_verification": {
            "verification_mode": "receipt_backed",
            "receipt_digest": receipt_digest,
            "verification_digest": prepared_manifest["source"]["verification_digest"],
        },
        "coordinate_policy": _copy_json(prepared_manifest["coordinate_policy"]),
        "scale_policy": _copy_json(prepared_manifest["scale_policy"]),
        "timing_policy": _copy_json(prepared_manifest["timing_policy"]),
        "temporal_alignment": _copy_json(prepared_manifest["temporal_alignment"]),
        "denominators": _copy_json(prepared_manifest["denominators"]),
        "denominator_policy": _copy_json(prepared_manifest["denominator_policy"]),
        "optional_fields": _copy_json(prepared_manifest["optional_fields"]),
        "invariants": _copy_json(prepared_manifest["invariants"]),
        "computation": {
            "computation_id": prepared_manifest["computation_id"],
            "prepared_schema_id": PREPARED_PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
            "prepared_schema_version": prepared_manifest["schema_version"],
            "prepared_manifest_sha256": canonical_json_sha256(prepared_manifest),
            "analysis_profile_path": receipt["analysis_profile_path"],
            "source_workflow_software_authority": _copy_json(
                receipt["software_authority"]
            ),
        },
    }
    return {**payload, "payload_digest": canonical_json_sha256(payload)}


def build_provider_chaser_distance_publication_plan(
    analysis_zarr: str | Path,
    *,
    receipt: Mapping[str, Any],
    run_name: str,
    expected_recording_id: str | None = None,
) -> ProviderChaserDistancePublicationPlan:
    """Read and prepare one exact receipt-backed publication plan."""

    archive = _canonical_archive(analysis_zarr)
    name = _run_name(run_name)
    run_path = _canonical_run_path(name)
    receipt_map = _strict_json_object(receipt, field_name="receipt")
    receipt_digest = _receipt_digest(receipt_map)
    validated_receipt = validate_chaser_proxy_candidate_receipt_for_source_load(
        receipt_map,
        expected_analysis_zarr=archive,
        expected_recording_id=expected_recording_id,
    )
    if (archive / run_path).exists():
        raise FileExistsError(f"Refusing to replace existing run: {archive / run_path}")
    recording_id = str(validated_receipt["recording_id"])
    handle = load_chaser_relative_frame_source_handle_from_receipt(
        archive,
        receipt=validated_receipt,
        expected_recording_id=recording_id,
        use_consolidated=True,
    )
    if not isinstance(handle, ChaserRelativeFrameSourceHandle):
        _fail("Receipt-backed source loader did not return its verified handle.")
    prepared = prepare_provider_chaser_distance_successor(handle)
    manifest = _publication_manifest(
        prepared,
        receipt=validated_receipt,
        receipt_digest=receipt_digest,
        run_name=name,
        run_path=run_path,
        analysis_zarr=archive,
    )
    run_provenance = build_writer_run_provenance(
        command="fisheye.analysis_workflows.provider_chaser_distance_publication",
        params={
            "run_name": name,
            "run_path": run_path,
            "publication_policy": PROVIDER_CHASER_DISTANCE_PUBLICATION_POLICY,
            "upstream_receipt_sha256": receipt_digest,
            "upstream_verification_mode": "receipt_backed",
        },
        input_run_ids={
            "candidate_chain_receipt": receipt_digest,
            "relative_frame": prepared.manifest["source"]["run_path"],
        },
    )
    return ProviderChaserDistancePublicationPlan(
        analysis_zarr=archive,
        run_name=name,
        run_path=run_path,
        receipt=MappingProxyType(_copy_json(validated_receipt)),
        receipt_digest=receipt_digest,
        prepared=prepared,
        manifest=MappingProxyType(_copy_json(manifest)),
        run_provenance=MappingProxyType(_copy_json(run_provenance)),
    )


def _write_local_run(
    plan: ProviderChaserDistancePublicationPlan,
    local_run_path: Path,
) -> None:
    if local_run_path.exists():
        raise FileExistsError(f"Local publication path already exists: {local_run_path}")
    local_run_path.parent.mkdir(parents=True, exist_ok=True)
    run = zarr.open_group(
        str(local_run_path), mode="w-", zarr_format=3, use_consolidated=False
    )
    mark_run_started(run, run_name=plan.run_name, stage="provider_chaser_distance")
    run.attrs.update(
        {
            "schema_id": PROVIDER_CHASER_DISTANCE_SCHEMA_ID,
            "schema_version": PROVIDER_CHASER_DISTANCE_SCHEMA_VERSION,
            "stage_selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "production_selector_activation": False,
            "registry_update": False,
            "run_provenance": json_attr_safe(dict(plan.run_provenance)),
            PROVIDER_CHASER_DISTANCE_MANIFEST_ATTR: json_attr_safe(
                dict(plan.manifest)
            ),
            PROVIDER_CHASER_DISTANCE_MANIFEST_DIGEST_ATTR: canonical_json_sha256(
                dict(plan.manifest)
            ),
        }
    )
    for path, values in plan.prepared.arrays.items():
        run.create_array(
            path,
            data=np.asarray(values),
            chunks=(
                max(1, min(int(values.shape[0]), 16384)),
                *tuple(int(size) for size in values.shape[1:]),
            ),
        )
    mark_run_complete(
        run,
        run_name=plan.run_name,
        run_provenance=dict(plan.run_provenance),
    )
    _validate_persistent_run(
        local_run_path,
        expected_manifest=plan.manifest,
        expected_run_path=plan.run_path,
    )


def publish_provider_chaser_distance_run(
    plan: ProviderChaserDistancePublicationPlan,
    *,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Atomically publish one complete, selector-ineligible provider run."""

    scratch_parent = (
        Path(scratch_root).expanduser().resolve()
        if scratch_root is not None
        else None
    )
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{plan.run_name}.provider_chaser_distance.",
        dir=str(scratch_parent) if scratch_parent is not None else None,
    ) as temporary:
        local_run_path = Path(temporary) / "run.zarr"
        _write_local_run(plan, local_run_path)
        parent_snapshot: dict[str, Any] | None = None

        def validate(path: Path) -> Mapping[str, Any]:
            return _compact_persistent_validation(
                _validate_persistent_run(
                    path,
                    expected_manifest=plan.manifest,
                    expected_run_path=plan.run_path,
                )
            )

        def prepare_parents(root: Any) -> tuple[Any]:
            nonlocal parent_snapshot
            analysis = root.require_group("analysis")
            parent = require_runs_parent(
                analysis,
                "provider_chaser_distance_runs",
                completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
            )
            if set(parent.attrs).intersection(_FORBIDDEN_SELECTOR_ATTRS):
                raise ProviderChaserDistancePublicationError(
                    "Provider chaser-distance parent already contains selector pointers."
                )
            if parent_snapshot is None:
                parent_snapshot = dict(parent.attrs)
            return (parent,)

        def complete(_root: Any, parent: Any, run: Any) -> None:
            run.attrs["stage_selector_eligible"] = False
            run.attrs["selection"] = "none"
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=plan.run_name,
                run_provenance=dict(plan.run_provenance),
            )

        def verify(root: Any) -> None:
            parent = root[PROVIDER_CHASER_DISTANCE_RUNS_PARENT_PATH]
            if parent_snapshot is None or dict(parent.attrs) != parent_snapshot:
                raise ProviderChaserDistancePublicationError(
                    "Selector-ineligible publication changed provider parent metadata."
                )
            _validate_persistent_run(
                plan.analysis_zarr / plan.run_path,
                expected_manifest=plan.manifest,
                expected_run_path=plan.run_path,
            )

        result = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.analysis_zarr,
                local_run_path=local_run_path,
                target_run_path=plan.analysis_zarr / plan.run_path,
                run_name=plan.run_name,
                lock_suffix="provider_chaser_distance_publication",
                publish_schema_id=PROVIDER_CHASER_DISTANCE_PUBLICATION_SCHEMA_ID,
                policy=PROVIDER_CHASER_DISTANCE_PUBLICATION_POLICY,
                rollback_policy="retain_failed_selector_ineligible_tombstone_v1",
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare_parents,
            complete_run=complete,
            verify_pointers=verify,
            payload_metadata={
                "recording_id": plan.prepared.manifest["recording_id"],
                "run_path": plan.run_path,
                "source_receipt_sha256": plan.receipt_digest,
                "selector_activation": "none",
                "upstream_verification_mode": "receipt_backed",
            },
        )
    consolidation = consolidate_metadata_capture_expected_warnings(
        plan.analysis_zarr
    )
    metadata_equivalence = validate_direct_consolidated_subtree(
        plan.analysis_zarr, subtree_path=plan.run_path
    ).to_json()
    final = load_provider_chaser_distance_source_handle(
        plan.analysis_zarr,
        run_name=plan.run_name,
        expected_recording_id=plan.prepared.manifest["recording_id"],
        use_consolidated=True,
    )
    return {
        "status": "published_selector_ineligible",
        "run_path": plan.run_path,
        "manifest_sha256": final.manifest_sha256,
        "receipt_digest": plan.receipt_digest,
        "row_count": final.n_rows,
        "array_count": len(final.arrays),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
        "consolidation": consolidation,
        "metadata_equivalence": metadata_equivalence,
        "upstream_dense_hash_recomputation": False,
        "atomic_publication": result,
    }


@dataclass(frozen=True, slots=True, init=False)
class ProviderChaserDistanceSourceHandle:
    """Read-only bounded view of one exact persistent provider run."""

    analysis_zarr: Path
    run_path: str
    run_name: str
    recording_id: str
    manifest: Mapping[str, Any] = field(repr=False)
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    metadata_equivalence: Mapping[str, Any] = field(repr=False)
    verification_mode: str
    _use_consolidated: bool = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _seal: object | None = None, **values: Any) -> None:
        if _seal is not _SOURCE_HANDLE_SEAL:
            raise TypeError("Provider chaser-distance handles require their loader.")
        for name, value in values.items():
            if name in {"manifest", "metadata_equivalence"}:
                value = MappingProxyType(_copy_json(value))
            elif name == "arrays":
                value = MappingProxyType(
                    {
                        key: _readonly_array(array)
                        for key, array in value.items()
                    }
                )
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _SOURCE_HANDLE_SEAL)

    @property
    def manifest_sha256(self) -> str:
        return canonical_json_sha256(dict(self.manifest))

    @property
    def n_rows(self) -> int:
        return int(self.manifest["dimensions"]["n_rows"])

    @property
    def n_frames(self) -> int:
        return int(self.manifest["dimensions"]["n_frames"])

    @property
    def n_chasers(self) -> int:
        return int(self.manifest["dimensions"]["n_chasers"])

    @property
    def source_receipt(self) -> Mapping[str, Any]:
        """Return the embedded bounded source authority, not an operations path."""

        return MappingProxyType(_copy_json(self.manifest["source_receipt"]))

    @property
    def source_receipt_sha256(self) -> str:
        return str(self.manifest["source_receipt_sha256"])

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider chaser-distance array {name!r}.") from exc

    def assert_current(self) -> None:
        refreshed = load_provider_chaser_distance_source_handle(
            self.analysis_zarr,
            run_name=self.run_name,
            expected_recording_id=self.recording_id,
            use_consolidated=self._use_consolidated,
        )
        if refreshed.manifest_sha256 != self.manifest_sha256:
            raise ProviderChaserDistancePublicationError(
                "Published provider chaser-distance run changed after sealing."
            )


_SOURCE_HANDLE_SEAL = object()


def _readonly_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    array.setflags(write=False)
    return array


def load_provider_chaser_distance_source_handle(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
    use_consolidated: bool = True,
    deep_audit: bool = False,
) -> ProviderChaserDistanceSourceHandle:
    """Load one exact persistent run without upstream or ordinary output rehashes."""

    if type(use_consolidated) is not bool:
        _fail("use_consolidated must be one exact boolean.")
    archive = _canonical_archive(analysis_zarr)
    name = _run_name(run_name)
    run_path = _canonical_run_path(name)
    try:
        metadata_equivalence = validate_direct_consolidated_subtree(
            archive, subtree_path=run_path
        ).to_json()
        root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
        run = root[run_path]
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        _fail(f"Unable to open exact provider chaser-distance run: {exc}")
    if not isinstance(run, zarr.Group):
        _fail("Exact provider chaser-distance path is not a Zarr group.")
    manifest, _manifest_digest = _manifest_from_run(run)
    if expected_recording_id is not None and manifest.get("recording_id") != expected_recording_id:
        _fail("Persistent run recording_id differs from the requested recording.")
    validation = _validate_persistent_run(
        archive / run_path,
        expected_manifest=manifest,
        expected_run_path=run_path,
        verify_content_hashes=deep_audit,
        run=run,
    )
    arrays = validation["arrays"]
    return ProviderChaserDistanceSourceHandle(
        analysis_zarr=archive,
        run_path=run_path,
        run_name=name,
        recording_id=str(manifest["recording_id"]),
        manifest=manifest,
        arrays=arrays,
        metadata_equivalence=metadata_equivalence,
        verification_mode=("deep_audit" if deep_audit else "bounded_publication"),
        _use_consolidated=use_consolidated,
        _seal=_SOURCE_HANDLE_SEAL,
    )


def deep_audit_provider_chaser_distance_run(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
) -> ProviderChaserDistanceSourceHandle:
    """Explicit maintenance path that recomputes every declared output hash."""

    return load_provider_chaser_distance_source_handle(
        analysis_zarr,
        run_name=run_name,
        expected_recording_id=expected_recording_id,
        use_consolidated=True,
        deep_audit=True,
    )


__all__ = [
    "PROVIDER_CHASER_DISTANCE_DEEP_AUDIT_POLICY",
    "PROVIDER_CHASER_DISTANCE_MANIFEST_ATTR",
    "PROVIDER_CHASER_DISTANCE_MANIFEST_DIGEST_ATTR",
    "PROVIDER_CHASER_DISTANCE_PUBLICATION_POLICY",
    "PROVIDER_CHASER_DISTANCE_PUBLICATION_SCHEMA_ID",
    "PROVIDER_CHASER_DISTANCE_PUBLICATION_SCHEMA_VERSION",
    "PROVIDER_CHASER_DISTANCE_RUNS_PARENT_PATH",
    "PROVIDER_CHASER_DISTANCE_RUNS_PREFIX",
    "ProviderChaserDistancePublicationError",
    "ProviderChaserDistancePublicationPlan",
    "ProviderChaserDistanceSourceHandle",
    "build_provider_chaser_distance_publication_plan",
    "deep_audit_provider_chaser_distance_run",
    "load_provider_chaser_distance_source_handle",
    "publish_provider_chaser_distance_run",
]
