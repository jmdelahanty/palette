"""Shared immutable publication boundary for composable chaser successors.

Controller trials, generalized bout response, gaze, escape/freeze, and the
full-profile envelope share the same lifecycle contract: one exact run name,
typed array payloads, bounded digest-bound metadata, atomic publication,
selector ineligibility, and strict direct/consolidated readback.  Keeping that
mechanism here avoids five subtly different selector or retry policies.
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

from fisheye.analysis_workflows.controller_trial_successor import (
    PreparedControllerTrials,
)
from fisheye.analysis_workflows.chaser_radial_near_field_successor import (
    PreparedChaserRadialNearField,
)
from fisheye.analysis_workflows.chaser_spatial_occupancy_successor import (
    PreparedChaserSpatialOccupancy,
)
from fisheye.analysis_workflows.escape_freeze_successor import (
    PreparedEscapeFreeze,
)
from fisheye.analysis_workflows.full_chaser_profile_successor import (
    PreparedFullChaserProfile,
)
from fisheye.analysis_workflows.gaze_tracking_successor import (
    PreparedGazeTracking,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    PreparedGeneralizedBoutResponse,
)
from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    VERIFICATION_MODE as EXACT_CHILD_VERIFICATION_MODE,
    read_exact_immutable_child_validation_receipt,
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


PUBLICATION_SCHEMA_ID = "palette.analysis.composable_chaser_successor.publication"
PUBLICATION_SCHEMA_VERSION = 1
STORAGE_SCHEMA_ID = "palette.analysis.composable_chaser_successor.run"
STORAGE_SCHEMA_VERSION = 1
PUBLICATION_POLICY = "immutable_typed_arrays_selector_ineligible_v1"
MANIFEST_ATTR = "composable_chaser_successor_manifest"
MANIFEST_DIGEST_ATTR = "composable_chaser_successor_manifest_sha256"
MAX_MANIFEST_BYTES = 262_144

_RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SELECTORS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "selected",
        "authoritative",
        "default",
    }
)
_HANDLE_SEAL = object()

_TYPE_INFO: Mapping[type[Any], tuple[str, str]] = MappingProxyType(
    {
        PreparedControllerTrials: (
            "controller_chase_trials",
            "analysis/controller_chase_trial_runs",
        ),
        PreparedGeneralizedBoutResponse: (
            "generalized_chaser_bout_response",
            "analysis/generalized_chaser_bout_response_runs",
        ),
        PreparedEscapeFreeze: (
            "chaser_escape_freeze",
            "analysis/chaser_escape_freeze_runs",
        ),
        PreparedGazeTracking: (
            "chaser_gaze_tracking",
            "analysis/chaser_gaze_tracking_runs",
        ),
        PreparedFullChaserProfile: (
            "chaser_full_profile",
            "analysis/chaser_full_profile_runs",
        ),
        PreparedChaserRadialNearField: (
            "chaser_radial_near_field",
            "analysis/chaser_radial_near_field_runs",
        ),
        PreparedChaserSpatialOccupancy: (
            "chaser_spatial_occupancy",
            "analysis/chaser_spatial_occupancy_runs",
        ),
    }
)


class ComposableChaserSuccessorPublicationError(ValueError):
    """Raised when a successor publication or readback is invalid."""


def _fail(message: str) -> None:
    raise ComposableChaserSuccessorPublicationError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _strict_json_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be one JSON object.")
    try:
        decoded = json.loads(
            json.dumps(
                _plain(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ComposableChaserSuccessorPublicationError(
            f"{name} is not strict JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):  # pragma: no cover
        _fail(f"{name} must decode to one object.")
    return decoded


def _run_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value.lower() in _SELECTORS
        or value in {".", ".."}
        or _RUN_RE.fullmatch(value) is None
    ):
        _fail("run_name must be one exact non-selector child name.")
    return value


def _archive(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {path}")
    return path


def _digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _prepared_info(prepared: object) -> tuple[str, str, str, Mapping[str, np.ndarray], dict[str, Any]]:
    info = _TYPE_INFO.get(type(prepared))
    if info is None:
        raise TypeError("prepared is not a supported composable chaser successor.")
    kind, parent_path = info
    recording_id = getattr(prepared, "recording_id", None)
    if type(recording_id) is not str or not recording_id.strip():
        _fail("Prepared successor lacks one exact recording ID.")
    arrays = getattr(prepared, "arrays", None)
    if not isinstance(arrays, Mapping) or any(
        type(name) is not str or not isinstance(value, np.ndarray)
        for name, value in arrays.items()
    ):
        _fail("Prepared successor lacks its typed array mapping.")
    scientific = _strict_json_object(
        getattr(prepared, "manifest", None),
        name="prepared scientific manifest",
    )
    scientific_payload = scientific.get("payload_digest")
    body = dict(scientific)
    body.pop("payload_digest", None)
    if scientific_payload != canonical_json_sha256(body):
        _fail("Prepared scientific manifest payload digest is stale.")
    if (
        scientific.get("selector_eligible") is not False
        or scientific.get("production_authority") is not False
        or scientific.get("registry_update") is not False
    ):
        _fail("Prepared successor is not selector-ineligible and non-authoritative.")
    return kind, parent_path, recording_id, arrays, scientific


@dataclass(frozen=True, slots=True)
class ComposableChaserSuccessorPublicationPlan:
    analysis_zarr: Path
    successor_kind: str
    parent_path: str
    run_name: str
    run_path: str
    recording_id: str
    prepared: Any = field(repr=False, compare=False)
    manifest: Mapping[str, Any] = field(repr=False)
    run_provenance: Mapping[str, Any] = field(repr=False)


def build_composable_chaser_successor_publication_plan(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    prepared: object,
) -> ComposableChaserSuccessorPublicationPlan:
    """Bind one prepared successor to one exact immutable run path."""

    archive = _archive(analysis_zarr)
    name = _run_name(run_name)
    kind, parent_path, recording_id, arrays, scientific = _prepared_info(prepared)
    run_path = f"{parent_path}/{name}"
    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    root_recording = root.attrs.get("recording_id")
    if root_recording is not None and root_recording != recording_id:
        _fail("Prepared successor belongs to another analysis archive recording.")
    try:
        root[run_path]
    except KeyError:
        pass
    else:
        raise FileExistsError(f"Successor target already exists: {run_path}")
    declarations = [
        {
            "path": array_name,
            "dtype": np.asarray(values).dtype.str,
            "shape": list(np.asarray(values).shape),
            "content_sha256": array_values_sha256(np.asarray(values)),
        }
        for array_name, values in sorted(arrays.items())
    ]
    publication_body = {
        "schema_id": STORAGE_SCHEMA_ID,
        "schema_version": STORAGE_SCHEMA_VERSION,
        "successor_kind": kind,
        "run_name": name,
        "run_path": run_path,
        "recording_id": recording_id,
        "scientific_manifest": scientific,
        "scientific_payload_sha256": scientific["payload_digest"],
        "array_declarations": declarations,
        "publication_policy": PUBLICATION_POLICY,
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "production_selector_activation": False,
        "registry_update": False,
    }
    manifest = {
        **publication_body,
        "payload_digest": canonical_json_sha256(publication_body),
    }
    encoded = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_MANIFEST_BYTES:
        _fail("Composable successor publication manifest exceeds its metadata bound.")
    provenance = build_writer_run_provenance(
        command="fisheye.analysis_workflows.composable_chaser_successor_publication",
        params={
            "successor_kind": kind,
            "run_name": name,
            "run_path": run_path,
            "publication_policy": PUBLICATION_POLICY,
            "scientific_payload_sha256": scientific["payload_digest"],
        },
        input_run_ids={
            "scientific_successor_payload": scientific["payload_digest"],
        },
        cwd=Path(__file__).resolve().parents[3],
    )
    return ComposableChaserSuccessorPublicationPlan(
        analysis_zarr=archive,
        successor_kind=kind,
        parent_path=parent_path,
        run_name=name,
        run_path=run_path,
        recording_id=recording_id,
        prepared=prepared,
        manifest=MappingProxyType(manifest),
        run_provenance=MappingProxyType(provenance),
    )


def _manifest_from_run(run: Any) -> dict[str, Any]:
    manifest = _strict_json_object(run.attrs.get(MANIFEST_ATTR), name=MANIFEST_ATTR)
    digest = _digest(run.attrs.get(MANIFEST_DIGEST_ATTR), name=MANIFEST_DIGEST_ATTR)
    if canonical_json_sha256(manifest) != digest:
        _fail("Persistent successor manifest digest is stale.")
    body = dict(manifest)
    payload = body.pop("payload_digest", None)
    if payload != canonical_json_sha256(body):
        _fail("Persistent successor manifest payload digest is stale.")
    return manifest


def _validate_persistent_run(
    path: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_run_path: str | None = None,
    verify_content_hashes: bool = True,
    run: Any | None = None,
) -> dict[str, Any]:
    group = (
        run
        if run is not None
        else zarr.open_group(
            str(path), mode="r", zarr_format=3, use_consolidated=False
        )
    )
    manifest = _manifest_from_run(group)
    if expected_manifest is not None and manifest != _plain(expected_manifest):
        _fail("Persistent successor manifest differs from the publication plan.")
    if expected_run_path is not None and manifest.get("run_path") != expected_run_path:
        _fail("Persistent successor run path differs from expectation.")
    if (
        group.attrs.get("stage_selector_eligible") is not False
        or group.attrs.get("production_authority") is not False
        or group.attrs.get("registry_update") is not False
        or group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or manifest.get("selector_eligible") is not False
        or manifest.get("production_authority") is not False
    ):
        _fail("Persistent successor is not complete and selector-ineligible.")
    if set(str(value) for value in group.group_keys()):
        _fail("Persistent successor contains unexpected nested groups.")
    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list):
        _fail("Persistent successor lacks array declarations.")
    by_path = {
        declaration.get("path"): declaration
        for declaration in declarations
        if isinstance(declaration, Mapping)
    }
    if len(by_path) != len(declarations) or set(group.array_keys()) != set(by_path):
        _fail("Persistent successor array inventory is missing, duplicated, or extra.")
    arrays: dict[str, np.ndarray] = {}
    for name, declaration in sorted(by_path.items()):
        if type(name) is not str or not name or "/" in name:
            _fail("Persistent successor array path is invalid.")
        value = np.asarray(group[name][:])
        if (
            value.dtype.str != declaration.get("dtype")
            or list(value.shape) != declaration.get("shape")
        ):
            _fail(f"Persistent successor array {name!r} metadata is stale.")
        if verify_content_hashes and array_values_sha256(value) != declaration.get(
            "content_sha256"
        ):
            _fail(f"Persistent successor array {name!r} content digest is stale.")
        copied = np.array(value, copy=True, order="C")
        copied.setflags(write=False)
        arrays[name] = copied
    scientific = _strict_json_object(
        manifest.get("scientific_manifest"), name="scientific_manifest"
    )
    if scientific.get("payload_digest") != manifest.get("scientific_payload_sha256"):
        _fail("Persistent scientific payload binding is stale.")
    return {
        "valid": True,
        "successor_kind": manifest["successor_kind"],
        "run_name": manifest["run_name"],
        "run_path": manifest["run_path"],
        "recording_id": manifest["recording_id"],
        "manifest": manifest,
        "manifest_sha256": canonical_json_sha256(manifest),
        "arrays": arrays,
    }


def _write_local_run(
    plan: ComposableChaserSuccessorPublicationPlan,
    local_path: Path,
) -> None:
    if local_path.exists():
        raise FileExistsError(f"Local successor path exists: {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run = zarr.open_group(
        str(local_path), mode="w-", zarr_format=3, use_consolidated=False
    )
    mark_run_started(run, run_name=plan.run_name, stage=plan.successor_kind)
    run.attrs.update(
        {
            "schema_id": STORAGE_SCHEMA_ID,
            "schema_version": STORAGE_SCHEMA_VERSION,
            "successor_kind": plan.successor_kind,
            "recording_id": plan.recording_id,
            "stage_selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "registry_update": False,
            "run_provenance": json_attr_safe(dict(plan.run_provenance)),
            MANIFEST_ATTR: json_attr_safe(dict(plan.manifest)),
            MANIFEST_DIGEST_ATTR: canonical_json_sha256(dict(plan.manifest)),
        }
    )
    for name, values in plan.prepared.arrays.items():
        value = np.asarray(values)
        chunks = (max(1, min(int(value.shape[0]), 16_384)), *value.shape[1:])
        run.create_array(name, data=value, chunks=chunks)
    mark_run_complete(
        run,
        run_name=plan.run_name,
        run_provenance=dict(plan.run_provenance),
    )
    _validate_persistent_run(
        local_path,
        expected_manifest=plan.manifest,
        expected_run_path=plan.run_path,
    )


def publish_composable_chaser_successor_run(
    plan: ComposableChaserSuccessorPublicationPlan,
    *,
    scratch_root: str | Path | None = None,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Atomically publish and consolidate one selector-ineligible successor."""

    if type(plan) is not ComposableChaserSuccessorPublicationPlan:
        raise TypeError("plan must be one composable successor publication plan.")
    scratch = (
        Path(scratch_root).expanduser().resolve() if scratch_root is not None else None
    )
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{plan.run_name}.{plan.successor_kind}.",
        dir=str(scratch) if scratch is not None else None,
    ) as temporary:
        local_path = Path(temporary) / "run.zarr"
        _write_local_run(plan, local_path)
        parent_snapshot: dict[str, Any] | None = None

        def validate(path: Path) -> Mapping[str, Any]:
            result = _validate_persistent_run(
                path,
                expected_manifest=plan.manifest,
                expected_run_path=plan.run_path,
            )
            return {key: value for key, value in result.items() if key not in {"arrays", "manifest"}}

        def prepare(root: Any) -> tuple[Any]:
            nonlocal parent_snapshot
            analysis = root.require_group("analysis")
            parent = require_runs_parent(
                analysis,
                plan.parent_path.split("/", 1)[1],
                completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
            )
            if set(parent.attrs).intersection(_SELECTORS):
                _fail("Successor parent contains a forbidden selector.")
            parent_snapshot = dict(parent.attrs)
            return (parent,)

        def complete(_root: Any, parent: Any, run: Any) -> None:
            run.attrs["stage_selector_eligible"] = False
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=plan.run_name,
                run_provenance=dict(plan.run_provenance),
            )

        def verify(root: Any) -> None:
            parent = root[plan.parent_path]
            if parent_snapshot is None or dict(parent.attrs) != parent_snapshot:
                _fail("Successor publication changed parent metadata.")
            _validate_persistent_run(
                plan.analysis_zarr / plan.run_path,
                expected_manifest=plan.manifest,
                expected_run_path=plan.run_path,
            )

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.analysis_zarr,
                local_run_path=local_path,
                target_run_path=plan.analysis_zarr / plan.run_path,
                run_name=plan.run_name,
                lock_suffix=f"{plan.successor_kind}-publication",
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
                "successor_kind": plan.successor_kind,
                "scientific_payload_sha256": plan.manifest[
                    "scientific_payload_sha256"
                ],
                "selector_activation": "none",
            },
        )
    consolidation = consolidate_metadata_capture_expected_warnings(plan.analysis_zarr)
    metadata = validate_direct_consolidated_subtree(
        plan.analysis_zarr, subtree_path=plan.run_path
    ).to_json()
    handle = load_composable_chaser_successor_source_handle(
        plan.analysis_zarr,
        successor_kind=plan.successor_kind,
        run_name=plan.run_name,
        expected_recording_id=plan.recording_id,
    )
    return {
        "status": "published_selector_ineligible",
        "successor_kind": handle.successor_kind,
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "scientific_payload_sha256": handle.scientific_payload_sha256,
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "consolidation": consolidation,
        "metadata_equivalence": metadata,
        "atomic_publication": publication,
    }


@dataclass(frozen=True, slots=True, init=False)
class ComposableChaserSuccessorSourceHandle:
    analysis_zarr: Path
    successor_kind: str
    parent_path: str
    run_name: str
    run_path: str
    recording_id: str
    manifest: Mapping[str, Any] = field(repr=False)
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    metadata_equivalence: Mapping[str, Any] = field(repr=False)
    deep_audited: bool
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _seal: object | None = None, **values: Any) -> None:
        if _seal is not _HANDLE_SEAL:
            raise TypeError("Composable successor handles require their strict loader.")
        for name, value in values.items():
            if name in {"manifest", "metadata_equivalence"}:
                value = _freeze(_plain(value))
            elif name == "arrays":
                copied = {}
                for key, array in value.items():
                    item = np.array(array, copy=True, order="C")
                    item.setflags(write=False)
                    copied[key] = item
                value = MappingProxyType(copied)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _HANDLE_SEAL)

    @property
    def manifest_sha256(self) -> str:
        return canonical_json_sha256(_plain(self.manifest))

    @property
    def scientific_manifest(self) -> Mapping[str, Any]:
        return self.manifest["scientific_manifest"]

    @property
    def scientific_payload_sha256(self) -> str:
        return str(self.manifest["scientific_payload_sha256"])

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown published successor array {name!r}.") from exc

    def prepared_successor(self) -> Any:
        """Rehydrate the exact prepared dependency after a deep content audit."""

        if self.deep_audited is not True:
            _fail(
                "Prepared successor reuse requires a deep-audited source handle "
                "whose array contents were rehashed."
            )
        scientific = self.scientific_manifest
        dimensions = scientific.get("dimensions")
        if not isinstance(dimensions, Mapping):
            _fail("Published scientific manifest lacks exact dimensions.")
        values = {
            "recording_id": self.recording_id,
            "arrays": self.arrays,
            "manifest": scientific,
        }
        if self.successor_kind == "controller_chase_trials":
            return PreparedControllerTrials(
                n_frames=int(dimensions["n_frames"]),
                n_chasers=int(dimensions["n_chasers"]),
                n_trials=int(dimensions["n_trials"]),
                **values,
            )
        if self.successor_kind == "generalized_chaser_bout_response":
            return PreparedGeneralizedBoutResponse(
                n_bouts=int(dimensions["n_bouts"]),
                n_chasers=int(dimensions["n_chasers"]),
                n_bout_chaser_rows=int(dimensions["n_bout_chaser_rows"]),
                **values,
            )
        if self.successor_kind == "chaser_escape_freeze":
            return PreparedEscapeFreeze(
                n_trials=int(dimensions["n_trials"]),
                n_events=int(dimensions["n_events"]),
                **values,
            )
        if self.successor_kind == "chaser_gaze_tracking":
            return PreparedGazeTracking(
                n_gaze_rows=int(dimensions["n_gaze_rows"]),
                n_summary_rows=int(dimensions["n_summary_rows"]),
                n_lock_events=int(dimensions["n_lock_events"]),
                **values,
            )
        if self.successor_kind == "chaser_full_profile":
            profile = scientific.get("normalized_profile")
            if not isinstance(profile, Mapping):
                _fail("Published full-profile successor lacks its normalized profile.")
            return PreparedFullChaserProfile(
                profile_id=str(profile["profile_id"]),
                profile_version=int(profile["profile_version"]),
                readiness=str(scientific["readiness"]),
                full_profile_complete=bool(scientific["full_profile_complete"]),
                **values,
            )
        if self.successor_kind == "chaser_radial_near_field":
            return PreparedChaserRadialNearField(
                n_epoch_chaser_rows=int(dimensions["n_epoch_chaser_rows"]),
                n_radial_rows=int(dimensions["n_radial_rows"]),
                n_cdf_rows=int(dimensions["n_cdf_rows"]),
                **values,
            )
        if self.successor_kind == "chaser_spatial_occupancy":
            return PreparedChaserSpatialOccupancy(
                n_providers=int(dimensions["n_providers"]),
                n_epochs=int(dimensions["n_epochs"]),
                grid_rows=int(dimensions["grid_rows"]),
                grid_columns=int(dimensions["grid_columns"]),
                **values,
            )
        _fail(f"Unsupported successor kind {self.successor_kind!r}.")

    def module_product_binding(self, *, module_id: str) -> Any:
        """Build a full-profile binding from this exact immutable product."""

        from fisheye.analysis_workflows.full_chaser_profile_successor import (
            ImmutableModuleProductBinding,
        )

        if self.deep_audited is not True:
            _fail(
                "Full-profile product bindings require a deep-audited source "
                "handle whose array contents were rehashed."
            )
        scientific = self.scientific_manifest.get("scientific_schema")
        if not isinstance(scientific, Mapping):
            scientific = self.scientific_manifest
        schema_id = scientific.get("schema_id")
        schema_version = scientific.get("schema_version")
        return ImmutableModuleProductBinding(
            module_id=module_id,
            schema_id=schema_id,
            schema_version=schema_version,
            run_path=self.run_path,
            manifest_sha256=self.manifest_sha256,
            payload_sha256=self.scientific_payload_sha256,
        )

    def assert_current(self) -> None:
        receipt_path = self.metadata_equivalence.get("receipt_path")
        refreshed = load_composable_chaser_successor_source_handle(
            self.analysis_zarr,
            successor_kind=self.successor_kind,
            run_name=self.run_name,
            expected_recording_id=self.recording_id,
            deep_audit=self.deep_audited,
            direct_validation_receipt=(
                str(receipt_path) if receipt_path is not None else None
            ),
        )
        if refreshed.manifest_sha256 != self.manifest_sha256:
            _fail("Published successor changed after handle creation.")


def load_composable_chaser_successor_source_handle(
    analysis_zarr: str | Path,
    *,
    successor_kind: str,
    run_name: str,
    expected_recording_id: str | None = None,
    use_consolidated: bool = True,
    deep_audit: bool = False,
    direct_validation_receipt: str | Path | None = None,
) -> ComposableChaserSuccessorSourceHandle:
    """Load one exact successor run without selector discovery."""

    parent_by_kind = {kind: parent for kind, parent in _TYPE_INFO.values()}
    if successor_kind not in parent_by_kind:
        _fail(f"Unknown composable successor kind {successor_kind!r}.")
    name = _run_name(run_name)
    archive = _archive(analysis_zarr)
    parent_path = parent_by_kind[successor_kind]
    run_path = f"{parent_path}/{name}"
    try:
        if direct_validation_receipt is None:
            metadata = validate_direct_consolidated_subtree(
                archive, subtree_path=run_path
            ).to_json()
            root = open_zarr_root(
                archive, mode="r", use_consolidated=use_consolidated
            )
            run = root[run_path]
        else:
            receipt_path = Path(direct_validation_receipt).expanduser().resolve()
            receipt = read_exact_immutable_child_validation_receipt(
                receipt_path,
                expected_analysis_zarr=archive,
                expected_run_path=run_path,
                expected_recording_id=expected_recording_id,
                expected_manifest_attr=MANIFEST_ATTR,
                expected_manifest_digest_attr=MANIFEST_DIGEST_ATTR,
            )
            metadata = {
                "verification_mode": EXACT_CHILD_VERIFICATION_MODE,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt["record_sha256"],
                "direct_metadata_inventory_sha256": receipt[
                    "direct_metadata_inventory"
                ]["inventory_sha256"],
                "archive_root_consolidated_metadata_reparse": False,
            }
            run = open_zarr_root(
                archive / run_path, mode="r", use_consolidated=False
            )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ComposableChaserSuccessorPublicationError(
            f"Unable to open exact successor run: {exc}"
        ) from exc
    validation = _validate_persistent_run(
        archive / run_path,
        expected_run_path=run_path,
        verify_content_hashes=deep_audit,
        run=run,
    )
    if validation["successor_kind"] != successor_kind:
        _fail("Published successor kind differs from the requested parent.")
    if (
        expected_recording_id is not None
        and validation["recording_id"] != expected_recording_id
    ):
        _fail("Published successor belongs to another recording.")
    return ComposableChaserSuccessorSourceHandle(
        analysis_zarr=archive,
        successor_kind=successor_kind,
        parent_path=parent_path,
        run_name=name,
        run_path=run_path,
        recording_id=validation["recording_id"],
        manifest=validation["manifest"],
        arrays=validation["arrays"],
        metadata_equivalence=metadata,
        deep_audited=deep_audit,
        _seal=_HANDLE_SEAL,
    )


__all__ = [
    "MANIFEST_ATTR",
    "MANIFEST_DIGEST_ATTR",
    "PUBLICATION_POLICY",
    "PUBLICATION_SCHEMA_ID",
    "PUBLICATION_SCHEMA_VERSION",
    "ComposableChaserSuccessorPublicationError",
    "ComposableChaserSuccessorPublicationPlan",
    "ComposableChaserSuccessorSourceHandle",
    "build_composable_chaser_successor_publication_plan",
    "load_composable_chaser_successor_source_handle",
    "publish_composable_chaser_successor_run",
]
