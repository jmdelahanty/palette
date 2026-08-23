"""Strict read-only handle for one immutable input-provenance proxy run."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis_workflows.materializers.chaser_input_provenance_proxy import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
)
from fisheye.shared.run_provenance import validate_run_provenance
from fisheye.shared.zarr.chaser_input_provenance_proxy_schema import (
    CHASER_INPUT_PROVENANCE_PROXY_LAYOUT,
    CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH,
    CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID,
    CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION,
    ChaserInputProvenanceProxyDimensions,
    validate_publication_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

SOURCE_HANDLE_SCHEMA_ID = "palette.chaser_input_provenance_proxy_source_handle"
SOURCE_HANDLE_SCHEMA_VERSION = 1
PUBLICATION_BINDING_SCHEMA_ID = (
    "palette.chaser_input_provenance_proxy_publication_binding"
)
PUBLICATION_BINDING_SCHEMA_VERSION = 1

_HANDLE_SEAL = object()
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SELECTOR_NAMES = frozenset(
    {
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
        "newest",
        "fallback",
    }
)


class ChaserInputProvenanceProxySourceHandleError(ValueError):
    """Raised when an exact immutable proxy run cannot be sealed."""


def _fail(message: str) -> None:
    raise ChaserInputProvenanceProxySourceHandleError(message)


def _run_name(value: object) -> str:
    if type(value) is not str or _RUN_NAME_RE.fullmatch(value) is None:
        _fail("run_name must be one bare exact run name.")
    if value.lower() in _SELECTOR_NAMES:
        _fail("run_name must not be a selector or fallback name.")
    return value


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return copy.deepcopy(value)


def _readonly(value: object, *, field: str) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    if array.dtype.hasobject or array.dtype.kind in {"U", "S"}:
        _fail(f"{field} is object/string typed.")
    array.setflags(write=False)
    return array


def _array_paths(run: Any) -> set[str]:
    group = run.get("arrays")
    if not isinstance(group, zarr.Group):
        _fail("Proxy run lacks its arrays group.")
    if set(str(name) for name in group.group_keys()):
        _fail("Proxy arrays group contains nested groups.")
    return set(str(name) for name in group.array_keys())


@dataclass(frozen=True, init=False, eq=False)
class ChaserInputProvenanceProxySourceHandle:
    """Verified immutable snapshot of one exact named proxy candidate."""

    analysis_zarr_path: Path
    run_name: str
    run_path: str
    recording_id: str
    dimensions: ChaserInputProvenanceProxyDimensions
    manifest: Mapping[str, Any] = field(repr=False)
    manifest_sha256: str
    acquisition_projection_record: Mapping[str, Any] = field(repr=False)
    acquisition_projection_record_sha256: str
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    provenance: Mapping[str, Any] = field(repr=False)
    metadata_equivalence: Mapping[str, Any] = field(repr=False)
    verification_digest: str
    selector_eligible: bool
    _use_consolidated: bool = field(repr=False, compare=False)
    _expected_recording_id: str | None = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(
        self, *, _verification_seal: object | None = None, **values: Any
    ) -> None:
        if _verification_seal is not _HANDLE_SEAL:
            _fail("Proxy source handles can only be minted by their strict loader.")
        for name, value in values.items():
            if name in {
                "manifest",
                "acquisition_projection_record",
                "provenance",
                "metadata_equivalence",
            }:
                value = _freeze(value)
            elif name == "arrays":
                value = MappingProxyType(
                    {
                        key: _readonly(child, field=f"arrays.{key}")
                        for key, child in value.items()
                    }
                )
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_verification_seal", _HANDLE_SEAL)

    @property
    def acquisition_frame_index(self) -> np.ndarray:
        return self.arrays["acquisition_frame_index"]

    @property
    def candidate_offsets(self) -> np.ndarray:
        return self.arrays["candidate_offsets"]

    @property
    def candidate_sample_count(self) -> np.ndarray:
        return self.arrays["candidate_sample_count"]

    @property
    def selected(self) -> np.ndarray:
        return self.arrays["selected"]

    @property
    def selected_native_sample_row_index(self) -> np.ndarray:
        return self.arrays["selected_native_sample_row_index"]

    @property
    def selected_source_stimulus_run_row_index(self) -> np.ndarray:
        return self.arrays["selected_source_stimulus_run_row_index"]

    @property
    def selected_chaser_position_xy(self) -> np.ndarray:
        return self.arrays["selected_chaser_position_xy"]

    @property
    def selected_chaser_valid(self) -> np.ndarray:
        return self.arrays["selected_chaser_valid"]

    @property
    def publication_binding_record(self) -> dict[str, Any]:
        projection = self.acquisition_projection_record
        return {
            "schema_id": PUBLICATION_BINDING_SCHEMA_ID,
            "schema_version": PUBLICATION_BINDING_SCHEMA_VERSION,
            "recording_id": self.recording_id,
            "run_path": self.run_path,
            "manifest_sha256": self.manifest_sha256,
            "acquisition_projection_record_sha256": (
                self.acquisition_projection_record_sha256
            ),
            "policy_id": projection["policy_id"],
            "temporal_alignment_class": projection["temporal_alignment_class"],
            "source_run_path": projection["source_run_path"],
            "source_manifest_sha256": projection["source_manifest_sha256"],
            "source_verification_digest": projection["source_verification_digest"],
            "n_frames": self.dimensions.n_frames,
            "n_candidates": self.dimensions.n_candidates,
            "n_chasers": self.dimensions.n_chasers,
            "selector_eligible": False,
            "selection": "none",
        }

    def array(self, name: str) -> np.ndarray:
        if type(name) is not str or name not in self.arrays:
            raise KeyError(f"Unknown proxy array {name!r}.")
        return self.arrays[name]

    def assert_current(self) -> None:
        refreshed = load_chaser_input_provenance_proxy_source_handle(
            self.analysis_zarr_path,
            run_name=self.run_name,
            expected_recording_id=self._expected_recording_id,
            use_consolidated=self._use_consolidated,
            expected_manifest_sha256=self.manifest_sha256,
        )
        if refreshed.verification_digest != self.verification_digest:
            _fail("Proxy candidate changed after the source handle was sealed.")

    def assert_verified(self) -> None:
        self.assert_current()


def load_chaser_input_provenance_proxy_source_handle(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
    use_consolidated: bool = True,
    expected_manifest_sha256: str | None = None,
) -> ChaserInputProvenanceProxySourceHandle:
    """Load one exact run; selectors, fallback names, and stale metadata fail."""

    archive = Path(analysis_zarr).expanduser().resolve()
    name = _run_name(run_name)
    run_path = f"{CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH}/{name}"
    if expected_manifest_sha256 is not None:
        expected_manifest_sha256 = _digest(
            expected_manifest_sha256,
            field="expected_manifest_sha256",
        )
    try:
        equivalence = validate_direct_consolidated_subtree(
            archive,
            subtree_path=run_path,
        )
        root = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=use_consolidated,
        )
        parent = root[CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH]
        run = root[run_path]
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        _fail(f"Unable to verify exact proxy run {run_path!r}: {exc}.")
    if not isinstance(parent, zarr.Group) or not isinstance(run, zarr.Group):
        _fail("Proxy parent or exact run path is not a Zarr group.")
    if _SELECTOR_NAMES.intersection(parent.attrs):
        _fail("Proxy candidate parent contains a selector or fallback attribute.")
    attrs = dict(run.attrs)
    if (
        attrs.get("schema_id") != CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID
        or attrs.get("schema_version") != CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION
        or attrs.get("layout") != CHASER_INPUT_PROVENANCE_PROXY_LAYOUT
    ):
        _fail("Proxy run schema identity is invalid.")
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("selector_eligible") is not False
        or attrs.get("selection") != "none"
    ):
        _fail("Proxy run is incomplete, selected, or selector-eligible.")
    manifest = attrs.get(MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        _fail("Proxy run manifest is missing.")
    manifest = dict(manifest)
    manifest_sha256 = _digest(
        attrs.get(MANIFEST_DIGEST_ATTR),
        field="manifest_sha256",
    )
    if canonical_json_sha256(manifest) != manifest_sha256:
        _fail("Proxy run manifest digest is stale.")
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        _fail("Proxy run manifest differs from the expected digest.")
    publication_manifest = {
        key: value for key, value in manifest.items() if key != "prepared_candidate"
    }
    try:
        normalized_manifest = validate_publication_manifest(publication_manifest)
    except (TypeError, ValueError) as exc:
        _fail(f"Proxy manifest validation failed: {exc}.")
    declaration_rows = publication_manifest.get("array_declarations")
    if not isinstance(declaration_rows, list):
        _fail("Proxy manifest array declarations are missing.")
    declared_names = {
        row.get("path") for row in declaration_rows if isinstance(row, Mapping)
    }
    if declared_names != _array_paths(run):
        _fail("Proxy array paths differ from the exact manifest declarations.")
    arrays = {
        str(name): _readonly(run[f"arrays/{name}"][...], field=f"arrays.{name}")
        for name in sorted(declared_names)
    }
    try:
        normalized_manifest = validate_publication_manifest(
            publication_manifest,
            arrays,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Proxy array content validation failed: {exc}.")
    recording_id = normalized_manifest["source"]["recording_id"]
    if root.attrs.get("recording_id") != recording_id:
        _fail("Proxy recording_id differs from the analysis archive root.")
    if expected_recording_id is not None and recording_id != expected_recording_id:
        _fail("Proxy recording_id differs from the caller expectation.")
    provenance = attrs.get("run_provenance")
    provenance_validation = validate_run_provenance(provenance)
    if not provenance_validation.valid:
        _fail(f"Proxy run provenance is invalid: {provenance_validation.errors}.")
    dimensions_record = normalized_manifest["dimensions"]
    dimensions = ChaserInputProvenanceProxyDimensions(
        n_frames=int(dimensions_record["frame"]),
        n_candidates=int(dimensions_record["candidate"]),
        n_chasers=int(dimensions_record["chaser"]),
    )
    verified_array_digests = {
        str(declaration["path"]): str(declaration["content_sha256"])
        for declaration in normalized_manifest["array_declarations"]
    }
    verification = {
        "schema_id": SOURCE_HANDLE_SCHEMA_ID,
        "schema_version": SOURCE_HANDLE_SCHEMA_VERSION,
        "run_path": run_path,
        "recording_id": recording_id,
        "manifest_sha256": manifest_sha256,
        "arrays": {
            name: verified_array_digests[name]
            for name in sorted(verified_array_digests)
        },
        "metadata_equivalence": equivalence.to_json(),
        "selector_eligible": False,
        "selection": "none",
    }
    return ChaserInputProvenanceProxySourceHandle(
        analysis_zarr_path=archive,
        run_name=name,
        run_path=run_path,
        recording_id=recording_id,
        dimensions=dimensions,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        acquisition_projection_record=normalized_manifest[
            "acquisition_projection_record"
        ],
        acquisition_projection_record_sha256=normalized_manifest[
            "acquisition_projection_record_sha256"
        ],
        arrays=arrays,
        provenance=provenance,
        metadata_equivalence=equivalence.to_json(),
        verification_digest=canonical_json_sha256(verification),
        selector_eligible=False,
        _use_consolidated=use_consolidated,
        _expected_recording_id=expected_recording_id,
        _verification_seal=_HANDLE_SEAL,
    )


__all__ = [
    "SOURCE_HANDLE_SCHEMA_ID",
    "SOURCE_HANDLE_SCHEMA_VERSION",
    "PUBLICATION_BINDING_SCHEMA_ID",
    "PUBLICATION_BINDING_SCHEMA_VERSION",
    "ChaserInputProvenanceProxySourceHandle",
    "ChaserInputProvenanceProxySourceHandleError",
    "load_chaser_input_provenance_proxy_source_handle",
]
