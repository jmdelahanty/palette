"""Sealed, exact input handle for one published body-frame run.

This module is deliberately a consumer-side boundary.  It names one run
directly, validates the immutable publication in place, and returns read-only
array snapshots.  It never resolves ``latest`` and it never changes a Zarr
selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import copy

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_manifest import (
    BODY_FRAME_RUN_MANIFEST_ATTRIBUTE,
    body_frame_recipe_from_manifest,
    body_frame_source_from_manifest,
    validate_body_frame_run_manifest,
)
from fisheye.shared.zarr.body_frame_schema import (
    BODY_FRAME_SCHEMA_V1,
    BodyFrameDimensions,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

BODY_FRAME_SOURCE_HANDLE_SCHEMA_ID = "palette.body_frame.source_handle"
BODY_FRAME_SOURCE_HANDLE_SCHEMA_VERSION = 1
BODY_FRAME_RUN_PREFIX = "analysis/body_frame_runs/"

_ARRAY_NAMES = BODY_FRAME_SCHEMA_V1.binding_paths
_REQUIRED_RUN_ATTRS = (
    "logical_schema",
    "source_keypoint_snapshot",
    "heading_recipe",
)
_BODY_FRAME_SOURCE_HANDLE_SEAL = object()


class BodyFrameSourceHandleError(ValueError):
    """Raised when an exact body-frame source cannot be safely bound."""


def _freeze_json(value: Any) -> Any:
    """Return a recursively immutable JSON-shaped value."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _array_snapshot(value: Any) -> np.ndarray:
    try:
        snapshot = np.array(value[...], copy=True)
    except (IndexError, KeyError, TypeError):
        snapshot = np.array(value, copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _require_exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise BodyFrameSourceHandleError(f"{name} must be the exact JSON boolean.")
    return value


def _require_run_path(value: object) -> str:
    if not isinstance(value, str):
        raise BodyFrameSourceHandleError("run_path must be a string.")
    if not value.startswith(BODY_FRAME_RUN_PREFIX):
        raise BodyFrameSourceHandleError(
            "run_path must name analysis/body_frame_runs/<run> exactly."
        )
    suffix = value[len(BODY_FRAME_RUN_PREFIX) :]
    if (
        not suffix
        or "/" in suffix
        or suffix in {".", ".."}
        or value.endswith("/")
        or "//" in value
    ):
        raise BodyFrameSourceHandleError(
            "run_path must name one explicit body-frame run, without a selector "
            "or fallback path."
        )
    return value


def _materialize_array_names(run: Any) -> set[str]:
    try:
        names = {str(name) for name in run.array_keys()}
    except (AttributeError, TypeError) as exc:
        raise BodyFrameSourceHandleError(
            "Body-frame run does not expose its exact array members."
        ) from exc
    try:
        groups = {str(name) for name in run.group_keys()}
    except (AttributeError, TypeError):
        groups = set()
    if groups:
        raise BodyFrameSourceHandleError(
            f"Body-frame run contains undeclared child groups: {sorted(groups)!r}."
        )
    return names


def _manifest_payload(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    errors = validate_body_frame_run_manifest(manifest)
    if errors:
        raise BodyFrameSourceHandleError(
            "Body-frame run_manifest validation failed: " + "; ".join(errors)
        )
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):  # pragma: no cover - validator guards it
        raise BodyFrameSourceHandleError("Body-frame run_manifest payload is absent.")
    return payload


def _dimensions_from_payload(payload: Mapping[str, Any]) -> BodyFrameDimensions:
    logical_schema = payload.get("logical_schema")
    if not isinstance(logical_schema, Mapping):
        raise BodyFrameSourceHandleError("Body-frame logical schema is absent.")
    dimensions = logical_schema.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise BodyFrameSourceHandleError("Body-frame dimensions are absent.")
    try:
        return BodyFrameDimensions(
            n_frames=dimensions["n_frames"],
            n_instances=dimensions["n_instances"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BodyFrameSourceHandleError(
            f"Body-frame dimensions are invalid: {exc}"
        ) from exc


def _validate_row_identity(
    arrays: Mapping[str, np.ndarray], dimensions: BodyFrameDimensions
) -> None:
    instance_key = arrays["instance_key"]
    source_rows = arrays["source_keypoint_row_ids"]
    source_signatures = arrays["source_keypoint_row_signature"]

    if np.unique(instance_key).size != dimensions.n_instances:
        raise BodyFrameSourceHandleError("Body-frame instance_key is duplicated.")
    if np.any(source_rows < 0):
        raise BodyFrameSourceHandleError(
            "Body-frame source_keypoint_row_ids contains a negative row."
        )
    expected_rows = np.arange(dimensions.n_instances, dtype=np.int64)
    if not np.array_equal(source_rows, expected_rows):
        raise BodyFrameSourceHandleError(
            "Body-frame source row identity is missing or reordered; the v1 "
            "source snapshot must be complete and row-for-row."
        )
    if np.unique(source_signatures, axis=0).shape[0] != dimensions.n_instances:
        raise BodyFrameSourceHandleError(
            "Body-frame source_keypoint_row_signature is duplicated."
        )


def _array_declarations(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping):
        raise BodyFrameSourceHandleError("Body-frame logical content is absent.")
    document = logical_content.get("document")
    if not isinstance(document, Mapping):
        raise BodyFrameSourceHandleError(
            "Body-frame logical content document is absent."
        )
    declarations = document.get("arrays")
    if not isinstance(declarations, Mapping):
        raise BodyFrameSourceHandleError(
            "Body-frame logical content array declarations are absent."
        )
    if set(declarations) != set(_ARRAY_NAMES):
        raise BodyFrameSourceHandleError(
            "Body-frame logical content does not declare the exact v1 arrays."
        )
    return declarations


@dataclass(frozen=True, init=False, eq=False)
class BodyFrameSourceHandle:
    """Immutable, verified binding to one exact body-frame source run."""

    analysis_zarr_path: Path
    run_path: str
    run_name: str
    selector_eligible: bool
    dimensions: BodyFrameDimensions
    run_manifest: Mapping[str, Any]
    source_snapshot: Mapping[str, Any]
    source_manifest_digest: str
    source_run_path: str
    source_skeleton_id: str
    source_skeleton_digest: str
    source_row_signatures_digest: str
    recipe_id: str
    recipe_digest: str
    recipe_skeleton_digest: str
    heading_computation_digest: str
    instance_key: np.ndarray
    frame_indices: np.ndarray
    frame_row_offsets: np.ndarray
    source_keypoint_row_ids: np.ndarray
    source_keypoint_row_signature: np.ndarray
    origin_xy: np.ndarray
    forward_axis_xy: np.ndarray
    left_axis_xy: np.ndarray
    axis_valid: np.ndarray
    heading_deg: np.ndarray
    verification_digest: str
    arrays: Mapping[str, np.ndarray]
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        analysis_zarr_path: Path,
        run_path: str,
        selector_eligible: bool,
        dimensions: BodyFrameDimensions,
        run_manifest: Mapping[str, Any],
        source_snapshot: Mapping[str, Any],
        source_manifest_digest: str,
        source_run_path: str,
        source_skeleton_id: str,
        source_skeleton_digest: str,
        source_row_signatures_digest: str,
        recipe_id: str,
        recipe_digest: str,
        recipe_skeleton_digest: str,
        heading_computation_digest: str,
        arrays: Mapping[str, np.ndarray],
        verification_digest: str,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BODY_FRAME_SOURCE_HANDLE_SEAL:
            raise TypeError("BodyFrameSourceHandle can only be created by its loader.")
        normalized_path = _require_run_path(run_path)
        snapshots = {name: _array_snapshot(arrays[name]) for name in _ARRAY_NAMES}
        frozen_arrays = MappingProxyType(snapshots)
        object.__setattr__(self, "analysis_zarr_path", Path(analysis_zarr_path))
        object.__setattr__(self, "run_path", normalized_path)
        object.__setattr__(self, "run_name", normalized_path.rsplit("/", 1)[1])
        object.__setattr__(self, "selector_eligible", selector_eligible)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "run_manifest", _freeze_json(run_manifest))
        object.__setattr__(self, "source_snapshot", _freeze_json(source_snapshot))
        for name, value in (
            ("source_manifest_digest", source_manifest_digest),
            ("source_run_path", source_run_path),
            ("source_skeleton_id", source_skeleton_id),
            ("source_skeleton_digest", source_skeleton_digest),
            ("source_row_signatures_digest", source_row_signatures_digest),
            ("recipe_id", recipe_id),
            ("recipe_digest", recipe_digest),
            ("recipe_skeleton_digest", recipe_skeleton_digest),
            ("heading_computation_digest", heading_computation_digest),
            ("verification_digest", verification_digest),
        ):
            object.__setattr__(self, name, str(value))
        for name in _ARRAY_NAMES:
            object.__setattr__(self, name, frozen_arrays[name])
        object.__setattr__(self, "arrays", frozen_arrays)
        object.__setattr__(self, "_verification_seal", _verification_seal)

    def array(self, name: str) -> np.ndarray:
        """Return one read-only array by its exact body-frame path."""

        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown body-frame array {name!r}.") from exc

    def assert_verified(self) -> None:
        """Reject objects that were not minted by the strict loader."""

        if self._verification_seal is not _BODY_FRAME_SOURCE_HANDLE_SEAL:
            raise BodyFrameSourceHandleError(
                "Body-frame source handle verification seal is absent."
            )


def load_body_frame_source_handle(
    analysis_zarr_path: Path | str,
    *,
    run_path: str,
    expected_selector_eligible: bool,
    use_consolidated: bool = True,
) -> BodyFrameSourceHandle:
    """Load and seal one exact body-frame run.

    ``run_path`` is intentionally required and is indexed directly.  No
    selector, ``latest`` attribute, or source-family discovery is consulted.
    ``use_consolidated`` defaults to the immutable-publication read mode; tests
    and active-development diagnostics may explicitly request ``False``.
    """

    if type(use_consolidated) is not bool:
        raise BodyFrameSourceHandleError(
            "use_consolidated must be the exact boolean metadata-read choice."
        )
    expected_eligible = _require_exact_bool(
        expected_selector_eligible, name="expected_selector_eligible"
    )
    exact_run_path = _require_run_path(run_path)
    archive_path = Path(analysis_zarr_path).expanduser().resolve()

    try:
        root = zarr.open_group(
            str(archive_path), mode="r", use_consolidated=use_consolidated
        )
        run = root[exact_run_path]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise BodyFrameSourceHandleError(
            f"Unable to open exact body-frame run {exact_run_path!r}: {exc}"
        ) from exc

    attrs = getattr(run, "attrs", {})
    if attrs.get("status") != RUN_STATUS_COMPLETE:
        raise BodyFrameSourceHandleError(
            f"Body-frame run {exact_run_path!r} is not complete."
        )
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR, RUN_COMPLETION_CONTRACT)
        != RUN_COMPLETION_CONTRACT
    ):
        raise BodyFrameSourceHandleError(
            "Body-frame run does not declare the completion contract."
        )
    completion_status = attrs.get(RUN_COMPLETION_STATUS_ATTR)
    if completion_status is not None and completion_status != RUN_STATUS_COMPLETE:
        raise BodyFrameSourceHandleError(
            "Body-frame run completion status is not complete."
        )
    actual_eligible = attrs.get("stage_selector_eligible")
    if actual_eligible is not expected_eligible:
        raise BodyFrameSourceHandleError(
            "Body-frame selector eligibility mismatch: expected "
            f"{expected_eligible!r}, found {actual_eligible!r}."
        )

    raw_manifest = attrs.get(BODY_FRAME_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(raw_manifest, Mapping):
        raise BodyFrameSourceHandleError("Body-frame run_manifest is missing.")
    manifest = copy.deepcopy(dict(raw_manifest))
    payload = _manifest_payload(manifest)
    if payload.get("run_id") != exact_run_path.rsplit("/", 1)[1]:
        raise BodyFrameSourceHandleError(
            "Body-frame run_manifest run_id does not match the explicit run path."
        )

    dimensions = _dimensions_from_payload(payload)
    expected_schema = payload["logical_schema"]
    if expected_schema != BODY_FRAME_SCHEMA_V1.as_manifest(dimensions=dimensions):
        raise BodyFrameSourceHandleError(
            "Body-frame run does not use the exact body-frame-v1 schema."
        )
    if attrs.get("logical_schema") != expected_schema:
        raise BodyFrameSourceHandleError(
            "Body-frame run logical_schema attribute differs from its manifest."
        )

    for attr_name in _REQUIRED_RUN_ATTRS[1:]:
        if attrs.get(attr_name) != payload[attr_name]:
            raise BodyFrameSourceHandleError(
                f"Body-frame run {attr_name!r} differs from its manifest."
            )

    array_names = _materialize_array_names(run)
    if array_names != set(_ARRAY_NAMES):
        raise BodyFrameSourceHandleError(
            "Body-frame run arrays do not match the exact v1 required set: "
            f"found {sorted(array_names)!r}."
        )
    arrays = {name: _array_snapshot(run[name]) for name in _ARRAY_NAMES}

    schema_issues = BODY_FRAME_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=None,
    )
    schema_issues = tuple(
        issue
        for issue in schema_issues
        if issue.code != "missing_source_keypoint_evidence"
    )
    if schema_issues:
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in schema_issues
        )
        raise BodyFrameSourceHandleError(
            f"Body-frame v1 array validation failed: {detail}"
        )
    _validate_row_identity(arrays, dimensions)

    declarations = _array_declarations(payload)
    for name in _ARRAY_NAMES:
        declaration = declarations[name]
        if not isinstance(declaration, Mapping):
            raise BodyFrameSourceHandleError(
                f"Body-frame array declaration is invalid at {name}."
            )
        observed_digest = sha256_array(arrays[name])
        if declaration.get("sha256") != observed_digest:
            raise BodyFrameSourceHandleError(
                f"Body-frame array content digest mismatch at {name}."
            )

    source_value = payload.get("source_keypoint_snapshot")
    recipe_value = payload.get("heading_recipe")
    if not isinstance(source_value, Mapping) or not isinstance(recipe_value, Mapping):
        raise BodyFrameSourceHandleError(
            "Body-frame source and heading recipe identities are incomplete."
        )
    try:
        source = body_frame_source_from_manifest(source_value)
        recipe = body_frame_recipe_from_manifest(recipe_value)
    except (TypeError, ValueError) as exc:
        raise BodyFrameSourceHandleError(
            f"Body-frame source or recipe identity is invalid: {exc}"
        ) from exc
    source_run_path = str(source_value["run_path"])
    verification_document = {
        "schema_id": BODY_FRAME_SOURCE_HANDLE_SCHEMA_ID,
        "schema_version": BODY_FRAME_SOURCE_HANDLE_SCHEMA_VERSION,
        "run_path": exact_run_path,
        "run_manifest_payload_digest": manifest["payload_digest"],
        "selector_eligible": expected_eligible,
        "dimensions": dimensions.as_manifest(),
        "arrays": {name: sha256_array(arrays[name]) for name in _ARRAY_NAMES},
        "source_manifest_digest": source.manifest_digest,
        "source_run_path": source_run_path,
        "source_skeleton_id": source.skeleton_id,
        "source_skeleton_digest": source.skeleton_digest,
        "source_row_signatures_digest": source.keypoint_row_signatures_digest,
        "recipe_id": recipe_value["recipe_id"],
        "recipe_digest": recipe.recipe_digest,
    }

    return BodyFrameSourceHandle(
        analysis_zarr_path=archive_path,
        run_path=exact_run_path,
        selector_eligible=expected_eligible,
        dimensions=dimensions,
        run_manifest=manifest,
        source_snapshot=source_value,
        source_manifest_digest=source.manifest_digest,
        source_run_path=source_run_path,
        source_skeleton_id=source.skeleton_id,
        source_skeleton_digest=source.skeleton_digest,
        source_row_signatures_digest=source.keypoint_row_signatures_digest,
        recipe_id=str(recipe_value["recipe_id"]),
        recipe_digest=recipe.recipe_digest,
        recipe_skeleton_digest=recipe.skeleton_digest,
        heading_computation_digest=recipe.heading_computation_digest,
        arrays=arrays,
        verification_digest=canonical_json_sha256(verification_document),
        _verification_seal=_BODY_FRAME_SOURCE_HANDLE_SEAL,
    )


def require_body_frame_source_handle(value: object) -> BodyFrameSourceHandle:
    """Require one loader-minted body-frame source handle."""

    if type(value) is not BodyFrameSourceHandle:
        raise BodyFrameSourceHandleError(
            "A verified BodyFrameSourceHandle is required."
        )
    value.assert_verified()
    return value


__all__ = [
    "BODY_FRAME_RUN_PREFIX",
    "BODY_FRAME_SOURCE_HANDLE_SCHEMA_ID",
    "BODY_FRAME_SOURCE_HANDLE_SCHEMA_VERSION",
    "BodyFrameSourceHandle",
    "BodyFrameSourceHandleError",
    "load_body_frame_source_handle",
    "require_body_frame_source_handle",
]
