"""Sealed, exact handles for published observation subject-position runs.

This module is a consumer boundary.  It accepts one caller-supplied run path,
validates that immutable publication in full, and returns a read-only handle
to the exact run and its array nodes.  It deliberately does not inspect or
resolve any parent selector such as ``latest``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import zarr

from fisheye.analysis_workflows.materializers.subject_position import (
    SUBJECT_POSITION_MANIFEST_ATTR,
    SUBJECT_POSITION_MANIFEST_DIGEST_ATTR,
    SUBJECT_POSITION_PARENT_PATH,
    validate_subject_position_manifest,
    validate_subject_position_run,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

_HANDLE_SEAL = object()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RESERVED_RUN_NAMES = frozenset(
    {"latest", "latest_complete", "latest_pending", "authoritative_run"}
)
_OBSERVATION_RUN_PREFIX = f"{SUBJECT_POSITION_PARENT_PATH}/"
_ARRAY_PATHS = (
    "instance_key",
    "source_acquisition_frame_index",
    "source_row_index",
    "position_xy",
    "valid",
    "failure_reason_codes",
)
_OPTIONAL_ARRAY_PATHS = (
    "support/source_points_xy",
    "support/source_points_valid",
    "support/source_point_reason_codes",
    "support/source_point_confidence",
)


class SubjectPositionSourceHandleError(ValueError):
    """Raised when an exact subject-position source cannot be sealed."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SubjectPositionSourceHandleError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _canonical_run_path(run_path: object) -> tuple[str, str]:
    if type(run_path) is not str:
        raise SubjectPositionSourceHandleError("run_path must be an exact string.")
    if (
        not run_path
        or run_path != run_path.strip()
        or run_path.startswith("/")
        or run_path.endswith("/")
        or "\\" in run_path
    ):
        raise SubjectPositionSourceHandleError(
            "run_path must be a relative canonical observation run path."
        )
    if not run_path.startswith(_OBSERVATION_RUN_PREFIX):
        raise SubjectPositionSourceHandleError(
            "run_path must be under analysis/subject_position_runs/observation."
        )
    run_name = run_path[len(_OBSERVATION_RUN_PREFIX) :]
    if (
        not run_name
        or "/" in run_name
        or _RUN_NAME_RE.fullmatch(run_name) is None
        or run_name in _RESERVED_RUN_NAMES
    ):
        raise SubjectPositionSourceHandleError(
            "run_path must name one concrete non-selector observation run."
        )
    canonical = f"{_OBSERVATION_RUN_PREFIX}{run_name}"
    if run_path != canonical:
        raise SubjectPositionSourceHandleError("run_path is not canonical.")
    return canonical, run_name


def _bind_record(
    payload: Mapping[str, Any],
    *,
    field_name: str,
) -> tuple[Mapping[str, Any], str]:
    binding = payload.get(field_name)
    expected_keys = {"record", "sha256"}
    if field_name == "coordinate":
        expected_keys.add("descriptor_sha256")
    if not isinstance(binding, Mapping) or set(binding) != expected_keys:
        raise SubjectPositionSourceHandleError(
            f"Subject-position {field_name} binding is not exact."
        )
    record = binding["record"]
    if not isinstance(record, Mapping):
        raise SubjectPositionSourceHandleError(
            f"Subject-position {field_name} record is not an object."
        )
    digest = _require_sha256(binding["sha256"], name=f"{field_name}_sha256")
    if canonical_json_sha256(record) != digest:
        raise SubjectPositionSourceHandleError(
            f"Subject-position {field_name} digest is stale."
        )
    return _freeze(record), digest


def _array_node(run_group: Any, path: str) -> zarr.Array:
    try:
        node = run_group[path]
    except (KeyError, ValueError) as exc:
        raise SubjectPositionSourceHandleError(
            f"Subject-position array is missing: {path!r}."
        ) from exc
    if not isinstance(node, zarr.Array):
        raise SubjectPositionSourceHandleError(
            f"Subject-position path is not an array: {path!r}."
        )
    return node


@dataclass(frozen=True, init=False)
class SubjectPositionSourceHandle:
    """Read-only binding to one verified observation position publication."""

    _analysis_zarr_path: Path = field(repr=False)
    _run_path: str
    _run_name: str
    _manifest: Mapping[str, Any] = field(repr=False)
    _manifest_sha256: str
    _decoded_content_sha256: str
    _estimator_record: Mapping[str, Any] = field(repr=False)
    _estimator_sha256: str
    _policy_record: Mapping[str, Any] = field(repr=False)
    _policy_sha256: str
    _source_record: Mapping[str, Any] = field(repr=False)
    _source_sha256: str
    _anatomy_record: Mapping[str, Any] = field(repr=False)
    _anatomy_sha256: str
    _coordinate_record: Mapping[str, Any] = field(repr=False)
    _coordinate_sha256: str
    _row_count: int
    _array_nodes: Mapping[str, zarr.Array] = field(repr=False, compare=False)
    _selector_eligible: bool
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        analysis_zarr_path: Path,
        run_path: str,
        run_name: str,
        manifest: Mapping[str, Any],
        manifest_sha256: str,
        decoded_content_sha256: str,
        estimator_record: Mapping[str, Any],
        estimator_sha256: str,
        policy_record: Mapping[str, Any],
        policy_sha256: str,
        source_record: Mapping[str, Any],
        source_sha256: str,
        anatomy_record: Mapping[str, Any],
        anatomy_sha256: str,
        coordinate_record: Mapping[str, Any],
        coordinate_sha256: str,
        row_count: int,
        array_nodes: Mapping[str, zarr.Array],
        selector_eligible: bool,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _HANDLE_SEAL:
            raise SubjectPositionSourceHandleError(
                "Subject-position source handles can only be constructed by the verified loader."
            )
        if type(row_count) is not int or row_count < 0:
            raise SubjectPositionSourceHandleError(
                "row_count must be a non-negative int."
            )
        if type(selector_eligible) is not bool:
            raise SubjectPositionSourceHandleError(
                "selector_eligible must be an exact bool."
            )
        object.__setattr__(self, "_analysis_zarr_path", Path(analysis_zarr_path))
        object.__setattr__(self, "_run_path", run_path)
        object.__setattr__(self, "_run_name", run_name)
        object.__setattr__(self, "_manifest", _freeze(manifest))
        object.__setattr__(self, "_manifest_sha256", manifest_sha256)
        object.__setattr__(self, "_decoded_content_sha256", decoded_content_sha256)
        object.__setattr__(self, "_estimator_record", _freeze(estimator_record))
        object.__setattr__(self, "_estimator_sha256", estimator_sha256)
        object.__setattr__(self, "_policy_record", _freeze(policy_record))
        object.__setattr__(self, "_policy_sha256", policy_sha256)
        object.__setattr__(self, "_source_record", _freeze(source_record))
        object.__setattr__(self, "_source_sha256", source_sha256)
        object.__setattr__(self, "_anatomy_record", _freeze(anatomy_record))
        object.__setattr__(self, "_anatomy_sha256", anatomy_sha256)
        object.__setattr__(self, "_coordinate_record", _freeze(coordinate_record))
        object.__setattr__(self, "_coordinate_sha256", coordinate_sha256)
        object.__setattr__(self, "_row_count", row_count)
        object.__setattr__(self, "_array_nodes", MappingProxyType(dict(array_nodes)))
        object.__setattr__(self, "_selector_eligible", selector_eligible)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def analysis_zarr_path(self) -> Path:
        return self._analysis_zarr_path

    @property
    def run_path(self) -> str:
        return self._run_path

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def subject_position_manifest_sha256(self) -> str:
        return self._manifest_sha256

    @property
    def manifest_digest(self) -> str:
        return self._manifest_sha256

    @property
    def decoded_content_sha256(self) -> str:
        return self._decoded_content_sha256

    @property
    def selector_eligible(self) -> bool:
        return self._selector_eligible

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def estimator_record(self) -> Mapping[str, Any]:
        return self._estimator_record

    @property
    def estimator_sha256(self) -> str:
        return self._estimator_sha256

    @property
    def policy_record(self) -> Mapping[str, Any]:
        return self._policy_record

    @property
    def policy_sha256(self) -> str:
        return self._policy_sha256

    @property
    def source_record(self) -> Mapping[str, Any]:
        return self._source_record

    @property
    def source_sha256(self) -> str:
        return self._source_sha256

    @property
    def anatomy_record(self) -> Mapping[str, Any]:
        return self._anatomy_record

    @property
    def anatomy_sha256(self) -> str:
        return self._anatomy_sha256

    @property
    def coordinate_record(self) -> Mapping[str, Any]:
        return self._coordinate_record

    @property
    def coordinate_sha256(self) -> str:
        return self._coordinate_sha256

    @property
    def array_nodes(self) -> Mapping[str, zarr.Array]:
        return self._array_nodes

    def _node(self, path: str) -> zarr.Array | None:
        return self._array_nodes.get(path)

    @property
    def instance_key_node(self) -> zarr.Array:
        return self._array_nodes["instance_key"]

    @property
    def source_acquisition_frame_index_node(self) -> zarr.Array:
        return self._array_nodes["source_acquisition_frame_index"]

    @property
    def source_row_index_node(self) -> zarr.Array:
        return self._array_nodes["source_row_index"]

    @property
    def position_xy_node(self) -> zarr.Array:
        return self._array_nodes["position_xy"]

    @property
    def valid_node(self) -> zarr.Array:
        return self._array_nodes["valid"]

    @property
    def failure_reason_codes_node(self) -> zarr.Array:
        return self._array_nodes["failure_reason_codes"]

    @property
    def source_points_xy_node(self) -> zarr.Array | None:
        return self._node("support/source_points_xy")

    @property
    def source_points_valid_node(self) -> zarr.Array | None:
        return self._node("support/source_points_valid")

    @property
    def source_point_reason_codes_node(self) -> zarr.Array | None:
        return self._node("support/source_point_reason_codes")

    @property
    def source_point_confidence_node(self) -> zarr.Array | None:
        return self._node("support/source_point_confidence")

    @property
    def source_points_xy(self) -> zarr.Array | None:
        return self.source_points_xy_node

    @property
    def source_points_valid(self) -> zarr.Array | None:
        return self.source_points_valid_node

    @property
    def source_point_reason_codes(self) -> zarr.Array | None:
        return self.source_point_reason_codes_node

    @property
    def source_point_confidence(self) -> zarr.Array | None:
        return self.source_point_confidence_node

    def assert_verified(self) -> None:
        """Reject objects that were not minted by the strict loader."""

        if self._seal is not _HANDLE_SEAL:
            raise SubjectPositionSourceHandleError(
                "Subject-position source handle verification seal is absent."
            )

    # Short aliases keep the handle convenient for consumers that already use
    # the logical array names.  They still return the exact validated nodes.
    @property
    def instance_key(self) -> zarr.Array:
        return self.instance_key_node

    @property
    def source_acquisition_frame_index(self) -> zarr.Array:
        return self.source_acquisition_frame_index_node

    @property
    def source_row_index(self) -> zarr.Array:
        return self.source_row_index_node

    @property
    def position_xy(self) -> zarr.Array:
        return self.position_xy_node

    @property
    def valid(self) -> zarr.Array:
        return self.valid_node

    @property
    def failure_reason_codes(self) -> zarr.Array:
        return self.failure_reason_codes_node


def load_subject_position_source_handle(
    analysis_zarr: str | Path,
    run_path: str,
    *,
    expected_selector_eligible: bool,
    use_consolidated: bool = True,
    expected_manifest_sha256: str | None = None,
) -> SubjectPositionSourceHandle:
    """Load and seal one exact, completed observation position run.

    ``expected_selector_eligible`` is intentionally required even though the
    current subject-position publication is always ineligible.  This makes a
    canary binding explicit at every call site and leaves no implicit route to
    a promoted selector.  ``use_consolidated`` defaults to the immutable
    publication read mode; tests may explicitly select direct metadata.
    """

    canonical_path, run_name = _canonical_run_path(run_path)
    if type(expected_selector_eligible) is not bool:
        raise SubjectPositionSourceHandleError(
            "expected_selector_eligible must be an explicit bool."
        )
    if type(use_consolidated) is not bool:
        raise SubjectPositionSourceHandleError(
            "use_consolidated must be an exact bool."
        )
    if expected_manifest_sha256 is not None:
        _require_sha256(expected_manifest_sha256, name="expected_manifest_sha256")

    archive = Path(analysis_zarr).expanduser().resolve()
    root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
    try:
        run_group = root[canonical_path]
    except (KeyError, ValueError) as exc:
        raise SubjectPositionSourceHandleError(
            f"Exact subject-position run does not exist: {canonical_path!r}."
        ) from exc

    manifest = run_group.attrs.get(SUBJECT_POSITION_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        raise SubjectPositionSourceHandleError(
            "Subject-position run lacks an exact manifest attribute."
        )

    # Invoke both strict existing validators.  The manifest validator checks
    # the envelope and every bound record; the run validator checks Zarr
    # metadata, completion, coordinates, arrays, and decoded content.
    manifest_result = validate_subject_position_manifest(
        manifest,
        expected_run_name=run_name,
        expected_status=RUN_STATUS_COMPLETE,
    )
    payload = manifest_result["payload"]
    if payload.get("run_path") != canonical_path:
        raise SubjectPositionSourceHandleError(
            "Subject-position manifest run_path differs from the requested exact path."
        )
    if run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise SubjectPositionSourceHandleError("Subject-position run is not complete.")
    selector_eligible = run_group.attrs.get("stage_selector_eligible")
    if type(selector_eligible) is not bool:
        raise SubjectPositionSourceHandleError(
            "Subject-position selector eligibility is not an exact bool."
        )
    if selector_eligible is not expected_selector_eligible:
        raise SubjectPositionSourceHandleError(
            "Subject-position selector eligibility differs from the explicit expectation."
        )

    manifest_sha256 = _require_sha256(
        manifest_result["manifest_sha256"], name="manifest_sha256"
    )
    if run_group.attrs.get(SUBJECT_POSITION_MANIFEST_DIGEST_ATTR) != manifest_sha256:
        raise SubjectPositionSourceHandleError(
            "Subject-position manifest attribute digest is stale."
        )
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise SubjectPositionSourceHandleError(
            "Subject-position manifest digest differs from the expected digest."
        )

    run_result = validate_subject_position_run(
        archive,
        canonical_path,
        use_consolidated=use_consolidated,
        expected_status=RUN_STATUS_COMPLETE,
        expected_manifest_sha256=manifest_sha256,
    )
    if payload.get("stage_selector_eligible") is not selector_eligible:
        raise SubjectPositionSourceHandleError(
            "Subject-position manifest selector eligibility is stale."
        )

    nodes: dict[str, zarr.Array] = {}
    declared_paths = {str(entry["path"]) for entry in payload["arrays"]}
    for path in (*_ARRAY_PATHS, *_OPTIONAL_ARRAY_PATHS):
        if path in declared_paths:
            nodes[path] = _array_node(run_group, path)
    if set(_ARRAY_PATHS) - set(
        nodes
    ):  # pragma: no cover - strict validator guards this
        raise SubjectPositionSourceHandleError(
            "Subject-position mandatory array is missing."
        )

    records: dict[str, tuple[Mapping[str, Any], str]] = {
        field_name: _bind_record(payload, field_name=field_name)
        for field_name in ("estimator", "policy", "source", "anatomy", "coordinate")
    }
    decoded_digest = _require_sha256(
        run_result["decoded_content_sha256"], name="decoded_content_sha256"
    )

    return SubjectPositionSourceHandle(
        analysis_zarr_path=archive,
        run_path=canonical_path,
        run_name=run_name,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        decoded_content_sha256=decoded_digest,
        estimator_record=records["estimator"][0],
        estimator_sha256=records["estimator"][1],
        policy_record=records["policy"][0],
        policy_sha256=records["policy"][1],
        source_record=records["source"][0],
        source_sha256=records["source"][1],
        anatomy_record=records["anatomy"][0],
        anatomy_sha256=records["anatomy"][1],
        coordinate_record=records["coordinate"][0],
        coordinate_sha256=records["coordinate"][1],
        row_count=int(run_result["row_count"]),
        array_nodes=nodes,
        selector_eligible=selector_eligible,
        _verification_seal=_HANDLE_SEAL,
    )


def require_subject_position_source_handle(
    value: object,
) -> SubjectPositionSourceHandle:
    """Require one loader-minted subject-position handle."""

    if type(value) is not SubjectPositionSourceHandle:
        raise SubjectPositionSourceHandleError(
            "A verified SubjectPositionSourceHandle is required."
        )
    value.assert_verified()
    return value


__all__ = [
    "SubjectPositionSourceHandle",
    "SubjectPositionSourceHandleError",
    "load_subject_position_source_handle",
    "require_subject_position_source_handle",
]
