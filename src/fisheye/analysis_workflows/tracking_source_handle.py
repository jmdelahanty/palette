"""Sealed handles for one exact immutable modern tracking run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.rowset_fingerprint import (
    build_rowset_fingerprint,
    instance_key_digest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.tracking.run_manifest import (
    TRACKING_RUN_MANIFEST_ATTR,
    TRACKING_RUN_MANIFEST_DIGEST_ATTR,
    TrackingRunManifestError,
    tracking_array_records,
    validate_tracking_run_manifest,
)


TRACKING_SOURCE_HANDLE_SCHEMA_ID = "palette.tracking_source_handle"
TRACKING_SOURCE_HANDLE_SCHEMA_VERSION = 1

_HANDLE_SEAL = object()
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RESERVED_RUN_NAMES = frozenset(
    {"latest", "latest_complete", "latest_pending", "authoritative_run"}
)


class TrackingSourceHandleError(ValueError):
    """Raised when an exact tracking run cannot become modern authority."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _canonical_run_path(value: object) -> tuple[str, str]:
    if type(value) is not str:
        raise TrackingSourceHandleError("run_path must be an exact string.")
    if (
        not value.startswith("tracking_runs/")
        or value.startswith("/")
        or value.endswith("/")
        or value.count("/") != 1
        or "\\" in value
        or value != value.strip()
    ):
        raise TrackingSourceHandleError(
            "run_path must name one exact tracking_runs/<run>."
        )
    name = value.split("/", 1)[1]
    if (
        _RUN_NAME_RE.fullmatch(name) is None
        or name in _RESERVED_RUN_NAMES
    ):
        raise TrackingSourceHandleError(
            "run_path must name one concrete non-selector tracking run."
        )
    return value, name


def _row_vector(
    run: Any,
    name: str,
    *,
    expected_rows: int | None = None,
) -> np.ndarray:
    try:
        values = np.asarray(run[name][:])
    except (KeyError, ValueError) as exc:
        raise TrackingSourceHandleError(
            f"Tracking run is missing required array {name!r}."
        ) from exc
    if values.ndim != 1:
        raise TrackingSourceHandleError(
            f"Tracking array {name!r} must be one-dimensional."
        )
    if expected_rows is not None and values.shape != (expected_rows,):
        raise TrackingSourceHandleError(
            f"Tracking array {name!r} is not row aligned."
        )
    return values


def _verify_source_lineage(
    payload: Mapping[str, Any],
    attrs: Mapping[str, Any],
    instance_key: np.ndarray,
) -> None:
    source = payload["source"]
    for name, value in source.items():
        if attrs.get(name) != value:
            raise TrackingSourceHandleError(
                f"Tracking source-lineage attr {name!r} differs from its manifest."
            )
    if source["source_rowset_fingerprint_status"] != "complete":
        raise TrackingSourceHandleError(
            "Modern tracking authority requires a complete keyed rowset fingerprint."
        )
    if source["source_rowset_row_count"] != int(instance_key.shape[0]):
        raise TrackingSourceHandleError(
            "Tracking source row count differs from persisted instance keys."
        )
    expected_key_digest = instance_key_digest(
        instance_key, expected_row_count=instance_key.shape[0]
    )
    if source["source_rowset_instance_key_digest"] != expected_key_digest:
        raise TrackingSourceHandleError(
            "Tracking source rowset digest differs from persisted instance keys."
        )
    fingerprint = source["source_rowset_fingerprint"]
    expected = build_rowset_fingerprint(
        source_rowset_path=source["source_rowset_path"],
        row_count=instance_key.shape[0],
        instance_keys=instance_key,
        source_edit_revision=source["source_rowset_edit_revision"],
    )
    if fingerprint != expected.fingerprint:
        raise TrackingSourceHandleError(
            "Tracking source rowset fingerprint is stale."
        )
    expected_attrs = expected.to_attrs()
    for name in (
        "source_rowset_fingerprint_schema_id",
        "source_rowset_fingerprint_schema_version",
        "source_rowset_fingerprint_canonicalization",
    ):
        if source[name] != expected_attrs[name]:
            raise TrackingSourceHandleError(
                "Tracking source rowset fingerprint contract differs."
            )


@dataclass(frozen=True)
class _VerifiedTrackingSnapshot:
    manifest: Mapping[str, Any]
    manifest_sha256: str
    selector_eligible: bool
    instance_key: np.ndarray
    track_ids: np.ndarray
    frame_indices: np.ndarray
    arena_ids: np.ndarray
    verification_digest: str


def _read_verified_snapshot(
    archive: Path,
    run_path: str,
    run_name: str,
    *,
    expected_selector_eligible: bool,
    use_consolidated: bool,
    expected_manifest_sha256: str | None,
) -> _VerifiedTrackingSnapshot:
    root = open_zarr_root(
        archive,
        mode="r",
        use_consolidated=use_consolidated,
    )
    try:
        run = root[run_path]
    except (KeyError, ValueError) as exc:
        raise TrackingSourceHandleError(
            f"Exact tracking run does not exist: {run_path!r}."
        ) from exc
    manifest = run.attrs.get(TRACKING_RUN_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        raise TrackingSourceHandleError(
            "Tracking run lacks the modern immutable manifest."
        )
    try:
        validated = validate_tracking_run_manifest(
            manifest,
            expected_run_name=run_name,
            expected_status=RUN_STATUS_COMPLETE,
        )
    except TrackingRunManifestError as exc:
        raise TrackingSourceHandleError(str(exc)) from exc
    payload = validated["payload"]
    digest = validated["manifest_sha256"]
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise TrackingSourceHandleError(
            "Tracking run manifest digest differs from the expected authority."
        )
    if run.attrs.get(TRACKING_RUN_MANIFEST_DIGEST_ATTR) != digest:
        raise TrackingSourceHandleError(
            "Tracking run manifest digest attr is stale."
        )
    if (
        run.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
    ):
        raise TrackingSourceHandleError(
            "Tracking run does not satisfy the strict completion contract."
        )
    eligible = run.attrs.get("stage_selector_eligible")
    if type(eligible) is not bool or eligible is not expected_selector_eligible:
        raise TrackingSourceHandleError(
            "Tracking selector eligibility differs from the explicit expectation."
        )
    if payload["stage_selector_eligible"] is not eligible:
        raise TrackingSourceHandleError(
            "Tracking manifest selector eligibility is stale."
        )
    for name in (
        "tracking_method",
        "tracking_identity_mode",
        "unassigned_track_id",
    ):
        if run.attrs.get(name) != payload[name]:
            raise TrackingSourceHandleError(
                f"Tracking attr {name!r} differs from its manifest."
            )
    configuration = payload["tracking_configuration"]
    if (
        run.attrs.get("tracker_parameters") != configuration["tracker_parameters"]
        or run.attrs.get("provenance") != configuration["provenance"]
    ):
        raise TrackingSourceHandleError(
            "Tracking configuration or provenance differs from its manifest."
        )
    actual_records = tracking_array_records(run)
    if actual_records != payload["arrays"]:
        raise TrackingSourceHandleError(
            "Tracking array content or declarations differ from the manifest."
        )
    if canonical_json_sha256(actual_records) != payload["decoded_content_sha256"]:
        raise TrackingSourceHandleError(
            "Tracking decoded content digest is stale."
        )

    instance_key = _row_vector(run, "instance_key")
    if instance_key.dtype != np.dtype("uint64"):
        raise TrackingSourceHandleError(
            "Modern tracking instance_key must be exact uint64[N]."
        )
    if np.unique(instance_key).shape[0] != instance_key.shape[0]:
        raise TrackingSourceHandleError(
            "Modern tracking instance_key contains duplicates."
        )
    if payload["tracking_identity_mode"] != "instance_key":
        raise TrackingSourceHandleError(
            "Legacy positional tracking cannot become modern motion authority."
        )
    rows = int(instance_key.shape[0])
    track_ids = _row_vector(run, "track_ids", expected_rows=rows)
    frame_indices = _row_vector(run, "frame_indices", expected_rows=rows)
    arena_ids = _row_vector(run, "arena_ids", expected_rows=rows)
    source_rows = _row_vector(run, "source_row_indices", expected_rows=rows)
    for name, values in {
        "track_ids": track_ids,
        "frame_indices": frame_indices,
        "arena_ids": arena_ids,
        "source_row_indices": source_rows,
    }.items():
        if values.dtype.kind != "i":
            raise TrackingSourceHandleError(
                f"Tracking array {name!r} must use a signed integer dtype."
            )
    if not np.array_equal(source_rows, np.arange(rows, dtype=source_rows.dtype)):
        raise TrackingSourceHandleError(
            "Tracking source_row_indices do not address the persisted rowset in order."
        )
    present = _row_vector(run, "track_ids_present")
    track_arenas = _row_vector(
        run, "track_arena_ids", expected_rows=int(present.shape[0])
    )
    if present.dtype.kind != "i" or track_arenas.dtype.kind != "i":
        raise TrackingSourceHandleError(
            "Tracking track-axis arrays must use signed integer dtypes."
        )
    if np.unique(present).shape[0] != present.shape[0]:
        raise TrackingSourceHandleError("Tracking track_ids_present is duplicated.")
    assigned_ids = np.unique(track_ids[track_ids != payload["unassigned_track_id"]])
    if not np.array_equal(
        np.sort(assigned_ids.astype(np.int64, copy=False)),
        np.sort(present.astype(np.int64, copy=False)),
    ):
        raise TrackingSourceHandleError(
            "Tracking row assignments differ from track_ids_present."
        )
    _verify_source_lineage(payload, run.attrs, instance_key)
    verification = {
        "schema_id": TRACKING_SOURCE_HANDLE_SCHEMA_ID,
        "schema_version": TRACKING_SOURCE_HANDLE_SCHEMA_VERSION,
        "run_path": run_path,
        "manifest_sha256": digest,
        "selector_eligible": eligible,
        "decoded_content_sha256": payload["decoded_content_sha256"],
        "source": payload["source"],
    }
    return _VerifiedTrackingSnapshot(
        manifest=_freeze(manifest),
        manifest_sha256=digest,
        selector_eligible=eligible,
        instance_key=_readonly(instance_key, dtype=np.dtype("uint64")),
        track_ids=_readonly(track_ids, dtype=np.dtype("int64")),
        frame_indices=_readonly(frame_indices, dtype=np.dtype("int64")),
        arena_ids=_readonly(arena_ids, dtype=np.dtype("int64")),
        verification_digest=canonical_json_sha256(verification),
    )


@dataclass(frozen=True, init=False)
class TrackingSourceHandle:
    """Read-only snapshot plus a live verifier for one exact tracking run."""

    _analysis_zarr_path: Path = field(repr=False)
    _run_path: str
    _run_name: str
    _manifest: Mapping[str, Any] = field(repr=False)
    _manifest_sha256: str
    _selector_eligible: bool
    _use_consolidated: bool
    _instance_key: np.ndarray = field(repr=False, compare=False)
    _track_ids: np.ndarray = field(repr=False, compare=False)
    _frame_indices: np.ndarray = field(repr=False, compare=False)
    _arena_ids: np.ndarray = field(repr=False, compare=False)
    _verification_digest: str
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        analysis_zarr_path: Path,
        run_path: str,
        run_name: str,
        snapshot: _VerifiedTrackingSnapshot,
        use_consolidated: bool,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _HANDLE_SEAL:
            raise TrackingSourceHandleError(
                "Tracking source handles can only be minted by the strict loader."
            )
        object.__setattr__(self, "_analysis_zarr_path", analysis_zarr_path)
        object.__setattr__(self, "_run_path", run_path)
        object.__setattr__(self, "_run_name", run_name)
        object.__setattr__(self, "_manifest", snapshot.manifest)
        object.__setattr__(self, "_manifest_sha256", snapshot.manifest_sha256)
        object.__setattr__(self, "_selector_eligible", snapshot.selector_eligible)
        object.__setattr__(self, "_use_consolidated", use_consolidated)
        object.__setattr__(self, "_instance_key", snapshot.instance_key)
        object.__setattr__(self, "_track_ids", snapshot.track_ids)
        object.__setattr__(self, "_frame_indices", snapshot.frame_indices)
        object.__setattr__(self, "_arena_ids", snapshot.arena_ids)
        object.__setattr__(self, "_verification_digest", snapshot.verification_digest)
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
    def selector_eligible(self) -> bool:
        return self._selector_eligible

    @property
    def instance_key(self) -> np.ndarray:
        return self._instance_key

    @property
    def track_ids(self) -> np.ndarray:
        return self._track_ids

    @property
    def frame_indices(self) -> np.ndarray:
        return self._frame_indices

    @property
    def arena_ids(self) -> np.ndarray:
        return self._arena_ids

    @property
    def verification_digest(self) -> str:
        return self._verification_digest

    def assert_current(self) -> None:
        """Reopen the exact run and reject mutation or stale consolidation."""

        if self._seal is not _HANDLE_SEAL:
            raise TrackingSourceHandleError(
                "Tracking source handle verification seal is absent."
            )
        refreshed = _read_verified_snapshot(
            self.analysis_zarr_path,
            self.run_path,
            self.run_name,
            expected_selector_eligible=self.selector_eligible,
            use_consolidated=self._use_consolidated,
            expected_manifest_sha256=self.manifest_sha256,
        )
        if refreshed.verification_digest != self.verification_digest:
            raise TrackingSourceHandleError(
                "Tracking authority changed after the source handle was sealed."
            )
        if self._use_consolidated:
            direct = _read_verified_snapshot(
                self.analysis_zarr_path,
                self.run_path,
                self.run_name,
                expected_selector_eligible=self.selector_eligible,
                use_consolidated=False,
                expected_manifest_sha256=self.manifest_sha256,
            )
            if direct.verification_digest != self.verification_digest:
                raise TrackingSourceHandleError(
                    "Tracking direct metadata differs from its published consolidated generation."
                )

    def assert_verified(self) -> None:
        self.assert_current()


def load_tracking_source_handle(
    analysis_zarr: str | Path,
    run_path: str,
    *,
    expected_selector_eligible: bool,
    use_consolidated: bool = True,
    expected_manifest_sha256: str | None = None,
) -> TrackingSourceHandle:
    """Load one explicit complete keyed tracking run without selector lookup."""

    exact_path, run_name = _canonical_run_path(run_path)
    if type(expected_selector_eligible) is not bool:
        raise TrackingSourceHandleError(
            "expected_selector_eligible must be an exact bool."
        )
    if type(use_consolidated) is not bool:
        raise TrackingSourceHandleError("use_consolidated must be an exact bool.")
    archive = Path(analysis_zarr).expanduser().resolve()
    snapshot = _read_verified_snapshot(
        archive,
        exact_path,
        run_name,
        expected_selector_eligible=expected_selector_eligible,
        use_consolidated=use_consolidated,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if use_consolidated:
        direct = _read_verified_snapshot(
            archive,
            exact_path,
            run_name,
            expected_selector_eligible=expected_selector_eligible,
            use_consolidated=False,
            expected_manifest_sha256=snapshot.manifest_sha256,
        )
        if direct.verification_digest != snapshot.verification_digest:
            raise TrackingSourceHandleError(
                "Tracking direct metadata differs from its published consolidated generation."
            )
    return TrackingSourceHandle(
        analysis_zarr_path=archive,
        run_path=exact_path,
        run_name=run_name,
        snapshot=snapshot,
        use_consolidated=use_consolidated,
        _verification_seal=_HANDLE_SEAL,
    )


def require_tracking_source_handle(value: object) -> TrackingSourceHandle:
    """Require a loader-minted handle and revalidate its live authority."""

    if type(value) is not TrackingSourceHandle:
        raise TrackingSourceHandleError(
            "A verified TrackingSourceHandle is required."
        )
    value.assert_current()
    return value


__all__ = [
    "TRACKING_SOURCE_HANDLE_SCHEMA_ID",
    "TRACKING_SOURCE_HANDLE_SCHEMA_VERSION",
    "TrackingSourceHandle",
    "TrackingSourceHandleError",
    "load_tracking_source_handle",
    "require_tracking_source_handle",
]
