"""Strict read-only access to one provider chaser-distance candidate.

The candidate writer intentionally publishes selector-ineligible evidence under
``analysis/provider_chaser_distance_candidate_runs/<run>``.  This module is the
consumer boundary for that evidence.  It accepts one caller-supplied bare run
name, verifies the immutable publication and its source authorities, copies the
native stimulus-sample arrays into read-only C-contiguous NumPy arrays, and
never resolves a selector or changes the scientific row axis.

In particular, ``source_acquisition_frame_index`` is lineage only.  Multiple
native stimulus samples may legitimately contain the same acquisition-frame
index; this reader preserves those rows and does not turn them into camera
frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis import provider_chaser_distance_candidates as writer
from fisheye.shared.run_provenance import validate_run_provenance
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    MetadataEquivalenceReceipt,
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


PROVIDER_CHASER_STIMULUS_SOURCE_HANDLE_SCHEMA_ID = (
    "palette.provider_chaser_stimulus_source_handle"
)
PROVIDER_CHASER_STIMULUS_SOURCE_HANDLE_SCHEMA_VERSION = 1
RUNS_PREFIX = f"{writer.PARENT_PATH}/"

_HANDLE_SEAL = object()
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SELECTOR_NAMES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative",
        "active",
        "active_run",
        "current",
        "current_run",
        "default",
        "default_run",
        "selected",
        "selected_run",
        "fallback",
        "newest",
    }
)

# These are the exact arrays emitted by the existing candidate writer.  Role
# arrays are a forward-compatible optional extension: the current canary does
# not emit one, but a future candidate may provide a declared per-chaser role
# without changing this native-source boundary.
_REQUIRED_ARRAY_PATHS = frozenset(
    {
        "samples/stimulus_frame_num",
        "samples/source_acquisition_frame_index",
        "samples/timestamp_ns",
        "samples/stimulus_epoch_window_id",
        "samples/source_stimulus_run_row_index",
        "samples/source_stimulus_source_row_index",
        "positions/source_position_run_row_index",
        "positions/source_position_source_row_index",
        "positions/source_position_instance_key",
        "positions/source_position_failure_reason_code",
        "positions/fish_position_source_camera_xy",
        "positions/fish_valid",
        "positions/fish_position_arena_xy",
        "positions/chaser_position_arena_xy",
        "positions/chaser_valid",
        "chasers/chaser_index",
        "distances/distance_px",
        "distances/distance_mm",
        "distances/nearest_chaser_index",
        "distances/nearest_distance_mm",
        "epoch_summary/window_id",
        "epoch_summary/label_bytes",
        "epoch_summary/start_frame",
        "epoch_summary/end_frame",
        "epoch_summary/valid_frame_count",
        "epoch_summary/mean_distance_mm",
        "epoch_summary/min_distance_mm",
        "epoch_summary/p05_distance_mm",
        "epoch_summary/p50_distance_mm",
        "epoch_summary/p95_distance_mm",
        "epoch_summary/fraction_within_threshold",
        "epoch_distributions/window_id",
        "epoch_distributions/chaser_index",
        "epoch_distributions/bin_edges_mm",
        "epoch_distributions/bin_centers_mm",
        "epoch_distributions/hist_counts",
        "epoch_distributions/hist_density",
        "epoch_distributions/valid_sample_count",
        "visualizations/distance_histogram_png",
        "visualizations/distance_trace_png",
    }
)
_OPTIONAL_ROLE_ARRAY_PATHS = frozenset(
    {
        "chasers/behavior_role",
        "chasers/behavior_role_code",
        "chasers/chaser_behavior_role",
        "chasers/chaser_behavior_role_code",
    }
)
_KNOWN_ARRAY_PATHS = _REQUIRED_ARRAY_PATHS | _OPTIONAL_ROLE_ARRAY_PATHS


class ProviderChaserStimulusSourceHandleError(ValueError):
    """Raised when an exact provider chaser source cannot be sealed."""


@dataclass(frozen=True, slots=True)
class ProviderChaserStimulusDimensions:
    """Declared axes of the native provider candidate."""

    total_frames: int
    n_samples: int
    n_chasers: int
    n_stimulus_rows: int
    n_fish_rows: int

    @property
    def sample_axis(self) -> str:
        return "stimulus_samples"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderChaserStimulusSourceHandleError(
            f"{field} must be a string-keyed mapping."
        )
    return value


def _require_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderChaserStimulusSourceHandleError(
            f"{field} must be one non-empty exact string."
        )
    return value


def _require_digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ProviderChaserStimulusSourceHandleError(
            f"{field} must be one lowercase SHA-256 digest."
        )
    return value


def _require_exact_bare_run_name(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderChaserStimulusSourceHandleError(
            "run_name must be one exact bare run-name string."
        )
    if (
        value in {".", ".."}
        or value in _SELECTOR_NAMES
        or "/" in value
        or "\\" in value
        or not _RUN_NAME_RE.fullmatch(value)
    ):
        raise ProviderChaserStimulusSourceHandleError(
            "run_name must be one concrete run name, not a selector, traversal, "
            "nested path, fallback, or newest alias."
        )
    return value


def _readonly_snapshot(node: Any, *, path: str) -> np.ndarray:
    try:
        dtype = np.dtype(node.dtype)
        shape = tuple(int(size) for size in node.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProviderChaserStimulusSourceHandleError(
            f"Array {path!r} has invalid typed metadata: {exc}."
        ) from exc
    if dtype.hasobject or dtype.kind not in {"b", "i", "u", "f"}:
        raise ProviderChaserStimulusSourceHandleError(
            f"Array {path!r} is not a supported typed numeric array: {dtype.str!r}."
        )
    try:
        value = np.array(node[:], dtype=dtype, copy=True, order="C")
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        raise ProviderChaserStimulusSourceHandleError(
            f"Unable to read typed array {path!r}: {exc}."
        ) from exc
    if value.shape != shape:
        raise ProviderChaserStimulusSourceHandleError(
            f"Array {path!r} changed shape while being read."
        )
    value.setflags(write=False)
    return value


def _array_node(group: Any, path: str) -> zarr.Array:
    try:
        node = group[path]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderChaserStimulusSourceHandleError(
            f"Declared array is missing: {path!r}."
        ) from exc
    if not isinstance(node, zarr.Array):
        raise ProviderChaserStimulusSourceHandleError(
            f"Declared path is not an array: {path!r}."
        )
    return node


def _array_paths(group: Any, *, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for name in group.array_keys():
        paths.add(f"{prefix}{name}")
    for name in group.group_keys():
        paths.update(_array_paths(group[name], prefix=f"{prefix}{name}/"))
    return paths


def _strict_manifest(run: Any, *, run_name: str, exact_run_path: str) -> tuple[Mapping[str, Any], str]:
    raw = run.attrs.get(writer.MANIFEST_ATTR)
    manifest = _require_mapping(raw, field=writer.MANIFEST_ATTR)
    if set(manifest) != {"schema_id", "schema_version", "payload", "payload_digest"}:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate manifest has missing or extra envelope fields."
        )
    if manifest.get("schema_id") != writer.MANIFEST_SCHEMA_ID or manifest.get(
        "schema_version"
    ) != writer.MANIFEST_SCHEMA_VERSION:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate manifest schema identity is invalid."
        )
    payload = _require_mapping(manifest.get("payload"), field="manifest.payload")
    payload_digest = _require_digest(
        manifest.get("payload_digest"), field="manifest.payload_digest"
    )
    if canonical_json_sha256(dict(payload)) != payload_digest:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate manifest payload digest is stale."
        )
    if run.attrs.get(writer.MANIFEST_DIGEST_ATTR) != payload_digest:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate manifest digest attribute is stale."
        )
    if set(payload) != {
        "schema_id",
        "schema_version",
        "run_name",
        "run_path",
        "status",
        "stage_selector_eligible",
        "method",
        "method_version",
        "source_authority",
        "parameters",
        "arrays",
        "publication",
    }:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate manifest payload has missing or extra fields."
        )
    if (
        payload["schema_id"] != writer.MANIFEST_SCHEMA_ID
        or payload["schema_version"] != writer.MANIFEST_SCHEMA_VERSION
        or payload["run_name"] != run_name
        or payload["run_path"] != exact_run_path
        or payload["status"] != RUN_STATUS_COMPLETE
        or payload["stage_selector_eligible"] is not False
        or payload["method"] != writer.METHOD
        or payload["method_version"] != writer.METHOD_VERSION
    ):
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate manifest does not bind the exact complete ineligible run."
        )
    if run.attrs.get("run_name") != run_name:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate run_name attribute is stale."
        )
    if run.attrs.get("stage_selector_eligible") is not False:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate is not selector-ineligible."
        )
    return manifest, payload_digest


def _validate_provenance(
    run: Any,
    *,
    source_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = run.attrs.get("run_provenance")
    result = validate_run_provenance(raw)
    if not result.valid:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate provenance is invalid: " + "; ".join(result.errors)
        )
    provenance = _require_mapping(raw, field="run_provenance")
    inputs = _require_mapping(
        provenance.get("input_run_ids"), field="run_provenance.input_run_ids"
    )
    expected = {
        "position": source_authority["position"]["run_path"],
        "stimulus": source_authority["stimulus"]["run_path"],
        "stimulus_epoch": source_authority["stimulus_epoch"]["run_path"],
    }
    if dict(inputs) != expected:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate provenance does not bind the exact source runs."
        )
    return _freeze(provenance)


def _validate_authorities(
    payload: Mapping[str, Any],
    *,
    recording_id: str,
) -> Mapping[str, Any]:
    authority = _require_mapping(payload.get("source_authority"), field="source_authority")
    if authority.get("schema_id") != "palette.provider_chaser_distance_candidate_source_authority" or authority.get("schema_version") != 1:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate source-authority schema identity is invalid."
        )
    if authority.get("recording_id") != recording_id:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate source authority belongs to another recording."
        )
    for key, value in _walk_mappings(authority):
        if key == "recording_id" and value != recording_id:
            raise ProviderChaserStimulusSourceHandleError(
                "A nested provider authority belongs to another recording."
            )
        if key.endswith("sha256") or key == "record_sha256":
            _require_digest(value, field=f"source_authority.{key}")
    for name in ("position", "stimulus", "stimulus_epoch"):
        _require_mapping(authority.get(name), field=f"source_authority.{name}")
    if not isinstance(authority.get("acquisition_frame_authority"), Mapping):
        raise ProviderChaserStimulusSourceHandleError(
            "source_authority.acquisition_frame_authority is missing."
        )
    if not isinstance(authority.get("fps_authority"), Mapping):
        raise ProviderChaserStimulusSourceHandleError(
            "source_authority.fps_authority is missing."
        )
    for name in ("total_frames", "stimulus_sample_count"):
        value = authority.get(name)
        if type(value) is not int or value < 0:
            raise ProviderChaserStimulusSourceHandleError(
                f"source_authority.{name} must be a non-negative integer."
            )
    return _freeze(authority)


def _walk_mappings(value: Mapping[str, Any]):
    for key, item in value.items():
        yield str(key), item
        if isinstance(item, Mapping):
            yield from _walk_mappings(item)


def _validate_declarations(
    run: Any,
    *,
    payload: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], list[Mapping[str, Any]]]:
    raw_declarations = payload.get("arrays")
    if not isinstance(raw_declarations, list) or not raw_declarations:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate manifest arrays must be a non-empty list."
        )
    declarations: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for raw in raw_declarations:
        declaration = _require_mapping(raw, field="manifest.array")
        if set(declaration) != {"path", "dtype", "shape", "sha256"}:
            raise ProviderChaserStimulusSourceHandleError(
                "Provider candidate array declarations have missing or extra fields."
            )
        path = _require_text(declaration.get("path"), field="array.path")
        if (
            path in paths
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or path not in _KNOWN_ARRAY_PATHS
        ):
            raise ProviderChaserStimulusSourceHandleError(
                f"Provider candidate array path is missing, extra, or non-canonical: {path!r}."
            )
        if type(declaration.get("dtype")) is not str:
            raise ProviderChaserStimulusSourceHandleError(
                f"Provider candidate array {path!r} has no exact dtype declaration."
            )
        shape = declaration.get("shape")
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ProviderChaserStimulusSourceHandleError(
                f"Provider candidate array {path!r} has an invalid shape declaration."
            )
        _require_digest(declaration.get("sha256"), field=f"array {path} sha256")
        paths.append(path)
        declarations.append(declaration)
    if paths != sorted(paths):
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate array declarations are reordered."
        )
    if not _REQUIRED_ARRAY_PATHS.issubset(paths):
        missing = sorted(_REQUIRED_ARRAY_PATHS - set(paths))
        raise ProviderChaserStimulusSourceHandleError(
            f"Provider candidate array declarations omit required arrays: {missing}."
        )
    actual_paths = _array_paths(run)
    if actual_paths != set(paths):
        missing = sorted(set(paths) - actual_paths)
        extra = sorted(actual_paths - set(paths))
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate arrays and declarations differ: "
            f"missing={missing}, extra={extra}."
        )
    arrays: dict[str, np.ndarray] = {}
    for declaration in declarations:
        path = str(declaration["path"])
        value = _readonly_snapshot(_array_node(run, path), path=path)
        if value.dtype.str != declaration["dtype"] or list(value.shape) != declaration["shape"]:
            raise ProviderChaserStimulusSourceHandleError(
                f"Provider candidate array {path!r} dtype or shape differs from its declaration."
            )
        if sha256_array(value) != declaration["sha256"]:
            raise ProviderChaserStimulusSourceHandleError(
                f"Provider candidate array {path!r} content digest differs from its declaration."
            )
        arrays[path] = value
    return arrays, declarations


def _validate_native_layout(
    arrays: Mapping[str, np.ndarray],
    *,
    total_frames: int,
    authority: Mapping[str, Any],
) -> tuple[ProviderChaserStimulusDimensions, Mapping[str, Any]]:
    samples = arrays["samples/stimulus_frame_num"]
    acquisition = arrays["samples/source_acquisition_frame_index"]
    timestamps = arrays["samples/timestamp_ns"]
    sample_count = int(samples.size)
    chaser_axis = arrays["chasers/chaser_index"]
    n_chasers = int(chaser_axis.size)
    if samples.dtype != np.dtype("<i8") or samples.ndim != 1 or sample_count <= 0:
        raise ProviderChaserStimulusSourceHandleError(
            "Native stimulus_frame_num must be a non-empty int64 sample axis."
        )
    if np.any(np.diff(samples) <= 0):
        raise ProviderChaserStimulusSourceHandleError(
            "Native stimulus_frame_num must remain strictly ordered and unique."
        )
    if acquisition.dtype != np.dtype("<i8") or acquisition.shape != samples.shape:
        raise ProviderChaserStimulusSourceHandleError(
            "Native acquisition-frame lineage is not aligned to stimulus samples."
        )
    if np.any(acquisition < 0) or np.any(acquisition >= total_frames):
        raise ProviderChaserStimulusSourceHandleError(
            "Native acquisition-frame lineage leaves the declared recording domain."
        )
    if timestamps.dtype != np.dtype("<i8") or timestamps.shape != samples.shape:
        raise ProviderChaserStimulusSourceHandleError(
            "Native timestamp_ns is not aligned to stimulus samples."
        )
    if np.any(np.diff(timestamps) < 0) or np.any(np.diff(acquisition) < 0):
        raise ProviderChaserStimulusSourceHandleError(
            "Native timestamps and acquisition lineage must be nondecreasing on "
            "the ordered stimulus sample axis."
        )
    if chaser_axis.dtype != np.dtype("<i2") or chaser_axis.ndim != 1 or n_chasers <= 0:
        raise ProviderChaserStimulusSourceHandleError(
            "Chaser identity axis is not a non-empty int16 axis."
        )
    if np.any(chaser_axis < 0) or np.unique(chaser_axis).size != n_chasers:
        raise ProviderChaserStimulusSourceHandleError(
            "Chaser identity axis is not unique and non-negative."
        )
    expected_pair = (sample_count, n_chasers)
    expected_position = (sample_count, n_chasers, 2)
    for path in (
        "samples/stimulus_epoch_window_id",
        "positions/source_position_run_row_index",
        "positions/source_position_source_row_index",
        "positions/source_position_instance_key",
        "positions/source_position_failure_reason_code",
        "positions/fish_valid",
        "positions/fish_position_source_camera_xy",
        "positions/fish_position_arena_xy",
    ):
        if arrays[path].shape[0] != sample_count:
            raise ProviderChaserStimulusSourceHandleError(
                f"Fish/sample lineage array {path!r} is not aligned to the native sample axis."
            )
    if arrays["positions/fish_position_source_camera_xy"].shape != (sample_count, 2):
        raise ProviderChaserStimulusSourceHandleError(
            "Fish source-camera positions do not have shape [sample, 2]."
        )
    if arrays["positions/fish_position_arena_xy"].shape != (sample_count, 2):
        raise ProviderChaserStimulusSourceHandleError(
            "Fish arena positions do not have shape [sample, 2]."
        )
    if arrays["positions/chaser_position_arena_xy"].shape != expected_position:
        raise ProviderChaserStimulusSourceHandleError(
            "Chaser positions do not preserve the complete sample-by-chaser axis."
        )
    if arrays["positions/chaser_valid"].shape != expected_pair or arrays["positions/chaser_valid"].dtype != np.dtype(bool):
        raise ProviderChaserStimulusSourceHandleError(
            "Chaser validity does not preserve the complete sample-by-chaser axis."
        )
    if arrays["positions/fish_valid"].dtype != np.dtype(bool):
        raise ProviderChaserStimulusSourceHandleError(
            "Fish validity must be an exact bool array."
        )
    for path in ("samples/source_stimulus_run_row_index", "samples/source_stimulus_source_row_index"):
        if (
            arrays[path].dtype != np.dtype("<i8")
            or arrays[path].shape != expected_pair
            or np.any(arrays[path] < 0)
        ):
            raise ProviderChaserStimulusSourceHandleError(
                f"Stimulus source-row lineage {path!r} is not nonnegative int64 "
                "sample-by-chaser evidence."
            )
    if np.unique(arrays["samples/source_stimulus_run_row_index"]).size != (
        sample_count * n_chasers
    ):
        raise ProviderChaserStimulusSourceHandleError(
            "Stimulus source run-row lineage is not unique per sample/chaser row."
        )
    for path in (
        "positions/source_position_run_row_index",
        "positions/source_position_source_row_index",
    ):
        if (
            arrays[path].dtype != np.dtype("<i8")
            or arrays[path].shape != (sample_count,)
            or np.any(arrays[path] < -1)
        ):
            raise ProviderChaserStimulusSourceHandleError(
                f"Fish provider lineage {path!r} is not aligned int64 evidence."
            )
    fish_projection_paths = (
        "positions/source_position_run_row_index",
        "positions/source_position_source_row_index",
        "positions/source_position_instance_key",
        "positions/source_position_failure_reason_code",
        "positions/fish_position_source_camera_xy",
        "positions/fish_position_arena_xy",
        "positions/fish_valid",
    )
    for acquisition_frame in np.unique(acquisition):
        rows = np.flatnonzero(acquisition == acquisition_frame)
        if rows.size < 2:
            continue
        reference = int(rows[0])
        for path in fish_projection_paths:
            values = arrays[path]
            left = values[reference]
            for row in rows[1:]:
                right = values[int(row)]
                if left.dtype.kind in {"f", "c"}:
                    equal = np.array_equal(left, right, equal_nan=True)
                else:
                    equal = np.array_equal(left, right)
                if not equal:
                    raise ProviderChaserStimulusSourceHandleError(
                        "Fish-side acquisition projection is contradictory for "
                        f"source_acquisition_frame_index={int(acquisition_frame)} "
                        f"at {path!r}."
                    )
    for path in (
        "distances/distance_px",
        "distances/distance_mm",
    ):
        if arrays[path].shape != expected_pair:
            raise ProviderChaserStimulusSourceHandleError(
                f"Derived distance array {path!r} is not sample-by-chaser aligned."
            )
    for path in _OPTIONAL_ROLE_ARRAY_PATHS.intersection(arrays):
        if arrays[path].shape not in {(n_chasers,), expected_pair}:
            raise ProviderChaserStimulusSourceHandleError(
                f"Optional chaser role array {path!r} has an incompatible axis."
            )
    if authority.get("total_frames") != total_frames:
        raise ProviderChaserStimulusSourceHandleError(
            "Source authority total_frames does not match the acquisition root."
        )
    if authority.get("stimulus_sample_count") != sample_count:
        raise ProviderChaserStimulusSourceHandleError(
            "Source authority stimulus_sample_count does not match the native sample axis."
        )
    dimensions = ProviderChaserStimulusDimensions(
        total_frames=total_frames,
        n_samples=sample_count,
        n_chasers=n_chasers,
        n_stimulus_rows=int(arrays["samples/source_stimulus_run_row_index"].size),
        n_fish_rows=int(arrays["positions/source_position_run_row_index"].size),
    )
    registries = MappingProxyType(
        {
            "chaser_index": tuple(int(value) for value in chaser_axis.tolist()),
            "sample_axis": "stimulus_frame_num",
            "source_row_axis": "source_stimulus_run_row_index",
        }
    )
    return dimensions, registries


def _metadata_equivalence(archive: Path, *, run_path: str) -> MetadataEquivalenceReceipt:
    try:
        return validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ProviderChaserStimulusSourceHandleError(
            f"Published candidate direct/consolidated metadata is missing, stale, "
            f"or divergent: {exc}."
        ) from exc


@dataclass(frozen=True, init=False, eq=False)
class ProviderChaserStimulusSourceHandle:
    """Verified immutable snapshot of one native provider candidate."""

    analysis_zarr_path: Path
    run_path: str
    run_name: str
    recording_id: str
    selector_eligible: bool
    dimensions: ProviderChaserStimulusDimensions
    manifest: Mapping[str, Any] = field(repr=False)
    manifest_sha256: str
    provenance: Mapping[str, Any] = field(repr=False)
    authorities: Mapping[str, Any] = field(repr=False)
    registries: Mapping[str, Any] = field(repr=False)
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    metadata_equivalence: Mapping[str, Any] = field(repr=False)
    verification_digest: str
    _use_consolidated: bool = field(repr=False, compare=False)
    _expected_recording_id: str | None = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _HANDLE_SEAL:
            raise ProviderChaserStimulusSourceHandleError(
                "Provider chaser stimulus handles can only be minted by their loader."
            )
        for name, value in values.items():
            if name in {"manifest", "provenance", "authorities", "registries", "metadata_equivalence"}:
                value = _freeze(value)
            elif name == "arrays":
                value = MappingProxyType(
                    {path: _readonly_snapshot(array, path=path) for path, array in value.items()}
                )
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_verification_seal", _HANDLE_SEAL)

    @property
    def source_authority(self) -> Mapping[str, Any]:
        return self.authorities

    @property
    def source_stimulus_run_path(self) -> str:
        return str(self.authorities["stimulus"]["run_path"])

    @property
    def source_stimulus_run_row_index(self) -> np.ndarray:
        return self.arrays["samples/source_stimulus_run_row_index"]

    @property
    def source_stimulus_source_row_index(self) -> np.ndarray:
        return self.arrays["samples/source_stimulus_source_row_index"]

    @property
    def stimulus_frame_num(self) -> np.ndarray:
        return self.arrays["samples/stimulus_frame_num"]

    @property
    def source_acquisition_frame_index(self) -> np.ndarray:
        return self.arrays["samples/source_acquisition_frame_index"]

    @property
    def timestamp_ns(self) -> np.ndarray:
        return self.arrays["samples/timestamp_ns"]

    @property
    def chaser_index(self) -> np.ndarray:
        return self.arrays["chasers/chaser_index"]

    @property
    def chaser_position_arena_xy(self) -> np.ndarray:
        return self.arrays["positions/chaser_position_arena_xy"]

    @property
    def chaser_valid(self) -> np.ndarray:
        return self.arrays["positions/chaser_valid"]

    @property
    def fish_position_source_camera_xy(self) -> np.ndarray:
        return self.arrays["positions/fish_position_source_camera_xy"]

    @property
    def fish_position_arena_xy(self) -> np.ndarray:
        return self.arrays["positions/fish_position_arena_xy"]

    @property
    def fish_valid(self) -> np.ndarray:
        return self.arrays["positions/fish_valid"]

    @property
    def fish_source_position_run_row_index(self) -> np.ndarray:
        return self.arrays["positions/source_position_run_row_index"]

    @property
    def fish_source_position_source_row_index(self) -> np.ndarray:
        return self.arrays["positions/source_position_source_row_index"]

    @property
    def fish_source_position_instance_key(self) -> np.ndarray:
        return self.arrays["positions/source_position_instance_key"]

    @property
    def behavior_roles(self) -> np.ndarray | None:
        for path in sorted(_OPTIONAL_ROLE_ARRAY_PATHS):
            if path in self.arrays:
                return self.arrays[path]
        return None

    def array(self, path: str) -> np.ndarray:
        if type(path) is not str or path not in self.arrays:
            raise KeyError(f"Unknown provider chaser stimulus array {path!r}.")
        return self.arrays[path]

    def assert_current(self) -> None:
        if self._verification_seal is not _HANDLE_SEAL:
            raise ProviderChaserStimulusSourceHandleError(
                "Provider chaser stimulus handle verification seal is absent."
            )
        refreshed = load_provider_chaser_stimulus_source_handle(
            self.analysis_zarr_path,
            run_name=self.run_name,
            expected_recording_id=self._expected_recording_id,
            use_consolidated=self._use_consolidated,
            expected_manifest_sha256=self.manifest_sha256,
        )
        if refreshed.verification_digest != self.verification_digest:
            raise ProviderChaserStimulusSourceHandleError(
                "Provider chaser stimulus candidate changed after the handle was sealed."
            )

    def assert_verified(self) -> None:
        self.assert_current()


def _load_once(
    archive: Path,
    *,
    exact_run_path: str,
    run_name: str,
    use_consolidated: bool,
    expected_recording_id: str | None,
    expected_manifest_sha256: str | None,
    equivalence: MetadataEquivalenceReceipt,
) -> ProviderChaserStimulusSourceHandle:
    try:
        root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
        run = root[exact_run_path]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ProviderChaserStimulusSourceHandleError(
            f"Unable to open exact provider candidate {exact_run_path!r}: {exc}."
        ) from exc
    if not isinstance(run, zarr.Group):
        raise ProviderChaserStimulusSourceHandleError(
            "Exact provider candidate path is not a Zarr group."
        )
    try:
        parent = root[writer.PARENT_PATH]
    except (KeyError, ValueError) as exc:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate parent namespace is missing."
        ) from exc
    if set(_SELECTOR_NAMES).intersection(parent.attrs):
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate namespace contains a selector or fallback attribute."
        )
    if (
        run.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or run.attrs.get("stage_selector_eligible") is not False
    ):
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate does not satisfy complete selector-ineligible lifecycle."
        )
    if (
        run.attrs.get("schema_id") != writer.SCHEMA_ID
        or run.attrs.get("schema_version") != writer.SCHEMA_VERSION
        or run.attrs.get("row_axis") != "stimulus_samples"
    ):
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate schema or native row-axis declaration is invalid."
        )
    manifest, manifest_sha256 = _strict_manifest(
        run, run_name=run_name, exact_run_path=exact_run_path
    )
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate manifest differs from the expected digest."
        )
    recording_id = _require_text(run.attrs.get("recording_id"), field="run.recording_id")
    if expected_recording_id is not None and recording_id != expected_recording_id:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate recording_id differs from the expected recording."
        )
    root_recording_id = _require_text(root.attrs.get("recording_id"), field="root.recording_id")
    if root_recording_id != recording_id:
        raise ProviderChaserStimulusSourceHandleError(
            "Provider candidate recording_id does not match the archive root."
        )
    authority = _validate_authorities(manifest["payload"], recording_id=recording_id)
    provenance = _validate_provenance(run, source_authority=authority)
    arrays, _declarations = _validate_declarations(run, payload=manifest["payload"])
    total_frames = int(authority["total_frames"])
    dimensions, registries = _validate_native_layout(
        arrays, total_frames=total_frames, authority=authority
    )
    verification = {
        "schema_id": PROVIDER_CHASER_STIMULUS_SOURCE_HANDLE_SCHEMA_ID,
        "schema_version": PROVIDER_CHASER_STIMULUS_SOURCE_HANDLE_SCHEMA_VERSION,
        "run_path": exact_run_path,
        "recording_id": recording_id,
        "manifest_sha256": manifest_sha256,
        "arrays": {path: sha256_array(value) for path, value in sorted(arrays.items())},
        "dimensions": {
            "total_frames": dimensions.total_frames,
            "n_samples": dimensions.n_samples,
            "n_chasers": dimensions.n_chasers,
        },
        "metadata_equivalence": equivalence.to_json(),
        "selector_eligible": False,
        "native_sample_axis": "stimulus_frame_num",
    }
    return ProviderChaserStimulusSourceHandle(
        analysis_zarr_path=archive,
        run_path=exact_run_path,
        run_name=run_name,
        recording_id=recording_id,
        selector_eligible=False,
        dimensions=dimensions,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        provenance=provenance,
        authorities=authority,
        registries=registries,
        arrays=arrays,
        metadata_equivalence=equivalence.to_json(),
        verification_digest=canonical_json_sha256(verification),
        _use_consolidated=use_consolidated,
        _expected_recording_id=expected_recording_id,
        _verification_seal=_HANDLE_SEAL,
    )


def load_provider_chaser_stimulus_source_handle(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
    use_consolidated: bool = True,
    expected_manifest_sha256: str | None = None,
) -> ProviderChaserStimulusSourceHandle:
    """Load one exact published native-sample candidate.

    Consolidated metadata is the default, and the persisted archive-root
    consolidated generation is always compared to direct metadata first.  The
    direct read used for that proof is not a fallback for the published read.
    """

    if type(use_consolidated) is not bool:
        raise ProviderChaserStimulusSourceHandleError(
            "use_consolidated must be the exact boolean metadata-read choice."
        )
    name = _require_exact_bare_run_name(run_name)
    if expected_recording_id is not None:
        _require_text(expected_recording_id, field="expected_recording_id")
    if expected_manifest_sha256 is not None:
        _require_digest(expected_manifest_sha256, field="expected_manifest_sha256")
    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}.")
    exact_run_path = f"{RUNS_PREFIX}{name}"
    equivalence = _metadata_equivalence(archive, run_path=exact_run_path)
    try:
        direct_root = open_zarr_root(archive, mode="r", use_consolidated=False)
        published_root = open_zarr_root(
            archive, mode="r", use_consolidated=use_consolidated
        )
        direct_recording_id = direct_root.attrs.get("recording_id")
        published_recording_id = published_root.attrs.get("recording_id")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ProviderChaserStimulusSourceHandleError(
            f"Unable to validate archive-root recording identity: {exc}."
        ) from exc
    if direct_recording_id != published_recording_id:
        raise ProviderChaserStimulusSourceHandleError(
            "Archive-root consolidated recording identity is stale."
        )
    snapshot = _load_once(
        archive,
        exact_run_path=exact_run_path,
        run_name=name,
        use_consolidated=use_consolidated,
        expected_recording_id=expected_recording_id,
        expected_manifest_sha256=expected_manifest_sha256,
        equivalence=equivalence,
    )
    if use_consolidated:
        direct = _load_once(
            archive,
            exact_run_path=exact_run_path,
            run_name=name,
            use_consolidated=False,
            expected_recording_id=expected_recording_id,
            expected_manifest_sha256=snapshot.manifest_sha256,
            equivalence=equivalence,
        )
        if direct.verification_digest != snapshot.verification_digest:
            raise ProviderChaserStimulusSourceHandleError(
                "Provider candidate direct and consolidated reads differ."
            )
    return snapshot


def require_provider_chaser_stimulus_source_handle(
    value: object,
) -> ProviderChaserStimulusSourceHandle:
    """Require a loader-minted handle and reverify its current publication."""

    if type(value) is not ProviderChaserStimulusSourceHandle:
        raise ProviderChaserStimulusSourceHandleError(
            "A verified ProviderChaserStimulusSourceHandle is required."
        )
    value.assert_verified()
    return value


__all__ = [
    "PROVIDER_CHASER_STIMULUS_SOURCE_HANDLE_SCHEMA_ID",
    "PROVIDER_CHASER_STIMULUS_SOURCE_HANDLE_SCHEMA_VERSION",
    "ProviderChaserStimulusDimensions",
    "ProviderChaserStimulusSourceHandle",
    "ProviderChaserStimulusSourceHandleError",
    "load_provider_chaser_stimulus_source_handle",
    "require_provider_chaser_stimulus_source_handle",
]
