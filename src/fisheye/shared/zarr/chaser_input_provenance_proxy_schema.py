"""Typed schema for immutable chaser input-provenance proxy candidates.

The in-memory selector is the scientific computation boundary.  This module
is the exact logical boundary immediately before Zarr publication: it keeps
all native candidates and selected rows in typed arrays, keeps the three
logical axes explicit, and binds the complete compact projection record to
the arrays by digest.

No physical-presentation claim is made here.  The projection record supplied
by the current pure result is copied dynamically, so additive fields in that
record cannot be silently dropped by this publication layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from fisheye.shared.chaser_input_provenance_proxy_contract import (
    BEHAVIORAL_DENOMINATOR,
    CAMERA_EXPOSURE_REFERENCE,
    CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE,
    ChaserInputProvenanceProxyResult,
    PHYSICAL_PRESENTATION_VERIFIED,
    PRESENTATION_TIMESTAMP_AVAILABLE,
    PROJECTION_RECORD_SCHEMA_ID,
    PROJECTION_RECORD_SCHEMA_VERSION,
    PROXY_POLICY_ID,
    SCIENTIFIC_USE_CLASS,
    TEMPORAL_ALIGNMENT_CLASS,
    TEMPORAL_ALIGNMENT_REQUIREMENT,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID = (
    "palette.analysis.chaser_input_provenance_proxy"
)
CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION = 1
CHASER_INPUT_PROVENANCE_PROXY_LAYOUT = (
    "frame_candidate_chaser_typed_evidence_v1"
)
CHASER_INPUT_PROVENANCE_PROXY_ARRAY_SCHEMA_ID = (
    "palette.analysis.chaser_input_provenance_proxy.arrays"
)
CHASER_INPUT_PROVENANCE_PROXY_ARRAY_SCHEMA_VERSION = 1
CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH = (
    "analysis/chaser_input_provenance_proxy_runs"
)
CHASER_INPUT_PROVENANCE_PROXY_MATERIALIZATION_SCHEMA_ID = (
    "palette.chaser_input_provenance_proxy_materialization"
)
CHASER_INPUT_PROVENANCE_PROXY_PUBLISH_SCHEMA_ID = (
    "palette.chaser_input_provenance_proxy_publish.v1"
)

FRAME_AXIS = "frame"
CANDIDATE_AXIS = "candidate"
CHASER_AXIS = "chaser"
FRAME_BOUNDARY_AXIS = "frame_boundary"
PROJECTION_RECORD_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# A single controlled code space is used for every reason array.  The
# manifest makes the scope readable, while the Zarr arrays remain numeric.
REASON_CODE_REGISTRY: Mapping[int, str] = MappingProxyType(
    {
        0: "selected",
        1: "no_complete_chaser_sample",
        2: "complete",
        3: "incomplete_chaser_sample",
    }
)
SELECTION_REASON_CODES: Mapping[str, int] = MappingProxyType(
    {value: code for code, value in REASON_CODE_REGISTRY.items() if code < 2}
)
CANDIDATE_REASON_CODES: Mapping[str, int] = MappingProxyType(
    {value: code for code, value in REASON_CODE_REGISTRY.items() if code >= 2}
)


class ChaserInputProvenanceProxySchemaError(ValueError):
    """Raised when a proxy result or manifest violates the typed contract."""


def _fail(message: str) -> None:
    raise ChaserInputProvenanceProxySchemaError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _strict_record(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{field} must be one non-empty mapping.")
    try:
        encoded = json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"{field} must be strict canonical JSON: {exc}")
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        _fail(f"{field} did not canonicalize to an object.")
    return decoded


def _exact_dtype(
    value: object,
    *,
    field: str,
    dtype: str,
    allow_text: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    expected = np.dtype(dtype)
    if array.dtype != expected:
        _fail(f"{field} must have exact dtype {expected.str}, got {array.dtype.str}.")
    if array.dtype.hasobject or (array.dtype.kind in {"U", "S"} and not allow_text):
        _fail(f"{field} cannot be an object or string array.")
    return array


def _shape(value: object, *, field: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        _fail(f"{field} must have exact shape {shape}, got {array.shape}.")
    return array


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_plain(value))


def _require_digest(value: object, *, field: str) -> str:
    if type(value) is not str or PROJECTION_RECORD_SHA256_RE.fullmatch(value) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True, slots=True)
class ChaserInputProvenanceProxyDimensions:
    """Concrete sizes for the frame, candidate, and chaser axes."""

    n_frames: int
    n_candidates: int
    n_chasers: int

    def __post_init__(self) -> None:
        for name in ("n_frames", "n_candidates", "n_chasers"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ChaserInputProvenanceProxySchemaError(
                    f"{name} must be one positive exact integer."
                )

    def as_manifest(self) -> dict[str, int]:
        return {
            "frame": self.n_frames,
            "candidate": self.n_candidates,
            "chaser": self.n_chasers,
            "frame_boundary": self.n_frames + 1,
        }


@dataclass(frozen=True, slots=True)
class ProxyArrayDeclaration:
    """One typed row-evidence array declaration and content digest."""

    path: str
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    content_sha256: str
    description: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "array_schema_id": CHASER_INPUT_PROVENANCE_PROXY_ARRAY_SCHEMA_ID,
            "array_schema_version": CHASER_INPUT_PROVENANCE_PROXY_ARRAY_SCHEMA_VERSION,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "content_sha256": self.content_sha256,
            "row_evidence_only": True,
            "description": self.description,
        }


_ARRAY_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "acquisition_frame_index": "Unique represented input-acquisition frame IDs.",
        "candidate_offsets": "Ragged candidate boundaries indexed by frame.",
        "candidate_sample_count": "Number of native candidates per represented frame.",
        "candidate_native_sample_row_index": "Native stimulus sample row for each candidate.",
        "candidate_stimulus_frame_num": "Stimulus frame number for each candidate.",
        "candidate_timestamp_ns_session": "Logged Citrus session timestamp for each candidate.",
        "candidate_source_acquisition_frame_index": "Input-acquisition provenance frame for each candidate.",
        "candidate_complete": "Whether every declared chaser is valid in the candidate sample.",
        "candidate_reason_code": "Controlled candidate reason code.",
        "candidate_source_stimulus_run_row_index": "Per-candidate, per-chaser source run-row lineage.",
        "candidate_source_stimulus_source_row_index": "Per-candidate, per-chaser source-row lineage.",
        "selected": "Whether a complete native sample was selected for each frame.",
        "selection_reason_code": "Controlled selection reason code.",
        "selected_native_sample_row_index": "Selected native stimulus sample row, or -1.",
        "selected_stimulus_frame_num": "Selected stimulus frame number, or -1.",
        "selected_timestamp_ns_session": "Selected logged session timestamp, or -1.",
        "selected_source_stimulus_run_row_index": "Selected per-chaser source run-row lineage, or -1.",
        "selected_source_stimulus_source_row_index": "Selected per-chaser source-row lineage, or -1.",
        "selected_chaser_index": "Full selected chaser identity matrix.",
        "selected_chaser_position_xy": (
            "Full selected chaser positions in arena_relative_canvas_px; these "
            "must be transformed through the exact published typed chain before "
            "source-camera use."
        ),
        "selected_chaser_valid": "Full selected chaser validity matrix.",
    }
)


def _array_values(result: ChaserInputProvenanceProxyResult) -> dict[str, np.ndarray]:
    """Return every published row-evidence array in stable contract order."""

    return {
        "acquisition_frame_index": np.asarray(result.acquisition_frame_index),
        "candidate_offsets": np.asarray(result.candidate_offsets),
        "candidate_sample_count": np.asarray(result.candidate_sample_count),
        "candidate_native_sample_row_index": np.asarray(
            result.candidate_native_sample_row_index
        ),
        "candidate_stimulus_frame_num": np.asarray(result.candidate_stimulus_frame_num),
        "candidate_timestamp_ns_session": np.asarray(
            result.candidate_timestamp_ns_session
        ),
        "candidate_source_acquisition_frame_index": np.asarray(
            result.candidate_source_acquisition_frame_index
        ),
        "candidate_complete": np.asarray(result.candidate_complete),
        "candidate_reason_code": np.asarray(result.candidate_reason_code),
        "candidate_source_stimulus_run_row_index": np.asarray(
            result.candidate_source_stimulus_run_row_index
        ),
        "candidate_source_stimulus_source_row_index": np.asarray(
            result.candidate_source_stimulus_source_row_index
        ),
        "selected": np.asarray(result.selected),
        "selection_reason_code": np.asarray(result.selection_reason_code),
        "selected_native_sample_row_index": np.asarray(
            result.selected_native_sample_row_index
        ),
        "selected_stimulus_frame_num": np.asarray(result.selected_stimulus_frame_num),
        "selected_timestamp_ns_session": np.asarray(
            result.selected_timestamp_ns_session
        ),
        "selected_source_stimulus_run_row_index": np.asarray(
            result.selected_source_stimulus_run_row_index
        ),
        "selected_source_stimulus_source_row_index": np.asarray(
            result.selected_source_stimulus_source_row_index
        ),
        "selected_chaser_index": np.asarray(result.selected_chaser_index),
        "selected_chaser_position_xy": np.asarray(result.selected_chaser_position_xy),
        "selected_chaser_valid": np.asarray(result.selected_chaser_valid),
    }


def _validate_projection_record(result: ChaserInputProvenanceProxyResult) -> dict[str, Any]:
    record = _strict_record(
        result.acquisition_projection_record,
        field="acquisition_projection_record",
    )
    if record.get("schema_id") != PROJECTION_RECORD_SCHEMA_ID:
        _fail("acquisition_projection_record.schema_id is not the pure proxy schema.")
    if record.get("schema_version") != PROJECTION_RECORD_SCHEMA_VERSION:
        _fail("acquisition_projection_record.schema_version is unsupported.")
    if record.get("recording_id") != result.recording_id:
        _fail("Projection recording_id differs from result recording_id.")
    expected_semantics = {
        "policy_id": PROXY_POLICY_ID,
        "temporal_alignment_requirement": TEMPORAL_ALIGNMENT_REQUIREMENT,
        "temporal_alignment_class": TEMPORAL_ALIGNMENT_CLASS,
        "physical_presentation_verified": PHYSICAL_PRESENTATION_VERIFIED,
        "presentation_timestamp_available": PRESENTATION_TIMESTAMP_AVAILABLE,
        "camera_presentation_clock_transform_available": (
            CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE
        ),
        "camera_exposure_reference": CAMERA_EXPOSURE_REFERENCE,
        "scientific_use_class": SCIENTIFIC_USE_CLASS,
        "behavioral_denominator": BEHAVIORAL_DENOMINATOR,
        "native_sample_axis": "stimulus_samples",
        "native_sample_rows_preserved": True,
        "source_acquisition_frame_field": "source_acquisition_frame_index",
        "missing_frame_rule": "no_carry_forward",
        "candidate_sample_row_index_is_zero_based": True,
    }
    for field, expected in expected_semantics.items():
        if record.get(field) != expected:
            _fail(f"Projection {field} has an unsupported value.")
    if record.get("chaser_count") != int(
        np.asarray(result.selected_chaser_index).shape[1]
    ):
        _fail("Projection chaser_count differs from the selected chaser axis.")
    if record.get("source_authority_id") != result.source_authority_id:
        _fail("Projection source_authority_id differs from the result.")
    for field, value in (
        ("source_authority_digest", result.source_authority_digest),
        ("source_manifest_sha256", result.source_manifest_sha256),
        ("source_verification_digest", result.source_verification_digest),
    ):
        if record.get(field) != value:
            _fail(f"Projection {field} differs from the result.")
        _require_digest(record.get(field), field=f"projection.{field}")
    if record.get("source_run_path") != result.source_run_path:
        _fail("Projection source_run_path differs from the result.")
    observed_digest = _mapping_sha256(record)
    declared_digest = _require_digest(
        result.acquisition_projection_record_sha256,
        field="acquisition_projection_record_sha256",
    )
    if observed_digest != declared_digest:
        _fail("acquisition_projection_record_sha256 does not match the exact record.")
    return record


def validate_proxy_result(
    result: ChaserInputProvenanceProxyResult,
    *,
    revalidate_source: Callable[
        [object], ChaserInputProvenanceProxyResult
    ] | None = None,
) -> tuple[ChaserInputProvenanceProxyDimensions, dict[str, np.ndarray], dict[str, Any]]:
    """Validate a result before it becomes an immutable storage payload.

    When ``revalidate_source`` is supplied, it reruns that analysis-layer pure
    selector against the result's verified source handle and compares every
    publication array and the full projection record. Materializers use this
    callback to reject a result object that was altered after computation
    without making the shared schema import a higher application layer.
    """

    if not isinstance(result, ChaserInputProvenanceProxyResult):
        _fail("result must be one ChaserInputProvenanceProxyResult.")
    arrays = _array_values(result)
    frame_values = _exact_dtype(
        arrays["acquisition_frame_index"],
        field="acquisition_frame_index",
        dtype="<i8",
    )
    n_frames = int(frame_values.size)
    if n_frames <= 0 or np.any(np.diff(frame_values) <= 0):
        _fail("acquisition_frame_index must be nonempty and strictly increasing.")
    candidate_offsets = _exact_dtype(
        arrays["candidate_offsets"], field="candidate_offsets", dtype="<i8"
    )
    candidate_count = _exact_dtype(
        arrays["candidate_sample_count"], field="candidate_sample_count", dtype="<i8"
    )
    if candidate_offsets.shape != (n_frames + 1,):
        _fail("candidate_offsets must have one boundary per frame plus one.")
    n_candidates = int(candidate_offsets[-1]) if candidate_offsets.size else -1
    if (
        candidate_offsets[0] != 0
        or n_candidates <= 0
        or np.any(np.diff(candidate_offsets) < 0)
        or np.any(candidate_offsets < 0)
        or candidate_offsets[-1] != n_candidates
    ):
        _fail("candidate_offsets violate nonnegative monotone ragged invariants.")
    if candidate_count.shape != (n_frames,) or not np.array_equal(
        np.diff(candidate_offsets), candidate_count
    ):
        _fail("candidate_sample_count differs from candidate_offsets.")
    if np.any(candidate_count <= 0):
        _fail("Every represented input-acquisition frame must retain a candidate.")
    n_chasers = int(np.asarray(arrays["selected_chaser_index"]).shape[1])
    if n_chasers <= 0:
        _fail("selected_chaser_index must declare a nonempty chaser axis.")
    dimensions = ChaserInputProvenanceProxyDimensions(
        n_frames=n_frames, n_candidates=n_candidates, n_chasers=n_chasers
    )
    expected_shapes: dict[str, tuple[int, ...]] = {
        "candidate_native_sample_row_index": (n_candidates,),
        "candidate_stimulus_frame_num": (n_candidates,),
        "candidate_timestamp_ns_session": (n_candidates,),
        "candidate_source_acquisition_frame_index": (n_candidates,),
        "candidate_complete": (n_candidates,),
        "candidate_reason_code": (n_candidates,),
        "candidate_source_stimulus_run_row_index": (n_candidates, n_chasers),
        "candidate_source_stimulus_source_row_index": (n_candidates, n_chasers),
        "selected": (n_frames,),
        "selection_reason_code": (n_frames,),
        "selected_native_sample_row_index": (n_frames,),
        "selected_stimulus_frame_num": (n_frames,),
        "selected_timestamp_ns_session": (n_frames,),
        "selected_source_stimulus_run_row_index": (n_frames, n_chasers),
        "selected_source_stimulus_source_row_index": (n_frames, n_chasers),
        "selected_chaser_index": (n_frames, n_chasers),
        "selected_chaser_position_xy": (n_frames, n_chasers, 2),
        "selected_chaser_valid": (n_frames, n_chasers),
    }
    expected_dtypes = {
        "candidate_native_sample_row_index": "<i8",
        "candidate_stimulus_frame_num": "<i8",
        "candidate_timestamp_ns_session": "<i8",
        "candidate_source_acquisition_frame_index": "<i8",
        "candidate_complete": "bool",
        "candidate_reason_code": "<U32",
        "candidate_source_stimulus_run_row_index": "<i8",
        "candidate_source_stimulus_source_row_index": "<i8",
        "selected": "bool",
        "selection_reason_code": "<U32",
        "selected_native_sample_row_index": "<i8",
        "selected_stimulus_frame_num": "<i8",
        "selected_timestamp_ns_session": "<i8",
        "selected_source_stimulus_run_row_index": "<i8",
        "selected_source_stimulus_source_row_index": "<i8",
        "selected_chaser_index": "<i2",
        "selected_chaser_position_xy": "<f8",
        "selected_chaser_valid": "bool",
    }
    for name, shape in expected_shapes.items():
        _shape(arrays[name], field=name, shape=shape)
        _exact_dtype(
            arrays[name],
            field=name,
            dtype=expected_dtypes[name],
            allow_text=name.endswith("reason_code"),
        )
    if np.any(arrays["candidate_reason_code"] == "") or np.any(
        ~np.isin(arrays["candidate_reason_code"], list(CANDIDATE_REASON_CODES))
    ):
        _fail("candidate_reason_code contains an unregistered reason.")
    if np.any(~np.isin(arrays["selection_reason_code"], list(SELECTION_REASON_CODES))):
        _fail("selection_reason_code contains an unregistered reason.")
    if np.any(arrays["candidate_native_sample_row_index"] < 0):
        _fail("candidate native sample rows must be nonnegative.")
    if np.any(np.diff(arrays["candidate_native_sample_row_index"]) <= 0):
        _fail("candidate native sample rows must remain strictly ordered and unique.")
    if np.any(
        arrays["candidate_source_acquisition_frame_index"]
        != np.repeat(frame_values, candidate_count)
    ):
        _fail("Candidate source acquisition frames do not match ragged frame groups.")
    if np.any(
        arrays["candidate_complete"]
        != (arrays["candidate_reason_code"] == "complete")
    ):
        _fail("Candidate completeness and reason text disagree.")
    for name in (
        "candidate_source_stimulus_run_row_index",
        "candidate_source_stimulus_source_row_index",
    ):
        if np.any(arrays[name] < 0):
            _fail(f"{name} must preserve nonnegative source-row lineage.")
    if np.unique(arrays["candidate_source_stimulus_run_row_index"]).size != (
        n_candidates * n_chasers
    ):
        _fail("Candidate source run-row lineage must be unique per chaser sample.")
    selected_rows = arrays["selected_native_sample_row_index"]
    selected = arrays["selected"]
    for frame_offset in range(n_frames):
        start = int(candidate_offsets[frame_offset])
        end = int(candidate_offsets[frame_offset + 1])
        row = int(selected_rows[frame_offset])
        if selected[frame_offset]:
            frame_candidate_rows = arrays["candidate_native_sample_row_index"][start:end]
            matches = np.flatnonzero(frame_candidate_rows == row)
            if row < 0 or matches.size != 1:
                _fail("Selected native row is not a candidate for its frame.")
            if arrays["selection_reason_code"][frame_offset] != "selected":
                _fail("Selected frame has a non-selected reason code.")
            candidate_offset = start + int(matches[0])
            for selected_name, candidate_name in (
                ("selected_stimulus_frame_num", "candidate_stimulus_frame_num"),
                (
                    "selected_timestamp_ns_session",
                    "candidate_timestamp_ns_session",
                ),
                (
                    "selected_source_stimulus_run_row_index",
                    "candidate_source_stimulus_run_row_index",
                ),
                (
                    "selected_source_stimulus_source_row_index",
                    "candidate_source_stimulus_source_row_index",
                ),
            ):
                if not np.array_equal(
                    arrays[selected_name][frame_offset],
                    arrays[candidate_name][candidate_offset],
                ):
                    _fail(
                        f"{selected_name} does not come from its selected candidate."
                    )
            if not arrays["candidate_complete"][candidate_offset]:
                _fail("Selected native row is not one complete all-chaser sample.")
            if not np.all(arrays["selected_chaser_valid"][frame_offset]):
                _fail("Selected proxy rows must retain every declared chaser.")
        else:
            if row != -1 or arrays["selection_reason_code"][frame_offset] != "no_complete_chaser_sample":
                _fail("Unselected frame must use the explicit no-complete sentinel.")
            if not np.all(np.isnan(arrays["selected_chaser_position_xy"][frame_offset])):
                _fail("Unselected frame positions must be NaN sentinels.")
            if np.any(arrays["selected_chaser_valid"][frame_offset]):
                _fail("Unselected frame validity must be false.")
            for name in (
                "selected_stimulus_frame_num",
                "selected_timestamp_ns_session",
                "selected_source_stimulus_run_row_index",
                "selected_source_stimulus_source_row_index",
            ):
                if np.any(arrays[name][frame_offset] != -1):
                    _fail(f"Unselected frame {name} must use the -1 sentinel.")
    if np.any(arrays["selected_chaser_valid"] & ~np.isfinite(arrays["selected_chaser_position_xy"]).all(axis=2)):
        _fail("A selected valid chaser position is nonfinite.")
    if np.any(arrays["selected_chaser_index"] < 0) or any(
        not np.array_equal(arrays["selected_chaser_index"][0], row)
        for row in arrays["selected_chaser_index"][1:]
    ):
        _fail("selected_chaser_index does not define one stable chaser axis.")
    record = _validate_projection_record(result)
    if record.get("chaser_count") != n_chasers:
        _fail("Projection chaser_count must equal the explicit chaser axis.")
    if record.get("native_sample_count") != n_candidates:
        _fail("Projection native_sample_count differs from preserved candidates.")
    if record.get("unique_acquisition_frame_count") != n_frames:
        _fail("Projection frame count differs from the frame axis.")
    if record.get("selected_acquisition_frame_count") != int(np.count_nonzero(selected)):
        _fail("Projection selected count differs from the selected axis.")

    if revalidate_source is not None:
        verifier = getattr(result.source_handle, "assert_verified", None)
        if not callable(verifier):
            _fail("Result source handle does not expose assert_verified().")
        try:
            verifier()
            fresh = revalidate_source(result.source_handle)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            _fail(f"Source revalidation failed: {exc}")
        fresh_arrays = _array_values(fresh)
        for name in arrays:
            if name.endswith("reason_code"):
                continue
            if not np.array_equal(arrays[name], fresh_arrays[name], equal_nan=True):
                _fail(f"Result array {name} differs from freshly verified source output.")
        if _mapping_sha256(_strict_record(fresh.acquisition_projection_record, field="fresh projection")) != result.acquisition_projection_record_sha256:
            _fail("Result projection record differs from freshly verified source output.")
    return dimensions, arrays, record


def build_array_declarations(
    result: ChaserInputProvenanceProxyResult,
) -> tuple[ProxyArrayDeclaration, ...]:
    dimensions, arrays, _record = validate_proxy_result(result)
    del dimensions
    axes: dict[str, tuple[str, ...]] = {
        "acquisition_frame_index": (FRAME_AXIS,),
        "candidate_offsets": (FRAME_BOUNDARY_AXIS,),
        "candidate_sample_count": (FRAME_AXIS,),
        **{
            name: (CANDIDATE_AXIS,)
            for name in (
                "candidate_native_sample_row_index",
                "candidate_stimulus_frame_num",
                "candidate_timestamp_ns_session",
                "candidate_source_acquisition_frame_index",
                "candidate_complete",
                "candidate_reason_code",
            )
        },
        **{
            name: (CANDIDATE_AXIS, CHASER_AXIS)
            for name in (
                "candidate_source_stimulus_run_row_index",
                "candidate_source_stimulus_source_row_index",
            )
        },
        **{
            name: (FRAME_AXIS,)
            for name in ("selected", "selection_reason_code", "selected_native_sample_row_index", "selected_stimulus_frame_num", "selected_timestamp_ns_session")
        },
        **{
            name: (FRAME_AXIS, CHASER_AXIS)
            for name in (
                "selected_source_stimulus_run_row_index",
                "selected_source_stimulus_source_row_index",
                "selected_chaser_index",
                "selected_chaser_valid",
            )
        },
        "selected_chaser_position_xy": (FRAME_AXIS, CHASER_AXIS, "coordinate"),
    }
    declarations: list[ProxyArrayDeclaration] = []
    for name, values in arrays.items():
        declarations.append(
            ProxyArrayDeclaration(
                path=name,
                dtype=values.dtype.str,
                shape=tuple(int(value) for value in values.shape),
                axes=axes[name],
                content_sha256=array_values_sha256(values),
                description=_ARRAY_DESCRIPTIONS[name],
            )
        )
    return tuple(declarations)


def encode_reason_codes(result: ChaserInputProvenanceProxyResult) -> dict[str, np.ndarray]:
    """Return a copied array map with all reason strings converted to uint8."""

    _dimensions, arrays, _record = validate_proxy_result(result)
    encoded = {name: np.array(value, copy=True, order="C") for name, value in arrays.items()}
    encoded["candidate_reason_code"] = np.asarray(
        [CANDIDATE_REASON_CODES[str(value)] for value in arrays["candidate_reason_code"]],
        dtype=np.uint8,
    )
    encoded["selection_reason_code"] = np.asarray(
        [SELECTION_REASON_CODES[str(value)] for value in arrays["selection_reason_code"]],
        dtype=np.uint8,
    )
    for value in encoded.values():
        value.setflags(write=False)
    return encoded


def build_publication_manifest(
    result: ChaserInputProvenanceProxyResult,
) -> dict[str, Any]:
    """Build the complete metadata manifest from the current pure result."""

    dimensions, arrays, record = validate_proxy_result(result)
    encoded = encode_reason_codes(result)
    declarations = []
    for declaration in build_array_declarations(result):
        values = encoded[declaration.path]
        declarations.append(
            ProxyArrayDeclaration(
                path=declaration.path,
                dtype=values.dtype.str,
                shape=declaration.shape,
                axes=declaration.axes,
                content_sha256=array_values_sha256(values),
                description=declaration.description,
            ).as_manifest()
        )
    manifest = {
        "schema_id": CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID,
        "schema_version": CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION,
        "layout": CHASER_INPUT_PROVENANCE_PROXY_LAYOUT,
        "parent_path": CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH,
        "selector_eligible": False,
        "selection": "none",
        "dimensions": dimensions.as_manifest(),
        "axis_contract": {
            "frame": "unique_input_acquisition_frame_index",
            "candidate": "ragged_native_stimulus_sample_evidence",
            "chaser": "declared_all_chaser_identity_axis",
            "frame_boundary": "candidate_offsets_boundaries",
        },
        "acquisition_projection_record": _plain(record),
        "acquisition_projection_record_sha256": result.acquisition_projection_record_sha256,
        "source": {
            "recording_id": result.recording_id,
            "source_authority_id": result.source_authority_id,
            "source_authority_digest": result.source_authority_digest,
            "source_manifest_sha256": result.source_manifest_sha256,
            "source_verification_digest": result.source_verification_digest,
            "source_run_path": result.source_run_path,
        },
        "reason_code_registry": {
            "encoding": "uint8",
            "codes": {str(code): reason for code, reason in REASON_CODE_REGISTRY.items()},
            "selection_arrays": ["selection_reason_code"],
            "candidate_arrays": ["candidate_reason_code"],
        },
        "denominator_policy": record["behavioral_denominator"],
        "no_carry_policy": record["missing_frame_rule"],
        "native_sample_policy": {
            "native_sample_rows_preserved": record.get("native_sample_rows_preserved"),
            "native_sample_axis": record.get("native_sample_axis"),
            "complete_sample_rule": record.get("complete_sample_rule"),
        },
        "array_declarations": declarations,
        "row_evidence_arrays_only": True,
    }
    return _strict_record(manifest, field="publication_manifest")


def validate_publication_manifest(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Validate a persisted manifest and, when supplied, its decoded arrays."""

    normalized = _strict_record(manifest, field="publication_manifest")
    if normalized.get("schema_id") != CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID:
        _fail("Manifest schema_id is invalid.")
    if normalized.get("schema_version") != CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION:
        _fail("Manifest schema_version is unsupported.")
    if normalized.get("layout") != CHASER_INPUT_PROVENANCE_PROXY_LAYOUT:
        _fail("Manifest layout is invalid.")
    if normalized.get("parent_path") != CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH:
        _fail("Manifest parent_path is invalid.")
    if normalized.get("selector_eligible") is not False or normalized.get("selection") != "none":
        _fail("Manifest must be explicitly selector-ineligible and unselected.")
    record = normalized.get("acquisition_projection_record")
    if not isinstance(record, Mapping):
        _fail("Manifest lacks acquisition_projection_record.")
    if record.get("schema_id") != PROJECTION_RECORD_SCHEMA_ID:
        _fail("Manifest projection record schema is invalid.")
    record_digest = _require_digest(
        normalized.get("acquisition_projection_record_sha256"),
        field="manifest.acquisition_projection_record_sha256",
    )
    if _mapping_sha256(record) != record_digest:
        _fail("Manifest acquisition projection digest is stale.")
    expected_record_values = {
        "policy_id": PROXY_POLICY_ID,
        "temporal_alignment_requirement": TEMPORAL_ALIGNMENT_REQUIREMENT,
        "temporal_alignment_class": TEMPORAL_ALIGNMENT_CLASS,
        "physical_presentation_verified": False,
        "presentation_timestamp_available": False,
        "camera_presentation_clock_transform_available": False,
        "camera_exposure_reference": CAMERA_EXPOSURE_REFERENCE,
        "scientific_use_class": SCIENTIFIC_USE_CLASS,
        "behavioral_denominator": BEHAVIORAL_DENOMINATOR,
        "native_sample_rows_preserved": True,
        "missing_frame_rule": "no_carry_forward",
    }
    for field, expected in expected_record_values.items():
        if record.get(field) != expected:
            _fail(f"Manifest projection record has invalid {field}.")
    dimensions = normalized.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        FRAME_AXIS,
        CANDIDATE_AXIS,
        CHASER_AXIS,
        FRAME_BOUNDARY_AXIS,
    }:
        _fail("Manifest dimensions are missing or inexact.")
    if any(type(value) is not int or value <= 0 for value in dimensions.values()):
        _fail("Manifest dimensions must be positive exact integers.")
    if dimensions[FRAME_BOUNDARY_AXIS] != dimensions[FRAME_AXIS] + 1:
        _fail("Manifest frame-boundary dimension is contradictory.")
    if record.get("unique_acquisition_frame_count") != dimensions[FRAME_AXIS]:
        _fail("Manifest projection/frame dimensions disagree.")
    if record.get("native_sample_count") != dimensions[CANDIDATE_AXIS]:
        _fail("Manifest projection/candidate dimensions disagree.")
    if record.get("chaser_count") != dimensions[CHASER_AXIS]:
        _fail("Manifest projection/chaser dimensions disagree.")
    source = normalized.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "recording_id",
        "source_authority_id",
        "source_authority_digest",
        "source_manifest_sha256",
        "source_verification_digest",
        "source_run_path",
    }:
        _fail("Manifest source binding is missing or inexact.")
    for field in source:
        record_field = field if field != "recording_id" else "recording_id"
        if source[field] != record.get(record_field):
            _fail(f"Manifest source {field} differs from the projection record.")
    for field in (
        "source_authority_digest",
        "source_manifest_sha256",
        "source_verification_digest",
    ):
        _require_digest(source[field], field=f"manifest.source.{field}")
    if normalized.get("denominator_policy") != BEHAVIORAL_DENOMINATOR:
        _fail("Manifest denominator policy is invalid.")
    if normalized.get("no_carry_policy") != "no_carry_forward":
        _fail("Manifest no-carry policy is invalid.")
    expected_reason_registry = {
        "encoding": "uint8",
        "codes": {str(code): reason for code, reason in REASON_CODE_REGISTRY.items()},
        "selection_arrays": ["selection_reason_code"],
        "candidate_arrays": ["candidate_reason_code"],
    }
    if normalized.get("reason_code_registry") != expected_reason_registry:
        _fail("Manifest reason-code registry is invalid.")
    declarations = normalized.get("array_declarations")
    if not isinstance(declarations, list) or not declarations:
        _fail("Manifest array_declarations must be a nonempty list.")
    paths: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            _fail("Manifest array declaration must be an object.")
        path = declaration.get("path")
        if type(path) is not str or not path or path in paths or "/" in path:
            _fail("Manifest array paths must be unique one-component names.")
        paths.add(path)
        _require_digest(declaration.get("content_sha256"), field=f"{path}.content_sha256")
        if declaration.get("row_evidence_only") is not True:
            _fail(f"{path} is not marked row-evidence-only.")
        if not isinstance(declaration.get("shape"), list) or not isinstance(declaration.get("axes"), list):
            _fail(f"{path} shape/axes declarations are invalid.")
        if (
            declaration.get("array_schema_id")
            != CHASER_INPUT_PROVENANCE_PROXY_ARRAY_SCHEMA_ID
            or declaration.get("array_schema_version")
            != CHASER_INPUT_PROVENANCE_PROXY_ARRAY_SCHEMA_VERSION
        ):
            _fail(f"{path} array schema identity is invalid.")
    if paths != set(_ARRAY_DESCRIPTIONS):
        _fail("Manifest array declarations do not cover the exact typed contract.")
    if arrays is not None:
        if set(arrays) != paths:
            _fail("Decoded array names differ from manifest declarations.")
        for declaration in declarations:
            name = str(declaration["path"])
            value = np.asarray(arrays[name])
            if value.dtype.hasobject or value.dtype.kind in {"U", "S"}:
                _fail(f"Decoded array {name} is object/string typed.")
            if value.dtype.str != declaration.get("dtype"):
                _fail(f"Decoded array {name} dtype differs from manifest.")
            if list(value.shape) != declaration.get("shape"):
                _fail(f"Decoded array {name} shape differs from manifest.")
            if array_values_sha256(value) != declaration.get("content_sha256"):
                _fail(f"Decoded array {name} content digest differs from manifest.")
    return normalized


__all__ = [
    "CANDIDATE_AXIS",
    "CANDIDATE_REASON_CODES",
    "CHASER_AXIS",
    "CHASER_INPUT_PROVENANCE_PROXY_ARRAY_SCHEMA_ID",
    "CHASER_INPUT_PROVENANCE_PROXY_ARRAY_SCHEMA_VERSION",
    "CHASER_INPUT_PROVENANCE_PROXY_LAYOUT",
    "CHASER_INPUT_PROVENANCE_PROXY_MATERIALIZATION_SCHEMA_ID",
    "CHASER_INPUT_PROVENANCE_PROXY_PARENT_PATH",
    "CHASER_INPUT_PROVENANCE_PROXY_PUBLISH_SCHEMA_ID",
    "CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_ID",
    "CHASER_INPUT_PROVENANCE_PROXY_SCHEMA_VERSION",
    "ChaserInputProvenanceProxyDimensions",
    "ChaserInputProvenanceProxySchemaError",
    "ProxyArrayDeclaration",
    "REASON_CODE_REGISTRY",
    "SELECTION_REASON_CODES",
    "FRAME_AXIS",
    "build_array_declarations",
    "build_publication_manifest",
    "encode_reason_codes",
    "validate_proxy_result",
    "validate_publication_manifest",
]
