"""Fail-closed visualization binding for protocol-authored chaser appearance."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.chaser_behavior import resolve_configured_chaser_behaviors
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root

APPEARANCE_SCHEMA_ID = "palette.visualization.chaser_appearance_projection"
APPEARANCE_SCHEMA_VERSION = 1
APPEARANCE_POLICY_ID = "protocol_rgba_independent_behavior_role_glyph_v1"
OCCURRENCE_SCHEMA_ID = "palette.chaser_relative_frame.chaser_occurrence_binding"
OCCURRENCE_POLICY_ID = "native_sample_declared_chaser_axis_v1"
CHASER_IDENTITY_POLICY_ID = "stimulus_run_scoped_chaser_index_v1"

PLOTLY_ROLE_SYMBOLS: Mapping[str, str] = MappingProxyType(
    {
        "aggressive": "star",
        "random_non_chasing": "diamond",
        "inert": "circle",
        "unknown": "x",
    }
)
MATPLOTLIB_ROLE_MARKERS: Mapping[str, str] = MappingProxyType(
    {
        "aggressive": "*",
        "random_non_chasing": "D",
        "inert": "o",
        "unknown": "X",
    }
)


class ChaserAppearanceProjectionError(ValueError):
    """Sealed chaser appearance evidence is incomplete or inconsistent."""


def _fail(message: str) -> None:
    raise ChaserAppearanceProjectionError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"{field} must be one non-empty string.")
    return value.strip()


def _digest(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return text


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        _fail(f"{field} must be one integer.")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif type(value) is str and value.isdigit():
        number = int(value)
    else:
        _fail(f"{field} must be one integer.")
    if number < minimum:
        _fail(f"{field} must be an integer greater than or equal to {minimum}.")
    return number


def _unit_channel(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(f"{field} must be one finite unit color channel.")
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        _fail(f"{field} must be one finite unit color channel.")
    return number


def _u8(channel: float) -> int:
    return int(round(float(channel) * 255.0))


def _experimental_color_hex(rgba: Sequence[float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(_u8(value) for value in rgba[:3]))


def _experimental_color_css(rgba: Sequence[float]) -> str:
    red, green, blue = (_u8(value) for value in rgba[:3])
    return f"rgba({red},{green},{blue},{float(rgba[3]):.6g})"


def _contrast_outline(rgba: Sequence[float]) -> str:
    red, green, blue = (float(value) for value in rgba[:3])
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#ffffff" if luminance < 0.45 else "#111111"


@dataclass(frozen=True, slots=True)
class ChaserAppearance:
    """Independent experimental-color, semantic-role, and identity channels."""

    identity_code: int
    chaser_index: int
    identity: str
    behavior_role_code: int
    behavior_role: str
    experimental_color_rgba: tuple[float, float, float, float]
    experimental_color_hex: str
    experimental_color_css: str
    plotly_role_symbol: str
    matplotlib_role_marker: str
    contrast_outline_hex: str

    def provenance_record(self) -> dict[str, Any]:
        return {
            "identity_code": self.identity_code,
            "chaser_index": self.chaser_index,
            "identity": self.identity,
            "behavior_role_code": self.behavior_role_code,
            "behavior_role": self.behavior_role,
            "experimental_color_rgba": list(self.experimental_color_rgba),
            "experimental_color_hex": self.experimental_color_hex,
            "plotly_role_symbol": self.plotly_role_symbol,
            "matplotlib_role_marker": self.matplotlib_role_marker,
            "contrast_outline_hex": self.contrast_outline_hex,
        }


@dataclass(frozen=True, slots=True)
class ChaserAppearanceProjection:
    """Digest-bound display projection of the exact protocol chaser list."""

    recording_id: str
    source_stimulus_run_path: str
    source_protocol_sha256: str
    occurrence_binding_sha256: str
    appearances: tuple[ChaserAppearance, ...]
    projection_sha256: str

    def by_identity_code(self) -> Mapping[int, ChaserAppearance]:
        return MappingProxyType(
            {appearance.identity_code: appearance for appearance in self.appearances}
        )

    def provenance_record(self) -> dict[str, Any]:
        return {
            "schema_id": APPEARANCE_SCHEMA_ID,
            "schema_version": APPEARANCE_SCHEMA_VERSION,
            "appearance_policy_id": APPEARANCE_POLICY_ID,
            "recording_id": self.recording_id,
            "source_stimulus_run_path": self.source_stimulus_run_path,
            "source_protocol_sha256": self.source_protocol_sha256,
            "occurrence_binding_sha256": self.occurrence_binding_sha256,
            "color_semantics": "experimental_protocol_rgba",
            "role_semantics": "independent_marker_shape_and_text",
            "color_role_independence": True,
            "chasers": [item.provenance_record() for item in self.appearances],
            "projection_sha256": self.projection_sha256,
        }


def _protocol_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        payload = _plain(value)
    elif type(value) is str:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            _fail(f"Bound stimulus protocol_json is invalid JSON: {exc}.")
    else:
        _fail("Bound stimulus run lacks one readable protocol_json payload.")
    if not isinstance(payload, Mapping):
        _fail("Bound stimulus protocol_json must decode to one object.")
    return payload


def _configured_chaser_records(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        _fail("Bound stimulus protocol lacks steps[].")
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        parameters = step.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        values = parameters.get("chasers")
        if isinstance(values, list):
            if any(not isinstance(value, Mapping) for value in values):
                _fail("Bound stimulus protocol chasers must each be one object.")
            return tuple(values)  # type: ignore[return-value]
    _fail("Bound stimulus protocol lacks steps[].parameters.chasers[].")


def _code_vector(
    values: Sequence[int] | np.ndarray, *, field: str, size: int
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (size,) or array.dtype.kind not in "iu":
        _fail(f"{field} must preserve the exact one-value-per-chaser column axis.")
    return array.astype(np.int64, copy=False)


def resolve_chaser_appearance_projection(
    *,
    relative_manifest: Mapping[str, Any],
    protocol_payload: Mapping[str, Any],
    identity_code_by_column: Sequence[int] | np.ndarray,
    behavior_role_code_by_column: Sequence[int] | np.ndarray,
    expected_recording_id: str | None = None,
) -> ChaserAppearanceProjection:
    """Resolve protocol colors only after exact occurrence and role equality checks."""

    manifest = _mapping(relative_manifest, field="relative manifest")
    recording_id = _text(manifest.get("recording_id"), field="recording_id")
    if expected_recording_id is not None and recording_id != _text(
        expected_recording_id, field="expected_recording_id"
    ):
        _fail("Relative manifest belongs to another recording.")
    dimensions = _mapping(manifest.get("dimensions"), field="dimensions")
    n_chasers = _integer(dimensions.get("n_chasers"), field="n_chasers", minimum=1)
    identity_codes = _code_vector(
        identity_code_by_column, field="identity_code_by_column", size=n_chasers
    )
    role_codes = _code_vector(
        behavior_role_code_by_column,
        field="behavior_role_code_by_column",
        size=n_chasers,
    )

    context = _mapping(manifest.get("context"), field="context")
    envelope = _mapping(context.get("chaser_occurrence"), field="chaser_occurrence")
    occurrence = _mapping(envelope.get("record"), field="chaser_occurrence.record")
    occurrence_sha256 = _digest(
        envelope.get("sha256"), field="chaser_occurrence.sha256"
    )
    if canonical_json_sha256(_plain(occurrence)) != occurrence_sha256:
        _fail("Chaser occurrence envelope digest is stale.")
    if (
        occurrence.get("schema_id") != OCCURRENCE_SCHEMA_ID
        or occurrence.get("schema_version") != 1
        or occurrence.get("recording_id") != recording_id
        or occurrence.get("occurrence_policy_id") != OCCURRENCE_POLICY_ID
        or occurrence.get("chaser_identity_policy_id") != CHASER_IDENTITY_POLICY_ID
    ):
        _fail(
            "Chaser occurrence record has an unsupported schema, policy, or recording."
        )
    source_path = _text(
        occurrence.get("source_stimulus_run_path"),
        field="source_stimulus_run_path",
    )
    prefix = "analysis/stimulus_runs/"
    run_name = source_path.removeprefix(prefix)
    if (
        not source_path.startswith(prefix)
        or not run_name
        or "/" in run_name
        or run_name.casefold()
        in {"latest", "authoritative", "authoritative_run", "selected"}
    ):
        _fail("Chaser occurrence does not name one exact stimulus-run child.")
    protocol_sha256 = _digest(
        occurrence.get("source_protocol_sha256"), field="source_protocol_sha256"
    )
    payload = _protocol_payload(protocol_payload)
    if canonical_json_sha256(_plain(payload)) != protocol_sha256:
        _fail("Bound stimulus protocol payload differs from the occurrence digest.")

    try:
        configured = resolve_configured_chaser_behaviors(payload)
    except ValueError as exc:
        _fail(f"Configured chaser behavior cannot be resolved: {exc}.")
    raw_records = _configured_chaser_records(payload)
    if len(configured) != n_chasers or len(raw_records) != n_chasers:
        _fail("Protocol chaser cardinality differs from the relative-frame axis.")
    configured_by_index = {item.chaser_index: item for item in configured}
    if len(configured_by_index) != n_chasers:
        _fail("Protocol chaser indices are not unique.")
    raw_by_index: dict[int, Mapping[str, Any]] = {}
    for fallback_index, record in enumerate(raw_records):
        raw_index = record.get("chaser_index", fallback_index)
        index = _integer(raw_index, field="protocol chaser_index", minimum=0)
        if index in raw_by_index:
            _fail("Protocol chaser indices are not unique.")
        raw_by_index[index] = record
    if set(raw_by_index) != set(configured_by_index):
        _fail("Protocol color rows differ from configured chaser identities.")

    registries = _mapping(
        manifest.get("identity_registries"), field="identity_registries"
    )
    identities = _mapping(registries.get("chaser"), field="identity_registries.chaser")
    roles = _mapping(
        registries.get("behavior_role"), field="identity_registries.behavior_role"
    )
    expected_identity_keys = {str(index) for index in range(1, n_chasers + 1)}
    if set(identities) != expected_identity_keys:
        _fail("Chaser identity registry does not match the exact chaser axis.")
    role_by_code = {
        _integer(code, field="behavior-role registry code", minimum=1): _text(
            label, field="behavior-role registry label"
        )
        for code, label in roles.items()
    }
    if not role_by_code or len(set(role_by_code.values())) != len(role_by_code):
        _fail("Behavior-role registry labels must be unique.")

    rows = occurrence.get("chasers")
    if not isinstance(rows, (list, tuple)) or len(rows) != n_chasers:
        _fail("Chaser occurrence rowset differs from the exact chaser axis.")
    occurrence_by_identity: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record = _mapping(row, field="chaser occurrence row")
        identity = _text(record.get("identity"), field="occurrence identity")
        if identity in occurrence_by_identity:
            _fail("Chaser occurrence identities are duplicated.")
        occurrence_by_identity[identity] = record

    appearances: list[ChaserAppearance] = []
    for column, (identity_code, role_code) in enumerate(
        zip(identity_codes.tolist(), role_codes.tolist(), strict=True)
    ):
        if identity_code != column + 1:
            _fail("Chaser identity codes do not preserve the exact column axis.")
        identity = _text(
            identities.get(str(identity_code)), field="chaser identity registry value"
        )
        occurrence_row = occurrence_by_identity.get(identity)
        if occurrence_row is None:
            _fail("Chaser identity registry is absent from the occurrence record.")
        index = _integer(
            occurrence_row.get("chaser_index"),
            field="occurrence chaser_index",
            minimum=0,
        )
        if identity != f"{run_name}:chaser_index:{index}":
            _fail("Chaser identity does not follow its declared stimulus-run policy.")
        behavior = configured_by_index.get(index)
        raw_record = raw_by_index.get(index)
        if behavior is None or raw_record is None:
            _fail("Occurrence chaser index is absent from the bound protocol.")
        role = _text(
            occurrence_row.get("behavior_role"), field="occurrence behavior_role"
        )
        declared_role = role_by_code.get(role_code)
        if (
            declared_role is None
            or declared_role != role
            or behavior.behavior_class != role
        ):
            _fail("Protocol, occurrence, and relative-frame behavior roles disagree.")
        color_fields = ("color_r", "color_g", "color_b", "color_a")
        if any(field not in raw_record for field in color_fields):
            _fail(
                "Protocol chaser lacks an explicit RGBA color; fallback is prohibited."
            )
        rgba = tuple(
            _unit_channel(raw_record[field], field=f"chaser {index} {field}")
            for field in color_fields
        )
        if tuple(float(value) for value in behavior.raw_color_rgba) != rgba:
            _fail("Resolved protocol color differs from its explicit RGBA record.")
        symbol = PLOTLY_ROLE_SYMBOLS.get(role)
        marker = MATPLOTLIB_ROLE_MARKERS.get(role)
        if symbol is None or marker is None:
            _fail(f"Behavior role {role!r} has no declared visualization glyph.")
        appearances.append(
            ChaserAppearance(
                identity_code=identity_code,
                chaser_index=index,
                identity=identity,
                behavior_role_code=role_code,
                behavior_role=role,
                experimental_color_rgba=rgba,  # type: ignore[arg-type]
                experimental_color_hex=_experimental_color_hex(rgba),
                experimental_color_css=_experimental_color_css(rgba),
                plotly_role_symbol=symbol,
                matplotlib_role_marker=marker,
                contrast_outline_hex=_contrast_outline(rgba),
            )
        )
    if set(occurrence_by_identity) != {item.identity for item in appearances}:
        _fail("Occurrence record contains a chaser outside the exact display axis.")

    body = {
        "schema_id": APPEARANCE_SCHEMA_ID,
        "schema_version": APPEARANCE_SCHEMA_VERSION,
        "appearance_policy_id": APPEARANCE_POLICY_ID,
        "recording_id": recording_id,
        "source_stimulus_run_path": source_path,
        "source_protocol_sha256": protocol_sha256,
        "occurrence_binding_sha256": occurrence_sha256,
        "color_semantics": "experimental_protocol_rgba",
        "role_semantics": "independent_marker_shape_and_text",
        "color_role_independence": True,
        "chasers": [item.provenance_record() for item in appearances],
    }
    return ChaserAppearanceProjection(
        recording_id=recording_id,
        source_stimulus_run_path=source_path,
        source_protocol_sha256=protocol_sha256,
        occurrence_binding_sha256=occurrence_sha256,
        appearances=tuple(appearances),
        projection_sha256=canonical_json_sha256(body),
    )


def load_chaser_appearance_projection(
    analysis_zarr: str | Path,
    *,
    relative_manifest: Mapping[str, Any],
    identity_code_by_column: Sequence[int] | np.ndarray,
    behavior_role_code_by_column: Sequence[int] | np.ndarray,
    expected_recording_id: str | None = None,
) -> ChaserAppearanceProjection:
    """Read only the digest-bound protocol attribute through published metadata."""

    manifest = _mapping(relative_manifest, field="relative manifest")
    context = _mapping(manifest.get("context"), field="context")
    envelope = _mapping(context.get("chaser_occurrence"), field="chaser_occurrence")
    occurrence = _mapping(envelope.get("record"), field="chaser_occurrence.record")
    source_path = _text(
        occurrence.get("source_stimulus_run_path"),
        field="source_stimulus_run_path",
    )
    root = open_zarr_root(
        Path(analysis_zarr).expanduser().resolve(),
        mode="r",
        use_consolidated=True,
    )
    try:
        stimulus = root[source_path]
    except KeyError:
        _fail("Bound exact stimulus run is absent from consolidated metadata.")
    return resolve_chaser_appearance_projection(
        relative_manifest=manifest,
        protocol_payload=_protocol_payload(stimulus.attrs.get("protocol_json")),
        identity_code_by_column=identity_code_by_column,
        behavior_role_code_by_column=behavior_role_code_by_column,
        expected_recording_id=expected_recording_id,
    )


__all__ = [
    "APPEARANCE_POLICY_ID",
    "APPEARANCE_SCHEMA_ID",
    "APPEARANCE_SCHEMA_VERSION",
    "MATPLOTLIB_ROLE_MARKERS",
    "PLOTLY_ROLE_SYMBOLS",
    "ChaserAppearance",
    "ChaserAppearanceProjection",
    "ChaserAppearanceProjectionError",
    "load_chaser_appearance_projection",
    "resolve_chaser_appearance_projection",
]
