"""Validated Citrus protocol semantic identity and ordered-step recipe.

The recording filename and Palette stimulus-run name are recording/run
locators.  They are not protocol identities.  Modern Citrus recordings carry
a protocol-snapshot contract that provides that identity:

``protocol_semantic_hash``
    SHA-256 of the exact UTF-8 bytes in ``protocol_semantic_json``.
``protocol_semantic_json``
    Calibration-independent semantic protocol identity.
``protocol_trial_index_json``
    Ordered, analysis-facing step and trial metadata bound to the semantic
    hash.
``protocol_trial_index_hash``
    Present and required for Citrus snapshot v2.  It authenticates the exact
    UTF-8 bytes in ``protocol_trial_index_json``.  Citrus v1 does not provide
    this dataset, so Palette computes and explicitly labels a local digest for
    that compatibility contract.

This module validates those three values as one indivisible contract.  It does
not derive a replacement semantic hash from ``protocol_definition_json`` and
does not infer a display context from a recording or step name.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np

from fisheye.shared.citrus_enums import STIMULUS_MODE


PROTOCOL_SEMANTIC_HASH_H5_PATH = "/protocol_snapshot/protocol_semantic_hash"
PROTOCOL_SEMANTIC_JSON_H5_PATH = "/protocol_snapshot/protocol_semantic_json"
PROTOCOL_TRIAL_INDEX_JSON_H5_PATH = (
    "/protocol_snapshot/protocol_trial_index_json"
)
PROTOCOL_TRIAL_INDEX_HASH_H5_PATH = (
    "/protocol_snapshot/protocol_trial_index_hash"
)

PROTOCOL_SEMANTIC_SCHEMA_ID = "citrus.protocol.semantic"
PROTOCOL_SEMANTIC_SCHEMA_VERSION = 1
PROTOCOL_SEMANTIC_NORMALIZATION_POLICY = "citrus.protocol.semantic.v1"
PROTOCOL_TRIAL_INDEX_SCHEMA_ID = "citrus.protocol.trial_index"
PROTOCOL_TRIAL_INDEX_SCHEMA_VERSIONS = frozenset({1, 2})
PROTOCOL_TRIAL_INDEX_NORMALIZATION_POLICIES = MappingProxyType(
    {
        1: "citrus.protocol.trial_index.v1",
        2: "citrus.protocol.trial_index.v2",
    }
)

PROTOCOL_SNAPSHOT_SCHEMA_ID = "citrus.protocol.snapshot"
PROTOCOL_SNAPSHOT_SCHEMA_VERSION = 2
PROTOCOL_SNAPSHOT_POLICY_ID = "citrus.protocol.snapshot.v2"
PROTOCOL_SNAPSHOT_VALID_STATUS = "valid"

TRIAL_INDEX_INTEGRITY_LOCAL = "palette_computed_not_producer_asserted"
TRIAL_INDEX_INTEGRITY_PRODUCER = "producer_asserted_exact_bytes"

PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_ID = "palette.stimulus.protocol_snapshot.v1"
PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_VERSION = 1
PALETTE_PROTOCOL_RECIPE_SCHEMA_ID = "palette.stimulus.protocol_recipe.v1"
PALETTE_PROTOCOL_RECIPE_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^sha256:([0-9a-f]{64})$")


class ProtocolSemanticContractError(ValueError):
    """Raised when a producer semantic protocol contract is incomplete or stale."""


def _decode_scalar_text(value: Any, *, name: str) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    elif isinstance(value, str):
        text = value
    elif hasattr(value, "item"):
        return _decode_scalar_text(value.item(), name=name)
    else:
        raise ProtocolSemanticContractError(f"{name} is not UTF-8 scalar text.")
    if not text:
        raise ProtocolSemanticContractError(f"{name} is empty.")
    return text


def _parse_object(text: str, *, name: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolSemanticContractError(
                    f"{name} contains duplicate JSON key {key!r}."
                )
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ProtocolSemanticContractError(f"{name} is not valid JSON.") from exc
    if not isinstance(value, Mapping):
        raise ProtocolSemanticContractError(f"{name} must contain one JSON object.")
    return value


def _require_exact(value: object, expected: object, *, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ProtocolSemanticContractError(
            f"{name} must be {expected!r}, got {value!r}."
        )


def _require_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ProtocolSemanticContractError(f"{name} must be one exact integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ProtocolSemanticContractError(
            f"{name} must be greater than or equal to {minimum}."
        )
    return result


def _require_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolSemanticContractError(f"{name} must be one JSON number.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolSemanticContractError(f"{name} must be finite.")
    return result


def _scaled_number(value: object, *, name: str) -> float:
    """Decode Citrus v1 scaled numbers without accepting booleans.

    Citrus deliberately serializes decimal scales such as ``"1e-3"`` as
    strings while serializing the scaled integer value as a JSON number.
    """

    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ProtocolSemanticContractError(
            f"{name} must be one finite scaled-number component."
        )
    if isinstance(value, str) and (not value or value != value.strip()):
        raise ProtocolSemanticContractError(
            f"{name} must be one canonical nonempty decimal string."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolSemanticContractError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ProtocolSemanticContractError(f"{name} must be finite.")
    return result


def _duration_seconds(value: object, *, name: str) -> float:
    if not isinstance(value, Mapping):
        raise ProtocolSemanticContractError(f"{name} must be a scaled duration object.")
    _require_exact(value.get("unit"), "s", name=f"{name}.unit")
    scale = _scaled_number(value.get("scale"), name=f"{name}.scale")
    raw = _scaled_number(value.get("value"), name=f"{name}.value")
    seconds = scale * raw
    if not math.isfinite(seconds) or seconds < 0:
        raise ProtocolSemanticContractError(
            f"{name} must resolve to a finite non-negative duration."
        )
    return seconds


def _optional_rgba8(features: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    color = features.get("resolved_color")
    if color is None:
        return None
    if not isinstance(color, Mapping):
        raise ProtocolSemanticContractError(
            "protocol trial-index resolved_color must be one object."
        )
    _require_exact(
        color.get("color_space"),
        "srgb",
        name="protocol trial-index resolved_color.color_space",
    )
    rgba = color.get("rgba8")
    if not isinstance(rgba, list) or len(rgba) != 4:
        raise ProtocolSemanticContractError(
            "protocol trial-index resolved_color.rgba8 must contain four uint8 values."
        )
    if any(type(value) is not int or not 0 <= value <= 255 for value in rgba):
        raise ProtocolSemanticContractError(
            "protocol trial-index resolved_color.rgba8 must contain four uint8 values."
        )
    return tuple(rgba)  # type: ignore[return-value]


def _display_context(
    *,
    stimulus_family: str,
    stimulus_mode: str,
    features: Mapping[str, Any],
) -> str:
    rgba8 = _optional_rgba8(features)
    color_name_raw = features.get("color_name")
    if color_name_raw is None:
        color_name = None
    elif (
        type(color_name_raw) is not str
        or not color_name_raw
        or color_name_raw != color_name_raw.strip()
        or color_name_raw != color_name_raw.lower()
    ):
        raise ProtocolSemanticContractError(
            "protocol trial-index color_name must be one canonical lowercase string."
        )
    else:
        color_name = color_name_raw
    if color_name == "black" and rgba8 is not None and rgba8 != (0, 0, 0, 255):
        raise ProtocolSemanticContractError(
            "protocol trial-index black name contradicts resolved_color.rgba8."
        )
    if color_name not in (None, "black") and rgba8 == (0, 0, 0, 255):
        raise ProtocolSemanticContractError(
            "protocol trial-index resolved black contradicts color_name."
        )
    if (
        stimulus_family == "solid_color"
        and stimulus_mode == "SOLID_BLACK"
        and rgba8 == (0, 0, 0, 255)
        and color_name in (None, "black")
    ):
        return "solid_black"
    if stimulus_family == "chaser" or stimulus_mode == "CHASER":
        return "chaser"
    return "other"


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class ProtocolStepIdentity:
    """One producer-authored step in exact recipe order."""

    step_index: int
    stimulus_mode_id: int
    stimulus_mode: str
    stimulus_family: str
    duration_s: float
    index_status: str
    display_context: str
    resolved_color_rgba8: tuple[int, int, int, int] | None

    def to_record(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "stimulus_mode_id": self.stimulus_mode_id,
            "stimulus_mode": self.stimulus_mode,
            "stimulus_family": self.stimulus_family,
            "duration_s": self.duration_s,
            "index_status": self.index_status,
            "display_context": self.display_context,
            "resolved_color_rgba8": (
                list(self.resolved_color_rgba8)
                if self.resolved_color_rgba8 is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ProtocolSemanticSnapshot:
    """Verified producer semantic identity plus ordered analysis recipe."""

    semantic_hash: str
    semantic_json: str
    trial_index_json: str
    trial_index_sha256: str
    trial_index_schema_version: int
    trial_index_integrity_status: str
    snapshot_schema_version: int
    snapshot_policy_id: str
    steps: tuple[ProtocolStepIdentity, ...]
    semantic_payload: Mapping[str, Any]
    trial_index_payload: Mapping[str, Any]

    @property
    def mode_sequence(self) -> tuple[str, ...]:
        return tuple(step.stimulus_mode for step in self.steps)

    @property
    def recipe_label(self) -> str:
        return " -> ".join(self.mode_sequence)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def recipe_record(self) -> dict[str, Any]:
        return {
            "schema_id": PALETTE_PROTOCOL_RECIPE_SCHEMA_ID,
            "schema_version": PALETTE_PROTOCOL_RECIPE_SCHEMA_VERSION,
            "protocol_semantic_hash": self.semantic_hash,
            "step_count": self.step_count,
            "mode_sequence": list(self.mode_sequence),
            "display_label": self.recipe_label,
            "steps": [step.to_record() for step in self.steps],
        }


def validate_protocol_semantic_snapshot(
    *,
    semantic_hash: str,
    semantic_json: str,
    trial_index_json: str,
    trial_index_hash: str | None = None,
    snapshot_schema_version: int = 1,
    snapshot_policy_id: str = "citrus.protocol.snapshot.legacy_v1",
) -> ProtocolSemanticSnapshot:
    """Validate exact producer bytes and return a typed immutable snapshot."""

    if type(snapshot_schema_version) is not int or snapshot_schema_version not in (1, 2):
        raise ProtocolSemanticContractError(
            "protocol snapshot schema version must be supported v1 or v2."
        )
    if snapshot_schema_version == PROTOCOL_SNAPSHOT_SCHEMA_VERSION:
        _require_exact(
            snapshot_policy_id,
            PROTOCOL_SNAPSHOT_POLICY_ID,
            name="protocol snapshot policy_id",
        )

    match = _SHA256_RE.fullmatch(semantic_hash)
    if match is None:
        raise ProtocolSemanticContractError(
            "protocol_semantic_hash must be 'sha256:' plus 64 lowercase hex digits."
        )
    observed = sha256(semantic_json.encode("utf-8")).hexdigest()
    if observed != match.group(1):
        raise ProtocolSemanticContractError(
            "protocol_semantic_json bytes do not match protocol_semantic_hash."
        )

    semantic = _parse_object(semantic_json, name="protocol_semantic_json")
    _require_exact(
        semantic.get("schema_id"),
        PROTOCOL_SEMANTIC_SCHEMA_ID,
        name="protocol_semantic_json.schema_id",
    )
    _require_exact(
        semantic.get("schema_version"),
        PROTOCOL_SEMANTIC_SCHEMA_VERSION,
        name="protocol_semantic_json.schema_version",
    )
    _require_exact(
        semantic.get("normalization_policy"),
        PROTOCOL_SEMANTIC_NORMALIZATION_POLICY,
        name="protocol_semantic_json.normalization_policy",
    )

    trial = _parse_object(trial_index_json, name="protocol_trial_index_json")
    _require_exact(
        trial.get("schema_id"),
        PROTOCOL_TRIAL_INDEX_SCHEMA_ID,
        name="protocol_trial_index_json.schema_id",
    )
    trial_schema_version = _require_int(
        trial.get("schema_version"),
        name="protocol_trial_index_json.schema_version",
        minimum=1,
    )
    if trial_schema_version not in PROTOCOL_TRIAL_INDEX_SCHEMA_VERSIONS:
        raise ProtocolSemanticContractError(
            "protocol_trial_index_json.schema_version is unsupported."
        )
    expected_trial_schema_version = 2 if snapshot_schema_version == 2 else 1
    _require_exact(
        trial_schema_version,
        expected_trial_schema_version,
        name="protocol_trial_index_json.schema_version",
    )
    _require_exact(
        trial.get("normalization_policy"),
        PROTOCOL_TRIAL_INDEX_NORMALIZATION_POLICIES[trial_schema_version],
        name="protocol_trial_index_json.normalization_policy",
    )
    _require_exact(
        trial.get("protocol_semantic_hash"),
        semantic_hash,
        name="protocol_trial_index_json.protocol_semantic_hash",
    )

    identity = semantic.get("identity")
    semantic_steps = identity.get("steps") if isinstance(identity, Mapping) else None
    trial_steps = trial.get("steps")
    if not isinstance(semantic_steps, list) or not isinstance(trial_steps, list):
        raise ProtocolSemanticContractError(
            "semantic identity and trial index must each contain an ordered steps array."
        )
    if len(semantic_steps) != len(trial_steps):
        raise ProtocolSemanticContractError(
            "semantic identity and trial index contain different step counts."
        )
    if not semantic_steps:
        raise ProtocolSemanticContractError(
            "semantic identity and trial index must contain at least one step."
        )

    steps: list[ProtocolStepIdentity] = []
    for expected_index, (semantic_step, trial_step) in enumerate(
        zip(semantic_steps, trial_steps)
    ):
        if not isinstance(semantic_step, Mapping) or not isinstance(trial_step, Mapping):
            raise ProtocolSemanticContractError(
                "protocol semantic steps must contain JSON objects."
            )
        trial_step_index = _require_int(
            trial_step.get("step_index"),
            name=f"trial step {expected_index}.step_index",
            minimum=0,
        )
        _require_exact(
            trial_step_index,
            expected_index,
            name=f"trial step {expected_index}.step_index",
        )
        semantic_mode_id = _require_int(
            semantic_step.get("stimulus_mode_id"),
            name=f"semantic step {expected_index}.stimulus_mode_id",
        )
        trial_mode_id = _require_int(
            trial_step.get("stimulus_mode_id"),
            name=f"trial step {expected_index}.stimulus_mode_id",
        )
        _require_exact(
            trial_mode_id,
            semantic_mode_id,
            name=f"trial step {expected_index}.stimulus_mode_id",
        )
        semantic_duration_s = _duration_seconds(
            semantic_step.get("duration"),
            name=f"semantic step {expected_index}.duration",
        )
        trial_duration_s = _require_number(
            trial_step.get("duration_s"),
            name=f"trial step {expected_index}.duration_s",
        )
        if trial_duration_s < 0:
            raise ProtocolSemanticContractError(
                f"trial step {expected_index}.duration_s must be non-negative."
            )
        if not math.isclose(
            semantic_duration_s,
            trial_duration_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ProtocolSemanticContractError(
                f"semantic and trial-index durations differ for step {expected_index}."
            )
        stimulus_mode = trial_step.get("stimulus_mode")
        stimulus_family = trial_step.get("stimulus_family")
        index_status = trial_step.get("index_status")
        if (
            type(stimulus_mode) is not str
            or not stimulus_mode
            or stimulus_mode != stimulus_mode.strip()
            or stimulus_mode != stimulus_mode.upper()
            or type(stimulus_family) is not str
            or not stimulus_family
            or stimulus_family != stimulus_family.strip()
            or stimulus_family != stimulus_family.lower()
            or type(index_status) is not str
            or not index_status
            or index_status != index_status.strip()
            or index_status != index_status.lower()
        ):
            raise ProtocolSemanticContractError(
                f"trial step {expected_index} lacks canonical mode, family, or index status."
            )
        expected_mode = STIMULUS_MODE.get(semantic_mode_id)
        if expected_mode is None:
            raise ProtocolSemanticContractError(
                f"semantic step {expected_index} uses unknown stimulus_mode_id "
                f"{semantic_mode_id}."
            )
        _require_exact(
            stimulus_mode,
            expected_mode,
            name=f"trial step {expected_index}.stimulus_mode",
        )
        expected_family = {4: "solid_color", 12: "chaser"}.get(semantic_mode_id)
        if expected_family is not None:
            _require_exact(
                stimulus_family,
                expected_family,
                name=f"trial step {expected_index}.stimulus_family",
            )
        features = trial_step.get("features")
        if not isinstance(features, Mapping):
            raise ProtocolSemanticContractError(
                f"trial step {expected_index}.features must be one object."
            )
        if semantic_mode_id == 4:
            parameters = semantic_step.get("parameters")
            if not isinstance(parameters, Mapping):
                raise ProtocolSemanticContractError(
                    f"semantic step {expected_index}.parameters must be one object."
                )
            _require_exact(
                parameters.get("color_type_id"),
                0,
                name=f"semantic step {expected_index}.parameters.color_type_id",
            )
        display_context = _display_context(
            stimulus_family=stimulus_family,
            stimulus_mode=stimulus_mode,
            features=features,
        )
        if semantic_mode_id == 4 and display_context != "solid_black":
            raise ProtocolSemanticContractError(
                f"semantic step {expected_index} lacks consistent sRGB black evidence."
            )
        steps.append(
            ProtocolStepIdentity(
                step_index=expected_index,
                stimulus_mode_id=trial_mode_id,
                stimulus_mode=stimulus_mode,
                stimulus_family=stimulus_family,
                duration_s=trial_duration_s,
                index_status=index_status,
                display_context=display_context,
                resolved_color_rgba8=_optional_rgba8(features),
            )
        )

    computed_trial_hash = "sha256:" + sha256(
        trial_index_json.encode("utf-8")
    ).hexdigest()
    if snapshot_schema_version == PROTOCOL_SNAPSHOT_SCHEMA_VERSION:
        if trial_index_hash is None:
            raise ProtocolSemanticContractError(
                "Citrus snapshot v2 requires protocol_trial_index_hash."
            )
        if _SHA256_RE.fullmatch(trial_index_hash) is None:
            raise ProtocolSemanticContractError(
                "protocol_trial_index_hash must be 'sha256:' plus 64 lowercase hex digits."
            )
        _require_exact(
            trial_index_hash,
            computed_trial_hash,
            name="protocol_trial_index_hash",
        )
        trial_integrity_status = TRIAL_INDEX_INTEGRITY_PRODUCER
    else:
        if trial_index_hash is not None:
            raise ProtocolSemanticContractError(
                "Legacy snapshot v1 must not be relabeled with a producer trial-index hash."
            )
        trial_integrity_status = TRIAL_INDEX_INTEGRITY_LOCAL

    return ProtocolSemanticSnapshot(
        semantic_hash=semantic_hash,
        semantic_json=semantic_json,
        trial_index_json=trial_index_json,
        trial_index_sha256=computed_trial_hash,
        trial_index_schema_version=trial_schema_version,
        trial_index_integrity_status=trial_integrity_status,
        snapshot_schema_version=snapshot_schema_version,
        snapshot_policy_id=snapshot_policy_id,
        steps=tuple(steps),
        semantic_payload=_freeze_json(semantic),  # type: ignore[arg-type]
        trial_index_payload=_freeze_json(trial),  # type: ignore[arg-type]
    )


def _materialized_utf8_array(group: Any, name: str) -> str:
    try:
        values = np.asarray(group[name][:])
    except (KeyError, TypeError) as exc:
        raise ProtocolSemanticContractError(
            f"Materialized protocol semantic snapshot lacks {name!r}."
        ) from exc
    if values.dtype != np.dtype(np.uint8) or values.ndim != 1:
        raise ProtocolSemanticContractError(
            f"Materialized protocol semantic snapshot {name!r} must be "
            "one-dimensional uint8."
        )
    try:
        return values.tobytes().decode("utf-8")
    except (TypeError, UnicodeDecodeError, ValueError) as exc:
        raise ProtocolSemanticContractError(
            f"Materialized protocol semantic snapshot {name!r} is not exact "
            "UTF-8 bytes."
        ) from exc


def _materialized_group_keys(group: Any) -> set[str]:
    try:
        return {str(name) for name in group.group_keys()}
    except Exception:
        try:
            return {
                str(name)
                for name in group.keys()
                if hasattr(group[name], "attrs") and not hasattr(group[name], "dtype")
            }
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            raise ProtocolSemanticContractError(
                "Materialized protocol semantic group inventory is unreadable."
            ) from exc


def read_materialized_protocol_semantic_snapshot(
    run_group: Any,
) -> ProtocolSemanticSnapshot:
    """Reload one exact verified Palette semantic snapshot and step binding.

    The caller chooses consolidated or unconsolidated traversal when opening
    ``run_group``.  This reader never repairs storage and accepts only the
    complete ``verified`` state; legacy or partially materialized runs are not
    semantic authorities.
    """

    attrs = run_group.attrs
    if attrs.get("protocol_semantic_status") != "verified":
        raise ProtocolSemanticContractError(
            "Materialized stimulus run is not a verified semantic authority."
        )
    semantic_attr_names = {
        "protocol_semantic_hash",
        "protocol_semantic_snapshot_path",
        "protocol_recipe_schema_id",
        "protocol_recipe_schema_version",
        "protocol_recipe_step_count",
        "protocol_recipe_mode_sequence",
        "protocol_recipe_label",
    }
    missing_attrs = sorted(name for name in semantic_attr_names if name not in attrs)
    if missing_attrs or "protocol_semantic_snapshot" not in run_group:
        missing = missing_attrs + (
            []
            if "protocol_semantic_snapshot" in run_group
            else ["protocol_semantic_snapshot"]
        )
        raise ProtocolSemanticContractError(
            "Verified stimulus run has partial semantic storage; missing "
            + ", ".join(missing)
            + "."
        )
    if attrs.get("protocol_semantic_snapshot_path") != "protocol_semantic_snapshot":
        raise ProtocolSemanticContractError(
            "Verified stimulus run points at an unexpected semantic snapshot path."
        )
    semantic_hash = attrs.get("protocol_semantic_hash")
    if type(semantic_hash) is not str:
        raise ProtocolSemanticContractError(
            "Verified stimulus run lacks an exact semantic hash."
        )
    snapshot_group = run_group["protocol_semantic_snapshot"]
    source_snapshot_schema_version = snapshot_group.attrs.get(
        "source_snapshot_schema_version",
        1,
    )
    if type(source_snapshot_schema_version) is not int:
        raise ProtocolSemanticContractError(
            "Materialized protocol snapshot has malformed source schema version."
        )
    source_snapshot_policy_id = snapshot_group.attrs.get(
        "source_snapshot_policy_id",
        "citrus.protocol.snapshot.legacy_v1",
    )
    if type(source_snapshot_policy_id) is not str:
        raise ProtocolSemanticContractError(
            "Materialized protocol snapshot has malformed source policy."
        )
    snapshot = validate_protocol_semantic_snapshot(
        semantic_hash=semantic_hash,
        semantic_json=_materialized_utf8_array(
            snapshot_group,
            "protocol_semantic_json_utf8",
        ),
        trial_index_json=_materialized_utf8_array(
            snapshot_group,
            "protocol_trial_index_json_utf8",
        ),
        trial_index_hash=(
            snapshot_group.attrs.get("protocol_trial_index_hash")
            if source_snapshot_schema_version == 2
            else None
        ),
        snapshot_schema_version=source_snapshot_schema_version,
        snapshot_policy_id=source_snapshot_policy_id,
    )
    recipe = snapshot.recipe_record()
    if (
        snapshot_group.attrs.get("schema_id")
        != PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_ID
        or snapshot_group.attrs.get("schema_version")
        != PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_VERSION
        or snapshot_group.attrs.get("source") != "citrus_h5_protocol_snapshot"
        or snapshot_group.attrs.get("protocol_semantic_hash") != semantic_hash
        or snapshot_group.attrs.get("protocol_trial_index_sha256")
        != snapshot.trial_index_sha256
        or snapshot_group.attrs.get("protocol_trial_index_integrity_status")
        != snapshot.trial_index_integrity_status
        or snapshot_group.attrs.get("source_snapshot_schema_version")
        != snapshot.snapshot_schema_version
        or snapshot_group.attrs.get("source_snapshot_policy_id")
        != snapshot.snapshot_policy_id
        or snapshot_group.attrs.get("source_trial_index_schema_version")
        != snapshot.trial_index_schema_version
        or snapshot_group.attrs.get("recipe") != recipe
        or attrs.get("protocol_recipe_schema_id")
        != PALETTE_PROTOCOL_RECIPE_SCHEMA_ID
        or attrs.get("protocol_recipe_schema_version")
        != PALETTE_PROTOCOL_RECIPE_SCHEMA_VERSION
        or attrs.get("protocol_recipe_step_count") != snapshot.step_count
        or list(attrs.get("protocol_recipe_mode_sequence", []))
        != list(snapshot.mode_sequence)
        or attrs.get("protocol_recipe_label") != snapshot.recipe_label
    ):
        raise ProtocolSemanticContractError(
            "Materialized protocol semantic attrs differ from the exact snapshot."
        )
    if snapshot.trial_index_integrity_status == TRIAL_INDEX_INTEGRITY_PRODUCER:
        if snapshot_group.attrs.get("protocol_trial_index_hash") != (
            snapshot.trial_index_sha256
        ):
            raise ProtocolSemanticContractError(
                "Materialized producer trial-index hash differs from the snapshot."
            )
    elif snapshot_group.attrs.get("palette_computed_trial_index_sha256") != (
        snapshot.trial_index_sha256
    ):
        raise ProtocolSemanticContractError(
            "Materialized Palette trial-index digest differs from the snapshot."
        )

    steps_group = run_group.get("steps")
    if steps_group is None:
        raise ProtocolSemanticContractError(
            "Verified protocol semantic snapshot has no materialized steps."
        )
    expected_names = {f"step_{step.step_index}" for step in snapshot.steps}
    if _materialized_group_keys(steps_group) != expected_names:
        raise ProtocolSemanticContractError(
            "Materialized stimulus steps differ from the exact semantic recipe."
        )
    if (
        steps_group.attrs.get("protocol_semantic_status") != "verified"
        or steps_group.attrs.get("protocol_semantic_hash") != semantic_hash
        or steps_group.attrs.get("protocol_recipe_step_count")
        != snapshot.step_count
        or list(steps_group.attrs.get("protocol_recipe_mode_sequence", []))
        != list(snapshot.mode_sequence)
    ):
        raise ProtocolSemanticContractError(
            "Materialized stimulus steps lack verified semantic parent binding."
        )
    for identity in snapshot.steps:
        step_attrs = steps_group[f"step_{identity.step_index}"].attrs
        expected_color = (
            list(identity.resolved_color_rgba8)
            if identity.resolved_color_rgba8 is not None
            else None
        )
        try:
            duration_s = float(step_attrs.get("duration_s"))
        except (TypeError, ValueError) as exc:
            raise ProtocolSemanticContractError(
                "Materialized stimulus step has malformed duration at "
                f"step_index={identity.step_index}."
            ) from exc
        if (
            step_attrs.get("protocol_semantic_status") != "verified"
            or step_attrs.get("protocol_semantic_hash") != semantic_hash
            or step_attrs.get("protocol_semantic_step_index")
            != identity.step_index
            or step_attrs.get("protocol_semantic_step_ref")
            != f"protocol_semantic_snapshot@recipe.steps[{identity.step_index}]"
            or step_attrs.get("step_index") != identity.step_index
            or step_attrs.get("stimulus_mode_id") != identity.stimulus_mode_id
            or step_attrs.get("stimulus_mode") != identity.stimulus_mode
            or not math.isclose(
                duration_s,
                identity.duration_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or step_attrs.get("stimulus_family") != identity.stimulus_family
            or step_attrs.get("protocol_trial_index_status")
            != identity.index_status
            or step_attrs.get("display_context") != identity.display_context
            or step_attrs.get("resolved_color_rgba8") != expected_color
        ):
            raise ProtocolSemanticContractError(
                "Materialized stimulus step semantic binding differs from the "
                f"exact snapshot at step_index={identity.step_index}."
            )
    return snapshot


def read_protocol_semantic_snapshot(
    h5: h5py.File,
) -> ProtocolSemanticSnapshot | None:
    """Read a modern Citrus contract or return ``None`` for a true legacy file.

    A partially present contract is corruption, not legacy absence.
    """

    paths = (
        PROTOCOL_SEMANTIC_HASH_H5_PATH,
        PROTOCOL_SEMANTIC_JSON_H5_PATH,
        PROTOCOL_TRIAL_INDEX_JSON_H5_PATH,
    )
    present = tuple(path in h5 for path in paths)
    if not any(present):
        return None
    if not all(present):
        missing = [path for path, available in zip(paths, present) if not available]
        raise ProtocolSemanticContractError(
            "Citrus protocol semantic contract is partial; missing "
            + ", ".join(missing)
            + "."
        )
    group = h5.get("/protocol_snapshot")
    if group is None:
        raise ProtocolSemanticContractError(
            "Citrus protocol semantic datasets lack /protocol_snapshot."
        )
    raw_schema_version = group.attrs.get("schema_version")
    if raw_schema_version is None:
        snapshot_schema_version = 1
        snapshot_policy_id = "citrus.protocol.snapshot.legacy_v1"
        trial_index_hash = None
        if PROTOCOL_TRIAL_INDEX_HASH_H5_PATH in h5:
            raise ProtocolSemanticContractError(
                "Legacy Citrus protocol snapshot unexpectedly contains a v2 "
                "trial-index hash."
            )
    else:
        if hasattr(raw_schema_version, "item"):
            raw_schema_version = raw_schema_version.item()
        snapshot_schema_version = _require_int(
            raw_schema_version,
            name="/protocol_snapshot@schema_version",
            minimum=1,
        )
        if snapshot_schema_version != PROTOCOL_SNAPSHOT_SCHEMA_VERSION:
            raise ProtocolSemanticContractError(
                "Citrus protocol snapshot schema version is unsupported."
            )
        _require_exact(
            _decode_scalar_text(
                group.attrs.get("schema_id"),
                name="/protocol_snapshot@schema_id",
            ),
            PROTOCOL_SNAPSHOT_SCHEMA_ID,
            name="/protocol_snapshot@schema_id",
        )
        snapshot_policy_id = _decode_scalar_text(
            group.attrs.get("policy_id"),
            name="/protocol_snapshot@policy_id",
        )
        _require_exact(
            snapshot_policy_id,
            PROTOCOL_SNAPSHOT_POLICY_ID,
            name="/protocol_snapshot@policy_id",
        )
        contract_status = _decode_scalar_text(
            group.attrs.get("contract_status"),
            name="/protocol_snapshot@contract_status",
        )
        if contract_status != PROTOCOL_SNAPSHOT_VALID_STATUS:
            raise ProtocolSemanticContractError(
                "Citrus snapshot v2 semantic identity is unavailable because "
                f"contract_status={contract_status!r}."
            )
        if PROTOCOL_TRIAL_INDEX_HASH_H5_PATH not in h5:
            raise ProtocolSemanticContractError(
                "Citrus snapshot v2 is missing protocol_trial_index_hash."
            )
        trial_index_hash = _decode_scalar_text(
            h5[PROTOCOL_TRIAL_INDEX_HASH_H5_PATH][()],
            name=PROTOCOL_TRIAL_INDEX_HASH_H5_PATH,
        )

    values = [_decode_scalar_text(h5[path][()], name=path) for path in paths]
    return validate_protocol_semantic_snapshot(
        semantic_hash=values[0],
        semantic_json=values[1],
        trial_index_json=values[2],
        trial_index_hash=trial_index_hash,
        snapshot_schema_version=snapshot_schema_version,
        snapshot_policy_id=snapshot_policy_id,
    )


__all__ = [
    "PALETTE_PROTOCOL_RECIPE_SCHEMA_ID",
    "PALETTE_PROTOCOL_RECIPE_SCHEMA_VERSION",
    "PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_ID",
    "PALETTE_PROTOCOL_SNAPSHOT_SCHEMA_VERSION",
    "PROTOCOL_SEMANTIC_HASH_H5_PATH",
    "PROTOCOL_SEMANTIC_JSON_H5_PATH",
    "PROTOCOL_TRIAL_INDEX_JSON_H5_PATH",
    "PROTOCOL_TRIAL_INDEX_HASH_H5_PATH",
    "TRIAL_INDEX_INTEGRITY_LOCAL",
    "TRIAL_INDEX_INTEGRITY_PRODUCER",
    "ProtocolSemanticContractError",
    "ProtocolSemanticSnapshot",
    "ProtocolStepIdentity",
    "read_materialized_protocol_semantic_snapshot",
    "read_protocol_semantic_snapshot",
    "validate_protocol_semantic_snapshot",
]
