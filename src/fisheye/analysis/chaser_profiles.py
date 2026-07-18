"""Versioned protocol adapters and protocol-neutral chaser analysis profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

PROTOCOL_PROFILE_SCHEMA_ID = "palette.chaser_protocol_profile"
PROTOCOL_PROFILE_SCHEMA_VERSION = 1
ANALYSIS_PROFILE_SCHEMA_ID = "palette.chaser_analysis_profile"
ANALYSIS_PROFILE_SCHEMA_VERSION = 1
PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
BOUNDARY_POLICY = "inclusive_start_exclusive_end_event_boundary"
ALLOWED_FALLBACKS = frozenset({"recording_start", "recording_end"})
ALLOWED_CARDINALITIES = frozenset({"recording", "per_chaser"})


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not PROFILE_ID_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must match {PROFILE_ID_PATTERN.pattern!r}: {text!r}")
    return text


def _string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{label} cannot be empty or contain empty values")
    return result


def _load_mapping(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    profile_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"profile must contain a mapping: {profile_path}")
    return profile_path, payload


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ChaserWindowDefinition:
    window_id: int
    label: str
    start_event: str
    end_event: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ChaserWindowDefinition":
        return cls(
            window_id=int(raw.get("window_id", -1)),
            label=_identifier(raw.get("label"), label="window label"),
            start_event=_identifier(raw.get("start_event"), label="window start_event"),
            end_event=_identifier(raw.get("end_event"), label="window end_event"),
        )


@dataclass(frozen=True)
class ResolvedChaserWindow:
    window_id: int
    label: str
    start_frame: int
    end_frame: int
    source_start_event_name: str
    source_end_event_name: str
    source_start_event_frame: int
    source_end_event_frame: int
    source_policy: str


@dataclass(frozen=True)
class ChaserProtocolProfile:
    profile_id: str
    profile_version: int
    source_adapter_id: str
    source_adapter_version: int
    role_resolver_id: str
    role_resolver_version: int
    window_policy_id: str
    window_policy_version: int
    boundary_policy: str
    event_aliases: Mapping[str, tuple[str, ...]]
    event_fallbacks: Mapping[str, str]
    windows: tuple[ChaserWindowDefinition, ...]
    analysis_parameters: Mapping[str, Mapping[str, Any]]
    source_path: str

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: str = "",
    ) -> "ChaserProtocolProfile":
        if str(raw.get("schema_id") or "") != PROTOCOL_PROFILE_SCHEMA_ID:
            raise ValueError(
                f"protocol profile schema_id must be {PROTOCOL_PROFILE_SCHEMA_ID!r}"
            )
        if int(raw.get("schema_version") or 0) != PROTOCOL_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"protocol profile schema_version must be {PROTOCOL_PROFILE_SCHEMA_VERSION}"
            )
        adapter = raw.get("source_adapter")
        roles = raw.get("role_resolver")
        policy = raw.get("window_policy")
        if (
            not isinstance(adapter, Mapping)
            or not isinstance(roles, Mapping)
            or not isinstance(policy, Mapping)
        ):
            raise ValueError(
                "source_adapter, role_resolver, and window_policy must be mappings"
            )
        aliases_raw = policy.get("event_aliases")
        if not isinstance(aliases_raw, Mapping):
            raise ValueError("window_policy.event_aliases must be a mapping")
        aliases = {
            _identifier(key, label="event alias key"): _string_tuple(
                value, label=f"event_aliases.{key}"
            )
            for key, value in aliases_raw.items()
        }
        fallbacks_raw = policy.get("event_fallbacks") or {}
        if not isinstance(fallbacks_raw, Mapping):
            raise ValueError("window_policy.event_fallbacks must be a mapping")
        fallbacks = {
            _identifier(key, label="fallback event key"): str(value).strip()
            for key, value in fallbacks_raw.items()
        }
        invalid_fallbacks = sorted(set(fallbacks.values()) - ALLOWED_FALLBACKS)
        if invalid_fallbacks:
            raise ValueError(
                "unsupported event fallback(s): " + ", ".join(invalid_fallbacks)
            )
        rows = policy.get("windows")
        if isinstance(rows, str) or not isinstance(rows, Sequence):
            raise ValueError("window_policy.windows must be a sequence")
        windows = tuple(
            ChaserWindowDefinition.from_mapping(row)
            for row in rows
            if isinstance(row, Mapping)
        )
        if len(windows) != len(rows) or not windows:
            raise ValueError("every window_policy.windows row must be a mapping")
        ids = [window.window_id for window in windows]
        labels = [window.label for window in windows]
        if len(set(ids)) != len(ids) or len(set(labels)) != len(labels):
            raise ValueError("window ids and labels must be unique")
        referenced = {
            value
            for window in windows
            for value in (window.start_event, window.end_event)
        }
        missing_aliases = sorted(referenced - set(aliases))
        if missing_aliases:
            raise ValueError(
                "window references missing event alias(es): "
                + ", ".join(missing_aliases)
            )
        parameters_raw = raw.get("analysis_parameters") or {}
        if not isinstance(parameters_raw, Mapping) or any(
            not isinstance(value, Mapping) for value in parameters_raw.values()
        ):
            raise ValueError("analysis_parameters must map module ids to mappings")
        boundary_policy = str(policy.get("boundary_policy") or "").strip()
        if boundary_policy != BOUNDARY_POLICY:
            raise ValueError(f"window boundary_policy must be {BOUNDARY_POLICY!r}")
        return cls(
            profile_id=_identifier(raw.get("profile_id"), label="profile_id"),
            profile_version=int(raw.get("profile_version") or 0),
            source_adapter_id=_identifier(adapter.get("id"), label="source_adapter.id"),
            source_adapter_version=int(adapter.get("version") or 0),
            role_resolver_id=_identifier(roles.get("id"), label="role_resolver.id"),
            role_resolver_version=int(roles.get("version") or 0),
            window_policy_id=_identifier(policy.get("id"), label="window_policy.id"),
            window_policy_version=int(policy.get("version") or 0),
            boundary_policy=boundary_policy,
            event_aliases=aliases,
            event_fallbacks=fallbacks,
            windows=windows,
            analysis_parameters={
                str(key): dict(value) for key, value in parameters_raw.items()
            },
            source_path=str(source_path),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": PROTOCOL_PROFILE_SCHEMA_ID,
            "schema_version": PROTOCOL_PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "source_adapter": {
                "id": self.source_adapter_id,
                "version": self.source_adapter_version,
            },
            "role_resolver": {
                "id": self.role_resolver_id,
                "version": self.role_resolver_version,
            },
            "window_policy": {
                "id": self.window_policy_id,
                "version": self.window_policy_version,
                "boundary_policy": self.boundary_policy,
                "event_aliases": {
                    key: list(value) for key, value in self.event_aliases.items()
                },
                "event_fallbacks": dict(self.event_fallbacks),
                "windows": [asdict(window) for window in self.windows],
            },
            "analysis_parameters": {
                key: dict(value) for key, value in self.analysis_parameters.items()
            },
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ChaserAnalysisModule:
    module_id: str
    implementation: str
    schema_id: str
    schema_version: int
    depends_on: tuple[str, ...]
    execution_cardinality: str
    default_enabled: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ChaserAnalysisModule":
        cardinality = str(raw.get("execution_cardinality") or "recording").strip()
        if cardinality not in ALLOWED_CARDINALITIES:
            raise ValueError(f"unsupported execution_cardinality: {cardinality!r}")
        dependencies = raw.get("depends_on") or []
        if isinstance(dependencies, str) or not isinstance(dependencies, Sequence):
            raise ValueError("module depends_on must be a sequence")
        schema_id = str(raw.get("schema_id") or "").strip()
        if not schema_id.startswith("palette."):
            raise ValueError(f"invalid module schema_id: {schema_id!r}")
        implementation = str(raw.get("implementation") or "").strip()
        if not implementation.startswith("fisheye.analysis."):
            raise ValueError(f"invalid module implementation: {implementation!r}")
        return cls(
            module_id=_identifier(raw.get("id"), label="module id"),
            implementation=implementation,
            schema_id=schema_id,
            schema_version=int(raw.get("schema_version") or 0),
            depends_on=tuple(
                _identifier(value, label="module dependency") for value in dependencies
            ),
            execution_cardinality=cardinality,
            default_enabled=bool(raw.get("default_enabled", False)),
        )


@dataclass(frozen=True)
class ChaserAnalysisProfile:
    profile_id: str
    profile_version: int
    modules: tuple[ChaserAnalysisModule, ...]
    source_path: str

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source_path: str = "",
    ) -> "ChaserAnalysisProfile":
        if str(raw.get("schema_id") or "") != ANALYSIS_PROFILE_SCHEMA_ID:
            raise ValueError(
                f"analysis profile schema_id must be {ANALYSIS_PROFILE_SCHEMA_ID!r}"
            )
        if int(raw.get("schema_version") or 0) != ANALYSIS_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"analysis profile schema_version must be {ANALYSIS_PROFILE_SCHEMA_VERSION}"
            )
        rows = raw.get("modules")
        if isinstance(rows, str) or not isinstance(rows, Sequence):
            raise ValueError("analysis profile modules must be a sequence")
        modules = tuple(
            ChaserAnalysisModule.from_mapping(row)
            for row in rows
            if isinstance(row, Mapping)
        )
        if len(modules) != len(rows) or not modules:
            raise ValueError("every analysis profile module must be a mapping")
        ids = {module.module_id for module in modules}
        if len(ids) != len(modules):
            raise ValueError("analysis module ids must be unique")
        for module in modules:
            missing = sorted(set(module.depends_on) - ids)
            if missing:
                raise ValueError(
                    f"module {module.module_id!r} references unknown dependencies: {missing}"
                )
        return cls(
            profile_id=_identifier(raw.get("profile_id"), label="profile_id"),
            profile_version=int(raw.get("profile_version") or 0),
            modules=modules,
            source_path=str(source_path),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": ANALYSIS_PROFILE_SCHEMA_ID,
            "schema_version": ANALYSIS_PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "modules": [asdict(module) for module in self.modules],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def _resolve_event(
    profile: ChaserProtocolProfile,
    event_frames: Mapping[str, int],
    event_key: str,
    *,
    total_frames: int,
) -> tuple[str, int, str | None]:
    for event_name in profile.event_aliases[event_key]:
        if event_name in event_frames:
            return event_name, int(event_frames[event_name]), None
    fallback = profile.event_fallbacks.get(event_key)
    if fallback == "recording_start":
        return "RECORDING_START_FALLBACK", 0, f"missing_{event_key}_used_frame_0"
    if fallback == "recording_end":
        return (
            "RECORDING_END_FALLBACK",
            max(0, int(total_frames)),
            f"missing_{event_key}_used_total_frames",
        )
    aliases = ", ".join(profile.event_aliases[event_key])
    raise ValueError(
        f"stimulus events do not include required {event_key!r} alias: {aliases}"
    )


def resolve_profile_windows(
    profile: ChaserProtocolProfile,
    event_frames: Mapping[str, int],
    *,
    total_frames: int,
) -> tuple[ResolvedChaserWindow, ...]:
    """Resolve configured event windows without embedding a protocol name in code."""

    resolved: dict[str, tuple[str, int, str | None]] = {}
    for event_key in profile.event_aliases:
        if any(
            event_key in (window.start_event, window.end_event)
            for window in profile.windows
        ):
            resolved[event_key] = _resolve_event(
                profile,
                event_frames,
                event_key,
                total_frames=int(total_frames),
            )
    max_frame = max(0, int(total_frames) - 1)
    result: list[ResolvedChaserWindow] = []
    for window in sorted(profile.windows, key=lambda value: value.window_id):
        start_name, start_boundary, start_note = resolved[window.start_event]
        end_name, end_boundary, end_note = resolved[window.end_event]
        start = max(0, int(start_boundary))
        end = min(max_frame, max(start, int(end_boundary) - 1))
        notes = [value for value in (start_note, end_note) if value]
        source_policy = profile.boundary_policy
        if notes:
            source_policy += ";" + ";".join(notes)
        result.append(
            ResolvedChaserWindow(
                window_id=int(window.window_id),
                label=window.label,
                start_frame=start,
                end_frame=end,
                source_start_event_name=start_name,
                source_end_event_name=end_name,
                source_start_event_frame=int(start_boundary),
                source_end_event_frame=int(end_boundary),
                source_policy=source_policy,
            )
        )
    return tuple(result)


def resolve_protocol_payload_path(
    payload: Mapping[str, Any], source_path: str
) -> Any | None:
    """Resolve the first value at a profile-owned dotted payload path.

    A segment ending in ``[]`` expands every mapping in that sequence. The
    optional ``protocol_json.`` prefix documents the source without becoming
    part of the decoded payload path.
    """

    normalized = str(source_path or "").strip().strip(".")
    if normalized.startswith("protocol_json."):
        normalized = normalized[len("protocol_json.") :]
    if not normalized:
        return None
    candidates: list[Any] = [payload]
    for raw_segment in normalized.split("."):
        expand_sequence = raw_segment.endswith("[]")
        key = raw_segment[:-2] if expand_sequence else raw_segment
        if not key:
            return None
        resolved: list[Any] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or key not in candidate:
                continue
            value = candidate[key]
            if expand_sequence:
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    resolved.extend(value)
            else:
                resolved.append(value)
        candidates = resolved
        if not candidates:
            return None
    return next((value for value in candidates if value is not None), None)


def load_chaser_protocol_profile(path: str | Path) -> ChaserProtocolProfile:
    profile_path, payload = _load_mapping(path)
    return ChaserProtocolProfile.from_mapping(payload, source_path=str(profile_path))


def load_chaser_analysis_profile(path: str | Path) -> ChaserAnalysisProfile:
    profile_path, payload = _load_mapping(path)
    return ChaserAnalysisProfile.from_mapping(payload, source_path=str(profile_path))


def default_goodcopbadcop_source_profile_path() -> Path:
    return Path(__file__).resolve().parent / "profiles" / "goodcopbadcop_source_v1.yaml"


def default_chaser_analysis_profile_path() -> Path:
    return Path(__file__).resolve().parent / "profiles" / "chaser_behavior_v1.yaml"


__all__ = [
    "ANALYSIS_PROFILE_SCHEMA_ID",
    "ANALYSIS_PROFILE_SCHEMA_VERSION",
    "BOUNDARY_POLICY",
    "PROTOCOL_PROFILE_SCHEMA_ID",
    "PROTOCOL_PROFILE_SCHEMA_VERSION",
    "ChaserAnalysisModule",
    "ChaserAnalysisProfile",
    "ChaserProtocolProfile",
    "ChaserWindowDefinition",
    "ResolvedChaserWindow",
    "default_chaser_analysis_profile_path",
    "default_goodcopbadcop_source_profile_path",
    "load_chaser_analysis_profile",
    "load_chaser_protocol_profile",
    "resolve_protocol_payload_path",
    "resolve_profile_windows",
]
