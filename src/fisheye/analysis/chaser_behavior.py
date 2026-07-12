"""Canonical chaser behavior vocabulary and protocol metadata resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Sequence


class ChaserBehaviorClass(IntEnum):
    UNKNOWN = 0
    AGGRESSIVE = 1
    RANDOM_NON_CHASING = 2
    INERT = 3


BEHAVIOR_CLASS_LABELS: dict[int, str] = {
    int(ChaserBehaviorClass.UNKNOWN): "unknown",
    int(ChaserBehaviorClass.AGGRESSIVE): "aggressive",
    int(ChaserBehaviorClass.RANDOM_NON_CHASING): "random_non_chasing",
    int(ChaserBehaviorClass.INERT): "inert",
}

LEGACY_BEHAVIOR_ALIASES: dict[str, str] = {"benign": "inert"}


@dataclass(frozen=True)
class ConfiguredChaserBehavior:
    chaser_index: int
    behavior_class_id: int
    behavior_class: str
    enable_chase: bool
    enable_random_movement: bool
    behavior_mode: int | None
    raw_color_rgba: tuple[float, float, float, float]
    start_position_preset: str
    end_position_preset: str


def canonical_behavior_label(value: Any) -> str:
    label = str(value or "unknown").strip().lower()
    return LEGACY_BEHAVIOR_ALIASES.get(label, label)


def configured_behavior_class_id(
    *,
    enable_chase: bool,
    enable_random_movement: bool,
) -> int:
    if enable_chase:
        return int(ChaserBehaviorClass.AGGRESSIVE)
    if enable_random_movement:
        return int(ChaserBehaviorClass.RANDOM_NON_CHASING)
    return int(ChaserBehaviorClass.INERT)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_chaser_parameters(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("protocol_json lacks steps[].")
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        parameters = step.get("parameters")
        if isinstance(parameters, Mapping) and isinstance(parameters.get("chasers"), list):
            return parameters
    raise ValueError("protocol_json lacks steps[].parameters.chasers[].")


def resolve_configured_chaser_behaviors(
    payload: Mapping[str, Any],
) -> tuple[ConfiguredChaserBehavior, ...]:
    """Resolve a variable-length chaser list using the acquisition enum vocabulary."""

    parameters = _first_chaser_parameters(payload)
    chasers = parameters.get("chasers")
    if not isinstance(chasers, list):
        raise ValueError("protocol_json chasers field is not a list.")
    resolved: list[ConfiguredChaserBehavior] = []
    for fallback_index, chaser in enumerate(chasers):
        if not isinstance(chaser, Mapping):
            continue
        enable_chase = bool(chaser.get("enable_chase", False))
        enable_random_movement = bool(chaser.get("enable_random_movement", False))
        class_id = configured_behavior_class_id(
            enable_chase=enable_chase,
            enable_random_movement=enable_random_movement,
        )
        explicit_index = _safe_int(chaser.get("chaser_index"))
        resolved.append(
            ConfiguredChaserBehavior(
                chaser_index=fallback_index if explicit_index is None else explicit_index,
                behavior_class_id=class_id,
                behavior_class=BEHAVIOR_CLASS_LABELS[class_id],
                enable_chase=enable_chase,
                enable_random_movement=enable_random_movement,
                behavior_mode=_safe_int(chaser.get("behavior_mode")),
                raw_color_rgba=(
                    _safe_float(chaser.get("color_r"), 0.0),
                    _safe_float(chaser.get("color_g"), 0.0),
                    _safe_float(chaser.get("color_b"), 0.0),
                    _safe_float(chaser.get("color_a"), 1.0),
                ),
                start_position_preset=str(chaser.get("start_position_preset") or ""),
                end_position_preset=str(chaser.get("end_position_preset") or ""),
            )
        )
    return tuple(resolved)


def behavior_counts(
    behaviors: Sequence[ConfiguredChaserBehavior],
) -> dict[str, int]:
    counts = {label: 0 for label in BEHAVIOR_CLASS_LABELS.values()}
    for behavior in behaviors:
        label = canonical_behavior_label(behavior.behavior_class)
        counts[label] = counts.get(label, 0) + 1
    return counts
