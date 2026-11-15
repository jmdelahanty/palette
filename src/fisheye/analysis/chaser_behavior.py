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
    # A chaser with a behavior_program: its behavior is the program's phases,
    # never the per-chaser fields (agent-contracts
    # citrus-crimson/chaser_behavior_class_contract.md).
    PROGRAMMED = 4


BEHAVIOR_CLASS_LABELS: dict[int, str] = {
    int(ChaserBehaviorClass.UNKNOWN): "unknown",
    int(ChaserBehaviorClass.AGGRESSIVE): "aggressive",
    int(ChaserBehaviorClass.RANDOM_NON_CHASING): "random_non_chasing",
    int(ChaserBehaviorClass.INERT): "inert",
    int(ChaserBehaviorClass.PROGRAMMED): "programmed",
}

# Citrus ChaserProperties::BehaviorMode: ACTIVE = 0, PASSIVE = 1.
PASSIVE_BEHAVIOR_MODE = 1

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


def behavior_class_label(class_id: Any) -> str:
    """The label of a known class id; an unknown id is refused, never relabelled."""

    try:
        key = int(class_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"chaser behavior class id is not an integer: {class_id!r}") from exc
    if key not in BEHAVIOR_CLASS_LABELS:
        raise ValueError(f"unknown chaser behavior class id: {key}")
    return BEHAVIOR_CLASS_LABELS[key]


def configured_behavior_class_id(
    *,
    enable_chase: bool,
    enable_random_movement: bool,
    has_behavior_program: bool = False,
    behavior_mode: int | None = None,
) -> int:
    # Citrus ResolveChaserBehaviorClass: a passive chaser is inert even with a
    # program; otherwise a program makes the chaser programmed and its
    # enable_chase/enable_random_movement fields are ignored at runtime.
    if has_behavior_program:
        if behavior_mode == PASSIVE_BEHAVIOR_MODE:
            return int(ChaserBehaviorClass.INERT)
        return int(ChaserBehaviorClass.PROGRAMMED)
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


def first_chaser_step_parameters(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Parameters of the first protocol step that configures chasers.

    Protocols may open with non-chaser steps (e.g. SOLID_BLACK), so the chaser
    step is found by content, never assumed to be ``steps[0]``.
    """
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

    parameters = first_chaser_step_parameters(payload)
    chasers = parameters.get("chasers")
    if not isinstance(chasers, list):
        raise ValueError("protocol_json chasers field is not a list.")
    resolved: list[ConfiguredChaserBehavior] = []
    for fallback_index, chaser in enumerate(chasers):
        if not isinstance(chaser, Mapping):
            continue
        enable_chase = bool(chaser.get("enable_chase", False))
        enable_random_movement = bool(chaser.get("enable_random_movement", False))
        behavior_mode = _safe_int(chaser.get("behavior_mode"))
        if "chaser_behavior_class_id" in chaser:
            # A recorded class (trial-index features) is taken as recorded.
            behavior_class_label(chaser["chaser_behavior_class_id"])
            class_id = int(chaser["chaser_behavior_class_id"])
        else:
            class_id = configured_behavior_class_id(
                enable_chase=enable_chase,
                enable_random_movement=enable_random_movement,
                has_behavior_program=chaser.get("behavior_program") is not None,
                behavior_mode=behavior_mode,
            )
        explicit_index = _safe_int(chaser.get("chaser_index"))
        resolved.append(
            ConfiguredChaserBehavior(
                chaser_index=fallback_index if explicit_index is None else explicit_index,
                behavior_class_id=class_id,
                behavior_class=behavior_class_label(class_id),
                enable_chase=enable_chase,
                enable_random_movement=enable_random_movement,
                behavior_mode=behavior_mode,
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
