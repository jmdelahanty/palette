"""Helpers for interpreting experiment setup metadata stored in Zarr attrs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .type_conversions import normalize_attr as _normalize_attr


@dataclass(frozen=True)
class ExperimentSetupInfo:
    """Resolved experiment setup with provenance."""

    setup_type: str
    num_dishes: Optional[int]
    source: str
    has_experiment_setup: bool


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    get_method = getattr(value, "get", None)
    if callable(get_method):
        return value  # type: ignore[return-value]
    return None


def infer_experiment_setup(attrs: Mapping[str, Any]) -> ExperimentSetupInfo:
    """Infer single- vs multi-dish mode using Zarr root attrs."""
    experiment_setup = attrs.get("experiment_setup")
    if experiment_setup:
        mapping = _as_mapping(experiment_setup)
        setup_type = None
        num_dishes = None
        if mapping is not None:
            setup_type = _normalize_attr(mapping.get("setup_type"))
            num_dishes = _coerce_int(mapping.get("num_dishes"))
        if setup_type == "single_dish" or num_dishes == 1:
            return ExperimentSetupInfo("single_dish", num_dishes or 1, "experiment_setup", True)
        if setup_type == "multi_dish" or (num_dishes is not None and num_dishes > 1):
            return ExperimentSetupInfo("multi_dish", num_dishes, "experiment_setup", True)
        return ExperimentSetupInfo(setup_type or "unknown", num_dishes, "experiment_setup", True)

    chamber = _normalize_attr(attrs.get("experimental_chamber"))
    if chamber:
        return ExperimentSetupInfo("single_dish", 1, "experimental_chamber", False)

    return ExperimentSetupInfo("unknown", None, "unknown", False)


def subdish_required(attrs: Mapping[str, Any]) -> bool:
    """Return True if sub-dish masks are required for spatial arena assignment."""
    info = infer_experiment_setup(attrs)
    return info.setup_type != "single_dish"
