"""Verified read-only projection for semantic-v2 epoch behavior summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Mapping

from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
    PARENT_PATH,
)
from fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle import (
    load_provider_epoch_behavior_summary_source_handle,
)

from .provenance import freeze, plain


class ExactEpochBehaviorProjectionError(ValueError):
    """The selected semantic epoch-behavior child is incompatible or changed."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactEpochBehaviorProjectionError(f"{label} must be one object.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactEpochBehaviorProjectionError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _exact_run_path(value: Any) -> tuple[str, str]:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ExactEpochBehaviorProjectionError(
            "Epoch-behavior run path must be one exact child."
        )
    prefix = f"{PARENT_PATH}/"
    name = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not name
        or "/" in name
        or name.casefold()
        in {
            "latest",
            "latest_complete",
            "latest_pending",
            "current",
            "selected",
            "authoritative",
            "authoritative_run",
            "default",
        }
    ):
        raise ExactEpochBehaviorProjectionError(
            "Epoch-behavior run path must be one exact non-selector child."
        )
    return value, name


def option_epoch_behavior_binding(option: Any) -> Mapping[str, Any]:
    """Validate and freeze the metadata-discovered semantic summary binding."""

    bindings = _mapping(option.spec.get("analysis_bindings"), label="analysis bindings")
    value = _mapping(bindings.get("epoch_behavior"), label="epoch behavior binding")
    expected = {
        "run_path",
        "manifest_sha256",
        "payload_digest",
        "source_protocol_semantic_selection",
        "source_provider_motion",
        "source_swim_bouts",
        "parameters",
        "dimensions",
        "array_declaration_count",
    }
    if set(value) != expected:
        raise ExactEpochBehaviorProjectionError(
            "Epoch-behavior binding has an unsupported field set."
        )
    run_path, _ = _exact_run_path(value.get("run_path"))
    parameters = dict(_mapping(value.get("parameters"), label="epoch parameters"))
    if (
        parameters.get("physical_speed_level")
        not in {"filtered", "smoothed", "averaged"}
        or parameters.get("rate_denominator") != "valid_tracked_duration_s"
        or parameters.get("spatial_metrics")
        != "omitted_requires_separately_selected_position_provider"
    ):
        raise ExactEpochBehaviorProjectionError(
            "Epoch-behavior speed, denominator, or spatial policy is incompatible."
        )
    dimensions = dict(_mapping(value.get("dimensions"), label="epoch dimensions"))
    if dimensions.get("n_epoch_rows") != 3:
        raise ExactEpochBehaviorProjectionError(
            "Epoch-behavior binding must contain three semantic epoch rows."
        )
    count = value.get("array_declaration_count")
    if type(count) is not int or count <= 0:
        raise ExactEpochBehaviorProjectionError(
            "Epoch-behavior array declaration count is invalid."
        )
    result = {
        "run_path": run_path,
        "manifest_sha256": _digest(
            value.get("manifest_sha256"), label="epoch manifest digest"
        ),
        "payload_digest": _digest(
            value.get("payload_digest"), label="epoch payload digest"
        ),
        "source_protocol_semantic_selection": dict(
            _mapping(
                value.get("source_protocol_semantic_selection"),
                label="epoch semantic source",
            )
        ),
        "source_provider_motion": dict(
            _mapping(value.get("source_provider_motion"), label="epoch motion source")
        ),
        "source_swim_bouts": dict(
            _mapping(value.get("source_swim_bouts"), label="epoch bout source")
        ),
        "parameters": parameters,
        "dimensions": dimensions,
        "array_declaration_count": count,
    }
    return freeze(result)


def load_exact_epoch_behavior(
    archive: Path,
    option: Any,
    *,
    spatial: Any,
    direct_validation_receipt: str | Path | None = None,
    required_array_paths: Collection[str] | None = None,
) -> Any:
    """Load one exact semantic-v2 summary and recheck its spatial epoch join."""

    binding = option_epoch_behavior_binding(option)
    spatial_sources = _mapping(
        spatial.scientific_manifest.get("sources"), label="spatial sources"
    )
    semantic = _mapping(
        spatial_sources.get("protocol_semantic_selection"),
        label="spatial semantic selection",
    )
    for field_name in ("run_path", "manifest_sha256"):
        if binding["source_protocol_semantic_selection"].get(
            field_name
        ) != semantic.get(field_name):
            raise ExactEpochBehaviorProjectionError(
                "Epoch behavior uses another semantic selection."
            )
    run_path, run_name = _exact_run_path(binding["run_path"])
    handle = load_provider_epoch_behavior_summary_source_handle(
        archive,
        run_name=run_name,
        expected_recording_id=spatial.recording_id,
        expected_semantic_selection=semantic,
        deep_audit=direct_validation_receipt is None,
        direct_validation_receipt=direct_validation_receipt,
        required_array_paths=required_array_paths,
    )
    observed = {
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "payload_digest": handle.payload_digest,
        "source_protocol_semantic_selection": handle.manifest["sources"][
            "protocol_semantic_selection"
        ],
        "source_provider_motion": handle.manifest["sources"]["provider_motion"],
        "source_swim_bouts": handle.manifest["sources"]["swim_bouts"],
        "parameters": handle.manifest["parameters"],
        "dimensions": handle.manifest["dimensions"],
        "array_declaration_count": len(handle.manifest["array_declarations"]),
    }
    if plain(observed) != plain(binding):
        raise ExactEpochBehaviorProjectionError(
            "Epoch-behavior child changed after metadata discovery."
        )
    if handle.run_path != run_path:
        raise ExactEpochBehaviorProjectionError(
            "Epoch-behavior loader resolved another run path."
        )
    return handle


__all__ = [
    "ExactEpochBehaviorProjectionError",
    "load_exact_epoch_behavior",
    "option_epoch_behavior_binding",
]
