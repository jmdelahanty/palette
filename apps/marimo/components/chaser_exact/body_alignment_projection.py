"""Verified read-only projection for persisted anatomical alignment summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Mapping

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    load_composable_chaser_successor_source_handle,
)

from ..chaser_exact_body_alignment_contract import (
    BODY_ALIGNMENT_PARENT,
    FORBIDDEN_SELECTORS,
    ExactBodyAlignmentContractError,
    validate_body_alignment_scientific_manifest,
)
from .provenance import freeze, plain


class ExactBodyAlignmentProjectionError(ValueError):
    """The selected alignment successor cannot produce a verified projection."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactBodyAlignmentProjectionError(f"{label} must be one object.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactBodyAlignmentProjectionError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _exact_run_path(value: Any) -> tuple[str, str]:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ExactBodyAlignmentProjectionError(
            "Body-alignment run path must be one exact child."
        )
    prefix = f"{BODY_ALIGNMENT_PARENT}/"
    name = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not name
        or "/" in name
        or name in {".", ".."}
        or name.casefold() in FORBIDDEN_SELECTORS
    ):
        raise ExactBodyAlignmentProjectionError(
            "Body-alignment run path must be one exact non-selector child."
        )
    return value, name


def option_body_alignment_binding(option: Any) -> Mapping[str, Any]:
    """Validate and freeze the metadata-discovered alignment binding."""

    bindings = _mapping(option.spec.get("analysis_bindings"), label="analysis bindings")
    value = _mapping(
        bindings.get("body_alignment_by_distance"), label="alignment binding"
    )
    expected = {
        "run_path",
        "manifest_sha256",
        "scientific_payload_sha256",
        "source_relative_frame",
        "source_protocol_semantic_selection",
        "source_fish_position_authority",
        "source_body_frame_authority",
        "distance_bin_recipe",
        "dimensions",
        "epoch_records",
        "identity_registries",
    }
    if set(value) != expected:
        raise ExactBodyAlignmentProjectionError(
            "Body-alignment binding has an unsupported field set."
        )
    run_path, _ = _exact_run_path(value.get("run_path"))
    result = {
        "run_path": run_path,
        "manifest_sha256": _digest(
            value.get("manifest_sha256"), label="alignment manifest digest"
        ),
        "scientific_payload_sha256": _digest(
            value.get("scientific_payload_sha256"),
            label="alignment scientific payload digest",
        ),
        "source_relative_frame": dict(
            _mapping(
                value.get("source_relative_frame"), label="alignment relative source"
            )
        ),
        "source_protocol_semantic_selection": dict(
            _mapping(
                value.get("source_protocol_semantic_selection"),
                label="alignment semantic source",
            )
        ),
        "source_fish_position_authority": dict(
            _mapping(
                value.get("source_fish_position_authority"),
                label="alignment position authority",
            )
        ),
        "source_body_frame_authority": dict(
            _mapping(
                value.get("source_body_frame_authority"),
                label="alignment body authority",
            )
        ),
        "distance_bin_recipe": dict(
            _mapping(
                value.get("distance_bin_recipe"), label="alignment distance-bin recipe"
            )
        ),
        "dimensions": dict(
            _mapping(value.get("dimensions"), label="alignment dimensions")
        ),
        "epoch_records": tuple(value.get("epoch_records", ())),
        "identity_registries": dict(
            _mapping(
                value.get("identity_registries"), label="alignment identity registries"
            )
        ),
    }
    return freeze(result)


def load_exact_body_alignment(
    archive: Path,
    option: Any,
    *,
    spatial: Any,
    expected_relative_binding: Mapping[str, Any],
    relative: Any,
    direct_validation_receipt: str | Path | None = None,
    required_array_names: Collection[str] | None = None,
) -> Any:
    """Load one exact persisted summary and recheck every sealed source join."""

    binding = option_body_alignment_binding(option)
    run_path, run_name = _exact_run_path(binding["run_path"])
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_body_alignment_by_distance",
        run_name=run_name,
        expected_recording_id=spatial.recording_id,
        deep_audit=direct_validation_receipt is None,
        direct_validation_receipt=direct_validation_receipt,
        required_array_names=required_array_names,
    )
    if (
        handle.run_path != run_path
        or handle.manifest_sha256 != binding["manifest_sha256"]
        or handle.scientific_payload_sha256 != binding["scientific_payload_sha256"]
    ):
        raise ExactBodyAlignmentProjectionError(
            "Body-alignment successor changed after metadata discovery."
        )
    spatial_sources = _mapping(
        spatial.scientific_manifest.get("sources"), label="spatial sources"
    )
    semantic = _mapping(
        spatial_sources.get("protocol_semantic_selection"),
        label="spatial semantic source",
    )
    authorities = _mapping(
        relative.run_manifest.get("source_authorities"),
        label="relative source authorities",
    )
    scale = _mapping(
        relative.run_manifest.get("scale_policy"), label="relative scale policy"
    )
    try:
        verified = validate_body_alignment_scientific_manifest(
            handle.scientific_manifest,
            expected_scientific_payload_sha256=handle.scientific_payload_sha256,
            expected_n_frames=relative.n_frames,
            expected_n_chasers=relative.n_chasers,
            expected_relative_binding=expected_relative_binding,
            expected_semantic_binding=semantic,
            expected_fish_position_authority=_mapping(
                authorities.get("fish_position"), label="fish-position authority"
            ),
            expected_body_frame_authority=_mapping(
                authorities.get("body_frame"), label="body-frame authority"
            ),
            expected_scale_policy=scale,
            expected_epoch_records=spatial.scientific_manifest.get("epoch_records"),
        )
    except ExactBodyAlignmentContractError as exc:
        raise ExactBodyAlignmentProjectionError(str(exc)) from exc
    observed = {
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "scientific_payload_sha256": handle.scientific_payload_sha256,
        **verified,
    }
    if plain(observed) != plain(binding):
        raise ExactBodyAlignmentProjectionError(
            "Body-alignment scientific sources differ from the selected exact bundle."
        )
    return handle


__all__ = [
    "ExactBodyAlignmentProjectionError",
    "load_exact_body_alignment",
    "option_body_alignment_binding",
]
