"""Verified read-only projections for paired exact chaser successors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    CHASER_RELATIVE_FRAME_RUN_PARENT_PATH,
    MANIFEST_ATTR as RELATIVE_MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR as RELATIVE_MANIFEST_DIGEST_ATTR,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    load_composable_chaser_successor_source_handle,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingProof,
    require_same_exact_relative_frame_child,
    validate_exact_relative_frame_binding,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

from ..common import normalize_path
from ..registry import CHASER_EXACT_SUCCESSOR_RENDERER, InteractiveSpecOption
from .controller_trial_projection import load_exact_controller_trials
from .provenance import build_projection_provenance, freeze, plain

PROVIDER_ROLES = ("keypoint", "detection")
RELATIVE_PARENT = CHASER_RELATIVE_FRAME_RUN_PARENT_PATH
RADIAL_PARENT = "analysis/chaser_radial_near_field_runs"
SPATIAL_PARENT = "analysis/chaser_spatial_occupancy_runs"

_FRAME_ARRAY_NAMES = (
    "acquisition_frame_id",
    "timestamp_ns",
    "timestamp_valid",
    "selection_member",
    "fish_position_xy_px",
    "fish_position_valid",
    "chaser_identity_code",
    "chaser_behavior_role_code",
    "chaser_occurrence_member",
    "chaser_position_xy_px",
    "chaser_position_valid",
    "relative_distance_physical",
    "relative_physical_valid",
)


class ExactChaserProjectionError(ValueError):
    """The selected exact successor cannot produce a verified projection."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactChaserProjectionError(f"{label} must be one object.")
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactChaserProjectionError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _exact_child_path(value: Any, *, parent: str, label: str) -> tuple[str, str]:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ExactChaserProjectionError(f"{label} must be one exact child path.")
    prefix = f"{parent}/"
    name = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not name
        or "/" in name
        or name in {".", ".."}
        or name.casefold()
        in {
            "latest",
            "latest_complete",
            "latest_pending",
            "current",
            "current_run",
            "selected",
            "authoritative",
            "authoritative_run",
            "default",
        }
    ):
        raise ExactChaserProjectionError(
            f"{label} must be one exact child below {parent!r}."
        )
    return value, name


def _option_bundle(option: InteractiveSpecOption) -> tuple[str, str]:
    if option.renderer != CHASER_EXACT_SUCCESSOR_RENDERER:
        raise ExactChaserProjectionError(
            "Selected source is not an exact chaser-successor bundle."
        )
    return _exact_child_path(
        normalize_path(option.run_path),
        parent=SPATIAL_PARENT,
        label="spatial bundle run",
    )


@dataclass(frozen=True)
class ExactChaserSelectionIdentity:
    """Complete reactive/cache identity for one selected exact-chaser view."""

    archive_path: str
    run_path: str
    bundle_manifest_sha256: str
    renderer: str
    schema_id: str | None
    analysis_id: str
    display_parameter_version: str
    display_parameters_sha256: str
    analysis_bindings_sha256: str


def build_exact_chaser_selection_identity(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
    *,
    analysis_id: str,
    display_parameter_version: str,
) -> ExactChaserSelectionIdentity:
    """Build an immutable identity without opening scientific arrays."""

    archive = Path(zarr_path).expanduser().resolve()
    run_path, _ = _option_bundle(option)
    manifest_sha256 = _digest(
        option.spec.get("bundle_manifest_sha256"), label="bundle option digest"
    )
    display_parameters = _mapping(
        option.spec.get("display_parameters", {}), label="display parameters"
    )
    analysis_bindings = _mapping(
        option.spec.get("analysis_bindings", {}), label="analysis bindings"
    )
    return ExactChaserSelectionIdentity(
        archive_path=str(archive),
        run_path=run_path,
        bundle_manifest_sha256=manifest_sha256,
        renderer=option.renderer,
        schema_id=option.schema_id,
        analysis_id=analysis_id,
        display_parameter_version=display_parameter_version,
        display_parameters_sha256=canonical_json_sha256(plain(display_parameters)),
        analysis_bindings_sha256=canonical_json_sha256(plain(analysis_bindings)),
    )


@dataclass(frozen=True)
class RelativeFrameProjection:
    run_path: str
    run_name: str
    recording_id: str
    manifest_sha256: str
    n_frames: int
    n_chasers: int
    source_authorities: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    def frame_chaser(self, name: str) -> np.ndarray:
        values = self.arrays[name]
        return values.reshape((self.n_frames, self.n_chasers) + values.shape[1:])

    def collapsed_frame(self, name: str) -> np.ndarray:
        values = self.frame_chaser(name)
        reference = values[:, :1, ...]
        if values.dtype.kind == "f":
            repeated = np.array_equal(
                values, np.broadcast_to(reference, values.shape), equal_nan=True
            )
        else:
            repeated = np.array_equal(values, np.broadcast_to(reference, values.shape))
        if not repeated:
            raise ExactChaserProjectionError(
                f"Exact relative-frame array {name!r} differs across chaser rows."
            )
        return values[:, 0, ...]


@dataclass(frozen=True)
class ExactChaserSuccessorProjection:
    analysis_id: str
    recording_id: str
    selection_identity: ExactChaserSelectionIdentity
    spatial: Any
    radials: tuple[Any, Any]
    relatives: tuple[RelativeFrameProjection, RelativeFrameProjection] | None
    controller_trials: Any | None
    provider_ids: tuple[str, str]
    epoch_records: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]


def _load_targeted_relative(
    archive: Path,
    *,
    run_path: str,
    expected_manifest_sha256: str,
    expected_recording_id: str,
) -> RelativeFrameProjection:
    """Deep-read only arrays required by these interactive display projections."""

    exact_path, run_name = _exact_child_path(
        run_path, parent=RELATIVE_PARENT, label="relative-frame run"
    )
    validate_direct_consolidated_subtree(archive, subtree_path=exact_path)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[exact_path]
    attrs = dict(getattr(run, "attrs", {}))
    manifest = _mapping(attrs.get(RELATIVE_MANIFEST_ATTR), label="relative manifest")
    manifest_digest = _digest(
        attrs.get(RELATIVE_MANIFEST_DIGEST_ATTR), label="relative manifest digest"
    )
    if manifest_digest != expected_manifest_sha256:
        raise ExactChaserProjectionError(
            "Relative-frame manifest digest differs from its bundle binding."
        )
    if canonical_json_sha256(plain(manifest)) != manifest_digest:
        raise ExactChaserProjectionError("Relative-frame manifest digest is stale.")
    if (
        attrs.get("schema_id") != "palette.analysis.chaser_relative_frame"
        or attrs.get("schema_version") != 1
        or attrs.get("run_path") != exact_path
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        raise ExactChaserProjectionError(
            "Relative-frame source is not one complete exact candidate."
        )
    if manifest.get("recording_id") != expected_recording_id:
        raise ExactChaserProjectionError(
            "Relative-frame source belongs to another recording."
        )
    dimensions = _mapping(manifest.get("dimensions"), label="relative dimensions")
    n_frames = int(dimensions.get("n_frames", 0))
    n_chasers = int(dimensions.get("n_chasers", 0))
    n_rows = int(dimensions.get("n_rows", 0))
    if n_frames <= 0 or n_chasers <= 0 or n_rows != n_frames * n_chasers:
        raise ExactChaserProjectionError("Relative-frame dimensions are invalid.")
    declarations = manifest.get("array_declarations")
    if not isinstance(declarations, list):
        raise ExactChaserProjectionError(
            "Relative-frame manifest lacks array declarations."
        )
    declaration_by_path = {
        item.get("path"): item for item in declarations if isinstance(item, Mapping)
    }
    if len(declaration_by_path) != len(declarations):
        raise ExactChaserProjectionError(
            "Relative-frame array declarations are duplicated or invalid."
        )
    base = run["base"]
    arrays: dict[str, np.ndarray] = {}
    for name in _FRAME_ARRAY_NAMES:
        path = f"base/{name}"
        declaration = _mapping(
            declaration_by_path.get(path), label=f"declaration {path}"
        )
        values = np.asarray(base[name][...])
        if (
            values.dtype.str != declaration.get("dtype")
            or list(values.shape) != declaration.get("shape")
            or values.shape[0] != n_rows
        ):
            raise ExactChaserProjectionError(
                f"Relative-frame array {path!r} metadata changed."
            )
        if array_values_sha256(values) != declaration.get("content_sha256"):
            raise ExactChaserProjectionError(
                f"Relative-frame array {path!r} content digest changed."
            )
        copied = np.array(values, copy=True, order="C")
        copied.setflags(write=False)
        arrays[name] = copied
    authorities = _mapping(
        manifest.get("source_authorities"), label="relative source authorities"
    )
    return RelativeFrameProjection(
        run_path=exact_path,
        run_name=run_name,
        recording_id=expected_recording_id,
        manifest_sha256=manifest_digest,
        n_frames=n_frames,
        n_chasers=n_chasers,
        source_authorities=freeze(authorities),
        arrays=MappingProxyType(arrays),
    )


def _source_records(spatial: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    scientific = spatial.scientific_manifest
    sources = _mapping(scientific.get("sources"), label="spatial sources")
    providers = sources.get("position_providers")
    if not isinstance(providers, (list, tuple)) or len(providers) != 2:
        raise ExactChaserProjectionError(
            "Spatial bundle must bind exactly two position providers."
        )
    records = tuple(_mapping(value, label="spatial provider") for value in providers)
    if tuple(record.get("provider_role") for record in records) != PROVIDER_ROLES:
        raise ExactChaserProjectionError(
            "Spatial bundle provider roles are not keypoint then detection."
        )
    return records  # type: ignore[return-value]


def _verify_bundle_children(
    spatial: Any,
    radials: Sequence[Any],
    relative_bindings: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], tuple[ExactRelativeFrameBindingProof, ...]]:
    records = _source_records(spatial)
    if len(radials) != 2 or len(relative_bindings) != 2:
        raise ExactChaserProjectionError("Exact chaser bundle is incomplete.")
    provider_ids: list[str] = []
    semantic_binding: Mapping[str, Any] | None = None
    geometry_binding: Mapping[str, Any] | None = None
    epoch_records: Any = None
    arena: Any = None
    relative_binding_proofs: list[ExactRelativeFrameBindingProof] = []
    for record, radial, relative_binding in zip(
        records, radials, relative_bindings, strict=True
    ):
        radial_binding = _mapping(
            record.get("radial_near_field"), label="radial binding"
        )
        expected_relative = _mapping(
            record.get("relative_frame"), label="relative binding"
        )
        require_same_exact_relative_frame_child(
            expected_relative,
            relative_binding,
            expected_label="spatial relative-frame binding",
            observed_label="projection relative-frame binding",
        )
        if radial.run_path != radial_binding.get(
            "run_path"
        ) or radial.manifest_sha256 != radial_binding.get("manifest_sha256"):
            raise ExactChaserProjectionError(
                "Spatial bundle child digest binding is stale."
            )
        scientific = radial.scientific_manifest
        provider = _mapping(
            scientific.get("position_provider"), label="radial provider"
        )
        if (
            provider.get("status") != "first_class_explicit_authority"
            or provider.get("provider_id") != record.get("provider_id")
            or provider.get("provider_digest") != record.get("provider_digest")
        ):
            raise ExactChaserProjectionError(
                "Radial successor position authority differs from the bundle."
            )
        radial_sources = _mapping(scientific.get("sources"), label="radial sources")
        relative_binding_proof = require_same_exact_relative_frame_child(
            expected_relative,
            _mapping(
                radial_sources.get("relative_frame"), label="radial relative source"
            ),
            expected_label="spatial relative-frame binding",
            observed_label="radial relative-frame binding",
        )
        local_semantic = _mapping(
            radial_sources.get("protocol_semantic_selection"),
            label="radial semantic selection",
        )
        local_geometry = _mapping(
            radial_sources.get("arena_geometry_and_scale"),
            label="radial arena geometry",
        )
        if semantic_binding is None:
            semantic_binding = local_semantic
            geometry_binding = local_geometry
            epoch_records = plain(scientific.get("epoch_records"))
            arena = plain(scientific.get("arena"))
        elif (
            dict(local_semantic) != dict(semantic_binding)
            or dict(local_geometry) != dict(geometry_binding or {})
            or plain(scientific.get("epoch_records")) != epoch_records
            or plain(scientific.get("arena")) != arena
        ):
            raise ExactChaserProjectionError(
                "Paired radial successors do not share exact epochs and arena."
            )
        provider_ids.append(str(provider["provider_id"]))
        relative_binding_proofs.append(relative_binding_proof)
    if len(set(provider_ids)) != 2:
        raise ExactChaserProjectionError(
            "Exact chaser bundle providers are not distinct."
        )
    spatial_sources = _mapping(
        spatial.scientific_manifest.get("sources"), label="spatial sources"
    )
    spatial_semantic = _mapping(
        spatial_sources.get("protocol_semantic_selection"),
        label="spatial semantic selection",
    )
    spatial_geometry = _mapping(
        spatial_sources.get("arena_geometry_and_scale"),
        label="spatial arena geometry",
    )
    if (
        dict(spatial_semantic) != dict(semantic_binding or {})
        or dict(spatial_geometry) != dict(geometry_binding or {})
        or plain(spatial.scientific_manifest.get("epoch_records")) != epoch_records
    ):
        raise ExactChaserProjectionError(
            "Spatial and radial successors use different semantic epochs or geometry."
        )
    return (provider_ids[0], provider_ids[1]), tuple(relative_binding_proofs)


def load_exact_chaser_projection(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
    *,
    selection_identity: ExactChaserSelectionIdentity,
    load_relative: bool,
    load_controller_trials: bool = False,
) -> ExactChaserSuccessorProjection:
    """Load one exact, selector-free visualization projection."""

    archive = Path(zarr_path).expanduser().resolve()
    if selection_identity.archive_path != str(archive):
        raise ExactChaserProjectionError(
            "Exact chaser selection identity names another archive."
        )
    spatial_path, spatial_name = _option_bundle(option)
    if selection_identity.run_path != spatial_path:
        raise ExactChaserProjectionError(
            "Exact chaser selection identity names another bundle."
        )
    spatial = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_spatial_occupancy",
        run_name=spatial_name,
        deep_audit=True,
    )
    if (
        spatial.run_path != spatial_path
        or spatial.manifest_sha256 != selection_identity.bundle_manifest_sha256
    ):
        raise ExactChaserProjectionError(
            "Selected exact chaser bundle changed after discovery."
        )
    records = _source_records(spatial)
    relative_bindings = tuple(
        _mapping(record.get("relative_frame"), label="relative binding")
        for record in records
    )
    radial_bindings = tuple(
        _mapping(record.get("radial_near_field"), label="radial binding")
        for record in records
    )
    radials = tuple(
        load_composable_chaser_successor_source_handle(
            archive,
            successor_kind="chaser_radial_near_field",
            run_name=_exact_child_path(
                binding.get("run_path"), parent=RADIAL_PARENT, label="radial run"
            )[1],
            expected_recording_id=spatial.recording_id,
            deep_audit=True,
        )
        for binding in radial_bindings
    )
    provider_ids, relative_binding_proofs = _verify_bundle_children(
        spatial, radials, relative_bindings
    )
    relatives = None
    if load_relative:
        relatives = tuple(
            _load_targeted_relative(
                archive,
                run_path=str(binding["run_path"]),
                expected_manifest_sha256=_digest(
                    binding.get("manifest_sha256"), label="relative binding digest"
                ),
                expected_recording_id=spatial.recording_id,
            )
            for binding in relative_bindings
        )
        for record, relative in zip(records, relatives, strict=True):
            authority = _mapping(
                relative.source_authorities.get("fish_position"),
                label="relative fish authority",
            )
            if authority.get("provider_id") != record.get(
                "provider_id"
            ) or authority.get("provider_digest") != record.get("provider_digest"):
                raise ExactChaserProjectionError(
                    "Relative-frame fish authority differs from the bundle."
                )
        if (
            relatives[0].n_frames != relatives[1].n_frames
            or relatives[0].n_chasers != relatives[1].n_chasers
        ):
            raise ExactChaserProjectionError("Paired relative-frame dimensions differ.")
        for name in (
            "acquisition_frame_id",
            "timestamp_ns",
            "timestamp_valid",
            "selection_member",
            "chaser_identity_code",
            "chaser_behavior_role_code",
            "chaser_occurrence_member",
            "chaser_position_xy_px",
            "chaser_position_valid",
        ):
            if not np.array_equal(relatives[0].arrays[name], relatives[1].arrays[name]):
                raise ExactChaserProjectionError(
                    f"Paired exact chaser evidence differs for {name!r}."
                )
    controller_trials = None
    if load_controller_trials:
        if relatives is None:
            raise ExactChaserProjectionError(
                "Controller-trial views require exact relative-frame sources."
            )
        controller_trials = load_exact_controller_trials(
            archive,
            option,
            spatial=spatial,
            expected_relative_binding=relative_bindings[0],
            relative=relatives[0],
        )
    epoch_records = spatial.scientific_manifest.get("epoch_records")
    if not isinstance(epoch_records, (list, tuple)) or not epoch_records:
        raise ExactChaserProjectionError("Exact chaser bundle lacks epoch records.")
    return ExactChaserSuccessorProjection(
        analysis_id=selection_identity.analysis_id,
        recording_id=spatial.recording_id,
        selection_identity=selection_identity,
        spatial=spatial,
        radials=radials,  # type: ignore[arg-type]
        relatives=relatives,  # type: ignore[arg-type]
        controller_trials=controller_trials,
        provider_ids=provider_ids,
        epoch_records=tuple(freeze(record) for record in epoch_records),
        provenance=build_projection_provenance(
            spatial=spatial,
            radials=radials,
            relative_bindings=relative_bindings,
            relative_binding_proofs=relative_binding_proofs,
            controller_trials=controller_trials,
        ),
    )


def identity_registry(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    registries = manifest.get("identity_registries")
    if not isinstance(registries, Mapping):
        return {}
    value = registries.get(name)
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "ExactChaserProjectionError",
    "ExactChaserSelectionIdentity",
    "ExactChaserSuccessorProjection",
    "RelativeFrameProjection",
    "build_exact_chaser_selection_identity",
    "identity_registry",
    "load_exact_chaser_projection",
]
