"""One mandatory preflight for the explicit native experimental input profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .appearance import validate_appearance_witness
from .common import PROFILE, canonical_json, digest, require
from .correspondence import BINDING, validate_component_correspondence
from .geometry import validate_geometry
from .identity import validate_recording_identity
from .integrity import (
    source_identity,
    validate_external_receipt,
    validate_internal_integrity,
)
from .protocol import validate_protocol
from .rows import validate_table_relations
from .schema import APPEARANCE, read_json

TABLE_COMPONENTS = {
    "frames": "/frames/stimulus",
    "chaser": "/components/chaser/states",
    "visual_appearance": APPEARANCE,
    "independent_motion_grid": "/components/independent_motion_grid/states",
    "moving_grating": "/components/moving_grating/states",
    "bounding_boxes": "/observations/bounding_boxes",
    "events": "/events/records",
    "runtime_trials": "/trials/trial_index",
    "region_routing": "/observations/region_routing",
    "region_candidates": "/observations/region_candidates",
    "display_submissions": "/timing/display_submissions/frame_submissions",
}


@dataclass(frozen=True)
class UnifiedH5Admission:
    profile: str
    source_identity: dict
    source_sha256: str
    finalization_receipt: dict
    dependency_count: int
    internal_manifest_sha256: str
    frame_count: int
    component_rows: dict[str, int]
    geometry_scope: str
    recording_id: str
    camera_serial: str
    protocol_semantic_sha256: str
    protocol_execution_sha256: str
    node_kinds: dict[str, str]
    selector_eligible: bool = False

    def manifest_claims(self):
        return {
            "profile": self.profile,
            "source_sha256": self.source_sha256,
            "finalization_receipt_sha256": digest(
                canonical_json(self.finalization_receipt)
            ),
            "dependency_count": self.dependency_count,
            "internal_manifest_sha256": self.internal_manifest_sha256,
            "frame_count": self.frame_count,
            "component_rows": self.component_rows,
            "geometry_scope": self.geometry_scope,
            "recording_id": self.recording_id,
            "camera_serial": self.camera_serial,
            "protocol_semantic_sha256": self.protocol_semantic_sha256,
            "protocol_execution_sha256": self.protocol_execution_sha256,
            "selector_eligible": False,
        }


def _accounting(h5, integrity, correspondence):
    outcomes = {value["component_id"]: value for value in integrity.component_outcomes}
    counts = {
        name: h5[path].shape[0] for name, path in TABLE_COMPONENTS.items() if path in h5
    }
    counts.update(
        correspondence=correspondence.mapped_frame_count
        + sum(correspondence.component_rows.values()),
        geometry=1,
        recording_association=1,
    )
    for name, count in counts.items():
        value = outcomes.get(name)
        require(
            value is not None
            and value["requirement"] == "required"
            and value["status"] == "complete"
            and value["expected_rows"] == value["written_rows"] == count,
            f"component_accounting_mismatch:{name}",
        )
    for name, value in outcomes.items():
        if name not in counts:
            require(
                name in TABLE_COMPONENTS
                and value["requirement"] == "not_applicable"
                and value["expected_rows"] == value["written_rows"] == 0,
                f"unresolved_component_accounting:{name}",
            )
    return counts


def validate_artifact(
    h5, *, source_h5: Path, finalization_receipt: dict
) -> UnifiedH5Admission:
    identity = validate_external_receipt(
        h5, source_h5=source_h5, receipt=finalization_receipt
    )
    integrity = validate_internal_integrity(h5)
    validate_table_relations(h5, integrity.table_descriptors)
    correspondence = validate_component_correspondence(h5)
    snapshot, execution = validate_protocol(h5)
    geometry = validate_geometry(
        h5, read_json(h5, BINDING, canonical=True), integrity.table_descriptors
    )
    validate_recording_identity(h5, finalization_receipt, geometry, snapshot)
    authored = read_json(h5, "/protocol/authored/protocol_trial_index_json")
    declared = any(
        chaser.get("appearance") is not None
        for step in authored["steps"]
        for chaser in step.get("features", {}).get("chasers", [])
    )
    require(not declared or APPEARANCE in h5, "declared_appearance_missing")
    if APPEARANCE in h5:
        validate_appearance_witness(h5)
    counts = _accounting(h5, integrity, correspondence)
    require(
        source_identity(h5, Path(source_h5)) == identity, "source_h5_generation_changed"
    )
    return UnifiedH5Admission(
        profile=PROFILE,
        source_identity=identity,
        source_sha256=finalization_receipt["contract"]["h5_artifact"]["sha256"],
        finalization_receipt=finalization_receipt,
        dependency_count=integrity.dependency_count,
        internal_manifest_sha256=integrity.manifest_sha256,
        frame_count=counts["frames"],
        component_rows=counts,
        geometry_scope=geometry["scope"],
        recording_id=correspondence.recording_id,
        camera_serial=correspondence.camera_serial,
        protocol_semantic_sha256=snapshot.semantic_hash,
        protocol_execution_sha256=execution.execution_hash,
        node_kinds=dict(integrity.nodes),
    )
