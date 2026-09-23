"""Submitted native identity consistency and exact external receipt agreement."""

from .common import SHA256_PATTERN, digest, exact_keys, require, text
from .correspondence import BINDING
from .hdf5_types import dataset_bytes
from .schema import contract, read_json


def validate_recording_identity(h5, outer, geometry, snapshot):
    definition = contract("experimental_h5_identity_claims_v1.json")["component"]
    claims = read_json(h5, definition["claims_path"], canonical=True)
    receipt = read_json(h5, definition["receipt_path"], canonical=True)
    binding = read_json(h5, BINDING, canonical=True)
    exact_keys(claims, definition["claims_schema"]["fields"], "recording_claims")
    require(
        claims["schema_id"] == definition["claims_schema"]["schema_id"]
        and type(claims["schema_version"]) is int
        and claims["schema_version"] == 1,
        "recording_claims_schema",
    )
    exact_keys(receipt, definition["receipt_fields"], "recording_association_receipt")
    require(
        receipt["schema_id"] == "citrus.experimental_h5.recording_association_receipt"
        and type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["status"] == "complete"
        and receipt["reason"] == ""
        and receipt["scope"] == definition["receipt_scope"],
        "recording_association_receipt_status",
    )
    require(
        receipt["claims_ref"] == definition["claims_path"]
        and receipt["claims_sha256"]
        == digest(dataset_bytes(h5[definition["claims_path"]]))
        and receipt["binding_sha256"] == digest(dataset_bytes(h5[BINDING])),
        "recording_association_digests",
    )
    for name, expected in {
        "recording_ref": BINDING + "#/recording_id",
        "experiment_ref": "/metadata/session#attribute:citrus_experiment_id",
        "session_ref": "/metadata/session#attribute:session_uuid",
    }.items():
        require(receipt[name] == expected, "recording_association_reference")
    require(
        text(claims["recording_id"], "recording_id") == binding["recording_id"],
        "recording_association_recording",
    )
    session, body = h5["/metadata/session"].attrs, outer["contract"]
    for name in (
        "observation_context_id",
        "citrus_experiment_id",
        "citrus_session_uuid",
    ):
        require(
            text(claims[name], name) == body[name],
            f"external_identity_disagreement:{name}",
        )
    require(
        claims["citrus_experiment_id"] == session["citrus_experiment_id"]
        and claims["citrus_session_uuid"] == session["session_uuid"],
        "recording_session_identity",
    )
    submitted = claims["finalized_observation_receipt"]
    require(
        receipt["source_receipt_validation"]
        == ("not_present" if submitted is None else "not_performed"),
        "source_receipt_scope",
    )
    if submitted is not None:
        exact_keys(
            submitted, ("receipt_id", "contract_sha256"), "source_receipt_reference"
        )
        value = text(submitted["contract_sha256"], "source_receipt_digest")
        require(
            SHA256_PATTERN.fullmatch(value)
            and submitted["receipt_id"] == "obsbindfin_" + value[7:],
            "source_receipt_identity",
        )
    target = body["target"]
    for name in ("arena_id", "canvas_name", "rig_id"):
        require(target[name] == session[name], f"external_target_mismatch:{name}")
    require(
        target["camera_id"] == binding["acquisition_camera_id"]
        and target["source_camera_stream_id"] == binding["camera_serial"],
        "external_target_camera_mismatch",
    )
    require(
        body["runtime_geometry_contract_sha256"] == geometry["runtime_sha256"],
        "external_geometry_mismatch",
    )
    require(
        body["protocol_semantic"]
        == {"status": "available", "semantic_sha256": snapshot.semantic_hash},
        "external_protocol_mismatch",
    )
    return claims
