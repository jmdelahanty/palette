"""Validate the exploratory chaser proxy chain and emit one exact receipt."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.chaser_profiles import (
    load_chaser_analysis_profile,
    resolve_chaser_analysis_modules,
)
from fisheye.analysis_workflows.chaser_input_provenance_proxy_source_handle import (
    load_chaser_input_provenance_proxy_source_handle,
)
from fisheye.analysis_workflows.chaser_profile_applicability import (
    CapabilityAssessment,
    CapabilityState,
    input_provenance_proxy_alignment_assessment,
    plan_chaser_profile_applicability,
)
from fisheye.analysis_workflows.chaser_proxy_relative_frame_adapter import (
    COORDINATE_POLICY_ID,
    TIMING_POLICY_ID,
    require_proxy_native_binding,
)
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    load_chaser_relative_frame_source_handle,
)
from fisheye.analysis_workflows.provider_chaser_stimulus_source_handle import (
    load_provider_chaser_stimulus_source_handle,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


RECEIPT_SCHEMA_ID = "palette.chaser_proxy_candidate_workflow_receipt"
RECEIPT_SCHEMA_VERSION = 1


class ChaserProxyCandidateReceiptError(ValueError):
    """Raised when one candidate chain or receipt is incomplete or stale."""


def _commit(value: object) -> str:
    commit = str(value or "").strip().lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ChaserProxyCandidateReceiptError(
            "palette_commit must be one full lowercase Git SHA."
        )
    return commit


def _exact_source_run_name(path: object) -> str:
    prefix = "analysis/provider_chaser_distance_candidate_runs/"
    if type(path) is not str or not path.startswith(prefix):
        raise ChaserProxyCandidateReceiptError(
            "Proxy source_run_path is not an exact native provider candidate."
        )
    name = path[len(prefix) :]
    if not name or "/" in name:
        raise ChaserProxyCandidateReceiptError(
            "Proxy source_run_path is not one exact native provider child."
        )
    return name


def _ready(capability_id: str, *, source: Mapping[str, Any]) -> CapabilityAssessment:
    return CapabilityAssessment(
        capability_id=capability_id,
        state=CapabilityState.READY,
        reason_code="validated_candidate_ready",
        evidence=source,
    )


def build_chaser_proxy_candidate_receipt(
    analysis_zarr: str | Path,
    *,
    proxy_run_name: str,
    relative_frame_run_name: str,
    analysis_profile_path: str | Path,
    palette_commit: str,
    expected_recording_id: str | None = None,
    expected_proxy_manifest_sha256: str | None = None,
    expected_relative_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Reopen all exact publications and prove their dependency chain."""

    archive = Path(analysis_zarr).expanduser().resolve()
    palette_commit = _commit(palette_commit)
    proxy = load_chaser_input_provenance_proxy_source_handle(
        archive,
        run_name=proxy_run_name,
        expected_recording_id=expected_recording_id,
        expected_manifest_sha256=expected_proxy_manifest_sha256,
        use_consolidated=True,
    )
    native = load_provider_chaser_stimulus_source_handle(
        archive,
        run_name=_exact_source_run_name(
            proxy.acquisition_projection_record["source_run_path"]
        ),
        expected_recording_id=proxy.recording_id,
        expected_manifest_sha256=proxy.acquisition_projection_record[
            "source_manifest_sha256"
        ],
        use_consolidated=True,
    )
    require_proxy_native_binding(proxy, native)
    relative = load_chaser_relative_frame_source_handle(
        archive,
        run_name=relative_frame_run_name,
        expected_recording_id=proxy.recording_id,
        use_consolidated=True,
    )
    if (
        expected_relative_manifest_sha256 is not None
        and relative.manifest_sha256 != expected_relative_manifest_sha256
    ):
        raise ChaserProxyCandidateReceiptError(
            "Relative-frame manifest differs from the expected digest."
        )
    context = relative.context
    publication = context["acquisition_projection_publication"]["record"]
    if publication != proxy.publication_binding_record:
        raise ChaserProxyCandidateReceiptError(
            "Relative-frame context does not bind the exact published proxy."
        )
    chaser_authority = relative.source_authorities["chaser_position"]
    expected_chaser_authority = {
        "source_authority_id": proxy.run_path,
        "source_digest": proxy.manifest_sha256,
        "provider_id": proxy.acquisition_projection_record["policy_id"],
        "provider_digest": proxy.acquisition_projection_record_sha256,
    }
    if any(
        chaser_authority.get(field) != expected
        for field, expected in expected_chaser_authority.items()
    ):
        raise ChaserProxyCandidateReceiptError(
            "Relative-frame chaser authority differs from the exact proxy."
        )
    transform = context["arena_to_source_camera_transform"]["record"]
    if (
        transform.get("transform_policy_id") != COORDINATE_POLICY_ID
        or transform.get("from_coordinate_space") != "arena_relative_canvas_px"
        or transform.get("to_coordinate_space") != "source_camera_image_px"
        or transform.get("no_reflection_or_heuristic_flip") is not True
    ):
        raise ChaserProxyCandidateReceiptError(
            "Relative-frame transform authority is missing or directionally invalid."
        )
    if (
        relative.manifest["timing_policy"].get("policy_id") != TIMING_POLICY_ID
        or relative.manifest["timing_policy"].get("timestamp_field") is not None
        or np.any(relative.base_array("timestamp_valid"))
    ):
        raise ChaserProxyCandidateReceiptError(
            "Candidate incorrectly claims camera timestamp availability."
        )
    if relative.selector_eligible is not False or relative.selection != "none":
        raise ChaserProxyCandidateReceiptError(
            "Relative-frame candidate unexpectedly became selectable."
        )

    profile = load_chaser_analysis_profile(analysis_profile_path)
    modules = resolve_chaser_analysis_modules(profile)
    proxy_assessment = input_provenance_proxy_alignment_assessment(
        proxy_projection_sha256=proxy.acquisition_projection_record_sha256,
        proxy_run_path=proxy.run_path,
        proxy_manifest_sha256=proxy.manifest_sha256,
    )
    relative_evidence = {
        "relative_frame_run_path": relative.run_path,
        "relative_frame_manifest_sha256": relative.manifest_sha256,
        "relative_frame_payload_digest": relative.payload_digest,
        "relative_frame_selector_eligible": False,
        "body_frame_available": relative.body_available,
    }
    capability_assessments = (
        proxy_assessment,
        _ready("position_series", source=relative_evidence),
        _ready("positioned_chaser", source=relative_evidence),
        _ready(
            "temporal_authority",
            source={
                "timing_policy_id": TIMING_POLICY_ID,
                "camera_timestamp_available": False,
                "frame_domain_available": True,
            },
        ),
    )
    applicability = plan_chaser_profile_applicability(
        recording_id=proxy.recording_id,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_sha256=profile.sha256,
        profile_scope=profile.profile_scope,
        selected_modules=modules,
        capability_assessments=capability_assessments,
    )
    body = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "complete_selector_ineligible_candidate_chain",
        "analysis_zarr": str(archive),
        "recording_id": proxy.recording_id,
        "analysis_profile_path": str(
            Path(analysis_profile_path).expanduser().resolve()
        ),
        "software_authority": {
            "repository": "palette",
            "commit": palette_commit,
        },
        "native_source": {
            "run_path": native.run_path,
            "manifest_sha256": native.manifest_sha256,
            "verification_digest": native.verification_digest,
        },
        "input_provenance_proxy": {
            "run_path": proxy.run_path,
            "manifest_sha256": proxy.manifest_sha256,
            "projection_sha256": proxy.acquisition_projection_record_sha256,
            "verification_digest": proxy.verification_digest,
            "selector_eligible": False,
            "selection": "none",
        },
        "relative_frame": {
            "run_path": relative.run_path,
            "manifest_sha256": relative.manifest_sha256,
            "payload_digest": relative.payload_digest,
            "verification_digest": relative.verification_digest,
            "selector_eligible": False,
            "selection": "none",
            "body_extension_present": relative.body_available,
        },
        "applicability_plan": applicability.as_envelope(),
        "production_authority": False,
        "registry_update": False,
        "production_selector_activation": False,
        "scientific_use_class": "exploratory_controller_input_provenance_proxy",
        "physical_presentation_verified": False,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def validate_chaser_proxy_candidate_receipt(
    receipt: object,
    *,
    expected_analysis_zarr: str | Path | None = None,
) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ChaserProxyCandidateReceiptError("Candidate receipt must be one object.")
    body = dict(receipt)
    digest = body.pop("record_sha256", None)
    if digest != canonical_json_sha256(body):
        raise ChaserProxyCandidateReceiptError("Candidate receipt digest is stale.")
    if (
        body.get("schema_id") != RECEIPT_SCHEMA_ID
        or body.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or body.get("status") != "complete_selector_ineligible_candidate_chain"
        or body.get("production_authority") is not False
        or body.get("registry_update") is not False
        or body.get("production_selector_activation") is not False
        or body.get("physical_presentation_verified") is not False
        or body.get("scientific_use_class")
        != "exploratory_controller_input_provenance_proxy"
    ):
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt identity or non-production state is invalid."
        )
    software = body.get("software_authority")
    if (
        not isinstance(software, Mapping)
        or set(software) != {"repository", "commit"}
        or software.get("repository") != "palette"
    ):
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt has invalid software authority."
        )
    palette_commit = _commit(software.get("commit"))
    if expected_analysis_zarr is not None and Path(
        str(body.get("analysis_zarr"))
    ).resolve() != Path(expected_analysis_zarr).expanduser().resolve():
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt names another analysis archive."
        )
    recomputed = build_chaser_proxy_candidate_receipt(
        body["analysis_zarr"],
        proxy_run_name=str(body["input_provenance_proxy"]["run_path"]).rsplit(
            "/", 1
        )[-1],
        relative_frame_run_name=str(body["relative_frame"]["run_path"]).rsplit(
            "/", 1
        )[-1],
        analysis_profile_path=body["analysis_profile_path"],
        palette_commit=palette_commit,
        expected_recording_id=str(body["recording_id"]),
        expected_proxy_manifest_sha256=body["input_provenance_proxy"][
            "manifest_sha256"
        ],
        expected_relative_manifest_sha256=body["relative_frame"][
            "manifest_sha256"
        ],
    )
    if recomputed != dict(receipt):
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt or one dependency changed after publication."
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--proxy-run-name", required=True)
    parser.add_argument("--relative-frame-run-name", required=True)
    parser.add_argument("--analysis-profile", type=Path, required=True)
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument("--expected-recording-id")
    parser.add_argument("--expected-proxy-manifest-sha256")
    parser.add_argument("--expected-relative-manifest-sha256")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_chaser_proxy_candidate_receipt(
        args.analysis_zarr,
        proxy_run_name=args.proxy_run_name,
        relative_frame_run_name=args.relative_frame_run_name,
        analysis_profile_path=args.analysis_profile,
        palette_commit=args.palette_commit,
        expected_recording_id=args.expected_recording_id,
        expected_proxy_manifest_sha256=args.expected_proxy_manifest_sha256,
        expected_relative_manifest_sha256=args.expected_relative_manifest_sha256,
    )
    write_json_atomic(args.output_json, receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RECEIPT_SCHEMA_ID",
    "RECEIPT_SCHEMA_VERSION",
    "ChaserProxyCandidateReceiptError",
    "build_chaser_proxy_candidate_receipt",
    "validate_chaser_proxy_candidate_receipt",
]
