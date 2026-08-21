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
# Version 2 adds the bounded source-loader authority table.  Version 1
# receipts remain valid historical deep-audit records but are intentionally
# ineligible for receipt-backed loading because they do not bind declarations
# or completion/metadata evidence.
RECEIPT_SCHEMA_VERSION = 2
_SHA256_HEX = frozenset("0123456789abcdef")
_SELECTOR_NAMES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "active",
        "active_run",
        "current",
        "current_run",
        "default",
        "default_run",
        "selected",
        "selected_run",
    }
)


class ChaserProxyCandidateReceiptError(ValueError):
    """Raised when one candidate chain or receipt is incomplete or stale."""


def _plain(value: Any) -> Any:
    """Return JSON-native values from a handle's frozen authority records."""

    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ChaserProxyCandidateReceiptError(
            f"{field} must be one lowercase SHA-256 digest."
        )
    return value


def _exact_path(value: object, *, prefix: str, field: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise ChaserProxyCandidateReceiptError(
            f"{field} must name one exact child of {prefix!r}."
        )
    name = value[len(prefix) :]
    if not name or "/" in name or name in {".", ".."} or name in _SELECTOR_NAMES:
        raise ChaserProxyCandidateReceiptError(
            f"{field} must name one concrete run, not a selector or path."
        )
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ChaserProxyCandidateReceiptError(f"{field} must be one JSON object.")
    return value


def _strict_temporal_caveats(value: object) -> dict[str, Any]:
    caveats = dict(_mapping(value, field="relative_frame.temporal_caveats"))
    expected = {
        "physical_presentation_verified": False,
        "presentation_timestamp_available": False,
        "camera_presentation_clock_transform_available": False,
        "camera_exposure_reference": "unknown",
        "scientific_use_class": "exploratory_controller_input_provenance_proxy",
    }
    if caveats != expected:
        raise ChaserProxyCandidateReceiptError(
            "Receipt temporal proxy caveats are missing, changed, or optimistic."
        )
    return caveats


def _strict_array_declarations(value: object) -> list[dict[str, Any]]:
    raw = value
    if not isinstance(raw, list) or not raw:
        raise ChaserProxyCandidateReceiptError(
            "relative_frame.array_declarations must be a non-empty list."
        )
    result: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, declaration in enumerate(raw):
        item = dict(_mapping(declaration, field=f"array_declarations[{index}]"))
        if set(item) != {"path", "dtype", "shape", "content_sha256"}:
            raise ChaserProxyCandidateReceiptError(
                "Receipt array declarations must contain exactly path, dtype, "
                "shape, and content_sha256."
            )
        path = item["path"]
        if (
            type(path) is not str
            or not path
            or path in paths
            or path.count("/") != 1
            or path.split("/", 1)[0] not in {"base", "body"}
        ):
            raise ChaserProxyCandidateReceiptError(
                f"Receipt array declaration path is invalid: {path!r}."
            )
        shape = item["shape"]
        if (
            not isinstance(shape, list)
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise ChaserProxyCandidateReceiptError(
                f"Receipt array declaration shape is invalid for {path!r}."
            )
        if type(item["dtype"]) is not str or not item["dtype"]:
            raise ChaserProxyCandidateReceiptError(
                f"Receipt array declaration dtype is invalid for {path!r}."
            )
        _digest(item["content_sha256"], field=f"{path}.content_sha256")
        paths.add(path)
        result.append(item)
    expected_order = sorted(path for path in paths if path.startswith("base/")) + sorted(
        path for path in paths if path.startswith("body/")
    )
    if [item["path"] for item in result] != expected_order:
        raise ChaserProxyCandidateReceiptError(
            "Receipt array declarations are not in canonical base/body order."
        )
    return result


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
            "publication_binding": _plain(proxy.publication_binding_record),
            "source_run_path": proxy.acquisition_projection_record[
                "source_run_path"
            ],
            "source_manifest_sha256": proxy.acquisition_projection_record[
                "source_manifest_sha256"
            ],
            "source_verification_digest": proxy.acquisition_projection_record[
                "source_verification_digest"
            ],
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
            "completion": _plain(relative.completion_authority),
            "metadata_equivalence": _plain(relative.metadata_equivalence),
            "array_declarations": _plain(relative.manifest["array_declarations"]),
            "timing_policy": _plain(relative.manifest["timing_policy"]),
            "temporal_caveats": {
                "physical_presentation_verified": False,
                "presentation_timestamp_available": False,
                "camera_presentation_clock_transform_available": False,
                "camera_exposure_reference": "unknown",
                "scientific_use_class": (
                    "exploratory_controller_input_provenance_proxy"
                ),
            },
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


def validate_chaser_proxy_candidate_receipt_for_source_load(
    receipt: object,
    *,
    expected_analysis_zarr: str | Path | None = None,
    expected_recording_id: str | None = None,
) -> Mapping[str, Any]:
    """Validate only the sealed receipt contract needed by a bounded source load.

    This intentionally does not reopen any dependency and does not call
    :func:`validate_chaser_proxy_candidate_receipt`.  The latter remains the
    explicit deep-audit validator that recomputes the entire dependency chain.
    The source loader validates the current relative-frame metadata and reads
    its arrays once after this bounded receipt envelope has passed.
    """

    if not isinstance(receipt, Mapping):
        raise ChaserProxyCandidateReceiptError("Candidate receipt must be one object.")
    body = dict(receipt)
    digest = body.pop("record_sha256", None)
    _digest(digest, field="record_sha256")
    if digest != canonical_json_sha256(body):
        raise ChaserProxyCandidateReceiptError("Candidate receipt digest is stale.")
    required = {
        "schema_id",
        "schema_version",
        "status",
        "analysis_zarr",
        "recording_id",
        "analysis_profile_path",
        "software_authority",
        "native_source",
        "input_provenance_proxy",
        "relative_frame",
        "applicability_plan",
        "production_authority",
        "registry_update",
        "production_selector_activation",
        "scientific_use_class",
        "physical_presentation_verified",
    }
    if set(body) != required:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt has missing or unexpected bounded-load fields."
        )
    if (
        body["schema_id"] != RECEIPT_SCHEMA_ID
        or body["schema_version"] != RECEIPT_SCHEMA_VERSION
        or body["status"] != "complete_selector_ineligible_candidate_chain"
        or body["production_authority"] is not False
        or body["registry_update"] is not False
        or body["production_selector_activation"] is not False
        or body["scientific_use_class"]
        != "exploratory_controller_input_provenance_proxy"
        or body["physical_presentation_verified"] is not False
    ):
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt identity or non-production state is invalid."
        )
    if type(body["recording_id"]) is not str or not body["recording_id"].strip():
        raise ChaserProxyCandidateReceiptError("Candidate receipt recording_id is invalid.")
    if expected_recording_id is not None and body["recording_id"] != expected_recording_id:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt recording_id differs from the requested recording."
        )
    archive_text = body["analysis_zarr"]
    if type(archive_text) is not str or not archive_text:
        raise ChaserProxyCandidateReceiptError("Candidate receipt analysis_zarr is invalid.")
    archive = str(Path(archive_text).expanduser().resolve())
    if archive != archive_text:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt analysis_zarr is not the canonical absolute archive path."
        )
    if expected_analysis_zarr is not None and archive != str(
        Path(expected_analysis_zarr).expanduser().resolve()
    ):
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt names another analysis archive."
        )
    if type(body["analysis_profile_path"]) is not str or not body["analysis_profile_path"]:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt analysis_profile_path is invalid."
        )
    software = _mapping(body["software_authority"], field="software_authority")
    if set(software) != {"repository", "commit"} or software["repository"] != "palette":
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt has invalid software authority."
        )
    _commit(software["commit"])

    native = _mapping(body["native_source"], field="native_source")
    if set(native) != {"run_path", "manifest_sha256", "verification_digest"}:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt native source authority is incomplete."
        )
    _exact_path(
        native["run_path"],
        prefix="analysis/provider_chaser_distance_candidate_runs/",
        field="native_source.run_path",
    )
    _digest(native["manifest_sha256"], field="native_source.manifest_sha256")
    _digest(native["verification_digest"], field="native_source.verification_digest")

    proxy = _mapping(body["input_provenance_proxy"], field="input_provenance_proxy")
    proxy_required = {
        "run_path",
        "manifest_sha256",
        "projection_sha256",
        "verification_digest",
        "publication_binding",
        "source_run_path",
        "source_manifest_sha256",
        "source_verification_digest",
        "selector_eligible",
        "selection",
    }
    if set(proxy) != proxy_required:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt proxy authority is incomplete."
        )
    _exact_path(
        proxy["run_path"],
        prefix="analysis/chaser_input_provenance_proxy_runs/",
        field="input_provenance_proxy.run_path",
    )
    for field in (
        "manifest_sha256",
        "projection_sha256",
        "verification_digest",
        "source_manifest_sha256",
        "source_verification_digest",
    ):
        _digest(proxy[field], field=f"input_provenance_proxy.{field}")
    _exact_path(
        proxy["source_run_path"],
        prefix="analysis/provider_chaser_distance_candidate_runs/",
        field="input_provenance_proxy.source_run_path",
    )
    if proxy["selector_eligible"] is not False or proxy["selection"] != "none":
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt proxy is not selector-ineligible with selection=none."
        )
    publication = _mapping(
        proxy["publication_binding"],
        field="input_provenance_proxy.publication_binding",
    )
    if publication.get("run_path") != proxy["run_path"] or publication.get(
        "manifest_sha256"
    ) != proxy["manifest_sha256"]:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt proxy publication binding is stale."
        )
    if publication.get("selector_eligible") is not False or publication.get(
        "selection"
    ) != "none":
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt proxy publication binding is selectable."
        )

    relative = _mapping(body["relative_frame"], field="relative_frame")
    relative_required = {
        "run_path",
        "manifest_sha256",
        "payload_digest",
        "verification_digest",
        "selector_eligible",
        "selection",
        "body_extension_present",
        "completion",
        "metadata_equivalence",
        "array_declarations",
        "timing_policy",
        "temporal_caveats",
    }
    if set(relative) != relative_required:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt relative-frame authority is incomplete."
        )
    _exact_path(
        relative["run_path"],
        prefix="analysis/chaser_relative_frame_runs/",
        field="relative_frame.run_path",
    )
    for field in ("manifest_sha256", "payload_digest", "verification_digest"):
        _digest(relative[field], field=f"relative_frame.{field}")
    if relative["selector_eligible"] is not False or relative["selection"] != "none":
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt relative frame is not selector-ineligible with selection=none."
        )
    if type(relative["body_extension_present"]) is not bool:
        raise ChaserProxyCandidateReceiptError(
            "relative_frame.body_extension_present must be an exact boolean."
        )
    completion = dict(_mapping(relative["completion"], field="relative_frame.completion"))
    if set(completion) != {"contract", "status", "epoch"}:
        raise ChaserProxyCandidateReceiptError(
            "relative_frame.completion must declare contract, status, and epoch."
        )
    if completion["contract"] != "palette.zarr_run_completion.v1" or completion[
        "status"
    ] != "complete" or type(completion["epoch"]) is not int or completion["epoch"] < 2:
        raise ChaserProxyCandidateReceiptError(
            "relative_frame.completion is not a strict provenance-bearing completion."
        )
    metadata = dict(
        _mapping(relative["metadata_equivalence"], field="relative_frame.metadata_equivalence")
    )
    if set(metadata) != {
        "schema_id",
        "schema_version",
        "subtree_path",
        "node_count",
        "group_count",
        "array_count",
        "declarations_sha256",
    }:
        raise ChaserProxyCandidateReceiptError(
            "relative_frame.metadata_equivalence is incomplete."
        )
    if metadata["schema_id"] != "palette.zarr.metadata_equivalence" or metadata[
        "schema_version"
    ] != 1 or metadata["subtree_path"] != relative["run_path"]:
        raise ChaserProxyCandidateReceiptError(
            "relative_frame.metadata_equivalence is bound to another subtree."
        )
    for field in ("node_count", "group_count", "array_count"):
        if type(metadata[field]) is not int or metadata[field] < 1:
            raise ChaserProxyCandidateReceiptError(
                f"relative_frame.metadata_equivalence.{field} is invalid."
            )
    _digest(
        metadata["declarations_sha256"],
        field="relative_frame.metadata_equivalence.declarations_sha256",
    )
    _strict_array_declarations(relative["array_declarations"])
    timing = _mapping(relative["timing_policy"], field="relative_frame.timing_policy")
    if timing.get("timestamp_field") is not None:
        raise ChaserProxyCandidateReceiptError(
            "Receipt timing policy claims a camera timestamp for the proxy path."
        )
    _strict_temporal_caveats(relative["temporal_caveats"])
    if proxy["source_run_path"] != native["run_path"] or proxy[
        "source_manifest_sha256"
    ] != native["manifest_sha256"] or proxy["source_verification_digest"] != native[
        "verification_digest"
    ]:
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt proxy/native source binding is inconsistent."
        )
    if body["recording_id"] != publication.get("recording_id"):
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt recording identity is inconsistent."
        )
    # Metadata-equivalence receipts intentionally do not carry recording_id;
    # accept that omission while rejecting a contradictory future extension.
    if metadata.get("recording_id") not in (None, body["recording_id"]):
        raise ChaserProxyCandidateReceiptError(
            "Candidate receipt metadata recording identity is inconsistent."
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
    "validate_chaser_proxy_candidate_receipt_for_source_load",
]
