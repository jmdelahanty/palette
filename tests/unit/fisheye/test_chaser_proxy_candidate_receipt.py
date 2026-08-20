from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis_workflows import chaser_proxy_candidate_receipt as receipt
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _fake_sources() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    projection = {
        "source_run_path": (
            "analysis/provider_chaser_distance_candidate_runs/native-v1"
        ),
        "source_manifest_sha256": "a" * 64,
        "policy_id": "latest_logged_cpu_state_per_input_acquisition_proxy_v1",
    }
    publication = {"run_path": "analysis/chaser_input_provenance_proxy_runs/proxy-v1"}
    proxy = SimpleNamespace(
        recording_id="recording-1",
        run_path="analysis/chaser_input_provenance_proxy_runs/proxy-v1",
        manifest_sha256="b" * 64,
        acquisition_projection_record=projection,
        acquisition_projection_record_sha256="c" * 64,
        publication_binding_record=publication,
        verification_digest="d" * 64,
    )
    native = SimpleNamespace(
        run_path="analysis/provider_chaser_distance_candidate_runs/native-v1",
        manifest_sha256="a" * 64,
        verification_digest="e" * 64,
    )
    relative = SimpleNamespace(
        run_path="analysis/chaser_relative_frame_runs/relative-v1",
        manifest_sha256="f" * 64,
        payload_digest="0" * 64,
        verification_digest="1" * 64,
        selector_eligible=False,
        selection="none",
        body_available=False,
        context={
            "acquisition_projection_publication": {"record": publication},
            "arena_to_source_camera_transform": {
                "record": {
                    "transform_policy_id": receipt.COORDINATE_POLICY_ID,
                    "from_coordinate_space": "arena_relative_canvas_px",
                    "to_coordinate_space": "source_camera_image_px",
                    "no_reflection_or_heuristic_flip": True,
                }
            },
        },
        source_authorities={
            "chaser_position": {
                "source_authority_id": proxy.run_path,
                "source_digest": proxy.manifest_sha256,
                "provider_id": projection["policy_id"],
                "provider_digest": proxy.acquisition_projection_record_sha256,
            }
        },
        manifest={
            "timing_policy": {
                "policy_id": receipt.TIMING_POLICY_ID,
                "timestamp_field": None,
            }
        },
        base_array=lambda name: np.zeros(2, dtype=bool),
    )
    return proxy, native, relative


def test_receipt_reopens_exact_chain_and_stays_non_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proxy, native, relative = _fake_sources()
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("profile_id: fixture\n", encoding="utf-8")
    checked: list[tuple[object, object]] = []
    assessments: list[object] = []

    monkeypatch.setattr(
        receipt,
        "load_chaser_input_provenance_proxy_source_handle",
        lambda *args, **kwargs: proxy,
    )
    monkeypatch.setattr(
        receipt,
        "load_provider_chaser_stimulus_source_handle",
        lambda *args, **kwargs: native,
    )
    monkeypatch.setattr(
        receipt,
        "load_chaser_relative_frame_source_handle",
        lambda *args, **kwargs: relative,
    )
    monkeypatch.setattr(
        receipt,
        "require_proxy_native_binding",
        lambda observed_proxy, observed_native: checked.append(
            (observed_proxy, observed_native)
        ),
    )
    profile = SimpleNamespace(
        profile_id="full",
        profile_version=3,
        sha256="2" * 64,
        profile_scope="full",
    )
    monkeypatch.setattr(receipt, "load_chaser_analysis_profile", lambda path: profile)
    monkeypatch.setattr(receipt, "resolve_chaser_analysis_modules", lambda value: ())

    class _Applicability:
        def as_envelope(self) -> dict[str, object]:
            return {"readiness": "blocked", "record_sha256": "3" * 64}

    def _plan(**kwargs: object) -> _Applicability:
        assessments.extend(kwargs["capability_assessments"])
        return _Applicability()

    monkeypatch.setattr(receipt, "plan_chaser_profile_applicability", _plan)

    result = receipt.build_chaser_proxy_candidate_receipt(
        archive,
        proxy_run_name="proxy-v1",
        relative_frame_run_name="relative-v1",
        analysis_profile_path=profile_path,
        palette_commit="4" * 40,
    )

    assert checked == [(proxy, native)]
    assert [item.capability_id for item in assessments] == [
        "chaser_temporal_alignment",
        "position_series",
        "positioned_chaser",
        "temporal_authority",
    ]
    assert result["status"] == "complete_selector_ineligible_candidate_chain"
    assert result["production_authority"] is False
    assert result["registry_update"] is False
    assert result["production_selector_activation"] is False
    assert result["physical_presentation_verified"] is False
    assert result["software_authority"] == {
        "repository": "palette",
        "commit": "4" * 40,
    }
    body = dict(result)
    digest = body.pop("record_sha256")
    assert digest == canonical_json_sha256(body)


def test_receipt_digest_tampering_fails_before_reopen() -> None:
    body = {
        "schema_id": receipt.RECEIPT_SCHEMA_ID,
        "schema_version": receipt.RECEIPT_SCHEMA_VERSION,
        "status": "complete_selector_ineligible_candidate_chain",
        "analysis_zarr": "/tmp/archive.zarr",
        "production_authority": False,
        "registry_update": False,
        "production_selector_activation": False,
    }
    candidate = {**body, "record_sha256": canonical_json_sha256(body)}
    candidate["production_authority"] = True

    with pytest.raises(receipt.ChaserProxyCandidateReceiptError, match="digest"):
        receipt.validate_chaser_proxy_candidate_receipt(candidate)
