from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fisheye.analysis_workflows import chaser_proxy_candidate_receipt as receipt_module
from fisheye.analysis_workflows import chaser_relative_frame_source_handle as handle_module
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    ChaserRelativeFrameSourceHandleError,
    load_chaser_relative_frame_source_handle,
    load_chaser_relative_frame_source_handle_from_receipt,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_chaser_relative_frame_source_handle import (
    _publish_proxy_bound,
)


def _receipt_for(archive: Path, tmp_path: Path) -> dict[str, object]:
    deep = load_chaser_relative_frame_source_handle(
        archive,
        run_name="candidate-v1",
        expected_recording_id="recording-1",
    )
    projection = deep.context["acquisition_projection"]["record"]
    publication = deep.context["acquisition_projection_publication"]["record"]
    native_run_path = projection["source_run_path"]
    native_manifest = projection["source_manifest_sha256"]
    native_verification = projection["source_verification_digest"]
    body: dict[str, object] = {
        "schema_id": receipt_module.RECEIPT_SCHEMA_ID,
        "schema_version": receipt_module.RECEIPT_SCHEMA_VERSION,
        "status": "complete_selector_ineligible_candidate_chain",
        "analysis_zarr": str(archive.resolve()),
        "recording_id": "recording-1",
        "analysis_profile_path": str((tmp_path / "profile.yaml").resolve()),
        "software_authority": {"repository": "palette", "commit": "a" * 40},
        "native_source": {
            "run_path": native_run_path,
            "manifest_sha256": native_manifest,
            "verification_digest": native_verification,
        },
        "input_provenance_proxy": {
            "run_path": publication["run_path"],
            "manifest_sha256": publication["manifest_sha256"],
            "projection_sha256": publication[
                "acquisition_projection_record_sha256"
            ],
            "verification_digest": "b" * 64,
            "publication_binding": dict(publication),
            "source_run_path": native_run_path,
            "source_manifest_sha256": native_manifest,
            "source_verification_digest": native_verification,
            "selector_eligible": False,
            "selection": "none",
        },
        "relative_frame": {
            "run_path": deep.run_path,
            "manifest_sha256": deep.manifest_sha256,
            "payload_digest": deep.payload_digest,
            "verification_digest": deep.verification_digest,
            "selector_eligible": False,
            "selection": "none",
            "body_extension_present": deep.body_available,
            "completion": dict(deep.completion_authority),
            "metadata_equivalence": dict(deep.metadata_equivalence),
            "array_declarations": receipt_module._plain(
                deep.manifest["array_declarations"]
            ),
            "timing_policy": dict(deep.manifest["timing_policy"]),
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
        "applicability_plan": {},
        "production_authority": False,
        "registry_update": False,
        "production_selector_activation": False,
        "scientific_use_class": "exploratory_controller_input_provenance_proxy",
        "physical_presentation_verified": False,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def _resigned(receipt: dict[str, object]) -> dict[str, object]:
    body = deepcopy(receipt)
    body.pop("record_sha256")
    return {**body, "record_sha256": canonical_json_sha256(body)}


def test_receipt_backed_load_reads_arrays_without_dense_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    receipt = _receipt_for(archive, tmp_path)

    def _unexpected_hash(value: object) -> str:
        raise AssertionError("receipt-backed load must not rehash dense arrays")

    monkeypatch.setattr(handle_module, "array_values_sha256", _unexpected_hash)
    monkeypatch.setattr(
        receipt_module,
        "validate_chaser_proxy_candidate_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bounded load called the deep receipt validator")
        ),
    )

    handle = load_chaser_relative_frame_source_handle_from_receipt(
        archive,
        receipt=receipt,
        expected_recording_id="recording-1",
    )

    assert handle.verification_mode == "receipt_backed"
    assert handle.receipt_digest == receipt["record_sha256"]
    assert handle.verification_digest == receipt["relative_frame"][
        "verification_digest"
    ]
    assert handle.verification_authority["receipt_digest"] == receipt["record_sha256"]
    assert handle.base_array("relative_distance_px").shape[0] == handle.n_rows
    assert handle.base_array("relative_distance_px").flags.writeable is False
    handle.assert_current()


def test_deep_loader_remains_explicitly_hashing(tmp_path: Path, monkeypatch) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    calls: list[int] = []
    original = handle_module.array_values_sha256

    def _counting_hash(value: object) -> str:
        calls.append(1)
        return original(value)

    monkeypatch.setattr(handle_module, "array_values_sha256", _counting_hash)
    handle = load_chaser_relative_frame_source_handle(
        archive, run_name="candidate-v1", expected_recording_id="recording-1"
    )
    assert handle.verification_mode == "deep_audit"
    assert handle.receipt_digest is None
    assert calls


@pytest.mark.parametrize(
    ("field_path", "value", "match"),
    [
        (("relative_frame", "manifest_sha256"), "0" * 64, "manifest digest"),
        (("relative_frame", "completion", "epoch"), 1, "completion"),
        (
            ("relative_frame", "metadata_equivalence", "array_count"),
            99,
            "metadata equivalence",
        ),
        (
            ("relative_frame", "temporal_caveats", "camera_exposure_reference"),
            "midpoint",
            "temporal",
        ),
        (("relative_frame", "selector_eligible"), True, "selector"),
    ],
)
def test_receipt_backed_load_rejects_resigned_stale_authority(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    candidate = _receipt_for(archive, tmp_path)
    target: object = candidate
    for key in field_path[:-1]:
        target = target[key]  # type: ignore[index]
    target[field_path[-1]] = value  # type: ignore[index]
    candidate = _resigned(candidate)

    with pytest.raises(
        (ChaserRelativeFrameSourceHandleError, receipt_module.ChaserProxyCandidateReceiptError),
        match=match,
    ):
        load_chaser_relative_frame_source_handle_from_receipt(
            archive, receipt=candidate, expected_recording_id="recording-1"
        )


def test_receipt_backed_load_rejects_path_and_self_digest_tampering(
    tmp_path: Path,
) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    candidate = _receipt_for(archive, tmp_path)
    candidate["relative_frame"]["run_path"] = (
        "analysis/chaser_relative_frame_runs/latest"
    )
    candidate = _resigned(candidate)
    with pytest.raises(receipt_module.ChaserProxyCandidateReceiptError, match="concrete"):
        load_chaser_relative_frame_source_handle_from_receipt(
            archive, receipt=candidate, expected_recording_id="recording-1"
        )

    stale = _receipt_for(archive, tmp_path)
    stale["relative_frame"]["payload_digest"] = "f" * 64
    with pytest.raises(receipt_module.ChaserProxyCandidateReceiptError, match="stale"):
        load_chaser_relative_frame_source_handle_from_receipt(
            archive, receipt=stale, expected_recording_id="recording-1"
        )


def test_receipt_backed_load_rejects_current_array_declaration_tampering(
    tmp_path: Path,
) -> None:
    archive = _publish_proxy_bound(tmp_path, timestamps=False)
    candidate = _receipt_for(archive, tmp_path)
    candidate["relative_frame"]["array_declarations"][0]["shape"] = [999]
    candidate = _resigned(candidate)

    with pytest.raises(
        ChaserRelativeFrameSourceHandleError, match="array declarations"
    ):
        load_chaser_relative_frame_source_handle_from_receipt(
            archive, receipt=candidate, expected_recording_id="recording-1"
        )
