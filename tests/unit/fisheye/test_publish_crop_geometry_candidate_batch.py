from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils import publish_crop_geometry_candidate_batch as batch


def _policy() -> CropGeometryPolicy:
    return CropGeometryPolicy(
        purpose="zebrafish_pose_subject_mask_input",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(348, 348),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )


def _activation_plan(tmp_path: Path) -> dict[str, object]:
    archive = tmp_path / "recording_analysis.zarr"
    return {
        "plan_digest": "a" * 64,
        "run_id": "refined_v2",
        "candidates": [
            {
                "analysis_zarr": str(archive.resolve()),
                "inspection": {
                    "analysis_zarr": str(archive.resolve()),
                    "recording_identity": "recording_one",
                    "run_id": "refined_v2",
                    "activation_manifest_digest": "b" * 64,
                    "logical_content_digest": "c" * 64,
                    "publication_owner_uuid": (
                        "11111111-1111-4111-8111-111111111111"
                    ),
                    "storage_profile_id": "detection_published_access_aware_v1",
                },
            }
        ],
    }


def _active(archive: Path) -> dict[str, object]:
    return {
        "analysis_zarr": str(archive.resolve()),
        "recording_identity": "recording_one",
        "run_id": "refined_v2",
        "manifest_digest": "b" * 64,
        "logical_content_digest": "c" * 64,
        "publication_owner_uuid": "11111111-1111-4111-8111-111111111111",
        "storage_profile_id": "detection_published_access_aware_v1",
        "selection_mode": "approved_authoritative_refined_v1",
    }


def _preflight(archive: Path) -> dict[str, object]:
    return {
        "status": "ready",
        "analysis_zarr": str(archive.resolve()),
        "refined_run_id": "refined_v2",
        "selection_mode": "approved_authoritative_refined_v1",
        "policy": _policy().as_manifest(),
        "dimensions": {"n_instances": 2},
        "pixel_authority": {
            "binding_document_digest": "d" * 64,
            "source_video_path": str(archive.parent / "source.mp4"),
            "authority": {
                "camera_identity": "cam2010095",
            },
        },
        "padding": {
            "padded_row_count": 1,
            "fully_contained_row_count": 1,
            "max_padding_ltrb": [1, 0, 0, 0],
            "examples": [],
        },
        "array_content_sha256": {"instance_key": "e" * 64},
    }


def _plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, object]:
    activation = _activation_plan(tmp_path)
    archive = Path(str(activation["candidates"][0]["analysis_zarr"]))
    monkeypatch.setattr(batch, "validate_activation_plan", lambda _plan: ())
    monkeypatch.setattr(
        batch,
        "inspect_active_refined_detection_authority",
        lambda **_kwargs: _active(archive),
    )
    monkeypatch.setattr(
        batch,
        "inspect_refined_detection_crop_preflight",
        lambda **_kwargs: _preflight(archive),
    )
    return batch.build_plan(
        activation_plan=activation,
        crop_run_id="crop_v2_348",
        policy=_policy(),
    )


def test_build_plan_binds_active_authority_camera_and_zero_padding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(monkeypatch, tmp_path)

    assert plan["source_activation_plan_digest"] == "a" * 64
    assert plan["storage_profile_id"] == "published_http_v1"
    assert plan["candidates"][0]["camera_identity"] == "cam2010095"
    assert plan["policy"]["payload"]["placement"]["fixed_size_wh"] == [348, 348]
    assert plan["policy"]["payload"]["placement"]["padding_mode"] == (
        "zero_outside_source_frame"
    )
    assert batch.validate_plan(plan) == ()


def test_validate_plan_rejects_recomputed_camera_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(monkeypatch, tmp_path)
    plan["candidates"][0]["camera_identity"] = "other"
    plan["plan_digest"] = canonical_json_sha256(batch._plan_payload(plan))

    assert "crop candidate 0 camera binding differs" in batch.validate_plan(plan)


def test_apply_rechecks_evidence_and_writes_selector_ineligible_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(monkeypatch, tmp_path)
    candidate = plan["candidates"][0]
    captured = {}
    monkeypatch.setattr(
        batch,
        "inspect_active_refined_detection_authority",
        lambda **_kwargs: candidate["active_authority"],
    )
    monkeypatch.setattr(
        batch,
        "inspect_refined_detection_crop_preflight",
        lambda **_kwargs: candidate["preflight"],
    )

    def publish(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "status": "complete",
            "recording_identity": "recording_one",
            "run_id": kwargs["run_id"],
            "selector_eligible": False,
            "registry_updated": False,
        }

    monkeypatch.setattr(batch, "publish_crop_geometry_production_candidate", publish)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = batch.apply_plan(
        plan,
        scratch_root=scratch,
        receipt_root=tmp_path / "receipts",
        only_recording_identities=frozenset({"recording_one"}),
    )

    assert result["completed_candidate_count"] == 1
    assert result["selector_activation"] == "none"
    assert result["registry_updated"] is False
    assert captured["expected_camera_identity"] == "cam2010095"
    assert captured["profile"].profile_id == "published_http_v1"
    assert (tmp_path / "receipts/recording_one.json").is_file()

