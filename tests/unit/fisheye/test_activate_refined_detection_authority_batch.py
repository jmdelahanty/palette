from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.utils import activate_refined_detection_authority_batch as activation
from fisheye.utils import publish_accept_all_refined_detection_batch as publication


def _source_inspection(path: Path) -> dict[str, object]:
    return {
        "analysis_zarr": str(path.resolve()),
        "recording_identity": "recording_one",
        "source": {"run_id": "canonical_v3"},
        "target": {"run_id": "refined_accept_all_v2"},
    }


def _publication_plan(tmp_path: Path) -> dict[str, object]:
    archive = tmp_path / "one_analysis.zarr"
    payload: dict[str, object] = {
        "schema_id": publication.PLAN_SCHEMA_ID,
        "schema_version": publication.PLAN_SCHEMA_VERSION,
        "created_at_utc": "2026-08-05T12:00:00+00:00",
        "canonical_successor_plan_digest": "a" * 64,
        "canonical_run_id": "canonical_v3",
        "refined_run_id": "refined_accept_all_v2",
        "candidate_count": 1,
        "candidates": [
            {
                "analysis_zarr": str(archive.resolve()),
                "inspection": _source_inspection(archive),
            }
        ],
    }
    return {
        **payload,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "plan_digest": canonical_json_sha256(payload),
    }


def _candidate_inspection(path: Path) -> dict[str, object]:
    return {
        "schema_id": "palette.refined_detection.authority_candidate_inspection",
        "schema_version": 1,
        "status": "ready",
        "analysis_zarr": str(path.resolve()),
        "recording_identity": "recording_one",
        "run_id": "refined_accept_all_v2",
        "manifest_digest_before": "b" * 64,
        "activation_manifest_digest": "c" * 64,
        "logical_content_digest": "d" * 64,
        "publication_owner_uuid": "11111111-1111-4111-8111-111111111111",
        "dimensions": {"n_frames": 4, "n_instances": 2},
        "storage_profile_id": "detection_published_access_aware_v1",
        "intended_use": "analysis",
        "authority_absent": True,
        "run_selector_eligible": False,
    }


def _activation_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, object]:
    source = _publication_plan(tmp_path)
    archive = Path(str(source["candidates"][0]["analysis_zarr"]))
    expected = _candidate_inspection(archive)
    monkeypatch.setattr(
        activation,
        "inspect_refined_detection_authority_candidate",
        lambda **_kwargs: expected,
    )
    return activation.build_plan(
        refined_publication_plan=source,
        approved_by="jeremy",
        approved_at_utc="2026-08-05T16:00:00+00:00",
        review_method="identity_preserving_accept_all_cohort_gate",
        git_sha="abcdef12",
    )


def test_build_plan_binds_frozen_refined_publication_cohort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _activation_plan(monkeypatch, tmp_path)

    assert plan["source_refined_publication_plan_digest"] == (
        _publication_plan(tmp_path)["plan_digest"]
    )
    assert plan["approval"]["intended_use"] == "analysis"
    assert plan["candidate_count"] == 1
    assert activation.validate_plan(plan) == ()


def test_validate_plan_rejects_recomputed_training_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _activation_plan(monkeypatch, tmp_path)
    plan["approval"]["intended_use"] = "training"
    plan["plan_digest"] = canonical_json_sha256(activation._plan_payload(plan))

    assert "activation intended_use must be analysis" in activation.validate_plan(plan)


def test_apply_plan_binds_receipt_to_frozen_inspection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _activation_plan(monkeypatch, tmp_path)
    received: dict[str, object] = {}

    def activate(**kwargs):  # noqa: ANN003, ANN202
        received.update(kwargs)
        return {
            "recording_identity": "recording_one",
            "run_id": "refined_accept_all_v2",
            "status": "complete",
        }

    monkeypatch.setattr(
        activation,
        "activate_refined_detection_authority",
        activate,
    )
    result = activation.apply_plan(
        plan,
        receipt_root=tmp_path / "receipts",
        only_recording_identities=frozenset({"recording_one"}),
    )

    assert result["completed_candidate_count"] == 1
    assert result["intended_use"] == "analysis"
    assert received["expected_inspection"] == plan["candidates"][0]["inspection"]
    assert received["approved_by"] == "jeremy"
    assert (tmp_path / "receipts/recording_one.json").is_file()

