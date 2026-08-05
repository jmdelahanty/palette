from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.utils import publish_accept_all_refined_detection_batch as batch
from fisheye.utils import publish_canonical_detection_successor_batch as canonical


def _canonical_inspection(path: Path, identity: str) -> dict[str, object]:
    return {
        "schema_id": "palette.canonical_detection_successor.source_inspection",
        "schema_version": 1,
        "status": "ready",
        "analysis_zarr": str(path.resolve()),
        "recording_identity": identity,
        "source_group_path": "detect_runs/source",
        "source_run_id": "source",
        "successor_run_id": "canonical_v3",
        "successor_group_path": "detect_runs/canonical_v3",
        "dimensions": {"n_frames": 4, "n_instances": 2},
        "source_instance_key_sha256": f"{1:064x}",
        "instance_key_policy": "preserved_from_source",
        "source_evidence": {"digest": f"{2:064x}"},
        "storage_profile_id": "detection_published_access_aware_v1",
        "coordinate_catalog": True,
        "selectors_before": {},
        "selector_eligible": False,
        "registry_updated": False,
    }


def _canonical_plan(tmp_path: Path) -> dict[str, object]:
    archive = tmp_path / "one_analysis.zarr"
    payload: dict[str, object] = {
        "schema_id": canonical.PLAN_SCHEMA_ID,
        "schema_version": canonical.PLAN_SCHEMA_VERSION,
        "created_at_utc": "2026-08-05T12:00:00Z",
        "registry_path": str(tmp_path / "registry.sqlite"),
        "scope_paths": [str(tmp_path)],
        "path_contains": "Batman",
        "successor_run_id": "canonical_v3",
        "candidate_count": 1,
        "candidates": [
            {
                "analysis_zarr": str(archive.resolve()),
                "inspection": _canonical_inspection(archive, "recording_one"),
            }
        ],
    }
    return {
        **payload,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "plan_digest": canonical_json_sha256(payload),
    }


def _refined_inspection(path: Path, identity: str) -> dict[str, object]:
    digest = f"{3:064x}"
    return {
        "schema_id": "palette.accept_all_refined_detection.source_inspection",
        "schema_version": 1,
        "status": "ready",
        "analysis_zarr": str(path.resolve()),
        "recording_identity": identity,
        "source": {
            "run_id": "canonical_v3",
            "instance_key_sha256": digest,
        },
        "target": {
            "run_id": "refined_accept_all_v2",
            "instance_key_sha256": digest,
        },
    }


def _refined_plan(tmp_path: Path) -> dict[str, object]:
    source = _canonical_plan(tmp_path)
    archive = Path(str(source["candidates"][0]["analysis_zarr"]))
    payload: dict[str, object] = {
        "schema_id": batch.PLAN_SCHEMA_ID,
        "schema_version": batch.PLAN_SCHEMA_VERSION,
        "created_at_utc": "2026-08-05T13:00:00Z",
        "canonical_successor_plan_digest": source["plan_digest"],
        "canonical_run_id": "canonical_v3",
        "refined_run_id": "refined_accept_all_v2",
        "candidate_count": 1,
        "candidates": [
            {
                "analysis_zarr": str(archive.resolve()),
                "inspection": _refined_inspection(archive, "recording_one"),
            }
        ],
    }
    return {
        **payload,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "plan_digest": canonical_json_sha256(payload),
    }


def test_build_plan_binds_frozen_canonical_cohort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _canonical_plan(tmp_path)
    expected = _refined_inspection(
        Path(str(source["candidates"][0]["analysis_zarr"])),
        "recording_one",
    )
    monkeypatch.setattr(
        batch,
        "inspect_accept_all_refined_detection_source",
        lambda **_kwargs: expected,
    )

    plan = batch.build_plan(
        canonical_successor_plan=source,
        refined_run_id="refined_accept_all_v2",
    )

    assert plan["canonical_successor_plan_digest"] == source["plan_digest"]
    assert plan["candidate_count"] == 1
    assert batch.validate_plan(plan) == ()


def test_validate_plan_rejects_recomputed_run_tampering(tmp_path: Path) -> None:
    plan = _refined_plan(tmp_path)
    plan["candidates"][0]["inspection"]["target"]["run_id"] = "other"
    plan["plan_digest"] = canonical_json_sha256(batch._plan_payload(plan))

    assert "candidate 0 refined run differs" in batch.validate_plan(plan)


def test_apply_plan_rechecks_inspection_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _refined_plan(tmp_path)
    drifted = dict(plan["candidates"][0]["inspection"])
    drifted["status"] = "drifted"
    published = False
    monkeypatch.setattr(
        batch,
        "inspect_accept_all_refined_detection_source",
        lambda **_kwargs: drifted,
    )

    def publish(**_kwargs):  # noqa: ANN003, ANN202
        nonlocal published
        published = True

    monkeypatch.setattr(
        batch,
        "publish_accept_all_refined_detection_successor",
        publish,
    )
    with pytest.raises(RuntimeError, match="inspection drifted"):
        batch.apply_plan(
            plan,
            scratch_root=tmp_path / "scratch",
            receipt_root=tmp_path / "receipts",
        )
    assert published is False


def test_apply_plan_writes_one_bound_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _refined_plan(tmp_path)
    expected = dict(plan["candidates"][0]["inspection"])
    observed_receipt: Path | None = None
    monkeypatch.setattr(
        batch,
        "inspect_accept_all_refined_detection_source",
        lambda **_kwargs: expected,
    )

    def publish(**kwargs):  # noqa: ANN003, ANN202
        nonlocal observed_receipt
        observed_receipt = kwargs["result_json"]
        return {"recording_identity": "recording_one", "status": "complete"}

    monkeypatch.setattr(
        batch,
        "publish_accept_all_refined_detection_successor",
        publish,
    )
    result = batch.apply_plan(
        plan,
        scratch_root=tmp_path / "scratch",
        receipt_root=tmp_path / "receipts",
        only_recording_identities=frozenset({"recording_one"}),
    )
    assert result["completed_candidate_count"] == 1
    assert observed_receipt == tmp_path / "receipts/recording_one.json"
    assert result["selector_activation"] == "none"
    assert result["registry_updated"] is False
