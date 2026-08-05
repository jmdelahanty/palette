from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.utils import publish_canonical_detection_successor_batch as batch
from fisheye.utils import preflight_canonical_detection_crops as crop_preflight


def _inspection(path: Path, identity: str) -> dict[str, object]:
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


def _plan(tmp_path: Path) -> dict[str, object]:
    archive = tmp_path / "one_analysis.zarr"
    payload: dict[str, object] = {
        "schema_id": batch.PLAN_SCHEMA_ID,
        "schema_version": batch.PLAN_SCHEMA_VERSION,
        "created_at_utc": "2026-08-05T12:00:00Z",
        "registry_path": str(tmp_path / "registry.sqlite"),
        "scope_paths": [str(tmp_path)],
        "path_contains": "Batman",
        "successor_run_id": "canonical_v3",
        "candidate_count": 1,
        "candidates": [
            {
                "analysis_zarr": str(archive.resolve()),
                "inspection": _inspection(archive, "recording_one"),
            }
        ],
    }
    return {
        **payload,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "plan_digest": canonical_json_sha256(payload),
    }


def test_validate_plan_rejects_recomputed_nested_tampering(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["candidates"][0]["inspection"]["successor_run_id"] = "other"
    payload = {
        key: value
        for key, value in plan.items()
        if key not in {"digest_algorithm", "plan_digest"}
    }
    plan["plan_digest"] = canonical_json_sha256(payload)

    errors = batch.validate_plan(plan)

    assert "candidate 0 successor run differs" in errors


def test_apply_plan_rechecks_frozen_inspection_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    drifted = dict(plan["candidates"][0]["inspection"])
    drifted["source_instance_key_sha256"] = f"{3:064x}"
    published = False

    monkeypatch.setattr(
        batch,
        "inspect_canonical_detection_successor_source",
        lambda **_kwargs: drifted,
    )

    def publish(**_kwargs):  # noqa: ANN003, ANN202
        nonlocal published
        published = True

    monkeypatch.setattr(batch, "publish_canonical_detection_successor", publish)

    with pytest.raises(RuntimeError, match="inspection drifted"):
        batch.apply_plan(
            plan,
            scratch_root=tmp_path / "scratch",
            receipt_root=tmp_path / "receipts",
        )

    assert published is False


def test_apply_plan_filters_and_writes_one_bound_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    expected = dict(plan["candidates"][0]["inspection"])
    observed_receipt: Path | None = None

    monkeypatch.setattr(
        batch,
        "inspect_canonical_detection_successor_source",
        lambda **_kwargs: expected,
    )

    def publish(**kwargs):  # noqa: ANN003, ANN202
        nonlocal observed_receipt
        observed_receipt = kwargs["result_json"]
        return {
            "recording_identity": "recording_one",
            "status": "complete",
        }

    monkeypatch.setattr(batch, "publish_canonical_detection_successor", publish)

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


def test_crop_cohort_preflight_aggregates_the_frozen_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    policy = CropGeometryPolicy(
        purpose="ordinary_zebrafish_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(348, 348),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )
    monkeypatch.setattr(
        crop_preflight,
        "inspect_canonical_detection_crop_preflight",
        lambda **_kwargs: {
            "dimensions": {"n_instances": 10},
            "padding": {
                "padded_row_count": 3,
                "max_padding_ltrb": [1, 2, 3, 4],
            },
        },
    )

    result = crop_preflight.inspect_cohort_plan(
        plan,
        policy=policy,
        allow_selector_ineligible_candidate=True,
    )

    assert result["archive_count"] == 1
    assert result["affected_archive_count"] == 1
    assert result["total_instance_count"] == 10
    assert result["total_padded_row_count"] == 3
    assert result["max_padding_ltrb"] == [1, 2, 3, 4]
    assert result["plan_digest"] == plan["plan_digest"]
    assert result["crop_zarr_writes"] is False
