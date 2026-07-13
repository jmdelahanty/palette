from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.utils import promote_refined_subject_mask_run as mod


RUN_NAME = "refined_subject_masks_complete_test_v001"


def _make_archives(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.zarr"
    target = tmp_path / "target.zarr"
    source_root = zarr.open_group(str(source), mode="w", use_consolidated=False)
    source_parent = source_root.require_group(mod.RUN_PARENT)
    source_run = source_parent.create_group(RUN_NAME)
    source_run.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "palette_run_name": RUN_NAME,
        }
    )
    source_run.create_array(
        "masks_roi",
        data=np.asarray(
            [
                [[[1, 0], [0, 1]]],
                [[[0, 1], [1, 0]]],
            ],
            dtype=np.uint8,
        ),
        chunks=(1, 1, 2, 2),
    )
    source_run.create_array(
        "frame_indices",
        data=np.asarray([10, 11], dtype=np.int64),
        chunks=(2,),
    )

    target_root = zarr.open_group(str(target), mode="w", use_consolidated=False)
    target_parent = target_root.require_group(mod.RUN_PARENT)
    target_parent.attrs.update(
        {
            "palette_completion_epoch": 2,
            "latest": "old_run",
            "latest_complete": "old_run",
            mod.REVIEW_POINTER: "old_run",
        }
    )
    return source, target


def _valid_contract(*_args, **_kwargs):
    return {
        "valid": True,
        "error_count": 0,
        "warning_count": 0,
        "errors": [],
        "warnings": [],
    }


def test_copy_promotion_validates_then_updates_pointers_and_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, target = _make_archives(tmp_path)
    monkeypatch.setattr(mod, "validate_refined_subject_mask_contract", _valid_contract)

    planned = mod.promote_refined_subject_mask_run(
        source_zarr=source,
        source_run=RUN_NAME,
        target_zarr=target,
        expected_rows=2,
    )
    assert planned["status"] == "planned"
    assert not (target / mod.RUN_PARENT / RUN_NAME).exists()

    promoted = mod.promote_refined_subject_mask_run(
        source_zarr=source,
        source_run=RUN_NAME,
        target_zarr=target,
        expected_rows=2,
        apply=True,
    )

    target_root = zarr.open_group(str(target), mode="r", use_consolidated=False)
    parent = target_root[mod.RUN_PARENT]
    copied = parent[RUN_NAME]
    assert promoted["status"] == "promoted"
    assert promoted["array_validation"]["all_values_equal"] is True
    assert promoted["source_inventory"] == promoted["target_inventory"]
    assert np.array_equal(
        copied["masks_roi"][:],
        zarr.open_group(str(source), mode="r", use_consolidated=False)[
            f"{mod.RUN_PARENT}/{RUN_NAME}/masks_roi"
        ][:],
    )
    assert parent.attrs["latest"] == RUN_NAME
    assert parent.attrs["latest_complete"] == RUN_NAME
    assert parent.attrs[mod.REVIEW_POINTER] == RUN_NAME
    receipt = target / mod.RUN_PARENT / ".imports" / f"{RUN_NAME}.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "promoted"


def test_registry_failure_restores_previous_parent_pointers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, target = _make_archives(tmp_path)
    monkeypatch.setattr(mod, "validate_refined_subject_mask_contract", _valid_contract)
    mod.promote_refined_subject_mask_run(
        source_zarr=source,
        source_run=RUN_NAME,
        target_zarr=target,
        expected_rows=2,
        apply=True,
    )
    target_root = zarr.open_group(str(target), mode="a", use_consolidated=False)
    parent = target_root[mod.RUN_PARENT]
    parent.attrs.update(
        {
            "latest": "old_run",
            "latest_complete": "old_run",
            mod.REVIEW_POINTER: "old_run",
        }
    )
    monkeypatch.setattr(mod, "emit_refined_subject_mask_stage_completion", lambda *_a, **_k: False)

    with pytest.raises(RuntimeError, match="registry completion"):
        mod.promote_refined_subject_mask_run(
            source_zarr=source,
            source_run=RUN_NAME,
            target_zarr=target,
            expected_rows=2,
            registry_path=tmp_path / "registry.sqlite",
            apply=True,
            resume_existing=True,
        )

    refreshed = zarr.open_group(str(target), mode="r", use_consolidated=False)[
        mod.RUN_PARENT
    ]
    assert refreshed.attrs["latest"] == "old_run"
    assert refreshed.attrs["latest_complete"] == "old_run"
    assert refreshed.attrs[mod.REVIEW_POINTER] == "old_run"


def test_corrected_canary_evidence_must_be_a_complete_pass(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "classification": "pass",
                "finalizer_rerun_required": False,
                "publication_rerun_required": False,
                "corrected_validation": {"all_checks_pass": True},
                "dense_content_audit": {"all_checks_pass": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a complete pass"):
        mod._validate_evidence(evidence)


def test_isolated_source_may_defer_only_missing_crop_context(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "validate_refined_subject_mask_contract",
        lambda *_a, **_k: {
            "valid": False,
            "errors": [{"code": "missing_source_crop_run"}],
        },
    )

    summary = mod._source_contract_summary(Path("isolated.zarr"), RUN_NAME)

    assert summary["target_context_validation_deferred"] is True
