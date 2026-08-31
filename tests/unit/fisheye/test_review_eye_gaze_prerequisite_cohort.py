from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.analysis_workflows.eye_gaze_source_handle import (
    validate_gaze_convention_review_receipt,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils import review_eye_gaze_prerequisite_cohort as review_mod
from fisheye.utils.materialize_composable_chaser_successor_cohort import (
    _load_eye_gaze_bindings,
)


HEX_A = "a" * 64
HEX_B = "b" * 64
PALETTE_COMMIT = "c" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _entry(index: int, root: Path) -> dict[str, object]:
    recording_id = f"recording_{index:03d}"
    return {
        "task_index": index,
        "recording_id": recording_id,
        "analysis_zarr": str(root / f"{recording_id}_analysis.zarr"),
        "outputs": {"eye_angle_run": "eye_angles_exact_v3"},
        "entry_sha256": f"{index:064x}",
    }


def _task(root: Path, count: int = 2) -> dict[str, object]:
    return {
        "task_sha256": HEX_A,
        "recording_count": count,
        "entries": [_entry(index, root) for index in range(1, count + 1)],
        "safety": dict(review_mod.PREREQUISITE_SAFETY),
    }


def _numeric(entry: dict[str, object], review_png: Path) -> dict[str, object]:
    run_name = entry["outputs"]["eye_angle_run"]  # type: ignore[index]
    return {
        "schema_id": "palette.gaze_convention_validation.v1",
        "schema_version": 1,
        "status": "pass",
        "read_only": True,
        "zarr_path": entry["analysis_zarr"],
        "eye_angle_run": run_name,
        "eye_angle_run_path": f"analysis/eye_angle_runs/{run_name}",
        "review_png": str(review_png.resolve()),
        "review_row_indices": [1, 4, 9],
        "checks": [{"name": "fixture", "passed": True}],
        "direction_assumption": {
            "name": "ellipse_axis_direction_assumption",
            "review_required": True,
        },
        "comparison_contract": {
            "coordinate_frame": "fish_body_frame",
            "zero": "fish_forward",
            "positive": "anatomical_left",
            "eye_angle_fields": [
                "left_gaze_signed_deg",
                "right_gaze_signed_deg",
            ],
        },
    }


def _write_member(root: Path, task: dict[str, object], index: int) -> Path:
    entry = task["entries"][index - 1]  # type: ignore[index]
    recording_id = entry["recording_id"]
    member = root / str(recording_id)
    member.mkdir(parents=True)
    png = member / "gaze_convention_review.png"
    png.write_bytes(f"png-{index}".encode())
    shape = {"schema_id": "shape", "status": "complete", "index": index}
    eye = {
        "schema_id": "eye",
        "status": "complete",
        "local_logical_manifest_sha256": f"{index + 100:064x}",
        "published_logical_manifest_sha256": f"{index + 100:064x}",
    }
    numeric = _numeric(entry, png)  # type: ignore[arg-type]
    _write_json(member / "subject_shape_result.json", shape)
    _write_json(member / "eye_angle_result.json", eye)
    _write_json(member / "gaze_convention_numeric_validation.json", numeric)
    body = {
        "schema_id": review_mod.MATERIALIZATION_RECEIPT_SCHEMA_ID,
        "schema_version": review_mod.MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": "2026-08-31T12:00:00+00:00",
        "task_sha256": task["task_sha256"],
        "entry_sha256": entry["entry_sha256"],
        "task_index": index,
        "recording_id": recording_id,
        "palette_commit": PALETTE_COMMIT,
        "rebinding_manifest_sha256": HEX_B,
        "subject_shape_result_sha256": canonical_json_sha256(shape),
        "eye_angle_result_sha256": canonical_json_sha256(eye),
        "numeric_validation_sha256": canonical_json_sha256(numeric),
        "review_png": str(png.resolve()),
        "review_png_sha256": review_mod._sha256_file(png),
        "human_gaze_direction_acceptance": False,
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "selector_activation": False,
    }
    receipt = {**body, "receipt_sha256": canonical_json_sha256(body)}
    path = member / "materialization_receipt.json"
    _write_json(path, receipt)
    return path


@pytest.fixture
def closed_cohort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    task = _task(tmp_path)
    smoke = tmp_path / "smoke"
    bulk = tmp_path / "bulk"
    _write_member(smoke, task, 1)
    _write_member(bulk, task, 2)
    monkeypatch.setattr(review_mod, "load_prerequisite_task", lambda source: task)
    return task, smoke, bulk


def _review_task(closed_cohort) -> dict[str, object]:
    task, smoke, bulk = closed_cohort
    return review_mod.build_review_task(
        task,
        receipt_roots=[smoke, bulk],
        eye_channel_variant="smoothed",
    )


def _accepted_decisions(review_task: dict[str, object]) -> dict[str, object]:
    decisions = review_mod.build_decision_template(review_task)
    decisions["reviewer"] = "reviewer@example.org"
    decisions["reviewed_at_utc"] = "2026-08-31T18:00:00+00:00"
    for row in decisions["entries"]:
        row["decision"] = "accepted"
    return decisions


def _rewrite_acceptance_manifest(output: Path, manifest: dict[str, object]) -> None:
    body = dict(manifest)
    body.pop("acceptance_sha256", None)
    _write_json(
        output / "acceptance_manifest.json",
        {**body, "acceptance_sha256": canonical_json_sha256(body)},
    )


def _rewrite_accepted_receipt(
    output: Path,
    manifest: dict[str, object],
    *,
    entry_index: int,
    receipt: dict[str, object],
) -> None:
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256", None)
    rewritten = {
        **receipt_body,
        "receipt_sha256": canonical_json_sha256(receipt_body),
    }
    accepted_entry = manifest["entries"][entry_index]
    receipt_path = Path(accepted_entry["convention_receipt"])
    _write_json(receipt_path, rewritten)
    accepted_entry["convention_receipt_file_sha256"] = review_mod._sha256_file(
        receipt_path
    )
    accepted_entry["convention_receipt_sha256"] = rewritten["receipt_sha256"]
    _rewrite_acceptance_manifest(output, manifest)


def test_review_task_and_template_remain_pending(closed_cohort) -> None:
    review_task = _review_task(closed_cohort)

    assert review_task["recording_count"] == 2
    assert review_task["review_status"] == review_mod.PENDING
    assert review_task["materialization_palette_commit"] == PALETTE_COMMIT
    assert review_task["review_task_sha256"] == review_mod._review_task_digest(
        review_task
    )
    assert [entry["review_status"] for entry in review_task["entries"]] == [
        review_mod.PENDING,
        review_mod.PENDING,
    ]

    template = review_mod.build_decision_template(review_task)
    assert template["reviewer"] == ""
    assert template["reviewed_at_utc"] == ""
    assert [row["decision"] for row in template["entries"]] == [
        "pending",
        "pending",
    ]


def test_review_task_loader_validates_subject_shape_binding(closed_cohort) -> None:
    review_task = _review_task(closed_cohort)
    review_task["entries"][0]["subject_shape_result"]["path"] = "relative.json"
    review_task["review_task_sha256"] = review_mod._review_task_digest(review_task)

    with pytest.raises(
        review_mod.EyeGazeCohortReviewError,
        match="canonical and absolute",
    ):
        review_mod.load_review_task(review_task)


def test_review_task_and_decisions_require_utc(closed_cohort, tmp_path: Path) -> None:
    review_task = _review_task(closed_cohort)
    review_task["created_at_utc"] = "2026-08-31T14:00:00-04:00"
    review_task["review_task_sha256"] = review_mod._review_task_digest(review_task)
    with pytest.raises(review_mod.EyeGazeCohortReviewError, match="must use UTC"):
        review_mod.load_review_task(review_task)

    review_task = _review_task(closed_cohort)
    decisions = _accepted_decisions(review_task)
    decisions["reviewed_at_utc"] = "2026-08-31T14:00:00-04:00"
    with pytest.raises(review_mod.EyeGazeCohortReviewError, match="must use UTC"):
        review_mod.accept_reviewed_cohort(
            review_task,
            decisions=decisions,
            output_root=tmp_path / "accepted-non-utc",
        )


def test_review_plan_requires_complete_exact_receipt_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path)
    receipts = tmp_path / "receipts"
    _write_member(receipts, task, 1)
    monkeypatch.setattr(review_mod, "load_prerequisite_task", lambda source: task)

    with pytest.raises(review_mod.EyeGazeCohortReviewError, match="exactly one"):
        review_mod.build_review_task(task, receipt_roots=[receipts])

    _write_member(receipts, task, 2)
    duplicate = tmp_path / "duplicate"
    _write_member(duplicate, task, 2)
    with pytest.raises(review_mod.EyeGazeCohortReviewError, match="exactly one"):
        review_mod.build_review_task(task, receipt_roots=[receipts, duplicate])


def test_review_plan_rejects_changed_png(closed_cohort) -> None:
    task, smoke, bulk = closed_cohort
    png = smoke / "recording_001" / "gaze_convention_review.png"
    png.write_bytes(b"changed")

    with pytest.raises(review_mod.EyeGazeCohortReviewError, match="Review PNG differs"):
        review_mod.build_review_task(task, receipt_roots=[smoke, bulk])


def test_accept_publishes_atomic_exact_receipts_and_bindings(
    closed_cohort, tmp_path: Path
) -> None:
    review_task = _review_task(closed_cohort)
    decisions = _accepted_decisions(review_task)
    output = tmp_path / "accepted"

    manifest = review_mod.accept_reviewed_cohort(
        review_task,
        decisions=decisions,
        output_root=output,
    )
    validated = review_mod.validate_acceptance_bundle(
        review_task, acceptance_root=output
    )

    assert manifest["status"] == "complete"
    assert manifest["recording_count"] == 2
    assert validated["status"] == "valid"
    bindings = json.loads((output / "eye_gaze_bindings.json").read_text())
    assert [row["recording_id"] for row in bindings] == [
        "recording_001",
        "recording_002",
    ]
    for task_entry, accepted in zip(review_task["entries"], manifest["entries"]):
        receipt = json.loads(Path(accepted["convention_receipt"]).read_text())
        assert receipt["biological_direction_review"]["status"] == "accepted"
        validate_gaze_convention_review_receipt(
            receipt,
            expected_run_path=task_entry["eye_angle_run_path"],
            expected_logical_sha256=task_entry["source_eye_logical_sha256"],
        )
        _write_json(
            Path(task_entry["analysis_zarr"])
            / task_entry["eye_angle_run_path"]
            / "zarr.json",
            {"attributes": {}},
        )

    frozen, source = _load_eye_gaze_bindings(output / "eye_gaze_bindings.json")
    assert set(frozen) == {"recording_001", "recording_002"}
    assert source["row_count"] == 2


@pytest.mark.parametrize("decision", ["pending", "rejected"])
def test_accept_requires_every_explicit_acceptance(
    closed_cohort, tmp_path: Path, decision: str
) -> None:
    review_task = _review_task(closed_cohort)
    decisions = _accepted_decisions(review_task)
    decisions["entries"][1]["decision"] = decision
    output = tmp_path / f"accepted-{decision}"

    with pytest.raises(review_mod.EyeGazeCohortReviewError):
        review_mod.accept_reviewed_cohort(
            review_task,
            decisions=decisions,
            output_root=output,
        )
    assert not output.exists()


def test_accept_rejects_wrong_review_hash(closed_cohort, tmp_path: Path) -> None:
    review_task = _review_task(closed_cohort)
    decisions = _accepted_decisions(review_task)
    decisions["entries"][0]["review_png_sha256"] = HEX_A
    output = tmp_path / "accepted"

    with pytest.raises(review_mod.EyeGazeCohortReviewError, match="does not bind"):
        review_mod.accept_reviewed_cohort(
            review_task,
            decisions=decisions,
            output_root=output,
        )
    assert not output.exists()


def test_accept_revalidates_sources_after_review_plan(
    closed_cohort, tmp_path: Path
) -> None:
    review_task = _review_task(closed_cohort)
    decisions = _accepted_decisions(review_task)
    numeric = Path(review_task["entries"][0]["numeric_validation"]["path"])
    numeric.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "accepted"

    with pytest.raises(review_mod.EyeGazeCohortReviewError, match="changed"):
        review_mod.accept_reviewed_cohort(
            review_task,
            decisions=decisions,
            output_root=output,
        )
    assert not output.exists()


def test_acceptance_validation_rejects_modified_bindings(
    closed_cohort, tmp_path: Path
) -> None:
    review_task = _review_task(closed_cohort)
    output = tmp_path / "accepted"
    review_mod.accept_reviewed_cohort(
        review_task,
        decisions=_accepted_decisions(review_task),
        output_root=output,
    )
    bindings = output / "eye_gaze_bindings.json"
    bindings.write_text("[]\n", encoding="utf-8")

    with pytest.raises(
        review_mod.EyeGazeCohortReviewError, match="bindings file changed"
    ):
        review_mod.validate_acceptance_bundle(review_task, acceptance_root=output)


def test_acceptance_validation_rejects_cross_bound_numeric_validation(
    closed_cohort, tmp_path: Path
) -> None:
    review_task = _review_task(closed_cohort)
    output = tmp_path / "accepted"
    manifest = review_mod.accept_reviewed_cohort(
        review_task,
        decisions=_accepted_decisions(review_task),
        output_root=output,
    )
    receipt_path = Path(manifest["entries"][0]["convention_receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["numeric_validation"]["checks"].append(
        {"name": "forged-extra-pass", "passed": True}
    )
    receipt["numeric_validation_sha256"] = canonical_json_sha256(
        receipt["numeric_validation"]
    )
    _rewrite_accepted_receipt(
        output,
        manifest,
        entry_index=0,
        receipt=receipt,
    )

    with pytest.raises(
        review_mod.EyeGazeCohortReviewError,
        match="frozen numeric validation",
    ):
        review_mod.validate_acceptance_bundle(review_task, acceptance_root=output)


def test_acceptance_validation_rejects_cross_bound_review_png(
    closed_cohort, tmp_path: Path
) -> None:
    review_task = _review_task(closed_cohort)
    output = tmp_path / "accepted"
    manifest = review_mod.accept_reviewed_cohort(
        review_task,
        decisions=_accepted_decisions(review_task),
        output_root=output,
    )
    receipt_path = Path(manifest["entries"][0]["convention_receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["biological_direction_review"]["review_artifact_sha256"] = HEX_A
    _rewrite_accepted_receipt(
        output,
        manifest,
        entry_index=0,
        receipt=receipt,
    )

    with pytest.raises(
        review_mod.EyeGazeCohortReviewError,
        match="frozen review PNG evidence",
    ):
        review_mod.validate_acceptance_bundle(review_task, acceptance_root=output)


def test_acceptance_validation_rejects_non_utc_manifest_time(
    closed_cohort, tmp_path: Path
) -> None:
    review_task = _review_task(closed_cohort)
    output = tmp_path / "accepted"
    manifest = review_mod.accept_reviewed_cohort(
        review_task,
        decisions=_accepted_decisions(review_task),
        output_root=output,
    )
    manifest["reviewed_at_utc"] = "2026-08-31T14:00:00-04:00"
    _rewrite_acceptance_manifest(output, manifest)

    with pytest.raises(review_mod.EyeGazeCohortReviewError, match="must use UTC"):
        review_mod.validate_acceptance_bundle(review_task, acceptance_root=output)


def test_acceptance_bundle_is_never_overwritten(closed_cohort, tmp_path: Path) -> None:
    review_task = _review_task(closed_cohort)
    output = tmp_path / "accepted"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        review_mod.accept_reviewed_cohort(
            review_task,
            decisions=_accepted_decisions(review_task),
            output_root=output,
        )
