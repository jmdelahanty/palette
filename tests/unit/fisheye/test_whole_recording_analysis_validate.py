from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fisheye.cluster import whole_recording_analysis as planner
from fisheye.cluster import whole_recording_analysis_validate as mod


class FakeArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.oindex = self

    def __getitem__(self, key):  # noqa: ANN001, ANN204
        return self.values[key]


class EllipsisRejectingOIndex:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.last_key = None

    def __getitem__(self, key):  # noqa: ANN001, ANN204
        self.last_key = key
        if any(item is Ellipsis for item in key):
            raise AssertionError("orthogonal selection must not contain Ellipsis")
        return self.values[key]


class EllipsisRejectingArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.oindex = EllipsisRejectingOIndex(self.values)


class FakeGroup(dict):
    def __init__(self, *args, attrs=None, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})

    def get(self, key, default=None):  # noqa: ANN001, ANN201
        node = self
        for part in str(key).split("/"):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _complete_group(**attrs) -> FakeGroup:  # noqa: ANN003
    return FakeGroup(attrs={"palette_run_completion_status": "complete", **attrs})


def _root() -> FakeGroup:
    raw_probabilities = np.full((4, 3, 2, 2), 128, dtype=np.uint8)
    raw = _complete_group(
        label_schema_id="subject_v1_union",
        mask_labels=list(mod.RAW_LABELS),
    )
    raw["mask_probs_roi"] = FakeArray(raw_probabilities)
    raw["available_channels"] = FakeArray(np.ones((3,), dtype=bool))
    raw["metrics"] = FakeGroup({"mask_present": FakeArray(np.ones((4, 3), dtype=bool))})

    refined = _complete_group(
        mask_labels=list(mod.REFINED_LABELS),
        component_registry={
            "schema_id": "palette.subject_mask.component_registry",
            "schema_version": 1,
            "labels": list(mod.REFINED_LABELS),
        },
        logical_schema={
            "schema_id": "palette.stage.refined_subject_mask_dense_core",
            "schema_version": 1,
            "components": {
                "schema_id": "palette.subject_mask.component_registry",
                "schema_version": 1,
                "labels": list(mod.REFINED_LABELS),
            },
        },
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoints_run="refined_keypoints_test",
        sampled_component_contours_status="computed",
        sampled_component_contours_requested=True,
        component_contours_requested=False,
    )
    refined["masks_roi"] = FakeArray(np.ones((4, 4, 2, 2), dtype=np.uint8))
    refined["available_channels"] = FakeArray(np.ones((4,), dtype=bool))
    refined["metrics"] = FakeGroup(
        {"mask_present": FakeArray(np.ones((4, 4), dtype=bool))}
    )
    refined["components"] = FakeGroup(
        {
            label: FakeGroup({"sampled_contours": FakeArray(np.ones((4, 2)))})
            for label in mod.REFINED_LABELS
        }
    )
    return FakeGroup(
        {
            "keypoints_runs": FakeGroup({"keypoints_test": _complete_group()}),
            "refined_keypoints_runs": FakeGroup(
                {"refined_keypoints_test": _complete_group()}
            ),
            "subject_mask_runs": FakeGroup({"subject_masks_test": raw}),
            "refined_subject_masks_runs": FakeGroup(
                {"refined_subject_masks_test": refined}
            ),
        }
    )


def _write_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": planner.PLAN_SCHEMA,
                "targets": [
                    {
                        "target_id": "target_0",
                        "analysis_zarr": "/groups/example_analysis.zarr",
                        "keypoint_run": "keypoints_test",
                        "refined_keypoint_run": "refined_keypoints_test",
                        "subject_masks": {
                            "subject_mask_run": "subject_masks_test",
                            "refined_subject_mask_run": "refined_subject_masks_test",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_validate_analysis_plan_checks_exact_dense_outputs(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()

    report = mod.validate_analysis_plan(
        plan_path,
        sample_rows=3,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {
            "valid": True,
            "errors": [],
            "warnings": [],
        },
    )

    assert report["status"] == "ok"
    assert report["ok_count"] == 1
    target = report["targets"][0]
    assert target["raw_masks"]["dtype"] == "uint8"
    assert target["refined_masks"]["sample_unique_values"] == [1]
    assert target["refined_masks"]["authoritative_surface"] == "masks_roi"
    assert target["refined_masks"]["component_schema"]["resolution_basis"] == (
        "exact_component_labels"
    )
    assert (
        target["refined_masks"]["component_schema"]["component_registry_present"]
        is True
    )
    assert (
        target["refined_masks"]["component_schema"]["logical_schema_components_present"]
        is True
    )
    assert target["subject_mask_component_completeness"]["status"] == "passed"
    assert report["component_completeness_status"] == "passed"


def test_read_rows_uses_explicit_slices_for_zarr_orthogonal_indexing() -> None:
    values = np.arange(5 * 3 * 2 * 2).reshape(5, 3, 2, 2)
    array = EllipsisRejectingArray(values)

    sampled = mod._read_rows(array, np.asarray([0, 3], dtype=np.int64))

    np.testing.assert_array_equal(sampled, values[[0, 3], :, :, :])
    assert array.oindex.last_key == (
        [0, 3],
        slice(None),
        slice(None),
        slice(None),
    )


def test_validate_analysis_plan_reports_all_zero_required_raw_component(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    raw = root["subject_mask_runs"]["subject_masks_test"]
    values = raw["mask_probs_roi"].values
    values[:, 0, :, :] = 0
    raw["metrics"]["mask_present"].values[:, 0] = False

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "ok"
    assert report["invalid_count"] == 0
    target = report["targets"][0]
    assert target["raw_masks"]["component_presence_policy"][
        "missing_required_components"
    ] == ["subject_body"]
    assert target["subject_mask_component_completeness"]["status"] == "failed"
    assert report["component_completeness_status"] == "failed"


def test_validate_analysis_plan_reports_absent_required_raw_component(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    raw = root["subject_mask_runs"]["subject_masks_test"]
    raw["mask_probs_roi"].values[:, 2, :, :] = 0
    raw["metrics"]["mask_present"].values[:, 2] = False

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "ok"
    raw_report = report["targets"][0]["raw_masks"]
    assert raw_report["mask_present_counts"]["swim_bladder"] == 0
    assert raw_report["component_schema"]["schema_id"] == "subject_v1_union"
    assert raw_report["component_presence_policy"]["optional_components"] == []
    assert raw_report["component_presence_policy"]["missing_required_components"] == [
        "swim_bladder"
    ]
    assert raw_report["sample_component_presence_policy"][
        "missing_required_components"
    ] == ["swim_bladder"]
    assert raw_report["component_completeness"]["status"] == "failed"
    assert raw_report["component_completeness"]["publication_blocking"] is False


def test_validate_analysis_plan_notes_metric_absence_without_invalidating(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    raw = root["subject_mask_runs"]["subject_masks_test"]
    raw["metrics"]["mask_present"].values[:, 0] = False

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "ok"
    completeness = report["targets"][0]["subject_mask_component_completeness"]
    assert completeness["status"] == "failed"
    assert completeness["failures"] == [
        {
            "stage": "raw_subject_masks",
            "code": "required_component_has_no_present_masks",
            "scope": "all_rows",
            "component": "subject_body",
        }
    ]


def test_validate_analysis_plan_reports_absent_required_refined_component(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    refined = root["refined_subject_masks_runs"]["refined_subject_masks_test"]
    refined["masks_roi"].values[:, 3, :, :] = 0
    refined["metrics"]["mask_present"].values[:, 3] = False

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "ok"
    refined_report = report["targets"][0]["refined_masks"]
    assert refined_report["mask_present_counts"]["swim_bladder"] == 0
    assert refined_report["component_schema"]["schema_id"] == "subject_v1_lr"
    assert refined_report["component_presence_policy"][
        "missing_required_components"
    ] == ["swim_bladder"]
    assert refined_report["sample_component_presence_policy"][
        "missing_required_components"
    ] == ["swim_bladder"]
    assert refined_report["component_completeness"]["status"] == "failed"


def test_validate_analysis_plan_notes_required_refined_component_absence(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    refined = root["refined_subject_masks_runs"]["refined_subject_masks_test"]
    refined["masks_roi"].values[:, 0, :, :] = 0
    refined["metrics"]["mask_present"].values[:, 0] = False

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "ok"
    completeness = report["targets"][0]["subject_mask_component_completeness"]
    assert completeness["status"] == "failed"
    assert completeness["failures"][0]["component"] == "subject_body"


def test_validate_analysis_plan_fails_when_required_schema_channel_unavailable(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    raw = root["subject_mask_runs"]["subject_masks_test"]
    raw["available_channels"].values[2] = False

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "invalid"
    assert "swim_bladder" in report["targets"][0]["error"]["message"]
    assert "marked unavailable" in report["targets"][0]["error"]["message"]


def test_validate_analysis_plan_fails_on_mask_schema_label_mismatch(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    raw = root["subject_mask_runs"]["subject_masks_test"]
    raw.attrs["mask_labels"] = ["subject_body", "eyes_union", "unknown_component"]

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "invalid"
    assert "do not match" in report["targets"][0]["error"]["message"]


def test_validate_analysis_plan_requires_authoritative_dense_masks(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    del root["refined_subject_masks_runs"]["refined_subject_masks_test"]["masks_roi"]

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "invalid"
    assert "authoritative dense masks_roi" in report["targets"][0]["error"]["message"]
