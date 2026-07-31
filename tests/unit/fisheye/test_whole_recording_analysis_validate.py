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
    return FakeGroup(
        attrs={"palette_run_completion_status": "complete", **attrs}
    )


def _root() -> FakeGroup:
    raw_probabilities = np.full((4, 3, 2, 2), 128, dtype=np.uint8)
    raw = _complete_group(
        mask_labels=list(mod.RAW_LABELS),
    )
    raw["mask_probs_roi"] = FakeArray(raw_probabilities)
    raw["metrics"] = FakeGroup(
        {"mask_present": FakeArray(np.ones((4, 3), dtype=bool))}
    )

    refined = _complete_group(
        mask_labels=list(mod.REFINED_LABELS),
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoints_run="refined_keypoints_test",
        sampled_component_contours_status="computed",
        sampled_component_contours_requested=True,
        component_contours_requested=False,
    )
    refined["masks_roi"] = FakeArray(np.ones((4, 4, 2, 2), dtype=np.uint8))
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
            "keypoints_runs": FakeGroup(
                {"keypoints_test": _complete_group()}
            ),
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


def test_validate_analysis_plan_fails_on_all_zero_raw_component(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    raw = root["subject_mask_runs"]["subject_masks_test"]
    values = raw["mask_probs_roi"].values
    values[:, 2, :, :] = 0

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "invalid"
    assert report["invalid_count"] == 1
    assert "swim_bladder" in report["targets"][0]["error"]["message"]


def test_validate_analysis_plan_requires_authoritative_dense_masks(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)
    root = _root()
    del root["refined_subject_masks_runs"]["refined_subject_masks_test"][
        "masks_roi"
    ]

    report = mod.validate_analysis_plan(
        plan_path,
        open_root_fn=lambda *_args, **_kwargs: root,
        contract_validator_fn=lambda *_args, **_kwargs: {"valid": True},
    )

    assert report["status"] == "invalid"
    assert "authoritative dense masks_roi" in report["targets"][0]["error"][
        "message"
    ]
