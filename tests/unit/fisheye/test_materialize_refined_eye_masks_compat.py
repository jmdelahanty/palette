from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.tune import eye_mask_review as eye_review_mod
from fisheye.tune import refined_subject_mask_review as review_mod
from fisheye.utils import materialize_refined_eye_masks_compat as compat_mod


class _FakeArray:
    def __init__(self, data: Any, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.chunks = chunks

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        if name in self._children:
            raise ValueError(f"{name} already exists")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if isinstance(existing, _FakeGroup):
            return existing
        if existing is not None:
            raise TypeError(f"{name} already exists and is not a group")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def create_array(
        self,
        name: str,
        *,
        data: Any | None = None,
        shape: tuple[int, ...] | None = None,
        chunks: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        fill_value: Any = 0,
        overwrite: bool = False,
        **_kwargs: object,
    ) -> _FakeArray:
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_array(
                tail,
                data=data,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                fill_value=fill_value,
                overwrite=overwrite,
            )
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            if shape is None:
                raise ValueError("shape is required when data is omitted")
            try:
                resolved_dtype = np.dtype(dtype) if dtype is not None else np.float32
            except TypeError:
                resolved_dtype = object
            data = np.full(shape, fill_value, dtype=resolved_dtype)
        arr = _FakeArray(data, chunks=chunks)
        self._children[name] = arr
        return arr

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def keys(self):
        return list(self._children.keys())

    def items(self):
        return self._children.items()

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str):
        if "/" in key:
            current: _FakeGroup | _FakeArray = self
            for token in key.split("/"):
                if not isinstance(current, _FakeGroup):
                    raise KeyError(key)
                current = current._children[token]
            return current
        return self._children[key]

    def __delitem__(self, key: str) -> None:
        if "/" in key:
            head, tail = key.split("/", 1)
            child = self._children[head]
            if not isinstance(child, _FakeGroup):
                raise KeyError(key)
            del child[tail]
            return
        del self._children[key]


def _patch_stage_provenance(monkeypatch) -> None:
    for module in (compat_mod, review_mod):
        monkeypatch.setattr(
            module,
            "get_git_info",
            lambda repo_path=None: {  # noqa: ARG005
                "commit_hash": "d" * 40,
                "short_hash": "dddddddd",
                "branch": "main",
                "is_dirty": False,
                "remote_url": "git@example.com:palette.git",
            },
        )
        monkeypatch.setattr(
            module,
            "get_environment_info",
            lambda **kwargs: {  # noqa: ARG005
                "environment": {"python": "3.12"},
                "platform": {
                    "hostname": "compat-host",
                    "system": "Linux",
                    "release": "6.8",
                    "python_version": "3.12.0",
                    "machine": "x86_64",
                },
            },
        )


def _build_materialize_root() -> _FakeGroup:
    root = _FakeGroup()

    keypoints_parent = root.create_group("refined_keypoints_runs")
    keypoints_parent.attrs["latest"] = "refined_kp_001"
    keypoints = keypoints_parent.create_group("refined_kp_001")
    keypoints.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[4.0, 4.0], [2.0, 3.0], [5.0, 3.0]],
                [[4.0, 4.0], [2.0, 4.0], [5.0, 4.0]],
            ],
            dtype=np.float32,
        ),
    )

    eye_parent = root.create_group("eye_masks_runs")
    eye_parent.attrs["latest"] = "eye_masks_001"
    eye_parent.create_group("eye_masks_001").attrs.update(
        {
            "method": "yolo_eye_segmentation",
            "eye_labels": ["eye_left", "eye_right"],
        }
    )

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "subject_masks_001"
    subject = subject_parent.create_group("subject_masks_001")
    subject.attrs.update(
        {
            "method": "eye_mask_runtime_projection",
            "source_eye_masks_run": "eye_masks_001",
            "source_refined_eye_masks_run": "refined_eye_masks_legacy_001",
            "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "label_schema_id": "subject_v1_lr",
        }
    )
    subject.create_array("available_channels", data=np.asarray([False, True, True, False], dtype=np.bool_))
    subject.create_array("detection_source", data=np.zeros((2,), dtype=np.int8))
    subject.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32))
    subject.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    subject.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32))
    subject_masks = np.zeros((2, 4, 8, 8), dtype=np.uint8)
    subject_masks[0, 1, 2:5, 1:4] = 1
    subject_masks[0, 2, 2:5, 4:7] = 1
    subject_masks[1, 1, 3:6, 1:4] = 1
    subject_masks[1, 2, 3:6, 4:7] = 1
    subject.create_array("masks_roi", data=subject_masks)

    refined_parent = root.create_group("refined_subject_masks_runs")
    refined_parent.attrs["latest"] = "refined_subject_masks_001"
    refined = refined_parent.create_group("refined_subject_masks_001")
    refined.attrs.update(
        {
            "mask_labels": ["eye_left", "eye_right"],
            "source_subject_mask_run": "subject_masks_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "source_keypoints_run": "refined_kp_001",
            "component_review_statuses": {
                "eye_left": {
                    "state": "approved",
                    "method": "manual",
                    "intended_use": "training",
                    "reviewer": "tester",
                    "timestamp_utc": "2026-04-02T00:00:00+00:00",
                },
                "eye_right": {
                    "state": "pending",
                    "method": "manual",
                    "intended_use": "training",
                    "reviewer": "tester",
                    "timestamp_utc": "2026-04-02T00:01:00+00:00",
                },
            },
        }
    )
    refined.create_array("masks_roi", data=np.asarray(subject_masks[:, 1:3], dtype=np.uint8))
    refined.create_array("edit_applied", data=np.asarray([[True, False], [False, False]], dtype=np.bool_))
    refined.create_array("detection_source", data=np.zeros((2,), dtype=np.int8))
    refined.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32))
    refined.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    refined.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32))
    refined.require_group("components/eye_left").create_array(
        "reason",
        data=np.asarray(["manual_correction", "copied_from_source"], dtype=object),
        overwrite=True,
    )
    refined.require_group("components/eye_right").create_array(
        "reason",
        data=np.asarray(["clean", "copied_from_source"], dtype=object),
        overwrite=True,
    )
    refined.require_group("components/eye_left/provenance").attrs.update(
        {
            "source_stage": "subject_mask_runs",
            "source_run": "subject_masks_001",
            "source_channels": ["eye_left"],
        }
    )
    refined.require_group("components/eye_right/provenance").attrs.update(
        {
            "source_stage": "subject_mask_runs",
            "source_run": "subject_masks_001",
            "source_channels": ["eye_right"],
        }
    )

    compat_parent = root.create_group("refined_eye_masks_runs")
    compat_parent.attrs["latest"] = "refined_eye_masks_legacy_001"
    compat_parent.create_group("refined_eye_masks_legacy_001").attrs.update(
        {
            "source_eye_masks_run": "eye_masks_001",
            "summary_statistics": {"refine": {"legacy": 1}},
            "success_min_eye_area_px": 12.0,
        }
    )
    return root


def _replace_refined_subject_masks_with_rle(root: _FakeGroup) -> np.ndarray:
    refined = root["refined_subject_masks_runs/refined_subject_masks_001"]
    dense = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    labels = [str(label) for label in refined.attrs["mask_labels"]]
    del refined["masks_roi"]
    write_component_rle_mask_store_from_dense(
        refined,
        dense,
        component_names=labels,
        encode_row_chunk_size=1,
    )
    return dense


def test_materialize_refined_eye_masks_compat_creates_derived_run_and_preserves_refine_summary(monkeypatch) -> None:
    _patch_stage_provenance(monkeypatch)
    root = _build_materialize_root()
    monkeypatch.setattr(compat_mod, "_open_root", lambda source, mode="a": root)  # noqa: ARG005
    monkeypatch.setattr(
        eye_review_mod,
        "_update_postprocess_summary",
        lambda _root, refined, *, print_summary: refined.attrs.__setitem__("postprocess_synced", True),  # noqa: ARG005
    )

    summary = compat_mod.materialize_refined_eye_masks_compat(
        root,
        refined_subject_run="refined_subject_masks_001",
    )

    assert summary["status"] == "updated"
    assert summary["target_run"] == "refined_eye_masks_legacy_001"
    assert root["refined_subject_masks_runs/refined_subject_masks_001"].attrs["compat_refined_eye_masks_run"] == (
        "refined_eye_masks_legacy_001"
    )

    run = root["refined_eye_masks_runs/refined_eye_masks_legacy_001"]
    assert run.attrs["method"] == compat_mod.MATERIALIZE_REFINED_EYE_MASKS_COMPAT_METHOD
    assert run.attrs["compatibility_role"] == "derived_from_refined_subject_masks"
    assert run.attrs["source_refined_subject_masks_run"] == "refined_subject_masks_001"
    assert run.attrs["source_subject_mask_run"] == "subject_masks_001"
    assert run.attrs["source_eye_masks_run"] == "eye_masks_001"
    assert run.attrs["summary_statistics"]["refine"] == {"legacy": 1}
    assert run.attrs["eye_mask_review_status"]["state"] == "pending"
    assert run.attrs["postprocess_synced"] is True
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][:], dtype=np.uint8),
        np.asarray(root["refined_subject_masks_runs/refined_subject_masks_001/masks_roi"][:], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][:], dtype=bool),
        np.asarray([[True, False], [False, False]], dtype=bool),
    )
    assert tuple(np.asarray(run["ellipse_success"][:], dtype=bool).shape) == (2, 2)
    assert tuple(np.asarray(run["metrics/area_refined"][:], dtype=np.float32).shape) == (2, 2)
    assert tuple(np.asarray(run["metrics/reason_bytes"][:], dtype=np.uint8).shape[:1]) == (2,)


def test_materialize_refined_eye_masks_compat_reads_compact_refined_subject_masks(monkeypatch) -> None:
    _patch_stage_provenance(monkeypatch)
    root = _build_materialize_root()
    dense = _replace_refined_subject_masks_with_rle(root)
    monkeypatch.setattr(compat_mod, "_open_root", lambda source, mode="a": root)  # noqa: ARG005
    monkeypatch.setattr(
        eye_review_mod,
        "_update_postprocess_summary",
        lambda _root, refined, *, print_summary: refined.attrs.__setitem__("postprocess_synced", True),  # noqa: ARG005
    )

    summary = compat_mod.materialize_refined_eye_masks_compat(
        root,
        refined_subject_run="refined_subject_masks_001",
    )

    assert summary["status"] == "updated"
    assert "masks_roi" not in root["refined_subject_masks_runs/refined_subject_masks_001"]
    run = root["refined_eye_masks_runs/refined_eye_masks_legacy_001"]
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][:], dtype=np.uint8), dense)
