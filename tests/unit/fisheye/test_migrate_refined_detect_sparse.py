from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.shared.refined_detect_curation import (
    build_source_detection_decision_summary,
    extract_present_curated_rows,
)
from fisheye.utils import migrate_refined_detect_sparse as mod


class _FakeArray:
    def __init__(self, data: Any, *, chunks: tuple[int, ...] | None = None) -> None:
        if isinstance(data, list) and data and isinstance(data[0], str):
            self._data = np.asarray(data, dtype=object)
        else:
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

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return self._children.keys()

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

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


def _write_sparse_group(
    group: _FakeGroup,
    *,
    frame_indices: np.ndarray,
    bbox_norm_coords: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    reason_labels: np.ndarray,
) -> None:
    frame_counts = np.bincount(frame_indices, minlength=4).astype(np.int32)
    group.create_array("frame_indices", data=frame_indices, overwrite=True)
    group.create_array("bbox_norm_coords", data=bbox_norm_coords, overwrite=True)
    group.create_array("scores", data=scores, overwrite=True)
    group.create_array("class_ids", data=class_ids, overwrite=True)
    group.create_array("frame_counts", data=frame_counts, overwrite=True)
    group.create_array("n_detections", data=frame_counts, overwrite=True)
    write_reason_columns(
        group,  # type: ignore[arg-type]
        np.asarray(reason_labels, dtype=object),
        chunk_size=max(1, int(frame_indices.shape[0])),
        overwrite=True,
    )


def _build_root(
    *,
    with_legacy_groups: bool = True,
    parent_latest: str = "refined_detect_001",
) -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["width"] = 320
    root.attrs["height"] = 240
    root.attrs["total_frames"] = 4

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect_group = detect_parent.create_group("detect_001")
    _write_sparse_group(
        detect_group,
        frame_indices=np.asarray([0, 1, 1, 3], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                [0.2, 0.2, 0.1, 0.1],
                [0.3, 0.4, 0.2, 0.2],
                [0.6, 0.4, 0.2, 0.2],
                [0.5, 0.7, 0.15, 0.15],
            ],
            dtype=np.float64,
        ),
        scores=np.asarray([0.95, 0.9, 0.85, 0.4], dtype=np.float32),
        class_ids=np.asarray([0, 0, 0, 0], dtype=np.int32),
        reason_labels=np.asarray(["raw", "raw", "raw", "raw"], dtype=object),
    )
    quality_parent = detect_group.create_group("quality_reports")
    quality_parent.attrs["latest"] = "quality_001"
    quality_group = quality_parent.create_group("quality_001")
    quality_group.create_array(
        "detection_quality_labels",
        data=np.asarray([0, 0, 0, 3], dtype=np.int8),
        overwrite=True,
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = parent_latest
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["manual_review_latest"] = "manual_a"
    refined.attrs["detect_review_status"] = {"state": "approved", "intended_use": "training"}
    if with_legacy_groups:
        manual = refined.create_group("manual_a")
        _write_sparse_group(
            manual,
            frame_indices=np.asarray([2], dtype=np.int32),
            bbox_norm_coords=np.asarray([[0.8, 0.5, 0.1, 0.1]], dtype=np.float64),
            scores=np.asarray([0.99], dtype=np.float32),
            class_ids=np.asarray([0], dtype=np.int32),
            reason_labels=np.asarray(["manual"], dtype=object),
        )
    if parent_latest != "refined_detect_001":
        latest_refined = refined_parent.create_group(parent_latest)
        latest_refined.attrs["source_detect_run"] = "detect_001"
    return root


def test_build_sparse_migration_plan_requires_explicit_override_for_legacy_groups() -> None:
    root = _build_root()

    try:
        mod.build_sparse_migration_plan(
            root,  # type: ignore[arg-type]
            refined_run_name="refined_detect_001",
        )
    except mod.SparseMigrationConflictError as exc:
        assert exc.legacy_sparse_groups == ["manual_a"]
        assert exc.output_refined_run_name == "refined_detect_001_sparse"
        assert "ignore-legacy-groups" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected legacy sparse groups to require explicit override.")


def test_build_sparse_migration_plan_materializes_successor_plan_when_override_enabled() -> None:
    root = _build_root()

    plan = mod.build_sparse_migration_plan(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        ignore_legacy_groups=True,
    )

    assert plan.source_refined_run_name == "refined_detect_001"
    assert plan.output_refined_run_name == "refined_detect_001_sparse"
    assert plan.source_detect_run == "detect_001"
    assert plan.source_quality_run == "quality_001"
    assert plan.parent_latest_refined_run_name == "refined_detect_001"
    assert plan.source_is_parent_latest is True
    assert plan.current_summary["has_curated_surface"] is False
    assert plan.current_summary["legacy_sparse_groups"] == ["manual_a"]
    assert plan.legacy_group_policy == "ignored_by_operator"
    assert plan.planned_summary["total_instances"] == 3
    assert plan.planned_summary["multi_instance_frames"] == 1
    assert plan.planned_summary["max_instances_per_frame"] == 2
    assert plan.planned_summary["source_detection_decision_counts"] == {"accepted": 3, "filtered": 1}
    assert plan.planned_summary["output_refined_run"] == "refined_detect_001_sparse"
    assert plan.planned_summary["ignored_legacy_groups"] == ["manual_a"]
    assert plan.planned_summary["source_is_parent_latest"] is True


def test_build_sparse_migration_plan_tracks_nonlatest_source_run() -> None:
    root = _build_root(with_legacy_groups=False, parent_latest="refined_detect_002")

    plan = mod.build_sparse_migration_plan(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
    )

    assert plan.source_refined_run_name == "refined_detect_001"
    assert plan.output_refined_run_name == "refined_detect_001_sparse"
    assert plan.parent_latest_refined_run_name == "refined_detect_002"
    assert plan.source_is_parent_latest is False
    assert plan.legacy_sparse_groups == []


def test_apply_sparse_migration_writes_successor_run_and_promotes_latest() -> None:
    root = _build_root()
    plan = mod.build_sparse_migration_plan(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
        ignore_legacy_groups=True,
    )

    result = mod.apply_sparse_migration(
        root,  # type: ignore[arg-type]
        zarr_path=Path("/tmp/canary.zarr"),
        plan=plan,
        command="migrate_refined_detect_sparse",
    )

    refined_parent = root["refined_detect_runs"]
    source_refined = refined_parent["refined_detect_001"]
    output_refined = refined_parent["refined_detect_001_sparse"]
    assert result["applied"] is True
    assert result["source_refined_run"] == "refined_detect_001"
    assert result["output_refined_run"] == "refined_detect_001_sparse"
    assert result["promoted_to_latest"] is True
    assert refined_parent.attrs["latest"] == "refined_detect_001_sparse"
    assert refined_parent.attrs["detect_review_status_latest"] == "refined_detect_001_sparse"

    assert "instances" not in source_refined
    assert "source_detections" not in source_refined
    assert "manual_a" in source_refined

    assert "instances" in output_refined
    assert "source_detections" in output_refined
    assert "manual_a" not in output_refined
    assert output_refined.attrs["source_quality_run"] == "quality_001"
    assert output_refined.attrs["migrated_from_refined_run"] == "refined_detect_001"
    assert output_refined.attrs["detect_review_status"]["state"] == "pending"
    assert output_refined.attrs["detect_review_status"]["resolved_group"] == "refined"
    assert output_refined.attrs["sparse_migration"]["ignored_legacy_groups"] == ["manual_a"]

    instances = extract_present_curated_rows(output_refined)  # type: ignore[arg-type]
    assert np.asarray(instances["frame_indices"], dtype=np.int32).tolist() == [0, 1, 1]

    decisions = build_source_detection_decision_summary(output_refined)  # type: ignore[arg-type]
    assert decisions["decision_accepted"] == 3
    assert decisions["decision_filtered"] == 1
    assert decisions["total_candidates"] == 4
    summary = dict(output_refined.attrs["summary_statistics"])
    assert summary["frames_multi_instance"] == 1
    assert summary["rows_ambiguous"] == 1


def test_apply_sparse_migration_blocks_default_promotion_for_nonlatest_source_run() -> None:
    root = _build_root(with_legacy_groups=False, parent_latest="refined_detect_002")
    plan = mod.build_sparse_migration_plan(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
    )

    try:
        mod.apply_sparse_migration(
            root,  # type: ignore[arg-type]
            zarr_path=Path("/tmp/canary.zarr"),
            plan=plan,
            command="migrate_refined_detect_sparse",
        )
    except mod.SparseMigrationConflictError as exc:
        assert exc.output_refined_run_name == "refined_detect_001_sparse"
        assert "--no-promote-latest" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected non-latest default promotion to be blocked.")


def test_apply_sparse_migration_nonlatest_succeeds_without_promotion() -> None:
    root = _build_root(with_legacy_groups=False, parent_latest="refined_detect_002")
    plan = mod.build_sparse_migration_plan(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
    )

    result = mod.apply_sparse_migration(
        root,  # type: ignore[arg-type]
        zarr_path=Path("/tmp/canary.zarr"),
        plan=plan,
        command="migrate_refined_detect_sparse",
        promote_latest=False,
    )

    refined_parent = root["refined_detect_runs"]
    assert result["promoted_to_latest"] is False
    assert refined_parent.attrs["latest"] == "refined_detect_002"
    assert refined_parent["refined_detect_001_sparse"].attrs["sparse_migration"]["promoted_to_latest"] is False


def test_apply_sparse_migration_nonlatest_can_force_promotion() -> None:
    root = _build_root(with_legacy_groups=False, parent_latest="refined_detect_002")
    plan = mod.build_sparse_migration_plan(
        root,  # type: ignore[arg-type]
        refined_run_name="refined_detect_001",
    )

    result = mod.apply_sparse_migration(
        root,  # type: ignore[arg-type]
        zarr_path=Path("/tmp/canary.zarr"),
        plan=plan,
        command="migrate_refined_detect_sparse",
        promote_latest=True,
        force_promote_nonlatest=True,
    )

    refined_parent = root["refined_detect_runs"]
    assert result["promoted_to_latest"] is True
    assert result["force_promote_nonlatest"] is True
    assert refined_parent.attrs["latest"] == "refined_detect_001_sparse"


def test_main_dry_run_emits_json_without_writing(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "canary.zarr"
    root = _build_root()

    monkeypatch.setattr(
        mod,
        "_open_root",
        lambda path, mode="r": root,  # noqa: ARG005
    )

    rc = mod.main([str(zarr_path), "--json", "--ignore-legacy-groups"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run"
    assert payload["source_refined_run"] == "refined_detect_001"
    assert payload["output_refined_run"] == "refined_detect_001_sparse"
    assert payload["planned"]["multi_instance_frames"] == 1
    refined = root["refined_detect_runs"]["refined_detect_001"]
    assert "instances" not in refined


def test_main_dry_run_reports_nonlatest_promotion_block_in_json(monkeypatch, capsys, tmp_path: Path) -> None:
    zarr_path = tmp_path / "canary.zarr"
    root = _build_root(with_legacy_groups=False, parent_latest="refined_detect_002")

    monkeypatch.setattr(
        mod,
        "_open_root",
        lambda path, mode="r": root,  # noqa: ARG005
    )

    rc = mod.main([str(zarr_path), "--json", "--refined-run", "refined_detect_001"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["source_refined_run"] == "refined_detect_001"
    assert payload["parent_latest_refined_run"] == "refined_detect_002"
    assert payload["source_is_parent_latest"] is False
    assert payload["promotion_requested"] is True
    assert payload["promotion_allowed"] is False
    assert "--no-promote-latest" in payload["promotion_blocked_reason"]
