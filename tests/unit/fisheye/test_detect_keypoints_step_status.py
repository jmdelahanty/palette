from __future__ import annotations

import json
from pathlib import Path

from fisheye.detection import detect_keypoints_traditional as trad_mod
from fisheye.detection import detect_keypoints_yolo as yolo_mod
from fisheye.registry.db import Registry, RegistryPaths, resolve_dataset_id
from fisheye.registry.status_ledger import upsert_recording_step_status


class _FakeGroup(dict):
    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, object] = {}

    def get(self, key: str, default: object = None) -> object:  # noqa: A003
        return super().get(key, default)


def _make_analysis_root(path: Path, *, session_uuid: str) -> tuple[_FakeGroup, Path]:
    resolved = path.expanduser().resolve()
    root = _FakeGroup()
    root.attrs["session_uuid"] = session_uuid
    root.attrs["recording_id"] = session_uuid
    root.attrs["zarr_use"] = "analysis"
    root.attrs["zarr_purpose"] = "analysis"
    return root, resolved


def _seed_downstream_status(
    *,
    registry_path: Path,
    dataset_id: str,
    session_uuid: str,
    zarr_path: Path,
) -> None:
    registry = Registry(registry_path)
    try:
        registry.upsert_dataset(
            dataset_id,
            session_uuid=session_uuid,
            zarr_path=zarr_path,
            recording_id=session_uuid,
            zarr_use="analysis",
            zarr_purpose="analysis",
        )
        upsert_recording_step_status(
            registry,
            dataset_id=dataset_id,
            recording_id=session_uuid,
            step_name="refined_keypoints",
            status="ok",
            run_name="refined_keypoints_seed",
            method="refine_keypoints",
            coverage_pct=91.0,
            details_json={"reason": "seed"},
            source="test_seed",
        )
    finally:
        registry.close()


def test_emit_keypoint_step_status_yolo_writes_status_and_cascades(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    root, zarr_path = _make_analysis_root(
        tmp_path / "recording_analysis.zarr",
        session_uuid="2026-02-28T01-02-03Z_arena_1",
    )
    dataset_id, _session_uuid = resolve_dataset_id(root, zarr_path)
    _seed_downstream_status(
        registry_path=registry_path,
        dataset_id=dataset_id,
        session_uuid=str(root.attrs["recording_id"]),
        zarr_path=zarr_path,
    )

    yolo_mod._emit_keypoint_step_status(
        root=root,
        zarr_path=zarr_path,
        run_name="keypoints_001",
        method="yolo_pose",
        coverage_pct=87.5,
        details={"reason": "present", "source_crop_run": "crop_001"},
        console=None,
        registry=registry_path,
    )

    registry = Registry(registry_path)
    try:
        keypoint_row = registry.conn.execute(
            """
            SELECT status, run_name, method, coverage_pct, source, details_json
            FROM recording_step_status
            WHERE dataset_id = ? AND step_name = 'keypoints';
            """,
            (dataset_id,),
        ).fetchone()
        assert keypoint_row is not None
        assert str(keypoint_row["status"]) == "ok"
        assert str(keypoint_row["run_name"]) == "keypoints_001"
        assert str(keypoint_row["method"]) == "yolo_pose"
        assert float(keypoint_row["coverage_pct"]) == 87.5
        assert str(keypoint_row["source"]) == "runtime_keypoints_detect"
        details = json.loads(str(keypoint_row["details_json"]))
        assert details["reason"] == "present"
        assert details["source_crop_run"] == "crop_001"

        refined_row = registry.conn.execute(
            """
            SELECT status, source, details_json
            FROM recording_step_status
            WHERE dataset_id = ? AND step_name = 'refined_keypoints';
            """,
            (dataset_id,),
        ).fetchone()
        assert refined_row is not None
        assert str(refined_row["status"]) == "missing"
        assert str(refined_row["source"]) == "runtime_cascade_invalidation:runtime_keypoints_detect"
        cascade_details = json.loads(str(refined_row["details_json"]))
        assert cascade_details["cascade_trigger_step"] == "keypoints"
        assert cascade_details["cascade_trigger_run"] == "keypoints_001"
    finally:
        registry.close()


def test_emit_keypoint_step_status_traditional_writes_status_and_cascades(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    root, zarr_path = _make_analysis_root(
        tmp_path / "recording_analysis_trad.zarr",
        session_uuid="2026-02-28T04-05-06Z_arena_2",
    )
    dataset_id, _session_uuid = resolve_dataset_id(root, zarr_path)
    _seed_downstream_status(
        registry_path=registry_path,
        dataset_id=dataset_id,
        session_uuid=str(root.attrs["recording_id"]),
        zarr_path=zarr_path,
    )

    trad_mod._emit_keypoint_step_status(
        root=root,
        zarr_path=zarr_path,
        run_name="keypoints_002",
        method="traditional_pose",
        coverage_pct=93.25,
        details={"reason": "present", "source_crop_run": "crop_002"},
        console=None,
        registry=registry_path,
    )

    registry = Registry(registry_path)
    try:
        keypoint_row = registry.conn.execute(
            """
            SELECT status, run_name, method, coverage_pct, source
            FROM recording_step_status
            WHERE dataset_id = ? AND step_name = 'keypoints';
            """,
            (dataset_id,),
        ).fetchone()
        assert keypoint_row is not None
        assert str(keypoint_row["status"]) == "ok"
        assert str(keypoint_row["run_name"]) == "keypoints_002"
        assert str(keypoint_row["method"]) == "traditional_pose"
        assert float(keypoint_row["coverage_pct"]) == 93.25
        assert str(keypoint_row["source"]) == "runtime_keypoints_detect"

        refined_row = registry.conn.execute(
            """
            SELECT status, source
            FROM recording_step_status
            WHERE dataset_id = ? AND step_name = 'refined_keypoints';
            """,
            (dataset_id,),
        ).fetchone()
        assert refined_row is not None
        assert str(refined_row["status"]) == "missing"
        assert str(refined_row["source"]) == "runtime_cascade_invalidation:runtime_keypoints_detect"
    finally:
        registry.close()


def test_emit_keypoint_step_status_traditional_noops_when_registry_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root, zarr_path = _make_analysis_root(
        tmp_path / "recording_analysis_no_registry.zarr",
        session_uuid="2026-02-28T07-08-09Z_arena_3",
    )
    missing_registry = tmp_path / "missing" / "palette_registry.sqlite"

    class _FakeRegistryPaths:
        @staticmethod
        def from_env(_default_root: Path) -> RegistryPaths:
            return RegistryPaths(path=missing_registry)

    monkeypatch.setattr(trad_mod, "RegistryPaths", _FakeRegistryPaths)

    trad_mod._emit_keypoint_step_status(
        root=root,
        zarr_path=zarr_path,
        run_name="keypoints_003",
        method="traditional_pose",
        coverage_pct=88.0,
        details={"reason": "present"},
        console=None,
        registry=None,
    )

    assert not missing_registry.exists()
