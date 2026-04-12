"""Tests for syncing keypoint profile summaries into registry rows."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import sync_keypoint_profile_registry as sync_module


class _FakeGroup:
    def __init__(
        self,
        *,
        children: dict[str, "_FakeGroup"] | None = None,
        attrs: dict[str, object] | None = None,
        get_misses: set[str] | None = None,
        item_misses: set[str] | None = None,
        store: object | None = None,
        path: str = "",
    ) -> None:
        self._children = dict(children or {})
        self.attrs = dict(attrs or {})
        self._get_misses = set(get_misses or set())
        self._item_misses = set(item_misses or set())
        self.store = object() if store is None else store
        self.path = path

    def get(self, name: str):
        if name in self._get_misses:
            return None
        return self._children.get(name)

    def __getitem__(self, name: str):
        if name in self._item_misses:
            raise KeyError(name)
        return self._children[name]

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def keys(self):
        return self._children.keys()

    def group_keys(self):
        return self._children.keys()


def _seed_dataset_row(
    registry_path: Path,
    *,
    dataset_id: str,
    zarr_path: Path,
    zarr_use: str,
    genotype: str | None = None,
    dpf_at_acquisition: int | None = None,
) -> None:
    registry = Registry(registry_path)
    registry.upsert_dataset(
        dataset_id,
        session_uuid=f"{dataset_id}_session",
        zarr_path=zarr_path,
        recording_id=f"{dataset_id}_recording",
        artifact_kind="source_recording",
        zarr_use=zarr_use,
    )
    if genotype is not None or dpf_at_acquisition is not None:
        registry.upsert_provenance(
            dataset_id,
            provenance={
                "genotype": genotype,
                "dpf_at_acquisition": dpf_at_acquisition,
            },
            context={},
            protocol_name=None,
            protocol_hash=None,
            acquisition={},
        )
    registry.close()


def _summary_payload(*, dataset_id: str, recording_id: str, zarr_use: str) -> dict[str, object]:
    return {
        "schema_name": "keypoint_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-12T03:00:00+00:00",
        "dataset": {
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
        },
        "source": {
            "keypoint_path": "refined_keypoints_runs/refined_keypoints_2026-02-12",
            "keypoint_method": "traditional_pose",
            "keypoint_run": "keypoints_2026-02-12",
            "skeleton_id": "fish_v1",
            "kpt_shape": [3, 2],
            "pose_schema_name": "traditional_v1",
            "pose_schema": {
                "name": "traditional_v1",
                "skeleton_id": "fish_v1",
                "kpt_shape": [3, 2],
                "edges": [[0, 1], [0, 2], [1, 2]],
                "metadata": {
                    "heading_computation": {
                        "version": 1,
                        "enabled": True,
                        "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"],
                    }
                },
            },
            "heading_computation_source": "pose_schema.metadata.heading_computation",
            "heading_computation": {
                "version": 1,
                "enabled": True,
                "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"],
            },
        },
        "quality": {
            "rows_total": 100,
            "rows_usable": 90,
            "usable_keypoints_total": 90,
            "usable_rate": 0.9,
            "confidence_valid_rate": 0.95,
            "geometry_valid_rate": 0.96,
        },
        "geometry": {
            "triangle_area": {"stats": {"p10": 0.01, "p50": 0.02, "p90": 0.03}},
            "min_angle": {"stats": {"p10": 10.0, "p50": 20.0, "p90": 30.0}},
            "heading": {"stats": {"p10": -0.4, "p50": 0.0, "p90": 0.4}},
        },
        "composition": {
            "rig_id": "rig_a",
            "camera_id": "cam_1",
            "arena_id": "arena_x",
            "dish_design": "cedar",
            "canvas_name": "canvas_a",
            "protocol_name": "DefaultScreen",
            "genotype": "Tg(elavl3:gcamp7f)",
            "dpf_at_acquisition": 7,
        },
    }


def test_sync_keypoint_profile_registry_apply_upserts_latest_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_a_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_a", zarr_path=zarr_path, zarr_use="training")

    summary = _summary_payload(dataset_id="dataset_a", recording_id="recording_a", zarr_use="training")
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("keypoint_profile_2026-02-12_03-00-00", summary, None),
    )

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_a",
            "--apply",
        ]
    )
    assert rc == 0

    registry = Registry(registry_path)
    try:
        rows = registry.query_keypoint_data_profile_latest(dataset_ids=["dataset_a"])
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["dataset_id"] == "dataset_a"
        assert row["profile_run"] == "keypoint_profile_2026-02-12_03-00-00"
        assert row["recording_id"] == "dataset_a_recording"
        assert row["zarr_use"] == "training"
        assert row["keypoint_method"] == "traditional_pose"
        assert row["source_keypoint_path"] == "refined_keypoints_runs/refined_keypoints_2026-02-12"
        assert row["source_keypoint_run"] == "keypoints_2026-02-12"
        assert row["skeleton_id"] == "fish_v1"
        assert row["kpt_shape"] == "[3,2]"
        assert row["pose_schema_name"] == "traditional_v1"
        assert '"name":"traditional_v1"' in str(row["pose_schema_json"])
        assert row["heading_computation_source"] == "pose_schema.metadata.heading_computation"
        assert '"enabled":true' in str(row["heading_computation_json"])
        assert row["rows_total"] == 100
        assert row["rows_usable"] == 90
        assert row["usable_keypoints_total"] == 90
        assert row["usable_rate"] == pytest.approx(0.9)
        assert row["confidence_valid_rate"] == pytest.approx(0.95)
        assert row["geometry_valid_rate"] == pytest.approx(0.96)
        assert row["triangle_area_p50"] == pytest.approx(0.02)
        assert row["min_angle_p50"] == pytest.approx(20.0)
        assert row["heading_p50"] == pytest.approx(0.0)
        assert row["genotype"] is None
        assert row["dpf_at_acquisition"] is None

        raw_row = registry.conn.execute(
            """
            SELECT *
            FROM keypoint_data_profile
            WHERE dataset_id = ?;
            """,
            ("dataset_a",),
        ).fetchone()
        assert raw_row is not None
        assert raw_row["recording_id"] == "recording_a"
        assert raw_row["genotype"] == "Tg(elavl3:gcamp7f)"
        assert int(raw_row["dpf_at_acquisition"]) == 7
    finally:
        registry.close()


def test_sync_keypoint_profile_registry_prefers_run_attrs_for_source_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_lineage_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(
        registry_path,
        dataset_id="dataset_lineage",
        zarr_path=zarr_path,
        zarr_use="training",
    )

    summary = _summary_payload(
        dataset_id="dataset_lineage",
        recording_id="summary_recording",
        zarr_use="summary_use",
    )
    source = dict(summary["source"])  # type: ignore[index]
    source["keypoint_path"] = "summary/keypoints/path"
    source["keypoint_method"] = "summary_method"
    source["keypoint_run"] = "summary_run"
    source["skeleton_id"] = "summary_skeleton"
    source["kpt_shape"] = [99, 99]
    summary["source"] = source
    summary["created_at_utc"] = "2026-02-12T01:00:00+00:00"

    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("keypoint_profile_2026-02-12_06-00-00", summary, None),
    )
    monkeypatch.setattr(
        sync_module,
        "_load_profile_run_attrs",
        lambda _root, _profile_run: {
            "created_at_utc": "2026-02-12T06:00:00+00:00",
            "source_recording_id": "attrs_recording",
            "source_zarr_use": "attrs_use",
            "source_keypoint_path": "refined_keypoints_runs/refined_keypoints_attrs",
            "source_keypoint_method": "traditional_pose",
            "source_keypoint_run": "keypoints_attrs",
            "source_skeleton_id": "fish_v2",
            "source_kpt_shape": [3, 2],
            "source_pose_schema_name": "traditional_v2",
            "source_pose_schema": {
                "name": "traditional_v2",
                "skeleton_id": "fish_v2",
                "kpt_shape": [3, 2],
            },
            "source_heading_computation_source": "heading_computation_override",
            "source_heading_computation": {"version": 1, "enabled": False},
        },
    )

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_lineage",
            "--apply",
        ]
    )
    assert rc == 0

    registry = Registry(registry_path)
    try:
        rows = registry.query_keypoint_data_profile_latest(dataset_ids=["dataset_lineage"])
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["keypoint_method"] == "traditional_pose"
        assert row["source_keypoint_path"] == "refined_keypoints_runs/refined_keypoints_attrs"
        assert row["source_keypoint_run"] == "keypoints_attrs"
        assert row["skeleton_id"] == "fish_v2"
        assert row["kpt_shape"] == "[3,2]"
        assert row["pose_schema_name"] == "traditional_v2"
        assert row["heading_computation_source"] == "heading_computation_override"
        assert str(row["heading_computation_json"]) == '{"enabled":false,"version":1}'

        raw_row = registry.conn.execute(
            """
            SELECT *
            FROM keypoint_data_profile
            WHERE dataset_id = ?;
            """,
            ("dataset_lineage",),
        ).fetchone()
        assert raw_row is not None
        assert raw_row["recording_id"] == "attrs_recording"
        assert raw_row["zarr_use"] == "attrs_use"
        assert raw_row["profile_created_utc"] == "2026-02-12T06:00:00+00:00"
        assert raw_row["keypoint_method"] == "traditional_pose"
        assert raw_row["source_keypoint_path"] == "refined_keypoints_runs/refined_keypoints_attrs"
        assert raw_row["source_keypoint_run"] == "keypoints_attrs"
        assert raw_row["skeleton_id"] == "fish_v2"
        assert raw_row["kpt_shape"] == "[3,2]"
        assert raw_row["pose_schema_name"] == "traditional_v2"
        assert raw_row["heading_computation_source"] == "heading_computation_override"
    finally:
        registry.close()


def test_sync_keypoint_profile_registry_dry_run_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_b_analysis.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_b", zarr_path=zarr_path, zarr_use="analysis")

    summary = _summary_payload(dataset_id="dataset_b", recording_id="recording_b", zarr_use="analysis")
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("keypoint_profile_2026-02-12_04-00-00", summary, None),
    )

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_b",
        ]
    )
    assert rc == 0

    registry = Registry(registry_path)
    try:
        count = registry.conn.execute("SELECT COUNT(*) AS n FROM keypoint_data_profile;").fetchone()
        assert count is not None
        assert int(count["n"]) == 0
    finally:
        registry.close()


def test_sync_keypoint_profile_registry_missing_profile_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_c_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_c", zarr_path=zarr_path, zarr_use="training")

    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(sync_module, "_latest_profile_summary", lambda _root: (None, None, "missing summary"))

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_c",
            "--apply",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "missing_profile" in out

    registry = Registry(registry_path)
    try:
        count = registry.conn.execute("SELECT COUNT(*) AS n FROM keypoint_data_profile;").fetchone()
        assert count is not None
        assert int(count["n"]) == 0
    finally:
        registry.close()


def test_sync_keypoint_profile_registry_falls_back_to_provenance_subject_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_d_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(
        registry_path,
        dataset_id="dataset_d",
        zarr_path=zarr_path,
        zarr_use="training",
        genotype="Tg(test:line)",
        dpf_at_acquisition=9,
    )

    summary = _summary_payload(dataset_id="dataset_d", recording_id="recording_d", zarr_use="training")
    composition = dict(summary["composition"])  # type: ignore[index]
    composition.pop("genotype", None)
    composition.pop("dpf_at_acquisition", None)
    summary["composition"] = composition
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("keypoint_profile_2026-02-12_05-00-00", summary, None),
    )

    rc = sync_module.main(
        [
            "--registry",
            str(registry_path),
            "--dataset-id",
            "dataset_d",
            "--apply",
        ]
    )
    assert rc == 0

    registry = Registry(registry_path)
    try:
        rows = registry.query_keypoint_data_profile_latest(dataset_ids=["dataset_d"])
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["genotype"] is None
        assert row["dpf_at_acquisition"] is None

        raw_row = registry.conn.execute(
            """
            SELECT *
            FROM keypoint_data_profile
            WHERE dataset_id = ?;
            """,
            ("dataset_d",),
        ).fetchone()
        assert raw_row is not None
        assert raw_row["genotype"] is None
        assert raw_row["dpf_at_acquisition"] is None
    finally:
        registry.close()


def test_latest_profile_summary_falls_back_to_keyed_group_lookup() -> None:
    summary_payload = {
        "schema_name": "keypoint_dataset_profile",
        "schema_version": "v1",
    }
    run_group = _FakeGroup(attrs={"profile_summary": summary_payload})
    runs_parent = _FakeGroup(
        children={"run_1": run_group},
        attrs={"latest": "run_1"},
        get_misses={"run_1"},
    )
    analysis = _FakeGroup(
        children={"keypoint_profile_runs": runs_parent},
        get_misses={"keypoint_profile_runs"},
    )
    root = _FakeGroup(
        children={"analysis": analysis},
        get_misses={"analysis"},
    )

    run_name, summary, error = sync_module._latest_profile_summary(root)  # type: ignore[arg-type]
    assert error is None
    assert run_name == "run_1"
    assert summary is not None
    assert summary["schema_name"] == "keypoint_dataset_profile"
