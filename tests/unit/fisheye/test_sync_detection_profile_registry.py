"""Tests for syncing detection profile summaries into registry rows."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import sync_detection_profile_registry as sync_module


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
        "schema_name": "detection_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-12T03:00:00+00:00",
        "dataset": {
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
        },
        "source": {
            "detection_path": "refined_detect_runs/refined_detect_2026-02-12/manual",
            "detection_type": "manual",
        },
        "coverage": {
            "frames_total": 100,
            "frames_with_detections": 95,
            "coverage_percent": 95.0,
        },
        "counts": {
            "detections_total": 950,
            "detections_per_frame": {
                "p50": 9.0,
                "p90": 10.0,
            },
        },
        "geometry_norm": {
            "w": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "h": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "area": {"p10": 0.01, "p50": 0.04, "p90": 0.09},
            "aspect_ratio": {"p10": 0.8, "p50": 1.0, "p90": 1.2},
        },
        "spatial": {
            "edge_proximity_rate": 0.03,
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


def test_sync_detection_profile_registry_apply_upserts_latest_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_a_training.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_a", zarr_path=zarr_path, zarr_use="training")

    summary = _summary_payload(dataset_id="dataset_a", recording_id="recording_a", zarr_use="training")
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("detection_profile_2026-02-12_03-00-00", summary, None),
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
        rows = registry.query_detection_data_profile_latest(dataset_ids=["dataset_a"])
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["dataset_id"] == "dataset_a"
        assert row["profile_run"] == "detection_profile_2026-02-12_03-00-00"
        assert row["recording_id"] == "recording_a"
        assert row["zarr_use"] == "training"
        assert row["detection_type"] == "manual"
        assert row["coverage_percent"] == pytest.approx(95.0)
        assert row["detections_total"] == 950
        assert row["genotype"] == "Tg(elavl3:gcamp7f)"
        assert int(row["dpf_at_acquisition"]) == 7
    finally:
        registry.close()


def test_sync_detection_profile_registry_dry_run_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "dataset_b_analysis.zarr"
    zarr_path.mkdir(parents=True)
    _seed_dataset_row(registry_path, dataset_id="dataset_b", zarr_path=zarr_path, zarr_use="analysis")

    summary = _summary_payload(dataset_id="dataset_b", recording_id="recording_b", zarr_use="analysis")
    monkeypatch.setattr(sync_module, "_open_root", lambda _: object())
    monkeypatch.setattr(
        sync_module,
        "_latest_profile_summary",
        lambda _root: ("detection_profile_2026-02-12_04-00-00", summary, None),
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
        count = registry.conn.execute("SELECT COUNT(*) AS n FROM detection_data_profile;").fetchone()
        assert count is not None
        assert int(count["n"]) == 0
    finally:
        registry.close()


def test_sync_detection_profile_registry_missing_profile_is_reported(
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
        count = registry.conn.execute("SELECT COUNT(*) AS n FROM detection_data_profile;").fetchone()
        assert count is not None
        assert int(count["n"]) == 0
    finally:
        registry.close()


def test_sync_detection_profile_registry_falls_back_to_provenance_subject_fields(
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
        lambda _root: ("detection_profile_2026-02-12_05-00-00", summary, None),
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
        rows = registry.query_detection_data_profile_latest(dataset_ids=["dataset_d"])
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["genotype"] == "Tg(test:line)"
        assert int(row["dpf_at_acquisition"]) == 9
    finally:
        registry.close()


def test_latest_profile_summary_falls_back_to_keyed_group_lookup() -> None:
    summary_payload = {
        "schema_name": "detection_dataset_profile",
        "schema_version": "v1",
    }
    run_group = _FakeGroup(attrs={"profile_summary": summary_payload})
    runs_parent = _FakeGroup(
        children={"run_1": run_group},
        attrs={"latest": "run_1"},
        get_misses={"run_1"},
    )
    analysis = _FakeGroup(
        children={"detection_profile_runs": runs_parent},
        get_misses={"detection_profile_runs"},
    )
    root = _FakeGroup(
        children={"analysis": analysis},
        get_misses={"analysis"},
    )

    run_name, summary, error = sync_module._latest_profile_summary(root)  # type: ignore[arg-type]
    assert error is None
    assert run_name == "run_1"
    assert summary is not None
    assert summary["schema_name"] == "detection_dataset_profile"


def test_latest_profile_summary_falls_back_to_open_group_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    store = object()
    summary_payload = {
        "schema_name": "detection_dataset_profile",
        "schema_version": "v1",
    }
    run_group = _FakeGroup(
        attrs={"profile_summary": summary_payload},
        store=store,
        path="analysis/detection_profile_runs/run_1",
    )
    runs_parent = _FakeGroup(
        children={"run_1": run_group},
        attrs={"latest": "run_1"},
        get_misses={"run_1"},
        item_misses={"run_1"},
        store=store,
        path="analysis/detection_profile_runs",
    )
    analysis = _FakeGroup(
        children={"detection_profile_runs": runs_parent},
        get_misses={"detection_profile_runs"},
        item_misses={"detection_profile_runs"},
        store=store,
        path="analysis",
    )
    root = _FakeGroup(
        children={"analysis": analysis},
        get_misses={"analysis"},
        item_misses={"analysis"},
        store=store,
        path="",
    )
    mapping = {
        "analysis": analysis,
        "analysis/detection_profile_runs": runs_parent,
        "analysis/detection_profile_runs/run_1": run_group,
    }

    def _fake_open_group(*_args, **kwargs):
        path = kwargs.get("path")
        if path in mapping:
            return mapping[path]
        raise KeyError(path)

    monkeypatch.setattr(sync_module.zarr, "open_group", _fake_open_group)

    run_name, summary, error = sync_module._latest_profile_summary(root)  # type: ignore[arg-type]
    assert error is None
    assert run_name == "run_1"
    assert summary is not None
    assert summary["schema_name"] == "detection_dataset_profile"
