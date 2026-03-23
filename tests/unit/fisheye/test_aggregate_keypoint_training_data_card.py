from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import aggregate_keypoint_training_data_card as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, key):
        return self._data[key]


class _FakeGroup:
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        self.attrs: dict[str, object] = dict(attrs or {})
        self._children: dict[str, object] = {}

    def add_group(self, name: str, *, attrs: dict[str, object] | None = None) -> "_FakeGroup":
        child = _FakeGroup(attrs=attrs)
        self._children[name] = child
        return child

    def add_array(self, name: str, data: np.ndarray) -> _FakeArray:
        arr = _FakeArray(np.asarray(data))
        self._children[name] = arr
        return arr

    def get(self, key: str):
        try:
            return self[key]
        except KeyError:
            return None

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str):
        if "/" in key:
            node: object = self
            for segment in key.split("/"):
                if not segment:
                    continue
                if not isinstance(node, _FakeGroup):
                    raise KeyError(key)
                node = node._children[segment]
            return node
        return self._children[key]

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]


def _seed_dataset(registry: Registry, *, dataset_id: str, recording_id: str, zarr_path: Path) -> None:
    zarr_path.mkdir(parents=True, exist_ok=True)
    registry.upsert_dataset(
        dataset_id,
        session_uuid=f"{dataset_id}_session",
        zarr_path=zarr_path,
        recording_id=recording_id,
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_provenance(
        dataset_id,
        provenance={},
        context={
            "rig_id": "rig_1",
            "arena_id": "arena_1",
            "camera_id": "camera_1",
            "canvas_name": "shadow",
        },
        protocol_name="DefaultScreen",
        protocol_hash=None,
        acquisition={
            "dish_design": "cedar",
        },
    )


def _seed_recording_context(
    registry: Registry,
    *,
    recording_id: str,
    session_uuid: str,
    root: Path,
    rig_id: str,
    arena_id: str,
    camera_id: str,
    canvas_name: str,
    protocol_name: str,
    dish_design: str,
) -> None:
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path,
            recording_type, recording_subtype, behavior_mode, artifact_schema_id,
            rig_id, arena_id, camera_id, canvas_name, protocol_name, dish_design,
            created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            recording_id,
            session_uuid,
            recording_id,
            str(root / "recordings" / recording_id),
            "behavior",
            "free",
            "free",
            "behavior_v1",
            rig_id,
            arena_id,
            camera_id,
            canvas_name,
            protocol_name,
            dish_design,
        ),
    )
    registry.conn.commit()


def _seed_keypoint_quality(
    registry: Registry,
    *,
    dataset_id: str,
    zarr_path: Path,
    refined_run: str = "refined_001",
    source_keypoint_run: str = "kp_001",
) -> None:
    registry.upsert_keypoint_quality(
        dataset_id=dataset_id,
        refined_run=refined_run,
        refined_created_utc="2026-02-24T00:00:00+00:00",
        source_keypoint_run=source_keypoint_run,
        keypoint_method="traditional_pose",
        review_state="approved",
        review_intended_use="training",
        review_reviewer="pytest",
        review_timestamp_utc="2026-02-24T00:00:00+00:00",
        review_method="manual",
        review_notes="ok",
        usable_keypoints=3,
        total_keypoints=4,
        usable_keypoints_rate=0.75,
        raw_keypoints_success_rate=0.75,
        raw_keypoints_successful=3,
        zarr_mtime_ns=int(zarr_path.stat().st_mtime_ns),
    )


def _seed_keypoint_profile(
    registry: Registry,
    *,
    dataset_id: str,
    recording_id: str,
    zarr_path: Path,
    profile_run: str = "keypoint_profile_001",
    source_keypoint_run: str = "kp_001",
    source_keypoint_path: str = "refined_keypoints_runs/refined_001",
    keypoint_method: str = "traditional_pose",
    zarr_mtime_ns: int | None = None,
    rows_total: int = 4,
    rows_usable: int = 3,
    usable_rate: float = 0.75,
    confidence_valid_rate: float = 0.75,
    geometry_valid_rate: float = 0.75,
    triangle_area_p10: float = 110.0,
    triangle_area_p50: float = 125.0,
    triangle_area_p90: float = 140.0,
    min_angle_p10: float = 18.0,
    min_angle_p50: float = 21.0,
    min_angle_p90: float = 24.0,
    heading_p10: float = 5.0,
    heading_p50: float = 12.5,
    heading_p90: float = 20.0,
) -> None:
    if zarr_mtime_ns is None:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    profile_summary = {
        "created_at_utc": "2026-02-24T00:00:00+00:00",
        "dataset": {
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "zarr_use": "training",
            "zarr_path": str(zarr_path),
        },
        "source": {
            "keypoint_path": source_keypoint_path,
            "keypoint_type": "refined",
            "keypoint_method": keypoint_method,
            "keypoint_run": source_keypoint_run,
            "refined_run": "refined_001",
            "skeleton_id": "traditional_v1",
            "kpt_shape": [3, 2],
        },
        "quality": {
            "rows_total": int(rows_total),
            "rows_usable": int(rows_usable),
            "usable_keypoints_total": int(rows_usable),
            "usable_rate": float(usable_rate),
            "confidence_valid_rate": float(confidence_valid_rate),
            "geometry_valid_rate": float(geometry_valid_rate),
        },
        "geometry": {
            "triangle_area": {
                "stats": {
                    "p10": float(triangle_area_p10),
                    "p50": float(triangle_area_p50),
                    "p90": float(triangle_area_p90),
                }
            },
            "min_angle": {
                "stats": {
                    "p10": float(min_angle_p10),
                    "p50": float(min_angle_p50),
                    "p90": float(min_angle_p90),
                }
            },
            "heading": {
                "stats": {
                    "p10": float(heading_p10),
                    "p50": float(heading_p50),
                    "p90": float(heading_p90),
                }
            },
        },
    }
    registry.upsert_keypoint_data_profile(
        dataset_id=dataset_id,
        profile_run=profile_run,
        recording_id=recording_id,
        zarr_use="training",
        keypoint_method=keypoint_method,
        source_keypoint_path=source_keypoint_path,
        source_keypoint_run=source_keypoint_run,
        skeleton_id="traditional_v1",
        kpt_shape="[3,2]",
        profile_created_utc="2026-02-24T00:00:00+00:00",
        rows_total=int(rows_total),
        rows_usable=int(rows_usable),
        usable_keypoints_total=int(rows_usable),
        usable_rate=float(usable_rate),
        confidence_valid_rate=float(confidence_valid_rate),
        geometry_valid_rate=float(geometry_valid_rate),
        triangle_area_p10=float(triangle_area_p10),
        triangle_area_p50=float(triangle_area_p50),
        triangle_area_p90=float(triangle_area_p90),
        min_angle_p10=float(min_angle_p10),
        min_angle_p50=float(min_angle_p50),
        min_angle_p90=float(min_angle_p90),
        heading_p10=float(heading_p10),
        heading_p50=float(heading_p50),
        heading_p90=float(heading_p90),
        rig_id="rig_1",
        camera_id="camera_1",
        arena_id="arena_1",
        dish_design="cedar",
        canvas_name="shadow",
        protocol_name="DefaultScreen",
        genotype=None,
        dpf_at_acquisition=None,
        profile_json=json.dumps(profile_summary),
        zarr_mtime_ns=int(zarr_mtime_ns),
    )


def _seed_subject_lineage(
    registry: Registry,
    *,
    dataset_id: str,
    recording_id: str,
    subject_id: str,
    genotype: str = "Tg(line_a)",
    dpf: int = 7,
) -> None:
    registry.conn.execute(
        "INSERT OR IGNORE INTO recordings (recording_id) VALUES (?);",
        (recording_id,),
    )
    cross_id = f"cross_{subject_id}"
    dish_id = f"dish_{subject_id}"
    registry.conn.execute(
        "INSERT OR IGNORE INTO crosses (cross_id, genotype) VALUES (?, ?);",
        (cross_id, genotype),
    )
    registry.conn.execute(
        "INSERT OR IGNORE INTO dishes (dish_id, cross_id) VALUES (?, ?);",
        (dish_id, cross_id),
    )
    registry.conn.execute(
        "INSERT OR IGNORE INTO subjects (subject_id, dish_id) VALUES (?, ?);",
        (subject_id, dish_id),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id,
            subject_id,
            dataset_id,
            dpf_at_acquisition,
            created_utc,
            updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(recording_id, subject_id) DO UPDATE SET
            dataset_id=excluded.dataset_id,
            dpf_at_acquisition=excluded.dpf_at_acquisition,
            updated_utc=excluded.updated_utc;
        """,
        (
            recording_id,
            subject_id,
            dataset_id,
            int(dpf),
            "2026-02-24T00:00:00+00:00",
            "2026-02-24T00:00:00+00:00",
        ),
    )
    registry.conn.commit()


def _make_pose_root(
    *,
    kpt_count: int,
    skeleton: list[list[int]],
    labels: list[str],
    n_rows: int = 4,
    triangle_area: list[float] | None = None,
    min_angle: list[float] | None = None,
    heading: list[float] | None = None,
    confidence_valid: list[bool] | None = None,
    geometry_valid: list[bool] | None = None,
    usable: list[bool] | None = None,
) -> _FakeGroup:
    if triangle_area is None:
        triangle_area = [110.0, 120.0, 130.0, 140.0]
    if min_angle is None:
        min_angle = [18.0, 20.0, 22.0, 24.0]
    if heading is None:
        heading = [5.0, 10.0, 15.0, 20.0]
    if confidence_valid is None:
        confidence_valid = [True, True, False, True]
    if geometry_valid is None:
        geometry_valid = [True, True, True, False]
    if usable is None:
        usable = [True, True, False, True]

    keypoints = np.zeros((n_rows, kpt_count, 2), dtype=np.float64)
    for idx in range(n_rows):
        for k in range(kpt_count):
            keypoints[idx, k, 0] = 8.0 + float(k) + float(idx) * 0.2
            keypoints[idx, k, 1] = 12.0 + float(k) * 0.5 + float(idx) * 0.1

    root = _FakeGroup(attrs={"session_uuid": "session_fake"})
    crop_parent = root.add_group("crop_runs", attrs={"latest": "crop_001"})
    crop_group = crop_parent.add_group("crop_001", attrs={"detection_source_type": "filtered"})
    crop_group.add_array("roi_images", np.zeros((n_rows, 64, 64), dtype=np.uint8))

    keypoints_parent = root.add_group("keypoints_runs", attrs={"latest": "kp_001"})
    kp_group = keypoints_parent.add_group(
        "kp_001",
        attrs={
            "method": "traditional_pose",
            "source_crop_run": "crop_001",
            "keypoint_labels": labels,
            "keypoint_skeleton": skeleton,
        },
    )
    kp_group.add_array("keypoints_roi", keypoints.astype(np.float32))

    refined_parent = root.add_group("refined_keypoints_runs", attrs={"latest": "refined_001"})
    refined_group = refined_parent.add_group(
        "refined_001",
        attrs={
            "source_keypoints_run": "kp_001",
            "created_utc": "2026-02-24T00:00:00+00:00",
            "summary_statistics": {
                "total_rois": n_rows,
                "flips_corrected": 1,
            },
        },
    )
    refined_group.add_array("keypoints_roi", keypoints.astype(np.float32))
    refined_group.add_array("triangle_area", np.asarray(triangle_area, dtype=np.float64))
    refined_group.add_array("min_angle", np.asarray(min_angle, dtype=np.float64))
    refined_group.add_array("heading", np.asarray(heading, dtype=np.float64))
    refined_group.add_array("confidence_valid", np.asarray(confidence_valid, dtype=np.bool_))
    refined_group.add_array("geometry_valid", np.asarray(geometry_valid, dtype=np.bool_))
    refined_group.add_array("usable_keypoints", np.asarray(usable, dtype=np.bool_))
    return root


def _make_merged_root(
    *,
    train: int,
    val: int,
    test: int = 0,
    source_dataset_ids: list[str] | None = None,
    source_dataset_idx: list[int] | None = None,
) -> _FakeGroup:
    root = _FakeGroup()
    split = root.add_group("splits")
    split.add_array("train_indices", np.arange(train, dtype=np.int64))
    split.add_array("val_indices", np.arange(train, train + val, dtype=np.int64))
    if test > 0:
        split.add_array("test_indices", np.arange(train + val, train + val + test, dtype=np.int64))

    total = int(train + val + test)
    source_ids = list(source_dataset_ids or ["dataset_0"])
    dataset_idx = list(source_dataset_idx or ([0] * total))
    if len(dataset_idx) != total:
        raise ValueError("source_dataset_idx length must match total split rows.")

    source_index = root.add_group("source_index")
    source_index.add_array("source_dataset_idx", np.asarray(dataset_idx, dtype=np.int64))
    source_index.add_array("source_dataset_id", np.asarray(source_ids, dtype=object))
    return root


def _patch_open_group(monkeypatch, mapping: dict[Path, _FakeGroup]) -> None:
    by_path = {str(path.resolve()): root for path, root in mapping.items()}

    def _fake_open_group(path: str, **_kwargs):
        key = str(Path(path).resolve())
        if key not in by_path:
            raise KeyError(f"Unexpected fake zarr path: {path}")
        return by_path[key]

    monkeypatch.setattr(mod.zarr, "open_group", _fake_open_group)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_mixed_skeleton_rejected(tmp_path: Path, monkeypatch, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "pose.manifest.json"
    output_path = tmp_path / "pose.data_card.json"
    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _seed_keypoint_profile(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_keypoint_profile(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    db.close()

    _write_manifest(
        manifest_path,
        {
            "set_id": "pose_mixed_v001",
            "datasets": [
                {"dataset_id": "dataset_a", "zarr_path": str(zarr_a), "keypoint_run": "kp_001"},
                {"dataset_id": "dataset_b", "zarr_path": str(zarr_b), "keypoint_run": "kp_001"},
            ],
        },
    )

    _patch_open_group(
        monkeypatch,
        {
            zarr_a: _make_pose_root(
                kpt_count=3,
                skeleton=[[0, 1], [1, 2]],
                labels=["swim_bladder", "eye_left", "eye_right"],
            ),
            zarr_b: _make_pose_root(
                kpt_count=4,
                skeleton=[[0, 1], [1, 2], [2, 3]],
                labels=["k0", "k1", "k2", "k3"],
            ),
        },
    )

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
        ]
    )
    assert rc == 1
    assert not output_path.exists()
    stdout = capsys.readouterr().out
    assert "Mixed skeleton identities" in stdout


def test_manifest_pose_schema_dims_do_not_false_fail_when_keypoint_count_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "pose.manifest.json"
    output_path = tmp_path / "pose.data_card.json"
    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _seed_keypoint_profile(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_keypoint_profile(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    db.close()

    _write_manifest(
        manifest_path,
        {
            "set_id": "pose_schema_dims_v001",
            "datasets": [
                {"dataset_id": "dataset_a", "zarr_path": str(zarr_a), "keypoint_run": "kp_001"},
                {"dataset_id": "dataset_b", "zarr_path": str(zarr_b), "keypoint_run": "kp_001"},
            ],
            "pose_schema": {
                "skeleton_id": "pose_schema:traditional_v1",
                "kpt_shape": [3, 3],
                "keypoint_labels": ["bladder", "eye_left", "eye_right"],
                "skeleton": None,
            },
        },
    )

    pose_root_a = _make_pose_root(
        kpt_count=3,
        skeleton=[[0, 1], [0, 2], [1, 2]],
        labels=["bladder", "eye_left", "eye_right"],
    )
    # Real keypoint runs can carry edges under pose_schema instead of keypoint_skeleton.
    del pose_root_a["keypoints_runs"]["kp_001"].attrs["keypoint_skeleton"]
    pose_root_a["keypoints_runs"]["kp_001"].attrs["pose_schema"] = {
        "name": "traditional_v1",
        "nodes": ["bladder", "eye_left", "eye_right"],
        "edges": [[0, 1], [0, 2], [1, 2]],
    }
    pose_root_b = _make_pose_root(
        kpt_count=3,
        skeleton=[[0, 1], [0, 2], [1, 2]],
        labels=["bladder", "eye_left", "eye_right"],
    )

    _patch_open_group(monkeypatch, {zarr_a: pose_root_a, zarr_b: pose_root_b})

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
        ]
    )
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["skeleton_graph_metrics"]["pose_schema"]["kpt_shape"] == [3, 3]
    assert payload["skeleton_graph_metrics"]["pose_schema"]["skeleton"] == [[0, 1], [0, 2], [1, 2]]
    assert "edge_0_1" in payload["skeleton_graph_metrics"]["edge_length_norm_stats"]


def test_subject_lineage_warn_and_require_policy(tmp_path: Path, monkeypatch, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "pose.manifest.json"
    output_path = tmp_path / "pose.data_card.json"
    zarr_a = tmp_path / "a_training.zarr"
    zarr_b = tmp_path / "b_training.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _seed_keypoint_profile(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_keypoint_profile(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _seed_subject_lineage(
        db,
        dataset_id="dataset_a",
        recording_id="rec_a",
        subject_id="subject_a",
        genotype="Tg(line_warn)",
        dpf=7,
    )
    db.close()

    _write_manifest(
        manifest_path,
        {
            "set_id": "pose_lineage_v001",
            "datasets": [
                {"dataset_id": "dataset_a", "zarr_path": str(zarr_a), "keypoint_run": "kp_001"},
                {"dataset_id": "dataset_b", "zarr_path": str(zarr_b), "keypoint_run": "kp_001"},
            ],
            "pose_schema": {
                "kpt_shape": [3, 2],
                "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
                "skeleton": [[0, 1], [0, 2]],
            },
        },
    )

    pose_root = _make_pose_root(
        kpt_count=3,
        skeleton=[[0, 1], [0, 2]],
        labels=["swim_bladder", "eye_left", "eye_right"],
    )
    _patch_open_group(monkeypatch, {zarr_a: pose_root, zarr_b: pose_root})

    rc_warn = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
        ]
    )
    assert rc_warn == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["subject_coverage"] == {
        "manifest_dataset_count": 2,
        "lineage_covered_dataset_count": 1,
        "missing_lineage_dataset_ids": ["dataset_b"],
    }
    assert payload["genotype_counts"] == {"Tg(line_warn)": 1}
    assert payload["dpf_stats"]["count"] == 1
    assert "Subject lineage coverage: 1/2 datasets" in capsys.readouterr().out

    output_path.unlink()
    rc_require = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--subject-lineage-policy",
            "require",
            "--no-plots",
        ]
    )
    assert rc_require == 1
    assert not output_path.exists()
    assert (
        "Missing subject lineage rows in recording_subject_overview for dataset_id(s): dataset_b"
        in capsys.readouterr().out
    )


def test_skeleton_graph_key_generation() -> None:
    pose_schema = mod._parse_pose_schema_mapping(  # noqa: SLF001
        {
            "kpt_shape": [4, 2],
            "keypoint_labels": ["swim_bladder", "eye_left", "eye_right", "tail"],
            "skeleton": [[0, 1], [1, 2], [1, 3]],
        }
    )
    graph_specs = mod._build_graph_metric_specs(pose_schema)  # noqa: SLF001
    edge_keys = {entry["key"] for entry in graph_specs["edges"]}
    angle_keys = {entry["key"] for entry in graph_specs["angles"]}

    assert edge_keys == {"edge_0_1", "edge_1_2", "edge_1_3"}
    assert angle_keys == {"angle_0_1_2", "angle_0_1_3", "angle_2_1_3"}
    aliases = {entry["alias"] for entry in graph_specs["edges"]}
    assert "edge_swim_bladder_eye_left" in aliases


def test_basic_payload_completeness(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "pose.manifest.json"
    output_path = tmp_path / "pose.data_card.json"
    zarr_path = tmp_path / "pose_training.zarr"
    merged_zarr = tmp_path / "pose_merged.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_pose", recording_id="rec_pose", zarr_path=zarr_path)
    _seed_keypoint_quality(db, dataset_id="dataset_pose", zarr_path=zarr_path)
    _seed_keypoint_profile(db, dataset_id="dataset_pose", recording_id="rec_pose", zarr_path=zarr_path)
    _seed_subject_lineage(
        db,
        dataset_id="dataset_pose",
        recording_id="rec_pose",
        subject_id="subject_pose",
        genotype="Tg(line_complete)",
        dpf=8,
    )
    db.close()

    _write_manifest(
        manifest_path,
        {
            "set_id": "pose_complete_v001",
            "set_version": 1,
            "datasets": [
                {
                    "dataset_id": "dataset_pose",
                    "zarr_path": str(zarr_path),
                    "keypoint_run": "kp_001",
                    "keypoints_total": 4,
                    "keypoints_successful": 3,
                    "keypoints_success_rate": 0.75,
                }
            ],
            "pose_schema": {
                "kpt_shape": [3, 2],
                "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
                "skeleton": [[0, 1], [0, 2]],
            },
        },
    )

    _patch_open_group(
        monkeypatch,
        {
            zarr_path: _make_pose_root(
                kpt_count=3,
                skeleton=[[0, 1], [0, 2]],
                labels=["swim_bladder", "eye_left", "eye_right"],
            ),
            merged_zarr: _make_merged_root(
                train=3,
                val=1,
                source_dataset_ids=["dataset_pose"],
                source_dataset_idx=[0, 0, 0, 0],
            ),
        },
    )

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--merged-zarr",
            str(merged_zarr),
            "--split",
            "train",
            "--no-plots",
        ]
    )
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    for key in (
        "selection",
        "quality",
        "geometry",
        "skeleton_graph_metrics",
        "spatial",
        "composition_counts",
        "subject_coverage",
        "genotype_counts",
        "dpf_stats",
        "dpf_histogram",
        "train_val_parity",
        "audit_freshness",
    ):
        assert key in payload

    assert payload["selection"]["dataset_count"] == 1
    assert payload["selection"]["rows_pre_gate"] == 4
    assert payload["selection"]["rows_post_gate"] == 4
    assert payload["selection"]["split_counts"] == {"train": 3, "val": 1}
    assert payload["quality"]["usable_keypoints_total"] == 3
    assert payload["quality"]["usable_keypoints_rate_overall"] == 0.75
    assert payload["quality"]["usable_keypoints_rate_histogram"] is not None
    assert payload["geometry"]["triangle_area"]["stats"]["count"] == 4
    assert "edge_0_1" in payload["skeleton_graph_metrics"]["edge_length_norm_stats"]
    assert "angle_1_0_2" in payload["skeleton_graph_metrics"]["angle_stats"]
    assert "0" in payload["spatial"]["landmark_center_heatmaps"]
    assert payload["spatial"]["landmark_center_heatmaps"]["0"]["alias"] == "swim_bladder"
    assert payload["subject_coverage"]["lineage_covered_dataset_count"] == 1
    assert payload["genotype_counts"] == {"Tg(line_complete)": 1}
    assert payload["dpf_stats"]["max"] == 8.0
    assert payload["train_val_parity"] is not None
    assert payload["train_val_parity"]["metrics"]["usable_keypoints_rate"]["delta"] == 0.0
    assert payload["audit_freshness"]["zarr_mtime_mismatch_count"] == 0


def test_train_val_parity_delta_from_merged_source_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "pose.manifest.json"
    output_path = tmp_path / "pose.data_card.json"
    merged_zarr = tmp_path / "pose_merged.zarr"
    zarr_a = tmp_path / "pose_a_training.zarr"
    zarr_b = tmp_path / "pose_b_training.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_a", recording_id="rec_a", zarr_path=zarr_a)
    _seed_dataset(db, dataset_id="dataset_b", recording_id="rec_b", zarr_path=zarr_b)
    _seed_keypoint_quality(db, dataset_id="dataset_a", zarr_path=zarr_a, source_keypoint_run="kp_001")
    _seed_keypoint_quality(db, dataset_id="dataset_b", zarr_path=zarr_b, source_keypoint_run="kp_001")
    _seed_keypoint_profile(
        db,
        dataset_id="dataset_a",
        recording_id="rec_a",
        zarr_path=zarr_a,
        source_keypoint_run="kp_001",
    )
    _seed_keypoint_profile(
        db,
        dataset_id="dataset_b",
        recording_id="rec_b",
        zarr_path=zarr_b,
        source_keypoint_run="kp_001",
        usable_rate=0.50,
        confidence_valid_rate=0.50,
        geometry_valid_rate=0.50,
        triangle_area_p50=105.0,
        triangle_area_p90=120.0,
        min_angle_p50=15.0,
        min_angle_p90=18.0,
    )
    _seed_subject_lineage(
        db,
        dataset_id="dataset_a",
        recording_id="rec_a",
        subject_id="subject_a",
        genotype="Tg(line_a)",
        dpf=7,
    )
    _seed_subject_lineage(
        db,
        dataset_id="dataset_b",
        recording_id="rec_b",
        subject_id="subject_b",
        genotype="Tg(line_b)",
        dpf=10,
    )
    db.close()

    _write_manifest(
        manifest_path,
        {
            "set_id": "pose_parity_v001",
            "datasets": [
                {"dataset_id": "dataset_a", "zarr_path": str(zarr_a), "keypoint_run": "kp_001"},
                {"dataset_id": "dataset_b", "zarr_path": str(zarr_b), "keypoint_run": "kp_001"},
            ],
            "pose_schema": {
                "kpt_shape": [3, 2],
                "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
                "skeleton": [[0, 1], [0, 2]],
            },
        },
    )

    _patch_open_group(
        monkeypatch,
        {
            zarr_a: _make_pose_root(
                kpt_count=3,
                skeleton=[[0, 1], [0, 2]],
                labels=["swim_bladder", "eye_left", "eye_right"],
            ),
            zarr_b: _make_pose_root(
                kpt_count=3,
                skeleton=[[0, 1], [0, 2]],
                labels=["swim_bladder", "eye_left", "eye_right"],
            ),
            merged_zarr: _make_merged_root(
                train=4,
                val=4,
                source_dataset_ids=["dataset_a", "dataset_b"],
                source_dataset_idx=[0, 0, 0, 0, 1, 1, 1, 1],
            ),
        },
    )

    rc = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--merged-zarr",
            str(merged_zarr),
            "--no-plots",
        ]
    )
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    parity = payload["train_val_parity"]
    assert parity is not None

    usable_rate_delta = parity["metrics"]["usable_keypoints_rate"]["delta"]
    assert usable_rate_delta == pytest.approx(0.25)
    triangle_p50_delta = parity["metrics"]["triangle_area_p50"]["delta"]
    assert triangle_p50_delta == pytest.approx(20.0)
    assert parity["lineage"]["genotype_mix_max_abs_delta"] == pytest.approx(1.0)
    assert parity["lineage"]["dpf_mean"]["delta"] == pytest.approx(3.0)


def test_missing_profile_rows_fail_closed_unless_fallback_enabled(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "pose.manifest.json"
    output_path = tmp_path / "pose.data_card.json"
    zarr_path = tmp_path / "pose_training.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_pose", recording_id="rec_pose", zarr_path=zarr_path)
    _seed_keypoint_quality(db, dataset_id="dataset_pose", zarr_path=zarr_path)
    db.close()

    _write_manifest(
        manifest_path,
        {
            "set_id": "pose_missing_profile_v001",
            "datasets": [
                {
                    "dataset_id": "dataset_pose",
                    "zarr_path": str(zarr_path),
                    "keypoint_run": "kp_001",
                }
            ],
            "pose_schema": {
                "kpt_shape": [3, 2],
                "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
                "skeleton": [[0, 1], [0, 2]],
            },
        },
    )
    _patch_open_group(
        monkeypatch,
        {
            zarr_path: _make_pose_root(
                kpt_count=3,
                skeleton=[[0, 1], [0, 2]],
                labels=["swim_bladder", "eye_left", "eye_right"],
            ),
        },
    )

    rc_fail = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
        ]
    )
    assert rc_fail == 1
    stdout_fail = capsys.readouterr().out
    assert "Missing keypoint_data_profile_latest rows" in stdout_fail
    assert "--refresh-keypoint-profiles" in stdout_fail

    rc_fallback = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--allow-profile-fallback-scan",
            "--no-plots",
        ]
    )
    assert rc_fallback == 0
    assert output_path.exists()


def test_query_dataset_context_rows_prefers_canonical_recording_context(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        zarr_path = tmp_path / "pose_training.zarr"
        _seed_dataset(registry, dataset_id="dataset_pose", recording_id="rec_pose", zarr_path=zarr_path)
        _seed_recording_context(
            registry,
            recording_id="rec_pose",
            session_uuid="dataset_pose_session",
            root=tmp_path,
            rig_id="rig_recording",
            arena_id="arena_recording",
            camera_id="camera_recording",
            canvas_name="canvas_recording",
            protocol_name="protocol_recording",
            dish_design="dish_design_recording",
        )

        rows = mod._query_dataset_context_rows(registry, dataset_ids=["dataset_pose"])  # noqa: SLF001
    finally:
        registry.close()

    assert rows["dataset_pose"]["rig_id"] == "rig_recording"
    assert rows["dataset_pose"]["arena_id"] == "arena_recording"
    assert rows["dataset_pose"]["camera_id"] == "camera_recording"
    assert rows["dataset_pose"]["canvas_name"] == "canvas_recording"
    assert rows["dataset_pose"]["protocol_name"] == "protocol_recording"
    assert rows["dataset_pose"]["dish_design"] == "dish_design_recording"


def test_profile_mtime_mismatch_fail_closed_unless_allowed(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "pose.manifest.json"
    output_path = tmp_path / "pose.data_card.json"
    zarr_path = tmp_path / "pose_training.zarr"

    db = Registry(registry_path)
    _seed_dataset(db, dataset_id="dataset_pose", recording_id="rec_pose", zarr_path=zarr_path)
    _seed_keypoint_quality(db, dataset_id="dataset_pose", zarr_path=zarr_path)
    _seed_keypoint_profile(
        db,
        dataset_id="dataset_pose",
        recording_id="rec_pose",
        zarr_path=zarr_path,
        zarr_mtime_ns=int(zarr_path.stat().st_mtime_ns) + 1,
    )
    db.close()

    _write_manifest(
        manifest_path,
        {
            "set_id": "pose_stale_profile_v001",
            "datasets": [
                {
                    "dataset_id": "dataset_pose",
                    "zarr_path": str(zarr_path),
                    "keypoint_run": "kp_001",
                }
            ],
            "pose_schema": {
                "kpt_shape": [3, 2],
                "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
                "skeleton": [[0, 1], [0, 2]],
            },
        },
    )
    _patch_open_group(
        monkeypatch,
        {
            zarr_path: _make_pose_root(
                kpt_count=3,
                skeleton=[[0, 1], [0, 2]],
                labels=["swim_bladder", "eye_left", "eye_right"],
            ),
        },
    )

    rc_fail = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--no-plots",
        ]
    )
    assert rc_fail == 1
    stdout_fail = capsys.readouterr().out
    assert "Stale keypoint_data_profile_latest row(s)" in stdout_fail
    assert "--refresh-keypoint-profiles" in stdout_fail

    rc_allow = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
            "--allow-profile-mtime-mismatch",
            "--no-plots",
        ]
    )
    assert rc_allow == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["audit_freshness"]["zarr_mtime_mismatch_count"] == 1
