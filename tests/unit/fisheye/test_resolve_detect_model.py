from __future__ import annotations

from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.registry import model_resolution as mod


def _target() -> mod.TargetProfile:
    return mod.TargetProfile(
        recording_id="recA",
        recording_type="behavior",
        recording_subtype="free",
        behavior_mode="free",
        rig_id="omnifin0",
        arena_id="arena_1",
        camera_id="2010093",
        canvas_name="DefaultScreen",
        protocol_name="cedar",
        dish_design="cedar",
        cross_id="17313",
        genotype="Casper_HHMI",
        dpf_at_acquisition=12,
    )


def _seed_source_recording_with_legacy_provenance(registry: Registry, *, root: Path) -> None:
    registry.upsert_dataset(
        dataset_id="dataset_ctx",
        session_uuid="session_ctx",
        zarr_path=root / "dataset_ctx.zarr",
        recording_id="recording_ctx",
        artifact_kind="source_recording",
    )
    registry.upsert_provenance(
        "dataset_ctx",
        provenance={
            "cross_id": "cross_legacy",
            "genotype": "genotype_legacy",
            "dpf_at_acquisition": 4,
            "snapshot_status": "complete",
        },
        context={
            "rig_id": "rig_legacy",
            "arena_id": "arena_legacy",
            "camera_id": "camera_legacy",
            "canvas_name": "canvas_legacy",
        },
        protocol_name="protocol_legacy",
        protocol_hash="hash_ctx",
        acquisition={"dish_design": "dish_design_legacy"},
        zarr_purpose="analysis",
    )
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
            "recording_ctx",
            "session_ctx",
            "recording_ctx",
            str(root / "recordings" / "recording_ctx"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
            "rig_recording",
            "arena_recording",
            "camera_recording",
            "canvas_recording",
            "protocol_recording",
            "dish_design_recording",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
        VALUES (?, ?, datetime('now'), datetime('now'));
        """,
        ("cross_ctx", "genotype_ctx"),
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
        VALUES (?, ?, datetime('now'), datetime('now'));
        """,
        ("dish_ctx", "cross_ctx"),
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, created_utc, updated_utc)
        VALUES (?, ?, datetime('now'), datetime('now'));
        """,
        ("subject_ctx", "dish_ctx"),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition,
            created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_ctx", "subject_ctx", "dataset_ctx", "dish_ctx", "cross_ctx", 8),
    )
    registry.conn.commit()


def test_score_rows_prefers_better_metadata_match() -> None:
    target = _target()
    good_rows = [
        {
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
            "rig_id": "omnifin0",
            "arena_id": "arena_1",
            "camera_id": "2010093",
            "canvas_name": "DefaultScreen",
            "protocol_name": "cedar",
            "dish_design": "cedar",
            "cross_id": "17313",
            "genotype": "Casper_HHMI",
            "dpf_at_acquisition": 12,
        }
    ]
    weak_rows = [
        {
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
            "rig_id": "omnifin0",
            "arena_id": "arena_2",
            "camera_id": "2010999",
            "canvas_name": "DefaultScreen",
            "protocol_name": "other",
            "dish_design": "other",
            "cross_id": "99999",
            "genotype": "Other",
            "dpf_at_acquisition": 8,
        }
    ]

    good_score, _, _ = mod._score_rows(good_rows, target)  # noqa: SLF001
    weak_score, _, _ = mod._score_rows(weak_rows, target)  # noqa: SLF001

    assert good_score > weak_score
    assert good_score == 1.0


def test_parse_dataset_ids_ignores_invalid_payloads() -> None:
    assert mod._parse_dataset_ids(None) == []  # noqa: SLF001
    assert mod._parse_dataset_ids("{}") == []  # noqa: SLF001
    assert mod._parse_dataset_ids('["a", "b", null, " "]') == ["a", "b"]  # noqa: SLF001


def test_matches_task_filters_pose_from_detect_default() -> None:
    assert mod._matches_task(  # noqa: SLF001
        run_id="omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb",
        set_id="detect_cedar_shadow_v007",
        model_path="/nvme1/models/detect/detect_cedar_shadow_v007/.../best.pt",
        task="detect",
    )
    assert not mod._matches_task(  # noqa: SLF001
        run_id="omnifin0_cedar_shadow_v004_pose_20260208-163716_c9dc72f5",
        set_id="pose_cedar_shadow_manual_gray_latest_traditional_894ad574_v004",
        model_path="/nvme1/models/pose/pose_cedar_shadow_manual_gray_latest_traditional_894ad574_v004/.../best.pt",
        task="detect",
    )


def test_matches_task_identifies_eye_mask_candidates() -> None:
    assert mod._matches_task(  # noqa: SLF001
        run_id="omnifin0_shadow_eye_masks_20260223-010203_abcd1234",
        set_id="eye_masks_shadow_v001",
        model_path="/nvme1/models/eye_masks/eye_masks_shadow_v001/weights/best.pt",
        task="eye_masks",
    )
    assert not mod._matches_task(  # noqa: SLF001
        run_id="omnifin0_cedar_shadow_v004_pose_20260208-163716_c9dc72f5",
        set_id="pose_cedar_shadow_manual_gray_latest_traditional_894ad574_v004",
        model_path="/nvme1/models/pose/pose_cedar_shadow_manual_gray_latest_traditional_894ad574_v004/.../best.pt",
        task="eye_masks",
    )


def test_resolve_recording_id_falls_back_to_recording_dir_for_alias(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    rec_dir = tmp_path / "recordings" / "2026-06-23T16-01-09Z_arena_2_RedScare"
    rec_dir.mkdir(parents=True)
    try:
        registry.conn.execute(
            """
            INSERT INTO recordings (
                recording_id, session_uuid, recording_name, recording_path,
                created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, datetime('now'), datetime('now'));
            """,
            (
                "2026-06-23T16-01-09Z_arena_2",
                "2026-06-23T16-01-09Z_arena_2",
                "2026-06-23T16-01-09Z_arena_2_RedScare",
                str(rec_dir),
            ),
        )
        registry.conn.commit()

        resolved = mod.resolve_recording_id(  # noqa: SLF001
            registry,
            recording_id="2026-06-23T16-01-09Z_arena_2_RedScare",
            recording_dir=rec_dir,
        )
    finally:
        registry.close()

    assert resolved == "2026-06-23T16-01-09Z_arena_2"


def test_load_target_profile_prefers_dataset_context_current(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _seed_source_recording_with_legacy_provenance(registry, root=tmp_path)
        profile = mod.load_target_profile(registry, "recording_ctx")  # noqa: SLF001
    finally:
        registry.close()

    assert profile.recording_id == "recording_ctx"
    assert profile.recording_type == "behavior"
    assert profile.recording_subtype == "free"
    assert profile.behavior_mode == "free"
    assert profile.rig_id == "rig_recording"
    assert profile.arena_id == "arena_recording"
    assert profile.camera_id == "camera_recording"
    assert profile.canvas_name == "canvas_recording"
    assert profile.protocol_name == "protocol_recording"
    assert profile.dish_design == "dish_design_recording"
    assert profile.cross_id == "cross_ctx"
    assert profile.genotype == "genotype_ctx"
    assert profile.dpf_at_acquisition == 8


def test_load_source_rows_for_set_prefers_dataset_context_current(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _seed_source_recording_with_legacy_provenance(registry, root=tmp_path)
        rows = mod._load_source_rows_for_set(registry, ["dataset_ctx"])  # noqa: SLF001
    finally:
        registry.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset_id"] == "dataset_ctx"
    assert row["rig_id"] == "rig_recording"
    assert row["arena_id"] == "arena_recording"
    assert row["camera_id"] == "camera_recording"
    assert row["canvas_name"] == "canvas_recording"
    assert row["protocol_name"] == "protocol_recording"
    assert row["dish_design"] == "dish_design_recording"
    assert row["cross_id"] == "cross_ctx"
    assert row["genotype"] == "genotype_ctx"
    assert row["dpf_at_acquisition"] == 8
