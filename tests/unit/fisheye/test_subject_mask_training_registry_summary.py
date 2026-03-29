"""Focused registry tests for subject-mask training-set summaries."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import check_training_registry as mod


def test_upsert_training_set_accepts_subject_masks_task_type(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_training_set(
        set_id="subject_masks_set_v001",
        name="subject masks",
        task_type="subject_masks",
        query_filter={"task_type": "subject_masks"},
        dataset_ids=["dataset_a"],
        invocation={"task_type": "subject_masks"},
    )

    row = registry.conn.execute(
        "SELECT task_type FROM training_sets WHERE set_id = ?",
        ("subject_masks_set_v001",),
    ).fetchone()
    assert row is not None
    assert row["task_type"] == "subject_masks"
    registry.close()


def test_load_set_rows_formats_eyes_only_subject_mask_summary(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_training_set(
        set_id="subject_masks_set_v001",
        name="subject masks",
        task_type="subject_masks",
        query_filter={"task_type": "subject_masks"},
        dataset_ids=["dataset_a"],
        invocation={
            "task_type": "subject_masks",
            "subject_mask_training_summary": {
                "coverage_class": "eyes_only",
                "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
                "supervised_row_counts": {
                    "subject_body": 0,
                    "eye_left": 227,
                    "eye_right": 227,
                    "swim_bladder": 0,
                },
            },
        },
    )

    rows = mod._load_set_rows(registry, "subject_masks_set_v001", limit=10)  # noqa: SLF001
    assert len(rows) == 1
    assert rows[0].task_type == "subject_masks"
    assert rows[0].data_summary == "eyes only"
    registry.close()
