from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import index_training_data_cards as mod


def _write_card(path: Path, *, schema_name: str, set_id: str, dataset_count: int, split: str) -> None:
    payload = {
        "schema_name": schema_name,
        "set_id": set_id,
        "selection": {"dataset_count": dataset_count, "split": split},
        "updated_utc": "2026-03-04T20:00:00+00:00",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_training_card_entries_detects_cards_and_plots(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    card_path = datasets_root / "pose_test_v001" / "pose_test_v001.data_card.json"
    plot_dir = card_path.parent / "pose_test_v001.data_card.plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "pose_test_v001.heading_distribution.png").write_bytes(b"PNG")
    _write_card(
        card_path,
        schema_name="keypoint_training_data_card",
        set_id="pose_test_v001",
        dataset_count=12,
        split="train",
    )

    entries = mod.collect_training_card_entries(datasets_root=datasets_root)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "pose"
    assert entry.schema_name == "keypoint_training_data_card"
    assert entry.set_id == "pose_test_v001"
    assert entry.dataset_count == 12
    assert entry.split == "train"
    assert len(entry.plot_paths) == 1
    assert entry.plot_paths[0].name.endswith(".png")


def test_render_training_card_index_html_includes_relative_links(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    card_path = datasets_root / "detect_test_v001" / "detect_test_v001.data_card.json"
    plot_dir = card_path.parent / "detect_test_v001.data_card.plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "detect_test_v001.w_norm.png").write_bytes(b"PNG")
    _write_card(
        card_path,
        schema_name="detection_training_data_card",
        set_id="detect_test_v001",
        dataset_count=3,
        split="train",
    )
    entries = mod.collect_training_card_entries(datasets_root=datasets_root)
    output_html = datasets_root / "_index" / "training_data_cards_index.html"

    html = mod.render_training_card_index_html(
        entries=entries,
        datasets_root=datasets_root,
        output_html=output_html,
        title="Test Index",
        thumb_width=200,
    )
    assert "detect_test_v001" in html
    assert "detection_training_data_card" in html
    assert "../detect_test_v001/detect_test_v001.data_card.json" in html
    assert "detect_test_v001.w_norm.png" in html


def test_main_writes_output_html(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    card_path = datasets_root / "eye_mask_test_v001" / "eye_mask_test_v001.data_card.json"
    _write_card(
        card_path,
        schema_name="eye_mask_training_data_card",
        set_id="eye_mask_test_v001",
        dataset_count=2,
        split="val",
    )
    output_html = tmp_path / "cards_index.html"

    rc = mod.main(
        [
            "--datasets-root",
            str(datasets_root),
            "--output-html",
            str(output_html),
            "--thumb-width",
            "280",
        ]
    )
    assert rc == 0
    assert output_html.exists()
    html = output_html.read_text(encoding="utf-8")
    assert "eye_mask_test_v001" in html
    assert "Eye-Mask" in html
