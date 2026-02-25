from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import plot_eye_mask_training_data_card as mod


def _sample_card_payload() -> dict[str, object]:
    return {
        "schema_name": "eye_mask_training_data_card",
        "schema_version": "v1",
        "set_id": "eye_mask_sample_v001",
        "selection": {"dataset_count": 3, "split": "train", "filters": {}},
        "quality": {
            "successful_roi_pair_rate_histogram": {
                "bin_edges": [0.0, 0.5, 1.0],
                "counts": [10, 90],
            }
        },
        "geometry": {
            "eye_separation_p50_histogram": {
                "bin_edges": [3.0, 4.0, 5.0, 6.0],
                "counts": [2, 5, 3],
            },
            "ellipse_major_p50_histogram": {
                "bin_edges": [6.0, 7.0, 8.0, 9.0],
                "counts": [2, 6, 2],
            },
            "ellipse_minor_p50_histogram": {
                "bin_edges": [3.0, 4.0, 5.0, 6.0],
                "counts": [1, 7, 2],
            },
            "left_area_p50_histogram": {
                "bin_edges": [120.0, 180.0, 240.0],
                "counts": [3, 7],
            },
            "right_area_p50_histogram": {
                "bin_edges": [125.0, 185.0, 245.0],
                "counts": [4, 6],
            },
            "union_area_p50_histogram": {
                "bin_edges": [260.0, 340.0, 420.0],
                "counts": [2, 8],
            },
            "area_lr_ratio_p50_histogram": {
                "bin_edges": [0.8, 0.95, 1.1],
                "counts": [5, 5],
            },
        },
        "spatial": {
            "center_heatmap": {
                "grid_h": 2,
                "grid_w": 2,
                "density": [0.1, 0.2, 0.3, 0.4],
            }
        },
        "composition": {
            "counts": {
                "camera_id": {
                    "cam_a": 2,
                    "cam_b": 1,
                }
            }
        },
        "subject_coverage": {
            "manifest_dataset_count": 3,
            "lineage_covered_dataset_count": 3,
            "missing_lineage_dataset_ids": [],
        },
        "parity": {
            "available": True,
            "metrics": {
                "successful_roi_pair_rate": {"train": 0.92, "val": 0.85, "delta": 0.07},
                "eye_separation_p50": {"train": 5.3, "val": 5.1, "delta": 0.2},
            },
        },
        "audit_freshness": {"profile_source": "registry_sql_view"},
        "genotype_counts": {
            "Tg(line_a)": 2,
            "wt": 1,
        },
        "dpf_histogram": {
            "bin_edges": [6.5, 7.5, 8.5],
            "counts": [1, 2],
        },
    }


def test_plot_eye_mask_training_data_card_writes_expected_pngs(tmp_path: Path) -> None:
    card_path = tmp_path / "eye_mask_sample.data_card.json"
    output_dir = tmp_path / "plots"
    card_path.write_text(json.dumps(_sample_card_payload()), encoding="utf-8")

    rc = mod.main(
        [
            "--card-json",
            str(card_path),
            "--outdir",
            str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / "eye_mask_sample_v001.usable_rate_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.eye_separation_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.ellipse_major_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.ellipse_minor_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.left_area_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.right_area_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.union_area_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.left_right_area_ratio_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.center_heatmap.png").exists()
    assert (output_dir / "eye_mask_sample_v001.composition_counts.png").exists()
    assert (output_dir / "eye_mask_sample_v001.parity_train_val_delta.png").exists()
    assert (output_dir / "eye_mask_sample_v001.genotype_counts.png").exists()
    assert (output_dir / "eye_mask_sample_v001.dpf_histogram.png").exists()


def test_plot_eye_mask_training_data_card_dry_run_does_not_write(tmp_path: Path) -> None:
    card_path = tmp_path / "eye_mask_sample.data_card.json"
    output_dir = tmp_path / "plots"
    card_path.write_text(json.dumps(_sample_card_payload()), encoding="utf-8")

    rc = mod.main(
        [
            "--card",
            str(card_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not output_dir.exists()


def test_plot_eye_mask_training_data_card_view_uses_existing_files(tmp_path: Path, monkeypatch) -> None:
    card_path = tmp_path / "eye_mask_sample.data_card.json"
    output_dir = tmp_path / "plots"
    payload = _sample_card_payload()
    card_path.write_text(json.dumps(payload), encoding="utf-8")

    expected = mod._expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix="eye_mask_sample_v001")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in expected:
        path.write_bytes(b"PNG")

    opened: list[Path] = []

    def _fake_open(paths):
        opened.extend(Path(path) for path in paths)
        return 0

    def _fail_generate(*, card_payload, output_dir, prefix, heatmap_bin_factor):
        _ = card_payload, output_dir, prefix, heatmap_bin_factor
        raise AssertionError("generate_eye_mask_training_data_card_plots should not be called for existing plots.")

    monkeypatch.setattr(mod, "_open_paths", _fake_open)
    monkeypatch.setattr(mod, "generate_eye_mask_training_data_card_plots", _fail_generate)

    rc = mod.main(["--card", str(card_path), "--output-dir", str(output_dir), "--view"])
    assert rc == 0
    assert sorted(opened) == sorted(expected)


def test_plot_eye_mask_training_data_card_skips_optional_empty_sections(tmp_path: Path) -> None:
    card_path = tmp_path / "eye_mask_sample.data_card.json"
    output_dir = tmp_path / "plots"
    payload = _sample_card_payload()
    payload["composition"] = {"counts": {}}
    payload["parity"] = {"available": False, "metrics": {}}
    payload["genotype_counts"] = {}
    payload["dpf_histogram"] = {}
    card_path.write_text(json.dumps(payload), encoding="utf-8")

    rc = mod.main(
        [
            "--card",
            str(card_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / "eye_mask_sample_v001.usable_rate_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.eye_separation_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.ellipse_major_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.ellipse_minor_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.left_area_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.right_area_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.union_area_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.left_right_area_ratio_distribution.png").exists()
    assert (output_dir / "eye_mask_sample_v001.center_heatmap.png").exists()
    assert not (output_dir / "eye_mask_sample_v001.composition_counts.png").exists()
    assert not (output_dir / "eye_mask_sample_v001.parity_train_val_delta.png").exists()
    assert not (output_dir / "eye_mask_sample_v001.genotype_counts.png").exists()
    assert not (output_dir / "eye_mask_sample_v001.dpf_histogram.png").exists()


def test_infer_integer_center_ticks_from_half_step_edges() -> None:
    edges = np.asarray([11.5, 12.5], dtype=np.float64)
    ticks = mod._infer_integer_center_ticks(edges)
    assert ticks is not None
    assert ticks.tolist() == [12.0]


def test_generate_eye_mask_training_data_card_plots_uses_integer_ticks_for_dpf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = {"dpf_histogram": {"bin_edges": [11.5, 12.5], "counts": [3]}}
    output_dir = tmp_path / "plots"
    calls: list[dict[str, object]] = []

    def _fake_plot_histogram(
        *,
        hist,
        title,
        xlabel,
        output_path,
        integer_center_ticks=False,
    ):
        _ = hist
        calls.append(
            {
                "title": title,
                "xlabel": xlabel,
                "output_path": Path(output_path),
                "integer_center_ticks": bool(integer_center_ticks),
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"PNG")

    monkeypatch.setattr(mod, "_plot_histogram", _fake_plot_histogram)

    generated = mod.generate_eye_mask_training_data_card_plots(
        card_payload=payload,
        output_dir=output_dir,
        prefix="eye_mask_sample_v001",
    )

    assert generated == [output_dir / "eye_mask_sample_v001.dpf_histogram.png"]
    assert calls == [
        {
            "title": "DPF Distribution",
            "xlabel": "DPF at acquisition",
            "output_path": output_dir / "eye_mask_sample_v001.dpf_histogram.png",
            "integer_center_ticks": True,
        }
    ]
