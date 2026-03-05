from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import plot_keypoint_training_data_card as mod


def _sample_card_payload() -> dict[str, object]:
    return {
        "schema_name": "keypoint_training_data_card",
        "schema_version": "v1",
        "set_id": "pose_sample_v001",
        "quality": {
            "usable_keypoints_rate_histogram": {
                "bin_edges": [0.0, 0.5, 1.0],
                "counts": [10, 90],
            }
        },
        "geometry": {
            "triangle_area_histogram": {
                "bin_edges": [0.0, 10.0, 20.0],
                "counts": [30, 70],
            },
            "min_angle_histogram": {
                "bin_edges": [0.0, 45.0, 90.0],
                "counts": [40, 60],
            },
            "heading_histogram": {
                "bin_edges": [0.0, 180.0, 360.0],
                "counts": [55, 45],
            },
        },
        "spatial": {
            "landmark_center_heatmaps": {
                "0": {
                    "alias": "eye_left",
                    "grid_h": 2,
                    "grid_w": 2,
                    "density": [0.1, 0.2, 0.3, 0.4],
                },
                "1": {
                    "alias": "eye_right",
                    "grid_h": 2,
                    "grid_w": 2,
                    "density": [0.4, 0.3, 0.2, 0.1],
                },
            }
        },
        "skeleton_graph_metrics": {
            "edge_length_norm_stats": {
                "edge_0_1": {
                    "alias": "edge_swim_bladder_eye_left",
                    "histogram": {
                        "bin_edges": [0.0, 0.5, 1.0],
                        "counts": [20, 80],
                    },
                },
                "edge_0_2": {
                    "alias": "edge_swim_bladder_eye_right",
                    "histogram": {
                        "bin_edges": [0.0, 0.4, 0.8],
                        "counts": [30, 70],
                    },
                },
            }
        },
        "genotype_counts": {
            "Tg(elavl3:gcamp7f)": 2,
            "wt": 1,
        },
        "dpf_histogram": {
            "bin_edges": [4.5, 5.5, 6.5, 7.5],
            "counts": [1, 1, 1],
        },
    }


def _sample_card_payload_modern() -> dict[str, object]:
    return {
        "schema_name": "keypoint_training_data_card",
        "schema_version": "v1",
        "set_id": "pose_sample_v001",
        "quality": {
            "usable_keypoints_rate_histogram": {
                "bin_edges": [0.0, 0.5, 1.0],
                "counts": [10, 90],
            }
        },
        "geometry": {
            "triangle_area": {
                "histogram": {
                    "bin_edges": [0.0, 10.0, 20.0],
                    "counts": [30, 70],
                }
            },
            "min_angle": {
                "histogram": {
                    "bin_edges": [0.0, 45.0, 90.0],
                    "counts": [40, 60],
                }
            },
            "heading": {
                "histogram": {
                    "bin_edges": [0.0, 180.0, 360.0],
                    "counts": [55, 45],
                }
            },
        },
        "spatial": {
            "landmark_center_heatmaps": {
                "0": {
                    "alias": "eye_left",
                    "grid_h": 2,
                    "grid_w": 2,
                    "density": [0.1, 0.2, 0.3, 0.4],
                },
                "1": {
                    "alias": "eye_right",
                    "grid_h": 2,
                    "grid_w": 2,
                    "density": [0.4, 0.3, 0.2, 0.1],
                },
            }
        },
        "skeleton_graph_metrics": {
            "edge_length_norm_stats": {
                "edge_0_1": {
                    "alias": "edge_swim_bladder_eye_left",
                    "histogram": {
                        "bin_edges": [0.0, 0.5, 1.0],
                        "counts": [20, 80],
                    },
                },
                "edge_0_2": {
                    "alias": "edge_swim_bladder_eye_right",
                    "histogram": {
                        "bin_edges": [0.0, 0.4, 0.8],
                        "counts": [30, 70],
                    },
                },
            }
        },
        "genotype_counts": {
            "Tg(elavl3:gcamp7f)": 2,
            "wt": 1,
        },
        "dpf_histogram": {
            "bin_edges": [4.5, 5.5, 6.5, 7.5],
            "counts": [1, 1, 1],
        },
    }


def test_plot_keypoint_training_data_card_writes_expected_pngs(tmp_path: Path) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
    output_dir = tmp_path / "plots"
    card_path.write_text(json.dumps(_sample_card_payload()), encoding="utf-8")

    rc = mod.main(
        [
            "--card",
            str(card_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / "pose_sample_v001.usable_rate_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.triangle_area_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.min_angle_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.heading_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.landmark_heatmap_panel.png").exists()
    assert (output_dir / "pose_sample_v001.edge_length_norm_panel.png").exists()
    assert (output_dir / "pose_sample_v001.genotype_counts.png").exists()
    assert (output_dir / "pose_sample_v001.dpf_histogram.png").exists()


def test_plot_keypoint_training_data_card_writes_expected_pngs_modern_schema(tmp_path: Path) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
    output_dir = tmp_path / "plots"
    card_path.write_text(json.dumps(_sample_card_payload_modern()), encoding="utf-8")

    rc = mod.main(
        [
            "--card",
            str(card_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / "pose_sample_v001.usable_rate_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.triangle_area_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.min_angle_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.heading_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.landmark_heatmap_panel.png").exists()
    assert (output_dir / "pose_sample_v001.edge_length_norm_panel.png").exists()
    assert (output_dir / "pose_sample_v001.genotype_counts.png").exists()
    assert (output_dir / "pose_sample_v001.dpf_histogram.png").exists()


def test_plot_keypoint_training_data_card_dry_run_does_not_write(tmp_path: Path) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
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


def test_plot_keypoint_training_data_card_view_uses_existing_files(tmp_path: Path, monkeypatch) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
    output_dir = tmp_path / "plots"
    payload = _sample_card_payload()
    card_path.write_text(json.dumps(payload), encoding="utf-8")

    expected = mod._expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix="pose_sample_v001")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in expected:
        path.write_bytes(b"PNG")

    opened: list[Path] = []

    def _fake_open(paths):
        opened.extend(Path(p) for p in paths)
        return 0

    def _fail_generate(*, card_payload, output_dir, prefix, heatmap_bin_factor):
        raise AssertionError("generate_keypoint_training_data_card_plots should not be called for existing plots.")

    monkeypatch.setattr(mod, "_open_paths", _fake_open)
    monkeypatch.setattr(mod, "generate_keypoint_training_data_card_plots", _fail_generate)

    rc = mod.main(["--card", str(card_path), "--output-dir", str(output_dir), "--view"])
    assert rc == 0
    assert sorted(opened) == sorted(expected)


def test_plot_keypoint_training_data_card_view_generates_when_missing(tmp_path: Path, monkeypatch) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
    output_dir = tmp_path / "plots"
    payload = _sample_card_payload()
    card_path.write_text(json.dumps(payload), encoding="utf-8")

    opened: list[Path] = []

    def _fake_open(paths):
        opened.extend(Path(p) for p in paths)
        return 0

    def _fake_generate(*, card_payload, output_dir, prefix, heatmap_bin_factor):
        assert int(heatmap_bin_factor) == 2
        paths = mod._expected_plot_paths(card_payload=card_payload, output_dir=output_dir, prefix=prefix)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            path.write_bytes(b"PNG")
        return paths

    monkeypatch.setattr(mod, "_open_paths", _fake_open)
    monkeypatch.setattr(mod, "generate_keypoint_training_data_card_plots", _fake_generate)

    rc = mod.main(["--card", str(card_path), "--output-dir", str(output_dir), "--view"])
    assert rc == 0
    expected = mod._expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix="pose_sample_v001")
    assert sorted(opened) == sorted(expected)
    for path in opened:
        assert path.exists()


def test_plot_keypoint_training_data_card_view_force_regenerates_existing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
    output_dir = tmp_path / "plots"
    payload = _sample_card_payload()
    card_path.write_text(json.dumps(payload), encoding="utf-8")

    expected = mod._expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix="pose_sample_v001")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in expected:
        path.write_bytes(b"OLD")

    opened: list[Path] = []
    generated_calls = {"count": 0}

    def _fake_open(paths):
        opened.extend(Path(p) for p in paths)
        return 0

    def _fake_generate(*, card_payload, output_dir, prefix, heatmap_bin_factor):
        generated_calls["count"] += 1
        paths = mod._expected_plot_paths(card_payload=card_payload, output_dir=output_dir, prefix=prefix)
        for path in paths:
            path.write_bytes(b"NEW")
        return paths

    monkeypatch.setattr(mod, "_open_paths", _fake_open)
    monkeypatch.setattr(mod, "generate_keypoint_training_data_card_plots", _fake_generate)

    rc = mod.main(["--card", str(card_path), "--output-dir", str(output_dir), "--view", "--force"])
    assert rc == 0
    assert generated_calls["count"] == 1
    assert sorted(opened) == sorted(expected)
    for path in expected:
        assert path.read_bytes() == b"NEW"


def test_plot_keypoint_training_data_card_rejects_view_with_dry_run(tmp_path: Path) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
    card_path.write_text(json.dumps(_sample_card_payload()), encoding="utf-8")

    with pytest.raises(SystemExit):
        mod.main(["--card", str(card_path), "--view", "--dry-run"])


def test_plot_keypoint_training_data_card_rejects_nonpositive_heatmap_bin_factor(tmp_path: Path) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
    card_path.write_text(json.dumps(_sample_card_payload()), encoding="utf-8")

    with pytest.raises(SystemExit):
        mod.main(["--card", str(card_path), "--heatmap-bin-factor", "0"])


def test_plot_keypoint_training_data_card_skips_empty_subject_aggregates(tmp_path: Path) -> None:
    card_path = tmp_path / "pose_sample.data_card.json"
    output_dir = tmp_path / "plots"
    payload = _sample_card_payload()
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
    assert (output_dir / "pose_sample_v001.usable_rate_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.triangle_area_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.min_angle_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.heading_distribution.png").exists()
    assert (output_dir / "pose_sample_v001.landmark_heatmap_panel.png").exists()
    assert (output_dir / "pose_sample_v001.edge_length_norm_panel.png").exists()
    assert not (output_dir / "pose_sample_v001.genotype_counts.png").exists()
    assert not (output_dir / "pose_sample_v001.dpf_histogram.png").exists()


def test_open_paths_invokes_xdg_open_via_subprocess_run(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"PNG")
    second.write_bytes(b"PNG")

    calls: list[list[str]] = []

    class _Completed:
        def __init__(self, returncode: int):
            self.returncode = returncode

    def _fake_run(command, check=False):
        calls.append(list(command))
        assert check is False
        return _Completed(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    open_errors = mod._open_paths([first, second])
    assert open_errors == 0
    assert calls == [
        ["xdg-open", str(first)],
        ["xdg-open", str(second)],
    ]


def test_infer_integer_center_ticks_from_half_step_edges() -> None:
    edges = np.asarray([11.5, 12.5], dtype=np.float64)
    ticks = mod._infer_integer_center_ticks(edges)
    assert ticks is not None
    assert ticks.tolist() == [12.0]


def test_infer_integer_center_ticks_returns_none_for_non_integer_centers() -> None:
    edges = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    ticks = mod._infer_integer_center_ticks(edges)
    assert ticks is None


def test_generate_keypoint_training_data_card_plots_uses_integer_ticks_for_dpf(
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

    generated = mod.generate_keypoint_training_data_card_plots(
        card_payload=payload,
        output_dir=output_dir,
        prefix="pose_sample_v001",
    )

    assert generated == [output_dir / "pose_sample_v001.dpf_histogram.png"]
    assert calls == [
        {
            "title": "DPF Distribution",
            "xlabel": "DPF at acquisition",
            "output_path": output_dir / "pose_sample_v001.dpf_histogram.png",
            "integer_center_ticks": True,
        }
    ]
