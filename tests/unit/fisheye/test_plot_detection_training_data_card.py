from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import plot_detection_training_data_card as mod


def _sample_card_payload() -> dict[str, object]:
    return {
        "schema_name": "detection_training_data_card",
        "schema_version": "v1",
        "set_id": "detect_sample_v001",
        "histograms_aggregate": {
            "w_norm": {"bin_edges": [0.0, 0.5, 1.0], "counts": [25, 75]},
            "h_norm": {"bin_edges": [0.0, 0.5, 1.0], "counts": [30, 70]},
            "area_norm": {"bin_edges": [0.0, 0.1, 1.0], "counts": [40, 60]},
            "aspect_ratio": {"bin_edges": [0.0, 1.0, 2.0], "counts": [55, 45]},
        },
        "spatial_aggregate": {
            "center_heatmap": {
                "grid_h": 2,
                "grid_w": 2,
                "density": [0.1, 0.2, 0.3, 0.4],
            }
        },
        "genotype_counts": {
            "Tg(elavl3:gcamp7f)": 2,
            "wt": 1,
        },
        "dpf_histogram": {
            "bin_edges": [5.0, 6.0, 7.0, 8.0],
            "counts": [1, 1, 1],
        },
    }


def test_plot_detection_training_data_card_writes_expected_pngs(tmp_path: Path) -> None:
    card_path = tmp_path / "detect_sample.data_card.json"
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
    assert (output_dir / "detect_sample_v001.w_norm.png").exists()
    assert (output_dir / "detect_sample_v001.h_norm.png").exists()
    assert (output_dir / "detect_sample_v001.area_norm.png").exists()
    assert (output_dir / "detect_sample_v001.aspect_ratio.png").exists()
    assert (output_dir / "detect_sample_v001.center_heatmap.png").exists()
    assert (output_dir / "detect_sample_v001.genotype_counts.png").exists()
    assert (output_dir / "detect_sample_v001.dpf_histogram.png").exists()


def test_plot_detection_training_data_card_dry_run_does_not_write(tmp_path: Path) -> None:
    card_path = tmp_path / "detect_sample.data_card.json"
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


def test_plot_detection_training_data_card_view_uses_existing_files(tmp_path: Path, monkeypatch) -> None:
    card_path = tmp_path / "detect_sample.data_card.json"
    output_dir = tmp_path / "plots"
    payload = _sample_card_payload()
    card_path.write_text(json.dumps(payload), encoding="utf-8")

    expected = mod._expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix="detect_sample_v001")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in expected:
        path.write_bytes(b"PNG")

    opened: list[Path] = []

    def _fake_open(paths):
        opened.extend(Path(p) for p in paths)
        return 0

    def _fail_generate(*, card_payload, output_dir, prefix, heatmap_bin_factor):
        raise AssertionError("generate_detection_training_data_card_plots should not be called for existing plots.")

    monkeypatch.setattr(mod, "_open_paths", _fake_open)
    monkeypatch.setattr(mod, "generate_detection_training_data_card_plots", _fail_generate)

    rc = mod.main(["--card", str(card_path), "--output-dir", str(output_dir), "--view"])
    assert rc == 0
    assert sorted(opened) == sorted(expected)


def test_plot_detection_training_data_card_view_generates_when_missing(tmp_path: Path, monkeypatch) -> None:
    card_path = tmp_path / "detect_sample.data_card.json"
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
    monkeypatch.setattr(mod, "generate_detection_training_data_card_plots", _fake_generate)

    rc = mod.main(["--card", str(card_path), "--output-dir", str(output_dir), "--view"])
    assert rc == 0
    expected = mod._expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix="detect_sample_v001")
    assert sorted(opened) == sorted(expected)
    for path in opened:
        assert path.exists()


def test_plot_detection_training_data_card_view_force_regenerates_existing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    card_path = tmp_path / "detect_sample.data_card.json"
    output_dir = tmp_path / "plots"
    payload = _sample_card_payload()
    card_path.write_text(json.dumps(payload), encoding="utf-8")

    expected = mod._expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix="detect_sample_v001")
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
    monkeypatch.setattr(mod, "generate_detection_training_data_card_plots", _fake_generate)

    rc = mod.main(["--card", str(card_path), "--output-dir", str(output_dir), "--view", "--force"])
    assert rc == 0
    assert generated_calls["count"] == 1
    assert sorted(opened) == sorted(expected)
    for path in expected:
        assert path.read_bytes() == b"NEW"


def test_plot_detection_training_data_card_rejects_view_with_dry_run(tmp_path: Path) -> None:
    card_path = tmp_path / "detect_sample.data_card.json"
    card_path.write_text(json.dumps(_sample_card_payload()), encoding="utf-8")

    with pytest.raises(SystemExit):
        mod.main(["--card", str(card_path), "--view", "--dry-run"])


def test_plot_detection_training_data_card_rejects_nonpositive_heatmap_bin_factor(tmp_path: Path) -> None:
    card_path = tmp_path / "detect_sample.data_card.json"
    card_path.write_text(json.dumps(_sample_card_payload()), encoding="utf-8")

    with pytest.raises(SystemExit):
        mod.main(["--card", str(card_path), "--heatmap-bin-factor", "0"])


def test_plot_detection_training_data_card_skips_empty_subject_aggregates(tmp_path: Path) -> None:
    card_path = tmp_path / "detect_sample.data_card.json"
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
    assert (output_dir / "detect_sample_v001.w_norm.png").exists()
    assert (output_dir / "detect_sample_v001.h_norm.png").exists()
    assert (output_dir / "detect_sample_v001.area_norm.png").exists()
    assert (output_dir / "detect_sample_v001.aspect_ratio.png").exists()
    assert (output_dir / "detect_sample_v001.center_heatmap.png").exists()
    assert not (output_dir / "detect_sample_v001.genotype_counts.png").exists()
    assert not (output_dir / "detect_sample_v001.dpf_histogram.png").exists()


def test_histogram_focus_xlim_zooms_to_occupied_bins() -> None:
    edges = np.linspace(0.0, 1.0, 11)
    counts = np.zeros(10, dtype=np.float64)
    counts[1] = 3.0
    counts[2] = 5.0

    xlim = mod._histogram_focus_xlim(edges=edges, counts=counts)
    assert xlim is not None
    assert xlim[0] == pytest.approx(0.09)
    assert xlim[1] == pytest.approx(0.31)


def test_histogram_focus_xlim_returns_full_range_when_empty() -> None:
    edges = np.linspace(0.0, 1.0, 6)
    counts = np.zeros(5, dtype=np.float64)

    xlim = mod._histogram_focus_xlim(edges=edges, counts=counts)
    assert xlim == (0.0, 1.0)
