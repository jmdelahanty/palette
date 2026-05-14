from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import plot_training_image_data_card as plot_mod


def test_plot_training_image_data_card_dry_run(tmp_path: Path, capsys) -> None:
    card_path = tmp_path / "training_image_card.json"
    card_path.write_text(
        json.dumps(
            {
                "schema_name": "training_image_data_card",
                "schema_version": "v1",
                "set_id": "detect_sleepyfish_v001",
                "metric_values": {"mean_intensity_p50": [100.0, 120.0]},
                "intensity_histogram_aggregate": {
                    "bin_edges": [0.0, 128.0, 256.0],
                    "counts": [7, 10],
                },
            }
        ),
        encoding="utf-8",
    )

    rc = plot_mod.main(["--card", str(card_path), "--dry-run"])

    assert rc == 0
    assert "mode=dry-run" in capsys.readouterr().out
