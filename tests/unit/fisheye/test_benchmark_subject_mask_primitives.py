import json

from fisheye.diagnostics import benchmark_subject_mask_primitives as mod


def test_benchmark_subject_mask_primitives_synthetic_fill_holes_smoke(capsys):
    rc = mod.main(
        [
            "--operation",
            "fill_holes",
            "--row-count",
            "8",
            "--height",
            "32",
            "--width",
            "32",
            "--repeat",
            "1",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "palette.subject_mask_primitive_benchmark_v1"
    assert payload["source"]["kind"] == "synthetic"
    assert [row["operation"] for row in payload["results"]] == ["fill_holes", "fill_holes"]
    assert payload["results"][0]["backend"] == "palette_cv2_flood_fill"
    assert payload["results"][0]["parity"] == "reference"
    assert payload["results"][1]["backend"] == "scipy_binary_fill_holes"
    assert payload["results"][1]["parity"] == "ok"
