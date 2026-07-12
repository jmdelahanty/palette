from __future__ import annotations

from pathlib import Path
import subprocess


def test_submit_analytics_export_bsub_renders_fail_closed_job(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    collection = tmp_path / "collection.manifest.json"
    collection.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "shared" / "palette_analytics"
    log_dir = tmp_path / "logs"
    run_id = "chaser_v2_test_20260712T000000Z"

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "submit_analytics_export_bsub.sh"),
            "--collection-manifest",
            str(collection),
            "--export-run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--palette-repo",
            str(repo),
            "--log-dir",
            str(log_dir),
            "--queue",
            "short",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "mode=render-only" in result.stdout
    assert f"export_run_id={run_id}" in result.stdout
    assert "bsub_command=" in result.stdout
    run_dir = log_dir / f"analytics_export_{run_id}"
    job_script = run_dir / "run_analytics_export.sh"
    text = job_script.read_text(encoding="utf-8")
    assert "fisheye.utils.export_cross_recording_analytics" in text
    assert "fisheye.utils.validate_analytics_export" in text
    assert "fisheye.utils.compute_group_statistics" in text
    assert "Palette commit mismatch" in text
    assert str(output_root) in text


def test_submit_analytics_export_bsub_rejects_unsafe_run_id(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    collection = tmp_path / "collection.manifest.json"
    collection.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "submit_analytics_export_bsub.sh"),
            "--collection-manifest",
            str(collection),
            "--export-run-id",
            "../unsafe",
            "--palette-repo",
            str(repo),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Unsafe --export-run-id" in result.stderr
