from __future__ import annotations

import os
from pathlib import Path
import subprocess

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/submit_recording_dish_rim_probe_bsub.sh"
)


def _fake_palette_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    palette = tmp_path / "palette"
    (palette / "scripts").mkdir(parents=True)
    (palette / "src/fisheye/diagnostics").mkdir(parents=True)
    (palette / "src/fisheye/utils").mkdir(parents=True)
    py = palette / "scripts/py"
    py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)
    (palette / "src/fisheye/diagnostics/probe_recording_dish_rim_fit.py").touch()
    (palette / "src/fisheye/utils/publish_arena_geometry_fit_review.py").touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"rev-parse HEAD"* ]]; then\n'
        "  printf '%064d\\n' 1\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return palette, env


def _recording(tmp_path: Path, *, with_zarr: bool) -> Path:
    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "recording_clip_index.json").write_text("{}\n", encoding="utf-8")
    if with_zarr:
        archive = recording / "zarr/recording_analysis.zarr"
        archive.mkdir(parents=True)
        (archive / "zarr.json").write_text("{}\n", encoding="utf-8")
    return recording


def _run(
    *,
    recording: Path,
    palette: Path,
    output_root: Path,
    probe_id: str,
    env: dict[str, str],
    diagnostic_only: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        "bash",
        str(SCRIPT),
        "--recording-dir",
        str(recording),
        "--probe-id",
        probe_id,
        "--output-root",
        str(output_root),
        "--palette-repo",
        str(palette),
    ]
    if diagnostic_only:
        command.append("--diagnostic-only")
    return subprocess.run(command, text=True, capture_output=True, env=env)


def test_clipped_probe_persists_fit_review_attempt_by_default(tmp_path: Path) -> None:
    palette, env = _fake_palette_repo(tmp_path)
    recording = _recording(tmp_path, with_zarr=True)
    result = _run(
        recording=recording,
        palette=palette,
        output_root=tmp_path / "output",
        probe_id="persist-default",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "persistence_mode=persist_fit_review_attempt" in result.stdout
    expected_zarr = recording / "zarr/recording_analysis.zarr"
    assert f"analysis_zarr={expected_zarr}" in result.stdout
    job = (tmp_path / "output/persist-default/run_probe.sh").read_text(encoding="utf-8")
    assert "fisheye.utils.publish_arena_geometry_fit_review" in job
    assert "--apply" in job


def test_clipped_probe_diagnostic_only_does_not_require_zarr(tmp_path: Path) -> None:
    palette, env = _fake_palette_repo(tmp_path)
    recording = _recording(tmp_path, with_zarr=False)
    result = _run(
        recording=recording,
        palette=palette,
        output_root=tmp_path / "output",
        probe_id="diagnostic-only",
        env=env,
        diagnostic_only=True,
    )

    assert result.returncode == 0, result.stderr
    assert "persistence_mode=diagnostic_only" in result.stdout
    job = (tmp_path / "output/diagnostic-only/run_probe.sh").read_text(encoding="utf-8")
    assert "PERSISTENCE_MODE=diagnostic_only" in job


def test_clipped_probe_default_fails_without_one_analysis_zarr(tmp_path: Path) -> None:
    palette, env = _fake_palette_repo(tmp_path)
    recording = _recording(tmp_path, with_zarr=False)
    result = _run(
        recording=recording,
        palette=palette,
        output_root=tmp_path / "output",
        probe_id="missing-zarr",
        env=env,
    )

    assert result.returncode == 2
    assert "requires exactly one *_analysis.zarr" in result.stderr
