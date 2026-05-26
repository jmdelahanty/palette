"""Submit sharded review-proxy video generation to LSF.

Shard jobs transcode disjoint clips and deliberately do not write the final
manifest. A finalizer waits for the shard array, verifies every expected proxy
exists, and writes the single authoritative manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.utils.build_review_proxy_videos import (
    DEFAULT_CRF,
    DEFAULT_PROXY_HEIGHT,
    DEFAULT_PROXY_WIDTH,
    ReviewProxyOptions,
    build_review_proxy_manifest,
)

SUBMISSION_SCHEMA = "palette.review_proxy_video_sharded_bsub_submission.v1"


@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    clip_ids: tuple[str, ...]


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_component(value: object, *, fallback: str = "run") -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return safe or fallback


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _quote_command(parts: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _parse_bsub_job_id(output: str) -> str:
    match = re.search(r"Job <(\d+)>", output)
    if not match:
        raise RuntimeError(f"Could not parse LSF job id from bsub output:\n{output}")
    return match.group(1)


def _partition_clip_ids(
    clip_ids: Sequence[str],
    *,
    shard_count: int | None,
    clips_per_shard: int | None,
) -> list[ShardSpec]:
    if not clip_ids:
        raise ValueError("No clips available to shard")
    if shard_count is not None and clips_per_shard is not None:
        raise ValueError("--shard-count and --clips-per-shard are mutually exclusive")
    if clips_per_shard is not None:
        if clips_per_shard <= 0:
            raise ValueError("--clips-per-shard must be positive")
        chunks = [tuple(clip_ids[i : i + clips_per_shard]) for i in range(0, len(clip_ids), clips_per_shard)]
    else:
        count = int(shard_count or 4)
        if count <= 0:
            raise ValueError("--shard-count must be positive")
        count = min(count, len(clip_ids))
        chunks = []
        for shard_index in range(count):
            start = (len(clip_ids) * shard_index) // count
            end = (len(clip_ids) * (shard_index + 1)) // count
            chunks.append(tuple(clip_ids[start:end]))
    return [ShardSpec(shard_index=i, clip_ids=chunk) for i, chunk in enumerate(chunks) if chunk]


def _builder_args(
    *,
    recording_dir: Path,
    output_dir: Path,
    proxy_run_id: str,
    proxy_width: int,
    proxy_height: int,
    encoder: str,
    preset: str,
    crf: int,
    hwaccel: str | None,
    scale_flags: str,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    no_probe: bool,
    overwrite: bool,
    skip_existing_valid: bool,
    clip_ids: Sequence[str] = (),
    camera_serials: Sequence[str] = (),
    apply: bool = False,
    defer_manifest: bool = False,
    write_manifest_only: bool = False,
    require_existing_proxies: bool = False,
    json_output: bool = True,
) -> list[str]:
    args: list[str] = [
        "scripts/py",
        "-m",
        "fisheye.utils.build_review_proxy_videos",
        str(recording_dir),
        "--output-dir",
        str(output_dir),
        "--proxy-run-id",
        str(proxy_run_id),
        "--proxy-width",
        str(int(proxy_width)),
        "--proxy-height",
        str(int(proxy_height)),
        "--encoder",
        str(encoder),
        "--preset",
        str(preset),
        "--crf",
        str(int(crf)),
        "--scale-flags",
        str(scale_flags),
        "--ffmpeg-bin",
        str(ffmpeg_bin),
        "--ffprobe-bin",
        str(ffprobe_bin),
    ]
    if hwaccel:
        args.extend(["--hwaccel", str(hwaccel)])
    if no_probe:
        args.append("--no-probe")
    if overwrite:
        args.append("--overwrite")
    if skip_existing_valid:
        args.append("--skip-existing-valid")
    for clip_id in clip_ids:
        args.extend(["--clip-id", str(clip_id)])
    for camera_serial in camera_serials:
        args.extend(["--camera-serial", str(camera_serial)])
    if apply:
        args.append("--apply")
    if defer_manifest:
        args.append("--defer-manifest")
    if write_manifest_only:
        args.append("--write-manifest-only")
    if require_existing_proxies:
        args.append("--require-existing-proxies")
    if json_output:
        args.append("--json")
    return args


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _bsub_args(
    *,
    job_name: str,
    ncores: int,
    mem_gb: int,
    walltime: str,
    queue: str,
    gpus: int,
    stdout: Path,
    stderr: Path,
    dependency: str | None = None,
) -> list[str]:
    args = [
        "-J",
        job_name,
        "-n",
        str(int(ncores)),
        "-W",
        str(walltime),
        "-R",
        f"rusage[mem={int(mem_gb)}G]",
        "-oo",
        str(stdout),
        "-eo",
        str(stderr),
    ]
    if queue:
        args.extend(["-q", queue])
    if gpus:
        args.extend(["-gpu", f"num={int(gpus)}"])
    if dependency:
        args.extend(["-w", dependency])
    return args


def _submit_bsub(args: Sequence[str], script: Path, *, cwd: Path) -> tuple[str, str]:
    result = subprocess.run(
        ["bsub", *map(str, args), "bash", str(script)],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"bsub failed with exit {result.returncode}:\n{output}")
    return _parse_bsub_job_id(output), output


def submit_review_proxy_videos_sharded(
    *,
    recording_dir: Path,
    output_dir: Path,
    proxy_run_id: str,
    run_dir: Path,
    repo_path: Path,
    proxy_width: int,
    proxy_height: int,
    encoder: str,
    preset: str,
    crf: int,
    hwaccel: str | None,
    scale_flags: str,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    no_probe: bool,
    overwrite: bool,
    skip_existing_valid: bool,
    camera_serials: Sequence[str],
    clip_ids: Sequence[str],
    shard_count: int | None,
    clips_per_shard: int | None,
    max_active: int,
    queue: str,
    ncores: int,
    mem_gb: int,
    gpus: int,
    walltime: str,
    finalizer_queue: str,
    finalizer_ncores: int,
    finalizer_mem_gb: int,
    finalizer_walltime: str,
    submit: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    repo_path = repo_path.expanduser().resolve()
    recording_dir = recording_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "scripts" / "shards").mkdir(parents=True, exist_ok=True)
    (run_dir / "summaries").mkdir(parents=True, exist_ok=True)

    planning_options = ReviewProxyOptions(
        output_dir=output_dir,
        proxy_run_id=proxy_run_id,
        proxy_width=proxy_width,
        proxy_height=proxy_height,
        encoder=encoder,
        preset=preset,
        crf=crf,
        hwaccel=hwaccel,
        scale_flags=scale_flags,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        probe=False,
    )
    plan_manifest = build_review_proxy_manifest(
        recording_dir,
        options=planning_options,
        clip_ids=clip_ids,
        camera_serials=camera_serials,
    )
    planned_clip_ids: list[str] = []
    for clip in plan_manifest["clips"]:
        clip_id = str(clip["clip_id"])
        if clip_id not in planned_clip_ids:
            planned_clip_ids.append(clip_id)
    shards = _partition_clip_ids(
        planned_clip_ids,
        shard_count=shard_count,
        clips_per_shard=clips_per_shard,
    )
    if max_active <= 0:
        raise ValueError("--max-active must be positive")
    max_active = min(max_active, len(shards))

    shard_script_dir = run_dir / "scripts" / "shards"
    shard_payloads: list[dict[str, Any]] = []
    for shard in shards:
        shard_name = f"shard_{shard.shard_index:06d}"
        summary_json = run_dir / "summaries" / f"{shard_name}.summary.json"
        builder_args = _builder_args(
            recording_dir=recording_dir,
            output_dir=output_dir,
            proxy_run_id=proxy_run_id,
            proxy_width=proxy_width,
            proxy_height=proxy_height,
            encoder=encoder,
            preset=preset,
            crf=crf,
            hwaccel=hwaccel,
            scale_flags=scale_flags,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            no_probe=no_probe,
            overwrite=overwrite,
            skip_existing_valid=skip_existing_valid,
            clip_ids=shard.clip_ids,
            camera_serials=camera_serials,
            apply=True,
            defer_manifest=True,
        )
        script = shard_script_dir / f"{shard_name}.sh"
        _write_text(
            script,
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f"cd {shlex.quote(str(repo_path))}",
                    f"mkdir -p {shlex.quote(str(summary_json.parent))}",
                    f"SUMMARY_JSON={shlex.quote(str(summary_json))}",
                    'echo "host=$(hostname)"',
                    f'echo "proxy_shard={shard_name}"',
                    'echo "summary_json=${SUMMARY_JSON}"',
                    f"{_quote_command(builder_args)} > \"${{SUMMARY_JSON}}\"",
                    'echo "summary_json=${SUMMARY_JSON}"',
                    "",
                ]
            ),
        )
        script.chmod(0o755)
        shard_payloads.append(
            {
                "shard_index": int(shard.shard_index),
                "clip_ids": list(shard.clip_ids),
                "script": str(script),
                "summary_template": str(summary_json),
                "command": _quote_command(builder_args),
            }
        )

    array_runner = run_dir / "scripts" / "run_review_proxy_shard_array.sh"
    _write_text(
        array_runner,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'INDEX="${LSB_JOBINDEX:-1}"',
                'if [[ "$INDEX" -lt 1 ]]; then echo "Invalid LSB_JOBINDEX=$INDEX" >&2; exit 2; fi',
                'SHARD_ZERO=$((INDEX - 1))',
                f"SHARD_DIR={shlex.quote(str(shard_script_dir))}",
                'SHARD_SCRIPT=$(printf "%s/shard_%06d.sh" "$SHARD_DIR" "$SHARD_ZERO")',
                'if [[ ! -x "$SHARD_SCRIPT" ]]; then echo "Missing shard script: $SHARD_SCRIPT" >&2; exit 2; fi',
                'bash "$SHARD_SCRIPT"',
                "",
            ]
        ),
    )
    array_runner.chmod(0o755)

    finalizer_summary = run_dir / "summaries" / "finalize.summary.json"
    finalizer_args = _builder_args(
        recording_dir=recording_dir,
        output_dir=output_dir,
        proxy_run_id=proxy_run_id,
        proxy_width=proxy_width,
        proxy_height=proxy_height,
        encoder=encoder,
        preset=preset,
        crf=crf,
        hwaccel=hwaccel,
        scale_flags=scale_flags,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        no_probe=no_probe,
        overwrite=overwrite,
        skip_existing_valid=False,
        clip_ids=clip_ids,
        camera_serials=camera_serials,
        write_manifest_only=True,
        require_existing_proxies=True,
    )
    finalizer_script = run_dir / "scripts" / "run_review_proxy_finalize.sh"
    _write_text(
        finalizer_script,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(repo_path))}",
                f"mkdir -p {shlex.quote(str(finalizer_summary.parent))}",
                f"SUMMARY_JSON={shlex.quote(str(finalizer_summary))}",
                'echo "host=$(hostname)"',
                'echo "stage=finalize_review_proxy_manifest"',
                'echo "summary_json=${SUMMARY_JSON}"',
                f"{_quote_command(finalizer_args)} > \"${{SUMMARY_JSON}}\"",
                'echo "summary_json=${SUMMARY_JSON}"',
                f"echo {shlex.quote('manifest=' + str(output_dir / 'manifest.json'))}",
                "",
            ]
        ),
    )
    finalizer_script.chmod(0o755)

    safe_run = _safe_component(proxy_run_id, fallback="proxy")
    shard_job_name = f"review_proxy_{safe_run}[1-{len(shards)}]%{max_active}"
    shard_bsub_args = _bsub_args(
        job_name=shard_job_name,
        ncores=ncores,
        mem_gb=mem_gb,
        walltime=walltime,
        queue=queue,
        gpus=gpus,
        stdout=run_dir / "logs" / "%J_%I.out",
        stderr=run_dir / "logs" / "%J_%I.err",
    )
    finalizer_bsub_args_template = _bsub_args(
        job_name=f"finalize_review_proxy_{safe_run}",
        ncores=finalizer_ncores,
        mem_gb=finalizer_mem_gb,
        walltime=finalizer_walltime,
        queue=finalizer_queue,
        gpus=0,
        stdout=run_dir / "logs" / "finalize_%J.out",
        stderr=run_dir / "logs" / "finalize_%J.err",
        dependency="done(<shard_jobid>)",
    )

    result: dict[str, Any] = {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "planned",
        "submit": bool(submit),
        "recording_dir": str(recording_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json"),
        "proxy_run_id": proxy_run_id,
        "run_dir": str(run_dir),
        "repo": str(repo_path),
        "clip_count": int(len(planned_clip_ids)),
        "shard_count": int(len(shards)),
        "max_active": int(max_active),
        "shards": shard_payloads,
        "array_runner": str(array_runner),
        "finalizer_script": str(finalizer_script),
        "shard_bsub_command": _quote_command(["bsub", *shard_bsub_args, "bash", array_runner]),
        "finalizer_bsub_command_template": _quote_command(
            ["bsub", *finalizer_bsub_args_template, "bash", finalizer_script]
        ),
        "finalizer_command": _quote_command(finalizer_args),
        "plan_manifest": plan_manifest,
    }

    if not submit:
        return result
    if not shutil_which("bsub"):
        raise RuntimeError("bsub not found in PATH. Is this an LSF cluster?")

    shard_job_id, shard_submit_output = _submit_bsub(shard_bsub_args, array_runner, cwd=repo_path)
    finalizer_bsub_args = _bsub_args(
        job_name=f"finalize_review_proxy_{safe_run}",
        ncores=finalizer_ncores,
        mem_gb=finalizer_mem_gb,
        walltime=finalizer_walltime,
        queue=finalizer_queue,
        gpus=0,
        stdout=run_dir / "logs" / "finalize_%J.out",
        stderr=run_dir / "logs" / "finalize_%J.err",
        dependency=f"done({shard_job_id})",
    )
    finalizer_job_id, finalizer_submit_output = _submit_bsub(finalizer_bsub_args, finalizer_script, cwd=repo_path)
    result.update(
        {
            "status": "submitted",
            "shard_job_id": shard_job_id,
            "shard_submit_output": shard_submit_output,
            "finalizer_job_id": finalizer_job_id,
            "finalizer_submit_output": finalizer_submit_output,
            "finalizer_bsub_command": _quote_command(["bsub", *finalizer_bsub_args, "bash", finalizer_script]),
        }
    )
    return result


def shutil_which(command: str) -> str | None:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for directory in paths:
        candidate = Path(directory) / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit sharded review-proxy video generation to LSF.")
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--proxy-run-id", default=None)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy-width", type=int, default=DEFAULT_PROXY_WIDTH)
    parser.add_argument("--proxy-height", type=int, default=DEFAULT_PROXY_HEIGHT)
    parser.add_argument("--encoder", default="h264_nvenc")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    parser.add_argument("--hwaccel", default="cuda")
    parser.add_argument("--scale-flags", default="bilinear")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing-valid", action="store_true", default=True)
    parser.add_argument("--no-skip-existing-valid", action="store_false", dest="skip_existing_valid")
    parser.add_argument("--clip-id", action="append", default=[])
    parser.add_argument("--camera-serial", action="append", default=[])
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--clips-per-shard", type=int, default=None)
    parser.add_argument("--max-active", type=int, default=4)
    parser.add_argument("--queue", default="gpu_l4")
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--walltime", default="2:00")
    parser.add_argument("--finalizer-queue", default="short")
    parser.add_argument("--finalizer-ncores", type=int, default=2)
    parser.add_argument("--finalizer-mem-gb", type=int, default=8)
    parser.add_argument("--finalizer-walltime", default="1:00")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"status: {result.get('status')}")
    print(f"submit: {result.get('submit')}")
    print(f"recording_dir: {result.get('recording_dir')}")
    print(f"output_dir: {result.get('output_dir')}")
    print(f"manifest_path: {result.get('manifest_path')}")
    print(f"run_dir: {result.get('run_dir')}")
    print(f"proxy_run_id: {result.get('proxy_run_id')}")
    print(f"clip_count: {result.get('clip_count')}")
    print(f"shard_count: {result.get('shard_count')} max_active={result.get('max_active')}")
    if result.get("shard_job_id"):
        print(f"shard_job_id: {result.get('shard_job_id')}")
    if result.get("finalizer_job_id"):
        print(f"finalizer_job_id: {result.get('finalizer_job_id')}")
    print(f"shard_bsub_command: {result.get('shard_bsub_command')}")
    print(f"finalizer_bsub_command: {result.get('finalizer_bsub_command') or result.get('finalizer_bsub_command_template')}")
    if not result.get("submit"):
        print("dry_run: no jobs submitted; pass --submit to call bsub")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    recording_dir = args.recording_dir.expanduser().resolve()
    proxy_run_id = args.proxy_run_id or ("review_proxy_" + _utc_run_id())
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else recording_dir / "derived" / "review_proxy" / "video_detect" / _safe_component(proxy_run_id, fallback="proxy")
    )
    run_id = _utc_run_id()
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir
        else output_dir / f"bsub_sharded_{run_id}"
    )
    result = submit_review_proxy_videos_sharded(
        recording_dir=recording_dir,
        output_dir=output_dir,
        proxy_run_id=proxy_run_id,
        run_dir=run_dir,
        repo_path=args.repo,
        proxy_width=int(args.proxy_width),
        proxy_height=int(args.proxy_height),
        encoder=str(args.encoder),
        preset=str(args.preset),
        crf=int(args.crf),
        hwaccel=str(args.hwaccel) if args.hwaccel else None,
        scale_flags=str(args.scale_flags),
        ffmpeg_bin=str(args.ffmpeg_bin),
        ffprobe_bin=str(args.ffprobe_bin),
        no_probe=bool(args.no_probe),
        overwrite=bool(args.overwrite),
        skip_existing_valid=bool(args.skip_existing_valid),
        camera_serials=tuple(args.camera_serial or ()),
        clip_ids=tuple(args.clip_id or ()),
        shard_count=args.shard_count,
        clips_per_shard=args.clips_per_shard,
        max_active=int(args.max_active),
        queue=str(args.queue),
        ncores=int(args.ncores),
        mem_gb=int(args.mem_gb),
        gpus=int(args.gpus),
        walltime=str(args.walltime),
        finalizer_queue=str(args.finalizer_queue),
        finalizer_ncores=int(args.finalizer_ncores),
        finalizer_mem_gb=int(args.finalizer_mem_gb),
        finalizer_walltime=str(args.finalizer_walltime),
        submit=bool(args.submit),
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    else:
        _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
