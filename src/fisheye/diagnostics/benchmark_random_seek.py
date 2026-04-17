#!/usr/bin/env python3
"""Benchmark random-seek latency for two videos (e.g. fixed vs not fixed).

This is intended to compare interactive/random frame access behavior, not
sequential decode throughput.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

import cv2

try:
    from fisheye.diagnostics.video.container import check_hevc_keyframe_flags
except ModuleNotFoundError:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _SRC_DIR = _THIS_DIR.parent.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from fisheye.diagnostics.video.container import check_hevc_keyframe_flags


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _build_random_positions(
    *,
    total_frames: int,
    samples: int,
    max_frame_fraction: float,
    seed: int,
) -> List[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    if samples <= 0:
        raise ValueError("samples must be > 0")
    if not (0 < max_frame_fraction <= 1.0):
        raise ValueError("max_frame_fraction must be in (0, 1]")

    max_frame_idx = max(0, min(total_frames - 1, int((total_frames - 1) * max_frame_fraction)))
    rng = random.Random(seed)
    return [rng.randint(0, max_frame_idx) for _ in range(samples)]


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for: {path}")
    return total_frames


def _video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Could not determine FPS for: {path}")
    return fps


def _seek_once_opencv(cap: cv2.VideoCapture, target_frame: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_frame))
    actual_before = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ok, frame = cap.read()
    t1 = time.perf_counter()
    actual_after = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    decoded_frame: Optional[int]
    if ok and frame is not None:
        decoded_frame = actual_after - 1 if actual_after > 0 else actual_before
    else:
        decoded_frame = None

    return {
        "requested_frame": int(target_frame),
        "ok": bool(ok and frame is not None),
        "latency_ms": float((t1 - t0) * 1000.0),
        "actual_before": int(actual_before),
        "actual_after": int(actual_after),
        "decoded_frame": decoded_frame,
    }


def _run_random_seek_opencv(
    *,
    video_path: Path,
    warmup_positions: List[int],
    timed_positions: List[int],
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    for frame_idx in warmup_positions:
        _seek_once_opencv(cap, frame_idx)

    samples: List[Dict[str, Any]] = []
    for frame_idx in timed_positions:
        sample = _seek_once_opencv(cap, frame_idx)
        decoded = sample.get("decoded_frame")
        if isinstance(decoded, int):
            sample["seek_error_frames"] = int(abs(decoded - frame_idx))
        else:
            sample["seek_error_frames"] = None
        samples.append(sample)

    cap.release()

    latencies = [float(s["latency_ms"]) for s in samples if bool(s.get("ok"))]
    errors = [
        int(s["seek_error_frames"])
        for s in samples
        if bool(s.get("ok")) and isinstance(s.get("seek_error_frames"), int)
    ]
    failures = len([s for s in samples if not bool(s.get("ok"))])
    inexact = len([e for e in errors if e > 0])

    return {
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "success_count": len(samples) - failures,
            "failure_count": failures,
            "success_rate": float((len(samples) - failures) / len(samples)) if samples else 0.0,
            "latency_ms": _stats(latencies),
            "seek_error_frames": _stats([float(e) for e in errors]),
            "inexact_seek_count": int(inexact),
        },
    }


def _seek_once_ffmpeg(
    *,
    video_path: Path,
    target_frame: int,
    fps: float,
    ffmpeg_bin: str,
) -> Dict[str, Any]:
    ts_seconds = float(target_frame / fps)
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{ts_seconds:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-an",
        "-f",
        "null",
        "-",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    t1 = time.perf_counter()
    return {
        "requested_frame": int(target_frame),
        "ok": bool(result.returncode == 0),
        "latency_ms": float((t1 - t0) * 1000.0),
        "actual_before": None,
        "actual_after": None,
        "decoded_frame": None,
        "returncode": int(result.returncode),
        "stderr": result.stderr.strip() if result.stderr else None,
    }


def _run_random_seek_ffmpeg(
    *,
    video_path: Path,
    warmup_positions: List[int],
    timed_positions: List[int],
    fps: float,
    ffmpeg_bin: str,
) -> Dict[str, Any]:
    for frame_idx in warmup_positions:
        _seek_once_ffmpeg(
            video_path=video_path,
            target_frame=frame_idx,
            fps=fps,
            ffmpeg_bin=ffmpeg_bin,
        )

    samples: List[Dict[str, Any]] = []
    for frame_idx in timed_positions:
        sample = _seek_once_ffmpeg(
            video_path=video_path,
            target_frame=frame_idx,
            fps=fps,
            ffmpeg_bin=ffmpeg_bin,
        )
        sample["seek_error_frames"] = None
        samples.append(sample)

    latencies = [float(s["latency_ms"]) for s in samples if bool(s.get("ok"))]
    failures = len([s for s in samples if not bool(s.get("ok"))])

    return {
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "success_count": len(samples) - failures,
            "failure_count": failures,
            "success_rate": float((len(samples) - failures) / len(samples)) if samples else 0.0,
            "latency_ms": _stats(latencies),
            "seek_error_frames": _stats([]),
            "inexact_seek_count": 0,
        },
    }


def _pick_slowest_samples(samples: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    ranked = sorted(samples, key=lambda row: float(row.get("latency_ms", 0.0)), reverse=True)
    return ranked[:limit]


def _format_stat(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _print_video_report(label: str, path: Path, check: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lat = summary["latency_ms"]
    err = summary["seek_error_frames"]
    print(f"\n[{label}] {path}")
    print(
        "  keyframe_flags: "
        f"codec={check.get('codec')} has_stss={check.get('has_stss')} needs_fix={check.get('needs_fix')}"
    )
    print(f"  keyframe_msg: {check.get('message')}")
    print(
        "  seek: "
        f"samples={summary['sample_count']} success={summary['success_count']} "
        f"failures={summary['failure_count']} success_rate={summary['success_rate']:.3f}"
    )
    print(
        "  latency_ms: "
        f"median={_format_stat(lat['median'])} p95={_format_stat(lat['p95'])} "
        f"p99={_format_stat(lat['p99'])} min={_format_stat(lat['min'])} "
        f"max={_format_stat(lat['max'])}"
    )
    print(
        "  seek_error_frames: "
        f"median={_format_stat(err['median'])} p95={_format_stat(err['p95'])} "
        f"inexact={summary['inexact_seek_count']}"
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark random seek latency between two videos (fixed vs not fixed).",
    )
    parser.add_argument("video_a", type=Path, help="First video path (e.g., not fixed).")
    parser.add_argument("video_b", type=Path, help="Second video path (e.g., fixed).")
    parser.add_argument("--label-a", default="A", help="Label for first video in report.")
    parser.add_argument("--label-b", default="B", help="Label for second video in report.")
    parser.add_argument(
        "--backend",
        choices=("opencv", "ffmpeg"),
        default="opencv",
        help="Seek backend (default: opencv).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg binary path for --backend ffmpeg (default: ffmpeg).",
    )
    parser.add_argument("--samples", type=int, default=300, help="Timed random seeks per video.")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup seek count per video.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible positions.")
    parser.add_argument(
        "--max-frame-fraction",
        type=float,
        default=0.98,
        help="Restrict sampled frame positions to this fraction of total frames (default: 0.98).",
    )
    parser.add_argument(
        "--slowest",
        type=int,
        default=5,
        help="Print N slowest samples per video (default: 5).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path with full benchmark payload.",
    )
    parser.add_argument(
        "--no-keyframe-check",
        action="store_true",
        help="Skip container stss check output.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    video_a = args.video_a.expanduser().resolve()
    video_b = args.video_b.expanduser().resolve()
    if not video_a.exists():
        raise FileNotFoundError(video_a)
    if not video_b.exists():
        raise FileNotFoundError(video_b)
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if not (0 < float(args.max_frame_fraction) <= 1.0):
        raise ValueError("--max-frame-fraction must be in (0, 1]")

    frames_a = _video_frame_count(video_a)
    frames_b = _video_frame_count(video_b)
    shared_frames = min(frames_a, frames_b)
    if shared_frames <= 0:
        raise RuntimeError("No frames available to benchmark")

    warmup_positions = _build_random_positions(
        total_frames=shared_frames,
        samples=max(1, int(args.warmup)) if args.warmup else 1,
        max_frame_fraction=float(args.max_frame_fraction),
        seed=int(args.seed) + 1,
    )
    if args.warmup == 0:
        warmup_positions = []
    timed_positions = _build_random_positions(
        total_frames=shared_frames,
        samples=int(args.samples),
        max_frame_fraction=float(args.max_frame_fraction),
        seed=int(args.seed),
    )

    print("Random-seek benchmark")
    print(f"  created_at_utc: {_utc_now_iso()}")
    print(f"  video_a: {video_a} (frames={frames_a})")
    print(f"  video_b: {video_b} (frames={frames_b})")
    print(f"  shared_frames: {shared_frames}")
    print(
        f"  backend={args.backend} samples={args.samples} warmup={args.warmup} "
        f"seed={args.seed} max_frame_fraction={args.max_frame_fraction}"
    )

    check_a = {} if args.no_keyframe_check else check_hevc_keyframe_flags(video_a)
    check_b = {} if args.no_keyframe_check else check_hevc_keyframe_flags(video_b)

    if args.backend == "opencv":
        result_a = _run_random_seek_opencv(
            video_path=video_a,
            warmup_positions=warmup_positions,
            timed_positions=timed_positions,
        )
        result_b = _run_random_seek_opencv(
            video_path=video_b,
            warmup_positions=warmup_positions,
            timed_positions=timed_positions,
        )
    else:
        fps_a = _video_fps(video_a)
        fps_b = _video_fps(video_b)
        if abs(fps_a - fps_b) > 1e-6:
            print(f"  note: fps differs (a={fps_a:.6f}, b={fps_b:.6f})")
        fps_shared = min(fps_a, fps_b)
        result_a = _run_random_seek_ffmpeg(
            video_path=video_a,
            warmup_positions=warmup_positions,
            timed_positions=timed_positions,
            fps=fps_shared,
            ffmpeg_bin=str(args.ffmpeg_bin),
        )
        result_b = _run_random_seek_ffmpeg(
            video_path=video_b,
            warmup_positions=warmup_positions,
            timed_positions=timed_positions,
            fps=fps_shared,
            ffmpeg_bin=str(args.ffmpeg_bin),
        )

    summary_a = result_a["summary"]
    summary_b = result_b["summary"]
    _print_video_report(args.label_a, video_a, check_a, summary_a)
    _print_video_report(args.label_b, video_b, check_b, summary_b)

    med_a = summary_a["latency_ms"]["median"]
    med_b = summary_b["latency_ms"]["median"]
    p95_a = summary_a["latency_ms"]["p95"]
    p95_b = summary_b["latency_ms"]["p95"]

    print("\nComparison")
    if isinstance(med_a, float) and isinstance(med_b, float) and med_b > 0:
        print(f"  median speedup ({args.label_b} vs {args.label_a}): {med_a / med_b:.3f}x")
    else:
        print("  median speedup: n/a")
    if isinstance(p95_a, float) and isinstance(p95_b, float) and p95_b > 0:
        print(f"  p95 speedup ({args.label_b} vs {args.label_a}): {p95_a / p95_b:.3f}x")
    else:
        print("  p95 speedup: n/a")

    slowest_a = _pick_slowest_samples(result_a["samples"], int(args.slowest))
    slowest_b = _pick_slowest_samples(result_b["samples"], int(args.slowest))
    if slowest_a:
        print(f"\nSlowest {len(slowest_a)} seeks ({args.label_a})")
        for row in slowest_a:
            print(
                f"  req={row['requested_frame']} latency_ms={row['latency_ms']:.3f} "
                f"ok={row['ok']} decoded={row['decoded_frame']}"
            )
    if slowest_b:
        print(f"\nSlowest {len(slowest_b)} seeks ({args.label_b})")
        for row in slowest_b:
            print(
                f"  req={row['requested_frame']} latency_ms={row['latency_ms']:.3f} "
                f"ok={row['ok']} decoded={row['decoded_frame']}"
            )

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "config": {
            "video_a": str(video_a),
            "video_b": str(video_b),
            "label_a": str(args.label_a),
            "label_b": str(args.label_b),
            "backend": str(args.backend),
            "ffmpeg_bin": str(args.ffmpeg_bin),
            "samples": int(args.samples),
            "warmup": int(args.warmup),
            "seed": int(args.seed),
            "max_frame_fraction": float(args.max_frame_fraction),
            "shared_frames": int(shared_frames),
            "frames_a": int(frames_a),
            "frames_b": int(frames_b),
        },
        "keyframe_check": {
            "video_a": check_a,
            "video_b": check_b,
        },
        "summary": {
            "video_a": summary_a,
            "video_b": summary_b,
            "comparison": {
                "median_speedup_b_vs_a": (med_a / med_b) if isinstance(med_a, float) and isinstance(med_b, float) and med_b > 0 else None,
                "p95_speedup_b_vs_a": (p95_a / p95_b) if isinstance(p95_a, float) and isinstance(p95_b, float) and p95_b > 0 else None,
            },
        },
        "slowest_samples": {
            "video_a": slowest_a,
            "video_b": slowest_b,
        },
        "timed_positions": timed_positions,
        "results": {
            "video_a": result_a["samples"],
            "video_b": result_b["samples"],
        },
    }

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote benchmark JSON: {output_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
