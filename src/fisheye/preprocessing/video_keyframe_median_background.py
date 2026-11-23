"""Background run computed from a recording's own video keyframes.

``compute_background`` samples frames from pixel arrays inside the archive.
Analysis archives of current recordings hold no video pixels, so this stage
decodes the recording's full-frame video instead: it takes N keyframes spread
evenly over a frame window, reduces them to a per-pixel median, and stores

- ``background_full``: the median at the camera's native raster;
- ``background_ds``: that image resized to the canonical 640 x 640 raster.

Only encoder-declared keyframes are decoded (one independent picture each),
so the cost is one pass over the file without decoding the frames between.

A subject that stays in one place for more than half of the sampled frames is
absorbed into the background. The window is the caller's choice and is
recorded; for a protocol with a no-stimulus period, sample that period.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import zarr

from ..shared.orange_frame_identity import keyframe_video_frame_indices
from ..shared.run_provenance import build_writer_run_provenance
from ..shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from ..shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)

METHOD = "video_keyframe_median"
RUN_PREFIX = "background_video_keyframe_median_"
# Canonical downsampled raster (configs/fisheye/default.yaml), as (height, width).
BACKGROUND_DS_SHAPE = (640, 640)
DEFAULT_KEYFRAME_COUNT = 61
MIN_KEYFRAME_COUNT = 3
# Luma planes that ffmpeg hands over as gray without a range conversion.
FULL_RANGE_PIXEL_FORMATS = ("yuvj420p", "gray")


def downsample_background(image: np.ndarray) -> np.ndarray:
    """The canonical 640 x 640 copy of a native-raster background (area average)."""

    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("background must be a 2-D uint8 image")
    height, width = BACKGROUND_DS_SHAPE
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)


def select_keyframes(
    keyframes: Sequence[int],
    *,
    count: int,
    first_frame: int | None = None,
    last_frame: int | None = None,
) -> tuple[int, ...]:
    """``count`` declared keyframes spread evenly over ``[first_frame, last_frame]``.

    Both ends are video frame indices and inclusive; ``None`` means the start
    or end of the video. Raises when the window holds fewer keyframes.
    """

    if count < MIN_KEYFRAME_COUNT:
        raise ValueError(f"at least {MIN_KEYFRAME_COUNT} keyframes are needed for a median")
    window = [
        int(frame)
        for frame in keyframes
        if (first_frame is None or frame >= first_frame) and (last_frame is None or frame <= last_frame)
    ]
    if len(window) < count:
        raise ValueError(f"the frame window holds {len(window)} keyframes, fewer than the {count} requested")
    last = len(window) - 1
    # Nearest keyframe to each of ``count`` equally spaced positions (halves round up).
    steps = count - 1
    return tuple(window[(2 * i * last + steps) // (2 * steps)] for i in range(count))


def probe_video(video_path: Path) -> dict[str, Any]:
    """Width, height, frame count and pixel format of the video's first stream."""

    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,pix_fmt,color_range,nb_frames",
            "-of", "json", str(video_path),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return {
        "codec_name": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt"),
        "color_range": stream.get("color_range"),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
    }


def decode_keyframes(
    video_path: Path,
    *,
    declared_keyframes: Sequence[int],
    wanted: Sequence[int],
    shape_hw: tuple[int, int],
    threads: int = 4,
) -> tuple[np.ndarray, str]:
    """Decode the ``wanted`` keyframes as luma; returns the stack and its SHA-256.

    ffmpeg emits every keyframe in order and nothing else; the n-th picture is
    the n-th declared keyframe. The pass stops after the last wanted one, and
    fails if the video yields fewer keyframes than were declared up to there.
    """

    position = {int(frame): index for index, frame in enumerate(declared_keyframes)}
    try:
        rows = {position[int(frame)]: row for row, frame in enumerate(wanted)}
    except KeyError as exc:
        raise ValueError(f"frame {exc.args[0]} is not a declared keyframe") from exc
    if len(rows) != len(wanted):
        raise ValueError("wanted keyframes must be unique")
    height, width = shape_hw
    frame_bytes = height * width
    stack = np.empty((len(wanted), height, width), dtype=np.uint8)
    process = subprocess.Popen(
        [
            # -nostdin and a closed stdin: ffmpeg otherwise reads its caller's standard
            # input for interactive commands and consumes whatever a loop or pipeline feeds.
            "ffmpeg", "-nostdin", "-v", "error", "-threads", str(int(threads)), "-skip_frame", "nokey",
            "-i", str(video_path), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        bufsize=frame_bytes,
    )
    try:
        last = max(rows)
        for index in range(last + 1):
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise RuntimeError(
                    f"{video_path.name}: decoding stopped at keyframe {index} of {last + 1} needed"
                )
            if index in rows:
                stack[rows[index]] = np.frombuffer(raw, np.uint8).reshape(height, width)
    finally:
        process.kill()
        process.wait()
    return stack, f"sha256:{hashlib.sha256(stack.tobytes(order='C')).hexdigest()}"


def write_video_keyframe_median_background(
    root: zarr.Group,
    *,
    background_full: np.ndarray,
    source: dict[str, Any],
    parameters: dict[str, Any],
    run_name: str | None = None,
) -> str:
    """Write one complete background run holding the median and its 640 x 640 copy."""

    image = np.ascontiguousarray(background_full)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("background_full must be a 2-D uint8 image")
    run_name = run_name or f"{RUN_PREFIX}{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}"
    parent = require_runs_parent(root, "background_runs")
    if run_name in parent:
        raise ValueError(f"background_runs/{run_name} already exists")
    group = parent.create_group(run_name)
    mark_run_started(group, run_name=run_name, stage="background")
    note_pending_latest(parent, run_name)
    for name, data in (("background_full", image), ("background_ds", downsample_background(image))):
        array = group.create_array(
            name,
            shape=data.shape,
            chunks=data.shape,
            dtype="uint8",
            fill_value=0,
            serializer=zarr.codecs.BytesCodec(),
            compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")],
        )
        array[:] = data
        if not np.array_equal(array[:], data):
            raise ValueError(f"background_runs/{run_name}/{name} does not read back equal")
    frames = [int(frame) for frame in source["source_frame_indices"]]
    group.attrs.update(
        {
            "background_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": METHOD,
            "background_source": "recording_video_keyframes",
            "parameters": parameters,
            "n_frames_used": len(frames),
            "total_frames": source.get("total_frames"),
            "source_frame_indices": frames,
            "video_keyframe_median": source,
        }
    )
    mark_run_complete(
        group,
        parent_group=parent,
        run_name=run_name,
        run_provenance=build_writer_run_provenance(
            command="compute_video_keyframe_median_background",
            params=parameters,
            input_run_ids={
                "source_video_relative_path": source.get("video_relative_path"),
                "decoded_keyframes_sha256": source.get("decoded_keyframes_sha256"),
            },
        ),
    )
    return run_name


def analysis_zarr_path(recording_dir: Path) -> Path:
    path = recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"
    if not (path / "zarr.json").is_file():
        raise ValueError(f"no analysis archive at {path}")
    return path


def plan_recording(
    recording_dir: Path,
    *,
    count: int = DEFAULT_KEYFRAME_COUNT,
    first_frame: int | None = None,
    last_frame: int | None = None,
    zarr_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve the video, its declared keyframes and the frames to sample. Reads only."""

    recording_dir = Path(recording_dir)
    zarr_path = Path(zarr_path) if zarr_path is not None else analysis_zarr_path(recording_dir)
    manifest = json.loads((recording_dir / "recording_manifest.json").read_text(encoding="utf-8"))
    camera = str(manifest.get("camera_id"))
    folder = recording_dir / "cams" / "acquisition" / "external_recorder"
    video = folder / f"Cam{camera}_external.mp4"
    keyframe_path = folder / f"Cam{camera}_external_keyframes.json"
    for path in (video, keyframe_path):
        if not path.is_file():
            raise ValueError(f"missing {path}")
    payload = json.loads(keyframe_path.read_text(encoding="utf-8"))
    declared = [int(frame) for frame in keyframe_video_frame_indices(payload, label=str(keyframe_path))]
    if declared != sorted(set(declared)):
        raise ValueError("declared keyframes must be strictly increasing")
    probe = probe_video(video)
    if probe["pix_fmt"] not in FULL_RANGE_PIXEL_FORMATS:
        raise ValueError(
            f"{video.name}: pixel format {probe['pix_fmt']!r} is not a full-range luma source; "
            "refusing to decode it as a background"
        )
    total = int(payload.get("total_frames") or 0)
    if probe["nb_frames"] is not None and total and probe["nb_frames"] != total:
        raise ValueError("keyframe summary frame count disagrees with the video")
    selected = select_keyframes(declared, count=count, first_frame=first_frame, last_frame=last_frame)
    return {
        "recording": recording_dir.name,
        "recording_dir": str(recording_dir),
        "zarr_path": str(zarr_path),
        "camera_serial": camera,
        "video_path": str(video),
        "video_relative_path": video.relative_to(recording_dir).as_posix(),
        "video_size_bytes": video.stat().st_size,
        "video_probe": probe,
        "total_frames": total or probe["nb_frames"],
        "declared_keyframe_count": len(declared),
        "declared_keyframes": declared,
        "window_first_frame": first_frame,
        "window_last_frame": last_frame,
        "source_frame_indices": list(selected),
    }


def compute_recording(
    recording_dir: Path,
    *,
    count: int = DEFAULT_KEYFRAME_COUNT,
    first_frame: int | None = None,
    last_frame: int | None = None,
    window_note: str | None = None,
    threads: int = 4,
) -> dict[str, Any]:
    """Compute and publish the keyframe-median background run for one recording."""

    plan = plan_recording(recording_dir, count=count, first_frame=first_frame, last_frame=last_frame)
    median, source, parameters = _median_of_plan(plan, window_note=window_note, threads=threads)
    stack_sha256 = source["decoded_keyframes_sha256"]
    zarr_path = Path(plan["zarr_path"])
    with archive_metadata_publication_lock(zarr_path):
        root = zarr.open_group(str(zarr_path), mode="r+", use_consolidated=False)
        run_name = write_video_keyframe_median_background(
            root, background_full=median, source=source, parameters=parameters
        )
        consolidate_metadata_capture_expected_warnings(zarr_path)
        published = zarr.open_group(str(zarr_path), mode="r", use_consolidated=True)
        runs = published["background_runs"]
        if run_name not in runs or not {"background_full", "background_ds"} <= set(runs[run_name].array_keys()):
            raise ValueError(f"{plan['recording']}: consolidated metadata lacks {run_name}")
        latest = runs.attrs.get("latest")
    summary = {key: value for key, value in plan.items() if key != "declared_keyframes"}
    return {
        **summary,
        "status": "written",
        "run_name": run_name,
        "latest": latest,
        "selected_as_latest": latest == run_name,
        "decoded_keyframes_sha256": stack_sha256,
        "background_full_mean": float(median.mean()),
    }


def _median_of_plan(
    plan: dict[str, Any], *, window_note: str | None, threads: int
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Decode a plan's keyframes; return the median image, its source record and parameters."""

    probe = plan["video_probe"]
    count = len(plan["source_frame_indices"])
    first_frame, last_frame = plan["window_first_frame"], plan["window_last_frame"]
    stack, stack_sha256 = decode_keyframes(
        Path(plan["video_path"]),
        declared_keyframes=plan["declared_keyframes"],
        wanted=plan["source_frame_indices"],
        shape_hw=(probe["height"], probe["width"]),
        threads=threads,
    )
    median = np.median(stack, axis=0, overwrite_input=True).astype(np.uint8)
    del stack
    parameters = {
        "method": METHOD,
        "keyframe_count": int(count),
        "window_first_frame": first_frame,
        "window_last_frame": last_frame,
        "window_note": window_note,
        "background_ds_shape": list(BACKGROUND_DS_SHAPE),
        "compute_full": True,
        "compute_ds": True,
    }
    source = {
        "video_relative_path": plan["video_relative_path"],
        "video_size_bytes": plan["video_size_bytes"],
        "video_probe": probe,
        "total_frames": plan["total_frames"],
        "declared_keyframe_count": plan["declared_keyframe_count"],
        "source_frame_indices": plan["source_frame_indices"],
        "decoded_keyframes_sha256": stack_sha256,
        "decoder": "ffmpeg -skip_frame nokey, luma as gray",
        "camera_serial": plan["camera_serial"],
    }
    return median, source, parameters


STATUS_WRITTEN = "written"
STATUS_ALREADY_PRESENT = "already_present"
STATUS_SKIPPED = "skipped"
STATUS_REFUSED = "refused"
IMPORT_WINDOW_NOTE = "whole recording (import default)"


def _whole_recording_run(root: zarr.Group) -> str | None:
    """Name of a complete whole-recording run of this method already in the archive."""

    if "background_runs" not in root:
        return None
    for name, group in sorted(root["background_runs"].groups()):
        attrs = dict(group.attrs)
        parameters = attrs.get("parameters") or {}
        if (
            attrs.get("method") == METHOD
            and "background_full" in group
            and "background_ds" in group
            and parameters.get("window_first_frame") is None
            and parameters.get("window_last_frame") is None
        ):
            return name
    return None


def full_frame_video_inputs(recording_dir: Path) -> tuple[Path, Path] | None:
    """The recording's full-frame video and keyframe list, or None when it has none."""

    recording_dir = Path(recording_dir)
    manifest_path = recording_dir / "recording_manifest.json"
    if not manifest_path.is_file():
        return None
    camera = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("camera_id"))
    folder = recording_dir / "cams" / "acquisition" / "external_recorder"
    video = folder / f"Cam{camera}_external.mp4"
    keyframe_path = folder / f"Cam{camera}_external_keyframes.json"
    if not video.is_file() or not keyframe_path.is_file():
        return None
    return video, keyframe_path


def import_recording_background(
    root: zarr.Group, recording_dir: Path, *, zarr_path: Path, threads: int = 4
) -> dict[str, Any] | None:
    """Import-time form: a whole-recording median written into an archive being built.

    Called while import still owns the archive, so it neither takes the
    publication lock nor consolidates; import consolidates when it finishes.

    - No full-frame external-recorder video or keyframe list: returns None.
    - Fewer than three declared keyframes: ``skipped``.
    - A video this stage refuses to decode (pixel format, keyframe list that
      disagrees with the video): ``refused`` with the reason; nothing is written.
    - A whole-recording run already present: ``already_present``.

    A recording with fewer than 61 keyframes uses all of them. A decode that
    fails part-way raises: the video import just accepted cannot be read.
    """

    recording_dir = Path(recording_dir)
    inputs = full_frame_video_inputs(recording_dir)
    if inputs is None:
        return None
    _, keyframe_path = inputs
    existing = _whole_recording_run(root)
    if existing is not None:
        return {"status": STATUS_ALREADY_PRESENT, "run_name": existing}
    try:
        payload = json.loads(keyframe_path.read_text(encoding="utf-8"))
        available = len(keyframe_video_frame_indices(payload, label=str(keyframe_path)))
        if available < MIN_KEYFRAME_COUNT:
            return {
                "status": STATUS_SKIPPED,
                "reason": "too_few_keyframes",
                "declared_keyframe_count": available,
            }
        plan = plan_recording(
            recording_dir, count=min(DEFAULT_KEYFRAME_COUNT, available), zarr_path=zarr_path
        )
    except ValueError as exc:
        return {"status": STATUS_REFUSED, "detail": str(exc)[:500]}
    median, source, parameters = _median_of_plan(plan, window_note=IMPORT_WINDOW_NOTE, threads=threads)
    run_name = write_video_keyframe_median_background(
        root, background_full=median, source=source, parameters=parameters
    )
    return {
        "status": STATUS_WRITTEN,
        "run_name": run_name,
        "keyframe_count": len(plan["source_frame_indices"]),
        "declared_keyframe_count": plan["declared_keyframe_count"],
        "decoded_keyframes_sha256": source["decoded_keyframes_sha256"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--keyframes", type=int, default=DEFAULT_KEYFRAME_COUNT, help="number of keyframes in the median")
    parser.add_argument("--first-frame", type=int, default=None, help="first video frame of the sampling window (inclusive)")
    parser.add_argument("--last-frame", type=int, default=None, help="last video frame of the sampling window (inclusive)")
    parser.add_argument("--window-note", default=None, help="why this window, stored with the run")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="report the frames that would be sampled; write nothing")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    kwargs = {"count": args.keyframes, "first_frame": args.first_frame, "last_frame": args.last_frame}
    if args.dry_run:
        plan = plan_recording(args.recording_dir, **kwargs)
        result = {key: value for key, value in plan.items() if key != "declared_keyframes"}
        result["status"] = "would_write"
    else:
        result = compute_recording(
            args.recording_dir, window_note=args.window_note, threads=args.threads, **kwargs
        )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "BACKGROUND_DS_SHAPE",
    "DEFAULT_KEYFRAME_COUNT",
    "METHOD",
    "RUN_PREFIX",
    "compute_recording",
    "decode_keyframes",
    "full_frame_video_inputs",
    "import_recording_background",
    "downsample_background",
    "plan_recording",
    "select_keyframes",
    "write_video_keyframe_median_background",
]


if __name__ == "__main__":
    raise SystemExit(main())
