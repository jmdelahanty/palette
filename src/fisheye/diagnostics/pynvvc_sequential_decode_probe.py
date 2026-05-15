#!/usr/bin/env python3
"""Probe PyNvVideoCodec sequential NVDEC decode behavior.

This diagnostic intentionally uses PyNvVideoCodec's low-level demuxer/decoder
path instead of random-access frame APIs. The goal is to measure whether a
sequential NVDEC path avoids Decord VideoReader's eager full-file frame index.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_exc(exc: BaseException) -> Dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _tensor_from_frame(frame: Any) -> Any:
    import torch

    return torch.from_dlpack(frame)


def _shape_of(value: Any) -> Optional[list[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(v) for v in shape]
    except Exception:
        return None


def _dtype_of(value: Any) -> Optional[str]:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


def _device_of(value: Any) -> Optional[str]:
    device = getattr(value, "device", None)
    return None if device is None else str(device)


def _cuda_interface_summary(frame: Any) -> Optional[Dict[str, Any]]:
    cuda_method = getattr(frame, "cuda", None)
    if not callable(cuda_method):
        return None
    try:
        cuda_value = cuda_method()
    except Exception as exc:
        return {"error": _format_exc(exc)}
    if isinstance(cuda_value, dict):
        return {
            "type": "dict",
            "keys": sorted(str(k) for k in cuda_value.keys()),
            "shape": cuda_value.get("shape"),
            "typestr": cuda_value.get("typestr"),
        }
    if isinstance(cuda_value, (list, tuple)):
        return {
            "type": type(cuda_value).__name__,
            "length": len(cuda_value),
            "items": [
                {
                    "type": type(item).__name__,
                    "keys": sorted(str(k) for k in item.keys()) if isinstance(item, dict) else None,
                    "shape": item.get("shape") if isinstance(item, dict) else None,
                    "typestr": item.get("typestr") if isinstance(item, dict) else None,
                }
                for item in cuda_value[:4]
            ],
        }
    return {"type": type(cuda_value).__name__, "repr": repr(cuda_value)[:200]}


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "status": "ok",
        "inputs": {
            "video_path": str(video_path),
            "max_frames": int(args.max_frames),
            "batch_size": int(args.batch_size),
            "gpu_id": int(args.gpu_id),
            "use_device_memory": bool(args.use_device_memory),
            "convert_dlpack": bool(args.convert_dlpack),
            "sync_cuda": bool(args.sync_cuda),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "stages": {},
        "summary": {},
        "first_frame": {},
        "batches": [],
    }

    t_import = time.perf_counter()
    try:
        import PyNvVideoCodec as nvc  # type: ignore
    except Exception as exc:
        payload["status"] = "import_error"
        payload["error"] = _format_exc(exc)
        return payload
    payload["stages"]["import_seconds"] = time.perf_counter() - t_import
    payload["environment"]["pynvvideocodec_module"] = getattr(nvc, "__file__", None)
    payload["environment"]["pynvvideocodec_version"] = getattr(nvc, "__version__", None)

    torch = None
    if args.convert_dlpack or args.sync_cuda:
        t_torch = time.perf_counter()
        try:
            import torch as torch_mod

            torch = torch_mod
        except Exception as exc:
            payload["status"] = "torch_import_error"
            payload["error"] = _format_exc(exc)
            return payload
        payload["stages"]["torch_import_seconds"] = time.perf_counter() - t_torch
        payload["environment"]["torch_version"] = getattr(torch, "__version__", None)
        payload["environment"]["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["environment"]["cuda_device_name"] = torch.cuda.get_device_name(0)

    t_demux = time.perf_counter()
    demuxer = nvc.CreateDemuxer(filename=str(video_path))
    payload["stages"]["demuxer_create_seconds"] = time.perf_counter() - t_demux

    demux_meta: Dict[str, Any] = {}
    for name in ("FrameRate", "Width", "Height", "GetNvCodecId", "ChromaFormat", "BitDepth"):
        method = getattr(demuxer, name, None)
        if callable(method):
            try:
                value = method()
                demux_meta[name] = str(value) if name == "GetNvCodecId" else value
            except Exception as exc:
                demux_meta[f"{name}_error"] = _format_exc(exc)
    payload["demuxer"] = demux_meta

    t_decoder = time.perf_counter()
    decoder = nvc.CreateDecoder(
        gpuid=int(args.gpu_id),
        codec=demuxer.GetNvCodecId(),
        usedevicememory=bool(args.use_device_memory),
    )
    payload["stages"]["decoder_create_seconds"] = time.perf_counter() - t_decoder

    frames_processed = 0
    packets_processed = 0
    batch_frames = 0
    batch_start: Optional[float] = None
    batch_index = 0
    dlpack_errors: list[Dict[str, str]] = []
    t_first_frame: Optional[float] = None
    t_decode = time.perf_counter()

    for packet in demuxer:
        packets_processed += 1
        decoded_frames = decoder.Decode(packet)
        for frame in decoded_frames:
            if t_first_frame is None:
                t_first_frame = time.perf_counter()
                payload["first_frame"]["seconds_from_decode_start"] = t_first_frame - t_decode
                payload["first_frame"]["type"] = type(frame).__name__
                payload["first_frame"]["shape"] = _shape_of(frame)
                payload["first_frame"]["dtype"] = _dtype_of(frame)
                payload["first_frame"]["cuda_interface"] = _cuda_interface_summary(frame)

            if batch_start is None:
                batch_start = time.perf_counter()

            if args.convert_dlpack:
                try:
                    tensor = _tensor_from_frame(frame)
                    if frames_processed == 0:
                        payload["first_frame"]["dlpack_tensor_shape"] = _shape_of(tensor)
                        payload["first_frame"]["dlpack_tensor_dtype"] = _dtype_of(tensor)
                        payload["first_frame"]["dlpack_tensor_device"] = _device_of(tensor)
                    del tensor
                except Exception as exc:
                    if len(dlpack_errors) < 5:
                        dlpack_errors.append(_format_exc(exc))

            frames_processed += 1
            batch_frames += 1

            if batch_frames == int(args.batch_size):
                if args.sync_cuda and torch is not None and torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_end = time.perf_counter()
                payload["batches"].append(
                    {
                        "batch_index": int(batch_index),
                        "frames": int(batch_frames),
                        "seconds": float(batch_end - (batch_start or batch_end)),
                    }
                )
                batch_index += 1
                batch_frames = 0
                batch_start = None

            if frames_processed >= int(args.max_frames):
                break
        if frames_processed >= int(args.max_frames):
            break

    if batch_frames > 0:
        if args.sync_cuda and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_end = time.perf_counter()
        payload["batches"].append(
            {
                "batch_index": int(batch_index),
                "frames": int(batch_frames),
                "seconds": float(batch_end - (batch_start or batch_end)),
            }
        )

    if args.flush and frames_processed < int(args.max_frames):
        for frame in decoder.Flush():
            del frame

    decode_seconds = time.perf_counter() - t_decode
    payload["stages"]["decode_loop_seconds"] = float(decode_seconds)
    payload["summary"] = {
        "packets_processed": int(packets_processed),
        "frames_processed": int(frames_processed),
        "batches_processed": int(len(payload["batches"])),
        "decode_fps": float(frames_processed / decode_seconds) if decode_seconds > 0 else 0.0,
        "dlpack_errors": dlpack_errors,
    }
    return payload


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe PyNvVideoCodec sequential demux/decode timings."
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument("--max-frames", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-device-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--convert-dlpack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flush", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    payload = run_probe(args)
    text = json.dumps(payload, indent=2)
    if args.output_json is not None:
        out = args.output_json.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
