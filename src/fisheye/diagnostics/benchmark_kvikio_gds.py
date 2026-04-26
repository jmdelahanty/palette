"""Probe KvikIO/GPUDirect Storage write behavior in isolated child processes.

KvikIO failures can happen during interpreter teardown after a successful write.
The parent process therefore never imports KvikIO directly; each risky probe runs
in a child process and reports the child return code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_MODULE = "fisheye.diagnostics.benchmark_kvikio_gds"
_PAGE_SIZE = 4096


@dataclass(frozen=True)
class ProbeResult:
    name: str
    returncode: int
    elapsed_seconds: float
    payload: dict[str, Any]
    stdout: str
    stderr: str


def _aligned_size(size_mib: int) -> int:
    size = max(1, int(size_mib)) * 1024 * 1024
    remainder = size % _PAGE_SIZE
    if remainder:
        size += _PAGE_SIZE - remainder
    return size


def _status_from_returncode(returncode: int) -> str:
    if returncode == 0:
        return "success"
    if returncode < 0:
        return f"signal {-returncode}"
    return f"failed({returncode})"


def _json_payload_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _child_command(
    mode: str,
    *,
    scratch_dir: Path,
    size_mib: int,
    os_exit: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        _MODULE,
        "_child",
        mode,
        "--scratch-dir",
        str(scratch_dir),
        "--size-mib",
        str(int(size_mib)),
    ]
    if os_exit:
        cmd.append("--os-exit")
    return cmd


def _run_child_probe(
    name: str,
    *,
    mode: str,
    scratch_dir: Path,
    size_mib: int,
    os_exit: bool = False,
    timeout_seconds: float = 60.0,
) -> ProbeResult:
    start = time.perf_counter()
    completed = subprocess.run(
        _child_command(mode, scratch_dir=scratch_dir, size_mib=size_mib, os_exit=os_exit),
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
    )
    elapsed = time.perf_counter() - start
    return ProbeResult(
        name=name,
        returncode=int(completed.returncode),
        elapsed_seconds=float(elapsed),
        payload=_json_payload_from_stdout(completed.stdout),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _emit_json(payload: dict[str, Any], *, os_exit: bool) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)
    if os_exit:
        os._exit(0)


def _child_availability(*, os_exit: bool) -> None:
    import kvikio
    import kvikio.cufile_driver as driver
    import kvikio.defaults

    payload: dict[str, Any] = {
        "kvikio_version": getattr(kvikio, "__version__", "unknown"),
        "compat_mode": kvikio.defaults.get("compat_mode"),
        "num_threads": kvikio.defaults.get("num_threads"),
        "task_size": kvikio.defaults.get("task_size"),
    }
    for key in ("is_gds_available", "allow_compat_mode", "major_version", "minor_version"):
        try:
            payload[key] = driver.get(key)
        except Exception as exc:  # pragma: no cover - depends on local driver state
            payload[key] = f"{type(exc).__name__}: {exc}"
    _emit_json(payload, os_exit=os_exit)


def _child_cufile_write(
    *,
    scratch_dir: Path,
    size_mib: int,
    use_gpu_buffer: bool,
    os_exit: bool,
) -> None:
    from kvikio import CuFile
    import kvikio.cufile_driver as driver

    scratch_dir.mkdir(parents=True, exist_ok=True)
    size = _aligned_size(size_mib)
    suffix = "gpu" if use_gpu_buffer else "host"
    path = scratch_dir / f"palette_kvikio_cufile_{suffix}_{os.getpid()}.bin"
    if path.exists():
        path.unlink()

    if use_gpu_buffer:
        import cupy as cp

        buf = cp.arange(size, dtype=cp.uint8)
        cp.cuda.Stream.null.synchronize()
    else:
        import numpy as np

        buf = np.arange(size, dtype=np.uint8)

    start = time.perf_counter()
    with CuFile(path, "w") as handle:
        written = int(handle.pwrite(buf, size=size, file_offset=0).get())
    if use_gpu_buffer:
        cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    payload = {
        "path": str(path),
        "buffer": suffix,
        "is_gds_available": driver.get("is_gds_available"),
        "bytes_requested": int(size),
        "bytes_written": int(written),
        "file_size": int(path.stat().st_size),
        "write_seconds": float(elapsed),
        "write_mib_per_second": float((written / (1024 * 1024)) / elapsed) if elapsed > 0 else None,
    }
    _emit_json(payload, os_exit=os_exit)


def _child_zarr_gds_store(
    *,
    scratch_dir: Path,
    size_mib: int,
    os_exit: bool,
) -> None:
    import numpy as np
    import zarr
    from kvikio.zarr import GDSStore

    scratch_dir.mkdir(parents=True, exist_ok=True)
    root_path = scratch_dir / f"palette_kvikio_gds_store_{os.getpid()}.zarr"
    if root_path.exists():
        import shutil

        shutil.rmtree(root_path)
    store = GDSStore(root=str(root_path), read_only=False)
    root = zarr.open_group(store=store, mode="w", use_consolidated=False)
    size = _aligned_size(size_mib)
    data = np.arange(size, dtype=np.uint8)
    arr = root.create_array("payload", shape=data.shape, chunks=(min(size, 1024 * 1024),), dtype=np.uint8)
    start = time.perf_counter()
    arr[:] = data
    elapsed = time.perf_counter() - start
    store.close()
    payload = {
        "path": str(root_path),
        "bytes_written": int(size),
        "write_seconds": float(elapsed),
        "write_mib_per_second": float((size / (1024 * 1024)) / elapsed) if elapsed > 0 else None,
    }
    _emit_json(payload, os_exit=os_exit)


def _run_child(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Internal child process for KvikIO/GDS probes.")
    parser.add_argument("mode", choices=("availability", "cufile-host", "cufile-gpu", "zarr-gds-store"))
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--size-mib", type=int, default=64)
    parser.add_argument("--os-exit", action="store_true", help="Bypass Python interpreter teardown after emitting JSON.")
    args = parser.parse_args(argv)

    if args.mode == "availability":
        _child_availability(os_exit=bool(args.os_exit))
    elif args.mode == "cufile-host":
        _child_cufile_write(
            scratch_dir=args.scratch_dir,
            size_mib=int(args.size_mib),
            use_gpu_buffer=False,
            os_exit=bool(args.os_exit),
        )
    elif args.mode == "cufile-gpu":
        _child_cufile_write(
            scratch_dir=args.scratch_dir,
            size_mib=int(args.size_mib),
            use_gpu_buffer=True,
            os_exit=bool(args.os_exit),
        )
    elif args.mode == "zarr-gds-store":
        _child_zarr_gds_store(
            scratch_dir=args.scratch_dir,
            size_mib=int(args.size_mib),
            os_exit=bool(args.os_exit),
        )
    return 0


def _print_result(result: ProbeResult) -> None:
    status = _status_from_returncode(result.returncode)
    payload_text = ""
    if result.payload:
        selected = {
            key: result.payload[key]
            for key in (
                "is_gds_available",
                "kvikio_version",
                "buffer",
                "bytes_written",
                "file_size",
                "write_mib_per_second",
            )
            if key in result.payload
        }
        payload_text = " " + json.dumps(selected, sort_keys=True)
    print(f"{result.name}: {status} ({result.elapsed_seconds:.2f}s){payload_text}")
    if result.returncode != 0 and result.stderr.strip():
        print(result.stderr.strip().splitlines()[-1])


def _run_parent(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark KvikIO/GPUDirect Storage write paths in child processes. "
            "Run outside the Codex sandbox for meaningful GDS results."
        )
    )
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp"), help="Scratch location for probe outputs.")
    parser.add_argument("--size-mib", type=int, default=64, help="Aligned payload size for write probes.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="Per-child timeout.")
    parser.add_argument(
        "--skip-normal-teardown",
        action="store_true",
        help="Only run os._exit child probes; this measures IO but cannot detect teardown crashes.",
    )
    args = parser.parse_args(argv)

    probes: list[tuple[str, str, bool]] = [("availability", "availability", False)]
    for mode in ("cufile-host", "cufile-gpu", "zarr-gds-store"):
        if not args.skip_normal_teardown:
            probes.append((f"{mode}:normal-teardown", mode, False))
        probes.append((f"{mode}:os-exit", mode, True))

    worst_returncode = 0
    for name, mode, os_exit in probes:
        try:
            result = _run_child_probe(
                name,
                mode=mode,
                scratch_dir=args.scratch_dir,
                size_mib=int(args.size_mib),
                os_exit=bool(os_exit),
                timeout_seconds=float(args.timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            result = ProbeResult(
                name=name,
                returncode=124,
                elapsed_seconds=float(args.timeout_seconds),
                payload={},
                stdout=exc.stdout or "",
                stderr=exc.stderr or "timeout",
            )
        _print_result(result)
        if result.returncode != 0:
            worst_returncode = result.returncode
    return 0 if worst_returncode == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "_child":
        return _run_child(args[1:])
    return _run_parent(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
