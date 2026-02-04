#!/usr/bin/env python3
"""
Repro for sync zarr.open_group hangs in this environment.

Expected behavior:
  - sync open_group (MemoryStore / LocalStore): HANG
  - async open_group (MemoryStore): success
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile


def _run_sync_mem() -> None:
    import zarr
    from zarr.storage import MemoryStore

    zarr.open_group(store=MemoryStore(), mode="w", zarr_format=3)


def _run_sync_local() -> None:
    import zarr
    from zarr.storage import LocalStore

    path = tempfile.mkdtemp()
    zarr.open_group(store=LocalStore(path), mode="w", zarr_format=3)


def _run_async_mem() -> None:
    import asyncio
    from zarr.api.asynchronous import open_group
    from zarr.storage import MemoryStore

    async def main() -> None:
        await open_group(store=MemoryStore(), mode="w", zarr_format=3)

    asyncio.run(main())


def _run_case(name: str, fn, timeout: int = 5) -> None:
    proc = mp.Process(target=fn)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        print(f"{name}: HANG after {timeout}s")
        proc.terminate()
        proc.join()
        return
    print(f"{name}: exit={proc.exitcode}")


def main() -> None:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    print(f"python {sys.version.split()[0]}")
    try:
        import zarr
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"zarr import failed: {exc}")
        return
    print(f"zarr {zarr.__version__}")

    _run_case("sync open_group MemoryStore", _run_sync_mem)
    _run_case("sync open_group LocalStore", _run_sync_local)
    _run_case("async open_group MemoryStore", _run_async_mem)


if __name__ == "__main__":
    main()
