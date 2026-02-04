#!/usr/bin/env python3
"""
Repro for asyncio thread-safe scheduling behavior in this environment.

Expected behavior:
  - call_soon_threadsafe should execute a callback.
  - run_coroutine_threadsafe should complete a simple coroutine.
"""

from __future__ import annotations

import asyncio
import threading
import time


def _start_loop(loop_ready: threading.Event, loop_holder: dict) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop_holder["loop"] = loop
    loop_ready.set()
    loop.run_forever()


def main() -> None:
    print(f"asyncio policy: {type(asyncio.get_event_loop_policy()).__name__}")

    loop_ready = threading.Event()
    loop_holder: dict = {}
    thread = threading.Thread(
        target=_start_loop,
        args=(loop_ready, loop_holder),
        name="asyncio_repro_loop",
        daemon=True,
    )
    thread.start()
    loop_ready.wait(timeout=2)

    loop = loop_holder.get("loop")
    if loop is None:
        print("FAILED: loop never started")
        return

    print(f"loop: {type(loop).__name__}, running={loop.is_running()}")

    callback_ran = threading.Event()

    def _cb() -> None:
        callback_ran.set()

    loop.call_soon_threadsafe(_cb)

    coro_ran = threading.Event()
    coro_result = {}

    async def _coro() -> int:
        return 123

    future = asyncio.run_coroutine_threadsafe(_coro(), loop)

    try:
        coro_result["value"] = future.result(timeout=2)
        coro_ran.set()
    except Exception as exc:  # pragma: no cover - repro behavior
        coro_result["error"] = repr(exc)

    print(f"call_soon_threadsafe: {'OK' if callback_ran.wait(2) else 'HANG'}")
    if coro_ran.is_set():
        print(f"run_coroutine_threadsafe: OK result={coro_result['value']}")
    else:
        print(f"run_coroutine_threadsafe: FAILED error={coro_result.get('error')}")

    # Clean shutdown attempt
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)


if __name__ == "__main__":
    main()
