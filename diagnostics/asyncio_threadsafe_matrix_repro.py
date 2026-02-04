#!/usr/bin/env python3
"""
Matrix repro for asyncio thread-safe scheduling across thread roles.

Cases:
  A) Loop runs in background thread, main thread calls call_soon_threadsafe/run_coroutine_threadsafe
  B) Loop runs in main thread, background thread calls call_soon_threadsafe/run_coroutine_threadsafe
"""

from __future__ import annotations

import asyncio
import threading


def _print_loop_details(loop: asyncio.AbstractEventLoop) -> None:
    csock = getattr(loop, "_csock", None)
    ssock = getattr(loop, "_ssock", None)
    csock_fd = getattr(csock, "fileno", lambda: None)()
    ssock_fd = getattr(ssock, "fileno", lambda: None)()
    print(f"  loop: {type(loop).__name__}, running={loop.is_running()}")
    print(f"  _csock fd: {csock_fd}, _ssock fd: {ssock_fd}")


def _case_a() -> None:
    print("Case A: loop in background thread; main thread schedules")
    loop_ready = threading.Event()
    loop_holder: dict = {}

    def run_loop() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        loop_ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, name="case_a_loop", daemon=True)
    thread.start()
    loop_ready.wait(timeout=2)

    loop = loop_holder.get("loop")
    if loop is None:
        print("  FAILED: loop never started")
        return

    _print_loop_details(loop)

    cb_ran = threading.Event()

    def _cb() -> None:
        cb_ran.set()

    loop.call_soon_threadsafe(_cb)
    cb_ok = cb_ran.wait(timeout=2)
    print(f"  call_soon_threadsafe: {'OK' if cb_ok else 'HANG'}")

    async def _coro() -> int:
        return 7

    coro_ok = False
    coro_err = None
    try:
        fut = asyncio.run_coroutine_threadsafe(_coro(), loop)
        coro_ok = fut.result(timeout=2) == 7
    except Exception as exc:  # pragma: no cover - repro behavior
        coro_err = repr(exc)
    if coro_ok:
        print("  run_coroutine_threadsafe: OK result=7")
    else:
        print(f"  run_coroutine_threadsafe: FAILED error={coro_err}")

    try:
        loop.call_soon_threadsafe(lambda: None)
        loop._write_to_self()
        print("  _write_to_self: OK")
    except Exception as exc:  # pragma: no cover - repro behavior
        print(f"  _write_to_self: FAILED error={exc!r}")

    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)


def _case_b() -> None:
    print("Case B: loop in main thread; background thread schedules")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _print_loop_details(loop)

    cb_ran = threading.Event()
    coro_ran = threading.Event()
    coro_err: dict = {}

    def _cb() -> None:
        cb_ran.set()

    async def _coro() -> int:
        return 11

    def schedule_from_thread() -> None:
        loop.call_soon_threadsafe(_cb)
        try:
            fut = asyncio.run_coroutine_threadsafe(_coro(), loop)
            if fut.result(timeout=2) == 11:
                coro_ran.set()
        except Exception as exc:  # pragma: no cover - repro behavior
            coro_err["exc"] = repr(exc)

    t = threading.Thread(target=schedule_from_thread, name="case_b_scheduler", daemon=True)
    t.start()

    loop.call_later(2, loop.stop)
    loop.run_forever()

    print(f"  call_soon_threadsafe: {'OK' if cb_ran.is_set() else 'HANG'}")
    if coro_ran.is_set():
        print("  run_coroutine_threadsafe: OK result=11")
    else:
        print(f"  run_coroutine_threadsafe: FAILED error={coro_err.get('exc')}")

    t.join(timeout=2)


if __name__ == "__main__":
    print(f"asyncio policy: {type(asyncio.get_event_loop_policy()).__name__}")
    _case_a()
    _case_b()
