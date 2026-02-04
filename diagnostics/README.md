# Diagnostics

## zarr_open_group_hang_repro.py

Purpose: reproduce a hang observed with sync `zarr.open_group` in this environment.

Run:

```bash
/home/delahantyj@hhmi.org/miniconda3/envs/palette-py311/bin/python \
  diagnostics/zarr_open_group_hang_repro.py
```

Expected output (approximate):

```
python 3.11.x
zarr 3.1.x
sync open_group MemoryStore: HANG after 5s
sync open_group LocalStore: HANG after 5s
async open_group MemoryStore: exit=0
```

## asyncio_threadsafe_repro.py

Purpose: check whether `call_soon_threadsafe` and `run_coroutine_threadsafe` actually
execute when a loop is running in a background thread.

Run:

```bash
/home/delahantyj@hhmi.org/miniconda3/envs/palette-py311/bin/python \
  diagnostics/asyncio_threadsafe_repro.py
```

Expected output (approximate):

```
asyncio policy: DefaultEventLoopPolicy
loop: _UnixSelectorEventLoop, running=True
call_soon_threadsafe: OK
run_coroutine_threadsafe: OK result=123
```

Observed output (palette-py311 on 2026-02-04) saved at:

- `diagnostics/asyncio_threadsafe_repro_output.txt`

## asyncio_threadsafe_matrix_repro.py

Purpose: check thread-safe scheduling behavior when the loop runs in a
background thread vs the main thread.

Run:

```bash
/home/delahantyj@hhmi.org/miniconda3/envs/palette-py311/bin/python \
  diagnostics/asyncio_threadsafe_matrix_repro.py
```

Observed output (palette-py311 on 2026-02-04) saved at:

- `diagnostics/asyncio_threadsafe_matrix_repro_output.txt`

Interpretation of the observed output:

- Case A (loop running in a background thread) shows `call_soon_threadsafe` and
  `run_coroutine_threadsafe` do not wake the loop; both time out. This means
  thread-safe scheduling into a background-loop is broken in this environment.
- Case B (loop running in the main thread) succeeds for both thread-safe calls.
  This suggests the failure is specific to *background-loop wakeups*, not a
  general asyncio failure.
- Zarr's sync wrapper relies on a background event loop thread; this failure
  explains why sync `zarr.open_group` hangs while the async API works.

Local shell verification (palette-py311 on 2026-02-04):

- `python diagnostics/asyncio_threadsafe_matrix_repro.py` reports OK in both cases.
- `python diagnostics/zarr_open_group_hang_repro.py` reports `exit=0` for sync/async.
- `python -m pytest tests/unit/fisheye/test_zarr_schema.py` passes (2 tests).
