#!/usr/bin/env python3
"""
Smoke-test that kvikIO is using true GPUDirect Storage (NVFS) and not POSIX compat.
- Forces compat OFF via kvikio defaults + env
- Measures write/read GB/s on GPU buffers
- Scans fresh /tmp/cufile_*.log for "posix path taken" / "bounce"
- Optionally runs gdscheck -p if present
"""

import os
import re
import sys
import glob
import time
import json
import shutil
import subprocess as sp

# ---------- Hard requirements you can tweak ----------
FILEPATH = os.environ.get("KVS_TEST_FILE", "/nvme1/kvikio_test.bin")
SIZE_MB  = int(os.environ.get("KVS_TEST_MB", "1024"))         # 1 GiB default
GPU_ID   = int(os.environ.get("KVS_TEST_GPU", "0"))
CUFILE_JSON = os.environ.get("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
# -----------------------------------------------------

def banner(msg):
    print("\n" + "=" * 64)
    print(msg)
    print("=" * 64)

def now_ns():
    return time.time_ns()

def newest_logs(since_ns):
    # Only consider cuFile logs created after we started the run
    logs = []
    for p in glob.glob("/tmp/cufile_*.log"):
        try:
            if os.path.getmtime(p) * 1e9 >= since_ns:
                logs.append(p)
        except Exception:
            pass
    return sorted(logs, key=os.path.getmtime)

def scan_logs(paths):
    txt = []
    for p in paths:
        try:
            with open(p, "r", errors="ignore") as f:
                txt.append(f.read())
        except Exception:
            pass
    blob = "\n".join(txt)
    # key phrases that indicate compat/bounce vs NVFS direct
    compat_hits = len(re.findall(r"posix path taken", blob, flags=re.IGNORECASE))
    bounce_hits = len(re.findall(r"bounce buffer", blob, flags=re.IGNORECASE))
    nvfs_hits   = len(re.findall(r"\bnvfs[_-]", blob, flags=re.IGNORECASE)) + \
                  len(re.findall(r"gds path taken", blob, flags=re.IGNORECASE))
    return compat_hits, bounce_hits, nvfs_hits

def show_gdscheck():
    gdscheck = shutil.which("gdscheck")
    if not gdscheck:
        print("gdscheck not found on PATH")
        return
    try:
        out = sp.run([gdscheck, "-p"], capture_output=True, text=True, timeout=8)
        focus = []
        keep = False
        for line in out.stdout.splitlines():
            if "DRIVER CONFIGURATION:" in line:
                keep = True
            if keep:
                focus.append(line)
            if keep and line.strip().startswith("="):
                break
        print("\n[gdscheck -p focus]")
        print("\n".join(focus) if focus else out.stdout[:400])
    except Exception as e:
        print(f"gdscheck -p failed: {e}")

def read_json_safe(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def align_4k(n):
    return (n // 4096) * 4096

def set_env_for_gds():
    # Strong hints to avoid compat
    os.environ["KVIKIO_COMPAT_MODE"] = "OFF"         # kvikIO hint
    os.environ["CUFILE_ENV_PATH_JSON"] = CUFILE_JSON # ensure cuFile reads our config
    # (Your /etc/cufile.json already has allow_compat_mode:false)

def main():
    banner("kvikIO GDS Capability Test")

    # 0) Env + config
    set_env_for_gds()
    print(f"CUFILE_ENV_PATH_JSON = {os.environ.get('CUFILE_ENV_PATH_JSON')}")
    print(f"KVIKIO_COMPAT_MODE   = {os.environ.get('KVIKIO_COMPAT_MODE')}")
    cfg = read_json_safe(os.environ["CUFILE_ENV_PATH_JSON"])
    props = cfg.get("properties", {})
    print(f"cufile.json properties: {json.dumps(props, indent=2) or '<none>'}")

    # 1) Imports
    try:
        import cupy as cp
        import kvikio
        from kvikio import CuFile
        import kvikio.defaults as kvd
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    # 2) Show basic versions
    print(f"\nCuPy:    {getattr(cp, '__version__', 'unknown')}")
    print(f"kvikIO:  {getattr(kvikio, '__version__', 'unknown')} @ {getattr(kvikio,'__file__','')}")
    # kvikIO defaults (API differs across versions; guard it)
    try:
        print(f"compat_mode(): {kvd.compat_mode()}")
    except Exception:
        pass
    for name in ("num_threads", "task_size"):
        if hasattr(kvd, name):
            try:
                print(f"{name}(): {getattr(kvd, name)()}")
            except Exception:
                pass

    # Try to force compat OFF via kvikIO defaults if available
    for fn in ("compat_mode_reset", "compat_mode_set"):
        if hasattr(kvd, fn):
            try:
                getattr(kvd, fn)(False)
                print(f"Set kvikIO defaults: {fn}(False)")
            except Exception:
                pass

    # 3) Optional: gdscheck focus
    show_gdscheck()

    # 4) Prepare data
    size = align_4k(SIZE_MB * 1024 * 1024)
    cp.cuda.Device(GPU_ID).use()
    data = cp.random.randint(0, 256, size, dtype=cp.uint8)
    readbuf = cp.empty_like(data)

    # 5) Remove any old logs so our scan is clean
    for p in newest_logs(0):
        try: os.remove(p)
        except Exception: pass

    # 6) GDS (expected) path
    banner("Test A — GDS forced (compat OFF)")
    t0 = now_ns()
    try:
        # WRITE
        t = time.perf_counter()
        with CuFile(FILEPATH, "wb") as f:
            n = f.write(data)
        cp.cuda.Stream.null.synchronize()
        wt = time.perf_counter() - t

        # READ
        t = time.perf_counter()
        with CuFile(FILEPATH, "rb") as f:
            m = f.read(readbuf)
        cp.cuda.Stream.null.synchronize()
        rt = time.perf_counter() - t

        gds_logs = newest_logs(t0)
        compat_hits, bounce_hits, nvfs_hits = scan_logs(gds_logs)

        print(f"Write: {SIZE_MB / wt:0.2f} GB/s  ({n} bytes)")
        print(f"Read : {SIZE_MB / rt:0.2f} GB/s  ({m} bytes)")
        print(f"Log hints → compat:{compat_hits}  bounce:{bounce_hits}  nvfs-ish:{nvfs_hits}")

        if n != size or m != size:
            print("✗ Size mismatch; aborting further checks.")
            sys.exit(2)

        # Basic data check (sampled to avoid full transfer back to host)
        ok = bool(int(cp.logical_and(data[:1_000_000] == readbuf[:1_000_000]).all().get()))
        print("Data check (first 1M):", "OK" if ok else "MISMATCH")

        gds_ok = (compat_hits == 0 and bounce_hits == 0)
        print("GDS verdict:", "✅ FAST-PATH (no POSIX/bounce seen)" if gds_ok else "⚠️ Compat hints seen in logs")

    except Exception as e:
        print(f"✗ GDS run failed: {e}")
        gds_ok = False

    # 7) Compat comparison (flip ON, then reset back OFF)
    banner("Test B — forced COMPAT for comparison")
    compat_speed_w = compat_speed_r = 0.0
    # try several API shapes to force compat on
    forced = False
    for fn in ("compat_mode_reset", "compat_mode_set"):
        if hasattr(kvd, fn):
            try:
                getattr(kvd, fn)(True)
                forced = True
                print(f"Set kvikIO defaults: {fn}(True)")
                break
            except Exception:
                pass
    os.environ["KVIKIO_COMPAT_MODE"] = "ON"

    try:
        t = time.perf_counter()
        with CuFile(FILEPATH, "wb") as f:
            f.write(data)
        cp.cuda.Stream.null.synchronize()
        compat_speed_w = SIZE_MB / (time.perf_counter() - t)

        t = time.perf_counter()
        with CuFile(FILEPATH, "rb") as f:
            f.read(readbuf)
        cp.cuda.Stream.null.synchronize()
        compat_speed_r = SIZE_MB / (time.perf_counter() - t)

        print(f"Compat Write: {compat_speed_w:0.2f} GB/s")
        print(f"Compat Read : {compat_speed_r:0.2f} GB/s")
    except Exception as e:
        print(f"(Compat) run failed: {e}")

    # reset back to GDS
    os.environ["KVIKIO_COMPAT_MODE"] = "OFF"
    for fn in ("compat_mode_reset", "compat_mode_set"):
        if hasattr(kvd, fn):
            try:
                getattr(kvd, fn)(False)
            except Exception:
                pass

    # 8) Cleanup + summary
    try:
        os.remove(FILEPATH)
    except Exception:
        pass

    banner("SUMMARY")
    print(f"GDS write/read: {SIZE_MB / wt:0.2f} / {SIZE_MB / rt:0.2f} GB/s")
    if compat_speed_w and compat_speed_r:
        print(f"Compat write/read: {compat_speed_w:0.2f} / {compat_speed_r:0.2f} GB/s")
        ws = (SIZE_MB / wt) / max(compat_speed_w, 1e-6)
        rs = (SIZE_MB / rt) / max(compat_speed_r, 1e-6)
        print(f"Speedups (GDS/Compat): write {ws:0.2f}×  read {rs:0.2f}×")
    print(f"Log hints (A): compat:{compat_hits}  bounce:{bounce_hits}  nvfs-ish:{nvfs_hits}")
    print("Verdict:", "✅ TRUE GDS" if gds_ok else "⚠️ Compat signals detected — check config/driver")

if __name__ == "__main__":
    main()
