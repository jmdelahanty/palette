#!/usr/bin/env python3
"""
Smoke-test that kvikIO uses true GPUDirect Storage (NVFS) and NOT POSIX compat.

What it does:
- Forces compat OFF via kvikIO defaults + env
- Writes/reads GPU buffers; reports GiB/s (with CUDA syncs)
- Scans fresh /tmp/cufile_*.log for "posix path taken"
- Optionally runs gdscheck -p (focus section)
- Optionally drops page cache before reads to avoid measuring RAM speed
- Frees GPU pools & exits cleanly to avoid teardown segfaults
"""

import os, re, sys, glob, time, json, shutil
import subprocess as sp

# ---------- Tunables ----------
FILEPATH        = os.environ.get("KVS_TEST_FILE", "/nvme1/kvikio_test.bin")
SIZE_MB         = int(os.environ.get("KVS_TEST_MB", "1024"))       # 1 GiB default
GPU_ID          = int(os.environ.get("KVS_TEST_GPU", "0"))
CUFILE_JSON     = os.environ.get("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
DROP_CACHES     = os.environ.get("KVS_TEST_DROP_CACHES", "1") == "1"  # needs sudo; set to "0" to disable
KV_THREADS      = int(os.environ.get("KVS_THREADS", "4"))          # kvikIO num_threads baseline
KV_TASK_SIZE_MB = int(os.environ.get("KVS_TASK_MB", "32"))         # kvikIO task_size baseline
VERIFY_SAMPLES  = 1_000_000  # compare the first 1M bytes on device
# ------------------------------

def banner(msg):
    print("\n" + "=" * 64)
    print(msg)
    print("=" * 64)

def gib_per_sec(nbytes, seconds):
    return (nbytes / (1024**3)) / max(seconds, 1e-9)

def now_ns():
    return time.time_ns()

def newest_logs(since_ns):
    logs = []
    for p in glob.glob("/tmp/cufile_*.log"):
        try:
            if os.path.getmtime(p) * 1e9 >= since_ns:
                logs.append(p)
        except Exception:
            pass
    return sorted(logs, key=os.path.getmtime)

def scan_logs(paths):
    blob = ""
    for p in paths:
        try:
            with open(p, "r", errors="ignore") as f:
                blob += f.read() + "\n"
        except Exception:
            pass
    compat_hits = len(re.findall(r"posix path taken", blob, flags=re.IGNORECASE))
    # "bounce buffer" appear at init/teardown; not fatal
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
        focus, keep = [], False
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

def try_drop_caches():
    """Drop page cache so compat-mode reads don't just measure RAM."""
    if not DROP_CACHES:
        return
    try:
        sp.run(["sync"], check=False)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        print("Dropped page cache (echo 3 > /proc/sys/vm/drop_caches).")
    except Exception:
        print("Note: couldn’t drop caches (need sudo); compat read may be cache-hot.")

# ---- kvikIO defaults compatibility helpers (new/old APIs) ----
def kv_get(name, default=None):
    import kvikio.defaults as kvd
    # New API: get("compat_mode")
    try:
        return kvd.get(name)
    except TypeError:
        pass
    # Mid API: get() -> dict
    try:
        cfg = kvd.get()
        return cfg.get(name, default)
    except Exception:
        pass
    # Old API: compat_mode()
    if name == "compat_mode" and hasattr(kvd, "compat_mode"):
        try:
            return kvd.compat_mode()
        except Exception:
            return default
    return default

def kv_set(name, value):
    import kvikio.defaults as kvd
    # New API: set("compat_mode", 0/1)
    try:
        kvd.set(name, value)
        return
    except TypeError:
        pass
    # Mid API: set(dict)
    try:
        cfg = kvd.get()
        cfg[name] = value
        kvd.set(cfg)
        return
    except Exception:
        pass
    # Old API: compat_mode_reset(True/False)
    if name == "compat_mode":
        for fn in ("compat_mode_reset", "compat_mode_set"):
            if hasattr(kvd, fn):
                try:
                    getattr(kvd, fn)(bool(value))
                    return
                except Exception:
                    continue
# ---------------------------------------------------------------

def main():
    banner("kvikIO GDS Capability Test")

    # 0) Env + config
    os.environ["KVIKIO_COMPAT_MODE"] = "OFF"
    os.environ["CUFILE_ENV_PATH_JSON"] = CUFILE_JSON
    print(f"Target file      : {FILEPATH}")
    print(f"Test size        : {SIZE_MB} MiB")
    print(f"GPU id           : {GPU_ID}")
    print(f"CUFILE_ENV_PATH_JSON = {os.environ.get('CUFILE_ENV_PATH_JSON')}")
    print(f"KVIKIO_COMPAT_MODE   = {os.environ.get('KVIKIO_COMPAT_MODE')}")
    if not FILEPATH.startswith("/nvme1/"):
        print("⚠️  NOTE: test path is not under /nvme1 — be sure it’s on your GDS-enabled mount.")

    cfg = read_json_safe(os.environ["CUFILE_ENV_PATH_JSON"])
    props = cfg.get("properties", {})
    print(f"cufile.json properties: {json.dumps(props, indent=2) or '<none>'}")

    # 1) Imports
    try:
        import cupy as cp
        import kvikio
        from kvikio import CuFile
        import kvikio.defaults as kvd  # noqa: F401  (used via kv_get/kv_set)
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    # 2) Versions + kvikIO defaults (set tuning & compat off)
    print(f"\nCuPy   : {getattr(cp, '__version__', 'unknown')}")
    print(f"kvikIO : {getattr(kvikio, '__version__', 'unknown')} @ {getattr(kvikio,'__file__','')}")
    print(f"compat_mode [initial]: {kv_get('compat_mode')}")
    kv_set("compat_mode", 0)
    kv_set("num_threads", KV_THREADS)
    kv_set("task_size",  KV_TASK_SIZE_MB * 1024 * 1024)
    print(f"kvikIO defaults now: compat_mode={kv_get('compat_mode')}  "
          f"num_threads={kv_get('num_threads')}  task_size={kv_get('task_size')}")

    # 3) Optional: gdscheck focus
    show_gdscheck()

    # 4) Prepare data
    size = align_4k(SIZE_MB * 1024 * 1024)
    cp.cuda.Device(GPU_ID).use()
    data    = cp.random.randint(0, 256, size, dtype=cp.uint8)
    readbuf = cp.empty_like(data)

    # 5) Clear old cuFile logs
    for p in newest_logs(0):
        try: os.remove(p)
        except Exception: pass

    # 6) GDS (expected) path
    banner("Test A — GDS forced (compat OFF)")
    t0 = now_ns()
    gds_write_gib = gds_read_gib = 0.0
    gds_ok = False

    try:
        # WRITE
        t = time.perf_counter()
        with CuFile(FILEPATH, "wb") as f:
            n = f.write(data)
        cp.cuda.Stream.null.synchronize()
        wt = time.perf_counter() - t
        gds_write_gib = gib_per_sec(n, wt)

        # READ (drop caches so we don't measure RAM)
        try_drop_caches()
        t = time.perf_counter()
        with CuFile(FILEPATH, "rb") as f:
            m = f.read(readbuf)
        cp.cuda.Stream.null.synchronize()
        rt = time.perf_counter() - t
        gds_read_gib = gib_per_sec(m, rt)

        gds_logs = newest_logs(t0)
        compat_hits, bounce_hits, nvfs_hits = scan_logs(gds_logs)

        print(f"Write: {gds_write_gib:0.2f} GiB/s  ({n} bytes in {wt:0.6f}s)")
        print(f"Read : {gds_read_gib:0.2f} GiB/s  ({m} bytes in {rt:0.6f}s)")
        print(f"Log hints → compat:{compat_hits}  bounce:{bounce_hits}  nvfs-ish:{nvfs_hits}")

        if n != size or m != size:
            print("✗ Size mismatch; aborting further checks.")
            sys.exit(2)

        # Device-side comparison of a sample prefix
        sample = min(VERIFY_SAMPLES, size)
        ok = bool(cp.all(data[:sample] == readbuf[:sample]).get())
        print("Data check (first 1M):", "OK" if ok else "MISMATCH")

        gds_ok = (compat_hits == 0)  # ignore "bounce" chatter
        print("GDS verdict:", "✅ FAST-PATH (no POSIX fallback detected)" if gds_ok
              else "⚠️ Compat signals seen in logs")
    except Exception as e:
        print(f"✗ GDS run failed: {e}")

    # 7) Compat comparison
    banner("Test B — forced COMPAT for comparison")
    compat_write_gib = compat_read_gib = 0.0

    kv_set("compat_mode", 1)
    print(f"compat_mode [forced 1]: {kv_get('compat_mode')}")
    os.environ["KVIKIO_COMPAT_MODE"] = "ON"

    try:
        t = time.perf_counter()
        with CuFile(FILEPATH, "wb") as f:
            f.write(data)
        cp.cuda.Stream.null.synchronize()
        compat_write_gib = gib_per_sec(size, time.perf_counter() - t)

        try_drop_caches()
        t = time.perf_counter()
        with CuFile(FILEPATH, "rb") as f:
            f.read(readbuf)
        cp.cuda.Stream.null.synchronize()
        compat_read_gib = gib_per_sec(size, time.perf_counter() - t)

        print(f"Compat Write: {compat_write_gib:0.2f} GiB/s")
        print(f"Compat Read : {compat_read_gib:0.2f} GiB/s")
    except Exception as e:
        print(f"(Compat) run failed: {e}")

    # reset back to GDS
    kv_set("compat_mode", 0)
    os.environ["KVIKIO_COMPAT_MODE"] = "OFF"

    # 8) Cleanup + summary (free pools to avoid segfaults)
    try: os.remove(FILEPATH)
    except Exception: pass

    banner("SUMMARY")
    print(f"GDS   write/read : {gds_write_gib:0.2f} / {gds_read_gib:0.2f} GiB/s")
    if compat_write_gib and compat_read_gib:
        ws = gds_write_gib / max(compat_write_gib, 1e-9)
        rs = gds_read_gib  / max(compat_read_gib,  1e-9)
        print(f"Compat write/read: {compat_write_gib:0.2f} / {compat_read_gib:0.2f} GiB/s")
        print(f"Speedups (GDS/Compat): write {ws:0.2f}×  read {rs:0.2f}×")

    print(f"Log hints (A): compat:{compat_hits if 'compat_hits' in locals() else '?'}  "
          f"bounce:{bounce_hits if 'bounce_hits' in locals() else '?'}  "
          f"nvfs-ish:{nvfs_hits if 'nvfs_hits' in locals() else '?'}")
    print("Verdict:", "✅ TRUE GDS" if gds_ok else "⚠️ Compat signals detected — check config/driver")

    # explicit cleanup to avoid teardown crashes
    try:
        import cupy as _cp
        del data, readbuf
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass
    time.sleep(0.1)

if __name__ == "__main__":
    main()
    # python crashes at exit probably from mess with fork+threads in kvikio and GPU memory pools/cuda contexts?
    os._exit(0)
