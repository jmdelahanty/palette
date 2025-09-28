# save as src/cupy_kvikio_bench.py
import os, time
import cupy as cp
import kvikio

# Fail fast if fallback would happen
os.environ.setdefault("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
os.environ["KVIKIO_COMPAT_MODE"] = "OFF"

PATH = "/nvme1/test/kvikio_bench.bin"

# tune these:
TOTAL_SIZE = 9 * 1024**3      # GiB total transfer
CHUNK_SIZE = 8 * 1024**2      # 8 MiB I/O size (aligned)
QDEPTH     = 8                # number of outstanding async I/Os

assert TOTAL_SIZE % CHUNK_SIZE == 0
n_chunks = TOTAL_SIZE // CHUNK_SIZE

# Generate a recognisable pattern on GPU
buf = cp.arange(CHUNK_SIZE, dtype=cp.uint8)
chk = int(buf[:100000].sum())

def bench_write():
    futures = []
    t0 = time.perf_counter()
    with kvikio.CuFile(PATH, "w") as f:
        for i in range(n_chunks):
            # queue up to QDEPTH ops, then drain
            futures.append(f.pwrite(buf, file_offset=i*CHUNK_SIZE))
            if len(futures) >= QDEPTH:
                # wait for them
                for fu in futures:
                    fu.get() if hasattr(fu, "get") else fu.result()
                futures.clear()
        # drain any remaining
        for fu in futures:
            fu.get() if hasattr(fu, "get") else fu.result()
    cp.cuda.runtime.deviceSynchronize()
    dt = time.perf_counter() - t0
    gbps = TOTAL_SIZE / dt / (1024**3)
    print(f"WRITE {TOTAL_SIZE//(1024**2)} MiB in {dt:.3f}s -> {gbps:.2f} GiB/s")

def bench_read_and_verify():
    out = cp.empty_like(buf)
    futures = []
    t0 = time.perf_counter()
    with kvikio.CuFile(PATH, "r") as f:
        for i in range(n_chunks):
            futures.append(f.pread(out, file_offset=i*CHUNK_SIZE))
            if len(futures) >= QDEPTH:
                for fu in futures:
                    fu.get() if hasattr(fu, "get") else fu.result()
                futures.clear()
        for fu in futures:
            fu.get() if hasattr(fu, "get") else fu.result()
    cp.cuda.runtime.deviceSynchronize()
    dt = time.perf_counter() - t0
    gbps = TOTAL_SIZE / dt / (1024**3)
    print(f"READ  {TOTAL_SIZE//(1024**2)} MiB in {dt:.3f}s -> {gbps:.2f} GiB/s")

    # quick content sanity: re-read first chunk and compare checksum
    with kvikio.CuFile(PATH, "r") as f:
        (f.pread(out, file_offset=0).get()
         if hasattr(f.pread(out,0), "get") else f.pread(out,0).result())
    ok = bool(int(out[:100000].sum()) == chk)
    print("Sanity checksum match:", ok)

if __name__ == "__main__":
    print("CuPy:", cp.__version__, "KvikIO:", getattr(kvikio, "__version__", "unknown"))
    print(f"TOTAL={TOTAL_SIZE//(1024**2)}MiB CHUNK={CHUNK_SIZE//(1024**2)}MiB QDEPTH={QDEPTH}")
    bench_write()
    bench_read_and_verify()


# CURRENT RUN OUTPUT:
# (palette-py311) delahantyj@delahantyj-ws1:~/gitrepos/palette$ export CUFILE_ENV_PATH_JSON=/etc/cufile.json
# export KVIKIO_COMPAT_MODE=OFF
# python src/cupy_kvikio_bench.py
# CuPy: 13.6.0 KvikIO: 25.08.00
# TOTAL=1024MiB CHUNK=8MiB QDEPTH=8
# WRITE 1024 MiB in 0.459s -> 2.18 GiB/s
# READ  1024 MiB in 0.330s -> 3.03 GiB/s
# Sanity checksum match: True

# (palette-py311) delahantyj@delahantyj-ws1:~/gitrepos/palette$ sudo cat /etc/cufile.json
# {
#   "version": 2,
#   "properties": {
#     "use_compat_mode": false,
#     "force_compat_mode": false,
#     "foce_odirect_mode": true,
#     "prefer_iouring": false
#   },
#   "profile": {
#     "cufile_stats": 1,
#     "nvtkx": false
#   },
#   "filesystems": {
#     "ext4": { "allow_odirect": true },
#     "xfs":  { "allow_odirect": true }
#   },
#   "mountPoints": [
#     { "path": "/nvme1", "fs": "ext4", "allow_odirect": true }
#   ]
# }
# (palette-py311) delahantyj@delahantyj-ws1:~/gitrepos/palette$ /usr/libexec/gds/tools/gdscheck -p | grep -E 'use_compat_mode|force_compat_mode|force_odirect_mode'
#  properties.use_compat_mode : true
#  properties.force_compat_mode : false
#  properties.force_odirect_mode : false
# (palette-py311) delahantyj@delahantyj-ws1:~/gitrepos/palette$ gdsio -f /nvme1/gdsio.bin -d 0 -w 4 -i 4M -s 4G -I 1 -x 5 -V
# IoType: WRITE XferType: GPUD_ASYNC Threads: 4 DataSetSize: 4182016/4194304(KiB) IOSize: 4096(KiB) Throughput: 5.383836 GiB/sec, Avg_Latency: 2850.102865 usecs ops: 1021 total_time 0.740788 secs
# Verifying data 
# IoType: READ XferType: GPUD_ASYNC Threads: 4 DataSetSize: 4182016/4194304(KiB) IOSize: 4096(KiB) Throughput: 6.543025 GiB/sec, Avg_Latency: 2379.658460 usecs ops: 1021 total_time 0.609547 secs
