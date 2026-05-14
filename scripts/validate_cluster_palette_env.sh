#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/validate_cluster_palette_env.sh [--video PATH]

Validate a Palette cluster environment from the repository root.

Checks:
  - scripts/py resolves the palette-py311 interpreter
  - PyTorch sees CUDA
  - NumPy stays within the Palette-supported range
  - Decord imports and libdecord.so links against the selected conda env FFmpeg
  - libnvcuvid is linked for NVDEC-capable Decord builds
  - core Python dependencies import
  - optional Decord GPU VideoReader smoke with --video

Examples:
  scripts/validate_cluster_palette_env.sh
  scripts/validate_cluster_palette_env.sh --video /groups/.../example.mp4
EOF
}

video_path=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      if [[ $# -lt 2 ]]; then
        echo "--video requires a path" >&2
        exit 2
      fi
      video_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
py="$repo_root/scripts/py"

if [[ ! -x "$py" ]]; then
  echo "Missing executable wrapper: $py" >&2
  exit 1
fi

echo "== Palette Cluster Environment Validation =="
echo "repo=$repo_root"
echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== nvidia-smi =="
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  echo
else
  echo "warning: nvidia-smi not found in PATH"
  echo
fi

echo "== Python / CUDA / imports =="
"$py" - <<'PY'
import importlib
import pathlib
import subprocess
import sys

print(f"python={sys.executable}")
print(f"prefix={sys.prefix}")

import numpy as np

version_parts = tuple(int(part) for part in np.__version__.split(".")[:2])
print(f"numpy={np.__version__} ({np.__file__})")
if version_parts >= (2, 3):
    raise SystemExit("NumPy must remain <2.3 for the current Palette environment.")

import torch

print(
    "torch="
    f"{torch.__version__} cuda={torch.version.cuda} "
    f"available={torch.cuda.is_available()}"
)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is not available.")
print(f"torch_gpu={torch.cuda.get_device_name(0)}")

for name in ["cv2", "ultralytics", "zarr", "pyarrow", "polars"]:
    importlib.import_module(name)
print("core_imports=ok")

import decord

print(f"decord={decord.__file__}")
lib_path = pathlib.Path(decord.__file__).parent / "libdecord.so"
if not lib_path.exists():
    raise SystemExit(f"Missing libdecord.so next to decord package: {lib_path}")
print(f"libdecord={lib_path}")

ldd = subprocess.run(
    ["ldd", str(lib_path)],
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(ldd.stdout)
if ldd.returncode != 0:
    raise SystemExit(f"ldd failed for {lib_path}")

env_lib = str(pathlib.Path(sys.prefix) / "lib")
if "libavformat" not in ldd.stdout:
    raise SystemExit("libdecord.so is not linked against libavformat.")
if env_lib not in ldd.stdout:
    raise SystemExit(
        "libdecord.so does not appear to resolve FFmpeg libraries from the active conda env."
    )
if "libnvcuvid" not in ldd.stdout:
    raise SystemExit(
        "libdecord.so is not linked against libnvcuvid; rebuild Decord with USE_CUDA=ON for NVDEC."
    )
print("decord_linkage=ok")
PY
echo

if [[ -n "$video_path" ]]; then
  if [[ ! -f "$video_path" ]]; then
    echo "Video does not exist: $video_path" >&2
    exit 1
  fi
  echo "== Decord GPU video smoke =="
  "$py" - "$video_path" <<'PY'
import sys

import decord
from decord import VideoReader, gpu

video = sys.argv[1]
decord.bridge.set_bridge("torch")
vr = VideoReader(video, ctx=gpu(0))
n = len(vr)
if n <= 0:
    raise SystemExit(f"VideoReader opened but reported no frames: {video}")
indices = list(range(min(4, n)))
batch = vr.get_batch(indices)
device = getattr(batch, "device", None)
print(f"video={video}")
print(f"frames={n}")
print(f"batch_shape={tuple(batch.shape)} device={device} dtype={batch.dtype}")
if device is None or getattr(device, "type", None) != "cuda":
    raise SystemExit("Decord GPU smoke did not return a CUDA torch tensor.")
print("decord_gpu_video=ok")
PY
else
  echo "Skipping Decord GPU video smoke; pass --video /path/to/file.mp4 to run it."
fi

echo
echo "cluster_palette_env=ok"
