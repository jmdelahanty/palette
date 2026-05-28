# Environment Setup

`environment.yml` is the portable Palette environment spec. It is intended for
fresh installs on workstations and Janelia compute nodes. It is not an exact
machine lockfile.

## Create The Environment

From the Palette repository root:

```bash
conda env create -n palette-py311 -f environment.yml
$HOME/miniforge3/envs/palette-py311/bin/python -m pip install -e .
```

If your machine uses Miniconda instead of Miniforge, replace
`$HOME/miniforge3` with `$HOME/miniconda3`.

Palette commands should then go through the repository wrapper:

```bash
scripts/py -c 'import sys; print(sys.executable)'
scripts/py -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
```

`scripts/py` searches both `$HOME/miniconda3/envs/palette-py311/bin/python` and
`$HOME/miniforge3/envs/palette-py311/bin/python`. Set `PALETTE_PYTHON` only if
your environment lives somewhere else. The wrapper also prepends the selected
environment's `lib/` directory to `LD_LIBRARY_PATH` before launching Python, so
native wheels such as Decord can resolve conda-provided shared libraries without
requiring `conda activate`.

## Optional Local Decord Wheel

Some Palette video paths can use Decord. Palette no longer tracks the Decord
source checkout or wheel in git; keep those artifacts outside the repository and
install a compatible wheel after the editable package if needed:

```bash
$HOME/miniforge3/envs/palette-py311/bin/python -m pip install /path/to/decord-0.6.0-cp311-cp311-linux_x86_64.whl
```

Use the matching Miniconda path if applicable.

The wheel is linked against FFmpeg 4.x (`libavformat.so.58`), so
`environment.yml` pins `ffmpeg=4.4.*`. The environment uses pip
`opencv-python-headless` instead of conda `opencv` because conda OpenCV builds
pull newer FFmpeg packages that conflict with Decord's FFmpeg 4.x ABI.

If Decord fails with `OSError: libavformat.so.58: cannot open shared object
file`, the simplest repair is to rebuild from the updated environment file:

```bash
git pull --ff-only
conda env remove -n palette-py311 -y
conda env create -n palette-py311 -f environment.yml
$HOME/miniforge3/envs/palette-py311/bin/python -m pip install -e .
$HOME/miniforge3/envs/palette-py311/bin/python -m pip install /path/to/decord-0.6.0-cp311-cp311-linux_x86_64.whl
scripts/py -c 'import cv2, decord; print("cv2/decord ok")'
```

To repair an existing environment in place instead:

```bash
git pull --ff-only
conda remove -n palette-py311 -y opencv py-opencv libopencv
conda install -n palette-py311 -c conda-forge 'numpy>=2,<2.3' 'ffmpeg=4.4.*'
$HOME/miniforge3/envs/palette-py311/bin/python -m pip install opencv-python-headless
scripts/py -c 'import cv2, decord; print("cv2/decord ok")'
```

## Failed Or Partial Environments

If `conda env create` failed partway through, remove the partial environment
before retrying:

```bash
conda env remove -n palette-py311 -y
conda env create -n palette-py311 -f environment.yml
```

## Exact Snapshots

`conda-packages-explicit.txt` and `pip-packages-exact.txt` are exact snapshots
from a known machine. They are useful for debugging provenance or recreating
that exact machine class, but they are not the default install path.

Do not use `pip-packages-exact.txt` as a general install input. It may contain
CUDA-specific wheel tags such as `torch==...+cu121` that require a custom PyTorch
wheel index and may conflict with the conda CUDA stack.

## Optional TensorRT / CUDA Tooling

The base environment intentionally avoids installing TensorRT, CuPy, or PyCUDA.
Those packages are useful for engine-building and low-level CUDA experiments,
but they are also more platform-sensitive. Install them only for workflows that
need them, and prefer documenting the exact command used in the run provenance.

## Janelia Cluster Notes

Create and validate the environment from an interactive compute node, not a
login-only workflow, when testing CUDA availability:

```bash
bsub -n 8 -gpu "num=1" -q gpu_l4 -W 2:00 -Is /bin/bash
umask 002
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export TBB_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENMP_NUM_THREADS=$LSB_DJOB_NUMPROC
```

Then run the create/validation commands above.

### Cluster Validation Script

After the environment is created, PyTorch CUDA is fixed to the `pytorch`
channel build, and Decord has been built on the cluster, run:

```bash
scripts/validate_cluster_palette_env.sh
```

For PyNvVideoCodec backend parity or default-promotion work, require the PyNv
stack explicitly:

```bash
scripts/validate_cluster_palette_env.sh --require-pynvvc
```

If a smoke MP4 is available, also verify Decord GPU decode:

```bash
scripts/validate_cluster_palette_env.sh --video /path/to/example.mp4
```

The script checks that:

- `scripts/py` resolves the `palette-py311` interpreter.
- PyTorch sees the assigned CUDA device.
- NumPy remains in the Palette-supported range (`<2.3`).
- Decord imports and `libdecord.so` resolves FFmpeg libraries from the selected
  conda environment.
- PyNvVideoCodec availability and NVIDIA video-driver libraries are reported;
  with `--require-pynvvc`, missing PyNvVideoCodec, `libnvcuvid`, or
  `libnvidia-encode` fails validation.
- Decord is linked against `libnvcuvid` for NVDEC-capable GPU decode.
- Core packages such as OpenCV, Ultralytics, Zarr, PyArrow, and Polars import.

This script is non-mutating. It validates the environment but does not install
packages or rebuild Decord.
