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

## Local Decord Wheel

Some Palette video paths use the local Decord wheel stored in this repository.
Install it after the editable package if needed:

```bash
$HOME/miniforge3/envs/palette-py311/bin/python -m pip install ./decord-0.6.0-cp311-cp311-linux_x86_64.whl
```

Use the matching Miniconda path if applicable.

The wheel is linked against FFmpeg 4.x (`libavformat.so.58`), so
`environment.yml` pins `ffmpeg=4.4.*`. If Decord fails with
`OSError: libavformat.so.58: cannot open shared object file`, repair an
existing environment with:

```bash
conda install -n palette-py311 -c conda-forge 'ffmpeg=4.4.*'
git pull --ff-only
scripts/py -c 'import decord; print("decord ok")'
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
