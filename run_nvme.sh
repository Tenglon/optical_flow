#!/usr/bin/env bash
# Stage project + venv + data onto the node-local NVMe (/nvmestore) and run
# a command there, keeping hot I/O (python imports, Inductor/Triton caches,
# intermediate files) off CephFS and the nearly-full home directory.
#
# Usage (inside an srun/sbatch allocation on a GPU node):
#   ./run_nvme.sh python ablation_gpu.py
#   srun -p gpu -w ilps-cn119 --gres=gpu:1 ... /fnwi_fs/.../run_nvme.sh python benchmark.py --backends torch_cuda
#
# First staging copies ~5 GB (venv); later runs rsync deltas only.
set -euo pipefail

SRC=/fnwi_fs/ivi/irlab/personal/tlong/code/optic_flow
DEST=/nvmestore/tlong/optic_flow
mkdir -p "$DEST" /nvmestore/tlong/cache

# Stage code + data + venv (delta-sync; venv first time is the big one).
rsync -a --delete --exclude='.git' --exclude='__pycache__' --exclude='slides' \
      "$SRC/" "$DEST/"

# All caches on NVMe: Inductor/Triton compile artifacts, XDG fallback.
export XDG_CACHE_HOME=/nvmestore/tlong/cache
export TORCHINDUCTOR_CACHE_DIR=/nvmestore/tlong/cache/torchinductor
export TRITON_CACHE_DIR=/nvmestore/tlong/cache/triton

# nvcc from the staged wheel for Inductor.
CUDIR="$DEST/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDIR/bin:$DEST/.venv/bin:$PATH"
export CUDA_HOME="$CUDIR"

cd "$DEST"
exec "$@"
