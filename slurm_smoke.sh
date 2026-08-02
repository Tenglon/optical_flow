#!/usr/bin/env bash
# GPU smoke test for the optic_flow project on the SLURM cluster.
# Verified working 2026-08-02 on ilps-cn119 (8x NVIDIA L40, driver 610.43.02, CUDA 13.3).
# Note: ilps-cn119 may be in power-save; allocation then sits in CONFIGURING for a few
# minutes while the node boots. Drop `-w ilps-cn119` to take any free GPU node, or use
# `--gres=gpu:nvidia_l40:1` / `--gres=gpu:nvidia_rtx_a6000:1` to pin a GPU type.
#
# Run from the project directory: ./slurm_smoke.sh

set -euo pipefail
cd "$(dirname "$0")"

srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=00:10:00 \
  "$HOME/.local/bin/uv" run --no-sync python - <<'EOF'
import time, torch
print("torch:", torch.__version__, "| wheel CUDA:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
dev = torch.device("cuda")
a = torch.randn(2000, 2000, device=dev); b = torch.randn(2000, 2000, device=dev)
for _ in range(3): c = a @ b
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10): c = a @ b
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 10
print(f"matmul 2000x2000 avg: {dt*1000:.3f} ms ({2*2000**3/dt/1e12:.2f} TFLOP/s)")
x = torch.randn(1, 3, 256, 256, device=dev); w = torch.randn(16, 3, 3, 3, device=dev)
print("conv2d ok:", tuple(torch.nn.functional.conv2d(x, w, padding=1).shape))
feat = torch.randn(1, 8, 64, 64, device=dev)
ys, xs = torch.meshgrid(torch.linspace(-1, 1, 32, device=dev),
                        torch.linspace(-1, 1, 32, device=dev), indexing="ij")
grid = torch.stack((xs, ys), -1).unsqueeze(0)
print("grid_sample ok:", tuple(torch.nn.functional.grid_sample(feat, grid, align_corners=True).shape))
EOF
