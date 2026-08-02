"""GPU batch-scaling test: ms/pair vs batch size for the torch implementations.

Replicates Middlebury pairs to batch sizes B and times each algo on CUDA.
  srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:45:00 \
       uv run --no-sync python scaling_gpu.py
"""
import time

import numpy as np
import torch

from benchmark_data import load_pairs
from benchmark import TVL1_PARAMS, DEEPFLOW_PARAMS
from torch_flow import calc_flow_tvl1
from torch_deepflow import calc_flow_deepflow

SIZES = [(240, 320), (480, 640)]
BATCHES = [1, 8, 32, 128]
REPS = 2

ALGOS = {
    "tvl1": lambda p, n: calc_flow_tvl1(p, n, **TVL1_PARAMS),
    "deepflow": lambda p, n: calc_flow_deepflow(p, n, **DEEPFLOW_PARAMS),
}


@torch.no_grad()
def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'algo':<9} {'res':>8} {'B':>4} {'total_s':>8} {'ms/pair':>9} {'fps':>7}")
    for hw in SIZES:
        pairs = load_pairs(max_pairs=4, gray=True, size=hw)
        base_p = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs])
        base_n = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs])
        for algo, fn in ALGOS.items():
            for B in BATCHES:
                reps = -(-B // len(pairs))                     # ceil
                p = base_p.repeat(reps, 1, 1)[:B].to(dev)
                n = base_n.repeat(reps, 1, 1)[:B].to(dev)
                try:
                    fn(p, n)                                   # warmup
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(REPS):
                        t0 = time.perf_counter()
                        fn(p, n)
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t0)
                    t = min(ts)
                    print(f"{algo:<9} {hw[0]}x{hw[1]:<4} {B:>4} {t:>8.2f} "
                          f"{t / B * 1e3:>9.2f} {B / t:>7.1f}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"{algo:<9} {hw[0]}x{hw[1]:<4} {B:>4}      OOM", flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
