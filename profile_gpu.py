"""Profile the torch optical-flow solvers on GPU: kernel counts, launch
overhead, GPU-busy fraction, top kernels.

  srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 \
       uv run --no-sync python profile_gpu.py
"""
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from benchmark import DEEPFLOW_PARAMS, TVL1_PARAMS
from benchmark_data import load_pairs
from torch_deepflow import calc_flow_deepflow
from torch_flow import calc_flow_tvl1

ALGOS = {
    "tvl1": lambda p, n: calc_flow_tvl1(p, n, **TVL1_PARAMS),
    "deepflow": lambda p, n: calc_flow_deepflow(p, n, **DEEPFLOW_PARAMS),
}
HW = (240, 320)
BATCHES = [8, 64]


def load_batch(B, dev):
    pairs = load_pairs(max_pairs=4, gray=True, size=HW)
    p = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs])
    n = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs])
    r = -(-B // len(pairs))
    return (p.repeat(r, 1, 1)[:B].to(dev), n.repeat(r, 1, 1)[:B].to(dev))


@torch.no_grad()
def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    for algo, fn in ALGOS.items():
        for B in BATCHES:
            p, n = load_batch(B, dev)
            fn(p, n)  # warmup / lazy init
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            fn(p, n)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                fn(p, n)
                torch.cuda.synchronize()

            evts = prof.key_averages()
            kernels = [e for e in evts if e.device_type == torch.autograd.DeviceType.CUDA]
            n_launch = sum(e.count for e in kernels)
            gpu_busy = sum(e.self_device_time_total for e in kernels) / 1e6  # s
            print(f"\n=== {algo}  B={B}  {HW[0]}x{HW[1]} ===")
            print(f"wall: {wall:.3f}s  ms/pair: {wall / B * 1e3:.2f}")
            print(f"kernel launches: {n_launch}  "
                  f"gpu busy: {gpu_busy:.3f}s  busy/wall: {gpu_busy / wall * 100:.0f}%  "
                  f"mean kernel: {gpu_busy / max(n_launch, 1) * 1e6:.1f} us")
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=12, max_name_column_width=45))
            mem = torch.cuda.max_memory_allocated() / 2**20
            print(f"peak mem: {mem:.0f} MiB")
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
