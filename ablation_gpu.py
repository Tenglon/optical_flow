"""Ablation: eager vs torch.compile (fusion) vs reduce-overhead (CUDA Graphs)
for the batched torch optical-flow solvers on GPU.

Variants:
  eager       -- baseline
  compile     -- torch.compile(dynamic=False): Inductor kernel fusion
  compile-ro  -- torch.compile(mode="reduce-overhead"): fusion + CUDA Graphs
                 (whole iteration loop replayed with one launch per graph)

  srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
       uv run --no-sync python ablation_gpu.py
"""
import time

import numpy as np
import torch

from benchmark import DEEPFLOW_PARAMS, TVL1_PARAMS
from benchmark_data import load_pairs
from torch_deepflow import calc_flow_deepflow
from torch_flow import calc_flow_tvl1

# Pyramid levels have distinct shapes -> one dynamo/inductor graph per shape.
torch._dynamo.config.cache_size_limit = 512

HW = (240, 320)
BATCHES = [8, 64]
REPS = 3

BASE = {
    "tvl1": (calc_flow_tvl1, TVL1_PARAMS),
    "deepflow": (calc_flow_deepflow, DEEPFLOW_PARAMS),
}


def load_batch(B, dev):
    pairs = load_pairs(max_pairs=4, gray=True, size=HW)
    p = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs])
    n = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs])
    r = -(-B // len(pairs))
    return (p.repeat(r, 1, 1)[:B].to(dev), n.repeat(r, 1, 1)[:B].to(dev))


@torch.no_grad()
def bench(fn, p, n, warmups=2):
    t0 = time.perf_counter()
    for _ in range(warmups):        # first call(s): compile + cudagraph capture
        out = fn(p, n)
        torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn(p, n)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return warm, min(ts), out


def main():
    assert torch.cuda.is_available()
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"{'algo':<9} {'variant':<11} {'B':>4} {'warmup_s':>9} "
          f"{'ms/pair':>9} {'fps':>7} {'vs_eager':>8} {'maxdiff':>9}")

    for algo, (base_fn, params) in BASE.items():
        variants = {
            "eager": base_fn,
            "compile": torch.compile(base_fn, dynamic=False),
            "compile-ro": torch.compile(base_fn, mode="reduce-overhead",
                                        dynamic=False),
        }
        for B in BATCHES:
            p, n = load_batch(B, "cuda")
            eager_ms, eager_out = None, None
            for name, fn in variants.items():
                try:
                    f = lambda a, b: fn(a, b, **params)
                    warm, t, out = bench(f, p, n)
                    ms = t / B * 1e3
                    if name == "eager":
                        eager_ms, eager_out = ms, out
                        speed, diff = 1.0, 0.0
                    else:
                        speed = eager_ms / ms
                        diff = (out - eager_out).abs().max().item()
                    print(f"{algo:<9} {name:<11} {B:>4} {warm:>9.1f} "
                          f"{ms:>9.2f} {1e3 / ms:>7.1f} {speed:>7.2f}x "
                          f"{diff:>9.2e}", flush=True)
                except Exception as e:  # noqa: BLE001 -- record and continue
                    print(f"{algo:<9} {name:<11} {B:>4} FAILED "
                          f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                finally:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
