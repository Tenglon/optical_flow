"""Post-optimization benchmark at higher resolutions.

TV-L1 (compile + early exit, fp32 and fp16) and DeepFlow (compile, fp32 and
fp16) at 480x640, 480x854, 720x1280 with the per-resolution best batch sizes.
"""
import time

import numpy as np
import torch

from benchmark_data import load_pairs
from torch_deepflow import calc_flow_deepflow
from torch_flow import calc_flow_tvl1

CASES = [
    # (algo, res, B list)
    ("tvl1", (480, 640), [4, 8, 16]),
    ("tvl1", (480, 854), [4, 8]),
    ("tvl1", (720, 1280), [2, 4, 8]),
    ("deepflow", (480, 640), [16]),
    ("deepflow", (480, 854), [16]),
    ("deepflow", (720, 1280), [8]),
]


def batch(hw, B, dt):
    pairs = load_pairs(max_pairs=8, gray=True, size=hw)
    p = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs])
    n = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs])
    r = -(-B // len(pairs))
    return (p.repeat(r, 1, 1)[:B].to("cuda", dt), n.repeat(r, 1, 1)[:B].to("cuda", dt))


@torch.no_grad()
def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'algo':<9} {'res':>9} {'dtype':<5} {'B':>3} {'ms/pair':>9} {'fps':>7} {'peakMiB':>8}")
    for algo, hw, Bs in CASES:
        fn = calc_flow_tvl1 if algo == "tvl1" else calc_flow_deepflow
        for dt in (torch.float32, torch.float16):
            for B in Bs:
                try:
                    p, n = batch(hw, B, dt)
                    torch.cuda.reset_peak_memory_stats()
                    fn(p, n, backend="compile")
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        out = fn(p, n, backend="compile")
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t0)
                    ok = torch.isfinite(out).all().item()
                    mem = torch.cuda.max_memory_allocated() / 2**20
                    t = min(ts)
                    tag = "fp32" if dt == torch.float32 else "fp16"
                    print(f"{algo:<9} {hw[0]}x{hw[1]:<5} {tag:<5} {B:>3} "
                          f"{t / B * 1e3:>9.2f} {B / t:>7.1f} {mem:>8.0f}"
                          + ("" if ok else "  NONFINITE!"), flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"{algo:<9} {hw[0]}x{hw[1]:<5} {dt} {B:>3}  OOM", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
