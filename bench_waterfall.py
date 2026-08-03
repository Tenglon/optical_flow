"""Waterfall benchmark for the three OpenCV-CUDA-inspired TV-L1 optimizations.

Variants (all backend="compile", 240x320, Middlebury pairs):

  orig       git-HEAD torch_flow.py: unfold+torch.median, convergence error
             every iteration, fixed trip count      -- the 54.5 ms/pair baseline
  nocat      new file, median=torch.median, early_exit=False
             -> isolates the torch.cat removal in _tvl1_step
  median     + 5x5 min/max selection network for the flow median filter
  med+split  + calc_error only on throttled check iterations (no loop break)
  all        + real early exit (break once the whole batch has converged)

Usage (GPU node):
    srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
         --time=00:59:00 ./run_nvme.sh python bench_waterfall.py --batch 16 --util
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import torch

import torch_flow
import torch_flow_orig
from benchmark_data import load_pairs

HW = (240, 320)
NPAIRS = 8
REPS = 3
PARAMS = dict(tau=0.25, lambda_=0.15, theta=0.3, nscales=5, warps=5, epsilon=0.01,
              inner_iterations=30, outer_iterations=10, scale_step=0.8,
              gamma=0.0, median_filtering=5)

_MEDIAN_NET = torch_flow._median_blur  # network version, captured before patching


def log(*a):
    print(*a, flush=True)


class Util:
    """Sampled GPU utilization for *this* device (nvidia-smi subprocess)."""

    def __init__(self, enabled=True):
        self.exe = shutil.which("nvidia-smi") if enabled else None
        self.idx = torch.cuda.current_device()
        self.rows: list[int] = []
        self.stop = threading.Event()
        self.th = None

    def _loop(self):
        while not self.stop.wait(0.1):
            try:
                out = subprocess.run(
                    [self.exe, f"--id={self.idx}", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                if out:
                    self.rows.append(int(out.splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        if self.exe:
            self.th = threading.Thread(target=self._loop, daemon=True)
            self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        if self.th:
            self.th.join(timeout=3)

    def mean(self):
        return float(np.mean(self.rows)) if self.rows else None


def make_variant(name):
    """Configure torch_flow for a variant; return a solver fn (p, n) -> flow."""
    torch_flow._COMPILED_FNS.clear()
    torch_flow._BREAK_ON_CONVERGENCE = name != "med+split"
    torch_flow._SPARSE_CHECKS = name in ("med+split", "all+split")
    torch_flow._median_blur = (_MEDIAN_NET if name != "nocat" else torch_flow._median_blur_sort)
    if name == "orig":
        torch_flow._median_blur = torch_flow._median_blur_sort
        return lambda p, n: torch_flow_orig.calc_flow_tvl1(p, n, backend="compile", **PARAMS)
    early = name in ("med+split", "all", "all+split")
    return lambda p, n: torch_flow.calc_flow_tvl1(p, n, backend="compile", early_exit=early, **PARAMS)


def load_batch(B, dev):
    pairs = load_pairs(max_pairs=NPAIRS, gray=True, size=HW)
    p = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs])
    n = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs])
    r = -(-B // len(pairs))
    return p.repeat(r, 1, 1)[:B].to(dev), n.repeat(r, 1, 1)[:B].to(dev), pairs


def cv2_reference(pairs):
    alg = cv2.optflow.DualTVL1OpticalFlow_create()  # defaults == PARAMS
    return [alg.calc(a, b, None).transpose(2, 0, 1) for a, b, _ in pairs]


@torch.no_grad()
def bench(fn, p, n, util_on):
    t0 = time.perf_counter()
    out = fn(p, n)
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    ts = []
    with Util(util_on) as u:
        for _ in range(REPS):
            t0 = time.perf_counter()
            out = fn(p, n)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return warm, min(ts), u.mean(), out


@torch.no_grad()
def count_steps(name, p, n):
    """Number of _tvl1_step calls actually executed (early-exit effectiveness)."""
    fn = make_variant(name)
    orig_resolve = torch_flow._resolve_backend
    box = [0]

    def counting(backend, dev):
        s, w, m = orig_resolve(backend, dev)

        def step(*a):
            box[0] += 1
            return s(*a)
        return step, w, m

    torch_flow._resolve_backend = counting
    try:
        fn(p, n)
        torch.cuda.synchronize()
    finally:
        torch_flow._resolve_backend = orig_resolve
    return box[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--variants", default="orig,nocat,median,med+split,all,all+split")
    ap.add_argument("--util", action="store_true")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--out", default="waterfall.jsonl")
    args = ap.parse_args()
    B = args.batch

    assert torch.cuda.is_available()
    torch._dynamo.config.cache_size_limit = 1024
    log(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  B={B} {HW}")

    p, n, pairs = load_batch(B, "cuda")
    t0 = time.perf_counter()
    ref = cv2_reference(pairs)
    log(f"cv2 CPU reference on {len(pairs)} pairs in {time.perf_counter()-t0:.1f}s "
        f"(batch tiled to B={B})")
    log(f"{'variant':<11} {'warm_s':>7} {'ms/pair':>9} {'fps':>8} {'util%':>6} "
        f"{'peakMiB':>8} {'EPEvcv2':>8} {'maxdiff':>8} {'vs orig':>8}")

    base_ms = None
    for name in args.variants.split(","):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            fn = make_variant(name)
            warm, t, util, out = bench(fn, p, n, args.util)
            ms = t / B * 1e3
            peak = torch.cuda.max_memory_allocated() / 2**20
            f = out[:len(pairs)].cpu().numpy()
            e = float(np.mean([np.linalg.norm(f[i] - ref[i], axis=0).mean() for i in range(len(pairs))]))
            md = float(np.max([np.abs(f[i] - ref[i]).max() for i in range(len(pairs))]))
            nsteps = count_steps(name, p, n) if args.count else None
            if base_ms is None:
                base_ms = ms
            log(f"{name:<11} {warm:>7.1f} {ms:>9.2f} {1e3/ms:>8.1f} "
                f"{(f'{util:.0f}' if util else '-'):>6} {peak:>8.0f} {e:>8.4f} {md:>8.3f} "
                f"{base_ms/ms:>7.2f}x" + (f"  steps={nsteps}" if nsteps else ""))
            with open(args.out, "a") as fh:
                fh.write(json.dumps(dict(B=B, variant=name, ms_per_pair=ms, warm_s=warm,
                                         util=util, peak_mib=peak, epe_vs_cv2=e,
                                         maxdiff_vs_cv2=md, steps=nsteps)) + "\n")
        except Exception as exc:  # noqa: BLE001
            import traceback
            log(f"{name:<11} FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
