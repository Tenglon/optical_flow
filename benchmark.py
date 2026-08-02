"""Benchmark dense optical flow backends: OpenCV CPU vs CUDA vs PyTorch.

Algorithms (--algo):
  tvl1       -- Dual TV-L1 (Zach et al. 2007), default
  deepflow   -- DeepFlow variational part (Weinzaepfel et al. 2013)
  farneback  -- Farneback polynomial expansion

Backends per algorithm:
  opencv_cpu   -- cv2.optflow.* / cv2.calcOpticalFlow* on CPU
  opencv_cuda  -- cv2.cuda.* (only if cv2 built with CUDA; pip wheels are NOT;
                  exists for tvl1/farneback, not for deepflow)
  torch_cpu    -- batched PyTorch reimplementation on CPU
  torch_cuda   -- batched PyTorch reimplementation on GPU

Usage:
  uv run python benchmark.py                        # tvl1, all available backends
  uv run python benchmark.py --algo deepflow
  uv run python benchmark.py --size 240 320 --backends opencv_cpu torch_cpu

On the login node (no GPU) only CPU backends run; submit to a GPU node with:
  srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=16G \
       ~/.local/bin/uv run --no-sync python benchmark.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

# Parameters shared by every backend of a given algorithm (OpenCV defaults).
TVL1_PARAMS = dict(tau=0.25, lambda_=0.15, theta=0.3, nscales=5, warps=5,
                   epsilon=0.01, inner_iterations=30, outer_iterations=10,
                   scale_step=0.8, gamma=0.0)
DEEPFLOW_PARAMS = dict(sigma=0.6, min_size=25, downscale_factor=0.95,
                       fixed_point_iterations=5, sor_iterations=25,
                       alpha=1.0, delta=0.5, gamma=5.0, omega=1.6)
FARNEBACK_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                        poly_n=5, poly_sigma=1.2)


def epe(flow_a: np.ndarray, flow_b: np.ndarray) -> float:
    """Mean endpoint error between two (H, W, 2) flows.

    Pixels where either flow has magnitude > 1e9 (Middlebury "unknown"
    marker) are excluded; NaNs are excluded the same way.
    """
    valid = (np.abs(flow_a).max(axis=-1) < 1e9) & (np.abs(flow_b).max(axis=-1) < 1e9)
    if not valid.any():
        return float("nan")
    return float(np.linalg.norm(flow_a - flow_b, axis=-1)[valid].mean())


def time_fn(fn, warmup: int, reps: int, sync=None) -> list[float]:
    """Run fn() warmup+reps times, return per-rep wall times (s) for the reps."""
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    return times


# --------------------------------------------------------------------------- #
# OpenCV backends. Each returns (flows: list[(H,W,2) ndarray], times: list[s]).
# --------------------------------------------------------------------------- #

def _cv_cpu_engine(algo):
    if algo == "tvl1":
        f = cv2.optflow.DualTVL1OpticalFlow_create(
            tau=TVL1_PARAMS["tau"], lambda_=TVL1_PARAMS["lambda_"],
            theta=TVL1_PARAMS["theta"], nscales=TVL1_PARAMS["nscales"],
            warps=TVL1_PARAMS["warps"], epsilon=TVL1_PARAMS["epsilon"],
            innnerIterations=TVL1_PARAMS["inner_iterations"],
            outerIterations=TVL1_PARAMS["outer_iterations"],
            scaleStep=TVL1_PARAMS["scale_step"], gamma=TVL1_PARAMS["gamma"])
        return lambda a, b: f.calc(a, b, None)
    if algo == "deepflow":
        f = cv2.optflow.createOptFlow_DeepFlow()
        return lambda a, b: f.calc(a, b, None)
    if algo == "farneback":
        return lambda a, b: cv2.calcOpticalFlowFarneback(
            a, b, None, flags=0, **FARNEBACK_PARAMS)
    raise ValueError(algo)


def run_opencv_cpu(algo, pairs, warmup, reps):
    calc = _cv_cpu_engine(algo)

    def once():
        return [calc(f1, f2) for f1, f2, _ in pairs]
    times = time_fn(once, warmup, reps)
    return once(), times


def run_opencv_cuda(algo, pairs, warmup, reps):
    if algo == "tvl1":
        f = cv2.cuda_OpticalFlowDual_TVL1.create(
            tau=TVL1_PARAMS["tau"], lambda_=TVL1_PARAMS["lambda_"],
            theta=TVL1_PARAMS["theta"], nscales=TVL1_PARAMS["nscales"],
            warps=TVL1_PARAMS["warps"], epsilon=TVL1_PARAMS["epsilon"],
            iterations=TVL1_PARAMS["outer_iterations"]
                       * TVL1_PARAMS["inner_iterations"],
            scaleStep=TVL1_PARAMS["scale_step"], gamma=TVL1_PARAMS["gamma"])
    elif algo == "farneback":
        f = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=FARNEBACK_PARAMS["levels"],
            pyrScale=FARNEBACK_PARAMS["pyr_scale"], fastPyramids=False,
            winSize=FARNEBACK_PARAMS["winsize"],
            numIters=FARNEBACK_PARAMS["iterations"],
            polyN=FARNEBACK_PARAMS["poly_n"],
            polySigma=FARNEBACK_PARAMS["poly_sigma"], flags=0)
    else:
        raise RuntimeError(f"no OpenCV CUDA implementation for {algo!r}")

    gpu_pairs = []
    for f1, f2, _ in pairs:
        g1, g2 = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
        g1.upload(f1)
        g2.upload(f2)
        gpu_pairs.append((g1, g2))

    def once():
        return [f.calc(g1, g2, None) for g1, g2 in gpu_pairs]
    times = time_fn(once, warmup, reps)
    return [o.download() for o in once()], times


# --------------------------------------------------------------------------- #
# PyTorch backends.
# --------------------------------------------------------------------------- #

def _torch_calc(algo):
    if algo == "tvl1":
        from torch_flow import calc_flow_tvl1
        return lambda p, n: calc_flow_tvl1(p, n, **TVL1_PARAMS)
    if algo == "deepflow":
        from torch_deepflow import calc_flow_deepflow
        return lambda p, n: calc_flow_deepflow(p, n, **DEEPFLOW_PARAMS)
    if algo == "farneback":
        from torch_flow import calc_flow_farneback
        return lambda p, n: calc_flow_farneback(p, n, **FARNEBACK_PARAMS)
    raise ValueError(algo)


def run_torch(algo, pairs, warmup, reps, device, batch=None):
    import torch

    calc = _torch_calc(algo)
    prev = torch.stack([torch.from_numpy(f1.astype(np.float32)) for f1, _, _ in pairs])
    nxt = torch.stack([torch.from_numpy(f2.astype(np.float32)) for _, f2, _ in pairs])
    prev, nxt = prev.to(device), nxt.to(device)
    if batch is None or batch >= len(pairs):
        batches = [(prev, nxt)]
    else:
        batches = [(prev[i:i + batch], nxt[i:i + batch])
                   for i in range(0, len(pairs), batch)]
    sync = torch.cuda.synchronize if device.startswith("cuda") else None

    @torch.no_grad()
    def once():
        return [calc(p, n) for p, n in batches]
    times = time_fn(once, warmup, reps, sync=sync)
    flows = torch.cat(once(), dim=0)                 # (B, 2, H, W)
    flows = flows.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 2)
    return list(flows), times


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", default="tvl1",
                    choices=["tvl1", "deepflow", "farneback"])
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=None)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None,
                    help="torch micro-batch size (default: all pairs in one batch)")
    ap.add_argument("--backends", nargs="+", default=None,
                    choices=["opencv_cpu", "opencv_cuda", "torch_cpu", "torch_cuda"])
    ap.add_argument("--opencv-threads", type=int, default=None,
                    help="cv2.setNumThreads for the CPU backend")
    args = ap.parse_args()

    from benchmark_data import load_pairs, load_flow_gt
    pairs = load_pairs(max_pairs=args.max_pairs, gray=True,
                       size=tuple(args.size) if args.size else None)
    n = len(pairs)
    h, w = pairs[0][0].shape[:2]
    params = {"tvl1": TVL1_PARAMS, "deepflow": DEEPFLOW_PARAMS,
              "farneback": FARNEBACK_PARAMS}[args.algo]
    print(f"algo: {args.algo}  pairs: {n}  resolution: {h}x{w}")
    print(f"params: {params}")
    print(f"reps: {args.reps} (+{args.warmup} warmup)")

    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    print(f"opencv threads: {cv2.getNumThreads()}")

    backends = args.backends
    if backends is None:
        backends = ["opencv_cpu", "torch_cpu"]
        if (args.algo != "deepflow" and hasattr(cv2, "cuda")
                and cv2.cuda.getCudaEnabledDeviceCount() > 0):
            backends.append("opencv_cuda")
        try:
            import torch
            if torch.cuda.is_available():
                backends.append("torch_cuda")
        except ImportError:
            pass
    print(f"backends: {backends}\n")

    def resized_gt(name, hw):
        g = load_flow_gt(name)
        if g is None or g.shape[:2] == hw:
            return g
        # Nearest-neighbour keeps the >1e9 "unknown" markers intact; scale
        # the flow vectors to the new pixel grid.
        sy, sx = hw[0] / g.shape[0], hw[1] / g.shape[1]
        g = cv2.resize(g, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        return g * np.array([sx, sy], dtype=g.dtype)

    gts = [resized_gt(name, f1.shape[:2]) for f1, _, name in pairs]
    results, ref_flows = {}, None
    for be in backends:
        try:
            if be == "opencv_cpu":
                flows, times = run_opencv_cpu(args.algo, pairs, args.warmup, args.reps)
                ref_flows = flows
            elif be == "opencv_cuda":
                flows, times = run_opencv_cuda(args.algo, pairs, args.warmup, args.reps)
            elif be == "torch_cpu":
                flows, times = run_torch(args.algo, pairs, args.warmup,
                                         args.reps, "cpu", args.batch)
            elif be == "torch_cuda":
                flows, times = run_torch(args.algo, pairs, args.warmup,
                                         args.reps, "cuda", args.batch)
        except Exception as e:  # noqa: BLE001 -- report and keep benchmarking
            print(f"[{be}] FAILED: {type(e).__name__}: {e}")
            continue
        results[be] = (flows, times)

    if not results:
        return
    if ref_flows is None:
        ref_flows = next(iter(results.values()))[0]

    print(f"{'backend':<12} {'ms/pair':>9} {'fps':>8} {'stdev':>7} "
          f"{'EPE vs cv2':>11} {'EPE vs GT':>10}")
    for be, (flows, times) in results.items():
        ms = statistics.mean(times) / n * 1e3
        sd = (statistics.stdev(times) / n * 1e3) if len(times) > 1 else 0.0
        e_ref = statistics.mean(epe(f, r) for f, r in zip(flows, ref_flows))
        gt_pairs = [(f, g) for f, g in zip(flows, gts) if g is not None]
        e_gt = (statistics.mean(epe(f, g) for f, g in gt_pairs)
                if gt_pairs else float("nan"))
        print(f"{be:<12} {ms:>9.2f} {1e3 / ms:>8.1f} {sd:>7.2f} "
              f"{e_ref:>11.3f} {e_gt:>10.3f}")


if __name__ == "__main__":
    main()
