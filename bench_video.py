"""End-to-end video throughput of the batched torch optical-flow solvers (L40).

What this measures, for 64 consecutive-frame pairs at 240x320 built from
Middlebury frames (rolled per timestep so the flow is a nontrivial, textured
translation field):

  a) GPU-resident : frames already float32 on cuda -> flow tensor on cuda.
     This is the "solver only" number and matches the usual batch benchmarks.
  b) End-to-end   : frames start as a pinned uint8 CPU tensor (what a decoder
     hands you), and the timed region covers H2D copy + uint8->float32 +
     flow + D2H copy of the flow back to CPU.  This is the number a video
     pipeline actually sees.
  c) chunk sweep  : chunk in {None, 16, 32} -- the flattened 64-pair batch is
     solved in micro-batches of that size, trading peak memory for throughput.

Two framings of the same 64 pairs are compared: (N=4, T=17) -- four short
clips -- and (T=65) -- one long clip.  The flattened pair batch is identical
(64 pairs), so any difference is the (N,T,H,W) reshape copy.

Timing: one warm-up call per (algo, backend, chunk) -- which also pays the
torch.compile cost, reported separately and excluded from the steady-state
numbers -- then >= 3 timed reps, min taken, cuda synchronised around each.
torch.cuda.empty_cache() + peak-memory reset between configs.

Usage (GPU node):
  srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
       --time=02:00:00 ./run_nvme.sh python bench_video.py
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import time

import numpy as np
import torch

# Inductor needs nvcc; on the compute nodes only the pip-wheel copy is usable.
_CUDIR = pathlib.Path(__file__).parent / ".venv/lib/python3.12/site-packages/nvidia/cu13"
if (_CUDIR / "bin/nvcc").exists():
    os.environ["PATH"] = f"{_CUDIR}/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", str(_CUDIR))

from benchmark import DEEPFLOW_PARAMS, TVL1_PARAMS  # noqa: E402
from benchmark_data import load_pairs  # noqa: E402
from torch_deepflow import calc_flow_deepflow, calc_flow_deepflow_video  # noqa: E402
from torch_flow import calc_flow_tvl1, calc_flow_tvl1_video  # noqa: E402

# Every (pyramid level shape x micro-batch size) specialises into its own
# graph; make sure dynamo never hits its recompile limit and falls back.
torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 2048)
if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
    torch._dynamo.config.accumulated_cache_size_limit = max(
        torch._dynamo.config.accumulated_cache_size_limit, 8192
    )

HW = (240, 320)
MIB = 1024.0 ** 2

ALGOS = {
    "tvl1": (calc_flow_tvl1, calc_flow_tvl1_video, TVL1_PARAMS),
    "deepflow": (calc_flow_deepflow, calc_flow_deepflow_video, DEEPFLOW_PARAMS),
}


# --------------------------------------------------------------------------- #
# Synthetic-but-textured sequences
# --------------------------------------------------------------------------- #
def make_video(n: int | None, t: int, hw=HW) -> torch.Tensor:
    """uint8 CPU frames: (n, t, H, W), or (t, H, W) when ``n is None``.

    Real Middlebury frames rolled by a per-timestep sub-linear + oscillating
    offset, so each consecutive pair carries a nontrivial (few-pixel, varying)
    translation on top of real image texture.
    """
    n_seq = 1 if n is None else n
    pairs = load_pairs(max_pairs=n_seq, gray=True, size=hw)
    if len(pairs) < n_seq:  # dataset smaller than requested -> cycle
        pairs = [pairs[i % len(pairs)] for i in range(n_seq)]
    seqs = []
    for i, (f1, _f2, _name) in enumerate(pairs[:n_seq]):
        base = torch.from_numpy(np.ascontiguousarray(f1))  # (H, W) uint8
        phase = 0.9 * i
        frames = []
        for k in range(t):
            dx = int(round(1.4 * k + 2.0 * math.sin(0.6 * k + phase)))
            dy = int(round(0.8 * k + 2.0 * math.cos(0.45 * k + phase)))
            frames.append(torch.roll(base, shifts=(dy, dx), dims=(0, 1)))
        seqs.append(torch.stack(frames))
    out = torch.stack(seqs)  # (n_seq, t, H, W)
    return out[0] if n is None else out


def n_pairs(frames: torch.Tensor) -> int:
    return frames.shape[0] * (frames.shape[1] - 1) if frames.dim() == 4 else frames.shape[0] - 1


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def time_min(fn, reps: int) -> float:
    """min wall time of ``reps`` synchronised calls."""
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def fresh():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Sanity: video path == pairwise path
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sanity(seed: int = 0, tol: float = 1e-4) -> bool:
    """video result == pairwise calc_flow_* on 3 random pairs (eager)."""
    rng = np.random.default_rng(seed)
    frames = make_video(None, 5).to("cuda").float()  # (5, H, W) -> 4 pairs
    ok = True
    for algo, (pair_fn, video_fn, params) in ALGOS.items():
        vid = video_fn(frames, chunk=None, backend="eager", **params)
        idx = rng.choice(frames.shape[0] - 1, size=3, replace=False)
        worst = 0.0
        for i in sorted(int(j) for j in idx):
            ref = pair_fn(frames[i : i + 1], frames[i + 1 : i + 2], backend="eager", **params)[0]
            worst = max(worst, (vid[i] - ref).abs().max().item())
        passed = worst <= tol
        ok &= passed
        print(f"  sanity {algo:<9} pairs={sorted(int(j) for j in idx)} "
              f"max|video-pairwise| = {worst:.3e}  {'OK' if passed else 'FAIL'}", flush=True)
        fresh()
    return ok


# --------------------------------------------------------------------------- #
# Transfer-only reference (how much of end-to-end is PCIe?)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def transfer_cost(frames_pin: torch.Tensor, reps: int) -> tuple[float, float]:
    """(H2D+float ms, D2H ms) for one 64-pair batch of frames / flows."""
    b = n_pairs(frames_pin)
    flow_gpu = torch.empty(b, 2, *HW, device="cuda", dtype=torch.float32)

    def h2d():
        frames_pin.to("cuda", non_blocking=True).float()

    def d2h():
        flow_gpu.to("cpu")

    out = (time_min(h2d, reps) * 1e3, time_min(d2h, reps) * 1e3)
    del flow_gpu
    fresh()
    return out


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    ap.add_argument("--chunks", nargs="+", default=["none", "16", "32"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-eager", action="store_true", help="skip the eager reference rows")
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()
    chunks = [None if c.lower() == "none" else int(c) for c in args.chunks]

    assert torch.cuda.is_available(), "needs a GPU node"
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"res {HW[0]}x{HW[1]}  reps {args.reps}\n", flush=True)

    if not args.no_sanity:
        print("sanity check (eager, video vs pairwise, tol 1e-4):", flush=True)
        ok = sanity()
        print(f"  -> {'ALL PASS' if ok else 'FAILURE'}\n", flush=True)

    cases = {"N4T17": make_video(4, 17), "T65": make_video(None, 65)}
    for name, f in cases.items():
        print(f"case {name:<6} frames {tuple(f.shape)} uint8 -> {n_pairs(f)} pairs", flush=True)

    # pinned CPU uint8 source + resident float32 cuda copy for each case
    src = {}
    for name, f in cases.items():
        pin = f.pin_memory()
        src[name] = (pin, pin.to("cuda").float())
    fresh()

    print("\ntransfer-only reference (pinned, per 64-pair batch):", flush=True)
    for name, (pin, _) in src.items():
        h2d_ms, d2h_ms = transfer_cost(pin, max(args.reps, 5))
        print(f"  {name:<6} H2D uint8+float {h2d_ms:7.2f} ms   "
              f"D2H flow (64,2,240,320) {d2h_ms:7.2f} ms   "
              f"total {h2d_ms + d2h_ms:7.2f} ms", flush=True)
    fresh()

    hdr = (f"\n{'algo':<9} {'case':<6} {'backend':<8} {'chunk':>5} "
           f"{'gpu_pair/s':>10} {'e2e_pair/s':>10} {'gpu_ms':>8} {'e2e_ms':>8} "
           f"{'peakRes':>8} {'peakE2E':>8} {'xfer%':>6}")
    print(hdr)
    print("-" * len(hdr.strip()), flush=True)

    warmups: dict[tuple, float] = {}
    rows: list[dict] = []

    for algo in args.algos:
        _pair_fn, video_fn, params = ALGOS[algo]
        backends = ["compile"] + ([] if args.no_eager else ["eager"])
        for backend in backends:
            # eager needs no compile sweep: one reference row only
            bk_chunks = chunks if backend == "compile" else [None]
            bk_cases = list(src) if backend == "compile" else ["N4T17"]
            for chunk in bk_chunks:
                for case in bk_cases:
                    pin, res = src[case]
                    b = n_pairs(pin)
                    kw = dict(chunk=chunk, backend=backend, **params)

                    def gpu_resident():
                        return video_fn(res, **kw)

                    def end_to_end():
                        g = pin.to("cuda", non_blocking=True).float()
                        return video_fn(g, **kw).to("cpu")

                    key = (algo, backend, chunk)
                    fresh()
                    t0 = time.perf_counter()
                    gpu_resident()          # warm-up (compiles on first hit)
                    torch.cuda.synchronize()
                    end_to_end()
                    torch.cuda.synchronize()
                    warm = time.perf_counter() - t0
                    warmups.setdefault(key, warm)

                    fresh()
                    t_gpu = time_min(gpu_resident, args.reps)
                    peak_res = torch.cuda.max_memory_allocated() / MIB
                    fresh()
                    t_e2e = time_min(end_to_end, args.reps)
                    peak_e2e = torch.cuda.max_memory_allocated() / MIB
                    fresh()

                    xfer = 100.0 * (t_e2e - t_gpu) / t_e2e
                    rows.append(dict(algo=algo, case=case, backend=backend,
                                     chunk=chunk, gpu=b / t_gpu, e2e=b / t_e2e,
                                     gpu_ms=t_gpu / b * 1e3, e2e_ms=t_e2e / b * 1e3,
                                     peak_res=peak_res, peak_e2e=peak_e2e, xfer=xfer))
                    print(f"{algo:<9} {case:<6} {backend:<8} "
                          f"{str(chunk):>5} {b / t_gpu:>10.1f} {b / t_e2e:>10.1f} "
                          f"{t_gpu / b * 1e3:>8.2f} {t_e2e / b * 1e3:>8.2f} "
                          f"{peak_res:>8.0f} {peak_e2e:>8.0f} {xfer:>5.1f}%",
                          flush=True)

    print("\nwarm-up (first call per algo/backend/chunk, incl. torch.compile; "
          "excluded above):", flush=True)
    for (algo, backend, chunk), w in warmups.items():
        print(f"  {algo:<9} {backend:<8} chunk={str(chunk):<5} {w:8.1f} s", flush=True)

    # ---- summary ---------------------------------------------------------- #
    print("\nrealistic single-L40 video throughput at 240x320 (compile backend, "
          "best chunk per algo):", flush=True)
    for algo in args.algos:
        cand = [r for r in rows if r["algo"] == algo and r["backend"] == "compile"]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["e2e"])
        eager = [r for r in rows if r["algo"] == algo and r["backend"] == "eager"]
        sp = f"  ({best['e2e'] / eager[0]['e2e']:.1f}x eager)" if eager else ""
        lo = min(cand, key=lambda r: r["peak_e2e"])
        print(f"  {algo:<9} best chunk={str(best['chunk']):<5} "
              f"{best['e2e']:6.1f} pairs/s end-to-end{sp}; "
              f"peak {best['peak_e2e']:.0f} MiB "
              f"(smallest footprint: chunk={str(lo['chunk'])} -> {lo['peak_e2e']:.0f} MiB "
              f"at {lo['e2e']:.1f} pairs/s)", flush=True)
    if rows:
        x = [r["xfer"] for r in rows if r["backend"] == "compile"]
        if x:
            print(f"  H2D+D2H + float conversion costs {min(x):.1f}-{max(x):.1f}% "
                  f"of the end-to-end time.", flush=True)


if __name__ == "__main__":
    main()
