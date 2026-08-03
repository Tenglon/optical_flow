"""OpenCV-CUDA optical-flow baseline (cv2.cuda_OpticalFlowDual_TVL1 / Farneback).

Standalone: imports only cv2 + numpy from the CUDA-enabled OpenCV env, never
torch and never the project's own modules (so it can run under a different
interpreter than the project .venv).

    /fnwi_fs/ivi/irlab/personal/tlong/conda_envs/opencv-cuda/bin/python \
        cv2_gpu_baseline.py --out cv2_gpu.jsonl

Three timing modes per (algo, resolution):
  naive      upload -> calc -> download for every pair (what a user writes first)
  resident   frames pre-uploaded as GpuMat, calc only (pure kernel throughput)
  streams    N cv2.cuda.Stream in round-robin over pre-uploaded frames

TV-L1 params match the project's CPU/torch reference: iterations=300 is the
CUDA solver's single loop count standing in for 10 outer x 30 inner.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PAIRS_DIR = os.path.join(DATA_DIR, "other-data")

SIZES: dict[str, tuple[int, int]] = {
    "240x320": (240, 320),
    "480x640": (480, 640),
}

# Matches the project's CPU/torch TV-L1 configuration.
TVL1 = dict(tau=0.25, lambda_=0.15, theta=0.3, nscales=5, warps=5,
            epsilon=0.01, iterations=300, scaleStep=0.8, gamma=0.0)
# CPU contrib equivalent: 300 = outerIterations(10) x innnerIterations(30).
TVL1_CPU_OUTER, TVL1_CPU_INNER = 10, 30

FARNEBACK = dict(numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
                 numIters=3, polyN=5, polySigma=1.2, flags=0)


# --------------------------------------------------------------------------- data
def load_pairs(max_pairs: int, size: tuple[int, int]):
    """(frame1, frame2, name) uint8 grayscale arrays resized to `size` (H, W)."""
    if not os.path.isdir(PAIRS_DIR):
        raise FileNotFoundError(f"{PAIRS_DIR} not found")
    h, w = size
    out = []
    for name in sorted(os.listdir(PAIRS_DIR)):
        seq = os.path.join(PAIRS_DIR, name)
        p1, p2 = os.path.join(seq, "frame10.png"), os.path.join(seq, "frame11.png")
        if not (os.path.isfile(p1) and os.path.isfile(p2)):
            continue
        f1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        f2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        if f1 is None or f2 is None:
            continue
        f1 = cv2.resize(f1, (w, h), interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(f2, (w, h), interpolation=cv2.INTER_AREA)
        out.append((f1, f2, name))
        if len(out) >= max_pairs:
            break
    return out


# ------------------------------------------------------------------- gpu sampling
def visible_gpu_index() -> str:
    """Physical GPU index to query.

    SLURM sets CUDA_VISIBLE_DEVICES to the *physical* index of the allocated
    GPU, while nvidia-smi --id addresses physical indices too -- so sampling
    index 0 would report a co-tenant's GPU on a shared node.
    """
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return v if v else "0"


class GpuSampler:
    """Background `nvidia-smi -lms` sampler over the measured region."""

    def __init__(self, index: str | None = None, period_ms: int = 100):
        self.index = visible_gpu_index() if index is None else str(index)
        self.period_ms = period_ms
        self.samples: list[int] = []
        self._proc = None
        self._thread = None

    def _read(self):
        for line in self._proc.stdout:
            line = line.strip()
            if line.isdigit():
                self.samples.append(int(line))

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--id={self.index}",
                 "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                 f"-lms={self.period_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()
        except FileNotFoundError:
            self._proc = None
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        return False

    def stats(self):
        s = self.samples
        if not s:
            return {"util_mean": None, "util_max": None, "util_n": 0}
        return {"util_mean": float(np.mean(s)), "util_max": int(max(s)),
                "util_n": len(s), "util_gpu_index": self.index}


# ------------------------------------------------------------------------ engines
def _cuda_class(*names):
    """First of `names` that exists as cv2.<n> or cv2.cuda.<n>.

    OpenCV exposes CUDA classes both as `cv2.cuda_Foo` and `cv2.cuda.Foo`
    depending on version; accept either so the script is not version-locked.
    """
    for n in names:
        if hasattr(cv2, n):
            return getattr(cv2, n)
        short = n[len("cuda_"):] if n.startswith("cuda_") else n
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, short):
            return getattr(cv2.cuda, short)
    raise AttributeError(f"none of {names} found in this cv2 build")


def gpu_mat(arr=None):
    cls = _cuda_class("cuda_GpuMat")
    g = cls()
    if arr is not None:
        g.upload(arr)
    return g


def make_gpu(algo: str):
    if algo == "tvl1":
        return _cuda_class("cuda_OpticalFlowDual_TVL1").create(**TVL1)
    if algo == "farneback":
        return _cuda_class("cuda_FarnebackOpticalFlow").create(**FARNEBACK)
    raise ValueError(algo)


def make_cpu_tvl1():
    """CPU contrib TV-L1 with the same configuration, or None if unavailable."""
    if not hasattr(cv2, "optflow"):
        return None
    return cv2.optflow.DualTVL1OpticalFlow_create(
        tau=TVL1["tau"], lambda_=TVL1["lambda_"], theta=TVL1["theta"],
        nscales=TVL1["nscales"], warps=TVL1["warps"], epsilon=TVL1["epsilon"],
        innnerIterations=TVL1_CPU_INNER,  # sic: upstream spelling
        outerIterations=TVL1_CPU_OUTER,
        scaleStep=TVL1["scaleStep"], gamma=TVL1["gamma"])


def sync():
    if hasattr(cv2.cuda, "Stream_Null"):
        cv2.cuda.Stream_Null().waitForCompletion()
    else:
        cv2.cuda.Stream.Null().waitForCompletion()


# ------------------------------------------------------------------------- timing
def _time_reps(one_rep, reps, util_seconds):
    """Warm up, time `reps` passes, then sample GPU util over a sustained window.

    The timed passes are far too short (tens of ms) for a 100 ms nvidia-smi
    poll to see anything, so utilisation is measured over a separate window
    that repeats the same work for at least `util_seconds`.
    """
    one_rep()  # warm-up
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        one_rep()
        sync()
        ts.append(time.perf_counter() - t0)

    util = {"util_mean": None, "util_max": None, "util_n": 0}
    if util_seconds > 0:
        with GpuSampler() as smp:
            t0 = time.perf_counter()
            n = 0
            while time.perf_counter() - t0 < util_seconds:
                one_rep()
                sync()
                n += 1
            util = smp.stats()
            util["util_window_s"] = time.perf_counter() - t0
            util["util_window_reps"] = n
    return ts, util


def bench_naive(flow, pairs, reps, util_seconds):
    """upload -> calc -> download per pair."""
    def one_rep():
        for f1, f2, _ in pairs:
            g1, g2 = gpu_mat(f1), gpu_mat(f2)
            res = flow.calc(g1, g2, None)
            res.download()
    return _time_reps(one_rep, reps, util_seconds)


def bench_resident(flow, gpu_pairs, reps, util_seconds):
    """calc only, frames already on device; result stays on device."""
    def one_rep():
        for g1, g2 in gpu_pairs:
            flow.calc(g1, g2, None)
    return _time_reps(one_rep, reps, util_seconds)


def bench_streams(algo, gpu_pairs, reps, n_streams, util_seconds):
    """One flow object + one Stream per lane, round-robin over pairs.

    Each engine keeps its own scratch buffers, so distinct objects are required
    for the lanes to actually overlap.
    """
    engines = [make_gpu(algo) for _ in range(n_streams)]
    streams = [cv2.cuda.Stream() for _ in range(n_streams)]
    dsts = [gpu_mat() for _ in range(n_streams)]

    def one_rep():
        for i, (g1, g2) in enumerate(gpu_pairs):
            k = i % n_streams
            engines[k].calc(g1, g2, dsts[k], stream=streams[k])
        for s in streams:
            s.waitForCompletion()

    return _time_reps(one_rep, reps, util_seconds)


# --------------------------------------------------------------------------- main
def build_cuda_lines():
    info = cv2.getBuildInformation()
    keep = []
    for line in info.splitlines():
        low = line.lower()
        if ("cuda" in low or "nvidia" in low or "cudnn" in low
                or "cufft" in low or "cublas" in low):
            keep.append(line.rstrip())
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["tvl1", "farneback"],
                    choices=["tvl1", "farneback"])
    ap.add_argument("--res", nargs="+", default=list(SIZES), choices=list(SIZES))
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--util-seconds", type=float, default=4.0,
                    help="sustained window for nvidia-smi util sampling (0=off)")
    ap.add_argument("--accuracy", action="store_true",
                    help="also compare CUDA TV-L1 vs CPU contrib TV-L1 (slow)")
    ap.add_argument("--acc-pairs", type=int, default=4)
    ap.add_argument("--out", default=None, help="append JSON lines here")
    args = ap.parse_args()

    n_dev = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"# opencv {cv2.__version__}  cuda devices={n_dev}", flush=True)
    if n_dev < 1:
        print("ERROR: no CUDA-enabled device visible to OpenCV", file=sys.stderr)
        sys.exit(1)
    cv2.cuda.printShortCudaDeviceInfo(cv2.cuda.getDevice())
    for line in build_cuda_lines():
        print("# " + line, flush=True)

    print(f"\n{'algo':<10} {'method':<9} {'res':>8} {'pairs':>5} {'ms/pair':>9} "
          f"{'rep_s_min':>10} {'spread':>7} {'util%':>7} {'utilmax':>8}",
          flush=True)

    for algo in args.algos:
        for res in args.res:
            hw = SIZES[res]
            pairs = load_pairs(args.pairs, hw)
            gpu_pairs = [(gpu_mat(f1), gpu_mat(f2)) for f1, f2, _ in pairs]
            flow = make_gpu(algo)

            us = args.util_seconds
            runs = [
                ("naive", lambda: bench_naive(flow, pairs, args.reps, us)),
                ("resident", lambda: bench_resident(flow, gpu_pairs, args.reps, us)),
                ("streams", lambda: bench_streams(algo, gpu_pairs, args.reps,
                                                  args.streams, us)),
            ]
            for method, fn in runs:
                ts, util = fn()
                t = min(ts)
                rec = {"algo": algo, "method": method, "res": res,
                       "px_per_pair": hw[0] * hw[1], "n_pairs": len(pairs),
                       "reps": args.reps, "rep_s_min": t,
                       "ms_per_pair": t / len(pairs) * 1e3,
                       "spread": (max(ts) - t) / t, "device": "cuda",
                       "opencv": cv2.__version__}
                if method == "streams":
                    rec["n_streams"] = args.streams
                rec.update(util)
                um = "-" if util["util_mean"] is None else f"{util['util_mean']:.0f}"
                ux = "-" if util["util_max"] is None else str(util["util_max"])
                print(f"{algo:<10} {method:<9} {res:>8} {len(pairs):>5} "
                      f"{rec['ms_per_pair']:>9.2f} {t:>10.3f} "
                      f"{rec['spread']:>7.3f} {um:>7} {ux:>8}", flush=True)
                print("RESULT " + json.dumps(rec), flush=True)
                if args.out:
                    with open(args.out, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")

    if args.accuracy:
        accuracy(args)


def accuracy(args) -> None:
    print("\n# accuracy: CUDA TV-L1 vs CPU contrib TV-L1 (mean EPE, px)",
          flush=True)
    cpu = make_cpu_tvl1()
    if cpu is None:
        print("# SKIPPED: cv2.optflow (contrib) not present in this build",
              flush=True)
        return
    gpu = make_gpu("tvl1")
    for res in args.res:
        hw = SIZES[res]
        for f1, f2, name in load_pairs(args.acc_pairs, hw):
            fg = gpu.calc(gpu_mat(f1), gpu_mat(f2), None).download()
            fc = cpu.calc(f1, f2, None)
            epe = float(np.mean(np.linalg.norm(fg - fc, axis=2)))
            mag = float(np.mean(np.linalg.norm(fc, axis=2)))
            rec = {"kind": "accuracy", "res": res, "seq": name,
                   "mean_epe_gpu_vs_cpu": epe, "mean_cpu_flow_mag": mag}
            print(f"{name:<12} {res:>8}  EPE={epe:7.4f}  "
                  f"|flow_cpu|={mag:7.4f}", flush=True)
            print("RESULT " + json.dumps(rec), flush=True)
            if args.out:
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
