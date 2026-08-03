"""OpenCV CPU reference timings per resolution (login-node script, no GPU).

Times `cv2.optflow.DualTVL1OpticalFlow` (benchmark.TVL1_PARAMS) and
`cv2.optflow.createOptFlow_DeepFlow` on real Middlebury pairs resized to each
benchmark resolution, giving the denominator for the GPU speedup table.

    ~/.local/bin/uv run --no-sync python res_cpu_ref.py --out res_cpu.jsonl

OpenCV barely parallelizes either solver (DeepFlow is single-threaded; TV-L1
parallelizes only a couple of inner loops), so the default thread count is left
alone and the numbers are effectively single-core.  Reported value is the min
over reps of the mean wall time per pair.
"""

from __future__ import annotations

import argparse
import json
import time

import cv2

from benchmark import TVL1_PARAMS
from benchmark_data import load_pairs

SIZES: dict[str, tuple[int, int]] = {
    "240x320": (240, 320),
    "480x640": (480, 640),
    "480x854": (480, 854),
    "720x1280": (720, 1280),
}


def make_engine(algo: str):
    if algo == "tvl1":
        f = cv2.optflow.DualTVL1OpticalFlow_create(
            tau=TVL1_PARAMS["tau"], lambda_=TVL1_PARAMS["lambda_"],
            theta=TVL1_PARAMS["theta"], nscales=TVL1_PARAMS["nscales"],
            warps=TVL1_PARAMS["warps"], epsilon=TVL1_PARAMS["epsilon"],
            innnerIterations=TVL1_PARAMS["inner_iterations"],  # sic: cv2 typo
            outerIterations=TVL1_PARAMS["outer_iterations"],
            scaleStep=TVL1_PARAMS["scale_step"], gamma=TVL1_PARAMS["gamma"])
    elif algo == "deepflow":
        f = cv2.optflow.createOptFlow_DeepFlow()
    else:
        raise ValueError(algo)
    return lambda a, b: f.calc(a, b, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["tvl1", "deepflow"],
                    choices=["tvl1", "deepflow"])
    ap.add_argument("--res", nargs="+", default=list(SIZES), choices=list(SIZES))
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=None,
                    help="cv2.setNumThreads(N); default = OpenCV's own choice")
    ap.add_argument("--out", default=None, help="append JSON lines here")
    args = ap.parse_args()

    if args.threads is not None:
        cv2.setNumThreads(args.threads)
    print(f"# opencv {cv2.__version__} threads={cv2.getNumThreads()} "
          f"cpus={cv2.getNumberOfCPUs()}", flush=True)
    print(f"{'algo':<9} {'res':>8} {'pairs':>5} {'reps':>4} {'ms/pair':>11} "
          f"{'rep_s_min':>10} {'spread':>7}", flush=True)

    for algo in args.algos:
        for res in args.res:
            hw = SIZES[res]
            pairs = load_pairs(max_pairs=args.pairs, gray=True, size=hw)
            calc = make_engine(algo)
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                for f1, f2, _ in pairs:
                    calc(f1, f2)
                ts.append(time.perf_counter() - t0)
            t = min(ts)
            rec = {"algo": algo, "res": res, "px_per_pair": hw[0] * hw[1],
                   "n_pairs": len(pairs), "reps": args.reps,
                   "rep_s_min": t, "ms_per_pair": t / len(pairs) * 1e3,
                   "spread": (max(ts) - t) / t,
                   "threads": cv2.getNumThreads(), "device": "cpu"}
            print(f"{algo:<9} {res:>8} {len(pairs):>5} {args.reps:>4} "
                  f"{rec['ms_per_pair']:>11.1f} {t:>10.2f} {rec['spread']:>7.3f}",
                  flush=True)
            print("RESULT " + json.dumps(rec), flush=True)
            if args.out:
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
