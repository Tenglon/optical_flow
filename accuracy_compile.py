"""Does backend="compile" / "cudagraphs" cost any accuracy vs backend="eager"?

Compares the three execution backends of the batched PyTorch TV-L1 and
DeepFlow ports against (a) Middlebury ground truth and (b) the OpenCV CPU
reference implementation, on the same 8 image pairs at a fixed resolution.

The worry: TV-L1's data term uses a hard three-way threshold, so a tiny
float reassociation introduced by Inductor's fusion can flip the branch at a
handful of pixels near motion discontinuities.  This script quantifies both
the aggregate effect (mean EPE) and the tail (per-pixel diff percentiles and
the fraction of pixels above 0.5 / 1 px).

Usage (GPU node):
    srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
         --time=00:59:00 ./run_nvme.sh python accuracy_compile.py
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import cv2
import numpy as np
import torch

from benchmark import DEEPFLOW_PARAMS, TVL1_PARAMS  # read-only import
from benchmark_data import load_flow_gt, load_pairs

BACKENDS = ["eager", "compile", "cudagraphs"]


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def epe(flow_a: np.ndarray, flow_b: np.ndarray) -> float:
    """Mean endpoint error over pixels valid in both flows (benchmark.epe)."""
    valid = (np.abs(flow_a).max(axis=-1) < 1e9) & (np.abs(flow_b).max(axis=-1) < 1e9)
    if not valid.any():
        return float("nan")
    return float(np.linalg.norm(flow_a - flow_b, axis=-1)[valid].mean())


def mean_epe(flows: list[np.ndarray], refs: list[np.ndarray | None]) -> float:
    vals = [epe(f, r) for f, r in zip(flows, refs) if r is not None]
    return float(np.mean(vals)) if vals else float("nan")


def diff_stats(flows_a: list[np.ndarray], flows_b: list[np.ndarray]) -> dict:
    """Per-pixel endpoint difference statistics between two flow lists."""
    d = np.concatenate([np.linalg.norm(a - b, axis=-1).ravel()
                        for a, b in zip(flows_a, flows_b)])
    comp = max(float(np.abs(a - b).max()) for a, b in zip(flows_a, flows_b))
    return {
        "max": float(d.max()),
        "max_comp": comp,
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "p999": float(np.percentile(d, 99.9)),
        "frac_gt_0p5": float((d > 0.5).mean()),
        "frac_gt_1": float((d > 1.0).mean()),
        "n": int(d.size),
    }


# --------------------------------------------------------------------------- #
# flow computation
# --------------------------------------------------------------------------- #

def cv2_flows(algo: str, pairs) -> list[np.ndarray]:
    if algo == "tvl1":
        f = cv2.optflow.DualTVL1OpticalFlow_create(
            tau=TVL1_PARAMS["tau"], lambda_=TVL1_PARAMS["lambda_"],
            theta=TVL1_PARAMS["theta"], nscales=TVL1_PARAMS["nscales"],
            warps=TVL1_PARAMS["warps"], epsilon=TVL1_PARAMS["epsilon"],
            innnerIterations=TVL1_PARAMS["inner_iterations"],   # sic: 3 n's
            outerIterations=TVL1_PARAMS["outer_iterations"],
            scaleStep=TVL1_PARAMS["scale_step"], gamma=TVL1_PARAMS["gamma"])
    elif algo == "deepflow":
        f = cv2.optflow.createOptFlow_DeepFlow()
    else:
        raise ValueError(algo)
    return [f.calc(a, b, None) for a, b, _ in pairs]


def torch_flows(algo: str, prev: torch.Tensor, nxt: torch.Tensor,
                backend: str) -> list[np.ndarray]:
    with torch.no_grad():
        if algo == "tvl1":
            from torch_flow import calc_flow_tvl1
            out = calc_flow_tvl1(prev, nxt, backend=backend, **TVL1_PARAMS)
        elif algo == "deepflow":
            from torch_deepflow import calc_flow_deepflow
            out = calc_flow_deepflow(prev, nxt, backend=backend, **DEEPFLOW_PARAMS)
        else:
            raise ValueError(algo)
    torch.cuda.synchronize()
    arr = out.permute(0, 2, 3, 1).float().cpu().numpy()  # (B, H, W, 2)
    return [np.ascontiguousarray(a) for a in arr]


# --------------------------------------------------------------------------- #

def resized_gt(name: str, hw: tuple[int, int]) -> np.ndarray | None:
    g = load_flow_gt(name)
    if g is None or g.shape[:2] == hw:
        return g
    sy, sx = hw[0] / g.shape[0], hw[1] / g.shape[1]
    g = cv2.resize(g, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return g * np.array([sx, sy], dtype=g.dtype)


def run_algo(algo: str, pairs, gts, device: str) -> None:
    log(f"\n{'=' * 78}\n{algo.upper()}\n{'=' * 78}")

    t0 = time.perf_counter()
    ref = cv2_flows(algo, pairs)
    log(f"cv2 CPU reference: {time.perf_counter() - t0:.1f}s")

    prev = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs]).to(device)
    nxt = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs]).to(device)

    flows: dict[str, list[np.ndarray]] = {}
    for be in BACKENDS:
        t0 = time.perf_counter()
        try:
            flows[be] = torch_flows(algo, prev, nxt, be)
        except Exception as e:  # noqa: BLE001 - report and continue
            log(f"[{algo}/{be}] FAILED: {type(e).__name__}: "
                f"{str(e).splitlines()[0] if str(e) else ''}")
            traceback.print_exc(limit=3, file=sys.stdout)
            continue
        log(f"[{algo}/{be}] ok ({time.perf_counter() - t0:.1f}s incl. compile)")

    if "eager" not in flows:
        log(f"{algo}: eager failed, nothing to compare")
        return

    # ---- accuracy table -------------------------------------------------- #
    n_gt = sum(g is not None for g in gts)
    log(f"\n-- accuracy ({len(pairs)} pairs, {n_gt} with GT) --")
    log(f"{'backend':<12} {'mean EPE vs GT':>15} {'mean EPE vs cv2':>16}")
    e_gt, e_cv = {}, {}
    log(f"{'cv2 (CPU)':<12} {mean_epe(ref, gts):>15.4f} {0.0:>16.4f}")
    for be in BACKENDS:
        if be not in flows:
            continue
        e_gt[be] = mean_epe(flows[be], gts)
        e_cv[be] = mean_epe(flows[be], ref)
        log(f"{'torch/' + be:<12} {e_gt[be]:>15.4f} {e_cv[be]:>16.4f}")

    # ---- flow diffs vs eager --------------------------------------------- #
    log("\n-- per-pixel flow diff (endpoint) vs backend='eager' --")
    log(f"{'comparison':<18} {'max':>10} {'max|comp|':>10} {'mean':>11} {'p99':>11} "
        f"{'p99.9':>11} {'%>0.5px':>10} {'%>1px':>10}")

    def diff_row(label, a, b):
        s = diff_stats(a, b)
        log(f"{label:<18} {s['max']:>10.4f} {s['max_comp']:>10.4f} {s['mean']:>11.3e} "
            f"{s['p99']:>11.3e} {s['p999']:>11.3e} "
            f"{100 * s['frac_gt_0p5']:>10.5f} {100 * s['frac_gt_1']:>10.5f}"
            f"   ({int(round(s['frac_gt_0p5'] * s['n']))}/{s['n']} px > 0.5)")
        return s

    stats = {}
    for be in BACKENDS:
        if be == "eager" or be not in flows:
            continue
        stats[be] = diff_row(be + " vs eager", flows[be], flows["eager"])
    # Baseline: how far eager already sits from the OpenCV CPU reference.  Any
    # compile-vs-eager tail much smaller than this is inside the existing
    # port-vs-OpenCV noise floor.
    base = diff_row("eager vs cv2", flows["eager"], ref)
    if "compile" in flows:
        diff_row("compile vs cv2", flows["compile"], ref)

    # ---- verdict ---------------------------------------------------------- #
    log("")
    for be in BACKENDS:
        if be == "eager" or be not in flows:
            continue
        d_gt = abs(e_gt[be] - e_gt["eager"])
        d_cv = abs(e_cv[be] - e_cv["eager"])
        s = stats[be]
        ok_epe = d_gt <= 0.01 and d_cv <= 0.01
        ok_tail = s["frac_gt_0p5"] < 1e-3
        log(f"VERDICT {algo}/{be}: dEPE(GT)={d_gt:.5f} px, dEPE(cv2)={d_cv:.5f} px "
            f"-> {'no change (<=0.01)' if ok_epe else 'CHANGED (>0.01)'}; "
            f"{100 * s['frac_gt_0p5']:.5f}% of pixels differ >0.5 px "
            f"-> {'confined to <0.1%' if ok_tail else 'NOT confined (>=0.1%)'} "
            f"[baseline eager-vs-cv2: {100 * base['frac_gt_0p5']:.5f}%]; "
            f"{'PASS' if (ok_epe and ok_tail) else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algos", nargs="+", default=["tvl1", "deepflow"],
                    choices=["tvl1", "deepflow"])
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=(240, 320))
    ap.add_argument("--max-pairs", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this script needs a GPU")
    log(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
        f"cv2 {cv2.__version__}")

    pairs = load_pairs(max_pairs=args.max_pairs, gray=True, size=tuple(args.size))
    hw = pairs[0][0].shape[:2]
    gts = [resized_gt(name, hw) for _, _, name in pairs]
    log(f"pairs ({len(pairs)}) at {hw[0]}x{hw[1]}: "
        + ", ".join(f"{n}{'' if g is not None else '(no GT)'}"
                    for (_, _, n), g in zip(pairs, gts)))
    log(f"TVL1_PARAMS:     {TVL1_PARAMS}")
    log(f"DEEPFLOW_PARAMS: {DEEPFLOW_PARAMS}")

    for algo in args.algos:
        run_algo(algo, pairs, gts, "cuda")


if __name__ == "__main__":
    main()
