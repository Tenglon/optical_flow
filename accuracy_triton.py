"""Correctness gates for backend="triton" (hand-fused TV-L1 kernels).

Why the gates are layered.  At default parameters TV-L1 is *chaotic* in the last
float bit: the data term's three-way threshold flips at a handful of pixels near
motion discontinuities, and 7 500 iterations amplify that into O(10 px)
disagreements on a few pixels.  This is not specific to Triton -- the existing
backend="compile" shows the same spread against backend="eager"
(see accuracy_compile.py).  So an absolute "max < 0.1 px" gate on the *full*
solve is unreachable for any backend, and the useful gates are:

  1. unit      -- one iteration, random state: the two kernels vs the eager
                  `_tvl1_step`.  Pure arithmetic, no amplification.
                  gate: max|diff| < 1e-5 on u / px / py, exact `active`.
  2. short     -- a deliberately short solve (2 scales, 1 warp, 5 iterations,
                  no median, epsilon=0) so nothing is amplified and no freeze
                  mask or early exit is in play.  gate: max < 1e-3 px.
  3. static    -- the full 7 500-iteration solve with epsilon=0: identical
                  trip counts, no freeze mask, no convergence reduction, so the
                  *only* difference left is float reassociation.  Reported
                  against the compile-vs-eager spread as the yardstick.
  4. default   -- the full solve at cv2 defaults (median 5, epsilon=0.01,
                  early exit).  gate: triton-vs-compile within 1.5x the
                  compile-vs-eager spread, i.e. inside the existing noise floor.
  5. cv2       -- mean EPE against the OpenCV CPU reference for all three
                  backends.  gate: |EPE_triton - EPE_compile| < 0.01 px.  This
                  is the gate that actually says "no accuracy was lost".
  6. batch     -- B=1 vs B=16 (same 8 pairs, tiled) bitwise identical.
  7. fp16      -- optional (--fp16): the half-precision path, triton vs compile.
                  gate: no NaN/Inf.  (The kernels always compute in fp32 and only
                  the traffic is fp16, so this is not a numerics gate.)

Every backend's output is additionally checked for NaN/Inf throughout.

Usage (GPU node):
    srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
         --time=00:59:00 ./run_nvme.sh python accuracy_triton.py
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np
import torch

import torch_flow
from benchmark_data import load_pairs

HW = (240, 320)
NPAIRS = 8
PARAMS = dict(tau=0.25, lambda_=0.15, theta=0.3, nscales=5, warps=5, epsilon=0.01,
              inner_iterations=30, outer_iterations=10, scale_step=0.8,
              gamma=0.0, median_filtering=5)
# short, unamplified solve -- the true arithmetic gate
SHORT = dict(PARAMS, nscales=2, warps=1, inner_iterations=5, outer_iterations=1,
             median_filtering=1, epsilon=0.0)
# full solve, no freeze mask / no early exit: identical trip counts everywhere
STATIC = dict(PARAMS, epsilon=0.0)

SHORT_GATE = 1e-3     # px, max componentwise
UNIT_GATE = 1e-5
NOISE_FACTOR = 1.5    # triton-vs-compile must stay within this x compile-vs-eager
EPE_GATE = 0.01       # px


def log(*a):
    print(*a, flush=True)


def load_batch(B: int, dev, hw=HW):
    pairs = load_pairs(max_pairs=NPAIRS, gray=True, size=hw)
    p = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs])
    n = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs])
    r = -(-B // len(pairs))
    return p.repeat(r, 1, 1)[:B].to(dev), n.repeat(r, 1, 1)[:B].to(dev), pairs


def flow_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """(mean endpoint diff, max endpoint diff, max componentwise diff) in px."""
    d = (a - b).float()
    ep = d.pow(2).sum(1).sqrt()
    return float(ep.mean()), float(ep.max()), float(d.abs().max())


# --------------------------------------------------------------------------- #
# 1. per-kernel unit test against the eager step
# --------------------------------------------------------------------------- #
@torch.no_grad()
def unit_kernels(dev, dtype=torch.float32, b=3, h=37, w=53, seed=0) -> bool:
    """Kernel-vs-eager on random state, one iteration, calc_error on and off."""
    import triton_tvl1

    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g).to(device=dev, dtype=dtype)

    l_t, theta, taut, scaled_eps = 0.045, 0.3, 0.25 / 0.3, 1e-4 * h * w
    ok = True
    for calc_error in (True, False):
        for n_frozen in (0, 1):
            u = rnd(b, 2, h, w)
            px, py = rnd(b, 2, h, w) * 0.1, rnd(b, 2, h, w) * 0.1
            i1w = rnd(b, 2, h, w) * 5.0
            rho_c = rnd(b, 1, h, w) * 5.0
            grad = i1w[:, 0:1] ** 2 + i1w[:, 1:2] ** 2
            # force some pixels into the |grad| <= eps branch
            grad = torch.where(rnd(b, 1, h, w) > 1.5, torch.zeros_like(grad), grad)
            pos = grad > torch_flow._FLT_EPS
            safe_grad = torch.where(pos, grad, torch.ones_like(grad))
            thr = l_t * grad
            active = torch.ones(b, 1, 1, 1, dtype=torch.bool, device=dev)
            if n_frozen:
                active[0] = False

            ref = torch_flow._tvl1_step(
                u.clone(), None, px.clone(), py.clone(), None, None, active.clone(),
                rho_c, i1w, thr, pos, safe_grad,
                l_t, theta, taut, 0.0, scaled_eps, calc_error)
            step = triton_tvl1.TritonTVL1Step(autotune=False)
            got = step(u.clone(), None, px.clone(), py.clone(), None, None, active.clone(),
                       rho_c, i1w, thr, pos, safe_grad,
                       l_t, theta, taut, 0.0, scaled_eps, calc_error)

            tag = f"calc_error={int(calc_error)} frozen={n_frozen}"
            for name, i in (("u ", 0), ("px", 2), ("py", 3)):
                d = float((ref[i] - got[i]).abs().max())
                good = d < UNIT_GATE
                ok &= good
                log(f"  [{tag}] max|{name}_ref - {name}_triton| = {d:.2e} "
                    f"{'ok' if good else 'FAIL'}")
            bad_act = bool((ref[6] != got[6]).any())
            ok &= not bad_act
            log(f"  [{tag}] active exact: {not bad_act}")
            if calc_error:
                rel = float((ref[7] - got[7].max()).abs()) / max(float(ref[7]), 1e-30)
                good = rel < UNIT_GATE
                ok &= good
                log(f"  [{tag}] err_max rel diff = {rel:.2e} {'ok' if good else 'FAIL'}")
    return ok


@torch.no_grad()
def solve(backend: str, p, n, params, early_exit=True, fresh=True):
    if fresh:
        torch_flow._COMPILED_FNS.pop("triton", None)
    t0 = time.perf_counter()
    out = torch_flow.calc_flow_tvl1(p, n, backend=backend, early_exit=early_exit, **params)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def triple(p, n, params, label, early_exit=True):
    """Solve with eager/compile/triton; print + return the pairwise spreads."""
    outs = {}
    for be in ("eager", "compile", "triton"):
        outs[be], t = solve(be, p, n, params, early_exit=early_exit)
        log(f"  [{label}/{be}] {t:.1f}s  non-finite={int((~torch.isfinite(outs[be])).sum())}"
            f"  range [{float(outs[be].min()):.3f}, {float(outs[be].max()):.3f}]")
    log(f"  {'pair':<18} {'mean px':>10} {'max px':>10} {'max comp':>10}")
    d = {}
    for a, b in (("compile", "eager"), ("triton", "eager"), ("triton", "compile")):
        d[(a, b)] = flow_diff(outs[a], outs[b])
        log(f"  {a + ' vs ' + b:<18} {d[(a, b)][0]:>10.3e} {d[(a, b)][1]:>10.3e} "
            f"{d[(a, b)][2]:>10.3e}")
    return outs, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip-cv2", action="store_true")
    ap.add_argument("--skip-static", action="store_true",
                    help="skip the epsilon=0 full solve (it is the slow one)")
    ap.add_argument("--fp16", action="store_true",
                    help="also compare the fp16 path (no NaN, triton vs compile)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log("no CUDA -- this script needs the GPU node")
        return 2
    log(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    import triton
    log(f"triton {triton.__version__}")
    torch._dynamo.config.cache_size_limit = 4096

    ok = True
    B = args.batch

    # ---------------- 1. unit ----------------
    log("\n== [1] per-kernel unit test vs eager _tvl1_step ==")
    try:
        ok &= unit_kernels("cuda")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        log("UNIT TEST RAISED -- Triton blocker, see traceback above")
        return 1

    p, n, pairs = load_batch(B, "cuda")

    # ---------------- 2. short solve (the arithmetic gate) ----------------
    log(f"\n== [2] short solve (2 scales, 1 warp, 5 iters, no median, eps=0), B={B} ==")
    _, d = triple(p, n, SHORT, "short", early_exit=False)
    g = d[("triton", "compile")][2] < SHORT_GATE
    ok &= g
    log(f"  GATE short: max comp {d[('triton', 'compile')][2]:.2e} < {SHORT_GATE} "
        f"-> {'PASS' if g else 'FAIL'}")

    # ---------------- 3. static full solve (eps=0: no freeze, no early exit) ----
    if not args.skip_static:
        log(f"\n== [3] full solve, epsilon=0 (identical 7500 trip counts), B={B} ==")
        _, ds = triple(p, n, STATIC, "static", early_exit=False)
        r = ds[("triton", "compile")][0] / max(ds[("compile", "eager")][0], 1e-30)
        log(f"  triton-vs-compile mean is {r:.2f}x the compile-vs-eager mean "
            "(pure float reassociation, amplified by 7500 threshold branches)")

    # ---------------- 4. default full solve ----------------
    log(f"\n== [4] full solve at cv2 defaults, {len(pairs)} pairs tiled to B={B} "
        f"@{HW[0]}x{HW[1]} ==")
    outs, dd = triple(p, n, PARAMS, "default")
    for be, f in outs.items():
        ok &= bool(torch.isfinite(f).all())
    m_tc = dd[("triton", "compile")][0]
    m_ce = dd[("compile", "eager")][0]
    g = m_tc <= NOISE_FACTOR * m_ce
    ok &= g
    log(f"  GATE noise floor: triton-vs-compile {m_tc:.3e} <= {NOISE_FACTOR}x "
        f"compile-vs-eager {m_ce:.3e} ({NOISE_FACTOR * m_ce:.3e}) -> {'PASS' if g else 'FAIL'}")
    log("  (the brief's absolute 1e-3 mean / 0.1 px max gate is not reachable by "
        "any backend here:\n   compile-vs-eager alone is "
        f"{m_ce:.3e} mean / {dd[('compile', 'eager')][1]:.2f} px max)")

    # ---------------- 5. cv2 reference ----------------
    if not args.skip_cv2:
        log("\n== [5] mean EPE vs the OpenCV CPU reference ==")
        import cv2
        t0 = time.perf_counter()
        alg = cv2.optflow.DualTVL1OpticalFlow_create()
        ref = torch.from_numpy(
            np.stack([alg.calc(a, b, None).transpose(2, 0, 1) for a, b, _ in pairs])).cuda()
        log(f"  cv2 CPU on {len(pairs)} pairs: {time.perf_counter()-t0:.1f}s")
        epes = {}
        for be, f in outs.items():
            m, x, _ = flow_diff(f[:len(pairs)], ref)
            epes[be] = m
            log(f"  [{be}] mean EPE = {m:.4f} px, max = {x:.3f} px")
        g = abs(epes["triton"] - epes["compile"]) < EPE_GATE
        ok &= g
        log(f"  GATE accuracy: |EPE_triton - EPE_compile| = "
            f"{abs(epes['triton'] - epes['compile']):.4f} < {EPE_GATE} -> "
            f"{'PASS' if g else 'FAIL'}")

    # ---------------- 6. batch consistency ----------------
    log(f"\n== [6] batch consistency (triton, B=1 vs B={B}) ==")
    torch_flow._COMPILED_FNS.pop("triton", None)
    f_b = torch_flow.calc_flow_tvl1(p, n, backend="triton", **PARAMS)
    per = torch.cat([torch_flow.calc_flow_tvl1(p[i:i + 1], n[i:i + 1], backend="triton", **PARAMS)
                     for i in range(B)])
    torch.cuda.synchronize()
    bd = float((f_b - per).abs().max())
    bitwise = torch.equal(f_b, per)
    ok &= bitwise
    log(f"  max|B={B} - B=1| = {bd:.3e}   bitwise identical: {bitwise} "
        f"-> {'PASS' if bitwise else 'FAIL'}")

    step = torch_flow._COMPILED_FNS["triton"][0]
    log("  block configs chosen:")
    for line in step.config_report():
        log(f"    {line}")

    # ---------------- 7. fp16 ----------------
    if args.fp16:
        log("\n== [7] fp16 (kernels promote to fp32 internally; traffic is fp16) ==")
        p16, n16 = p.half(), n.half()
        f16 = {}
        for be in ("compile", "triton"):
            f16[be], t = solve(be, p16, n16, PARAMS)
            bad = int((~torch.isfinite(f16[be])).sum())
            ok &= bad == 0
            log(f"  [{be}/fp16] {t:.1f}s  non-finite={bad}  "
                f"range [{float(f16[be].min()):.3f}, {float(f16[be].max()):.3f}]")
        m, x, c = flow_diff(f16["triton"].float(), f16["compile"].float())
        log(f"  triton/fp16 vs compile/fp16: mean {m:.3e} max {x:.3e} maxcomp {c:.3e} px")
        for be in ("compile", "triton"):
            m32, _, _ = flow_diff(f16[be].float(), outs[be].float())
            log(f"  [{be}] fp16 vs its own fp32: mean {m32:.3e} px")

    log(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
