"""Is bf16 (or fp16 / TF32) worth it for the batched TV-L1 and DeepFlow solvers?

Both ports are iterative variational solvers made of elementwise kernels plus a
handful of grid_sample / interpolate / median ops.  They are memory-bandwidth
and kernel-launch bound, never matmul bound, so the *theoretical* ceiling from
halving the element size is ~2x on the bandwidth-bound part and ~1x on the
launch-bound part.  This script measures where reality lands and what it costs
in accuracy.

Variants measured per algorithm (240x320, B in {8, 64}, backend="compile" as
the primary path, one eager reference row):

  fp32-eager     eager fp32, TF32 off                      (reference)
  fp32           compile fp32, TF32 explicitly OFF         (BASELINE)
  fp32-tf32      compile fp32, TF32 allowed (matmul+cudnn)
  bf16           inputs cast to bfloat16 (0..255 kept)
  bf16-autocast  fp32 inputs inside torch.autocast(bf16)
  fp16           inputs cast to float16 (0..255 kept)
  fp16-0to1      inputs cast to float16 and scaled to 0..1

DeepFlow only, clearly labelled EXPLORATORY:

  *-driver       torch_deepflow.calc_flow_deepflow hard-casts its inputs to
                 float32 (torch_deepflow.py:441-442) and allocates the coarsest
                 flow as float32 (line 478), so a bf16/fp16 *input cast* cannot
                 reach the solver at all.  The "-driver" rows re-run the same
                 20-line public wrapper from *this* file with the input dtype
                 preserved, reusing torch_deepflow's own unmodified internals
                 (_gaussian_blur / _resize / _variational_refinement /
                 _build_system / _sor_step).  No solver file is edited.

Usage (GPU node, one algo per job keeps each run inside the srun window):
    srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
         --time=01:00:00 ./run_nvme.sh python bf16_study.py --algos tvl1
    ... --algos deepflow
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import traceback
import warnings
from dataclasses import dataclass

import numpy as np
import torch

import torch_deepflow as tdf
from benchmark import DEEPFLOW_PARAMS, TVL1_PARAMS  # read-only import
from benchmark_data import load_flow_gt, load_pairs
from torch_deepflow import calc_flow_deepflow
from torch_flow import calc_flow_tvl1

HW = (240, 320)
BATCHES = (8, 64)
REPS = 3
ACC_B = 8          # accuracy is measured on the B=8 batch of distinct pairs
L40_BW_GBS = 864.0  # L40 GDDR6 peak bandwidth, GB/s (ECC on is a bit lower)


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
@dataclass
class Variant:
    name: str
    dtype: torch.dtype = torch.float32
    backend: str = "compile"
    tf32: bool = False
    autocast: bool = False
    scale: float = 1.0
    driver: bool = False           # deepflow: bypass the fp32 hard-cast wrapper
    batches: tuple = BATCHES
    prio: int = 1                  # lower runs first / survives the time budget
    note: str = ""


def variants_for(algo: str) -> list[Variant]:
    v = [
        Variant("fp32", prio=0, note="BASELINE compile fp32, TF32 off"),
        Variant("fp32-eager", backend="eager", prio=0, batches=(8,),
                note="eager fp32 reference"),
        Variant("fp32-0to1", scale=1.0 / 255.0, prio=3, batches=(8,),
                note="control: fp32 solver, inputs rescaled to 0..1"),
        Variant("bf16", dtype=torch.bfloat16, prio=1,
                note="input cast to bf16"),
        Variant("fp32-tf32", tf32=True, prio=2,
                note="TF32 allowed (matmul + cudnn)"),
        Variant("fp16", dtype=torch.float16, prio=2,
                note="input cast to fp16, 0..255"),
        Variant("fp16-0to1", dtype=torch.float16, scale=1.0 / 255.0, prio=3,
                note="input cast to fp16, rescaled to 0..1"),
        Variant("bf16-autocast", autocast=True, prio=3,
                note="fp32 inputs inside autocast(bf16)"),
    ]
    if algo == "deepflow":
        # DeepFlow builds ~44 pyramid levels => ~88 Inductor graphs per
        # (dtype, B), ~10x TV-L1's warm-up.  Only the rows that need a *new*
        # dtype pay that; reorder so the cheap/decisive ones land first and
        # the compile-hungry ones sit behind the time budget.
        prios = {"fp32": 0, "fp32-eager": 0,
                 "bf16": 1, "fp16": 1,               # free: wrapper upcasts to fp32
                 "fp32-0to1": 1, "fp16-0to1": 1,     # free: same fp32 graphs
                 "fp32-tf32": 3, "bf16-autocast": 3}
        for var in v:
            var.prio = prios.get(var.name, var.prio)
            if var.name in ("fp32-tf32", "bf16-autocast", "fp16", "fp16-0to1"):
                var.batches = (8,)
        v += [
            Variant("fp32-driver", driver=True, prio=1,
                    note="EXPLORATORY control: out-of-file wrapper, fp32"),
            Variant("bf16-driver", dtype=torch.bfloat16, driver=True, prio=1,
                    note="EXPLORATORY: solver actually runs in bf16"),
            Variant("fp16-driver", dtype=torch.float16, driver=True, prio=2,
                    batches=(8,),
                    note="EXPLORATORY: solver actually runs in fp16, 0..255"),
            Variant("fp16-0to1-driver", dtype=torch.float16, driver=True,
                    scale=1.0 / 255.0, prio=2, batches=(8,),
                    note="EXPLORATORY: fp16 solver, inputs rescaled to 0..1"),
        ]
    return v


# --------------------------------------------------------------------------- #
# environment knobs
# --------------------------------------------------------------------------- #
def set_tf32(on: bool) -> None:
    """Toggle TF32 for both matmul and cudnn (conv), old and new API."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.backends.cuda.matmul.allow_tf32 = on
        torch.backends.cudnn.allow_tf32 = on
        for obj, attr in ((getattr(torch.backends.cuda, "matmul", None), "fp32_precision"),
                          (getattr(torch.backends.cudnn, "conv", None), "fp32_precision")):
            if obj is not None and hasattr(obj, attr):
                try:
                    setattr(obj, attr, "tf32" if on else "ieee")
                except Exception:  # noqa: BLE001 - read-only on some builds
                    pass


def raise_dynamo_limits() -> None:
    """DeepFlow compiles ~2 graphs per pyramid level per (dtype, B): make room."""
    import torch._dynamo as dyn

    for name, val in (("cache_size_limit", 8192), ("recompile_limit", 8192),
                      ("accumulated_cache_size_limit", 262144),
                      ("accumulated_recompile_limit", 262144)):
        if hasattr(dyn.config, name):
            try:
                setattr(dyn.config, name, val)
            except Exception:  # noqa: BLE001
                pass


def dynamo_graphs() -> int:
    from torch._dynamo.utils import counters
    return int(counters.get("stats", {}).get("unique_graphs", 0))


def clear_gpu() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# DeepFlow: out-of-file, dtype-preserving copy of the public wrapper
# --------------------------------------------------------------------------- #
def deepflow_driver(prev: torch.Tensor, next: torch.Tensor, *, sigma: float,
                    min_size: int, downscale_factor: float,
                    fixed_point_iterations: int, sor_iterations: int,
                    alpha: float, delta: float, gamma: float, omega: float,
                    backend: str) -> torch.Tensor:
    """calc_flow_deepflow without the ``.to(torch.float32)`` on lines 441-442.

    Byte-for-byte the same control flow; every heavy op is torch_deepflow's own
    unmodified helper.  Only the dtype of the images / initial flow differs.
    """
    sys_fn, sor_fn = tdf._resolve_backend(backend)
    dev, dt = prev.device, prev.dtype
    I0, I1 = prev, next.to(dt)
    B, H, W = I0.shape

    I0 = tdf._gaussian_blur(I0, sigma)
    I1 = tdf._gaussian_blur(I1, sigma)

    sizes: list[tuple[int, int]] = [(H, W)]
    h, w = H, W
    while True:
        nh = int(h * downscale_factor + 0.5)
        nw = int(w * downscale_factor + 0.5)
        if nh <= min_size or nw <= min_size:
            break
        sizes.append((nh, nw))
        h, w = nh, nw

    pyr0, pyr1 = [I0], [I1]
    for s in sizes[1:]:
        pyr0.append(tdf._resize(pyr0[-1].unsqueeze(1), s).squeeze(1))
        pyr1.append(tdf._resize(pyr1[-1].unsqueeze(1), s).squeeze(1))

    vr_kwargs = dict(alpha=4.0 * alpha, delta=delta / 3.0, gamma=gamma / 3.0,
                     fixed_point_iterations=fixed_point_iterations,
                     sor_iterations=sor_iterations, omega=omega,
                     sys_fn=sys_fn, sor_fn=sor_fn)

    flow = torch.zeros(B, 2, *sizes[-1], device=dev, dtype=dt)
    for level in range(len(sizes) - 1, -1, -1):
        flow = tdf._variational_refinement(pyr0[level], pyr1[level], flow, **vr_kwargs)
        if level > 0:
            flow = tdf._resize(flow, sizes[level - 1]) * (1.0 / downscale_factor)
    return flow


def make_fn(algo: str, v: Variant):
    if algo == "tvl1":
        return lambda p, n: calc_flow_tvl1(p, n, backend=v.backend, **TVL1_PARAMS)
    if v.driver:
        return lambda p, n: deepflow_driver(p, n, backend=v.backend, **DEEPFLOW_PARAMS)
    return lambda p, n: calc_flow_deepflow(p, n, backend=v.backend, **DEEPFLOW_PARAMS)


# --------------------------------------------------------------------------- #
# data / metrics
# --------------------------------------------------------------------------- #
def resized_gt(name: str, hw: tuple[int, int]) -> np.ndarray | None:
    import cv2
    g = load_flow_gt(name)
    if g is None or g.shape[:2] == hw:
        return g
    sy, sx = hw[0] / g.shape[0], hw[1] / g.shape[1]
    g = cv2.resize(g, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return g * np.array([sx, sy], dtype=g.dtype)


def flows_to_numpy(out: torch.Tensor) -> list[np.ndarray]:
    a = out.detach().permute(0, 2, 3, 1).float().cpu().numpy()  # (B, H, W, 2)
    return [np.ascontiguousarray(x) for x in a]


def _valid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
            & (np.abs(a).max(axis=-1) < 1e9) & (np.abs(b).max(axis=-1) < 1e9))


def epe(a: np.ndarray, b: np.ndarray) -> float:
    m = _valid(a, b)
    if not m.any():
        return float("nan")
    return float(np.linalg.norm(a - b, axis=-1)[m].mean())


def mean_epe(flows, refs) -> float:
    vals = [epe(f, r) for f, r in zip(flows, refs) if r is not None]
    return float(np.mean(vals)) if vals else float("nan")


def diff_stats(flows, base) -> dict:
    ds, bad, tot = [], 0, 0
    for a, b in zip(flows, base):
        m = _valid(a, b)
        bad += int((~np.isfinite(a)).any(axis=-1).sum())
        tot += m.size
        ds.append(np.linalg.norm(a - b, axis=-1)[m])
    d = np.concatenate(ds) if ds else np.zeros(1)
    if d.size == 0:
        return {"mean": float("nan"), "p99": float("nan"), "max": float("nan"),
                "nonfinite_frac": bad / max(tot, 1)}
    return {"mean": float(d.mean()), "p99": float(np.percentile(d, 99)),
            "max": float(d.max()), "nonfinite_frac": bad / max(tot, 1)}


def sanity(flows) -> dict:
    a = np.concatenate([f.ravel() for f in flows])
    fin = np.isfinite(a)
    mag = np.concatenate([np.linalg.norm(f, axis=-1).ravel() for f in flows])
    mag = mag[np.isfinite(mag)]
    return {"nonfinite_frac": float((~fin).mean()),
            "mean_mag": float(mag.mean()) if mag.size else float("nan"),
            "max_mag": float(mag.max()) if mag.size else float("nan")}


# --------------------------------------------------------------------------- #
# roofline (order-of-magnitude only)
# --------------------------------------------------------------------------- #
def roofline_ms(algo: str, B: int, itemsize: int) -> tuple[float, int]:
    """Rough bandwidth-limited time for the hot inner loop, and its call count.

    Counts only the dominant inner step (TV-L1 primal-dual step / DeepFlow SOR
    sweep) and models its traffic as ``CH`` full-size planes read+written.
    Deliberately crude: it fixes the *scale* of the bandwidth ceiling so the
    measured bf16 speedup can be read against "2x if bandwidth bound".
    """
    if algo == "tvl1":
        sizes = [HW]
        for _ in range(1, TVL1_PARAMS["nscales"]):
            h2 = int(round(sizes[-1][0] * TVL1_PARAMS["scale_step"]))
            w2 = int(round(sizes[-1][1] * TVL1_PARAMS["scale_step"]))
            if h2 < 16 or w2 < 16:
                break
            sizes.append((h2, w2))
        per_level = (TVL1_PARAMS["warps"] * TVL1_PARAMS["outer_iterations"]
                     * TVL1_PARAMS["inner_iterations"])
        calls = per_level * len(sizes)
        area = per_level * sum(h * w for h, w in sizes)
        ch = 18.5   # u,p,rho_c,I1wx,I1wy,thr,safe_grad,pos read + u,p written
    else:
        sizes, (h, w) = [HW], HW
        while True:
            nh = int(h * DEEPFLOW_PARAMS["downscale_factor"] + 0.5)
            nw = int(w * DEEPFLOW_PARAMS["downscale_factor"] + 0.5)
            if nh <= DEEPFLOW_PARAMS["min_size"] or nw <= DEEPFLOW_PARAMS["min_size"]:
                break
            sizes.append((nh, nw))
            h, w = nh, nw
        per_level = (DEEPFLOW_PARAMS["fixed_point_iterations"]
                     * DEEPFLOW_PARAMS["sor_iterations"])
        calls = per_level * len(sizes)
        area = per_level * sum(h * w for h, w in sizes)
        ch = 24.0   # d + wL,wC,wT,A11,A12,A22,b1,b2 read, d written, x2 colors
    byts = ch * area * B * itemsize
    return byts / (L40_BW_GBS * 1e9) * 1e3, calls


# --------------------------------------------------------------------------- #
# one measurement cell
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_cell(algo: str, v: Variant, B: int, base32: tuple[torch.Tensor, torch.Tensor],
             want_flows: bool) -> tuple[dict, list[np.ndarray] | None]:
    rec = {"algo": algo, "variant": v.name, "B": B, "dtype": str(v.dtype).split(".")[-1],
           "backend": v.backend, "tf32": v.tf32, "autocast": v.autocast,
           "scale": v.scale, "driver": v.driver, "status": "ok"}
    fn = make_fn(algo, v)
    set_tf32(v.tf32)
    p32, n32 = base32
    r = -(-B // p32.shape[0])
    p = (p32.repeat(r, 1, 1)[:B] * v.scale).to(v.dtype)
    n = (n32.repeat(r, 1, 1)[:B] * v.scale).to(v.dtype)
    flows = None
    try:
        clear_gpu()
        g0 = dynamo_graphs()
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if v.autocast else contextlib.nullcontext())
        t0 = time.perf_counter()
        with ctx:
            out = fn(p, n)
        torch.cuda.synchronize()
        rec["warmup_s"] = round(time.perf_counter() - t0, 2)
        rec["new_graphs"] = dynamo_graphs() - g0
        rec["out_dtype"] = str(out.dtype).split(".")[-1]
        if want_flows:
            flows = flows_to_numpy(out)
        del out

        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            with ctx:
                o = fn(p, n)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            del o
        t = min(ts)
        rl, calls = roofline_ms(algo, B, torch.empty((), dtype=v.dtype).element_size())
        rec.update(call_s=round(t, 5), ms_per_pair=round(t / B * 1e3, 3),
                   spread=round((max(ts) - t) / t, 4),
                   peak_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                   roofline_ms_per_pair=round(rl / B, 3), inner_calls=calls,
                   roofline_frac=round(rl / (t * 1e3), 3))
    except Exception as exc:  # noqa: BLE001 - a failed dtype is itself a result
        rec["status"] = "ERR"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        log(f"    !! {rec['error']}")
        traceback.print_exc(limit=4)
    finally:
        del p, n
        set_tf32(False)
        clear_gpu()
    return rec, flows


# --------------------------------------------------------------------------- #
# autocast introspection
# --------------------------------------------------------------------------- #
def autocast_probe() -> None:
    import torch.nn.functional as F
    log("\n-- what does autocast(cuda, bf16) actually cast? (fp32 inputs) --")
    x = torch.randn(1, 1, 8, 8, device="cuda")
    g = torch.rand(1, 8, 8, 2, device="cuda") * 2 - 1
    k = torch.randn(1, 1, 3, 3, device="cuda")
    probes = {
        "conv2d": lambda: F.conv2d(x, k, padding=1),
        "grid_sample(bicubic)": lambda: F.grid_sample(x, g, mode="bicubic", align_corners=True),
        "grid_sample(bilinear)": lambda: F.grid_sample(x, g, mode="bilinear", align_corners=True),
        "interpolate(bilinear)": lambda: F.interpolate(x, size=(4, 4), mode="bilinear"),
        "pad(replicate)": lambda: F.pad(x, (1, 1, 1, 1), mode="replicate"),
        "mul/add": lambda: x * x + x,
        "div": lambda: x / (x * x + 1.0),
        "sqrt": lambda: torch.sqrt(x * x + 1.0),
        "hypot": lambda: torch.hypot(x, x),
        "median(unfold)": lambda: x.unfold(2, 3, 1).unfold(3, 3, 1)
                                   .reshape(1, 1, 6, 6, 9).median(dim=-1).values,
        "where": lambda: torch.where(x > 0, x, -x),
    }
    log(f"{'op':<24} {'no autocast':>12} {'autocast bf16':>14}")
    for name, f in probes.items():
        with torch.no_grad():
            d0 = str(f().dtype).split(".")[-1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                d1 = str(f().dtype).split(".")[-1]
        log(f"{name:<24} {d0:>12} {d1:>14}{'   <-- cast' if d0 != d1 else ''}")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_algo(algo: str, pairs, gts, out_rows: list, budget_s: float,
             only: list[str] | None = None,
             batches: tuple[int, ...] = BATCHES) -> None:
    log(f"\n{'=' * 100}\n{algo.upper()}  ({HW[0]}x{HW[1]})\n{'=' * 100}")
    p32 = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs]).cuda()
    n32 = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs]).cuda()
    base = (p32, n32)

    vs = sorted(variants_for(algo), key=lambda v: v.prio)
    if only:
        vs = [v for v in vs if v.name in only]
    acc_flows: dict[str, list[np.ndarray]] = {}
    t_start = time.perf_counter()

    for B in batches:
        for v in vs:
            if B not in v.batches and not only:
                continue
            spent = time.perf_counter() - t_start
            if spent > budget_s and v.prio > 1:
                log(f"  [skip {algo}/{v.name}/B={B}] time budget "
                    f"({spent / 60:.1f} min > {budget_s / 60:.1f} min)")
                continue
            log(f"  [{algo}/{v.name}/B={B}] {v.note}")
            want = (B == ACC_B)
            rec, flows = run_cell(algo, v, B, base, want)
            if want and flows is not None:
                acc_flows[v.name] = flows
                rec.update(sanity(flows))
            out_rows.append(rec)
            if rec["status"] == "ok":
                log(f"    {rec['ms_per_pair']:.2f} ms/pair  peak {rec['peak_mib']:.0f} MiB  "
                    f"warmup {rec['warmup_s']:.1f}s ({rec['new_graphs']} new graphs)  "
                    f"out={rec['out_dtype']}  roofline {rec['roofline_ms_per_pair']:.2f} "
                    f"ms/pair ({100 * rec['roofline_frac']:.0f}% of measured)")

    # ---- accuracy vs GT and vs the fp32-compile baseline ------------------ #
    ref = acc_flows.get("fp32")
    for rec in out_rows:
        if rec["algo"] != algo or rec["B"] != ACC_B or rec["status"] != "ok":
            continue
        f = acc_flows.get(rec["variant"])
        if f is None:
            continue
        rec["epe_gt"] = mean_epe(f, gts)
        if ref is not None:
            rec["epe_vs_fp32"] = mean_epe(f, ref)
            rec.update({f"d_{k}": val for k, val in diff_stats(f, ref).items()})
            rec["bit_identical_to_fp32"] = bool(
                all(np.array_equal(a, b) for a, b in zip(f, ref)))

    print_table(algo, out_rows)


def print_table(algo: str, rows: list) -> None:
    rows = [r for r in rows if r["algo"] == algo]
    log(f"\n---- {algo} results ----")
    hdr = (f"{'variant':<18} {'B':>3} {'ms/pair':>9} {'vs fp32':>8} {'peakMiB':>8} "
           f"{'warmup s':>9} {'graphs':>7} {'EPE_GT':>8} {'EPE_f32':>8} "
           f"{'p99diff':>9} {'maxdiff':>9} {'NaN%':>7} {'rooflin%':>8}")
    log(hdr)
    log("-" * len(hdr))
    base = {r["B"]: r.get("ms_per_pair") for r in rows if r["variant"] == "fp32"}
    for r in rows:
        if r["status"] != "ok":
            log(f"{r['variant']:<18} {r['B']:>3}  FAILED: {r.get('error', '')[:80]}")
            continue
        b = base.get(r["B"])
        sp = f"{b / r['ms_per_pair']:.2f}x" if b and r.get("ms_per_pair") else "-"

        def g(k, fmt=".4f"):
            val = r.get(k)
            return format(val, fmt) if isinstance(val, float) and math.isfinite(val) else "-"
        log(f"{r['variant']:<18} {r['B']:>3} {r['ms_per_pair']:>9.2f} {sp:>8} "
            f"{r['peak_mib']:>8.0f} {r['warmup_s']:>9.1f} {r['new_graphs']:>7} "
            f"{g('epe_gt', '.4f'):>8} {g('epe_vs_fp32', '.4f'):>8} "
            f"{g('d_p99', '.4f'):>9} {g('d_max', '.3f'):>9} "
            f"{g('nonfinite_frac', '.2%'):>7} "
            f"{100 * r.get('roofline_frac', float('nan')):>7.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algos", nargs="+", default=["tvl1", "deepflow"],
                    choices=["tvl1", "deepflow"])
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these variant names (bypasses per-variant batch limits)")
    ap.add_argument("--batches", nargs="+", type=int, default=list(BATCHES))
    ap.add_argument("--budget-min", type=float, default=22.0,
                    help="per-algo soft budget; prio>1 variants are skipped past it")
    ap.add_argument("--out", default="/fnwi_fs/ivi/irlab/personal/tlong/code/"
                                    "optic_flow/bf16_study_results.jsonl")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this study needs a GPU")
    raise_dynamo_limits()
    set_tf32(False)
    log(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
        f"cap {torch.cuda.get_device_capability(0)}")
    log(f"TVL1_PARAMS:     {TVL1_PARAMS}")
    log(f"DEEPFLOW_PARAMS: {DEEPFLOW_PARAMS}")
    log(f"default TF32 flags: matmul={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn={torch.backends.cudnn.allow_tf32} (forced off for the baseline)")

    pairs = load_pairs(max_pairs=args.max_pairs, gray=True, size=HW)
    gts = [resized_gt(n, HW) for _, _, n in pairs]
    log(f"pairs ({len(pairs)}, {sum(g is not None for g in gts)} with GT): "
        + ", ".join(f"{n}{'' if g is not None else '*'}" for (_, _, n), g in zip(pairs, gts)))

    autocast_probe()

    rows: list[dict] = []
    t0 = time.perf_counter()
    for algo in args.algos:
        run_algo(algo, pairs, gts, rows, args.budget_min * 60,
                 only=args.only, batches=tuple(args.batches))
    log(f"\ntotal wall: {(time.perf_counter() - t0) / 60:.1f} min")

    path = os.path.abspath(args.out)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
